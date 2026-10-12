#include "editor/asset_workspace.hpp"
#include "editor/mesh_edit.hpp"
#include "physics/convex_decomposition.hpp"
#include "physics/jolt_collision.hpp"
#include "world/world_scene.hpp"
#include "world/picking_geometry.hpp"
#include <ImGuizmo.h>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <iostream>
#include <set>
#include <stdexcept>
#include <utility>

namespace openf4::editor
{
    struct AssetWorkspace::Impl
    {
        EditorDocument& world;
        std::uint32_t model;
        assets::ESMModelKind baseKind;
        AssetDocument asset;
        std::vector<std::string> messages;
        std::vector<render::SceneMaterial> gpuMaterials;
        std::vector<std::pair<std::shared_ptr<const assets::ModelImage>, std::shared_ptr<const assets::ModelImage>>> images;
        std::vector<render::SceneInstance> objects;
        std::vector<world::detail::PickingGeometry> picking;
        std::set<std::size_t> surfaces{0};
        std::optional<AssetCollisionPart> partKind;
        std::size_t part = 0, system = 0, body = 0;
        glm::vec3 center{}, minimum{}, maximum{};
        float radius = 100, yaw = 0.5f, pitch = 0.3f, distance = 350, aspect = 1;
        ImVec2 viewportPosition{}, viewportSize{1,1};
        std::uint32_t firstInstance = 0;
        std::uint64_t previewRevision = ~std::uint64_t{};
        bool installed = false, showCollision = true, gesture = false, gizmoDrag = false, finishGesture = false;
        ImGuizmo::OPERATION operation = ImGuizmo::TRANSLATE;
        std::unique_ptr<physics::JoltWorld> physics;
        physics::BodyHandle previewBody;
        glm::mat4 simulation{1}, bodyLocal{1};
        bool playing = false, playRequested = false;
        float accumulator = 0;
        std::array<char, 1024> texturePath{};
        // ---- PROTOTYPE 0.1 mesh editing (cube spawner + vertex/edge/triangle transforms) ----
        std::shared_ptr<const assets::ModelGeometry> previewGeometry;
        bool meshMode = false;
        mesh::Element element = mesh::Element::vertex;
        std::set<std::uint32_t> meshSelection;
        std::optional<std::size_t> topologySurface;
        std::shared_ptr<const assets::ModelGeometry> topologyGeometry;
        mesh::Topology topology;
        glm::mat4 meshPivot{1}, meshStartPivot{1};
        std::vector<glm::vec3> meshStartPositions;
        std::vector<std::uint32_t> meshDragPoints;
        // ---- PROTOTYPE collision mode: pick/transform/delete collision parts in the viewport ----
        bool collisionMode = false;
        // Convex-part element editing inside collision mode (nullopt = whole-shape gizmo).
        std::optional<mesh::Element> collisionElement;
        std::set<std::uint32_t> collisionSelection;
        mesh::Topology collisionTopology;               // world engine space (root applied)
        std::optional<std::array<std::size_t,3>> collisionTopologyKey;
        std::uint64_t collisionTopologyRevision = ~std::uint64_t{};
        glm::mat4 collisionRoot{1}, collisionStartRoot{1}, collisionPivot{1}, collisionStartPivot{1};
        std::vector<glm::vec3> collisionStartPositions;
        std::vector<std::uint32_t> collisionDragPoints;
        std::uint64_t convexityRevision = ~std::uint64_t{};
        std::set<std::array<std::size_t,3>> nonConvexParts;
        std::shared_ptr<const assets::ModelGeometry> surfaceReportGeometry;
        std::set<std::size_t> surfaceReportSelection;
        std::optional<physics::ConvexityReport> surfaceReport;
        int splitMaxPieces = 16, splitResolution = 40; float splitTolerancePercent = 2.0f;

        Impl(EditorDocument& doc, std::uint32_t id, const std::filesystem::path& data)
            : world(doc), model(id), baseKind(doc.model(id)?doc.model(id)->kind:assets::ESMModelKind::staticObject),
              asset(source(doc,id), path(doc,id), data, baseKind==assets::ESMModelKind::movableStatic)
        {
            world::detail::PickingGeometry bounds(asset.sourceGeometry());
            const auto box = bounds.bounds(glm::mat4(1));
            minimum = glm::vec3(box.minimum); maximum = glm::vec3(box.maximum);
            center = (minimum + maximum) * 0.5f; center.x = -center.x;
            radius = std::max(glm::length(maximum - minimum) * 0.5f, 10.0f);
            distance = radius * 3;
            for (const auto& surface : asset.sourceGeometry().surfaces)
            {
                assets::ModelGeometry geometry; geometry.surfaces.push_back(surface);
                picking.emplace_back(geometry);
            }
        }
        Impl(EditorDocument& doc, const std::filesystem::path&)
            : world(doc), model(0), baseKind(assets::ESMModelKind::staticObject), asset(AssetDocument::createEmpty())
        {
            minimum=glm::vec3(-64); maximum=glm::vec3(64); center={}; radius=64; distance=radius*3;
            surfaces.clear();
        }
        static std::filesystem::path source(EditorDocument& doc, std::uint32_t id)
        { return doc.sourceEsm(id).value_or(doc.path().string()); }
        static std::string path(EditorDocument& doc, std::uint32_t id)
        {
            const auto found = doc.models().find(id);
            if (found == doc.models().end()) throw std::runtime_error("Selected model no longer exists");
            return found->second;
        }
        template<class F> void attempt(F&& action)
        {
            try { action(); }
            catch (const std::exception& error)
            {
                if (messages.empty() || messages.back() != error.what())
                { messages.push_back(error.what()); std::cerr << "Asset editor: " << error.what() << '\n'; }
                if (gesture) { asset.cancelTransaction(); gesture = gizmoDrag = false; }
            }
        }
        glm::vec3 camera() const
        { return center + distance * glm::vec3(std::sin(yaw)*std::cos(pitch), std::sin(pitch), std::cos(yaw)*std::cos(pitch)); }
        glm::mat4 view() const { return glm::lookAtRH(camera(), center, {0,1,0}); }
        glm::mat4 projection() const
        {
            auto result = render::reverseZInfinitePerspectiveRH_ZO(glm::radians(55.0f), aspect, 0.1f);
            result[1][1] *= -1;
            return result;
        }
        void reset()
        { physics.reset(); playing = false; accumulator = 0; simulation = glm::mat4(1); }
        void start()
        {
            if (physics) { playing = true; return; }
            const auto& systems = asset.collisionSystems();
            if (systems.size() != 1 || systems[0].physics.bodies.size() != 1)
                throw std::runtime_error("Physics preview requires one mapped rigid body; other native assets remain inspectable");
            const auto& collision = systems[0].physics.bodies[0];
            auto next = std::make_unique<physics::JoltWorld>();
            (void)next->addStaticBox({radius*5, 5, radius*5}, {(minimum.x+maximum.x)*0.5f, minimum.y-25, (minimum.z+maximum.z)*0.5f});
            bodyLocal = physics::havokBodyLocalTransform(collision);
            previewBody = next->addCollisionBody(collision, glm::vec3(bodyLocal[3]), glm::quat_cast(bodyLocal));
            physics = std::move(next); playing = true;
        }
        bool texturesChanged() const
        {
            if (images.size() != asset.materials().size()) return true;
            for (std::size_t i=0;i<images.size();++i)
                if (images[i].first != asset.materials()[i].diffuseImage || images[i].second != asset.materials()[i].normalImage) return true;
            return false;
        }
        void rebuild(render::VulkanRenderer& renderer)
        {
            render::PackedMesh mesh;
            std::vector<render::SceneTexture> textures;
            std::vector<render::SceneMaterial> materials;
            std::vector<render::SceneInstance> instances;
            std::vector<decltype(images)::value_type> nextImages;
            picking.clear();
            for (const auto& surface : asset.sourceGeometry().surfaces)
            { assets::ModelGeometry geometry; geometry.surfaces.push_back(surface); picking.emplace_back(geometry); }
            for (std::size_t i=0;i<asset.sourceGeometry().surfaces.size();++i)
            {
                const auto& source = asset.sourceGeometry().surfaces[i];
                const auto& material = asset.materials()[i];
                render::SceneMaterial gpu; gpu.baseColor=material.baseColor; gpu.alphaCutoff=material.alphaCutoff; gpu.doubleSided=material.doubleSided;
                if (material.diffuseImage) { textures.push_back(world::modelSceneTexture(*material.diffuseImage)); gpu.diffuseTexture=static_cast<std::uint32_t>(textures.size()); }
                if (material.normalImage) { textures.push_back(world::modelSceneTexture(*material.normalImage)); gpu.normalTexture=static_cast<std::uint32_t>(textures.size()); }
                materials.push_back(gpu);
                nextImages.emplace_back(material.diffuseImage,material.normalImage);
                const auto range=render::appendMesh(mesh,render::buildMeshlets(source.vertices,source.indices,static_cast<std::uint32_t>(i)));
                render::SceneInstance instance; instance.bounds=range.bounds;
                instance.meshletOffset=range.meshletOffset; instance.meshletCount=range.meshletCount;
                instances.push_back(instance);
            }
            auto ground=render::makeCubeMesh();
            for(auto& m:ground.meshlets) m.materialIndex=static_cast<std::uint32_t>(materials.size());
            render::SceneMaterial floor; floor.baseColor={0.18f,0.2f,0.22f,1}; materials.push_back(floor);
            const auto floorRange=render::appendMesh(mesh,std::move(ground));
            render::SceneInstance floorInstance;
            floorInstance.transform=glm::translate(glm::mat4(1),glm::vec3((minimum.x+maximum.x)*0.5f,minimum.y-25,(minimum.z+maximum.z)*0.5f))
                *glm::scale(glm::mat4(1),glm::vec3(radius*5,5,radius*5));
            floorInstance.bounds=glm::vec4(glm::vec3(floorInstance.transform[3]),radius*8);
            floorInstance.meshletOffset=floorRange.meshletOffset; floorInstance.meshletCount=floorRange.meshletCount;
            instances.push_back(floorInstance);
            installed=false;
            firstInstance=renderer.replacePreviewScene(std::move(mesh),instances,materials,textures);
            objects=std::move(instances); gpuMaterials=std::move(materials); images=std::move(nextImages);
            installed=true; previewRevision=asset.revision(); previewGeometry=asset.geometryHandle();
        }
        void update(render::VulkanRenderer& renderer,float delta)
        {
            const bool startRequested = std::exchange(playRequested, false);
            // Mesh edits replace the geometry snapshot; the preview is rebuilt (prototype: full re-upload).
            if (!installed || texturesChanged() || previewGeometry != asset.geometryHandle()) { reset(); rebuild(renderer); }
            if (previewRevision != asset.revision())
            {
                reset();
                for(std::size_t i=0;i<asset.materials().size();++i)
                {
                    auto material=gpuMaterials[i];
                    material.baseColor=asset.materials()[i].baseColor; material.alphaCutoff=asset.materials()[i].alphaCutoff; material.doubleSided=asset.materials()[i].doubleSided;
                    if(material.diffuseTexture) material.diffuseTexture+=renderer.previewTextureOffset()-1;
                    if(material.normalTexture) material.normalTexture+=renderer.previewTextureOffset()-1;
                    renderer.updateMaterial(renderer.previewMaterialOffset()+static_cast<std::uint32_t>(i),material);
                }
                previewRevision=asset.revision();
            }
            if (startRequested) start();
            if(playing && physics)
            {
                accumulator+=std::min(delta,0.1f);
                while(accumulator>=1.0f/60) { physics->step(1.0f/60); accumulator-=1.0f/60; }
                simulation=physics->bodyTransform(previewBody)*glm::inverse(bodyLocal);
            }
            for(std::size_t i=0;i<asset.materials().size();++i)
            {
                auto sphere=objects[i].bounds; sphere=glm::vec4(glm::vec3(simulation*glm::vec4(glm::vec3(sphere),1)),sphere.w);
                renderer.updateInstanceTransform(firstInstance+static_cast<std::uint32_t>(i),simulation,sphere);
            }
            std::vector<std::uint32_t> selected;
            if(!partKind && !meshMode && !collisionMode) for(auto i:surfaces) if(i<asset.materials().size()) selected.push_back(firstInstance+static_cast<std::uint32_t>(i));
            renderer.setSelection(selected,selected.empty()?std::nullopt:std::optional(selected.front()));
        }
        void trackWidget()
        {
            if(ImGui::IsItemActivated() && !gesture) { asset.beginTransaction(); gesture=true; }
            if(ImGui::IsItemDeactivated() && gesture && !gizmoDrag) finishGesture=true;
        }
        bool drag(const char* label,float* value,float speed,float lo,float hi)
        { const bool changed=ImGui::DragFloat(label,value,speed,lo,hi); trackWidget(); return changed; }
        void save()
        {
            if(gesture) { asset.endTransaction(); gesture=gizmoDrag=false; }
            if(!model)
            {
                // First save of a blank asset: publish under the FormID the new base will take.
                const auto package=asset.saveNative(world.path(),world.nextFormId());
                model=world.addNativeModel(package.modelPath,recordKind());
                baseKind=world.model(model)->kind;
                asset.markSaved();
                messages.insert(messages.end(),package.diagnostics.begin(),package.diagnostics.end());
                messages.push_back("New asset saved as "+std::string(baseKind==assets::ESMModelKind::movableStatic?"MSTT":"STAT")
                    +" base "+std::to_string(model)+". Place it from the Asset Browser; Save World to persist it.");
                return;
            }
            if(!world.model(model)) throw std::runtime_error("Asset base was removed by world undo; reopen it from the world before saving");
            const auto package=asset.saveNative(world.path(),model);
            model=world.assignNativeModel(model,package.modelPath,recordKind());
            baseKind=world.model(model)->kind;
            asset.markSaved();
            messages.insert(messages.end(),package.diagnostics.begin(),package.diagnostics.end());
            messages.push_back("Native package saved. Save World to persist its model assignment.");
        }
        assets::ESMModelKind recordKind() const
        {
            // Collision-only edits preserve both native motion flags and the existing ESM record type.
            if(!asset.capabilities().motion)
                return baseKind;
            return asset.movable()?assets::ESMModelKind::movableStatic:assets::ESMModelKind::staticObject;
        }
        void outline()
        {
            ImGui::Begin("Outliner");
            ImGui::TextUnformatted("Asset surfaces");
            for(std::size_t i=0;i<asset.materials().size();++i)
            {
                const auto label="Surface "+std::to_string(i);
                if(ImGui::Selectable(label.c_str(),!partKind && surfaces.contains(i)))
                { if(!ImGui::GetIO().KeyCtrl) surfaces.clear(); if(!surfaces.erase(i)) surfaces.insert(i); partKind.reset(); collisionMode=false; }
            }
            if(asset.materials().empty()) ImGui::TextDisabled("  (empty - Mesh > Add cube)");
            ImGui::SeparatorText("Collision bodies");
            refreshConvexity();
            const auto& systems=asset.collisionSystems();
            for(std::size_t s=0;s<systems.size();++s) for(std::size_t b=0;b<systems[s].physics.bodies.size();++b)
            {
                ImGui::PushID(static_cast<int>(s)); ImGui::PushID(static_cast<int>(b));
                if(ImGui::Selectable(("Body "+std::to_string(s)+":"+std::to_string(b)).c_str(),s==system && b==body && !partKind))
                {system=s;body=b;partKind.reset();if(asset.capabilities().collision) setCollisionMode(true);}
                const auto& rigid=systems[s].physics.bodies[b];
                const auto row=[&](AssetCollisionPart kind,std::size_t index,const char* name){
                    const bool dented=kind==AssetCollisionPart::convex && nonConvexParts.contains({s,b,index});
                    if(ImGui::Selectable((std::string("  ")+name+" "+std::to_string(index)+(dented?"  (NOT CONVEX)":"")).c_str(),partKind==kind && part==index && system==s && body==b))
                    {setCollisionMode(asset.capabilities().collision);partKind=kind;part=index;system=s;body=b;}
                };
                for(std::size_t i=0;i<rigid.parts.size();++i) row(AssetCollisionPart::convex,i,"Convex");
                for(std::size_t i=0;i<rigid.capsules.size();++i) row(AssetCollisionPart::capsule,i,"Capsule");
                if(rigid.triangleMesh) row(AssetCollisionPart::triangleMesh,0,"Triangle mesh");
                ImGui::PopID(); ImGui::PopID();
            }
            ImGui::End();
        }
        void inspector()
        {
            ImGui::Begin("Inspector");
            if(!model) ImGui::TextWrapped("New asset (no world base yet)%s",asset.dirty()?" *":"");
            else if(asset.modelPath().empty()) ImGui::TextWrapped("%s%s",world.models().contains(model)?world.models().at(model).c_str():"(removed)",asset.dirty()?" *":"");
            else ImGui::TextWrapped("%s%s",asset.modelPath().c_str(),asset.dirty()?" *":"");
            const auto& cap=asset.capabilities();
            if(!cap.reason.empty()) ImGui::TextWrapped("%s",cap.reason.c_str());
            ImGui::BeginDisabled(!cap.materials && !cap.collision);
            if(ImGui::Button("Save Native NIF")) attempt([&]{save();});
            ImGui::EndDisabled();
            ImGui::Checkbox("Show collision",&showCollision);
            if(ImGui::Button(playing?"Pause":"Play")) {if(playing) playing=false; else playRequested=true;}
            ImGui::SameLine(); if(ImGui::Button("Reset")) reset();
            ImGui::TextDisabled("Simulation is never saved.");
            meshPanel();
            ImGui::BeginDisabled(!cap.materials);
            if(!surfaces.empty() && *surfaces.begin()<asset.materials().size() && ImGui::CollapsingHeader("Material",ImGuiTreeNodeFlags_DefaultOpen))
            {
                const auto index=*surfaces.begin(); auto material=asset.materials()[index]; bool changed=false;
                changed|=ImGui::ColorEdit4("Tint",glm::value_ptr(material.baseColor)); trackWidget();
                changed|=drag("Alpha cutoff",&material.alphaCutoff,0.005f,0,1);
                changed|=ImGui::Checkbox("Double sided",&material.doubleSided);
                if(changed) attempt([&]{asset.setMaterial(index,material);});
                ImGui::TextWrapped("Diffuse: %s",material.diffuseTexture.c_str());
                ImGui::TextWrapped("Normal: %s",material.normalTexture.c_str());
                ImGui::InputText("Texture path",texturePath.data(),texturePath.size());
                if(ImGui::Button("Set diffuse")) attempt([&]{asset.setTexture(index,texturePath.data(),false);});
                ImGui::SameLine(); if(ImGui::Button("Set normal")) attempt([&]{asset.setTexture(index,texturePath.data(),true);});
                ImGui::TextWrapped("PNG, JPEG or DDS. Native alpha blending/PBR appearance is not fully previewed; unsupported metadata is retained.");
            }
            ImGui::EndDisabled();
            ImGui::BeginDisabled(!cap.collision || playing);
            if(ImGui::CollapsingHeader("Collision",ImGuiTreeNodeFlags_DefaultOpen))
            {
                bool editCollision=collisionMode;
                if(ImGui::Checkbox("Edit collision (C)",&editCollision)) setCollisionMode(editCollision);
                if(collisionMode)
                {
                    int op=operation==ImGuizmo::TRANSLATE?0:operation==ImGuizmo::ROTATE?1:2;
                    ImGui::RadioButton("Move (W)##col",&op,0); ImGui::SameLine();
                    ImGui::RadioButton("Rotate (E)##col",&op,1); ImGui::SameLine();
                    ImGui::RadioButton("Scale (R)##col",&op,2);
                    operation=op==0?ImGuizmo::TRANSLATE:op==1?ImGuizmo::ROTATE:ImGuizmo::SCALE;
                    int ce=collisionElement?static_cast<int>(*collisionElement)+1:0;
                    ImGui::RadioButton("Shape (0)",&ce,0); ImGui::SameLine();
                    ImGui::RadioButton("Vertex (1)##col",&ce,1); ImGui::SameLine();
                    ImGui::RadioButton("Edge (2)##col",&ce,2); ImGui::SameLine();
                    ImGui::RadioButton("Triangle (3)##col",&ce,3);
                    setCollisionElement(ce?std::optional(static_cast<mesh::Element>(ce-1)):std::nullopt);
                    ImGui::TextDisabled("LMB pick a shape | Delete removes it | Esc cancels a drag");
                    ImGui::TextDisabled("1/2/3 edit vertices/edges/triangles of the picked convex part (Shift toggles).");
                    ImGui::TextDisabled("Triangle-mesh collision can be picked/deleted, not transformed.");
                }
                bool movable=recordKind()==assets::ESMModelKind::movableStatic; ImGui::BeginDisabled(!cap.motion);
                if(ImGui::Checkbox("Movable prop (MSTT)",&movable)) attempt([&]{asset.setMovable(movable);});
                ImGui::EndDisabled();
                if(recordKind()!=baseKind)
                    ImGui::TextWrapped("World base is still %s. Save Native NIF to make it %s; placements are retargeted.",
                        baseKind==assets::ESMModelKind::movableStatic?"MSTT":"STAT",
                        recordKind()==assets::ESMModelKind::movableStatic?"MSTT":"STAT");
                const std::vector<std::size_t> selected(surfaces.begin(),surfaces.end());
                if(ImGui::Button("Convex from surfaces")) attempt([&]{asset.generateCollision(selected,true,system,body);partKind.reset();});
                ImGui::SameLine(); ImGui::BeginDisabled(selected.empty() || gesture);
                if(ImGui::Button("Convex pieces (split)")) attempt([&]{
                    const auto count=asset.generateConvexDecomposition(selected,splitSettings(),system,body);partKind.reset();collisionSelection.clear();
                    messages.push_back("Generated "+std::to_string(count)+" convex piece(s) from the selected surface(s).");});
                ImGui::EndDisabled();
                if(ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
                    ImGui::SetTooltip("Approximate convex decomposition of the selected surfaces into several convex parts.");
                surfaceConvexityWarning(selected);
                ImGui::BeginDisabled(asset.movable());
                if(ImGui::Button("Triangle mesh from surfaces")) attempt([&]{asset.generateCollision(selected,false,system,body);partKind.reset();});
                ImGui::EndDisabled();
                if(ImGui::Button("Add box")) attempt([&]{
                    const auto inverseRoot=glm::inverse(selectedBodyTransform());
                    glm::vec3 low(std::numeric_limits<float>::max()), high(-std::numeric_limits<float>::max());
                    for(int corner=0;corner<8;++corner)
                    {
                        const auto point=glm::vec3(inverseRoot*glm::vec4(
                            corner&1?maximum.x:minimum.x,corner&2?maximum.y:minimum.y,corner&4?maximum.z:minimum.z,1));
                        low=glm::min(low,point);high=glm::max(high,point);
                    }
                    asset.addBox((low+high)*.5f,glm::max((high-low)*.5f,glm::vec3(1)),system,body);
                });
                ImGui::SameLine(); if(ImGui::Button("Add capsule")) attempt([&]{
                    const auto inverseRoot=glm::inverse(selectedBodyTransform());
                    auto c=(minimum+maximum)*0.5f; asset.addCapsule(
                        glm::vec3(inverseRoot*glm::vec4(c-glm::vec3(0,radius*.25f,0),1)),
                        glm::vec3(inverseRoot*glm::vec4(c+glm::vec3(0,radius*.25f,0),1)),radius*.15f,system,body);
                });
                const auto& systems=asset.collisionSystems();
                if(system<systems.size() && body<systems[system].physics.bodies.size())
                {
                    auto properties=asset.bodyProperties(system,body); bool changed=false;
                    if(systems[system].physics.bodies[body].dynamic) changed|=drag("Mass (kg)",&properties.mass,.1f,.001f,1000000);
                    changed|=drag("Friction",&properties.friction,.01f,0,10);
                    changed|=drag("Restitution",&properties.restitution,.01f,0,1);
                    changed|=drag("Linear damping",&properties.linearDamping,.005f,0,10);
                    changed|=drag("Angular damping",&properties.angularDamping,.005f,0,10);
                    changed|=drag("Gravity",&properties.gravityFactor,.01f,0,10);
                    if(changed) attempt([&]{asset.setBodyProperties(system,body,properties);});
                    ImGui::TextWrapped("Changing mass or collision geometry recalculates center of mass and inertia using uniform density.");
                    if(partKind)
                    {
                        const auto& rigid=asset.collisionSystems()[system].physics.bodies[body];
                        if(*partKind==AssetCollisionPart::capsule && part<rigid.capsules.size())
                        {
                            const auto shape=rigid.capsules[part].capsule;
                            float shapeRadius=shape.radius*assets::bethesdaUnitsPerMeter;
                            float length=glm::length(shape.pointB-shape.pointA)*assets::bethesdaUnitsPerMeter;
                            bool resize=drag("Capsule radius",&shapeRadius,.5f,.01f,1e6f);
                            resize|=drag("Capsule segment",&length,.5f,.01f,1e6f);
                            if(resize) attempt([&]{
                                const auto center=(shape.pointA+shape.pointB)*.5f;
                                const auto axis=glm::length(shape.pointB-shape.pointA)>1e-6f?glm::normalize(shape.pointB-shape.pointA):glm::vec3(0,0,1);
                                asset.setCapsuleGeometry(part,
                                    assets::havokToEnginePosition(center)-assets::swapBethesdaAxes(axis)*length*.5f,
                                    assets::havokToEnginePosition(center)+assets::swapBethesdaAxes(axis)*length*.5f,
                                    shapeRadius,system,body);
                            });
                        }
                        if(*partKind==AssetCollisionPart::convex && part<rigid.parts.size()) convexPanel();
                        if (*partKind != AssetCollisionPart::triangleMesh) attempt([&]{
                            auto transform=asset.partTransform(*partKind,part,system,body); float p[3],r[3],s[3];
                            ImGuizmo::DecomposeMatrixToComponents(glm::value_ptr(transform),p,r,s);
                            bool edit=ImGui::DragFloat3("Part position",p,1); trackWidget();
                            edit|=ImGui::DragFloat3("Part rotation",r,.5f); trackWidget();
                            edit|=ImGui::DragFloat3("Part scale",s,.01f,.001f,1000); trackWidget();
                            if(edit) {ImGuizmo::RecomposeMatrixFromComponents(p,r,s,glm::value_ptr(transform));asset.setPartTransform(*partKind,part,transform,system,body);}
                        });
                        if(ImGui::Button("Delete part")) attempt([&]{asset.erasePart(*partKind,part,system,body);partKind.reset();});
                    }
                }
            }
            ImGui::EndDisabled();
            for(const auto& message:asset.diagnostics()) ImGui::TextWrapped("%s",message.c_str());
            for(const auto& message:messages) ImGui::TextWrapped("%s",message.c_str());
            if(finishGesture && gesture) { asset.endTransaction();gesture=false; }
            finishGesture=false;
            ImGui::End();
        }
        // ---------------- PROTOTYPE 0.1 mesh editing ----------------
        std::optional<std::size_t> editSurface() const
        {
            if(surfaces.empty() || *surfaces.begin()>=asset.sourceGeometry().surfaces.size()) return std::nullopt;
            return *surfaces.begin();
        }
        void setMeshMode(bool on)
        {
            if(gizmoDrag) return;
            meshMode=on && asset.meshEditable(); meshSelection.clear();
            if(meshMode) { partKind.reset(); collisionMode=false; }
        }
        void setCollisionMode(bool on)
        {
            if(gizmoDrag) return;
            collisionMode=on && asset.capabilities().collision;
            collisionSelection.clear(); collisionTopologyKey.reset();
            if(collisionMode) { meshMode=false; meshSelection.clear(); showCollision=true; }
            else partKind.reset();
        }
        void deleteSurfaces()
        {
            if(gesture) throw std::runtime_error("Finish the current edit before deleting surfaces");
            std::vector<std::size_t> doomed;
            for(auto i:surfaces) if(i<asset.materials().size()) doomed.push_back(i);
            if(doomed.empty()) throw std::runtime_error("Select a surface to delete");
            std::ranges::sort(doomed,std::greater<>());
            asset.beginTransaction();
            try { for(auto i:doomed) asset.eraseSurface(i); }
            catch(...) { asset.cancelTransaction(); throw; }
            asset.endTransaction();
            meshSelection.clear(); topologySurface.reset(); topologyGeometry.reset();
            surfaces.clear(); if(!asset.materials().empty()) surfaces.insert(std::min(doomed.back(),asset.materials().size()-1));
            messages.push_back("Deleted "+std::to_string(doomed.size())+" surface(s). Collision is unchanged; regenerate it if needed.");
        }
        std::optional<glm::vec3> projectEngine(glm::vec3 point) const
        {
            point.x=-point.x;
            const auto clip=projection()*view()*glm::vec4(point,1);
            if(clip.w<=.1f) return std::nullopt;
            return glm::vec3(viewportPosition.x+(clip.x/clip.w+1)*.5f*viewportSize.x,viewportPosition.y+(clip.y/clip.w+1)*.5f*viewportSize.y,clip.w);
        }
        // Screen-space pick of collision shapes: convex/triangle-mesh faces by inside test,
        // capsules by distance to their projected segment. Nearest (by depth) wins.
        void pickCollision(const ImGuiIO& io)
        {
            const glm::vec2 mouse{io.MousePos.x,io.MousePos.y};
            struct Hit { AssetCollisionPart kind; std::size_t part, system, body; float depth; };
            std::optional<Hit> best;
            const auto consider=[&](Hit hit){ if(!best || hit.depth<best->depth) best=hit; };
            const auto edge=[](glm::vec2 p,glm::vec2 q,glm::vec2 r){return (q.x-p.x)*(r.y-p.y)-(q.y-p.y)*(r.x-p.x);};
            const auto inside=[&](glm::vec2 a,glm::vec2 b,glm::vec2 c){
                const float d1=edge(a,b,mouse),d2=edge(b,c,mouse),d3=edge(c,a,mouse);
                return !((d1<0||d2<0||d3<0)&&(d1>0||d2>0||d3>0)); };
            const float focal=viewportSize.y*.5f/std::tan(glm::radians(55.0f)*.5f);
            const auto& systems=asset.collisionSystems();
            for(std::size_t s=0;s<systems.size();++s) for(std::size_t b=0;b<systems[s].physics.bodies.size();++b)
            {
                const auto& rigid=systems[s].physics.bodies[b]; const auto root=simulation*physics::havokBodyLocalTransform(rigid);
                const auto at=[&](const glm::mat4& transform,glm::vec3 v){
                    return projectEngine(glm::vec3(root*glm::vec4(assets::havokToEnginePosition(glm::vec3(transform*glm::vec4(v,1))),1))); };
                const auto faces=[&](const auto& vertices,const auto& indices,const glm::mat4& transform,AssetCollisionPart kind,std::size_t index){
                    for(std::size_t i=0;i+2<indices.size();i+=3)
                    {
                        const auto a=at(transform,vertices.at(indices[i])),bb=at(transform,vertices.at(indices[i+1])),c=at(transform,vertices.at(indices[i+2]));
                        if(a && bb && c && inside(glm::vec2(*a),glm::vec2(*bb),glm::vec2(*c)))
                            consider({kind,index,s,b,(a->z+bb->z+c->z)/3});
                    }
                };
                for(std::size_t i=0;i<rigid.parts.size();++i)
                    faces(rigid.parts[i].polytope.vertices,rigid.parts[i].polytope.indices,
                        rigid.parts[i].transform*glm::scale(glm::mat4(1),rigid.parts[i].scale),AssetCollisionPart::convex,i);
                if(rigid.triangleMesh) faces(rigid.triangleMesh->vertices,rigid.triangleMesh->indices,glm::mat4(1),AssetCollisionPart::triangleMesh,0);
                for(std::size_t i=0;i<rigid.capsules.size();++i)
                {
                    const auto& p=rigid.capsules[i]; const auto transform=p.transform*glm::scale(glm::mat4(1),p.scale);
                    const auto a=at(transform,p.capsule.pointA),c=at(transform,p.capsule.pointB);
                    if(!a || !c) continue;
                    const glm::vec2 pa(*a),pc(*c); const auto segment=pc-pa;
                    const float t=glm::dot(segment,segment)>1e-6f?std::clamp(glm::dot(mouse-pa,segment)/glm::dot(segment,segment),0.0f,1.0f):0.0f;
                    const float depth=glm::mix(a->z,c->z,t);
                    const float scale=std::max({p.scale.x,p.scale.y,p.scale.z});
                    const float pixels=std::max(8.0f,p.capsule.radius*scale*assets::bethesdaUnitsPerMeter*focal/std::max(depth,.1f));
                    if(glm::length(mouse-(pa+segment*t))<=pixels) consider({AssetCollisionPart::capsule,i,s,b,depth});
                }
            }
            if(!best) { partKind.reset(); return; }
            partKind=best->kind; part=best->part; system=best->system; body=best->body;
        }
        // ---------------- PROTOTYPE convex collision element editing ----------------
        bool convexSelected() const
        {
            if(!collisionMode || partKind!=AssetCollisionPart::convex) return false;
            const auto& systems=asset.collisionSystems();
            return system<systems.size() && body<systems[system].physics.bodies.size()
                && part<systems[system].physics.bodies[body].parts.size();
        }
        void setCollisionElement(std::optional<mesh::Element> next)
        { if(next!=collisionElement && !gizmoDrag) {collisionElement=next;collisionSelection.clear();} }
        physics::ConvexDecompositionSettings splitSettings() const
        {
            physics::ConvexDecompositionSettings settings;
            settings.maxPieces=static_cast<unsigned>(std::clamp(splitMaxPieces,1,64));
            settings.resolution=static_cast<unsigned>(std::clamp(splitResolution,8,96));
            settings.concavityTolerance=std::clamp(splitTolerancePercent,.1f,50.0f)/100.0f;
            return settings;
        }
        void refreshConvexity()
        {
            if(convexityRevision==asset.revision()) return;
            convexityRevision=asset.revision(); nonConvexParts.clear();
            const auto& systems=asset.collisionSystems();
            for(std::size_t s=0;s<systems.size();++s) for(std::size_t b=0;b<systems[s].physics.bodies.size();++b)
                for(std::size_t i=0;i<systems[s].physics.bodies[b].parts.size();++i)
                    if(!asset.convexPartIsConvex(i,s,b)) nonConvexParts.insert({s,b,i});
        }
        void refreshCollisionTopology()
        {
            if(!convexSelected()) { collisionTopologyKey.reset(); collisionSelection.clear(); return; }
            const std::array<std::size_t,3> key{system,body,part};
            const auto& rigid=asset.collisionSystems()[system].physics.bodies[body];
            const auto root=simulation*physics::havokBodyLocalTransform(rigid);
            if(collisionTopologyKey==key && collisionTopologyRevision==asset.revision() && collisionRoot==root) return;
            assets::ModelSurface surface;
            for(const auto p:asset.convexPartPositions(part,system,body))
            { render::SourceVertex vertex; vertex.position=glm::vec3(root*glm::vec4(p,1)); surface.vertices.push_back(vertex); }
            surface.indices=asset.convexPartIndices(part,system,body);
            auto next=mesh::buildTopology(surface);
            const auto kind=collisionElement.value_or(mesh::Element::vertex);
            if(collisionTopologyKey!=key || mesh::elementCount(next,kind)!=mesh::elementCount(collisionTopology,kind)
                || next.vertexPoint.size()!=collisionTopology.vertexPoint.size()) collisionSelection.clear();
            collisionTopology=std::move(next); collisionTopologyKey=key; collisionTopologyRevision=asset.revision(); collisionRoot=root;
            const auto count=mesh::elementCount(collisionTopology,kind);
            std::erase_if(collisionSelection,[count](std::uint32_t e){return e>=count;});
        }
        void pickCollisionElement(const std::vector<mesh::ScreenPoint>& screen,const ImGuiIO& io)
        {
            const bool add=io.KeyShift||io.KeyCtrl;
            const auto hit=mesh::pickElement(collisionTopology,*collisionElement,screen,{io.MousePos.x,io.MousePos.y},8.0f);
            if(!add) collisionSelection.clear();
            if(hit) { if(add && collisionSelection.erase(*hit)) return; collisionSelection.insert(*hit); return; }
            // Missing every element of this part lets the click pick another shape.
            if(!add) pickCollision(io);
        }
        void collisionElementGizmo()
        {
            if(!collisionElement || collisionSelection.empty() || !convexSelected() || !collisionTopologyKey) return;
            glm::mat4 reflect(1);reflect[0][0]=-1;
            if(!gizmoDrag)
            {
                const std::vector<std::uint32_t> selection(collisionSelection.begin(),collisionSelection.end());
                collisionDragPoints=mesh::selectedPoints(collisionTopology,*collisionElement,selection);
                auto c=mesh::centroid(collisionTopology,collisionDragPoints); c.x=-c.x;
                collisionPivot=collisionStartPivot=glm::translate(glm::mat4(1),c);
                collisionStartRoot=collisionRoot;
                collisionStartPositions.clear();
                for(const auto p:asset.convexPartPositions(part,system,body)) collisionStartPositions.push_back(glm::vec3(collisionStartRoot*glm::vec4(p,1)));
            }
            auto v=view();auto p=glm::perspectiveRH_NO(glm::radians(55.0f),aspect,.1f,1e7f);
            ImGuizmo::Enable(true);
            ImGuizmo::SetOrthographic(false);ImGuizmo::SetDrawlist();ImGuizmo::SetRect(viewportPosition.x,viewportPosition.y,viewportSize.x,viewportSize.y);
            const bool changed=ImGuizmo::Manipulate(glm::value_ptr(v),glm::value_ptr(p),operation,
                operation==ImGuizmo::SCALE?ImGuizmo::LOCAL:ImGuizmo::WORLD,glm::value_ptr(collisionPivot));
            if(ImGuizmo::IsUsing() && !gizmoDrag) {asset.beginTransaction();gesture=gizmoDrag=true;}
            if(changed && gizmoDrag)
            {
                const auto delta=reflect*collisionPivot*glm::inverse(collisionStartPivot)*reflect;
                auto positions=mesh::transformPoints(collisionTopology,collisionStartPositions,collisionDragPoints,delta);
                const auto inverse=glm::inverse(collisionStartRoot);
                for(auto& position:positions) position=glm::vec3(inverse*glm::vec4(position,1));
                asset.setConvexPartPositions(part,positions,system,body);
            }
            if(gizmoDrag && !ImGuizmo::IsUsing()) {asset.endTransaction();gesture=gizmoDrag=false;}
        }
        void splitSettingsUi()
        {
            if(!ImGui::TreeNode("Split settings")) return;
            ImGui::SliderInt("Max pieces",&splitMaxPieces,1,64);
            ImGui::SliderFloat("Concavity tolerance %",&splitTolerancePercent,.1f,20.0f,"%.1f");
            ImGui::SliderInt("Voxel resolution",&splitResolution,16,96);
            ImGui::TextDisabled("Lower tolerance / higher resolution = tighter fit, more pieces, slower.");
            ImGui::TreePop();
        }
        void convexPanel()
        {
            const auto& polytope=asset.collisionSystems()[system].physics.bodies[body].parts[part].polytope;
            ImGui::SeparatorText("Convex part");
            ImGui::Text("Part %zu: %zu vertices, %zu triangles",part,polytope.vertices.size(),polytope.indices.size()/3);
            physics::ConvexityReport report;
            try { report=asset.convexPartConvexity(part,system,body); } catch(const std::exception&) { report.convex=asset.convexPartIsConvex(part,system,body); }
            if(report.convex) ImGui::TextColored({.45f,.9f,.55f,1},"Convex - exports to Fallout 4.");
            else
            {
                ImGui::PushStyleColor(ImGuiCol_Text,IM_COL32(255,95,80,255));
                ImGui::TextWrapped("WARNING: this part is NOT convex (deepest dent %.2f units). Fallout 4 only accepts convex "
                    "parts: the preview uses its hull and Save is blocked until you split or re-hull it.",report.depth);
                ImGui::PopStyleColor();
            }
            ImGui::BeginDisabled(gesture);
            if(ImGui::Button("Split optimally")) attempt([&]{
                const auto count=asset.splitConvexPart(part,splitSettings(),system,body);
                collisionSelection.clear(); collisionTopologyKey.reset();
                messages.push_back(count>1?"Split convex part "+std::to_string(part)+" into "+std::to_string(count)+" convex pieces (parts "
                    +std::to_string(part)+"-"+std::to_string(part+count-1)+").":"Part was already convex within tolerance; re-hulled as 1 piece.");
            });
            if(ImGui::IsItemHovered()) ImGui::SetTooltip("Approximate convex decomposition (voxel hierarchical cuts, V-HACD style).\nPieces keep this part's transform.");
            ImGui::SameLine();
            if(ImGui::Button("Re-hull")) attempt([&]{asset.rehullConvexPart(part,system,body);collisionSelection.clear();collisionTopologyKey.reset();});
            if(ImGui::IsItemHovered()) ImGui::SetTooltip("Wrap the part in the convex hull of its vertices (fills dents).");
            ImGui::EndDisabled();
            splitSettingsUi();
        }
        void surfaceConvexityWarning(const std::vector<std::size_t>& selected)
        {
            if(selected.empty() || asset.sourceGeometry().surfaces.empty()) return;
            const std::set<std::size_t> key(selected.begin(),selected.end());
            if(surfaceReportGeometry!=asset.geometryHandle() || surfaceReportSelection!=key)
            {
                surfaceReportGeometry=asset.geometryHandle(); surfaceReportSelection=key; surfaceReport.reset();
                try { surfaceReport=asset.surfaceConvexity(selected); } catch(const std::exception&) {}
            }
            if(surfaceReport && !surfaceReport->convex)
            {
                ImGui::PushStyleColor(ImGuiCol_Text,IM_COL32(255,170,60,255));
                ImGui::TextWrapped("Selected surface(s) are not convex (deepest dent %.1f units). \"Convex from surfaces\" fills the "
                    "concavities with one hull; \"Convex pieces (split)\" keeps them.",surfaceReport->depth);
                ImGui::PopStyleColor();
                splitSettingsUi();
            }
        }
        void deletePart()
        {
            if(!partKind || gesture) return;
            asset.erasePart(*partKind,part,system,body); partKind.reset();
        }
        void setElement(mesh::Element next)
        { if(next!=element && !gizmoDrag) {element=next;meshSelection.clear();} }
        void spawnCube()
        {
            if(gesture) throw std::runtime_error("Finish the current edit before adding a cube");
            glm::vec3 at=center; at.x=-at.x; // orbit target, back to engine space
            AssetMaterial material; material.baseColor={.72f,.72f,.75f,1};
            const auto index=asset.addSurface(mesh::makeCube(at,assets::bethesdaUnitsPerMeter),std::move(material));
            surfaces={index}; partKind.reset(); meshMode=true; collisionMode=false; meshSelection.clear();
        }
        void refreshTopology()
        {
            const auto s=editSurface(); const auto geometry=asset.geometryHandle();
            if(!s) { topologySurface.reset(); topologyGeometry.reset(); meshSelection.clear(); return; }
            if(topologySurface==s && topologyGeometry==geometry) return;
            const auto& surface=geometry->surfaces[*s];
            if(topologySurface==s && topologyGeometry && *s<topologyGeometry->surfaces.size()
                && topologyGeometry->surfaces[*s].vertices.size()==surface.vertices.size()
                && topologyGeometry->surfaces[*s].indices.size()==surface.indices.size())
                mesh::refreshPoints(topology,surface);   // positions moved; keep welding + selection
            else
            { topology=mesh::buildTopology(surface); topologySurface=s; meshSelection.clear(); }
            topologyGeometry=geometry;
            const auto count=mesh::elementCount(topology,element);
            std::erase_if(meshSelection,[count](std::uint32_t e){return e>=count;});
        }
        std::vector<mesh::ScreenPoint> projectPoints(const mesh::Topology& topo) const
        {
            std::vector<mesh::ScreenPoint> result(topo.points.size());
            const auto vp=projection()*view();
            for(std::size_t i=0;i<result.size();++i)
            {
                auto clip=vp*glm::vec4(-topo.points[i].x,topo.points[i].y,topo.points[i].z,1);
                auto& out=result[i]; out.depth=clip.w; out.visible=clip.w>.1f;
                if(!out.visible) continue;
                clip/=clip.w;
                out.position={viewportPosition.x+(clip.x+1)*.5f*viewportSize.x,viewportPosition.y+(clip.y+1)*.5f*viewportSize.y};
            }
            return result;
        }
        void meshOverlay(const mesh::Topology& topo,mesh::Element elem,const std::set<std::uint32_t>& selected,
            const std::vector<mesh::ScreenPoint>& screen)
        {
            auto* draw=ImGui::GetWindowDrawList();
            const auto at=[&](std::uint32_t p){return ImVec2(screen[p].position.x,screen[p].position.y);};
            const ImU32 wire=IM_COL32(190,215,255,120), hot=IM_COL32(255,165,40,255);
            const std::vector<std::uint32_t> selection(selected.begin(),selected.end());
            if(elem==mesh::Element::triangle) for(auto t:selection)
            {
                const auto& tri=topo.triangles[t];
                if(screen[tri.x].visible && screen[tri.y].visible && screen[tri.z].visible)
                    draw->AddTriangleFilled(at(tri.x),at(tri.y),at(tri.z),IM_COL32(255,165,40,90));
            }
            if(topo.edges.size()<=200000) for(const auto& e:topo.edges)
                if(screen[e.x].visible && screen[e.y].visible) draw->AddLine(at(e.x),at(e.y),wire,1.0f);
            std::vector<char> hotPoint(topo.points.size(),0);
            for(auto p:mesh::selectedPoints(topo,elem,selection)) hotPoint[p]=1;
            if(elem==mesh::Element::edge) for(auto e:selection)
            {
                const auto& edge=topo.edges[e];
                if(screen[edge.x].visible && screen[edge.y].visible) draw->AddLine(at(edge.x),at(edge.y),hot,3.0f);
            }
            if(elem==mesh::Element::triangle) for(auto t:selection) for(int k=0;k<3;++k)
            {
                const auto a=topo.triangles[t][k],b=topo.triangles[t][(k+1)%3];
                if(screen[a].visible && screen[b].visible) draw->AddLine(at(a),at(b),hot,2.0f);
            }
            if(elem==mesh::Element::vertex && topo.points.size()<=100000) for(std::uint32_t p=0;p<screen.size();++p)
            {
                if(!screen[p].visible) continue;
                const float r=hotPoint[p]?4.0f:2.5f;
                draw->AddRectFilled({at(p).x-r,at(p).y-r},{at(p).x+r,at(p).y+r},hotPoint[p]?hot:IM_COL32(20,30,45,230));
                if(!hotPoint[p]) draw->AddRect({at(p).x-r,at(p).y-r},{at(p).x+r,at(p).y+r},wire);
            }
        }
        void pickMesh(const std::vector<mesh::ScreenPoint>& screen,const ImGuiIO& io)
        {
            const bool add=io.KeyShift||io.KeyCtrl;
            const auto hit=mesh::pickElement(topology,element,screen,{io.MousePos.x,io.MousePos.y},8.0f);
            if(!add) meshSelection.clear();
            if(hit) { if(add && meshSelection.erase(*hit)) return; meshSelection.insert(*hit); return; }
            // Clicking empty space on another surface switches the edited surface.
            if(!add) if(auto picked=pickSurface(io); picked && *picked!=*surfaces.begin()) surfaces={*picked};
        }
        void meshGizmo()
        {
            const auto s=editSurface();
            if(!s || meshSelection.empty() || !asset.meshEditable()) return;
            glm::mat4 reflect(1);reflect[0][0]=-1;
            if(!gizmoDrag)
            {
                // Pivot at the selection centroid with world axes; captured as the drag origin.
                const std::vector<std::uint32_t> selection(meshSelection.begin(),meshSelection.end());
                meshDragPoints=mesh::selectedPoints(topology,element,selection);
                auto c=mesh::centroid(topology,meshDragPoints); c.x=-c.x;
                meshPivot=meshStartPivot=glm::translate(glm::mat4(1),c);
                meshStartPositions.clear();
                for(const auto& vertex:asset.sourceGeometry().surfaces[*s].vertices) meshStartPositions.push_back(vertex.position);
            }
            auto v=view();auto p=glm::perspectiveRH_NO(glm::radians(55.0f),aspect,.1f,1e7f);
            ImGuizmo::Enable(true);
            ImGuizmo::SetOrthographic(false);ImGuizmo::SetDrawlist();ImGuizmo::SetRect(viewportPosition.x,viewportPosition.y,viewportSize.x,viewportSize.y);
            const bool changed=ImGuizmo::Manipulate(glm::value_ptr(v),glm::value_ptr(p),operation,
                operation==ImGuizmo::SCALE?ImGuizmo::LOCAL:ImGuizmo::WORLD,glm::value_ptr(meshPivot));
            if(ImGuizmo::IsUsing() && !gizmoDrag) {asset.beginTransaction();gesture=gizmoDrag=true;}
            if(changed && gizmoDrag)
            {
                // Delta measured from the captured start, mapped back from render to engine space.
                const auto delta=reflect*meshPivot*glm::inverse(meshStartPivot)*reflect;
                asset.setSurfacePositions(*s,mesh::transformPoints(topology,meshStartPositions,meshDragPoints,delta));
            }
            if(gizmoDrag && !ImGuizmo::IsUsing()) {asset.endTransaction();gesture=gizmoDrag=false;}
        }
        void meshPanel()
        {
            if(!ImGui::CollapsingHeader("Mesh (prototype 0.1)",ImGuiTreeNodeFlags_DefaultOpen)) return;
            if(!asset.meshEditable())
            {
                ImGui::TextWrapped("Geometry is read-only for this asset%s%s",
                    asset.capabilities().reason.empty()?"":": ",asset.capabilities().reason.c_str());
                return;
            }
            ImGui::BeginDisabled(playing || gesture);
            if(ImGui::Button("Add cube")) attempt([&]{spawnCube();});
            ImGui::EndDisabled();
            ImGui::SameLine(); ImGui::TextDisabled("1 m cube at the view target");
            ImGui::BeginDisabled(playing || gesture || surfaces.empty() || asset.materials().empty());
            if(ImGui::Button(surfaces.size()>1?"Delete surfaces":"Delete surface")) attempt([&]{deleteSurfaces();});
            ImGui::EndDisabled();
            if(ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
                ImGui::SetTooltip("Removes the selected surface(s) and their materials (Delete key outside mesh/collision edit).\nCollision shapes are not changed.");
            bool on=meshMode;
            ImGui::BeginDisabled(playing);
            if(ImGui::Checkbox("Edit mesh (Tab)",&on)) setMeshMode(on);
            ImGui::EndDisabled();
            if(!meshMode) return;
            int e=static_cast<int>(element);
            ImGui::RadioButton("Vertex (1)",&e,0); ImGui::SameLine();
            ImGui::RadioButton("Edge (2)",&e,1); ImGui::SameLine();
            ImGui::RadioButton("Triangle (3)",&e,2);
            setElement(static_cast<mesh::Element>(e));
            int op=operation==ImGuizmo::TRANSLATE?0:operation==ImGuizmo::ROTATE?1:2;
            ImGui::RadioButton("Move (W)",&op,0); ImGui::SameLine();
            ImGui::RadioButton("Rotate (E)",&op,1); ImGui::SameLine();
            ImGui::RadioButton("Scale (R)",&op,2);
            operation=op==0?ImGuizmo::TRANSLATE:op==1?ImGuizmo::ROTATE:ImGuizmo::SCALE;
            if(const auto s=editSurface(); s && topologySurface==s)
                ImGui::Text("Surface %zu: %zu points, %zu edges, %zu triangles, %zu selected",*s,
                    topology.points.size(),topology.edges.size(),topology.triangles.size(),meshSelection.size());
            else ImGui::TextUnformatted("Select a surface in the Outliner or viewport.");
            if(ImGui::Button("Clear selection")) meshSelection.clear();
            ImGui::TextDisabled("LMB select | Shift/Ctrl+LMB toggle | Esc cancels a drag");
            ImGui::TextDisabled("Coincident vertices are welded; selection is x-ray (no occlusion).");
        }
        std::optional<std::size_t> pickSurface(const ImGuiIO& io) const
        {
            const glm::vec2 ndc{(io.MousePos.x-viewportPosition.x)/viewportSize.x*2-1,(io.MousePos.y-viewportPosition.y)/viewportSize.y*2-1};
            const auto inv=glm::inverse(projection()*view());auto near=inv*glm::vec4(ndc,1,1);near/=near.w;
            auto origin=camera();auto direction=glm::normalize(glm::vec3(near)-origin);origin.x=-origin.x;direction.x=-direction.x;
            double closest=std::numeric_limits<double>::infinity();std::optional<std::size_t> picked;
            for(std::size_t i=0;i<picking.size();++i) if(auto hit=picking[i].raycast(origin,direction,simulation,closest)) {closest=*hit;picked=i;}
            return picked;
        }
        glm::mat4 selectedBodyTransform() const
        {
            const auto& systems=asset.collisionSystems();
            if(system>=systems.size() || body>=systems[system].physics.bodies.size()) return glm::mat4(1);
            return physics::havokBodyLocalTransform(systems[system].physics.bodies[body]);
        }
        void line(glm::vec3 a,glm::vec3 b,ImU32 color)
        {
            a.x=-a.x;b.x=-b.x;
            const auto vp=projection()*view(); auto pa=vp*glm::vec4(a,1),pb=vp*glm::vec4(b,1);
            if(pa.w<=.1f || pb.w<=.1f) return;
            pa/=pa.w;pb/=pb.w;
            ImGui::GetWindowDrawList()->AddLine({viewportPosition.x+(pa.x+1)*.5f*viewportSize.x,viewportPosition.y+(pa.y+1)*.5f*viewportSize.y},
                {viewportPosition.x+(pb.x+1)*.5f*viewportSize.x,viewportPosition.y+(pb.y+1)*.5f*viewportSize.y},color,1.5f);
        }
        void collisionOverlay()
        {
            if(!showCollision) return;
            const auto& systems=asset.collisionSystems();
            for(std::size_t s=0;s<systems.size();++s) for(std::size_t b=0;b<systems[s].physics.bodies.size();++b)
            {
                const auto& rigid=systems[s].physics.bodies[b]; const auto root=simulation*physics::havokBodyLocalTransform(rigid);
                const auto drawTriangles=[&](const auto& vertices,const auto& indices,const glm::mat4& transform,ImU32 color){
                    for(std::size_t i=0;i+2<indices.size();i+=3) for(int edge=0;edge<3;++edge)
                        line(glm::vec3(root*glm::vec4(assets::havokToEnginePosition(glm::vec3(transform*glm::vec4(vertices.at(indices[i+edge]),1))),1)),
                            glm::vec3(root*glm::vec4(assets::havokToEnginePosition(glm::vec3(transform*glm::vec4(vertices.at(indices[i+(edge+1)%3]),1))),1)),color);
                };
                for(std::size_t i=0;i<rigid.parts.size();++i)
                {
                    const auto& p=rigid.parts[i]; drawTriangles(p.polytope.vertices,p.polytope.indices,p.transform*glm::scale(glm::mat4(1),p.scale),
                        partKind==AssetCollisionPart::convex && part==i && system==s && body==b?IM_COL32(255,210,45,230):IM_COL32(70,230,190,200));
                }
                if(rigid.triangleMesh) drawTriangles(rigid.triangleMesh->vertices,rigid.triangleMesh->indices,glm::mat4(1),IM_COL32(70,230,190,170));
                for(const auto& p:rigid.capsules)
                {
                    const auto transform=p.transform*glm::scale(glm::mat4(1),p.scale);
                    const auto point=[&](glm::vec3 value){return glm::vec3(root*glm::vec4(assets::havokToEnginePosition(glm::vec3(transform*glm::vec4(value,1))),1));};
                    auto axis=p.capsule.pointB-p.capsule.pointA;
                    axis=glm::length(axis)>1e-6f?glm::normalize(axis):glm::vec3(0,0,1);
                    const auto u=glm::normalize(glm::cross(axis,std::abs(axis.z)<.9f?glm::vec3(0,0,1):glm::vec3(0,1,0)));
                    const auto v=glm::cross(axis,u); const auto r=p.capsule.radius;
                    for(int n=0;n<24;++n)
                    {
                        const float a=n*glm::two_pi<float>()/24,next=(n+1)*glm::two_pi<float>()/24;
                        const auto da=r*(std::cos(a)*u+std::sin(a)*v),db=r*(std::cos(next)*u+std::sin(next)*v);
                        for(const auto& c:{p.capsule.pointA,p.capsule.pointB}) line(point(c+da),point(c+db),IM_COL32(70,230,190,220));
                        if(n%6==0) line(point(p.capsule.pointA+da),point(p.capsule.pointB+da),IM_COL32(70,230,190,220));
                    }
                    for(const auto tangent:{u,v}) for(int n=0;n<12;++n)
                    {
                        const float a=n*glm::pi<float>()/12,next=(n+1)*glm::pi<float>()/12;
                        for(int end=0;end<2;++end)
                        {
                            const auto c=end?p.capsule.pointB:p.capsule.pointA;
                            const auto direction=end?axis:-axis;
                            line(point(c+r*(std::cos(a)*tangent+std::sin(a)*direction)),
                                point(c+r*(std::cos(next)*tangent+std::sin(next)*direction)),IM_COL32(70,230,190,220));
                        }
                    }
                }
            }
        }
        void viewport()
        {
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding,{0,0});
            ImGui::Begin("Viewport",nullptr,ImGuiWindowFlags_NoBackground|ImGuiWindowFlags_NoScrollbar|ImGuiWindowFlags_NoScrollWithMouse);
            viewportPosition=ImGui::GetCursorScreenPos();viewportSize=ImGui::GetContentRegionAvail();
            viewportSize.x=std::max(viewportSize.x,1.0f);viewportSize.y=std::max(viewportSize.y,1.0f);aspect=viewportSize.x/viewportSize.y;
            ImGui::Dummy(viewportSize); const bool hovered=ImGui::IsItemHovered();
            const auto& io=ImGui::GetIO();
            if(hovered && !gizmoDrag && !io.WantTextInput)
            {
                if(ImGui::IsMouseDragging(ImGuiMouseButton_Right)) {yaw-=io.MouseDelta.x*.008f;pitch=std::clamp(pitch+io.MouseDelta.y*.008f,-1.5f,1.5f);}
                if(ImGui::IsMouseDragging(ImGuiMouseButton_Middle))
                {
                    const auto inv=glm::inverse(view());center+=glm::vec3(inv[0])*(-io.MouseDelta.x*distance*.002f)+glm::vec3(inv[1])*(io.MouseDelta.y*distance*.002f);
                }
                distance=std::clamp(distance*std::pow(.85f,io.MouseWheel),1.0f,1e7f);
                if(ImGui::IsKeyPressed(ImGuiKey_F)) {center=(minimum+maximum)*.5f;center.x=-center.x;distance=radius*3;}
            }
            const bool editingCollision=collisionMode && !playing && asset.capabilities().collision;
            if(editingCollision && !showCollision) showCollision=true;
            collisionOverlay();
            std::vector<mesh::ScreenPoint> screen;
            const bool editingMesh=meshMode && !playing && asset.meshEditable();
            if(editingMesh)
            {
                attempt([&]{refreshTopology();});
                if(topologySurface) { screen=projectPoints(topology); meshOverlay(topology,element,meshSelection,screen); attempt([&]{meshGizmo();}); }
            }
            const bool editingConvexElements=editingCollision && collisionElement && convexSelected();
            if(editingConvexElements)
            {
                attempt([&]{refreshCollisionTopology();});
                if(collisionTopologyKey)
                {
                    screen=projectPoints(collisionTopology);
                    meshOverlay(collisionTopology,*collisionElement,collisionSelection,screen);
                    attempt([&]{collisionElementGizmo();});
                }
            }
            if(!meshMode && !editingConvexElements && partKind && *partKind != AssetCollisionPart::triangleMesh && asset.capabilities().collision && !playing)
                attempt([&]{
                    glm::mat4 reflect(1);reflect[0][0]=-1;
                    const auto& rigid=asset.collisionSystems().at(system).physics.bodies.at(body);
                    const auto root=physics::havokBodyLocalTransform(rigid);
                    auto partMatrix=reflect*root*asset.partTransform(*partKind,part,system,body)*reflect;
                    auto v=view();auto p=glm::perspectiveRH_NO(glm::radians(55.0f),aspect,.1f,1e7f);
                    ImGuizmo::Enable(true);
                    ImGuizmo::SetOrthographic(false);ImGuizmo::SetDrawlist();ImGuizmo::SetRect(viewportPosition.x,viewportPosition.y,viewportSize.x,viewportSize.y);
                    const bool changed=ImGuizmo::Manipulate(glm::value_ptr(v),glm::value_ptr(p),operation,ImGuizmo::LOCAL,glm::value_ptr(partMatrix));
                    if(ImGuizmo::IsUsing() && !gizmoDrag) {asset.beginTransaction();gesture=gizmoDrag=true;}
                    if(changed) asset.setPartTransform(*partKind,part,glm::inverse(root)*reflect*partMatrix*reflect,system,body);
                    if(gizmoDrag && !ImGuizmo::IsUsing()) {asset.endTransaction();gesture=gizmoDrag=false;}
                });
            if(hovered && !gizmoDrag && !ImGuizmo::IsOver() && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
            {
                if(editingCollision)
                {
                    if(editingConvexElements && collisionTopologyKey) pickCollisionElement(screen,io);
                    else pickCollision(io);
                }
                else if(editingMesh && topologySurface) pickMesh(screen,io);
                else if(!editingMesh) if(auto picked=pickSurface(io)) {if(!io.KeyCtrl) surfaces.clear();surfaces.insert(*picked);partKind.reset();}
            }
            ImGui::GetWindowDrawList()->AddText({viewportPosition.x+12,viewportPosition.y+10},IM_COL32(230,235,240,255),editingCollision
                ?(editingConvexElements?"Convex element edit | LMB select | Shift+LMB toggle | 1/2/3 vertex/edge/triangle | 0 whole shape | W/E/R move/rotate/scale | C exit"
                    :"Collision edit (prototype) | LMB pick shape | 1/2/3 edit convex vertices/edges/triangles | W/E/R move/rotate/scale | Delete remove shape | C exit")
                :editingMesh
                ?"Mesh edit (prototype) | LMB select | Shift+LMB toggle | 1/2/3 vertex/edge/triangle | W/E/R move/rotate/scale | Tab exit"
                :"Asset | RMB orbit | MMB pan | Wheel zoom | F frame | W/E/R gizmo | Tab mesh edit | C collision edit | Del delete surface");
            ImGui::End();ImGui::PopStyleVar();
        }
    };

    AssetWorkspace::AssetWorkspace(EditorDocument& world,std::uint32_t model,const std::filesystem::path& data)
        :mImpl(std::make_unique<Impl>(world,model,data)){}
    AssetWorkspace::AssetWorkspace(EditorDocument& world,const std::filesystem::path& data)
        :mImpl(std::make_unique<Impl>(world,data)){}
    AssetWorkspace::~AssetWorkspace()=default;
    void AssetWorkspace::draw(render::VulkanRenderer& renderer,float delta,ImVec2& position,ImVec2& size)
    {
        mImpl->outline();mImpl->inspector();mImpl->attempt([&]{mImpl->update(renderer,delta);});mImpl->viewport();
        position=mImpl->viewportPosition;size=mImpl->viewportSize;
        if(!ImGui::GetIO().WantTextInput)
        {
            if(ImGui::IsKeyPressed(ImGuiKey_Escape)) cancelGesture();
            if(!mImpl->gesture)
            {
                if(ImGui::GetIO().KeyCtrl)
                {
                    if(ImGui::IsKeyPressed(ImGuiKey_S)) mImpl->attempt([&]{save();});
                    if(ImGui::IsKeyPressed(ImGuiKey_Z)) history(false);
                    if(ImGui::IsKeyPressed(ImGuiKey_Y)) history(true);
                }
                else
                {
                    if(ImGui::IsKeyPressed(ImGuiKey_W)) mImpl->operation=ImGuizmo::TRANSLATE;
                    if(ImGui::IsKeyPressed(ImGuiKey_E)) mImpl->operation=ImGuizmo::ROTATE;
                    if(ImGui::IsKeyPressed(ImGuiKey_R)) mImpl->operation=ImGuizmo::SCALE;
                    if(ImGui::IsKeyPressed(ImGuiKey_Tab,false) && mImpl->asset.meshEditable() && !mImpl->playing) mImpl->setMeshMode(!mImpl->meshMode);
                    if(ImGui::IsKeyPressed(ImGuiKey_C,false) && mImpl->asset.capabilities().collision && !mImpl->playing) mImpl->setCollisionMode(!mImpl->collisionMode);
                    if(ImGui::IsKeyPressed(ImGuiKey_Delete,false) && !mImpl->playing)
                    {
                        if(mImpl->collisionMode)
                        {
                            if(mImpl->collisionElement && !mImpl->collisionSelection.empty())
                                mImpl->messages.push_back("Delete removes whole shapes; press 0 (Shape) first. Vertex deletion is not supported yet.");
                            else mImpl->attempt([&]{mImpl->deletePart();});
                        }
                        else if(!mImpl->meshMode && mImpl->asset.meshEditable() && !mImpl->surfaces.empty()) mImpl->attempt([&]{mImpl->deleteSurfaces();});
                    }
                    if(mImpl->collisionMode && !mImpl->meshMode)
                    {
                        if(ImGui::IsKeyPressed(ImGuiKey_0,false)) mImpl->setCollisionElement(std::nullopt);
                        if(ImGui::IsKeyPressed(ImGuiKey_1,false)) mImpl->setCollisionElement(mesh::Element::vertex);
                        if(ImGui::IsKeyPressed(ImGuiKey_2,false)) mImpl->setCollisionElement(mesh::Element::edge);
                        if(ImGui::IsKeyPressed(ImGuiKey_3,false)) mImpl->setCollisionElement(mesh::Element::triangle);
                    }
                    if(mImpl->meshMode)
                    {
                        if(ImGui::IsKeyPressed(ImGuiKey_1,false)) mImpl->setElement(mesh::Element::vertex);
                        if(ImGui::IsKeyPressed(ImGuiKey_2,false)) mImpl->setElement(mesh::Element::edge);
                        if(ImGui::IsKeyPressed(ImGuiKey_3,false)) mImpl->setElement(mesh::Element::triangle);
                    }
                }
            }
        }
    }
    void AssetWorkspace::suspend(render::VulkanRenderer& renderer)
    {cancelGesture();mImpl->reset();renderer.endPreview();mImpl->installed=false;}
    void AssetWorkspace::save(){mImpl->save();}
    void AssetWorkspace::makeUnique(std::uint32_t reference)
    {
        const auto package=mImpl->asset.saveNative(mImpl->world.path(),mImpl->world.nextFormId());
        mImpl->model=mImpl->world.makeUnique(reference,package.modelPath,mImpl->recordKind());
        mImpl->baseKind=mImpl->world.model(mImpl->model)->kind;
        mImpl->asset.markSaved();
    }
    void AssetWorkspace::history(bool redo)
    {
        cancelGesture();if(redo)(void)mImpl->asset.redo();else(void)mImpl->asset.undo();mImpl->partKind.reset();mImpl->reset();
        mImpl->collisionSelection.clear();mImpl->collisionTopologyKey.reset();
        std::erase_if(mImpl->surfaces,[&](std::size_t i){return i>=mImpl->asset.materials().size();});
    }
    void AssetWorkspace::cancelGesture()
    {if(mImpl->gesture)mImpl->asset.cancelTransaction();mImpl->gesture=mImpl->gizmoDrag=false;}
    bool AssetWorkspace::dirty()const{return mImpl->asset.dirty();}
    bool AssetWorkspace::renderable()const{return mImpl->installed;}
    glm::mat4 AssetWorkspace::view()const{return mImpl->view();}
    glm::mat4 AssetWorkspace::projection()const{return mImpl->projection();}
    glm::vec3 AssetWorkspace::cameraPosition()const{return mImpl->camera();}
    AssetDocument& AssetWorkspace::document(){return mImpl->asset;}
    std::uint32_t AssetWorkspace::modelId()const{return mImpl->model;}
    void AssetWorkspace::smokeEdit()
    {
        auto& a=mImpl->asset;a.beginTransaction();a.generateCollision(std::array<std::size_t,1>{0},true);a.setMovable(true,5);
        auto material=a.materials()[0];material.baseColor={.2f,.45f,.8f,1};a.setMaterial(0,material);a.endTransaction();
        if(!a.undo() || !a.redo() || !a.dirty())throw std::runtime_error("Asset smoke undo/redo failed");
        mImpl->playRequested=true;
    }
    void AssetWorkspace::smokeVerifyPhysics()
    {
        auto& preview=*mImpl;
        if(!preview.physics || !preview.playing) throw std::runtime_error("Asset smoke physics did not start");
        const auto revision=preview.asset.revision();const bool dirty=preview.asset.dirty();
        const auto initial=preview.bodyLocal[3].y;
        for(int i=0;i<120;++i) preview.physics->step(1.0f/60);
        const auto position=preview.physics->bodyTransform(preview.previewBody)[3];
        if(!std::isfinite(position.y) || position.y>=initial-.1f || position.y<preview.minimum.y-preview.radius*2)
            throw std::runtime_error("Asset smoke body did not fall onto preview ground");
        preview.reset();
        if(preview.simulation!=glm::mat4(1) || preview.asset.revision()!=revision || preview.asset.dirty()!=dirty)
            throw std::runtime_error("Asset preview reset changed authored state");
    }
}
