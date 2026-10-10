#include "editor/asset_workspace.hpp"
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
            installed=true; previewRevision=asset.revision();
        }
        void update(render::VulkanRenderer& renderer,float delta)
        {
            const bool startRequested = std::exchange(playRequested, false);
            if (!installed || texturesChanged()) { reset(); rebuild(renderer); }
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
            if(!partKind) for(auto i:surfaces) selected.push_back(firstInstance+static_cast<std::uint32_t>(i));
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
                { if(!ImGui::GetIO().KeyCtrl) surfaces.clear(); if(!surfaces.erase(i)) surfaces.insert(i); partKind.reset(); }
            }
            ImGui::SeparatorText("Collision bodies");
            const auto& systems=asset.collisionSystems();
            for(std::size_t s=0;s<systems.size();++s) for(std::size_t b=0;b<systems[s].physics.bodies.size();++b)
            {
                ImGui::PushID(static_cast<int>(s)); ImGui::PushID(static_cast<int>(b));
                if(ImGui::Selectable(("Body "+std::to_string(s)+":"+std::to_string(b)).c_str(),s==system && b==body && !partKind))
                {system=s;body=b;partKind.reset();}
                const auto& rigid=systems[s].physics.bodies[b];
                const auto row=[&](AssetCollisionPart kind,std::size_t index,const char* name){
                    if(ImGui::Selectable((std::string("  ")+name+" "+std::to_string(index)).c_str(),partKind==kind && part==index && system==s && body==b))
                    {partKind=kind;part=index;system=s;body=b;}
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
            ImGui::TextWrapped("%s%s",asset.modelPath().c_str(),asset.dirty()?" *":"");
            const auto& cap=asset.capabilities();
            if(!cap.reason.empty()) ImGui::TextWrapped("%s",cap.reason.c_str());
            ImGui::BeginDisabled(!cap.materials && !cap.collision);
            if(ImGui::Button("Save Native NIF")) attempt([&]{save();});
            ImGui::EndDisabled();
            ImGui::Checkbox("Show collision",&showCollision);
            if(ImGui::Button(playing?"Pause":"Play")) {if(playing) playing=false; else playRequested=true;}
            ImGui::SameLine(); if(ImGui::Button("Reset")) reset();
            ImGui::TextDisabled("Simulation is never saved.");
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
                bool movable=recordKind()==assets::ESMModelKind::movableStatic; ImGui::BeginDisabled(!cap.motion);
                if(ImGui::Checkbox("Movable prop (MSTT)",&movable)) attempt([&]{asset.setMovable(movable);});
                ImGui::EndDisabled();
                const std::vector<std::size_t> selected(surfaces.begin(),surfaces.end());
                if(ImGui::Button("Convex from surfaces")) attempt([&]{asset.generateCollision(selected,true,system,body);partKind.reset();});
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
            collisionOverlay();
            if(partKind && *partKind != AssetCollisionPart::triangleMesh && asset.capabilities().collision && !playing)
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
            if(hovered && !ImGuizmo::IsOver() && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
            {
                const glm::vec2 ndc{(io.MousePos.x-viewportPosition.x)/viewportSize.x*2-1,(io.MousePos.y-viewportPosition.y)/viewportSize.y*2-1};
                const auto inv=glm::inverse(projection()*view());auto near=inv*glm::vec4(ndc,1,1);near/=near.w;
                auto origin=camera();auto direction=glm::normalize(glm::vec3(near)-origin);origin.x=-origin.x;direction.x=-direction.x;
                double closest=std::numeric_limits<double>::infinity();std::optional<std::size_t> picked;
                for(std::size_t i=0;i<picking.size();++i) if(auto hit=picking[i].raycast(origin,direction,simulation,closest)) {closest=*hit;picked=i;}
                if(picked) {if(!io.KeyCtrl) surfaces.clear();surfaces.insert(*picked);partKind.reset();}
            }
            ImGui::GetWindowDrawList()->AddText({viewportPosition.x+12,viewportPosition.y+10},IM_COL32(230,235,240,255),"Asset | RMB orbit | MMB pan | Wheel zoom | F frame | W/E/R gizmo");
            ImGui::End();ImGui::PopStyleVar();
        }
    };

    AssetWorkspace::AssetWorkspace(EditorDocument& world,std::uint32_t model,const std::filesystem::path& data)
        :mImpl(std::make_unique<Impl>(world,model,data)){}
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
    {cancelGesture();if(redo)(void)mImpl->asset.redo();else(void)mImpl->asset.undo();mImpl->partKind.reset();mImpl->reset();}
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
