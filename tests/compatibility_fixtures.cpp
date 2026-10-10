#include "compatibility_fixtures.hpp"

#include "assets/asset_resolver.hpp"
#include "assets/esm_document.hpp"
#include "assets/esm_editor.hpp"
#include "editor/asset_document.hpp"

#include <NifFile.hpp>
#include <ExtraData.hpp>
#include <bhk.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

#include <glm/ext/matrix_transform.hpp>

namespace openf4::tests
{
    namespace
    {
        using namespace assets;
        using Bytes = std::vector<std::byte>;
        constexpr float units = 69.99125f;
        constexpr std::uint32_t firstModel = 0x01000900, firstReference = 0x01000a00;
        constexpr std::array<std::uint32_t, 36> triangles{
            0,2,1,0,3,2,4,5,6,4,6,7,0,1,5,0,5,4,
            1,2,6,1,6,5,2,3,7,2,7,6,3,0,4,3,4,7};

        void require(bool condition, const std::string& message)
        {
            if (!condition) throw std::runtime_error("Compatibility: " + message);
        }

        Bytes read(const std::filesystem::path& path)
        {
            std::ifstream file(path, std::ios::binary | std::ios::ate);
            require(bool(file) && file.tellg() >= 0, "cannot read " + path.string());
            Bytes result(static_cast<std::size_t>(file.tellg()));
            file.seekg(0);
            require(bool(file.read(reinterpret_cast<char*>(result.data()), result.size())), "truncated fixture");
            return result;
        }

        void write(const std::filesystem::path& path, std::span<const std::byte> bytes)
        {
            std::filesystem::create_directories(path.parent_path());
            std::ofstream file(path, std::ios::binary);
            require(bool(file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size()).flush()), "cannot write fixture");
        }

        std::string clean(std::string text)
        {
            for (char& c : text) if (c == '\t' || c == '\n' || c == '\r') c = ' ';
            return text;
        }

        struct Reports
        {
            std::filesystem::path root;
            std::ofstream cases, cycles;
            explicit Reports(const std::filesystem::path& path) : root(path),
                cases(path / "cases.tsv"), cycles(path / "cycles.tsv")
            {
                require(bool(cases) && bool(cycles), "cannot open reports");
                cases << "case\tprofile\tstatus\tdetail\n";
                cycles << "case\tgeneration\tstatus\tposition_error_units\tcom_error_m\ttensor_error_kg_m2\tmodel_path\n";
            }
            void row(std::string_view name, std::string_view profile, std::string_view status, const std::string& detail)
            {
                cases << name << '\t' << profile << '\t' << status << '\t' << clean(detail) << '\n'; cases.flush();
            }
            void summary(const std::string& status)
            {
                std::ofstream file(root / "summary.txt");
                file << status << "\nE0 is exported from explicit source data; E1..E10 open only the previous native package.\n"
                    "Generated profiles compare every generation with source and E0. No source geometry is regenerated during cycling.\n"
                    "Fallout 4 results require a new manual run; package generation is not an in-game compatibility claim.\n";
            }
        };

        struct Box
        {
            glm::vec3 center{}, half{};
            glm::quat orientation{1,0,0,0};
            float mass = 0;
        };
        struct Fixture
        {
            std::string name;
            ModelGeometry geometry;
            HavokPhysicsSystem physics;
            std::vector<Box> boxes;
            glm::vec3 expectedCom{};
            glm::mat3 expectedTensor{0};
            bool doorway = false;
        };

        // Oracle axes/units are explicit here, not calls into the importer/exporter conversion helpers.
        glm::vec3 engine(glm::vec3 point) { return {point.x * units, point.z * units, point.y * units}; }
        std::array<glm::vec3, 8> corners(glm::vec3 h)
        {
            return {{{-h.x,-h.y,-h.z},{h.x,-h.y,-h.z},{h.x,h.y,-h.z},{-h.x,h.y,-h.z},
                {-h.x,-h.y,h.z},{h.x,-h.y,h.z},{h.x,h.y,h.z},{-h.x,h.y,h.z}}};
        }
        glm::mat3 diagonal(glm::vec3 d)
        {
            glm::mat3 value(0); for (int i=0; i<3; ++i) value[i][i]=d[i]; return value;
        }
        glm::mat3 tensor(const HavokRigidBody& body)
        {
            const auto r = glm::mat3_cast(body.inertiaOrientation);
            return r * diagonal(body.inertiaDiagonal) * glm::transpose(r);
        }
        float tensorError(const glm::mat3& a, const glm::mat3& b)
        {
            float error = 0;
            for (int i=0; i<3; ++i) for (int j=0; j<3; ++j) error+=(a[i][j]-b[i][j])*(a[i][j]-b[i][j]);
            return std::sqrt(error);
        }

        void addBox(Fixture& fixture, Box box)
        {
            fixture.boxes.push_back(box);
            auto& body = fixture.physics.bodies[0];
            auto& part = body.parts.emplace_back();
            part.transform = glm::translate(glm::mat4(1),box.center) * glm::mat4_cast(box.orientation);
            const auto vertices = corners(box.half);
            part.polytope.vertices.assign(vertices.begin(),vertices.end());
            part.polytope.indices.assign(triangles.begin(),triangles.end());
            auto& surface = fixture.geometry.surfaces.emplace_back();
            surface.metallicFactor = 0; surface.roughnessFactor = 1;
            surface.baseColor={.25f,.5f,.75f,1};
            surface.doubleSided = fixture.boxes.size() % 2 != 0;
            surface.alphaCutoff = .25f;
            ModelImage image;
            image.width = image.height = 2; image.srgb = true; image.mipOffsets = {0};
            image.pixels = {std::byte{255},std::byte{0},std::byte{0},std::byte{255},
                std::byte{0},std::byte{255},std::byte{0},std::byte{255},
                std::byte{0},std::byte{0},std::byte{255},std::byte{255},
                std::byte{255},std::byte{255},std::byte{0},std::byte{255}};
            surface.diffuseImage=std::move(image);
            ModelImage normal;
            normal.width=normal.height=1; normal.mipOffsets={0};
            normal.pixels={std::byte{128},std::byte{128},std::byte{255},std::byte{255}};
            surface.normalImage=std::move(normal);
            for (std::size_t t=0; t<triangles.size(); t+=3)
            {
                std::array<glm::vec3,3> p;
                for (int j=0; j<3; ++j)
                    p[j] = engine(body.position + body.rotation * (box.center + box.orientation * vertices[triangles[t+j]]));
                // The engine mirrors X at draw time. Preserve Bethesda winding through the Y/Z reflection.
                const auto normal = -glm::normalize(glm::cross(p[1]-p[0],p[2]-p[0]));
                for (int j=0; j<3; ++j)
                {
                    render::SourceVertex v;
                    v.position=p[j]; v.normal=normal; v.uv={j==1 ? 1.f : 0.f,j==2 ? 1.f : 0.f};
                    surface.indices.push_back(static_cast<std::uint32_t>(surface.vertices.size()));
                    surface.vertices.push_back(v);
                }
            }
        }

        void massOracle(Fixture& fixture, glm::quat principalFrame)
        {
            auto& body=fixture.physics.bodies[0];
            body.mass=0;
            for (const auto& box : fixture.boxes) { body.mass+=box.mass; fixture.expectedCom+=box.center*box.mass; }
            fixture.expectedCom/=body.mass;
            for (const auto& box : fixture.boxes)
            {
                const auto h=box.half;
                const glm::vec3 local = box.mass/3.f * glm::vec3(h.y*h.y+h.z*h.z,h.x*h.x+h.z*h.z,h.x*h.x+h.y*h.y);
                const auto rotation=glm::mat3_cast(box.orientation);
                const auto offset=box.center-fixture.expectedCom;
                fixture.expectedTensor += rotation*diagonal(local)*glm::transpose(rotation)
                    + box.mass*(glm::dot(offset,offset)*glm::mat3(1)-glm::outerProduct(offset,offset));
            }
            const auto r=glm::mat3_cast(principalFrame);
            const auto principal=glm::transpose(r)*fixture.expectedTensor*r;
            body.centerOfMass=fixture.expectedCom;
            body.inertiaOrientation=principalFrame;
            body.inertiaDiagonal={principal[0][0],principal[1][1],principal[2][2]};
            require(tensorError(tensor(body),fixture.expectedTensor)<.00002f,"analytic compound frame is not principal");
        }

        std::vector<Fixture> sourceFixtures()
        {
            std::vector<Fixture> result(4);
            for (auto& fixture : result)
            {
                auto& body=fixture.physics.bodies.emplace_back();
                body.dynamic=true; body.friction=.713f; body.restitution=0;
                body.collisionFilterInfo=5; body.linearDamping=.12f; body.angularDamping=.2f;
            }
            auto& asymmetric=result[0]; asymmetric.name="asymmetric";
            auto& first=asymmetric.physics.bodies[0];
            first.position={.07f,-.11f,.13f}; first.rotation=glm::angleAxis(.19f,glm::normalize(glm::vec3(2,1,3)));
            const auto orientation=glm::angleAxis(.31f,glm::normalize(glm::vec3(1,3,2)));
            addBox(asymmetric,{{.23f,-.17f,.51f},{.41f,.63f,.87f},orientation,7.f});
            massOracle(asymmetric,orientation);

            auto& doorway=result[1]; doorway.name="doorway"; doorway.doorway=true;
            auto& door=doorway.physics.bodies[0]; door.dynamic=false; door.collisionFilterInfo=1;
            addBox(doorway,{{-1.2f,0,1.6f},{.7f,.25f,1.6f}});
            addBox(doorway,{{1.2f,0,1.6f},{.3f,.25f,1.6f}});
            addBox(doorway,{{.2f,0,2.8f},{.7f,.25f,.4f}});
            door.triangleMesh.emplace();
            for (const auto& box : doorway.boxes)
            {
                const auto base=static_cast<std::uint32_t>(door.triangleMesh->vertices.size());
                for (const auto p : corners(box.half)) door.triangleMesh->vertices.push_back(p+box.center);
                for (const auto i : triangles) door.triangleMesh->indices.push_back(base+i);
            }
            door.parts.clear();

            const auto frame=glm::angleAxis(.37f,glm::normalize(glm::vec3(1,2,3)));
            for (int i=2; i<4; ++i)
            {
                auto& compound=result[i]; compound.name=i==2 ? "unequal-compound" : "nonuniform-compound";
                const float a=i==2 ? 2.f : 7.f, b=i==2 ? 6.f : 1.f;
                addBox(compound,{frame*glm::vec3(-.7f,0,.5f),{.3f,.4f,.5f},frame,a});
                addBox(compound,{frame*glm::vec3(.5f,0,.5f),{.6f,.5f,.6f},frame,b});
                massOracle(compound,frame);
            }
            return result;
        }

        bool rayHits(const HavokTriangleMesh& mesh, glm::vec3 origin, glm::vec3 direction={0,1,0})
        {
            for (std::size_t i=0; i<mesh.indices.size(); i+=3)
            {
                const auto a=mesh.vertices[mesh.indices[i]], b=mesh.vertices[mesh.indices[i+1]], c=mesh.vertices[mesh.indices[i+2]];
                const auto e1=b-a,e2=c-a,p=glm::cross(direction,e2);
                const float d=glm::dot(e1,p);
                if (std::abs(d)<1e-7f) continue;
                const auto t=origin-a; const float u=glm::dot(t,p)/d;
                const auto q=glm::cross(t,e1); const float v=glm::dot(direction,q)/d, distance=glm::dot(e2,q)/d;
                if (u>=0 && v>=0 && u+v<=1 && distance>=0) return true;
            }
            return false;
        }

        std::vector<glm::vec3> collisionVertices(const HavokRigidBody& body)
        {
            std::vector<glm::vec3> result;
            for (const auto& part : body.parts)
                for (const auto p : part.polytope.vertices)
                    result.push_back(body.position+body.rotation*glm::vec3(part.transform*glm::vec4(p*part.scale,1)));
            if (body.triangleMesh)
                for (const auto p : body.triangleMesh->vertices) result.push_back(body.position+body.rotation*p);
            return result;
        }

        struct Errors { float position=0,com=0,inertia=0; };
        Errors verify(const Fixture& source, const ModelGeometry& geometry, const HavokPhysicsSystem& physics)
        {
            Errors errors;
            require(geometry.surfaces.size()==source.geometry.surfaces.size(),source.name+": surface count");
            for (std::size_t s=0; s<geometry.surfaces.size(); ++s)
            {
                const auto& a=source.geometry.surfaces[s]; const auto& b=geometry.surfaces[s];
                require(a.indices==b.indices && a.vertices.size()==b.vertices.size(),source.name+": winding/topology");
                require(a.doubleSided==b.doubleSided && a.alphaBlend==b.alphaBlend
                    && std::abs(a.alphaCutoff-b.alphaCutoff)<=1.f/255.f,source.name+": material flags");
                for (std::size_t v=0; v<a.vertices.size(); ++v)
                {
                    errors.position=std::max(errors.position,glm::length(a.vertices[v].position-b.vertices[v].position));
                    require(glm::length(a.vertices[v].normal-b.vertices[v].normal)<.022f,source.name+": normal orientation");
                    require(glm::length(a.vertices[v].uv-b.vertices[v].uv)<.002f,source.name+": UV orientation");
                }
            }
            require(errors.position<.002f,source.name+": render vertex axes/scale/position");
            require(physics.bodies.size()==1,source.name+": body count");
            const auto& expected=source.physics.bodies[0]; const auto& body=physics.bodies[0];
            require(body.bodyIndex==0 && body.dynamic==expected.dynamic && body.collisionFilterInfo==expected.collisionFilterInfo
                && body.materialId==expected.materialId,source.name+": identity/motion/filter/material");
            const auto nativeContact=[](float v) { return std::bit_cast<float>(std::bit_cast<std::uint32_t>(v)&0xffff0000u); };
            require(body.friction==nativeContact(expected.friction) && body.restitution==nativeContact(expected.restitution),source.name+": contact quantization");
            const double orientationError=2*std::acos(std::min(1.0,std::abs(glm::dot(
                glm::normalize(glm::dquat(body.rotation)),glm::normalize(glm::dquat(expected.rotation))))));
            require(glm::length(body.position-expected.position)<.000003f
                && orientationError<=glm::radians(.05),source.name+": body transform");
            if (body.dynamic)
            {
                errors.com=glm::length(body.centerOfMass-source.expectedCom);
                errors.inertia=tensorError(tensor(body),source.expectedTensor);
                require(std::abs(body.mass-expected.mass)<.00001f && errors.com<.00002f && errors.inertia<.00008f,
                    source.name+": independently calculated mass/COM/full inertia tensor");
                require(body.linearDamping==expected.linearDamping && body.angularDamping==expected.angularDamping
                    && body.gravityFactor==expected.gravityFactor && body.maxLinearVelocity==expected.maxLinearVelocity
                    && body.maxAngularVelocity==expected.maxAngularVelocity,source.name+": motion properties");
            }
            require(body.parts.size()==expected.parts.size() && body.capsules.empty()
                && bool(body.triangleMesh)==bool(expected.triangleMesh),source.name+": collision topology");
            const auto vertices=collisionVertices(body), originals=collisionVertices(expected);
            require(vertices.size()==originals.size(),source.name+": collision vertex count");
            const float tolerance=source.doorway ? .004f : .00003f;
            for (const auto v : originals)
                require(std::ranges::any_of(vertices,[&](glm::vec3 other) { return glm::length(v-other)<=tolerance; }),source.name+": collision shape/axes/units");
            if (source.doorway)
            {
                for (float side : {-1.f,1.f})
                {
                    const glm::vec3 direction(0,-side,0);
                    require(!rayHits(*body.triangleMesh,{.2f,side*2,1},direction)
                        && !rayHits(*body.triangleMesh,{.8f,side*2,2.2f},direction),"doorway opening sealed");
                    require(rayHits(*body.triangleMesh,{-1.2f,side*2,1},direction)
                        && rayHits(*body.triangleMesh,{1.2f,side*2,1},direction)
                        && rayHits(*body.triangleMesh,{.2f,side*2,2.8f},direction),"doorway supports missing");
                }
            }
            return errors;
        }

        void expectedFailure(const std::function<void()>& operation, const std::string& name)
        {
            try { operation(); } catch (const std::exception&) { return; }
            throw std::runtime_error("Compatibility: negative probe was accepted: "+name);
        }

        Bytes nonCollisionBytes(const std::filesystem::path& path)
        {
            auto bytes=read(path);
            nifly::NifFile file(path);
            const auto& header=file.GetHeader();
            std::size_t cursor=0;
            const auto skip=[&](std::size_t amount) {
                require(cursor<=bytes.size() && amount<=bytes.size()-cursor,"preservation NIF header bounds"); cursor+=amount;
            };
            const auto number=[&]<class T>() {
                const auto offset=cursor; skip(sizeof(T)); T value; std::memcpy(&value,bytes.data()+offset,sizeof(T)); return value;
            };
            while (cursor<bytes.size() && bytes[cursor]!=std::byte{'\n'}) skip(1);
            skip(1);
            require(number.operator()<std::uint32_t>()==0x14020007 && number.operator()<std::uint8_t>()==1
                && number.operator()<std::uint32_t>()==12,"preservation FO4 NIF header");
            const auto count=number.operator()<std::uint32_t>();
            require(count==header.GetNumBlocks() && number.operator()<std::uint32_t>()==130,"preservation NIF version/count");
            for (int i=0;i<4;++i) skip(number.operator()<std::uint8_t>());
            const auto types=number.operator()<std::uint16_t>();
            for (unsigned i=0;i<types;++i) skip(number.operator()<std::uint32_t>());
            skip(static_cast<std::size_t>(count)*2);
            const auto sizeTable=cursor;
            std::optional<std::pair<std::size_t,std::size_t>> payload;
            std::size_t sizeField=0;
            for (std::uint32_t i=0;i<header.GetNumBlocks();++i)
                if (const auto* physics=header.GetBlock<nifly::bhkPhysicsSystem>(i))
                {
                    require(!payload && !physics->data.empty(),"preservation fixture requires one nonempty collision block");
                    Bytes needle(4+physics->data.size());
                    const auto length=static_cast<std::uint32_t>(physics->data.size());
                    std::memcpy(needle.data(),&length,4); std::memcpy(needle.data()+4,physics->data.data(),length);
                    const auto begin=std::search(bytes.begin(),bytes.end(),needle.begin(),needle.end());
                    require(begin!=bytes.end() && std::search(begin+1,bytes.end(),needle.begin(),needle.end())==bytes.end(),"collision payload is not unique");
                    payload={{static_cast<std::size_t>(begin-bytes.begin()),needle.size()}};
                    sizeField=sizeTable+i*4;
                }
            require(payload && sizeField+4<=payload->first && payload->first+payload->second<=bytes.size(),"invalid preserved collision span");
            bytes.erase(bytes.begin()+payload->first,bytes.begin()+payload->first+payload->second);
            bytes.erase(bytes.begin()+sizeField,bytes.begin()+sizeField+4);
            return bytes;
        }

        void validateProofPlugin(const std::filesystem::path& path)
        {
            const auto bytes=read(path);
            const auto number=[&]<class T>(std::size_t offset) {
                require(offset<=bytes.size() && sizeof(T)<=bytes.size()-offset,"proof plugin number out of bounds");
                T value; std::memcpy(&value,bytes.data()+offset,sizeof(T)); return value;
            };
            const auto integer=[&](std::size_t offset) { return number.operator()<std::uint32_t>(offset); };
            const auto tag=[&](std::size_t offset) {
                require(offset<=bytes.size() && 4<=bytes.size()-offset,"proof plugin tag out of bounds");
                return std::string(reinterpret_cast<const char*>(bytes.data()+offset),4);
            };
            require(tag(0)=="TES4" && integer(8)==0 && integer(12)==0 && tag(24)=="HEDR"
                && integer(34)==6 && integer(38)==0x904,"proof plugin TES4 counts/IDs");
            std::size_t cursor=24+integer(4),count=0,groups=0;
            std::array<bool,4> seen{};
            while (cursor<bytes.size())
            {
                require(tag(cursor)=="GRUP" && integer(cursor+12)==0,"proof plugin top group");
                const auto type=tag(cursor+8); const auto end=cursor+integer(cursor+4);
                require((type=="STAT" || type=="MSTT") && end<=bytes.size() && end>=cursor+24,"proof plugin group bounds");
                cursor+=24; ++groups;
                while (cursor<end)
                {
                    const auto id=integer(cursor+12);
                    require(tag(cursor)==type && integer(cursor+8)==0 && id>=firstModel && id<firstModel+4
                        && !seen[id-firstModel],"proof plugin grouped unique FormID");
                    seen[id-firstModel]=true;
                    require(type==(id==firstModel+1 ? "STAT":"MSTT"),"proof plugin model kind");
                    const auto recordEnd=cursor+24+integer(cursor+4);
                    require(recordEnd<=end,"proof plugin record bounds"); cursor+=24;
                    bool edid=false,obnd=false,modl=false,data=false,full=type=="STAT";
                    while (cursor<recordEnd)
                    {
                        require(cursor+6<=recordEnd,"proof plugin subrecord header");
                        const auto name=tag(cursor); const auto size=number.operator()<std::uint16_t>(cursor+4); cursor+=6;
                        require(cursor+size<=recordEnd,"proof plugin subrecord bounds");
                        edid|=name=="EDID" && size>1; obnd|=name=="OBND" && size==12;
                        modl|=name=="MODL" && size>1 && bytes[cursor+size-1]==std::byte{};
                        full|=name=="FULL" && size>1;
                        data|=type=="STAT" ? name=="DNAM" && size==16 : name=="DATA" && size==1;
                        cursor+=size;
                    }
                    require(edid && obnd && modl && data && full,"proof plugin required fields"); ++count;
                }
            }
            require(cursor==bytes.size() && count==4 && groups==2,"proof plugin total counts");
        }

        // Minimal disposable plugin: base records only, no save-game or world overrides.
        void proofPlugin(const std::filesystem::path& path,const std::vector<Fixture>& fixtures,
            const std::vector<NativeModelPackage>& packages)
        {
            const auto number=[]<class T>(Bytes& b,T v) { const auto* p=reinterpret_cast<const std::byte*>(&v); b.insert(b.end(),p,p+sizeof(T)); };
            const auto tag=[](Bytes& b,std::string_view s) { for (const auto c:s) b.push_back(std::byte(c)); };
            const auto sub=[&](Bytes& b,std::string_view t,const Bytes& p) { tag(b,t); number(b,static_cast<std::uint16_t>(p.size())); b.insert(b.end(),p.begin(),p.end()); };
            const auto text=[&](Bytes& b,std::string_view t,std::string s) { Bytes p; for (auto c:s) p.push_back(std::byte(c)); p.push_back(std::byte{}); sub(b,t,p); };
            const auto record=[&](std::string_view t,std::uint32_t id,const Bytes& body) {
                Bytes b; tag(b,t); number(b,static_cast<std::uint32_t>(body.size())); number(b,0u); number(b,id);
                number(b,0u); number(b,std::uint16_t{131}); number(b,std::uint16_t{0}); b.insert(b.end(),body.begin(),body.end()); return b;
            };
            Bytes header,hedr; number(hedr,1.f); number(hedr,static_cast<std::uint32_t>(fixtures.size()+2)); number(hedr,0x904u);
            sub(header,"HEDR",hedr); text(header,"CNAM","OpenF4 compatibility harness");
            text(header,"SNAM","Disposable native compatibility fixtures; no world overrides");
            text(header,"MAST","Fallout4.esm"); sub(header,"DATA",Bytes(8)); auto output=record("TES4",0,header);
            for (const bool dynamic : {false,true})
            {
                Bytes children;
                const std::string type=dynamic ? "MSTT" : "STAT";
                for (std::size_t i=0;i<fixtures.size();++i)
                {
                    if (fixtures[i].physics.bodies[0].dynamic!=dynamic) continue;
                    const auto generation=path.stem().string().substr(std::string("Fallout4-").size());
                    Bytes body; const auto name="OF4Compat"+generation+"_"+std::to_string(i); text(body,"EDID",name);
                    glm::vec3 lo(std::numeric_limits<float>::max()),hi(-std::numeric_limits<float>::max());
                    for (const auto& s:fixtures[i].geometry.surfaces) for (const auto& v:s.vertices)
                    { lo=glm::min(lo,v.position); hi=glm::max(hi,v.position); }
                    Bytes bounds;
                    for (const auto axis : {0,2,1}) number(bounds,static_cast<std::int16_t>(std::floor(lo[axis])));
                    for (const auto axis : {0,2,1}) number(bounds,static_cast<std::int16_t>(std::ceil(hi[axis])));
                    sub(body,"OBND",bounds); if (dynamic) text(body,"FULL",name);
                    auto model=packages[i].modelPath; require(model.starts_with("meshes/"),"proof NIF namespace");
                    model=model.substr(7); std::ranges::replace(model,'/','\\'); text(body,"MODL",model);
                    if (dynamic) sub(body,"DATA",Bytes{std::byte{1}});
                    else { Bytes d; number(d,90.f); number(d,0u); number(d,0.f); number(d,0.f); sub(body,"DNAM",d); }
                    const auto bytes=record(type,firstModel+static_cast<std::uint32_t>(i),body);
                    children.insert(children.end(),bytes.begin(),bytes.end());
                }
                Bytes group; tag(group,"GRUP"); number(group,static_cast<std::uint32_t>(children.size()+24));
                tag(group,type); number(group,0u); number(group,std::uint64_t{0}); group.insert(group.end(),children.begin(),children.end());
                output.insert(output.end(),group.begin(),group.end());
            }
            write(path,output);
            validateProofPlugin(path);
        }

        void proofWorld(const std::filesystem::path& path,const std::vector<Fixture>& fixtures,
            const std::vector<NativeModelPackage>& packages)
        {
            saveEditableLandscape(path,createEditableLandscape());
            EsmDocument document(path);
            std::vector<ESMModel> models; std::vector<ESMReference> references;
            constexpr std::array<float,4> x{512,1400,2300,3200},z{300,0,400,400};
            for (std::size_t i=0;i<fixtures.size();++i)
            {
                ESMModel model; model.formId=firstModel+static_cast<std::uint32_t>(i); model.editorId="OF4Compat"+std::to_string(i);
                model.path=packages[i].modelPath; model.kind=fixtures[i].physics.bodies[0].dynamic ? ESMModelKind::movableStatic : ESMModelKind::staticObject;
                models.push_back(model);
                ESMReference reference; reference.placement.formId=firstReference+static_cast<std::uint32_t>(i);
                reference.placement.baseObject=model.formId; reference.placement.cell={0,0};
                reference.placement.position[0]=x[i]; reference.placement.position[1]=512; reference.placement.position[2]=z[i];
                reference.editorId="OF4CompatRef"+std::to_string(i); references.push_back(reference);
            }
            document.setModels(models); document.setReferences(references); document.setNextFormId(0x01000a04); document.save();
            EsmDocument reload(path);
            require(reload.models().size()==models.size() && reload.references().size()==references.size(),"proof world semantic reload");
            for (const auto& model:models)
                require(std::ranges::any_of(reload.models(),[&](const auto& m) { return m.formId==model.formId && m.kind==model.kind; }),"proof STAT/MSTT type changed");
        }

        void sourceReport(const std::filesystem::path& root,const std::vector<Fixture>& fixtures)
        {
            std::ofstream file(root/"source-fixtures.json"); file << std::setprecision(9);
            file << "{\"schema\":1,\"unitsPerMeter\":69.99125,\"groundY\":0,\"generations\":10,"
                "\"tolerances\":{\"vertexUnits\":0.002,\"normal\":0.022,\"uv\":0.002,\"convexMeters\":0.00003,"
                "\"triangleMeters\":0.004,\"comMeters\":0.00002,\"tensorFrobeniusKgM2\":0.00008,\"orientationDegrees\":0.05,\"alpha\":0.003921569},"
                "\"contactEncoding\":\"upper 16 float bits (truncate, not IEEE half)\","
                "\"sourceBaseColor\":[0.25,0.5,0.75,1],\"sourceTextureRGBA\":[255,0,0,255,0,255,0,255,0,0,255,255,255,255,0,255],"
                "\"expectedBakedTextureRGBA\":[137,0,0,255,0,188,0,255,0,0,225,255,137,188,0,255],\"fixtures\":[";
            for (std::size_t i=0;i<fixtures.size();++i)
            {
                const auto& f=fixtures[i]; const auto& b=f.physics.bodies[0];
                if (i) file << ',';
                file << "{\"name\":\""<<f.name<<"\",\"modelId\":"<<firstModel+i<<",\"referenceId\":"<<firstReference+i
                    <<",\"dynamic\":"<<(b.dynamic?"true":"false")<<",\"mass\":"<<b.mass<<",\"com\":["
                    <<f.expectedCom.x<<','<<f.expectedCom.y<<','<<f.expectedCom.z<<"],\"tensorColumnMajor\":[";
                for (int c=0;c<3;++c) for (int r=0;r<3;++r) { if (c||r) file<<','; file<<f.expectedTensor[c][r]; }
                file << "],\"boxes\":[";
                for (std::size_t j=0;j<f.boxes.size();++j)
                {
                    const auto& box=f.boxes[j]; if (j) file<<',';
                    file << "{\"center\":["<<box.center.x<<','<<box.center.y<<','<<box.center.z<<"],\"halfExtents\":["
                        <<box.half.x<<','<<box.half.y<<','<<box.half.z<<"],\"rotationWXYZ\":["<<box.orientation.w<<','
                        <<box.orientation.x<<','<<box.orientation.y<<','<<box.orientation.z<<"],\"mass\":"<<box.mass<<'}';
                }
                file << "],\"bodyPosition\":["<<b.position.x<<','<<b.position.y<<','<<b.position.z<<"],\"bodyRotationWXYZ\":["
                    <<b.rotation.w<<','<<b.rotation.x<<','<<b.rotation.y<<','<<b.rotation.z<<"]}";
            }
            file << "],\"doorwayProbes\":{\"directions\":[[0,1,0],[0,-1,0]],\"clearXZ\":[[0.2,1],[0.8,2.2]],"
                "\"blockedXZ\":[[-1.2,1],[1.2,1],[0.2,2.8]],\"originY\":[-2,2]},"
                "\"numericPlacementProbes\":{\"scales\":[0.25,1,4],\"translationEngine\":[100000,-25000,31000]}}\n";
            require(bool(file.flush()),"cannot write source oracle");
        }
    }

    void runCompatibilityTests(const std::filesystem::path& outputRoot)
    {
        require(!std::filesystem::exists(outputRoot),"output directory must be new");
        std::filesystem::create_directories(outputRoot);
        Reports reports(outputRoot);
        std::string current="source-oracle",profile="Generated";
        try
        {
            const auto fixtures=sourceFixtures(); sourceReport(outputRoot,fixtures);
            std::vector<NativeModelPackage> first,last;
            for (std::size_t i=0;i<fixtures.size();++i)
            {
                const auto& fixture=fixtures[i]; current=fixture.name;
                auto package=exportNativeModelAsset(outputRoot/"E0.esm",firstModel+static_cast<std::uint32_t>(i),fixture.geometry,&fixture.physics);
                first.push_back(package);
                std::optional<ModelGeometry> baselineGeometry;
                std::optional<HavokPhysicsSystem> baselinePhysics;
                std::vector<editor::AssetMaterial> baselineMaterials;
                for (int generation=0;generation<=10;++generation)
                {
                    const auto world=outputRoot/("E"+std::to_string(generation)+".esm");
                    editor::AssetDocument document(world,package.modelPath);
                    require(document.capabilities().materials && document.capabilities().collision,"generated source became read-only: "+document.capabilities().reason);
                    require(document.collisionSystems().size()==1,"native attachment count");
                    const auto& physics=document.collisionSystems()[0].physics;
                    const auto error=verify(fixture,document.sourceGeometry(),physics);
                    for (std::size_t s=0;s<document.materials().size();++s)
                    {
                        const auto& material=document.materials()[s]; const auto& expected=fixture.geometry.surfaces[s];
                        const auto imageMatches=[](const auto& native,const auto& input,bool tinted) {
                            if (!native || !input || native->width!=input->width || native->height!=input->height
                                || native->pixels.size()<input->pixels.size()) return false;
                            auto expected=input->pixels;
                            // IEC sRGB encodings of linear .25/.5/.75, independently rounded to 8 bits.
                            constexpr std::array<std::byte,3> factor{std::byte{137},std::byte{188},std::byte{225}};
                            if (tinted) for (std::size_t p=0;p<expected.size();p+=4)
                                for (std::size_t c=0;c<3;++c) if (expected[p+c]==std::byte{255}) expected[p+c]=factor[c];
                            return std::equal(expected.begin(),expected.end(),native->pixels.begin());
                        };
                        require(imageMatches(material.diffuseImage,expected.diffuseImage,true)
                            && imageMatches(material.normalImage,expected.normalImage,false)
                            && material.baseColor==glm::vec4(1),fixture.name+": source diffuse tint/normal texels");
                    }
                    if (!baselineGeometry)
                    {
                        baselineGeometry=document.sourceGeometry(); baselinePhysics=physics; baselineMaterials=document.materials();
                    }
                    else
                    {
                        Fixture baseline=fixture; baseline.geometry=*baselineGeometry; baseline.physics=*baselinePhysics;
                        baseline.expectedCom=baselinePhysics->bodies[0].centerOfMass; baseline.expectedTensor=tensor(baselinePhysics->bodies[0]);
                        (void)verify(baseline,document.sourceGeometry(),physics);
                        for (std::size_t s=0;s<baselineMaterials.size();++s)
                        {
                            const auto& a=baselineMaterials[s]; const auto& b=document.materials()[s];
                            require(a.diffuseImage && b.diffuseImage && a.diffuseImage->pixels==b.diffuseImage->pixels
                                && a.normalImage && b.normalImage && a.normalImage->pixels==b.normalImage->pixels
                                && a.baseColor==b.baseColor,fixture.name+": diffuse/normal bytes/tint drift");
                        }
                    }
                    reports.cycles<<fixture.name<<'\t'<<generation<<"\tPass\t"<<error.position<<'\t'<<error.com<<'\t'<<error.inertia<<'\t'<<package.modelPath<<'\n';
                    reports.cycles.flush();
                    if (generation<10)
                        package=document.saveNative(outputRoot/("E"+std::to_string(generation+1)+".esm"),firstModel+static_cast<std::uint32_t>(i));
                }
                last.push_back(package);
                reports.row("SER03-SER07-"+current,profile,"Pass","E0 direct native export and ten fresh native AssetDocument cycles match source and E0; explicit axes, topology, normals, UVs, source texture texels, contact quantization and full inertia oracle");
            }

            current="SER03-transformed-landmarks";
            const auto placement=glm::translate(glm::mat4(1),glm::vec3(100000,-25000,31000))
                *glm::rotate(glm::mat4(1),.43f,glm::normalize(glm::vec3(2,3,1)));
            for (std::size_t i=0;i<fixtures.size();++i)
            {
                editor::AssetDocument document(outputRoot/"E10.esm",last[i].modelPath);
                for (float scale : {.25f,1.f,4.f})
                    for (std::size_t s=0;s<fixtures[i].geometry.surfaces.size();++s)
                        for (std::size_t v=0;v<fixtures[i].geometry.surfaces[s].vertices.size();++v)
                        {
                            const auto expected=placement*glm::vec4(fixtures[i].geometry.surfaces[s].vertices[v].position*scale,1);
                            const auto actual=placement*glm::vec4(document.sourceGeometry().surfaces[s].vertices[v].position*scale,1);
                            require(glm::length(expected-actual)<.02f,"scaled far-origin render landmark drift");
                        }
            }
            reports.row(current,"Generated","Pass","Numeric 0.25/1/4 scale and rotated far-origin landmarks match source within 0.02 engine units; not separate in-game scale tests");
            reports.row("scaled-Fallout4-placements","Fallout4","Not run","0.25/1/4 and far-origin in-game placements require fresh manual observations");

            current="negative-oracles";
            auto wrong=fixtures[0].physics; wrong.bodies[0].centerOfMass.x+=.2f;
            const auto wrongDecoded=loadHavokPhysicsSystem(serializeHavokPhysicsSystem(wrong));
            expectedFailure([&] { (void)verify(fixtures[0],fixtures[0].geometry,wrongDecoded); },"valid wrong COM");
            wrong=fixtures[2].physics; wrong.bodies[0].inertiaOrientation=glm::quat(1,0,0,0);
            const auto wrongTensor=loadHavokPhysicsSystem(serializeHavokPhysicsSystem(wrong));
            expectedFailure([&] { (void)verify(fixtures[2],fixtures[2].geometry,wrongTensor); },"valid wrong inertia orientation");
            wrong=fixtures[2].physics; wrong.bodies[0].parts[0].transform[3].x+=.2f;
            const auto shiftedChild=loadHavokPhysicsSystem(serializeHavokPhysicsSystem(wrong));
            expectedFailure([&] { (void)verify(fixtures[2],fixtures[2].geometry,shiftedChild); },"valid shifted child");
            auto reflected=fixtures[0].geometry;
            for (auto& surface:reflected.surfaces) for (auto& vertex:surface.vertices) vertex.position.x=-vertex.position.x;
            const auto reflectedPackage=exportNativeModelAsset(outputRoot/"negative.esm",0x01000921,reflected,&fixtures[0].physics);
            editor::AssetDocument reflectedAsset(outputRoot/"negative.esm",reflectedPackage.modelPath);
            expectedFailure([&] { (void)verify(fixtures[0],reflectedAsset.sourceGeometry(),reflectedAsset.collisionSystems()[0].physics); },"valid reflected axis");
            auto wrongGeometry=fixtures[0].geometry; wrongGeometry.surfaces[0].vertices[0].position.x+=10;
            const auto wrongPackage=exportNativeModelAsset(outputRoot/"negative.esm",0x01000920,wrongGeometry,&fixtures[0].physics);
            editor::AssetDocument wrongAsset(outputRoot/"negative.esm",wrongPackage.modelPath);
            expectedFailure([&] { (void)verify(fixtures[0],wrongAsset.sourceGeometry(),wrongAsset.collisionSystems()[0].physics); },"valid wrong render position");
            auto sealed=fixtures[1].physics;
            sealed.bodies[0].triangleMesh->vertices={{-.69f,0,0},{.69f,0,0},{.69f,0,2.4f},{-.69f,0,2.4f}};
            sealed.bodies[0].triangleMesh->indices={0,1,2,0,2,3};
            const auto sealedDecoded=loadHavokPhysicsSystem(serializeHavokPhysicsSystem(sealed));
            require(rayHits(*sealedDecoded.bodies[0].triangleMesh,{0,-2,1}),"sealed valid doorway negative probe did not hit");
            expectedFailure([&] { (void)verify(fixtures[1],fixtures[1].geometry,sealedDecoded); },"valid sealed doorway");
            reports.row(current,profile,"Pass","Validly parsed wrong COM, principal orientation, shifted child, reflected axis, render vertex and sealed doorway are rejected by independent expectations");

            current="foreign-native"; profile="Preserved";
            nifly::NifFile foreign(first[2].assetRoot/first[2].modelPath);
            foreign.GetRootNode()->name.get()="Independent preservation fixture";
            auto note=std::make_unique<nifly::NiStringExtraData>();
            note->name.get()="CompatibilitySentinel"; note->stringData.get()="untouched foreign author metadata";
            foreign.GetRootNode()->extraDataRefs.AddBlockRef(foreign.GetHeader().AddBlock(std::move(note)));
            const auto foreignPath=first[2].assetRoot/"meshes/compatibility-foreign.nif";
            require(foreign.Save(foreignPath)==0,"cannot write foreign profile");
            const auto untouched=nonCollisionBytes(foreignPath);
            auto previousWorld=outputRoot/"E0.esm"; std::string previousModel="meshes/compatibility-foreign.nif";
            for (int generation=0;generation<=10;++generation)
            {
                editor::AssetDocument document(previousWorld,previousModel);
                require(!document.capabilities().materials && document.capabilities().collision && !document.capabilities().motion,"foreign capability guard");
                (void)verify(fixtures[2],document.sourceGeometry(),document.collisionSystems()[0].physics);
                nifly::NifFile check(AssetResolver::projectAssetRoot(previousWorld)/previousModel);
                bool retained=false;
                for (std::uint32_t block=0;block<check.GetHeader().GetNumBlocks();++block)
                    if (const auto* extra=check.GetHeader().GetBlock<nifly::NiStringExtraData>(block))
                        retained|=extra->name.get()=="CompatibilitySentinel" && extra->stringData.get()=="untouched foreign author metadata";
                require(retained,"foreign extra data was flattened");
                require(nonCollisionBytes(AssetResolver::projectAssetRoot(previousWorld)/previousModel)==untouched,
                    "noncollision blocks/header/footer changed in preserved profile");
                reports.cycles<<current<<'\t'<<generation<<"\tPass\t0\t0\t0\t"<<previousModel<<'\n'; reports.cycles.flush();
                if (generation<10)
                {
                    const auto next=outputRoot/("P"+std::to_string(generation+1)+".esm");
                    const auto package=document.saveNative(next,0x01000910); previousWorld=next; previousModel=package.modelPath;
                }
            }
            reports.row("SER07-"+current,profile,"Pass","Ten collision-only native saves preserve every byte outside the physics payload and its size field, plus attachments and analytic compound mass frame");

            current="unsupported-ragdoll"; profile="Rejected";
            foreign.GetRootNode()->extraDataRefs.AddBlockRef(foreign.GetHeader().AddBlock(std::make_unique<nifly::bhkRagdollSystem>()));
            const auto rejectedPath=first[2].assetRoot/"meshes/compatibility-rejected.nif";
            require(foreign.Save(rejectedPath)==0,"cannot write rejected profile");
            editor::AssetDocument rejected(outputRoot/"E0.esm","meshes/compatibility-rejected.nif");
            require(!rejected.capabilities().collision && !rejected.capabilities().materials && !rejected.capabilities().reason.empty(),"unsupported source not read-only");
            const auto rejectedBytes=read(rejectedPath);
            expectedFailure([&] { (void)rejected.saveNative(outputRoot/"rejected.esm",0x01000930); },"ragdoll native save");
            require(read(rejectedPath)==rejectedBytes && !rejected.dirty(),"rejected export changed source/document");
            reports.row(current,profile,"Pass",rejected.capabilities().reason);

            current="proof-worlds"; profile="Generated";
            proofWorld(outputRoot/"E0.esm",fixtures,first); proofWorld(outputRoot/"E10.esm",fixtures,last);
            proofPlugin(outputRoot/"Fallout4-E0.esp",fixtures,first); proofPlugin(outputRoot/"Fallout4-E10.esp",fixtures,last);
            reports.row(current,profile,"Pass","E0/E10 native packages and four saved STAT/MSTT references reload from matching sidecars; disposable FO4 ESP bases generated");
            reports.row("OpenF4-runtime","Runtime","Not run","Run normal openf4 with OPENF4_COMPATIBILITY_CAPTURE_DIR; compile-time fixture checks do not substitute for runtime simulation");
            reports.row("Fallout4-E0","Fallout4","Blocked","Requires fresh isolated in-game material, doorway and physics observations");
            reports.row("Fallout4-E10","Fallout4","Blocked","Requires fresh isolated in-game material, doorway and physics observations");
            reports.summary("Pass: all automated structural/oracle/cycle checks completed.");
        }
        catch (const std::exception& error)
        {
            reports.row(current,profile,"Fail",error.what());
            reports.summary("Fail: "+std::string(error.what()));
            throw;
        }
    }
}
