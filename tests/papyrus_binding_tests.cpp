#include "assets/papyrus_binding.hpp"
#include "assets/esm_document.hpp"
#include "assets/fo4_world_writer.hpp"
#include "assets/fo4_export.hpp"
#include "assets/native_model_export.hpp"
#include "editor/editor_document.hpp"
#include "fixtures/papyrus/compiled.hpp"
#include "script/papyrus_vm.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace openf4::assets;
using openf4::editor::EditorDocument;
using Bytes = std::vector<std::byte>;
void require(bool value, const char* error) { if (!value) throw std::runtime_error(error); }
template<class F> void rejects(F operation, const char* error) {
    try { operation(); } catch (const std::exception&) { return; } throw std::runtime_error(error);
}
Bytes read(const std::filesystem::path& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate); require(bool(f), "Cannot read fixture");
    Bytes b(static_cast<std::size_t>(f.tellg())); f.seekg(0); f.read(reinterpret_cast<char*>(b.data()), b.size()); require(bool(f), "Short fixture"); return b;
}
void write(const std::filesystem::path& path, const Bytes& b) {
    std::ofstream f(path, std::ios::binary); f.write(reinterpret_cast<const char*>(b.data()), b.size()); require(bool(f), "Cannot write fixture");
}
template<class T> T at(const Bytes& b, std::size_t p) {
    require(p + sizeof(T) <= b.size(), "Fixture bounds"); T value; std::memcpy(&value, b.data()+p, sizeof(T)); return value;
}
template<class T> void put(Bytes& bytes, T value) {
    const auto* p=reinterpret_cast<const std::byte*>(&value);bytes.insert(bytes.end(),p,p+sizeof(T));
}
PapyrusBindings binding(std::uint32_t target = 0) {
    PapyrusBindings result; result.present = true;
    result.scripts.push_back({"OpenF4:Wake", 0, {
        {"Enabled", 1, true}, {"Count", 1, std::int32_t{3}}, {"Strength", 1, 2.5f},
        {"Message", 1, std::string("Hello")}, {"Target", 1, PapyrusObject{target}}, {"Nothing", 1, std::monostate{}}}});
    return result;
}
void codec() {
    auto value = binding(0x01000901); const auto encoded = encodePapyrusBindings(value);
    require(at<std::uint16_t>(encoded,0)==6 && at<std::uint16_t>(encoded,2)==2 && at<std::uint16_t>(encoded,4)==1,"Wrong FO4 header");
    auto decoded=decodePapyrusBindings(encoded,"source.esm");
    require(decoded.editable && decoded.scripts==value.scripts && decoded.sourceEsm=="source.esm","VMAD semantic round trip failed");
    require(encodePapyrusBindings(decoded)==encoded,"VMAD byte round trip failed");
    remapPapyrusBindings(decoded,[](std::uint32_t id){return id? id+0x01000000 : id;});
    require(std::get<PapyrusObject>(decoded.scripts[0].properties[4].value).formId==0x02000901,"Object property did not remap");
    require(encodePapyrusBindings(decoded)==encoded,"Remapping altered preserved original bytes");
    require(encodePapyrusBindings(decoded,false)!=encoded,"Remapped semantic serialization ignored remap");
    for (std::size_t length=0;length<encoded.size();++length) {
        const auto truncated=decodePapyrusBindings(std::span(encoded).first(length));
        require(!truncated.editable,"Truncated VMAD accepted");
        require(encodePapyrusBindings(truncated)==Bytes(encoded.begin(),encoded.begin()+length),"Opaque VMAD not preserved");
    }
    auto future=encoded;future[0]=std::byte{99};const auto unknown=decodePapyrusBindings(future);
    require(!unknown.editable&&encodePapyrusBindings(unknown)==future,"Unknown version was flattened");
    rejects([&]{(void)encodePapyrusBindings(unknown,false);},"Unsupported VMAD was remapped");
    auto duplicate=value;duplicate.scripts.push_back(duplicate.scripts.front());duplicate.scripts.back().name="openf4:wake";
    rejects([&]{(void)encodePapyrusBindings(duplicate);},"Case-insensitive duplicate accepted");
    duplicate=value;duplicate.scripts[0].properties.push_back(duplicate.scripts[0].properties[0]);
    rejects([&]{(void)encodePapyrusBindings(duplicate);},"Duplicate property accepted");
    require(papyrusScriptPath("OpenF4:Wake")=="scripts/OpenF4/Wake.pex","Namespace path incorrect");
    for(const auto name:{"../Wake","C:/Wake","A::B",":Wake","Wake:","A/B"})
        rejects([&]{(void)papyrusScriptPath(name);},"Unsafe script path accepted");
    auto ref=value;ref.scripts[0].flags=1;ref.scripts[0].properties={{"Enabled",1,false},{"Message",3,std::string{}}};
    const auto merged=effectivePapyrusScripts(value,ref);
    require(merged.size()==1&&merged[0].properties.size()==5&&!std::get<bool>(merged[0].properties[0].value),"Inherited property override/removal failed");
    ref.scripts[0].flags=3;require(effectivePapyrusScripts(value,ref).empty(),"Inherited script removal failed");
}
void document(const std::filesystem::path& root) {
    const auto path=root/"bindings.esm"; EditorDocument editor(path);
    const auto model=editor.addModel("models/test.glb",{},ESMModelKind::movableStatic);
    const auto ref=editor.place(model,glm::mat4(1));editor.save();
    editor.setModelScripts(model,binding(ref));require(editor.dirty(),"Script edit not dirty");
    require(editor.undo()&&!editor.model(model)->scripts.present,"Base script undo failed");
    require(editor.redo()&&editor.model(model)->scripts.present,"Base script redo failed");
    auto overrides=binding(model);overrides.scripts[0].flags=1;overrides.scripts[0].properties={{"Enabled",1,false}};
    editor.setReferenceScripts(ref,overrides);const auto copy=editor.duplicate(ref);editor.save();
    EsmDocument saved(path);require(saved.models()[0].scripts.scripts==binding(ref).scripts,"Base VMAD did not reload");
    require(saved.references().size()==2&&saved.references()[0].scripts.present&&saved.references()[1].scripts.present,"Duplicate lost reference VMAD");
    require(saved.references()[0].scripts.scripts==saved.references()[1].scripts.scripts,"Duplicate changed script properties");
    EditorDocument reopened(path);reopened.setLocked(copy,true);
    rejects([&]{reopened.setReferenceScripts(copy,{});},"Locked script edit accepted");
    const auto before=read(path);auto bad=binding();bad.scripts.push_back(bad.scripts[0]);
    rejects([&]{reopened.setModelScripts(model,bad);},"Invalid command accepted");
    require(read(path)==before,"Failed command wrote world");
    reopened.setModelScripts(model,{});reopened.save();require(!EsmDocument(path).models()[0].scripts.present,"Detach left VMAD");
    require(reopened.undo(),"Detach undo unavailable");reopened.save();require(EsmDocument(path).models()[0].scripts.present,"Detach undo failed to restore VMAD");
    reopened.setReferenceScripts(ref,{});reopened.save();require(!EsmDocument(path).references()[0].scripts.present,"Reference detach left VMAD");
    require(reopened.undo(),"Reference detach undo unavailable");reopened.save();require(EsmDocument(path).references()[0].scripts.present,"Reference detach undo failed to restore VMAD");
}
void inheritance(const std::filesystem::path& root) {
    const auto path=root/"inherited.esm";EditorDocument original(path);
    const auto model=original.addModel("models/shared.glb");const auto ref=original.place(model,glm::mat4(1));
    original.setModelScripts(model,binding(ref));original.save();
    std::filesystem::create_directories(original.assetRoot()/"models");write(original.assetRoot()/"models/shared.glb",{std::byte{1}});
    auto protectedBytes=read(path);const std::string author="OpenF4";
    const auto authorAt=std::search(protectedBytes.begin(),protectedBytes.end(),reinterpret_cast<const std::byte*>(author.data()),reinterpret_cast<const std::byte*>(author.data()+author.size()));
    require(authorAt!=protectedBytes.end(),"Missing author");*authorAt=std::byte{'X'};write(path,protectedBytes);
    EditorDocument overlay(path);auto scripts=overlay.model(model)->scripts;scripts.scripts[0].properties[0].value=false;
    overlay.setModelScripts(model,scripts);overlay.save();
    require(read(path)==protectedBytes,"Script override changed protected master");
    auto stack=loadEsmMasterStack(overlay.path());
    require(stack.models.size()==1&&stack.models[0].formId==model&&stack.models[0].sourceEsm==path.string(),"Script-only override redirected identity or geometry ownership");
    require(stack.models[0].recordOwnerEsm==overlay.path().string()&&!std::get<bool>(stack.models[0].scripts.scripts[0].properties[0].value),"Inherited override did not persist");
    require(overlay.undo(),"Inherited script undo unavailable");overlay.save();
    require(EsmDocument(overlay.path()).models().empty(),"Undo retained script-only override");
    stack=loadEsmMasterStack(overlay.path());require(std::get<bool>(stack.models[0].scripts.scripts[0].properties[0].value),"Undo failed to reveal master scripts");
}
void rawPreservation(const std::filesystem::path& root) {
    const auto path=root/"opaque.esm";EditorDocument initial(path);const auto model=initial.addModel("models/x.glb");
    const auto ref=initial.place(model,glm::mat4(1));initial.setReferenceScripts(ref,binding());initial.save();
    auto malformed=read(path);const auto wire=encodePapyrusBindings(binding());
    const auto found=std::search(malformed.begin(),malformed.end(),wire.begin(),wire.end());require(found!=malformed.end(),"Missing VMAD fixture");
    *found=std::byte{99};write(path,malformed);
    EditorDocument editor(path);require(!editor.references()[0].scripts.editable,"Unsupported VMAD editable");
    rejects([&]{editor.setReferenceScripts(ref,binding());},"Opaque script replacement accepted");
    auto moved=glm::mat4(1);moved[3].x=256;editor.setTransform(ref,moved);editor.save();
    const auto after=EsmDocument(path).references()[0].scripts;
    require(!after.editable&&after.originalBytes[0]==std::byte{99},"Transform edit destroyed opaque VMAD");
}
void nativeWriter(const std::filesystem::path& root) {
    Fo4WorldWriteRequest request;request.editorId="OF4_Scripts";request.worldFormId=0x01000800;
    request.persistentCellFormId=0x01000801;request.cells.push_back({{0,0},0x01000802,0x01000803});
    request.loadCell=[](CellCoord c)->std::optional<LandscapeCell>{if(c!=CellCoord{0,0})return std::nullopt;LandscapeCell cell;cell.cell=c;cell.hasHeights=true;return cell;};
    request.models.push_back({0x01000804,ESMModelKind::movableStatic,"ScriptedBase","meshes/script.nif",{},"ScriptedBase",binding(0x01000805)});
    ESMReference ref;ref.placement.formId=0x01000805;ref.placement.baseObject=0x01000804;ref.scripts=binding(0x01000804);request.references.push_back(ref);
    request.arrival.placement.formId=0x01000806;request.arrival.placement.position[2]=64;
    const auto output=root/"scripts.esp";writeFo4WorldPlugin(output,request);
    EsmDocument reloaded(output);require(reloaded.models()[0].scripts.scripts==request.models[0].scripts.scripts,"Native base lost VMAD");
    const auto found=std::ranges::find_if(reloaded.references(),[](const auto& r){return r.placement.formId==0x01000805;});
    require(found!=reloaded.references().end()&&found->scripts.scripts==ref.scripts.scripts,"Native reference lost VMAD");
    auto& object=std::get<PapyrusObject>(request.references[0].scripts.scripts[0].properties[4].value);object.formId=0x01009876;
    rejects([&]{writeFo4WorldPlugin(root/"invalid-script.esp",request);},"Unexported object property accepted");
    require(!std::filesystem::exists(root/"invalid-script.esp"),"Invalid property published plugin");
    object.formId=0;object.alias=2;
    rejects([&]{writeFo4WorldPlugin(root/"alias-script.esp",request);},"Unsupported quest alias exported");
}
Bytes stubPex(const std::string& name) {
    // Test-only native-class placeholder, with no game-owned code or source.
    Bytes bytes;put(bytes,std::uint32_t{0xfa57c0de});put(bytes,std::uint8_t{3});put(bytes,std::uint8_t{9});put(bytes,std::uint16_t{2});put(bytes,std::uint64_t{});
    const auto text=[&](std::string_view value){put(bytes,static_cast<std::uint16_t>(value.size()));for(char c:value)put(bytes,c);};
    text("");text("");text("");put(bytes,std::uint16_t{2});text(name);text("");
    put(bytes,std::uint8_t{});put(bytes,std::uint16_t{});put(bytes,std::uint16_t{1});put(bytes,std::uint16_t{0});
    put(bytes,std::uint32_t{23});put(bytes,std::uint16_t{1});put(bytes,std::uint16_t{1});put(bytes,std::uint8_t{});put(bytes,std::uint32_t{});put(bytes,std::uint16_t{1});
    for(int i=0;i<4;++i)put(bytes,std::uint16_t{});
    require(openf4::script::inspectPex({reinterpret_cast<const std::uint8_t*>(bytes.data()),bytes.size()}).name==name,"Invalid stub PEX fixture");return bytes;
}
void nativeExport(const std::filesystem::path& root) {
    const auto path=root/"script-export.esm";EditorDocument editor(path);
    ModelGeometry geometry;auto& surface=geometry.surfaces.emplace_back();surface.metallicFactor=0;
    for(const auto p:{glm::vec3{0,0,0},glm::vec3{64,0,0},glm::vec3{0,0,64}}){openf4::render::SourceVertex v;v.position=p;v.normal={0,1,0};surface.vertices.push_back(v);}surface.indices={0,1,2};
    HavokPhysicsSystem physics;auto& body=physics.bodies.emplace_back();body.triangleMesh.emplace();body.triangleMesh->vertices={{0,0,0},{1,0,0},{0,1,0}};body.triangleMesh->indices={0,1,2};
    const auto model=editor.addModel("models/placeholder.glb");const auto native=exportNativeModelAsset(path,model,geometry,&physics);
    (void)editor.assignNativeModel(model,native.modelPath,ESMModelKind::staticObject);const auto reference=editor.place(model,glm::mat4(1));
    PapyrusBindings scripts;scripts.present=true;scripts.scripts={{"OpenF4VmFixture",0,{{"Counter",1,std::int32_t{4}},{"Impulse",1,2.f},{"Target",1,PapyrusObject{reference}}}}};
    editor.setReferenceScripts(reference,scripts);editor.save();
    const auto scriptRoot=editor.assetRoot()/"scripts";std::filesystem::create_directories(scriptRoot);
    const auto store=[&](const std::filesystem::path& target,const auto& source){Bytes b(source.size());std::memcpy(b.data(),source.data(),b.size());write(target,b);};
    store(scriptRoot/"OpenF4VmFixture.pex",openf4::test::papyrus::OpenF4VmFixture);
    store(scriptRoot/"OpenF4VmBase.pex",openf4::test::papyrus::OpenF4VmBase);
    const auto data=root/"script-native-data";std::filesystem::create_directories(data/"scripts");
    write(data/"scripts/ObjectReference.pex",stubPex("ObjectReference"));write(data/"scripts/Utility.pex",stubPex("Utility"));
    Fo4ExportOptions options;options.dataDirectory=data;
    const auto preflight=inspectFo4Export(path,options);
    if(!preflight.canWriteWorldProof()){for(const auto& error:preflight.blockers)std::cerr<<error<<'\n';throw std::runtime_error("Script preflight failed");}
    const auto output=exportFo4WorldProof(path,root/"script-package",options);
    require(read(output.packageDirectory/"Data/scripts/OpenF4VmFixture.pex")==read(scriptRoot/"OpenF4VmFixture.pex"),"PEX changed during export");
    require(read(output.packageDirectory/"Data/scripts/openf4vmbase.pex")==read(scriptRoot/"OpenF4VmBase.pex"),"Project parent PEX missing");
    require(!std::filesystem::exists(output.packageDirectory/"Data/scripts/objectreference.pex"),"Native script was redistributed");
    const auto refs=EsmDocument(output.pluginPath).references();const auto found=std::ranges::find_if(refs,[&](const auto& r){return r.placement.formId==reference;});
    require(found!=refs.end()&&found->scripts.scripts==scripts.scripts,"Export changed VMAD binding");
    auto bad=scripts;bad.scripts[0].properties[0].value=true;editor.setReferenceScripts(reference,bad);editor.save();
    require(!inspectFo4Export(path,options).canWriteWorldProof(),"PEX/VMAD property mismatch accepted");
    bad=scripts;std::get<PapyrusObject>(bad.scripts[0].properties[2].value).formId=0x0100f000;editor.setReferenceScripts(reference,bad);editor.save();
    require(!inspectFo4Export(path,options).canWriteWorldProof(),"Unexported script object property passed preflight");
    editor.setReferenceScripts(reference,scripts);editor.save();
    std::filesystem::rename(scriptRoot/"OpenF4VmBase.pex",scriptRoot/"OpenF4VmBase.saved");
    require(!inspectFo4Export(path,options).canWriteWorldProof(),"Missing project script parent accepted");
}
int main() try {
    const auto root=std::filesystem::temp_directory_path()/ ("openf4-papyrus-bindings-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    require(std::filesystem::create_directory(root),"Cannot create test directory");
    codec();document(root);inheritance(root);rawPreservation(root);nativeWriter(root);
#ifdef _WIN32
    nativeExport(root);
#endif
    std::filesystem::remove_all(root);std::cout<<"Papyrus binding tests passed\n";return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
