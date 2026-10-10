#include "script/object_scripts.hpp"
#include "script/papyrus_compiler.hpp"
#include "editor/editor_document.hpp"
#include "assets/fo4_install.hpp"
#include "assets/fo4_export.hpp"
#include "assets/native_model_export.hpp"
#include "physics/jolt_collision.hpp"

#include <array>
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <glm/ext/matrix_transform.hpp>

namespace
{
    void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
    std::string read(const std::filesystem::path& path)
    {
        std::ifstream file(path, std::ios::binary);
        return {std::istreambuf_iterator<char>(file), {}};
    }
}

int main()
{
    using namespace openf4;
    const auto data = assets::findFallout4Data();
    if (!data || !std::filesystem::is_regular_file(data->parent_path() / "Papyrus Compiler/PapyrusCompiler.exe"))
    {
        std::cout << "Creation Kit compiler not installed; compiled-script integration skipped\n";
        return 77;
    }
    const auto root = std::filesystem::temp_directory_path() / ("openf4-papyrus-"
        + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(root);
    try
    {
        const auto worldPath = root / "proof.esm";
        editor::EditorDocument document(worldPath);
        const auto model = document.addModel("models/proof.nif", {}, assets::ESMModelKind::movableStatic);
        const auto ref = document.place(model, glm::translate(glm::mat4(1), glm::vec3(0, 300, 0)));
        script::createWakeScript(worldPath, "OF4Lifecycle");
        const auto source = script::scriptSourcePath(worldPath, "OF4Lifecycle");
        {
            std::ofstream file(source);
            file << "Scriptname OF4Lifecycle extends ObjectReference\n"
                "Int Property Starts = 0 Auto\n"
                "Event OnInit()\n Starts += 1\n Debug.Trace(\"INIT\")\nEndEvent\n"
                "Event OnLoad()\n Debug.Trace(\"LOAD\")\n SetMotionType(1, true)\n"
                " ApplyHavokImpulse(0.0, 0.0, 1.0, 0.0)\n Utility.Wait(0.05)\n"
                " Debug.Trace(\"AFTER\")\nEndEvent\n"
                "Event OnUnload()\n Debug.Trace(\"UNLOAD\")\nEndEvent\n";
        }
        const auto compiled = script::compilePapyrus(worldPath, *data, "OF4Lifecycle");
        require(compiled.success, compiled.diagnostics.c_str());
        const auto savedPex = read(compiled.output);
        const auto savedSource = read(source);
        {
            std::ofstream file(source); file << "this is not Papyrus";
        }
        const auto failed = script::compilePapyrus(worldPath, *data, "OF4Lifecycle");
        require(!failed.success && read(compiled.output) == savedPex, "Failed compile replaced working bytecode");
        { std::ofstream file(source); file << savedSource; }
        try { (void)script::scriptSourcePath(worldPath, "../escape"); throw std::logic_error("Unsafe name accepted"); }
        catch (const std::runtime_error&) {}
        assets::PapyrusBindings bindings;
        bindings.present = true;
        bindings.scripts.push_back({"OF4Lifecycle", 0, {{"Starts", 1, std::int32_t(5)}}});
        document.setReferenceScripts(ref, bindings);
        document.save();
        physics::JoltWorld physics;
        (void)physics.addStaticBox({1000, 10, 1000}, {0, -10, 0});
        std::array<glm::vec3, 8> vertices;
        for (unsigned i = 0; i < vertices.size(); ++i)
            vertices[i] = {i & 1 ? 35 : -35, i & 2 ? 35 : -35, i & 4 ? 35 : -35};
        assets::HavokRigidBody sourceBody;
        sourceBody.parts.push_back({glm::mat4(1), glm::vec3(1), physics::makeHavokConvexHull(vertices)});
        sourceBody.dynamic = true; sourceBody.mass = 4;
        sourceBody.collisionFilterInfo = 5; // Match the Weapon layer used in the validated native prop test.
        physics::updateHavokMassProperties(sourceBody);
        assets::ModelGeometry geometry;
        auto& surface = geometry.surfaces.emplace_back();
        const auto cube = render::makeCubeMesh();
        for (const auto& vertex : cube.vertices)
        {
            render::SourceVertex v;
            v.position = glm::vec3(vertex.position) * 35.0f;
            v.normal = vertex.normal; v.uv = vertex.texCoord;
            surface.vertices.push_back(v);
        }
        surface.indices = cube.indices;
        surface.metallicFactor = 0;
        assets::HavokPhysicsSystem nativePhysics;
        nativePhysics.bodies.push_back(sourceBody);
        const auto native = assets::exportNativeModelAsset(worldPath, model, geometry, &nativePhysics);
        (void)document.assignNativeModel(model, native.modelPath, assets::ESMModelKind::movableStatic);
        document.save();
        const auto stack = assets::loadEsmMasterStack(worldPath);
        assets::Fo4ExportOptions exportOptions;
        exportOptions.dataDirectory = data;
        const auto exported = assets::exportFo4WorldProof(worldPath, root / "fallout-proof", exportOptions);
        require(read(exported.packageDirectory / "Data/scripts/OF4Lifecycle.pex") == savedPex,
            "Export changed compiled bytecode");
        const auto body = physics.addCollisionBody(sourceBody, {0, 300, 0});
        require(physics.setMotionType(body, 0), "Cannot fix test body");
        require(physics.setMotionType(body, 1, false), "Cannot suspend test body");
        bool loaded = false;
        int init = 0, loads = 0, unloads = 0, waits = 0, impulses = 0;
        std::vector<std::string> errors;
        script::ObjectScriptHost host;
        host.loaded = [&](std::uint32_t id) { return id == ref && loaded; };
        host.position = [&](std::uint32_t) { return glm::vec3(physics.bodyTransform(body)[3]); };
        host.motion = [&](std::uint32_t id, std::uint32_t type, bool activate) {
            require(id == ref && physics.setMotionType(body, type, activate), "Wrong script motion target");
        };
        host.impulse = [&](std::uint32_t id, glm::vec3 direction, float magnitude) {
            require(id == ref && direction == glm::vec3(0, 1, 0) && magnitude == 0,
                "Papyrus impulse axes or magnitude changed");
            require(physics.applyHavokImpulse(body, direction, magnitude), "Script impulse rejected"); ++impulses;
        };
        host.diagnostic = [&](std::string_view text) {
            if (text == "Papyrus: INIT") ++init;
            else if (text == "Papyrus: LOAD") ++loads;
            else if (text == "Papyrus: UNLOAD") ++unloads;
            else if (text == "Papyrus: AFTER") ++waits;
            else errors.emplace_back(text);
        };
        script::ObjectScripts scripts(stack, worldPath, *data, std::move(host));
        scripts.update(0.1f);
        require(init == 0, "Script ran before reference and surroundings were ready");
        loaded = true;
        for (int i = 0; i < 120; ++i) { scripts.update(1.0f / 60); physics.step(1.0f / 60); }
        for (const auto& error : errors) std::cerr << error << '\n';
        require(errors.empty(), "Compiled script execution produced diagnostics");
        require(init == 1 && loads == 1 && waits == 1 && impulses == 1, "Missing lifecycle/native/latent script call");
        require(physics.bodyTransform(body)[3].y < 100 && physics.bodyTransform(body)[3].y > 20,
            "Script did not wake body or collide with ground");
        loaded = false; scripts.update(0.1f);
        loaded = true;
        for (int i = 0; i < 20; ++i) scripts.update(0.01f);
        require(init == 1 && loads == 2 && unloads == 1 && waits == 2,
            "Cell residency restarted script state or lost lifecycle events");
        std::cout << "Papyrus CK compile, persistence, runtime, Jolt wake and residency passed\n";
        std::cout << "Proof world: " << worldPath << "\nFallout proof: " << exported.packageDirectory << '\n';
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << "\nRetained test evidence: " << root << '\n';
        return 1;
    }
}
