#include "assets/fo4_export.hpp"
#include "assets/esm_editor.hpp"
#include "assets/esm_document.hpp"
#include "assets/fo4_install.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

#ifdef _WIN32
#define NOMINMAX
#include <Windows.h>
#endif

namespace
{
    using namespace openf4::assets;
    void require(bool result, const char* error) { if (!result) throw std::runtime_error(error); }
    template<class F> void rejects(F operation, const char* error)
    {
        try { operation(); } catch (const std::exception&) { return; }
        throw std::runtime_error(error);
    }
    std::string read(const std::filesystem::path& path)
    {
        std::ifstream input(path, std::ios::binary);
        if (!input) throw std::runtime_error("Missing test file");
        return {std::istreambuf_iterator<char>(input), {}};
    }
    void tests(const std::filesystem::path& root)
    {
        const auto world = root / "proof.esm";
        saveEditableLandscape(world, createEditableLandscape());
        const auto before = read(world);
        auto inspection = inspectFo4Export(world);
        require(inspection.canWriteWorldProof() && inspection.cells == 9
            && inspection.references == 0 && !inspection.limitations.empty(), "Incorrect world proof preflight");
        const auto exported = exportFo4WorldProof(world, root / "package");
        require(read(world) == before, "Export changed the source ESM");
        require(std::filesystem::exists(exported.pluginPath), "No exported plugin");
        const auto report = read(exported.packageDirectory / "export-report.txt");
        require(report.find("NOT A COMPLETE PLAYABLE-WORLD EXPORT") != report.npos
            && report.find("Source SHA-256") != report.npos && report.find("Output SHA-256") != report.npos,
            "Proof report failed to declare missing capabilities/evidence");
        EsmDocument native(exported.pluginPath);
        require(native.terrainCells().size() == 9 && native.references().size() == 1,
            "Native output lost cells/arrival");
        require(native.worldFormId() == EditableLandscape::worldFormId, "Export changed existing world identity");
        require(native.references().front().placement.position[2] == 64, "Arrival is not above terrain");
        require(exported.arrivalCell == CellCoord{0,0}, "Default arrival moved to the wrong cell");
        const auto pluginBefore = read(exported.pluginPath);
        rejects([&] { (void)exportFo4WorldProof(world, root / "package"); }, "Export overwrote an existing package");
        require(read(exported.pluginPath) == pluginBefore, "Failed export modified previous package");
        rejects([&] { (void)exportFo4WorldProof(world, root / "proof.assets" / "export"); },
            "Export wrote into authored assets");
        require(!std::filesystem::exists(root / "proof.assets"), "Rejected destination changed authored assets");
        Fo4ExportOptions options;
        options.dataDirectory = root / "Fallout4Data";
        rejects([&] { (void)exportFo4WorldProof(world, *options.dataDirectory / "export", options); },
            "Export auto-installed into Fallout Data");
        require(!std::filesystem::exists(*options.dataDirectory), "Rejected destination created Fallout Data");
        options = {};
        options.arrivalCell = CellCoord{99,99};
        require(!inspectFo4Export(world, options).canWriteWorldProof(), "Absent arrival was accepted");
        rejects([&] { (void)exportFo4WorldProof(world, root / "missing", options); }, "Absent arrival exported");
        require(!std::filesystem::exists(root / "missing"), "Failed preflight published output");
        options = {}; options.cancelled = [] { return true; };
        rejects([&] { (void)inspectFo4Export(world, options); }, "Preflight swallowed cancellation as a blocker");
        rejects([&] { (void)exportFo4WorldProof(world, root / "cancelled", options); }, "Cancel ignored");
        require(!std::filesystem::exists(root / "cancelled"), "Cancelled export published");
        bool cancel = false;
        options.cancelled = [&] { return cancel; };
        options.progress = [&](const auto& progress) { if (progress.phase == "Writing native world") cancel = true; };
        rejects([&] { (void)exportFo4WorldProof(world, root / "late-cancel", options); }, "Late cancel ignored");
        require(!std::filesystem::exists(root / "late-cancel"), "Late cancellation published");
        bool retained = false;
        for (const auto& entry : std::filesystem::directory_iterator(root))
            retained |= entry.path().filename().string().starts_with("late-cancel.incomplete-");
        require(retained, "Cancellation lost diagnostic staging");
        cancel = false;
        options.progress = [&](const auto& progress) { if (progress.phase == "Publishing package") cancel = true; };
        rejects([&] { (void)exportFo4WorldProof(world, root / "publish-cancel", options); }, "Publication cancellation ignored");
        require(!std::filesystem::exists(root / "publish-cancel"), "Publication cancellation published output");
        options = {};
        const auto competingDestination = root / "competing-package";
        options.progress = [&](const auto& progress) {
            if (progress.phase == "Publishing package")
            {
                std::filesystem::create_directory(competingDestination);
                std::filesystem::copy_file(world, competingDestination / "keep.esm");
            }
        };
        rejects([&] { (void)exportFo4WorldProof(world, competingDestination, options); },
            "Publication overwrote a newly created destination");
        require(read(competingDestination / "keep.esm") == before
            && !std::filesystem::exists(competingDestination / "Data"),
            "Failed publication changed the competing destination");
        options = {};
        options.progress = [&](const auto& progress) {
            if (progress.phase != "Writing native world") return;
            auto changed = loadEditableLandscape(world);
            for (auto& h : changed.heights) h = 8;
            saveEditableLandscape(world, changed);
        };
        rejects([&] { (void)exportFo4WorldProof(world, root / "changed", options); }, "Changed input was published");
        require(!std::filesystem::exists(root / "changed"), "Source change published output");
        saveEditableLandscape(world, createEditableLandscape());
        // The native writer cannot flatten arbitrary extra records; preflight must say so.
        const auto unknownWorld = root / "unknown.esm";
        std::filesystem::copy_file(world, unknownWorld);
        {
            std::ofstream append(unknownWorld, std::ios::binary | std::ios::app);
            const std::string unknown = std::string("ZZZZ") + std::string(20, '\0');
            append.write(unknown.data(), unknown.size());
        }
        require(!inspectFo4Export(unknownWorld).canWriteWorldProof(), "Unknown authored record was silently discarded");
        const auto overlay = createEditorOverlay(world);
        require(!inspectFo4Export(overlay).canWriteWorldProof(), "Unimplemented overlay export was accepted");
        const auto exported2 = exportFo4WorldProof(world, root / "package-2");
        require(read(exported2.pluginPath) == pluginBefore, "Unchanged source did not generate deterministic plugin bytes");
        auto lowTerrain = createEditableLandscape();
        for (auto& height : lowTerrain.heights) height = -4096;
        const auto dryWorld = root / "dry.esm";
        saveEditableLandscape(dryWorld, lowTerrain);
        const auto dryPackage = exportFo4WorldProof(dryWorld, root / "dry-package");
        EsmDocument dryNative(dryPackage.pluginPath);
        require(dryNative.references().front().placement.position[2] == -4032,
            "Dry terrain below default water height was treated as flooded");
#ifdef _WIN32
        std::jthread unlock;
        options = {};
        options.progress = [&](const auto& progress) {
            if (progress.phase != "Publishing package") return;
            for (const auto& entry : std::filesystem::directory_iterator(root))
                if (entry.path().filename().string().starts_with("shared-directory.incomplete-"))
                {
                    const auto handle = CreateFileW(entry.path().c_str(), FILE_LIST_DIRECTORY,
                        FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, nullptr);
                    require(handle != INVALID_HANDLE_VALUE, "Cannot simulate scanner publication lock");
                    unlock = std::jthread([handle] {
                        std::this_thread::sleep_for(std::chrono::milliseconds(150)); CloseHandle(handle);
                    });
                    return;
                }
            throw std::runtime_error("Missing publication staging directory");
        };
        const auto shared = exportFo4WorldProof(world, root / "shared-directory", options);
        require(std::filesystem::exists(shared.pluginPath), "Transient Windows sharing lock prevented publication");
#endif
    }
}

int main(int argc, char** argv)
{
    std::filesystem::path root;
    try
    {
        if (argc == 4 && std::string(argv[1]) == "--export-proof")
        {
            Fo4ExportOptions options; options.dataDirectory = findFallout4Data();
            options.progress = [](const auto& progress) {
                std::cout << progress.phase << ": " << progress.completed << '/' << progress.total << '\n';
            };
            const auto result = exportFo4WorldProof(argv[2], argv[3], options);
            std::cout << "World geometry proof (no navigation/LOD certification): " << result.pluginPath.string() << '\n';
            return 0;
        }
        if (argc != 1) throw std::runtime_error("Usage: openf4-fo4-export-tests [--export-proof world.esm new-package-directory]");
#ifndef _WIN32
        std::cout << "Windows proof hashing is required\n"; return 77;
#endif
        root = std::filesystem::temp_directory_path() / ("openf4-fo4-export-" +
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directory(root);
        tests(root);
        std::filesystem::remove_all(root);
        std::cout << "Fallout export tests passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << "\nRetained test artifacts: " << root.string() << '\n';
        return 1;
    }
}
