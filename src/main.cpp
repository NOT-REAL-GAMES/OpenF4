#include "assets/ba2_archive.hpp"
#include "assets/asset_resolver.hpp"
#include "assets/dem_import.hpp"
#include "assets/esm_document.hpp"
#include "assets/fo4_install.hpp"
#include "assets/nif_model.hpp"
#include "physics/jolt_collision.hpp"
#include "render/mesh_builder.hpp"
#include "render/vulkan_renderer.hpp"
#include "world/world_scene.hpp"

#include <SDL3/SDL.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <limits>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>

namespace
{
    constexpr double worldCellSize = 4096.0;
    bool headless = false;
    std::optional<std::filesystem::path> compatibilityOutput;

    void jsonNumber(std::ostream& out, double value)
    {
        if (std::isfinite(value)) out << value;
        else out << "null";
    }

    template<class Vector> void jsonVector(std::ostream& out, const Vector& value)
    {
        out << '[';
        for (int i = 0; i < value.length(); ++i)
        {
            if (i) out << ',';
            jsonNumber(out, value[i]);
        }
        out << ']';
    }

    void jsonMatrix(std::ostream& out, const glm::mat4& value)
    {
        out << '[';
        for (int column = 0; column < 4; ++column)
        {
            if (column) out << ',';
            jsonVector(out, value[column]);
        }
        out << ']';
    }

    double matrixDrift(const glm::mat4& a, const glm::mat4& b)
    {
        double result = 0;
        for (int c = 0; c < 4; ++c)
            for (int r = 0; r < 4; ++r)
            {
                if (!std::isfinite(a[c][r]) || !std::isfinite(b[c][r]))
                    return std::numeric_limits<double>::infinity();
                result = std::max(result, std::abs(static_cast<double>(a[c][r]) - b[c][r]));
            }
        return result;
    }

    struct CompatibilityRun
    {
        using State = openf4::world::ObjectCompatibilityState;
        static constexpr int simulationSteps = 360;
        static constexpr double groundTolerance = 4.2;
        struct Reference
        {
            std::uint32_t formId = 0;
            std::uint32_t baseId = 0;
            std::string model;
            bool movable = false;
            State initial;
            State final;
            double staticDrift = 0;
            int restingSteps = 0;
        };
        std::filesystem::path world;
        std::vector<Reference> references;
        std::chrono::steady_clock::time_point started = std::chrono::steady_clock::now();
        bool ready = false;
        bool done = false;
        bool passed = false;
        int steps = 0;
        std::string reason;

        CompatibilityRun(const std::filesystem::path& path,
            const openf4::assets::ESMMasterStack& records) : world(path)
        {
            for (const auto& record : records.references)
            {
                if (record.deleted) continue;
                const auto model = std::ranges::find(records.models, record.placement.baseObject,
                    &openf4::assets::ESMModel::formId);
                if (model == records.models.end())
                    throw std::runtime_error("Compatibility reference has no base model");
                const auto source = model->sourceEsm.empty() ? path : std::filesystem::path(model->sourceEsm);
                if (!std::filesystem::equivalent(source, path))
                    throw std::runtime_error("Compatibility fixtures must own every reference's model");
                references.push_back({record.placement.formId, record.placement.baseObject, model->path,
                    model->kind == openf4::assets::ESMModelKind::movableStatic});
            }
            if (references.empty()) throw std::runtime_error("Compatibility world has no references");
        }

        void fail(std::string message)
        {
            done = true; passed = false; reason = std::move(message);
        }

        bool beforeStep(const openf4::world::ObjectScene& objects, std::size_t terrainPending)
        {
            if (done) return false;
            if (std::chrono::steady_clock::now() - started > std::chrono::seconds(60))
            {
                fail("Timed out waiting for fixture residency or post-readiness simulation");
                return false;
            }
            if (!objects.diagnostics().empty())
            {
                fail("Runtime asset diagnostics were emitted");
                return false;
            }
            bool allReady = objects.pending() == 0 && terrainPending == 0;
            for (auto& reference : references)
            {
                reference.final = objects.compatibilityState(reference.formId);
                const auto& state = reference.final;
                if (state.collisionFailed)
                {
                    fail("Fixture native collision failed to install");
                    return false;
                }
                if (state.installed && state.collisionReady && state.expectedBodies == 0)
                {
                    fail("Fixture installed without any native collision bodies");
                    return false;
                }
                allReady = allReady && state.installed && state.collisionReady && state.surroundingsReady
                    && state.expectedBodies > 0 && state.bodies.size() == state.expectedBodies
                    && state.movable == reference.movable;
            }
            if (!allReady)
            {
                if (ready) fail("Fixture lost residency/readiness during its measured simulation");
                return false;
            }
            if (!ready)
            {
                ready = true;
                for (auto& reference : references) reference.initial = reference.final;
            }
            return true;
        }

        void afterStep(const openf4::world::ObjectScene& objects)
        {
            ++steps;
            for (auto& reference : references)
            {
                reference.final = objects.compatibilityState(reference.formId);
                const auto& state = reference.final;
                if (!state.installed || !state.collisionReady || !state.surroundingsReady
                    || state.bodies.size() != reference.initial.bodies.size())
                {
                    fail("Fixture lost installed render or physics state");
                    return;
                }
                if (!reference.movable)
                {
                    reference.staticDrift = std::max(reference.staticDrift,
                        matrixDrift(state.transform, reference.initial.transform));
                    for (std::size_t i = 0; i < state.bodies.size(); ++i)
                        reference.staticDrift = std::max(reference.staticDrift,
                            matrixDrift(state.bodies[i].transform, reference.initial.bodies[i].transform));
                    if (reference.staticDrift > 0.001)
                    {
                        fail("STAT render or native body moved during simulation");
                        return;
                    }
                }
                else
                {
                    const auto& body = state.bodies.front();
                    const bool onLand = state.geometry && state.landHeight
                        && std::abs(state.geometry->minimum.y - *state.landHeight) <= groundTolerance;
                    const bool resting = onLand && glm::length(body.linearVelocity) < 3.5f
                        && glm::length(body.angularVelocity) < 0.05f
                        && std::isfinite(matrixDrift(body.transform, reference.initial.bodies.front().transform));
                    reference.restingSteps = resting ? reference.restingSteps + 1 : 0;
                }
            }
            if (steps < simulationSteps) return;
            for (const auto& reference : references)
                if (reference.movable)
                {
                    const auto& initialBody = reference.initial.bodies.front();
                    const auto& finalBody = reference.final.bodies.front();
                    if (reference.restingSteps < 30
                        || !(initialBody.transform[3].y - finalBody.transform[3].y > 1.0f))
                    {
                        fail("MSTT did not fall and remain resting on LAND for the final 30 steps");
                        return;
                    }
                }
            passed = done = true;
            reason = "All expected references and native bodies installed; STAT immobile; MSTT fell and rested on LAND";
        }

        static void writeState(std::ostream& out, const State& state)
        {
            out << "{\"installed\":" << state.installed << ",\"movable\":" << state.movable
                << ",\"collisionReady\":" << state.collisionReady
                << ",\"surroundingsReady\":" << state.surroundingsReady
                << ",\"collisionFailed\":" << state.collisionFailed
                << ",\"expectedBodies\":" << state.expectedBodies << ",\"transform\":";
            jsonMatrix(out, state.transform);
            out << ",\"authoredTransform\":"; jsonMatrix(out, state.authoredTransform);
            out << ",\"landHeight\":";
            if (state.landHeight) jsonNumber(out, *state.landHeight); else out << "null";
            out << ",\"geometryMinimum\":";
            if (state.geometry) jsonVector(out, state.geometry->minimum); else out << "null";
            out << ",\"geometryMaximum\":";
            if (state.geometry) jsonVector(out, state.geometry->maximum); else out << "null";
            out << ",\"bodies\":[";
            for (std::size_t i = 0; i < state.bodies.size(); ++i)
            {
                if (i) out << ',';
                out << "{\"transform\":"; jsonMatrix(out, state.bodies[i].transform);
                out << ",\"linearVelocity\":"; jsonVector(out, state.bodies[i].linearVelocity);
                out << ",\"angularVelocity\":"; jsonVector(out, state.bodies[i].angularVelocity);
                out << '}';
            }
            out << "]}";
        }

        void write(const openf4::world::ObjectScene& objects, std::size_t terrainPending, int frames)
        {
            if (!done) fail("Compatibility run was interrupted before completing its checks");
            std::ofstream out(*compatibilityOutput / "runtime.json", std::ios::binary);
            out << std::boolalpha << std::setprecision(17)
                << "{\"schema\":1,\"status\":" << openf4::assets::compatibilityJsonString(passed ? "Pass" : "Fail")
                << ",\"reason\":" << openf4::assets::compatibilityJsonString(reason)
                << ",\"world\":" << openf4::assets::compatibilityJsonString(world.generic_string())
                << ",\"ready\":" << ready << ",\"renderedFrames\":" << frames
                << ",\"simulationSteps\":" << steps << ",\"requiredSimulationSteps\":" << simulationSteps
                << ",\"stepSeconds\":" << 1.0 / 60.0 << ",\"groundToleranceUnits\":" << groundTolerance
                << ",\"groundCheck\":\"fell_and_resting_on_land_not_contact_callback\""
                << ",\"pendingObjects\":" << objects.pending() << ",\"pendingTerrain\":" << terrainPending
                << ",\"loadedObjects\":" << objects.loaded() << ",\"diagnostics\":[";
            for (std::size_t i = 0; i < objects.diagnostics().size(); ++i)
            {
                if (i) out << ',';
                out << openf4::assets::compatibilityJsonString(objects.diagnostics()[i]);
            }
            out << "],\"references\":[";
            for (std::size_t i = 0; i < references.size(); ++i)
            {
                if (i) out << ',';
                const auto& reference = references[i];
                out << "{\"formId\":" << reference.formId << ",\"baseId\":" << reference.baseId
                    << ",\"modelPath\":" << openf4::assets::compatibilityJsonString(reference.model)
                    << ",\"kind\":" << openf4::assets::compatibilityJsonString(reference.movable ? "MSTT" : "STAT")
                    << ",\"maximumStaticDrift\":";
                jsonNumber(out, reference.staticDrift);
                out << ",\"consecutiveRestingSteps\":" << reference.restingSteps << ",\"initial\":";
                writeState(out, reference.initial);
                out << ",\"final\":"; writeState(out, objects.compatibilityState(reference.formId));
                out << '}';
            }
            out << "]}\n";
            out.close();
            if (!out) throw std::runtime_error("Cannot write runtime compatibility evidence");
        }
    };

    glm::dvec3 cellAlignedRenderOrigin(const glm::dvec3& position)
    {
        const double cellX = std::floor(-position.x / worldCellSize);
        const double cellY = std::floor(position.z / worldCellSize);
        return {-cellX * worldCellSize, 0.0, cellY * worldCellSize};
    }

    struct SdlGuard
    {
        SdlGuard()
        {
            if (!SDL_Init(SDL_INIT_VIDEO))
                throw std::runtime_error(SDL_GetError());
        }
        ~SdlGuard() { SDL_Quit(); }
    };

    struct ArchivedModel
    {
        std::string path;
        std::vector<std::byte> bytes;
    };

    std::optional<ArchivedModel> loadArchivedModel(
        const std::filesystem::path& data, const std::optional<std::string>& requested)
    {
        constexpr std::array archives{"Fallout4 - Meshes.ba2", "Fallout4 - MeshesExtra.ba2"};
        for (const char* archiveName : archives)
        {
            const std::filesystem::path archivePath = data / archiveName;
            if (!std::filesystem::is_regular_file(archivePath))
                continue;
            openf4::assets::Ba2Archive archive(archivePath);
            if (requested)
            {
                if (archive.contains(*requested))
                    return ArchivedModel{*requested, archive.extract(*requested)};
                continue;
            }

            std::vector<std::string> candidates;
            for (const auto& [path, entry] : archive.entries())
            {
                const bool usefulSize = entry.unpackedSize >= 20'000 && entry.unpackedSize <= 1'000'000;
                if (usefulSize && path.starts_with("meshes\\architecture\\") && path.ends_with(".nif")
                    && path.find("lod") == std::string::npos
                    && path.find("precombined") == std::string::npos
                    && path.find("marker") == std::string::npos)
                    candidates.push_back(path);
            }
            if (!candidates.empty())
            {
                std::ranges::sort(candidates);
                return ArchivedModel{candidates.front(), archive.extract(candidates.front())};
            }
        }
        if (requested)
            throw std::runtime_error("Model was not found in Fallout 4 mesh archives: " + *requested);
        return std::nullopt;
    }

    openf4::render::PackedMesh makeOverlayMesh(std::span<const glm::vec2> points,
        std::span<const std::uint32_t> indices, std::uint32_t material, float depth)
    {
        std::vector<openf4::render::SourceVertex> vertices;
        vertices.reserve(points.size());
        for (const glm::vec2 point : points)
            vertices.push_back({glm::vec3(point, depth)});
        return openf4::render::buildMeshlets(vertices, indices, material);
    }
}

int main(int argc, char** argv)
try
{
    std::cout << std::unitbuf;
    std::cerr << std::unitbuf;
    const char* compatibilityEnvironment = std::getenv("OPENF4_COMPATIBILITY_CAPTURE_DIR");
    const bool compatibilityRequested = compatibilityEnvironment && *compatibilityEnvironment;
    if (compatibilityRequested) headless = true; // Never block an automated failure on a dialog.
    int benchmarkFrames = 0;
    bool cubesOnly = false;
    bool resizeTest = false;
    bool cellCrossTest = false;
    bool landOnly = false;
    bool physicsSample = false;
    bool editorConflict = false;
    int worldRadius = 1;
    std::size_t maxObjects = std::numeric_limits<std::size_t>::max();
    std::optional<std::string> requestedModel;
    std::optional<openf4::assets::CellCoord> requestedCell;
    std::optional<std::filesystem::path> editorPath;
    std::optional<std::filesystem::path> masterPath;
    std::optional<std::filesystem::path> demImportPath;
    std::array<std::filesystem::path, 4> demSources;
    bool inspectDemOnly = false;
    bool bakeDemBtr = false;
    for (int index = 1; index < argc; ++index)
    {
        if (std::string_view(argv[index]) == "--import-dem"
            || std::string_view(argv[index]) == "--inspect-dem")
        {
            if (index != 1 || (argc != 7 && argc != 8))
                throw std::runtime_error(
                    "DEM import expects an output .esm followed by exactly four GeoTIFF paths");
            inspectDemOnly = std::string_view(argv[index]) == "--inspect-dem";
            if (index + 1 < argc && std::string_view(argv[index + 1]) == "--bake-btr")
            {
                if (inspectDemOnly)
                    throw std::runtime_error("--bake-btr is only valid with --import-dem");
                bakeDemBtr = true;
                ++index;
            }
            if ((!bakeDemBtr && argc != 7) || (bakeDemBtr && argc != 8))
                throw std::runtime_error(
                    "--bake-btr must immediately follow --import-dem");
            demImportPath = argv[++index];
            for (auto& source : demSources)
                source = argv[++index];
        }
        else if (std::string_view(argv[index]) == "--benchmark")
        {
            editorConflict = true;
            if (++index >= argc)
                throw std::runtime_error("--benchmark expects a positive frame count");
            const std::string_view value(argv[index]);
            const auto result = std::from_chars(value.data(), value.data() + value.size(), benchmarkFrames);
            if (result.ec != std::errc{} || benchmarkFrames < 1)
                throw std::runtime_error("--benchmark expects a positive frame count");
        }
        else if (std::string_view(argv[index]) == "--model")
        {
            editorConflict = true;
            if (++index >= argc)
                throw std::runtime_error("--model expects an archive-relative NIF path");
            requestedModel = argv[index];
        }
        else if (std::string_view(argv[index]) == "--cubes")
        {
            editorConflict = true;
            cubesOnly = true;
        }
        else if (std::string_view(argv[index]) == "--resize-test")
        {
            editorConflict = true;
            resizeTest = true;
        }
        else if (std::string_view(argv[index]) == "--cell-cross-test")
        {
            editorConflict = true;
            cellCrossTest = true;
        }
        else if (std::string_view(argv[index]) == "--land-only")
        {
            editorConflict = true;
            landOnly = true;
        }
        else if (std::string_view(argv[index]) == "--physics-sample")
        {
            editorConflict = true;
            physicsSample = true;
        }
        else if (std::string_view(argv[index]) == "--radius")
        {
            editorConflict = true;
            if (++index >= argc)
                throw std::runtime_error("--radius expects a cell radius");
            const std::string_view value(argv[index]);
            const auto result = std::from_chars(value.data(), value.data() + value.size(), worldRadius);
            if (result.ec != std::errc{} || worldRadius < 0 || worldRadius > 8)
                throw std::runtime_error("--radius expects a value from 0 through 8");
        }
        else if (std::string_view(argv[index]) == "--max-objects")
        {
            editorConflict = true;
            if (++index >= argc)
                throw std::runtime_error("--max-objects expects a positive count");
            const std::string_view value(argv[index]);
            const auto result = std::from_chars(value.data(), value.data() + value.size(), maxObjects);
            if (result.ec != std::errc{} || maxObjects == 0)
                throw std::runtime_error("--max-objects expects a positive count");
        }
        else if (std::string_view(argv[index]) == "--cell")
        {
            editorConflict = true;
            if (index + 2 >= argc)
                throw std::runtime_error("--cell expects X and Y coordinates");
            openf4::assets::CellCoord cell;
            for (int component = 0; component < 2; ++component)
            {
                const std::string_view value(argv[++index]);
                auto& destination = component == 0 ? cell.x : cell.y;
                const auto result = std::from_chars(value.data(), value.data() + value.size(), destination);
                if (result.ec != std::errc{})
                    throw std::runtime_error("--cell expects integer X and Y coordinates");
            }
            requestedCell = cell;
        }
        else if (std::string_view(argv[index]) == "--editor")
        {
            if (++index >= argc)
                throw std::runtime_error("--editor expects an output .esm path");
            if (editorPath)
                throw std::runtime_error("--editor may only be specified once");
            editorPath = argv[index];
        }
        else if (index == 1
            && !std::string_view(argv[index]).starts_with('-'))
        {
            std::filesystem::path candidate = std::filesystem::absolute(argv[index]);
            std::string extension = candidate.extension().string();
            std::ranges::transform(extension, extension.begin(), [](unsigned char value) {
                return static_cast<char>(std::tolower(value));
            });
            if (extension != ".esm" || !std::filesystem::is_regular_file(candidate))
                throw std::runtime_error(
                    "The positional argument must be an existing master .esm file");
            masterPath = std::move(candidate);
        }
        else
            throw std::runtime_error("Unknown command line option: " + std::string(argv[index]));
    }
    if (editorPath && editorConflict)
        throw std::runtime_error(
            "--editor is mutually exclusive with model, physics, benchmark, and streamed-world options");
    if (compatibilityRequested)
    {
        if (!masterPath || editorPath || demImportPath || cubesOnly || requestedModel
            || physicsSample || landOnly || cellCrossTest || resizeTest || requestedCell)
            throw std::runtime_error("Compatibility capture requires only a fixture master ESM (and optional --benchmark)");
        openf4::assets::EsmDocument fixture(*masterPath);
        if (!fixture.isOpenF4Owned())
            throw std::runtime_error("Compatibility capture requires an OpenF4-owned fixture world");
        openf4::assets::beginCompatibilityAssetCapture(*masterPath, compatibilityEnvironment);
        compatibilityOutput = std::filesystem::absolute(compatibilityEnvironment);
    }
    if (demImportPath)
    {
        headless = true;
        const auto summary = inspectDemOnly
            ? openf4::assets::inspectDemLandscape(*demImportPath, demSources)
            : openf4::assets::importDemLandscape(
                *demImportPath, demSources, {.bakeBtr = bakeDemBtr});
        std::cout << (inspectDemOnly ? "DEM import preflight: " : "DEM import complete: ")
                  << summary.cellCount << " cells in "
                  << summary.shards.size() << " terrain ESMs ("
                  << summary.widthMeters / 1000.0 << " x "
                  << summary.heightMeters / 1000.0 << " km, estimated "
                  << summary.estimatedBytes / (1024.0 * 1024.0 * 1024.0) << " GiB)\n";
        for (std::size_t index = 0; index < summary.shards.size(); ++index)
            std::cout << "  shard " << index << ": " << summary.shards[index].cellCount
                      << " cells -> " << summary.shards[index].path.string() << '\n';
        if (!inspectDemOnly)
        {
            if (summary.btrCount != 0)
                std::cout << "BTRs: " << summary.btrCount << " level-32 tiles in "
                          << summary.btrDirectory.string() << '\n';
            std::cout << "Open with: openf4.exe --editor \""
                      << summary.editorPath.string() << "\"\n";
        }
        return 0;
    }
    bool demViewer = masterPath
        && openf4::assets::DemLandscapeProject::recognizes(*masterPath);

    SdlGuard sdl;
    SDL_Window* window = SDL_CreateWindow(
        "OpenF4", 1600, 900,
        SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY);
    if (!window)
        throw std::runtime_error(SDL_GetError());

    const auto falloutData = openf4::assets::findFallout4Data();
    auto assetData = falloutData;
    if (masterPath && (std::filesystem::is_regular_file(
            masterPath->parent_path() / "Fallout4 - Meshes.ba2") || !assetData))
        assetData = masterPath->parent_path();
    std::optional<openf4::assets::ESMMasterStack> projectRecords;
    std::optional<std::filesystem::path> terrainMaster = masterPath;
    bool projectTerrain = false;
    if (masterPath)
    {
        openf4::assets::EsmDocument source(*masterPath);
        if (source.isOpenF4Owned())
        {
            projectRecords = openf4::assets::loadEsmMasterStack(*masterPath, assetData);
            terrainMaster = projectRecords->basePath;
            openf4::assets::EsmDocument terrainSource(*terrainMaster);
            projectTerrain = terrainSource.isOpenF4Owned();
            demViewer = projectTerrain;
        }
    }
    if ((editorPath || demViewer) && !falloutData)
        throw std::runtime_error(
            "Landscape loading requires Fallout 4 data (set FALLOUT4_PATH)");
    if (physicsSample && (cubesOnly || requestedModel || !assetData))
        throw std::runtime_error(
            "--physics-sample requires Fallout 4 data and the streamed world scene");
    openf4::render::PackedMesh mesh;
    std::vector<openf4::render::SceneInstance> instances;
    std::vector<openf4::render::SceneMaterial> materials;
    std::vector<openf4::render::SceneTexture> textures;
    glm::vec3 initialPosition{};
    glm::vec3 initialTarget{};
    float sceneScale = 100.0f;
    bool usingNif = false;
    bool usingWorld = false;
    std::unique_ptr<openf4::physics::JoltWorld> physicsWorld;
    std::unique_ptr<openf4::world::WorldStreamer> worldStreamer;
    std::unique_ptr<openf4::world::LandscapeEditor> landscapeEditor;
    std::unique_ptr<openf4::world::ObjectScene> projectObjects;
    std::vector<openf4::assets::PlacedReference> projectReferences;
    std::map<std::uint32_t, std::string> projectModels;
    std::unordered_map<std::uint32_t, std::filesystem::path> projectModelSources;
    bool landscapeEditing = false;
    if (editorPath || demViewer)
    {
        landscapeEditing = editorPath.has_value();
        const auto& landscapePath = landscapeEditing ? *editorPath : *terrainMaster;
        landscapeEditor = std::make_unique<openf4::world::LandscapeEditor>(
            *falloutData, landscapePath, !landscapeEditing);
        auto world = landscapeEditor->takeInitialScene();
        mesh = std::move(world.mesh);
        instances = std::move(world.instances);
        materials = std::move(world.materials);
        textures = std::move(world.textures);
        initialPosition = world.cameraPosition;
        initialTarget = world.cameraTarget;
        sceneScale = world.sceneScale;
        usingWorld = true;
        std::cout << (landscapeEditing ? "Editor: " : "DEM world: ")
                  << std::filesystem::absolute(landscapePath).string()
                  << " (" << world.landscapeCells << " LAND cells, " << materials.size() << " materials, "
                  << textures.size() << " textures)\n";
        if (landscapeEditing)
            std::cout << "Editor controls: 1 raise, 2 lower, 3 smooth, 4 flatten; "
                         "left-drag sculpt; wheel radius; Shift+wheel strength; "
                         "Ctrl+Z undo; Ctrl+Y redo; Ctrl+S save\n";
    }
    else if (!cubesOnly && !requestedModel && assetData)
    {
        worldStreamer = std::make_unique<openf4::world::WorldStreamer>(
            *assetData, requestedCell, worldRadius, !landOnly && !projectRecords,
            maxObjects, terrainMaster);
        auto world = worldStreamer->takeInitialScene();
        mesh = std::move(world.mesh);
        instances = std::move(world.instances);
        materials = std::move(world.materials);
        textures = std::move(world.textures);
        initialPosition = world.cameraPosition;
        initialTarget = world.cameraTarget;
        sceneScale = world.sceneScale;
        usingWorld = true;
        std::cout << "World: cell " << world.centerCell.x << ", " << world.centerCell.y
                  << " radius " << worldRadius << " (" << world.landscapeCells
                  << " LAND cells + " << world.landscapeLodCells << " coarse cells, "
                  << worldStreamer->pendingObjects() << " jobs queued, "
                  << materials.size() << " materials, " << textures.size() << " textures)\n";
    }
    else if (!cubesOnly && falloutData)
    {
        if (auto archived = loadArchivedModel(*falloutData, requestedModel))
        {
            const auto geometry = openf4::assets::loadNifModel(archived->bytes);
            std::size_t vertexCount = 0;
            for (const auto& surface : geometry.surfaces)
            {
                vertexCount += surface.vertices.size();
                (void)openf4::render::appendMesh(mesh,
                    openf4::render::buildMeshlets(surface.vertices, surface.indices));
            }
            usingNif = true;
            std::cout << "NIF: " << archived->path << " (" << vertexCount
                      << " vertices, " << mesh.meshlets.size() << " meshlets)\n";
        }
    }
    if (projectRecords && !landOnly)
    {
        projectObjects = std::make_unique<openf4::world::ObjectScene>(
            assetData.value_or(masterPath->parent_path()), *masterPath);
        for (const auto& model : projectRecords->models)
        {
            projectModels.emplace(model.formId, model.path);
            projectModelSources.emplace(model.formId,
                model.sourceEsm.empty() ? *masterPath : std::filesystem::path(model.sourceEsm));
        }
        for (const auto& reference : projectRecords->references)
            projectReferences.push_back(reference.placement);
    }
    if (!usingWorld)
    {
        if (mesh.meshlets.empty())
            mesh = openf4::render::makeCubeMesh();
        const int radius = usingNif ? worldRadius : 40;
        const float spacing = usingNif ? std::max(mesh.bounds.w * 2.4f, 100.0f) : 4.0f;
        instances.reserve((radius * 2 + 1) * (radius * 2 + 1));
        for (int z = -radius; z <= radius; ++z)
        {
            for (int x = -radius; x <= radius; ++x)
            {
                openf4::render::SceneInstance instance;
                const float height = usingNif ? 0.0f : 1.0f + static_cast<float>((x * x + z * z) % 12);
                const glm::vec3 translation(x * spacing, height, z * spacing);
                instance.transform = glm::translate(glm::mat4(1.0f), translation);
                if (!usingNif)
                    instance.transform *= glm::scale(glm::mat4(1.0f), glm::vec3(1.5f, height, 1.5f));
                const glm::vec3 worldCenter = glm::vec3(
                    instance.transform * glm::vec4(glm::vec3(mesh.bounds), 1.0f));
                const float scale = std::max(glm::length(glm::vec3(instance.transform[0])),
                    std::max(glm::length(glm::vec3(instance.transform[1])),
                        glm::length(glm::vec3(instance.transform[2]))));
                instance.bounds = glm::vec4(worldCenter, mesh.bounds.w * scale);
                instances.push_back(instance);
            }
        }
        sceneScale = spacing * static_cast<float>(radius);
        initialTarget = glm::vec3(mesh.bounds);
        initialTarget.x = -initialTarget.x;
        initialPosition = initialTarget + glm::vec3(0.0f, sceneScale * 0.3f, sceneScale * 0.8f);
    }

    auto renderer = std::make_unique<openf4::render::VulkanRenderer>(
        window, mesh, instances, materials, textures);
    if (worldStreamer)
    {
        physicsWorld = std::make_unique<openf4::physics::JoltWorld>();
        if (const auto target = worldStreamer->enablePhysics(
                *renderer, *physicsWorld, physicsSample))
        {
            initialTarget = *target;
            initialPosition = initialTarget + glm::vec3(500.0f, 320.0f, 500.0f);
        }
    }
    if (projectObjects && !landscapeEditing)
    {
        if (!physicsWorld)
            physicsWorld = std::make_unique<openf4::physics::JoltWorld>();
        if (landscapeEditor)
            landscapeEditor->enablePhysics(*physicsWorld);
        projectObjects->enablePhysics(*physicsWorld, projectRecords->models,
            landscapeEditor.get(), worldStreamer.get());
        projectObjects->enableScripts(*projectRecords);
    }
    std::optional<CompatibilityRun> compatibility;
    if (compatibilityRequested)
    {
        if (!projectObjects || !projectRecords || !landscapeEditor)
            throw std::runtime_error("Compatibility capture requires the normal project terrain/object runtime");
        compatibility.emplace(*masterPath, *projectRecords);
    }
    std::optional<std::uint32_t> physicsPlayButton;
    std::optional<std::uint32_t> physicsPauseButton;
    if (physicsWorld && physicsSample)
    {
        openf4::render::SceneMaterial background;
        background.baseColor = {0.035f, 0.05f, 0.06f, 1.0f};
        openf4::render::SceneMaterial icon;
        icon.baseColor = {0.28f, 0.95f, 0.55f, 1.0f};
        const std::uint32_t backgroundMaterial = renderer->addMaterial(background);
        const std::uint32_t iconMaterial = renderer->addMaterial(icon);
        const auto addOverlay = [&](openf4::render::PackedMesh overlay, bool active) {
            const auto range = renderer->addMesh(std::move(overlay));
            openf4::render::SceneInstance instance;
            instance.bounds = range.bounds;
            instance.meshletOffset = range.meshletOffset;
            instance.meshletCount = range.meshletCount;
            instance.active = active;
            instance.overlay = true;
            return renderer->addInstance(instance);
        };
        constexpr std::array backgroundPoints{
            glm::vec2{16.0f, 16.0f}, glm::vec2{16.0f, 72.0f},
            glm::vec2{72.0f, 72.0f}, glm::vec2{72.0f, 16.0f}};
        constexpr std::array quadIndices{
            0u, 1u, 2u, 0u, 2u, 3u, 0u, 2u, 1u, 0u, 3u, 2u};
        (void)addOverlay(makeOverlayMesh(
            backgroundPoints, quadIndices, backgroundMaterial, 0.999f), true);
        constexpr std::array playPoints{
            glm::vec2{34.0f, 29.0f}, glm::vec2{34.0f, 59.0f},
            glm::vec2{59.0f, 44.0f}};
        constexpr std::array triangleIndices{0u, 1u, 2u, 0u, 2u, 1u};
        physicsPlayButton = addOverlay(
            makeOverlayMesh(playPoints, triangleIndices, iconMaterial, 1.0f), true);
        constexpr std::array pausePoints{
            glm::vec2{32.0f, 29.0f}, glm::vec2{32.0f, 59.0f},
            glm::vec2{40.0f, 59.0f}, glm::vec2{40.0f, 29.0f},
            glm::vec2{48.0f, 29.0f}, glm::vec2{48.0f, 59.0f},
            glm::vec2{56.0f, 59.0f}, glm::vec2{56.0f, 29.0f}};
        constexpr std::array pauseIndices{
            0u, 1u, 2u, 0u, 2u, 3u, 0u, 2u, 1u, 0u, 3u, 2u,
            4u, 5u, 6u, 4u, 6u, 7u, 4u, 6u, 5u, 4u, 7u, 6u};
        physicsPauseButton = addOverlay(
            makeOverlayMesh(pausePoints, pauseIndices, iconMaterial, 1.0f), false);
    }
    std::cout << "GPU: " << renderer->deviceName() << '\n';
    std::cout << "Textures: " << renderer->stats().residentTextures << " resident, "
              << renderer->stats().sparseTextures << " sparse\n";
    std::cout << "Texture memory: "
              << renderer->stats().textureResidentBytes / (1024.0 * 1024.0) << " / "
              << renderer->stats().textureVirtualBytes / (1024.0 * 1024.0)
              << " MiB resident / virtual\n";
    std::cout << "Geometry: " << mesh.vertices.size() << " vertices, "
              << mesh.meshlets.size() << " meshlets, " << instances.size()
              << " instances, " << renderer->stats().sourceCandidates << " candidates\n";
    if (falloutData)
        std::cout << "Fallout 4 Data: " << falloutData->string() << '\n';
    else
        std::cout << "Fallout 4 Data: not found (set FALLOUT4_PATH)\n";
    if (masterPath)
        std::cout << "Master ESM: " << masterPath->string() << '\n';

    bool running = true;
    bool physicsRunning = projectObjects && !landscapeEditing;
    bool editorSculpting = false;
    bool mouseCaptured = !physicsPlayButton && !compatibility;
    SDL_SetWindowRelativeMouseMode(window, mouseCaptured);
    const auto setPhysicsRunning = [&](bool active) {
        physicsRunning = active;
        if (!physicsPlayButton || !physicsPauseButton)
            return;
        const std::array play{*physicsPlayButton};
        const std::array pause{*physicsPauseButton};
        renderer->setInstancesActive(play, !active);
        renderer->setInstancesActive(pause, active);
    };
    glm::dvec3 position = initialPosition;
    const glm::vec3 initialForward = glm::normalize(initialTarget - initialPosition);
    float yaw = std::atan2(initialForward.x, initialForward.z);
    float pitch = std::asin(initialForward.y);
    auto previous = std::chrono::steady_clock::now();
    auto lastTitleUpdate = previous;
    int renderedFrames = 0;
    std::vector<double> benchmarkFrameTimes;
    std::vector<double> benchmarkStreamTimes;
    benchmarkFrameTimes.reserve(static_cast<std::size_t>(benchmarkFrames));
    benchmarkStreamTimes.reserve(static_cast<std::size_t>(benchmarkFrames));
    const int cellCrossFrame = benchmarkFrames > 180 ? benchmarkFrames - 60 : 60;

    while (running)
    {
        const auto frameStart = std::chrono::steady_clock::now();
        SDL_Event event;
        while (SDL_PollEvent(&event))
        {
            if (event.type == SDL_EVENT_QUIT)
                running = false;
            else if (event.type == SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED)
                renderer->resize();
            else if (compatibility)
                continue;
            else if (event.type == SDL_EVENT_KEY_DOWN && event.key.key == SDLK_ESCAPE)
            {
                mouseCaptured = !mouseCaptured;
                SDL_SetWindowRelativeMouseMode(window, mouseCaptured);
            }
            else if (event.type == SDL_EVENT_KEY_DOWN && landscapeEditing
                && landscapeEditor && !event.key.repeat)
            {
                const bool control = (event.key.mod & SDL_KMOD_CTRL) != 0;
                if (control && event.key.key == SDLK_S)
                    landscapeEditor->save(*renderer);
                else if (control && event.key.key == SDLK_Z)
                    (void)landscapeEditor->undo(*renderer);
                else if (control && event.key.key == SDLK_Y)
                    (void)landscapeEditor->redo(*renderer);
                else if (event.key.key == SDLK_1)
                    landscapeEditor->setTool(openf4::world::LandscapeTool::raise);
                else if (event.key.key == SDLK_2)
                    landscapeEditor->setTool(openf4::world::LandscapeTool::lower);
                else if (event.key.key == SDLK_3)
                    landscapeEditor->setTool(openf4::world::LandscapeTool::smooth);
                else if (event.key.key == SDLK_4)
                    landscapeEditor->setTool(openf4::world::LandscapeTool::flatten);
            }
            else if (event.type == SDL_EVENT_MOUSE_WHEEL && landscapeEditing
                && landscapeEditor)
            {
                if ((SDL_GetModState() & SDL_KMOD_SHIFT) != 0)
                    landscapeEditor->adjustStrength(event.wheel.y);
                else
                    landscapeEditor->adjustRadius(event.wheel.y);
            }
            else if (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN && landscapeEditing
                && landscapeEditor
                && event.button.button == SDL_BUTTON_LEFT)
                editorSculpting = true;
            else if (event.type == SDL_EVENT_MOUSE_BUTTON_UP && landscapeEditing
                && landscapeEditor
                && event.button.button == SDL_BUTTON_LEFT)
                editorSculpting = false;
            else if (event.type == SDL_EVENT_KEY_DOWN && event.key.key == SDLK_P
                && !event.key.repeat && physicsPlayButton)
                setPhysicsRunning(!physicsRunning);
            else if (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN
                && event.button.button == SDL_BUTTON_LEFT && !mouseCaptured
                && physicsPlayButton && event.button.x >= 16.0f && event.button.x <= 72.0f
                && event.button.y >= 16.0f && event.button.y <= 72.0f)
                setPhysicsRunning(!physicsRunning);
            else if (event.type == SDL_EVENT_MOUSE_MOTION && mouseCaptured)
            {
                yaw -= event.motion.xrel * 0.002f;
                pitch = std::clamp(pitch - event.motion.yrel * 0.002f, -1.5f, 1.5f);
            }
        }

        const auto now = std::chrono::steady_clock::now();
        const float delta = std::min(std::chrono::duration<float>(now - previous).count(), 0.1f);
        previous = now;
        const glm::vec3 forward(
            std::cos(pitch) * std::sin(yaw), std::sin(pitch), std::cos(pitch) * std::cos(yaw));
        const glm::vec3 right = glm::normalize(glm::cross(forward, glm::vec3(0.0f, 1.0f, 0.0f)));
        const bool* keys = SDL_GetKeyboardState(nullptr);
        const float baseSpeed = std::max(25.0f, sceneScale * 0.15f);
        const float speed = compatibility ? 0.0f
            : baseSpeed * (keys[SDL_SCANCODE_LSHIFT] ? 4.0f : 1.0f) * delta;
        position += glm::dvec3(forward) * static_cast<double>(speed)
            * static_cast<double>(keys[SDL_SCANCODE_W] - keys[SDL_SCANCODE_S]);
        position += glm::dvec3(right) * static_cast<double>(speed)
            * static_cast<double>(keys[SDL_SCANCODE_D] - keys[SDL_SCANCODE_A]);
        position.y += speed * static_cast<float>(keys[SDL_SCANCODE_SPACE] - keys[SDL_SCANCODE_C]);

        const auto streamStart = std::chrono::steady_clock::now();
        if (physicsWorld && physicsRunning && !projectObjects)
            physicsWorld->step(delta);
        if (landscapeEditor)
            landscapeEditor->update(*renderer, position, forward, delta, editorSculpting);
        if (projectObjects)
            projectObjects->update(*renderer, projectReferences, projectModels,
                projectModelSources, position);
        if (worldStreamer)
        {
            worldStreamer->update(*renderer, position);
            if (renderedFrames > 0 && worldStreamer->pendingObjects() > 0)
            {
                constexpr std::size_t maxStreamItemsPerFrame = 256;
                constexpr auto streamCpuBudget = std::chrono::milliseconds(2);
                const auto deadline = std::chrono::steady_clock::now() + streamCpuBudget;
                for (std::size_t item = 0; item < maxStreamItemsPerFrame; ++item)
                {
                    if (!worldStreamer->streamNext(*renderer))
                        break;
                    if (std::chrono::steady_clock::now() >= deadline)
                        break;
                }
            }
        }
        if (compatibility)
        {
            if (compatibility->beforeStep(*projectObjects, landscapeEditor->pendingWork()))
            {
                projectObjects->stepPhysics(*renderer, 1.0f / 60.0f);
                compatibility->afterStep(*projectObjects);
            }
            if (compatibility->done) running = false;
        }
        else if (projectObjects && physicsRunning)
            projectObjects->stepPhysics(*renderer, delta);
        const auto streamEnd = std::chrono::steady_clock::now();

        int width = 0;
        int height = 0;
        SDL_GetWindowSizeInPixels(window, &width, &height);
        if (width == 0 || height == 0)
        {
            SDL_Delay(16);
            continue;
        }
        const glm::dvec3 worldOrigin = usingWorld
            ? cellAlignedRenderOrigin(position) : glm::dvec3(0.0);
        const glm::vec3 localPosition(position - worldOrigin);
        const auto view = glm::lookAtRH(
            localPosition, localPosition + forward, glm::vec3(0.0f, 1.0f, 0.0f));
        auto projection = usingWorld
            ? openf4::render::reverseZInfinitePerspectiveRH_ZO(glm::radians(70.0f),
                static_cast<float>(width) / static_cast<float>(height), 0.1f)
            : openf4::render::reverseZPerspectiveRH_ZO(glm::radians(70.0f),
                static_cast<float>(width) / static_cast<float>(height), 0.1f,
                std::max(5000.0f, sceneScale * 8.0f));
        projection[1][1] *= -1.0f;
        renderer->render(view, projection, localPosition, glm::vec3(worldOrigin));
        if (benchmarkFrames > 0)
        {
            benchmarkFrameTimes.push_back(
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - frameStart).count());
            benchmarkStreamTimes.push_back(
                std::chrono::duration<double, std::milli>(streamEnd - streamStart).count());
        }
        ++renderedFrames;
        if (cellCrossTest && renderedFrames == cellCrossFrame)
            position.x -= 4096.0f;
        if (resizeTest && renderedFrames == 10)
            SDL_SetWindowSize(window, 1280, 720);
        if (!compatibility && benchmarkFrames > 0 && renderedFrames >= benchmarkFrames)
            running = false;

        if (now - lastTitleUpdate > std::chrono::milliseconds(250))
        {
            const auto& stats = renderer->stats();
            const std::string title = landscapeEditing && landscapeEditor
                ? landscapeEditor->status()
                : "OpenF4 | CPU " + std::to_string(stats.cpuFrameMs).substr(0, 5)
                + " ms | GPU " + std::to_string(stats.gpuFrameMs).substr(0, 5) + " ms"
                + (physicsPlayButton
                    ? (physicsRunning ? " | Physics running" : " | Physics paused") : "")
                + (worldStreamer ? " | loaded "
                    + std::to_string(worldStreamer->streamedObjects()) : "");
            SDL_SetWindowTitle(window, title.c_str());
            lastTitleUpdate = now;
        }
    }

    const auto finalStats = renderer->stats();
    if (compatibility)
        compatibility->write(*projectObjects, landscapeEditor->pendingWork(), renderedFrames);
    if (landscapeEditing && landscapeEditor && landscapeEditor->dirty())
        landscapeEditor->save(*renderer);
    if (projectObjects)
        std::cout << "Project objects: " << projectObjects->loaded() << " loaded, "
                  << projectObjects->pending() << " pending\n";
    if (projectObjects && physicsWorld)
        std::cout << "Project physics: " << physicsWorld->dynamicBodyPositions().size()
                  << " resident movable bodies\n";
    if (landscapeEditor)
        std::cout << "Terrain: " << landscapeEditor->residentCells() << " resident cells, "
                  << landscapeEditor->pendingWork() << " pending\n";
    if (worldStreamer)
        std::cout << "Streamer final: " << worldStreamer->streamedObjects()
                  << " loaded, " << worldStreamer->pendingObjects() << " pending\n";
    const auto reportTimings = [](std::string_view label, std::vector<double> values) {
        if (values.empty())
            return;
        std::ranges::sort(values);
        double total = 0.0;
        for (const double value : values)
            total += value;
        const std::size_t p95 = std::min(values.size() - 1,
            static_cast<std::size_t>(static_cast<double>(values.size()) * 0.95));
        std::cout << label << ": mean " << total / static_cast<double>(values.size())
                  << " ms, p95 " << values[p95] << " ms, max " << values.back() << " ms\n";
    };
    if (cellCrossTest && benchmarkFrameTimes.size() > static_cast<std::size_t>(cellCrossFrame))
    {
        std::cout << "Cell crossing: " << benchmarkFrameTimes[cellCrossFrame]
                  << " ms whole frame, " << benchmarkStreamTimes[cellCrossFrame]
                  << " ms streaming CPU\n";
    }
    reportTimings("Whole frame", std::move(benchmarkFrameTimes));
    reportTimings("Streaming CPU", std::move(benchmarkStreamTimes));
    std::cout << "Final CPU frame: " << finalStats.cpuFrameMs << " ms\n"
              << "Final GPU frame: " << finalStats.gpuFrameMs << " ms\n"
              << "Visible meshlets: " << finalStats.visibleMeshlets << " / "
              << finalStats.sourceCandidates << '\n'
              << "Textures: " << finalStats.residentTextures << " resident, "
              << finalStats.sparseTextures << " sparse\n"
              << "Texture memory: "
              << finalStats.textureResidentBytes / (1024.0 * 1024.0) << " / "
              << finalStats.textureVirtualBytes / (1024.0 * 1024.0)
              << " MiB resident / virtual\n";
    renderer.reset();
    SDL_DestroyWindow(window);
    return compatibility && !compatibility->passed ? 1 : 0;
}
catch (const std::exception& error)
{
    if (compatibilityOutput)
    {
        std::ofstream report(*compatibilityOutput / "runtime.json", std::ios::binary);
        report << "{\"schema\":1,\"status\":\"Fail\",\"reason\":"
            << openf4::assets::compatibilityJsonString(error.what()) << "}\n";
    }
    if (!headless)
        SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "OpenF4", error.what(), nullptr);
    std::cerr << error.what() << '\n';
    return 1;
}
