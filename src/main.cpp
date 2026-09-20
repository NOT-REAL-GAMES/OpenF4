#include "assets/ba2_archive.hpp"
#include "assets/dem_import.hpp"
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
#include <exception>
#include <filesystem>
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
            empire::assets::Ba2Archive archive(archivePath);
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

    empire::render::PackedMesh makeOverlayMesh(std::span<const glm::vec2> points,
        std::span<const std::uint32_t> indices, std::uint32_t material, float depth)
    {
        std::vector<empire::render::SourceVertex> vertices;
        vertices.reserve(points.size());
        for (const glm::vec2 point : points)
            vertices.push_back({glm::vec3(point, depth)});
        return empire::render::buildMeshlets(vertices, indices, material);
    }
}

int main(int argc, char** argv)
try
{
    std::cout << std::unitbuf;
    std::cerr << std::unitbuf;
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
    std::optional<empire::assets::CellCoord> requestedCell;
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
            empire::assets::CellCoord cell;
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
        else if (index == 1 && argc == 2
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
    if (demImportPath)
    {
        headless = true;
        const auto summary = inspectDemOnly
            ? empire::assets::inspectDemLandscape(*demImportPath, demSources)
            : empire::assets::importDemLandscape(
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
            std::cout << "Open with: empire.exe --editor \""
                      << summary.editorPath.string() << "\"\n";
        }
        return 0;
    }
    const bool demViewer = masterPath
        && empire::assets::DemLandscapeProject::recognizes(*masterPath);

    SdlGuard sdl;
    SDL_Window* window = SDL_CreateWindow(
        "EMPIRE", 1600, 900,
        SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY);
    if (!window)
        throw std::runtime_error(SDL_GetError());

    const auto falloutData = empire::assets::findFallout4Data();
    auto assetData = falloutData;
    if (masterPath && (std::filesystem::is_regular_file(
            masterPath->parent_path() / "Fallout4 - Meshes.ba2") || !assetData))
        assetData = masterPath->parent_path();
    if ((editorPath || demViewer) && !falloutData)
        throw std::runtime_error(
            "Landscape loading requires Fallout 4 data (set FALLOUT4_PATH)");
    if (physicsSample && (cubesOnly || requestedModel || !assetData))
        throw std::runtime_error(
            "--physics-sample requires Fallout 4 data and the streamed world scene");
    empire::render::PackedMesh mesh;
    std::vector<empire::render::SceneInstance> instances;
    std::vector<empire::render::SceneMaterial> materials;
    std::vector<empire::render::SceneTexture> textures;
    glm::vec3 initialPosition{};
    glm::vec3 initialTarget{};
    float sceneScale = 100.0f;
    bool usingNif = false;
    bool usingWorld = false;
    std::unique_ptr<empire::physics::JoltWorld> physicsWorld;
    std::unique_ptr<empire::world::WorldStreamer> worldStreamer;
    std::unique_ptr<empire::world::LandscapeEditor> landscapeEditor;
    bool landscapeEditing = false;
    if (editorPath || demViewer)
    {
        landscapeEditing = editorPath.has_value();
        const auto& landscapePath = landscapeEditing ? *editorPath : *masterPath;
        landscapeEditor = std::make_unique<empire::world::LandscapeEditor>(
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
        worldStreamer = std::make_unique<empire::world::WorldStreamer>(
            *assetData, requestedCell, worldRadius, !landOnly, maxObjects, masterPath);
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
            const auto geometry = empire::assets::loadNifModel(archived->bytes);
            std::size_t vertexCount = 0;
            for (const auto& surface : geometry.surfaces)
            {
                vertexCount += surface.vertices.size();
                (void)empire::render::appendMesh(mesh,
                    empire::render::buildMeshlets(surface.vertices, surface.indices));
            }
            usingNif = true;
            std::cout << "NIF: " << archived->path << " (" << vertexCount
                      << " vertices, " << mesh.meshlets.size() << " meshlets)\n";
        }
    }
    if (!usingWorld)
    {
        if (mesh.meshlets.empty())
            mesh = empire::render::makeCubeMesh();
        const int radius = usingNif ? worldRadius : 40;
        const float spacing = usingNif ? std::max(mesh.bounds.w * 2.4f, 100.0f) : 4.0f;
        instances.reserve((radius * 2 + 1) * (radius * 2 + 1));
        for (int z = -radius; z <= radius; ++z)
        {
            for (int x = -radius; x <= radius; ++x)
            {
                empire::render::SceneInstance instance;
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

    auto renderer = std::make_unique<empire::render::VulkanRenderer>(
        window, mesh, instances, materials, textures);
    if (worldStreamer)
    {
        physicsWorld = std::make_unique<empire::physics::JoltWorld>();
        if (const auto target = worldStreamer->enablePhysics(
                *renderer, *physicsWorld, physicsSample))
        {
            initialTarget = *target;
            initialPosition = initialTarget + glm::vec3(500.0f, 320.0f, 500.0f);
        }
    }
    std::optional<std::uint32_t> physicsPlayButton;
    std::optional<std::uint32_t> physicsPauseButton;
    if (physicsWorld && physicsSample)
    {
        empire::render::SceneMaterial background;
        background.baseColor = {0.035f, 0.05f, 0.06f, 1.0f};
        empire::render::SceneMaterial icon;
        icon.baseColor = {0.28f, 0.95f, 0.55f, 1.0f};
        const std::uint32_t backgroundMaterial = renderer->addMaterial(background);
        const std::uint32_t iconMaterial = renderer->addMaterial(icon);
        const auto addOverlay = [&](empire::render::PackedMesh overlay, bool active) {
            const auto range = renderer->addMesh(std::move(overlay));
            empire::render::SceneInstance instance;
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
    bool physicsRunning = false;
    bool editorSculpting = false;
    bool mouseCaptured = !physicsPlayButton;
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
                    landscapeEditor->setTool(empire::world::LandscapeTool::raise);
                else if (event.key.key == SDLK_2)
                    landscapeEditor->setTool(empire::world::LandscapeTool::lower);
                else if (event.key.key == SDLK_3)
                    landscapeEditor->setTool(empire::world::LandscapeTool::smooth);
                else if (event.key.key == SDLK_4)
                    landscapeEditor->setTool(empire::world::LandscapeTool::flatten);
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
        const float speed = baseSpeed * (keys[SDL_SCANCODE_LSHIFT] ? 4.0f : 1.0f) * delta;
        position += glm::dvec3(forward) * static_cast<double>(speed)
            * static_cast<double>(keys[SDL_SCANCODE_W] - keys[SDL_SCANCODE_S]);
        position += glm::dvec3(right) * static_cast<double>(speed)
            * static_cast<double>(keys[SDL_SCANCODE_D] - keys[SDL_SCANCODE_A]);
        position.y += speed * static_cast<float>(keys[SDL_SCANCODE_SPACE] - keys[SDL_SCANCODE_C]);

        const auto streamStart = std::chrono::steady_clock::now();
        if (physicsWorld && physicsRunning)
            physicsWorld->step(delta);
        if (landscapeEditor)
            landscapeEditor->update(*renderer, position, forward, delta, editorSculpting);
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
            ? empire::render::reverseZInfinitePerspectiveRH_ZO(glm::radians(70.0f),
                static_cast<float>(width) / static_cast<float>(height), 0.1f)
            : empire::render::reverseZPerspectiveRH_ZO(glm::radians(70.0f),
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
        if (benchmarkFrames > 0 && renderedFrames >= benchmarkFrames)
            running = false;

        if (now - lastTitleUpdate > std::chrono::milliseconds(250))
        {
            const auto& stats = renderer->stats();
            const std::string title = landscapeEditing && landscapeEditor
                ? landscapeEditor->status()
                : "EMPIRE | CPU " + std::to_string(stats.cpuFrameMs).substr(0, 5)
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
    if (landscapeEditing && landscapeEditor && landscapeEditor->dirty())
        landscapeEditor->save(*renderer);
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
    return 0;
}
catch (const std::exception& error)
{
    if (!headless)
        SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "EMPIRE", error.what(), nullptr);
    std::cerr << error.what() << '\n';
    return 1;
}
