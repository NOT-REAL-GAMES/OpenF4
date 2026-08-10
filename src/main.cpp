#include "assets/ba2_archive.hpp"
#include "assets/fo4_install.hpp"
#include "assets/nif_model.hpp"
#include "render/mesh_builder.hpp"
#include "render/vulkan_renderer.hpp"
#include "world/world_scene.hpp"

#include <SDL3/SDL.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <limits>
#include <optional>
#include <ranges>
#include <string>
#include <string_view>
#include <vector>

#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>

namespace
{
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
}

int main(int argc, char** argv)
try
{
    std::cout << std::unitbuf;
    std::cerr << std::unitbuf;
    int benchmarkFrames = 0;
    bool cubesOnly = false;
    bool resizeTest = false;
    bool landOnly = false;
    int worldRadius = 1;
    std::size_t maxObjects = std::numeric_limits<std::size_t>::max();
    std::optional<std::string> requestedModel;
    std::optional<empire::assets::CellCoord> requestedCell;
    for (int index = 1; index < argc; ++index)
    {
        if (std::string_view(argv[index]) == "--benchmark")
        {
            if (++index >= argc)
                throw std::runtime_error("--benchmark expects a positive frame count");
            const std::string_view value(argv[index]);
            const auto result = std::from_chars(value.data(), value.data() + value.size(), benchmarkFrames);
            if (result.ec != std::errc{} || benchmarkFrames < 1)
                throw std::runtime_error("--benchmark expects a positive frame count");
        }
        else if (std::string_view(argv[index]) == "--model")
        {
            if (++index >= argc)
                throw std::runtime_error("--model expects an archive-relative NIF path");
            requestedModel = argv[index];
        }
        else if (std::string_view(argv[index]) == "--cubes")
            cubesOnly = true;
        else if (std::string_view(argv[index]) == "--resize-test")
            resizeTest = true;
        else if (std::string_view(argv[index]) == "--land-only")
            landOnly = true;
        else if (std::string_view(argv[index]) == "--radius")
        {
            if (++index >= argc)
                throw std::runtime_error("--radius expects a cell radius");
            const std::string_view value(argv[index]);
            const auto result = std::from_chars(value.data(), value.data() + value.size(), worldRadius);
            if (result.ec != std::errc{} || worldRadius < 0 || worldRadius > 8)
                throw std::runtime_error("--radius expects a value from 0 through 8");
        }
        else if (std::string_view(argv[index]) == "--max-objects")
        {
            if (++index >= argc)
                throw std::runtime_error("--max-objects expects a positive count");
            const std::string_view value(argv[index]);
            const auto result = std::from_chars(value.data(), value.data() + value.size(), maxObjects);
            if (result.ec != std::errc{} || maxObjects == 0)
                throw std::runtime_error("--max-objects expects a positive count");
        }
        else if (std::string_view(argv[index]) == "--cell")
        {
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
        else
            throw std::runtime_error("Unknown command line option: " + std::string(argv[index]));
    }
    SdlGuard sdl;
    SDL_Window* window = SDL_CreateWindow(
        "EMPIRE", 1600, 900,
        SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY);
    if (!window)
        throw std::runtime_error(SDL_GetError());

    const auto falloutData = empire::assets::findFallout4Data();
    empire::render::PackedMesh mesh;
    std::vector<empire::render::SceneInstance> instances;
    std::vector<empire::render::SceneMaterial> materials;
    std::vector<empire::render::SceneTexture> textures;
    glm::vec3 initialPosition{};
    glm::vec3 initialTarget{};
    float sceneScale = 100.0f;
    bool usingNif = false;
    bool usingWorld = false;
    std::unique_ptr<empire::world::WorldStreamer> worldStreamer;
    if (!cubesOnly && !requestedModel && falloutData)
    {
        worldStreamer = std::make_unique<empire::world::WorldStreamer>(
            *falloutData, requestedCell, worldRadius);
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
                  << " LAND cells, " << worldStreamer->pendingObjects() << " objects queued, "
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
    std::cout << "GPU: " << renderer->deviceName() << '\n';
    std::cout << "Textures: " << renderer->stats().residentTextures << " resident, "
              << renderer->stats().sparseTextures << " sparse\n";
    std::cout << "Geometry: " << mesh.vertices.size() << " vertices, "
              << mesh.meshlets.size() << " meshlets, " << instances.size()
              << " instances, " << renderer->stats().sourceCandidates << " candidates\n";
    if (falloutData)
        std::cout << "Fallout 4 Data: " << falloutData->string() << '\n';
    else
        std::cout << "Fallout 4 Data: not found (set FALLOUT4_PATH)\n";

    bool running = true;
    bool mouseCaptured = true;
    SDL_SetWindowRelativeMouseMode(window, true);
    glm::vec3 position = initialPosition;
    const glm::vec3 initialForward = glm::normalize(initialTarget - initialPosition);
    float yaw = std::atan2(initialForward.x, initialForward.z);
    float pitch = std::asin(initialForward.y);
    auto previous = std::chrono::steady_clock::now();
    auto lastTitleUpdate = previous;
    int renderedFrames = 0;

    while (running)
    {
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
            else if (event.type == SDL_EVENT_MOUSE_MOTION && mouseCaptured)
            {
                yaw -= event.motion.xrel * 0.002f;
                pitch = std::clamp(pitch - event.motion.yrel * 0.002f, -1.5f, 1.5f);
            }
        }

        const auto now = std::chrono::steady_clock::now();
        const float delta = std::min(std::chrono::duration<float>(now - previous).count(), 0.1f);
        previous = now;
        if (worldStreamer && !landOnly && renderedFrames > 0
            && worldStreamer->pendingObjects() > 0
            && worldStreamer->streamedObjects() < maxObjects)
            (void)worldStreamer->streamNext(*renderer);
        const glm::vec3 forward(
            std::cos(pitch) * std::sin(yaw), std::sin(pitch), std::cos(pitch) * std::cos(yaw));
        const glm::vec3 right = glm::normalize(glm::cross(forward, glm::vec3(0.0f, 1.0f, 0.0f)));
        const bool* keys = SDL_GetKeyboardState(nullptr);
        const float baseSpeed = std::max(25.0f, sceneScale * 0.15f);
        const float speed = baseSpeed * (keys[SDL_SCANCODE_LSHIFT] ? 4.0f : 1.0f) * delta;
        position += forward * speed * static_cast<float>(keys[SDL_SCANCODE_W] - keys[SDL_SCANCODE_S]);
        position += right * speed * static_cast<float>(keys[SDL_SCANCODE_D] - keys[SDL_SCANCODE_A]);
        position.y += speed * static_cast<float>(keys[SDL_SCANCODE_SPACE] - keys[SDL_SCANCODE_C]);

        int width = 0;
        int height = 0;
        SDL_GetWindowSizeInPixels(window, &width, &height);
        if (width == 0 || height == 0)
        {
            SDL_Delay(16);
            continue;
        }
        const auto view = glm::lookAtRH(position, position + forward, glm::vec3(0.0f, 1.0f, 0.0f));
        auto projection = glm::perspectiveRH_ZO(glm::radians(70.0f),
            static_cast<float>(width) / static_cast<float>(height), 0.1f,
            std::max(5000.0f, sceneScale * 8.0f));
        projection[1][1] *= -1.0f;
        renderer->render(view, projection, position);
        ++renderedFrames;
        if (resizeTest && renderedFrames == 10)
            SDL_SetWindowSize(window, 1280, 720);
        if (benchmarkFrames > 0 && renderedFrames >= benchmarkFrames)
            running = false;

        if (now - lastTitleUpdate > std::chrono::milliseconds(250))
        {
            const auto& stats = renderer->stats();
            const std::string title = "EMPIRE | CPU " + std::to_string(stats.cpuFrameMs).substr(0, 5)
                + " ms | GPU " + std::to_string(stats.gpuFrameMs).substr(0, 5) + " ms"
                + (worldStreamer ? " | loaded "
                    + std::to_string(worldStreamer->streamedObjects()) : "");
            SDL_SetWindowTitle(window, title.c_str());
            lastTitleUpdate = now;
        }
    }

    const auto finalStats = renderer->stats();
    std::cout << "Final CPU frame: " << finalStats.cpuFrameMs << " ms\n"
              << "Final GPU frame: " << finalStats.gpuFrameMs << " ms\n"
              << "Visible meshlets: " << finalStats.visibleMeshlets << " / "
              << finalStats.sourceCandidates << '\n'
              << "Textures: " << finalStats.residentTextures << " resident, "
              << finalStats.sparseTextures << " sparse\n";
    renderer.reset();
    SDL_DestroyWindow(window);
    return 0;
}
catch (const std::exception& error)
{
    SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "EMPIRE", error.what(), nullptr);
    std::cerr << error.what() << '\n';
    return 1;
}
