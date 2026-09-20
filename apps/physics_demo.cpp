#include "assets/ba2_archive.hpp"
#include "assets/fo4_install.hpp"
#include "assets/havok_packfile.hpp"
#include "physics/jolt_collision.hpp"
#include "render/mesh_builder.hpp"
#include "render/vulkan_renderer.hpp"

#include <SDL3/SDL.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <exception>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

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

    empire::assets::HavokRigidBody fallbackBox()
    {
        empire::assets::HavokRigidBody result;
        result.parts.emplace_back();
        auto& hull = result.parts.front().polytope;
        constexpr std::array<float, 2> signs{-1.0f, 1.0f};
        for (const float z : signs)
            for (const float y : signs)
                for (const float x : signs)
                    hull.vertices.emplace_back(x * 0.35f, y * 0.25f, z * 0.2f);
        hull.convexRadius = 0.01f;
        return result;
    }

    empire::assets::HavokRigidBody loadAmmoBoxBody()
    {
        const auto data = empire::assets::findFallout4Data();
        if (!data)
        {
            std::cout << "Fallout 4 not found; using built-in box collision\n";
            return fallbackBox();
        }
        const std::filesystem::path archivePath = *data / "Fallout4 - Meshes.ba2";
        empire::assets::Ba2Archive archive(archivePath);
        constexpr std::string_view path = "Meshes\\SetDressing\\Ammo\\AmmoBoxNotSkinned.nif";
        const auto systems = empire::assets::loadNifPhysicsSystems(archive.extract(path));
        if (systems.empty() || systems.front().bodies.empty())
            throw std::runtime_error("Fallout 4 ammo box has no decoded Havok body");
        std::cout << "Collision: " << path << " ("
                  << systems.front().bodies.front().parts.front().polytope.vertices.size()
                  << " hull vertices)\n";
        return systems.front().bodies.front();
    }

    struct CompressedMeshScene
    {
        std::vector<empire::assets::HavokRigidBody> bodies;
        empire::render::PackedMesh mesh;
        glm::vec3 supportPoint{};
        std::size_t sections = 0;
        std::size_t vertices = 0;
        std::size_t triangles = 0;
        std::size_t runs = 0;
    };

    CompressedMeshScene loadCompressedMeshScene()
    {
        const auto data = empire::assets::findFallout4Data();
        if (!data)
            throw std::runtime_error("Fallout 4 is required for --compressed-mesh");
        empire::assets::Ba2Archive archive(*data / "Fallout4 - Meshes.ba2");
        constexpr std::string_view path =
            "Meshes\\Architecture\\Unique\\OldStateHouse.nif";
        const auto systems = empire::assets::loadNifPhysicsSystems(archive.extract(path));

        CompressedMeshScene result;
        std::vector<empire::render::SourceVertex> vertices;
        std::vector<std::uint32_t> indices;
        float bestSupportArea = 0.0f;
        for (const auto& system : systems)
            for (const auto& body : system.bodies)
            {
                if (!body.triangleMesh)
                    continue;
                const auto& collision = *body.triangleMesh;
                const glm::mat4 bodyTransform = empire::physics::havokBodyLocalTransform(body);
                if (vertices.size() > std::numeric_limits<std::uint32_t>::max()
                    - collision.vertices.size())
                    throw std::runtime_error("OldStateHouse collision mesh is too large");
                const std::uint32_t base = static_cast<std::uint32_t>(vertices.size());
                for (const glm::vec3& vertex : collision.vertices)
                {
                    empire::render::SourceVertex source;
                    source.position = glm::vec3(bodyTransform * glm::vec4(
                        glm::vec3(vertex.x, vertex.z, vertex.y)
                            * empire::physics::BethesdaUnitsPerMeter,
                        1.0f));
                    source.normal = glm::vec3(0.0f);
                    vertices.push_back(source);
                }
                for (std::size_t index = 0; index < collision.indices.size(); index += 3)
                {
                    const std::uint32_t a = base + collision.indices[index];
                    const std::uint32_t b = base + collision.indices[index + 2];
                    const std::uint32_t c = base + collision.indices[index + 1];
                    indices.insert(indices.end(), {a, b, c});
                    const glm::vec3 normal = glm::cross(
                        vertices[b].position - vertices[a].position,
                        vertices[c].position - vertices[a].position);
                    vertices[a].normal += normal;
                    vertices[b].normal += normal;
                    vertices[c].normal += normal;
                    const float area = glm::length(normal);
                    if (area > 0.0f && normal.y / area > 0.9f
                        && normal.y > bestSupportArea)
                    {
                        bestSupportArea = normal.y;
                        result.supportPoint = (vertices[a].position
                            + vertices[b].position + vertices[c].position) / 3.0f;
                    }
                }
                result.sections += collision.sectionCount;
                result.vertices += collision.vertices.size();
                result.triangles += collision.indices.size() / 3;
                result.runs += collision.primitiveDataRuns.size();
                result.bodies.push_back(body);
            }
        if (result.bodies.empty() || indices.empty() || bestSupportArea <= 0.0f)
            throw std::runtime_error("OldStateHouse has no usable compressed collision mesh");
        for (auto& vertex : vertices)
        {
            const float lengthSquared = glm::dot(vertex.normal, vertex.normal);
            vertex.normal = lengthSquared > 0.0f
                ? vertex.normal / std::sqrt(lengthSquared)
                : glm::vec3(0.0f, 1.0f, 0.0f);
        }
        result.mesh = empire::render::buildMeshlets(vertices, indices);
        return result;
    }

    glm::vec3 centerHull(empire::assets::HavokConvexPolytope& hull)
    {
        glm::vec3 minimum(std::numeric_limits<float>::max());
        glm::vec3 maximum(std::numeric_limits<float>::lowest());
        for (const glm::vec3& vertex : hull.vertices)
        {
            minimum = glm::min(minimum, vertex);
            maximum = glm::max(maximum, vertex);
        }
        const glm::vec3 center = (minimum + maximum) * 0.5f;
        for (glm::vec3& vertex : hull.vertices)
            vertex -= center;
        const glm::vec3 extent = (maximum - minimum) * 0.5f;
        return {extent.x, extent.z, extent.y};
    }

    empire::render::SceneInstance makeInstance(const empire::render::MeshRange& mesh,
        const glm::mat4& transform)
    {
        empire::render::SceneInstance result;
        result.transform = transform;
        result.meshletOffset = mesh.meshletOffset;
        result.meshletCount = mesh.meshletCount;
        const glm::vec3 center = glm::vec3(transform * glm::vec4(glm::vec3(mesh.bounds), 1.0f));
        const float scale = std::max(glm::length(glm::vec3(transform[0])),
            std::max(glm::length(glm::vec3(transform[1])), glm::length(glm::vec3(transform[2]))));
        result.bounds = glm::vec4(center, mesh.bounds.w * scale);
        return result;
    }
}

int main(int argc, char** argv)
try
{
    int frameLimit = 0;
    bool compressedMesh = false;
    for (int argument = 1; argument < argc; ++argument)
    {
        const std::string_view option(argv[argument]);
        if (option == "--compressed-mesh")
            compressedMesh = true;
        else if (option == "--frames" && argument + 1 < argc)
        {
            const std::string_view value(argv[++argument]);
            const auto parsed = std::from_chars(
                value.data(), value.data() + value.size(), frameLimit);
            if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size()
                || frameLimit < 1)
                throw std::runtime_error("--frames expects a positive count");
        }
        else
            throw std::runtime_error(
                "Usage: empire-physics-demo [--compressed-mesh] [--frames count]");
    }

    SdlGuard sdl;
    SDL_Window* window = SDL_CreateWindow(compressedMesh
            ? "EMPIRE - Havok Compressed Mesh Demo" : "EMPIRE - Jolt Havok Demo",
        1280, 720,
        SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY);
    if (!window)
        throw std::runtime_error(SDL_GetError());

    empire::physics::JoltWorld physics;
    auto bodyDefinition = loadAmmoBoxBody();
    auto& hull = bodyDefinition.parts.front().polytope;
    const glm::vec3 boxHalfExtent = centerHull(hull) * empire::physics::BethesdaUnitsPerMeter;
    const auto collisionShape = empire::physics::makeJoltCollisionShape(bodyDefinition);
    constexpr float units = empire::physics::BethesdaUnitsPerMeter;

    empire::render::PackedMesh sceneMesh;
    empire::render::MeshRange staticRange;
    glm::vec3 dropOrigin(0.0f);
    glm::vec3 camera(7.0f * units, 5.0f * units, 9.0f * units);
    glm::vec3 cameraTarget(0.0f, 1.2f * units, 0.0f);
    float cameraFar = 100.0f * units;

    if (compressedMesh)
    {
        auto collision = loadCompressedMeshScene();
        staticRange = empire::render::appendMesh(sceneMesh, std::move(collision.mesh));
        dropOrigin = collision.supportPoint;
        camera = dropOrigin + glm::vec3(13.0f, 8.0f, 15.0f) * units;
        cameraTarget = dropOrigin;
        cameraFar = 300.0f * units;
        for (const auto& body : collision.bodies)
        {
            const auto shape = empire::physics::makeJoltCollisionShape(body);
            const glm::mat4 transform = empire::physics::havokBodyLocalTransform(body);
            const glm::quat rotation = glm::normalize(glm::quat_cast(glm::mat3(transform)));
            (void)physics.addStaticBody(shape, glm::vec3(transform[3]), rotation);
        }
        std::cout << "Collision: Meshes\\Architecture\\Unique\\OldStateHouse.nif ("
                  << collision.bodies.size() << " bodies, " << collision.sections
                  << " sections, " << collision.vertices << " vertices, "
                  << collision.triangles << " triangles, " << collision.runs
                  << " primitive runs)\n";
    }
    else
    {
        (void)physics.addStaticBox({8.0f * units, 0.25f * units, 8.0f * units},
            {0.0f, -0.25f * units, 0.0f});
        staticRange = empire::render::appendMesh(
            sceneMesh, empire::render::makeCubeMesh());
    }

    auto boxMesh = empire::render::makeCubeMesh();
    for (auto& meshlet : boxMesh.meshlets)
        meshlet.materialIndex = 1;
    const auto boxRange = empire::render::appendMesh(sceneMesh, std::move(boxMesh));

    std::vector<empire::render::SceneMaterial> materials(2);
    materials[0].baseColor = {0.14f, 0.17f, 0.2f, 1.0f};
    materials[1].baseColor = {0.85f, 0.42f, 0.12f, 1.0f};
    std::vector<empire::render::SceneInstance> instances;
    instances.push_back(makeInstance(staticRange, compressedMesh
            ? glm::mat4(1.0f)
            : glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -0.25f * units, 0.0f))
                * glm::scale(glm::mat4(1.0f),
                    glm::vec3(8.0f, 0.25f, 8.0f) * units)));

    std::vector<empire::physics::BodyHandle> bodies;
    const int layers = compressedMesh ? 4 : 6;
    const int columns = compressedMesh ? 4 : 5;
    for (int layer = 0; layer < layers; ++layer)
    {
        for (int column = 0; column < columns; ++column)
        {
            const glm::vec3 position = dropOrigin + glm::vec3(
                (static_cast<float>(column) - (static_cast<float>(columns) - 1.0f) * 0.5f)
                    * boxHalfExtent.x * 2.2f,
                (compressedMesh ? 1.5f * units : 0.75f * units)
                    + static_cast<float>(layer) * boxHalfExtent.y * 2.4f,
                (layer % 2 == 0 ? -0.15f : 0.15f) * units);
            const glm::quat rotation = glm::angleAxis(
                (layer % 2 == 0 ? 0.04f : -0.04f), glm::vec3(0.0f, 1.0f, 0.0f));
            const auto body = physics.addDynamicBody(collisionShape, position, rotation);
            bodies.push_back(body);
            instances.push_back(makeInstance(boxRange,
                physics.bodyTransform(body) * glm::scale(glm::mat4(1.0f), boxHalfExtent)));
        }
    }

    empire::render::VulkanRenderer renderer(window, sceneMesh, instances, materials);
    std::cout << "GPU: " << renderer.deviceName() << '\n'
              << "Jolt: dropping " << bodies.size() << " decoded ammo-box hulls\n"
              << "Press Escape to quit\n";

    bool running = true;
    int renderedFrames = 0;
    auto previous = std::chrono::steady_clock::now();
    while (running)
    {
        SDL_Event event;
        while (SDL_PollEvent(&event))
        {
            if (event.type == SDL_EVENT_QUIT
                || (event.type == SDL_EVENT_KEY_DOWN && event.key.key == SDLK_ESCAPE))
                running = false;
            else if (event.type == SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED)
                renderer.resize();
        }

        const auto now = std::chrono::steady_clock::now();
        const float delta = std::min(std::chrono::duration<float>(now - previous).count(), 0.1f);
        previous = now;
        physics.step(delta);
        for (std::size_t index = 0; index < bodies.size(); ++index)
        {
            const glm::mat4 transform = physics.bodyTransform(bodies[index])
                * glm::scale(glm::mat4(1.0f), boxHalfExtent);
            const auto updated = makeInstance(boxRange, transform);
            renderer.updateInstanceTransform(
                static_cast<std::uint32_t>(index + 1), updated.transform, updated.bounds);
        }

        int width = 0;
        int height = 0;
        SDL_GetWindowSizeInPixels(window, &width, &height);
        if (width > 0 && height > 0)
        {
            const auto view = glm::lookAtRH(camera, cameraTarget,
                {0.0f, 1.0f, 0.0f});
            auto projection = empire::render::reverseZPerspectiveRH_ZO(glm::radians(65.0f),
                static_cast<float>(width) / static_cast<float>(height),
                0.05f * units, cameraFar);
            projection[1][1] *= -1.0f;
            renderer.render(view, projection, camera);
            ++renderedFrames;
        }
        if (frameLimit > 0 && renderedFrames >= frameLimit)
            running = false;
    }

    SDL_DestroyWindow(window);
    return 0;
}
catch (const std::exception& error)
{
    std::cerr << error.what() << '\n';
    SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "EMPIRE", error.what(), nullptr);
    return 1;
}
