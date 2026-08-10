#pragma once

#include "render/mesh_builder.hpp"
#include "render/vulkan_renderer.hpp"

#include <filesystem>
#include <limits>
#include <optional>
#include <memory>
#include <string_view>
#include <vector>

#include <glm/glm.hpp>

#include "assets/esm_world.hpp"

namespace empire::world
{
    struct WorldScene
    {
        render::PackedMesh mesh;
        std::vector<render::SceneInstance> instances;
        std::vector<render::SceneMaterial> materials;
        std::vector<render::SceneTexture> textures;
        glm::vec3 cameraPosition{};
        glm::vec3 cameraTarget{};
        float sceneScale = 4096.0f;
        assets::CellCoord centerCell{};
        std::size_t landscapeCells = 0;
        std::size_t placedObjects = 0;
    };

    [[nodiscard]] WorldScene loadWorldScene(const std::filesystem::path& dataDirectory,
        std::optional<assets::CellCoord> requestedCenter = std::nullopt, int cellRadius = 1,
        bool includeObjects = true,
        std::size_t maxPlacedObjects = std::numeric_limits<std::size_t>::max());
    [[nodiscard]] bool isRenderableModelPath(std::string_view path);
    void eulerToQuaternion(float rotX, float rotY, float rotZ,
        float& qx, float& qy, float& qz, float& qw);
    [[nodiscard]] glm::mat4 bethesdaReferenceTransform(const assets::PlacedReference& reference);

    class WorldStreamer
    {
    public:
        WorldStreamer(const std::filesystem::path& dataDirectory,
            std::optional<assets::CellCoord> requestedCenter = std::nullopt, int cellRadius = 1);
        ~WorldStreamer();
        WorldStreamer(WorldStreamer&&) noexcept;
        WorldStreamer& operator=(WorldStreamer&&) noexcept;
        WorldStreamer(const WorldStreamer&) = delete;
        WorldStreamer& operator=(const WorldStreamer&) = delete;

        [[nodiscard]] WorldScene takeInitialScene();
        [[nodiscard]] bool streamNext(render::VulkanRenderer& renderer);
        [[nodiscard]] std::size_t streamedObjects() const noexcept;
        [[nodiscard]] std::size_t pendingObjects() const noexcept;

    private:
        struct Impl;
        std::unique_ptr<Impl> mImpl;
    };
}
