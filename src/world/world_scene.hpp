#pragma once

#include "render/mesh_builder.hpp"
#include "render/vulkan_renderer.hpp"
#include "physics/jolt_collision.hpp"

#include <filesystem>
#include <limits>
#include <optional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <glm/glm.hpp>

#include "assets/esm_world.hpp"
#include "assets/esm_editor.hpp"

namespace empire::world
{
    enum class LandscapeTool
    {
        raise,
        lower,
        smooth,
        flatten
    };

    namespace detail
    {
        inline constexpr float demHorizonFadeStart = 80.0f * 4096.0f;
        inline constexpr float demHorizonFadeEnd = 92.0f * 4096.0f;

        [[nodiscard]] std::optional<assets::CellCoord> demBtrOrigin(
            const std::filesystem::path& path);
        [[nodiscard]] bool repairLegacyDemBtrGrid(
            std::span<render::SourceVertex> vertices,
            std::span<const float> fallbackHeights = {});
        [[nodiscard]] inline glm::vec4 demTerrainLodRange(bool coarse) noexcept
        {
            constexpr float fadeStart = 3.0f * 4096.0f;
            constexpr float fadeEnd = 4.0f * 4096.0f;
            return coarse
                ? glm::vec4(fadeStart, fadeEnd,
                    demHorizonFadeStart, demHorizonFadeEnd)
                : glm::vec4(0.0f, 0.0f, fadeStart, fadeEnd);
        }

        [[nodiscard]] inline glm::vec4 demWaterLodRange() noexcept
        {
            return {0.0f, 0.0f, demHorizonFadeStart, demHorizonFadeEnd};
        }
        [[nodiscard]] bool applyLandscapeBrush(assets::EditableLandscape& landscape,
            glm::vec2 center, LandscapeTool tool, float radius, float amount,
            float flattenHeight = 0.0f);
        void quantizeLandscape(assets::EditableLandscape& landscape);

        class LandscapeHistory
        {
        public:
            void commit(const assets::EditableLandscape& before);
            void addCell(assets::CellCoord coord);
            [[nodiscard]] bool undo(assets::EditableLandscape& current);
            [[nodiscard]] bool redo(assets::EditableLandscape& current);

        private:
            std::vector<assets::EditableLandscape> mUndo;
            std::vector<assets::EditableLandscape> mRedo;
        };
    }

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
        std::size_t landscapeLodCells = 0;
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
            std::optional<assets::CellCoord> requestedCenter = std::nullopt, int cellRadius = 1,
            bool includeObjects = true,
            std::size_t maxObjects = std::numeric_limits<std::size_t>::max(),
            std::optional<std::filesystem::path> masterPath = std::nullopt);
        ~WorldStreamer();
        WorldStreamer(WorldStreamer&&) noexcept;
        WorldStreamer& operator=(WorldStreamer&&) noexcept;
        WorldStreamer(const WorldStreamer&) = delete;
        WorldStreamer& operator=(const WorldStreamer&) = delete;

        [[nodiscard]] WorldScene takeInitialScene();
        [[nodiscard]] std::optional<glm::vec3> enablePhysics(
            render::VulkanRenderer& renderer, physics::JoltWorld& world,
            bool spawnPhysicsSample = false);
        void update(render::VulkanRenderer& renderer, const glm::dvec3& playerPosition);
        [[nodiscard]] bool streamNext(render::VulkanRenderer& renderer);
        [[nodiscard]] std::size_t streamedObjects() const noexcept;
        [[nodiscard]] std::size_t pendingObjects() const noexcept;

    private:
        struct Impl;
        std::unique_ptr<Impl> mImpl;
    };

    class LandscapeEditor
    {
    public:
        LandscapeEditor(const std::filesystem::path& dataDirectory,
            std::filesystem::path outputPath, bool readOnly = false);
        ~LandscapeEditor();
        LandscapeEditor(LandscapeEditor&&) noexcept;
        LandscapeEditor& operator=(LandscapeEditor&&) noexcept;
        LandscapeEditor(const LandscapeEditor&) = delete;
        LandscapeEditor& operator=(const LandscapeEditor&) = delete;

        [[nodiscard]] WorldScene takeInitialScene();
        void setTool(LandscapeTool tool) noexcept;
        void adjustRadius(float wheelSteps) noexcept;
        void adjustStrength(float wheelSteps) noexcept;
        void update(render::VulkanRenderer& renderer, const glm::dvec3& cameraPosition,
            const glm::vec3& cameraForward, float deltaSeconds, bool sculpting);
        [[nodiscard]] bool undo(render::VulkanRenderer& renderer);
        [[nodiscard]] bool redo(render::VulkanRenderer& renderer);
        void save(render::VulkanRenderer& renderer);
        [[nodiscard]] bool dirty() const noexcept;
        [[nodiscard]] std::string status() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> mImpl;
    };
}
