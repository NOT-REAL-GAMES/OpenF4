#pragma once

#include "render/mesh_builder.hpp"
#include "render/vulkan_renderer.hpp"
#include "physics/jolt_collision.hpp"
#include "world/picking_geometry.hpp"

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

namespace openf4::world
{
    enum class LandscapeTool
    {
        raise,
        lower,
        smooth,
        flatten
    };

    struct ObjectPick
    {
        std::uint32_t formId = 0;
        double distance = 0.0;
    };

    struct LandscapeSurface
    {
        double height = 0.0;
        glm::vec3 normal{0.0f, 1.0f, 0.0f};
    };

    namespace detail
    {
        inline constexpr float demHorizonFadeStart = 80.0f * 4096.0f;
        inline constexpr float demHorizonFadeEnd = 92.0f * 4096.0f;

        [[nodiscard]] std::optional<assets::CellCoord> demBtrOrigin(
            const std::filesystem::path& path);
        [[nodiscard]] bool preferDemBtr(const std::filesystem::path& candidate,
            const std::filesystem::path& current);
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

        [[nodiscard]] inline glm::vec4 demBtrLodRange(int level) noexcept
        {
            constexpr float cell = 4096.0f;
            switch (level)
            {
            case 4: return {3.0f * cell, 4.0f * cell, 8.0f * cell, 12.0f * cell};
            case 8: return {8.0f * cell, 12.0f * cell, 16.0f * cell, 24.0f * cell};
            case 16: return {16.0f * cell, 24.0f * cell, 32.0f * cell, 48.0f * cell};
            default: return {32.0f * cell, 48.0f * cell,
                demHorizonFadeStart, demHorizonFadeEnd};
            }
        }

        [[nodiscard]] inline glm::vec4 demWaterLodRange() noexcept
        {
            return {0.0f, 0.0f, demHorizonFadeStart, demHorizonFadeEnd};
        }
        [[nodiscard]] bool applyLandscapeBrush(assets::EditableLandscape& landscape,
            glm::vec2 center, LandscapeTool tool, float radius, float amount,
            float flattenHeight = 0.0f);
        void quantizeLandscape(assets::EditableLandscape& landscape);
        [[nodiscard]] std::optional<LandscapeSurface> landscapeSurface(
            const assets::EditableLandscape& landscape, glm::dvec2 engineXZ);

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
    // Inverse of bethesdaReferenceTransform for authored (non-SCOL) references.
    void setBethesdaReferenceTransform(assets::PlacedReference& reference,
        const glm::mat4& transform);

    // Shared live projection of authored references, used by the editor and runtime.
    // Camera positions include the renderer's X reflection; bounds do not.
    class ObjectScene
    {
    public:
        ObjectScene(const std::filesystem::path& dataDirectory,
            const std::filesystem::path& masterPath);
        ~ObjectScene();
        ObjectScene(const ObjectScene&) = delete;
        ObjectScene& operator=(const ObjectScene&) = delete;
        void update(render::VulkanRenderer& renderer,
            std::span<const assets::PlacedReference> references,
            const std::map<std::uint32_t, std::string>& models,
            const std::unordered_map<std::uint32_t, std::filesystem::path>& modelSources,
            const glm::dvec3& cameraPosition,
            std::span<const std::uint32_t> hiddenForms = {});
        [[nodiscard]] std::optional<glm::vec4> bounds(std::uint32_t formId) const;
        // Picking and bounds use absolute engine coordinates, without camera reflection.
        [[nodiscard]] std::optional<ObjectPick> pick(const glm::dvec3& origin,
            const glm::dvec3& unitDirection, std::span<const std::uint32_t> excludedForms = {},
            double maxDistance = std::numeric_limits<double>::infinity()) const;
        [[nodiscard]] std::optional<GeometryBounds> geometryBounds(std::uint32_t formId,
            std::optional<glm::mat4> transform = std::nullopt) const;
        [[nodiscard]] std::optional<std::uint32_t> rendererInstance(std::uint32_t formId) const;
        [[nodiscard]] std::size_t pending() const noexcept;
        [[nodiscard]] std::size_t loaded() const noexcept;
        [[nodiscard]] const std::vector<std::string>& diagnostics() const noexcept;
        void refreshDerivedLod(render::VulkanRenderer& renderer,
            std::span<const std::filesystem::path> publishedPaths);

    private:
        struct Impl;
        std::unique_ptr<Impl> mImpl;
    };

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
        void setBrushEnabled(bool enabled) noexcept;
        void setExternalSave(bool enabled) noexcept;
        [[nodiscard]] std::optional<glm::vec3> raycast(
            const glm::dvec3& cameraPosition, const glm::vec3& direction) const;
        [[nodiscard]] std::optional<LandscapeSurface> surfaceAt(glm::dvec2 engineXZ) const;
        [[nodiscard]] assets::EditableLandscape snapshot() const;
        [[nodiscard]] std::size_t residentCells() const noexcept;
        [[nodiscard]] std::size_t pendingWork() const noexcept;
        [[nodiscard]] std::uint32_t nextFormId() const noexcept;
        void setNextFormId(std::uint32_t next) noexcept;
        void applySnapshot(render::VulkanRenderer& renderer,
            const assets::EditableLandscape& snapshot);
        void refreshDerivedLod(render::VulkanRenderer& renderer,
            std::span<const std::filesystem::path> publishedPaths);
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
