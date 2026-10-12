#pragma once

#include "render/mesh_builder.hpp"

#include <SDL3/SDL.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <glm/ext/matrix_clip_space.hpp>
#include <vulkan/vulkan.h>

namespace openf4::render
{
    inline constexpr std::uint32_t maxSceneTextures = 16384;

    namespace detail
    {
        inline constexpr std::uint32_t sparseTextureMaxDimension = 1024;

        [[nodiscard]] constexpr std::uint32_t sparseFirstResidentMip(
            std::uint32_t width, std::uint32_t height, std::uint32_t mipLevels,
            std::uint32_t mipTailFirstLod)
        {
            if (mipLevels == 0)
                return 0;
            std::uint32_t mip = 0;
            while (mip + 1 < mipLevels
                && (width > sparseTextureMaxDimension
                    || height > sparseTextureMaxDimension))
            {
                width = width > 1 ? width / 2 : 1;
                height = height > 1 ? height / 2 : 1;
                ++mip;
            }
            return mip < mipTailFirstLod ? mip : mipTailFirstLod;
        }

        // Virtual texturing: the first mip a streamed texture makes resident. It is the finest
        // mip no larger than startDimension, but never deeper than the sparse mip tail (the
        // tail is bound and uploaded as one unit anyway) and never finer than targetMip.
        // Finer mips are bound and uploaded one level at a time after the texture is visible.
        [[nodiscard]] constexpr std::uint32_t textureStreamInitialMip(
            std::uint32_t width, std::uint32_t height, std::uint32_t mipLevels,
            std::uint32_t targetMip, std::uint32_t mipTailFirstLod, std::uint32_t startDimension)
        {
            if (mipLevels == 0)
                return 0;
            std::uint32_t mip = 0;
            while (mip + 1 < mipLevels && (width > startDimension || height > startDimension))
            {
                width = width > 1 ? width / 2 : 1;
                height = height > 1 ? height / 2 : 1;
                ++mip;
            }
            if (mip > mipTailFirstLod)
                mip = mipTailFirstLod;
            return mip < targetMip ? targetMip : mip;
        }
    }

    [[nodiscard]] inline glm::mat4 reverseZPerspectiveRH_ZO(
        float fovY, float aspect, float nearPlane, float farPlane)
    {
        return glm::perspectiveRH_ZO(fovY, aspect, farPlane, nearPlane);
    }

    [[nodiscard]] inline glm::mat4 reverseZInfinitePerspectiveRH_ZO(
        float fovY, float aspect, float nearPlane)
    {
        const float focalLength = 1.0f / std::tan(fovY * 0.5f);
        glm::mat4 result(0.0f);
        result[0][0] = focalLength / aspect;
        result[1][1] = focalLength;
        result[2][3] = -1.0f;
        result[3][2] = nearPlane;
        return result;
    }

    enum class TextureFormat
    {
        rgba8,
        rgba8Srgb,
        bc1,
        bc1Srgb,
        bc2,
        bc2Srgb,
        bc3,
        bc3Srgb,
        bc5,
        bc7,
        bc7Srgb,
        bgra8
    };

    struct SceneTexture
    {
        std::uint32_t width = 1;
        std::uint32_t height = 1;
        std::uint32_t mipLevels = 1;
        TextureFormat format = TextureFormat::rgba8;
        std::vector<std::byte> pixels;
        std::vector<std::uint32_t> mipOffsets;
    };

    struct SceneMaterial
    {
        glm::vec4 baseColor{1.0f};
        std::uint32_t diffuseTexture = 0;
        std::uint32_t normalTexture = 0;
        std::uint32_t grayscaleTexture = 0;
        glm::uvec4 terrainTextures{};
        glm::uvec4 terrainTexturesExtra{};
        float grayscaleToPaletteScale = 0.5f;
        float alphaCutoff = 0.0f;
        bool grayscaleToPalette = false;
        bool terrain = false;
        bool terrainLod = false;
        bool water = false;
        bool decal = false;
        bool doubleSided = false;
        bool normalMapped = false;
    };

    struct SceneInstance
    {
        glm::mat4 transform{1.0f};
        glm::vec4 bounds{};
        glm::vec4 lodRange{0.0f, 0.0f, 1.0e9f, 1.0e9f};
        std::uint32_t meshletOffset = 0;
        std::uint32_t meshletCount = 0;
        std::uint32_t skinOffset = 0;
        std::uint32_t skinCount = 0;
        bool active = true;
        bool overlay = false;
    };

    struct RendererStats
    {
        double cpuFrameMs = 0.0;
        double gpuFrameMs = 0.0;
        // GPU pass split (timestamps): cull compute, shadow pass, world draws, UI/selection/present.
        // With FIFO present the world-draw slice also contains the swapchain acquire wait.
        double gpuCullMs = 0.0;
        double gpuShadowMs = 0.0;
        double gpuWorldMs = 0.0;
        double gpuOverlayMs = 0.0;
        std::uint32_t sourceCandidates = 0;
        std::uint32_t visibleMeshlets = 0;
        std::uint32_t shadowMeshlets = 0;
        std::uint32_t residentTextures = 0;
        std::uint32_t sparseTextures = 0;
        std::uint64_t textureVirtualBytes = 0;
        std::uint64_t textureResidentBytes = 0;
        // Texture streaming diagnostics (CPU): addTexture work since the previous frame,
        // this frame's upload recording, and the texture bytes copied this frame.
        double textureCreateMs = 0.0;
        double streamUploadMs = 0.0;
        std::uint64_t textureUploadBytes = 0;
        std::uint32_t streamingTextures = 0;
        double uploadGrowMs = 0.0;      // descriptor/buffer growth inside the upload this frame
        double sparseBindMs = 0.0;      // frame-thread sparse bind cost (binder hand-off, or vkQueueBindSparse when sync)
        double sparsePoolStallMs = 0.0; // waits for a sparse page block since the previous frame
        double frameWaitMs = 0.0;       // CPU wait on this frame slot's fence
        double retireMs = 0.0;          // freeing retired uploads/buffers/views at frame start
        double submitWaitMs = 0.0;      // frame thread waiting on the GPU submit thread (slot reuse)
        double submitThreadMs = 0.0;    // slowest submit+present on the submit thread since last frame
        // Unified GPU memory budget (OPENF4_MEMORY_PRESET / --memory-preset): scene buffers (+ cull
        // lists, in-flight growths) and resident textures together never exceed memoryBudgetBytes.
        // The texture budget is what the scene leaves: the residency pass raises a distance-based mip
        // bias and unloads finer mips to stay under it, and promotions/new textures wait instead.
        std::uint64_t memoryBudgetBytes = 0;      // total, scene + textures (0 = uncapped)
        std::uint64_t sceneCapBytes = 0;          // most the scene may take (budget - texture floor)
        std::uint64_t sceneAllocatedBytes = 0;    // scene buffers as allocated (reservation included)
        std::uint64_t sceneUsedBytes = 0;         // scene buffers as populated
        std::uint64_t scenePeakUsedBytes = 0;
        std::uint64_t textureBudgetBytes = 0;     // live: budget - scene allocation
        std::uint64_t textureBudgetMinBytes = 0;  // lowest live texture budget seen
        std::uint64_t texturePeakResidentBytes = 0;
        float textureMipBias = 0.0f;              // budget-driven extra bias (log2 units)
        std::uint32_t textureEvictions = 0;       // mip unloads since startup
        std::uint32_t textureBudgetDeferrals = 0; // promotions/creations held back by the budget
        std::uint32_t sceneCapRejects = 0;        // scene appends refused by the scene cap
        double residencyMs = 0.0;                 // residency pass (parallel) this frame
        const char* memoryPreset = "medium";
    };

    // The editor owns its UI backend; the renderer only supplies the Vulkan objects
    // and records the UI draw list at the safe point in its existing frame.
    struct UiContext
    {
        VkInstance instance = VK_NULL_HANDLE;
        VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
        VkDevice device = VK_NULL_HANDLE;
        VkQueue queue = VK_NULL_HANDLE;
        std::uint32_t queueFamily = 0;
        VkFormat colorFormat = VK_FORMAT_UNDEFINED;
        VkFormat depthFormat = VK_FORMAT_UNDEFINED;
        std::uint32_t minImageCount = 0;
        std::uint32_t imageCount = 0;
        std::uint64_t swapchainGeneration = 0;
    };

    struct RenderViewport
    {
        std::int32_t x = 0;
        std::int32_t y = 0;
        std::uint32_t width = 0;
        std::uint32_t height = 0;
    };

    using UiDrawCallback = std::function<void(VkCommandBuffer)>;

    class VulkanRenderer
    {
    public:
        VulkanRenderer(SDL_Window* window, const PackedMesh& mesh,
            std::span<const SceneInstance> instances,
            std::span<const SceneMaterial> materials = {},
            std::span<const SceneTexture> textures = {});
        ~VulkanRenderer();
        VulkanRenderer(const VulkanRenderer&) = delete;
        VulkanRenderer& operator=(const VulkanRenderer&) = delete;

        void resize();
        void render(const glm::mat4& view, const glm::mat4& projection,
            const glm::vec3& cameraPosition, const glm::vec3& worldOrigin = {},
            std::optional<RenderViewport> viewport = std::nullopt,
            const UiDrawCallback& drawUi = {});
        [[nodiscard]] std::uint32_t addTexture(SceneTexture texture);
        [[nodiscard]] std::uint32_t addMaterial(const SceneMaterial& material);
        void updateMaterial(std::uint32_t material, const SceneMaterial& value);
        // Preview owns a recyclable tail of the scene. World uploads pause until endPreview.
        // Texture index zero remains the default; preview textures are numbered from one.
        [[nodiscard]] std::uint32_t replacePreviewScene(PackedMesh mesh,
            std::span<const SceneInstance> instances, std::span<const SceneMaterial> materials,
            std::span<const SceneTexture> textures = {});
        void endPreview();
        [[nodiscard]] std::uint32_t previewMaterialOffset() const noexcept;
        [[nodiscard]] std::uint32_t previewTextureOffset() const noexcept;
        [[nodiscard]] MeshRange addMesh(PackedMesh mesh);
        void updateMesh(const MeshRange& range, PackedMesh mesh);
        [[nodiscard]] std::uint32_t addInstance(SceneInstance instance);
        [[nodiscard]] std::uint32_t addSkinPalette(std::span<const glm::mat4> transforms);
        void updateSkinPalette(std::uint32_t offset, std::span<const glm::mat4> transforms);
        void updateInstanceTransform(std::uint32_t instance,
            const glm::mat4& transform, const glm::vec4& bounds);
        void updateInstanceLodRange(std::uint32_t instance, glm::vec4 lodRange);
        void setInstancesActive(std::span<const std::uint32_t> instances, bool active);
        // Transient editor state. The active instance must be in the selection.
        // An empty selection performs no mask/composite work.
        void setSelection(std::span<const std::uint32_t> instances,
            std::optional<std::uint32_t> activeInstance = std::nullopt);
        [[nodiscard]] const RendererStats& stats() const noexcept;
        [[nodiscard]] const std::string& deviceName() const noexcept;
        [[nodiscard]] UiContext uiContext() const noexcept;

    private:
        struct Impl;
        std::unique_ptr<Impl> mImpl;
    };
}
