#pragma once

#include "render/mesh_builder.hpp"

#include <SDL3/SDL.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <glm/ext/matrix_clip_space.hpp>

namespace empire::render
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
        std::uint32_t sourceCandidates = 0;
        std::uint32_t visibleMeshlets = 0;
        std::uint32_t residentTextures = 0;
        std::uint32_t sparseTextures = 0;
        std::uint64_t textureVirtualBytes = 0;
        std::uint64_t textureResidentBytes = 0;
    };

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
            const glm::vec3& cameraPosition, const glm::vec3& worldOrigin = {});
        [[nodiscard]] std::uint32_t addTexture(SceneTexture texture);
        [[nodiscard]] std::uint32_t addMaterial(const SceneMaterial& material);
        [[nodiscard]] MeshRange addMesh(PackedMesh mesh);
        void updateMesh(const MeshRange& range, PackedMesh mesh);
        [[nodiscard]] std::uint32_t addInstance(SceneInstance instance);
        [[nodiscard]] std::uint32_t addSkinPalette(std::span<const glm::mat4> transforms);
        void updateSkinPalette(std::uint32_t offset, std::span<const glm::mat4> transforms);
        void updateInstanceTransform(std::uint32_t instance,
            const glm::mat4& transform, const glm::vec4& bounds);
        void setInstancesActive(std::span<const std::uint32_t> instances, bool active);
        [[nodiscard]] const RendererStats& stats() const noexcept;
        [[nodiscard]] const std::string& deviceName() const noexcept;

    private:
        struct Impl;
        std::unique_ptr<Impl> mImpl;
    };
}
