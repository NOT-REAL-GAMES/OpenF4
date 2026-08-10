#pragma once

#include "render/mesh_builder.hpp"

#include <SDL3/SDL.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <glm/glm.hpp>

namespace empire::render
{
    inline constexpr std::uint32_t maxSceneTextures = 4096;

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
        float grayscaleToPaletteScale = 0.5f;
        float alphaCutoff = 0.0f;
        bool grayscaleToPalette = false;
        bool terrain = false;
    };

    struct SceneInstance
    {
        glm::mat4 transform{1.0f};
        glm::vec4 bounds{};
        glm::vec4 lodRange{0.0f, 0.0f, 1.0e9f, 1.0e9f};
        std::uint32_t meshletOffset = 0;
        std::uint32_t meshletCount = 0;
    };

    struct RendererStats
    {
        double cpuFrameMs = 0.0;
        double gpuFrameMs = 0.0;
        std::uint32_t sourceCandidates = 0;
        std::uint32_t visibleMeshlets = 0;
        std::uint32_t residentTextures = 0;
        std::uint32_t sparseTextures = 0;
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
        void render(const glm::mat4& view, const glm::mat4& projection, const glm::vec3& cameraPosition);
        [[nodiscard]] std::uint32_t addTexture(const SceneTexture& texture);
        [[nodiscard]] std::uint32_t addMaterial(const SceneMaterial& material);
        [[nodiscard]] MeshRange addMesh(PackedMesh mesh);
        void addInstance(SceneInstance instance);
        [[nodiscard]] const RendererStats& stats() const noexcept;
        [[nodiscard]] const std::string& deviceName() const noexcept;

    private:
        struct Impl;
        std::unique_ptr<Impl> mImpl;
    };
}
