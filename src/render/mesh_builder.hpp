#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include <glm/glm.hpp>

namespace openf4::render
{
    struct SourceVertex
    {
        glm::vec3 position{};
        glm::vec3 normal{0.0f, 0.0f, 1.0f};
        glm::vec2 uv{};
        glm::vec4 blendWeights{1.0f, 0.0f, 0.0f, 0.0f};
        glm::vec4 blendWeightsExtra{};
        glm::vec4 color{1.0f};
        glm::uvec4 boneIndices{};
        glm::vec4 boneWeights{};
    };

    struct alignas(16) GpuVertex
    {
        glm::vec4 position{};
        glm::vec3 normal{};
        std::uint32_t blendWeights = 0x000000FFu;
        glm::vec2 texCoord{};
        std::uint32_t blendWeightsExtra = 0;
        std::uint32_t color = 0xFFFFFFFFu;
        std::uint32_t boneIndices = 0;
        std::uint32_t boneWeights = 0;
        std::uint32_t padding[2]{};
    };

    struct GpuMeshlet
    {
        std::uint32_t vertexOffset = 0;
        std::uint32_t indexOffset = 0;
        std::uint32_t vertexCount = 0;
        std::uint32_t triangleCount = 0;
        glm::vec4 sphere{};
        std::uint32_t materialIndex = 0;
        std::uint32_t padding[3]{};
    };

    struct PackedMesh
    {
        std::vector<GpuVertex> vertices;
        std::vector<std::uint32_t> indices;
        std::vector<GpuMeshlet> meshlets;
        glm::vec4 bounds{};
    };

    struct MeshRange
    {
        std::uint32_t vertexOffset = 0;
        std::uint32_t vertexCount = 0;
        std::uint32_t indexOffset = 0;
        std::uint32_t indexCount = 0;
        std::uint32_t meshletOffset = 0;
        std::uint32_t meshletCount = 0;
        glm::vec4 bounds{};
    };

    namespace detail
    {
        [[nodiscard]] bool fixedTopologyReplacement(const MeshRange& range,
            std::span<const GpuMeshlet> currentMeshlets, const PackedMesh& replacement);
    }

    [[nodiscard]] PackedMesh buildMeshlets(
        std::span<const SourceVertex> vertices, std::span<const std::uint32_t> indices,
        std::uint32_t materialIndex = 0);
    [[nodiscard]] PackedMesh makeCubeMesh();
    [[nodiscard]] MeshRange appendMesh(PackedMesh& destination, PackedMesh source);
}
