#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include <glm/glm.hpp>

namespace empire::render
{
    struct SourceVertex
    {
        glm::vec3 position{};
        glm::vec3 normal{0.0f, 0.0f, 1.0f};
        glm::vec2 uv{};
        glm::vec4 blendWeights{1.0f, 0.0f, 0.0f, 0.0f};
    };

    struct GpuVertex
    {
        glm::vec4 position{};
        glm::vec4 normalAndColor{};
        glm::vec4 texCoord{};
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
        std::uint32_t meshletOffset = 0;
        std::uint32_t meshletCount = 0;
        glm::vec4 bounds{};
    };

    [[nodiscard]] PackedMesh buildMeshlets(
        std::span<const SourceVertex> vertices, std::span<const std::uint32_t> indices,
        std::uint32_t materialIndex = 0);
    [[nodiscard]] PackedMesh makeCubeMesh();
    [[nodiscard]] MeshRange appendMesh(PackedMesh& destination, PackedMesh source);
}
