#include "render/mesh_builder.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace empire::render
{
    PackedMesh buildMeshlets(
        std::span<const SourceVertex> vertices, std::span<const std::uint32_t> indices,
        std::uint32_t materialIndex)
    {
        if (vertices.empty() || indices.empty() || indices.size() % 3 != 0)
            throw std::invalid_argument("Mesh input must contain vertices and complete triangles");
        if (std::ranges::any_of(indices,
            [vertexCount = vertices.size()](std::uint32_t index) { return index >= vertexCount; }))
            throw std::invalid_argument("Mesh index is outside the vertex array");
        for (const auto& vertex : vertices)
        {
            const auto finite3 = [](const glm::vec3& value) {
                return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
            };
            if (!finite3(vertex.position) || !finite3(vertex.normal)
                || !std::isfinite(vertex.uv.x) || !std::isfinite(vertex.uv.y)
                || !std::isfinite(vertex.blendWeights.x) || !std::isfinite(vertex.blendWeights.y)
                || !std::isfinite(vertex.blendWeights.z) || !std::isfinite(vertex.blendWeights.w)
                || std::max({std::abs(vertex.position.x), std::abs(vertex.position.y),
                    std::abs(vertex.position.z)}) > 10'000'000.0f)
                throw std::invalid_argument("Mesh contains a non-finite or unbounded vertex");
        }

        PackedMesh result;
        glm::vec3 minimum = vertices.front().position;
        glm::vec3 maximum = vertices.front().position;
        for (const auto& vertex : vertices)
        {
            minimum = glm::min(minimum, vertex.position);
            maximum = glm::max(maximum, vertex.position);
        }
        const glm::vec3 center = (minimum + maximum) * 0.5f;
        float radius = 0.0f;
        for (const auto& vertex : vertices)
            radius = std::max(radius, glm::distance(center, vertex.position));
        result.bounds = glm::vec4(center, radius);

        constexpr std::size_t maxVertices = 64;
        constexpr std::size_t maxTriangles = 124;
        std::vector<std::uint32_t> sourceVertices;
        std::vector<std::uint32_t> localIndices;
        sourceVertices.reserve(maxVertices);
        localIndices.reserve(maxTriangles * 3);

        const auto flush = [&]()
        {
            if (localIndices.empty())
                return;
            GpuMeshlet meshlet;
            meshlet.vertexOffset = static_cast<std::uint32_t>(result.vertices.size());
            meshlet.indexOffset = static_cast<std::uint32_t>(result.indices.size());
            meshlet.vertexCount = static_cast<std::uint32_t>(sourceVertices.size());
            meshlet.triangleCount = static_cast<std::uint32_t>(localIndices.size() / 3);
            meshlet.materialIndex = materialIndex;
            glm::vec3 meshletMinimum = vertices[sourceVertices.front()].position;
            glm::vec3 meshletMaximum = meshletMinimum;
            for (const std::uint32_t sourceIndex : sourceVertices)
            {
                const SourceVertex& vertex = vertices[sourceIndex];
                const float normalLength = glm::length(vertex.normal);
                glm::vec4 weights = glm::max(vertex.blendWeights, glm::vec4(0.0f));
                const float weightSum = weights.x + weights.y + weights.z + weights.w;
                weights = weightSum > 0.0f
                    ? weights / weightSum : glm::vec4(1.0f, 0.0f, 0.0f, 0.0f);
                result.vertices.push_back({
                    glm::vec4(vertex.position, 1.0f),
                    glm::vec4(normalLength > 0.0f
                        ? vertex.normal / normalLength : glm::vec3(0.0f, 1.0f, 0.0f), weights.x),
                    glm::vec4(vertex.uv, weights.y, weights.z)
                });
                meshletMinimum = glm::min(meshletMinimum, vertex.position);
                meshletMaximum = glm::max(meshletMaximum, vertex.position);
            }
            result.indices.insert(result.indices.end(), localIndices.begin(), localIndices.end());
            const glm::vec3 meshletCenter = (meshletMinimum + meshletMaximum) * 0.5f;
            float meshletRadius = 0.0f;
            for (const std::uint32_t sourceIndex : sourceVertices)
                meshletRadius = std::max(meshletRadius,
                    glm::distance(meshletCenter, vertices[sourceIndex].position));
            meshlet.sphere = glm::vec4(meshletCenter, meshletRadius);
            result.meshlets.push_back(meshlet);
            sourceVertices.clear();
            localIndices.clear();
        };

        for (std::size_t triangle = 0; triangle < indices.size(); triangle += 3)
        {
            std::array<std::uint32_t, 3> local{};
            std::size_t newVertices = 0;
            for (std::size_t corner = 0; corner < 3; ++corner)
            {
                const auto found = std::ranges::find(sourceVertices, indices[triangle + corner]);
                if (found == sourceVertices.end())
                    ++newVertices;
                else
                    local[corner] = static_cast<std::uint32_t>(found - sourceVertices.begin());
            }
            if (localIndices.size() / 3 == maxTriangles
                || sourceVertices.size() + newVertices > maxVertices)
                flush();
            for (std::size_t corner = 0; corner < 3; ++corner)
            {
                const std::uint32_t sourceIndex = indices[triangle + corner];
                const auto found = std::ranges::find(sourceVertices, sourceIndex);
                if (found == sourceVertices.end())
                {
                    local[corner] = static_cast<std::uint32_t>(sourceVertices.size());
                    sourceVertices.push_back(sourceIndex);
                }
                else
                    local[corner] = static_cast<std::uint32_t>(found - sourceVertices.begin());
            }
            localIndices.insert(localIndices.end(), local.begin(), local.end());
        }
        flush();
        return result;
    }

    PackedMesh makeCubeMesh()
    {
        constexpr std::array positions{
            glm::vec3{-1.0f, -1.0f,  1.0f}, glm::vec3{ 1.0f, -1.0f,  1.0f},
            glm::vec3{ 1.0f,  1.0f,  1.0f}, glm::vec3{-1.0f,  1.0f,  1.0f},
            glm::vec3{ 1.0f, -1.0f, -1.0f}, glm::vec3{-1.0f, -1.0f, -1.0f},
            glm::vec3{-1.0f,  1.0f, -1.0f}, glm::vec3{ 1.0f,  1.0f, -1.0f},
            glm::vec3{-1.0f, -1.0f, -1.0f}, glm::vec3{-1.0f, -1.0f,  1.0f},
            glm::vec3{-1.0f,  1.0f,  1.0f}, glm::vec3{-1.0f,  1.0f, -1.0f},
            glm::vec3{ 1.0f, -1.0f,  1.0f}, glm::vec3{ 1.0f, -1.0f, -1.0f},
            glm::vec3{ 1.0f,  1.0f, -1.0f}, glm::vec3{ 1.0f,  1.0f,  1.0f},
            glm::vec3{-1.0f,  1.0f,  1.0f}, glm::vec3{ 1.0f,  1.0f,  1.0f},
            glm::vec3{ 1.0f,  1.0f, -1.0f}, glm::vec3{-1.0f,  1.0f, -1.0f},
            glm::vec3{-1.0f, -1.0f, -1.0f}, glm::vec3{ 1.0f, -1.0f, -1.0f},
            glm::vec3{ 1.0f, -1.0f,  1.0f}, glm::vec3{-1.0f, -1.0f,  1.0f}
        };
        constexpr std::array normals{
            glm::vec3{0, 0, 1}, glm::vec3{0, 0, -1}, glm::vec3{-1, 0, 0},
            glm::vec3{1, 0, 0}, glm::vec3{0, 1, 0}, glm::vec3{0, -1, 0}
        };
        std::vector<SourceVertex> vertices;
        vertices.reserve(24);
        for (std::size_t face = 0; face < 6; ++face)
        {
            for (std::size_t corner = 0; corner < 4; ++corner)
                vertices.push_back({positions[face * 4 + corner], normals[face], {}});
        }
        std::vector<std::uint32_t> indices;
        indices.reserve(36);
        for (std::uint32_t face = 0; face < 6; ++face)
        {
            const std::uint32_t base = face * 4;
            indices.insert(indices.end(), {base, base + 1, base + 2, base, base + 2, base + 3});
        }
        return buildMeshlets(vertices, indices);
    }

    MeshRange appendMesh(PackedMesh& destination, PackedMesh source)
    {
        MeshRange range;
        range.meshletOffset = static_cast<std::uint32_t>(destination.meshlets.size());
        range.meshletCount = static_cast<std::uint32_t>(source.meshlets.size());
        range.bounds = source.bounds;
        const std::uint32_t vertexOffset = static_cast<std::uint32_t>(destination.vertices.size());
        const std::uint32_t indexOffset = static_cast<std::uint32_t>(destination.indices.size());
        for (auto& meshlet : source.meshlets)
        {
            meshlet.vertexOffset += vertexOffset;
            meshlet.indexOffset += indexOffset;
        }
        destination.vertices.insert(destination.vertices.end(),
            std::make_move_iterator(source.vertices.begin()), std::make_move_iterator(source.vertices.end()));
        destination.indices.insert(destination.indices.end(), source.indices.begin(), source.indices.end());
        destination.meshlets.insert(destination.meshlets.end(),
            std::make_move_iterator(source.meshlets.begin()), std::make_move_iterator(source.meshlets.end()));
        if (range.meshletOffset == 0)
            destination.bounds = range.bounds;
        return range;
    }
}
