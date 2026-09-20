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
            const auto finite4 = [](const glm::vec4& value) {
                return std::isfinite(value.x) && std::isfinite(value.y)
                    && std::isfinite(value.z) && std::isfinite(value.w);
            };
            if (!finite3(vertex.position)
                || std::max({std::abs(vertex.position.x), std::abs(vertex.position.y),
                    std::abs(vertex.position.z)}) > 10'000'000.0f)
                throw std::invalid_argument("Mesh contains a non-finite or unbounded position");
            if (!finite3(vertex.normal))
                throw std::invalid_argument("Mesh contains a non-finite normal");
            if (!std::isfinite(vertex.uv.x) || !std::isfinite(vertex.uv.y))
                throw std::invalid_argument("Mesh contains a non-finite texture coordinate");
            if (!finite4(vertex.blendWeights) || !finite4(vertex.blendWeightsExtra))
                throw std::invalid_argument("Mesh contains non-finite blend weights");
            if (!finite4(vertex.color))
                throw std::invalid_argument("Mesh contains a non-finite vertex color");
            if (!finite4(vertex.boneWeights)
                || glm::any(glm::greaterThan(vertex.boneIndices, glm::uvec4(255))))
                throw std::invalid_argument("Mesh contains invalid skin weights");
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
                glm::vec4 extraWeights = glm::max(vertex.blendWeightsExtra, glm::vec4(0.0f));
                glm::vec4 boneWeights = glm::max(vertex.boneWeights, glm::vec4(0.0f));
                const float weightSum = weights.x + weights.y + weights.z + weights.w
                    + extraWeights.x + extraWeights.y + extraWeights.z + extraWeights.w;
                if (weightSum <= 0.0f)
                    weights = glm::vec4(1.0f, 0.0f, 0.0f, 0.0f);
                const float boneWeightSum = boneWeights.x + boneWeights.y
                    + boneWeights.z + boneWeights.w;
                if (boneWeightSum > 0.0f)
                    boneWeights /= boneWeightSum;
                const auto packUnorm4x8 = [](const glm::vec4& value) {
                    std::uint32_t packed = 0;
                    for (glm::length_t component = 0; component < value.length(); ++component)
                    {
                        const auto byte = static_cast<std::uint32_t>(std::lround(
                            std::clamp(value[component], 0.0f, 1.0f) * 255.0f));
                        packed |= byte << (component * 8);
                    }
                    return packed;
                };
                result.vertices.push_back({
                    glm::vec4(vertex.position, 1.0f),
                    normalLength > 0.0f
                        ? vertex.normal / normalLength : glm::vec3(0.0f, 1.0f, 0.0f),
                    packUnorm4x8(weights), vertex.uv, packUnorm4x8(extraWeights),
                    packUnorm4x8(glm::clamp(vertex.color, glm::vec4(0.0f), glm::vec4(1.0f))),
                    vertex.boneIndices.x | (vertex.boneIndices.y << 8)
                        | (vertex.boneIndices.z << 16) | (vertex.boneIndices.w << 24),
                    packUnorm4x8(boneWeights)
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
        const bool destinationEmpty = destination.meshlets.empty();
        MeshRange range;
        range.vertexOffset = static_cast<std::uint32_t>(destination.vertices.size());
        range.vertexCount = static_cast<std::uint32_t>(source.vertices.size());
        range.indexOffset = static_cast<std::uint32_t>(destination.indices.size());
        range.indexCount = static_cast<std::uint32_t>(source.indices.size());
        range.meshletOffset = static_cast<std::uint32_t>(destination.meshlets.size());
        range.meshletCount = static_cast<std::uint32_t>(source.meshlets.size());
        range.bounds = source.bounds;
        for (auto& meshlet : source.meshlets)
        {
            meshlet.vertexOffset += range.vertexOffset;
            meshlet.indexOffset += range.indexOffset;
        }
        destination.vertices.insert(destination.vertices.end(),
            std::make_move_iterator(source.vertices.begin()), std::make_move_iterator(source.vertices.end()));
        destination.indices.insert(destination.indices.end(), source.indices.begin(), source.indices.end());
        destination.meshlets.insert(destination.meshlets.end(),
            std::make_move_iterator(source.meshlets.begin()), std::make_move_iterator(source.meshlets.end()));
        if (destinationEmpty)
            destination.bounds = range.bounds;
        else if (range.meshletCount != 0)
        {
            const glm::vec3 destinationCenter(destination.bounds);
            const glm::vec3 sourceCenter(range.bounds);
            const float destinationRadius = destination.bounds.w;
            const float sourceRadius = range.bounds.w;
            const glm::vec3 offset = sourceCenter - destinationCenter;
            const float distance = glm::length(offset);
            if (destinationRadius < distance + sourceRadius)
            {
                if (sourceRadius >= distance + destinationRadius)
                    destination.bounds = range.bounds;
                else if (distance > 0.0f)
                {
                    const float radius = (distance + destinationRadius + sourceRadius) * 0.5f;
                    destination.bounds = glm::vec4(
                        destinationCenter + offset * ((radius - destinationRadius) / distance),
                        radius);
                }
            }
        }
        return range;
    }

    bool detail::fixedTopologyReplacement(const MeshRange& range,
        std::span<const GpuMeshlet> currentMeshlets, const PackedMesh& replacement)
    {
        if (range.vertexCount != replacement.vertices.size()
            || range.indexCount != replacement.indices.size()
            || range.meshletCount != replacement.meshlets.size()
            || static_cast<std::uint64_t>(range.meshletOffset) + range.meshletCount
                > currentMeshlets.size())
            return false;
        for (std::size_t i = 0; i < replacement.meshlets.size(); ++i)
        {
            const auto& previous = currentMeshlets[range.meshletOffset + i];
            const auto& next = replacement.meshlets[i];
            if (next.vertexCount != previous.vertexCount
                || next.triangleCount != previous.triangleCount)
                return false;
        }
        return true;
    }
}
