#include "render/mesh_builder.hpp"
#include "world/world_scene.hpp"

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace
{
    void require(bool condition, const char* message)
    {
        if (!condition)
            throw std::runtime_error(message);
    }
}

int main()
try
{
    static_assert(sizeof(empire::render::GpuVertex) == 48);
    static_assert(sizeof(empire::render::GpuMeshlet) == 48);
    static_assert(offsetof(empire::render::GpuMeshlet, materialIndex) == 32);
    require(!empire::world::isRenderableModelPath("meshes\\markers\\xmarker.nif"),
        "editor marker path was not filtered");
    require(!empire::world::isRenderableModelPath("meshes\\precombined\\city.nif"),
        "precombined path was not filtered");
    require(empire::world::isRenderableModelPath("meshes\\setdressing\\shoppingcart.nif"),
        "ordinary model path was filtered");

    empire::assets::PlacedReference rotatedReference;
    constexpr float halfPi = 1.57079632679f;
    rotatedReference.rotation[0] = halfPi;
    rotatedReference.rotation[1] = halfPi;
    float qx = 0.0f;
    float qy = 0.0f;
    float qz = 0.0f;
    float qw = 0.0f;
    empire::world::eulerToQuaternion(halfPi, halfPi, 0.0f, qx, qy, qz, qw);
    require(std::abs(qx - 0.5f) < 0.0001f && std::abs(qy - 0.5f) < 0.0001f
            && std::abs(qz + 0.5f) < 0.0001f && std::abs(qw - 0.5f) < 0.0001f,
        "Bethesda ZYX quaternion components changed");
    const glm::vec3 rotated = glm::vec3(empire::world::bethesdaReferenceTransform(rotatedReference)
        * glm::vec4(1.0f, 0.0f, 0.0f, 0.0f));
    require(glm::length(rotated - glm::vec3(0.0f, 0.0f, 1.0f)) < 0.0001f,
        "Bethesda quaternion axis remap is wrong");
    rotatedReference.rotation[0] = 0.0f;
    rotatedReference.rotation[1] = 0.0f;
    rotatedReference.rotation[2] = halfPi;
    const glm::vec3 yawed = glm::vec3(empire::world::bethesdaReferenceTransform(rotatedReference)
        * glm::vec4(1.0f, 0.0f, 0.0f, 0.0f));
    require(glm::length(yawed - glm::vec3(0.0f, 0.0f, -1.0f)) < 0.0001f,
        "Bethesda Z rotation conversion is wrong");

    empire::assets::PlacedReference translatedReference;
    translatedReference.position[0] = 10.0f;
    translatedReference.position[1] = 20.0f;
    translatedReference.position[2] = 30.0f;
    const glm::vec3 translated = glm::vec3(
        empire::world::bethesdaReferenceTransform(translatedReference)[3]);
    require(glm::length(translated - glm::vec3(10.0f, 30.0f, 20.0f)) < 0.0001f,
        "Bethesda reference position mapping is wrong");

    std::vector<empire::render::SourceVertex> vertices;
    std::vector<std::uint32_t> indices;
    for (std::uint32_t triangle = 0; triangle < 300; ++triangle)
    {
        const float x = static_cast<float>(triangle % 30);
        const float z = static_cast<float>(triangle / 30);
        const std::uint32_t first = static_cast<std::uint32_t>(vertices.size());
        vertices.push_back({{x, 0, z}, {0, 1, 0}, {0, 0}});
        vertices.push_back({{x + 0.8f, 0, z}, {0, 1, 0}, {1, 0}});
        vertices.push_back({{x, 0, z + 0.8f}, {0, 1, 0}, {0, 1}});
        indices.insert(indices.end(), {first, first + 1, first + 2});
    }
    vertices.front().blendWeights = {0.1f, 0.2f, 0.3f, 0.4f};
    const auto mesh = empire::render::buildMeshlets(vertices, indices, 7);
    require(std::abs(mesh.vertices.front().normalAndColor.w - 0.1f) < 0.0001f
            && std::abs(mesh.vertices.front().texCoord.z - 0.2f) < 0.0001f
            && std::abs(mesh.vertices.front().texCoord.w - 0.3f) < 0.0001f,
        "terrain blend weights were not packed into the GPU vertex");
    require(mesh.meshlets.size() > 1, "test mesh did not span multiple meshlets");
    for (const auto& meshlet : mesh.meshlets)
    {
        require(meshlet.vertexCount <= 64, "meshlet vertex limit exceeded");
        require(meshlet.triangleCount <= 124, "meshlet triangle limit exceeded");
        require(meshlet.materialIndex == 7, "meshlet material was lost");
        require(meshlet.vertexOffset + meshlet.vertexCount <= mesh.vertices.size(),
            "meshlet vertex range is invalid");
        require(meshlet.indexOffset + meshlet.triangleCount * 3 <= mesh.indices.size(),
            "meshlet index range is invalid");
        for (std::uint32_t index = 0; index < meshlet.triangleCount * 3; ++index)
            require(mesh.indices[meshlet.indexOffset + index] < meshlet.vertexCount,
                "meshlet local index is invalid");
    }
    std::cout << "Mesh builder tests passed\n";
    return 0;
}
catch (const std::exception& error)
{
    std::cerr << error.what() << '\n';
    return 1;
}
