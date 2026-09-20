#include "render/mesh_builder.hpp"
#include "render/vulkan_renderer.hpp"
#include "world/world_scene.hpp"

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <limits>
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
    static_assert(sizeof(empire::render::GpuVertex) == 64);
    static_assert(offsetof(empire::render::GpuVertex, blendWeights) == 28);
    static_assert(offsetof(empire::render::GpuVertex, blendWeightsExtra) == 40);
    static_assert(offsetof(empire::render::GpuVertex, color) == 44);
    static_assert(offsetof(empire::render::GpuVertex, boneIndices) == 48);
    static_assert(offsetof(empire::render::GpuVertex, boneWeights) == 52);
    static_assert(sizeof(empire::render::GpuMeshlet) == 48);
    static_assert(offsetof(empire::render::GpuMeshlet, materialIndex) == 32);
    static_assert(empire::render::detail::sparseFirstResidentMip(1024, 1024, 11, 6) == 0);
    static_assert(empire::render::detail::sparseFirstResidentMip(2048, 2048, 12, 6) == 1);
    static_assert(empire::render::detail::sparseFirstResidentMip(4096, 4096, 13, 6) == 2);
    static_assert(empire::render::detail::sparseFirstResidentMip(8192, 8192, 2, 2) == 1);
    static_assert(empire::render::detail::sparseFirstResidentMip(4096, 4096, 13, 1) == 1);

    constexpr float nearPlane = 0.1f;
    constexpr float farPlane = 100000.0f;
    const glm::mat4 reverseProjection = empire::render::reverseZPerspectiveRH_ZO(
        glm::radians(70.0f), 16.0f / 9.0f, nearPlane, farPlane);
    const glm::vec4 nearClip = reverseProjection * glm::vec4(0.0f, 0.0f, -nearPlane, 1.0f);
    const glm::vec4 farClip = reverseProjection * glm::vec4(0.0f, 0.0f, -farPlane, 1.0f);
    require(std::abs(nearClip.z / nearClip.w - 1.0f) < 0.0001f,
        "reverse-Z near plane did not map to depth 1");
    require(std::abs(farClip.z / farClip.w) < 0.0001f,
        "reverse-Z far plane did not map to depth 0");

    const glm::mat4 infiniteProjection = empire::render::reverseZInfinitePerspectiveRH_ZO(
        glm::radians(70.0f), 16.0f / 9.0f, nearPlane);
    const glm::vec4 infiniteNear = infiniteProjection
        * glm::vec4(0.0f, 0.0f, -nearPlane, 1.0f);
    const glm::vec4 infiniteFar = infiniteProjection
        * glm::vec4(0.0f, 0.0f, -1.0e9f, 1.0f);
    require(std::abs(infiniteNear.z / infiniteNear.w - 1.0f) < 0.0001f,
        "infinite reverse-Z near plane did not map to depth 1");
    require(infiniteFar.z / infiniteFar.w > 0.0f
            && infiniteFar.z / infiniteFar.w < 0.0000001f,
        "infinite reverse-Z distance did not approach depth 0");

    require(!empire::world::isRenderableModelPath("meshes\\markers\\xmarker.nif"),
        "editor marker path was not filtered");
    require(!empire::world::isRenderableModelPath("meshes\\precombined\\city.nif"),
        "precombined path was not filtered");
    require(empire::world::isRenderableModelPath("meshes\\setdressing\\shoppingcart.nif"),
        "ordinary model path was filtered");
    require(empire::world::isRenderableModelPath("meshes\\scol\\fallout4.esm\\cm0005094c.nif"),
        "combined SCOL model path was filtered");
    const auto demBtr = empire::world::detail::demBtrOrigin(
        "NYC.btr/r00c00/EmpireWorld.32.-32.64.btr");
    require(demBtr == empire::assets::CellCoord{-32, 64},
        "DEM BTR tile filename was not recognized");
    require(!empire::world::detail::demBtrOrigin("EmpireWorld.16.-32.64.btr"),
        "unsupported DEM BTR level was recognized");

    const glm::vec4 fineDemRange = empire::world::detail::demTerrainLodRange(false);
    const glm::vec4 coarseDemRange = empire::world::detail::demTerrainLodRange(true);
    require(fineDemRange == glm::vec4(0.0f, 0.0f, 12288.0f, 16384.0f),
        "DEM LAND fade range changed");
    require(coarseDemRange == glm::vec4(12288.0f, 16384.0f,
            empire::world::detail::demHorizonFadeStart,
            empire::world::detail::demHorizonFadeEnd),
        "DEM BTR fade range changed");
    require(fineDemRange.z == coarseDemRange.x
            && fineDemRange.w == coarseDemRange.y,
        "DEM LAND and BTR fade endpoints are not complementary");
    require(empire::world::detail::demWaterLodRange()
            == glm::vec4(0.0f, 0.0f,
                empire::world::detail::demHorizonFadeStart,
                empire::world::detail::demHorizonFadeEnd),
        "DEM water does not share the BTR horizon fade");
    require(empire::world::detail::demHorizonFadeEnd < 96.0f * 4096.0f,
        "DEM horizon fade reaches beyond guaranteed BTR residency");

    std::vector<empire::render::SourceVertex> legacyBtr(33 * 33);
    for (std::size_t y = 0; y < 33; ++y)
        for (std::size_t x = 0; x < 33; ++x)
            legacyBtr[y * 33 + x].position = {
                x < 16 ? static_cast<float>(x * 4096) : std::numeric_limits<float>::infinity(),
                static_cast<float>(x + y) * 8.0f,
                y < 16 ? static_cast<float>(y * 4096) : std::numeric_limits<float>::infinity()};
    require(empire::world::detail::repairLegacyDemBtrGrid(legacyBtr),
        "legacy overflowing DEM BTR was not repaired");
    require(legacyBtr.back().position == glm::vec3(131072.0f, 512.0f, 131072.0f),
        "legacy DEM BTR grid dimensions changed during repair");
    require(std::isfinite(legacyBtr.back().normal.x)
            && std::isfinite(legacyBtr.back().normal.y)
            && std::isfinite(legacyBtr.back().normal.z),
        "legacy DEM BTR repair produced an invalid normal");
    require(!empire::world::detail::repairLegacyDemBtrGrid(legacyBtr),
        "valid DEM BTR was repaired a second time");
    legacyBtr.front().position.y = std::numeric_limits<float>::infinity();
    std::vector<float> fallbackHeights(33 * 33, 384.0f);
    require(empire::world::detail::repairLegacyDemBtrGrid(legacyBtr, fallbackHeights),
        "legacy DEM BTR height overflow was not repaired from LAND");
    require(legacyBtr.front().position.y == 384.0f,
        "legacy DEM BTR LAND height fallback changed");

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

    empire::assets::PlacedReference collectionPart;
    collectionPart.rotation[2] = halfPi;
    collectionPart.scale = 2.0f;
    collectionPart.localPosition[0] = 10.0f;
    collectionPart.localRotation[2] = halfPi;
    collectionPart.localScale = 0.5f;
    collectionPart.hasLocalTransform = true;
    const auto collectionTransform = empire::world::bethesdaReferenceTransform(collectionPart);
    require(glm::length(glm::vec3(collectionTransform[3]) - glm::vec3(0.0f, 0.0f, 20.0f))
            < 0.0001f,
        "SCOL local translation was not rotated by its parent");
    require(std::abs(glm::length(glm::vec3(collectionTransform[0])) - 1.0f) < 0.0001f,
        "SCOL parent and local scales were not composed");
    require(glm::length(glm::vec3(collectionTransform * glm::vec4(1.0f, 0.0f, 0.0f, 0.0f))
            - glm::vec3(-1.0f, 0.0f, 0.0f)) < 0.0001f,
        "SCOL parent and local rotations were not composed");

    const auto boundedMesh = [](glm::vec4 bounds) {
        empire::render::PackedMesh result;
        result.meshlets.resize(1);
        result.bounds = bounds;
        return result;
    };
    empire::render::PackedMesh combined;
    const auto firstRange = empire::render::appendMesh(
        combined, boundedMesh({0.0f, 0.0f, 0.0f, 1.0f}));
    require(firstRange.bounds == glm::vec4(0.0f, 0.0f, 0.0f, 1.0f)
            && combined.bounds == firstRange.bounds,
        "first appended mesh bounds changed");
    const auto secondRange = empire::render::appendMesh(
        combined, boundedMesh({4.0f, 0.0f, 0.0f, 1.0f}));
    require(secondRange.bounds == glm::vec4(4.0f, 0.0f, 0.0f, 1.0f),
        "appendMesh changed the source range bounds");
    require(glm::length(glm::vec3(combined.bounds) - glm::vec3(2.0f, 0.0f, 0.0f)) < 0.0001f
            && std::abs(combined.bounds.w - 3.0f) < 0.0001f,
        "disjoint mesh bounds were not unioned");
    (void)empire::render::appendMesh(
        combined, boundedMesh({2.0f, 0.0f, 0.0f, 0.5f}));
    require(glm::length(glm::vec3(combined.bounds) - glm::vec3(2.0f, 0.0f, 0.0f)) < 0.0001f
            && std::abs(combined.bounds.w - 3.0f) < 0.0001f,
        "contained mesh bounds enlarged the union");
    (void)empire::render::appendMesh(
        combined, boundedMesh({2.0f, 0.0f, 0.0f, 5.0f}));
    require(combined.bounds == glm::vec4(2.0f, 0.0f, 0.0f, 5.0f),
        "containing mesh bounds did not replace the union");
    (void)empire::render::appendMesh(
        combined, boundedMesh({2.0f, 0.0f, 0.0f, 7.0f}));
    require(combined.bounds == glm::vec4(2.0f, 0.0f, 0.0f, 7.0f),
        "coincident mesh bounds were not unioned safely");
    (void)empire::render::appendMesh(
        combined, boundedMesh({12.0f, 0.0f, 0.0f, 1.0f}));
    require(glm::length(glm::vec3(combined.bounds) - glm::vec3(4.0f, 0.0f, 0.0f)) < 0.0001f
            && std::abs(combined.bounds.w - 9.0f) < 0.0001f,
        "multi-append mesh bounds were not unioned");

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
    vertices.front().blendWeights = {64.0f / 255.0f, 32.0f / 255.0f,
        16.0f / 255.0f, 8.0f / 255.0f};
    vertices.front().blendWeightsExtra = {4.0f / 255.0f, 2.0f / 255.0f,
        1.0f / 255.0f, 1.0f};
    vertices.front().color = {17.0f / 255.0f, 34.0f / 255.0f,
        51.0f / 255.0f, 68.0f / 255.0f};
    vertices.front().boneIndices = {1, 7, 12, 255};
    vertices.front().boneWeights = {0.5f, 0.25f, 0.125f, 0.125f};
    const auto mesh = empire::render::buildMeshlets(vertices, indices, 7);
    require(mesh.vertices.front().blendWeights == 0x08102040u
            && mesh.vertices.front().blendWeightsExtra == 0xFF010204u,
        "terrain blend weights were normalized instead of packed independently");
    require(mesh.vertices.front().color == 0x44332211u,
        "terrain vertex color was not packed into the GPU vertex");
    require(mesh.vertices.front().boneIndices == 0xFF0C0701u,
        "skin bone indices were not packed into the GPU vertex");
    require(mesh.vertices.front().boneWeights == 0x20204080u,
        "skin weights were not normalized and packed into the GPU vertex");
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

    const auto cube = empire::render::makeCubeMesh();
    empire::render::PackedMesh topologyScene;
    const auto topologyRange = empire::render::appendMesh(topologyScene, cube);
    require(topologyRange.vertexOffset == 0
            && topologyRange.vertexCount == cube.vertices.size()
            && topologyRange.indexOffset == 0
            && topologyRange.indexCount == cube.indices.size(),
        "MeshRange did not retain its vertex/index spans");
    require(empire::render::detail::fixedTopologyReplacement(
            topologyRange, topologyScene.meshlets, cube),
        "valid fixed-topology mesh replacement was rejected");
    auto changedTopology = cube;
    changedTopology.vertices.pop_back();
    require(!empire::render::detail::fixedTopologyReplacement(
            topologyRange, topologyScene.meshlets, changedTopology),
        "mesh replacement accepted a changed vertex span");
    changedTopology = cube;
    --changedTopology.meshlets.front().triangleCount;
    require(!empire::render::detail::fixedTopologyReplacement(
            topologyRange, topologyScene.meshlets, changedTopology),
        "mesh replacement accepted a changed meshlet partition");

    auto landscape = empire::assets::createEditableLandscape();
    require(empire::world::detail::applyLandscapeBrush(landscape, {0.0f, 0.0f},
            empire::world::LandscapeTool::raise, 512.0f, 256.0f),
        "raise brush made no change");
    require(landscape.height(32, 32) == 256.0f, "raise brush strength changed");
    (void)empire::world::detail::applyLandscapeBrush(landscape, {0.0f, 0.0f},
        empire::world::LandscapeTool::lower, 512.0f, 128.0f);
    require(landscape.height(32, 32) == 128.0f, "lower brush strength changed");
    (void)empire::world::detail::applyLandscapeBrush(landscape, {0.0f, 0.0f},
        empire::world::LandscapeTool::flatten, 512.0f, 1024.0f, 64.0f);
    require(landscape.height(32, 32) == 64.0f, "flatten brush target changed");

    landscape = empire::assets::createEditableLandscape();
    landscape.height(32, 32) = 512.0f;
    (void)empire::world::detail::applyLandscapeBrush(landscape, {0.0f, 0.0f},
        empire::world::LandscapeTool::smooth, 256.0f, 1024.0f);
    require(landscape.height(32, 32) == 0.0f, "smooth brush ignored the eight-neighbor average");
    landscape.height(1, 0) = 4096.0f;
    empire::world::detail::quantizeLandscape(landscape);
    require(landscape.height(1, 0) == 1016.0f,
        "landscape quantization did not clamp the signed VHGT delta");

    empire::world::detail::LandscapeHistory history;
    auto beforeStroke = empire::assets::createEditableLandscape();
    landscape = beforeStroke;
    landscape.height(10, 10) = 8.0f;
    history.commit(beforeStroke);
    require(landscape.addCell({2, 0}), "history test could not add a CELL");
    history.addCell({2, 0});
    require(history.undo(landscape) && landscape.height(10, 10) == 0.0f
            && landscape.containsCell({2, 0}),
        "landscape undo failed");
    require(history.redo(landscape) && landscape.height(10, 10) == 8.0f
            && landscape.containsCell({2, 0}),
        "landscape redo failed");
    std::cout << "Mesh builder tests passed\n";
    return 0;
}
catch (const std::exception& error)
{
    std::cerr << error.what() << '\n';
    return 1;
}
