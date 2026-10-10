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

#include <glm/ext/matrix_transform.hpp>

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
    {
        openf4::assets::ModelGeometry geometry;
        auto& surface = geometry.surfaces.emplace_back();
        for (int triangle = 0; triangle < 12; ++triangle)
        {
            const float z = -3.0f * triangle;
            for (const glm::vec3 position : {glm::vec3{0.0f, 0.0f, z},
                     glm::vec3{2.0f, 0.0f, z}, glm::vec3{0.0f, 2.0f, z}})
            {
                surface.indices.push_back(static_cast<std::uint32_t>(surface.vertices.size()));
                surface.vertices.push_back({position});
            }
        }
        // Unreferenced source vertices must not influence grounding or selection bounds.
        surface.vertices.push_back({glm::vec3(1.0e8f)});
        const openf4::world::detail::PickingGeometry picking(geometry);
        const auto hit = picking.raycast({0.5, 0.5, 5.0}, {0.0, 0.0, -1.0}, glm::mat4(1.0f));
        require(hit && std::abs(*hit - 5.0) < 1.0e-9,
            "BVH picking did not select the nearest triangle");
        require(!picking.raycast({1.5, 1.5, 5.0}, {0.0, 0.0, -1.0}, glm::mat4(1.0f)),
            "Picking accepted a sphere/box hit outside the model triangles");
        require(!picking.raycast({0.5, 0.5, 5.0}, {0.0, 0.0, -1.0}, glm::mat4(1.0f), 4.0),
            "Picking ignored the terrain-occlusion distance limit");
        const auto inside = picking.raycast({0.5, 0.5, -1.0}, {0.0, 0.0, -1.0}, glm::mat4(1.0f));
        require(inside && std::abs(*inside - 2.0) < 1.0e-9,
            "Picking from inside the model bounds missed the next triangle");
        const auto transform = glm::translate(glm::mat4(1.0f), glm::vec3(4000000.0f, 20.0f, -8000000.0f))
            * glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f))
            * glm::scale(glm::mat4(1.0f), glm::vec3(3.0f));
        const glm::dmat4 preciseTransform(transform);
        const auto origin = glm::dvec3(preciseTransform * glm::dvec4(0.5, 0.5, 5.0, 1.0));
        const auto direction = glm::normalize(glm::dvec3(preciseTransform * glm::dvec4(0.0, 0.0, -1.0, 0.0)));
        const auto transformedHit = picking.raycast(origin, direction, transform);
        require(transformedHit && std::abs(*transformedHit - 15.0) < 1.0e-5,
            "Large-coordinate rotated/scaled picking lost world-space distance");
        require(!picking.raycast(origin, direction, transform, 14.0),
            "Scaled picking interpreted the distance limit in local units");
        require(!picking.raycast(origin, {}, transform)
            && !picking.raycast(origin, direction, glm::mat4(0.0f)),
            "Picking accepted an invalid ray or singular transform");

        surface.indices.resize(3);
        const openf4::world::detail::PickingGeometry triangle(geometry);
        const auto bounds = triangle.bounds(glm::rotate(glm::mat4(1.0f),
            glm::radians(45.0f), glm::vec3(0.0f, 0.0f, 1.0f)));
        require(std::abs(bounds.maximum.y - std::sqrt(2.0)) < 1.0e-6,
            "Geometry bounds used a transformed box or unreferenced vertices instead of the geometry");
    }
    {
        auto landscape = openf4::assets::createEditableLandscape();
        landscape.setSample({0, 0}, 0.0f);
        landscape.setSample({1, 0}, 128.0f);
        landscape.setSample({0, 1}, 256.0f);
        landscape.setSample({1, 1}, 512.0f);
        const auto near = openf4::world::detail::landscapeSurface(landscape, {32.0, 32.0});
        const auto far = openf4::world::detail::landscapeSurface(landscape, {96.0, 96.0});
        require(near && far && std::abs(near->height - 96.0) < 1.0e-9
            && std::abs(far->height - 352.0) < 1.0e-9,
            "LAND surface query does not match the rendered two-triangle topology");
        require(glm::length(near->normal - glm::normalize(glm::vec3(-1.0f, 1.0f, -2.0f))) < 1.0e-6f
            && glm::length(far->normal - glm::normalize(glm::vec3(-2.0f, 1.0f, -3.0f))) < 1.0e-6f,
            "LAND grounding normal did not match its triangle slope");
        const auto flat = openf4::world::detail::landscapeSurface(landscape, {4096.0, 64.0});
        require(flat && flat->height == 0.0 && flat->normal == glm::vec3(0.0f, 1.0f, 0.0f),
            "Flat LAND query failed at a cell seam");
        require(!openf4::world::detail::landscapeSurface(landscape, {-8192.0, 0.0})
            && !openf4::world::detail::landscapeSurface(landscape,
                {std::numeric_limits<double>::infinity(), 0.0})
            && landscape.cellForms.size() == 9,
            "Grounding query created missing terrain or accepted invalid coordinates");
    }
    for (const glm::vec3 angles : {glm::vec3(0.0f), glm::vec3(0.2f, -0.7f, 1.1f),
             glm::vec3(1.2f, 1.5707963f, -0.4f), glm::vec3(-2.0f, -1.5707963f, 0.8f)})
    {
        openf4::assets::PlacedReference source;
        source.position[0] = -4096.0f;
        source.position[1] = 8192.0f;
        source.position[2] = 126.0f;
        source.scale = 2.5f;
        for (int i = 0; i < 3; ++i)
            source.rotation[i] = angles[i];
        const auto matrix = openf4::world::bethesdaReferenceTransform(source);
        openf4::assets::PlacedReference recovered;
        openf4::world::setBethesdaReferenceTransform(recovered, matrix);
        const auto roundtrip = openf4::world::bethesdaReferenceTransform(recovered);
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                require(std::abs(matrix[i][j] - roundtrip[i][j]) < 0.002f,
                    "Authored transform changed after Bethesda round-trip");
        require(recovered.cell == openf4::assets::CellCoord{-1, 2},
            "Reference cell migration did not use Bethesda horizontal axes");
    }
    {
        auto invalid = glm::mat4(1.0f);
        invalid[0][0] = 2.0f;
        openf4::assets::PlacedReference reference;
        bool rejected = false;
        try { openf4::world::setBethesdaReferenceTransform(reference, invalid); }
        catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "ESM transform accepted unsupported nonuniform scale");
    }
    static_assert(sizeof(openf4::render::GpuVertex) == 64);
    require(!openf4::render::SceneMaterial{}.normalMapped,
        "Normal mapping must remain opt-in for existing Commonwealth materials");
    static_assert(offsetof(openf4::render::GpuVertex, blendWeights) == 28);
    static_assert(offsetof(openf4::render::GpuVertex, blendWeightsExtra) == 40);
    static_assert(offsetof(openf4::render::GpuVertex, color) == 44);
    static_assert(offsetof(openf4::render::GpuVertex, boneIndices) == 48);
    static_assert(offsetof(openf4::render::GpuVertex, boneWeights) == 52);
    static_assert(sizeof(openf4::render::GpuMeshlet) == 48);
    static_assert(offsetof(openf4::render::GpuMeshlet, materialIndex) == 32);
    static_assert(openf4::render::detail::sparseFirstResidentMip(1024, 1024, 11, 6) == 0);
    static_assert(openf4::render::detail::sparseFirstResidentMip(2048, 2048, 12, 6) == 1);
    static_assert(openf4::render::detail::sparseFirstResidentMip(4096, 4096, 13, 6) == 2);
    static_assert(openf4::render::detail::sparseFirstResidentMip(8192, 8192, 2, 2) == 1);
    static_assert(openf4::render::detail::sparseFirstResidentMip(4096, 4096, 13, 1) == 1);

    constexpr float nearPlane = 0.1f;
    constexpr float farPlane = 100000.0f;
    const glm::mat4 reverseProjection = openf4::render::reverseZPerspectiveRH_ZO(
        glm::radians(70.0f), 16.0f / 9.0f, nearPlane, farPlane);
    const glm::vec4 nearClip = reverseProjection * glm::vec4(0.0f, 0.0f, -nearPlane, 1.0f);
    const glm::vec4 farClip = reverseProjection * glm::vec4(0.0f, 0.0f, -farPlane, 1.0f);
    require(std::abs(nearClip.z / nearClip.w - 1.0f) < 0.0001f,
        "reverse-Z near plane did not map to depth 1");
    require(std::abs(farClip.z / farClip.w) < 0.0001f,
        "reverse-Z far plane did not map to depth 0");

    const glm::mat4 infiniteProjection = openf4::render::reverseZInfinitePerspectiveRH_ZO(
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

    require(!openf4::world::isRenderableModelPath("meshes\\markers\\xmarker.nif"),
        "editor marker path was not filtered");
    require(!openf4::world::isRenderableModelPath("meshes\\precombined\\city.nif"),
        "precombined path was not filtered");
    require(openf4::world::isRenderableModelPath("meshes\\setdressing\\shoppingcart.nif"),
        "ordinary model path was filtered");
    require(openf4::world::isRenderableModelPath("meshes\\scol\\fallout4.esm\\cm0005094c.nif"),
        "combined SCOL model path was filtered");
    const auto demBtr = openf4::world::detail::demBtrOrigin(
        "NYC.btr/r00c00/OpenF4World.32.-32.64.btr");
    require(demBtr == openf4::assets::CellCoord{-32, 64},
        "DEM BTR tile filename was not recognized");
    const std::filesystem::path legacyBtrPath = "NYC.btr/r00c00/EmpireWorld.32.-32.64.btr";
    const std::filesystem::path currentBtrPath = "NYC.btr/r00c00/OpenF4World.32.-32.64.btr";
    require(openf4::world::detail::demBtrOrigin(legacyBtrPath) == demBtr,
        "Legacy DEM BTR tile filename was not recognized");
    require(openf4::world::detail::preferDemBtr(currentBtrPath, legacyBtrPath)
        && !openf4::world::detail::preferDemBtr(legacyBtrPath, currentBtrPath)
        && !openf4::world::detail::preferDemBtr("openf4world.32.0.0.btr", legacyBtrPath),
        "Mixed BTR discovery must prefer OpenF4 only for the same tile");
    require(!openf4::world::detail::demBtrOrigin("OpenF4World.16.-32.64.btr"),
        "unsupported DEM BTR level was recognized");

    const glm::vec4 fineDemRange = openf4::world::detail::demTerrainLodRange(false);
    const glm::vec4 coarseDemRange = openf4::world::detail::demTerrainLodRange(true);
    require(fineDemRange == glm::vec4(0.0f, 0.0f, 12288.0f, 16384.0f),
        "DEM LAND fade range changed");
    require(coarseDemRange == glm::vec4(12288.0f, 16384.0f,
            openf4::world::detail::demHorizonFadeStart,
            openf4::world::detail::demHorizonFadeEnd),
        "DEM BTR fade range changed");
    require(fineDemRange.z == coarseDemRange.x
            && fineDemRange.w == coarseDemRange.y,
        "DEM LAND and BTR fade endpoints are not complementary");
    require(openf4::world::detail::demWaterLodRange()
            == glm::vec4(0.0f, 0.0f,
                openf4::world::detail::demHorizonFadeStart,
                openf4::world::detail::demHorizonFadeEnd),
        "DEM water does not share the BTR horizon fade");
    require(openf4::world::detail::demHorizonFadeEnd < 96.0f * 4096.0f,
        "DEM horizon fade reaches beyond guaranteed BTR residency");

    {
        using openf4::assets::LodTerrainTile;
        using openf4::world::detail::selectDemTerrainLod;
        const LodTerrainTile fine{4, {0, 0}}, medium{8, {0, 0}},
            distant{16, {0, 0}}, coarse{32, {0, 0}};
        std::map<LodTerrainTile, bool> desired{
            {fine, true}, {medium, true}, {distant, true}, {coarse, true}};
        auto selected = selectDemTerrainLod(desired, {});
        require(selected.size() == 4,
            "Clipped worlds must use resident layers without requiring 49 tiles per level");
        for (const auto key : {fine, medium, distant, coarse})
            require(selected.at(key) == openf4::world::detail::demBtrLodRange(key.level),
                "Complete terrain hierarchy changed its fade thresholds");
        desired[medium] = false;
        selected = selectDemTerrainLod(desired, selected);
        require(!selected.contains(medium) && selected.contains(fine)
            && selected.at(distant).x == selected.at(fine).z
            && selected.at(distant).y == selected.at(fine).w,
            "A loading intermediate layer must be bridged by its loaded ancestor");
        const LodTerrainTile unpublishedSibling{4, {4, 0}};
        desired[unpublishedSibling] = false;
        selected = selectDemTerrainLod(desired, selected);
        require(!selected.contains(fine) && selected.at(distant).x == fineDemRange.z,
            "An indexed but unpublished sibling must retain ancestor coverage after cancellation");
        desired[fine] = false;
        selected = selectDemTerrainLod(desired, selected, false);
        require(selected.at(distant).x == 0.0f && selected.at(distant).y == 0.0f,
            "Coarse terrain must cover the inner window while LAND is loading");
        desired[coarse] = false;
        selected = selectDemTerrainLod(desired, selected);
        require(selected.at(distant).w == openf4::world::detail::demHorizonFadeEnd,
            "Loaded descendants must remain visible while a parent loads");
        const LodTerrainTile negative{32, {-32, -32}};
        desired[negative] = true;
        selected = selectDemTerrainLod(desired, selected);
        require(selected.contains(negative) && selected.contains(distant),
            "Negative-coordinate regions must advance independently");
        const auto retained = selectDemTerrainLod({{coarse, false}}, selected);
        require(retained.contains(distant) && retained.contains(negative),
            "Pending replacement roots must retain the displayed hierarchy");
        const auto retainedWithoutLand = selectDemTerrainLod({{coarse, false}}, selected, false);
        require(retainedWithoutLand.at(distant).x == 0.0f,
            "Retained descendants must cover LAND during a pending root handoff");

        std::vector<openf4::render::SourceVertex> cellVertices(4);
        cellVertices[0].position = {0.0f, 0.0f, 0.0f};
        cellVertices[1].position = {4096.0f, 0.0f, 0.0f};
        cellVertices[2].position = {0.0f, 0.0f, 4096.0f};
        cellVertices[3].position = {4096.0f, 0.0f, 4096.0f};
        const std::vector<std::uint32_t> cellIndices{0, 1, 2, 1, 3, 2};
        const auto clipped = openf4::render::buildMeshlets(cellVertices, cellIndices);
        const auto coverage = openf4::world::detail::demTerrainCoverage(clipped, negative);
        require(coverage.size() == 1 && coverage.contains({-32, -32})
            && !coverage.contains({-31, -32}),
            "BTR fallback must leave absent or newly authored LAND cells visible");
    }

    std::vector<openf4::render::SourceVertex> legacyBtr(33 * 33);
    for (std::size_t y = 0; y < 33; ++y)
        for (std::size_t x = 0; x < 33; ++x)
            legacyBtr[y * 33 + x].position = {
                x < 16 ? static_cast<float>(x * 4096) : std::numeric_limits<float>::infinity(),
                static_cast<float>(x + y) * 8.0f,
                y < 16 ? static_cast<float>(y * 4096) : std::numeric_limits<float>::infinity()};
    require(openf4::world::detail::repairLegacyDemBtrGrid(legacyBtr),
        "legacy overflowing DEM BTR was not repaired");
    require(legacyBtr.back().position == glm::vec3(131072.0f, 512.0f, 131072.0f),
        "legacy DEM BTR grid dimensions changed during repair");
    require(std::isfinite(legacyBtr.back().normal.x)
            && std::isfinite(legacyBtr.back().normal.y)
            && std::isfinite(legacyBtr.back().normal.z),
        "legacy DEM BTR repair produced an invalid normal");
    require(!openf4::world::detail::repairLegacyDemBtrGrid(legacyBtr),
        "valid DEM BTR was repaired a second time");
    legacyBtr.front().position.y = std::numeric_limits<float>::infinity();
    std::vector<float> fallbackHeights(33 * 33, 384.0f);
    require(openf4::world::detail::repairLegacyDemBtrGrid(legacyBtr, fallbackHeights),
        "legacy DEM BTR height overflow was not repaired from LAND");
    require(legacyBtr.front().position.y == 384.0f,
        "legacy DEM BTR LAND height fallback changed");

    openf4::assets::PlacedReference rotatedReference;
    constexpr float halfPi = 1.57079632679f;
    rotatedReference.rotation[0] = halfPi;
    rotatedReference.rotation[1] = halfPi;
    float qx = 0.0f;
    float qy = 0.0f;
    float qz = 0.0f;
    float qw = 0.0f;
    openf4::world::eulerToQuaternion(halfPi, halfPi, 0.0f, qx, qy, qz, qw);
    require(std::abs(qx - 0.5f) < 0.0001f && std::abs(qy - 0.5f) < 0.0001f
            && std::abs(qz + 0.5f) < 0.0001f && std::abs(qw - 0.5f) < 0.0001f,
        "Bethesda ZYX quaternion components changed");
    const glm::vec3 rotated = glm::vec3(openf4::world::bethesdaReferenceTransform(rotatedReference)
        * glm::vec4(1.0f, 0.0f, 0.0f, 0.0f));
    require(glm::length(rotated - glm::vec3(0.0f, 0.0f, 1.0f)) < 0.0001f,
        "Bethesda quaternion axis remap is wrong");
    rotatedReference.rotation[0] = 0.0f;
    rotatedReference.rotation[1] = 0.0f;
    rotatedReference.rotation[2] = halfPi;
    const glm::vec3 yawed = glm::vec3(openf4::world::bethesdaReferenceTransform(rotatedReference)
        * glm::vec4(1.0f, 0.0f, 0.0f, 0.0f));
    require(glm::length(yawed - glm::vec3(0.0f, 0.0f, -1.0f)) < 0.0001f,
        "Bethesda Z rotation conversion is wrong");

    openf4::assets::PlacedReference translatedReference;
    translatedReference.position[0] = 10.0f;
    translatedReference.position[1] = 20.0f;
    translatedReference.position[2] = 30.0f;
    const glm::vec3 translated = glm::vec3(
        openf4::world::bethesdaReferenceTransform(translatedReference)[3]);
    require(glm::length(translated - glm::vec3(10.0f, 30.0f, 20.0f)) < 0.0001f,
        "Bethesda reference position mapping is wrong");

    openf4::assets::PlacedReference collectionPart;
    collectionPart.rotation[2] = halfPi;
    collectionPart.scale = 2.0f;
    collectionPart.localPosition[0] = 10.0f;
    collectionPart.localRotation[2] = halfPi;
    collectionPart.localScale = 0.5f;
    collectionPart.hasLocalTransform = true;
    const auto collectionTransform = openf4::world::bethesdaReferenceTransform(collectionPart);
    require(glm::length(glm::vec3(collectionTransform[3]) - glm::vec3(0.0f, 0.0f, 20.0f))
            < 0.0001f,
        "SCOL local translation was not rotated by its parent");
    require(std::abs(glm::length(glm::vec3(collectionTransform[0])) - 1.0f) < 0.0001f,
        "SCOL parent and local scales were not composed");
    require(glm::length(glm::vec3(collectionTransform * glm::vec4(1.0f, 0.0f, 0.0f, 0.0f))
            - glm::vec3(-1.0f, 0.0f, 0.0f)) < 0.0001f,
        "SCOL parent and local rotations were not composed");

    const auto boundedMesh = [](glm::vec4 bounds) {
        openf4::render::PackedMesh result;
        result.meshlets.resize(1);
        result.bounds = bounds;
        return result;
    };
    {
        const glm::vec4 original(10.0f, -20.0f, 30.0f, 100.0f);
        auto sceneBounds = original;
        for (int instance = 0; instance < 10000; ++instance)
            sceneBounds = openf4::render::unionBoundingSpheres(sceneBounds,
                {10.0f + instance % 20, -20.0f, 30.0f, 5.0f});
        require(sceneBounds == original,
            "Repeated contained instances inflated the scene/shadow bounds");
        require(openf4::render::unionBoundingSpheres(original, original) == original,
            "Coincident scene bounds grew during preview replacement");
    }
    openf4::render::PackedMesh combined;
    const auto firstRange = openf4::render::appendMesh(
        combined, boundedMesh({0.0f, 0.0f, 0.0f, 1.0f}));
    require(firstRange.bounds == glm::vec4(0.0f, 0.0f, 0.0f, 1.0f)
            && combined.bounds == firstRange.bounds,
        "first appended mesh bounds changed");
    const auto secondRange = openf4::render::appendMesh(
        combined, boundedMesh({4.0f, 0.0f, 0.0f, 1.0f}));
    require(secondRange.bounds == glm::vec4(4.0f, 0.0f, 0.0f, 1.0f),
        "appendMesh changed the source range bounds");
    require(glm::length(glm::vec3(combined.bounds) - glm::vec3(2.0f, 0.0f, 0.0f)) < 0.0001f
            && std::abs(combined.bounds.w - 3.0f) < 0.0001f,
        "disjoint mesh bounds were not unioned");
    (void)openf4::render::appendMesh(
        combined, boundedMesh({2.0f, 0.0f, 0.0f, 0.5f}));
    require(glm::length(glm::vec3(combined.bounds) - glm::vec3(2.0f, 0.0f, 0.0f)) < 0.0001f
            && std::abs(combined.bounds.w - 3.0f) < 0.0001f,
        "contained mesh bounds enlarged the union");
    (void)openf4::render::appendMesh(
        combined, boundedMesh({2.0f, 0.0f, 0.0f, 5.0f}));
    require(combined.bounds == glm::vec4(2.0f, 0.0f, 0.0f, 5.0f),
        "containing mesh bounds did not replace the union");
    (void)openf4::render::appendMesh(
        combined, boundedMesh({2.0f, 0.0f, 0.0f, 7.0f}));
    require(combined.bounds == glm::vec4(2.0f, 0.0f, 0.0f, 7.0f),
        "coincident mesh bounds were not unioned safely");
    (void)openf4::render::appendMesh(
        combined, boundedMesh({12.0f, 0.0f, 0.0f, 1.0f}));
    require(glm::length(glm::vec3(combined.bounds) - glm::vec3(4.0f, 0.0f, 0.0f)) < 0.0001f
            && std::abs(combined.bounds.w - 9.0f) < 0.0001f,
        "multi-append mesh bounds were not unioned");

    std::vector<openf4::render::SourceVertex> vertices;
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
    const auto mesh = openf4::render::buildMeshlets(vertices, indices, 7);
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

    const auto cube = openf4::render::makeCubeMesh();
    for (const auto& meshlet : cube.meshlets)
        for (std::uint32_t triangle = 0; triangle < meshlet.triangleCount * 3; triangle += 3)
        {
            const auto& first = cube.vertices[meshlet.vertexOffset + cube.indices[meshlet.indexOffset + triangle]];
            auto a = glm::vec3(first.position);
            auto b = glm::vec3(cube.vertices[meshlet.vertexOffset + cube.indices[meshlet.indexOffset + triangle + 1]].position);
            auto c = glm::vec3(cube.vertices[meshlet.vertexOffset + cube.indices[meshlet.indexOffset + triangle + 2]].position);
            auto normal = first.normal;
            require(glm::dot(a, normal) > 0, "Cube normal points inward");
            a.x = -a.x; b.x = -b.x; c.x = -c.x; normal.x = -normal.x;
            require(glm::dot(glm::cross(b - a, c - a), normal) > 0,
                "Cube winding disagrees with outward normals after renderer reflection");
        }
    openf4::render::PackedMesh topologyScene;
    const auto topologyRange = openf4::render::appendMesh(topologyScene, cube);
    require(topologyRange.vertexOffset == 0
            && topologyRange.vertexCount == cube.vertices.size()
            && topologyRange.indexOffset == 0
            && topologyRange.indexCount == cube.indices.size(),
        "MeshRange did not retain its vertex/index spans");
    require(openf4::render::detail::fixedTopologyReplacement(
            topologyRange, topologyScene.meshlets, cube),
        "valid fixed-topology mesh replacement was rejected");
    auto changedTopology = cube;
    changedTopology.vertices.pop_back();
    require(!openf4::render::detail::fixedTopologyReplacement(
            topologyRange, topologyScene.meshlets, changedTopology),
        "mesh replacement accepted a changed vertex span");
    changedTopology = cube;
    --changedTopology.meshlets.front().triangleCount;
    require(!openf4::render::detail::fixedTopologyReplacement(
            topologyRange, topologyScene.meshlets, changedTopology),
        "mesh replacement accepted a changed meshlet partition");

    auto landscape = openf4::assets::createEditableLandscape();
    require(openf4::world::detail::applyLandscapeBrush(landscape, {0.0f, 0.0f},
            openf4::world::LandscapeTool::raise, 512.0f, 256.0f),
        "raise brush made no change");
    require(landscape.height(32, 32) == 256.0f, "raise brush strength changed");
    (void)openf4::world::detail::applyLandscapeBrush(landscape, {0.0f, 0.0f},
        openf4::world::LandscapeTool::lower, 512.0f, 128.0f);
    require(landscape.height(32, 32) == 128.0f, "lower brush strength changed");
    (void)openf4::world::detail::applyLandscapeBrush(landscape, {0.0f, 0.0f},
        openf4::world::LandscapeTool::flatten, 512.0f, 1024.0f, 64.0f);
    require(landscape.height(32, 32) == 64.0f, "flatten brush target changed");

    landscape = openf4::assets::createEditableLandscape();
    landscape.height(32, 32) = 512.0f;
    (void)openf4::world::detail::applyLandscapeBrush(landscape, {0.0f, 0.0f},
        openf4::world::LandscapeTool::smooth, 256.0f, 1024.0f);
    require(landscape.height(32, 32) == 0.0f, "smooth brush ignored the eight-neighbor average");
    landscape.height(1, 0) = 4096.0f;
    openf4::world::detail::quantizeLandscape(landscape);
    require(landscape.height(1, 0) == 1016.0f,
        "landscape quantization did not clamp the signed VHGT delta");

    openf4::world::detail::LandscapeHistory history;
    auto beforeStroke = openf4::assets::createEditableLandscape();
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
