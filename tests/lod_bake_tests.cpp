#include "assets/asset_resolver.hpp"
#include "assets/esm_document.hpp"
#include "assets/esm_editor.hpp"
#include "assets/lod_bake.hpp"
#include "assets/lod_statics.hpp"
#include "assets/lod_terrain.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace
{
    using namespace openf4::assets;

    void require(bool condition, const char* message)
    {
        if (!condition) throw std::runtime_error(message);
    }

    template<class T> void append(std::vector<std::byte>& bytes, T value)
    {
        const auto* first = reinterpret_cast<const std::byte*>(&value);
        bytes.insert(bytes.end(), first, first + sizeof(T));
    }

    void writeTriangle(const std::filesystem::path& path)
    {
        std::vector<std::byte> binary;
        for (const float value : {0.0f, 0.0f, 0.0f,
                1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f}) append(binary, value);
        for (const std::uint16_t value : {0, 1, 2}) append(binary, value);
        while (binary.size() % 4) binary.push_back(std::byte{});
        std::string json = R"({"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0]}],"nodes":[{"mesh":0}],"buffers":[{"byteLength":44}],"bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":36},{"buffer":0,"byteOffset":36,"byteLength":6}],"accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3","min":[0,0,0],"max":[1,1,0]},{"bufferView":1,"componentType":5123,"count":3,"type":"SCALAR"}],"meshes":[{"primitives":[{"attributes":{"POSITION":0},"indices":1}]}]})";
        while (json.size() % 4) json += ' ';
        std::vector<std::byte> bytes;
        append(bytes, std::uint32_t{0x46546c67});
        append(bytes, std::uint32_t{2});
        append(bytes, static_cast<std::uint32_t>(28 + json.size() + binary.size()));
        append(bytes, static_cast<std::uint32_t>(json.size()));
        append(bytes, std::uint32_t{0x4e4f534a});
        for (const char value : json) bytes.push_back(std::byte(value));
        append(bytes, static_cast<std::uint32_t>(binary.size()));
        append(bytes, std::uint32_t{0x004e4942});
        bytes.insert(bytes.end(), binary.begin(), binary.end());
        std::filesystem::create_directories(path.parent_path());
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        output.write(reinterpret_cast<const char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
        require(bool(output), "Could not create GLB test fixture");
    }

    void testIncrementalBake()
    {
        const auto folder = std::filesystem::temp_directory_path()
            / ("openf4-lod-index-" + std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(folder);
        const auto world = folder / "Test.esm";
        auto landscape = createEditableLandscape();
        for (int y = 4; y <= 8; ++y)
            for (int x = 4; x <= 8; ++x)
                (void)landscape.addCell({x, y});
        saveEditableLandscape(world, landscape);
        TerrainLodSource terrain(world);
        StaticLodSource statics(world);
        LodBuildIndex index(world);

        const auto first = index.preflight(terrain, statics);
        require(first.full && first.terrainTiles == terrain.tiles().size()
            && first.staticTiles == 0 && first.estimatedBytes != 0,
            "Missing build index did not require a full terrain bake");
        index.markSavedChanges(std::vector<CellCoord>{{0, 0}}, {});
        require(!std::filesystem::exists(LodBuildIndex::indexPath(world)),
            "A save created an index before the first bake");

        unsigned steps = 0;
        LodBakeOptions cancel;
        cancel.cancelled = [&] { return steps >= 2; };
        cancel.progress = [&](const LodBakeProgress&) { ++steps; };
        const auto partial = index.bake(terrain, statics, cancel);
        require(partial.cancelled && partial.terrainBaked == 2 && partial.revision == 2,
            "Cancellation did not preserve completed tiles");
        const auto resumed = index.bake(terrain, statics);
        require(!resumed.cancelled && resumed.terrainBaked == first.terrainTiles - 2,
            "Interrupted bake did not resume remaining tiles");
        require(index.preflight(terrain, statics).totalTiles() == 0,
            "Unchanged sources still require tile work");

        const std::vector<CellCoord> terrainChanges{{6, 6}};
        const std::vector<CellCoord> staticChanges{{-1, 0}, {4, 4}};
        index.markSavedChanges(terrainChanges, staticChanges);
        const auto changed = index.preflight(terrain, statics);
        require(!changed.full && changed.terrainTiles > 0
            && changed.terrainTiles < first.terrainTiles,
            "A tracked terrain edit did not produce a changed-only plan");
        require(changed.staticTiles == 2,
            "Old and new static cells did not invalidate both tile origins");
        const auto rebaked = index.bake(terrain, statics);
        require(rebaked.terrainBaked == changed.terrainTiles
            && rebaked.removedTiles == changed.staticTiles,
            "Changed-only bake did not clear every dirty tile");
        require(index.preflight(terrain, statics).totalTiles() == 0,
            "Changed-only bake left stale work");

        std::filesystem::last_write_time(world,
            std::filesystem::last_write_time(world) + std::chrono::seconds(2));
        const auto external = index.preflight(terrain, statics);
        require(external.full && external.terrainTiles == first.terrainTiles,
            "External ESM modification did not invalidate published terrain");
    }

    void testStaticMembership()
    {
        const auto folder = std::filesystem::temp_directory_path()
            / ("openf4-lod-members-" + std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(folder);
        const auto world = folder / "Test.esm";
        auto landscape = createEditableLandscape();
        (void)landscape.addCell({4, 4});
        saveEditableLandscape(world, landscape);
        const auto modelPath = AssetResolver::projectAssetRoot(world)
            / "models" / "triangle.glb";
        writeTriangle(modelPath);
        LodBuildIndex index(world);
        TerrainLodSource terrain(world);
        StaticLodSource empty(world);
        LodBakeOptions staticOnly;
        staticOnly.terrain = false;
        (void)index.bake(terrain, empty, staticOnly);

        EsmDocument document(world);
        const auto modelId = document.allocateFormId();
        const auto referenceId = document.allocateFormId();
        document.setModels({ESMModel{modelId, "TestTriangle",
            "models/triangle.glb", {}, world.string()}});
        PlacedReference placement;
        placement.formId = referenceId;
        placement.instanceId = referenceId;
        placement.baseObject = modelId;
        placement.cell = {0, 0};
        placement.position[0] = 512;
        placement.position[1] = 512;
        document.setReferences({ESMReference{placement, "TestPlacement"}});
        document.save();
        index.markSavedChanges({}, std::vector<CellCoord>{{0, 0}});
        StaticLodSource placed(world);
        require(!index.preflight(terrain, placed, staticOnly).full,
            "Editor-owned model addition triggered an unrelated full rebake");
        const auto first = index.bake(terrain, placed, staticOnly);
        require(first.staticBaked == 1 && !first.cancelled,
            "A saved reference was not included in its static tile");
        const auto oldTile = LodBuildIndex::staticTilePath(world, {{0, 0}});
        require(std::filesystem::is_regular_file(oldTile),
            "Published static tile is missing");

        std::filesystem::last_write_time(modelPath,
            std::filesystem::last_write_time(modelPath) + std::chrono::seconds(2));
        StaticLodSource changedModel(world);
        const auto affected = index.preflight(terrain, changedModel, staticOnly);
        require(!affected.full && affected.staticTiles == 1,
            "Known model-file edit did not invalidate just its member tile");
        (void)index.bake(terrain, changedModel, staticOnly);

        placement.cell = {4, 4};
        placement.position[0] = 4 * 4096 + 512;
        placement.position[1] = 4 * 4096 + 512;
        EsmDocument moved(world);
        moved.setReferences({ESMReference{placement, "TestPlacement"}});
        moved.save();
        index.markSavedChanges({}, std::vector<CellCoord>{{0, 0}, {4, 4}});
        StaticLodSource movedSource(world);
        const auto movePlan = index.preflight(terrain, movedSource, staticOnly);
        require(!movePlan.full && movePlan.staticTiles == 2,
            "Moving a reference did not invalidate old and new tiles");
        const auto move = index.bake(terrain, movedSource, staticOnly);
        const auto newTile = LodBuildIndex::staticTilePath(world, {{4, 4}});
        require(move.staticBaked == 1 && move.removedTiles == 1
            && !std::filesystem::exists(oldTile) && std::filesystem::exists(newTile)
            && std::find(move.publishedPaths.begin(), move.publishedPaths.end(), oldTile)
                != move.publishedPaths.end(),
            "Moving a reference left its old coarse tile behind");

        EsmDocument deleted(world);
        deleted.setReferences({});
        deleted.save();
        index.markSavedChanges({}, std::vector<CellCoord>{{4, 4}});
        StaticLodSource none(world);
        const auto removal = index.bake(terrain, none, staticOnly);
        require(removal.removedTiles == 1 && !std::filesystem::exists(newTile),
            "Deleting the final reference left a stale coarse tile");
    }
}

int main()
{
    try
    {
        testIncrementalBake();
        testStaticMembership();
        std::cout << "LOD build index tests passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "LOD build index tests failed: " << error.what() << '\n';
        return 1;
    }
}
