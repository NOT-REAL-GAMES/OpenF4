#include "assets/esm_document.hpp"
#include "assets/esm_editor.hpp"
#include "assets/lod_statics.hpp"
#include "assets/model_import.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    void require(bool okay, const char* why)
    {
        if (!okay) throw std::runtime_error(why);
    }

    template<class T>
    void append(std::vector<std::byte>& bytes, const T& value)
    {
        const auto* first = reinterpret_cast<const std::byte*>(&value);
        bytes.insert(bytes.end(), first, first + sizeof(value));
    }

    void write(const std::filesystem::path& path, const std::vector<std::byte>& bytes)
    {
        std::ofstream file(path, std::ios::binary);
        file.write(reinterpret_cast<const char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
        require(bool(file), "Could not write static LOD fixture");
    }

    void makeSource(const std::filesystem::path& directory)
    {
        std::filesystem::create_directories(directory);
        std::vector<std::byte> bytes;
        for (float value : {0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f, 0.f})
            append(bytes, value);
        for (int i = 0; i < 3; ++i)
            for (float value : {0.f, 0.f, 1.f}) append(bytes, value);
        for (float value : {0.f, 0.f, 1.f, 0.f, 0.f, 1.f})
            append(bytes, value);
        for (std::uint16_t value : {0, 1, 2}) append(bytes, value);
        write(directory / "mesh.bin", bytes);
        std::ofstream json(directory / "mesh.gltf");
        json << R"({"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0]}],
"nodes":[{"mesh":0}],"buffers":[{"uri":"mesh.bin","byteLength":102}],
"bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":36},
{"buffer":0,"byteOffset":36,"byteLength":36},
{"buffer":0,"byteOffset":72,"byteLength":24},
{"buffer":0,"byteOffset":96,"byteLength":6}],
"accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3","min":[0,0,0],"max":[1,1,0]},
{"bufferView":1,"componentType":5126,"count":3,"type":"VEC3"},
{"bufferView":2,"componentType":5126,"count":3,"type":"VEC2"},
{"bufferView":3,"componentType":5123,"count":3,"type":"SCALAR"}],
"materials":[{"pbrMetallicRoughness":{"baseColorFactor":[0.2,0.4,0.6,0.5],
"metallicFactor":0.25,"roughnessFactor":0.75},"alphaMode":"BLEND"}],
"meshes":[{"primitives":[{"attributes":{"POSITION":0,"NORMAL":1,"TEXCOORD_0":2},
"indices":3,"material":0}]}]})";
        require(bool(json), "Could not write glTF fixture");
    }

    void testBake(const std::filesystem::path& root)
    {
        const auto world = root / "world.esm";
        openf4::assets::saveEditableLandscape(world,
            openf4::assets::createEditableLandscape());
        const auto asset = root / "world.assets" / "models";
        makeSource(asset);
        openf4::assets::EsmDocument document(world);
        const auto modelId = document.allocateFormId();
        const auto firstId = document.allocateFormId();
        const auto secondId = document.allocateFormId();
        document.setModels({{modelId, "LodTest", "models/mesh.gltf"}});
        std::vector<openf4::assets::ESMReference> refs(2);
        refs[0].placement.formId = firstId;
        refs[0].placement.baseObject = modelId;
        refs[0].placement.cell = {-1, -1};
        refs[0].placement.position[0] = -4096.0f;
        refs[0].placement.position[1] = -4096.0f;
        refs[0].placement.scale = 1.0f;
        refs[1] = refs[0];
        refs[1].placement.formId = secondId;
        refs[1].placement.position[0] += 512.0f;
        document.setReferences(refs);
        document.save();

        openf4::assets::StaticLodSource source(world);
        const auto tiles = source.tiles();
        require(tiles.size() == 1
            && tiles.front().origin == openf4::assets::CellCoord{-4, -4},
            "Negative CELLs did not floor into the same level-4 tile");
        require(source.members(tiles.front()) == std::vector<std::uint32_t>{firstId, secondId},
            "Tile lost reference identities");
        const auto output = root / "world.assets" / "lod" / "tiles" / "tile.glb";
        source.bake(tiles.front(), output);
        require(std::filesystem::is_regular_file(output), "Static LOD GLB was not written");
        const auto imported = openf4::assets::importModel(output);
        require(imported.geometry.surfaces.size() == 2,
            "GLB lost a named reference part on Assimp reimport");
        const auto parts = openf4::assets::loadLodStaticParts(output);
        require(parts.size() == 2 && parts[0].referenceFormId == firstId
            && parts[1].referenceFormId == secondId,
            "GLB loader lost per-REFR identity needed for facade handoff");
        for (const auto& surface : imported.geometry.surfaces)
        {
            require(surface.indices.size() == 3, "Static simplifier lost a triangle");
            require(surface.alphaBlend && std::abs(surface.baseColor.a - 0.5f) < 0.01f,
                "Static LOD lost alpha material");
        }
        const auto x = imported.geometry.surfaces[0].vertices[0].position.x;
        require(std::abs(x - 12288.0f) < 0.1f,
            "Baked geometry failed tile-local coordinate round trip");
        const auto files = source.sourceFiles();
        require(std::find(files.begin(), files.end(), world) != files.end(),
            "Static source fingerprint omitted ESM");
    }
}

int main()
{
    const auto root = std::filesystem::temp_directory_path()
        / ("openf4-static-lod-" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(root);
    try
    {
        testBake(root);
        std::filesystem::remove_all(root);
    }
    catch (...)
    {
        std::filesystem::remove_all(root);
        throw;
    }
}
