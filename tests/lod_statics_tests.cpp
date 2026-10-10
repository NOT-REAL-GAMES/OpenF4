#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "assets/esm_document.hpp"
#include "assets/esm_editor.hpp"
#include "assets/lod_statics.hpp"
#include "assets/model_import.hpp"

#include <DirectXTex.h>
#include <NifFile.hpp>
#include <stb_image_write.h>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
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
        const std::array<std::uint8_t, 16> pixels{255, 0, 0, 255, 0, 255, 0, 255,
            0, 0, 255, 255, 255, 255, 255, 255};
        require(stbi_write_png((directory / "diffuse.png").string().c_str(),
            2, 2, 4, pixels.data(), 8) != 0, "Could not write texture fixture");
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
"metallicFactor":0.25,"roughnessFactor":0.75,"baseColorTexture":{"index":0}},"alphaMode":"BLEND"}],
"textures":[{"source":0}],"images":[{"uri":"diffuse.png"}],
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
        const auto movableId = document.allocateFormId();
        const auto movableRefId = document.allocateFormId();
        document.setModels({{modelId, "LodTest", "models/mesh.gltf"},
            {movableId, "MovableNotLod", "models/mesh.gltf", {}, {}, openf4::assets::ESMModelKind::movableStatic}});
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
        refs[1].placement.rotation[2] = 0.4f;
        refs[1].placement.scale = 1.25f;
        auto movable = refs[0];
        movable.placement.formId = movableRefId; movable.placement.baseObject = movableId;
        movable.placement.cell = {1, 1}; movable.placement.position[0] = movable.placement.position[1] = 4096;
        refs.push_back(movable);
        document.setReferences(refs);
        document.save();

        openf4::assets::StaticLodSource source(world);
        const auto tiles = source.tiles();
        require(tiles.size() == 1
            && tiles.front().origin == openf4::assets::CellCoord{-4, -4},
            "Negative CELLs did not floor into one tile, or MSTT leaked into static LOD");
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
        const auto original = openf4::assets::importModel(asset / "mesh.gltf");
        const auto transform = glm::translate(glm::mat4(1), glm::vec3(12800, 0, 12288))
            * glm::rotate(glm::mat4(1), 0.4f, glm::vec3(0, 1, 0))
            * glm::scale(glm::mat4(1), glm::vec3(1.25f));
        for (std::size_t i = 0; i < original.geometry.surfaces[0].vertices.size(); ++i)
        {
            const auto& vertex = original.geometry.surfaces[0].vertices[i];
            const auto& baked = parts[1].geometry.surfaces[0].vertices[i];
            require(glm::length(baked.position - glm::vec3(transform * glm::vec4(vertex.position, 1))) < 0.05f,
                "Baked REFR position/rotation/scale did not round-trip");
        }
        for (const auto& surface : imported.geometry.surfaces)
        {
            require(surface.indices.size() == 3, "Static simplifier lost a triangle");
            require(surface.alphaBlend && std::abs(surface.baseColor.a - 0.5f) < 0.01f,
                "Static LOD lost alpha material");
            require(surface.diffuseImage && surface.diffuseImage->width == 2
                && surface.diffuseImage->pixels == original.geometry.surfaces[0].diffuseImage->pixels,
                "Baked static texture pixels changed");
            for (std::size_t i = 0; i < surface.vertices.size(); ++i)
                require(glm::length(surface.vertices[i].uv - original.geometry.surfaces[0].vertices[i].uv) < 0.001f,
                    "Baked static UVs flipped on reload");
            auto a = surface.vertices[surface.indices[0]].position;
            auto b = surface.vertices[surface.indices[1]].position;
            auto c = surface.vertices[surface.indices[2]].position;
            auto normal = surface.vertices[surface.indices[0]].normal;
            a.x = -a.x; b.x = -b.x; c.x = -c.x; normal.x = -normal.x;
            require(glm::dot(glm::cross(b - a, c - a), normal) > 0,
                "Baked static winding disagrees with its normals after renderer conversion");
        }
        require(std::distance(std::filesystem::directory_iterator(root / "world.assets/lod/textures"),
            std::filesystem::directory_iterator{}) == 1, "Shared diffuse texture was duplicated per REFR");
        const auto x = imported.geometry.surfaces[0].vertices[0].position.x;
        require(std::abs(x - 12288.0f) < 0.1f,
            "Baked geometry failed tile-local coordinate round trip");
        const auto files = source.sourceFiles();
        require(std::find(files.begin(), files.end(), world) != files.end(),
            "Static source fingerprint omitted ESM");

        const auto middle = openf4::assets::createEditorOverlay(world);
        openf4::assets::EsmDocument moved(middle);
        auto movedReference = *std::ranges::find_if(document.references(),
            [&](const auto& reference) { return reference.placement.formId == firstId; });
        movedReference.placement.position[0] += 256.0f;
        const std::array inheritedCell{*document.terrainCell({-1, -1})};
        moved.replaceTerrainCells(inheritedCell);
        moved.setReferences({movedReference});
        moved.save();
        const auto overlay = openf4::assets::createEditorOverlay(middle);
        const auto inherited = openf4::assets::StaticLodSource(overlay).sourceFiles();
        require(std::find(inherited.begin(), inherited.end(), middle) != inherited.end(),
            "REFR-only intermediate master omitted from source fingerprints");
    }

    void testSimplification(const std::filesystem::path& root)
    {
        const auto world = root / "grid.esm";
        openf4::assets::saveEditableLandscape(world, openf4::assets::createEditableLandscape());
        const auto asset = root / "grid.assets";
        std::filesystem::create_directories(asset);
        std::ofstream obj(asset / "grid.obj");
        for (int y = 0; y <= 16; ++y)
            for (int x = 0; x <= 16; ++x) obj << "v " << x << " 0 " << y << '\n';
        for (int y = 0; y < 16; ++y)
            for (int x = 0; x < 16; ++x)
            {
                const auto a = y * 17 + x + 1, b = a + 1, c = a + 17, d = c + 1;
                obj << "f " << a << ' ' << c << ' ' << b << '\n'
                    << "f " << b << ' ' << c << ' ' << d << '\n';
            }
        obj.close();
        openf4::assets::EsmDocument document(world);
        const auto model = document.allocateFormId();
        openf4::assets::ESMReference ref;
        ref.placement.formId = document.allocateFormId();
        ref.placement.baseObject = model;
        ref.placement.scale = 1;
        document.setModels({{model, "Grid", "grid.obj"}});
        document.setReferences({ref});
        document.save();
        const auto output = asset / "lod/tiles/grid.glb";
        const openf4::assets::StaticLodSource source(world);
        require(source.bake({{0, 0}}, output), "Grid did not bake");
        const auto mesh = openf4::assets::loadLodStaticParts(output).front().geometry.surfaces.front();
        require(mesh.indices.size() <= 512 * 3 / 4 && mesh.vertices.size() < 289,
            "Planar LOD did not reduce triangles and compact unused vertices");
        const auto files = openf4::assets::StaticLodSource(world).sourceFiles();
        require(std::find(files.begin(), files.end(), output) == files.end(),
            "Derived output became a source dependency for a sidecar-root model");
    }

    void testAnimatedSnapshot(const std::filesystem::path& root)
    {
        const auto world = root / "animated.esm";
        const auto asset = root / "animated.assets/models";
        openf4::assets::saveEditableLandscape(world, openf4::assets::createEditableLandscape());
        makeSource(asset);
        std::vector<std::byte> animation;
        for (float value : {0.f, 1.f, 0.f, 0.f, 0.f, 8.f, 0.f, 0.f}) append(animation, value);
        write(asset / "animation.bin", animation);
        std::ifstream input(asset / "mesh.gltf");
        std::string json((std::istreambuf_iterator<char>(input)), {});
        input.close();
        const auto replace = [&](std::string_view old, std::string_view value) {
            const auto where = json.find(old);
            require(where != std::string::npos, "Animation fixture patch did not match");
            json.replace(where, old.size(), value);
        };
        replace(R"("nodes":[{"mesh":0}])", R"("nodes":[{"mesh":0,"name":"AnimatedNode"}])");
        replace(R"("buffers":[{"uri":"mesh.bin","byteLength":102}])",
            R"("buffers":[{"uri":"mesh.bin","byteLength":102},{"uri":"animation.bin","byteLength":32}])");
        replace(R"({"buffer":0,"byteOffset":96,"byteLength":6}])",
            R"({"buffer":0,"byteOffset":96,"byteLength":6},{"buffer":1,"byteOffset":0,"byteLength":8},{"buffer":1,"byteOffset":8,"byteLength":24}])");
        replace(R"({"bufferView":3,"componentType":5123,"count":3,"type":"SCALAR"}])",
            R"({"bufferView":3,"componentType":5123,"count":3,"type":"SCALAR"},{"bufferView":4,"componentType":5126,"count":2,"type":"SCALAR","min":[0],"max":[1]},{"bufferView":5,"componentType":5126,"count":2,"type":"VEC3"}])");
        replace(R"("materials":)", R"("animations":[{"samplers":[{"input":4,"output":5,"interpolation":"LINEAR"}],"channels":[{"sampler":0,"target":{"node":0,"path":"translation"}}]}],"materials":)");
        std::ofstream sourceFile(asset / "mesh.gltf");
        sourceFile << json;
        sourceFile.close();
        const auto original = openf4::assets::importModel(asset / "mesh.gltf");
        require(!original.geometry.animations.empty(), "Animated fixture lost its source animation");
        openf4::assets::EsmDocument document(world);
        const auto model = document.allocateFormId();
        openf4::assets::ESMReference ref;
        ref.placement.formId = document.allocateFormId(); ref.placement.baseObject = model;
        ref.placement.scale = 1;
        document.setModels({{model, "Animated", "models/mesh.gltf"}});
        document.setReferences({ref}); document.save();
        const openf4::assets::StaticLodSource source(world);
        const auto output = root / "animated.assets/lod/tiles/animated.glb";
        require(source.bake({{0, 0}}, output), "Animated model was skipped instead of snapshot baked");
        const auto baked = openf4::assets::importModel(output);
        require(baked.geometry.animations.empty(), "Static tile retained animation channels");
        for (std::size_t i = 0; i < original.geometry.surfaces[0].vertices.size(); ++i)
            require(glm::length(baked.geometry.surfaces[0].vertices[i].position
                - original.geometry.surfaces[0].vertices[i].position) < 0.001f,
                "Animated static snapshot did not preserve the source bind pose");
        const auto messages = source.diagnostics();
        require(std::any_of(messages.begin(), messages.end(), [](const auto& message) {
            return message.find("snapshot") != std::string::npos;
        }), "Animated bake omitted its snapshot diagnostic");
        const auto files = source.sourceFilesForTile({{0, 0}});
        require(std::find(files.begin(), files.end(), asset / "animation.bin") != files.end(),
            "Imported external buffer missing from source fingerprints");
    }

    void testBethesdaMaterial(const std::filesystem::path& root)
    {
        const auto world = root / "nif.esm";
        openf4::assets::saveEditableLandscape(world, openf4::assets::createEditableLandscape());
        const auto data = root / "Data";
        for (const auto* directory : {"meshes", "materials", "textures"})
            std::filesystem::create_directories(data / directory);
        std::vector<nifly::Vector3> vertices{{0, 0, 0}, {128, 0, 0}, {0, 128, 0}};
        std::vector<nifly::Vector3> normals(3, {0, 0, 1});
        std::vector<nifly::Vector2> uvs{{0, 0}, {1, 0}, {0, 1}};
        std::vector<nifly::Triangle> triangles{{0, 1, 2}};
        nifly::NifFile nif;
        nif.Create(nifly::NiVersion::getFO4());
        auto* shape = nif.CreateShapeFromData("Material test", &vertices, &triangles, &uvs, &normals);
        require(shape != nullptr, "Could not create NIF fixture");
        auto* shader = nif.GetShader(shape);
        require(shader != nullptr, "NIF fixture lacks shader");
        shader->name.get() = "material.bgsm";
        shader->SetWetMaterialName("");
        require(nif.Save(data / "meshes/model.nif") == 0, "Could not save NIF fixture");
        std::vector<std::byte> bgsm;
        append(bgsm, std::uint32_t{0x4D534742});
        append(bgsm, std::uint32_t{2});
        append(bgsm, std::uint32_t{});
        for (int i = 0; i < 5; ++i) append(bgsm, 0.0f);
        append(bgsm, std::uint8_t{});
        append(bgsm, std::uint32_t{}); append(bgsm, std::uint32_t{});
        append(bgsm, std::uint8_t{128});
        for (int i = 0; i < 11; ++i) append(bgsm, std::uint8_t(i == 0 || i == 6));
        append(bgsm, 0.0f); append(bgsm, std::uint8_t{});
        append(bgsm, 0.0f); append(bgsm, std::uint8_t{});
        for (const auto* path : {"texture.dds", "", "", ""})
        {
            append(bgsm, static_cast<std::uint32_t>(std::strlen(path) + 1));
            for (std::size_t i = 0; i <= std::strlen(path); ++i) bgsm.push_back(std::byte(path[i]));
        }
        write(data / "materials/material.bgsm", bgsm);
        std::array<std::uint8_t, 64> pixels{};
        for (std::size_t i = 0; i < pixels.size(); i += 4)
        {
            pixels[i] = 32; pixels[i + 1] = 96; pixels[i + 2] = 192; pixels[i + 3] = 128;
        }
        const DirectX::Image image{4, 4, DXGI_FORMAT_R8G8B8A8_UNORM, 16, 64, pixels.data()};
        DirectX::ScratchImage compressed;
        require(SUCCEEDED(DirectX::Compress(image, DXGI_FORMAT_BC7_UNORM,
            DirectX::TEX_COMPRESS_DEFAULT, 0.5f, compressed)), "Could not compress BC7 fixture");
        DirectX::Blob blob;
        require(SUCCEEDED(DirectX::SaveToDDSMemory(*compressed.GetImage(0, 0, 0),
            DirectX::DDS_FLAGS_NONE, blob)), "Could not write BC7 fixture");
        const auto* first = reinterpret_cast<const std::byte*>(blob.GetBufferPointer());
        write(data / "textures/texture.dds", {first, first + blob.GetBufferSize()});

        openf4::assets::EsmDocument document(world);
        const auto model = document.allocateFormId();
        openf4::assets::ESMReference ref;
        ref.placement.formId = document.allocateFormId(); ref.placement.baseObject = model;
        ref.placement.scale = 1;
        document.setModels({{model, "NifTest", "model.nif"}});
        document.setReferences({ref}); document.save();
        const openf4::assets::StaticLodSource source(world, data);
        const auto files = source.sourceFilesForTile({{0, 0}});
        for (const auto* dependency : {"meshes/model.nif", "materials/material.bgsm", "textures/texture.dds"})
            require(std::find(files.begin(), files.end(), data / dependency) != files.end(),
                "Loose NIF/material/texture dependency missing from fingerprint");
        const auto output = root / "nif.assets/lod/tiles/material.glb";
        require(source.bake({{0, 0}}, output), "BGSM/BC7 tile was skipped");
        const auto surface = openf4::assets::loadLodStaticParts(output).front().geometry.surfaces.front();
        require(surface.diffuseImage && surface.doubleSided
            && std::abs(surface.alphaCutoff - 128.f / 255.f) < 0.001f,
            "BGSM texture/alpha/two-sided material failed round trip");
        require(std::abs(std::to_integer<int>(surface.diffuseImage->pixels[0]) - 32) <= 2
            && std::abs(std::to_integer<int>(surface.diffuseImage->pixels[3]) - 128) <= 2,
            "BC7 diffuse/alpha channels were not preserved");
        const auto overrideMaterial = root / "nif.assets/materials/material.bgsm";
        require(!std::filesystem::exists(overrideMaterial)
            && std::find(files.begin(), files.end(), overrideMaterial) != files.end(),
            "Absent project material override was not fingerprinted above loose Data");
        std::filesystem::create_directories(overrideMaterial.parent_path());
        bgsm[41] = std::byte{64};
        write(overrideMaterial, bgsm);
        const openf4::assets::StaticLodSource overridden(world, data);
        require(overridden.bake({{0, 0}}, output), "Project material override failed to bake");
        const auto replaced = openf4::assets::loadLodStaticParts(output).front().geometry.surfaces.front();
        require(std::abs(replaced.alphaCutoff - 64.f / 255.f) < 0.001f,
            "New project material did not override the earlier loose Data source");
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
        testSimplification(root);
        testAnimatedSnapshot(root);
        testBethesdaMaterial(root);
        std::filesystem::remove_all(root);
    }
    catch (const std::exception& error)
    {
        std::filesystem::remove_all(root);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
