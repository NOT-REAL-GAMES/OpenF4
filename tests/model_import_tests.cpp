#include "assets/asset_resolver.hpp"
#include "assets/model_import.hpp"

#include <zlib.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace
{
    void require(bool condition, const char* message)
    {
        if (!condition)
            throw std::runtime_error(message);
    }

    struct TemporaryDirectory
    {
        std::filesystem::path path = std::filesystem::temp_directory_path()
            / ("openf4-model-import-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        TemporaryDirectory() { std::filesystem::create_directory(path); }
        ~TemporaryDirectory() { std::error_code ignored; std::filesystem::remove_all(path, ignored); }
    };

    template<class T>
    void append(std::vector<std::byte>& bytes, const T& value)
    {
        const auto* begin = reinterpret_cast<const std::byte*>(&value);
        bytes.insert(bytes.end(), begin, begin + sizeof(value));
    }

    void write(const std::filesystem::path& path, std::span<const std::byte> bytes)
    {
        std::filesystem::create_directories(path.parent_path());
        std::ofstream stream(path, std::ios::binary);
        stream.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        require(static_cast<bool>(stream), "Fixture write failed");
    }

    void write(const std::filesystem::path& path, std::string_view value)
    {
        write(path, {reinterpret_cast<const std::byte*>(value.data()), value.size()});
    }

    std::vector<std::byte> png(Bytef alpha = 128)
    {
        const auto be = [](std::uint32_t value) {
            return ((value & 0xff) << 24) | ((value & 0xff00) << 8)
                | ((value & 0xff0000) >> 8) | ((value & 0xff000000) >> 24);
        };
        std::vector<std::byte> bytes;
        for (unsigned char value : {137, 80, 78, 71, 13, 10, 26, 10})
            bytes.push_back(std::byte(value));
        const auto chunk = [&](std::string_view tag, const std::vector<std::byte>& data) {
            append(bytes, be(static_cast<std::uint32_t>(data.size())));
            const auto offset = bytes.size();
            for (char character : tag)
                bytes.push_back(std::byte(character));
            bytes.insert(bytes.end(), data.begin(), data.end());
            append(bytes, be(crc32(0, reinterpret_cast<const Bytef*>(bytes.data() + offset),
                static_cast<uInt>(bytes.size() - offset))));
        };
        std::vector<std::byte> header;
        append(header, be(1));
        append(header, be(1));
        for (int value : {8, 6, 0, 0, 0})
            header.push_back(std::byte(value));
        chunk("IHDR", header);
        const std::array<Bytef, 5> rgba{0, 255, 64, 32, alpha};
        uLongf length = compressBound(static_cast<uLong>(rgba.size()));
        std::vector<std::byte> compressed(length);
        require(compress2(reinterpret_cast<Bytef*>(compressed.data()), &length,
            rgba.data(), static_cast<uLong>(rgba.size()), Z_BEST_SPEED) == Z_OK, "PNG compression failed");
        compressed.resize(length);
        chunk("IDAT", compressed);
        chunk("IEND", {});
        return bytes;
    }

    std::vector<std::byte> triangle()
    {
        std::vector<std::byte> bytes;
        for (float value : {0.f,0.f,0.f, 1.f,0.f,0.f, 0.f,1.f,0.f}) append(bytes, value);
        for (float value : {0.f,0.f,1.f, 0.f,0.f,1.f, 0.f,0.f,1.f}) append(bytes, value);
        for (float value : {0.f,0.f, 1.f,0.f, 0.f,1.f}) append(bytes, value);
        for (std::uint16_t value : {0, 1, 2}) append(bytes, value);
        while (bytes.size() % 4) bytes.push_back(std::byte{});
        return bytes;
    }

    std::string gltf(std::size_t bufferLength, bool embedded, std::size_t imageLength = 0,
        std::string_view imageUri = "../textures/color%20pixel.png")
    {
        std::ostringstream json;
        json << R"({"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0]}],
"nodes":[{"translation":[1,2,3],"children":[1]},{"scale":[2,3,4],"mesh":0}],
"buffers":[{"byteLength":)" << bufferLength;
        if (!embedded) json << R"(,"uri":"geometry.bin")";
        json << R"(}],"bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":36},
{"buffer":0,"byteOffset":36,"byteLength":36},{"buffer":0,"byteOffset":72,"byteLength":24},
{"buffer":0,"byteOffset":96,"byteLength":6})";
        if (embedded) json << R"(,{"buffer":0,"byteOffset":104,"byteLength":)" << imageLength << '}';
        json << R"(],"accessors":[
{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3","min":[0,0,0],"max":[1,1,0]},
{"bufferView":1,"componentType":5126,"count":3,"type":"VEC3"},
{"bufferView":2,"componentType":5126,"count":3,"type":"VEC2"},
{"bufferView":3,"componentType":5123,"count":3,"type":"SCALAR"}],"images":[{)";
        if (embedded) json << R"("bufferView":4,"mimeType":"image/png")";
        else json << "\"uri\":\"" << imageUri << '"';
        json << R"(}],"textures":[{"source":0}],"materials":[
{"pbrMetallicRoughness":{"baseColorFactor":[0.25,0.5,0.75,0.8],"baseColorTexture":{"index":0},
"metallicFactor":0.2,"roughnessFactor":0.6},"alphaMode":"MASK","alphaCutoff":0.3,"doubleSided":true},
{"pbrMetallicRoughness":{"baseColorFactor":[1,0,0,0.5]},"alphaMode":"BLEND"}],
"meshes":[{"primitives":[{"attributes":{"POSITION":0,"NORMAL":1,"TEXCOORD_0":2},"indices":3,"material":0},
{"attributes":{"POSITION":0,"NORMAL":1,"TEXCOORD_0":2},"indices":3,"material":1}]}]})";
        return json.str();
    }

    std::filesystem::path writeGltf(const std::filesystem::path& root)
    {
        const auto bytes = triangle();
        write(root / "models" / "geometry.bin", bytes);
        write(root / "textures" / "color pixel.png", png());
        const auto source = root / "models" / "triangle.gltf";
        write(source, gltf(bytes.size(), false));
        return source;
    }

    void writeGlbPayload(const std::filesystem::path& source, std::string json,
        std::vector<std::byte> buffer)
    {
        while (json.size() % 4) json += ' ';
        while (buffer.size() % 4) buffer.push_back(std::byte{});
        std::vector<std::byte> bytes;
        append(bytes, std::uint32_t{0x46546c67});
        append(bytes, std::uint32_t{2});
        append(bytes, static_cast<std::uint32_t>(28 + json.size() + buffer.size()));
        append(bytes, static_cast<std::uint32_t>(json.size()));
        append(bytes, std::uint32_t{0x4e4f534a});
        for (const char value : json) bytes.push_back(std::byte(value));
        append(bytes, static_cast<std::uint32_t>(buffer.size()));
        append(bytes, std::uint32_t{0x004e4942});
        bytes.insert(bytes.end(), buffer.begin(), buffer.end());
        write(source, bytes);
    }

    std::filesystem::path writeGlb(const std::filesystem::path& root)
    {
        auto buffer = triangle();
        const auto image = png();
        buffer.insert(buffer.end(), image.begin(), image.end());
        const auto source = root / "triangle.glb";
        auto json = gltf(buffer.size(), true, image.size());
        writeGlbPayload(source, std::move(json), std::move(buffer));
        return source;
    }

    std::filesystem::path writeCube(const std::filesystem::path& root)
    {
        // One meter tall, with its origin on the ground. Independent face
        // vertices exercise normals/UVs and produce a useful placement fixture.
        const std::array<glm::vec3, 6> normals{{
            {1,0,0}, {-1,0,0}, {0,1,0}, {0,-1,0}, {0,0,1}, {0,0,-1}}};
        const std::array<glm::vec3, 6> horizontal{{
            {0,0,-1}, {0,0,1}, {1,0,0}, {1,0,0}, {1,0,0}, {-1,0,0}}};
        const std::array<glm::vec2, 4> corners{{{-1,-1}, {1,-1}, {1,1}, {-1,1}}};
        std::vector<std::byte> buffer;
        for (std::size_t face = 0; face < normals.size(); ++face)
        {
            const auto vertical = glm::cross(normals[face], horizontal[face]);
            for (const auto& corner : corners)
            {
                const auto position = glm::vec3(0, 0.5f, 0) + normals[face] * 0.5f
                    + horizontal[face] * (corner.x * 0.5f) + vertical * (corner.y * 0.5f);
                for (int axis = 0; axis < 3; ++axis) append(buffer, position[axis]);
            }
        }
        const auto normalsOffset = buffer.size();
        for (const auto& normal : normals)
            for (int vertex = 0; vertex < 4; ++vertex)
                for (int axis = 0; axis < 3; ++axis) append(buffer, normal[axis]);
        const auto uvOffset = buffer.size();
        for (int face = 0; face < 6; ++face)
            for (const auto& corner : corners)
            {
                append(buffer, (corner.x + 1) * 0.5f);
                append(buffer, (corner.y + 1) * 0.5f);
            }
        const auto indexOffset = buffer.size();
        for (int face = 0; face < 6; ++face)
            for (int index : {0, 1, 2, 0, 2, 3})
                append(buffer, static_cast<std::uint16_t>(face * 4 + index));
        const auto imageOffset = buffer.size();
        const auto image = png(255);
        buffer.insert(buffer.end(), image.begin(), image.end());
        std::ostringstream json;
        json << R"({"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0]}],
"nodes":[{"mesh":0}],"buffers":[{"byteLength":)" << buffer.size() << R"(}],"bufferViews":[
{"buffer":0,"byteOffset":0,"byteLength":288},
{"buffer":0,"byteOffset":)" << normalsOffset << R"(,"byteLength":288},
{"buffer":0,"byteOffset":)" << uvOffset << R"(,"byteLength":192},
{"buffer":0,"byteOffset":)" << indexOffset << R"(,"byteLength":72},
{"buffer":0,"byteOffset":)" << imageOffset << R"(,"byteLength":)" << image.size() << R"(}],
"accessors":[{"bufferView":0,"componentType":5126,"count":24,"type":"VEC3","min":[-0.5,0,-0.5],"max":[0.5,1,0.5]},
{"bufferView":1,"componentType":5126,"count":24,"type":"VEC3"},
{"bufferView":2,"componentType":5126,"count":24,"type":"VEC2"},
{"bufferView":3,"componentType":5123,"count":36,"type":"SCALAR"}],
"images":[{"bufferView":4,"mimeType":"image/png"}],"textures":[{"source":0}],
"materials":[{"pbrMetallicRoughness":{"baseColorTexture":{"index":0},"metallicFactor":0,"roughnessFactor":0.8}}],
"meshes":[{"primitives":[{"attributes":{"POSITION":0,"NORMAL":1,"TEXCOORD_0":2},"indices":3,"material":0}]}]})";
        const auto source = root / "cube.glb";
        writeGlbPayload(source, json.str(), std::move(buffer));
        return source;
    }

    bool near(float first, float second) { return std::abs(first - second) < 0.001f; }

    void checkGeometry(const openf4::assets::ModelImportResult& model)
    {
        require(model.geometry.surfaces.size() == 2, "Multi-primitive glTF must retain both surfaces");
        const auto& surface = model.geometry.surfaces.front();
        require(surface.vertices.size() == 3 && surface.indices.size() == 3, "Triangle topology changed");
        const float unit = openf4::assets::gltfBethesdaUnitsPerMeter;
        const auto& first = surface.vertices[0];
        require(near(first.position.x, -unit) && near(first.position.y, 2 * unit)
            && near(first.position.z, 3 * unit), "Nested node translation / meter conversion incorrect");
        require(near(surface.vertices[1].position.x, -3 * unit)
            && near(surface.vertices[2].position.y, 5 * unit), "Nested nonuniform scale incorrect");
        require(near(first.normal.z, 1) && near(first.uv.y, 0), "Normal or glTF texture convention incorrect");
        auto a = surface.vertices[surface.indices[0]].position;
        auto b = surface.vertices[surface.indices[1]].position;
        auto c = surface.vertices[surface.indices[2]].position;
        a.x = -a.x; b.x = -b.x; c.x = -c.x;
        require(glm::dot(glm::cross(b - a, c - a), first.normal) > 0,
            "Imported winding must face the normal after renderer X reflection");
        require(surface.tangents.size() == 3, "Tangents were not retained");
        require(near(surface.baseColor.r, 0.25f) && near(surface.baseColor.a, 0.8f)
            && near(surface.alphaCutoff, 0.3f) && surface.doubleSided, "PBR color/alpha metadata lost");
        require(near(surface.metallicFactor, 0.2f) && near(surface.roughnessFactor, 0.6f), "PBR factors lost");
        require(surface.diffuseImage && surface.diffuseImage->width == 1
            && surface.diffuseImage->height == 1 && surface.diffuseImage->srgb,
            "PNG texture was not decoded");
        require(surface.diffuseImage->pixels[0] == std::byte{255}
            && surface.diffuseImage->pixels[1] == std::byte{64}
            && surface.diffuseImage->pixels[3] == std::byte{128}, "PNG channels changed");
        const auto& fallback = model.geometry.surfaces[1];
        require(fallback.alphaBlend && fallback.diffuseImage
            && fallback.diffuseImage->pixels == std::vector<std::byte>(4, std::byte{255}),
            "Untextured imported surfaces need an untinted white fallback");
    }

    void testImport(const std::filesystem::path& root)
    {
        const auto external = writeGltf(root / "external");
        const auto embedded = writeGlb(root / "embedded");
        checkGeometry(openf4::assets::importModel(external));
        checkGeometry(openf4::assets::importModelAsync(embedded).get());
        const auto world = root / "World.esm";
        const auto copied = openf4::assets::importAndCopyModelAsset(external, world, 0x01000820);
        require(copied.modelPath == "models/01000820/assets/models/triangle.gltf",
            "Sibling resource closure did not preserve relative hierarchy");
        const auto copiedGlb = openf4::assets::importAndCopyModelAsset(embedded, world, 0x01000821);
        require(copiedGlb.modelPath == "models/01000821/source.glb", "Simple imported model path is unstable");
        // Make originals inaccessible, proving copied sources are self-contained.
        std::filesystem::rename(root / "external", root / "external-moved");
        std::filesystem::rename(root / "embedded", root / "embedded-moved");
        checkGeometry(openf4::assets::importModel(copied.sourcePath));
        checkGeometry(openf4::assets::importModel(copiedGlb.sourcePath));
        bool rejected = false;
        try { (void)openf4::assets::importAndCopyModelAsset(copiedGlb.sourcePath, world, 0x01000821); }
        catch (const std::exception&) { rejected = true; }
        require(rejected, "Import must not overwrite an existing FormID's assets");
        write(root / "invalid.glb", "invalid glb");
        rejected = false;
        try { (void)openf4::assets::importAndCopyModelAsset(root / "invalid.glb", world, 0x01000822); }
        catch (const std::exception&) { rejected = true; }
        require(rejected && !std::filesystem::exists(copied.assetRoot / "models" / "01000822"),
            "Failed imports must not publish assets");

        const auto cube = openf4::assets::importModel(writeCube(root / "cube"));
        require(cube.geometry.surfaces.size() == 1 && cube.geometry.surfaces[0].indices.size() == 36,
            "Portable cube fixture lost faces");
        const auto& cubeSurface = cube.geometry.surfaces[0];
        for (std::size_t index = 0; index < cubeSurface.indices.size(); index += 3)
        {
            const auto& vertex = cubeSurface.vertices[cubeSurface.indices[index]];
            auto a = vertex.position;
            auto b = cubeSurface.vertices[cubeSurface.indices[index + 1]].position;
            auto c = cubeSurface.vertices[cubeSurface.indices[index + 2]].position;
            auto normal = vertex.normal;
            a.x = -a.x; b.x = -b.x; c.x = -c.x; normal.x = -normal.x;
            require(glm::dot(glm::cross(b - a, c - a), normal) > 0,
                "Cube faces must face their normals after renderer reflection");
        }

        const auto mirrored = writeGltf(root / "mirrored");
        auto json = gltf(triangle().size(), false);
        json.replace(json.find("[2,3,4]"), 7, "[-2,3,4]");
        write(mirrored, json);
        const auto negative = openf4::assets::importModel(mirrored);
        const auto& reflected = negative.geometry.surfaces.front();
        auto a = reflected.vertices[reflected.indices[0]].position;
        auto b = reflected.vertices[reflected.indices[1]].position;
        auto c = reflected.vertices[reflected.indices[2]].position;
        a.x = -a.x; b.x = -b.x; c.x = -c.x;
        require(glm::dot(glm::cross(b - a, c - a), reflected.vertices[0].normal) > 0,
            "Negative node scale inverted face winding");

        auto broken = triangle();
        const auto nonFinite = std::numeric_limits<float>::infinity();
        std::memcpy(broken.data() + 72, &nonFinite, sizeof(nonFinite));
        write(mirrored.parent_path() / "geometry.bin", broken);
        rejected = false;
        try { (void)openf4::assets::importModel(mirrored); }
        catch (const std::exception&) { rejected = true; }
        require(rejected, "Non-finite imported UV coordinates must be rejected");

        broken = triangle();
        const std::uint16_t badIndex = 4000;
        std::memcpy(broken.data() + 96, &badIndex, sizeof(badIndex));
        write(mirrored.parent_path() / "geometry.bin", broken);
        rejected = false;
        try { (void)openf4::assets::importModel(mirrored); }
        catch (const std::exception&) { rejected = true; }
        require(rejected, "Out-of-range triangle indices must be rejected");
    }

    void writeArchive(const std::filesystem::path& path)
    {
        const std::string name = "models\\sample.bin";
        std::vector<std::byte> bytes;
        append(bytes, std::uint32_t{0x58445442});
        append(bytes, std::uint32_t{1});
        append(bytes, std::uint32_t{0x4c524e47});
        append(bytes, std::uint32_t{1});
        append(bytes, std::uint64_t{61});
        for (int index = 0; index < 4; ++index) append(bytes, std::uint32_t{});
        append(bytes, std::uint64_t{60});
        append(bytes, std::uint32_t{});
        append(bytes, std::uint32_t{1});
        append(bytes, std::uint32_t{});
        bytes.push_back(std::byte{'a'});
        append(bytes, static_cast<std::uint16_t>(name.size()));
        for (char value : name) bytes.push_back(std::byte(value));
        write(path, bytes);
    }

    void testResolver(const std::filesystem::path& root)
    {
        const auto master = root / "World.esm";
        const auto data = root / "Data";
        const auto assets = openf4::assets::AssetResolver::projectAssetRoot(master);
        writeArchive(data / "Fallout4 - Meshes.ba2");
        write(data / "models" / "sample.bin", "d");
        write(assets / "models" / "sample.bin", "p");
        openf4::assets::AssetResolver resolver(master, data);
        require(resolver.contains("MODELS/SAMPLE.BIN"), "Asset paths must be case-insensitive");
        require(resolver.read("models/sample.bin").source == openf4::assets::AssetSource::project
            && resolver.extract("models/sample.bin")[0] == std::byte{'p'}, "Project asset precedence wrong");
        std::filesystem::remove(assets / "models" / "sample.bin");
        require(resolver.read("models/sample.bin").source == openf4::assets::AssetSource::looseData
            && resolver.extract("models/sample.bin")[0] == std::byte{'d'}, "Loose Data precedence wrong");
        std::filesystem::remove(data / "models" / "sample.bin");
        require(resolver.read("models/sample.bin").source == openf4::assets::AssetSource::archive
            && resolver.extract("models/sample.bin")[0] == std::byte{'a'}, "Archive fallback wrong");
        require(resolver.contains("./models/./sample.bin")
            && resolver.extract("./models/./sample.bin")[0] == std::byte{'a'},
            "Archive lookup must use the same normalized path as loose lookup");
        std::vector<std::future<void>> readers;
        for (int worker = 0; worker < 8; ++worker)
            readers.emplace_back(std::async(std::launch::async, [&] {
                openf4::assets::AssetResolver shared(master, data);
                for (int read = 0; read < 100; ++read)
                    require(shared.extract("models/sample.bin") == resolver.extract("models/sample.bin"),
                        "Concurrent shared archive reads changed payload");
            }));
        for (auto& reader : readers) reader.get();
        require(resolver.files("models\\", ".bin") == std::vector<std::string>{"models\\sample.bin"},
            "Discovery must deduplicate normalized asset paths");
        bool rejected = false;
        try { (void)resolver.read("../outside.bin"); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "Asset resolution must reject traversal");
        openf4::assets::AssetResolver projectOnly(master);
        require(!projectOnly.contains("models/missing.glb"), "Project-only missing path handling wrong");
    }
}

int main(int argc, char** argv)
{
    try
    {
        if (argc == 3 && std::string_view(argv[1]) == "--write-fixture")
        {
            const auto source = writeCube(std::filesystem::path(argv[2]));
            (void)openf4::assets::importModel(source);
            std::cout << source.string() << '\n';
            return 0;
        }
        TemporaryDirectory temp;
        testImport(temp.path / "import");
        testResolver(temp.path / "resolver");
        std::cout << "Model import and asset resolver tests passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
