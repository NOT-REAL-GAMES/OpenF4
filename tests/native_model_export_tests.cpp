#include "assets/native_model_export.hpp"
#include "assets/coordinates.hpp"
#include "assets/dds_texture.hpp"
#include "assets/material_file.hpp"

#include <NifFile.hpp>
#include <bhk.hpp>
#include <ExtraData.hpp>

#include <algorithm>
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

#include <glm/gtc/matrix_transform.hpp>

namespace
{
    using namespace openf4::assets;
    void require(bool value, const char* message)
    {
        if (!value) throw std::runtime_error(message);
    }

    template<class F> void rejects(F&& action, const char* message)
    {
        try { action(); }
        catch (const std::exception&) { return; }
        throw std::runtime_error(message);
    }

    std::vector<std::byte> read(const std::filesystem::path& path)
    {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        require(bool(file), "Cannot read fixture");
        const auto size = file.tellg();
        std::vector<std::byte> bytes(static_cast<std::size_t>(size));
        file.seekg(0);
        require(bool(file.read(reinterpret_cast<char*>(bytes.data()), size)), "Cannot read fixture data");
        return bytes;
    }

    constexpr std::array<std::uint32_t, 36> boxTriangles{
        0,2,1,0,3,2, 4,5,6,4,6,7, 0,1,5,0,5,4,
        1,2,6,1,6,5, 2,3,7,2,7,6, 3,0,4,3,4,7};

    std::vector<glm::vec3> boxVertices(glm::vec3 minimum, glm::vec3 maximum)
    {
        return {{minimum.x,minimum.y,minimum.z},{maximum.x,minimum.y,minimum.z},
            {maximum.x,maximum.y,minimum.z},{minimum.x,maximum.y,minimum.z},
            {minimum.x,minimum.y,maximum.z},{maximum.x,minimum.y,maximum.z},
            {maximum.x,maximum.y,maximum.z},{minimum.x,maximum.y,maximum.z}};
    }

    void addBox(ModelSurface& surface, glm::vec3 minimum, glm::vec3 maximum)
    {
        const auto vertices = boxVertices(minimum, maximum);
        for (std::size_t i = 0; i < boxTriangles.size(); i += 3)
        {
            const auto a = vertices[boxTriangles[i]], b = vertices[boxTriangles[i+1]], c = vertices[boxTriangles[i+2]];
            const auto normal = swapBethesdaAxes(glm::normalize(glm::cross(b-a,c-a)));
            for (int j = 0; j < 3; ++j)
            {
                openf4::render::SourceVertex v;
                v.position = havokToEnginePosition(vertices[boxTriangles[i+j]]);
                v.normal = normal; v.uv = {j == 1 ? 1 : 0, j == 2 ? 1 : 0};
                surface.indices.push_back(static_cast<std::uint32_t>(surface.vertices.size()));
                surface.vertices.push_back(v);
            }
        }
    }

    ModelGeometry cube()
    {
        ModelGeometry geometry;
        auto& surface = geometry.surfaces.emplace_back();
        addBox(surface, {-0.5f,-0.5f,0}, {0.5f,0.5f,1});
        surface.metallicFactor = 0; surface.roughnessFactor = 1;
        surface.baseColor = {0.2f, 0.4f, 0.8f, 0.75f};
        surface.alphaCutoff = 0.3f; surface.doubleSided = true;
        return geometry;
    }

    HavokPhysicsSystem cubeCollision(bool dynamic = false)
    {
        HavokPhysicsSystem system;
        auto& body = system.bodies.emplace_back();
        body.dynamic = dynamic; body.mass = dynamic ? 1 : 0;
        body.centerOfMass = {0,0,0.5f}; body.inertiaDiagonal = glm::vec3(1.0f / 6.0f);
        body.collisionFilterInfo = dynamic ? 5 : 1;
        auto& hull = body.parts.emplace_back().polytope;
        hull.vertices = boxVertices({-0.5f,-0.5f,0}, {0.5f,0.5f,1});
        hull.indices.assign(boxTriangles.begin(), boxTriangles.end());
        hull.convexRadius = 0.005f;
        return system;
    }

    void testPackage(const std::filesystem::path& root)
    {
        const auto geometry = cube();
        const auto collision = cubeCollision();
        const auto package = exportNativeModelAsset(root / "native.esm", 0x01000800, geometry, &collision);
        const auto nifPath = package.assetRoot / package.modelPath;
        const auto bytes = read(nifPath);
        const auto reloaded = loadNifModel(bytes);
        require(reloaded.surfaces.size() == 1, "Native surface count changed");
        require(reloaded.surfaces[0].doubleSided && !reloaded.surfaces[0].alphaBlend
            && std::abs(reloaded.surfaces[0].alphaCutoff-0.3f) < 0.004f, "NIF alpha/two-sided flags lost");
        require(reloaded.surfaces[0].indices == geometry.surfaces[0].indices, "NIF winding changed");
        for (std::size_t i = 0; i < geometry.surfaces[0].vertices.size(); ++i)
        {
            const auto& before = geometry.surfaces[0].vertices[i];
            const auto& after = reloaded.surfaces[0].vertices[i];
            require(glm::length(before.position-after.position) < 0.001f, "NIF position/units round trip failed");
            require(glm::length(before.normal-after.normal) < 0.02f, "NIF normal round trip failed");
            require(glm::length(before.uv-after.uv) < 0.002f, "NIF UV round trip failed");
        }
        const auto systems = loadNifPhysicsSystems(bytes);
        require(systems.size() == 1 && systems[0].bodies.size() == 1, "NIF physics attachment lost");
        nifly::NifFile nif(nifPath);
        // Inspect native coordinates directly so a matching importer bug cannot hide inverted faces.
        for (auto* shape : nif.GetShapes())
        {
            const auto* positions = nif.GetVertsForShape(shape);
            const auto* normals = nif.GetNormalsForShape(shape);
            std::vector<nifly::Triangle> triangles;
            require(positions && normals && shape->GetTriangles(triangles), "Native winding fixture lost geometry");
            const auto position = [&](std::uint16_t index) {
                const auto& v = positions->at(index); return glm::vec3(v.x, v.y, v.z);
            };
            for (const auto& triangle : triangles)
            {
                const auto a = position(triangle.p1), b = position(triangle.p2), c = position(triangle.p3);
                const auto& normal = normals->at(triangle.p1);
                require(glm::dot(glm::cross(b - a, c - a), glm::vec3(normal.x, normal.y, normal.z)) > 0,
                    "Exported native face winding disagrees with its outward normal");
            }
        }
        auto& header = nif.GetHeader();
        const auto* object = header.GetBlock<nifly::bhkNPCollisionObject>(nif.GetRootNode()->collisionRef);
        require(object && object->flags == 0x0080 && object->bodyID == 0
            && object->targetRef.index == nif.GetBlockID(nif.GetRootNode())
            && header.GetBlock<nifly::bhkPhysicsSystem>(object->bodyRef), "Invalid FO4 collision node/dataID/bodyID attachment");
        auto nativeMaterialPath = reloaded.surfaces[0].materialPath;
        std::replace(nativeMaterialPath.begin(), nativeMaterialPath.end(), '\\', '/');
        require(nativeMaterialPath.starts_with("materials/"), "NIF material path lacks the Fallout Materials namespace");
        const auto materialPath = package.assetRoot / nativeMaterialPath;
        const auto materialBytes = read(materialPath);
        std::size_t textureString = 63; // BGSM v2 texture table follows the fixed material header.
        for (int slot = 0; slot < 9; ++slot)
        {
            require(textureString + 4 <= materialBytes.size(), "Truncated BGSM texture string length");
            std::uint32_t length; std::memcpy(&length, materialBytes.data() + textureString, 4);
            textureString += 4;
            require(length > 0 && length <= materialBytes.size() - textureString
                && materialBytes[textureString + length - 1] == std::byte{},
                "Fallout BGSM texture strings must include a terminating NUL");
            textureString += length;
        }
        const auto mat = loadMaterial(materialBytes);
        require(mat.twoSided && std::abs(mat.alphaCutoff-0.3f) < 0.004f, "Native alpha/two-sided settings lost");
        std::string diffuse = mat.diffuseTexture, normal = mat.normalTexture;
        std::replace(diffuse.begin(), diffuse.end(), '\\', '/');
        std::replace(normal.begin(), normal.end(), '\\', '/');
        require(!diffuse.starts_with("textures/") && !normal.starts_with("textures/"),
            "BGSM texture paths must be relative to the Fallout Textures directory");
        const auto dds = loadDdsTexture(read(package.assetRoot / "textures" / diffuse));
        require(dds.width == 1 && dds.height == 1 && dds.pixels.size() == 4, "Native DDS invalid");
        require(std::to_integer<unsigned>(dds.pixels[0]) > 110
            && std::to_integer<unsigned>(dds.pixels[0]) < 130, "Base color not applied in linear space");
        require(std::to_integer<unsigned>(dds.pixels[3]) == 191, "Base alpha was not preserved");
        require(loadDdsTexture(read(package.assetRoot / "textures" / normal)).pixels[2] == std::byte{255}, "Default normal is invalid");
        const auto second = exportNativeModelAsset(root / "native.esm", 0x01000800, geometry);
        require(second.modelPath != package.modelPath && read(nifPath) == bytes, "Native versioning overwrote usable asset");
        require(!std::filesystem::exists(root / "native.esm"), "Export unexpectedly wrote world document");

        auto modified = collision; modified.bodies[0].friction = 0.25f;
        const auto replacement = replaceNativeNifCollision(bytes, object->bodyRef.index, modified);
        require(std::abs(loadNifPhysicsSystems(replacement)[0].bodies[0].friction - 0.25f) < 0.002f,
            "Collision replacement was not applied");
        const auto reverse = replaceNativeNifCollision(replacement, object->bodyRef.index, collision);
        require(reverse == bytes, "Collision-only edit rewrote untouched NIF bytes");
        auto unknown = bytes;
        const std::string typeName = "BSXFlags";
        auto marker = std::search(unknown.begin(), unknown.end(),
            reinterpret_cast<const std::byte*>(typeName.data()),
            reinterpret_cast<const std::byte*>(typeName.data() + typeName.size()));
        require(marker != unknown.end(), "Unknown block fixture target not found");
        std::memcpy(&*marker, "OpaqueXX", typeName.size());
        unknown.push_back(std::byte{0xee});
        const auto modifiedUnknown = replaceNativeNifCollision(unknown, object->bodyRef.index, modified);
        require(replaceNativeNifCollision(modifiedUnknown, object->bodyRef.index, collision) == unknown,
            "Unknown blocks or trailing bytes changed during collision-only update");
        auto taggedSystem = collision; taggedSystem.bodies[0].collisionFilterInfo = 0x43210001;
        auto unsupported = replaceNativeNifCollision(bytes, object->bodyRef.index, taggedSystem);
        const std::array<std::byte, 4> filterMarker{std::byte{1}, std::byte{0}, std::byte{0x21}, std::byte{0x43}};
        auto bodyFilter = std::search(unsupported.begin(), unsupported.end(), filterMarker.begin(), filterMarker.end());
        require(bodyFilter != unsupported.end() && std::distance(bodyFilter, unsupported.end()) >= 8,
            "Body flags fixture target not found");
        auto noncanonical = unsupported;
        require(std::distance(unsupported.begin(), bodyFilter) >= 4, "Body quality fixture target not found");
        noncanonical[static_cast<std::size_t>(std::distance(unsupported.begin(), bodyFilter)) - 4] = std::byte{2};
        require(loadNifPhysicsSystems(noncanonical)[0].nativeRewriteBlockReason.empty(),
            "Body quality fixture should remain readable without a represented-field rejection");
        rejects([&] { (void)replaceNativeNifCollision(noncanonical, object->bodyRef.index, collision); },
            "Canonical-source guard discarded an unrepresented Havok body quality setting");
        bodyFilter[4] = std::byte{1}; // An unsupported source-body flag must not disappear during rewrite.
        require(!loadNifPhysicsSystems(unsupported)[0].nativeRewriteBlockReason.empty(), "Unsupported source metadata was not detected");
        rejects([&] { (void)replaceNativeNifCollision(unsupported, object->bodyRef.index, collision); },
            "Fresh authored replacement bypassed unsupported source preservation");
        auto truncated = bytes; truncated.resize(bytes.size() / 2);
        rejects([&] { (void)replaceNativeNifCollision(truncated, object->bodyRef.index, collision); }, "Truncated NIF accepted");
        rejects([&] { (void)replaceNativeNifCollision(bytes, 0, collision); }, "Wrong collision block accepted");
        auto changedBodies = collision; changedBodies.bodies.push_back(collision.bodies[0]);
        rejects([&] { (void)replaceNativeNifCollision(bytes, object->bodyRef.index, changedBodies); }, "Body attachment change accepted");
    }

    void testAuthoredCollisionReload(const std::filesystem::path& root)
    {
        const auto check = [&](const char* name, const HavokPhysicsSystem& collision) {
            const auto package = exportNativeModelAsset(root / name, 0x01000900, cube(), &collision);
            const auto path = package.assetRoot / package.modelPath;
            const auto original = read(path);
            nifly::NifFile nif(path);
            const auto* attachment = nif.GetHeader().GetBlock<nifly::bhkNPCollisionObject>(nif.GetRootNode()->collisionRef);
            require(attachment, "Authored collision reload lost its NIF attachment");
            auto edited = loadNifPhysicsSystems(original).front();
            edited.bodies.front().friction = 0.25f;
            try
            {
                const auto saved = replaceNativeNifCollision(original, attachment->bodyRef.index, edited);
                auto reopened = loadNifPhysicsSystems(saved).front();
                reopened.bodies.front().friction = 0.75f;
                const auto savedAgain = replaceNativeNifCollision(saved, attachment->bodyRef.index, reopened);
                require(loadNifPhysicsSystems(savedAgain).front().bodies.front().friction == 0.75f,
                    "A second authored collision save did not preserve the edit");
            }
            catch (const std::exception& error)
            {
                throw std::runtime_error(std::string(name) + ": " + error.what());
            }
        };

        auto mesh = cubeCollision();
        mesh.bodies.front().parts.clear();
        auto& triangles = mesh.bodies.front().triangleMesh.emplace();
        for (unsigned i = 0; i < 160; ++i)
        {
            const auto index = static_cast<unsigned>(triangles.vertices.size());
            const float x = float(i) * 0.37f;
            triangles.vertices.insert(triangles.vertices.end(), {{x,-0.122f,0.13f},
                {x+0.271f,0.117f,0.09f}, {x+0.022f,0.912f,0.12f}});
            triangles.indices.insert(triangles.indices.end(), {index,index+1,index+2});
        }
        check("mesh-reload.esm", mesh);

        auto rotated = cubeCollision(true);
        rotated.bodies.front().position = {0.121f,-0.23f,0.734f};
        rotated.bodies.front().rotation = glm::angleAxis(0.87f, glm::normalize(glm::vec3(1,2,3)));
        rotated.bodies.front().inertiaOrientation = glm::angleAxis(0.23f, glm::normalize(glm::vec3(3,1,2)));
        rotated.bodies.front().centerOfMass = {0.123f,-0.112f,0.51f};
        check("rotated-reload.esm", rotated);

        auto compound = cubeCollision();
        compound.bodies.front().parts.push_back(compound.bodies.front().parts.front());
        compound.bodies.front().parts.back().transform = glm::translate(glm::mat4(1), glm::vec3(0.234f,0.141f,0.53f))
            * glm::mat4_cast(glm::angleAxis(0.53f, glm::normalize(glm::vec3(3,2,1))));
        check("compound-reload.esm", compound);
    }

    void testRejections(const std::filesystem::path& root)
    {
        auto geometry = cube();
        geometry.skinBones.push_back({"bone", glm::mat4(1)});
        rejects([&] { (void)exportNativeModelAsset(root / "bad.esm", 1, geometry); }, "Skinned conversion was not blocked");
        geometry.skinBones.clear(); geometry.animations.emplace_back();
        rejects([&] { (void)exportNativeModelAsset(root / "bad.esm", 1, geometry); }, "Animated conversion was not blocked");
        geometry.animations.clear(); geometry.surfaces[0].indices[0] = 99999;
        rejects([&] { (void)exportNativeModelAsset(root / "bad.esm", 1, geometry); }, "Bad indices accepted");
        geometry = cube(); geometry.surfaces[0].vertices[0].position.x = std::numeric_limits<float>::quiet_NaN();
        rejects([&] { (void)exportNativeModelAsset(root / "bad.esm", 1, geometry); }, "NaN accepted");
        require(!std::filesystem::exists(root / "bad.assets"), "Invalid asset wrote files");
        geometry = cube();
        ModelImage invalid; invalid.width = invalid.height = 4;
        geometry.surfaces[0].diffuseImage = invalid;
        rejects([&] { (void)exportNativeModelAsset(root / "incomplete.esm", 1, geometry); }, "Truncated pixels accepted");
        const auto path = root / "incomplete.assets";
        require(std::filesystem::is_empty(path), "Failed native export left staged/published data");
        const auto blocked = root / "blocked.assets";
        std::filesystem::create_directory(blocked);
        {
            std::ofstream marker(blocked / "meshes", std::ios::binary);
            marker << "existing user file";
        }
        const auto preserved = read(blocked / "meshes");
        rejects([&] { (void)exportNativeModelAsset(root / "blocked.esm", 1, cube()); }, "Publication failure was not reported");
        require(read(blocked / "meshes") == preserved, "Failed publication damaged existing user data");
        for (const auto& entry : std::filesystem::recursive_directory_iterator(blocked))
            require(entry.is_directory() || entry.path() == blocked / "meshes",
                "Failed publication retained a partial native package");
    }

    void testTexturesAndChunks(const std::filesystem::path& root)
    {
        auto geometry = cube();
        auto& surface = geometry.surfaces[0];
        surface.vertices.clear(); surface.indices.clear();
        for (unsigned i = 0; i < 22000; ++i)
            for (unsigned c = 0; c < 3; ++c)
            {
                openf4::render::SourceVertex vertex;
                vertex.position = {static_cast<float>(i % 100) + (c == 1 ? 0.5f : 0),
                    0, static_cast<float>(i / 100) + (c == 2 ? 0.5f : 0)};
                vertex.normal = {0, 1, 0};
                surface.indices.push_back(static_cast<std::uint32_t>(surface.vertices.size()));
                surface.vertices.push_back(vertex);
            }
        ModelImage image; image.width = image.height = 4; image.srgb = true;
        image.pixels.resize(4 * 4 * 4, std::byte{255});
        surface.diffuseImage = image; surface.baseColor = glm::vec4(1); surface.alphaCutoff = 0;
        surface.alphaBlend = true;
        surface.normalImage = image; surface.normalImage->srgb = false;
        for (std::size_t i = 0; i < image.pixels.size(); i += 4)
        {
            surface.normalImage->pixels[i] = std::byte{128};
            surface.normalImage->pixels[i+1] = std::byte{64};
        }
        const auto package = exportNativeModelAsset(root / "chunks.esm", 1, geometry);
        const auto reloaded = loadNifModel(read(package.assetRoot / package.modelPath));
        require(reloaded.surfaces.size() == 2, "NIF 16-bit index chunks were not split");
        std::size_t triangles = 0;
        for (const auto& part : reloaded.surfaces)
        {
            triangles += part.indices.size() / 3;
            require(part.vertices.size() <= 65535, "NIF vertex index overflow");
            require(part.alphaBlend && part.doubleSided && part.alphaCutoff == 0, "NIF blend flags lost");
        }
        require(triangles == 22000, "NIF split lost triangles");
        auto materialName = reloaded.surfaces[0].materialPath;
        std::replace(materialName.begin(), materialName.end(), '\\', '/');
        auto mat = loadMaterial(read(package.assetRoot / materialName));
        std::replace(mat.diffuseTexture.begin(), mat.diffuseTexture.end(), '\\', '/');
        std::replace(mat.normalTexture.begin(), mat.normalTexture.end(), '\\', '/');
        const auto dds = loadDdsTexture(read(package.assetRoot / "textures" / mat.diffuseTexture));
        require(dds.mipLevels == 3 && dds.width == 4, "Native DDS mip chain missing");
        const auto normal = loadDdsTexture(read(package.assetRoot / "textures" / mat.normalTexture));
        require(normal.pixels[1] == std::byte{64}, "Authored normal map was changed");
    }

    void proofPlugin(const std::filesystem::path& path, const NativeModelPackage& cubePackage,
        const NativeModelPackage& doorwayPackage, const NativeModelPackage& compoundPackage)
    {
        using Bytes = std::vector<std::byte>;
        const auto number = []<class T>(Bytes& bytes, T value) {
            const auto* begin = reinterpret_cast<const std::byte*>(&value);
            bytes.insert(bytes.end(), begin, begin + sizeof(T));
        };
        const auto tag = [](Bytes& bytes, std::string_view name) {
            require(name.size() == 4, "Invalid fixture record tag");
            for (const char c : name) bytes.push_back(std::byte(c));
        };
        const auto sub = [&](Bytes& bytes, std::string_view name, const Bytes& value) {
            require(value.size() <= 65535, "Fixture subrecord exceeds size limit");
            tag(bytes, name); number(bytes, static_cast<std::uint16_t>(value.size()));
            bytes.insert(bytes.end(), value.begin(), value.end());
        };
        const auto text = [&](Bytes& bytes, std::string_view name, std::string value) {
            Bytes payload;
            for (char c : value) payload.push_back(std::byte(c));
            payload.push_back(std::byte{}); sub(bytes, name, payload);
        };
        const auto record = [&](std::string_view name, std::uint32_t id, const Bytes& body) {
            Bytes result; tag(result, name); number(result, static_cast<std::uint32_t>(body.size()));
            number(result, std::uint32_t{0}); number(result, id); number(result, std::uint32_t{0});
            number(result, std::uint16_t{131}); number(result, std::uint16_t{0});
            result.insert(result.end(), body.begin(), body.end()); return result;
        };
        const auto group = [&](std::string_view name, const Bytes& children) {
            Bytes result; tag(result, "GRUP"); number(result, static_cast<std::uint32_t>(children.size() + 24));
            tag(result, name); number(result, std::uint32_t{0}); number(result, std::uint64_t{0});
            result.insert(result.end(), children.begin(), children.end()); return result;
        };
        // xEdit wbDefinitionsFO4: STAT requires OBND/DNAM; MSTT requires OBND/DATA.
        const auto model = [&](std::string_view type, std::uint32_t id, const std::string& editorId,
            const NativeModelPackage& package, std::array<std::int16_t, 6> bounds) {
            Bytes body; text(body, "EDID", editorId);
            Bytes boundBytes; for (auto value : bounds) number(boundBytes, value);
            sub(body, "OBND", boundBytes);
            if (type == "MSTT") text(body, "FULL", editorId);
            require(package.modelPath.starts_with("meshes/"), "Fixture NIF is outside the Meshes namespace");
            auto modelPath = package.modelPath.substr(7);
            std::replace(modelPath.begin(), modelPath.end(), '/', '\\');
            text(body, "MODL", modelPath);
            if (type == "STAT")
            {
                Bytes direction;
                number(direction, 90.0f); number(direction, std::uint32_t{0});
                number(direction, 0.0f); number(direction, 0.0f);
                sub(body, "DNAM", direction);
            }
            else sub(body, "DATA", Bytes{std::byte{1}});
            return record(type, id, body);
        };
        Bytes header, hedr;
        number(hedr, 1.0f); number(hedr, std::uint32_t{5}); number(hedr, std::uint32_t{0x803});
        sub(header, "HEDR", hedr); text(header, "CNAM", "OpenF4 Native Collision Proof");
        text(header, "SNAM", "Disposable native asset validation fixtures; no world overrides.");
        text(header, "MAST", "Fallout4.esm"); sub(header, "DATA", Bytes(8));
        auto output = record("TES4", 0, header);
        const auto statics = group("STAT", model("STAT", 0x01000801, "OF4NativeProofDoorway",
            doorwayPackage, {-106,-19,-1,106,19,211}));
        auto movable = model("MSTT", 0x01000800, "OF4NativeProofCube", cubePackage, {-36,-36,-1,36,36,71});
        const auto compound = model("MSTT", 0x01000802, "OF4NativeProofCompound", compoundPackage,
            {-71,-36,-1,71,36,71});
        movable.insert(movable.end(), compound.begin(), compound.end());
        const auto mstt = group("MSTT", movable);
        output.insert(output.end(), statics.begin(), statics.end());
        output.insert(output.end(), mstt.begin(), mstt.end());
        std::ofstream file(path, std::ios::binary);
        require(bool(file.write(reinterpret_cast<const char*>(output.data()), output.size()).flush()),
            "Cannot write disposable Fallout 4 proof plugin");
    }

    void testProofPlugin(const std::filesystem::path& path)
    {
        const auto bytes = read(path);
        const auto tag = [&](std::size_t offset) {
            require(offset + 4 <= bytes.size(), "Truncated fixture tag");
            return std::string(reinterpret_cast<const char*>(bytes.data() + offset), 4);
        };
        const auto integer = [&](std::size_t offset) {
            require(offset + 4 <= bytes.size(), "Truncated fixture integer");
            std::uint32_t value; std::memcpy(&value, bytes.data() + offset, 4); return value;
        };
        require(tag(0) == "TES4" && integer(8) == 0 && integer(12) == 0, "Invalid proof plugin header");
        require(tag(24) == "HEDR" && integer(34) == 5 && integer(38) == 0x803,
            "Invalid proof plugin record count/next FormID");
        std::size_t cursor = 24 + integer(4), count = 0;
        while (cursor < bytes.size())
        {
            require(tag(cursor) == "GRUP" && integer(cursor + 12) == 0, "Invalid proof plugin top group");
            const auto type = tag(cursor + 8);
            const auto groupEnd = cursor + integer(cursor + 4);
            require(groupEnd <= bytes.size(), "Invalid proof plugin group size");
            cursor += 24;
            while (cursor < groupEnd)
            {
                require(tag(cursor) == type && integer(cursor+8) == 0, "Fixture record escaped its group");
                const auto formId = integer(cursor + 12);
                require(formId >= 0x01000800 && formId <= 0x01000802, "Invalid fixture self FormID");
                const auto recordEnd = cursor + 24 + integer(cursor+4);
                require(recordEnd <= groupEnd, "Fixture record exceeds group size");
                cursor += 24;
                bool editorId = false, bounds = false, modelPath = false, requiredData = false;
                while (cursor < recordEnd)
                {
                    require(cursor + 6 <= recordEnd, "Truncated fixture subrecord");
                    std::uint16_t size; std::memcpy(&size, bytes.data()+cursor+4, 2);
                    const auto name = tag(cursor); cursor += 6;
                    require(cursor + size <= recordEnd, "Fixture subrecord exceeds record size");
                    editorId |= name == "EDID";
                    bounds |= name == "OBND" && size == 12;
                    modelPath |= name == "MODL" && size > 1 && bytes[cursor+size-1] == std::byte{};
                    requiredData |= type == "STAT" ? name == "DNAM" && size == 16 : name == "DATA" && size == 1;
                    cursor += size;
                }
                require(editorId && bounds && modelPath && requiredData, "Missing required FO4 fixture record fields");
                ++count;
            }
        }
        require(count == 3 && cursor == bytes.size(), "Incorrect fixture record/group counts");
    }

    void fixtures(const std::filesystem::path& root)
    {
        require(!std::filesystem::exists(root), "Fixture output directory already exists");
        auto geometry = cube();
        geometry.surfaces[0].alphaCutoff = 0; geometry.surfaces[0].baseColor.a = 1;
        const auto dynamic = cubeCollision(true);
        const auto dynamicPackage = exportNativeModelAsset(root / "NativeFixtures.esm", 0x01000800, geometry, &dynamic);
        std::cout << "Movable cube: " << (dynamicPackage.assetRoot / dynamicPackage.modelPath).string() << '\n';
        ModelGeometry doorway;
        auto& wall = doorway.surfaces.emplace_back(); wall.metallicFactor = 0; wall.roughnessFactor = 1;
        addBox(wall, {-1.5f,-0.25f,0}, {-0.7f,0.25f,3});
        addBox(wall, {0.7f,-0.25f,0}, {1.5f,0.25f,3});
        addBox(wall, {-0.7f,-0.25f,2.3f}, {0.7f,0.25f,3});
        HavokPhysicsSystem collision;
        auto& body = collision.bodies.emplace_back();
        body.triangleMesh.emplace();
        for (const auto& vertex : wall.vertices) body.triangleMesh->vertices.push_back(engineToHavokPosition(vertex.position));
        body.triangleMesh->indices = wall.indices;
        const auto staticPackage = exportNativeModelAsset(root / "NativeFixtures.esm", 0x01000801, doorway, &collision);
        std::cout << "Static doorway: " << (staticPackage.assetRoot / staticPackage.modelPath).string() << '\n';
        ModelGeometry compoundGeometry;
        auto& coarse = compoundGeometry.surfaces.emplace_back();
        coarse.metallicFactor = 0; coarse.roughnessFactor = 1;
        addBox(coarse, {-1,-0.5f,0}, {0,0.5f,1});
        addBox(coarse, {0,-0.5f,0}, {1,0.5f,1});
        auto compound = cubeCollision(true);
        compound.bodies[0].parts.clear();
        for (const float x : {-1.0f, 0.0f})
        {
            auto& hull = compound.bodies[0].parts.emplace_back().polytope;
            hull.vertices = boxVertices({x,-0.5f,0}, {x+1,0.5f,1});
            hull.indices.assign(boxTriangles.begin(), boxTriangles.end());
            hull.convexRadius = 0.005f;
        }
        compound.bodies[0].inertiaDiagonal = {1.0f/6.0f, 5.0f/12.0f, 5.0f/12.0f};
        const auto compoundPackage = exportNativeModelAsset(root / "NativeFixtures.esm", 0x01000802, compoundGeometry, &compound);
        std::cout << "Movable compound: " << (compoundPackage.assetRoot / compoundPackage.modelPath).string() << '\n';
        const auto plugin = root / "OpenF4NativeCollisionProof.esp";
        proofPlugin(plugin, dynamicPackage, staticPackage, compoundPackage);
        testProofPlugin(plugin);
        std::cout << "Disposable game plugin: " << plugin.string() << '\n'
            << "01000800 MSTT OF4NativeProofCube\n01000801 STAT OF4NativeProofDoorway\n"
            << "01000802 MSTT OF4NativeProofCompound\n"
            << "Fresh main menu: coc qasmoke; help OF4NativeProof 4; player.placeatme <resolved FormID>\n";
    }
}

int main(int argc, char** argv)
{
    try
    {
        if (argc == 3 && std::string_view(argv[1]) == "--export-fixtures")
        {
            fixtures(std::filesystem::absolute(argv[2]));
            return 0;
        }
        const auto root = std::filesystem::temp_directory_path() / ("openf4-native-export-tests-"
            + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        require(std::filesystem::create_directory(root), "Cannot create test directory");
        testPackage(root); testAuthoredCollisionReload(root); testRejections(root); testTexturesAndChunks(root); fixtures(root / "game-proof");
        std::filesystem::remove_all(root);
        std::cout << "Native model export tests passed\n";
        return 0;
    }
    catch (const std::exception& exception)
    {
        std::cerr << exception.what() << '\n';
        return 1;
    }
}
