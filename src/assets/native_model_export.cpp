#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "assets/native_model_export.hpp"

#include "assets/asset_resolver.hpp"
#include "assets/coordinates.hpp"
#include "assets/material_file.hpp"

#include <DirectXTex.h>
#include <ExtraData.hpp>
#include <NifFile.hpp>
#include <Shaders.hpp>
#include <bhk.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <set>
#include <stdexcept>
#include <unordered_map>

namespace openf4::assets
{
    namespace
    {
        template<class T> void append(std::vector<std::byte>& bytes, const T& value)
        {
            const auto* first = reinterpret_cast<const std::byte*>(&value);
            bytes.insert(bytes.end(), first, first + sizeof(T));
        }

        void string(std::vector<std::byte>& bytes, std::string value)
        {
            std::replace(value.begin(), value.end(), '/', '\\');
            append(bytes, static_cast<std::uint32_t>(value.size() + 1));
            for (char c : value) bytes.push_back(std::byte(c));
            bytes.push_back(std::byte{});
        }

        void write(const std::filesystem::path& path, std::span<const std::byte> bytes)
        {
            std::filesystem::create_directories(path.parent_path());
            std::ofstream output(path, std::ios::binary | std::ios::trunc);
            if (!output || !output.write(reinterpret_cast<const char*>(bytes.data()),
                    static_cast<std::streamsize>(bytes.size())) || !output.flush())
                throw std::runtime_error("Cannot write native asset: " + path.string());
        }

        template<glm::length_t N> bool finite(const glm::vec<N, float>& value)
        {
            for (glm::length_t i = 0; i < N; ++i)
                if (!std::isfinite(value[i])) return false;
            return true;
        }

        void validateGeometry(const ModelGeometry& geometry)
        {
            if (!geometry.skinBones.empty() || !geometry.animations.empty())
                throw std::runtime_error("Native static export cannot convert skinned or animated geometry");
            if (!geometry.attachmentPoints.empty() || !geometry.attachmentTargets.empty())
                throw std::runtime_error("Native static export cannot discard attachment metadata; use collision-only editing");
            if (geometry.surfaces.empty()) throw std::runtime_error("Native model has no surfaces");
            for (const auto& surface : geometry.surfaces)
            {
                if (surface.vertices.empty() || surface.indices.empty() || surface.indices.size() % 3)
                    throw std::runtime_error("Native model requires indexed triangles");
                if (!finite(surface.baseColor) || !std::isfinite(surface.alphaCutoff)
                    || !std::isfinite(surface.roughnessFactor) || !std::isfinite(surface.metallicFactor)
                    || glm::any(glm::lessThan(surface.baseColor, glm::vec4(0)))
                    || glm::any(glm::greaterThan(surface.baseColor, glm::vec4(1)))
                    || surface.alphaCutoff < 0 || surface.alphaCutoff > 1)
                    throw std::runtime_error("Native model has invalid material factors");
                for (const auto& v : surface.vertices)
                    if (!finite(v.position) || !finite(v.normal) || !finite(v.uv) || !finite(v.color)
                        || glm::length(v.normal) < 0.00001f || glm::any(glm::greaterThan(glm::abs(v.position), glm::vec3(1e7f)))
                        || glm::any(glm::notEqual(v.boneWeights, glm::vec4(0))))
                        throw std::runtime_error("Native model has invalid or skinned vertex data");
                for (auto index : surface.indices)
                    if (index >= surface.vertices.size())
                        throw std::runtime_error("Native model has an out-of-range triangle index");
                for (std::size_t i = 0; i < surface.indices.size(); i += 3)
                {
                    const auto a = surface.vertices[surface.indices[i]].position;
                    const auto b = surface.vertices[surface.indices[i + 1]].position;
                    const auto c = surface.vertices[surface.indices[i + 2]].position;
                    if (glm::length(glm::cross(b - a, c - a)) < 1e-8f)
                        throw std::runtime_error("Native model contains a degenerate triangle");
                }
                if (!surface.diffuseImage && !surface.diffuseTexture.empty())
                    throw std::runtime_error("Resolve diffuse texture pixels before native export");
                if (!surface.normalImage && !surface.normalTexture.empty())
                    throw std::runtime_error("Resolve normal texture pixels before native export");
            }
        }

        // BGSM v2 field layout, shared with the Fallout 4 Material Editor format.
        std::vector<std::byte> material(const ModelSurface& surface,
            const std::string& diffuse, const std::string& normal)
        {
            std::vector<std::byte> bytes;
            const auto flag = [&](bool v) { append(bytes, static_cast<std::uint8_t>(v)); };
            const auto scalar = [&](float v) { append(bytes, v); };
            append(bytes, std::uint32_t{0x4d534742}); append(bytes, std::uint32_t{2});
            append(bytes, std::uint32_t{3}); // Repeat U and V.
            scalar(0); scalar(0); scalar(1); scalar(1); scalar(1);
            flag(surface.alphaBlend);
            append(bytes, std::uint32_t(surface.alphaBlend ? 6 : 0));
            append(bytes, std::uint32_t(surface.alphaBlend ? 7 : 0));
            append(bytes, static_cast<std::uint8_t>(std::lround(surface.alphaCutoff * 255)));
            flag(surface.alphaCutoff > 0); flag(!surface.alphaBlend); flag(true);
            flag(false); flag(false); flag(surface.decal); flag(surface.doubleSided);
            flag(false); flag(false); flag(false); flag(false); scalar(0);
            flag(false); scalar(1); flag(false);
            // BGSM paths are relative to Textures; NIF texture slots are relative to Data.
            string(bytes, diffuse.substr(9)); string(bytes, normal.substr(9));
            for (int i = 0; i < 7; ++i) string(bytes, "");
            flag(false); // Editor alpha reference.
            flag(false); scalar(2); scalar(0); flag(false); scalar(0);
            flag(false); scalar(1); scalar(1); scalar(1); scalar(0); scalar(0); scalar(5);
            for (int i = 0; i < 6; ++i) scalar(-1); // Inherit wetness defaults.
            string(bytes, ""); flag(false); flag(false); scalar(1);
            flag(false); flag(false); flag(false);
            flag(true); flag(false); flag(true); flag(false); flag(false); flag(false);
            flag(false); flag(false); flag(false); scalar(1); scalar(1); scalar(1);
            flag(false); flag(false); flag(false); flag(false);
            for (int i = 0; i < 5; ++i) scalar(0); // Displacement/tessellation disabled.
            scalar(1); flag(false);
            return bytes;
        }

        float linear(float value)
        {
            return value <= 0.04045f ? value / 12.92f : std::pow((value + 0.055f) / 1.055f, 2.4f);
        }

        float srgb(float value)
        {
            return value <= 0.0031308f ? value * 12.92f : 1.055f * std::pow(value, 1 / 2.4f) - 0.055f;
        }

        void texture(const std::filesystem::path& path, const ModelImage* input,
            glm::vec4 factor, bool normal)
        {
            const std::array<std::uint8_t, 4> fallback = normal
                ? std::array<std::uint8_t, 4>{128, 128, 255, 255}
                : std::array<std::uint8_t, 4>{255, 255, 255, 255};
            DirectX::Image source{1, 1, DXGI_FORMAT_R8G8B8A8_UNORM, 4, 4,
                const_cast<std::uint8_t*>(fallback.data())};
            if (input)
            {
                if (!input->width || !input->height || input->width > 16384 || input->height > 16384)
                    throw std::runtime_error("Invalid native texture dimensions");
                source.width = input->width; source.height = input->height;
                source.format = input->encoding == ModelImageEncoding::rgba8
                    ? DXGI_FORMAT_R8G8B8A8_UNORM : DirectX::MakeLinear(static_cast<DXGI_FORMAT>(input->dxgiFormat));
                if (FAILED(DirectX::ComputePitch(source.format, source.width, source.height,
                        source.rowPitch, source.slicePitch)) || source.slicePitch > input->pixels.size())
                    throw std::runtime_error("Truncated native texture pixels");
                source.pixels = reinterpret_cast<std::uint8_t*>(const_cast<std::byte*>(input->pixels.data()));
            }
            DirectX::ScratchImage converted;
            if (source.format != DXGI_FORMAT_R8G8B8A8_UNORM)
            {
                const auto result = DirectX::IsCompressed(source.format)
                    ? DirectX::Decompress(source, DXGI_FORMAT_R8G8B8A8_UNORM, converted)
                    : DirectX::Convert(source, DXGI_FORMAT_R8G8B8A8_UNORM, DirectX::TEX_FILTER_DEFAULT, 0.5f, converted);
                if (FAILED(result)) throw std::runtime_error("Cannot decode native source texture");
                source = *converted.GetImage(0, 0, 0);
            }
            std::vector<std::uint8_t> pixels(source.width * source.height * 4);
            for (std::size_t y = 0; y < source.height; ++y)
                for (std::size_t x = 0; x < source.width; ++x)
                    for (int c = 0; c < 4; ++c)
                    {
                        float value = source.pixels[y * source.rowPitch + x * 4 + c] / 255.0f;
                        if (!normal && c < 3)
                            value = srgb((!input || input->srgb ? linear(value) : value) * factor[c]);
                        else if (!normal) value *= factor[c];
                        pixels[(y * source.width + x) * 4 + c] = static_cast<std::uint8_t>(
                            std::lround(std::clamp(value, 0.0f, 1.0f) * 255));
                    }
            source.pixels = pixels.data(); source.rowPitch = source.width * 4;
            source.slicePitch = source.rowPitch * source.height;
            source.format = normal ? DXGI_FORMAT_R8G8B8A8_UNORM : DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
            DirectX::ScratchImage mips;
            DirectX::Blob dds;
            HRESULT result;
            if (source.width == 1 && source.height == 1)
                result = DirectX::SaveToDDSMemory(source, DirectX::DDS_FLAGS_FORCE_DX10_EXT, dds);
            else
            {
                if (FAILED(DirectX::GenerateMipMaps(source, DirectX::TEX_FILTER_FORCE_NON_WIC, 0, mips)))
                    throw std::runtime_error("Cannot generate native texture mipmaps");
                result = DirectX::SaveToDDSMemory(mips.GetImages(), mips.GetImageCount(),
                    mips.GetMetadata(), DirectX::DDS_FLAGS_FORCE_DX10_EXT, dds);
            }
            if (FAILED(result))
                throw std::runtime_error("Cannot encode native DDS texture");
            write(path, {reinterpret_cast<const std::byte*>(dds.GetBufferPointer()), dds.GetBufferSize()});
        }

        void addShapes(nifly::NifFile& file, const ModelSurface& surface,
            std::size_t surfaceIndex, std::string materialName, std::string diffuse, std::string normal)
        {
            std::vector<nifly::Vector3> positions, normals;
            std::vector<nifly::Vector2> uvs;
            std::vector<nifly::Color4> colors;
            std::vector<nifly::Triangle> triangles;
            std::unordered_map<std::uint32_t, std::uint16_t> local;
            unsigned chunk = 0;
            const auto flush = [&] {
                if (triangles.empty()) return;
                auto* shape = file.CreateShapeFromData("Surface " + std::to_string(surfaceIndex)
                    + " part " + std::to_string(chunk++), &positions, &triangles, &uvs, &normals);
                if (!shape) throw std::runtime_error("Cannot create native NIF shape");
                if (auto* bst = dynamic_cast<nifly::BSTriShape*>(shape)) bst->SetFullPrecision(true);
                file.SetColorsForShape(shape, colors);
                auto* shader = file.GetShader(shape);
                shader->name.get() = "materials\\" + materialName;
                std::replace(shader->name.get().begin(), shader->name.get().end(), '/', '\\');
                shader->SetWetMaterialName("");
                shader->SetDoubleSided(surface.doubleSided);
                shader->SetVertexColors(true); shader->SetVertexAlpha(true);
                shader->SetSpecularStrength(0);
                std::replace(diffuse.begin(), diffuse.end(), '/', '\\');
                std::replace(normal.begin(), normal.end(), '/', '\\');
                file.SetTextureSlot(shape, diffuse, 0); file.SetTextureSlot(shape, normal, 1);
                if (surface.alphaBlend || surface.alphaCutoff > 0)
                {
                    auto alpha = std::make_unique<nifly::NiAlphaProperty>();
                    alpha->flags = surface.alphaBlend ? 4845 : 4844;
                    if (surface.alphaCutoff > 0) alpha->flags |= 0x0200;
                    else alpha->flags &= ~0x0200;
                    alpha->threshold = static_cast<std::uint8_t>(std::lround(surface.alphaCutoff * 255));
                    shape->AlphaPropertyRef()->index = file.GetHeader().AddBlock(std::move(alpha));
                }
                positions.clear(); normals.clear(); uvs.clear(); colors.clear(); triangles.clear(); local.clear();
            };
            for (std::size_t i = 0; i < surface.indices.size(); i += 3)
            {
                if (positions.size() > 65532) flush();
                std::array<std::uint16_t, 3> tri{};
                for (int c = 0; c < 3; ++c)
                {
                    const auto source = surface.indices[i + c];
                    auto [it, inserted] = local.try_emplace(source, static_cast<std::uint16_t>(positions.size()));
                    if (inserted)
                    {
                        const auto& v = surface.vertices[source];
                        const auto p = swapBethesdaAxes(v.position);
                        const auto n = swapBethesdaAxes(glm::normalize(v.normal));
                        positions.emplace_back(p.x, p.y, p.z);
                        normals.emplace_back(n.x, n.y, n.z); uvs.emplace_back(v.uv.x, v.uv.y);
                        colors.emplace_back(v.color.r, v.color.g, v.color.b, v.color.a);
                    }
                    tri[c] = it->second;
                }
                triangles.emplace_back(tri[0], tri[1], tri[2]);
            }
            flush();
        }

        void addCollision(nifly::NifFile& file, const HavokPhysicsSystem& collision)
        {
            if (collision.bodies.size() != 1 || collision.bodies.front().bodyIndex != 0)
                throw std::runtime_error("Native static export requires one root body; use compound pieces within that body");
            const auto bytes = serializeHavokPhysicsSystem(collision);
            auto physics = std::make_unique<nifly::bhkPhysicsSystem>(static_cast<std::uint32_t>(bytes.size()));
            std::memcpy(physics->data.data(), bytes.data(), bytes.size());
            auto& header = file.GetHeader();
            const auto physicsIndex = header.AddBlock(std::move(physics));
            auto* root = file.GetRootNode();
            auto object = std::make_unique<nifly::bhkNPCollisionObject>();
            object->flags = 0x0080;
            object->bodyRef.index = physicsIndex; object->bodyID = 0;
            object->targetRef.index = file.GetBlockID(root);
            root->collisionRef.index = header.AddBlock(std::move(object));
            auto flags = std::make_unique<nifly::BSXFlags>();
            flags->name.get() = "BSX";
            flags->integerData = nifly::BSX_HAVOK | nifly::BSX_ARTICULATED;
            if (collision.bodies.front().dynamic)
                flags->integerData |= nifly::BSX_DYNAMIC;
            root->extraDataRefs.AddBlockRef(header.AddBlock(std::move(flags)));
        }

        struct RawNif
        {
            struct Block { std::string type; std::size_t offset, size, sizeField; };
            std::vector<Block> blocks;
        };

        RawNif indexNif(std::span<const std::byte> bytes)
        {
            std::size_t cursor = 0;
            const auto skip = [&](std::size_t amount) {
                if (amount > bytes.size() - cursor) throw std::runtime_error("Truncated native NIF");
                cursor += amount;
            };
            const auto number = [&]<class T>() {
                const auto start = cursor; skip(sizeof(T));
                T value; std::memcpy(&value, bytes.data() + start, sizeof(T)); return value;
            };
            while (cursor < bytes.size() && bytes[cursor] != std::byte{'\n'}) skip(1);
            skip(1);
            if (number.operator()<std::uint32_t>() != 0x14020007 || number.operator()<std::uint8_t>() != 1
                || number.operator()<std::uint32_t>() != 12)
                throw std::runtime_error("Collision replacement requires a little-endian Fallout 4 NIF");
            const auto count = number.operator()<std::uint32_t>();
            if (count > bytes.size() / 4) throw std::runtime_error("Invalid NIF block count");
            const auto version = number.operator()<std::uint32_t>();
            if (version != 130) throw std::runtime_error("Collision replacement supports Fallout 4 NIF stream version 130 only");
            for (int i = 0; i < 4; ++i) skip(number.operator()<std::uint8_t>());
            const auto typeCount = number.operator()<std::uint16_t>();
            std::vector<std::string> types;
            for (unsigned i = 0; i < typeCount; ++i)
            {
                const auto size = number.operator()<std::uint32_t>();
                const auto start = cursor; skip(size);
                types.emplace_back(reinterpret_cast<const char*>(bytes.data() + start), size);
            }
            RawNif result;
            for (std::uint32_t i = 0; i < count; ++i)
            {
                const auto type = number.operator()<std::uint16_t>();
                if (type >= types.size()) throw std::runtime_error("Invalid native NIF block type");
                result.blocks.push_back({types[type], 0, 0, 0});
            }
            for (auto& block : result.blocks)
            {
                block.sizeField = cursor; block.size = number.operator()<std::uint32_t>();
            }
            const auto strings = number.operator()<std::uint32_t>(); skip(4);
            if (strings > bytes.size() / 4) throw std::runtime_error("Invalid NIF string count");
            for (std::uint32_t i = 0; i < strings; ++i) skip(number.operator()<std::uint32_t>());
            const auto groups = number.operator()<std::uint32_t>();
            if (groups > bytes.size() / 4) throw std::runtime_error("Invalid NIF group count");
            skip(static_cast<std::size_t>(groups) * 4);
            for (auto& block : result.blocks) { block.offset = cursor; skip(block.size); }
            const auto roots = number.operator()<std::uint32_t>();
            if (roots > count) throw std::runtime_error("Invalid NIF root count");
            for (std::uint32_t i = 0; i < roots; ++i)
                if (number.operator()<std::uint32_t>() >= count) throw std::runtime_error("Invalid NIF root reference");
            return result;
        }
    }

    NativeModelPackage exportNativeModelAsset(const std::filesystem::path& worldEsm,
        std::uint32_t formId, const ModelGeometry& geometry, const HavokPhysicsSystem* collision)
    {
        validateGeometry(geometry);
        if (!formId || worldEsm.empty()) throw std::runtime_error("Native export needs an asset FormID and world path");
        NativeModelPackage result;
        result.assetRoot = std::filesystem::absolute(AssetResolver::projectAssetRoot(worldEsm));
        std::ostringstream id; id << std::hex << std::setfill('0') << std::setw(8) << formId;
        static std::atomic_uint64_t sequence{};
        const auto version = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())
            + "-" + std::to_string(sequence++);
        const auto relative = std::filesystem::path("openf4") / id.str() / version;
        std::filesystem::create_directories(result.assetRoot);
        const auto staging = result.assetRoot / (".native-stage-" + version);
        if (!std::filesystem::create_directory(staging)) throw std::runtime_error("Native export staging collision");
        std::vector<std::filesystem::path> published;
        try
        {
            nifly::NifFile file; file.Create(nifly::NiVersion::getFO4());
            file.GetRootNode()->name.get() = "OpenF4 Native Model";
            if (collision) addCollision(file, *collision);
            for (std::size_t i = 0; i < geometry.surfaces.size(); ++i)
            {
                const auto& surface = geometry.surfaces[i];
                const auto name = "surface" + std::to_string(i);
                const auto diffuse = (std::filesystem::path("textures") / relative / (name + "_d.dds")).generic_string();
                const auto normal = (std::filesystem::path("textures") / relative / (name + "_n.dds")).generic_string();
                const auto materialName = (relative / (name + ".bgsm")).generic_string();
                texture(staging / diffuse, surface.diffuseImage ? &*surface.diffuseImage : nullptr, surface.baseColor, false);
                texture(staging / normal, surface.normalImage ? &*surface.normalImage : nullptr, glm::vec4(1), true);
                const auto bgsm = material(surface, diffuse, normal);
                (void)loadMaterial(bgsm);
                write(staging / "materials" / materialName, bgsm);
                addShapes(file, surface, i, materialName, diffuse, normal);
                if (surface.metallicRoughnessImage || surface.metallicFactor != 0 || surface.roughnessFactor != 1)
                    result.diagnostics.push_back("Surface " + std::to_string(i)
                        + ": metallic/roughness material conversion is not supported; exported non-specular base color and normal");
            }
            result.modelPath = (std::filesystem::path("meshes") / relative / "model.nif").generic_string();
            const auto nifPath = staging / result.modelPath;
            std::filesystem::create_directories(nifPath.parent_path());
            if (file.Save(nifPath) != 0) throw std::runtime_error("Cannot serialize native NIF");
            nifly::NifFile validation(nifPath);
            if (!validation.IsValid() || validation.GetShapes().empty())
                throw std::runtime_error("Native NIF failed post-write validation");
            // No reference can resolve the new NIF until all its dependencies are published.
            for (const auto* category : {"textures", "materials", "meshes"})
            {
                const auto destination = result.assetRoot / category / relative;
                std::filesystem::create_directories(destination.parent_path());
                std::filesystem::rename(staging / category / relative, destination);
                published.push_back(destination);
            }
            std::filesystem::remove_all(staging);
            return result;
        }
        catch (...)
        {
            std::error_code ignored;
            for (const auto& path : published) std::filesystem::remove_all(path, ignored);
            std::filesystem::remove_all(staging, ignored);
            throw;
        }
    }

    std::vector<std::byte> replaceNativeNifCollision(std::span<const std::byte> original,
        std::uint32_t physicsBlockIndex, const HavokPhysicsSystem& collision)
    {
        const auto index = indexNif(original);
        if (physicsBlockIndex >= index.blocks.size() || index.blocks[physicsBlockIndex].type != "bhkPhysicsSystem")
            throw std::runtime_error("Selected NIF block is not a bhkPhysicsSystem");
        const auto& block = index.blocks[physicsBlockIndex];
        if (block.size < 4) throw std::runtime_error("Truncated NIF physics block");
        std::uint32_t length; std::memcpy(&length, original.data() + block.offset, 4);
        if (length != block.size - 4) throw std::runtime_error("Invalid NIF physics payload size");
        const auto previousBytes = original.subspan(block.offset + 4, length);
        const auto previous = loadEditableHavokPhysicsSystem(previousBytes);
        if (previous.bodies.size() != collision.bodies.size())
            throw std::runtime_error("Collision replacement cannot change body/node attachment count");
        for (std::size_t i = 0; i < previous.bodies.size(); ++i)
            if (previous.bodies[i].bodyIndex != collision.bodies[i].bodyIndex
                || previous.bodies[i].dynamic != collision.bodies[i].dynamic)
                throw std::runtime_error("Collision replacement cannot change body identity or NIF motion flags");
        const auto bytes = serializeHavokPhysicsSystem(collision);
        if (bytes.size() > std::numeric_limits<std::uint32_t>::max() - 4)
            throw std::runtime_error("Native collision payload is too large");
        std::vector<std::byte> result(original.begin(), original.begin() + block.offset);
        append(result, static_cast<std::uint32_t>(bytes.size()));
        result.insert(result.end(), bytes.begin(), bytes.end());
        result.insert(result.end(), original.begin() + block.offset + block.size, original.end());
        const auto newSize = static_cast<std::uint32_t>(bytes.size() + 4);
        std::memcpy(result.data() + block.sizeField, &newSize, 4);
        return result;
    }

    bool nativeMaterialIsEditable(std::span<const std::byte> bytes)
    {
        try
        {
            const auto value = loadMaterial(bytes);
            ModelSurface surface;
            surface.alphaBlend = value.alphaBlend;
            surface.alphaCutoff = value.alphaTest ? value.alphaCutoff : 0;
            surface.doubleSided = value.twoSided; surface.decal = value.decal;
            auto diffuse = value.diffuseTexture, normal = value.normalTexture;
            std::ranges::replace(diffuse, '\\', '/'); std::ranges::replace(normal, '\\', '/');
            if (!diffuse.starts_with("textures/")) diffuse = "textures/" + diffuse;
            if (!normal.starts_with("textures/")) normal = "textures/" + normal;
            const auto canonical = material(surface, diffuse, normal);
            return bytes.size() == canonical.size() && std::ranges::equal(bytes, canonical);
        }
        catch (const std::exception&) { return false; }
    }

    NativeModelInspection inspectNativeModel(std::span<const std::byte> bytes)
    {
        const auto raw = indexNif(bytes);
        std::istringstream input(std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size()),
            std::ios::binary);
        nifly::NifFile file(input);
        if (!file.IsValid()) throw std::runtime_error("Cannot inspect malformed native NIF");
        NativeModelInspection result;
        result.authoredByOpenF4 = file.GetRootNode()
            && file.GetRootNode()->name.get() == "OpenF4 Native Model";
        const std::set<std::string> authoredTypes{"NiNode", "BSTriShape", "BSSubIndexTriShape", "BSLightingShaderProperty",
            "BSShaderTextureSet", "NiAlphaProperty", "BSXFlags", "bhkPhysicsSystem", "bhkNPCollisionObject"};
        for (std::size_t i = 0; i < raw.blocks.size(); ++i)
        {
            const auto& block = raw.blocks[i];
            if (!authoredTypes.contains(block.type)) result.authoredByOpenF4 = false;
            if (block.type.find("Controller") != std::string::npos || block.type.find("Sequence") != std::string::npos
                || block.type.find("Interpolator") != std::string::npos || block.type == "BSBehaviorGraphExtraData")
                result.hasAnimation = true;
            if (block.type.find("Ragdoll") != std::string::npos || block.type.find("Constraint") != std::string::npos)
                result.collisionReadOnlyReason = "Native ragdoll and constraint editing is unsupported";
            if (block.type != "bhkPhysicsSystem") continue;
            try
            {
                if (block.size < 4) throw std::runtime_error("Truncated native physics block");
                std::uint32_t size; std::memcpy(&size, bytes.data() + block.offset, 4);
                if (size != block.size - 4) throw std::runtime_error("Invalid native physics payload");
                const auto payload = bytes.subspan(block.offset + 4, size);
                result.collision.push_back({static_cast<std::uint32_t>(i), loadHavokPhysicsSystem(payload)});
                (void)loadEditableHavokPhysicsSystem(payload);
            }
            catch (const std::exception& e)
            {
                result.collisionReadOnlyReason = e.what();
                result.authoredByOpenF4 = false;
            }
        }
        if (result.collision.empty() && !result.authoredByOpenF4)
            result.collisionReadOnlyReason = "Native NIF has no editable collision attachment";
        std::set<std::string> dependencies;
        const auto add = [&](std::string path) {
            if (path.empty()) return;
            path = AssetResolver::normalizePath(path);
            std::ranges::replace(path, '\\', '/');
            const auto extension = std::filesystem::path(path).extension().string();
            if (extension == ".bgsm" || extension == ".bgem")
            {
                if (!path.starts_with("materials/")) path = "materials/" + path;
            }
            else if (extension == ".dds")
            {
                if (!path.starts_with("textures/")) path = "textures/" + path;
            }
            else return;
            dependencies.insert(std::move(path));
        };
        for (std::uint32_t i = 0; i < file.GetHeader().GetStringCount(); ++i)
            add(file.GetHeader().GetStringById(i));
        for (auto* shape : file.GetShapes())
            for (const auto& path : file.GetTexturePathRefs(shape)) add(path.get());
        result.dependencies.assign(dependencies.begin(), dependencies.end());
        return result;
    }

    NativeModelPackage exportNativeCollisionAsset(const std::filesystem::path& worldEsm,
        std::uint32_t formId, std::span<const std::byte> original,
        std::span<const NativeCollisionBlock> collision, const std::filesystem::path& sourceEsm,
        std::optional<std::filesystem::path> dataDirectory)
    {
        if (!formId || worldEsm.empty()) throw std::runtime_error("Native export needs a world and FormID");
        const auto inspection = inspectNativeModel(original);
        if (!inspection.collisionReadOnlyReason.empty())
            throw std::runtime_error(inspection.collisionReadOnlyReason);
        if (inspection.collision.size() != collision.size())
            throw std::runtime_error("Native collision save cannot change attachment count");
        std::vector<std::byte> bytes(original.begin(), original.end());
        for (std::size_t i = 0; i < collision.size(); ++i)
        {
            if (collision[i].blockIndex != inspection.collision[i].blockIndex)
                throw std::runtime_error("Native collision save cannot change attachment identities");
            bytes = replaceNativeNifCollision(bytes, collision[i].blockIndex, collision[i].physics);
        }
        AssetResolver source(sourceEsm, std::move(dataDirectory));
        NativeModelPackage result;
        result.assetRoot = std::filesystem::absolute(AssetResolver::projectAssetRoot(worldEsm));
        std::map<std::string, std::vector<std::byte>> dependencies;
        const auto read = [](const std::filesystem::path& path) {
            std::ifstream input(path, std::ios::binary | std::ios::ate);
            if (!input || input.tellg() < 0) throw std::runtime_error("Cannot read native dependency: " + path.string());
            std::vector<std::byte> value(static_cast<std::size_t>(input.tellg()));
            input.seekg(0);
            if (!input.read(reinterpret_cast<char*>(value.data()), static_cast<std::streamsize>(value.size())))
                throw std::runtime_error("Cannot read complete native dependency: " + path.string());
            return value;
        };
        std::vector<std::string> pending = inspection.dependencies;
        std::set<std::string> visited;
        for (std::size_t i = 0; i < pending.size(); ++i)
        {
            auto path = AssetResolver::normalizePath(pending[i]);
            std::ranges::replace(path, '\\', '/');
            if (!visited.insert(path).second) continue;
            auto asset = source.read(path);
            if (std::filesystem::exists(result.assetRoot / path) && read(result.assetRoot / path) != asset.bytes)
                throw std::runtime_error("Native dependency conflicts with existing project file: " + path);
            // Game Data dependencies retain their original paths; only project files need publication.
            if (asset.source == AssetSource::project) dependencies.emplace(path, asset.bytes);
            const auto extension = std::filesystem::path(path).extension().string();
            if (extension != ".bgsm" && extension != ".bgem") continue;
            (void)loadMaterial(asset.bytes);
            // All material asset references are length-prefixed strings; retain optional texture/root fields too.
            for (std::size_t offset = 0; offset + 4 <= asset.bytes.size(); ++offset)
            {
                std::uint32_t length; std::memcpy(&length, asset.bytes.data() + offset, 4);
                if (length < 5 || length > 4096 || length > asset.bytes.size() - offset - 4) continue;
                std::string value(reinterpret_cast<const char*>(asset.bytes.data() + offset + 4), length);
                if (!value.empty() && value.back() == '\0') value.pop_back();
                if (value.empty() || std::ranges::any_of(value, [](unsigned char c) { return c < 32 || c == 127; })) continue;
                value = AssetResolver::normalizePath(value);
                std::ranges::replace(value, '\\', '/');
                const auto ext = std::filesystem::path(value).extension().string();
                const char* prefix = ext == ".dds" ? "textures/" :
                    (ext == ".bgsm" || ext == ".bgem" ? "materials/" : nullptr);
                if (!prefix) continue;
                if (!value.starts_with(prefix)) value = std::string(prefix) + value;
                pending.push_back(std::move(value));
            }
        }
        static std::atomic_uint64_t sequence{};
        const auto version = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())
            + "-raw-" + std::to_string(sequence++);
        std::ostringstream id; id << std::hex << std::setfill('0') << std::setw(8) << formId;
        result.modelPath = "meshes/openf4/" + id.str() + "/" + version + "/model.nif";
        std::filesystem::create_directories(result.assetRoot);
        const auto staging = result.assetRoot / (".native-stage-" + version);
        if (!std::filesystem::create_directory(staging)) throw std::runtime_error("Native export staging collision");
        std::vector<std::filesystem::path> published;
        try
        {
            for (const auto& [path, value] : dependencies) write(staging / path, value);
            write(staging / "model.nif", bytes);
            for (const auto& [path, value] : dependencies)
            {
                const auto destination = result.assetRoot / path;
                if (std::filesystem::exists(destination))
                {
                    if (read(destination) != value) throw std::runtime_error("Native dependency changed during export: " + path);
                    continue;
                }
                std::filesystem::create_directories(destination.parent_path());
                // A same-volume hard link publishes the complete file without replacing a concurrent writer.
                std::filesystem::create_hard_link(staging / path, destination);
                published.push_back(destination);
            }
            const auto destination = result.assetRoot / result.modelPath;
            std::filesystem::create_directories(destination.parent_path());
            std::filesystem::rename(staging / "model.nif", destination);
            std::error_code ignored;
            std::filesystem::remove_all(staging, ignored);
            return result;
        }
        catch (...)
        {
            std::error_code ignored;
            for (const auto& path : published) std::filesystem::remove(path, ignored);
            std::filesystem::remove_all(staging, ignored);
            throw;
        }
    }
}
