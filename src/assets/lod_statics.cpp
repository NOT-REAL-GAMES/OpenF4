#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "assets/lod_statics.hpp"

#include "assets/asset_resolver.hpp"
#include "assets/dds_texture.hpp"
#include "assets/esm_document.hpp"
#include "assets/model_import.hpp"
#include "assets/material_file.hpp"
#include "assets/nif_model.hpp"

#include <assimp/GltfMaterial.h>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <meshoptimizer.h>
#include <DirectXTex.h>

#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <algorithm>
#include <atomic>
#include <charconv>
#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <map>
#include <ranges>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/quaternion.hpp>

namespace openf4::assets
{
    namespace
    {
        constexpr float cellSize = 4096.0f;
        constexpr float unitsPerMeter = gltfBethesdaUnitsPerMeter;

        std::int32_t tileBase(std::int32_t cell)
        {
            // C++ integer division truncates toward zero, unlike CELL tile ownership.
            return static_cast<std::int32_t>(std::floor(static_cast<double>(cell) / 4.0) * 4.0);
        }

        LodStaticTile tileOf(CellCoord cell)
        {
            return {{tileBase(cell.x), tileBase(cell.y)}};
        }

        std::string extension(std::string value)
        {
            std::ranges::transform(value, value.begin(), [](unsigned char c) {
                return static_cast<char>(std::tolower(c));
            });
            return value;
        }

        std::string materialPath(std::string_view path)
        {
            auto result = AssetResolver::normalizePath(path);
            if (!result.starts_with("materials\\")) result = "materials\\" + result;
            if (!result.ends_with(".bgsm") && !result.ends_with(".bgem")) result += ".bgsm";
            return result;
        }

        std::string texturePath(std::string_view path)
        {
            auto result = AssetResolver::normalizePath(path);
            if (!result.empty() && !result.starts_with("textures\\")) result = "textures\\" + result;
            return result;
        }

        glm::mat4 referenceTransform(const PlacedReference& reference)
        {
            const auto rotation = [](const float (&r)[3]) {
                const float cx = std::cos(r[0] * 0.5f), sx = std::sin(r[0] * 0.5f);
                const float cy = std::cos(r[1] * 0.5f), sy = std::sin(r[1] * 0.5f);
                const float cz = std::cos(r[2] * 0.5f), sz = std::sin(r[2] * 0.5f);
                return glm::quat(cx * cy * cz + sx * sy * sz,
                    sx * cy * cz - cx * sy * sz,
                    cx * sy * cz + sx * cy * sz,
                    cx * cy * sz - sx * sy * cz);
            };
            glm::quat orientation = rotation(reference.rotation);
            glm::vec3 position(reference.position[0], reference.position[1], reference.position[2]);
            float scale = reference.scale;
            if (reference.hasLocalTransform)
            {
                position += orientation * (glm::vec3(reference.localPosition[0],
                    reference.localPosition[1], reference.localPosition[2]) * scale);
                orientation *= rotation(reference.localRotation);
                scale *= reference.localScale;
            }
            const glm::quat engineRotation(orientation.w, orientation.x, orientation.z, orientation.y);
            return glm::translate(glm::mat4(1.0f), {position.x, position.z, position.y})
                * glm::mat4_cast(engineRotation)
                * glm::scale(glm::mat4(1.0f), glm::vec3(scale));
        }

        struct Vertex
        {
            float position[3], normal[3], uv[2], color[4];
        };
        static_assert(sizeof(Vertex) == 48);

        glm::vec3 toGltf(glm::vec3 engine)
        {
            return {-engine.x / unitsPerMeter, engine.y / unitsPerMeter,
                engine.z / unitsPerMeter};
        }

        ModelImage rgbaImage(const DdsTexture& dds)
        {
            if (!dds.width || !dds.height || dds.width > 16384 || dds.height > 16384)
                throw std::runtime_error("LOD texture exceeds PNG decoder limits");
            const auto format = DirectX::MakeLinear(DirectX::MakeTypelessUNORM(
                static_cast<DXGI_FORMAT>(dds.dxgiFormat)));
            std::size_t rowPitch = 0, slicePitch = 0;
            if (FAILED(DirectX::ComputePitch(format, dds.width, dds.height, rowPitch, slicePitch))
                || slicePitch > dds.pixels.size())
                throw std::runtime_error("LOD texture has a truncated or unsupported DDS payload");
            DirectX::Image source{dds.width, dds.height, format, rowPitch, slicePitch,
                reinterpret_cast<std::uint8_t*>(const_cast<std::byte*>(dds.pixels.data()))};
            DirectX::ScratchImage decoded;
            const DirectX::Image* image = &source;
            if (format != DXGI_FORMAT_R8G8B8A8_UNORM)
            {
                const auto result = DirectX::IsCompressed(format)
                    ? DirectX::Decompress(source, DXGI_FORMAT_R8G8B8A8_UNORM, decoded)
                    : DirectX::Convert(source, DXGI_FORMAT_R8G8B8A8_UNORM,
                        DirectX::TEX_FILTER_DEFAULT, 0.5f, decoded);
                if (FAILED(result))
                    throw std::runtime_error("Cannot decode diffuse DDS format " + std::to_string(dds.dxgiFormat));
                image = decoded.GetImage(0, 0, 0);
            }
            ModelImage result;
            result.width = dds.width;
            result.height = dds.height;
            result.dxgiFormat = 29;
            result.srgb = true;
            result.mipOffsets = {0};
            result.pixels.resize(static_cast<std::size_t>(dds.width) * dds.height * 4);
            for (std::size_t y = 0; y < dds.height; ++y)
                std::memcpy(result.pixels.data() + y * dds.width * 4,
                    image->pixels + y * image->rowPitch, static_cast<std::size_t>(dds.width) * 4);
            return result;
        }

        std::uint64_t hashImage(const ModelImage& image)
        {
            std::uint64_t hash = 14695981039346656037ull;
            const auto add = [&](std::uint8_t value) {
                hash = (hash ^ value) * 1099511628211ull;
            };
            for (int i = 0; i < 4; ++i) add(static_cast<std::uint8_t>(image.width >> (i * 8)));
            for (int i = 0; i < 4; ++i) add(static_cast<std::uint8_t>(image.height >> (i * 8)));
            for (auto value : image.pixels) add(std::to_integer<std::uint8_t>(value));
            return hash;
        }

        std::string pngName(const ModelImage& image)
        {
            std::ostringstream out;
            out << std::hex << std::setw(16) << std::setfill('0') << hashImage(image) << ".png";
            return out.str();
        }

        void publishPng(const std::filesystem::path& path, const ModelImage& image)
        {
            if (std::filesystem::exists(path)) return;
            std::vector<std::byte> bytes;
            auto callback = [](void* context, void* data, int length) {
                auto& target = *static_cast<std::vector<std::byte>*>(context);
                const auto* first = static_cast<std::byte*>(data);
                target.insert(target.end(), first, first + length);
            };
            if (!stbi_write_png_to_func(callback, &bytes,
                    static_cast<int>(image.width), static_cast<int>(image.height), 4,
                    image.pixels.data(), static_cast<int>(image.width * 4)))
                throw std::runtime_error("PNG encoder rejected LOD texture");
            static std::atomic_uint64_t sequence{0};
            std::filesystem::create_directories(path.parent_path());
            auto temporary = path;
            temporary += ".tmp." + std::to_string(sequence.fetch_add(1));
            try
            {
                std::ofstream stream(temporary, std::ios::binary);
                stream.write(reinterpret_cast<const char*>(bytes.data()),
                    static_cast<std::streamsize>(bytes.size()));
                stream.close();
                if (!stream) throw std::runtime_error("Failed to write shared LOD texture");
                if (!std::filesystem::exists(path))
                    std::filesystem::rename(temporary, path);
                else
                    std::filesystem::remove(temporary);
            }
            catch (...)
            {
                std::error_code ignored;
                std::filesystem::remove(temporary, ignored);
                if (!std::filesystem::exists(path)) throw;
            }
        }

        template<class T>
        void append(std::vector<std::byte>& target, const T& source)
        {
            const auto* begin = reinterpret_cast<const std::byte*>(&source);
            target.insert(target.end(), begin, begin + sizeof(source));
        }

        template<class T>
        void appendArray(std::vector<std::byte>& target, const std::vector<T>& source)
        {
            const auto* begin = reinterpret_cast<const std::byte*>(source.data());
            target.insert(target.end(), begin, begin + source.size() * sizeof(T));
        }

        void pad(std::vector<std::byte>& data, std::byte value)
        {
            while (data.size() % 4) data.push_back(value);
        }

        struct GlbBuilder
        {
            std::vector<std::byte> binary;
            std::vector<std::string> views, accessors, materials, meshes, nodes, images, textures;
            std::map<std::uint64_t, int> imageIds;
            const std::filesystem::path& world;
            const std::filesystem::path& output;
            std::set<std::string>* diagnostics = nullptr;

            template<class T>
            int addView(const std::vector<T>& data, int target, int stride = 0)
            {
                pad(binary, std::byte{0});
                const auto offset = binary.size();
                appendArray(binary, data);
                std::ostringstream json;
                json << "{\"buffer\":0,\"byteOffset\":" << offset
                    << ",\"byteLength\":" << data.size() * sizeof(T)
                    << ",\"target\":" << target;
                if (stride) json << ",\"byteStride\":" << stride;
                json << '}';
                views.push_back(json.str());
                return static_cast<int>(views.size() - 1);
            }

            int addAccessor(int view, int offset, int component, int count,
                std::string_view type, std::string_view bounds = {})
            {
                std::ostringstream json;
                json << "{\"bufferView\":" << view << ",\"byteOffset\":" << offset
                    << ",\"componentType\":" << component << ",\"count\":" << count
                    << ",\"type\":\"" << type << "\"";
                if (!bounds.empty()) json << ',' << bounds;
                json << '}';
                accessors.push_back(json.str());
                return static_cast<int>(accessors.size() - 1);
            }

            int texture(const ModelImage& image)
            {
                const auto hash = hashImage(image);
                if (auto found = imageIds.find(hash); found != imageIds.end()) return found->second;
                const auto file = AssetResolver::projectAssetRoot(world) / "lod" / "textures" / pngName(image);
                publishPng(file, image);
                const auto uri = std::filesystem::relative(file, output.parent_path()).generic_string();
                images.push_back("{\"uri\":\"" + uri + "\"}");
                textures.push_back("{\"source\":" + std::to_string(images.size() - 1) + "}");
                const int index = static_cast<int>(textures.size() - 1);
                imageIds.emplace(hash, index);
                return index;
            }

            int material(const ModelSurface& surface, std::optional<ModelImage> diffuse)
            {
                std::ostringstream json;
                json.imbue(std::locale::classic());
                json << std::setprecision(std::numeric_limits<float>::max_digits10);
                json << "{\"pbrMetallicRoughness\":{\"baseColorFactor\":["
                    << surface.baseColor.r << ',' << surface.baseColor.g << ','
                    << surface.baseColor.b << ',' << surface.baseColor.a << "],"
                    << "\"metallicFactor\":" << std::clamp(surface.metallicFactor, 0.0f, 1.0f)
                    << ",\"roughnessFactor\":" << std::clamp(surface.roughnessFactor, 0.0f, 1.0f);
                if (diffuse) json << ",\"baseColorTexture\":{\"index\":" << texture(*diffuse) << '}';
                json << "},\"doubleSided\":" << (surface.doubleSided ? "true" : "false");
                if (surface.alphaBlend) json << ",\"alphaMode\":\"BLEND\"";
                else if (surface.alphaCutoff > 0.0f)
                    json << ",\"alphaMode\":\"MASK\",\"alphaCutoff\":" << surface.alphaCutoff;
                json << '}';
                materials.push_back(json.str());
                return static_cast<int>(materials.size() - 1);
            }

            int mesh(const ModelGeometry& geometry,
                const AssetResolver& resolver, const std::string& name)
            {
                std::vector<std::string> primitives;
                for (const auto& surface : geometry.surfaces)
                {
                    if (surface.vertices.empty() || surface.indices.empty()) continue;
                    if (surface.indices.size() % 3 != 0)
                        throw std::runtime_error("Static mesh has non-triangle indices");
                    std::vector<Vertex> vertices;
                    vertices.reserve(surface.vertices.size());
                    for (const auto& source : surface.vertices)
                    {
                        const auto position = toGltf(source.position);
                        const auto normal = glm::normalize(glm::vec3(-source.normal.x,
                            source.normal.y, source.normal.z));
                        if (!std::isfinite(position.x) || !std::isfinite(position.y)
                            || !std::isfinite(position.z) || !std::isfinite(normal.x)
                            || !std::isfinite(normal.y) || !std::isfinite(normal.z))
                            throw std::runtime_error("Static mesh contains a non-finite vertex");
                        vertices.push_back({{position.x, position.y, position.z},
                            {normal.x, normal.y, normal.z}, {source.uv.x, source.uv.y},
                            {source.color.r, source.color.g, source.color.b, source.color.a}});
                    }
                    for (auto index : surface.indices)
                        if (index >= vertices.size())
                            throw std::runtime_error("Static mesh contains an invalid index");
                    const auto target = std::max<std::size_t>(3,
                        (surface.indices.size() / 12) * 3);
                    const float weights[9]{0.5f, 0.5f, 0.5f, 10.0f, 10.0f,
                        1.0f, 1.0f, 1.0f, 1.0f};
                    std::vector<std::uint32_t> indices(surface.indices.size());
                    float error = 0;
                    const auto count = meshopt_simplifyWithAttributes(indices.data(),
                        surface.indices.data(), surface.indices.size(),
                        &vertices.front().position[0], vertices.size(), sizeof(Vertex),
                        &vertices.front().normal[0], sizeof(Vertex), weights, 9, nullptr,
                        target, 0.01f, 0, &error);
                    if (count == 0 || count % 3 != 0)
                        throw std::runtime_error("Static mesh simplification produced no triangles");
                    indices.resize(count);
                    if (count > target)
                    {
                        const auto message = "Static LOD: " + name + " retained "
                            + std::to_string(count / 3) + " triangles over target "
                            + std::to_string(target / 3) + " at 1% error limit";
                        if (diagnostics) diagnostics->insert(message);
                    }
                    // Remove vertices made unreachable by simplification before serializing.
                    vertices.resize(meshopt_optimizeVertexFetch(vertices.data(), indices.data(),
                        indices.size(), vertices.data(), vertices.size(), sizeof(Vertex)));
                    const auto vertexView = addView(vertices, 34962, sizeof(Vertex));
                    const auto indexView = addView(indices, 34963);
                    glm::vec3 min(std::numeric_limits<float>::max());
                    glm::vec3 max(-std::numeric_limits<float>::max());
                    for (const auto& vertex : vertices)
                    {
                        const glm::vec3 position(vertex.position[0], vertex.position[1], vertex.position[2]);
                        min = glm::min(min, position);
                        max = glm::max(max, position);
                    }
                    std::ostringstream bounds;
                    bounds.imbue(std::locale::classic());
                    bounds << std::setprecision(std::numeric_limits<float>::max_digits10);
                    bounds << "\"min\":[" << min.x << ',' << min.y << ',' << min.z
                        << "],\"max\":[" << max.x << ',' << max.y << ',' << max.z << ']';
                    const int position = addAccessor(vertexView, 0, 5126,
                        static_cast<int>(vertices.size()), "VEC3", bounds.str());
                    const int normal = addAccessor(vertexView, 12, 5126,
                        static_cast<int>(vertices.size()), "VEC3");
                    const int uv = addAccessor(vertexView, 24, 5126,
                        static_cast<int>(vertices.size()), "VEC2");
                    const int color = addAccessor(vertexView, 32, 5126,
                        static_cast<int>(vertices.size()), "VEC4");
                    const int index = addAccessor(indexView, 0, 5125,
                        static_cast<int>(indices.size()), "SCALAR");
                    ModelSurface materialSurface;
                    materialSurface.baseColor = surface.baseColor;
                    materialSurface.alphaBlend = surface.alphaBlend;
                    materialSurface.alphaCutoff = surface.alphaCutoff;
                    materialSurface.doubleSided = surface.doubleSided;
                    materialSurface.metallicFactor = surface.metallicFactor;
                    materialSurface.roughnessFactor = surface.roughnessFactor;
                    auto diffusePath = surface.diffuseTexture;
                    if (!surface.materialPath.empty())
                    {
                        try
                        {
                            const auto loaded = loadMaterial(resolver.extract(materialPath(surface.materialPath)));
                            if (!loaded.diffuseTexture.empty()) diffusePath = loaded.diffuseTexture;
                            materialSurface.baseColor *= loaded.baseColor;
                            materialSurface.alphaCutoff = std::max(surface.alphaCutoff, loaded.alphaCutoff);
                            materialSurface.doubleSided = surface.doubleSided || loaded.twoSided;
                        }
                        catch (const std::exception& error)
                        {
                            if (diagnostics) diagnostics->insert("Static LOD material fallback: "
                                + surface.materialPath + ": " + error.what());
                        }
                    }
                    std::optional<ModelImage> diffuse = surface.diffuseImage;
                    if (!diffuse && !diffusePath.empty())
                    {
                        const auto bytes = resolver.extractTexture(texturePath(diffusePath));
                        diffuse = rgbaImage(loadDdsTexture(bytes));
                    }
                    if (diffuse && diffuse->encoding != ModelImageEncoding::rgba8)
                        diffuse = rgbaImage({diffuse->width, diffuse->height,
                            diffuse->mipLevels, diffuse->dxgiFormat,
                            diffuse->pixels, diffuse->mipOffsets});
                    const int mat = material(materialSurface, std::move(diffuse));
                    std::ostringstream primitive;
                    primitive << "{\"attributes\":{\"POSITION\":" << position
                        << ",\"NORMAL\":" << normal << ",\"TEXCOORD_0\":" << uv
                        << ",\"COLOR_0\":" << color << "},\"indices\":" << index
                        << ",\"material\":" << mat << '}';
                    primitives.push_back(primitive.str());
                }
                if (primitives.empty()) throw std::runtime_error("Static model has no triangle surfaces");
                std::ostringstream json;
                json << "{\"primitives\":[";
                for (std::size_t i = 0; i < primitives.size(); ++i)
                {
                    if (i) json << ',';
                    json << primitives[i];
                }
                json << "]}";
                meshes.push_back(json.str());
                return static_cast<int>(meshes.size() - 1);
            }

            void write() const
            {
                const auto array = [](std::ostringstream& json, std::string_view name,
                    const std::vector<std::string>& values, bool comma = true) {
                    json << '\"' << name << "\":[";
                    for (std::size_t i = 0; i < values.size(); ++i)
                    {
                        if (i) json << ',';
                        json << values[i];
                    }
                    json << ']';
                    if (comma) json << ',';
                };
                std::ostringstream json;
                json.imbue(std::locale::classic());
                json << "{\"asset\":{\"version\":\"2.0\",\"generator\":\"OpenF4\"},"
                    << "\"scene\":0,\"scenes\":[{\"nodes\":[";
                for (std::size_t i = 0; i < nodes.size(); ++i)
                {
                    if (i) json << ',';
                    json << i;
                }
                json << "]}],";
                array(json, "nodes", nodes);
                array(json, "meshes", meshes);
                array(json, "materials", materials);
                array(json, "textures", textures);
                array(json, "images", images);
                array(json, "bufferViews", views);
                array(json, "accessors", accessors);
                json << "\"buffers\":[{\"byteLength\":" << binary.size() << "}]}";
                std::vector<std::byte> jsonBytes;
                for (const auto character : json.str()) jsonBytes.push_back(std::byte(character));
                pad(jsonBytes, std::byte{' '});
                auto binaryBytes = binary;
                pad(binaryBytes, std::byte{0});
                if (jsonBytes.size() + binaryBytes.size() + 28 > std::numeric_limits<std::uint32_t>::max())
                    throw std::runtime_error("Static LOD GLB exceeds 4 GiB");
                std::filesystem::create_directories(output.parent_path());
                std::ofstream stream(output, std::ios::binary | std::ios::trunc);
                if (!stream) throw std::runtime_error("Cannot create static LOD GLB");
                const auto u32 = [&](std::uint32_t value) {
                    stream.write(reinterpret_cast<const char*>(&value), 4);
                };
                u32(0x46546C67u);
                u32(2u);
                u32(static_cast<std::uint32_t>(jsonBytes.size() + binaryBytes.size() + 28));
                u32(static_cast<std::uint32_t>(jsonBytes.size()));
                u32(0x4E4F534Au);
                stream.write(reinterpret_cast<const char*>(jsonBytes.data()),
                    static_cast<std::streamsize>(jsonBytes.size()));
                u32(static_cast<std::uint32_t>(binaryBytes.size()));
                u32(0x004E4942u);
                stream.write(reinterpret_cast<const char*>(binaryBytes.data()),
                    static_cast<std::streamsize>(binaryBytes.size()));
                stream.flush();
                if (!stream) throw std::runtime_error("Unable to finish static LOD GLB");
            }
        };

        ModelGeometry loadStaticModel(const ESMModel& model, const AssetResolver& resolver,
            std::set<std::string>& diagnostics)
        {
            auto path = model.path;
            const auto ext = extension(std::filesystem::path(path).extension().string());
            if (ext == ".nif" || ext == ".bto" || ext == ".btr")
            {
                if (!resolver.contains(path))
                    path = "meshes/" + path;
                return loadNifModel(resolver.extract(path));
            }
            const auto loose = resolver.resolveLoose(path);
            if (!loose) throw std::runtime_error("imported source missing");
            ModelImportResult imported;
            try { imported = importModel(*loose); }
            catch (const std::exception& error)
            {
                throw std::runtime_error(std::string(error.what())
                    + "; after restoring missing resources, use Bake All LOD or --full");
            }
            for (const auto& warning : imported.warnings)
                diagnostics.insert("Static LOD " + model.path + ": " + warning);
            // Animated models are now supported; their animation data is preserved
            // but not used for LOD baking (static snapshots are used instead).
            return std::move(imported.geometry);
        }

        glm::mat4 matrix(const aiMatrix4x4& value)
        {
            return {value.a1, value.b1, value.c1, value.d1,
                value.a2, value.b2, value.c2, value.d2,
                value.a3, value.b3, value.c3, value.d3,
                value.a4, value.b4, value.c4, value.d4};
        }

        ModelSurface readPartSurface(const aiScene& scene, const aiMesh& mesh,
            const glm::mat4& transform, const std::filesystem::path& tilePath)
        {
            if (!mesh.HasPositions() || !mesh.HasNormals())
                throw std::runtime_error("Baked static part lacks positions or normals");
            if (mesh.mMaterialIndex >= scene.mNumMaterials)
                throw std::runtime_error("Baked static part has an invalid material");
            const auto* material = scene.mMaterials[mesh.mMaterialIndex];
            ModelSurface surface;
            aiColor4D color(1.0f);
            if (material->Get(AI_MATKEY_BASE_COLOR, color) != AI_SUCCESS)
                material->Get(AI_MATKEY_COLOR_DIFFUSE, color);
            surface.baseColor = {color.r, color.g, color.b, color.a};
            material->Get(AI_MATKEY_METALLIC_FACTOR, surface.metallicFactor);
            material->Get(AI_MATKEY_ROUGHNESS_FACTOR, surface.roughnessFactor);
            if (!std::isfinite(color.r) || !std::isfinite(color.g) || !std::isfinite(color.b)
                || !std::isfinite(color.a) || !std::isfinite(surface.metallicFactor)
                || !std::isfinite(surface.roughnessFactor))
                throw std::runtime_error("Baked static part contains a non-finite material");
            aiString alphaMode;
            material->Get(AI_MATKEY_GLTF_ALPHAMODE, alphaMode);
            surface.alphaBlend = std::string_view(alphaMode.C_Str()) == "BLEND";
            if (std::string_view(alphaMode.C_Str()) == "MASK")
            {
                surface.alphaCutoff = 0.5f;
                material->Get(AI_MATKEY_GLTF_ALPHACUTOFF, surface.alphaCutoff);
                if (!std::isfinite(surface.alphaCutoff))
                    throw std::runtime_error("Baked static part contains an invalid alpha cutoff");
            }
            int doubleSided = 0;
            material->Get(AI_MATKEY_TWOSIDED, doubleSided);
            surface.doubleSided = doubleSided != 0;
            const auto imageType = material->GetTextureCount(aiTextureType_BASE_COLOR)
                ? aiTextureType_BASE_COLOR : aiTextureType_DIFFUSE;
            aiString imagePath;
            if (material->GetTexture(imageType, 0, &imagePath) == AI_SUCCESS)
            {
                const auto absolute = tilePath.parent_path() / imagePath.C_Str();
                std::ifstream stream(absolute, std::ios::binary | std::ios::ate);
                if (!stream) throw std::runtime_error("Baked static PNG is missing: " + absolute.string());
                const auto length = stream.tellg();
                if (length <= 0 || length > std::numeric_limits<int>::max())
                    throw std::runtime_error("Baked static PNG has invalid size");
                std::vector<std::byte> encoded(static_cast<std::size_t>(length));
                stream.seekg(0);
                stream.read(reinterpret_cast<char*>(encoded.data()), length);
                if (!stream) throw std::runtime_error("Baked static PNG is truncated");
                int width = 0, height = 0, channels = 0;
                if (!stbi_info_from_memory(reinterpret_cast<const stbi_uc*>(encoded.data()),
                        static_cast<int>(encoded.size()), &width, &height, &channels)
                    || width <= 0 || height <= 0 || width > 16384 || height > 16384)
                    throw std::runtime_error("Baked static PNG has invalid dimensions");
                std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> decoded(
                    stbi_load_from_memory(reinterpret_cast<const stbi_uc*>(encoded.data()),
                        static_cast<int>(encoded.size()), &width, &height, &channels, 4),
                    stbi_image_free);
                if (!decoded || width <= 0 || height <= 0
                    || width > 16384 || height > 16384)
                    throw std::runtime_error("Baked static PNG cannot be decoded");
                ModelImage image;
                image.width = static_cast<std::uint32_t>(width);
                image.height = static_cast<std::uint32_t>(height);
                image.dxgiFormat = 29;
                image.srgb = true;
                image.mipOffsets = {0};
                const auto* first = reinterpret_cast<const std::byte*>(decoded.get());
                image.pixels.assign(first, first + static_cast<std::size_t>(width) * height * 4);
                surface.diffuseImage = std::move(image);
            }
            const auto normalMatrix = glm::inverseTranspose(glm::mat3(transform));
            surface.vertices.reserve(mesh.mNumVertices);
            for (unsigned vertex = 0; vertex < mesh.mNumVertices; ++vertex)
            {
                render::SourceVertex value;
                const auto& p = mesh.mVertices[vertex];
                const auto& n = mesh.mNormals[vertex];
                value.position = glm::vec3(transform * glm::vec4(p.x, p.y, p.z, 1));
                value.normal = glm::normalize(normalMatrix * glm::vec3(n.x, n.y, n.z));
                if (mesh.HasTextureCoords(0))
                    value.uv = {mesh.mTextureCoords[0][vertex].x,
                        mesh.mTextureCoords[0][vertex].y};
                if (mesh.HasVertexColors(0))
                {
                    const auto& c = mesh.mColors[0][vertex];
                    value.color = {c.r, c.g, c.b, c.a};
                }
                if (!std::isfinite(value.position.x) || !std::isfinite(value.position.y)
                    || !std::isfinite(value.position.z) || !std::isfinite(value.normal.x)
                    || !std::isfinite(value.normal.y) || !std::isfinite(value.normal.z)
                    || !std::isfinite(value.uv.x) || !std::isfinite(value.uv.y)
                    || !std::isfinite(value.color.r) || !std::isfinite(value.color.g)
                    || !std::isfinite(value.color.b) || !std::isfinite(value.color.a)
                    || glm::any(glm::greaterThan(glm::abs(value.position), glm::vec3(10'000'000))))
                    throw std::runtime_error("Baked static part contains a non-finite or unbounded vertex");
                surface.vertices.push_back(value);
            }
            const bool reverse = glm::determinant(glm::mat3(transform)) > 0.0f;
            for (unsigned faceIndex = 0; faceIndex < mesh.mNumFaces; ++faceIndex)
            {
                const auto& face = mesh.mFaces[faceIndex];
                if (face.mNumIndices != 3) continue;
                if (face.mIndices[0] >= mesh.mNumVertices
                    || face.mIndices[1] >= mesh.mNumVertices
                    || face.mIndices[2] >= mesh.mNumVertices)
                    throw std::runtime_error("Baked static part has invalid indices");
                surface.indices.insert(surface.indices.end(),
                    {face.mIndices[0], face.mIndices[reverse ? 2 : 1],
                        face.mIndices[reverse ? 1 : 2]});
            }
            return surface;
        }
    }

    std::vector<LodStaticPart> loadLodStaticParts(const std::filesystem::path& tilePath)
    {
        Assimp::Importer importer;
        const auto scene = importer.ReadFile(tilePath.string(),
            aiProcess_Triangulate | aiProcess_ValidateDataStructure
                | aiProcess_GenSmoothNormals | aiProcess_FlipUVs);
        if (!scene || !scene->mRootNode)
            throw std::runtime_error("Static LOD GLB is invalid: "
                + std::string(importer.GetErrorString()));
        std::vector<LodStaticPart> parts;
        std::set<std::uint32_t> forms;
        glm::mat4 conversion(1.0f);
        conversion[0][0] = -unitsPerMeter;
        conversion[1][1] = unitsPerMeter;
        conversion[2][2] = unitsPerMeter;
        const auto visit = [&](auto&& self, const aiNode& node,
            const glm::mat4& parent) -> void
        {
            const auto global = parent * matrix(node.mTransformation);
            const std::string_view name(node.mName.C_Str());
            if (name.starts_with("ref_"))
            {
                if (name.size() != 12)
                    throw std::runtime_error("Baked static node has malformed REFR identity");
                std::uint32_t form = 0;
                const auto [end, error] = std::from_chars(
                    name.data() + 4, name.data() + 12, form, 16);
                if (error != std::errc{} || end != name.data() + 12
                    || !form || !forms.insert(form).second)
                    throw std::runtime_error("Baked static node has duplicate or invalid REFR identity");
                LodStaticPart part;
                part.referenceFormId = form;
                for (unsigned i = 0; i < node.mNumMeshes; ++i)
                {
                    const auto index = node.mMeshes[i];
                    if (index >= scene->mNumMeshes)
                        throw std::runtime_error("Baked static node has an invalid mesh index");
                    auto surface = readPartSurface(*scene, *scene->mMeshes[index],
                        conversion * global, tilePath);
                    if (!surface.indices.empty())
                        part.geometry.surfaces.push_back(std::move(surface));
                }
                if (part.geometry.surfaces.empty())
                    throw std::runtime_error("Baked static REFR has no renderable geometry");
                parts.push_back(std::move(part));
            }
            else
                for (unsigned i = 0; i < node.mNumChildren; ++i)
                    self(self, *node.mChildren[i], global);
        };
        visit(visit, *scene->mRootNode, glm::mat4(1.0f));
        if (parts.empty()) throw std::runtime_error("Static LOD GLB has no REFR parts");
        return parts;
    }

    struct StaticLodSource::Impl
    {
        std::filesystem::path world;
        std::optional<std::filesystem::path> data;
        ESMMasterStack stack;
        std::map<std::uint32_t, ESMModel> models;
        std::map<LodStaticTile, std::vector<ESMReference>> byTile;
        mutable std::set<std::string> warnings;
        mutable std::map<std::filesystem::path, std::unique_ptr<AssetResolver>> resolvers;
        mutable std::map<std::uint32_t, std::vector<std::filesystem::path>> dependencies;

        const AssetResolver& resolver(const std::filesystem::path& source) const
        {
            auto& result = resolvers[source];
            if (!result) result = std::make_unique<AssetResolver>(source, data);
            return *result;
        }

        const std::vector<std::filesystem::path>& modelDependencies(const ESMModel& model) const
        {
            if (const auto found = dependencies.find(model.formId); found != dependencies.end())
                return found->second;
            const auto source = model.sourceEsm.empty() ? world : std::filesystem::path(model.sourceEsm);
            const auto& assets = resolver(source);
            std::set<std::filesystem::path> files{source};
            const auto add = [&](const std::string& path) {
                if (path.empty()) return;
                if (const auto loose = assets.resolveLoose(path)) files.insert(*loose);
                // Missing higher-priority overrides must invalidate existing archive/Data products too.
                auto relative = path;
                std::ranges::replace(relative, '\\', '/');
                files.insert(AssetResolver::projectAssetRoot(source) / relative);
                if (data) files.insert(*data / relative);
            };
            auto path = model.path;
            const auto ext = extension(std::filesystem::path(path).extension().string());
            const bool nif = ext == ".nif" || ext == ".btr" || ext == ".bto";
            add(path);
            if (nif && !assets.contains(path)) path = "meshes/" + path;
            add(path);
            if (const auto loose = assets.resolveLoose(path); loose && !nif)
            {
                try
                {
                    for (const auto& dependency : importModel(*loose).sourceFiles)
                        files.insert(dependency);
                }
                // ponytail: a failed parse cannot expose its dependency closure; use a full bake after repairing missing resources.
                catch (const std::exception&) {} // Bake reports unreadable model diagnostics.
            }
            if (nif)
            {
                try
                {
                    const auto geometry = loadNifModel(assets.extract(path));
                    for (const auto& surface : geometry.surfaces)
                    {
                        add(texturePath(surface.diffuseTexture));
                        if (surface.materialPath.empty()) continue;
                        const auto material = materialPath(surface.materialPath);
                        add(material);
                        try { add(texturePath(loadMaterial(assets.extract(material)).diffuseTexture)); }
                        catch (const std::exception&) {} // Bake reports material fallback diagnostics.
                    }
                }
                catch (const std::exception&) {} // Bake reports unreadable model diagnostics.
            }
            return dependencies.emplace(model.formId,
                std::vector<std::filesystem::path>(files.begin(), files.end())).first->second;
        }

        Impl(const std::filesystem::path& input, std::optional<std::filesystem::path> dataDirectory)
            : world(std::filesystem::absolute(input)), data(std::move(dataDirectory)),
              stack(loadEsmMasterStack(world, data))
        {
            for (auto model : stack.models) models.emplace(model.formId, std::move(model));
            for (auto reference : stack.references)
                if (!reference.deleted && !reference.disabled && models.contains(reference.placement.baseObject)
                    && models.at(reference.placement.baseObject).kind == ESMModelKind::staticObject)
                    byTile[tileOf(reference.placement.cell)].push_back(std::move(reference));
        }
    };

    StaticLodSource::StaticLodSource(const std::filesystem::path& world,
        std::optional<std::filesystem::path> data)
        : mImpl(std::make_unique<Impl>(world, std::move(data))) {}
    StaticLodSource::~StaticLodSource() = default;
    StaticLodSource::StaticLodSource(StaticLodSource&&) noexcept = default;
    StaticLodSource& StaticLodSource::operator=(StaticLodSource&&) noexcept = default;

    std::vector<LodStaticTile> StaticLodSource::tiles() const
    {
        std::vector<LodStaticTile> result;
        for (const auto& [tile, refs] : mImpl->byTile) result.push_back(tile);
        return result;
    }

    std::vector<std::uint32_t> StaticLodSource::members(LodStaticTile tile) const
    {
        std::vector<std::uint32_t> result;
        if (auto found = mImpl->byTile.find(tile); found != mImpl->byTile.end())
            for (const auto& ref : found->second)
                result.push_back(ref.placement.formId);
        return result;
    }

    std::vector<std::filesystem::path> StaticLodSource::sourceFiles() const
    {
        std::set<std::filesystem::path> result(mImpl->stack.sourcePaths.begin(), mImpl->stack.sourcePaths.end());
        result.insert(mImpl->world);
        for (const auto& [tile, refs] : mImpl->byTile)
            for (const auto& file : sourceFilesForTile(tile))
                result.insert(file);
        if (mImpl->data && std::filesystem::is_directory(*mImpl->data))
            for (const auto& entry : std::filesystem::directory_iterator(*mImpl->data))
                if (entry.is_regular_file()
                    && extension(entry.path().extension().string()) == ".ba2")
                    result.insert(entry.path());
        return {result.begin(), result.end()};
    }

    std::vector<std::filesystem::path> StaticLodSource::sourceFilesForTile(
        LodStaticTile tile) const
    {
        std::set<std::filesystem::path> result{mImpl->world};
        const auto found = mImpl->byTile.find(tile);
        if (found == mImpl->byTile.end()) return {result.begin(), result.end()};
        std::set<std::uint32_t> seen;
        for (const auto& ref : found->second)
        {
            if (!seen.insert(ref.placement.baseObject).second) continue;
            const auto& model = mImpl->models.at(ref.placement.baseObject);
            for (const auto& file : mImpl->modelDependencies(model)) result.insert(file);
        }
        return {result.begin(), result.end()};
    }

    std::vector<std::string> StaticLodSource::diagnostics() const
    {
        return {mImpl->warnings.begin(), mImpl->warnings.end()};
    }

    bool StaticLodSource::bake(LodStaticTile tile,
        const std::filesystem::path& outputPath) const
    {
        if (tile.origin.x != tileBase(tile.origin.x)
            || tile.origin.y != tileBase(tile.origin.y))
            throw std::invalid_argument("Static LOD tile origin must align to four cells");
        const auto found = mImpl->byTile.find(tile);
        if (found == mImpl->byTile.end())
            throw std::invalid_argument("Static LOD tile has no saved references");
        GlbBuilder glb{{}, {}, {}, {}, {}, {}, {}, {}, {},
            mImpl->world, outputPath, &mImpl->warnings};
        std::map<std::uint32_t, int> meshByBase;
        glm::mat4 gltfToEngine(1.0f), engineToGltf(1.0f);
        gltfToEngine[0][0] = -unitsPerMeter;
        gltfToEngine[1][1] = unitsPerMeter;
        gltfToEngine[2][2] = unitsPerMeter;
        engineToGltf[0][0] = -1.0f / unitsPerMeter;
        engineToGltf[1][1] = 1.0f / unitsPerMeter;
        engineToGltf[2][2] = 1.0f / unitsPerMeter;
        const glm::vec3 origin(tile.origin.x * cellSize, 0.0f,
            tile.origin.y * cellSize);
        for (const auto& ref : found->second)
        {
            const auto& model = mImpl->models.at(ref.placement.baseObject);
            auto mesh = meshByBase.find(model.formId);
            if (mesh == meshByBase.end())
            {
                try
                {
                    const auto source = model.sourceEsm.empty() ? mImpl->world
                        : std::filesystem::path(model.sourceEsm);
                    const auto& resolver = mImpl->resolver(source);
                    const auto geometry = loadStaticModel(model, resolver, mImpl->warnings);
                    if (!geometry.skinBones.empty() || !geometry.animations.empty())
                        mImpl->warnings.insert("Static LOD uses bind-pose snapshot: " + model.path);
                    mesh = meshByBase.emplace(model.formId,
                        glb.mesh(geometry, resolver, model.path)).first;
                }
                catch (const std::exception& error)
                {
                    const auto message = "Static LOD skipped STAT "
                        + std::to_string(model.formId) + " (" + model.path + "): "
                        + error.what();
                    mImpl->warnings.insert(message);
                    meshByBase.emplace(model.formId, -1);
                    mesh = meshByBase.find(model.formId);
                }
            }
            if (mesh->second < 0) continue;
            const auto matrix = engineToGltf
                * glm::translate(glm::mat4(1.0f), -origin)
                * referenceTransform(ref.placement) * gltfToEngine;
            std::ostringstream json;
            json.imbue(std::locale::classic());
            json << std::setprecision(std::numeric_limits<float>::max_digits10);
            json << "{\"name\":\"ref_" << std::uppercase << std::hex
                << std::setw(8) << std::setfill('0') << ref.placement.formId
                << std::dec << "\",\"mesh\":" << mesh->second << ",\"matrix\":[";
            for (int column = 0; column < 4; ++column)
                for (int row = 0; row < 4; ++row)
                {
                    const auto value = matrix[column][row];
                    if (!std::isfinite(value))
                        throw std::runtime_error("Static LOD reference has a non-finite transform");
                    if (column || row) json << ',';
                    json << value;
                }
            json << "]}";
            glb.nodes.push_back(json.str());
        }
        if (glb.nodes.empty())
            return false;
        glb.write();
        return true;
    }
}
