#include "assets/lod_statics.hpp"

#include "assets/asset_resolver.hpp"
#include "assets/dds_texture.hpp"
#include "assets/esm_document.hpp"
#include "assets/model_import.hpp"
#include "assets/nif_model.hpp"

#include <assimp/GltfMaterial.h>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <meshoptimizer.h>

#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
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

        std::uint16_t word(const std::byte* source)
        {
            return static_cast<std::uint16_t>(std::to_integer<unsigned char>(source[0])
                | (std::to_integer<unsigned char>(source[1]) << 8));
        }

        std::array<std::uint8_t, 4> rgb565(std::uint16_t value)
        {
            const auto r = (value >> 11) & 31u, g = (value >> 5) & 63u, b = value & 31u;
            return {static_cast<std::uint8_t>((r << 3) | (r >> 2)),
                static_cast<std::uint8_t>((g << 2) | (g >> 4)),
                static_cast<std::uint8_t>((b << 3) | (b >> 2)), 255};
        }

        ModelImage rgbaImage(const DdsTexture& dds)
        {
            if (dds.width > 16384 || dds.height > 16384
                || static_cast<std::uint64_t>(dds.width) * dds.height > 16384ull * 16384ull)
                throw std::runtime_error("LOD texture exceeds PNG decoder limits");
            ModelImage result;
            result.width = dds.width;
            result.height = dds.height;
            result.pixels.resize(static_cast<std::size_t>(dds.width) * dds.height * 4);
            if (dds.dxgiFormat == 28 || dds.dxgiFormat == 29 || dds.dxgiFormat == 87)
            {
                std::copy_n(dds.pixels.begin(), result.pixels.size(), result.pixels.begin());
                if (dds.dxgiFormat == 87)
                    for (std::size_t i = 0; i < result.pixels.size(); i += 4)
                        std::swap(result.pixels[i], result.pixels[i + 2]);
                return result;
            }
            const bool bc1 = dds.dxgiFormat == 70 || dds.dxgiFormat == 71 || dds.dxgiFormat == 72;
            const bool bc2 = dds.dxgiFormat >= 73 && dds.dxgiFormat <= 75;
            const bool bc3 = dds.dxgiFormat >= 76 && dds.dxgiFormat <= 78;
            if (!bc1 && !bc2 && !bc3)
                throw std::runtime_error("Unsupported diffuse DDS compression " + std::to_string(dds.dxgiFormat));
            const std::size_t blockSize = bc1 ? 8 : 16;
            for (std::uint32_t by = 0; by < (dds.height + 3) / 4; ++by)
                for (std::uint32_t bx = 0; bx < (dds.width + 3) / 4; ++bx)
                {
                    const auto* block = dds.pixels.data()
                        + (static_cast<std::size_t>(by) * ((dds.width + 3) / 4) + bx) * blockSize;
                    const auto* colorBlock = block + (bc1 ? 0 : 8);
                    const auto c0 = word(colorBlock), c1 = word(colorBlock + 2);
                    std::array<std::array<std::uint8_t, 4>, 4> palette{
                        rgb565(c0), rgb565(c1), {}, {}};
                    for (int channel = 0; channel < 3; ++channel)
                    {
                        if (c0 > c1 || bc2 || bc3)
                        {
                            palette[2][channel] = static_cast<std::uint8_t>(
                                (2 * palette[0][channel] + palette[1][channel]) / 3);
                            palette[3][channel] = static_cast<std::uint8_t>(
                                (palette[0][channel] + 2 * palette[1][channel]) / 3);
                        }
                        else
                            palette[2][channel] = static_cast<std::uint8_t>(
                                (palette[0][channel] + palette[1][channel]) / 2);
                    }
                    palette[2][3] = 255;
                    palette[3][3] = (c0 > c1 || bc2 || bc3) ? 255 : 0;
                    std::uint32_t colorBits = 0;
                    std::memcpy(&colorBits, colorBlock + 4, 4);
                    std::array<std::uint8_t, 8> alphas{};
                    std::uint64_t alphaBits = 0;
                    if (bc2)
                        std::memcpy(&alphaBits, block, 8);
                    if (bc3)
                    {
                        alphas[0] = std::to_integer<std::uint8_t>(block[0]);
                        alphas[1] = std::to_integer<std::uint8_t>(block[1]);
                        for (int i = 2; i < 8; ++i)
                            alphas[i] = static_cast<std::uint8_t>(
                                alphas[0] > alphas[1]
                                ? ((8 - i) * alphas[0] + (i - 1) * alphas[1]) / 7
                                : (i < 6 ? ((6 - i) * alphas[0] + (i - 1) * alphas[1]) / 5
                                    : (i == 6 ? 0 : 255)));
                        for (int i = 0; i < 6; ++i)
                            alphaBits |= static_cast<std::uint64_t>(
                                std::to_integer<std::uint8_t>(block[2 + i])) << (8 * i);
                    }
                    for (std::uint32_t y = 0; y < 4; ++y)
                        for (std::uint32_t x = 0; x < 4; ++x)
                        {
                            const auto px = bx * 4 + x, py = by * 4 + y;
                            if (px >= dds.width || py >= dds.height) continue;
                            const auto k = y * 4 + x;
                            auto rgba = palette[(colorBits >> (2 * k)) & 3u];
                            if (bc2) rgba[3] = static_cast<std::uint8_t>(
                                ((alphaBits >> (4 * k)) & 15u) * 17u);
                            if (bc3) rgba[3] = alphas[(alphaBits >> (3 * k)) & 7u];
                            const auto dst = (static_cast<std::size_t>(py) * dds.width + px) * 4;
                            for (int channel = 0; channel < 4; ++channel)
                                result.pixels[dst + channel] = std::byte(rgba[channel]);
                        }
                }
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
            std::vector<std::string>* diagnostics = nullptr;

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
                        std::cerr << message << '\n';
                        if (diagnostics) diagnostics->push_back(message);
                    }
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
                    std::optional<ModelImage> diffuse = surface.diffuseImage;
                    if (!diffuse && !surface.diffuseTexture.empty())
                    {
                        const auto bytes = resolver.extractTexture(surface.diffuseTexture);
                        diffuse = rgbaImage(loadDdsTexture(bytes));
                    }
                    if (diffuse && diffuse->encoding != ModelImageEncoding::rgba8)
                        diffuse = rgbaImage({diffuse->width, diffuse->height,
                            diffuse->mipLevels, diffuse->dxgiFormat,
                            diffuse->pixels, diffuse->mipOffsets});
                    const int mat = material(surface, std::move(diffuse));
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

        ModelGeometry loadStaticModel(const ESMModel& model, const AssetResolver& resolver)
        {
            auto path = model.path;
            const auto ext = extension(std::filesystem::path(path).extension().string());
            if (ext == ".nif" || ext == ".bto" || ext == ".btr")
            {
                if (!resolver.contains(path))
                    path = "meshes/" + path;
                auto geometry = loadNifModel(resolver.extract(path));
                if (!geometry.skinBones.empty() || !geometry.attachmentTargets.empty())
                    throw std::runtime_error("animated or skinned NIF");
                return geometry;
            }
            const auto loose = resolver.resolveLoose(path);
            if (!loose) throw std::runtime_error("imported source missing");
            auto imported = importModel(*loose);
            if (std::ranges::any_of(imported.warnings, [](const auto& warning) {
                return warning.find("Animations") != std::string::npos;
            }))
                throw std::runtime_error("animated imported model");
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
            aiString alphaMode;
            material->Get(AI_MATKEY_GLTF_ALPHAMODE, alphaMode);
            surface.alphaBlend = std::string_view(alphaMode.C_Str()) == "BLEND";
            if (std::string_view(alphaMode.C_Str()) == "MASK")
            {
                surface.alphaCutoff = 0.5f;
                material->Get(AI_MATKEY_GLTF_ALPHACUTOFF, surface.alphaCutoff);
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
                    || !forms.insert(form).second)
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
        mutable std::vector<std::string> warnings;

        Impl(const std::filesystem::path& input, std::optional<std::filesystem::path> dataDirectory)
            : world(std::filesystem::absolute(input)), data(std::move(dataDirectory)),
              stack(loadEsmMasterStack(world, data))
        {
            for (auto model : stack.models) models.emplace(model.formId, std::move(model));
            for (auto reference : stack.references)
                if (!reference.deleted && models.contains(reference.placement.baseObject))
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
        std::set<std::filesystem::path> result{mImpl->world, mImpl->stack.basePath};
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
            const auto source = model.sourceEsm.empty() ? mImpl->world
                : std::filesystem::path(model.sourceEsm);
            result.insert(source);
            AssetResolver resolver(source, mImpl->data);
            const auto loose = resolver.resolveLoose(model.path);
            if (!loose) continue;
            result.insert(*loose);
            const auto project = AssetResolver::projectAssetRoot(source);
            const auto relative = loose->lexically_relative(project);
            if (!relative.empty() && !relative.is_absolute()
                && *relative.begin() != "..")
                for (const auto& entry : std::filesystem::recursive_directory_iterator(
                    loose->parent_path(), std::filesystem::directory_options::skip_permission_denied))
                    if (entry.is_regular_file()) result.insert(entry.path());
        }
        return {result.begin(), result.end()};
    }

    std::vector<std::string> StaticLodSource::diagnostics() const
    {
        return mImpl->warnings;
    }

    void StaticLodSource::bake(LodStaticTile tile,
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
                    AssetResolver resolver(source, mImpl->data);
                    const auto geometry = loadStaticModel(model, resolver);
                    mesh = meshByBase.emplace(model.formId,
                        glb.mesh(geometry, resolver, model.path)).first;
                }
                catch (const std::exception& error)
                {
                    const auto message = "Static LOD skipped STAT "
                        + std::to_string(model.formId) + " (" + model.path + "): "
                        + error.what();
                    std::cerr << message << '\n';
                    mImpl->warnings.push_back(message);
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
            throw std::runtime_error("Static LOD tile contains no resolvable static models");
        glb.write();
    }
}
