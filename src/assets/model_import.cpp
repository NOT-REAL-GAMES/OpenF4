#include "assets/model_import.hpp"

#include "assets/asset_resolver.hpp"
#include "assets/dds_texture.hpp"

#include <assimp/DefaultIOSystem.h>
#include <assimp/GltfMaterial.h>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <algorithm>
#include <functional>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>

#include <glm/gtc/matrix_inverse.hpp>

namespace openf4::assets
{
    namespace
    {
        std::string pathText(const std::filesystem::path& path)
        {
            const auto utf8 = path.u8string();
            return {reinterpret_cast<const char*>(utf8.data()), utf8.size()};
        }

        std::string decodeUri(std::string value)
        {
            // glTF paths are URI-escaped, unlike ordinary filesystem paths.
            std::string decoded;
            for (std::size_t index = 0; index < value.size(); ++index)
            {
                if (value[index] == '%' && index + 2 < value.size())
                {
                    const auto hex = [](char digit) -> int {
                        if (digit >= '0' && digit <= '9') return digit - '0';
                        if (digit >= 'a' && digit <= 'f') return digit - 'a' + 10;
                        if (digit >= 'A' && digit <= 'F') return digit - 'A' + 10;
                        return -1;
                    };
                    const int first = hex(value[index + 1]);
                    const int second = hex(value[index + 2]);
                    if (first >= 0 && second >= 0)
                    {
                        const char character = static_cast<char>(first * 16 + second);
                        if (character == '\0')
                            throw std::runtime_error("Model resource URI contains a null byte");
                        decoded += character;
                        index += 2;
                        continue;
                    }
                }
                decoded += value[index] == '\\' ? '/' : value[index];
            }
            return decoded;
        }

        class TrackingIo final : public Assimp::IOSystem
        {
        public:
            explicit TrackingIo(std::set<std::filesystem::path>& files) : mFiles(files) {}
            bool Exists(const char* path) const override { return mDefault.Exists(path); }
            char getOsSeparator() const override { return mDefault.getOsSeparator(); }
            Assimp::IOStream* Open(const char* path, const char* mode = "rb") override
            {
                auto* stream = mDefault.Open(path, mode);
                if (stream)
                    mFiles.insert(std::filesystem::weakly_canonical(std::filesystem::u8path(path)));
                return stream;
            }
            void Close(Assimp::IOStream* stream) override { mDefault.Close(stream); }
            bool ComparePaths(const char* first, const char* second) const override
            {
                return mDefault.ComparePaths(first, second);
            }
        private:
            Assimp::DefaultIOSystem mDefault;
            std::set<std::filesystem::path>& mFiles;
        };

        std::vector<std::byte> readBytes(const std::filesystem::path& path)
        {
            std::ifstream file(path, std::ios::binary | std::ios::ate);
            if (!file)
                throw std::runtime_error("Unable to read model resource: " + path.string());
            const auto size = file.tellg();
            if (size < 0 || static_cast<std::uint64_t>(size) > std::numeric_limits<int>::max())
                throw std::runtime_error("Model image is too large: " + path.string());
            std::vector<std::byte> bytes(static_cast<std::size_t>(size));
            file.seekg(0);
            if (!file.read(reinterpret_cast<char*>(bytes.data()), size))
                throw std::runtime_error("Truncated model resource: " + path.string());
            return bytes;
        }

        ModelImage decodeImage(std::span<const std::byte> bytes, bool srgb)
        {
            ModelImage result;
            result.srgb = srgb;
            if (bytes.size() >= 4 && std::memcmp(bytes.data(), "DDS ", 4) == 0)
            {
                auto dds = loadDdsTexture(bytes);
                result.encoding = ModelImageEncoding::dds;
                result.width = dds.width;
                result.height = dds.height;
                result.mipLevels = dds.mipLevels;
                result.dxgiFormat = dds.dxgiFormat;
                result.pixels = std::move(dds.pixels);
                result.mipOffsets = std::move(dds.mipOffsets);
                return result;
            }
            if (bytes.size() > std::numeric_limits<int>::max())
                throw std::runtime_error("Model image exceeds decoder size limit");
            int width = 0, height = 0, channels = 0;
            if (!stbi_info_from_memory(reinterpret_cast<const stbi_uc*>(bytes.data()),
                    static_cast<int>(bytes.size()), &width, &height, &channels)
                || width <= 0 || height <= 0 || width > 16384 || height > 16384)
                throw std::runtime_error("Unsupported, malformed, or oversized model image");
            std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> pixels(
                stbi_load_from_memory(reinterpret_cast<const stbi_uc*>(bytes.data()),
                    static_cast<int>(bytes.size()), &width, &height, &channels, 4), stbi_image_free);
            if (!pixels)
                throw std::runtime_error("Unable to decode model image");
            result.width = static_cast<std::uint32_t>(width);
            result.height = static_cast<std::uint32_t>(height);
            result.dxgiFormat = srgb ? 29 : 28;
            result.mipOffsets = {0};
            const auto* begin = reinterpret_cast<const std::byte*>(pixels.get());
            result.pixels.assign(begin, begin + static_cast<std::size_t>(width) * height * 4);
            return result;
        }

        std::optional<ModelImage> materialImage(const aiScene& scene, const aiMaterial& material,
            aiTextureType type, const std::filesystem::path& source, bool srgb,
            std::string& texturePath, std::set<std::filesystem::path>& dependencies,
            std::vector<std::string>& warnings)
        {
            aiString texture;
            if (material.GetTexture(type, 0, &texture) != AI_SUCCESS)
                return std::nullopt;
            texturePath = texture.C_Str();
            if (texturePath.find("://") != std::string::npos)
                throw std::runtime_error("Network model resources are unsupported: " + texturePath);
            try
            {
                if (const aiTexture* embedded = scene.GetEmbeddedTexture(texture.C_Str()))
                {
                    if (embedded->mHeight == 0)
                        return decodeImage({reinterpret_cast<const std::byte*>(embedded->pcData),
                            embedded->mWidth}, srgb);
                    if (embedded->mWidth > 16384 || embedded->mHeight > 16384)
                        throw std::runtime_error("Embedded image dimensions exceed 16384");
                    ModelImage result;
                    result.srgb = srgb;
                    result.dxgiFormat = srgb ? 29 : 28;
                    result.width = embedded->mWidth;
                    result.height = embedded->mHeight;
                    result.mipOffsets = {0};
                    result.pixels.resize(static_cast<std::size_t>(result.width) * result.height * 4);
                    for (std::size_t index = 0; index < result.pixels.size() / 4; ++index)
                    {
                        const auto& texel = embedded->pcData[index];
                        result.pixels[index * 4] = std::byte(texel.r);
                        result.pixels[index * 4 + 1] = std::byte(texel.g);
                        result.pixels[index * 4 + 2] = std::byte(texel.b);
                        result.pixels[index * 4 + 3] = std::byte(texel.a);
                    }
                    return result;
                }
                const auto path = std::filesystem::weakly_canonical(source.parent_path()
                    / std::filesystem::u8path(decodeUri(texturePath)));
                auto bytes = readBytes(path);
                dependencies.insert(path);
                return decodeImage(bytes, srgb);
            }
            catch (const std::exception& error)
            {
                warnings.push_back("Texture " + texturePath + ": " + error.what());
                return std::nullopt;
            }
        }

        glm::mat4 matrix(const aiMatrix4x4& value)
        {
            return {value.a1, value.b1, value.c1, value.d1,
                value.a2, value.b2, value.c2, value.d2,
                value.a3, value.b3, value.c3, value.d3,
                value.a4, value.b4, value.c4, value.d4};
        }

        // Convert Assimp's aiVector3D to glm::vec3
        glm::vec3 convert(const aiVector3t<float>& value)
        {
            return {value.x, value.y, value.z};
        }

        // Convert Assimp's aiQuaternion to glm::quat
        glm::quat convert(const aiQuaterniont<float>& value)
        {
            return {value.w, value.x, value.y, value.z};
        }

        // Extract animation data from an Assimp scene
        void extractAnimations(const aiScene& scene, ModelGeometry& result)
        {
            if (!scene.HasAnimations())
                return;

            // Build a mapping from node names to their global transforms
            std::unordered_map<std::string, glm::mat4> nodeGlobalTransforms;

            // First pass: compute global transforms for all nodes
            std::function<void(const aiNode*, const glm::mat4&)> computeGlobalTransforms =
                [&](const aiNode* node, const glm::mat4& parentTransform) {
                    glm::mat4 global = parentTransform * matrix(node->mTransformation);
                    if (node->mName.C_Str())
                        nodeGlobalTransforms[node->mName.C_Str()] = global;
                    for (unsigned i = 0; i < node->mNumChildren; ++i)
                        computeGlobalTransforms(node->mChildren[i], global);
                };
            computeGlobalTransforms(scene.mRootNode, glm::mat4(1.0f));

            // Process each animation
            for (unsigned animIdx = 0; animIdx < scene.mNumAnimations; ++animIdx)
            {
                const aiAnimation* aiAnim = scene.mAnimations[animIdx];
                ModelAnimationClip clip;
                clip.name = aiAnim->mName.C_Str();
                
                // Determine duration and ticks per second
                clip.duration = static_cast<float>(aiAnim->mDuration);
                clip.ticksPerSecond = aiAnim->mTicksPerSecond > 0 ? static_cast<float>(aiAnim->mTicksPerSecond) : 24.0f;

                // Process each animation channel (one per bone/node)
                for (unsigned chanIdx = 0; chanIdx < aiAnim->mNumChannels; ++chanIdx)
                {
                    const aiNodeAnim* aiChannel = aiAnim->mChannels[chanIdx];
                    ModelAnimationChannel channel;
                    channel.boneName = aiChannel->mNodeName.C_Str();

                    // Collect all unique keyframe times
                    std::vector<float> allTimes;
                    for (unsigned i = 0; i < aiChannel->mNumPositionKeys; ++i)
                        allTimes.push_back(static_cast<float>(static_cast<double>(aiChannel->mPositionKeys[i].mTime)));
                    for (unsigned i = 0; i < aiChannel->mNumRotationKeys; ++i)
                        allTimes.push_back(static_cast<float>(static_cast<double>(aiChannel->mRotationKeys[i].mTime)));
                    for (unsigned i = 0; i < aiChannel->mNumScalingKeys; ++i)
                        allTimes.push_back(static_cast<float>(static_cast<double>(aiChannel->mScalingKeys[i].mTime)));
                    
                    std::sort(allTimes.begin(), allTimes.end());
                    allTimes.erase(std::unique(allTimes.begin(), allTimes.end()), allTimes.end());

                    channel.times = allTimes;
                    channel.positions.resize(allTimes.size());
                    channel.rotations.resize(allTimes.size());
                    channel.scales.resize(allTimes.size());

                    // For each time, interpolate the position, rotation, and scale
                    for (std::size_t i = 0; i < allTimes.size(); ++i)
                    {
                        float time = allTimes[i];
                        
                        // Interpolate position
                        if (aiChannel->mNumPositionKeys > 0)
                        {
                            unsigned idx = 0;
                            for (unsigned j = 0; j < aiChannel->mNumPositionKeys - 1; ++j)
                            {
                                if (aiChannel->mPositionKeys[j].mTime <= time && 
                                    aiChannel->mPositionKeys[j + 1].mTime >= time)
                                {
                                    idx = j;
                                    break;
                                }
                            }
                            if (aiChannel->mNumPositionKeys > 0)
                            {
                                float t = 0.0f;
                                if (aiChannel->mNumPositionKeys > 1 && idx + 1 < aiChannel->mNumPositionKeys)
                                {
                                    float t1 = static_cast<float>(aiChannel->mPositionKeys[idx].mTime);
                                    float t2 = static_cast<float>(aiChannel->mPositionKeys[idx + 1].mTime);
                                    if (t2 > t1)
                                        t = (time - t1) / (t2 - t1);
                                }
                                aiVector3t<float> pos = aiChannel->mPositionKeys[idx].mValue;
                                if (aiChannel->mNumPositionKeys > 1 && idx + 1 < aiChannel->mNumPositionKeys)
                                {
                                    aiVector3t<float> pos2 = aiChannel->mPositionKeys[idx + 1].mValue;
                                    pos = pos + (pos2 - pos) * t;
                                }
                                channel.positions[i] = convert(pos);
                            }
                        }

                        // Interpolate rotation (use SLERP for quaternions)
                        if (aiChannel->mNumRotationKeys > 0)
                        {
                            unsigned idx = 0;
                            for (unsigned j = 0; j < aiChannel->mNumRotationKeys - 1; ++j)
                            {
                                if (aiChannel->mRotationKeys[j].mTime <= time && 
                                    aiChannel->mRotationKeys[j + 1].mTime >= time)
                                {
                                    idx = j;
                                    break;
                                }
                            }
                            if (aiChannel->mNumRotationKeys > 0)
                            {
                                float t = 0.0f;
                                if (aiChannel->mNumRotationKeys > 1 && idx + 1 < aiChannel->mNumRotationKeys)
                                {
                                    float t1 = static_cast<float>(aiChannel->mRotationKeys[idx].mTime);
                                    float t2 = static_cast<float>(aiChannel->mRotationKeys[idx + 1].mTime);
                                    if (t2 > t1)
                                        t = (time - t1) / (t2 - t1);
                                }
                                aiQuaterniont<float> rot = aiChannel->mRotationKeys[idx].mValue;
                                if (aiChannel->mNumRotationKeys > 1 && idx + 1 < aiChannel->mNumRotationKeys)
                                {
                                    aiQuaterniont<float> rot2 = aiChannel->mRotationKeys[idx + 1].mValue;
                                    aiQuaterniont<float>::Interpolate(rot, rot, rot2, t);
                                }
                                channel.rotations[i] = convert(rot);
                            }
                        }

                        // Interpolate scale
                        if (aiChannel->mNumScalingKeys > 0)
                        {
                            unsigned idx = 0;
                            for (unsigned j = 0; j < aiChannel->mNumScalingKeys - 1; ++j)
                            {
                                if (aiChannel->mScalingKeys[j].mTime <= time && 
                                    aiChannel->mScalingKeys[j + 1].mTime >= time)
                                {
                                    idx = j;
                                    break;
                                }
                            }
                            if (aiChannel->mNumScalingKeys > 0)
                            {
                                float t = 0.0f;
                                if (aiChannel->mNumScalingKeys > 1 && idx + 1 < aiChannel->mNumScalingKeys)
                                {
                                    float t1 = static_cast<float>(aiChannel->mScalingKeys[idx].mTime);
                                    float t2 = static_cast<float>(aiChannel->mScalingKeys[idx + 1].mTime);
                                    if (t2 > t1)
                                        t = (time - t1) / (t2 - t1);
                                }
                                aiVector3t<float> scale = aiChannel->mScalingKeys[idx].mValue;
                                if (aiChannel->mNumScalingKeys > 1 && idx + 1 < aiChannel->mNumScalingKeys)
                                {
                                    aiVector3t<float> scale2 = aiChannel->mScalingKeys[idx + 1].mValue;
                                    scale = scale + (scale2 - scale) * t;
                                }
                                channel.scales[i] = convert(scale);
                            }
                        }
                    }

                    // Store channel index for this bone
                    clip.boneIndex[channel.boneName] = clip.channels.size();
                    clip.channels.push_back(std::move(channel));
                }

                if (!clip.channels.empty())
                {
                    clip.name = clip.name.empty() ? "Animation_" + std::to_string(result.animations.size()) : clip.name;
                    result.animations.push_back(std::move(clip));
                }
            }
        }

        bool finite(const glm::vec3& value)
        {
            return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
        }

        glm::vec3 direction(const glm::vec3& value)
        {
            const float length = glm::length(value);
            if (!finite(value) || !std::isfinite(length) || length < 1e-10f)
                throw std::runtime_error("Imported model contains an invalid normal or tangent");
            return value / length;
        }

        ModelImportResult import(const std::filesystem::path& input, ModelImportOptions options,
            std::set<std::filesystem::path>& dependencies)
        {
            const auto source = std::filesystem::canonical(input);
            ModelImportResult result;
            result.sourcePath = source;
            result.format = source.extension().string();
            Assimp::Importer importer;
            importer.SetIOHandler(new TrackingIo(dependencies));
            unsigned flags = aiProcess_Triangulate | aiProcess_JoinIdenticalVertices
                | aiProcess_ValidateDataStructure | aiProcess_SortByPType | aiProcess_FlipUVs;
            if (options.generateNormals)
                flags |= aiProcess_GenSmoothNormals;
            if (options.generateTangents)
                flags |= aiProcess_CalcTangentSpace;
            const aiScene* scene = importer.ReadFile(pathText(source), flags);
            if (!scene || !scene->mRootNode || (scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE))
                throw std::runtime_error("Model import failed: " + std::string(importer.GetErrorString()));
            extractAnimations(*scene, result.geometry);

            std::vector<ModelSurface> materials(scene->mNumMaterials);
            for (unsigned index = 0; index < scene->mNumMaterials; ++index)
            {
                const auto& material = *scene->mMaterials[index];
                auto& surface = materials[index];
                aiColor4D color(1.0f);
                if (material.Get(AI_MATKEY_BASE_COLOR, color) != AI_SUCCESS)
                    material.Get(AI_MATKEY_COLOR_DIFFUSE, color);
                surface.baseColor = {color.r, color.g, color.b, color.a};
                material.Get(AI_MATKEY_METALLIC_FACTOR, surface.metallicFactor);
                material.Get(AI_MATKEY_ROUGHNESS_FACTOR, surface.roughnessFactor);
                aiString alphaMode;
                material.Get(AI_MATKEY_GLTF_ALPHAMODE, alphaMode);
                if (std::string_view(alphaMode.C_Str()) == "MASK")
                {
                    surface.alphaCutoff = 0.5f;
                    material.Get(AI_MATKEY_GLTF_ALPHACUTOFF, surface.alphaCutoff);
                }
                surface.alphaBlend = std::string_view(alphaMode.C_Str()) == "BLEND";
                if (!finite(glm::vec3(surface.baseColor)) || !std::isfinite(surface.baseColor.a)
                    || !std::isfinite(surface.metallicFactor) || !std::isfinite(surface.roughnessFactor)
                    || !std::isfinite(surface.alphaCutoff))
                    throw std::runtime_error("Imported model contains non-finite material properties");
                int doubleSided = 0;
                material.Get(AI_MATKEY_TWOSIDED, doubleSided);
                surface.doubleSided = doubleSided != 0;
                const auto diffuseType = material.GetTextureCount(aiTextureType_BASE_COLOR)
                    ? aiTextureType_BASE_COLOR : aiTextureType_DIFFUSE;
                surface.diffuseImage = materialImage(*scene, material, diffuseType, source, true,
                    surface.diffuseTexture, dependencies, result.warnings);
                if (!surface.diffuseImage)
                {
                    ModelImage white;
                    white.width = white.height = 1;
                    white.srgb = true;
                    white.dxgiFormat = 29;
                    white.mipOffsets = {0};
                    white.pixels.assign(4, std::byte{255});
                    surface.diffuseImage = std::move(white);
                }
                surface.normalImage = materialImage(*scene, material, aiTextureType_NORMALS, source, false,
                    surface.normalTexture, dependencies, result.warnings);
                std::string metallicTexture;
                surface.metallicRoughnessImage = materialImage(*scene, material,
                    aiTextureType_GLTF_METALLIC_ROUGHNESS, source, false,
                    metallicTexture, dependencies, result.warnings);
                // Preserve every referenced texture, including channels the first renderer pass ignores.
                for (unsigned type = aiTextureType_NONE + 1; type <= AI_TEXTURE_TYPE_MAX; ++type)
                    for (unsigned slot = 0; slot < material.GetTextureCount(static_cast<aiTextureType>(type)); ++slot)
                    {
                        aiString path;
                        if (material.GetTexture(static_cast<aiTextureType>(type), slot, &path) != AI_SUCCESS
                            || scene->GetEmbeddedTexture(path.C_Str()))
                            continue;
                        if (std::string_view(path.C_Str()).find("://") != std::string::npos)
                            throw std::runtime_error("Network model resources are unsupported");
                        const auto resource = std::filesystem::weakly_canonical(source.parent_path()
                            / std::filesystem::u8path(decodeUri(path.C_Str())));
                        result.sourceFiles.push_back(resource);
                        if (std::filesystem::is_regular_file(resource))
                            dependencies.insert(resource);
                    }
            }

            glm::mat4 conversion(gltfBethesdaUnitsPerMeter);
            conversion[0][0] = -gltfBethesdaUnitsPerMeter;
            conversion[3][3] = 1.0f;
            const auto visit = [&](auto&& self, const aiNode& node, const glm::mat4& parent) -> void {
                const glm::mat4 global = parent * matrix(node.mTransformation);
                const glm::mat4 transform = conversion * global;
                const float determinant = glm::determinant(glm::mat3(global));
                if (!std::isfinite(determinant) || std::abs(determinant) < 1e-12f)
                    throw std::runtime_error("Imported model contains a singular node transform");
                const auto normalMatrix = glm::inverseTranspose(glm::mat3(transform));
                for (unsigned meshIndex = 0; meshIndex < node.mNumMeshes; ++meshIndex)
                {
                    const auto& mesh = *scene->mMeshes[node.mMeshes[meshIndex]];
                    if (!(mesh.mPrimitiveTypes & aiPrimitiveType_TRIANGLE))
                        continue;
                    if (!mesh.HasPositions() || !mesh.HasNormals())
                        throw std::runtime_error("Imported triangle mesh has no positions or normals");
                    auto surface = materials.at(mesh.mMaterialIndex);
                    surface.vertices.reserve(mesh.mNumVertices);
                    if (mesh.HasTangentsAndBitangents())
                        surface.tangents.reserve(mesh.mNumVertices);
                    for (unsigned index = 0; index < mesh.mNumVertices; ++index)
                    {
                        const auto& position = mesh.mVertices[index];
                        const auto& normal = mesh.mNormals[index];
                        render::SourceVertex vertex;
                        vertex.position = glm::vec3(transform * glm::vec4(position.x, position.y, position.z, 1));
                        vertex.normal = direction(normalMatrix * glm::vec3(normal.x, normal.y, normal.z));
                        if (!finite(vertex.position)
                            || glm::any(glm::greaterThan(glm::abs(vertex.position), glm::vec3(10'000'000.0f))))
                            throw std::runtime_error("Imported model contains a non-finite or unbounded vertex");
                        if (mesh.HasTextureCoords(0))
                            vertex.uv = {mesh.mTextureCoords[0][index].x, mesh.mTextureCoords[0][index].y};
                        if (!std::isfinite(vertex.uv.x) || !std::isfinite(vertex.uv.y))
                            throw std::runtime_error("Imported model contains non-finite texture coordinates");
                        if (mesh.HasVertexColors(0))
                        {
                            const auto& color = mesh.mColors[0][index];
                            vertex.color = {color.r, color.g, color.b, color.a};
                            if (!finite(glm::vec3(vertex.color)) || !std::isfinite(vertex.color.a))
                                throw std::runtime_error("Imported model contains non-finite vertex colors");
                        }
                        surface.vertices.push_back(vertex);
                        if (mesh.HasTangentsAndBitangents())
                        {
                            const auto& t = mesh.mTangents[index];
                            const auto& b = mesh.mBitangents[index];
                            const auto tangent = direction(glm::mat3(transform) * glm::vec3(t.x, t.y, t.z));
                            const auto bitangent = glm::mat3(transform) * glm::vec3(b.x, b.y, b.z);
                            surface.tangents.emplace_back(tangent,
                                glm::dot(glm::cross(vertex.normal, tangent), bitangent) < 0 ? -1.0f : 1.0f);
                        }
                    }
                    for (unsigned faceIndex = 0; faceIndex < mesh.mNumFaces; ++faceIndex)
                    {
                        const auto& face = mesh.mFaces[faceIndex];
                        if (face.mNumIndices != 3)
                            continue;
                        if (face.mIndices[0] >= mesh.mNumVertices || face.mIndices[1] >= mesh.mNumVertices
                            || face.mIndices[2] >= mesh.mNumVertices)
                            throw std::runtime_error("Imported model contains an out-of-range triangle index");
                        // The renderer supplies the second reflection. Only a node's negative
                        // scale requires reversing glTF winding at import time.
                        surface.indices.insert(surface.indices.end(), {face.mIndices[0],
                            face.mIndices[determinant < 0 ? 2 : 1], face.mIndices[determinant < 0 ? 1 : 2]});
                    }
                    if (!surface.indices.empty())
                        result.geometry.surfaces.push_back(std::move(surface));
                }
                for (unsigned child = 0; child < node.mNumChildren; ++child)
                    self(self, *node.mChildren[child], global);
            };
            visit(visit, *scene->mRootNode, glm::mat4(1.0f));
            if (result.geometry.surfaces.empty())
                throw std::runtime_error("Imported model contains no renderable triangles");
            return result;
        }
    }

    ModelImage decodeModelImage(std::span<const std::byte> bytes, bool srgb)
    {
        return decodeImage(bytes, srgb);
    }

    ModelImportResult importModel(const std::filesystem::path& source, ModelImportOptions options)
    {
        std::set<std::filesystem::path> dependencies;
        auto result = import(source, options, dependencies);
        dependencies.insert(result.sourceFiles.begin(), result.sourceFiles.end());
        result.sourceFiles.assign(dependencies.begin(), dependencies.end());
        return result;
    }

    std::future<ModelImportResult> importModelAsync(std::filesystem::path source, ModelImportOptions options)
    {
        return std::async(std::launch::async, [source = std::move(source), options] {
            return importModel(source, options);
        });
    }

    ImportedModelAsset importAndCopyModelAsset(const std::filesystem::path& source,
        const std::filesystem::path& worldEsm, std::uint32_t formId, ModelImportOptions options)
    {
        if (!formId)
            throw std::invalid_argument("Imported model must have a nonzero FormID");
        std::set<std::filesystem::path> dependencies;
        auto preview = import(source, options, dependencies);
        const auto original = preview.sourcePath;
        auto sourceDirectory = original.parent_path();
        const auto beneath = [](const std::filesystem::path& path, const std::filesystem::path& directory) {
            const auto relative = path.lexically_relative(directory);
            return !relative.empty() && !relative.is_absolute() && *relative.begin() != "..";
        };
        for (const auto& dependency : dependencies)
            while (!beneath(dependency, sourceDirectory))
            {
                const auto parent = sourceDirectory.parent_path();
                if (parent == sourceDirectory || parent.empty())
                    throw std::runtime_error("Model dependency is on a different filesystem: " + dependency.string());
                sourceDirectory = parent;
            }
        const auto root = std::filesystem::absolute(AssetResolver::projectAssetRoot(worldEsm));
        std::ostringstream id;
        id << std::hex << std::setw(8) << std::setfill('0') << formId;
        const auto destination = root / "models" / id.str();
        if (std::filesystem::exists(destination))
            throw std::runtime_error("An imported asset already owns FormID " + id.str());
        std::filesystem::create_directories(destination.parent_path());
        static std::atomic_uint64_t sequence{0};
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        const auto staging = destination.parent_path() / (".import-" + id.str() + "-"
            + std::to_string(stamp) + "-" + std::to_string(sequence.fetch_add(1)));
        if (!std::filesystem::create_directory(staging))
            throw std::runtime_error("Unable to create model import staging directory");
        try
        {
            const bool nested = sourceDirectory != original.parent_path();
            const auto payload = nested ? staging / "assets" : staging;
            for (const auto& dependency : dependencies)
            {
                const auto relative = dependency.lexically_relative(sourceDirectory);
                const auto target = payload / relative;
                std::filesystem::create_directories(target.parent_path());
                std::filesystem::copy_file(dependency, target);
            }
            std::string extension = original.extension().string();
            std::ranges::transform(extension, extension.begin(), [](unsigned char character) {
                return static_cast<char>(std::tolower(character));
            });
            const auto stagedSource = nested ? payload / original.lexically_relative(sourceDirectory)
                : staging / ("source" + extension);
            if (!nested && stagedSource != staging / original.filename())
                std::filesystem::copy_file(original, stagedSource);
            std::set<std::filesystem::path> stagedDependencies;
            auto imported = import(stagedSource, options, stagedDependencies);
            for (const auto& dependency : stagedDependencies)
                if (!beneath(dependency, staging))
                    throw std::runtime_error("Model uses an absolute external resource path; "
                        "use relative paths before importing: " + dependency.string());
            const auto relativeSource = stagedSource.lexically_relative(staging);
            std::filesystem::rename(staging, destination);
            imported.sourcePath = destination / relativeSource;
            stagedDependencies.insert(imported.sourceFiles.begin(), imported.sourceFiles.end());
            imported.sourceFiles.clear();
            for (const auto& dependency : stagedDependencies)
                imported.sourceFiles.push_back(destination / dependency.lexically_relative(staging));
            ImportedModelAsset result;
            result.assetRoot = root;
            result.sourcePath = imported.sourcePath;
            result.modelPath = (std::filesystem::path("models") / id.str() / relativeSource).generic_string();
            result.imported = std::move(imported);
            return result;
        }
        catch (...)
        {
            // This newly-created staging directory is owned exclusively by this operation.
            std::error_code ignored;
            std::filesystem::remove_all(staging, ignored);
            throw;
        }
    }
}
