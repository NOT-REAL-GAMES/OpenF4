#include "world/world_scene.hpp"

#include "assets/ba2_archive.hpp"
#include "assets/dds_texture.hpp"
#include "assets/dx10_archive.hpp"
#include "assets/material_file.hpp"
#include "assets/nif_model.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <unordered_map>

#include <glm/ext/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

namespace empire::world
{
    bool isRenderableModelPath(std::string_view path)
    {
        std::string value(path);
        std::replace(value.begin(), value.end(), '\\', '/');
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
            return static_cast<char>(std::tolower(character));
        });
        constexpr std::array blocked{
            "occluder", "roombounds", "roombound", "marker", "trigger", "_col.nif",
            "collision", "invisible", "ambient", "fxlight", "previs", "precombined",
            "cloud", "fog", "volume", "volumetric", "/fx/", "fxmeshes", "/effects/",
            "godrays", "debug", "editor", "helper", "dummy", "placeholder",
            "activateparent", "primitive", "intarrow", "navcut", "decals/roads",
            "defaultobject", "constructionline", "workshopsnap", "snappoint",
            "snapmarker", "workshopline", "wireconnector", "wireend", "powerconnector",
            "workshopcable", "constructobject", "bz_", "/scol/", "leveldesign", "/sky/",
            "worldspaceconnector", "wsconnector", "teleportdoor", "interiordoor",
            "exteriordoor", "axis", "origin", "gizmo", "movablestatic", "movableobject",
            "billboard", "impostor", "treelod", "_bb.nif", "_flat.nif", "flat_", "decal"};
        return std::ranges::none_of(blocked,
            [&](std::string_view blockedPart) { return value.find(blockedPart) != std::string::npos; });
    }

    namespace
    {
        constexpr float cellSize = 4096.0f;
        constexpr float vertexSpacing = 128.0f;
        constexpr std::size_t textureLimit = render::maxSceneTextures;

        std::string assetPath(std::string_view value, std::string_view prefix,
            std::string_view extension)
        {
            std::string result(value);
            std::replace(result.begin(), result.end(), '/', '\\');
            std::transform(result.begin(), result.end(), result.begin(), [](unsigned char character) {
                return static_cast<char>(std::tolower(character));
            });
            while (!result.empty() && result.front() == '\\')
                result.erase(result.begin());
            const auto embeddedPrefix = result.rfind(prefix);
            if (embeddedPrefix != std::string::npos)
                result.erase(0, embeddedPrefix);
            else if (!result.starts_with(prefix))
                result.insert(0, prefix);
            if (!result.ends_with(extension))
                result.append(extension);
            return result;
        }

        class Archives
        {
        public:
            explicit Archives(const std::filesystem::path& data)
            {
                for (const char* name : {"Fallout4 - MeshesExtra.ba2", "Fallout4 - Meshes.ba2",
                         "Fallout4 - Materials.ba2"})
                {
                    const auto path = data / name;
                    if (std::filesystem::is_regular_file(path))
                        general.emplace_back(std::make_unique<assets::Ba2Archive>(path));
                }
                constexpr std::array textureNames{
                    "Fallout4 - TexturesPatch.ba2", "Fallout4 - Textures1.ba2",
                    "Fallout4 - Textures2.ba2", "Fallout4 - Textures3.ba2",
                    "Fallout4 - Textures4.ba2", "Fallout4 - Textures5.ba2",
                    "Fallout4 - Textures6.ba2", "Fallout4 - Textures7.ba2",
                    "Fallout4 - Textures8.ba2", "Fallout4 - Textures9.ba2"};
                for (const char* name : textureNames)
                {
                    const auto path = data / name;
                    if (std::filesystem::is_regular_file(path))
                        textures.emplace_back(std::make_unique<assets::Dx10Archive>(path));
                }
            }

            std::vector<std::byte> extract(std::string_view path) const
            {
                for (const auto& archive : general)
                    if (archive->contains(path))
                        return archive->extract(path);
                throw std::runtime_error("Asset is not present in Fallout 4 archives: " + std::string(path));
            }

            std::vector<std::byte> extractTexture(std::string_view path) const
            {
                for (const auto& archive : textures)
                    if (archive->contains(path))
                        return archive->extractDds(path);
                throw std::runtime_error("Texture is not present in Fallout 4 archives: " + std::string(path));
            }

        private:
            std::vector<std::unique_ptr<assets::Ba2Archive>> general;
            std::vector<std::unique_ptr<assets::Dx10Archive>> textures;
        };

        render::TextureFormat textureFormat(std::uint32_t dxgiFormat)
        {
            switch (dxgiFormat)
            {
                case 28: return render::TextureFormat::rgba8;
                case 29: return render::TextureFormat::rgba8Srgb;
                case 70: case 71: return render::TextureFormat::bc1;
                case 72: return render::TextureFormat::bc1Srgb;
                case 73: case 74: return render::TextureFormat::bc2;
                case 75: return render::TextureFormat::bc2Srgb;
                case 76: case 77: return render::TextureFormat::bc3;
                case 78: return render::TextureFormat::bc3Srgb;
                case 82: case 83: return render::TextureFormat::bc5;
                case 97: case 98: return render::TextureFormat::bc7;
                case 99: return render::TextureFormat::bc7Srgb;
                case 87: return render::TextureFormat::bgra8;
                default: throw std::runtime_error("Unsupported scene texture format");
            }
        }

        class Materials
        {
        public:
            Materials(const Archives& archives, WorldScene& scene)
                : mArchives(archives), mScene(scene)
            {
                render::SceneTexture fallback;
                fallback.pixels = {std::byte{0xB3}, std::byte{0xB8}, std::byte{0xAD}, std::byte{0xFF}};
                fallback.mipOffsets = {0};
                mScene.textures.push_back(std::move(fallback));
                mScene.materials.push_back({});
            }

            std::uint32_t add(std::string_view materialPath, std::string_view directDiffuse,
                std::string_view directNormal, float directAlphaCutoff = 0.0f,
                const assets::MaterialSwapRecord* materialSwap = nullptr)
            {
                const auto resolved = materialSwap && !materialPath.empty()
                    ? materialSwap->resolve(materialPath)
                    : assets::ResolvedMaterialSwap{std::string(materialPath), std::nullopt, false};
                std::string normalizedMaterial = assetPath(resolved.path, "materials\\", {});
                if (!normalizedMaterial.empty() && !normalizedMaterial.ends_with(".bgsm")
                    && !normalizedMaterial.ends_with(".bgem"))
                    normalizedMaterial += ".bgsm";
                std::string key = !materialPath.empty()
                    ? normalizedMaterial
                    : std::string(directDiffuse) + '\n' + std::string(directNormal);
                if (!materialPath.empty() && (!directDiffuse.empty() || !directNormal.empty()))
                    key += '\n' + std::string(directDiffuse) + '\n' + std::string(directNormal);
                if (resolved.colorRemapIndex)
                    key += '\n' + std::to_string(*resolved.colorRemapIndex);
                if (directAlphaCutoff > 0.0f)
                    key += '\n' + std::to_string(directAlphaCutoff);
                if (const auto found = mMaterialIndices.find(key); found != mMaterialIndices.end())
                    return found->second;

                assets::MaterialInfo source;
                source.diffuseTexture = directDiffuse;
                source.normalTexture = directNormal;
                if (!materialPath.empty())
                {
                    try
                    {
                        auto loaded = assets::loadMaterial(mArchives.extract(normalizedMaterial));
                        if (loaded.diffuseTexture.empty())
                            loaded.diffuseTexture = source.diffuseTexture;
                        if (loaded.normalTexture.empty())
                            loaded.normalTexture = source.normalTexture;
                        source = std::move(loaded);
                    }
                    catch (const std::exception& error)
                    {
                        warn("Material", key, error);
                    }
                }
                source.alphaCutoff = std::max(source.alphaCutoff, directAlphaCutoff);
                if (resolved.colorRemapIndex && source.grayscaleToPalette)
                    source.grayscaleToPaletteScale
                        = std::clamp(*resolved.colorRemapIndex, 0.0f, 1.0f);
                render::SceneMaterial material;
                material.baseColor = source.baseColor;
                material.diffuseTexture = addTexture(source.diffuseTexture);
                material.normalTexture = addTexture(source.normalTexture);
                material.grayscaleTexture = addTexture(source.grayscaleTexture);
                material.grayscaleToPalette = source.grayscaleToPalette
                    && material.grayscaleTexture != 0;
                material.grayscaleToPaletteScale = source.grayscaleToPaletteScale;
                material.alphaCutoff = source.alphaCutoff;
                if (material.alphaCutoff > 0.0f && material.diffuseTexture == 0)
                    warn("Alpha texture", key,
                        std::runtime_error("alpha-masked material has no diffuse texture"));
                const auto index = static_cast<std::uint32_t>(mScene.materials.size());
                mScene.materials.push_back(material);
                mMaterialIndices.emplace(key, index);
                return index;
            }

            std::uint32_t addTerrain(const std::array<std::string, 4>& diffuse)
            {
                std::string key = "terrain";
                for (const auto& path : diffuse)
                    key += '\n' + path;
                if (const auto found = mMaterialIndices.find(key); found != mMaterialIndices.end())
                    return found->second;

                render::SceneMaterial material;
                material.terrain = true;
                std::uint32_t firstTexture = 0;
                for (glm::length_t slot = 0; slot < material.terrainTextures.length(); ++slot)
                {
                    material.terrainTextures[slot] = addTexture(diffuse[slot]);
                    if (firstTexture == 0 && material.terrainTextures[slot] != 0)
                        firstTexture = material.terrainTextures[slot];
                }
                if (firstTexture == 0)
                {
                    mMaterialIndices.emplace(std::move(key), 0);
                    return 0;
                }
                for (glm::length_t slot = 0; slot < material.terrainTextures.length(); ++slot)
                    if (material.terrainTextures[slot] == 0)
                        material.terrainTextures[slot] = firstTexture;

                const auto index = static_cast<std::uint32_t>(mScene.materials.size());
                mScene.materials.push_back(material);
                mMaterialIndices.emplace(std::move(key), index);
                return index;
            }

            std::uint32_t addDynamic(render::VulkanRenderer& renderer,
                std::string_view materialPath, std::string_view directDiffuse,
                std::string_view directNormal, float directAlphaCutoff = 0.0f,
                const assets::MaterialSwapRecord* materialSwap = nullptr)
            {
                const auto resolved = materialSwap && !materialPath.empty()
                    ? materialSwap->resolve(materialPath)
                    : assets::ResolvedMaterialSwap{std::string(materialPath), std::nullopt, false};
                std::string normalizedMaterial = assetPath(resolved.path, "materials\\", {});
                if (!normalizedMaterial.empty() && !normalizedMaterial.ends_with(".bgsm")
                    && !normalizedMaterial.ends_with(".bgem"))
                    normalizedMaterial += ".bgsm";
                std::string key = !materialPath.empty()
                    ? normalizedMaterial
                    : std::string(directDiffuse) + '\n' + std::string(directNormal);
                if (!materialPath.empty() && (!directDiffuse.empty() || !directNormal.empty()))
                    key += '\n' + std::string(directDiffuse) + '\n' + std::string(directNormal);
                if (resolved.colorRemapIndex)
                    key += '\n' + std::to_string(*resolved.colorRemapIndex);
                if (directAlphaCutoff > 0.0f)
                    key += '\n' + std::to_string(directAlphaCutoff);
                if (const auto found = mMaterialIndices.find(key); found != mMaterialIndices.end())
                    return found->second;

                assets::MaterialInfo source;
                source.diffuseTexture = directDiffuse;
                source.normalTexture = directNormal;
                if (!materialPath.empty())
                {
                    try
                    {
                        auto loaded = assets::loadMaterial(mArchives.extract(normalizedMaterial));
                        if (loaded.diffuseTexture.empty())
                            loaded.diffuseTexture = source.diffuseTexture;
                        if (loaded.normalTexture.empty())
                            loaded.normalTexture = source.normalTexture;
                        source = std::move(loaded);
                    }
                    catch (const std::exception& error)
                    {
                        warn("Material", key, error);
                    }
                }
                source.alphaCutoff = std::max(source.alphaCutoff, directAlphaCutoff);
                if (resolved.colorRemapIndex && source.grayscaleToPalette)
                    source.grayscaleToPaletteScale
                        = std::clamp(*resolved.colorRemapIndex, 0.0f, 1.0f);
                render::SceneMaterial material;
                material.baseColor = source.baseColor;
                material.diffuseTexture = addTextureDynamic(renderer, source.diffuseTexture);
                material.normalTexture = addTextureDynamic(renderer, source.normalTexture);
                material.grayscaleTexture = addTextureDynamic(renderer, source.grayscaleTexture);
                material.grayscaleToPalette = source.grayscaleToPalette
                    && material.grayscaleTexture != 0;
                material.grayscaleToPaletteScale = source.grayscaleToPaletteScale;
                material.alphaCutoff = source.alphaCutoff;
                if (material.alphaCutoff > 0.0f && material.diffuseTexture == 0)
                    warn("Alpha texture", key,
                        std::runtime_error("alpha-masked material has no diffuse texture"));
                const auto index = renderer.addMaterial(material);
                mMaterialIndices.emplace(key, index);
                return index;
            }

        private:
            const Archives& mArchives;
            WorldScene& mScene;
            std::unordered_map<std::string, std::uint32_t> mTextureIndices;
            std::unordered_map<std::string, std::uint32_t> mMaterialIndices;
            std::size_t mWarningCount = 0;

            void warn(std::string_view category, std::string_view path, const std::exception& error)
            {
                if (mWarningCount++ < 20)
                    std::cerr << category << " fallback: " << path << ": " << error.what() << '\n';
                else if (mWarningCount == 21)
                    std::cerr << "Further asset fallback messages suppressed\n";
            }

            std::uint32_t addTexture(std::string_view value)
            {
                if (value.empty())
                    return 0;
                const std::string path = assetPath(value, "textures\\", ".dds");
                if (const auto found = mTextureIndices.find(path); found != mTextureIndices.end())
                    return found->second;
                if (mScene.textures.size() >= textureLimit)
                    return 0;
                try
                {
                    const auto decoded = assets::loadDdsTexture(mArchives.extractTexture(path));
                    render::SceneTexture texture;
                    texture.width = decoded.width;
                    texture.height = decoded.height;
                    texture.mipLevels = decoded.mipLevels;
                    texture.format = textureFormat(decoded.dxgiFormat);
                    texture.pixels = decoded.pixels;
                    texture.mipOffsets = decoded.mipOffsets;
                    const auto index = static_cast<std::uint32_t>(mScene.textures.size());
                    mScene.textures.push_back(std::move(texture));
                    mTextureIndices.emplace(path, index);
                    return index;
                }
                catch (const std::exception& error)
                {
                    warn("Texture", path, error);
                    mTextureIndices.emplace(path, 0);
                    return 0;
                }
            }

            std::uint32_t addTextureDynamic(
                render::VulkanRenderer& renderer, std::string_view value)
            {
                if (value.empty())
                    return 0;
                const std::string path = assetPath(value, "textures\\", ".dds");
                if (const auto found = mTextureIndices.find(path); found != mTextureIndices.end())
                    return found->second;
                if (mTextureIndices.size() + 1 >= textureLimit)
                    return 0;
                try
                {
                    const auto decoded = assets::loadDdsTexture(mArchives.extractTexture(path));
                    render::SceneTexture texture;
                    texture.width = decoded.width;
                    texture.height = decoded.height;
                    texture.mipLevels = decoded.mipLevels;
                    texture.format = textureFormat(decoded.dxgiFormat);
                    texture.pixels = decoded.pixels;
                    texture.mipOffsets = decoded.mipOffsets;
                    const auto index = renderer.addTexture(texture);
                    mTextureIndices.emplace(path, index);
                    return index;
                }
                catch (const std::exception& error)
                {
                    warn("Texture", path, error);
                    mTextureIndices.emplace(path, 0);
                    return 0;
                }
            }
        };

        glm::vec4 boundsOf(const assets::ModelGeometry& geometry)
        {
            bool first = true;
            glm::vec3 minimum{};
            glm::vec3 maximum{};
            for (const auto& surface : geometry.surfaces)
            {
                for (const auto& vertex : surface.vertices)
                {
                    if (first)
                    {
                        minimum = maximum = vertex.position;
                        first = false;
                    }
                    else
                    {
                        minimum = glm::min(minimum, vertex.position);
                        maximum = glm::max(maximum, vertex.position);
                    }
                }
            }
            if (first)
                return {};
            const glm::vec3 center = (minimum + maximum) * 0.5f;
            float radius = 0.0f;
            for (const auto& surface : geometry.surfaces)
                for (const auto& vertex : surface.vertices)
                    radius = std::max(radius, glm::distance(center, vertex.position));
            return glm::vec4(center, radius);
        }

        struct TerrainSurface
        {
            std::uint32_t material = 0;
            std::array<glm::vec4, 17 * 17> weights{};
        };

        TerrainSurface terrainSurface(const assets::LandscapeCell& landscape, int quadrant,
            const assets::ExteriorWorld& world, Materials& materials)
        {
            struct SelectedLayer
            {
                const assets::LandscapeCell::TextureLayer* source = nullptr;
                glm::length_t slot = 0;
            };

            TerrainSurface result;
            std::array<std::string, 4> diffuse;
            std::vector<SelectedLayer> selected;
            std::unordered_map<std::uint32_t, glm::length_t> textureSlots;
            const auto select = [&](const assets::LandscapeCell::TextureLayer& layer)
            {
                if (layer.quadrant != quadrant || layer.texture == 0)
                    return;
                const auto ltex = world.landscapeTextures.find(layer.texture);
                if (ltex == world.landscapeTextures.end())
                    return;
                const auto txst = world.textureSets.find(ltex->second);
                if (txst == world.textureSets.end() || txst->second.diffuse.empty())
                    return;

                auto slot = textureSlots.find(layer.texture);
                if (slot == textureSlots.end())
                {
                    if (textureSlots.size() == diffuse.size())
                        return;
                    const auto index = static_cast<glm::length_t>(textureSlots.size());
                    slot = textureSlots.emplace(layer.texture, index).first;
                    diffuse[index] = txst->second.diffuse;
                }
                selected.push_back({&layer, slot->second});
            };
            for (const auto& layer : landscape.textureLayers)
                if (layer.base)
                    select(layer);
            for (const auto& layer : landscape.textureLayers)
                if (!layer.base)
                    select(layer);

            const auto base = std::ranges::find_if(selected,
                [](const SelectedLayer& layer) { return layer.source->base; });
            const glm::length_t baseSlot = base == selected.end() ? 0 : base->slot;
            for (auto& weights : result.weights)
            {
                weights = glm::vec4(0.0f);
                weights[baseSlot] = 1.0f;
            }
            for (const auto& layer : selected)
            {
                if (layer.source->base || layer.slot == baseSlot)
                    continue;
                for (const auto& point : layer.source->opacityPoints)
                {
                    if (point.position >= result.weights.size() || !std::isfinite(point.opacity))
                        continue;
                    auto& weights = result.weights[point.position];
                    const float opacity = std::clamp(point.opacity, 0.0f, 1.0f);
                    const float transfer = std::min(opacity, weights[baseSlot]);
                    weights[baseSlot] -= transfer;
                    weights[layer.slot] += opacity;
                }
            }
            for (auto& weights : result.weights)
            {
                const float sum = weights.x + weights.y + weights.z + weights.w;
                if (sum > 0.0f)
                    weights /= sum;
                else
                    weights[baseSlot] = 1.0f;
            }
            result.material = materials.addTerrain(diffuse);
            return result;
        }

        render::PackedMesh terrainQuadrant(const assets::LandscapeCell& landscape, int quadrant,
            std::uint32_t material, const std::array<glm::vec4, 17 * 17>& blendWeights)
        {
            constexpr int side = 17;
            const int startX = (quadrant & 1) * 16;
            const int startY = (quadrant >> 1) * 16;
            std::vector<render::SourceVertex> vertices;
            vertices.reserve(side * side);
            for (int localY = 0; localY < side; ++localY)
            {
                for (int localX = 0; localX < side; ++localX)
                {
                    const int x = startX + localX;
                    const int y = startY + localY;
                    const auto heightAt = [&](int sampleX, int sampleY) {
                        sampleX = std::clamp(sampleX, 0, 32);
                        sampleY = std::clamp(sampleY, 0, 32);
                        return landscape.heights[static_cast<std::size_t>(sampleY) * 33 + sampleX];
                    };
                    const float worldX = landscape.cell.x * cellSize + x * vertexSpacing;
                    const float worldY = landscape.cell.y * cellSize + y * vertexSpacing;
                    render::SourceVertex vertex;
                    vertex.position = {worldX, heightAt(x, y), worldY};
                    vertex.normal = glm::normalize(glm::vec3{
                        -(heightAt(x + 1, y) - heightAt(x - 1, y)) / (2.0f * vertexSpacing),
                        1.0f,
                        -(heightAt(x, y + 1) - heightAt(x, y - 1)) / (2.0f * vertexSpacing)});
                    vertex.uv = {worldX / 512.0f, worldY / 512.0f};
                    vertex.blendWeights = blendWeights[localY * side + localX];
                    vertices.push_back(vertex);
                }
            }
            std::vector<std::uint32_t> indices;
            indices.reserve(16 * 16 * 6);
            for (std::uint32_t y = 0; y < 16; ++y)
            {
                for (std::uint32_t x = 0; x < 16; ++x)
                {
                    const std::uint32_t topLeft = y * side + x;
                    const std::uint32_t topRight = topLeft + 1;
                    const std::uint32_t bottomLeft = topLeft + side;
                    const std::uint32_t bottomRight = bottomLeft + 1;
                    indices.insert(indices.end(), {
                        topLeft, topRight, bottomLeft, topRight, bottomRight, bottomLeft});
                }
            }
            return render::buildMeshlets(vertices, indices, material);
        }

        glm::mat4 referenceTransform(const assets::PlacedReference& reference)
        {
            float qx = 0.0f;
            float qy = 0.0f;
            float qz = 0.0f;
            float qw = 1.0f;
            eulerToQuaternion(reference.rotation[0], reference.rotation[1],
                reference.rotation[2], qx, qy, qz, qw);

            // Preserve the old renderer's Bethesda Z-up to engine Y-up component swap.
            const glm::quat engineRotation(qw, qx, qz, qy);
            const glm::vec3 enginePosition(reference.position[0], reference.position[2],
                reference.position[1]);
            return glm::translate(glm::mat4(1.0f), enginePosition)
                * glm::mat4_cast(engineRotation)
                * glm::scale(glm::mat4(1.0f), glm::vec3(reference.scale));
        }

        render::SceneInstance makeInstance(const render::MeshRange& range, const glm::mat4& transform)
        {
            render::SceneInstance instance;
            instance.transform = transform;
            const glm::vec3 center = glm::vec3(transform * glm::vec4(glm::vec3(range.bounds), 1.0f));
            const float scale = std::max(glm::length(glm::vec3(transform[0])),
                std::max(glm::length(glm::vec3(transform[1])), glm::length(glm::vec3(transform[2]))));
            instance.bounds = glm::vec4(center, range.bounds.w * scale);
            instance.meshletOffset = range.meshletOffset;
            instance.meshletCount = range.meshletCount;
            for (int column = 0; column < 4; ++column)
                for (int row = 0; row < 4; ++row)
                    if (!std::isfinite(instance.transform[column][row]))
                        throw std::runtime_error("Placed reference has a non-finite transform");
            if (!std::isfinite(instance.bounds.x) || !std::isfinite(instance.bounds.y)
                || !std::isfinite(instance.bounds.z) || !std::isfinite(instance.bounds.w)
                || instance.bounds.w > 10'000'000.0f)
                throw std::runtime_error("Placed reference has invalid bounds");
            return instance;
        }

        std::uint32_t selectedMaterialSwap(
            const assets::ExteriorWorld& world, const assets::PlacedReference& reference)
        {
            if (reference.materialSwap != 0)
                return reference.materialSwap;
            const auto baseSwap = world.baseMaterialSwaps.find(reference.baseObject);
            return baseSwap == world.baseMaterialSwaps.end() ? 0 : baseSwap->second;
        }

        const assets::MaterialSwapRecord* findMaterialSwap(
            const assets::ExteriorWorld& world, std::uint32_t formId)
        {
            const auto found = world.materialSwaps.find(formId);
            return found == world.materialSwaps.end() ? nullptr : &found->second;
        }

        std::uint64_t modelVariantKey(std::uint32_t baseObject, std::uint32_t materialSwap)
        {
            return (static_cast<std::uint64_t>(baseObject) << 32u) | materialSwap;
        }
    }

    struct WorldStreamer::Impl
    {
        assets::ExteriorWorld world;
        WorldScene initial;
        Archives archives;
        Materials materials;
        std::vector<assets::PlacedReference> references;
        std::unordered_map<std::uint64_t, std::optional<render::MeshRange>> modelRanges;
        std::size_t nextReference = 0;
        std::size_t loadedObjects = 0;
        std::size_t warningCount = 0;
        bool initialTaken = false;

        Impl(const std::filesystem::path& dataDirectory,
            std::optional<assets::CellCoord> requestedCenter, int cellRadius)
            : world(assets::loadCommonwealth(dataDirectory / "Fallout4.esm"))
            , archives(dataDirectory)
            , materials(archives, initial)
        {
            if (cellRadius < 0 || cellRadius > 8)
                throw std::runtime_error("World cell radius must be between 0 and 8");
            initial.centerCell = requestedCenter.value_or(world.suggestedCenter());
            initial.sceneScale = cellSize * static_cast<float>(cellRadius * 2 + 1);
            const auto selected = [&](assets::CellCoord cell) {
                return std::abs(cell.x - initial.centerCell.x) <= cellRadius
                    && std::abs(cell.y - initial.centerCell.y) <= cellRadius;
            };

            float centerHeight = 0.0f;
            if (const auto center = world.landscapes.find(initial.centerCell);
                center != world.landscapes.end())
                centerHeight = center->second.heights[16 * 33 + 16];
            for (const auto& [cell, landscape] : world.landscapes)
            {
                if (!selected(cell))
                    continue;
                ++initial.landscapeCells;
                for (int quadrant = 0; quadrant < 4; ++quadrant)
                {
                    const auto surface = terrainSurface(landscape, quadrant, world, materials);
                    const auto range = render::appendMesh(initial.mesh,
                        terrainQuadrant(landscape, quadrant, surface.material, surface.weights));
                    initial.instances.push_back(makeInstance(range, glm::mat4(1.0f)));
                }
            }
            for (const auto& reference : world.references)
                if (selected(reference.cell))
                    references.push_back(reference);
            const auto swappedReferences = std::ranges::count_if(references,
                [&](const auto& reference) { return selectedMaterialSwap(world, reference) != 0; });
            std::size_t substitutions = 0;
            for (const auto& [formId, materialSwap] : world.materialSwaps)
            {
                (void)formId;
                substitutions += materialSwap.substitutions.size();
            }
            std::cout << "ESM material swaps: " << world.materialSwaps.size() << " records, "
                      << substitutions << " substitutions, " << swappedReferences
                      << " selected references\n";
            const glm::vec2 centerPosition(
                (initial.centerCell.x + 0.5f) * cellSize,
                (initial.centerCell.y + 0.5f) * cellSize);
            std::ranges::sort(references, [&](const auto& left, const auto& right) {
                const glm::vec2 leftPosition(left.position[0], left.position[1]);
                const glm::vec2 rightPosition(right.position[0], right.position[1]);
                const glm::vec2 leftDelta = leftPosition - centerPosition;
                const glm::vec2 rightDelta = rightPosition - centerPosition;
                return glm::dot(leftDelta, leftDelta) < glm::dot(rightDelta, rightDelta);
            });
            if (initial.mesh.meshlets.empty())
                throw std::runtime_error("Selected Commonwealth cells contain no LAND geometry");
            initial.cameraTarget = {
                -centerPosition.x, centerHeight, centerPosition.y};
            initial.cameraPosition = initial.cameraTarget + glm::vec3(0.0f, 1800.0f, 2600.0f);
        }

        bool streamNext(render::VulkanRenderer& renderer)
        {
            while (nextReference < references.size())
            {
                const auto reference = references[nextReference++];
                try
                {
                    const auto materialSwapId = selectedMaterialSwap(world, reference);
                    const auto* materialSwap = findMaterialSwap(world, materialSwapId);
                    auto [rangeIterator, inserted] = modelRanges.try_emplace(
                        modelVariantKey(reference.baseObject, materialSwapId));
                    if (inserted)
                    {
                        const auto model = world.models.find(reference.baseObject);
                        if (model == world.models.end())
                            continue;
                        const std::string path = assetPath(model->second, "meshes\\", ".nif");
                        if (!isRenderableModelPath(path))
                            continue;
                        const auto geometry = assets::loadNifModel(archives.extract(path));
                        render::PackedMesh packed;
                        for (const auto& surface : geometry.surfaces)
                        {
                            if (surface.vertices.empty() || surface.indices.empty())
                                continue;
                            const auto material = materials.addDynamic(renderer,
                                surface.materialPath, surface.diffuseTexture, surface.normalTexture,
                                surface.alphaCutoff, materialSwap);
                            (void)render::appendMesh(packed,
                                render::buildMeshlets(surface.vertices, surface.indices, material));
                        }
                        if (!packed.meshlets.empty())
                        {
                            packed.bounds = boundsOf(geometry);
                            rangeIterator->second = renderer.addMesh(std::move(packed));
                        }
                    }
                    if (!rangeIterator->second)
                        continue;
                    renderer.addInstance(makeInstance(
                        *rangeIterator->second, referenceTransform(reference)));
                    ++loadedObjects;
                    return true;
                }
                catch (const std::exception& error)
                {
                    if (warningCount++ < 20)
                        std::cerr << "Streamed object skipped (" << std::hex << reference.formId
                                  << std::dec << "): " << error.what() << '\n';
                    else if (warningCount == 21)
                        std::cerr << "Further streamed-object messages suppressed\n";
                }
            }
            return false;
        }
    };

    void eulerToQuaternion(float rotX, float rotY, float rotZ,
        float& qx, float& qy, float& qz, float& qw)
    {
        float cx = std::cos(rotX * 0.5f), sx = std::sin(rotX * 0.5f);
        float cy = std::cos(rotY * 0.5f), sy = std::sin(rotY * 0.5f);
        float cz = std::cos(rotZ * 0.5f), sz = std::sin(rotZ * 0.5f);

        // ZYX order quaternion multiplication
        qw = cx * cy * cz + sx * sy * sz;
        qx = sx * cy * cz - cx * sy * sz;
        qy = cx * sy * cz + sx * cy * sz;
        qz = cx * cy * sz - sx * sy * cz;
    }

    glm::mat4 bethesdaReferenceTransform(const assets::PlacedReference& reference)
    {
        return referenceTransform(reference);
    }

    WorldStreamer::WorldStreamer(const std::filesystem::path& dataDirectory,
        std::optional<assets::CellCoord> requestedCenter, int cellRadius)
        : mImpl(std::make_unique<Impl>(dataDirectory, requestedCenter, cellRadius))
    {
    }

    WorldStreamer::~WorldStreamer() = default;
    WorldStreamer::WorldStreamer(WorldStreamer&&) noexcept = default;
    WorldStreamer& WorldStreamer::operator=(WorldStreamer&&) noexcept = default;

    WorldScene WorldStreamer::takeInitialScene()
    {
        if (mImpl->initialTaken)
            throw std::runtime_error("Initial streamed world scene was already taken");
        mImpl->initialTaken = true;
        return std::move(mImpl->initial);
    }

    bool WorldStreamer::streamNext(render::VulkanRenderer& renderer)
    {
        if (!mImpl->initialTaken)
            throw std::runtime_error("Initial scene must be installed before streaming objects");
        return mImpl->streamNext(renderer);
    }

    std::size_t WorldStreamer::streamedObjects() const noexcept
    {
        return mImpl->loadedObjects;
    }

    std::size_t WorldStreamer::pendingObjects() const noexcept
    {
        return mImpl->references.size() - mImpl->nextReference;
    }

    WorldScene loadWorldScene(const std::filesystem::path& dataDirectory,
        std::optional<assets::CellCoord> requestedCenter, int cellRadius, bool includeObjects,
        std::size_t maxPlacedObjects)
    {
        if (cellRadius < 0 || cellRadius > 8)
            throw std::runtime_error("World cell radius must be between 0 and 8");
        const auto world = assets::loadCommonwealth(dataDirectory / "Fallout4.esm");
        WorldScene result;
        result.centerCell = requestedCenter.value_or(world.suggestedCenter());
        result.sceneScale = cellSize * static_cast<float>(cellRadius * 2 + 1);
        Archives archives(dataDirectory);
        Materials materials(archives, result);

        const auto selected = [&](assets::CellCoord cell) {
            return std::abs(cell.x - result.centerCell.x) <= cellRadius
                && std::abs(cell.y - result.centerCell.y) <= cellRadius;
        };

        float centerHeight = 0.0f;
        if (const auto center = world.landscapes.find(result.centerCell); center != world.landscapes.end())
            centerHeight = center->second.heights[16 * 33 + 16];
        for (const auto& [cell, landscape] : world.landscapes)
        {
            if (!selected(cell))
                continue;
            ++result.landscapeCells;
            for (int quadrant = 0; quadrant < 4; ++quadrant)
            {
                const auto surface = terrainSurface(landscape, quadrant, world, materials);
                const auto range = render::appendMesh(result.mesh,
                    terrainQuadrant(landscape, quadrant, surface.material, surface.weights));
                result.instances.push_back(makeInstance(range, glm::mat4(1.0f)));
            }
        }

        std::unordered_map<std::uint64_t, std::optional<render::MeshRange>> modelRanges;
        for (const auto& reference : world.references)
        {
            if (!includeObjects)
                break;
            if (result.placedObjects >= maxPlacedObjects)
                break;
            if (!selected(reference.cell))
                continue;
            const auto materialSwapId = selectedMaterialSwap(world, reference);
            const auto* materialSwap = findMaterialSwap(world, materialSwapId);
            auto [rangeIterator, inserted] = modelRanges.try_emplace(
                modelVariantKey(reference.baseObject, materialSwapId));
            if (inserted)
            {
                const auto model = world.models.find(reference.baseObject);
                if (model != world.models.end())
                {
                    try
                    {
                        const std::string path = assetPath(model->second, "meshes\\", ".nif");
                        if (!isRenderableModelPath(path))
                            continue;
                        const auto geometry = assets::loadNifModel(archives.extract(path));
                        render::MeshRange modelRange;
                        modelRange.meshletOffset = static_cast<std::uint32_t>(result.mesh.meshlets.size());
                        modelRange.bounds = boundsOf(geometry);
                        for (const auto& surface : geometry.surfaces)
                        {
                            if (surface.vertices.empty() || surface.indices.empty())
                                continue;
                            const auto material = materials.add(
                                surface.materialPath, surface.diffuseTexture, surface.normalTexture,
                                surface.alphaCutoff, materialSwap);
                            (void)render::appendMesh(result.mesh,
                                render::buildMeshlets(surface.vertices, surface.indices, material));
                        }
                        modelRange.meshletCount = static_cast<std::uint32_t>(result.mesh.meshlets.size())
                            - modelRange.meshletOffset;
                        if (modelRange.meshletCount != 0)
                            rangeIterator->second = modelRange;
                    }
                    catch (const std::exception& error)
                    {
                        std::cerr << "Model skipped: " << model->second << ": " << error.what() << '\n';
                    }
                }
            }
            if (rangeIterator->second)
            {
                result.instances.push_back(makeInstance(*rangeIterator->second, referenceTransform(reference)));
                ++result.placedObjects;
            }
        }

        if (result.mesh.meshlets.empty())
            throw std::runtime_error("Selected Commonwealth cells contain no renderable geometry");
        result.cameraTarget = {
            -(result.centerCell.x + 0.5f) * cellSize, centerHeight,
            (result.centerCell.y + 0.5f) * cellSize};
        result.cameraPosition = result.cameraTarget + glm::vec3(0.0f, 1800.0f, 2600.0f);
        return result;
    }
}
