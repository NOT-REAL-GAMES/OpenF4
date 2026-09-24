#include "world/world_scene.hpp"

#include "assets/ba2_archive.hpp"
#include "assets/asset_resolver.hpp"
#include "assets/dds_texture.hpp"
#include "assets/dem_import.hpp"
#include "assets/dx10_archive.hpp"
#include "assets/havok_packfile.hpp"
#include "assets/material_file.hpp"
#include "assets/nif_model.hpp"
#include "assets/model_import.hpp"
#include "physics/jolt_collision.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <cmath>
#include <chrono>
#include <deque>
#include <future>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <set>
#include <shared_mutex>
#include <thread>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

#include <glm/ext/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

namespace openf4::world
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
            "workshopcable", "constructobject", "bz_", "leveldesign", "/sky/",
            "worldspaceconnector", "wsconnector", "teleportdoor", "interiordoor",
            "exteriordoor", "axis", "origin", "gizmo", "movablestatic", "movableobject",
            "billboard", "impostor", "treelod", "_bb.nif", "_flat.nif", "flat_", "decal"};
        return std::ranges::none_of(blocked,
            [&](std::string_view blockedPart) { return value.find(blockedPart) != std::string::npos; });
    }

    std::optional<assets::CellCoord> detail::demBtrOrigin(
        const std::filesystem::path& btrPath)
    {
        std::string name = btrPath.filename().string();
        std::ranges::transform(name, name.begin(), [](unsigned char value) {
            return static_cast<char>(std::tolower(value));
        });
        const std::string_view prefix = name.starts_with("openf4world.32.")
            ? "openf4world.32." : "empireworld.32.";
        constexpr std::string_view suffix = ".btr";
        std::string_view value(name);
        if (!value.starts_with(prefix) || !value.ends_with(suffix))
            return std::nullopt;
        value.remove_prefix(prefix.size());
        value.remove_suffix(suffix.size());
        const std::size_t separator = value.find('.');
        if (separator == std::string_view::npos)
            return std::nullopt;
        assets::CellCoord result;
        const auto x = value.substr(0, separator);
        const auto y = value.substr(separator + 1);
        const auto parsedX = std::from_chars(x.data(), x.data() + x.size(), result.x);
        const auto parsedY = std::from_chars(y.data(), y.data() + y.size(), result.y);
        if (parsedX.ec != std::errc{} || parsedX.ptr != x.data() + x.size()
            || parsedY.ec != std::errc{} || parsedY.ptr != y.data() + y.size())
            return std::nullopt;
        return result;
    }

    bool detail::preferDemBtr(const std::filesystem::path& candidate,
        const std::filesystem::path& current)
    {
        const auto candidateOrigin = demBtrOrigin(candidate);
        if (!candidateOrigin || candidateOrigin != demBtrOrigin(current)) return false;
        // Both paths have validated prefixes; O distinguishes the current branding.
        const auto isCurrent = [](const std::filesystem::path& path) {
            return std::tolower(static_cast<unsigned char>(path.filename().string().front())) == 'o';
        };
        return isCurrent(candidate) && !isCurrent(current);
    }

    bool detail::repairLegacyDemBtrGrid(std::span<render::SourceVertex> vertices,
        std::span<const float> fallbackHeights)
    {
        const auto finitePosition = [](const render::SourceVertex& vertex) {
            return std::isfinite(vertex.position.x) && std::isfinite(vertex.position.y)
                && std::isfinite(vertex.position.z);
        };
        if (std::ranges::all_of(vertices, finitePosition))
            return false;

        constexpr std::size_t side = 33;
        constexpr float spacing = 4096.0f;
        if (vertices.size() != side * side)
            throw std::runtime_error("DEM BTR has invalid non-grid vertex positions");
        const bool invalidHeight = std::ranges::any_of(vertices,
            [](const render::SourceVertex& vertex) {
                return !std::isfinite(vertex.position.y);
            });
        if (invalidHeight)
        {
            if (fallbackHeights.size() != vertices.size()
                || std::ranges::any_of(fallbackHeights,
                    [](float height) { return !std::isfinite(height); }))
                throw std::runtime_error("DEM BTR has a non-finite terrain height");
            for (std::size_t index = 0; index < vertices.size(); ++index)
                vertices[index].position.y = fallbackHeights[index];
        }

        const auto height = [&](int x, int y) {
            x = std::clamp(x, 0, 32);
            y = std::clamp(y, 0, 32);
            return vertices[static_cast<std::size_t>(y) * side
                + static_cast<std::size_t>(x)].position.y;
        };
        for (int y = 0; y < 33; ++y)
        {
            for (int x = 0; x < 33; ++x)
            {
                auto& vertex = vertices[static_cast<std::size_t>(y) * side
                    + static_cast<std::size_t>(x)];
                vertex.position.x = static_cast<float>(x) * spacing;
                vertex.position.z = static_cast<float>(y) * spacing;
                vertex.normal = glm::normalize(glm::vec3(
                    height(x - 1, y) - height(x + 1, y), 2.0f * spacing,
                    height(x, y - 1) - height(x, y + 1)));
                vertex.uv = {static_cast<float>(x) / 32.0f,
                    static_cast<float>(y) / 32.0f};
            }
        }
        return true;
    }

    bool detail::applyLandscapeBrush(assets::EditableLandscape& landscape,
        glm::vec2 center, LandscapeTool tool, float radius, float amount, float flattenHeight)
    {
        if (!std::isfinite(center.x) || !std::isfinite(center.y)
            || !std::isfinite(radius) || !std::isfinite(amount) || radius <= 0.0f || amount <= 0.0f)
            return false;
        constexpr float spacing = 128.0f;
        constexpr float negativeDelta = 128.0f * 8.0f;
        constexpr float positiveDelta = 127.0f * 8.0f;
        const auto previous = landscape;
        const std::int32_t firstX = static_cast<std::int32_t>(
            std::floor((center.x - radius) / spacing));
        const std::int32_t lastX = static_cast<std::int32_t>(
            std::ceil((center.x + radius) / spacing));
        const std::int32_t firstY = static_cast<std::int32_t>(
            std::floor((center.y - radius) / spacing));
        const std::int32_t lastY = static_cast<std::int32_t>(
            std::ceil((center.y + radius) / spacing));
        bool changed = false;
        for (std::int32_t y = firstY; y <= lastY; ++y)
        {
            for (std::int32_t x = firstX; x <= lastX; ++x)
            {
            const assets::EditableLandscape::SampleCoord sample{x, y};
            if (!previous.hasSample(sample))
                continue;
            const glm::vec2 position(static_cast<float>(sample.x) * spacing,
                static_cast<float>(sample.y) * spacing);
            const float distance = glm::distance(position, center);
            if (distance > radius)
                continue;
            const float step = amount * (1.0f - distance / radius);
            const float oldHeight = previous.sample(sample);
            float target = oldHeight;
            switch (tool)
            {
            case LandscapeTool::raise:
                target += step;
                break;
            case LandscapeTool::lower:
                target -= step;
                break;
            case LandscapeTool::smooth:
            {
                float sum = 0.0f;
                int count = 0;
                for (int oy = -1; oy <= 1; ++oy)
                {
                    for (int ox = -1; ox <= 1; ++ox)
                    {
                        if ((ox == 0 && oy == 0)
                            || !previous.hasSample({sample.x + ox, sample.y + oy}))
                            continue;
                        sum += previous.sample({sample.x + ox, sample.y + oy});
                        ++count;
                    }
                }
                if (count != 0)
                {
                    const float average = sum / static_cast<float>(count);
                    target += std::clamp(average - target, -step, step);
                }
                break;
            }
            case LandscapeTool::flatten:
                target += std::clamp(flattenHeight - target, -step, step);
                break;
            }

            float low = -std::numeric_limits<float>::max();
            float high = std::numeric_limits<float>::max();
            const auto constrain = [&](assets::EditableLandscape::SampleCoord neighbor,
                                       bool after) {
                if (!previous.hasSample(neighbor))
                    return;
                const float value = previous.sample(neighbor);
                low = std::max(low, value - (after ? negativeDelta : positiveDelta));
                high = std::min(high, value + (after ? positiveDelta : negativeDelta));
            };
            constrain({sample.x - 1, sample.y}, true);
            constrain({sample.x + 1, sample.y}, false);
            if (sample.x % 32 == 0)
            {
                constrain({sample.x, sample.y - 1}, true);
                constrain({sample.x, sample.y + 1}, false);
            }
            target = std::clamp(target, low, high);
            if (target != oldHeight)
            {
                landscape.setSample(sample, target);
                changed = true;
            }
            }
        }
        return changed;
    }

    void detail::quantizeLandscape(assets::EditableLandscape& landscape)
    {
        constexpr float scale = 8.0f;
        constexpr float minimumDelta = -128.0f * scale;
        constexpr float maximumDelta = 127.0f * scale;
        const auto quantize = [=](float& height) {
            if (!std::isfinite(height))
                throw std::runtime_error("Landscape contains a non-finite height");
            height = std::round(height / scale) * scale;
        };
        for (float& height : landscape.heights)
            quantize(height);
        for (auto& [coord, height] : landscape.additionalHeights)
        {
            (void)coord;
            quantize(height);
        }

        const std::size_t maxPasses = landscape.cellForms.size() * 4 + 4;
        for (std::size_t pass = 0; pass < maxPasses; ++pass)
        {
            bool changed = false;
            for (const auto& [cell, forms] : landscape.cellForms)
            {
                (void)forms;
                for (std::int32_t y = 0; y <= 32; ++y)
                {
                    for (std::int32_t x = 0; x <= 32; ++x)
                    {
                        if (x == 0 && y == 0)
                            continue;
                        const assets::EditableLandscape::SampleCoord current{
                            cell.x * 32 + x, cell.y * 32 + y};
                        const assets::EditableLandscape::SampleCoord previous = x == 0
                            ? assets::EditableLandscape::SampleCoord{current.x, current.y - 1}
                            : assets::EditableLandscape::SampleCoord{current.x - 1, current.y};
                        const float oldHeight = landscape.sample(current);
                        const float previousHeight = landscape.sample(previous);
                        const float height = std::clamp(oldHeight,
                            previousHeight + minimumDelta, previousHeight + maximumDelta);
                        if (height != oldHeight)
                        {
                            landscape.setSample(current, height);
                            changed = true;
                        }
                    }
                }
            }
            if (!changed)
                return;
        }
        throw std::runtime_error("Landscape cannot satisfy the signed VHGT delta range");
    }

    void detail::LandscapeHistory::commit(const assets::EditableLandscape& before)
    {
        if (mUndo.size() == 64)
            mUndo.erase(mUndo.begin());
        mUndo.push_back(before);
        mRedo.clear();
    }

    void detail::LandscapeHistory::addCell(assets::CellCoord coord)
    {
        for (auto& snapshot : mUndo)
            (void)snapshot.addCell(coord);
        for (auto& snapshot : mRedo)
            (void)snapshot.addCell(coord);
    }

    bool detail::LandscapeHistory::undo(assets::EditableLandscape& current)
    {
        if (mUndo.empty())
            return false;
        mRedo.push_back(current);
        current = mUndo.back();
        mUndo.pop_back();
        return true;
    }

    bool detail::LandscapeHistory::redo(assets::EditableLandscape& current)
    {
        if (mRedo.empty())
            return false;
        if (mUndo.size() == 64)
            mUndo.erase(mUndo.begin());
        mUndo.push_back(current);
        current = mRedo.back();
        mRedo.pop_back();
        return true;
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

        class Archives : public assets::AssetResolver
        {
        public:
            explicit Archives(const std::filesystem::path& data,
                const std::filesystem::path& master = {})
                : AssetResolver(master.empty() ? data / "Fallout4.esm" : master, data) {}
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

        render::SceneTexture sceneTexture(const assets::ModelImage& image)
        {
            render::SceneTexture result;
            result.width = image.width;
            result.height = image.height;
            result.mipLevels = image.mipLevels;
            result.format = image.encoding == assets::ModelImageEncoding::dds
                ? textureFormat(image.dxgiFormat)
                : image.srgb ? render::TextureFormat::rgba8Srgb : render::TextureFormat::rgba8;
            result.pixels = image.pixels;
            result.mipOffsets = image.mipOffsets;
            return result;
        }

        struct PreparedTexture
        {
            std::string path;
            std::optional<render::SceneTexture> texture;
            std::string error;
        };

        struct PreparedMaterial
        {
            std::string key;
            assets::MaterialInfo source;
            std::array<PreparedTexture, 3> textures;
            std::vector<std::string> warnings;
        };

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

            PreparedMaterial prepareSurface(const assets::ModelSurface& surface,
                const std::string& key, const assets::MaterialSwapRecord* swap = nullptr) const
            {
                auto prepared = prepare(surface.materialPath, surface.diffuseTexture,
                    surface.normalTexture, surface.alphaCutoff, swap, surface.decal);
                prepared.key = key;
                prepared.source.baseColor *= surface.baseColor;
                if (surface.alphaBlend)
                {
                    prepared.source.alphaCutoff = 0.5f;
                    prepared.warnings.push_back("Imported alpha blend uses a 0.5 cutout fallback; sorted transparency is not yet supported");
                }
                if (surface.doubleSided)
                    prepared.warnings.push_back("Imported double-sided material uses back-face culling in the current renderer");
                const auto copyImage = [&](const std::optional<assets::ModelImage>& image,
                                           std::size_t slot) {
                    if (!image)
                        return;
                    prepared.textures[slot] = {key + "/image" + std::to_string(slot),
                        sceneTexture(*image), {}};
                };
                copyImage(surface.diffuseImage, 0);
                copyImage(surface.normalImage, 1);
                return prepared;
            }

            std::uint32_t add(std::string_view materialPath, std::string_view directDiffuse,
                std::string_view directNormal, float directAlphaCutoff = 0.0f,
                const assets::MaterialSwapRecord* materialSwap = nullptr,
                bool terrainLod = false, bool directDecal = false)
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
                if (terrainLod)
                    key += "\nterrain-lod";
                if (directDecal)
                    key += "\ndecal";
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
                source.decal = source.decal || directDecal;
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
                material.terrainLod = terrainLod;
                material.decal = source.decal;
                if (material.alphaCutoff > 0.0f && material.diffuseTexture == 0)
                    warn("Alpha texture", key,
                        std::runtime_error("alpha-masked material has no diffuse texture"));
                const auto index = static_cast<std::uint32_t>(mScene.materials.size());
                mScene.materials.push_back(material);
                const auto inserted = mMaterialIndices.emplace(key, index).first;
                rememberMaterial(inserted->first);
                return index;
            }

            std::string terrainDiffuse(std::string_view materialPath)
            {
                if (materialPath.empty())
                    return {};

                std::string path = assetPath(materialPath, "materials\\", {});
                if (!path.ends_with(".bgsm") && !path.ends_with(".bgem"))
                    path += ".bgsm";
                if (const auto found = mTerrainDiffusePaths.find(path);
                    found != mTerrainDiffusePaths.end())
                    return found->second;
                try
                {
                    const std::string diffuse = assets::loadMaterial(mArchives.extract(path)).diffuseTexture;
                    mTerrainDiffusePaths.emplace(std::move(path), diffuse);
                    return diffuse;
                }
                catch (const std::exception& error)
                {
                    warn("Terrain material", path, error);
                    mTerrainDiffusePaths.emplace(std::move(path), std::string{});
                    return {};
                }
            }

            std::string terrainDiffuse(const assets::TextureSetRecord& textureSet)
            {
                return textureSet.material.empty()
                    ? textureSet.diffuse : terrainDiffuse(textureSet.material);
            }

            std::uint32_t addTerrain(const std::array<std::string, 8>& diffuse)
            {
                std::string key = "terrain";
                for (const auto& path : diffuse)
                    key += '\n' + path;
                if (const auto found = mMaterialIndices.find(key); found != mMaterialIndices.end())
                    return found->second;

                render::SceneMaterial material;
                material.terrain = true;
                std::uint32_t firstTexture = 0;
                for (std::size_t slot = 0; slot < diffuse.size(); ++slot)
                {
                    const auto component = static_cast<glm::length_t>(slot % 4);
                    auto& texture = slot < 4 ? material.terrainTextures[component]
                                             : material.terrainTexturesExtra[component];
                    texture = addTexture(diffuse[slot]);
                    if (firstTexture == 0 && texture != 0)
                        firstTexture = texture;
                }
                if (firstTexture == 0)
                {
                    const auto inserted = mMaterialIndices.emplace(std::move(key), 0).first;
                    rememberMaterial(inserted->first);
                    return 0;
                }
                for (std::size_t slot = 0; slot < diffuse.size(); ++slot)
                {
                    const auto component = static_cast<glm::length_t>(slot % 4);
                    auto& texture = slot < 4 ? material.terrainTextures[component]
                                             : material.terrainTexturesExtra[component];
                    if (texture == 0)
                        texture = firstTexture;
                }

                const auto index = static_cast<std::uint32_t>(mScene.materials.size());
                mScene.materials.push_back(material);
                const auto inserted = mMaterialIndices.emplace(std::move(key), index).first;
                rememberMaterial(inserted->first);
                return index;
            }

            std::uint32_t addTerrainDynamic(render::VulkanRenderer& renderer,
                const std::array<std::string, 8>& diffuse)
            {
                std::string key = "terrain";
                for (const auto& path : diffuse)
                    key += '\n' + path;
                if (const auto found = mMaterialIndices.find(key); found != mMaterialIndices.end())
                    return found->second;

                render::SceneMaterial material;
                material.terrain = true;
                std::uint32_t firstTexture = 0;
                for (std::size_t slot = 0; slot < diffuse.size(); ++slot)
                {
                    const auto component = static_cast<glm::length_t>(slot % 4);
                    auto& texture = slot < 4 ? material.terrainTextures[component]
                                             : material.terrainTexturesExtra[component];
                    texture = addTextureDynamic(renderer, diffuse[slot]);
                    if (firstTexture == 0 && texture != 0)
                        firstTexture = texture;
                }
                if (firstTexture == 0)
                {
                    const auto inserted = mMaterialIndices.emplace(std::move(key), 0).first;
                    rememberMaterial(inserted->first);
                    return 0;
                }
                for (std::size_t slot = 0; slot < diffuse.size(); ++slot)
                {
                    const auto component = static_cast<glm::length_t>(slot % 4);
                    auto& texture = slot < 4 ? material.terrainTextures[component]
                                             : material.terrainTexturesExtra[component];
                    if (texture == 0)
                        texture = firstTexture;
                }
                const auto index = renderer.addMaterial(material);
                const auto inserted = mMaterialIndices.emplace(std::move(key), index).first;
                rememberMaterial(inserted->first);
                return index;
            }

            std::uint32_t addDynamic(render::VulkanRenderer& renderer,
                std::string_view materialPath, std::string_view directDiffuse,
                std::string_view directNormal, float directAlphaCutoff = 0.0f,
                const assets::MaterialSwapRecord* materialSwap = nullptr)
            {
                return addPreparedDynamic(renderer, prepare(materialPath, directDiffuse,
                    directNormal, directAlphaCutoff, materialSwap));
            }

            PreparedMaterial prepare(std::string_view materialPath,
                std::string_view directDiffuse, std::string_view directNormal,
                float directAlphaCutoff = 0.0f,
                const assets::MaterialSwapRecord* materialSwap = nullptr,
                bool directDecal = false) const
            {
                const auto resolved = materialSwap && !materialPath.empty()
                    ? materialSwap->resolve(materialPath)
                    : assets::ResolvedMaterialSwap{std::string(materialPath), std::nullopt, false};
                std::string normalizedMaterial = assetPath(resolved.path, "materials\\", {});
                if (!normalizedMaterial.empty() && !normalizedMaterial.ends_with(".bgsm")
                    && !normalizedMaterial.ends_with(".bgem"))
                    normalizedMaterial += ".bgsm";
                PreparedMaterial result;
                result.key = !materialPath.empty()
                    ? normalizedMaterial
                    : std::string(directDiffuse) + '\n' + std::string(directNormal);
                if (!materialPath.empty() && (!directDiffuse.empty() || !directNormal.empty()))
                    result.key += '\n' + std::string(directDiffuse) + '\n' + std::string(directNormal);
                if (resolved.colorRemapIndex)
                    result.key += '\n' + std::to_string(*resolved.colorRemapIndex);
                if (directAlphaCutoff > 0.0f)
                    result.key += '\n' + std::to_string(directAlphaCutoff);
                if (directDecal)
                    result.key += "\ndecal";
                if (materialResident(result.key))
                    return result;

                result.source.diffuseTexture = directDiffuse;
                result.source.normalTexture = directNormal;
                if (!materialPath.empty())
                {
                    try
                    {
                        auto loaded = assets::loadMaterial(mArchives.extract(normalizedMaterial));
                        if (loaded.diffuseTexture.empty())
                            loaded.diffuseTexture = result.source.diffuseTexture;
                        if (loaded.normalTexture.empty())
                            loaded.normalTexture = result.source.normalTexture;
                        result.source = std::move(loaded);
                    }
                    catch (const std::exception& error)
                    {
                        result.warnings.push_back("Material fallback: " + result.key + ": "
                            + error.what());
                    }
                }
                result.source.alphaCutoff = std::max(result.source.alphaCutoff, directAlphaCutoff);
                result.source.decal = result.source.decal || directDecal;
                if (resolved.colorRemapIndex && result.source.grayscaleToPalette)
                    result.source.grayscaleToPaletteScale
                        = std::clamp(*resolved.colorRemapIndex, 0.0f, 1.0f);
                result.textures = {
                    prepareTexture(result.source.diffuseTexture),
                    prepareTexture(result.source.normalTexture),
                    prepareTexture(result.source.grayscaleTexture)};
                return result;
            }

            std::uint32_t addPreparedDynamic(
                render::VulkanRenderer& renderer, PreparedMaterial prepared,
                bool terrainLod = false)
            {
                if (terrainLod)
                    prepared.key += "\nterrain-lod";
                if (const auto found = mMaterialIndices.find(prepared.key);
                    found != mMaterialIndices.end())
                    return found->second;
                for (const auto& warning : prepared.warnings)
                    warn(warning);
                render::SceneMaterial material;
                material.baseColor = prepared.source.baseColor;
                material.diffuseTexture = addPreparedTexture(renderer, std::move(prepared.textures[0]));
                material.normalTexture = addPreparedTexture(renderer, std::move(prepared.textures[1]));
                material.grayscaleTexture = addPreparedTexture(renderer, std::move(prepared.textures[2]));
                material.grayscaleToPalette = prepared.source.grayscaleToPalette
                    && material.grayscaleTexture != 0;
                material.grayscaleToPaletteScale = prepared.source.grayscaleToPaletteScale;
                material.alphaCutoff = prepared.source.alphaCutoff;
                material.terrainLod = terrainLod;
                material.decal = prepared.source.decal;
                if (material.alphaCutoff > 0.0f && material.diffuseTexture == 0)
                    warn("Alpha texture fallback: " + prepared.key
                        + ": alpha-masked material has no diffuse texture");
                const auto index = renderer.addMaterial(material);
                const auto inserted = mMaterialIndices.emplace(std::move(prepared.key), index).first;
                rememberMaterial(inserted->first);
                return index;
            }

            std::uint32_t addPrepared(PreparedMaterial prepared, bool terrainLod = false)
            {
                if (terrainLod)
                    prepared.key += "\nterrain-lod";
                if (const auto found = mMaterialIndices.find(prepared.key);
                    found != mMaterialIndices.end())
                    return found->second;
                for (const auto& warning : prepared.warnings)
                    warn(warning);
                render::SceneMaterial material;
                material.baseColor = prepared.source.baseColor;
                material.diffuseTexture = addPreparedTexture(std::move(prepared.textures[0]));
                material.normalTexture = addPreparedTexture(std::move(prepared.textures[1]));
                material.grayscaleTexture = addPreparedTexture(std::move(prepared.textures[2]));
                material.grayscaleToPalette = prepared.source.grayscaleToPalette
                    && material.grayscaleTexture != 0;
                material.grayscaleToPaletteScale = prepared.source.grayscaleToPaletteScale;
                material.alphaCutoff = prepared.source.alphaCutoff;
                material.terrainLod = terrainLod;
                material.decal = prepared.source.decal;
                if (material.alphaCutoff > 0.0f && material.diffuseTexture == 0)
                    warn("Alpha texture fallback: " + prepared.key
                        + ": alpha-masked material has no diffuse texture");
                const auto index = static_cast<std::uint32_t>(mScene.materials.size());
                mScene.materials.push_back(material);
                const auto inserted = mMaterialIndices.emplace(std::move(prepared.key), index).first;
                rememberMaterial(inserted->first);
                return index;
            }

        private:
            const Archives& mArchives;
            WorldScene& mScene;
            std::unordered_map<std::string, std::uint32_t> mTextureIndices;
            std::unordered_map<std::string, std::uint32_t> mMaterialIndices;
            std::unordered_map<std::string, std::string> mTerrainDiffusePaths;
            mutable std::shared_mutex mResidentMutex;
            std::unordered_set<std::string> mResidentTextures;
            std::unordered_set<std::string> mResidentMaterials;
            std::size_t mWarningCount = 0;

            bool materialResident(std::string_view key) const
            {
                std::shared_lock lock(mResidentMutex);
                return mResidentMaterials.contains(std::string(key));
            }

            bool textureResident(std::string_view path) const
            {
                std::shared_lock lock(mResidentMutex);
                return mResidentTextures.contains(std::string(path));
            }

            void rememberMaterial(const std::string& key)
            {
                std::unique_lock lock(mResidentMutex);
                mResidentMaterials.insert(key);
            }

            void rememberTexture(const std::string& path)
            {
                std::unique_lock lock(mResidentMutex);
                mResidentTextures.insert(path);
            }

            PreparedTexture prepareTexture(std::string_view value) const
            {
                if (value.empty())
                    return {};
                PreparedTexture result;
                result.path = assetPath(value, "textures\\", ".dds");
                if (textureResident(result.path))
                    return result;
                try
                {
                    const auto decoded = assets::loadDdsTexture(mArchives.extractTexture(result.path));
                    render::SceneTexture texture;
                    texture.width = decoded.width;
                    texture.height = decoded.height;
                    texture.mipLevels = decoded.mipLevels;
                    texture.format = textureFormat(decoded.dxgiFormat);
                    texture.pixels = decoded.pixels;
                    texture.mipOffsets = decoded.mipOffsets;
                    result.texture = std::move(texture);
                }
                catch (const std::exception& error)
                {
                    result.error = "Texture fallback: " + result.path + ": " + error.what();
                }
                return result;
            }

            std::uint32_t addPreparedTexture(
                render::VulkanRenderer& renderer, PreparedTexture prepared)
            {
                if (prepared.path.empty())
                    return 0;
                if (const auto found = mTextureIndices.find(prepared.path);
                    found != mTextureIndices.end())
                    return found->second;
                if (mTextureIndices.size() + 1 >= textureLimit)
                    return 0;
                if (!prepared.texture)
                {
                    warn(prepared.error);
                    const auto inserted = mTextureIndices.emplace(std::move(prepared.path), 0).first;
                    rememberTexture(inserted->first);
                    return 0;
                }
                const auto index = renderer.addTexture(std::move(*prepared.texture));
                const auto inserted = mTextureIndices.emplace(std::move(prepared.path), index).first;
                rememberTexture(inserted->first);
                return index;
            }

            std::uint32_t addPreparedTexture(PreparedTexture prepared)
            {
                if (prepared.path.empty())
                    return 0;
                if (const auto found = mTextureIndices.find(prepared.path);
                    found != mTextureIndices.end())
                    return found->second;
                if (mScene.textures.size() >= textureLimit)
                    return 0;
                if (!prepared.texture)
                {
                    warn(prepared.error);
                    const auto inserted = mTextureIndices.emplace(std::move(prepared.path), 0).first;
                    rememberTexture(inserted->first);
                    return 0;
                }
                const auto index = static_cast<std::uint32_t>(mScene.textures.size());
                mScene.textures.push_back(std::move(*prepared.texture));
                const auto inserted = mTextureIndices.emplace(std::move(prepared.path), index).first;
                rememberTexture(inserted->first);
                return index;
            }

            void warn(std::string_view message)
            {
                if (mWarningCount++ < 20)
                    std::cerr << message << '\n';
                else if (mWarningCount == 21)
                    std::cerr << "Further asset fallback messages suppressed\n";
            }

            void warn(std::string_view category, std::string_view path, const std::exception& error)
            {
                warn(std::string(category) + " fallback: " + std::string(path)
                    + ": " + error.what());
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
                    const auto inserted = mTextureIndices.emplace(path, index).first;
                    rememberTexture(inserted->first);
                    return index;
                }
                catch (const std::exception& error)
                {
                    warn("Texture", path, error);
                    const auto inserted = mTextureIndices.emplace(path, 0).first;
                    rememberTexture(inserted->first);
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
                    const auto index = renderer.addTexture(std::move(texture));
                    const auto inserted = mTextureIndices.emplace(path, index).first;
                    rememberTexture(inserted->first);
                    return index;
                }
                catch (const std::exception& error)
                {
                    warn("Texture", path, error);
                    const auto inserted = mTextureIndices.emplace(path, 0).first;
                    rememberTexture(inserted->first);
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

        std::string connectionKey(std::string_view name)
        {
            std::string result(name);
            std::transform(result.begin(), result.end(), result.begin(),
                [](unsigned char character) { return static_cast<char>(std::tolower(character)); });
            if (result.size() > 2 && result[1] == '-')
                result.erase(0, 2);
            return result;
        }

        void attachModel(assets::ModelGeometry& parent, assets::ModelGeometry child)
        {
            const auto point = std::ranges::find_if(parent.attachmentPoints,
                [&](const auto& candidate) {
                    return std::ranges::any_of(child.attachmentTargets,
                        [&](const std::string& target) {
                            return connectionKey(candidate.name) == connectionKey(target);
                        });
                });
            if (point == parent.attachmentPoints.end())
                throw std::runtime_error("NIF attachment has no matching connection point");
            const glm::mat3 normalTransform = glm::transpose(glm::inverse(glm::mat3(point->transform)));
            for (auto& surface : child.surfaces)
            {
                for (auto& vertex : surface.vertices)
                {
                    vertex.position = glm::vec3(point->transform * glm::vec4(vertex.position, 1.0f));
                    vertex.normal = glm::normalize(normalTransform * vertex.normal);
                }
                parent.surfaces.push_back(std::move(surface));
            }
        }

        struct TerrainSurface
        {
            std::uint32_t material = 0;
            std::array<std::array<float, 8>, assets::LandscapeCell::side
                * assets::LandscapeCell::side> weights{};
        };

        constexpr std::optional<std::size_t> terrainWeightIndex(
            std::uint8_t quadrant, std::uint16_t position)
        {
            constexpr int quadrantSide = 17;
            if (quadrant >= 4 || position >= quadrantSide * quadrantSide)
                return std::nullopt;
            const int localX = position % 17;
            const int localY = position / 17;
            const int x = (quadrant & 1) * 16 + localX;
            const int y = (quadrant >> 1) * 16 + localY;
            return static_cast<std::size_t>(y * 33 + x);
        }

        static_assert(terrainWeightIndex(0, 0) == 0);
        static_assert(terrainWeightIndex(0, 16) == 16);
        static_assert(terrainWeightIndex(1, 0) == 16);
        static_assert(terrainWeightIndex(2, 0) == 16 * 33);
        static_assert(terrainWeightIndex(3, 16 * 17 + 16) == 32 * 33 + 32);
        static_assert(!terrainWeightIndex(0, 17 * 17));

        TerrainSurface terrainSurface(const assets::LandscapeCell& landscape,
            const assets::ExteriorWorld& world, Materials& materials,
            render::VulkanRenderer* renderer = nullptr)
        {
            struct SelectedLayer
            {
                const assets::LandscapeCell::TextureLayer* source = nullptr;
                std::size_t slot = 0;
            };

            TerrainSurface result;
            std::array<std::string, 8> diffuse;
            std::vector<SelectedLayer> selected;
            std::unordered_map<std::string, std::size_t> textureSlots;
            const auto texturePath = [&](std::uint32_t texture) -> std::string {
                if (const auto direct = world.landscapeMaterials.find(texture);
                    direct != world.landscapeMaterials.end())
                    return materials.terrainDiffuse(direct->second);
                const auto ltex = world.landscapeTextures.find(texture);
                if (ltex == world.landscapeTextures.end())
                    return {};
                const auto txst = world.textureSets.find(ltex->second);
                return txst == world.textureSets.end()
                    ? std::string{} : materials.terrainDiffuse(txst->second);
            };
            const auto addTextureSlot = [&](std::uint32_t texture) -> std::optional<std::size_t> {
                std::string path = texturePath(texture);
                if (path.empty())
                    return std::nullopt;
                auto slot = textureSlots.find(path);
                if (slot == textureSlots.end())
                {
                    if (textureSlots.size() == diffuse.size())
                        return std::nullopt;
                    const std::size_t index = textureSlots.size();
                    diffuse[index] = path;
                    slot = textureSlots.emplace(std::move(path), index).first;
                }
                return slot->second;
            };
            const auto select = [&](const assets::LandscapeCell::TextureLayer& layer) {
                if (const auto slot = addTextureSlot(layer.texture))
                    selected.push_back({&layer, *slot});
            };
            const auto defaultSlot = addTextureSlot(world.defaultLandscapeTexture);
            for (const auto& layer : landscape.textureLayers)
                if (layer.base)
                    select(layer);
            for (const auto& layer : landscape.textureLayers)
                if (!layer.base)
                    select(layer);
            std::stable_sort(selected.begin(), selected.end(),
                [](const SelectedLayer& left, const SelectedLayer& right) {
                    if (left.source->base != right.source->base)
                        return left.source->base;
                    if (left.source->quadrant != right.source->quadrant)
                        return left.source->quadrant < right.source->quadrant;
                    return left.source->layer < right.source->layer;
                });

            std::array<std::size_t, 4> baseSlots{};
            baseSlots.fill(defaultSlot.value_or(0));
            for (const auto& layer : selected)
                if (layer.source->base && layer.source->quadrant < baseSlots.size())
                    baseSlots[layer.source->quadrant] = layer.slot;
            for (int quadrant = 0; quadrant < 4; ++quadrant)
            {
                for (int localY = 0; localY < 17; ++localY)
                {
                    for (int localX = 0; localX < 17; ++localX)
                    {
                        const auto index = terrainWeightIndex(
                            static_cast<std::uint8_t>(quadrant),
                            static_cast<std::uint16_t>(localY * 17 + localX));
                        auto& weights = result.weights[*index];
                        weights.fill(0.0f);
                        weights[baseSlots[quadrant]] = 1.0f;
                    }
                }
            }
            for (const auto& layer : selected)
            {
                if (layer.source->base || layer.source->quadrant >= baseSlots.size())
                    continue;
                for (const auto& point : layer.source->opacityPoints)
                {
                    const auto index = terrainWeightIndex(layer.source->quadrant, point.position);
                    if (!index || !std::isfinite(point.opacity))
                        continue;
                    auto& weights = result.weights[*index];
                    const float opacity = std::clamp(point.opacity, 0.0f, 1.0f);
                    float& baseWeight = weights[baseSlots[layer.source->quadrant]];
                    baseWeight -= std::min(baseWeight, opacity);
                    weights[layer.slot] += opacity;
                }
            }
            for (auto& weights : result.weights)
            {
                const float sum = std::accumulate(weights.begin(), weights.end(), 0.0f);
                if (sum <= 0.0f)
                    weights[defaultSlot.value_or(0)] = 1.0f;
            }
            result.material = renderer
                ? materials.addTerrainDynamic(*renderer, diffuse) : materials.addTerrain(diffuse);
            return result;
        }

        render::PackedMesh terrainCell(
            const assets::LandscapeCell& landscape, const TerrainSurface& surface, int sampleStep = 1,
            const assets::EditableLandscape* editable = nullptr)
        {
            if (sampleStep < 1 || 32 % sampleStep != 0)
                throw std::runtime_error("Terrain sample step must divide 32");
            const int side = 32 / sampleStep + 1;
            std::vector<render::SourceVertex> vertices;
            vertices.reserve(side * side);
            for (int y = 0; y < side; ++y)
            {
                for (int x = 0; x < side; ++x)
                {
                    const int sampleX = x * sampleStep;
                    const int sampleY = y * sampleStep;
                    const auto heightAt = [&](int sampleX, int sampleY) {
                        if (editable)
                        {
                            const assets::EditableLandscape::SampleCoord global{
                                landscape.cell.x * 32 + sampleX,
                                landscape.cell.y * 32 + sampleY};
                            if (editable->hasSample(global))
                                return editable->sample(global);
                            return editable->sample({landscape.cell.x * 32
                                    + std::clamp(sampleX, 0, 32),
                                landscape.cell.y * 32 + std::clamp(sampleY, 0, 32)});
                        }
                        sampleX = std::clamp(sampleX, 0, 32);
                        sampleY = std::clamp(sampleY, 0, 32);
                        return landscape.heights[static_cast<std::size_t>(sampleY) * 33 + sampleX];
                    };
                    const float worldX = landscape.cell.x * cellSize + sampleX * vertexSpacing;
                    const float worldY = landscape.cell.y * cellSize + sampleY * vertexSpacing;
                    render::SourceVertex vertex;
                    vertex.position = {worldX, heightAt(sampleX, sampleY), worldY};
                    vertex.normal = glm::normalize(glm::vec3{
                        -(heightAt(sampleX + sampleStep, sampleY)
                            - heightAt(sampleX - sampleStep, sampleY))
                            / (2.0f * sampleStep * vertexSpacing),
                        1.0f,
                        -(heightAt(sampleX, sampleY + sampleStep)
                            - heightAt(sampleX, sampleY - sampleStep))
                            / (2.0f * sampleStep * vertexSpacing)});
                    constexpr float terrainTileSize = cellSize / 12.0f;
                    vertex.uv = {worldX / terrainTileSize, worldY / terrainTileSize};
                    const auto& weights = surface.weights[sampleY * 33 + sampleX];
                    vertex.blendWeights = {weights[0], weights[1], weights[2], weights[3]};
                    vertex.blendWeightsExtra = {weights[4], weights[5], weights[6], weights[7]};
                    if (landscape.hasColors)
                    {
                        const auto& color = landscape.colors[sampleY * 33 + sampleX];
                        vertex.color = glm::vec4(color.r, color.g, color.b, 255.0f) / 255.0f;
                    }
                    vertices.push_back(vertex);
                }
            }
            std::vector<std::uint32_t> indices;
            const std::uint32_t quads = static_cast<std::uint32_t>(side - 1);
            indices.reserve(quads * quads * 6);
            for (std::uint32_t y = 0; y < quads; ++y)
            {
                for (std::uint32_t x = 0; x < quads; ++x)
                {
                    const std::uint32_t topLeft = y * side + x;
                    const std::uint32_t topRight = topLeft + 1;
                    const std::uint32_t bottomLeft = topLeft + side;
                    const std::uint32_t bottomRight = bottomLeft + 1;
                    indices.insert(indices.end(), {
                        topLeft, topRight, bottomLeft, topRight, bottomRight, bottomLeft});
                }
            }
            return render::buildMeshlets(vertices, indices, surface.material);
        }

        render::PackedMesh waterCells(
            const std::map<assets::CellCoord, float>& cells, std::uint32_t material)
        {
            constexpr float surfaceBias = 2.0f;
            std::vector<render::SourceVertex> vertices;
            std::vector<std::uint32_t> indices;
            vertices.reserve(cells.size() * 4);
            indices.reserve(cells.size() * 6);
            for (const auto& [cell, sourceHeight] : cells)
            {
                if (!std::isfinite(sourceHeight) || std::abs(sourceHeight) >= 1.0e30f)
                    continue;
                const float minimumX = cell.x * cellSize;
                const float minimumZ = cell.y * cellSize;
                const float maximumX = minimumX + cellSize;
                const float maximumZ = minimumZ + cellSize;
                const float height = sourceHeight + surfaceBias;
                const std::uint32_t first = static_cast<std::uint32_t>(vertices.size());
                for (const glm::vec2 corner : std::array{
                         glm::vec2{minimumX, minimumZ}, glm::vec2{maximumX, minimumZ},
                         glm::vec2{minimumX, maximumZ}, glm::vec2{maximumX, maximumZ}})
                {
                    render::SourceVertex vertex;
                    vertex.position = {corner.x, height, corner.y};
                    vertex.normal = {0.0f, 1.0f, 0.0f};
                    vertex.uv = corner / cellSize;
                    vertices.push_back(vertex);
                }
                indices.insert(indices.end(), {
                    first, first + 1, first + 2, first + 1, first + 3, first + 2});
            }
            return render::buildMeshlets(vertices, indices, material);
        }

        render::PackedMesh waterPlane(
            float halfExtent, float sourceHeight, std::uint32_t material)
        {
            constexpr float surfaceBias = 2.0f;
            const float height = sourceHeight + surfaceBias;
            std::array<render::SourceVertex, 4> vertices;
            for (std::size_t index = 0; index < vertices.size(); ++index)
            {
                const glm::vec2 corner{
                    (index & 1u) != 0 ? halfExtent : -halfExtent,
                    (index & 2u) != 0 ? halfExtent : -halfExtent};
                vertices[index].position = {corner.x, height, corner.y};
                vertices[index].normal = {0.0f, 1.0f, 0.0f};
                vertices[index].uv = corner / cellSize;
            }
            constexpr std::array<std::uint32_t, 6> indices{0, 1, 2, 1, 3, 2};
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

            glm::quat bethesdaRotation(qw, qx, qy, qz);
            glm::vec3 bethesdaPosition(
                reference.position[0], reference.position[1], reference.position[2]);
            float scale = reference.scale;
            if (reference.hasLocalTransform)
            {
                float localQx = 0.0f;
                float localQy = 0.0f;
                float localQz = 0.0f;
                float localQw = 1.0f;
                eulerToQuaternion(reference.localRotation[0], reference.localRotation[1],
                    reference.localRotation[2], localQx, localQy, localQz, localQw);
                bethesdaPosition += bethesdaRotation * (glm::vec3(reference.localPosition[0],
                    reference.localPosition[1], reference.localPosition[2]) * scale);
                bethesdaRotation *= glm::quat(localQw, localQx, localQy, localQz);
                scale *= reference.localScale;
            }

            // Preserve the old renderer's Bethesda Z-up to engine Y-up component swap.
            const glm::quat engineRotation(bethesdaRotation.w, bethesdaRotation.x,
                bethesdaRotation.z, bethesdaRotation.y);
            const glm::vec3 enginePosition(
                bethesdaPosition.x, bethesdaPosition.z, bethesdaPosition.y);
            return glm::translate(glm::mat4(1.0f), enginePosition)
                * glm::mat4_cast(engineRotation)
                * glm::scale(glm::mat4(1.0f), glm::vec3(scale));
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

        glm::vec4 ragdollPoseBounds(std::span<const glm::mat4> transforms)
        {
            if (transforms.empty())
                throw std::runtime_error("Ragdoll pose has no body transforms");
            glm::vec3 minimum(transforms.front()[3]);
            glm::vec3 maximum = minimum;
            for (const glm::mat4& transform : transforms)
            {
                const glm::vec3 position(transform[3]);
                minimum = glm::min(minimum, position);
                maximum = glm::max(maximum, position);
            }
            const glm::vec3 center = (minimum + maximum) * 0.5f;
            float radius = 120.0f;
            for (const glm::mat4& transform : transforms)
                radius = std::max(radius, glm::distance(center, glm::vec3(transform[3])) + 80.0f);
            return glm::vec4(center, radius);
        }

        render::PackedMesh makeCapsuleDebugMesh(
            const assets::HavokRigidBody& body, std::uint32_t materialIndex)
        {
            constexpr std::uint32_t segments = 12;
            constexpr std::uint32_t hemisphereRings = 4;
            constexpr float pi = 3.14159265358979323846f;
            std::vector<render::SourceVertex> vertices;
            std::vector<std::uint32_t> indices;

            glm::mat4 axes(1.0f);
            axes[1] = glm::vec4(0.0f, 0.0f, 1.0f, 0.0f);
            axes[2] = glm::vec4(0.0f, 1.0f, 0.0f, 0.0f);
            const auto swapAxes = [](const glm::vec3& value) {
                return glm::vec3(value.x, value.z, value.y);
            };

            for (const auto& part : body.capsules)
            {
                if (!glm::all(glm::epsilonEqual(
                        part.scale, glm::vec3(part.scale.x), 0.00001f)))
                    throw std::runtime_error(
                        "Ragdoll debug mesh cannot render a non-uniform capsule");
                const glm::mat4 transform = axes * part.transform * axes;
                const auto point = [&](const glm::vec3& value) {
                    return glm::vec3(transform * glm::vec4(
                        swapAxes(value) * part.scale.x, 1.0f))
                        * physics::BethesdaUnitsPerMeter;
                };
                const glm::vec3 pointA = point(part.capsule.pointA);
                const glm::vec3 pointB = point(part.capsule.pointB);
                const float radius = part.capsule.radius * part.scale.x
                    * physics::BethesdaUnitsPerMeter;
                if (!std::isfinite(radius) || radius <= 0.0f)
                    throw std::runtime_error("Ragdoll debug capsule has an invalid radius");

                const glm::vec3 delta = pointB - pointA;
                const float length = glm::length(delta);
                const glm::vec3 axis = length > 0.00001f
                    ? delta / length : glm::vec3(0.0f, 1.0f, 0.0f);
                const glm::vec3 reference = std::abs(axis.y) < 0.9f
                    ? glm::vec3(0.0f, 1.0f, 0.0f)
                    : glm::vec3(1.0f, 0.0f, 0.0f);
                const glm::vec3 tangent = glm::normalize(glm::cross(reference, axis));
                const glm::vec3 bitangent = glm::cross(axis, tangent);

                const std::uint32_t bottom = static_cast<std::uint32_t>(vertices.size());
                vertices.push_back({pointA - axis * radius, -axis});
                std::vector<std::uint32_t> rings;
                const auto addRing = [&](const glm::vec3& center, float angle) {
                    const std::uint32_t start = static_cast<std::uint32_t>(vertices.size());
                    rings.push_back(start);
                    const float radialScale = std::cos(angle);
                    const float axialScale = std::sin(angle);
                    for (std::uint32_t segment = 0; segment < segments; ++segment)
                    {
                        const float around = 2.0f * pi
                            * static_cast<float>(segment) / static_cast<float>(segments);
                        const glm::vec3 radial = tangent * std::cos(around)
                            + bitangent * std::sin(around);
                        render::SourceVertex vertex;
                        vertex.position = center + axis * (axialScale * radius)
                            + radial * (radialScale * radius);
                        vertex.normal = glm::normalize(
                            axis * axialScale + radial * radialScale);
                        vertex.uv = {static_cast<float>(segment)
                                / static_cast<float>(segments),
                            axialScale * 0.5f + 0.5f};
                        vertices.push_back(vertex);
                    }
                };

                const float step = (pi * 0.5f) / static_cast<float>(hemisphereRings);
                for (std::uint32_t ring = 1; ring <= hemisphereRings; ++ring)
                    addRing(pointA, -pi * 0.5f + static_cast<float>(ring) * step);
                if (length > 0.00001f)
                    addRing(pointB, 0.0f);
                for (std::uint32_t ring = 1; ring < hemisphereRings; ++ring)
                    addRing(pointB, static_cast<float>(ring) * step);

                const std::uint32_t top = static_cast<std::uint32_t>(vertices.size());
                vertices.push_back({pointB + axis * radius, axis});
                for (std::uint32_t segment = 0; segment < segments; ++segment)
                {
                    const std::uint32_t next = (segment + 1) % segments;
                    indices.insert(indices.end(), {
                        bottom, rings.front() + segment, rings.front() + next});
                }
                for (std::size_t ring = 0; ring + 1 < rings.size(); ++ring)
                {
                    for (std::uint32_t segment = 0; segment < segments; ++segment)
                    {
                        const std::uint32_t next = (segment + 1) % segments;
                        indices.insert(indices.end(), {rings[ring] + segment,
                            rings[ring + 1] + segment, rings[ring + 1] + next,
                            rings[ring] + segment, rings[ring + 1] + next,
                            rings[ring] + next});
                    }
                }
                for (std::uint32_t segment = 0; segment < segments; ++segment)
                {
                    const std::uint32_t next = (segment + 1) % segments;
                    indices.insert(indices.end(), {
                        rings.back() + segment, top, rings.back() + next});
                }
            }
            if (vertices.empty())
                throw std::runtime_error("Ragdoll body has no capsule debug geometry");
            return render::buildMeshlets(vertices, indices, materialIndex);
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
        enum class LodLoadState
        {
            discovered,
            queued,
            loading,
            loaded,
            failed
        };

        struct CellState
        {
            std::optional<std::uint32_t> fullTerrain;
            std::optional<std::uint32_t> lodTerrain;
            std::optional<physics::BodyHandle> terrainBody;
            std::vector<std::uint32_t> fullObjects;
            std::vector<std::uint32_t> lodObjects;
            std::vector<std::uint64_t> fullReferenceIds;
            std::size_t pendingLodReplacements = 0;
            bool hasLodReplacements = false;
            bool fullTerrainQueued = false;
            bool lodTerrainQueued = false;
            bool terrainPhysicsQueued = false;
        };

        struct LodTile
        {
            static constexpr int side = 4;

            struct ObjectPart
            {
                std::uint32_t instance = 0;
                glm::vec4 bounds{};
                std::vector<assets::CellCoord> replacementCells;
            };

            assets::CellCoord origin;
            int level = 4;
            std::string terrainPath;
            std::string objectsPath;
            std::optional<std::uint32_t> terrain;
            std::array<std::vector<ObjectPart>, side * side> objects;
            LodLoadState terrainState = LodLoadState::discovered;
            LodLoadState objectsState = LodLoadState::failed;
        };

        struct LodRequest
        {
            std::size_t tile = 0;
            bool objects = false;
        };

        struct TerrainRequest
        {
            assets::CellCoord cell;
            bool lod = false;
        };

        struct PreparedSurface
        {
            render::PackedMesh mesh;
            PreparedMaterial material;
        };

        struct PreparedModel
        {
            struct Collision
            {
                std::vector<assets::HavokRigidBody> bodies;
                std::string error;
            };

            glm::vec4 bounds{};
            std::vector<PreparedSurface> surfaces;
            std::optional<Collision> collision;
            std::size_t removedWaterTriangles = 0;
        };

        struct ReferenceKey
        {
            std::uint64_t instanceId = 0;
            bool lod = false;

            auto operator<=>(const ReferenceKey&) const = default;
        };

        struct ReferenceRequest
        {
            assets::PlacedReference reference;
            bool lod = false;
        };

        struct ModelKey
        {
            std::uint64_t variant = 0;
            bool lod = false;

            auto operator<=>(const ModelKey&) const = default;
        };

        struct ModelState
        {
            std::optional<render::MeshRange> range;
            std::vector<ReferenceRequest> waiters;
            bool loading = false;
            bool failed = false;
        };

        struct PendingJob
        {
            ModelKey key;
            std::string path;
            bool preparesCollision = false;
            std::future<PreparedModel> future;
        };

        struct LodJob
        {
            LodRequest request;
            std::future<PreparedModel> future;
        };

        struct CachedCollisionBody
        {
            JPH::ShapeRefC shape;
            glm::mat4 localTransform{1.0f};
        };

        struct CollisionState
        {
            std::vector<CachedCollisionBody> bodies;
            bool loading = false;
            bool ready = false;
            bool failed = false;
        };

        struct FullReferencePhysics
        {
            assets::PlacedReference reference;
            std::string collisionPath;
            std::vector<physics::BodyHandle> bodies;
            bool ready = false;
            bool failed = false;
        };

        struct PhysicsSample
        {
            physics::BodyHandle body;
            std::uint32_t instance = 0;
            render::MeshRange range;
            glm::mat4 bodyLocal{1.0f};
        };

        struct PhysicsRagdollSample
        {
            physics::RagdollHandle ragdoll;
            std::vector<std::uint32_t> instances;
            std::vector<render::MeshRange> ranges;
            std::optional<std::uint32_t> skinnedInstance;
            std::uint32_t paletteOffset = 0;
            std::vector<std::uint32_t> paletteBodies;
            std::vector<glm::mat4> paletteFromInitialBodies;
        };

        assets::ExteriorWorld world;
        WorldScene initial;
        Archives archives;
        Materials materials;
        std::map<assets::CellCoord, std::vector<std::size_t>> referencesByCell;
        std::map<assets::CellCoord, CellState> cells;
        std::vector<LodTile> lodTiles;
        std::map<assets::CellCoord, std::size_t> lodTileByCell;
        std::map<std::tuple<int, int, int>, std::size_t> lodTileByKey;
        std::set<std::size_t> displayedLodTiles;
        std::deque<LodRequest> lodRequests;
        std::vector<LodJob> lodJobs;
        std::deque<TerrainRequest> terrainRequests;
        std::deque<assets::CellCoord> terrainPhysicsRequests;
        std::deque<ReferenceRequest> referenceRequests;
        std::deque<ReferenceRequest> readyReferences;
        std::deque<std::uint64_t> physicsReferences;
        std::deque<physics::BodyHandle> physicsRemovals;
        std::map<ModelKey, ModelState> models;
        std::map<std::string, CollisionState> collisions;
        std::map<std::string, std::vector<std::uint64_t>> referencesByCollisionPath;
        std::map<std::uint64_t, FullReferencePhysics> fullReferencePhysics;
        std::map<ReferenceKey, std::uint32_t> loadedReferences;
        std::set<ReferenceKey> queuedReferences;
        std::set<std::uint64_t> queuedPhysicsReferences;
        std::set<ReferenceKey> rejectedReferences;
        std::set<assets::CellCoord> completedBtoHandoffs;
        std::set<assets::CellCoord> terrainPhysicsCells;
        std::map<assets::CellCoord, std::vector<std::uint32_t>> btoPartsByReplacementCell;
        std::map<assets::CellCoord, std::vector<glm::vec4>> btoReplacementBounds;
        std::unordered_map<std::uint32_t, std::vector<assets::CellCoord>> btoReplacementCells;
        std::vector<PendingJob> jobs;
        physics::JoltWorld* physicsWorld = nullptr;
        std::vector<PhysicsSample> physicsSamples;
        std::vector<PhysicsRagdollSample> physicsRagdollSamples;
        assets::CellCoord centerCell{};
        assets::CellCoord physicsOriginCell{};
        glm::dvec2 streamPosition{};
        int fullRadius = 1;
        int lodRadius = 5;
        std::size_t maxObjects = std::numeric_limits<std::size_t>::max();
        std::size_t workerCount = 2;
        std::size_t loadedObjects = 0;
        std::size_t warningCount = 0;
        bool includeObjects = true;
        bool initialTaken = false;

        Impl(const std::filesystem::path& dataDirectory,
            std::optional<assets::CellCoord> requestedCenter, int cellRadius,
            bool loadObjects, std::size_t objectLimit,
            const std::optional<std::filesystem::path>& masterPath)
            // ponytail: positional ESM loading reads that file directly. Merge its MAST
            // dependency graph when inherited records are needed for standalone mod loading.
            : world(masterPath ? assets::loadExteriorWorld(*masterPath)
                               : assets::loadCommonwealth(dataDirectory / "Fallout4.esm"))
            , archives(dataDirectory, masterPath.value_or(dataDirectory / "Fallout4.esm"))
            , materials(archives, initial)
            , fullRadius(cellRadius)
            , lodRadius(std::min(cellRadius + 4, 8))
            , maxObjects(objectLimit)
            , workerCount(std::clamp<std::size_t>(
                std::max(1u, std::thread::hardware_concurrency()) / 2u, 2u, 16u))
            , includeObjects(loadObjects)
        {
            if (cellRadius < 0 || cellRadius > 8)
                throw std::runtime_error("World cell radius must be between 0 and 8");
            centerCell = requestedCenter.value_or(world.suggestedCenter());
            streamPosition = {
                (centerCell.x + 0.5f) * cellSize,
                (centerCell.y + 0.5f) * cellSize};
            initial.centerCell = centerCell;
            initial.sceneScale = cellSize * static_cast<float>(lodRadius * 2 + 1);
            for (std::size_t index = 0; index < world.references.size(); ++index)
            {
                const auto& reference = world.references[index];
                referencesByCell[reference.cell].push_back(index);
                if (reference.replacesDistantLod)
                {
                    auto& state = cells[reference.cell];
                    state.hasLodReplacements = true;
                    ++state.pendingLodReplacements;
                }
            }
            if (world.worldFormId == 0x0000003Cu)
                preloadCommonwealthLod();
            preloadCommonwealthWater();

            float centerHeight = 0.0f;
            if (const auto center = world.landscapes.find(centerCell);
                center != world.landscapes.end())
                centerHeight = center->second.heights[16 * 33 + 16];
            for (const auto& [cell, landscape] : world.landscapes)
            {
                const int distance = cellDistance(cell);
                if (distance > lodRadius)
                    continue;
                const auto surface = terrainSurface(landscape, world, materials);
                auto& state = cells[cell];

                // ponytail: decimated LAND covers the near LOD ring; use BTR tiles for a world-scale horizon.
                const auto lodRange = render::appendMesh(
                    initial.mesh, terrainCell(landscape, surface, 4));
                auto lodInstance = makeInstance(lodRange, glm::mat4(1.0f));
                lodInstance.active = distance > fullRadius
                    && !commonwealthTerrainCovers(cell);
                state.lodTerrain = static_cast<std::uint32_t>(initial.instances.size());
                initial.instances.push_back(lodInstance);
                ++initial.landscapeLodCells;

                if (distance <= fullRadius)
                {
                    const auto range = render::appendMesh(
                        initial.mesh, terrainCell(landscape, surface));
                    state.fullTerrain = static_cast<std::uint32_t>(initial.instances.size());
                    initial.instances.push_back(makeInstance(range, glm::mat4(1.0f)));
                    ++initial.landscapeCells;
                }
            }
            preloadReferenceLod();
            queueDesiredReferences();

            const auto swappedReferences = std::ranges::count_if(referenceRequests,
                [&](const ReferenceRequest& request) {
                    return selectedMaterialSwap(world, request.reference) != 0;
                });
            std::size_t substitutions = 0;
            for (const auto& [formId, materialSwap] : world.materialSwaps)
            {
                (void)formId;
                substitutions += materialSwap.substitutions.size();
            }
            std::cout << "ESM material swaps: " << world.materialSwaps.size() << " records, "
                      << substitutions << " substitutions, " << swappedReferences
                      << " selected references\n";
            if (includeObjects)
                std::cout << "Streamer: " << workerCount
                          << " parallel CPU jobs, time-budgeted GPU batches\n";
            if (initial.mesh.meshlets.empty())
                throw std::runtime_error("Selected Commonwealth cells contain no LAND geometry");
            initial.cameraTarget = {
                -streamPosition.x, centerHeight, streamPosition.y};
            initial.cameraPosition = initial.cameraTarget + glm::vec3(0.0f, 1800.0f, 1600.0f);
        }

        ~Impl()
        {
            if (!physicsWorld)
                return;
            for (const auto& sample : physicsSamples)
                physicsWorld->removeBody(sample.body);
            for (const auto& sample : physicsRagdollSamples)
                physicsWorld->removeRagdoll(sample.ragdoll);
            for (const auto body : physicsRemovals)
                physicsWorld->removeBody(body);
            for (auto& [cell, state] : cells)
            {
                (void)cell;
                if (state.terrainBody)
                    physicsWorld->removeBody(*state.terrainBody);
            }
            for (auto& [instanceId, reference] : fullReferencePhysics)
            {
                (void)instanceId;
                for (const auto body : reference.bodies)
                    physicsWorld->removeBody(body);
            }
        }

        int cellDistance(assets::CellCoord cell) const
        {
            return cellDistance(cell, centerCell);
        }

        static int cellDistance(assets::CellCoord cell, assets::CellCoord center)
        {
            return std::max(std::abs(cell.x - center.x), std::abs(cell.y - center.y));
        }

        bool fullCell(assets::CellCoord cell) const
        {
            return cellDistance(cell) <= fullRadius;
        }

        bool fullPrefetchCell(assets::CellCoord cell) const
        {
            return cellDistance(cell) <= std::min(fullRadius + 1, lodRadius);
        }

        bool lodCell(assets::CellCoord cell) const
        {
            const int distance = cellDistance(cell);
            return distance > fullRadius && distance <= lodRadius;
        }

        void syncTerrainPhysics(bool installFullCells = false)
        {
            if (!physicsWorld)
                return;

            std::set<assets::CellCoord> occupiedCells;
            for (const glm::vec3 position : physicsWorld->dynamicBodyPositions())
            {
                const assets::CellCoord occupied{
                    static_cast<std::int32_t>(std::floor(position.x / cellSize)),
                    static_cast<std::int32_t>(std::floor(position.z / cellSize))};
                for (int y = occupied.y - 1; y <= occupied.y + 1; ++y)
                    for (int x = occupied.x - 1; x <= occupied.x + 1; ++x)
                        occupiedCells.insert({x, y});
            }
            const auto wanted = [&](assets::CellCoord cell) {
                return fullPrefetchCell(cell) || occupiedCells.contains(cell);
            };

            for (const auto cell : terrainPhysicsRequests)
                cells[cell].terrainPhysicsQueued = false;
            terrainPhysicsRequests.clear();
            for (auto iterator = terrainPhysicsCells.begin();
                 iterator != terrainPhysicsCells.end();)
            {
                const assets::CellCoord cell = *iterator;
                if (!wanted(cell))
                {
                    auto& state = cells[cell];
                    physicsRemovals.push_back(*state.terrainBody);
                    state.terrainBody.reset();
                    iterator = terrainPhysicsCells.erase(iterator);
                }
                else
                    ++iterator;
            }

            const auto queueCell = [&](assets::CellCoord cell) {
                auto& state = cells[cell];
                if (state.terrainBody || state.terrainPhysicsQueued)
                    return;
                const auto landscape = world.landscapes.find(cell);
                if (landscape == world.landscapes.end() || !landscape->second.hasHeights)
                    return;
                if (installFullCells && fullCell(cell))
                {
                    state.terrainBody = physicsWorld->addStaticHeightField(
                        landscape->second.heights, assets::LandscapeCell::side, vertexSpacing,
                        {cell.x * cellSize, 0.0f, cell.y * cellSize});
                    terrainPhysicsCells.insert(cell);
                }
                else
                {
                    state.terrainPhysicsQueued = true;
                    if (fullCell(cell) || occupiedCells.contains(cell))
                        terrainPhysicsRequests.push_front(cell);
                    else
                        terrainPhysicsRequests.push_back(cell);
                }
            };
            const int radius = std::min(fullRadius + 1, lodRadius);
            for (int y = centerCell.y - radius; y <= centerCell.y + radius; ++y)
            {
                for (int x = centerCell.x - radius; x <= centerCell.x + radius; ++x)
                    queueCell({x, y});
            }
            for (const assets::CellCoord cell : occupiedCells)
                queueCell(cell);
        }

        bool installTerrainPhysics()
        {
            while (!terrainPhysicsRequests.empty())
            {
                const assets::CellCoord cell = terrainPhysicsRequests.front();
                terrainPhysicsRequests.pop_front();
                auto& state = cells[cell];
                state.terrainPhysicsQueued = false;
                if (!physicsWorld || state.terrainBody)
                    continue;
                const auto landscape = world.landscapes.find(cell);
                if (landscape == world.landscapes.end() || !landscape->second.hasHeights)
                    continue;
                state.terrainBody = physicsWorld->addStaticHeightField(
                    landscape->second.heights, assets::LandscapeCell::side, vertexSpacing,
                    {cell.x * cellSize, 0.0f, cell.y * cellSize});
                terrainPhysicsCells.insert(cell);
                return true;
            }
            return false;
        }

        void setReferencePhysicsReady(render::VulkanRenderer& renderer,
            FullReferencePhysics& reference, bool ready)
        {
            (void)renderer;
            if (reference.ready == ready)
                return;
            reference.ready = ready;
        }

        void removeReferencePhysics(render::VulkanRenderer& renderer,
            FullReferencePhysics& reference)
        {
            queuedPhysicsReferences.erase(reference.reference.instanceId);
            physicsRemovals.insert(
                physicsRemovals.end(), reference.bodies.begin(), reference.bodies.end());
            reference.bodies.clear();
            setReferencePhysicsReady(renderer, reference, false);
        }

        bool removeStalePhysics()
        {
            if (!physicsWorld || physicsRemovals.empty())
                return false;
            physicsWorld->removeBody(physicsRemovals.front());
            physicsRemovals.pop_front();
            return true;
        }

        void syncReferencePhysics(render::VulkanRenderer& renderer, std::uint64_t instanceId)
        {
            const auto found = fullReferencePhysics.find(instanceId);
            if (found == fullReferencePhysics.end() || !physicsWorld)
                return;
            auto& reference = found->second;
            if (!fullPrefetchCell(reference.reference.cell))
            {
                removeReferencePhysics(renderer, reference);
                return;
            }
            if (reference.ready || reference.failed)
                return;

            const auto collision = collisions.find(reference.collisionPath);
            if (collision == collisions.end() || collision->second.loading)
                return;
            if (collision->second.failed)
            {
                reference.failed = true;
                return;
            }
            if (!collision->second.ready)
                return;
            if (collision->second.bodies.empty())
            {
                setReferencePhysicsReady(renderer, reference, true);
                return;
            }
            if (queuedPhysicsReferences.insert(instanceId).second)
                physicsReferences.push_back(instanceId);
        }

        void syncCellPhysics(render::VulkanRenderer& renderer, assets::CellCoord cell)
        {
            const auto found = cells.find(cell);
            if (found == cells.end())
                return;
            for (const std::uint64_t instanceId : found->second.fullReferenceIds)
                syncReferencePhysics(renderer, instanceId);
        }

        void syncCollisionPhysics(render::VulkanRenderer& renderer, const std::string& path)
        {
            const auto found = referencesByCollisionPath.find(path);
            if (found == referencesByCollisionPath.end())
                return;
            for (const std::uint64_t instanceId : found->second)
                syncReferencePhysics(renderer, instanceId);
        }

        void failCollision(render::VulkanRenderer& renderer,
            const std::string& path, std::string_view error)
        {
            auto& collision = collisions[path];
            collision.loading = false;
            collision.ready = false;
            collision.failed = true;
            collision.bodies.clear();
            syncCollisionPhysics(renderer, path);
            if (warningCount++ < 20)
                std::cerr << "Streamed collision skipped (" << path << "): " << error << '\n';
            else if (warningCount == 21)
                std::cerr << "Further streamed-object messages suppressed\n";
        }

        void completeCollision(render::VulkanRenderer& renderer,
            const std::string& path, PreparedModel::Collision prepared)
        {
            if (!prepared.error.empty())
            {
                failCollision(renderer, path, prepared.error);
                return;
            }

            auto& collision = collisions[path];
            collision.loading = false;
            collision.bodies.clear();
            try
            {
                collision.bodies.reserve(prepared.bodies.size());
                for (auto& body : prepared.bodies)
                {
                    body.dynamic = false;
                    collision.bodies.push_back({
                        physics::makeJoltCollisionShape(body),
                        physics::havokBodyLocalTransform(body)});
                }
                collision.ready = true;
                collision.failed = false;
            }
            catch (const std::exception& error)
            {
                failCollision(renderer, path, error.what());
                return;
            }
            syncCollisionPhysics(renderer, path);
        }

        bool installReferencePhysics(render::VulkanRenderer& renderer)
        {
            while (!physicsReferences.empty())
            {
                const std::uint64_t instanceId = physicsReferences.front();
                physicsReferences.pop_front();
                if (queuedPhysicsReferences.erase(instanceId) == 0)
                    continue;
                const auto found = fullReferencePhysics.find(instanceId);
                if (found == fullReferencePhysics.end())
                    continue;
                auto& reference = found->second;
                if (!physicsWorld || !fullPrefetchCell(reference.reference.cell)
                    || reference.ready || reference.failed)
                    continue;
                const auto collision = collisions.find(reference.collisionPath);
                if (collision == collisions.end() || !collision->second.ready)
                    continue;

                std::vector<physics::BodyHandle> created;
                try
                {
                    const float referenceScale = reference.reference.scale
                        * (reference.reference.hasLocalTransform
                            ? reference.reference.localScale : 1.0f);
                    if (!std::isfinite(referenceScale) || referenceScale <= 0.0f)
                        throw std::runtime_error("placed collision has an invalid scale");
                    const glm::mat4 placed = referenceTransform(reference.reference);
                    created.reserve(collision->second.bodies.size());
                    for (const auto& body : collision->second.bodies)
                    {
                        const glm::mat4 worldTransform = placed * body.localTransform;
                        glm::mat3 rotationMatrix;
                        for (glm::length_t column = 0; column < 3; ++column)
                            rotationMatrix[column] = glm::normalize(glm::vec3(worldTransform[column]));
                        const glm::quat rotation = glm::normalize(glm::quat_cast(rotationMatrix));
                        const auto shape = physics::scaleJoltCollisionShape(
                            body.shape, glm::vec3(referenceScale));
                        created.push_back(physicsWorld->addStaticBody(
                            shape, glm::vec3(worldTransform[3]), rotation));
                    }
                    reference.bodies = std::move(created);
                    setReferencePhysicsReady(renderer, reference, true);
                }
                catch (const std::exception& error)
                {
                    for (const auto body : created)
                        physicsWorld->removeBody(body);
                    reference.failed = true;
                    if (warningCount++ < 20)
                        std::cerr << "Placed collision skipped (" << std::hex
                                  << reference.reference.formId << std::dec << "): "
                                  << error.what() << '\n';
                    else if (warningCount == 21)
                        std::cerr << "Further streamed-object messages suppressed\n";
                }
                return true;
            }
            return false;
        }

        std::optional<glm::vec3> enablePhysics(
            render::VulkanRenderer& renderer, physics::JoltWorld& worldPhysics,
            bool spawnPhysicsSample)
        {
            if (physicsWorld)
                throw std::runtime_error("World physics was already enabled");
            physicsWorld = &worldPhysics;
            physicsOriginCell = centerCell;
            physicsWorld->setOrigin(
                {centerCell.x * cellSize, 0.0f, centerCell.y * cellSize});
            syncTerrainPhysics(true);
            const int physicsRadius = std::min(fullRadius + 1, lodRadius);
            for (int y = centerCell.y - physicsRadius; y <= centerCell.y + physicsRadius; ++y)
                for (int x = centerCell.x - physicsRadius; x <= centerCell.x + physicsRadius; ++x)
                    syncCellPhysics(renderer, {x, y});
            if (!spawnPhysicsSample)
                return std::nullopt;

            const auto landscape = world.landscapes.find(centerCell);
            const float centerHeight = landscape == world.landscapes.end()
                ? 0.0f : landscape->second.heights[16 * assets::LandscapeCell::side + 16];
            const glm::vec3 sampleCenter((centerCell.x + 0.5f) * cellSize,
                centerHeight + 120.0f, (centerCell.y + 0.5f) * cellSize);

            const auto buildSampleMesh = [&](assets::ModelGeometry geometry,
                                             std::string_view label) {
                render::PackedMesh packed;
                for (const auto& surface : geometry.surfaces)
                {
                    auto mesh = render::buildMeshlets(surface.vertices, surface.indices);
                    const auto material = materials.addPreparedDynamic(renderer,
                        materials.prepare(surface.materialPath, surface.diffuseTexture,
                            surface.normalTexture, surface.alphaCutoff, nullptr, surface.decal));
                    for (auto& meshlet : mesh.meshlets)
                        meshlet.materialIndex = material;
                    (void)render::appendMesh(packed, std::move(mesh));
                }
                if (packed.meshlets.empty())
                    throw std::runtime_error(std::string(label) + " has no renderable geometry");
                return renderer.addMesh(std::move(packed));
            };

            struct SampleAsset
            {
                std::string label;
                assets::ModelGeometry geometry;
                assets::HavokRigidBody body;
            };

            const auto loadSample = [&](std::string_view label, std::string_view path,
                                        std::initializer_list<std::string_view> attachments = {}) {
                const auto bytes = archives.extract(path);
                auto geometry = assets::loadNifModel(bytes);
                for (const std::string_view attachment : attachments)
                    attachModel(geometry, assets::loadNifModel(archives.extract(attachment)));
                auto systems = assets::loadNifPhysicsSystems(bytes);
                if (systems.size() != 1 || systems.front().bodies.size() != 1)
                    throw std::runtime_error(std::string(label)
                        + " does not have one Havok rigid body");
                return SampleAsset{
                    std::string(label), std::move(geometry),
                    std::move(systems.front().bodies.front())};
            };

            const auto addSample = [&](SampleAsset asset, const glm::mat4& bodyWorld,
                                       bool dynamic) {
                const auto range = buildSampleMesh(std::move(asset.geometry), asset.label);
                const auto shape = physics::makeJoltCollisionShape(asset.body);
                const glm::mat4 bodyLocal = physics::havokBodyLocalTransform(asset.body);
                glm::mat3 rotationMatrix;
                for (glm::length_t column = 0; column < 3; ++column)
                    rotationMatrix[column] = glm::normalize(glm::vec3(bodyWorld[column]));
                const glm::quat rotation = glm::normalize(glm::quat_cast(rotationMatrix));
                const float dynamicMass = asset.body.mass > 0.0f ? asset.body.mass : 10.0f;
                const auto body = dynamic
                    ? physicsWorld->addDynamicBody(shape, glm::vec3(bodyWorld[3]), rotation,
                        dynamicMass, asset.body.friction, asset.body.restitution)
                    : physicsWorld->addStaticBody(shape, glm::vec3(bodyWorld[3]), rotation);
                const auto instance = renderer.addInstance(makeInstance(range,
                    physicsWorld->bodyTransform(body) * glm::inverse(bodyLocal)));
                physicsSamples.push_back(PhysicsSample{body, instance, range, bodyLocal});
                std::cout << "  " << asset.label << ": " << asset.body.parts.size()
                          << " convex, " << asset.body.capsules.size() << " capsule, "
                          << (asset.body.triangleMesh ? 1 : 0) << " compressed mesh";
                if (dynamic && asset.body.mass <= 0.0f)
                    std::cout << ", promoted static fixture with 10 kg demo mass";
                std::cout << '\n';
            };

            constexpr std::string_view platformPath =
                "Meshes\\SetDressing\\Greebs\\WallGreeb1x1_01.nif";
            auto platform = loadSample("mixed WallGreeb platform", platformPath);
            if (!platform.body.triangleMesh)
                throw std::runtime_error("WallGreeb platform has no compressed mesh");

            glm::vec3 supportCenter{};
            glm::vec3 supportNormal{};
            float supportArea = 0.0f;
            const auto& platformMesh = *platform.body.triangleMesh;
            for (std::size_t index = 0; index < platformMesh.indices.size(); index += 3)
            {
                const auto vertex = [&](std::uint32_t vertexIndex) {
                    const glm::vec3& value = platformMesh.vertices[vertexIndex];
                    return glm::vec3(value.x, value.z, value.y)
                        * physics::BethesdaUnitsPerMeter;
                };
                const glm::vec3 a = vertex(platformMesh.indices[index]);
                const glm::vec3 b = vertex(platformMesh.indices[index + 2]);
                const glm::vec3 c = vertex(platformMesh.indices[index + 1]);
                const glm::vec3 normal = glm::cross(b - a, c - a);
                const float area = glm::length(normal);
                if (area > supportArea)
                {
                    supportArea = area;
                    supportCenter = (a + b + c) / 3.0f;
                    supportNormal = normal / area;
                }
            }
            if (supportArea <= 0.0f)
                throw std::runtime_error("WallGreeb platform has no usable collision triangle");

            const auto rotationBetween = [](glm::vec3 from, glm::vec3 to) {
                from = glm::normalize(from);
                to = glm::normalize(to);
                const float cosine = glm::clamp(glm::dot(from, to), -1.0f, 1.0f);
                if (cosine < -0.9999f)
                {
                    glm::vec3 axis = glm::cross(from, glm::vec3(1.0f, 0.0f, 0.0f));
                    if (glm::dot(axis, axis) < 0.0001f)
                        axis = glm::cross(from, glm::vec3(0.0f, 0.0f, 1.0f));
                    return glm::angleAxis(glm::pi<float>(), glm::normalize(axis));
                }
                const glm::vec3 axis = glm::cross(from, to);
                return glm::normalize(glm::quat(1.0f + cosine, axis.x, axis.y, axis.z));
            };
            const glm::quat platformRotation = rotationBetween(
                supportNormal, glm::vec3(0.0f, 1.0f, 0.0f));
            const glm::vec3 platformPosition = sampleCenter - platformRotation * supportCenter;
            const glm::mat4 platformBodyWorld = glm::translate(
                glm::mat4(1.0f), platformPosition) * glm::mat4_cast(platformRotation);
            addSample(std::move(platform), platformBodyWorld, false);

            const auto dynamicBodyWorld = [&](const glm::vec3& offset,
                                               const assets::HavokRigidBody& sourceBody) {
                return glm::translate(glm::mat4(1.0f), sampleCenter + offset)
                    * physics::havokBodyLocalTransform(sourceBody);
            };
            const auto addFallingSample = [&](std::string_view label, std::string_view path,
                                              const glm::vec3& offset,
                                              std::initializer_list<std::string_view> attachments = {}) {
                auto asset = loadSample(label, path, attachments);
                const glm::mat4 bodyWorld = dynamicBodyWorld(offset, asset.body);
                addSample(std::move(asset), bodyWorld, true);
            };

            addFallingSample("assembled 10mm pistol",
                "Meshes\\Weapons\\10mmPistol\\10MMPistol.nif", {-65.0f, 420.0f, 0.0f}, {
                    "Meshes\\Weapons\\10mmPistol\\10mmGrip.nif",
                    "Meshes\\Weapons\\10mmPistol\\10mmMag01.nif"});
            addFallingSample("plain hknpConvexShape",
                "Meshes\\Architecture\\Buildings\\DecoKit\\DecoMainA1x1WinC01.nif",
                {65.0f, 610.0f, 0.0f});
            addFallingSample("convex and capsule compound",
                "Meshes\\SetDressing\\Greebs\\Greebles01.nif", {0.0f, 800.0f, 25.0f});

            constexpr std::string_view skeletonPath =
                "Meshes\\Actors\\Character\\CharacterAssets\\skeleton.nif";
            auto ragdolls = assets::loadNifRagdolls(archives.extract(skeletonPath));
            if (ragdolls.size() != 1)
                throw std::runtime_error("Human skeleton does not contain one Havok ragdoll");
            auto ragdoll = std::move(ragdolls.front());
            const glm::vec3 ragdollPosition = sampleCenter + glm::vec3(220.0f, 620.0f, 0.0f);
            const auto ragdollHandle = physicsWorld->addRagdoll(
                ragdoll, ragdollPosition);
            try
            {
                PhysicsRagdollSample sample;
                sample.ragdoll = ragdollHandle;
                const auto transforms = physicsWorld->ragdollBodyTransforms(ragdollHandle);
                try
                {
                    constexpr std::string_view bodyPath =
                        "Meshes\\Actors\\Character\\CharacterAssets\\MaleBody.nif";
                    auto bodyGeometry = assets::loadNifModel(archives.extract(bodyPath));
                    if (bodyGeometry.skinBones.empty())
                        throw std::runtime_error("MaleBody.nif contains no skin bones");
                    const auto skinBones = bodyGeometry.skinBones;
                    const auto range = buildSampleMesh(std::move(bodyGeometry), "male body");
                    sample.paletteBodies.reserve(skinBones.size());
                    sample.paletteFromInitialBodies.reserve(skinBones.size());

                    std::vector<glm::vec3> bindPositions;
                    bindPositions.reserve(skinBones.size());
                    for (const auto& skinBone : skinBones)
                        bindPositions.emplace_back(glm::inverse(skinBone.skinToBone)[3]);
                    std::vector<glm::vec3> bodyPositions;
                    bodyPositions.reserve(ragdoll.bones.size());
                    for (std::size_t bone = 0; bone < ragdoll.bones.size(); ++bone)
                        bodyPositions.emplace_back(physics::havokBodyLocalTransform(
                            ragdoll.bodies[ragdoll.boneToBody[bone]])[3]);
                    const auto byX = [](const glm::vec3& left, const glm::vec3& right) {
                        return left.x < right.x;
                    };
                    const auto [skinLeft, skinRight] = std::ranges::minmax_element(
                        bindPositions, byX);
                    const auto [bodyLeft, bodyRight] = std::ranges::minmax_element(
                        bodyPositions, byX);
                    const glm::vec3 skinToRagdoll = ((*bodyLeft - *skinLeft)
                        + (*bodyRight - *skinRight)) * 0.5f;
                    const glm::mat4 initialSkinWorld = glm::translate(
                        glm::mat4(1.0f), ragdollPosition + skinToRagdoll);

                    std::vector<glm::mat4> palette;
                    palette.reserve(skinBones.size());
                    std::set<std::uint32_t> mappedBodies;
                    for (std::size_t skin = 0; skin < skinBones.size(); ++skin)
                    {
                        const glm::vec3 bindPosition = bindPositions[skin] + skinToRagdoll;
                        std::uint32_t closestBone = 0;
                        float closestDistance = std::numeric_limits<float>::max();
                        for (std::uint32_t bone = 0; bone < ragdoll.bones.size(); ++bone)
                        {
                            const glm::vec3 offset = bindPosition - bodyPositions[bone];
                            const float distance = glm::dot(offset, offset);
                            if (distance < closestDistance)
                            {
                                closestDistance = distance;
                                closestBone = bone;
                            }
                        }
                        sample.paletteBodies.push_back(closestBone);
                        mappedBodies.insert(closestBone);
                        sample.paletteFromInitialBodies.push_back(
                            glm::inverse(transforms[closestBone]) * initialSkinWorld);
                        palette.push_back(transforms[closestBone]
                            * sample.paletteFromInitialBodies.back());
                    }
                    if (mappedBodies.size() < ragdoll.bones.size() / 2)
                        throw std::runtime_error("Human skin did not align with the ragdoll bind pose");
                    sample.paletteOffset = renderer.addSkinPalette(palette);
                    render::SceneInstance instance = makeInstance(range, glm::mat4(1.0f));
                    instance.bounds = ragdollPoseBounds(transforms);
                    instance.skinOffset = sample.paletteOffset;
                    instance.skinCount = static_cast<std::uint32_t>(palette.size());
                    sample.skinnedInstance = renderer.addInstance(instance);
                    std::cout << "  human skin: " << skinBones.size()
                              << " render bones mapped onto " << mappedBodies.size()
                              << " ragdoll limbs\n";
                }
                catch (const std::exception& error)
                {
                    std::cerr << "Human skin unavailable; using capsule debug rig: "
                              << error.what() << '\n';
                    render::SceneMaterial ragdollMaterial;
                    ragdollMaterial.baseColor = {0.10f, 0.38f, 0.52f, 1.0f};
                    const std::uint32_t ragdollMaterialIndex =
                        renderer.addMaterial(ragdollMaterial);
                    sample.instances.reserve(ragdoll.bones.size());
                    sample.ranges.reserve(ragdoll.bones.size());
                    for (std::size_t bone = 0; bone < ragdoll.bones.size(); ++bone)
                    {
                        const std::uint32_t body = ragdoll.boneToBody[bone];
                        const auto range = renderer.addMesh(makeCapsuleDebugMesh(
                            ragdoll.bodies[body], ragdollMaterialIndex));
                        sample.ranges.push_back(range);
                        sample.instances.push_back(renderer.addInstance(
                            makeInstance(range, transforms[bone])));
                    }
                }
                physicsRagdollSamples.push_back(std::move(sample));
            }
            catch (...)
            {
                physicsWorld->removeRagdoll(ragdollHandle);
                throw;
            }
            const auto ragdollJointCount = ragdoll.constraints.size();
            std::cout << "  human ragdoll: " << ragdoll.bones.size()
                      << " capsule limbs, " << ragdollJointCount << " joints\n";

            std::cout << "Physics sample: " << physicsSamples.size()
                      << " decoded Fallout 4 bodies and " << physicsRagdollSamples.size()
                      << " ragdoll above LAND cell " << centerCell.x << ", "
                      << centerCell.y << '\n';
            return glm::vec3(-sampleCenter.x, sampleCenter.y + 160.0f,
                sampleCenter.z + 760.0f);
        }

        static int tileOrigin(int coordinate, int level)
        {
            int result = coordinate / level;
            if (coordinate < 0 && coordinate % level != 0)
                --result;
            return result * level;
        }

        static std::optional<LodTile> commonwealthTerrainTile(std::string_view path)
        {
            constexpr std::string_view prefix =
                "meshes\\terrain\\commonwealth\\commonwealth.";
            constexpr std::string_view suffix = ".btr";
            if (!path.starts_with(prefix) || !path.ends_with(suffix))
                return std::nullopt;

            path.remove_prefix(prefix.size());
            path.remove_suffix(suffix.size());
            std::array<int, 3> values{};
            for (std::size_t index = 0; index < values.size(); ++index)
            {
                const auto separator = path.find('.');
                const auto component = path.substr(0, separator);
                const auto [end, error] = std::from_chars(
                    component.data(), component.data() + component.size(), values[index]);
                if (error != std::errc{} || end != component.data() + component.size()
                    || (index + 1 < values.size() && separator == std::string_view::npos))
                    return std::nullopt;
                path = separator == std::string_view::npos
                    ? std::string_view{} : path.substr(separator + 1);
            }
            if (!path.empty() || (values[0] != 4 && values[0] != 8
                    && values[0] != 16 && values[0] != 32))
                return std::nullopt;
            return LodTile{{values[1], values[2]}, values[0]};
        }

        static bool tileIntersectsRadius(
            const LodTile& tile, int radius, assets::CellCoord center)
        {
            const int maximumX = tile.origin.x + tile.level - 1;
            const int maximumY = tile.origin.y + tile.level - 1;
            return maximumX >= center.x - radius && tile.origin.x <= center.x + radius
                && maximumY >= center.y - radius && tile.origin.y <= center.y + radius;
        }

        static bool tileInsideRadius(
            const LodTile& tile, int radius, assets::CellCoord center)
        {
            const int maximumX = tile.origin.x + tile.level - 1;
            const int maximumY = tile.origin.y + tile.level - 1;
            return tile.origin.x >= center.x - radius && maximumX <= center.x + radius
                && tile.origin.y >= center.y - radius && maximumY <= center.y + radius;
        }

        bool tileActive(const LodTile& tile, assets::CellCoord center) const
        {
            if (tile.level < 32)
            {
                const int parentLevel = tile.level * 2;
                const LodTile parent{{tileOrigin(tile.origin.x, parentLevel),
                                          tileOrigin(tile.origin.y, parentLevel)},
                    parentLevel};
                if (!tileIntersectsRadius(parent, parentLevel * 2, center))
                    return false;
            }
            if (tile.level > 4 && tileIntersectsRadius(tile, tile.level * 2, center))
                return false;
            return tile.level > 4 || !tileIntersectsRadius(tile, fullRadius, center)
                || !tileInsideRadius(tile, lodRadius, center);
        }

        bool tileActive(const LodTile& tile) const
        {
            return tileActive(tile, centerCell);
        }

        const LodTile* commonwealthTile(assets::CellCoord cell) const
        {
            const auto found = lodTileByCell.find(cell);
            return found == lodTileByCell.end() ? nullptr : &lodTiles[found->second];
        }

        bool commonwealthObjectsNearby(
            const LodTile& tile, assets::CellCoord center) const
        {
            return tile.level == LodTile::side && tileIntersectsRadius(tile, lodRadius, center);
        }

        static bool commonwealthObjectsDesired(const LodTile& tile)
        {
            return tile.level == LodTile::side;
        }

        void addLoadedDescendants(const LodTile& tile, std::set<std::size_t>& selected) const
        {
            if (tile.level <= LodTile::side)
                return;
            const int childLevel = tile.level / 2;
            for (const int y : {tile.origin.y, tile.origin.y + childLevel})
            {
                for (const int x : {tile.origin.x, tile.origin.x + childLevel})
                {
                    const auto child = lodTileByKey.find({childLevel, x, y});
                    if (child == lodTileByKey.end())
                        continue;
                    const auto& childTile = lodTiles[child->second];
                    if (childTile.terrainState == LodLoadState::loaded)
                        selected.insert(child->second);
                    else
                        addLoadedDescendants(childTile, selected);
                }
            }
        }

        std::set<std::size_t> selectedCommonwealthLod() const
        {
            std::set<std::size_t> selected;
            for (std::size_t index = 0; index < lodTiles.size(); ++index)
            {
                const auto& tile = lodTiles[index];
                if (!tileActive(tile))
                    continue;
                if (tile.terrainState == LodLoadState::loaded)
                {
                    selected.insert(index);
                    continue;
                }

                bool foundAncestor = false;
                for (int level = tile.level * 2; level <= 32; level *= 2)
                {
                    const auto ancestor = lodTileByKey.find({level,
                        tileOrigin(tile.origin.x, level), tileOrigin(tile.origin.y, level)});
                    if (ancestor != lodTileByKey.end()
                        && lodTiles[ancestor->second].terrainState == LodLoadState::loaded)
                    {
                        selected.insert(ancestor->second);
                        foundAncestor = true;
                        break;
                    }
                }
                if (!foundAncestor)
                    addLoadedDescendants(tile, selected);
            }
            std::erase_if(selected, [&](std::size_t index) {
                const auto& tile = lodTiles[index];
                for (int level = tile.level * 2; level <= 32; level *= 2)
                {
                    const auto ancestor = lodTileByKey.find({level,
                        tileOrigin(tile.origin.x, level), tileOrigin(tile.origin.y, level)});
                    if (ancestor != lodTileByKey.end() && selected.contains(ancestor->second))
                        return true;
                }
                return false;
            });
            return selected;
        }

        void refreshCommonwealthLod(render::VulkanRenderer& renderer)
        {
            displayedLodTiles = selectedCommonwealthLod();
            std::vector<std::uint32_t> enable;
            std::vector<std::uint32_t> disable;
            for (std::size_t index = 0; index < lodTiles.size(); ++index)
            {
                const auto& tile = lodTiles[index];
                if (!tile.terrain)
                    continue;
                (displayedLodTiles.contains(index) ? enable : disable).push_back(*tile.terrain);
            }
            renderer.setInstancesActive(disable, false);
            renderer.setInstancesActive(enable, true);
        }

        bool commonwealthTerrainCovers(assets::CellCoord cell) const
        {
            return std::ranges::any_of(displayedLodTiles, [&](std::size_t index) {
                const auto& tile = lodTiles[index];
                return cell.x >= tile.origin.x && cell.x < tile.origin.x + tile.level
                    && cell.y >= tile.origin.y && cell.y < tile.origin.y + tile.level;
            });
        }

        bool commonwealthObjectsCover(assets::CellCoord cell) const
        {
            const auto* tile = commonwealthTile(cell);
            if (!tile || !commonwealthObjectsDesired(*tile)
                || tile->objectsState != LodLoadState::loaded || replacementReady(cell))
                return false;
            const int x = cell.x - tile->origin.x;
            const int y = cell.y - tile->origin.y;
            const auto& parts = tile->objects[y * LodTile::side + x];
            return std::ranges::any_of(parts, [&](const LodTile::ObjectPart& part) {
                return std::ranges::none_of(part.replacementCells,
                    [&](assets::CellCoord replacement) { return replacementReady(replacement); });
            });
        }

        void preloadCommonwealthWater()
        {
            if (world.waterHeights.empty())
                return;
            render::SceneMaterial water;
            water.baseColor = {0.025f, 0.10f, 0.14f, 1.0f};
            water.water = true;
            const auto material = static_cast<std::uint32_t>(initial.materials.size());
            initial.materials.push_back(water);
            const auto range = render::appendMesh(initial.mesh,
                waterCells(world.waterHeights, material));
            initial.instances.push_back(makeInstance(range, glm::mat4(1.0f)));
            const auto [minimum, maximum] = std::ranges::minmax_element(
                world.waterHeights, {}, [](const auto& entry) { return entry.second; });
            std::cout << "Commonwealth water: " << world.waterHeights.size()
                      << " cells, heights " << minimum->second << " to "
                      << maximum->second << '\n';
        }

        bool fullObjectsReady(assets::CellCoord cell) const
        {
            const auto found = cells.find(cell);
            return found != cells.end() && found->second.hasLodReplacements
                && found->second.pendingLodReplacements == 0;
        }

        bool replacementReady(assets::CellCoord cell) const
        {
            return fullCell(cell) && fullObjectsReady(cell);
        }

        render::MeshRange appendInitialModel(PreparedModel prepared, bool terrainLod = false)
        {
            render::PackedMesh packed;
            for (auto& surface : prepared.surfaces)
            {
                const auto material = materials.addPrepared(
                    std::move(surface.material), terrainLod);
                for (auto& meshlet : surface.mesh.meshlets)
                    meshlet.materialIndex = material;
                (void)render::appendMesh(packed, std::move(surface.mesh));
            }
            packed.bounds = prepared.bounds;
            return render::appendMesh(initial.mesh, std::move(packed));
        }

        static void appendMeshlet(render::PackedMesh& destination,
            const render::PackedMesh& source, const render::GpuMeshlet& sourceMeshlet)
        {
            auto meshlet = sourceMeshlet;
            meshlet.vertexOffset = static_cast<std::uint32_t>(destination.vertices.size());
            meshlet.indexOffset = static_cast<std::uint32_t>(destination.indices.size());
            destination.vertices.insert(destination.vertices.end(),
                source.vertices.begin() + sourceMeshlet.vertexOffset,
                source.vertices.begin() + sourceMeshlet.vertexOffset + sourceMeshlet.vertexCount);
            destination.indices.insert(destination.indices.end(),
                source.indices.begin() + sourceMeshlet.indexOffset,
                source.indices.begin() + sourceMeshlet.indexOffset + sourceMeshlet.triangleCount * 3);
            destination.meshlets.push_back(meshlet);
        }

        template <typename AddMaterial>
        static std::array<render::PackedMesh, LodTile::side * LodTile::side>
        packBtoCells(PreparedModel prepared, const LodTile& tile, AddMaterial&& addMaterial)
        {
            std::array<render::PackedMesh, LodTile::side * LodTile::side> cells;
            for (auto& surface : prepared.surfaces)
            {
                const auto material = addMaterial(std::move(surface.material));
                for (auto& meshlet : surface.mesh.meshlets)
                {
                    meshlet.materialIndex = material;
                    const int cellX = static_cast<int>(std::floor(meshlet.sphere.x / cellSize));
                    const int cellY = static_cast<int>(std::floor(meshlet.sphere.z / cellSize));
                    const int localX = std::clamp(cellX - tile.origin.x, 0, LodTile::side - 1);
                    const int localY = std::clamp(cellY - tile.origin.y, 0, LodTile::side - 1);
                    render::PackedMesh packed;
                    appendMeshlet(packed, surface.mesh, meshlet);
                    packed.bounds = meshlet.sphere;
                    (void)render::appendMesh(
                        cells[localY * LodTile::side + localX], std::move(packed));
                }
            }
            return cells;
        }

        void appendInitialObjects(PreparedModel prepared, LodTile& tile)
        {
            auto btoCells = packBtoCells(std::move(prepared), tile,
                [&](PreparedMaterial material) {
                    return materials.addPrepared(std::move(material));
                });
            for (std::size_t index = 0; index < btoCells.size(); ++index)
            {
                if (btoCells[index].meshlets.empty())
                    continue;
                const auto range = render::appendMesh(initial.mesh, std::move(btoCells[index]));
                auto instance = makeInstance(range, glm::mat4(1.0f));
                const auto instanceIndex = static_cast<std::uint32_t>(initial.instances.size());
                tile.objects[index].push_back({instanceIndex, instance.bounds, {}});
                initial.instances.push_back(instance);
            }
        }

        void preloadCommonwealthLod()
        {
            struct Request
            {
                std::size_t tile = 0;
                bool objects = false;
            };
            std::vector<Request> requests;
            constexpr std::string_view terrainPrefix =
                "meshes\\terrain\\commonwealth\\commonwealth.";
            for (const auto& path : archives.files(terrainPrefix, ".btr"))
            {
                const auto tile = commonwealthTerrainTile(path);
                if (!tile)
                    continue;
                const std::size_t tileIndex = lodTiles.size();
                lodTiles.push_back(*tile);
                lodTiles.back().terrainPath = path;
                lodTileByKey[{tile->level, tile->origin.x, tile->origin.y}] = tileIndex;
                if (tile->level == 4)
                {
                    const std::string objects = "meshes\\terrain\\commonwealth\\objects\\commonwealth.4."
                        + std::to_string(tile->origin.x) + "." + std::to_string(tile->origin.y) + ".bto";
                    if (archives.contains(objects))
                    {
                        lodTiles.back().objectsPath = objects;
                        lodTiles.back().objectsState = LodLoadState::discovered;
                    }
                    for (int cellY = tile->origin.y; cellY < tile->origin.y + tile->level; ++cellY)
                        for (int cellX = tile->origin.x; cellX < tile->origin.x + tile->level; ++cellX)
                            lodTileByCell[{cellX, cellY}] = tileIndex;
                }
            }

            for (std::size_t index = 0; index < lodTiles.size(); ++index)
            {
                auto& tile = lodTiles[index];
                if (tileActive(tile))
                {
                    tile.terrainState = LodLoadState::queued;
                    requests.push_back({index, false});
                }
            }
            for (std::size_t index = 0; index < lodTiles.size(); ++index)
            {
                auto& tile = lodTiles[index];
                if (tile.objectsState == LodLoadState::discovered
                    && commonwealthObjectsNearby(tile, centerCell))
                {
                    tile.objectsState = LodLoadState::queued;
                    requests.push_back({index, true});
                }
            }

            std::size_t loaded = 0;
            std::size_t loadedTerrain = 0;
            std::size_t removedWaterTriangles = 0;
            for (std::size_t offset = 0; offset < requests.size(); offset += workerCount)
            {
                struct Job
                {
                    const Request* request = nullptr;
                    std::future<PreparedModel> future;
                };
                std::vector<Job> preloadJobs;
                const std::size_t end = std::min(offset + workerCount, requests.size());
                preloadJobs.reserve(end - offset);
                for (std::size_t index = offset; index < end; ++index)
                {
                    const Request* request = &requests[index];
                    auto& tile = lodTiles[request->tile];
                    (request->objects ? tile.objectsState : tile.terrainState)
                        = LodLoadState::loading;
                    const std::string path = request->objects
                        ? tile.objectsPath : tile.terrainPath;
                    preloadJobs.push_back({request, std::async(std::launch::async,
                        [this, path] { return prepareModel(path, {}, false); })});
                }
                for (auto& job : preloadJobs)
                {
                    auto& tile = lodTiles[job.request->tile];
                    auto& state = job.request->objects ? tile.objectsState : tile.terrainState;
                    try
                    {
                        auto prepared = job.future.get();
                        removedWaterTriangles += prepared.removedWaterTriangles;
                        if (job.request->objects)
                            appendInitialObjects(std::move(prepared), tile);
                        else
                        {
                            const auto range = appendInitialModel(std::move(prepared), true);
                            const glm::mat4 transform = glm::translate(glm::mat4(1.0f),
                                glm::vec3(tile.origin.x * cellSize, 0.0f,
                                    tile.origin.y * cellSize));
                            auto instance = makeInstance(range, transform);
                            instance.active = tileActive(tile);
                            tile.terrain = static_cast<std::uint32_t>(initial.instances.size());
                            initial.instances.push_back(instance);
                            ++loadedTerrain;
                        }
                        state = LodLoadState::loaded;
                        ++loaded;
                    }
                    catch (const std::exception& error)
                    {
                        state = LodLoadState::failed;
                        if (warningCount++ < 20)
                            std::cerr << "Commonwealth LOD skipped: "
                                      << (job.request->objects ? tile.objectsPath : tile.terrainPath)
                                      << ": " << error.what() << '\n';
                    }
                }
            }
            displayedLodTiles = selectedCommonwealthLod();
            for (std::size_t index = 0; index < lodTiles.size(); ++index)
                if (lodTiles[index].terrain)
                    initial.instances[*lodTiles[index].terrain].active
                        = displayedLodTiles.contains(index);
            const auto btoCount = std::ranges::count_if(
                lodTiles, [](const LodTile& tile) { return !tile.objectsPath.empty(); });
            std::cout << "Commonwealth LOD preload: " << loadedTerrain << " active BTRs, "
                      << loaded - loadedTerrain << " / " << btoCount << " BTOs ("
                      << lodTiles.size()
                      << " BTRs discovered, "
                      << removedWaterTriangles << " baked water triangles removed)\n";
        }

        void preloadReferenceLod()
        {
            struct Group
            {
                ModelKey key;
                std::string path;
                std::vector<ReferenceRequest> references;
            };
            std::vector<Group> groups;
            std::map<ModelKey, std::size_t> groupIndices;
            std::size_t scheduledInstances = 0;
            for (int distance = fullRadius + 1; distance <= lodRadius; ++distance)
            {
                for (int y = centerCell.y - distance; y <= centerCell.y + distance; ++y)
                {
                    for (int x = centerCell.x - distance; x <= centerCell.x + distance; ++x)
                    {
                        if (std::max(std::abs(x - centerCell.x), std::abs(y - centerCell.y)) != distance)
                            continue;
                        const auto found = referencesByCell.find({x, y});
                        if (found == referencesByCell.end())
                            continue;
                        for (const std::size_t referenceIndex : found->second)
                        {
                            if (loadedObjects + scheduledInstances >= maxObjects)
                                break;
                            ReferenceRequest request{world.references[referenceIndex], true};
                            if (!desired(request))
                                continue;
                            const auto path = modelPath(request);
                            if (!path || !isRenderableModelPath(*path))
                            {
                                rejectedReferences.insert(referenceKey(request));
                                continue;
                            }
                            const ModelKey key = modelKey(request);
                            auto [group, inserted] = groupIndices.emplace(key, groups.size());
                            if (inserted)
                                groups.push_back({key, *path, {}});
                            groups[group->second].references.push_back(std::move(request));
                            ++scheduledInstances;
                        }
                    }
                }
            }

            std::size_t loadedModels = 0;
            std::size_t loadedInstances = 0;
            for (std::size_t offset = 0; offset < groups.size(); offset += workerCount)
            {
                struct Job
                {
                    Group* group = nullptr;
                    std::future<PreparedModel> future;
                };
                std::vector<Job> preloadJobs;
                const std::size_t end = std::min(offset + workerCount, groups.size());
                preloadJobs.reserve(end - offset);
                for (std::size_t index = offset; index < end; ++index)
                {
                    Group* group = &groups[index];
                    preloadJobs.push_back({group, std::async(std::launch::async,
                        [this, path = group->path, key = group->key] {
                            return prepareModel(path, key, false);
                        })});
                }
                for (auto& job : preloadJobs)
                {
                    auto& state = models[job.group->key];
                    try
                    {
                        state.range = appendInitialModel(job.future.get());
                        ++loadedModels;
                        for (const auto& request : job.group->references)
                        {
                            try
                            {
                                const auto instance = static_cast<std::uint32_t>(initial.instances.size());
                                initial.instances.push_back(makeInstance(
                                    *state.range, referenceTransform(request.reference)));
                                loadedReferences.emplace(referenceKey(request), instance);
                                cells[request.reference.cell].lodObjects.push_back(instance);
                                ++loadedObjects;
                                ++loadedInstances;
                            }
                            catch (const std::exception& error)
                            {
                                rejectedReferences.insert(referenceKey(request));
                                warnReference(request.reference.formId, error);
                            }
                        }
                    }
                    catch (const std::exception& error)
                    {
                        state.failed = true;
                        for (const auto& request : job.group->references)
                            rejectedReferences.insert(referenceKey(request));
                        warnReference(job.group->references.empty()
                            ? 0 : job.group->references.front().reference.formId, error);
                    }
                }
            }
            std::cout << "Reference LOD preload: " << loadedModels << " / " << groups.size()
                      << " model variants, " << loadedInstances << " instances\n";
        }

        bool desired(const ReferenceRequest& request) const
        {
            return request.lod ? lodCell(request.reference.cell)
                    && !commonwealthObjectsCover(request.reference.cell)
                    && modelPath(request).has_value()
                               : fullPrefetchCell(request.reference.cell);
        }

        ReferenceKey referenceKey(const ReferenceRequest& request) const
        {
            return {request.reference.instanceId, request.lod};
        }

        ModelKey modelKey(const ReferenceRequest& request) const
        {
            return {modelVariantKey(request.reference.baseObject,
                selectedMaterialSwap(world, request.reference)), request.lod};
        }

        std::optional<std::string> modelPath(const ReferenceRequest& request) const
        {
            if (request.lod)
            {
                const auto found = world.modelLods.find(request.reference.baseObject);
                if (found == world.modelLods.end())
                    return std::nullopt;
                for (auto lod = found->second.rbegin(); lod != found->second.rend(); ++lod)
                {
                    if (lod->empty())
                        continue;
                    auto path = assetPath(*lod, "meshes\\", ".nif");
                    if (archives.contains(path))
                        return path;
                }
                return std::nullopt;
            }
            const auto model = world.models.find(request.reference.baseObject);
            if (model != world.models.end())
            {
                if (archives.contains(model->second))
                    return model->second;
                auto path = assetPath(model->second, "meshes\\", ".nif");
                if (archives.contains(path))
                    return path;
            }
            const auto lods = world.modelLods.find(request.reference.baseObject);
            if (lods != world.modelLods.end())
            {
                for (const auto& lod : lods->second)
                {
                    if (lod.empty())
                        continue;
                    auto path = assetPath(lod, "meshes\\", ".nif");
                    if (archives.contains(path))
                        return path;
                }
            }
            return std::nullopt;
        }

        int requestPriority(const ReferenceRequest& request) const
        {
            if (request.lod)
                return 3;
            if (request.reference.cell == centerCell)
                return 0;
            return fullCell(request.reference.cell) ? 1 : 2;
        }

        bool requestNearer(const ReferenceRequest& left, const ReferenceRequest& right) const
        {
            const int leftPriority = requestPriority(left);
            const int rightPriority = requestPriority(right);
            if (leftPriority != rightPriority)
                return leftPriority < rightPriority;
            const auto distanceSquared = [&](const ReferenceRequest& request) {
                const double x = request.reference.position[0] - streamPosition.x;
                const double y = request.reference.position[1] - streamPosition.y;
                return x * x + y * y;
            };
            const double leftDistance = distanceSquared(left);
            const double rightDistance = distanceSquared(right);
            return leftDistance != rightDistance
                ? leftDistance < rightDistance
                : left.reference.instanceId < right.reference.instanceId;
        }

        void queueDesiredReferences()
        {
            if (!includeObjects || loadedObjects + queuedReferences.size() >= maxObjects)
                return;
            constexpr std::size_t queuedReferenceLimit = 2048;
            const std::size_t remainingObjects = maxObjects - loadedObjects;
            const std::size_t queueTarget = std::min(queuedReferenceLimit, remainingObjects);
            if (queuedReferences.size() >= queueTarget)
                return;
            const std::size_t available = queueTarget - queuedReferences.size();
            std::vector<ReferenceRequest> candidates;
            for (int y = centerCell.y - lodRadius; y <= centerCell.y + lodRadius; ++y)
            {
                for (int x = centerCell.x - lodRadius; x <= centerCell.x + lodRadius; ++x)
                {
                    const auto found = referencesByCell.find({x, y});
                    if (found == referencesByCell.end())
                        continue;
                    for (const std::size_t index : found->second)
                    {
                        const auto& reference = world.references[index];
                        for (const bool lod : {false, true})
                        {
                            ReferenceRequest request{reference, lod};
                            const auto key = referenceKey(request);
                            if (!desired(request) || loadedReferences.contains(key)
                                || rejectedReferences.contains(key) || queuedReferences.contains(key))
                                continue;
                            candidates.push_back(std::move(request));
                        }
                    }
                }
            }
            const auto nearer = [this](const auto& left, const auto& right) {
                return requestNearer(left, right);
            };
            if (candidates.size() > available)
            {
                std::ranges::nth_element(candidates,
                    candidates.begin() + static_cast<std::ptrdiff_t>(available), nearer);
                candidates.resize(available);
            }
            std::ranges::sort(candidates, nearer);
            for (auto& request : candidates)
            {
                if (loadedObjects + queuedReferences.size() >= maxObjects)
                    break;
                if (queuedReferences.insert(referenceKey(request)).second)
                    referenceRequests.push_back(std::move(request));
            }
        }

        void queueTerrain()
        {
            for (int y = centerCell.y - lodRadius; y <= centerCell.y + lodRadius; ++y)
            {
                for (int x = centerCell.x - lodRadius; x <= centerCell.x + lodRadius; ++x)
                {
                    const assets::CellCoord cell{x, y};
                    if (!world.landscapes.contains(cell))
                        continue;
                    auto& state = cells[cell];
                    if (fullPrefetchCell(cell) && !state.fullTerrain && !state.fullTerrainQueued)
                    {
                        state.fullTerrainQueued = true;
                        terrainRequests.push_front({cell, false});
                    }
                    if (lodCell(cell) && !commonwealthTerrainCovers(cell)
                        && !state.lodTerrain && !state.lodTerrainQueued)
                    {
                        state.lodTerrainQueued = true;
                        terrainRequests.push_back({cell, true});
                    }
                }
            }
        }

        void queueCommonwealthLod()
        {
            for (std::size_t index = 0; index < lodTiles.size(); ++index)
            {
                auto& tile = lodTiles[index];
                if (tileActive(tile) && tile.terrainState == LodLoadState::discovered)
                {
                    tile.terrainState = LodLoadState::queued;
                    lodRequests.push_back({index, false});
                }
            }
            for (std::size_t index = 0; index < lodTiles.size(); ++index)
            {
                auto& tile = lodTiles[index];
                if (commonwealthObjectsNearby(tile, centerCell)
                    && tile.objectsState == LodLoadState::discovered)
                {
                    tile.objectsState = LodLoadState::queued;
                    lodRequests.push_back({index, true});
                }
            }
        }

        void update(render::VulkanRenderer& renderer, const glm::dvec3& playerPosition)
        {
            for (const auto& sample : physicsSamples)
            {
                const glm::mat4 transform = physicsWorld->bodyTransform(sample.body)
                    * glm::inverse(sample.bodyLocal);
                const auto instance = makeInstance(sample.range, transform);
                renderer.updateInstanceTransform(
                    sample.instance, instance.transform, instance.bounds);
            }
            for (const auto& sample : physicsRagdollSamples)
            {
                const auto transforms = physicsWorld->ragdollBodyTransforms(sample.ragdoll);
                if (sample.skinnedInstance)
                {
                    std::vector<glm::mat4> palette;
                    palette.reserve(sample.paletteBodies.size());
                    for (std::size_t bone = 0; bone < sample.paletteBodies.size(); ++bone)
                    {
                        if (sample.paletteBodies[bone] >= transforms.size())
                            throw std::runtime_error("Jolt ragdoll pose size changed");
                        palette.push_back(transforms[sample.paletteBodies[bone]]
                            * sample.paletteFromInitialBodies[bone]);
                    }
                    renderer.updateSkinPalette(sample.paletteOffset, palette);
                    renderer.updateInstanceTransform(*sample.skinnedInstance,
                        glm::mat4(1.0f), ragdollPoseBounds(transforms));
                }
                else
                {
                    if (transforms.size() != sample.instances.size())
                        throw std::runtime_error("Jolt ragdoll pose size changed");
                    for (std::size_t limb = 0; limb < transforms.size(); ++limb)
                    {
                        const auto instance = makeInstance(sample.ranges[limb], transforms[limb]);
                        renderer.updateInstanceTransform(sample.instances[limb],
                            instance.transform, instance.bounds);
                    }
                }
            }
            streamPosition = {-playerPosition.x, playerPosition.z};
            const assets::CellCoord playerCell{
                static_cast<std::int32_t>(std::floor(-playerPosition.x / cellSize)),
                static_cast<std::int32_t>(std::floor(playerPosition.z / cellSize))};
            if (playerCell == centerCell)
                return;
            const assets::CellCoord previousCenter = centerCell;
            centerCell = playerCell;
            std::cout << "Streamer center: " << centerCell.x << ", " << centerCell.y << '\n';
            constexpr int physicsOriginRadius = 32;
            if (physicsWorld && cellDistance(centerCell, physicsOriginCell) >= physicsOriginRadius)
            {
                physicsOriginCell = centerCell;
                physicsWorld->setOrigin(
                    {centerCell.x * cellSize, 0.0f, centerCell.y * cellSize});
            }
            syncTerrainPhysics();

            const auto pruneAndSort = [&](auto& requests) {
                std::erase_if(requests, [&](const ReferenceRequest& request) {
                    if (desired(request))
                        return false;
                    queuedReferences.erase(referenceKey(request));
                    return true;
                });
                std::ranges::sort(requests, [this](const auto& left, const auto& right) {
                    return requestNearer(left, right);
                });
            };
            pruneAndSort(referenceRequests);
            pruneAndSort(readyReferences);
            for (auto& [key, state] : models)
            {
                (void)key;
                pruneAndSort(state.waiters);
            }

            std::set<assets::CellCoord> affectedCells;
            const auto addCells = [&](assets::CellCoord center) {
                for (int y = center.y - lodRadius; y <= center.y + lodRadius; ++y)
                    for (int x = center.x - lodRadius; x <= center.x + lodRadius; ++x)
                        affectedCells.insert({x, y});
            };
            addCells(previousCenter);
            addCells(centerCell);
            for (const assets::CellCoord cell : affectedCells)
            {
                const bool wasPrefetched = cellDistance(cell, previousCenter)
                    <= std::min(fullRadius + 1, lodRadius);
                if (wasPrefetched != fullPrefetchCell(cell))
                    syncCellPhysics(renderer, cell);
            }

            std::vector<std::uint32_t> enable;
            std::vector<std::uint32_t> disable;
            const auto select = [&](const std::optional<std::uint32_t>& instance, bool active) {
                if (instance)
                    (active ? enable : disable).push_back(*instance);
            };
            queueCommonwealthLod();
            refreshCommonwealthLod(renderer);
            for (const auto& tile : lodTiles)
            {
                const bool desired = commonwealthObjectsDesired(tile);
                for (std::size_t index = 0; index < tile.objects.size(); ++index)
                {
                    const assets::CellCoord cell{
                        tile.origin.x + static_cast<int>(index % LodTile::side),
                        tile.origin.y + static_cast<int>(index / LodTile::side)};
                    for (const auto& part : tile.objects[index])
                    {
                        const bool active = desired && !replacementReady(cell)
                            && std::ranges::none_of(
                            part.replacementCells, [&](assets::CellCoord replacement) {
                                return replacementReady(replacement);
                            });
                        (active ? enable : disable).push_back(part.instance);
                    }
                }
            }

            for (const assets::CellCoord cell : affectedCells)
            {
                const auto found = cells.find(cell);
                if (found == cells.end())
                    continue;
                auto& state = found->second;
                const bool full = fullCell(cell);
                const bool lod = lodCell(cell);
                select(state.fullTerrain, full);
                select(state.lodTerrain, lod && !commonwealthTerrainCovers(cell));
                (full ? enable : disable).insert(
                    (full ? enable : disable).end(), state.fullObjects.begin(), state.fullObjects.end());
                const bool lodObjectsActive = (lod || (full && !fullObjectsReady(cell)))
                    && !commonwealthObjectsCover(cell);
                (lodObjectsActive ? enable : disable).insert(
                    (lodObjectsActive ? enable : disable).end(),
                    state.lodObjects.begin(), state.lodObjects.end());
            }
            renderer.setInstancesActive(disable, false);
            renderer.setInstancesActive(enable, true);
            queueTerrain();
            queueDesiredReferences();
        }

        static void rebuildTerrainLodNormals(assets::ModelGeometry& geometry)
        {
            for (auto& surface : geometry.surfaces)
            {
                for (auto& vertex : surface.vertices)
                    vertex.normal = {};
                for (std::size_t index = 0; index + 2 < surface.indices.size(); index += 3)
                {
                    const std::array triangle{
                        surface.indices[index], surface.indices[index + 1], surface.indices[index + 2]};
                    const glm::vec3 normal = -glm::cross(
                        surface.vertices[triangle[1]].position - surface.vertices[triangle[0]].position,
                        surface.vertices[triangle[2]].position - surface.vertices[triangle[0]].position);
                    if (normal.y <= 0.0001f)
                        continue;
                    for (const auto vertex : triangle)
                        surface.vertices[vertex].normal += normal;
                }
                for (auto& vertex : surface.vertices)
                    vertex.normal = glm::length(vertex.normal) > 0.0f
                        ? glm::normalize(vertex.normal) : glm::vec3(0.0f, 1.0f, 0.0f);
            }
        }

        PreparedModel prepareModel(
            const std::string& path, const ModelKey& key, bool prepareCollision) const
        {
            auto extension = std::filesystem::path(path).extension().string();
            std::ranges::transform(extension, extension.begin(), [](unsigned char c) {
                return static_cast<char>(std::tolower(c));
            });
            const bool isNif = extension == ".nif" || extension == ".btr" || extension == ".bto";
            const auto bytes = isNif ? archives.extract(path) : std::vector<std::byte>{};
            assets::ModelGeometry geometry;
            if (isNif)
                geometry = assets::loadNifModel(bytes);
            else if (const auto loose = archives.resolveLoose(path))
                geometry = assets::importModel(*loose).geometry;
            else
                throw std::runtime_error("Imported model source not found: " + path);
            PreparedModel result;
            if (prepareCollision && isNif)
            {
                result.collision.emplace();
                try
                {
                    auto systems = assets::loadNifPhysicsSystems(bytes);
                    for (auto& system : systems)
                        for (auto& body : system.bodies)
                            result.collision->bodies.push_back(std::move(body));
                }
                catch (const std::exception& error)
                {
                    result.collision->error = error.what();
                }
            }
            std::size_t removedWaterTriangles = 0;
            if (path.ends_with(".btr"))
            {
                if (const auto tile = commonwealthTerrainTile(path))
                {
                    constexpr float waterTolerance = 8.0f;
                    for (auto& surface : geometry.surfaces)
                    {
                        std::vector<std::uint32_t> retained;
                        retained.reserve(surface.indices.size());
                        for (std::size_t index = 0; index + 2 < surface.indices.size(); index += 3)
                        {
                            const std::array triangle{
                                surface.indices[index], surface.indices[index + 1],
                                surface.indices[index + 2]};
                            const glm::vec3 center = (surface.vertices[triangle[0]].position
                                + surface.vertices[triangle[1]].position
                                + surface.vertices[triangle[2]].position) / 3.0f;
                            const assets::CellCoord cell{
                                tile->origin.x + static_cast<std::int32_t>(
                                    std::floor(center.x / cellSize)),
                                tile->origin.y + static_cast<std::int32_t>(
                                    std::floor(center.z / cellSize))};
                            const auto water = world.waterHeights.find(cell);
                            const bool bakedWater = water != world.waterHeights.end()
                                && std::ranges::all_of(triangle, [&](std::uint32_t vertex) {
                                    return std::abs(surface.vertices[vertex].position.y
                                        - water->second) <= waterTolerance;
                                });
                            if (bakedWater)
                                ++removedWaterTriangles;
                            else
                                retained.insert(retained.end(), triangle.begin(), triangle.end());
                        }
                        surface.indices = std::move(retained);
                    }
                }
                rebuildTerrainLodNormals(geometry);
            }
            const auto materialSwapId = static_cast<std::uint32_t>(key.variant);
            const auto* materialSwap = findMaterialSwap(world, materialSwapId);
            result.removedWaterTriangles = removedWaterTriangles;
            result.bounds = boundsOf(geometry);
            result.surfaces.reserve(geometry.surfaces.size());
            for (const auto& surface : geometry.surfaces)
            {
                if (surface.vertices.empty() || surface.indices.empty())
                    continue;
                result.surfaces.push_back({
                    render::buildMeshlets(surface.vertices, surface.indices),
                    isNif ? materials.prepare(surface.materialPath, surface.diffuseTexture,
                        surface.normalTexture, surface.alphaCutoff, materialSwap, surface.decal)
                        : materials.prepareSurface(surface,
                            path + "/" + std::to_string(result.surfaces.size()), materialSwap)});
            }
            if (result.surfaces.empty())
                throw std::runtime_error("NIF contains no streamable surfaces");
            return result;
        }

        std::optional<std::size_t> closestBackgroundBto() const
        {
            std::optional<std::size_t> closest;
            int closestDistance = std::numeric_limits<int>::max();
            for (std::size_t index = 0; index < lodTiles.size(); ++index)
            {
                const auto& tile = lodTiles[index];
                if (tile.objectsState != LodLoadState::discovered)
                    continue;
                const int maximumX = tile.origin.x + tile.level - 1;
                const int maximumY = tile.origin.y + tile.level - 1;
                const int distanceX = std::max({tile.origin.x - centerCell.x,
                    centerCell.x - maximumX, 0});
                const int distanceY = std::max({tile.origin.y - centerCell.y,
                    centerCell.y - maximumY, 0});
                const int distance = std::max(distanceX, distanceY);
                if (distance < closestDistance)
                {
                    closest = index;
                    closestDistance = distance;
                }
            }
            return closest;
        }

        void scheduleJobs(render::VulkanRenderer& renderer)
        {
            const auto startLodJob = [&](LodRequest request) {
                auto& tile = lodTiles[request.tile];
                auto& state = request.objects ? tile.objectsState : tile.terrainState;
                if (state != LodLoadState::queued)
                    return;
                state = LodLoadState::loading;
                const std::string path = request.objects
                    ? tile.objectsPath : tile.terrainPath;
                lodJobs.push_back({request, std::async(std::launch::async,
                    [this, path] { return prepareModel(path, {}, false); })});
            };
            while (jobs.size() + lodJobs.size() < workerCount && !lodRequests.empty())
            {
                const LodRequest request = lodRequests.front();
                lodRequests.pop_front();
                startLodJob(request);
            }
            const std::size_t referenceLimit = closestBackgroundBto() && workerCount > 1
                ? workerCount - 1 : workerCount;
            while (jobs.size() + lodJobs.size() < referenceLimit && !referenceRequests.empty())
            {
                const auto request = referenceRequests.front();
                referenceRequests.pop_front();
                const auto referenceKeyValue = referenceKey(request);
                if (!desired(request))
                {
                    queuedReferences.erase(referenceKeyValue);
                    continue;
                }
                const auto key = modelKey(request);
                auto& state = models[key];
                if (state.range)
                {
                    readyReferences.push_back(request);
                    continue;
                }
                if (state.failed)
                {
                    queuedReferences.erase(referenceKeyValue);
                    rejectReference(renderer, request);
                    continue;
                }
                state.waiters.push_back(request);
                if (state.loading)
                    continue;
                const auto path = modelPath(request);
                if (!path)
                {
                    state.failed = true;
                    for (const auto& waiter : state.waiters)
                    {
                        queuedReferences.erase(referenceKey(waiter));
                        rejectReference(renderer, waiter);
                    }
                    state.waiters.clear();
                    continue;
                }
                if (!isRenderableModelPath(*path))
                {
                    state.failed = true;
                    for (const auto& waiter : state.waiters)
                    {
                        queuedReferences.erase(referenceKey(waiter));
                        rejectReference(renderer, waiter);
                    }
                    state.waiters.clear();
                    continue;
                }
                state.loading = true;
                bool preparesCollision = false;
                if (!request.lod)
                {
                    auto [collision, inserted] = collisions.try_emplace(*path);
                    if (inserted)
                    {
                        collision->second.loading = true;
                        preparesCollision = true;
                    }
                }
                jobs.push_back({key, *path, preparesCollision,
                    std::async(std::launch::async,
                        [this, path = *path, key, preparesCollision] {
                            return prepareModel(path, key, preparesCollision);
                        })});
            }
            while (jobs.size() + lodJobs.size() < workerCount)
            {
                const auto tile = closestBackgroundBto();
                if (!tile)
                    break;
                lodTiles[*tile].objectsState = LodLoadState::queued;
                startLodJob({*tile, true});
            }
        }

        bool completeLod(render::VulkanRenderer& renderer)
        {
            for (std::size_t index = 0; index < lodJobs.size(); ++index)
            {
                if (lodJobs[index].future.wait_for(std::chrono::seconds(0))
                    != std::future_status::ready)
                    continue;
                const LodRequest request = lodJobs[index].request;
                auto& tile = lodTiles[request.tile];
                auto& state = request.objects ? tile.objectsState : tile.terrainState;
                const std::string path = request.objects
                    ? tile.objectsPath : tile.terrainPath;
                try
                {
                    auto prepared = lodJobs[index].future.get();
                    if (request.objects)
                    {
                        auto btoCells = packBtoCells(std::move(prepared), tile,
                            [&](PreparedMaterial material) {
                                return materials.addPreparedDynamic(
                                    renderer, std::move(material));
                            });
                        for (std::size_t cell = 0; cell < btoCells.size(); ++cell)
                        {
                            if (btoCells[cell].meshlets.empty())
                                continue;
                            const auto range = renderer.addMesh(std::move(btoCells[cell]));
                            auto sceneInstance = makeInstance(range, glm::mat4(1.0f));
                            const assets::CellCoord coveredCell{
                                tile.origin.x + static_cast<int>(cell % LodTile::side),
                                tile.origin.y + static_cast<int>(cell / LodTile::side)};
                            sceneInstance.active = commonwealthObjectsDesired(tile)
                                && !replacementReady(coveredCell);
                            const auto instance = renderer.addInstance(sceneInstance);
                            tile.objects[cell].push_back(
                                {instance, sceneInstance.bounds, {}});
                        }
                        state = LodLoadState::loaded;
                        for (const auto& [replacementCell, bounds] : btoReplacementBounds)
                            for (const auto& bound : bounds)
                                replaceBtoParts(renderer, bound, replacementCell);
                    }
                    else
                    {
                        render::PackedMesh packed;
                        for (auto& surface : prepared.surfaces)
                        {
                            const auto material = materials.addPreparedDynamic(
                                renderer, std::move(surface.material), true);
                            for (auto& meshlet : surface.mesh.meshlets)
                                meshlet.materialIndex = material;
                            (void)render::appendMesh(packed, std::move(surface.mesh));
                        }
                        packed.bounds = prepared.bounds;
                        const auto range = renderer.addMesh(std::move(packed));
                        const glm::mat4 transform = glm::translate(glm::mat4(1.0f),
                            glm::vec3(tile.origin.x * cellSize, 0.0f,
                                tile.origin.y * cellSize));
                        auto sceneInstance = makeInstance(range, transform);
                        sceneInstance.active = false;
                        tile.terrain = renderer.addInstance(sceneInstance);
                        state = LodLoadState::loaded;
                        refreshCommonwealthLod(renderer);
                    }
                }
                catch (const std::exception& error)
                {
                    state = LodLoadState::failed;
                    if (warningCount++ < 20)
                        std::cerr << "Commonwealth LOD skipped: " << path
                                  << ": " << error.what() << '\n';
                }
                lodJobs[index] = std::move(lodJobs.back());
                lodJobs.pop_back();
                return true;
            }
            return false;
        }

        bool completeModel(render::VulkanRenderer& renderer)
        {
            for (std::size_t index = 0; index < jobs.size(); ++index)
            {
                if (jobs[index].future.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
                    continue;
                const ModelKey key = jobs[index].key;
                const std::string path = jobs[index].path;
                const bool preparesCollision = jobs[index].preparesCollision;
                auto& state = models[key];
                state.loading = false;
                try
                {
                    auto prepared = jobs[index].future.get();
                    if (prepared.collision)
                        completeCollision(renderer, path, std::move(*prepared.collision));
                    if (state.waiters.empty())
                    {
                        jobs[index] = std::move(jobs.back());
                        jobs.pop_back();
                        return true;
                    }
                    render::PackedMesh packed;
                    for (auto& surface : prepared.surfaces)
                    {
                        const auto material = materials.addPreparedDynamic(
                            renderer, std::move(surface.material));
                        for (auto& meshlet : surface.mesh.meshlets)
                            meshlet.materialIndex = material;
                        (void)render::appendMesh(packed, std::move(surface.mesh));
                    }
                    packed.bounds = prepared.bounds;
                    state.range = renderer.addMesh(std::move(packed));
                    for (const auto& waiter : state.waiters)
                    {
                        if (desired(waiter))
                            readyReferences.push_back(waiter);
                        else
                            queuedReferences.erase(referenceKey(waiter));
                    }
                    std::ranges::sort(readyReferences,
                        [this](const auto& left, const auto& right) {
                            return requestNearer(left, right);
                        });
                }
                catch (const std::exception& error)
                {
                    if (preparesCollision)
                        failCollision(renderer, path, error.what());
                    state.failed = true;
                    for (const auto& waiter : state.waiters)
                    {
                        queuedReferences.erase(referenceKey(waiter));
                        rejectReference(renderer, waiter);
                    }
                    warnReference(state.waiters.empty() ? 0
                        : state.waiters.front().reference.formId, error);
                }
                state.waiters.clear();
                jobs[index] = std::move(jobs.back());
                jobs.pop_back();
                return true;
            }
            return false;
        }

        bool installTerrain(render::VulkanRenderer& renderer)
        {
            while (!terrainRequests.empty())
            {
                const TerrainRequest request = terrainRequests.front();
                terrainRequests.pop_front();
                auto& state = cells[request.cell];
                (request.lod ? state.lodTerrainQueued : state.fullTerrainQueued) = false;
                if ((request.lod && (!lodCell(request.cell)
                        || commonwealthTerrainCovers(request.cell)))
                    || (!request.lod && !fullPrefetchCell(request.cell)))
                    continue;
                const auto landscape = world.landscapes.find(request.cell);
                if (landscape == world.landscapes.end())
                    continue;
                const auto surface = terrainSurface(
                    landscape->second, world, materials, &renderer);
                const auto range = renderer.addMesh(
                    terrainCell(landscape->second, surface, request.lod ? 4 : 1));
                auto sceneInstance = makeInstance(range, glm::mat4(1.0f));
                sceneInstance.active = request.lod ? lodCell(request.cell)
                    : fullCell(request.cell);
                const auto instance = renderer.addInstance(sceneInstance);
                (request.lod ? state.lodTerrain : state.fullTerrain) = instance;
                return true;
            }
            return false;
        }

        void completeBtoHandoff(
            render::VulkanRenderer& renderer, assets::CellCoord replacementCell)
        {
            if (!replacementReady(replacementCell))
                return;
            if (!completedBtoHandoffs.insert(replacementCell).second)
                return;
            std::vector<std::uint32_t> instances;
            if (const auto tile = lodTileByCell.find(replacementCell);
                tile != lodTileByCell.end())
            {
                const auto& lodTile = lodTiles[tile->second];
                const int localX = replacementCell.x - lodTile.origin.x;
                const int localY = replacementCell.y - lodTile.origin.y;
                for (const auto& part : lodTile.objects[localY * LodTile::side + localX])
                    instances.push_back(part.instance);
            }
            if (const auto parts = btoPartsByReplacementCell.find(replacementCell);
                parts != btoPartsByReplacementCell.end())
                instances.insert(instances.end(), parts->second.begin(), parts->second.end());
            renderer.setInstancesActive(instances, false);
        }

        void rejectReference(
            render::VulkanRenderer& renderer, const ReferenceRequest& request)
        {
            (void)renderer;
            if (!rejectedReferences.insert(referenceKey(request)).second)
                return;
        }

        void replaceBtoParts(render::VulkanRenderer& renderer, const glm::vec4& bounds,
            assets::CellCoord replacementCell)
        {
            completedBtoHandoffs.erase(replacementCell);
            const int firstX = static_cast<int>(std::floor((bounds.x - bounds.w) / cellSize)) - 1;
            const int lastX = static_cast<int>(std::floor((bounds.x + bounds.w) / cellSize)) + 1;
            const int firstY = static_cast<int>(std::floor((bounds.z - bounds.w) / cellSize)) - 1;
            const int lastY = static_cast<int>(std::floor((bounds.z + bounds.w) / cellSize)) + 1;
            for (int y = firstY; y <= lastY; ++y)
            {
                for (int x = firstX; x <= lastX; ++x)
                {
                    const assets::CellCoord cell{x, y};
                    const auto found = lodTileByCell.find(cell);
                    if (found == lodTileByCell.end())
                        continue;
                    auto& tile = lodTiles[found->second];
                    const int localX = x - tile.origin.x;
                    const int localY = y - tile.origin.y;
                    for (auto& part : tile.objects[localY * LodTile::side + localX])
                    {
                        if (glm::distance(glm::vec3(bounds), glm::vec3(part.bounds))
                            > bounds.w + part.bounds.w + 64.0f)
                            continue;
                        if (std::ranges::find(part.replacementCells, replacementCell)
                            == part.replacementCells.end())
                        {
                            part.replacementCells.push_back(replacementCell);
                            btoPartsByReplacementCell[replacementCell].push_back(part.instance);
                            btoReplacementCells[part.instance].push_back(replacementCell);
                        }
                    }
                }
            }

            completeBtoHandoff(renderer, replacementCell);
        }

        bool installReference(render::VulkanRenderer& renderer)
        {
            while (!readyReferences.empty())
            {
                const auto request = readyReferences.front();
                readyReferences.pop_front();
                const auto key = referenceKey(request);
                if (!desired(request) || loadedReferences.contains(key))
                {
                    queuedReferences.erase(key);
                    continue;
                }
                const auto model = models.find(modelKey(request));
                if (model == models.end() || !model->second.range)
                {
                    queuedReferences.erase(key);
                    rejectReference(renderer, request);
                    continue;
                }
                const auto path = request.lod ? std::optional<std::string>{} : modelPath(request);
                if (!request.lod && !path)
                {
                    queuedReferences.erase(key);
                    rejectReference(renderer, request);
                    continue;
                }
                auto sceneInstance = makeInstance(
                    *model->second.range, referenceTransform(request.reference));
                sceneInstance.active = request.lod ? lodCell(request.reference.cell)
                    && !commonwealthObjectsCover(request.reference.cell)
                                                   : fullCell(request.reference.cell);
                const auto instance = renderer.addInstance(sceneInstance);
                loadedReferences.emplace(key, instance);
                auto& state = cells[request.reference.cell];
                (request.lod ? state.lodObjects : state.fullObjects).push_back(instance);
                queuedReferences.erase(key);
                ++loadedObjects;
                if (!request.lod && request.reference.replacesDistantLod)
                {
                    if (state.pendingLodReplacements > 0)
                        --state.pendingLodReplacements;
                    btoReplacementBounds[request.reference.cell].push_back(sceneInstance.bounds);
                    replaceBtoParts(renderer, sceneInstance.bounds, request.reference.cell);
                }
                if (!request.lod)
                {
                    state.fullReferenceIds.push_back(request.reference.instanceId);
                    referencesByCollisionPath[*path].push_back(request.reference.instanceId);
                    fullReferencePhysics.emplace(request.reference.instanceId,
                        FullReferencePhysics{request.reference, *path});
                    syncReferencePhysics(renderer, request.reference.instanceId);
                }
                if (!request.lod && fullCell(request.reference.cell)
                    && fullObjectsReady(request.reference.cell))
                {
                    renderer.setInstancesActive(state.lodObjects, false);
                }
                return true;
            }
            return false;
        }

        void warnReference(std::uint32_t formId, const std::exception& error)
        {
            if (warningCount++ < 20)
                std::cerr << "Streamed object skipped (" << std::hex << formId
                          << std::dec << "): " << error.what() << '\n';
            else if (warningCount == 21)
                std::cerr << "Further streamed-object messages suppressed\n";
        }

        bool streamNext(render::VulkanRenderer& renderer)
        {
            scheduleJobs(renderer);
            if (removeStalePhysics())
                return true;
            if (installTerrainPhysics())
                return true;
            if (completeLod(renderer))
                return true;
            if (installTerrain(renderer))
                return true;
            if (completeModel(renderer))
                return true;
            if (installReference(renderer))
                return true;
            if (installReferencePhysics(renderer))
                return true;
            if (lodRequests.empty() && lodJobs.empty() && referenceRequests.empty()
                && readyReferences.empty() && jobs.empty() && physicsReferences.empty())
            {
                queueCommonwealthLod();
                queueDesiredReferences();
                scheduleJobs(renderer);
            }
            return false;
        }
    };

    struct ObjectScene::Impl
    {
        struct Resources
        {
            WorldScene scratch;
            Archives archives;
            Materials materials;
            Resources(const std::filesystem::path& data, const std::filesystem::path& master)
                : archives(data, master), materials(archives, scratch) {}
        };
        struct Surface
        {
            render::PackedMesh mesh;
            PreparedMaterial material;
        };
        struct Prepared
        {
            std::vector<Surface> surfaces;
            std::vector<std::string> warnings;
        };
        struct Model
        {
            std::shared_ptr<Resources> resources;
            std::future<Prepared> job;
            std::optional<render::MeshRange> range;
            bool failed = false;
        };
        struct Instance
        {
            std::uint32_t id = 0;
            std::string model;
            glm::mat4 transform{1.0f};
            glm::vec4 bounds{};
            bool active = false;
        };
        std::filesystem::path data;
        std::filesystem::path master;
        std::map<std::filesystem::path, std::shared_ptr<Resources>> resources;
        std::map<std::string, Model> models;
        std::map<std::uint32_t, Instance> instances;
        std::vector<std::string> messages;
        std::size_t pending = 0;
        std::size_t active = 0;

        Impl(std::filesystem::path dataPath, std::filesystem::path masterPath)
            : data(std::move(dataPath)), master(std::move(masterPath)) {}

        void warn(const std::string& message)
        {
            std::cerr << "Object stream: " << message << '\n';
            if (messages.size() == 100)
                messages.erase(messages.begin());
            messages.push_back(message);
        }

        void update(render::VulkanRenderer& renderer,
            std::span<const assets::PlacedReference> references,
            const std::map<std::uint32_t, std::string>& paths,
            const std::unordered_map<std::uint32_t, std::filesystem::path>& sources,
            const glm::dvec3& camera, std::span<const std::uint32_t> hidden)
        {
            std::size_t running = 0;
            bool installed = false;
            for (auto& [key, model] : models)
            {
                if (!model.job.valid())
                    continue;
                if (installed || model.job.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
                {
                    ++running;
                    continue;
                }
                installed = true;
                try
                {
                    auto prepared = model.job.get();
                    render::PackedMesh combined;
                    for (auto& surface : prepared.surfaces)
                    {
                        const auto material = model.resources->materials.addPreparedDynamic(
                            renderer, std::move(surface.material));
                        for (auto& meshlet : surface.mesh.meshlets)
                            meshlet.materialIndex = material;
                        (void)render::appendMesh(combined, std::move(surface.mesh));
                    }
                    model.range = renderer.addMesh(std::move(combined));
                    for (const auto& warning : prepared.warnings)
                        warn(warning);
                }
                catch (const std::exception& error)
                {
                    model.failed = true;
                    warn(key + ": " + error.what());
                }
            }
            const assets::CellCoord center{
                static_cast<std::int32_t>(std::floor(-camera.x / cellSize)),
                static_cast<std::int32_t>(std::floor(camera.z / cellSize))};
            const std::set<std::uint32_t> hiddenSet(hidden.begin(), hidden.end());
            std::set<std::uint32_t> visible;
            pending = active = 0;
            const std::size_t workerLimit = std::clamp<std::size_t>(
                std::max(1u, std::thread::hardware_concurrency()) / 2u, 2u, 16u);
            for (const auto& reference : references)
            {
                if (hiddenSet.contains(reference.formId)
                    || std::abs(static_cast<std::int64_t>(reference.cell.x) - center.x) > 4
                    || std::abs(static_cast<std::int64_t>(reference.cell.y) - center.y) > 4)
                    continue;
                const auto path = paths.find(reference.baseObject);
                if (path == paths.end())
                    continue;
                const auto source = sources.find(reference.baseObject);
                const auto& sourcePath = source == sources.end() ? master : source->second;
                const auto key = sourcePath.generic_string() + ":" + path->second;
                auto& model = models[key];
                if (!model.resources)
                {
                    auto& pack = resources[sourcePath];
                    if (!pack)
                        pack = std::make_shared<Resources>(data, sourcePath);
                    model.resources = pack;
                }
                if (model.failed)
                    continue;
                if (!model.range)
                {
                    ++pending;
                    if (!model.job.valid() && running < workerLimit)
                    {
                        ++running;
                        model.job = std::async(std::launch::async,
                            [pack = model.resources, relative = path->second, key] {
                                Prepared result;
                                assets::ModelGeometry geometry;
                                std::string extension = std::filesystem::path(relative).extension().string();
                                std::ranges::transform(extension, extension.begin(), [](unsigned char c) {
                                    return static_cast<char>(std::tolower(c));
                                });
                                if (extension == ".nif" || extension == ".bto" || extension == ".btr")
                                {
                                    const auto nifPath = pack->archives.contains(relative) ? relative
                                        : assetPath(relative, "meshes\\", ".nif");
                                    geometry = assets::loadNifModel(pack->archives.extract(nifPath));
                                }
                                else
                                {
                                    const auto loose = pack->archives.resolveLoose(relative);
                                    if (!loose)
                                        throw std::runtime_error("Imported model source is missing: " + relative);
                                    auto imported = assets::importModel(*loose);
                                    geometry = std::move(imported.geometry);
                                    result.warnings = std::move(imported.warnings);
                                }
                                for (std::size_t i = 0; i < geometry.surfaces.size(); ++i)
                                {
                                    const auto& surface = geometry.surfaces[i];
                                    if (surface.vertices.empty() || surface.indices.empty())
                                        continue;
                                    result.surfaces.push_back({render::buildMeshlets(
                                        surface.vertices, surface.indices), pack->materials.prepareSurface(
                                            surface, key + "/" + std::to_string(i))});
                                }
                                if (result.surfaces.empty())
                                    throw std::runtime_error("Model contains no renderable triangles");
                                return result;
                            });
                    }
                    continue;
                }
                const auto transform = referenceTransform(reference);
                const auto projected = makeInstance(*model.range, transform);
                auto found = instances.find(reference.formId);
                if (found != instances.end() && found->second.model != key)
                {
                    const std::array ids{found->second.id};
                    renderer.setInstancesActive(ids, false);
                    instances.erase(found);
                    found = instances.end();
                }
                if (found == instances.end())
                {
                    found = instances.emplace(reference.formId, Instance{
                        renderer.addInstance(projected), key, transform, projected.bounds, true}).first;
                }
                auto& instance = found->second;
                if (instance.transform != transform)
                {
                    renderer.updateInstanceTransform(instance.id, transform, projected.bounds);
                    instance.transform = transform;
                    instance.bounds = projected.bounds;
                }
                visible.insert(reference.formId);
                ++active;
            }
            for (auto& [formId, instance] : instances)
            {
                const bool wanted = visible.contains(formId);
                if (instance.active != wanted)
                {
                    const std::array ids{instance.id};
                    renderer.setInstancesActive(ids, wanted);
                    instance.active = wanted;
                }
            }
        }
    };

    ObjectScene::ObjectScene(const std::filesystem::path& dataDirectory,
        const std::filesystem::path& masterPath)
        : mImpl(std::make_unique<Impl>(dataDirectory, masterPath)) {}
    ObjectScene::~ObjectScene() = default;
    void ObjectScene::update(render::VulkanRenderer& renderer,
        std::span<const assets::PlacedReference> references,
        const std::map<std::uint32_t, std::string>& models,
        const std::unordered_map<std::uint32_t, std::filesystem::path>& sources,
        const glm::dvec3& camera, std::span<const std::uint32_t> hidden)
    {
        mImpl->update(renderer, references, models, sources, camera, hidden);
    }
    std::optional<glm::vec4> ObjectScene::bounds(std::uint32_t formId) const
    {
        const auto found = mImpl->instances.find(formId);
        return found == mImpl->instances.end() || !found->second.active
            ? std::nullopt : std::optional(found->second.bounds);
    }
    std::size_t ObjectScene::pending() const noexcept { return mImpl->pending; }
    std::size_t ObjectScene::loaded() const noexcept { return mImpl->active; }
    const std::vector<std::string>& ObjectScene::diagnostics() const noexcept { return mImpl->messages; }

    struct LandscapeEditor::Impl
    {
        static constexpr std::int32_t demCellRadius = 4;
        // Seven level-32 tiles per axis guarantee the full 92-cell horizon fade even
        // when the camera is at the edge of its center tile.
        static constexpr std::int32_t demBtrRadius = 3;
        // A camera-following plane this wide reaches the sub-pixel horizon with an
        // infinite projection while remaining comfortably inside mesh-coordinate limits.
        static constexpr float demWaterHalfExtent = cellSize * 512.0f;

        struct DemBtrTile
        {
            std::filesystem::path path;
            std::optional<std::uint32_t> instance;
            bool queued = false;
            bool failed = false;
        };

        struct PreparedDemCells
        {
            std::vector<assets::CellCoord> requested;
            std::vector<assets::DemLandscapeCell> cells;
            std::string error;
        };

        struct PreparedDemBtr
        {
            assets::CellCoord origin{};
            std::optional<render::PackedMesh> mesh;
            std::string error;
        };

        static void insertCell(assets::EditableLandscape& document,
            const assets::DemLandscapeCell& imported)
        {
            const assets::CellCoord coord = imported.landscape.cell;
            if (!document.cellForms.emplace(coord, imported.forms).second)
                return;
            const auto hasOwner = [&](assets::EditableLandscape::SampleCoord sample) {
                for (std::int32_t ownerY = coord.y - 1; ownerY <= coord.y + 1; ++ownerY)
                {
                    for (std::int32_t ownerX = coord.x - 1; ownerX <= coord.x + 1; ++ownerX)
                    {
                        const assets::CellCoord owner{ownerX, ownerY};
                        if (owner == coord || !document.containsCell(owner))
                            continue;
                        if (sample.x >= owner.x * 32 && sample.x <= owner.x * 32 + 32
                            && sample.y >= owner.y * 32 && sample.y <= owner.y * 32 + 32)
                            return true;
                    }
                }
                return false;
            };
            for (std::size_t y = 0; y < assets::LandscapeCell::side; ++y)
            {
                for (std::size_t x = 0; x < assets::LandscapeCell::side; ++x)
                {
                    const assets::EditableLandscape::SampleCoord sample{
                        coord.x * 32 + static_cast<std::int32_t>(x),
                        coord.y * 32 + static_cast<std::int32_t>(y)};
                    if (!hasOwner(sample))
                        document.setSample(sample,
                            imported.landscape.heights[y * assets::LandscapeCell::side + x]);
                }
            }
        }

        static std::unique_ptr<assets::DemLandscapeProject> openDemProject(
            const std::filesystem::path& path)
        {
            return assets::DemLandscapeProject::recognizes(path)
                ? std::make_unique<assets::DemLandscapeProject>(path) : nullptr;
        }

        static assets::EditableLandscape openDocument(const std::filesystem::path& path,
            assets::DemLandscapeProject* demProject)
        {
            if (demProject)
            {
                assets::EditableLandscape result;
                result.nextFormId = demProject->nextFormId();
                for (std::int32_t y = -demCellRadius; y <= demCellRadius; ++y)
                    for (std::int32_t x = -demCellRadius; x <= demCellRadius; ++x)
                        if (const auto cell = demProject->cell({x, y}))
                            insertCell(result, *cell);
                if (!result.containsCell({0, 0}))
                    throw std::runtime_error("DEM project does not cover Times Square CELL 0, 0");
                return result;
            }
            if (std::filesystem::exists(path))
                return assets::loadEditableLandscape(path);
            auto created = assets::createEditableLandscape();
            assets::saveEditableLandscape(path, created);
            return created;
        }

        std::filesystem::path path;
        std::unique_ptr<assets::DemLandscapeProject> demProject;
        assets::EditableLandscape document;
        assets::EditableLandscape saved;
        assets::ExteriorWorld materialWorld;
        WorldScene initial;
        Archives archives;
        Materials materials;
        std::map<assets::CellCoord, TerrainSurface> surfaces;
        std::map<assets::CellCoord, render::MeshRange> terrainRanges;
        std::map<assets::CellCoord, std::uint32_t> terrainInstances;
        std::set<assets::CellCoord> activeCells;
        std::map<assets::CellCoord, DemBtrTile> demBtrTiles;
        std::set<assets::CellCoord> activeDemBtrs;
        std::deque<assets::CellCoord> demCellQueue;
        std::set<assets::CellCoord> queuedDemCells;
        std::set<assets::CellCoord> unavailableDemCells;
        std::deque<assets::CellCoord> demBtrQueue;
        mutable std::mutex demIoMutex;
        std::future<PreparedDemCells> demCellJob;
        std::future<PreparedDemBtr> demBtrJob;
        std::set<assets::CellCoord> dirtyCells;
        std::optional<assets::CellCoord> activeCenter;
        render::MeshRange brushRange;
        render::MeshRange waterRange;
        std::uint32_t brushInstance = 0;
        std::optional<std::uint32_t> waterInstance;
        std::uint32_t brushMaterial = 0;
        std::uint32_t waterMaterial = 0;
        detail::LandscapeHistory history;
        // ponytail: DEM undo snapshots cover the visited-cell cache, not the eleven-million-cell
        // source. Replace this with sparse per-stroke deltas only if the cache becomes costly.
        LandscapeTool tool = LandscapeTool::raise;
        float radius = 512.0f;
        float strength = 256.0f;
        float flattenHeight = 0.0f;
        bool initialTaken = false;
        bool strokeActive = false;
        bool brushVisible = false;
        bool brushEnabled = true;
        bool externalSave = false;
        bool readOnly = false;
        assets::EditableLandscape strokeBefore;
        glm::vec2 brushCenter{std::numeric_limits<float>::max()};
        float displayedRadius = 0.0f;

        [[nodiscard]] static std::int32_t demBtrTileOrigin(std::int32_t coordinate)
        {
            std::int32_t tile = coordinate / 32;
            if (coordinate < 0 && coordinate % 32 != 0)
                --tile;
            return tile * 32;
        }

        void discoverDemBtrs()
        {
            const std::filesystem::path directory = path.parent_path()
                / (path.stem().string() + ".btr");
            if (!std::filesystem::is_directory(directory))
                return;
            for (const auto& entry : std::filesystem::recursive_directory_iterator(directory,
                     std::filesystem::directory_options::skip_permission_denied))
            {
                if (!entry.is_regular_file())
                    continue;
                if (const auto origin = detail::demBtrOrigin(entry.path()))
                {
                    const auto [tile, inserted] = demBtrTiles.try_emplace(*origin, DemBtrTile{entry.path()});
                    if (!inserted && detail::preferDemBtr(entry.path(), tile->second.path))
                        tile->second = DemBtrTile{entry.path()};
                }
            }
        }

        [[nodiscard]] std::optional<assets::DemLandscapeCell> loadDemCell(
            assets::CellCoord coord) const
        {
            std::scoped_lock lock(demIoMutex);
            return demProject->cell(coord);
        }

        [[nodiscard]] render::PackedMesh loadDemBtrMesh(
            const std::filesystem::path& btrPath, std::uint32_t material) const
        {
            std::ifstream input(btrPath, std::ios::binary | std::ios::ate);
            if (!input)
                throw std::runtime_error("Could not open DEM BTR: " + btrPath.string());
            const std::streamoff byteCount = input.tellg();
            if (byteCount <= 0)
                throw std::runtime_error("DEM BTR is empty: " + btrPath.string());
            std::vector<std::byte> bytes(static_cast<std::size_t>(byteCount));
            input.seekg(0);
            input.read(reinterpret_cast<char*>(bytes.data()), byteCount);
            if (!input)
                throw std::runtime_error("Could not read DEM BTR: " + btrPath.string());
            auto geometry = assets::loadNifModel(bytes);
            std::vector<float> fallbackHeights;
            const bool needsHeightFallback = std::ranges::any_of(geometry.surfaces,
                [](const assets::ModelSurface& surface) {
                    return std::ranges::any_of(surface.vertices,
                        [](const render::SourceVertex& vertex) {
                            return !std::isfinite(vertex.position.y);
                        });
                });
            if (needsHeightFallback)
            {
                const auto origin = detail::demBtrOrigin(btrPath);
                if (!origin || !demProject)
                    throw std::runtime_error("DEM BTR height overflow has no LAND fallback");
                fallbackHeights.reserve(33 * 33);
                for (std::int32_t y = 0; y < 33; ++y)
                {
                    for (std::int32_t x = 0; x < 33; ++x)
                    {
                        const assets::CellCoord vertexCell{origin->x + x, origin->y + y};
                        std::optional<float> height;
                        for (std::int32_t oy = 0; oy >= -1 && !height; --oy)
                        {
                            for (std::int32_t ox = 0; ox >= -1 && !height; --ox)
                            {
                                if (const auto cell = loadDemCell(
                                        {vertexCell.x + ox, vertexCell.y + oy}))
                                {
                                    const std::size_t sampleX = ox == 0 ? 0 : 32;
                                    const std::size_t sampleY = oy == 0 ? 0 : 32;
                                    height = cell->landscape.heights[
                                        sampleY * assets::LandscapeCell::side + sampleX];
                                }
                            }
                        }
                        if (!height)
                            throw std::runtime_error(
                                "DEM BTR height overflow is outside the LAND mosaic");
                        fallbackHeights.push_back(*height);
                    }
                }
            }
            render::PackedMesh mesh;
            for (auto& surface : geometry.surfaces)
                if (!surface.vertices.empty() && !surface.indices.empty())
                {
                    (void)detail::repairLegacyDemBtrGrid(surface.vertices, fallbackHeights);
                    (void)render::appendMesh(mesh,
                        render::buildMeshlets(surface.vertices, surface.indices, material));
                }
            if (mesh.meshlets.empty())
                throw std::runtime_error("DEM BTR contains no renderable terrain: "
                    + btrPath.string());
            return mesh;
        }

        [[nodiscard]] std::set<assets::CellCoord> desiredDemBtrs(
            assets::CellCoord center) const
        {
            std::set<assets::CellCoord> result;
            const assets::CellCoord centerTile{
                demBtrTileOrigin(center.x), demBtrTileOrigin(center.y)};
            for (std::int32_t y = -demBtrRadius; y <= demBtrRadius; ++y)
                for (std::int32_t x = -demBtrRadius; x <= demBtrRadius; ++x)
                {
                    const assets::CellCoord origin{
                        centerTile.x + x * 32, centerTile.y + y * 32};
                    if (demBtrTiles.contains(origin))
                        result.insert(origin);
                }
            return result;
        }

        [[nodiscard]] static glm::mat4 demBtrTransform(assets::CellCoord origin)
        {
            // Keep the coarse surface just below exact LAND where they overlap.
            return glm::translate(glm::mat4(1.0f),
                glm::vec3(origin.x * cellSize, -32.0f, origin.y * cellSize));
        }

        [[nodiscard]] static glm::mat4 demWaterTransform(assets::CellCoord center)
        {
            return glm::translate(glm::mat4(1.0f), glm::vec3(
                (static_cast<float>(center.x) + 0.5f) * cellSize, 0.0f,
                (static_cast<float>(center.y) + 0.5f) * cellSize));
        }

        [[nodiscard]] render::SceneInstance makeDemTerrainInstance(
            const render::MeshRange& range, const glm::mat4& transform, bool coarse) const
        {
            auto instance = makeInstance(range, transform);
            if (demProject)
                instance.lodRange = detail::demTerrainLodRange(coarse);
            return instance;
        }

        void appendInitialDemBtrs(assets::CellCoord center)
        {
            discoverDemBtrs();
            if (surfaces.empty())
                throw std::runtime_error("DEM BTR has no LAND material");
            const std::uint32_t material = surfaces.begin()->second.material;
            for (const assets::CellCoord origin : desiredDemBtrs(center))
            {
                auto& tile = demBtrTiles.at(origin);
                try
                {
                    const auto range = render::appendMesh(initial.mesh,
                        loadDemBtrMesh(tile.path, material));
                    tile.instance = static_cast<std::uint32_t>(initial.instances.size());
                    initial.instances.push_back(
                        makeDemTerrainInstance(range, demBtrTransform(origin), true));
                    activeDemBtrs.insert(origin);
                }
                catch (const std::exception& error)
                {
                    tile.failed = true;
                    std::cerr << "DEM BTR skipped: " << error.what() << '\n';
                }
            }
            std::cout << "DEM BTR viewer: " << activeDemBtrs.size() << " / "
                      << demBtrTiles.size() << " level-32 tiles resident\n";
        }

        void queueDemCells(assets::CellCoord center)
        {
            for (std::int32_t y = center.y - demCellRadius;
                y <= center.y + demCellRadius; ++y)
            {
                for (std::int32_t x = center.x - demCellRadius;
                    x <= center.x + demCellRadius; ++x)
                {
                    const assets::CellCoord coord{x, y};
                    if (!document.containsCell(coord) && !unavailableDemCells.contains(coord)
                        && queuedDemCells.insert(coord).second)
                        demCellQueue.push_back(coord);
                }
            }
        }

        void startDemCellJob()
        {
            if (demCellJob.valid() || demCellQueue.empty())
                return;
            constexpr std::size_t batchSize = 16;
            std::vector<assets::CellCoord> requested;
            while (!demCellQueue.empty() && requested.size() < batchSize)
            {
                requested.push_back(demCellQueue.front());
                demCellQueue.pop_front();
            }
            demCellJob = std::async(std::launch::async,
                [this, requested = std::move(requested)]() mutable {
                    PreparedDemCells result;
                    result.requested = std::move(requested);
                    try
                    {
                        std::scoped_lock lock(demIoMutex);
                        for (const assets::CellCoord coord : result.requested)
                            if (const auto cell = demProject->cell(coord))
                                result.cells.push_back(*cell);
                    }
                    catch (const std::exception& error)
                    {
                        result.error = error.what();
                    }
                    return result;
                });
        }

        void pollDemCells(render::VulkanRenderer& renderer)
        {
            if (!demCellJob.valid()
                || demCellJob.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
                return;
            auto prepared = demCellJob.get();
            std::set<assets::CellCoord> found;
            for (const auto& cell : prepared.cells)
                found.insert(cell.landscape.cell);
            for (const assets::CellCoord coord : prepared.requested)
            {
                queuedDemCells.erase(coord);
                if (!found.contains(coord))
                    unavailableDemCells.insert(coord);
            }
            if (!prepared.error.empty())
                std::cerr << "DEM LAND streaming failed: " << prepared.error << '\n';
            for (const auto& cell : prepared.cells)
                addRuntimeCell(renderer, cell, false);
        }

        [[nodiscard]] bool demWindowResolved(assets::CellCoord center) const
        {
            for (std::int32_t y = center.y - demCellRadius;
                y <= center.y + demCellRadius; ++y)
                for (std::int32_t x = center.x - demCellRadius;
                    x <= center.x + demCellRadius; ++x)
                    if (!document.containsCell({x, y})
                        && !unavailableDemCells.contains({x, y}))
                        return false;
            return true;
        }

        void startDemBtrJob()
        {
            if (demBtrJob.valid() || demBtrQueue.empty() || demCellJob.valid()
                || !demCellQueue.empty())
                return;
            if (surfaces.empty())
                throw std::runtime_error("DEM BTR has no LAND material");
            const assets::CellCoord origin = demBtrQueue.front();
            demBtrQueue.pop_front();
            const auto btrPath = demBtrTiles.at(origin).path;
            const std::uint32_t material = surfaces.begin()->second.material;
            demBtrJob = std::async(std::launch::async,
                [this, origin, btrPath, material] {
                    PreparedDemBtr result;
                    result.origin = origin;
                    try
                    {
                        result.mesh = loadDemBtrMesh(btrPath, material);
                    }
                    catch (const std::exception& error)
                    {
                        result.error = error.what();
                    }
                    return result;
                });
        }

        void pollDemBtr(render::VulkanRenderer& renderer)
        {
            if (!demBtrJob.valid()
                || demBtrJob.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
                return;
            auto prepared = demBtrJob.get();
            auto& tile = demBtrTiles.at(prepared.origin);
            tile.queued = false;
            if (prepared.mesh)
            {
                const auto range = renderer.addMesh(std::move(*prepared.mesh));
                auto instance = makeDemTerrainInstance(
                    range, demBtrTransform(prepared.origin), true);
                instance.active = false;
                tile.instance = renderer.addInstance(instance);
            }
            else
            {
                tile.failed = true;
                std::cerr << "DEM BTR skipped: " << prepared.error << '\n';
            }
        }

        void updateDemBtrs(render::VulkanRenderer& renderer, assets::CellCoord center,
            std::vector<std::uint32_t>& activate, std::vector<std::uint32_t>& deactivate)
        {
            pollDemBtr(renderer);
            const auto desired = desiredDemBtrs(center);
            std::set<assets::CellCoord> residentDesired;
            bool replacementReady = true;
            for (const assets::CellCoord origin : desired)
            {
                auto& tile = demBtrTiles.at(origin);
                if (!tile.instance && !tile.failed && !tile.queued)
                {
                    tile.queued = true;
                    demBtrQueue.push_back(origin);
                }
                if (!tile.instance)
                {
                    replacementReady = replacementReady && tile.failed;
                    continue;
                }
                residentDesired.insert(origin);
                if (!activeDemBtrs.contains(origin))
                    activate.push_back(*tile.instance);
            }
            startDemBtrJob();
            if (replacementReady)
            {
                for (const assets::CellCoord origin : activeDemBtrs)
                    if (!residentDesired.contains(origin))
                        deactivate.push_back(*demBtrTiles.at(origin).instance);
                activeDemBtrs = std::move(residentDesired);
            }
            else
                activeDemBtrs.insert(residentDesired.begin(), residentDesired.end());
        }

        Impl(const std::filesystem::path& dataDirectory, std::filesystem::path outputPath,
            bool viewOnly)
            : path(std::filesystem::absolute(std::move(outputPath))),
              demProject(openDemProject(path)), document(openDocument(path, demProject.get())),
              saved(document), materialWorld(assets::loadCommonwealth(dataDirectory / "Fallout4.esm")),
              archives(dataDirectory), materials(archives, initial), readOnly(viewOnly)
        {
            materialWorld.models.clear();
            materialWorld.modelLods.clear();
            materialWorld.staticCollections.clear();
            materialWorld.references.clear();
            materialWorld.landscapes.clear();
            materialWorld.waterHeights.clear();

            if (demProject)
            {
                render::SceneMaterial water;
                water.baseColor = {0.025f, 0.10f, 0.14f, 1.0f};
                water.water = true;
                waterMaterial = static_cast<std::uint32_t>(initial.materials.size());
                initial.materials.push_back(water);
                waterRange = render::appendMesh(initial.mesh, waterPlane(
                    demWaterHalfExtent, assets::demDefaultWaterHeight, waterMaterial));
                waterInstance = static_cast<std::uint32_t>(initial.instances.size());
                auto instance = makeInstance(waterRange, demWaterTransform({0, 0}));
                instance.lodRange = detail::demWaterLodRange();
                initial.instances.push_back(instance);
            }

            for (const auto& [coord, forms] : document.cellForms)
            {
                (void)forms;
                const auto landscape = document.cell(coord);
                const auto surface = terrainSurface(landscape, materialWorld, materials);
                surfaces.emplace(coord, surface);
                const auto range = render::appendMesh(
                    initial.mesh, terrainCell(landscape, surface, 1, &document));
                terrainRanges.emplace(coord, range);
                terrainInstances.emplace(coord,
                    static_cast<std::uint32_t>(initial.instances.size()));
                initial.instances.push_back(
                    makeDemTerrainInstance(range, glm::mat4(1.0f), false));
                activeCells.insert(coord);
            }

            if (demProject)
                appendInitialDemBtrs({0, 0});
            if (!readOnly)
            {
                render::SceneMaterial brushSceneMaterial;
                brushSceneMaterial.baseColor = {0.95f, 0.2f, 0.04f, 1.0f};
                brushMaterial = static_cast<std::uint32_t>(initial.materials.size());
                initial.materials.push_back(brushSceneMaterial);
                brushRange = render::appendMesh(initial.mesh,
                    makeBrushMesh({2048.0f, 2048.0f}, radius, brushMaterial));
                brushInstance = static_cast<std::uint32_t>(initial.instances.size());
                auto instance = makeInstance(brushRange, glm::mat4(1.0f));
                instance.active = false;
                initial.instances.push_back(instance);
            }

            const float centerHeight = document.cell({0, 0}).heights[16 * 33 + 16];
            initial.cameraTarget = {-2048.0f, centerHeight, 2048.0f};
            initial.cameraPosition = initial.cameraTarget + glm::vec3(0.0f, 3000.0f, 4200.0f);
            initial.sceneScale = demProject
                ? cellSize * static_cast<float>(demCellRadius * 2 + 1) : 12288.0f;
            initial.centerCell = {0, 0};
            initial.landscapeCells = document.cellForms.size();
            if (demProject)
                activeCenter = assets::CellCoord{0, 0};
        }

        ~Impl()
        {
            if (readOnly || externalSave || !isDirty())
                return;
            try
            {
                detail::quantizeLandscape(document);
                if (demProject)
                    demProject->save(document, dirtyCells);
                else
                    assets::saveEditableLandscape(path, document);
            }
            catch (...)
            {
            }
        }

        [[nodiscard]] bool isDirty() const noexcept
        {
            if (readOnly)
                return false;
            return demProject ? !dirtyCells.empty() : document != saved;
        }

        [[nodiscard]] float sampleHeight(float x, float y) const
        {
            const float gx = x / vertexSpacing;
            const float gy = y / vertexSpacing;
            const std::int32_t x0 = static_cast<std::int32_t>(std::floor(gx));
            const std::int32_t y0 = static_cast<std::int32_t>(std::floor(gy));
            const std::int32_t x1 = x0 + 1;
            const std::int32_t y1 = y0 + 1;
            const float tx = gx - static_cast<float>(x0);
            const float ty = gy - static_cast<float>(y0);
            const std::array coordinates{
                assets::EditableLandscape::SampleCoord{x0, y0},
                assets::EditableLandscape::SampleCoord{x1, y0},
                assets::EditableLandscape::SampleCoord{x0, y1},
                assets::EditableLandscape::SampleCoord{x1, y1}};
            if (std::ranges::all_of(coordinates,
                    [&](const auto coord) { return document.hasSample(coord); }))
                return glm::mix(glm::mix(document.sample(coordinates[0]),
                    document.sample(coordinates[1]), tx), glm::mix(document.sample(coordinates[2]),
                    document.sample(coordinates[3]), tx), ty);

            const assets::CellCoord cell{
                static_cast<std::int32_t>(std::floor(x / cellSize)),
                static_cast<std::int32_t>(std::floor(y / cellSize))};
            if (!document.containsCell(cell))
                return 0.0f;
            return document.sample({std::clamp(x0, cell.x * 32, cell.x * 32 + 32),
                std::clamp(y0, cell.y * 32, cell.y * 32 + 32)});
        }

        [[nodiscard]] render::PackedMesh makeBrushMesh(
            glm::vec2 center, float brushRadius, std::uint32_t material) const
        {
            constexpr std::uint32_t segments = 64;
            constexpr float pi = 3.14159265358979323846f;
            constexpr float lift = 12.0f;
            const float halfWidth = std::min(20.0f, brushRadius * 0.08f);
            std::vector<render::SourceVertex> vertices;
            std::vector<std::uint32_t> indices;
            vertices.reserve((segments + 1) * 2);
            indices.reserve(segments * 6);
            for (std::uint32_t segment = 0; segment <= segments; ++segment)
            {
                const float angle = 2.0f * pi * static_cast<float>(segment)
                    / static_cast<float>(segments);
                const glm::vec2 direction(std::cos(angle), std::sin(angle));
                for (const float ringRadius : {brushRadius - halfWidth, brushRadius + halfWidth})
                {
                    const glm::vec2 position = center + direction * ringRadius;
                    render::SourceVertex vertex;
                    vertex.position = {position.x, sampleHeight(position.x, position.y) + lift,
                        position.y};
                    vertex.normal = {0.0f, 1.0f, 0.0f};
                    vertices.push_back(vertex);
                }
            }
            for (std::uint32_t segment = 0; segment < segments; ++segment)
            {
                const std::uint32_t first = segment * 2;
                indices.insert(indices.end(), {
                    first, first + 1, first + 2, first + 1, first + 3, first + 2});
            }
            return render::buildMeshlets(vertices, indices, material);
        }

        [[nodiscard]] static std::optional<float> rayTriangle(
            const glm::vec3& origin, const glm::vec3& direction,
            const glm::vec3& a, const glm::vec3& b, const glm::vec3& c)
        {
            constexpr float epsilon = 0.00001f;
            const glm::vec3 edge1 = b - a;
            const glm::vec3 edge2 = c - a;
            const glm::vec3 cross = glm::cross(direction, edge2);
            const float determinant = glm::dot(edge1, cross);
            if (std::abs(determinant) < epsilon)
                return std::nullopt;
            const float inverse = 1.0f / determinant;
            const glm::vec3 distance = origin - a;
            const float u = glm::dot(distance, cross) * inverse;
            if (u < 0.0f || u > 1.0f)
                return std::nullopt;
            const glm::vec3 q = glm::cross(distance, edge1);
            const float v = glm::dot(direction, q) * inverse;
            if (v < 0.0f || u + v > 1.0f)
                return std::nullopt;
            const float hit = glm::dot(edge2, q) * inverse;
            return hit > epsilon ? std::optional<float>(hit) : std::nullopt;
        }

        [[nodiscard]] std::optional<glm::vec3> raycast(
            const glm::dvec3& cameraPosition, const glm::vec3& cameraForward) const
        {
            const glm::vec3 origin(-static_cast<float>(cameraPosition.x),
                static_cast<float>(cameraPosition.y), static_cast<float>(cameraPosition.z));
            const glm::vec3 direction(-cameraForward.x, cameraForward.y, cameraForward.z);
            float closest = std::numeric_limits<float>::max();
            // ponytail: linear per-cell raycasts are enough for small editor ESMs; add a
            // spatial index only if user-created cell counts make this measurable.
            for (const auto& [cell, forms] : document.cellForms)
            {
                (void)forms;
                if (demProject && !activeCells.contains(cell))
                    continue;
                const auto landscape = document.cell(cell);
                for (std::size_t y = 0; y < 32; ++y)
                {
                    for (std::size_t x = 0; x < 32; ++x)
                    {
                        const float x0 = cell.x * cellSize + static_cast<float>(x) * vertexSpacing;
                        const float x1 = x0 + vertexSpacing;
                        const float z0 = cell.y * cellSize + static_cast<float>(y) * vertexSpacing;
                        const float z1 = z0 + vertexSpacing;
                        const glm::vec3 a{x0, landscape.heights[y * 33 + x], z0};
                        const glm::vec3 b{x1, landscape.heights[y * 33 + x + 1], z0};
                        const glm::vec3 c{x0, landscape.heights[(y + 1) * 33 + x], z1};
                        const glm::vec3 d{x1, landscape.heights[(y + 1) * 33 + x + 1], z1};
                        for (const auto hit : {rayTriangle(origin, direction, a, b, c),
                                 rayTriangle(origin, direction, b, d, c)})
                            if (hit && *hit < closest)
                                closest = *hit;
                    }
                }
            }
            return closest == std::numeric_limits<float>::max()
                ? std::nullopt : std::optional(origin + direction * closest);
        }

        void setBrushVisible(render::VulkanRenderer& renderer, bool visible)
        {
            if (brushVisible == visible)
                return;
            const std::array instances{brushInstance};
            renderer.setInstancesActive(instances, visible);
            brushVisible = visible;
        }

        void updateBrush(render::VulkanRenderer& renderer, const std::optional<glm::vec3>& hit)
        {
            setBrushVisible(renderer, hit.has_value());
            if (!hit)
                return;
            const glm::vec2 center(hit->x, hit->z);
            if (glm::distance(center, brushCenter) < 1.0f && displayedRadius == radius)
                return;
            auto mesh = makeBrushMesh(center, radius, brushMaterial);
            const glm::vec4 bounds = mesh.bounds;
            renderer.updateMesh(brushRange, std::move(mesh));
            renderer.updateInstanceTransform(brushInstance, glm::mat4(1.0f), bounds);
            brushCenter = center;
            displayedRadius = radius;
        }

        [[nodiscard]] std::set<assets::CellCoord> changedCells(
            const assets::EditableLandscape& before) const
        {
            std::set<assets::CellCoord> owners;
            for (const auto& [coord, forms] : document.cellForms)
            {
                (void)forms;
                if (!before.containsCell(coord)
                    || before.cell(coord).heights != document.cell(coord).heights)
                    owners.insert(coord);
            }
            std::set<assets::CellCoord> result = owners;
            for (const auto owner : owners)
            {
                for (int y = owner.y - 1; y <= owner.y + 1; ++y)
                    for (int x = owner.x - 1; x <= owner.x + 1; ++x)
                        if (document.containsCell({x, y}))
                            result.insert({x, y});
            }
            return result;
        }

        void refreshChanged(render::VulkanRenderer& renderer,
            const assets::EditableLandscape& before)
        {
            for (const auto coord : changedCells(before))
            {
                auto mesh = terrainCell(document.cell(coord), surfaces.at(coord), 1, &document);
                const glm::vec4 bounds = mesh.bounds;
                renderer.updateMesh(terrainRanges.at(coord), std::move(mesh));
                renderer.updateInstanceTransform(
                    terrainInstances.at(coord), glm::mat4(1.0f), bounds);
            }
        }

        void refreshAll(render::VulkanRenderer& renderer)
        {
            for (const auto& [coord, forms] : document.cellForms)
            {
                (void)forms;
                auto mesh = terrainCell(document.cell(coord), surfaces.at(coord), 1, &document);
                const glm::vec4 bounds = mesh.bounds;
                renderer.updateMesh(terrainRanges.at(coord), std::move(mesh));
                renderer.updateInstanceTransform(
                    terrainInstances.at(coord), glm::mat4(1.0f), bounds);
            }
            brushCenter = glm::vec2(std::numeric_limits<float>::max());
        }

        void finishStroke(render::VulkanRenderer& renderer)
        {
            if (!strokeActive)
                return;
            const auto beforeQuantize = document;
            detail::quantizeLandscape(document);
            refreshChanged(renderer, beforeQuantize);
            if (document != strokeBefore)
            {
                if (demProject)
                {
                    const auto changed = changedCells(strokeBefore);
                    dirtyCells.insert(changed.begin(), changed.end());
                }
                history.commit(strokeBefore);
            }
            strokeActive = false;
        }

        void addRuntimeCell(render::VulkanRenderer& renderer,
            const assets::DemLandscapeCell& imported, bool active = true)
        {
            const assets::CellCoord coord = imported.landscape.cell;
            insertCell(document, imported);
            const auto landscape = document.cell(coord);
            const auto surface = terrainSurface(landscape, materialWorld, materials, &renderer);
            surfaces.emplace(coord, surface);
            const auto range = renderer.addMesh(terrainCell(landscape, surface, 1, &document));
            terrainRanges.emplace(coord, range);
            auto instance = makeDemTerrainInstance(range, glm::mat4(1.0f), false);
            instance.active = active;
            terrainInstances.emplace(coord, renderer.addInstance(instance));
        }

        void updateDemWindow(render::VulkanRenderer& renderer, assets::CellCoord center)
        {
            const bool centerChanged = activeCenter != center;
            if (centerChanged)
                finishStroke(renderer);
            pollDemCells(renderer);
            queueDemCells(center);
            startDemCellJob();
            std::set<assets::CellCoord> desired;
            for (std::int32_t y = center.y - demCellRadius;
                y <= center.y + demCellRadius; ++y)
            {
                for (std::int32_t x = center.x - demCellRadius;
                    x <= center.x + demCellRadius; ++x)
                {
                    const assets::CellCoord coord{x, y};
                    if (document.containsCell(coord))
                        desired.insert(coord);
                }
            }
            if (centerChanged && !readOnly && !document.containsCell(center)
                && !demProject->contains(center))
            {
                if (!document.addCell(center))
                    throw std::runtime_error("Could not create missing DEM editor CELL");
                dirtyCells.insert(center);
                history.addCell(center);
                const assets::DemLandscapeCell created{document.cell(center),
                    document.cellForms.at(center)};
                addRuntimeCell(renderer, created);
                desired.insert(center);
                std::cout << "Editor created CELL " << center.x << ", " << center.y << '\n';
            }
            std::vector<std::uint32_t> activate;
            std::vector<std::uint32_t> deactivate;
            const bool switchLand = demWindowResolved(center);
            if (switchLand)
            {
                for (const auto& [coord, instance] : terrainInstances)
                {
                    if (desired.contains(coord) && !activeCells.contains(coord))
                        activate.push_back(instance);
                    else if (!desired.contains(coord) && activeCells.contains(coord))
                        deactivate.push_back(instance);
                }
            }
            updateDemBtrs(renderer, center, activate, deactivate);
            if (centerChanged && waterInstance)
            {
                auto water = makeInstance(waterRange, demWaterTransform(center));
                water.lodRange = detail::demWaterLodRange();
                renderer.updateInstanceTransform(
                    *waterInstance, water.transform, water.bounds);
            }
            if (!activate.empty())
                renderer.setInstancesActive(activate, true);
            if (!deactivate.empty())
                renderer.setInstancesActive(deactivate, false);
            if (switchLand)
                activeCells = std::move(desired);
            if (centerChanged)
            {
                activeCenter = center;
                brushCenter = glm::vec2(std::numeric_limits<float>::max());
            }
        }

        void ensureCameraCell(render::VulkanRenderer& renderer, const glm::dvec3& cameraPosition)
        {
            const assets::CellCoord coord{
                static_cast<std::int32_t>(std::floor(-cameraPosition.x / cellSize)),
                static_cast<std::int32_t>(std::floor(cameraPosition.z / cellSize))};
            if (demProject)
            {
                updateDemWindow(renderer, coord);
                return;
            }
            if (readOnly)
                return;
            if (document.containsCell(coord))
                return;
            finishStroke(renderer);
            if (!document.addCell(coord))
                return;
            history.addCell(coord);
            const auto landscape = document.cell(coord);
            const auto surface = terrainSurface(landscape, materialWorld, materials, &renderer);
            surfaces.emplace(coord, surface);
            const auto range = renderer.addMesh(terrainCell(landscape, surface, 1, &document));
            terrainRanges.emplace(coord, range);
            terrainInstances.emplace(coord, renderer.addInstance(makeInstance(range, glm::mat4(1.0f))));
            brushCenter = glm::vec2(std::numeric_limits<float>::max());
            std::cout << "Editor created CELL " << coord.x << ", " << coord.y << '\n';
        }

        void update(render::VulkanRenderer& renderer, const glm::dvec3& cameraPosition,
            const glm::vec3& cameraForward, float deltaSeconds, bool sculpting)
        {
            ensureCameraCell(renderer, cameraPosition);
            if (readOnly)
                return;
            if (!brushEnabled)
            {
                finishStroke(renderer);
                setBrushVisible(renderer, false);
                return;
            }
            const auto hit = raycast(cameraPosition, cameraForward);
            if (sculpting && hit)
            {
                if (!strokeActive)
                {
                    strokeBefore = document;
                    flattenHeight = hit->y;
                    strokeActive = true;
                }
                const auto before = document;
                if (detail::applyLandscapeBrush(document, {hit->x, hit->z}, tool,
                        radius, strength * deltaSeconds, flattenHeight))
                    refreshChanged(renderer, before);
            }
            else if (!sculpting)
                finishStroke(renderer);
            updateBrush(renderer, hit);
        }

        [[nodiscard]] bool undo(render::VulkanRenderer& renderer)
        {
            if (readOnly)
                return false;
            finishStroke(renderer);
            if (!history.undo(document))
                return false;
            if (demProject)
                for (const auto& [coord, forms] : document.cellForms)
                {
                    (void)forms;
                    dirtyCells.insert(coord);
                }
            refreshAll(renderer);
            return true;
        }

        [[nodiscard]] bool redo(render::VulkanRenderer& renderer)
        {
            if (readOnly)
                return false;
            finishStroke(renderer);
            if (!history.redo(document))
                return false;
            if (demProject)
                for (const auto& [coord, forms] : document.cellForms)
                {
                    (void)forms;
                    dirtyCells.insert(coord);
                }
            refreshAll(renderer);
            return true;
        }

        void save(render::VulkanRenderer& renderer)
        {
            if (readOnly)
                return;
            finishStroke(renderer);
            if (demProject)
            {
                demProject->save(document, dirtyCells);
                dirtyCells.clear();
            }
            else
                assets::saveEditableLandscape(path, document);
            saved = document;
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

    void setBethesdaReferenceTransform(assets::PlacedReference& reference,
        const glm::mat4& transform)
    {
        for (int column = 0; column < 4; ++column)
            for (int row = 0; row < 4; ++row)
                if (!std::isfinite(transform[column][row]))
                    throw std::runtime_error("Reference transform must be finite");
        const glm::vec3 scales(glm::length(glm::vec3(transform[0])),
            glm::length(glm::vec3(transform[1])), glm::length(glm::vec3(transform[2])));
        const float scale = (scales.x + scales.y + scales.z) / 3.0f;
        if (scale < 0.0001f || glm::any(glm::greaterThan(glm::abs(scales - scale),
                glm::vec3(scale * 0.001f))) || glm::determinant(glm::mat3(transform)) <= 0.0f)
            throw std::runtime_error("References require positive uniform scale");
        const glm::mat3 engineRotation = glm::mat3(transform) / scale;
        if (std::abs(glm::dot(engineRotation[0], engineRotation[1])) > 0.001f
            || std::abs(glm::dot(engineRotation[0], engineRotation[2])) > 0.001f
            || std::abs(glm::dot(engineRotation[1], engineRotation[2])) > 0.001f)
            throw std::runtime_error("References cannot contain shear");
        const glm::quat engine = glm::normalize(glm::quat_cast(engineRotation));
        const glm::mat3 rotation = glm::mat3_cast(glm::quat(engine.w, engine.x, engine.z, engine.y));
        const float y = std::asin(glm::clamp(-rotation[0][2], -1.0f, 1.0f));
        const bool singular = std::abs(std::cos(y)) < 0.0001f;
        const float x = singular ? std::atan2(-rotation[2][1], rotation[1][1])
            : std::atan2(rotation[1][2], rotation[2][2]);
        const float z = singular ? 0.0f : std::atan2(rotation[0][1], rotation[0][0]);
        const double cellX = std::floor(static_cast<double>(transform[3].x) / cellSize);
        const double cellY = std::floor(static_cast<double>(transform[3].z) / cellSize);
        if (cellX < INT32_MIN || cellX > INT32_MAX || cellY < INT32_MIN || cellY > INT32_MAX)
            throw std::runtime_error("Reference position exceeds CELL coordinate range");
        reference.position[0] = transform[3].x;
        reference.position[1] = transform[3].z;
        reference.position[2] = transform[3].y;
        reference.rotation[0] = x;
        reference.rotation[1] = y;
        reference.rotation[2] = z;
        reference.scale = scale;
        reference.hasLocalTransform = false;
        reference.cell = {static_cast<std::int32_t>(cellX), static_cast<std::int32_t>(cellY)};
    }

    WorldStreamer::WorldStreamer(const std::filesystem::path& dataDirectory,
        std::optional<assets::CellCoord> requestedCenter, int cellRadius,
        bool includeObjects, std::size_t maxObjects,
        std::optional<std::filesystem::path> masterPath)
        : mImpl(std::make_unique<Impl>(dataDirectory, requestedCenter, cellRadius,
            includeObjects, maxObjects, masterPath))
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

    std::optional<glm::vec3> WorldStreamer::enablePhysics(
        render::VulkanRenderer& renderer, physics::JoltWorld& world, bool spawnPhysicsSample)
    {
        if (!mImpl->initialTaken)
            throw std::runtime_error("Initial scene must be installed before enabling physics");
        return mImpl->enablePhysics(renderer, world, spawnPhysicsSample);
    }

    bool WorldStreamer::streamNext(render::VulkanRenderer& renderer)
    {
        if (!mImpl->initialTaken)
            throw std::runtime_error("Initial scene must be installed before streaming objects");
        return mImpl->streamNext(renderer);
    }

    void WorldStreamer::update(
        render::VulkanRenderer& renderer, const glm::dvec3& playerPosition)
    {
        if (!mImpl->initialTaken)
            throw std::runtime_error("Initial scene must be installed before updating the streamer");
        mImpl->update(renderer, playerPosition);
    }

    std::size_t WorldStreamer::streamedObjects() const noexcept
    {
        return mImpl->loadedObjects;
    }

    std::size_t WorldStreamer::pendingObjects() const noexcept
    {
            return mImpl->terrainRequests.size() + mImpl->referenceRequests.size()
                + mImpl->readyReferences.size() + mImpl->physicsReferences.size()
                + mImpl->terrainPhysicsRequests.size() + mImpl->physicsRemovals.size()
                + mImpl->jobs.size() + mImpl->lodRequests.size() + mImpl->lodJobs.size()
                + std::ranges::count_if(mImpl->lodTiles, [](const auto& tile) {
                    return tile.objectsState == WorldStreamer::Impl::LodLoadState::discovered;
                });
    }

    LandscapeEditor::LandscapeEditor(const std::filesystem::path& dataDirectory,
        std::filesystem::path outputPath, bool readOnly)
        : mImpl(std::make_unique<Impl>(dataDirectory, std::move(outputPath), readOnly))
    {
    }

    LandscapeEditor::~LandscapeEditor() = default;
    LandscapeEditor::LandscapeEditor(LandscapeEditor&&) noexcept = default;
    LandscapeEditor& LandscapeEditor::operator=(LandscapeEditor&&) noexcept = default;

    WorldScene LandscapeEditor::takeInitialScene()
    {
        if (mImpl->initialTaken)
            throw std::runtime_error("Initial editor scene was already taken");
        mImpl->initialTaken = true;
        return std::move(mImpl->initial);
    }

    void LandscapeEditor::setTool(LandscapeTool tool) noexcept
    {
        mImpl->tool = tool;
    }

    void LandscapeEditor::adjustRadius(float wheelSteps) noexcept
    {
        mImpl->radius = std::clamp(mImpl->radius + wheelSteps * 128.0f, 128.0f, 4096.0f);
        mImpl->brushCenter = glm::vec2(std::numeric_limits<float>::max());
    }

    void LandscapeEditor::adjustStrength(float wheelSteps) noexcept
    {
        mImpl->strength = std::clamp(mImpl->strength + wheelSteps * 64.0f, 64.0f, 4096.0f);
    }

    void LandscapeEditor::update(render::VulkanRenderer& renderer,
        const glm::dvec3& cameraPosition, const glm::vec3& cameraForward,
        float deltaSeconds, bool sculpting)
    {
        if (!mImpl->initialTaken)
            throw std::runtime_error("Initial editor scene must be installed before editing");
        mImpl->update(renderer, cameraPosition, cameraForward, deltaSeconds, sculpting);
    }

    bool LandscapeEditor::undo(render::VulkanRenderer& renderer)
    {
        return mImpl->undo(renderer);
    }

    bool LandscapeEditor::redo(render::VulkanRenderer& renderer)
    {
        return mImpl->redo(renderer);
    }

    void LandscapeEditor::save(render::VulkanRenderer& renderer)
    {
        mImpl->save(renderer);
    }

    void LandscapeEditor::setBrushEnabled(bool enabled) noexcept
    {
        mImpl->brushEnabled = enabled;
    }

    void LandscapeEditor::setExternalSave(bool enabled) noexcept
    {
        mImpl->externalSave = enabled;
    }

    std::optional<glm::vec3> LandscapeEditor::raycast(
        const glm::dvec3& cameraPosition, const glm::vec3& direction) const
    {
        return mImpl->raycast(cameraPosition, direction);
    }

    assets::EditableLandscape LandscapeEditor::snapshot() const
    {
        return mImpl->document;
    }

    std::size_t LandscapeEditor::residentCells() const noexcept
    {
        return mImpl->document.cellForms.size();
    }

    std::size_t LandscapeEditor::pendingWork() const noexcept
    {
        return mImpl->queuedDemCells.size() + mImpl->demBtrQueue.size()
            + static_cast<std::size_t>(mImpl->demBtrJob.valid());
    }

    std::uint32_t LandscapeEditor::nextFormId() const noexcept
    {
        return mImpl->document.nextFormId;
    }

    void LandscapeEditor::setNextFormId(std::uint32_t next) noexcept
    {
        mImpl->document.nextFormId = std::max(mImpl->document.nextFormId, next);
    }

    void LandscapeEditor::applySnapshot(render::VulkanRenderer& renderer,
        const assets::EditableLandscape& snapshot)
    {
        mImpl->finishStroke(renderer);
        auto restored = snapshot;
        // Streaming is not an authoring command. Undoing an older stroke must not
        // remove cells that arrived from the stream after that snapshot was taken.
        for (const auto& [cell, forms] : mImpl->document.cellForms)
            if (!restored.containsCell(cell))
                Impl::insertCell(restored, {mImpl->document.cell(cell), forms});
        restored.nextFormId = std::max(restored.nextFormId, mImpl->document.nextFormId);
        for (const auto& [cell, forms] : restored.cellForms)
            if (!mImpl->terrainInstances.contains(cell))
                mImpl->addRuntimeCell(renderer, {restored.cell(cell), forms});
        for (const auto& [cell, instance] : mImpl->terrainInstances)
        {
            const std::array ids{instance};
            renderer.setInstancesActive(ids, restored.containsCell(cell)
                && (!mImpl->demProject || mImpl->activeCells.contains(cell)));
            mImpl->dirtyCells.insert(cell);
        }
        mImpl->document = std::move(restored);
        mImpl->refreshAll(renderer);
    }

    bool LandscapeEditor::dirty() const noexcept
    {
        return mImpl->isDirty();
    }

    std::string LandscapeEditor::status() const
    {
        const char* tool = "Raise";
        switch (mImpl->tool)
        {
        case LandscapeTool::raise: tool = "Raise"; break;
        case LandscapeTool::lower: tool = "Lower"; break;
        case LandscapeTool::smooth: tool = "Smooth"; break;
        case LandscapeTool::flatten: tool = "Flatten"; break;
        }
        const std::uint64_t cells = mImpl->demProject
            ? mImpl->demProject->importedCellCount()
            : mImpl->document.cellForms.size();
        return std::string("OpenF4 Editor | ") + tool
            + " | cells " + std::to_string(cells)
            + " | radius " + std::to_string(static_cast<int>(std::lround(mImpl->radius)))
            + " | strength " + std::to_string(static_cast<int>(std::lround(mImpl->strength)))
            + (dirty() ? " | modified | " : " | saved | ") + mImpl->path.string();
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
            const auto surface = terrainSurface(landscape, world, materials);
            const auto range = render::appendMesh(result.mesh, terrainCell(landscape, surface));
            result.instances.push_back(makeInstance(range, glm::mat4(1.0f)));
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
                        modelRange.vertexOffset = static_cast<std::uint32_t>(result.mesh.vertices.size());
                        modelRange.indexOffset = static_cast<std::uint32_t>(result.mesh.indices.size());
                        modelRange.meshletOffset = static_cast<std::uint32_t>(result.mesh.meshlets.size());
                        modelRange.bounds = boundsOf(geometry);
                        for (const auto& surface : geometry.surfaces)
                        {
                            if (surface.vertices.empty() || surface.indices.empty())
                                continue;
                            const auto material = materials.add(
                                surface.materialPath, surface.diffuseTexture, surface.normalTexture,
                                surface.alphaCutoff, materialSwap, false, surface.decal);
                            (void)render::appendMesh(result.mesh,
                                render::buildMeshlets(surface.vertices, surface.indices, material));
                        }
                        modelRange.meshletCount = static_cast<std::uint32_t>(result.mesh.meshlets.size())
                            - modelRange.meshletOffset;
                        modelRange.vertexCount = static_cast<std::uint32_t>(result.mesh.vertices.size())
                            - modelRange.vertexOffset;
                        modelRange.indexCount = static_cast<std::uint32_t>(result.mesh.indices.size())
                            - modelRange.indexOffset;
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
