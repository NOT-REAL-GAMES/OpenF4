#include "assets/fo4_asset_package.hpp"

#include "assets/coordinates.hpp"
#include "assets/material_file.hpp"
#include "assets/model_import.hpp"
#include "assets/native_model_export.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>

namespace openf4::assets
{
    namespace
    {
        std::string normalized(std::string value)
        {
            if (value.empty() || value.find(':') != value.npos
                || std::ranges::any_of(value, [](unsigned char c) { return c < 32 || c == 127; }))
                throw std::runtime_error("Invalid native package asset path");
            std::ranges::replace(value, '\\', '/');
            if (value.front() == '/') throw std::runtime_error("Native package asset paths must be relative");
            std::size_t start = 0;
            while (start < value.size())
            {
                const auto end = value.find('/', start);
                const auto part = value.substr(start, end == value.npos ? end : end - start);
                if (part.empty() || part == "." || part == ".." || part.back() == '.' || part.back() == ' ')
                    throw std::runtime_error("Native package asset path contains traversal or ambiguous components");
                if (end == value.npos) break;
                start = end + 1;
            }
            value = AssetResolver::normalizePath(value); std::ranges::replace(value, '\\', '/');
            return value;
        }

        std::string assetPath(std::string value)
        {
            value = normalized(std::move(value));
            const auto extension = std::filesystem::path(value).extension().string();
            const char* prefix = extension == ".nif" ? "meshes/" : extension == ".dds" ? "textures/"
                : extension == ".bgsm" || extension == ".bgem" ? "materials/" : nullptr;
            if (!prefix) throw std::runtime_error("Unsupported native dependency: " + value);
            if (!value.starts_with(prefix)) value = std::string(prefix) + value;
            return value;
        }

        std::vector<std::byte> read(const std::filesystem::path& path)
        {
            std::ifstream input(path, std::ios::binary | std::ios::ate);
            if (!input || input.tellg() < 0) throw std::runtime_error("Cannot read package file: " + path.string());
            std::vector<std::byte> bytes(static_cast<std::size_t>(input.tellg())); input.seekg(0);
            if (!input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size())))
                throw std::runtime_error("Cannot read complete package file: " + path.string());
            return bytes;
        }

        // Inspect each component, including on case-sensitive hosts: FO4 paths are case-insensitive.
        std::filesystem::path existingPath(const std::filesystem::path& root, const std::string& relative)
        {
            auto path = std::filesystem::absolute(root);
            for (auto parent = path; !parent.empty();)
            {
                if (std::filesystem::is_symlink(std::filesystem::symlink_status(parent)))
                    throw std::runtime_error("Native package path contains a link: " + parent.string());
                const auto next = parent.parent_path(); if (next == parent) break; parent = next;
            }
            for (const auto& component : std::filesystem::path(relative))
            {
                std::optional<std::filesystem::path> found;
                if (std::filesystem::exists(path))
                {
                    if (!std::filesystem::is_directory(path)) throw std::runtime_error("Asset directory conflicts with a file: " + path.string());
                    for (const auto& entry : std::filesystem::directory_iterator(path))
                        if (AssetResolver::normalizePath(entry.path().filename().string()) == component.string())
                        {
                            if (found) throw std::runtime_error("Case-folded asset path is ambiguous: " + relative);
                            if (entry.is_symlink()) throw std::runtime_error("Native package asset is a link: " + relative);
                            found = entry.path();
                        }
                }
                path = found ? *found : path / component;
            }
            return path;
        }

        struct Bounds
        {
            glm::vec3 minimum{std::numeric_limits<float>::max()}, maximum{std::numeric_limits<float>::lowest()};
            bool any = false;
            void add(glm::vec3 point)
            {
                for (int axis = 0; axis < 3; ++axis)
                    if (!std::isfinite(point[axis])) throw std::runtime_error("Native asset has non-finite bounds");
                minimum = glm::min(minimum, point); maximum = glm::max(maximum, point); any = true;
            }
        };

        bool hasShape(const HavokRigidBody& body)
        {
            return !body.parts.empty() || !body.capsules.empty() || body.triangleMesh.has_value();
        }

        bool baseArchive(std::string name)
        {
            name = AssetResolver::normalizePath(name);
            if (name == "fallout4 - meshesextra.ba2" || name == "fallout4 - meshes.ba2"
                || name == "fallout4 - materials.ba2" || name == "fallout4 - texturespatch.ba2") return true;
            for (int i = 1; i <= 9; ++i) if (name == "fallout4 - textures" + std::to_string(i) + ".ba2") return true;
            return false;
        }

        void collisionBounds(Bounds& bounds, const HavokRigidBody& body)
        {
            const auto add = [&](glm::vec3 p, float radius = 0) {
                p = (body.position + body.rotation * p) * bethesdaUnitsPerMeter;
                bounds.add(p - glm::vec3(radius * bethesdaUnitsPerMeter));
                bounds.add(p + glm::vec3(radius * bethesdaUnitsPerMeter));
            };
            for (const auto& part : body.parts)
            {
                if (part.polytope.vertices.empty()) throw std::runtime_error("Native collision has an empty convex part");
                const float radius = part.polytope.convexRadius * std::max({part.scale.x, part.scale.y, part.scale.z});
                for (const auto p : part.polytope.vertices)
                    add(glm::vec3(part.transform * glm::vec4(p * part.scale, 1)), radius);
            }
            for (const auto& part : body.capsules)
            {
                const float radius = part.capsule.radius * std::max({part.scale.x, part.scale.y, part.scale.z});
                if (!std::isfinite(radius) || radius <= 0) throw std::runtime_error("Native collision has an invalid capsule");
                add(glm::vec3(part.transform * glm::vec4(part.capsule.pointA * part.scale, 1)), radius);
                add(glm::vec3(part.transform * glm::vec4(part.capsule.pointB * part.scale, 1)), radius);
            }
            if (body.triangleMesh)
            {
                if (body.triangleMesh->vertices.empty() || body.triangleMesh->indices.empty())
                    throw std::runtime_error("Native collision has an empty triangle mesh");
                for (const auto p : body.triangleMesh->vertices) add(p);
            }
        }

        struct Prepared
        {
            Fo4StagedModel model;
            std::map<std::string, AssetBytes> files;
        };

        Prepared prepare(const ESMModel& model, std::optional<std::filesystem::path> dataDirectory)
        {
            if (model.sourceEsm.empty()) throw std::runtime_error("Native model is missing its owning source ESM");
            if (!normalized(model.path).ends_with(".nif"))
                throw std::runtime_error("Save Native NIF in Asset workspace with authored collision before exporting this model");
            AssetResolver resolver(model.sourceEsm, dataDirectory);
            const auto nativePath = assetPath(model.path);
            auto source = resolver.read(nativePath);
            const auto inspection = inspectNativeModel(source.bytes);
            const auto geometry = loadNifModel(source.bytes);
            if (inspection.collision.empty())
                throw std::runtime_error("Save Native NIF in Asset workspace with authored collision: model has no supported collision");
            if (inspection.collisionReadOnlyReason.find("ragdoll") != std::string::npos
                || inspection.collisionReadOnlyReason.find("constraint") != std::string::npos)
                throw std::runtime_error("Ragdoll/constraint collision is outside the native package profile");

            Prepared result;
            if (inspection.hasAnimation || !geometry.skinBones.empty() || !geometry.animations.empty()
                || !geometry.attachmentPoints.empty() || !geometry.attachmentTargets.empty())
                throw std::runtime_error("Native model has animation, skinning or attachments whose dependency closure is not supported by the world geometry proof; use a standalone static native asset");
            result.model.modelPath = nativePath.substr(7);
            std::ranges::replace(result.model.modelPath, '/', '\\');
            Bounds bounds;
            for (const auto& surface : geometry.surfaces)
                for (const auto& vertex : surface.vertices) bounds.add(swapBethesdaAxes(vertex.position));
            if (!bounds.any) throw std::runtime_error("Native asset has no render geometry");
            for (int axis = 0; axis < 3; ++axis)
            {
                const double minimum = std::floor(bounds.minimum[axis]), maximum = std::ceil(bounds.maximum[axis]);
                if (minimum < std::numeric_limits<std::int16_t>::min() || maximum > std::numeric_limits<std::int16_t>::max())
                    throw std::runtime_error("Native asset bounds exceed Bethesda OBND limits");
                result.model.bounds[axis] = static_cast<std::int16_t>(minimum);
                result.model.bounds[axis + 3] = static_cast<std::int16_t>(maximum);
            }
            for (const auto& system : loadNifPhysicsSystems(source.bytes))
            {
                if (system.bodies.empty()) throw std::runtime_error("Native asset has no collision bodies");
                for (const auto& body : system.bodies)
                {
                    if (!hasShape(body)) throw std::runtime_error("Native asset has a body without collision");
                    if (body.dynamic != (model.kind == ESMModelKind::movableStatic))
                        throw std::runtime_error("Native collision motion does not match STAT/MSTT; fix it in the Asset workspace");
                    if (body.dynamic && (body.triangleMesh || !std::isfinite(body.mass) || body.mass <= 0
                        || glm::any(glm::lessThanEqual(body.inertiaDiagonal, glm::vec3(0)))))
                        throw std::runtime_error("MSTT requires authored dynamic convex/capsule collision with positive mass and inertia");
                    if (body.dynamic)
                    {
                        for (int axis = 0; axis < 3; ++axis)
                            if (!std::isfinite(body.inertiaDiagonal[axis]) || !std::isfinite(body.centerOfMass[axis]))
                                throw std::runtime_error("MSTT has an invalid authored mass frame");
                        const float length = glm::length(body.inertiaOrientation);
                        if (!std::isfinite(length) || std::abs(length - 1) > .001f)
                            throw std::runtime_error("MSTT has an invalid authored inertia rotation");
                    }
                    collisionBounds(bounds, body);
                }
                result.model.collision.push_back(system);
            }
            if (!inspection.collisionReadOnlyReason.empty())
                result.model.diagnostics.push_back("Native collision copied unchanged; editing guard: " + inspection.collisionReadOnlyReason);
            result.model.bethesdaMinimum = bounds.minimum; result.model.bethesdaMaximum = bounds.maximum;
            std::vector<std::string> pending{nativePath};
            pending.insert(pending.end(), inspection.dependencies.begin(), inspection.dependencies.end());
            for (const auto& surface : geometry.surfaces)
                for (const auto& dependency : {surface.materialPath, surface.diffuseTexture, surface.normalTexture})
                    if (!dependency.empty()) pending.push_back(assetPath(dependency));
            for (std::size_t i = 0; i < pending.size(); ++i)
            {
                const auto path = assetPath(pending[i]);
                if (result.files.contains(path)) continue;
                auto asset = path == nativePath ? std::move(source) : resolver.read(path);
                if (asset.source == AssetSource::project)
                {
                    const auto expected = existingPath(AssetResolver::projectAssetRoot(model.sourceEsm), path);
                    if (!std::filesystem::equivalent(expected, asset.resolvedPath))
                        throw std::runtime_error("Native dependency resolved outside its project: " + path);
                    result.model.copiedFiles.push_back(path);
                }
                else
                {
                    if (asset.source == AssetSource::looseData)
                        throw std::runtime_error("Loose Data dependency is not attested by Fallout4.esm; copy it into the owning project: " + path);
                    if (!dataDirectory || asset.resolvedPath.empty() || !baseArchive(asset.resolvedPath.filename().string())
                        || std::filesystem::weakly_canonical(asset.resolvedPath.parent_path()) != std::filesystem::weakly_canonical(*dataDirectory))
                        throw std::runtime_error("Native dependency is not from a verified base Fallout 4 archive: " + path);
                    result.model.stockDependencies.push_back(path);
                }
                result.model.dependencies.push_back({path, asset.resolvedPath, asset.source, asset.bytes.size()});
                const auto extension = std::filesystem::path(path).extension().string();
                if (extension == ".dds") (void)decodeModelImage(asset.bytes, false);
                else if (extension == ".bgsm" || extension == ".bgem")
                {
                    (void)loadMaterial(asset.bytes);
                    // Preserve optional texture slots and root materials not exposed by the editor.
                    for (std::size_t offset = 0; offset + 4 <= asset.bytes.size(); ++offset)
                    {
                        std::uint32_t length; std::memcpy(&length, asset.bytes.data() + offset, 4);
                        if (length < 5 || length > 4096 || length > asset.bytes.size() - offset - 4) continue;
                        std::string value(reinterpret_cast<const char*>(asset.bytes.data() + offset + 4), length);
                        if (!value.empty() && value.back() == '\0') value.pop_back();
                        if (std::ranges::any_of(value, [](unsigned char c) { return c < 32 || c == 127; })) continue;
                        auto extension = std::filesystem::path(AssetResolver::normalizePath(value)).extension().string();
                        if (extension == ".dds" || extension == ".bgsm" || extension == ".bgem") pending.push_back(assetPath(value));
                    }
                }
                result.files.emplace(path, std::move(asset));
            }
            return result;
        }
    }

    Fo4StagedModel inspectFo4ModelAsset(const ESMModel& model, std::optional<std::filesystem::path> dataDirectory)
    {
        return prepare(model, std::move(dataDirectory)).model;
    }

    Fo4StagedModel stageFo4ModelAsset(const ESMModel& model, const std::filesystem::path& dataStagingDirectory,
        std::optional<std::filesystem::path> dataDirectory)
    {
        if (dataStagingDirectory.empty()) throw std::runtime_error("Native package needs a staging directory");
        auto prepared = prepare(model, std::move(dataDirectory));
        std::map<std::string, std::filesystem::path> targets;
        for (const auto& [path, asset] : prepared.files)
        {
            const auto target = existingPath(dataStagingDirectory, path);
            if (std::filesystem::exists(target))
            {
                if (!std::filesystem::is_regular_file(target) || read(target) != asset.bytes)
                    throw std::runtime_error("Native package path conflict: " + path);
            }
            else if (asset.source == AssetSource::project) targets.emplace(path, target);
        }
        if (targets.empty()) return std::move(prepared.model);
        static std::atomic_uint64_t sequence{};
        const auto temporary = dataStagingDirectory / (".openf4-assets-" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()) + "-" + std::to_string(sequence++));
        std::filesystem::create_directories(dataStagingDirectory);
        if (!std::filesystem::create_directory(temporary)) throw std::runtime_error("Native package staging collision");
        try
        {
            unsigned index = 0;
            for (const auto& [path, target] : targets)
            {
                const auto staged = temporary / std::to_string(index++);
                std::ofstream output(staged, std::ios::binary);
                const auto& bytes = prepared.files.at(path).bytes;
                output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
                output.close();
                if (!output) throw std::runtime_error("Cannot stage native asset: " + path);
                std::filesystem::create_directories(target.parent_path());
                std::filesystem::create_hard_link(staged, target);
            }
            std::filesystem::remove_all(temporary);
        }
        catch (...)
        {
            std::error_code ignored; std::filesystem::remove_all(temporary, ignored);
            throw;
        }
        return std::move(prepared.model);
    }
}
