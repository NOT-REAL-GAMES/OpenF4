#include "editor/asset_document.hpp"

#include "assets/asset_resolver.hpp"
#include "assets/coordinates.hpp"
#include "assets/material_file.hpp"
#include "assets/model_import.hpp"
#include "editor/mesh_edit.hpp"
#include "physics/convex_decomposition.hpp"
#include "physics/jolt_collision.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <functional>
#include <stdexcept>

#include <glm/ext/matrix_transform.hpp>

namespace openf4::editor
{
    namespace
    {
        std::vector<std::byte> readFile(const std::filesystem::path& path)
        {
            std::ifstream file(path, std::ios::binary | std::ios::ate);
            if (!file || file.tellg() < 0) throw std::runtime_error("Cannot read asset: " + path.string());
            std::vector<std::byte> bytes(static_cast<std::size_t>(file.tellg()));
            file.seekg(0);
            if (!file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size())))
                throw std::runtime_error("Cannot read complete asset: " + path.string());
            return bytes;
        }

        std::string assetPath(std::string path, std::string_view prefix)
        {
            path = assets::AssetResolver::normalizePath(path);
            std::ranges::replace(path, '\\', '/');
            if (!path.starts_with(prefix)) path = std::string(prefix) + path;
            return path;
        }

        template<glm::length_t N> bool finite(const glm::vec<N, float>& value)
        {
            for (glm::length_t i = 0; i < N; ++i) if (!std::isfinite(value[i])) return false;
            return true;
        }

        bool hasShape(const assets::HavokRigidBody& body)
        {
            return !body.parts.empty() || !body.capsules.empty() || body.triangleMesh.has_value();
        }

        void validateBody(const assets::HavokRigidBody& body)
        {
            if (!finite(body.position) || !std::isfinite(body.mass)
                || (body.dynamic ? body.mass <= 0 : body.mass != 0)
                || !std::isfinite(body.friction) || body.friction < 0
                || !std::isfinite(body.restitution) || body.restitution < 0 || body.restitution > 1
                || !std::isfinite(body.linearDamping) || body.linearDamping < 0
                || !std::isfinite(body.angularDamping) || body.angularDamping < 0
                || !std::isfinite(body.gravityFactor))
                throw std::runtime_error("Invalid collision body properties");
            if (hasShape(body))
            {
                (void)physics::makeJoltCollisionShape(body);
                auto validation = body; validation.bodyIndex = 0;
                // Vertex-edited non-convex parts are previewed through their hull and refused
                // at save (see requireConvexParts); validate everything else for export now.
                std::erase_if(validation.parts, [](const assets::HavokConvexPart& part) {
                    return !physics::polytopeIsConvex(part.polytope.vertices, part.polytope.indices); });
                if (hasShape(validation))
                    (void)assets::serializeHavokPhysicsSystem({{std::move(validation)}, {}});
            }
            else if (body.dynamic) throw std::runtime_error("Generate collision before making the asset movable");
        }

        glm::mat4 partMatrix(const assets::HavokConvexPart& part)
        {
            return glm::scale(part.transform, part.scale);
        }

        void requireConvexParts(const std::vector<AssetCollisionSystem>& systems)
        {
            for (std::size_t s = 0; s < systems.size(); ++s)
                for (std::size_t b = 0; b < systems[s].physics.bodies.size(); ++b)
                {
                    const auto& parts = systems[s].physics.bodies[b].parts;
                    for (std::size_t i = 0; i < parts.size(); ++i)
                        if (!physics::polytopeIsConvex(parts[i].polytope.vertices, parts[i].polytope.indices))
                            throw std::runtime_error("Convex collision part " + std::to_string(i) + " (body "
                                + std::to_string(s) + ":" + std::to_string(b) + ") is not convex. Use Split optimally"
                                " or Re-hull in collision edit (C) before saving; Fallout 4 requires convex parts.");
                }
        }

        // Body-local engine-space triangles of the given surfaces.
        void gatherSurfaces(const assets::ModelGeometry& geometry, std::span<const std::size_t> surfaces,
            const glm::mat4& local, std::vector<glm::vec3>& vertices, std::vector<std::uint32_t>& indices)
        {
            for (auto index : surfaces)
            {
                const auto& surface = geometry.surfaces.at(index);
                const auto base = static_cast<std::uint32_t>(vertices.size());
                for (const auto& vertex : surface.vertices) vertices.push_back(glm::vec3(local * glm::vec4(vertex.position, 1)));
                for (const auto vertex : surface.indices)
                {
                    if (vertex >= surface.vertices.size()) throw std::runtime_error("Surface has invalid triangle indices");
                    indices.push_back(base + vertex);
                }
            }
        }

        glm::mat4 toPartMatrix(const glm::mat4& transform, glm::vec3& scale)
        {
            for (int i = 0; i < 4; ++i)
                if (!finite(transform[i])) throw std::runtime_error("Collision transform is not finite");
            if (glm::length(glm::vec4(transform[0].w, transform[1].w, transform[2].w, transform[3].w - 1)) > 1e-5f)
                throw std::runtime_error("Collision transform must be affine");
            auto native = assets::swapBethesdaAxes(transform);
            native[3] = glm::vec4(assets::engineToHavokPosition(glm::vec3(transform[3])), 1);
            for (int i = 0; i < 3; ++i)
            {
                scale[i] = glm::length(glm::vec3(native[i]));
                if (scale[i] <= 0) throw std::runtime_error("Collision scale must be positive");
                native[i] /= scale[i];
            }
            return native;
        }
    }

    struct AssetDocument::Impl
    {
        struct State
        {
            std::vector<AssetMaterial> materials;
            std::vector<AssetCollisionSystem> collision;
            // Copy-on-write: unchanged by material/collision edits, replaced by mesh edits.
            std::shared_ptr<const assets::ModelGeometry> geometry;
            std::uint64_t serial = 0;
        };
        std::filesystem::path sourceEsm;
        std::string modelPath;
        std::optional<std::filesystem::path> dataDirectory;
        std::vector<std::byte> bytes;
        AssetCapabilities capabilities;
        std::vector<std::string> diagnostics;
        bool preserveNative = false;
        State state;
        std::vector<State> past, future;
        std::optional<State> transaction;
        unsigned transactionDepth = 0;
        std::uint64_t revision = 0, serial = 0, savedSerial = 0;

        void initialize(assets::ModelGeometry geometry, bool allowEmpty = false)
        {
            if (geometry.surfaces.empty() && !allowEmpty) throw std::runtime_error("Asset contains no renderable surfaces");
            for (auto& surface : geometry.surfaces)
            {
                AssetMaterial material{surface.baseColor, surface.alphaCutoff, surface.alphaBlend,
                    surface.doubleSided, surface.diffuseTexture, surface.normalTexture};
                if (surface.diffuseImage)
                    material.diffuseImage = std::make_shared<const assets::ModelImage>(std::move(*surface.diffuseImage));
                if (surface.normalImage)
                    material.normalImage = std::make_shared<const assets::ModelImage>(std::move(*surface.normalImage));
                surface.diffuseImage.reset(); surface.normalImage.reset();
                state.materials.push_back(std::move(material));
            }
            if ((!geometry.skinBones.empty() || !geometry.animations.empty()
                    || !geometry.attachmentPoints.empty() || !geometry.attachmentTargets.empty()) && !preserveNative)
            {
                capabilities = {false, false, false,
                    "Animated, skinned, or attachment-bearing imports cannot be converted losslessly"};
            }
            state.geometry = std::make_shared<const assets::ModelGeometry>(std::move(geometry));
            if (state.collision.empty())
                state.collision.push_back({std::nullopt, {{assets::HavokRigidBody{}}, {}}});
        }

        void requireMesh() const
        {
            if (!capabilities.materials)
                throw std::runtime_error(capabilities.reason.empty() ? "This asset's geometry is read-only" : capabilities.reason);
        }

        void requireCollision() const
        {
            if (!capabilities.collision) throw std::runtime_error(capabilities.reason);
        }

        void commit(State next)
        {
            if (!transaction)
            {
                past.push_back(state);
                if (past.size() > 64) past.erase(past.begin());
                future.clear();
            }
            next.serial = ++serial; state = std::move(next); ++revision;
        }

        void replaceBody(std::size_t system, std::size_t body, assets::HavokRigidBody value, bool deriveMass)
        {
            requireCollision();
            const auto& original = state.collision.at(system).physics.bodies.at(body);
            if (value.bodyIndex != original.bodyIndex || (!capabilities.motion && value.dynamic != original.dynamic))
                throw std::runtime_error("Cannot change native body identities or motion flags");
            if (preserveNative && !hasShape(value)) throw std::runtime_error("Native body attachment cannot be removed");
            if (deriveMass) physics::updateHavokMassProperties(value);
            validateBody(value);
            auto next = state; next.collision[system].physics.bodies[body] = std::move(value);
            commit(std::move(next));
        }
    };

    AssetDocument::AssetDocument(assets::ModelGeometry geometry, bool allowEmpty) : mImpl(std::make_unique<Impl>())
    {
        mImpl->initialize(std::move(geometry), allowEmpty);
    }
    AssetDocument AssetDocument::createEmpty() { return AssetDocument(assets::ModelGeometry{}, true); }

    AssetDocument::AssetDocument(std::filesystem::path sourceEsm, std::string modelPath,
        std::optional<std::filesystem::path> dataDirectory, std::optional<bool> movableRecordHint)
        : mImpl(std::make_unique<Impl>())
    {
        auto& data = *mImpl;
        data.sourceEsm = std::move(sourceEsm); data.modelPath = std::move(modelPath);
        data.dataDirectory = std::move(dataDirectory);
        assets::AssetResolver resolver(data.sourceEsm, data.dataDirectory);
        assets::ModelGeometry geometry;
        const auto normalized = assets::AssetResolver::normalizePath(data.modelPath);
        if (normalized.ends_with(".nif"))
        {
            data.bytes = resolver.extract(assetPath(data.modelPath, "meshes/"));
            geometry = assets::loadNifModel(data.bytes);
            try
            {
                const auto native = assets::inspectNativeModel(data.bytes);
                data.preserveNative = !native.authoredByOpenF4;
                data.capabilities.materials = !data.preserveNative;
                data.capabilities.motion = !data.preserveNative;
                data.capabilities.collision = native.collisionReadOnlyReason.empty();
                data.capabilities.reason = native.collisionReadOnlyReason;
                if (data.preserveNative && data.capabilities.reason.empty())
                    data.capabilities.reason = "Existing native NIF: collision-only edits preserve all other bytes and attachments";
                for (const auto& system : native.collision)
                    data.state.collision.push_back({system.blockIndex, system.physics});
            }
            catch (const std::exception& e)
            {
                data.preserveNative = true;
                data.capabilities = {false, false, false, e.what()};
            }
            for (auto& surface : geometry.surfaces)
            {
                try
                {
                    if (!surface.materialPath.empty())
                    {
                        const auto materialBytes = resolver.extract(assetPath(surface.materialPath, "materials/"));
                        const auto material = assets::loadMaterial(materialBytes);
                        if (data.capabilities.materials && !assets::nativeMaterialIsEditable(materialBytes))
                        {
                            data.preserveNative = true; data.capabilities.materials = false;
                            data.capabilities.motion = false;
                            data.capabilities.collision = !data.state.collision.empty();
                            data.capabilities.reason = "Native material contains unsupported fields; collision-only edits preserve it unchanged";
                        }
                        if (!material.diffuseTexture.empty()) surface.diffuseTexture = material.diffuseTexture;
                        if (!material.normalTexture.empty()) surface.normalTexture = material.normalTexture;
                        surface.baseColor *= material.baseColor;
                        surface.alphaCutoff = material.alphaTest ? material.alphaCutoff : 0;
                        surface.alphaBlend = material.alphaBlend;
                        surface.doubleSided = material.twoSided;
                        if (material.grayscaleToPalette)
                            data.diagnostics.push_back("Grayscale palette preview is not available in the asset workspace");
                    }
                    if (!surface.diffuseTexture.empty())
                        surface.diffuseImage = assets::decodeModelImage(
                            resolver.extractTexture(assetPath(surface.diffuseTexture, "textures/")), true);
                    if (!surface.normalTexture.empty())
                        surface.normalImage = assets::decodeModelImage(
                            resolver.extractTexture(assetPath(surface.normalTexture, "textures/")), false);
                }
                catch (const std::exception& e)
                {
                    data.diagnostics.push_back(e.what());
                    if (data.capabilities.materials)
                    {
                        data.preserveNative = true; data.capabilities.materials = false;
                        data.capabilities.motion = false;
                        data.capabilities.collision = data.capabilities.collision && !data.state.collision.empty();
                        data.capabilities.reason = "Native material or texture could not be loaded; only source-preserving collision edits are allowed";
                    }
                }
            }
        }
        else
        {
            const auto path = resolver.resolveLoose(data.modelPath);
            if (!path) throw std::runtime_error("Imported model must resolve to a project or loose file");
            auto imported = assets::importModel(*path);
            geometry = std::move(imported.geometry); data.diagnostics = std::move(imported.warnings);
            data.bytes = readFile(*path);
        }
        data.initialize(std::move(geometry));
        if (movableRecordHint && data.capabilities.motion)
        {
            if (*movableRecordHint && std::ranges::any_of(data.state.collision, [](const auto& system) {
                return std::ranges::any_of(system.physics.bodies, [](const auto& body) { return body.triangleMesh.has_value(); });
            }))
            {
                data.capabilities = {false, false, false,
                    "MSTT record has static triangle collision; this incompatible native asset is inspect-only"};
                data.diagnostics.push_back(data.capabilities.reason);
                return;
            }
            for (auto& system : data.state.collision)
                for (auto& body : system.physics.bodies)
                {
                    if (body.dynamic != *movableRecordHint)
                    {
                        body.dynamic = *movableRecordHint;
                        body.mass = body.dynamic ? 10.0f : 0.0f;
                        if (hasShape(body) || !body.dynamic) physics::updateHavokMassProperties(body);
                    }
                    if (hasShape(body)) validateBody(body);
                    else if (body.dynamic) data.diagnostics.push_back("Source movable record had no collision; its native version requires an authored shape");
                }
        }
    }

    AssetDocument::~AssetDocument() = default;
    AssetDocument::AssetDocument(AssetDocument&&) noexcept = default;
    AssetDocument& AssetDocument::operator=(AssetDocument&&) noexcept = default;
    const assets::ModelGeometry& AssetDocument::sourceGeometry() const noexcept { return *mImpl->state.geometry; }
    std::shared_ptr<const assets::ModelGeometry> AssetDocument::geometryHandle() const noexcept { return mImpl->state.geometry; }
    bool AssetDocument::meshEditable() const noexcept { return mImpl->capabilities.materials; }
    const std::vector<AssetMaterial>& AssetDocument::materials() const noexcept { return mImpl->state.materials; }
    const std::vector<AssetCollisionSystem>& AssetDocument::collisionSystems() const noexcept { return mImpl->state.collision; }
    const AssetCapabilities& AssetDocument::capabilities() const noexcept { return mImpl->capabilities; }
    const std::vector<std::string>& AssetDocument::diagnostics() const noexcept { return mImpl->diagnostics; }
    const std::filesystem::path& AssetDocument::sourceEsm() const noexcept { return mImpl->sourceEsm; }
    const std::string& AssetDocument::modelPath() const noexcept { return mImpl->modelPath; }
    std::span<const std::byte> AssetDocument::sourceBytes() const noexcept { return mImpl->bytes; }
    bool AssetDocument::dirty() const noexcept { return mImpl->state.serial != mImpl->savedSerial; }
    std::uint64_t AssetDocument::revision() const noexcept { return mImpl->revision; }
    bool AssetDocument::movable() const noexcept
    {
        for (const auto& system : mImpl->state.collision)
            for (const auto& body : system.physics.bodies) if (body.dynamic) return true;
        return false;
    }

    void AssetDocument::beginTransaction()
    {
        if (!mImpl->transactionDepth++) mImpl->transaction = mImpl->state;
    }
    void AssetDocument::endTransaction()
    {
        if (!mImpl->transactionDepth) throw std::runtime_error("No asset transaction is active");
        if (--mImpl->transactionDepth) return;
        if (mImpl->transaction->serial != mImpl->state.serial)
        {
            mImpl->past.push_back(std::move(*mImpl->transaction));
            if (mImpl->past.size() > 64) mImpl->past.erase(mImpl->past.begin());
            mImpl->future.clear();
        }
        mImpl->transaction.reset();
    }
    void AssetDocument::cancelTransaction()
    {
        if (!mImpl->transaction) return;
        mImpl->state = std::move(*mImpl->transaction); mImpl->transaction.reset();
        mImpl->transactionDepth = 0; ++mImpl->revision;
    }
    bool AssetDocument::undo()
    {
        if (mImpl->transaction || mImpl->past.empty()) return false;
        mImpl->future.push_back(std::move(mImpl->state)); mImpl->state = std::move(mImpl->past.back());
        mImpl->past.pop_back(); ++mImpl->revision; return true;
    }
    bool AssetDocument::redo()
    {
        if (mImpl->transaction || mImpl->future.empty()) return false;
        mImpl->past.push_back(std::move(mImpl->state)); mImpl->state = std::move(mImpl->future.back());
        mImpl->future.pop_back(); ++mImpl->revision; return true;
    }
    void AssetDocument::markSaved()
    {
        if (mImpl->transaction) throw std::runtime_error("Finish the asset gesture before saving");
        mImpl->savedSerial = mImpl->state.serial;
    }

    void AssetDocument::setMaterial(std::size_t surface, AssetMaterial material)
    {
        if (!mImpl->capabilities.materials) throw std::runtime_error(mImpl->capabilities.reason);
        if (!finite(material.baseColor) || glm::any(glm::lessThan(material.baseColor, glm::vec4(0)))
            || glm::any(glm::greaterThan(material.baseColor, glm::vec4(1)))
            || !std::isfinite(material.alphaCutoff) || material.alphaCutoff < 0 || material.alphaCutoff > 1)
            throw std::runtime_error("Invalid asset material factors");
        if (mImpl->state.materials.at(surface) == material) return;
        auto next = mImpl->state; next.materials[surface] = std::move(material); mImpl->commit(std::move(next));
    }
    void AssetDocument::setTexture(std::size_t surface, const std::filesystem::path& texture, bool normal)
    {
        if (!mImpl->capabilities.materials) throw std::runtime_error(mImpl->capabilities.reason);
        auto material = mImpl->state.materials.at(surface);
        auto& image = normal ? material.normalImage : material.diffuseImage;
        auto& path = normal ? material.normalTexture : material.diffuseTexture;
        if (texture.empty()) { image.reset(); path.clear(); }
        else
        {
            image = std::make_shared<const assets::ModelImage>(assets::decodeModelImage(readFile(texture), !normal));
            path = texture.string();
        }
        setMaterial(surface, std::move(material));
    }

    AssetBodyProperties AssetDocument::bodyProperties(std::size_t system, std::size_t body) const
    {
        const auto& value = mImpl->state.collision.at(system).physics.bodies.at(body);
        return {value.mass, value.friction, value.restitution, value.linearDamping, value.angularDamping, value.gravityFactor};
    }
    void AssetDocument::setBodyProperties(std::size_t system, std::size_t body, const AssetBodyProperties& value)
    {
        mImpl->requireCollision();
        if (bodyProperties(system, body) == value) return;
        auto next = mImpl->state.collision.at(system).physics.bodies.at(body);
        const bool massChanged = next.mass != value.mass;
        if (!next.dynamic && massChanged) throw std::runtime_error("Static collision has no editable mass");
        next.mass = value.mass; next.friction = value.friction; next.restitution = value.restitution;
        next.linearDamping = value.linearDamping; next.angularDamping = value.angularDamping;
        next.gravityFactor = value.gravityFactor;
        mImpl->replaceBody(system, body, std::move(next), massChanged);
    }
    void AssetDocument::setMovable(bool movable, float mass)
    {
        mImpl->requireCollision();
        const bool changed = std::ranges::any_of(mImpl->state.collision, [movable](const auto& system) {
            return std::ranges::any_of(system.physics.bodies, [movable](const auto& body) { return body.dynamic != movable; });
        });
        if (!changed) return;
        if (!mImpl->capabilities.motion) throw std::runtime_error("Existing NIF motion flags are preserved");
        auto next = mImpl->state;
        for (auto& system : next.collision)
            for (auto& body : system.physics.bodies)
            {
                if (body.dynamic == movable) continue;
                body.dynamic = movable; body.mass = movable ? mass : 0;
                physics::updateHavokMassProperties(body); validateBody(body);
            }
        mImpl->commit(std::move(next));
    }

    void AssetDocument::generateCollision(std::span<const std::size_t> surfaces, bool convex,
        std::size_t system, std::size_t bodyIndex)
    {
        mImpl->requireCollision();
        if (surfaces.empty()) throw std::runtime_error("Select at least one surface");
        auto body = mImpl->state.collision.at(system).physics.bodies.at(bodyIndex);
        if (!convex && body.dynamic) throw std::runtime_error("Triangle collision requires a static body");
        std::vector<glm::vec3> vertices;
        std::vector<std::uint32_t> indices;
        gatherSurfaces(*mImpl->state.geometry, surfaces, glm::inverse(physics::havokBodyLocalTransform(body)), vertices, indices);
        body.parts.clear(); body.capsules.clear(); body.triangleMesh.reset();
        if (convex) body.parts.push_back({glm::mat4(1), glm::vec3(1), physics::makeHavokConvexHull(vertices)});
        else
        {
            assets::HavokTriangleMesh mesh;
            for (const auto position : vertices) mesh.vertices.push_back(assets::engineToHavokPosition(position));
            mesh.indices = std::move(indices);
            body.triangleMesh = std::move(mesh);
        }
        mImpl->replaceBody(system, bodyIndex, std::move(body), true);
    }

    void AssetDocument::addBox(glm::vec3 center, glm::vec3 halfExtent, std::size_t system, std::size_t bodyIndex)
    {
        mImpl->requireCollision();
        if (!finite(center) || !finite(halfExtent) || glm::any(glm::lessThanEqual(halfExtent, glm::vec3(0))))
            throw std::runtime_error("Box half extents must be positive and finite");
        auto body = mImpl->state.collision.at(system).physics.bodies.at(bodyIndex);
        if (body.triangleMesh) throw std::runtime_error("Replace triangle collision with convex collision before adding parts");
        std::array<glm::vec3, 8> points;
        for (unsigned i = 0; i < 8; ++i)
            points[i] = halfExtent * glm::vec3(i & 1 ? 1 : -1, i & 2 ? 1 : -1, i & 4 ? 1 : -1);
        assets::HavokConvexPart part;
        part.polytope = physics::makeHavokConvexHull(points);
        part.transform[3] = glm::vec4(assets::engineToHavokPosition(center), 1);
        body.parts.push_back(std::move(part)); mImpl->replaceBody(system, bodyIndex, std::move(body), true);
    }
    void AssetDocument::addCapsule(glm::vec3 pointA, glm::vec3 pointB, float radius,
        std::size_t system, std::size_t bodyIndex)
    {
        mImpl->requireCollision();
        if (!finite(pointA) || !finite(pointB) || !std::isfinite(radius) || radius <= 0)
            throw std::runtime_error("Capsule endpoints and radius must be finite with a positive radius");
        auto body = mImpl->state.collision.at(system).physics.bodies.at(bodyIndex);
        if (body.triangleMesh) throw std::runtime_error("Replace triangle collision with convex collision before adding parts");
        body.capsules.push_back({glm::mat4(1), glm::vec3(1), {assets::engineToHavokPosition(pointA),
            assets::engineToHavokPosition(pointB), radius / assets::bethesdaUnitsPerMeter}});
        mImpl->replaceBody(system, bodyIndex, std::move(body), true);
    }
    void AssetDocument::setCapsuleGeometry(std::size_t part, glm::vec3 pointA, glm::vec3 pointB, float radius,
        std::size_t system, std::size_t bodyIndex)
    {
        mImpl->requireCollision();
        if (!finite(pointA) || !finite(pointB) || !std::isfinite(radius) || radius <= 0)
            throw std::runtime_error("Capsule endpoints and radius must be finite with a positive radius");
        auto body = mImpl->state.collision.at(system).physics.bodies.at(bodyIndex);
        auto& shape = body.capsules.at(part).capsule;
        const assets::HavokCapsule next{assets::engineToHavokPosition(pointA),
            assets::engineToHavokPosition(pointB), radius / assets::bethesdaUnitsPerMeter};
        if (shape.pointA == next.pointA && shape.pointB == next.pointB && shape.radius == next.radius) return;
        shape = next;
        mImpl->replaceBody(system, bodyIndex, std::move(body), true);
    }
    void AssetDocument::erasePart(AssetCollisionPart kind, std::size_t part, std::size_t system, std::size_t bodyIndex)
    {
        auto body = mImpl->state.collision.at(system).physics.bodies.at(bodyIndex);
        if (kind == AssetCollisionPart::convex)
        {
            (void)body.parts.at(part); body.parts.erase(body.parts.begin() + part);
        }
        else if (kind == AssetCollisionPart::capsule)
        {
            (void)body.capsules.at(part); body.capsules.erase(body.capsules.begin() + part);
        }
        else
        {
            if (part || !body.triangleMesh) throw std::runtime_error("Invalid triangle collision part");
            body.triangleMesh.reset();
        }
        mImpl->replaceBody(system, bodyIndex, std::move(body), true);
    }
    glm::mat4 AssetDocument::partTransform(AssetCollisionPart kind, std::size_t part,
        std::size_t system, std::size_t bodyIndex) const
    {
        const auto& body = mImpl->state.collision.at(system).physics.bodies.at(bodyIndex);
        glm::mat4 transform;
        if (kind == AssetCollisionPart::convex)
        {
            const auto& shape = body.parts.at(part); transform = glm::scale(shape.transform, shape.scale);
        }
        else if (kind == AssetCollisionPart::capsule)
        {
            const auto& shape = body.capsules.at(part); transform = glm::scale(shape.transform, shape.scale);
        }
        else throw std::runtime_error("Triangle collision has no independent part transform");
        const auto translation = assets::havokToEnginePosition(glm::vec3(transform[3]));
        transform = assets::swapBethesdaAxes(transform); transform[3] = glm::vec4(translation, 1);
        return transform;
    }
    void AssetDocument::setPartTransform(AssetCollisionPart kind, std::size_t part,
        const glm::mat4& transform, std::size_t system, std::size_t bodyIndex)
    {
        mImpl->requireCollision();
        if (partTransform(kind, part, system, bodyIndex) == transform) return;
        auto body = mImpl->state.collision.at(system).physics.bodies.at(bodyIndex);
        glm::vec3 scale;
        const auto native = toPartMatrix(transform, scale);
        if (kind == AssetCollisionPart::convex)
        {
            auto& shape = body.parts.at(part); shape.transform = native; shape.scale = scale;
        }
        else if (kind == AssetCollisionPart::capsule)
        {
            auto& shape = body.capsules.at(part); shape.transform = native; shape.scale = scale;
        }
        else throw std::runtime_error("Triangle collision has no independent part transform");
        mImpl->replaceBody(system, bodyIndex, std::move(body), true);
    }

    std::vector<glm::vec3> AssetDocument::convexPartPositions(std::size_t part, std::size_t system, std::size_t bodyIndex) const
    {
        const auto& shape = mImpl->state.collision.at(system).physics.bodies.at(bodyIndex).parts.at(part);
        const auto matrix = partMatrix(shape);
        std::vector<glm::vec3> result;
        result.reserve(shape.polytope.vertices.size());
        for (const auto v : shape.polytope.vertices)
            result.push_back(assets::havokToEnginePosition(glm::vec3(matrix * glm::vec4(v, 1))));
        return result;
    }
    std::vector<std::uint32_t> AssetDocument::convexPartIndices(std::size_t part, std::size_t system, std::size_t bodyIndex) const
    {
        return mImpl->state.collision.at(system).physics.bodies.at(bodyIndex).parts.at(part).polytope.indices;
    }
    void AssetDocument::setConvexPartPositions(std::size_t part, std::span<const glm::vec3> positions,
        std::size_t system, std::size_t bodyIndex)
    {
        mImpl->requireCollision();
        const auto current = convexPartPositions(part, system, bodyIndex);
        if (positions.size() != current.size()) throw std::runtime_error("Convex vertex count mismatch");
        bool changed = false;
        for (std::size_t i = 0; i < positions.size(); ++i)
        {
            if (!finite(positions[i]) || glm::any(glm::greaterThan(glm::abs(positions[i]), glm::vec3(1e7f))))
                throw std::runtime_error("Collision vertices must be finite and within 1e7 units");
            changed |= positions[i] != current[i];
        }
        if (!changed) return;
        auto body = mImpl->state.collision.at(system).physics.bodies.at(bodyIndex);
        auto& shape = body.parts.at(part);
        const auto inverse = glm::inverse(partMatrix(shape));
        // Untouched vertices keep their exact stored value (no float round-trip drift).
        for (std::size_t i = 0; i < positions.size(); ++i)
            if (positions[i] != current[i])
                shape.polytope.vertices[i] = glm::vec3(inverse * glm::vec4(assets::engineToHavokPosition(positions[i]), 1));
        mImpl->replaceBody(system, bodyIndex, std::move(body), true);
    }
    bool AssetDocument::convexPartIsConvex(std::size_t part, std::size_t system, std::size_t bodyIndex) const
    {
        const auto& polytope = mImpl->state.collision.at(system).physics.bodies.at(bodyIndex).parts.at(part).polytope;
        return physics::polytopeIsConvex(polytope.vertices, polytope.indices);
    }
    physics::ConvexityReport AssetDocument::convexPartConvexity(std::size_t part, std::size_t system, std::size_t bodyIndex) const
    {
        const auto positions = convexPartPositions(part, system, bodyIndex);
        const auto& polytope = mImpl->state.collision.at(system).physics.bodies.at(bodyIndex).parts.at(part).polytope;
        auto report = physics::measureConvexity(positions, polytope.indices, 0.0f);
        report.convex = physics::polytopeIsConvex(polytope.vertices, polytope.indices);
        if (report.convex) report.depth = 0;
        return report;
    }
    bool AssetDocument::collisionConvex() const
    {
        try { requireConvexParts(mImpl->state.collision); return true; }
        catch (const std::runtime_error&) { return false; }
    }
    void AssetDocument::rehullConvexPart(std::size_t part, std::size_t system, std::size_t bodyIndex)
    {
        mImpl->requireCollision();
        auto body = mImpl->state.collision.at(system).physics.bodies.at(bodyIndex);
        auto& shape = body.parts.at(part);
        std::vector<glm::vec3> local;
        for (const auto v : shape.polytope.vertices) local.push_back(assets::havokToEnginePosition(v));
        const float radius = shape.polytope.convexRadius;
        shape.polytope = physics::makeHavokConvexHull(local);
        shape.polytope.convexRadius = radius;
        mImpl->replaceBody(system, bodyIndex, std::move(body), true);
    }
    std::size_t AssetDocument::splitConvexPart(std::size_t part, const physics::ConvexDecompositionSettings& settings,
        std::size_t system, std::size_t bodyIndex)
    {
        mImpl->requireCollision();
        auto body = mImpl->state.collision.at(system).physics.bodies.at(bodyIndex);
        const auto source = body.parts.at(part);
        if (source.polytope.indices.empty()) throw std::runtime_error("Part has no faces; it is already its own convex hull");
        // Decompose in the polytope's own frame so every piece keeps the part transform.
        std::vector<glm::vec3> local;
        for (const auto v : source.polytope.vertices) local.push_back(assets::havokToEnginePosition(v));
        const auto pieces = physics::decomposeConvex(local, source.polytope.indices, settings);
        std::vector<assets::HavokConvexPart> parts;
        for (const auto& piece : pieces)
        {
            assets::HavokConvexPart next{source.transform, source.scale, physics::makeHavokConvexHull(piece)};
            next.polytope.convexRadius = source.polytope.convexRadius;
            parts.push_back(std::move(next));
        }
        body.parts.erase(body.parts.begin() + static_cast<std::ptrdiff_t>(part));
        body.parts.insert(body.parts.begin() + static_cast<std::ptrdiff_t>(part), parts.begin(), parts.end());
        mImpl->replaceBody(system, bodyIndex, std::move(body), true);
        return parts.size();
    }
    physics::ConvexityReport AssetDocument::surfaceConvexity(std::span<const std::size_t> surfaces) const
    {
        std::vector<glm::vec3> vertices; std::vector<std::uint32_t> indices;
        gatherSurfaces(*mImpl->state.geometry, surfaces, glm::mat4(1), vertices, indices);
        return physics::measureConvexity(vertices, indices);
    }
    std::size_t AssetDocument::generateConvexDecomposition(std::span<const std::size_t> surfaces,
        const physics::ConvexDecompositionSettings& settings, std::size_t system, std::size_t bodyIndex)
    {
        mImpl->requireCollision();
        if (surfaces.empty()) throw std::runtime_error("Select at least one surface");
        auto body = mImpl->state.collision.at(system).physics.bodies.at(bodyIndex);
        std::vector<glm::vec3> vertices; std::vector<std::uint32_t> indices;
        gatherSurfaces(*mImpl->state.geometry, surfaces, glm::inverse(physics::havokBodyLocalTransform(body)), vertices, indices);
        const auto pieces = physics::decomposeConvex(vertices, indices, settings);
        body.parts.clear(); body.capsules.clear(); body.triangleMesh.reset();
        for (const auto& piece : pieces) body.parts.push_back({glm::mat4(1), glm::vec3(1), physics::makeHavokConvexHull(piece)});
        mImpl->replaceBody(system, bodyIndex, std::move(body), true);
        return pieces.size();
    }

    std::size_t AssetDocument::addSurface(assets::ModelSurface surface, AssetMaterial material)
    {
        mImpl->requireMesh();
        if (surface.vertices.empty() || surface.indices.empty() || surface.indices.size() % 3)
            throw std::runtime_error("New surface requires indexed triangles");
        for (const auto index : surface.indices)
            if (index >= surface.vertices.size()) throw std::runtime_error("New surface has invalid triangle indices");
        for (const auto& vertex : surface.vertices)
            if (!finite(vertex.position) || !finite(vertex.normal)) throw std::runtime_error("New surface has non-finite vertices");
        surface.diffuseImage.reset(); surface.normalImage.reset();
        auto next = mImpl->state;
        auto geometry = std::make_shared<assets::ModelGeometry>(*next.geometry);
        geometry->surfaces.push_back(std::move(surface));
        next.geometry = std::move(geometry);
        next.materials.push_back(std::move(material));
        const auto index = next.materials.size() - 1;
        mImpl->commit(std::move(next));
        return index;
    }
    void AssetDocument::setSurfacePositions(std::size_t surface, std::span<const glm::vec3> positions)
    {
        mImpl->requireMesh();
        const auto& current = mImpl->state.geometry->surfaces.at(surface);
        if (positions.size() != current.vertices.size()) throw std::runtime_error("Vertex count mismatch");
        bool changed = false;
        for (std::size_t i = 0; i < positions.size(); ++i)
        {
            if (!finite(positions[i]) || glm::any(glm::greaterThan(glm::abs(positions[i]), glm::vec3(1e7f))))
                throw std::runtime_error("Vertex positions must be finite and within 1e7 units");
            changed |= positions[i] != current.vertices[i].position;
        }
        if (!changed) return;
        auto next = mImpl->state;
        auto geometry = std::make_shared<assets::ModelGeometry>(*next.geometry);
        mesh::applyPositions(geometry->surfaces[surface], positions);
        next.geometry = std::move(geometry);
        mImpl->commit(std::move(next));
    }

    void AssetDocument::eraseSurface(std::size_t surface)
    {
        mImpl->requireMesh();
        if (surface >= mImpl->state.geometry->surfaces.size() || surface >= mImpl->state.materials.size())
            throw std::runtime_error("Surface no longer exists");
        auto next = mImpl->state;
        auto geometry = std::make_shared<assets::ModelGeometry>(*next.geometry);
        geometry->surfaces.erase(geometry->surfaces.begin() + static_cast<std::ptrdiff_t>(surface));
        next.geometry = std::move(geometry);
        next.materials.erase(next.materials.begin() + static_cast<std::ptrdiff_t>(surface));
        mImpl->commit(std::move(next));
    }

    assets::ModelGeometry AssetDocument::geometryForExport() const
    {
        auto result = *mImpl->state.geometry;
        for (std::size_t i = 0; i < result.surfaces.size(); ++i)
        {
            auto& surface = result.surfaces[i]; const auto& material = mImpl->state.materials[i];
            surface.baseColor = material.baseColor; surface.alphaCutoff = material.alphaCutoff;
            surface.alphaBlend = material.alphaBlend; surface.doubleSided = material.doubleSided;
            surface.diffuseTexture = material.diffuseTexture; surface.normalTexture = material.normalTexture;
            if (material.diffuseImage) surface.diffuseImage = *material.diffuseImage;
            else surface.diffuseImage.reset();
            if (material.normalImage) surface.normalImage = *material.normalImage;
            else surface.normalImage.reset();
        }
        return result;
    }
    assets::NativeModelPackage AssetDocument::saveNative(const std::filesystem::path& worldEsm, std::uint32_t formId) const
    {
        if (mImpl->transaction) throw std::runtime_error("Finish the asset gesture before saving");
        if (!mImpl->capabilities.materials && !mImpl->capabilities.collision)
            throw std::runtime_error(mImpl->capabilities.reason);
        requireConvexParts(mImpl->state.collision);
        if (mImpl->preserveNative)
        {
            std::vector<assets::NativeCollisionBlock> blocks;
            for (const auto& collision : mImpl->state.collision)
            {
                if (!collision.blockIndex) throw std::runtime_error("Missing native collision attachment");
                blocks.push_back({*collision.blockIndex, collision.physics});
            }
            return assets::exportNativeCollisionAsset(worldEsm, formId, mImpl->bytes, blocks,
                mImpl->sourceEsm, mImpl->dataDirectory);
        }
        if (mImpl->state.geometry->surfaces.empty())
            throw std::runtime_error("Asset has no surfaces; add a cube (Mesh > Add cube) before saving");
        const auto& collision = mImpl->state.collision.front().physics;
        const bool anyShape = std::ranges::any_of(collision.bodies, hasShape);
        if (movable() && !anyShape) throw std::runtime_error("Generate collision before saving a movable asset");
        return assets::exportNativeModelAsset(worldEsm, formId, geometryForExport(), anyShape ? &collision : nullptr);
    }
}
