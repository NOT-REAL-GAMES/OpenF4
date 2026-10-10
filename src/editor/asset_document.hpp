#pragma once

#include "assets/native_model_export.hpp"

#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace openf4::editor
{
    struct AssetMaterial
    {
        glm::vec4 baseColor{1};
        float alphaCutoff = 0;
        bool alphaBlend = false;
        bool doubleSided = false;
        std::string diffuseTexture;
        std::string normalTexture;
        std::shared_ptr<const assets::ModelImage> diffuseImage;
        std::shared_ptr<const assets::ModelImage> normalImage;
        bool operator==(const AssetMaterial&) const = default;
    };

    struct AssetCapabilities
    {
        bool materials = true;
        bool collision = true;
        bool motion = true;
        std::string reason;
    };

    struct AssetCollisionSystem
    {
        // Imported/new assets have no original NIF block to preserve.
        std::optional<std::uint32_t> blockIndex;
        assets::HavokPhysicsSystem physics;
    };

    enum class AssetCollisionPart { convex, capsule, triangleMesh };

    struct AssetBodyProperties
    {
        float mass = 0;
        float friction = 0.5f;
        float restitution = 0.4f;
        float linearDamping = 0.1f;
        float angularDamping = 0.05f;
        float gravityFactor = 1;
        bool operator==(const AssetBodyProperties&) const = default;
    };

    class AssetDocument
    {
    public:
        AssetDocument(std::filesystem::path sourceEsm, std::string modelPath,
            std::optional<std::filesystem::path> dataDirectory = std::nullopt,
            std::optional<bool> movableRecordHint = std::nullopt);
        explicit AssetDocument(assets::ModelGeometry geometry);
        ~AssetDocument();
        AssetDocument(AssetDocument&&) noexcept;
        AssetDocument& operator=(AssetDocument&&) noexcept;

        [[nodiscard]] const assets::ModelGeometry& sourceGeometry() const noexcept;
        [[nodiscard]] const std::vector<AssetMaterial>& materials() const noexcept;
        [[nodiscard]] const std::vector<AssetCollisionSystem>& collisionSystems() const noexcept;
        [[nodiscard]] const AssetCapabilities& capabilities() const noexcept;
        [[nodiscard]] const std::vector<std::string>& diagnostics() const noexcept;
        [[nodiscard]] const std::filesystem::path& sourceEsm() const noexcept;
        [[nodiscard]] const std::string& modelPath() const noexcept;
        [[nodiscard]] std::span<const std::byte> sourceBytes() const noexcept;
        [[nodiscard]] bool movable() const noexcept;
        [[nodiscard]] bool dirty() const noexcept;
        [[nodiscard]] std::uint64_t revision() const noexcept;

        void beginTransaction();
        void endTransaction();
        void cancelTransaction();
        [[nodiscard]] bool undo();
        [[nodiscard]] bool redo();
        void markSaved();

        void setMaterial(std::size_t surface, AssetMaterial material);
        void setTexture(std::size_t surface, const std::filesystem::path& texture, bool normal);
        [[nodiscard]] AssetBodyProperties bodyProperties(std::size_t system, std::size_t body) const;
        // Only mass/shape edits and actual motion transitions derive uniform-density mass properties.
        void setBodyProperties(std::size_t system, std::size_t body, const AssetBodyProperties& value);
        void setMovable(bool movable, float mass = 10.0f);
        // Generation replaces the selected body's shape; vertex positions use engine units.
        void generateCollision(std::span<const std::size_t> surfaces, bool convex,
            std::size_t system = 0, std::size_t body = 0);
        void addBox(glm::vec3 center, glm::vec3 halfExtent,
            std::size_t system = 0, std::size_t body = 0);
        void addCapsule(glm::vec3 pointA, glm::vec3 pointB, float radius,
            std::size_t system = 0, std::size_t body = 0);
        void setCapsuleGeometry(std::size_t part, glm::vec3 pointA, glm::vec3 pointB, float radius,
            std::size_t system = 0, std::size_t body = 0);
        void erasePart(AssetCollisionPart kind, std::size_t part,
            std::size_t system = 0, std::size_t body = 0);
        // Part matrices are body-local engine-space matrices, including positive scale.
        [[nodiscard]] glm::mat4 partTransform(AssetCollisionPart kind, std::size_t part,
            std::size_t system = 0, std::size_t body = 0) const;
        void setPartTransform(AssetCollisionPart kind, std::size_t part, const glm::mat4& transform,
            std::size_t system = 0, std::size_t body = 0);

        [[nodiscard]] assets::ModelGeometry geometryForExport() const;
        // Does not mark saved: caller first assigns the package to the world transaction.
        [[nodiscard]] assets::NativeModelPackage saveNative(
            const std::filesystem::path& worldEsm, std::uint32_t formId) const;

    private:
        struct Impl;
        std::unique_ptr<Impl> mImpl;
    };
}
