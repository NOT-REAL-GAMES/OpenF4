#pragma once

#include "assets/havok_packfile.hpp"
#include "assets/nif_model.hpp"

#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace openf4::assets
{
    struct NativeCollisionBlock
    {
        std::uint32_t blockIndex = 0;
        HavokPhysicsSystem physics;
    };

    struct NativeModelInspection
    {
        bool authoredByOpenF4 = false;
        bool hasAnimation = false;
        std::vector<NativeCollisionBlock> collision;
        std::vector<std::string> dependencies;
        std::string collisionReadOnlyReason;
    };

    struct NativeModelPackage
    {
        std::filesystem::path assetRoot;
        std::string modelPath;
        std::vector<std::string> diagnostics;
    };

    // Publishes a new immutable version; never rewrites the source model or ESM.
    // Geometry is engine Y-up in Bethesda units; collision is Havok Z-up in meters.
    [[nodiscard]] NativeModelPackage exportNativeModelAsset(
        const std::filesystem::path& worldEsm, std::uint32_t formId,
        const ModelGeometry& geometry, const HavokPhysicsSystem* collision = nullptr);

    // Replaces only an existing bhkPhysicsSystem payload and its block-size entry.
    // Untouched bytes, including unknown blocks and footer data, remain exact.
    // Body identity/count and node attachments must remain unchanged.
    [[nodiscard]] std::vector<std::byte> replaceNativeNifCollision(
        std::span<const std::byte> original, std::uint32_t physicsBlockIndex,
        const HavokPhysicsSystem& collision);

    [[nodiscard]] NativeModelInspection inspectNativeModel(std::span<const std::byte> bytes);
    [[nodiscard]] bool nativeMaterialIsEditable(std::span<const std::byte> bytes);
    // Publishes raw collision-patched NIF plus project dependencies; never replaces an existing path.
    [[nodiscard]] NativeModelPackage exportNativeCollisionAsset(
        const std::filesystem::path& worldEsm, std::uint32_t formId,
        std::span<const std::byte> original, std::span<const NativeCollisionBlock> collision,
        const std::filesystem::path& sourceEsm,
        std::optional<std::filesystem::path> dataDirectory = std::nullopt);
}
