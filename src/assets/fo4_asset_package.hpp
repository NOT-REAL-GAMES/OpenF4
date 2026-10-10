#pragma once

#include "assets/asset_resolver.hpp"
#include "assets/esm_document.hpp"
#include "assets/havok_packfile.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace openf4::assets
{
    struct Fo4AssetDependency
    {
        std::string dataPath;
        std::filesystem::path resolvedPath;
        AssetSource source = AssetSource::project;
        std::uint64_t bytes = 0;
    };

    struct Fo4StagedModel
    {
        // Bethesda MODL path: relative to Data/meshes, with backslash separators.
        std::string modelPath;
        // OBND order: min X/Y/Z, max X/Y/Z. Z-up NIF units, no renderer X reflection.
        std::array<std::int16_t, 6> bounds{};
        // Full-precision union of render and collision bounds for arrival clearance.
        glm::vec3 bethesdaMinimum{}, bethesdaMaximum{};
        std::vector<HavokPhysicsSystem> collision;
        std::vector<Fo4AssetDependency> dependencies;
        std::vector<std::string> copiedFiles;
        std::vector<std::string> stockDependencies;
        std::vector<std::string> diagnostics;
    };

    // Read-only complete dependency/collision preflight; model.sourceEsm is its owning master.
    [[nodiscard]] Fo4StagedModel inspectFo4ModelAsset(const ESMModel& model,
        std::optional<std::filesystem::path> dataDirectory = std::nullopt);

    // Caller owns a fresh package staging directory and its atomic publication/rollback.
    // Copies project-owned bytes verbatim; identical existing paths are reused, never overwritten.
    [[nodiscard]] Fo4StagedModel stageFo4ModelAsset(const ESMModel& model,
        const std::filesystem::path& dataStagingDirectory,
        std::optional<std::filesystem::path> dataDirectory = std::nullopt);
}
