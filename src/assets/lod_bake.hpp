#pragma once

#include "assets/esm_world.hpp"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <span>
#include <string>
#include <vector>

namespace openf4::assets
{
    class TerrainLodSource;
    class StaticLodSource;
    struct LodTerrainTile;
    struct LodStaticTile;

    struct LodBakeProgress
    {
        std::uint64_t completed = 0;
        std::uint64_t total = 0;
        std::filesystem::path output;
    };

    struct LodBakeOptions
    {
        bool full = false;
        bool terrain = true;
        bool statics = true;
        std::function<bool()> cancelled;
        std::function<void(const LodBakeProgress&)> progress;
    };

    struct LodBakePreflight
    {
        std::uint64_t terrainTiles = 0;
        std::uint64_t staticTiles = 0;
        std::uint64_t estimatedBytes = 0;
        bool full = false;

        [[nodiscard]] std::uint64_t totalTiles() const noexcept
        {
            return terrainTiles + staticTiles;
        }
    };

    struct LodBakeResult
    {
        std::uint64_t terrainBaked = 0;
        std::uint64_t staticBaked = 0;
        std::uint64_t removedTiles = 0;
        std::uint64_t revision = 0;
        bool cancelled = false;
        // Paths changed by the bake, including tiles retired because their
        // last source CELL or REFR disappeared.
        std::vector<std::filesystem::path> publishedPaths;
        std::vector<std::string> diagnostics;
    };

    // The SQLite file is derived build state. Source ESMs and imported assets
    // remain authoritative; a missing index means the first bake is full.
    class LodBuildIndex
    {
    public:
        explicit LodBuildIndex(std::filesystem::path worldEsm);

        [[nodiscard]] static std::filesystem::path indexPath(
            const std::filesystem::path& worldEsm);
        [[nodiscard]] static std::filesystem::path terrainTilePath(
            const std::filesystem::path& worldEsm, const LodTerrainTile& tile);
        [[nodiscard]] static std::filesystem::path staticTilePath(
            const std::filesystem::path& worldEsm, const LodStaticTile& tile);

        [[nodiscard]] LodBakePreflight preflight(const TerrainLodSource& terrain,
            const StaticLodSource& statics, const LodBakeOptions& options = {}) const;
        [[nodiscard]] LodBakeResult bake(const TerrainLodSource& terrain,
            const StaticLodSource& statics, const LodBakeOptions& options = {}) const;

        // Call only after an authoring save succeeded. Include old and new
        // reference cells, and every cell using a changed model, in staticCells.
        // No-op before the first bake.
        void markSavedChanges(std::span<const CellCoord> terrainCells,
            std::span<const CellCoord> staticCells,
            std::span<const std::filesystem::path> sourceFiles = {}) const;
        [[nodiscard]] std::uint64_t revision() const;

    private:
        std::filesystem::path mWorldEsm;
    };
}
