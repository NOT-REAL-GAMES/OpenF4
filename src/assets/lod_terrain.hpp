#pragma once

#include "assets/esm_world.hpp"

#include <compare>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace openf4::assets
{
    struct LodTerrainTile
    {
        int level = 0;
        CellCoord origin{};

        auto operator<=>(const LodTerrainTile&) const = default;
    };

    // The import baker and the editor baker write the same NIF terrain grid.
    void writeTerrainBtr(const std::filesystem::path& path, float spacing,
        const std::function<std::optional<float>(int, int)>& height,
        const std::function<bool(int, int)>& covered);

    class TerrainLodSource
    {
    public:
        explicit TerrainLodSource(const std::filesystem::path& worldEsm);
        ~TerrainLodSource();
        TerrainLodSource(TerrainLodSource&&) noexcept;
        TerrainLodSource& operator=(TerrainLodSource&&) noexcept;
        TerrainLodSource(const TerrainLodSource&) = delete;
        TerrainLodSource& operator=(const TerrainLodSource&) = delete;

        [[nodiscard]] std::vector<LodTerrainTile> tiles() const;
        [[nodiscard]] std::vector<std::filesystem::path> sourceFiles() const;
        [[nodiscard]] std::optional<std::pair<CellCoord, CellCoord>> shardCellBounds(
            const std::filesystem::path& source) const;
        void bake(const LodTerrainTile& tile, const std::filesystem::path& outputPath) const;

    private:
        struct Impl;
        std::unique_ptr<Impl> mImpl;
    };
}
