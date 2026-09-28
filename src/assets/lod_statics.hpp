#pragma once

#include "assets/esm_world.hpp"
#include "assets/nif_model.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace openf4::assets
{
    // A four-cell exterior tile; origin is the south-west CELL coordinate.
    struct LodStaticTile
    {
        CellCoord origin;
        auto operator<=>(const LodStaticTile&) const = default;
    };

    struct LodStaticPart
    {
        std::uint32_t referenceFormId = 0;
        ModelGeometry geometry;
    };

    // Reads the named REFR parts of an OpenF4-baked standard GLB independently.
    [[nodiscard]] std::vector<LodStaticPart> loadLodStaticParts(
        const std::filesystem::path& tilePath);

    class StaticLodSource
    {
    public:
        explicit StaticLodSource(const std::filesystem::path& worldEsm,
            std::optional<std::filesystem::path> dataDirectory = std::nullopt);
        ~StaticLodSource();
        StaticLodSource(StaticLodSource&&) noexcept;
        StaticLodSource& operator=(StaticLodSource&&) noexcept;
        StaticLodSource(const StaticLodSource&) = delete;
        StaticLodSource& operator=(const StaticLodSource&) = delete;

        [[nodiscard]] std::vector<LodStaticTile> tiles() const;
        [[nodiscard]] std::vector<std::uint32_t> members(LodStaticTile tile) const;
        [[nodiscard]] std::vector<std::filesystem::path> sourceFiles() const;
        [[nodiscard]] std::vector<std::filesystem::path> sourceFilesForTile(
            LodStaticTile tile) const;
        [[nodiscard]] std::vector<std::string> diagnostics() const;
        // Writes a single glTF 2 GLB. The caller publishes the path atomically.
        void bake(LodStaticTile tile, const std::filesystem::path& outputPath) const;

    private:
        struct Impl;
        std::unique_ptr<Impl> mImpl;
    };
}
