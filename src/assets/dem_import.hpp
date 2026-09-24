#pragma once

#include "assets/esm_editor.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace openf4::assets
{
    inline constexpr std::uint64_t maxDemCellsPerEsm = 1'048'576;
    inline constexpr double demBethesdaUnitsPerMeter = 69.99125;
    inline constexpr float demDefaultWaterHeight = 0.0f;

    struct DemImportShard
    {
        std::filesystem::path path;
        CellCoord minimum{};
        CellCoord maximum{};
        std::uint64_t cellCount = 0;
        std::uint64_t estimatedBytes = 0;
    };

    struct DemImportSummary
    {
        std::filesystem::path editorPath;
        std::filesystem::path worldPath;
        CellCoord minimumCell{};
        CellCoord maximumCell{};
        double widthMeters = 0.0;
        double heightMeters = 0.0;
        std::uint64_t cellCount = 0;
        std::uint64_t estimatedBytes = 0;
        std::filesystem::path btrDirectory;
        std::uint64_t btrCount = 0;
        std::vector<DemImportShard> shards;
    };

    struct DemImportOptions
    {
        bool bakeBtr = false;
    };

    [[nodiscard]] DemImportSummary inspectDemLandscape(
        const std::filesystem::path& editorPath,
        const std::array<std::filesystem::path, 4>& sources);
    [[nodiscard]] DemImportSummary importDemLandscape(
        const std::filesystem::path& editorPath,
        const std::array<std::filesystem::path, 4>& sources,
        DemImportOptions options = {});

    struct DemLandscapeCell
    {
        LandscapeCell landscape;
        EditableLandscape::CellForms forms;
    };

    class DemLandscapeProject
    {
    public:
        explicit DemLandscapeProject(const std::filesystem::path& editorPath);
        ~DemLandscapeProject();
        DemLandscapeProject(DemLandscapeProject&&) noexcept;
        DemLandscapeProject& operator=(DemLandscapeProject&&) noexcept;
        DemLandscapeProject(const DemLandscapeProject&) = delete;
        DemLandscapeProject& operator=(const DemLandscapeProject&) = delete;

        [[nodiscard]] static bool recognizes(const std::filesystem::path& editorPath);
        [[nodiscard]] bool contains(CellCoord coord) const;
        [[nodiscard]] std::optional<DemLandscapeCell> cell(CellCoord coord) const;
        [[nodiscard]] std::uint32_t nextFormId() const noexcept;
        [[nodiscard]] std::uint64_t importedCellCount() const noexcept;
        [[nodiscard]] std::size_t overlayCellCount() const noexcept;
        void save(const EditableLandscape& landscape,
            const std::set<CellCoord>& changedCells);

    private:
        struct Impl;
        std::unique_ptr<Impl> mImpl;
    };
}
