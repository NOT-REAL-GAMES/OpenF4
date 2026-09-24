#pragma once

#include "assets/esm_world.hpp"

#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <span>
#include <optional>
#include <string>
#include <vector>

namespace openf4::assets
{
    // The small set of authored records OpenF4 currently owns.  Everything else in an
    // ESM remains raw data when this document is saved.
    struct ESMModel
    {
        std::uint32_t formId = 0;
        std::string editorId;
        std::string path;
        std::string sourcePath;
        std::string sourceEsm;

        bool operator==(const ESMModel&) const = default;
    };

    struct ESMReference
    {
        PlacedReference placement;
        std::string editorId;

        bool deleted = false;
        bool operator==(const ESMReference&) const = default;
    };

    struct ESMTerrainCell
    {
        LandscapeCell landscape;
        std::uint32_t cellFormId = 0;
        std::uint32_t landFormId = 0;
    };

    // A raw-preserving index over one OpenF4-owned ESM.  It understands enough of
    // the TES4 hierarchy to patch STAT, REFR, and LAND records, and preserves every
    // record/group it does not own byte-for-byte.
    class EsmDocument
    {
    public:
        explicit EsmDocument(const std::filesystem::path& path);
        ~EsmDocument();
        EsmDocument(EsmDocument&&) noexcept;
        EsmDocument& operator=(EsmDocument&&) noexcept;
        EsmDocument(const EsmDocument&) = delete;
        EsmDocument& operator=(const EsmDocument&) = delete;

        [[nodiscard]] const std::filesystem::path& path() const noexcept;
        [[nodiscard]] std::uint32_t worldFormId() const noexcept;
        [[nodiscard]] std::span<const std::string> masters() const noexcept;
        [[nodiscard]] std::uint32_t nextFormId() const noexcept;
        [[nodiscard]] bool isDemOverlay() const noexcept;
        [[nodiscard]] bool isOpenF4Owned() const noexcept;
        [[nodiscard]] std::optional<ESMTerrainCell> terrainCell(CellCoord coord) const;
        [[nodiscard]] const std::vector<ESMModel>& models() const noexcept;
        [[nodiscard]] const std::vector<ESMReference>& references() const noexcept;
        [[nodiscard]] const std::map<CellCoord, ESMTerrainCell>& terrainCells() const noexcept;

        [[nodiscard]] std::uint32_t allocateFormId();
        void setNextFormId(std::uint32_t nextFormId);
        void setModels(std::vector<ESMModel> models);
        void setReferences(std::vector<ESMReference> references);
        void replaceTerrainCells(std::span<const ESMTerrainCell> cells);
        void setPath(std::filesystem::path path);
        void save();

    private:
        struct Impl;
        std::unique_ptr<Impl> mImpl;
    };

    struct ESMMasterStack
    {
        std::filesystem::path basePath;
        std::vector<ESMModel> models;
        std::vector<ESMReference> references;
        std::uint32_t worldFormId = 0;
    };
    [[nodiscard]] ESMMasterStack loadEsmMasterStack(const std::filesystem::path& path,
        std::optional<std::filesystem::path> dataDirectory = std::nullopt);
    [[nodiscard]] std::filesystem::path createEditorOverlay(const std::filesystem::path& master);
}
