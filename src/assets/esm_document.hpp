#pragma once

#include "assets/esm_world.hpp"
#include "assets/papyrus_binding.hpp"

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
    enum class ESMModelKind { staticObject, movableStatic };

    struct ESMModel
    {
        std::uint32_t formId = 0;
        std::string editorId;
        std::string path;
        std::string sourcePath;
        std::string sourceEsm;
        ESMModelKind kind = ESMModelKind::staticObject;
        // Original record provenance survives a project override or Make Unique.
        std::string recordSourceEsm;
        std::uint32_t recordSourceFormId = 0;
        PapyrusBindings scripts;
        // Record overrides need not change the sidecar owning unchanged geometry.
        std::string recordOwnerEsm;

        bool operator==(const ESMModel&) const = default;
    };

    struct ESMRecordSource
    {
        std::string esm;
        std::uint32_t formId = 0;
        bool operator==(const ESMRecordSource&) const = default;
    };

    // Read-only index of an NPC_ record: enough to list, place and label actors. The record
    // itself is never rewritten by OpenF4 yet (no NPC_ authoring in this slice).
    struct ESMActorBase
    {
        std::uint32_t formId = 0;
        std::string editorId;
        std::string name;          // FULL; empty for localized files (string-table IDs)
        std::uint32_t race = 0;    // RNAM
        std::string sourceEsm;     // last file in the stack that contains the record
        bool operator==(const ESMActorBase&) const = default;
    };

    // REFR and ACHR share this representation; placement.actor selects the record type.
    struct ESMReference
    {
        PlacedReference placement;
        std::string editorId;

        bool deleted = false;
        ESMRecordSource recordSource;
        std::uint32_t ownerCellFormId = 0;
        std::uint32_t childGroupType = 9;
        PapyrusBindings scripts;
        // REFR record flag 0x800 "Initially Disabled". Plain Delete in a plugin sets this on a
        // (same-FormID) override instead of erasing anything; the record itself stays.
        bool disabled = false;
        bool operator==(const ESMReference&) const = default;
    };

    struct ESMTerrainCell
    {
        LandscapeCell landscape;
        std::uint32_t cellFormId = 0;
        std::uint32_t landFormId = 0;
        ESMRecordSource cellSource;
        ESMRecordSource landSource;
    };

    // A raw-preserving index over one OpenF4-owned ESM.  It understands enough of
    // the TES4 hierarchy to patch STAT, MSTT, REFR, ACHR and LAND records (and to index NPC_
    // bases read-only), and preserves every
    // record/group it does not own byte-for-byte.
    class EsmDocument
    {
    public:
        explicit EsmDocument(const std::filesystem::path& path);
        // Indexes one specific worldspace (FormID in this file's own space). The WRLD record
        // itself may live only in a master; save() then writes a parent WRLD override taken
        // from `worldSource` the first time a CELL/REFR override needs it.
        EsmDocument(const std::filesystem::path& path, std::optional<std::uint32_t> worldFormId);
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
        [[nodiscard]] bool isLocalized() const noexcept;
        [[nodiscard]] bool containsRecord(std::uint32_t formId) const noexcept;
        [[nodiscard]] bool hasWorldRecord() const noexcept;
        void setWorldSource(ESMRecordSource source);
        [[nodiscard]] std::optional<ESMTerrainCell> terrainCell(CellCoord coord) const;
        [[nodiscard]] const std::vector<ESMModel>& models() const noexcept;
        [[nodiscard]] const std::vector<ESMReference>& references() const noexcept;
        [[nodiscard]] const std::vector<ESMActorBase>& actorBases() const noexcept;
        [[nodiscard]] const std::map<CellCoord, ESMTerrainCell>& terrainCells() const noexcept;

        [[nodiscard]] std::uint32_t allocateFormId();
        void setNextFormId(std::uint32_t nextFormId);
        void setModels(std::vector<ESMModel> models);
        void setReferences(std::vector<ESMReference> references);
        void replaceTerrainCells(std::span<const ESMTerrainCell> cells);
        void setPath(std::filesystem::path path);
        void save();

    private:
        friend std::filesystem::path createEditorOverlay(const std::filesystem::path& master);
        struct Impl;
        std::unique_ptr<Impl> mImpl;
    };

    struct ESMMasterStack
    {
        // Active file's MAST list followed by its own filename: index == FormID high byte.
        std::vector<std::string> canonicalNames;
        std::filesystem::path basePath;
        std::vector<std::filesystem::path> sourcePaths;
        std::vector<ESMModel> models;
        std::vector<ESMReference> references;
        // NPC_ bases from every file in the stack, FormIDs in the active file's space.
        std::vector<ESMActorBase> actorBases;
        std::uint32_t worldFormId = 0;
        // For every REFR the selected (last) file overrides or deletes: the version that wins
        // in the remaining load order without it. Revert to load order restores this.
        std::vector<ESMReference> shadowedReferences;
    };
    [[nodiscard]] ESMMasterStack loadEsmMasterStack(const std::filesystem::path& path,
        std::optional<std::filesystem::path> dataDirectory = std::nullopt);
    struct ESMStackOptions
    {
        std::optional<std::filesystem::path> dataDirectory;
        // Worldspace in the selected file's FormID space; nullopt keeps the automatic choice.
        std::optional<std::uint32_t> worldFormId;
        // Files resolved by the Load Order panel; matched by filename before folder lookup.
        std::vector<std::filesystem::path> resolvedFiles;
        // Plain (non-OpenF4) plugins edited in place: their base is the master owning the world.
        bool directPlugin = false;
    };
    [[nodiscard]] ESMMasterStack loadEsmMasterStack(const std::filesystem::path& path, const ESMStackOptions& options);
    [[nodiscard]] std::filesystem::path createEditorOverlay(const std::filesystem::path& master);
}
