#pragma once

#include "assets/esm_document.hpp"
#include "assets/esm_editor.hpp"

#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <glm/mat4x4.hpp>

namespace openf4::editor
{
    struct EditorReference
    {
        assets::PlacedReference placement;
        std::string name;
        glm::vec4 bounds{};
        bool visible = true;
        bool locked = false;
        assets::ESMRecordSource recordSource;
        std::uint32_t ownerCellFormId = 0;
        std::uint32_t childGroupType = 9;
        assets::PapyrusBindings scripts;
        // REFR "Initially Disabled" (0x800): what plain Delete sets in a plugin.
        bool disabled = false;
        bool operator==(const EditorReference&) const = default;
    };

    struct EditorOpenOptions
    {
        std::optional<std::filesystem::path> dataDirectory;
        // Edit exactly this file (Load Order "Make Active File"). Without it, a non-OpenF4
        // file is never touched: an automatic <name>.openf4.esm overlay is created instead.
        bool directEdit = false;
        // Worldspace in this file's FormID space; nullopt keeps the automatic choice.
        std::optional<std::uint32_t> worldFormId;
        // Files resolved by the load order; matched by filename before folder lookup.
        std::vector<std::filesystem::path> resolvedFiles;
        // Initial write policy (see setWritePolicy).
        bool writable = true;
        std::string readOnlyReason;
    };

    // Where a record comes from and what editing it will do. All names are filenames.
    struct RecordProvenance
    {
        std::uint32_t formId = 0;            // in the active file's FormID space
        std::string definingFile;            // owner of the FormID (its MAST slot)
        std::string winningFile;             // last file in the loaded stack containing it
        std::string activeFile;              // current edit target
        bool definedByActive = false;        // a genuinely new record of the active file
        bool activeHasOverride = false;      // active file already stores an override
        bool editCreatesOverride = false;    // saving an edit adds a NEW override record
        bool pendingChange = false;          // unsaved edit relative to the active file
        bool writable = false;
        std::string blockedReason;
        // The active file stores (or will store) its own version of a record another file
        // defines, and dropping it lets the remaining load order decide the record's state.
        bool canRevertToLoadOrder = false;
        std::string revertWinner;            // file whose version wins after the revert
        bool revertPending = false;          // reverted in this session; the override is dropped on save
    };

    // What Delete (hard = false) or Ctrl+Shift+Delete (hard = true) does to one reference.
    enum class ReferenceDeleteAction
    {
        blocked,          // read-only target or locked reference: nothing happens
        disable,          // set "Initially Disabled"; a record another file defines gets a same-FormID override
        alreadyDisabled,  // plain Delete on a reference that is already disabled: nothing to do
        removeRecord,     // the active file's own record is removed from the active file
        deletedOverride,  // a record another file defines gets a Deleted-flag (0x20) override in the active file
    };
    struct ReferenceDeletePlan
    {
        ReferenceDeleteAction action = ReferenceDeleteAction::blocked;
        bool hard = false;
        bool requiresConfirmation = false;  // every hard delete
        bool crashRisk = false;             // Deleted-flag override of a master/other-file record
        bool erasesFromMaster = false;      // removes a record from a master (only under Enable Master Editing)
        std::string summary;
        std::string warning;
        std::string blockedReason;
        RecordProvenance provenance;
    };

    // Authoring state for one OpenF4 ESM.  Transforms exposed here are engine
    // transforms (Y-up); ESM serialization performs the Bethesda conversion.
    class EditorDocument
    {
    public:
        explicit EditorDocument(std::filesystem::path path,
            std::optional<std::filesystem::path> dataDirectory = std::nullopt);
        ~EditorDocument();
        EditorDocument(std::filesystem::path path, EditorOpenOptions options);
        EditorDocument(EditorDocument&&) noexcept;
        EditorDocument& operator=(EditorDocument&&) noexcept;

        [[nodiscard]] const std::filesystem::path& path() const noexcept;
        [[nodiscard]] const std::filesystem::path& basePath() const noexcept;
        [[nodiscard]] bool hasEditableTerrain() const noexcept;
        [[nodiscard]] std::filesystem::path assetRoot() const;
        [[nodiscard]] std::uint32_t worldFormId() const noexcept;
        [[nodiscard]] const std::vector<EditorReference>& references() const noexcept;
        // Hash-indexed lookup by FormID (nullptr when absent). Stays correct across edits.
        [[nodiscard]] const EditorReference* findReference(std::uint32_t id) const noexcept;
        // Changes whenever references/models may have changed (edit, undo, redo, cancel, save).
        // Unique across documents, so callers can cache derived per-frame data by it.
        [[nodiscard]] std::uint64_t revision() const noexcept;
        [[nodiscard]] const std::map<std::uint32_t, std::string>& models() const noexcept;
        [[nodiscard]] const assets::ESMModel* model(std::uint32_t id) const noexcept;
        [[nodiscard]] std::optional<std::string> sourcePath(std::uint32_t model) const;
        [[nodiscard]] std::optional<std::string> sourceEsm(std::uint32_t model) const;
        [[nodiscard]] glm::mat4 engineTransform(std::uint32_t reference) const;
        [[nodiscard]] std::uint32_t nextFormId() const noexcept;
        void setNextFormId(std::uint32_t nextFormId);
        void synchronizeTerrain(const assets::EditableLandscape& terrain, bool recordUndo = true);
        [[nodiscard]] const std::optional<assets::EditableLandscape>& terrainSnapshot() const noexcept;

        [[nodiscard]] std::uint32_t importModel(const std::filesystem::path& source);
        [[nodiscard]] std::uint32_t reserveModelId();
        [[nodiscard]] std::uint32_t commitImportedModel(std::uint32_t formId,
            std::string modelPath, std::string sourcePath, const glm::mat4& transform);
        void beginTransaction();
        void endTransaction();
        // Cancel the whole active transaction, including nested operations and
        // any CELL/FormID allocations, without creating an undo entry.
        void cancelTransaction();
        [[nodiscard]] std::uint32_t addModel(std::string projectRelativePath,
            std::string sourcePath = {}, assets::ESMModelKind kind = assets::ESMModelKind::staticObject);
        // New base for an already-published native package (e.g. a blank asset's first save).
        [[nodiscard]] std::uint32_t addNativeModel(std::string projectRelativePath, assets::ESMModelKind kind);
        // Publication precedes these commands; they never write or remove asset files.
        [[nodiscard]] std::uint32_t assignNativeModel(std::uint32_t model,
            std::string projectRelativePath, assets::ESMModelKind kind);
        [[nodiscard]] std::uint32_t makeUnique(std::uint32_t reference,
            std::string projectRelativePath, assets::ESMModelKind kind);
        void retarget(std::uint32_t reference, std::uint32_t model);
        void setModelScripts(std::uint32_t model, assets::PapyrusBindings scripts);
        // Start Awake default: when enabled, any editor command that creates an MSTT base or
        // turns a base into an MSTT (addModel, assignNativeModel STAT->MSTT, makeUnique as
        // MSTT) attaches assets::wakeOnLoadScriptName to that base in the same undo step, so
        // every placement inherits it. Off by default so library/test callers stay literal.
        void setStartAwakeByDefault(bool enabled) noexcept;
        [[nodiscard]] bool startAwakeByDefault() const noexcept;
        // One-time upgrade for worlds authored before the default: attaches the wake script to
        // MSTT bases this ESM itself defines (its own FormID prefix) that lack it. Never touches
        // master records (Fallout4.esm etc.) or read-only VMAD. One undo step; returns the count.
        std::size_t migrateStartAwake();
        // Incremented whenever the document attaches the wake script automatically.
        [[nodiscard]] std::uint64_t wakeAttachments() const noexcept;
        void setReferenceScripts(std::uint32_t reference, assets::PapyrusBindings scripts);
        [[nodiscard]] std::uint32_t place(std::uint32_t model, const glm::mat4& transform,
            std::string name = {});
        // Actors (prototype). NPC_ bases visible through the loaded stack, FormIDs in the active
        // file's space. Read-only: NPC_ records are never written in this slice.
        [[nodiscard]] const std::map<std::uint32_t, assets::ESMActorBase>& actorBases() const noexcept;
        [[nodiscard]] const assets::ESMActorBase* actorBase(std::uint32_t id) const noexcept;
        // New ACHR in the active file's own FormID slot. The base must be a known NPC_, or a
        // FormID owned by a master this session did not index (e.g. Fallout4.esm behind an
        // OpenF4 world); STAT/MSTT bases and unknown own-slot IDs are refused.
        [[nodiscard]] std::uint32_t placeActor(std::uint32_t npcBase, const glm::mat4& transform,
            std::string name = {});
        // Change an ACHR's NPC_ base (same validation as placeActor). Overrides keep their FormID.
        void setActorBase(std::uint32_t reference, std::uint32_t npcBase);
        void setTransform(std::uint32_t reference, const glm::mat4& transform);
        // Low-level removal without any policy or confirmation (tests, smoke). Editor UI must
        // go through deleteReference so plugins never lose records to a plain Delete.
        void erase(std::uint32_t reference);
        // Delete policy. Plain Delete (hard = false) never erases a record from a plugin or a
        // master: it disables the reference ("Initially Disabled"), creating or updating a
        // same-FormID override when another file defines it. Only an OpenF4-native world's
        // (non-plugin, OpenF4-authored ESM) own records are removed by plain Delete.
        // Ctrl+Shift+Delete (hard = true) removes the active file's own record, or writes a
        // Deleted-flag override for a record another file defines, and always requires
        // confirmation. Read-only targets and locked references are blocked.
        [[nodiscard]] ReferenceDeletePlan planReferenceDelete(std::uint32_t reference, bool hard) const;
        // Applies planReferenceDelete. Throws when blocked, or when hard and !confirmed.
        ReferenceDeleteAction deleteReference(std::uint32_t reference, bool hard, bool confirmed = false);
        // Initially Disabled toggle (plain Delete / Inspector). Undoable; creates an override as needed.
        void setDisabled(std::uint32_t reference, bool disabled);
        // The active file is an ordinary plugin (no ESM flag, not .esm/.esl).
        [[nodiscard]] bool isPluginFile() const noexcept;
        // Revert to load order: drop the active file's version of a record another file defines
        // (move/disable/edit override, or a Deleted-flag override) so the prior load order wins
        // again. Undoable; the override record is removed from the active file on save. Never
        // touches the master. Throws for records the active file itself defines (use Delete),
        // locked references, read-only targets, and records without an active-file version.
        void revertToLoadOrder(std::uint32_t reference);
        // References hidden by a Deleted-flag override in the active file (saved or pending)
        // that revertToLoadOrder can restore: FormID and editor name.
        [[nodiscard]] std::vector<std::pair<std::uint32_t, std::string>> revertibleDeletions() const;
        [[nodiscard]] std::uint32_t duplicate(std::uint32_t reference);
        void setVisible(std::uint32_t reference, bool visible);
        void setLocked(std::uint32_t reference, bool locked);

        // Single choke point for "no accidental writes": while not writable every mutation,
        // undo/redo, save and saveAs throws `reason` and leaves files and state untouched.
        void setWritePolicy(bool writable, std::string reason = {});
        [[nodiscard]] bool writable() const noexcept;
        [[nodiscard]] const std::string& readOnlyReason() const noexcept;
        [[nodiscard]] bool isOpenF4Owned() const noexcept;
        // Selected world in basePath()'s own FormID space (nullopt: automatic choice).
        [[nodiscard]] std::optional<std::uint32_t> baseWorldFormId() const noexcept;
        // Active file's MAST list followed by the active filename (FormID slot order).
        [[nodiscard]] const std::vector<std::string>& loadedFiles() const noexcept;
        [[nodiscard]] RecordProvenance referenceProvenance(std::uint32_t reference) const;
        [[nodiscard]] RecordProvenance modelProvenance(std::uint32_t model) const;

        [[nodiscard]] bool undo();
        [[nodiscard]] bool redo();
        [[nodiscard]] bool dirty() const noexcept;
        void save();
        void saveAs(std::filesystem::path path);

    private:
        struct Impl;
        std::unique_ptr<Impl> mImpl;
    };
}
