#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace openf4::assets
{
    // TES4 header facts needed before a plugin may join a load order.
    struct PluginHeader
    {
        std::filesystem::path path;
        std::string name;                  // filename as it appears in MAST entries
        std::vector<std::string> masters;  // MAST order == FormID high-byte order
        std::uint32_t flags = 0;
        bool openf4Owned = false;          // CNAM "OpenF4" (project files this editor authored)
        bool localized = false;            // TES4 0x80
        bool light = false;                // TES4 0x200 / .esl
        // Fallout 4 treats the ESM flag and the .esm/.esl extensions as master files;
        // they always load before ordinary plugins.
        [[nodiscard]] bool isMasterFile() const noexcept;
    };
    [[nodiscard]] PluginHeader readPluginHeader(const std::filesystem::path& path);

    struct WorldspaceInfo
    {
        std::uint32_t formId = 0;      // in the requesting file's FormID space (see listWorldspaces)
        std::string editorId;
        std::string definingFile;      // owner of the FormID (master slot), not merely an override
    };
    // WRLD records in one file, FormIDs in that file's own space. Children groups are skipped.
    [[nodiscard]] std::vector<WorldspaceInfo> listWorldspaces(const std::filesystem::path& path);

    [[nodiscard]] std::string foldedName(std::string_view value);

    // Ordered, dependency-validated set of loaded files plus the single active edit
    // target. Nothing here is persisted: Enable Master Editing never survives a session.
    class LoadOrder
    {
    public:
        struct Entry
        {
            PluginHeader header;
            std::vector<std::string> missingMasters;  // MAST names not present in this load order
        };
        struct AddResult
        {
            std::size_t index = 0;
            std::vector<std::string> missingMasters;
            // Missing masters (transitively) found in the Data folder or beside the plugin,
            // already in a valid insertion order. Offer these to the user.
            std::vector<std::filesystem::path> resolvable;
            std::vector<std::string> unresolvable;
        };

        explicit LoadOrder(std::optional<std::filesystem::path> dataDirectory = std::nullopt);

        [[nodiscard]] const std::vector<Entry>& entries() const noexcept { return mEntries; }
        [[nodiscard]] std::size_t size() const noexcept { return mEntries.size(); }
        [[nodiscard]] std::optional<std::size_t> find(std::string_view name) const;

        // Adds a file at the first position that satisfies master-before-plugin and
        // every MAST dependency. Duplicate filenames are refused.
        AddResult add(const std::filesystem::path& path);
        // Finds (transitively) the missing masters of entry `index` on disk.
        [[nodiscard]] AddResult resolveMissingMasters(std::size_t index) const;
        // Adds every resolvable missing master of entry `index`; returns how many were added.
        std::size_t addMissingMasters(std::size_t index);

        // Reorder. Refused (throws, order unchanged) if any file would load before one of
        // its masters, or an ordinary plugin would precede a master file.
        [[nodiscard]] bool canMove(std::size_t from, std::size_t to, std::string* reason = nullptr) const;
        void move(std::size_t from, std::size_t to);
        // Refused while another loaded file lists this one as a master.
        void remove(std::size_t index);
        void clear();
        // Human-readable problems; empty means the order is consistent.
        [[nodiscard]] std::vector<std::string> validate() const;

        // Make Active File. Missing masters or an invalid order refuse. Always clears
        // master editing. A protected master becomes the target in read-only browse mode.
        void makeActive(std::size_t index);
        // Legacy single-file open: adopts an already-opened document as the active target
        // without requiring every master to be listed (missing ones are reported, not hidden).
        void adoptActive(const std::filesystem::path& path);
        [[nodiscard]] std::optional<std::size_t> active() const noexcept { return mActive; }
        [[nodiscard]] const Entry* activeEntry() const noexcept;

        // A protected file is a non-OpenF4 master (Fallout4.esm, DLC, third-party ESMs), any
        // light/ESL file, or any file another loaded plugin depends on.
        [[nodiscard]] bool isProtected(std::size_t index) const;
        [[nodiscard]] std::vector<std::string> dependents(std::size_t index) const;

        // Typed confirmation must equal the active filename exactly. Creates a timestamped
        // byte-for-byte backup beside the file first; returns the backup path.
        std::filesystem::path enableMasterEditing(std::string_view typedConfirmation);
        void disableMasterEditing() noexcept;
        [[nodiscard]] bool masterEditingEnabled() const noexcept { return mMasterEditing; }
        [[nodiscard]] const std::filesystem::path& masterBackup() const noexcept { return mBackup; }

        // Gate for every mutation and save: `documentPath` must be the active file, and the
        // active file must be an ordinary plugin or have master editing explicitly enabled.
        [[nodiscard]] bool canWrite(const std::filesystem::path& documentPath, std::string* reason = nullptr) const;

        // Paths of the active file's MAST list (in MAST order), resolved through this load
        // order, followed by the active file itself.
        [[nodiscard]] std::vector<std::filesystem::path> activeStack() const;
        // Worldspaces visible to the active file, FormIDs in the active file's space.
        [[nodiscard]] std::vector<WorldspaceInfo> activeWorldspaces() const;

        // Loaded files outside the active file's master stack that also contain the record
        // `objectId` owned by `ownerName`. Each result is "name (loads before|after active)".
        // Large files (> 64 MiB) are not indexed and are reported with "(not indexed)".
        [[nodiscard]] std::vector<std::string> otherOverrides(std::string_view ownerName,
            std::uint32_t objectId) const;

    private:
        void refreshMissing();
        [[nodiscard]] std::optional<std::filesystem::path> locate(std::string_view name,
            const std::filesystem::path& besides) const;
        std::optional<std::filesystem::path> mData;
        std::vector<Entry> mEntries;
        std::optional<std::size_t> mActive;
        bool mMasterEditing = false;
        std::filesystem::path mBackup;
        struct Index { bool indexed = false; std::unordered_set<std::uint64_t> records; std::vector<std::string> owners; };
        mutable std::map<std::string, Index> mIndexes;  // folded filename -> record index
    };

    // Creates the timestamped backup used by Enable Master Editing; refuses to overwrite.
    std::filesystem::path createTimestampedBackup(const std::filesystem::path& file);
}
