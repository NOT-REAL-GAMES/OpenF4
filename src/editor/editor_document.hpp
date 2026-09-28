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
        bool operator==(const EditorReference&) const = default;
    };

    // Authoring state for one OpenF4 ESM.  Transforms exposed here are engine
    // transforms (Y-up); ESM serialization performs the Bethesda conversion.
    class EditorDocument
    {
    public:
        explicit EditorDocument(std::filesystem::path path,
            std::optional<std::filesystem::path> dataDirectory = std::nullopt);
        ~EditorDocument();
        EditorDocument(EditorDocument&&) noexcept;
        EditorDocument& operator=(EditorDocument&&) noexcept;

        [[nodiscard]] const std::filesystem::path& path() const noexcept;
        [[nodiscard]] const std::filesystem::path& basePath() const noexcept;
        [[nodiscard]] bool hasEditableTerrain() const noexcept;
        [[nodiscard]] std::filesystem::path assetRoot() const;
        [[nodiscard]] std::uint32_t worldFormId() const noexcept;
        [[nodiscard]] const std::vector<EditorReference>& references() const noexcept;
        [[nodiscard]] const std::map<std::uint32_t, std::string>& models() const noexcept;
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
            std::string sourcePath = {});
        [[nodiscard]] std::uint32_t place(std::uint32_t model, const glm::mat4& transform,
            std::string name = {});
        void setTransform(std::uint32_t reference, const glm::mat4& transform);
        void erase(std::uint32_t reference);
        [[nodiscard]] std::uint32_t duplicate(std::uint32_t reference);
        void setVisible(std::uint32_t reference, bool visible);
        void setLocked(std::uint32_t reference, bool locked);

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
