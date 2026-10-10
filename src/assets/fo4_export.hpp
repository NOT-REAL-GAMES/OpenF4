#pragma once

#include "assets/esm_world.hpp"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace openf4::assets
{
    struct Fo4ExportProgress
    {
        std::string phase;
        std::uint64_t completed = 0, total = 0;
    };

    struct Fo4ExportOptions
    {
        std::optional<std::filesystem::path> dataDirectory;
        std::optional<CellCoord> arrivalCell;
        std::function<bool()> cancelled;
        std::function<void(const Fo4ExportProgress&)> progress;
    };

    struct Fo4ExportPreflight
    {
        std::string name;
        std::uint64_t cells = 0, references = 0, models = 0, estimatedBytes = 0;
        std::vector<std::string> blockers;
        std::vector<std::string> limitations;
        [[nodiscard]] bool canWriteWorldProof() const noexcept { return blockers.empty(); }
    };

    struct Fo4ExportResult
    {
        std::filesystem::path packageDirectory, pluginPath;
        std::string worldEditorId, arrivalEditorId;
        CellCoord arrivalCell{};
    };

    [[nodiscard]] Fo4ExportPreflight inspectFo4Export(const std::filesystem::path& world,
        const Fo4ExportOptions& options = {});

    // The first compatibility gate, NOT a complete playable-world certification.
    // Never includes implied navigation/LOD support; the report lists those missing gates.
    // Destination must not exist. Failed/cancelled staging is retained for diagnosis.
    [[nodiscard]] Fo4ExportResult exportFo4WorldProof(const std::filesystem::path& world,
        const std::filesystem::path& destination, const Fo4ExportOptions& options = {});
}
