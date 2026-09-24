#pragma once

#include "assets/nif_model.hpp"

#include <cstdint>
#include <filesystem>
#include <future>
#include <string>
#include <vector>

namespace openf4::assets
{
    struct ModelImportOptions
    {
        bool generateNormals = true;
        bool generateTangents = true;
    };

    struct ModelImportResult
    {
        ModelGeometry geometry;
        std::filesystem::path sourcePath;
        std::string format;
        std::vector<std::string> warnings;
    };

    struct ImportedModelAsset
    {
        ModelImportResult imported;
        std::filesystem::path assetRoot;
        std::filesystem::path sourcePath;
        // ESM-facing path, relative to <world>.assets.
        std::string modelPath;
    };

    // CPU-only imports are safe to invoke from the stream/import worker pool.
    [[nodiscard]] ModelImportResult importModel(
        const std::filesystem::path& source, ModelImportOptions options = {});
    [[nodiscard]] std::future<ModelImportResult> importModelAsync(
        std::filesystem::path source, ModelImportOptions options = {});

    // Copies a self-contained source/dependency closure, validates the staged
    // asset through Assimp, then publishes it as <world>.assets/models/<id>/.
    [[nodiscard]] ImportedModelAsset importAndCopyModelAsset(
        const std::filesystem::path& source, const std::filesystem::path& worldEsm,
        std::uint32_t formId, ModelImportOptions options = {});
}
