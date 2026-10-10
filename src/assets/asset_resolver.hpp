#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace openf4::assets
{
    // Explicit test instrumentation; only the selected world's sidecar is captured.
    // Configure once, before constructing resolvers or starting worker threads.
    void beginCompatibilityAssetCapture(const std::filesystem::path& worldEsm,
        const std::filesystem::path& outputDirectory);
    void endCompatibilityAssetCapture(); // After scoped resolvers and workers are destroyed.
    [[nodiscard]] std::string compatibilityJsonString(std::string_view value);

    enum class AssetSource
    {
        project,
        looseData,
        archive
    };

    struct AssetBytes
    {
        std::vector<std::byte> bytes;
        std::filesystem::path resolvedPath;
        AssetSource source = AssetSource::project;
    };

    // Resolves paths stored in an OpenF4 ESM.  Project assets take precedence
    // over loose Fallout Data files, which take precedence over BA2 archives.
    class AssetResolver
    {
    public:
        AssetResolver(const std::filesystem::path& worldEsm,
            std::optional<std::filesystem::path> dataDirectory = std::nullopt);
        ~AssetResolver();
        AssetResolver(AssetResolver&&) noexcept;
        AssetResolver& operator=(AssetResolver&&) noexcept;
        AssetResolver(const AssetResolver&) = delete;
        AssetResolver& operator=(const AssetResolver&) = delete;

        [[nodiscard]] static std::filesystem::path projectAssetRoot(
            const std::filesystem::path& worldEsm);
        [[nodiscard]] static std::string normalizePath(std::string_view path);

        [[nodiscard]] bool contains(std::string_view path) const;
        [[nodiscard]] AssetBytes read(std::string_view path) const;
        [[nodiscard]] std::vector<std::byte> extract(std::string_view path) const;
        [[nodiscard]] std::vector<std::byte> extractTexture(std::string_view path) const;
        [[nodiscard]] std::optional<std::filesystem::path> resolveLoose(
            std::string_view path) const;
        [[nodiscard]] std::vector<std::string> files(
            std::string_view prefix, std::string_view suffix) const;

    private:
        struct Impl;
        std::unique_ptr<Impl> mImpl;
    };
}
