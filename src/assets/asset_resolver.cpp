#include "assets/asset_resolver.hpp"

#include "assets/ba2_archive.hpp"
#include "assets/dx10_archive.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>

namespace openf4::assets
{
    namespace
    {
        std::vector<std::byte> readFile(const std::filesystem::path& path)
        {
            std::ifstream file(path, std::ios::binary | std::ios::ate);
            if (!file)
                throw std::runtime_error("Unable to read asset: " + path.string());
            const auto length = file.tellg();
            if (length < 0)
                throw std::runtime_error("Unable to determine asset size: " + path.string());
            std::vector<std::byte> bytes(static_cast<std::size_t>(length));
            file.seekg(0);
            if (!file.read(reinterpret_cast<char*>(bytes.data()), length))
                throw std::runtime_error("Unable to read complete asset: " + path.string());
            return bytes;
        }

        std::filesystem::path relativePath(std::string_view path)
        {
            std::string value(path);
            if (value.find('\0') != std::string::npos)
                throw std::invalid_argument("Asset path contains a null byte");
            std::ranges::replace(value, '\\', '/');
            const std::filesystem::path result(value);
            if (result.empty() || result.is_absolute() || result.has_root_name())
                throw std::invalid_argument("Asset path must be relative: " + value);
            for (const auto& component : result)
                if (component == "..")
                    throw std::invalid_argument("Asset path must not leave its asset root: " + value);
            return result.lexically_normal();
        }

        std::optional<std::filesystem::path> loosePath(
            const std::filesystem::path& root, const std::filesystem::path& relative)
        {
            if (root.empty())
                return std::nullopt;
            auto path = root / relative;
            if (std::filesystem::is_regular_file(path))
                return path;
#ifndef _WIN32
            // Bethesda paths are case-insensitive, including on non-Windows hosts.
            path = root;
            for (const auto& component : relative)
            {
                if (component == ".")
                    continue;
                const auto expected = AssetResolver::normalizePath(component.string());
                bool found = false;
                std::error_code error;
                for (const auto& entry : std::filesystem::directory_iterator(path, error))
                    if (AssetResolver::normalizePath(entry.path().filename().string()) == expected)
                    {
                        path = entry.path();
                        found = true;
                        break;
                    }
                if (!found)
                    return std::nullopt;
            }
            if (std::filesystem::is_regular_file(path))
                return path;
#endif
            return std::nullopt;
        }

        struct Archives
        {
            std::vector<std::unique_ptr<Ba2Archive>> general;
            std::vector<std::unique_ptr<Dx10Archive>> textures;

            explicit Archives(const std::filesystem::path& data)
            {
                for (const char* name : {"Fallout4 - MeshesExtra.ba2", "Fallout4 - Meshes.ba2",
                         "Fallout4 - Materials.ba2"})
                    if (const auto path = data / name; std::filesystem::is_regular_file(path))
                        general.emplace_back(std::make_unique<Ba2Archive>(path));
                for (const char* name : {"Fallout4 - TexturesPatch.ba2", "Fallout4 - Textures1.ba2",
                         "Fallout4 - Textures2.ba2", "Fallout4 - Textures3.ba2",
                         "Fallout4 - Textures4.ba2", "Fallout4 - Textures5.ba2",
                         "Fallout4 - Textures6.ba2", "Fallout4 - Textures7.ba2",
                         "Fallout4 - Textures8.ba2", "Fallout4 - Textures9.ba2"})
                    if (const auto path = data / name; std::filesystem::is_regular_file(path))
                        textures.emplace_back(std::make_unique<Dx10Archive>(path));
            }
        };

        std::shared_ptr<Archives> sharedArchives(const std::filesystem::path& data)
        {
            static std::mutex mutex;
            static std::map<std::filesystem::path, std::weak_ptr<Archives>> cache;
            const std::lock_guard lock(mutex);
            const auto key = std::filesystem::weakly_canonical(data);
            auto& weak = cache[key];
            auto archives = weak.lock();
            if (!archives)
                weak = archives = std::make_shared<Archives>(key);
            return archives;
        }
    }

    struct AssetResolver::Impl
    {
        std::filesystem::path project;
        std::filesystem::path data;
        std::shared_ptr<Archives> archives;
    };

    AssetResolver::AssetResolver(const std::filesystem::path& worldEsm,
        std::optional<std::filesystem::path> dataDirectory)
        : mImpl(std::make_unique<Impl>())
    {
        mImpl->project = projectAssetRoot(worldEsm);
        if (!dataDirectory)
            return;
        mImpl->data = *dataDirectory;
        mImpl->archives = sharedArchives(mImpl->data);
    }

    AssetResolver::~AssetResolver() = default;
    AssetResolver::AssetResolver(AssetResolver&&) noexcept = default;
    AssetResolver& AssetResolver::operator=(AssetResolver&&) noexcept = default;

    std::filesystem::path AssetResolver::projectAssetRoot(const std::filesystem::path& worldEsm)
    {
        auto result = worldEsm;
        result.replace_extension(".assets");
        return result;
    }

    std::string AssetResolver::normalizePath(std::string_view path)
    {
        std::string result(path);
        std::ranges::replace(result, '/', '\\');
        std::ranges::transform(result, result.begin(), [](unsigned char value) {
            return static_cast<char>(std::tolower(value));
        });
        while (result.starts_with(".\\"))
            result.erase(0, 2);
        return result;
    }

    std::optional<std::filesystem::path> AssetResolver::resolveLoose(std::string_view path) const
    {
        const auto relative = relativePath(path);
        if (auto found = loosePath(mImpl->project, relative))
            return found;
        return loosePath(mImpl->data, relative);
    }

    bool AssetResolver::contains(std::string_view path) const
    {
        if (resolveLoose(path))
            return true;
        if (!mImpl->archives)
            return false;
        const auto normalized = relativePath(path).generic_string();
        return std::ranges::any_of(mImpl->archives->general,
                   [&](const auto& archive) { return archive->contains(normalized); })
            || std::ranges::any_of(mImpl->archives->textures,
                   [&](const auto& archive) { return archive->contains(normalized); });
    }

    AssetBytes AssetResolver::read(std::string_view path) const
    {
        const auto relative = relativePath(path);
        if (const auto found = loosePath(mImpl->project, relative))
            return {readFile(*found), *found, AssetSource::project};
        if (const auto found = loosePath(mImpl->data, relative))
            return {readFile(*found), *found, AssetSource::looseData};
        const auto normalized = relative.generic_string();
        if (mImpl->archives)
        for (const auto& archive : mImpl->archives->general)
            if (archive->contains(normalized))
                return {archive->extract(normalized), {}, AssetSource::archive};
        if (mImpl->archives)
        for (const auto& archive : mImpl->archives->textures)
            if (archive->contains(normalized))
                return {archive->extractDds(normalized), {}, AssetSource::archive};
        throw std::runtime_error("Asset is not present in project, Data, or archives: "
            + std::string(path));
    }

    std::vector<std::byte> AssetResolver::extract(std::string_view path) const
    {
        return read(path).bytes;
    }

    std::vector<std::byte> AssetResolver::extractTexture(std::string_view path) const
    {
        return read(path).bytes;
    }

    std::vector<std::string> AssetResolver::files(std::string_view prefix, std::string_view suffix) const
    {
        const auto normalizedPrefix = normalizePath(prefix);
        const auto normalizedSuffix = normalizePath(suffix);
        std::set<std::string> matches;
        const auto add = [&](std::string_view path) {
            auto value = normalizePath(path);
            if (value.starts_with(normalizedPrefix) && value.ends_with(normalizedSuffix))
                matches.insert(std::move(value));
        };
        if (mImpl->archives)
        for (const auto& archive : mImpl->archives->general)
            for (const auto& [path, entry] : archive->entries())
                add(path);
        for (const auto& root : {mImpl->project, mImpl->data})
        {
            if (root.empty() || !std::filesystem::is_directory(root))
                continue;
            for (const auto& entry : std::filesystem::recursive_directory_iterator(root,
                     std::filesystem::directory_options::skip_permission_denied))
                if (entry.is_regular_file())
                    add(entry.path().lexically_relative(root).generic_string());
        }
        return {matches.begin(), matches.end()};
    }
}
