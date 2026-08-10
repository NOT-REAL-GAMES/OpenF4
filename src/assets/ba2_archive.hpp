#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace empire::assets
{
    class Ba2Archive
    {
    public:
        struct Entry
        {
            std::uint64_t offset = 0;
            std::uint32_t packedSize = 0;
            std::uint32_t unpackedSize = 0;
        };

        explicit Ba2Archive(const std::filesystem::path& path);

        [[nodiscard]] bool contains(std::string_view path) const;
        [[nodiscard]] std::vector<std::byte> extract(std::string_view path) const;
        [[nodiscard]] const std::unordered_map<std::string, Entry>& entries() const noexcept;

    private:
        std::filesystem::path mPath;
        std::unordered_map<std::string, Entry> mEntries;
    };

    [[nodiscard]] std::vector<std::byte> decompressBa2Payload(
        std::span<const std::byte> packed, std::size_t unpackedSize);
}
