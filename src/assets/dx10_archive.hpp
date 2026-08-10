#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace empire::assets
{
    class Dx10Archive
    {
    public:
        struct Chunk
        {
            std::uint64_t offset = 0;
            std::uint32_t packedSize = 0;
            std::uint32_t unpackedSize = 0;
            std::uint16_t firstMip = 0;
            std::uint16_t lastMip = 0;
        };

        struct Entry
        {
            std::uint16_t width = 0;
            std::uint16_t height = 0;
            std::uint8_t mipCount = 0;
            std::uint8_t dxgiFormat = 0;
            std::uint16_t cubeMap = 0;
            std::vector<Chunk> chunks;
        };

        explicit Dx10Archive(const std::filesystem::path& path);

        [[nodiscard]] bool contains(std::string_view path) const;
        [[nodiscard]] std::vector<std::byte> extractDds(std::string_view path) const;

    private:
        enum class Compression { zlib, lz4 };

        std::filesystem::path mPath;
        Compression mCompression = Compression::zlib;
        std::unordered_map<std::string, Entry> mEntries;
    };
}
