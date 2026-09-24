#include "assets/dx10_archive.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>

#include <lz4.h>
#include <zlib.h>

namespace openf4::assets
{
    namespace
    {
        template <class T>
        T read(std::istream& stream)
        {
            T value{};
            stream.read(reinterpret_cast<char*>(&value), sizeof(value));
            if (!stream)
                throw std::runtime_error("Unexpected end of DX10 BA2 archive");
            return value;
        }

        constexpr std::uint32_t fourCc(char a, char b, char c, char d)
        {
            return static_cast<std::uint32_t>(a)
                | (static_cast<std::uint32_t>(b) << 8u)
                | (static_cast<std::uint32_t>(c) << 16u)
                | (static_cast<std::uint32_t>(d) << 24u);
        }

        std::string normalized(std::string_view path)
        {
            std::string result(path);
            std::replace(result.begin(), result.end(), '/', '\\');
            std::ranges::transform(result, result.begin(), [](unsigned char value) {
                return static_cast<char>(std::tolower(value));
            });
            return result;
        }

#pragma pack(push, 1)
        struct DdsPixelFormat
        {
            std::uint32_t size = 32;
            std::uint32_t flags = 0x4;
            std::uint32_t fourCC = fourCc('D', 'X', '1', '0');
            std::uint32_t rgbBitCount = 0;
            std::uint32_t redMask = 0;
            std::uint32_t greenMask = 0;
            std::uint32_t blueMask = 0;
            std::uint32_t alphaMask = 0;
        };

        struct DdsHeader
        {
            std::uint32_t size = 124;
            std::uint32_t flags = 0x00001007 | 0x00020000 | 0x00080000;
            std::uint32_t height = 0;
            std::uint32_t width = 0;
            std::uint32_t pitchOrLinearSize = 0;
            std::uint32_t depth = 0;
            std::uint32_t mipMapCount = 0;
            std::uint32_t reserved[11]{};
            DdsPixelFormat pixelFormat{};
            std::uint32_t caps = 0x00001000;
            std::uint32_t caps2 = 0;
            std::uint32_t caps3 = 0;
            std::uint32_t caps4 = 0;
            std::uint32_t reserved2 = 0;
            std::uint32_t dxgiFormat = 0;
            std::uint32_t resourceDimension = 3;
            std::uint32_t miscFlags = 0;
            std::uint32_t arraySize = 1;
            std::uint32_t miscFlags2 = 0;
        };
#pragma pack(pop)

        static_assert(sizeof(DdsHeader) == 144);
    }

    Dx10Archive::Dx10Archive(const std::filesystem::path& path)
        : mPath(path)
    {
        std::ifstream stream(path, std::ios::binary);
        if (!stream)
            throw std::runtime_error("Unable to open DX10 BA2 archive: " + path.string());
        if (read<std::uint32_t>(stream) != fourCc('B', 'T', 'D', 'X'))
            throw std::runtime_error("Invalid DX10 BA2 signature");
        const std::uint32_t version = read<std::uint32_t>(stream);
        if (read<std::uint32_t>(stream) != fourCc('D', 'X', '1', '0'))
            throw std::runtime_error("Archive is not a DX10 BA2");
        const std::uint32_t fileCount = read<std::uint32_t>(stream);
        const std::uint64_t nameTableOffset = read<std::uint64_t>(stream);
        if (version == 3)
        {
            [[maybe_unused]] const auto unknown = read<std::uint64_t>(stream);
            mCompression = read<std::uint32_t>(stream) == 3 ? Compression::lz4 : Compression::zlib;
        }
        else if (version != 1 && version != 7 && version != 8)
            throw std::runtime_error("Unsupported DX10 BA2 version: " + std::to_string(version));

        std::vector<Entry> records;
        records.reserve(fileCount);
        for (std::uint32_t index = 0; index < fileCount; ++index)
        {
            [[maybe_unused]] const auto nameHash = read<std::uint32_t>(stream);
            [[maybe_unused]] const auto extensionHash = read<std::uint32_t>(stream);
            [[maybe_unused]] const auto directoryHash = read<std::uint32_t>(stream);
            [[maybe_unused]] const auto unknown = read<std::uint8_t>(stream);
            const std::uint8_t chunkCount = read<std::uint8_t>(stream);
            if (read<std::uint16_t>(stream) != 24)
                throw std::runtime_error("Invalid DX10 BA2 chunk header size");
            Entry entry;
            entry.height = read<std::uint16_t>(stream);
            entry.width = read<std::uint16_t>(stream);
            entry.mipCount = read<std::uint8_t>(stream);
            entry.dxgiFormat = read<std::uint8_t>(stream);
            entry.cubeMap = read<std::uint16_t>(stream);
            entry.chunks.reserve(chunkCount);
            for (std::uint8_t chunkIndex = 0; chunkIndex < chunkCount; ++chunkIndex)
            {
                Chunk chunk;
                chunk.offset = read<std::uint64_t>(stream);
                chunk.packedSize = read<std::uint32_t>(stream);
                chunk.unpackedSize = read<std::uint32_t>(stream);
                chunk.firstMip = read<std::uint16_t>(stream);
                chunk.lastMip = read<std::uint16_t>(stream);
                if (read<std::uint32_t>(stream) != 0xBAADF00D)
                    throw std::runtime_error("Invalid DX10 BA2 chunk sentinel");
                entry.chunks.push_back(chunk);
            }
            records.push_back(std::move(entry));
        }

        stream.seekg(static_cast<std::streamoff>(nameTableOffset));
        if (!stream)
            throw std::runtime_error("Invalid DX10 BA2 name table offset");
        for (auto& entry : records)
        {
            const std::uint16_t size = read<std::uint16_t>(stream);
            std::string name(size, '\0');
            stream.read(name.data(), size);
            if (!stream)
                throw std::runtime_error("Truncated DX10 BA2 name table");
            mEntries.emplace(normalized(name), std::move(entry));
        }
    }

    bool Dx10Archive::contains(std::string_view path) const
    {
        return mEntries.contains(normalized(path));
    }

    std::vector<std::byte> Dx10Archive::extractDds(std::string_view path) const
    {
        const auto iterator = mEntries.find(normalized(path));
        if (iterator == mEntries.end())
            throw std::runtime_error("Texture is not present in DX10 BA2: " + std::string(path));
        const Entry& entry = iterator->second;
        DdsHeader header;
        header.width = entry.width;
        header.height = entry.height;
        header.mipMapCount = entry.mipCount;
        header.pitchOrLinearSize = static_cast<std::uint32_t>(entry.width) * entry.height;
        header.dxgiFormat = entry.dxgiFormat;
        if (entry.mipCount > 1)
            header.caps |= 0x00400008;
        if (entry.cubeMap == 2049)
        {
            header.caps |= 0x8;
            header.caps2 = 0x0000FE00;
            header.miscFlags = 0x4;
        }

        std::size_t payloadSize = 0;
        for (const Chunk& chunk : entry.chunks)
            payloadSize += chunk.unpackedSize;
        std::vector<std::byte> result(sizeof(std::uint32_t) + sizeof(header) + payloadSize);
        const std::uint32_t magic = fourCc('D', 'D', 'S', ' ');
        std::memcpy(result.data(), &magic, sizeof(magic));
        std::memcpy(result.data() + sizeof(magic), &header, sizeof(header));

        std::ifstream stream(mPath, std::ios::binary);
        std::size_t destination = sizeof(magic) + sizeof(header);
        for (const Chunk& chunk : entry.chunks)
        {
            const std::uint32_t storedSize = chunk.packedSize == 0
                ? chunk.unpackedSize : chunk.packedSize;
            std::vector<std::byte> stored(storedSize);
            stream.seekg(static_cast<std::streamoff>(chunk.offset));
            stream.read(reinterpret_cast<char*>(stored.data()), stored.size());
            if (!stream)
                throw std::runtime_error("Truncated DX10 BA2 texture chunk");
            if (chunk.packedSize == 0)
                std::memcpy(result.data() + destination, stored.data(), stored.size());
            else
            {
                std::vector<std::byte> unpacked(chunk.unpackedSize);
                bool decoded = false;
                if (mCompression == Compression::lz4
                    && stored.size() <= static_cast<std::size_t>(std::numeric_limits<int>::max())
                    && unpacked.size() <= static_cast<std::size_t>(std::numeric_limits<int>::max()))
                {
                    decoded = LZ4_decompress_safe(reinterpret_cast<const char*>(stored.data()),
                        reinterpret_cast<char*>(unpacked.data()), static_cast<int>(stored.size()),
                        static_cast<int>(unpacked.size())) == static_cast<int>(unpacked.size());
                }
                else if (mCompression == Compression::zlib)
                {
                    uLongf outputSize = static_cast<uLongf>(unpacked.size());
                    decoded = uncompress(reinterpret_cast<Bytef*>(unpacked.data()), &outputSize,
                        reinterpret_cast<const Bytef*>(stored.data()), static_cast<uLong>(stored.size())) == Z_OK
                        && outputSize == unpacked.size();
                }
                if (!decoded)
                    throw std::runtime_error("DX10 BA2 texture chunk decompression failed");
                std::memcpy(result.data() + destination, unpacked.data(), unpacked.size());
            }
            destination += chunk.unpackedSize;
        }
        return result;
    }
}
