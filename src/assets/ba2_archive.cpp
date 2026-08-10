#include "assets/ba2_archive.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <limits>
#include <stdexcept>

#include <lz4.h>
#include <zlib.h>

namespace empire::assets
{
    namespace
    {
        template <class T>
        T read(std::istream& stream)
        {
            T value{};
            stream.read(reinterpret_cast<char*>(&value), sizeof(value));
            if (!stream)
                throw std::runtime_error("Unexpected end of BA2 archive");
            return value;
        }

        std::string normalized(std::string_view path)
        {
            std::string result(path);
            std::replace(result.begin(), result.end(), '/', '\\');
            std::transform(result.begin(), result.end(), result.begin(), [](unsigned char value) {
                return static_cast<char>(std::tolower(value));
            });
            return result;
        }

        constexpr std::uint32_t fourCc(char a, char b, char c, char d)
        {
            return static_cast<std::uint32_t>(a)
                | (static_cast<std::uint32_t>(b) << 8u)
                | (static_cast<std::uint32_t>(c) << 16u)
                | (static_cast<std::uint32_t>(d) << 24u);
        }
    }

    Ba2Archive::Ba2Archive(const std::filesystem::path& path)
        : mPath(path)
    {
        std::ifstream stream(path, std::ios::binary);
        if (!stream)
            throw std::runtime_error("Unable to open BA2 archive: " + path.string());

        if (read<std::uint32_t>(stream) != fourCc('B', 'T', 'D', 'X'))
            throw std::runtime_error("Invalid BA2 signature: " + path.string());
        [[maybe_unused]] const auto version = read<std::uint32_t>(stream);
        if (read<std::uint32_t>(stream) != fourCc('G', 'N', 'R', 'L'))
            throw std::runtime_error("Only GNRL BA2 archives are supported by this reader");

        const auto fileCount = read<std::uint32_t>(stream);
        const auto nameTableOffset = read<std::uint64_t>(stream);
        std::vector<Entry> records;
        records.reserve(fileCount);
        for (std::uint32_t index = 0; index < fileCount; ++index)
        {
            [[maybe_unused]] const auto nameHash = read<std::uint32_t>(stream);
            [[maybe_unused]] const auto extensionHash = read<std::uint32_t>(stream);
            [[maybe_unused]] const auto directoryHash = read<std::uint32_t>(stream);
            [[maybe_unused]] const auto flags = read<std::uint32_t>(stream);
            Entry entry;
            entry.offset = read<std::uint64_t>(stream);
            entry.packedSize = read<std::uint32_t>(stream);
            entry.unpackedSize = read<std::uint32_t>(stream);
            [[maybe_unused]] const auto sentinel = read<std::uint32_t>(stream);
            records.push_back(entry);
        }

        stream.seekg(static_cast<std::streamoff>(nameTableOffset));
        if (!stream)
            throw std::runtime_error("Invalid BA2 name table offset");
        for (const auto& entry : records)
        {
            const auto length = read<std::uint16_t>(stream);
            std::string name(length, '\0');
            stream.read(name.data(), length);
            if (!stream)
                throw std::runtime_error("Truncated BA2 name table");
            mEntries.emplace(normalized(name), entry);
        }
    }

    bool Ba2Archive::contains(std::string_view path) const
    {
        return mEntries.contains(normalized(path));
    }

    std::vector<std::byte> Ba2Archive::extract(std::string_view path) const
    {
        const auto iterator = mEntries.find(normalized(path));
        if (iterator == mEntries.end())
            throw std::runtime_error("File is not present in BA2 archive: " + std::string(path));

        const Entry& entry = iterator->second;
        const std::size_t storedSize = entry.packedSize == 0 ? entry.unpackedSize : entry.packedSize;
        std::vector<std::byte> bytes(storedSize);
        std::ifstream stream(mPath, std::ios::binary);
        stream.seekg(static_cast<std::streamoff>(entry.offset));
        stream.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        if (!stream)
            throw std::runtime_error("Truncated BA2 payload: " + std::string(path));
        if (entry.packedSize == 0)
            return bytes;
        return decompressBa2Payload(bytes, entry.unpackedSize);
    }

    const std::unordered_map<std::string, Ba2Archive::Entry>& Ba2Archive::entries() const noexcept
    {
        return mEntries;
    }

    std::vector<std::byte> decompressBa2Payload(
        std::span<const std::byte> packed, std::size_t unpackedSize)
    {
        if (packed.size() > std::numeric_limits<uInt>::max()
            || unpackedSize > std::numeric_limits<uInt>::max())
            throw std::runtime_error("BA2 payload exceeds decompressor limits");

        std::vector<std::byte> output(unpackedSize);
        uLongf zlibSize = static_cast<uLongf>(unpackedSize);
        const auto zlibResult = uncompress(
            reinterpret_cast<Bytef*>(output.data()), &zlibSize,
            reinterpret_cast<const Bytef*>(packed.data()), static_cast<uLong>(packed.size()));
        if (zlibResult == Z_OK && zlibSize == unpackedSize)
            return output;

        if (packed.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())
            || unpackedSize > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw std::runtime_error("BA2 LZ4 payload exceeds API limits");
        const int lz4Size = LZ4_decompress_safe(
            reinterpret_cast<const char*>(packed.data()), reinterpret_cast<char*>(output.data()),
            static_cast<int>(packed.size()), static_cast<int>(unpackedSize));
        if (lz4Size != static_cast<int>(unpackedSize))
            throw std::runtime_error("BA2 payload is neither valid zlib nor raw LZ4 data");
        return output;
    }
}
