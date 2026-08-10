#include "assets/ba2_archive.hpp"
#include "assets/dx10_archive.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

#include <lz4.h>
#include <zlib.h>

namespace
{
    void require(bool condition, std::string_view message)
    {
        if (!condition)
            throw std::runtime_error(std::string(message));
    }

    template <class T>
    void write(std::ostream& output, T value)
    {
        output.write(reinterpret_cast<const char*>(&value), sizeof(value));
    }

    constexpr std::uint32_t fourCc(char a, char b, char c, char d)
    {
        return static_cast<std::uint32_t>(a)
            | (static_cast<std::uint32_t>(b) << 8u)
            | (static_cast<std::uint32_t>(c) << 16u)
            | (static_cast<std::uint32_t>(d) << 24u);
    }

    void testZlib()
    {
        constexpr std::string_view input = "zlib BA2 payload remains supported";
        std::vector<std::byte> packed(compressBound(static_cast<uLong>(input.size())));
        uLongf packedSize = packed.size();
        require(compress2(
            reinterpret_cast<Bytef*>(packed.data()), &packedSize,
            reinterpret_cast<const Bytef*>(input.data()), static_cast<uLong>(input.size()),
            Z_BEST_SPEED) == Z_OK, "zlib test compression failed");
        packed.resize(packedSize);
        const auto output = empire::assets::decompressBa2Payload(packed, input.size());
        require(std::memcmp(output.data(), input.data(), input.size()) == 0, "zlib round trip failed");
    }

    void testLz4()
    {
        constexpr std::string_view input = "raw LZ4 BA2 payload is decoded without zlib";
        std::vector<std::byte> packed(LZ4_compressBound(static_cast<int>(input.size())));
        const int packedSize = LZ4_compress_default(
            input.data(), reinterpret_cast<char*>(packed.data()),
            static_cast<int>(input.size()), static_cast<int>(packed.size()));
        require(packedSize > 0, "LZ4 test compression failed");
        packed.resize(static_cast<std::size_t>(packedSize));
        const auto output = empire::assets::decompressBa2Payload(packed, input.size());
        require(std::memcmp(output.data(), input.data(), input.size()) == 0, "LZ4 round trip failed");
    }

    void testOffset64()
    {
        const auto path = std::filesystem::temp_directory_path() / "empire-ba2-offset-test.ba2";
        constexpr std::uint64_t expectedOffset = 0x00000001'00000020ull;
        constexpr std::string_view name = "meshes\\test.nif";
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        write(output, fourCc('B', 'T', 'D', 'X'));
        write<std::uint32_t>(output, 1);
        write(output, fourCc('G', 'N', 'R', 'L'));
        write<std::uint32_t>(output, 1);
        write<std::uint64_t>(output, 60);
        write<std::uint32_t>(output, 0);
        write<std::uint32_t>(output, 0);
        write<std::uint32_t>(output, 0);
        write<std::uint32_t>(output, 0);
        write(output, expectedOffset);
        write<std::uint32_t>(output, 0);
        write<std::uint32_t>(output, 4);
        write<std::uint32_t>(output, 0xBAADF00D);
        write<std::uint16_t>(output, static_cast<std::uint16_t>(name.size()));
        output.write(name.data(), static_cast<std::streamsize>(name.size()));
        output.close();

        const empire::assets::Ba2Archive archive(path);
        require(archive.contains("Meshes/Test.nif"), "BA2 path normalization failed");
        require(archive.entries().at("meshes\\test.nif").offset == expectedOffset,
            "BA2 64-bit offset was truncated");
        std::filesystem::remove(path);
    }

    void testDx10Lz4()
    {
        constexpr std::string_view input = "DX10 LZ4 texture chunk";
        std::vector<char> packed(LZ4_compressBound(static_cast<int>(input.size())));
        const int packedSize = LZ4_compress_default(input.data(), packed.data(),
            static_cast<int>(input.size()), static_cast<int>(packed.size()));
        require(packedSize > 0, "DX10 LZ4 test compression failed");
        packed.resize(static_cast<std::size_t>(packedSize));

        const auto path = std::filesystem::temp_directory_path() / "empire-dx10-lz4-test.ba2";
        constexpr std::string_view name = "textures\\test.dds";
        constexpr std::uint64_t payloadOffset = 84;
        const std::uint64_t nameTableOffset = payloadOffset + packed.size();
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        write(output, fourCc('B', 'T', 'D', 'X'));
        write<std::uint32_t>(output, 3);
        write(output, fourCc('D', 'X', '1', '0'));
        write<std::uint32_t>(output, 1);
        write(output, nameTableOffset);
        write<std::uint64_t>(output, 0);
        write<std::uint32_t>(output, 3);
        write<std::uint32_t>(output, 0);
        write<std::uint32_t>(output, 0);
        write<std::uint32_t>(output, 0);
        write<std::uint8_t>(output, 0);
        write<std::uint8_t>(output, 1);
        write<std::uint16_t>(output, 24);
        write<std::uint16_t>(output, 4);
        write<std::uint16_t>(output, 4);
        write<std::uint8_t>(output, 1);
        write<std::uint8_t>(output, 28);
        write<std::uint16_t>(output, 0);
        write(output, payloadOffset);
        write<std::uint32_t>(output, static_cast<std::uint32_t>(packed.size()));
        write<std::uint32_t>(output, static_cast<std::uint32_t>(input.size()));
        write<std::uint16_t>(output, 0);
        write<std::uint16_t>(output, 0);
        write<std::uint32_t>(output, 0xBAADF00D);
        output.write(packed.data(), static_cast<std::streamsize>(packed.size()));
        write<std::uint16_t>(output, static_cast<std::uint16_t>(name.size()));
        output.write(name.data(), static_cast<std::streamsize>(name.size()));
        output.close();

        const empire::assets::Dx10Archive archive(path);
        require(archive.contains("Textures/Test.dds"), "DX10 BA2 path normalization failed");
        const auto dds = archive.extractDds(name);
        require(dds.size() == 148 + input.size(), "DX10 DDS reconstruction size is wrong");
        require(std::memcmp(dds.data() + 148, input.data(), input.size()) == 0,
            "DX10 LZ4 payload was not reconstructed");
        std::filesystem::remove(path);
    }
}

int main()
try
{
    testZlib();
    testLz4();
    testOffset64();
    testDx10Lz4();
    std::cout << "BA2 tests passed\n";
    return 0;
}
catch (const std::exception& error)
{
    std::cerr << error.what() << '\n';
    return 1;
}
