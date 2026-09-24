#include "assets/dds_texture.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace openf4::assets
{
    namespace
    {
        std::uint32_t read32(std::span<const std::byte> bytes, std::size_t offset)
        {
            if (offset + 4 > bytes.size())
                throw std::runtime_error("Truncated DDS header");
            std::uint32_t value = 0;
            std::memcpy(&value, bytes.data() + offset, sizeof(value));
            return value;
        }

        std::uint64_t mipSize(std::uint32_t width, std::uint32_t height, std::uint32_t format)
        {
            switch (format)
            {
                case 70: case 71: case 72:
                    return static_cast<std::uint64_t>((width + 3) / 4) * ((height + 3) / 4) * 8;
                case 73: case 74: case 75: case 76: case 77: case 78:
                case 82: case 83: case 97: case 98: case 99:
                    return static_cast<std::uint64_t>((width + 3) / 4) * ((height + 3) / 4) * 16;
                case 28: case 29: case 87:
                    return static_cast<std::uint64_t>(width) * height * 4;
                default:
                    throw std::runtime_error("Unsupported DDS DXGI format: " + std::to_string(format));
            }
        }
    }

    DdsTexture loadDdsTexture(std::span<const std::byte> bytes)
    {
        constexpr std::uint32_t ddsMagic = 0x20534444;
        constexpr std::uint32_t dx10Magic = 0x30315844;
        constexpr std::size_t payloadOffset = 148;
        if (bytes.size() < payloadOffset || read32(bytes, 0) != ddsMagic
            || read32(bytes, 84) != dx10Magic)
            throw std::runtime_error("DDS texture is not a DX10 DDS image");

        DdsTexture result;
        result.height = read32(bytes, 12);
        result.width = read32(bytes, 16);
        result.mipLevels = std::max(1u, read32(bytes, 28));
        result.dxgiFormat = read32(bytes, 128);
        if (result.width == 0 || result.height == 0)
            throw std::runtime_error("DDS texture has zero extent");

        std::uint64_t offset = 0;
        std::uint32_t width = result.width;
        std::uint32_t height = result.height;
        result.mipOffsets.reserve(result.mipLevels);
        for (std::uint32_t mip = 0; mip < result.mipLevels; ++mip)
        {
            if (offset > std::numeric_limits<std::uint32_t>::max())
                throw std::runtime_error("DDS texture is too large");
            result.mipOffsets.push_back(static_cast<std::uint32_t>(offset));
            offset += mipSize(width, height, result.dxgiFormat);
            width = std::max(1u, width / 2);
            height = std::max(1u, height / 2);
        }
        if (payloadOffset + offset > bytes.size())
            throw std::runtime_error("DDS mip data is truncated");
        result.pixels.assign(bytes.begin() + payloadOffset, bytes.begin() + payloadOffset + offset);
        return result;
    }
}
