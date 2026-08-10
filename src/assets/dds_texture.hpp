#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace empire::assets
{
    struct DdsTexture
    {
        std::uint32_t width = 0;
        std::uint32_t height = 0;
        std::uint32_t mipLevels = 0;
        std::uint32_t dxgiFormat = 0;
        std::vector<std::byte> pixels;
        std::vector<std::uint32_t> mipOffsets;
    };

    [[nodiscard]] DdsTexture loadDdsTexture(std::span<const std::byte> bytes);
}
