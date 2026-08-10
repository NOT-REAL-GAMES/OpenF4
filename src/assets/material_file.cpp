#include "assets/material_file.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string_view>

namespace empire::assets
{
    namespace
    {
        class Reader
        {
        public:
            explicit Reader(std::span<const std::byte> bytes) : mBytes(bytes) {}

            template <class T>
            T read()
            {
                if (mOffset + sizeof(T) > mBytes.size())
                    throw std::runtime_error("Truncated BGSM/BGEM material");
                T value{};
                std::memcpy(&value, mBytes.data() + mOffset, sizeof(T));
                mOffset += sizeof(T);
                return value;
            }

            void skip(std::size_t size)
            {
                if (mOffset + size > mBytes.size())
                    throw std::runtime_error("Truncated BGSM/BGEM material");
                mOffset += size;
            }

            std::string string()
            {
                const std::uint32_t size = read<std::uint32_t>();
                if (size > 4096 || mOffset + size > mBytes.size())
                    throw std::runtime_error("Invalid BGSM/BGEM string");
                std::string value(reinterpret_cast<const char*>(mBytes.data() + mOffset), size);
                mOffset += size;
                if (const auto terminator = value.find('\0'); terminator != std::string::npos)
                    value.resize(terminator);
                return value;
            }

        private:
            std::span<const std::byte> mBytes;
            std::size_t mOffset = 0;
        };

        void readBase(Reader& reader, std::uint32_t version, MaterialInfo& material)
        {
            reader.skip(4 + 8 + 8 + 4); // Clamp, UV offset/scale, transparency.
            const bool alphaBlend = reader.read<std::uint8_t>() != 0;
            reader.skip(4 + 4); // Source and destination blend modes.
            const float alphaThreshold = reader.read<std::uint8_t>() / 255.0f;
            const bool alphaTest = reader.read<std::uint8_t>() != 0;
            material.alphaCutoff = alphaTest ? alphaThreshold : (alphaBlend ? 1.0f / 255.0f : 0.0f);
            reader.skip(4); // Depth and screen-space reflection flags.
            reader.skip(1); // Decal.
            material.twoSided = reader.read<std::uint8_t>() != 0;
            reader.skip(4); // Decal fade, non-occluder, refraction flags.
            reader.skip(4); // Refraction power.
            reader.skip(version < 10 ? 5 : 1);
            material.grayscaleToPalette = reader.read<std::uint8_t>() != 0;
            if (version >= 6)
                reader.skip(1); // Color write mask.
        }
    }

    MaterialInfo loadMaterial(std::span<const std::byte> bytes)
    {
        Reader reader(bytes);
        const std::uint32_t magic = reader.read<std::uint32_t>();
        constexpr std::uint32_t bgsm = 0x4D534742;
        constexpr std::uint32_t bgem = 0x4D454742;
        if (magic != bgsm && magic != bgem)
            throw std::runtime_error("Unsupported material signature");

        const std::uint32_t version = reader.read<std::uint32_t>();
        MaterialInfo result;
        readBase(reader, version, result);
        result.diffuseTexture = reader.string();
        if (magic == bgsm)
        {
            result.normalTexture = reader.string();
            reader.string(); // Smooth/specular map.
            result.grayscaleTexture = reader.string();
            try
            {
                const int additionalTextures = version >= 3 ? (version >= 17 ? 6 : 5) : 5;
                for (int index = 0; index < additionalTextures; ++index)
                    reader.string();
                reader.skip(1); // Editor alpha threshold.
                reader.skip(version >= 8 ? 23 : 14); // Translucency or rim-lighting block.
                reader.skip(25); // Specular block.
                reader.skip(version < 10 ? 24 : 20); // Wetness block.
                if (version >= 3)
                {
                    reader.skip(1); // PBR enabled.
                    if (version >= 9)
                        reader.skip(5); // Custom porosity.
                }
                reader.string(); // Root material.
                reader.skip(1); // Anisotropic lighting.
                if (reader.read<std::uint8_t>() != 0)
                    reader.skip(12); // Emittance color.
                reader.skip(4 + 2); // Emittance multiplier and normal/emittance flags.
                if (version >= 12)
                {
                    reader.skip(4); // Luminance emittance.
                    if (version >= 13)
                        reader.skip(9); // Adaptive emissive flag and exposure parameters.
                }
                else if (version < 8)
                    reader.skip(1); // Back lighting.
                reader.skip(5 + 1); // Shadow/fade flags and glow-map enabled.
                if (version < 7)
                    reader.skip(2);
                reader.skip(1 + 12); // Hair flag and tint.
                reader.skip(3 + 1); // Tree/face/skin flags and tessellation.
                if (version < 3)
                    reader.skip(32);
                const float scale = reader.read<float>();
                result.grayscaleToPaletteScale = scale > 0.01f ? scale : 0.5f;
            }
            catch (const std::runtime_error&)
            {
                // Fallout material versions add optional tail fields. Texture slots above remain valid.
            }
        }
        else
        {
            reader.string(); // Grayscale map.
            reader.string(); // Environment map.
            result.normalTexture = reader.string();
        }
        return result;
    }
}
