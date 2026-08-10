#include "assets/dds_texture.hpp"
#include "assets/esm_world.hpp"
#include "assets/material_file.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace
{
    void require(bool condition, const char* message)
    {
        if (!condition)
            throw std::runtime_error(message);
    }

    constexpr std::uint32_t fourCc(char a, char b, char c, char d)
    {
        return static_cast<std::uint32_t>(a)
            | (static_cast<std::uint32_t>(b) << 8u)
            | (static_cast<std::uint32_t>(c) << 16u)
            | (static_cast<std::uint32_t>(d) << 24u);
    }

    template <class T>
    void append(std::vector<std::byte>& bytes, const T& value)
    {
        const auto* begin = reinterpret_cast<const std::byte*>(&value);
        bytes.insert(bytes.end(), begin, begin + sizeof(value));
    }

    void appendString(std::vector<std::byte>& bytes, std::string_view value)
    {
        append(bytes, static_cast<std::uint32_t>(value.size()));
        bytes.insert(bytes.end(), reinterpret_cast<const std::byte*>(value.data()),
            reinterpret_cast<const std::byte*>(value.data() + value.size()));
    }

    void appendZeros(std::vector<std::byte>& bytes, std::size_t size)
    {
        bytes.insert(bytes.end(), size, std::byte{});
    }

    void subrecord(std::vector<std::byte>& destination, std::uint32_t type,
        std::span<const std::byte> data)
    {
        append(destination, type);
        append(destination, static_cast<std::uint16_t>(data.size()));
        destination.insert(destination.end(), data.begin(), data.end());
    }

    std::vector<std::byte> record(std::uint32_t type, std::uint32_t formId,
        const std::vector<std::byte>& data)
    {
        std::vector<std::byte> result;
        append(result, type);
        append(result, static_cast<std::uint32_t>(data.size()));
        append(result, std::uint32_t{});
        append(result, formId);
        appendZeros(result, 8);
        result.insert(result.end(), data.begin(), data.end());
        return result;
    }

    std::vector<std::byte> group(std::uint32_t label, std::uint32_t type,
        const std::vector<std::byte>& children)
    {
        std::vector<std::byte> result;
        append(result, fourCc('G', 'R', 'U', 'P'));
        append(result, static_cast<std::uint32_t>(24 + children.size()));
        append(result, label);
        append(result, type);
        appendZeros(result, 8);
        result.insert(result.end(), children.begin(), children.end());
        return result;
    }

    void testMaterialPalette()
    {
        std::vector<std::byte> bytes;
        append(bytes, fourCc('B', 'G', 'S', 'M'));
        append(bytes, std::uint32_t{2});
        append(bytes, std::uint32_t{}); // Clamp.
        for (int index = 0; index < 5; ++index) append(bytes, 0.0f);
        append(bytes, std::uint8_t{});
        append(bytes, std::uint32_t{});
        append(bytes, std::uint32_t{});
        append(bytes, std::uint8_t{128});
        for (int index = 0; index < 11; ++index)
            append(bytes, static_cast<std::uint8_t>(index == 0 || index == 6));
        append(bytes, 0.0f);
        append(bytes, std::uint8_t{});
        append(bytes, 0.0f);
        append(bytes, std::uint8_t{1});
        appendString(bytes, "textures\\test_d.dds");
        appendString(bytes, "textures\\test_n.dds");
        appendString(bytes, "");
        appendString(bytes, "textures\\test_palette.dds");
        for (int index = 0; index < 5; ++index) appendString(bytes, "");
        appendZeros(bytes, 1 + 14 + 25 + 24);
        appendString(bytes, "");
        appendZeros(bytes, 1);
        append(bytes, std::uint8_t{}); // Emit disabled.
        appendZeros(bytes, 4 + 2 + 1 + 5 + 1 + 2 + 1 + 12 + 3 + 1 + 32);
        append(bytes, 0.75f);

        const auto material = empire::assets::loadMaterial(bytes);
        require(material.diffuseTexture == "textures\\test_d.dds", "BGSM diffuse path failed");
        require(material.normalTexture == "textures\\test_n.dds", "BGSM normal path failed");
        require(material.grayscaleTexture == "textures\\test_palette.dds", "BGSM palette path failed");
        require(material.grayscaleToPalette, "BGSM palette flag failed");
        require(material.twoSided, "BGSM two-sided flag failed");
        require(std::abs(material.alphaCutoff - 128.0f / 255.0f) < 0.001f,
            "BGSM alpha-test threshold failed");
        require(std::abs(material.grayscaleToPaletteScale - 0.75f) < 0.001f,
            "BGSM palette row failed");
    }

    void testDdsMips()
    {
        std::vector<std::byte> bytes(148 + 16);
        const auto put = [&](std::size_t offset, std::uint32_t value) {
            std::memcpy(bytes.data() + offset, &value, sizeof(value));
        };
        put(0, fourCc('D', 'D', 'S', ' '));
        put(4, 124);
        put(12, 4);
        put(16, 4);
        put(28, 2);
        put(84, fourCc('D', 'X', '1', '0'));
        put(128, 71);
        const auto texture = empire::assets::loadDdsTexture(bytes);
        require(texture.width == 4 && texture.height == 4, "DDS dimensions failed");
        require(texture.mipLevels == 2, "DDS mip count failed");
        require(texture.mipOffsets == std::vector<std::uint32_t>({0, 8}), "DDS mip offsets failed");
        require(texture.pixels.size() == 16, "DDS payload size failed");
    }

    void testExteriorWorld()
    {
        constexpr std::uint32_t baseId = 0x100;
        constexpr std::uint32_t cellId = 0x200;
        constexpr std::uint32_t ltexId = 0x300;
        constexpr std::uint32_t overlayLtexId = 0x301;
        constexpr std::uint32_t txstId = 0x400;
        constexpr std::uint32_t baseSwapId = 0x700;
        constexpr std::uint32_t referenceSwapId = 0x701;
        std::vector<std::byte> modelData;
        const std::string_view model = "architecture\\test.nif\0";
        subrecord(modelData, fourCc('M', 'O', 'D', 'L'),
            {reinterpret_cast<const std::byte*>(model.data()), model.size()});
        std::array<std::byte, 4> baseSwapBytes{};
        std::memcpy(baseSwapBytes.data(), &baseSwapId, 4);
        subrecord(modelData, fourCc('M', 'O', 'D', 'S'), baseSwapBytes);

        const auto makeSwap = [](std::string_view editorId, std::string_view original,
                                  std::string_view replacement, float paletteRow) {
            std::vector<std::byte> data;
            subrecord(data, fourCc('E', 'D', 'I', 'D'),
                {reinterpret_cast<const std::byte*>(editorId.data()), editorId.size()});
            subrecord(data, fourCc('B', 'N', 'A', 'M'),
                {reinterpret_cast<const std::byte*>(original.data()), original.size()});
            subrecord(data, fourCc('S', 'N', 'A', 'M'),
                {reinterpret_cast<const std::byte*>(replacement.data()), replacement.size()});
            std::array<std::byte, sizeof(float)> color{};
            std::memcpy(color.data(), &paletteRow, sizeof(paletteRow));
            subrecord(data, fourCc('C', 'N', 'A', 'M'), color);
            return data;
        };
        const auto baseSwapData = makeSwap("BaseBlue", "architecture\\base.bgsm",
            "architecture\\blue.bgsm", 0.25f);
        const auto referenceSwapData = makeSwap("ReferenceRed", "vehicles\\car.bgsm",
            "C:\\Fallout4\\Build\\PC\\Data\\materials\\vehicles\\red.bgsm", 0.75f);

        std::vector<std::byte> txstData;
        const std::string_view diffuse = "textures\\land_d.dds\0";
        subrecord(txstData, fourCc('T', 'X', '0', '0'),
            {reinterpret_cast<const std::byte*>(diffuse.data()), diffuse.size()});
        std::vector<std::byte> ltexData;
        std::array<std::byte, 4> txstBytes{};
        std::memcpy(txstBytes.data(), &txstId, 4);
        subrecord(ltexData, fourCc('T', 'N', 'A', 'M'), txstBytes);

        std::vector<std::byte> cellData;
        std::array<std::int32_t, 2> coordinates{2, -3};
        subrecord(cellData, fourCc('X', 'C', 'L', 'C'),
            {reinterpret_cast<const std::byte*>(coordinates.data()), sizeof(coordinates)});
        std::vector<std::byte> referenceData;
        std::array<std::byte, 4> baseBytes{};
        std::memcpy(baseBytes.data(), &baseId, 4);
        subrecord(referenceData, fourCc('N', 'A', 'M', 'E'), baseBytes);
        std::array<std::byte, 4> referenceSwapBytes{};
        std::memcpy(referenceSwapBytes.data(), &referenceSwapId, 4);
        subrecord(referenceData, fourCc('X', 'M', 'S', 'P'), referenceSwapBytes);
        std::array<float, 6> transform{10, 20, 30, 0, 0, 0};
        subrecord(referenceData, fourCc('D', 'A', 'T', 'A'),
            {reinterpret_cast<const std::byte*>(transform.data()), sizeof(transform)});
        std::vector<std::byte> landData;
        std::vector<std::byte> heights;
        append(heights, 10.0f);
        appendZeros(heights, 1089);
        subrecord(landData, fourCc('V', 'H', 'G', 'T'), heights);
        std::array<std::byte, 8> baseTexture{};
        std::memcpy(baseTexture.data(), &ltexId, 4);
        subrecord(landData, fourCc('B', 'T', 'X', 'T'), baseTexture);
        std::array<std::byte, 8> overlayTexture{};
        std::memcpy(overlayTexture.data(), &overlayLtexId, 4);
        overlayTexture[6] = std::byte{1};
        subrecord(landData, fourCc('A', 'T', 'X', 'T'), overlayTexture);
        std::array<std::byte, 8> opacity{};
        const std::uint16_t opacityPosition = 18;
        const float opacityValue = 0.625f;
        std::memcpy(opacity.data(), &opacityPosition, sizeof(opacityPosition));
        std::memcpy(opacity.data() + 4, &opacityValue, sizeof(opacityValue));
        subrecord(landData, fourCc('V', 'T', 'X', 'T'), opacity);

        std::vector<std::byte> cellChildren;
        const auto reference = record(fourCc('R', 'E', 'F', 'R'), 0x500, referenceData);
        const auto landscape = record(fourCc('L', 'A', 'N', 'D'), 0x600, landData);
        cellChildren.insert(cellChildren.end(), reference.begin(), reference.end());
        cellChildren.insert(cellChildren.end(), landscape.begin(), landscape.end());
        std::vector<std::byte> worldChildren = record(fourCc('C', 'E', 'L', 'L'), cellId, cellData);
        const auto temporary = group(cellId, 9, cellChildren);
        const auto cellGroup = group(cellId, 6, temporary);
        worldChildren.insert(worldChildren.end(), cellGroup.begin(), cellGroup.end());

        std::vector<std::byte> file;
        std::vector<std::byte> swapChildren = record(
            fourCc('M', 'S', 'W', 'P'), baseSwapId, baseSwapData);
        const auto referenceSwap = record(
            fourCc('M', 'S', 'W', 'P'), referenceSwapId, referenceSwapData);
        swapChildren.insert(swapChildren.end(), referenceSwap.begin(), referenceSwap.end());
        for (const auto& top : {
            group(fourCc('S', 'T', 'A', 'T'), 0, record(fourCc('S', 'T', 'A', 'T'), baseId, modelData)),
            group(fourCc('M', 'S', 'W', 'P'), 0, swapChildren),
            group(fourCc('T', 'X', 'S', 'T'), 0, record(fourCc('T', 'X', 'S', 'T'), txstId, txstData)),
            group(fourCc('L', 'T', 'E', 'X'), 0, record(fourCc('L', 'T', 'E', 'X'), ltexId, ltexData)),
            group(fourCc('W', 'R', 'L', 'D'), 0, group(0x3C, 1, worldChildren))})
            file.insert(file.end(), top.begin(), top.end());
        const auto path = std::filesystem::temp_directory_path() / "empire-esm-world-test.esm";
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        output.write(reinterpret_cast<const char*>(file.data()), static_cast<std::streamsize>(file.size()));
        output.close();
        const auto world = empire::assets::loadCommonwealth(path);
        std::filesystem::remove(path);
        require(world.models.at(baseId) == "architecture\\test.nif", "ESM model failed");
        require(world.baseMaterialSwaps.at(baseId) == baseSwapId, "ESM base MODS failed");
        require(world.references.size() == 1, "ESM reference failed");
        require(world.references.front().materialSwap == referenceSwapId, "ESM reference XMSP failed");
        require(world.references.front().cell == empire::assets::CellCoord{2, -3}, "ESM cell failed");
        const auto swapped = world.materialSwaps.at(referenceSwapId).resolve(
            "C:/Fallout4/Data/materials/vehicles/car.bgsm");
        require(swapped.matched && swapped.path == "vehicles\\red.bgsm",
            "ESM MSWP material replacement failed");
        require(swapped.colorRemapIndex && std::abs(*swapped.colorRemapIndex - 0.75f) < 0.001f,
            "ESM MSWP CNAM failed");
        require(world.landscapes.at({2, -3}).baseTextures[0] == ltexId, "ESM LAND texture failed");
        const auto& layers = world.landscapes.at({2, -3}).textureLayers;
        require(layers.size() == 2 && layers[0].base && !layers[1].base,
            "ESM LAND layers failed");
        require(layers[1].texture == overlayLtexId && layers[1].layer == 1,
            "ESM LAND ATXT failed");
        require(layers[1].opacityPoints.size() == 1
                && layers[1].opacityPoints[0].position == opacityPosition
                && std::abs(layers[1].opacityPoints[0].opacity - opacityValue) < 0.001f,
            "ESM LAND VTXT failed");
        require(std::abs(world.landscapes.at({2, -3}).heights[0] - 80.0f) < 0.001f,
            "ESM LAND height failed");
    }
}

int main()
try
{
    testMaterialPalette();
    testDdsMips();
    testExteriorWorld();
    std::cout << "Asset format tests passed\n";
    return 0;
}
catch (const std::exception& error)
{
    std::cerr << error.what() << '\n';
    return 1;
}
