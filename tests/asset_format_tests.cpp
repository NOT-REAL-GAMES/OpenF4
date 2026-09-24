#include "assets/dds_texture.hpp"
#include "assets/dem_import.hpp"
#include "assets/esm_editor.hpp"
#include "assets/esm_world.hpp"
#include "assets/material_file.hpp"

#include <geotiffio.h>
#include <geovalues.h>
#include <tiffio.h>
#include <xtiffio.h>

#include <algorithm>
#include <array>
#include <chrono>
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
            append(bytes, static_cast<std::uint8_t>(index == 0 || index == 5 || index == 6));
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

        const auto material = openf4::assets::loadMaterial(bytes);
        require(material.diffuseTexture == "textures\\test_d.dds", "BGSM diffuse path failed");
        require(material.normalTexture == "textures\\test_n.dds", "BGSM normal path failed");
        require(material.grayscaleTexture == "textures\\test_palette.dds", "BGSM palette path failed");
        require(material.grayscaleToPalette, "BGSM palette flag failed");
        require(material.twoSided, "BGSM two-sided flag failed");
        require(material.decal, "BGSM decal flag failed");
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
        const auto texture = openf4::assets::loadDdsTexture(bytes);
        require(texture.width == 4 && texture.height == 4, "DDS dimensions failed");
        require(texture.mipLevels == 2, "DDS mip count failed");
        require(texture.mipOffsets == std::vector<std::uint32_t>({0, 8}), "DDS mip offsets failed");
        require(texture.pixels.size() == 16, "DDS payload size failed");
    }

    void testExteriorWorld()
    {
        constexpr std::uint32_t baseId = 0x100;
        constexpr std::uint32_t collectionId = 0x101;
        constexpr std::uint32_t lodProxyId = 0x102;
        constexpr std::uint32_t combinedCollectionId = 0x103;
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
        std::array<char, 4 * 260> lodPaths{};
        const std::string_view lodModel = "architecture\\test_lod.nif";
        std::memcpy(lodPaths.data() + 3 * 260, lodModel.data(), lodModel.size());
        subrecord(modelData, fourCc('M', 'N', 'A', 'M'),
            {reinterpret_cast<const std::byte*>(lodPaths.data()), lodPaths.size()});
        std::array<std::byte, 4> baseSwapBytes{};
        std::memcpy(baseSwapBytes.data(), &baseSwapId, 4);
        subrecord(modelData, fourCc('M', 'O', 'D', 'S'), baseSwapBytes);

        std::vector<std::byte> lodProxyData;
        const std::string_view lodProxyEditorId = "Cambridge49_Bld02LOD\0";
        subrecord(lodProxyData, fourCc('E', 'D', 'I', 'D'),
            {reinterpret_cast<const std::byte*>(lodProxyEditorId.data()), lodProxyEditorId.size()});
        subrecord(lodProxyData, fourCc('M', 'O', 'D', 'L'),
            {reinterpret_cast<const std::byte*>(model.data()), model.size()});

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
        const std::string_view terrainMaterial = "materials\\landscape\\land.bgsm\0";
        subrecord(txstData, fourCc('M', 'N', 'A', 'M'),
            {reinterpret_cast<const std::byte*>(terrainMaterial.data()), terrainMaterial.size()});
        std::vector<std::byte> ltexData;
        std::array<std::byte, 4> txstBytes{};
        std::memcpy(txstBytes.data(), &txstId, 4);
        subrecord(ltexData, fourCc('T', 'N', 'A', 'M'), txstBytes);
        const std::string_view directTerrainMaterial = "materials\\landscape\\direct.bgsm\0";
        subrecord(ltexData, fourCc('B', 'N', 'A', 'M'),
            {reinterpret_cast<const std::byte*>(directTerrainMaterial.data()),
                directTerrainMaterial.size()});
        std::vector<std::byte> worldData;

        std::vector<std::byte> collectionData;
        std::array<std::byte, 4> partBaseBytes{};
        std::memcpy(partBaseBytes.data(), &baseId, 4);
        subrecord(collectionData, fourCc('O', 'N', 'A', 'M'), partBaseBytes);
        std::array<float, 14> partTransform{
            100, 200, 300, 0, 0, 0.5f, 2.0f,
            -100, -200, 0, 0, 0, -0.75f, 1.5f};
        subrecord(collectionData, fourCc('D', 'A', 'T', 'A'),
            {reinterpret_cast<const std::byte*>(partTransform.data()), sizeof(partTransform)});
        auto combinedCollectionData = collectionData;
        const std::string_view combinedModel = "scol\\fallout4.esm\\cm00000103.nif\0";
        subrecord(combinedCollectionData, fourCc('M', 'O', 'D', 'L'),
            {reinterpret_cast<const std::byte*>(combinedModel.data()), combinedModel.size()});

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
        std::array<float, 6> transform{8202, -12268, 30, 0, 0, 0};
        subrecord(referenceData, fourCc('D', 'A', 'T', 'A'),
            {reinterpret_cast<const std::byte*>(transform.data()), sizeof(transform)});
        std::vector<std::byte> collectionReferenceData;
        std::memcpy(baseBytes.data(), &collectionId, 4);
        subrecord(collectionReferenceData, fourCc('N', 'A', 'M', 'E'), baseBytes);
        subrecord(collectionReferenceData, fourCc('D', 'A', 'T', 'A'),
            {reinterpret_cast<const std::byte*>(transform.data()), sizeof(transform)});
        std::vector<std::byte> combinedCollectionReferenceData;
        std::memcpy(baseBytes.data(), &combinedCollectionId, 4);
        subrecord(combinedCollectionReferenceData, fourCc('N', 'A', 'M', 'E'), baseBytes);
        subrecord(combinedCollectionReferenceData, fourCc('D', 'A', 'T', 'A'),
            {reinterpret_cast<const std::byte*>(transform.data()), sizeof(transform)});
        std::vector<std::byte> lodProxyReferenceData;
        std::memcpy(baseBytes.data(), &lodProxyId, 4);
        subrecord(lodProxyReferenceData, fourCc('N', 'A', 'M', 'E'), baseBytes);
        subrecord(lodProxyReferenceData, fourCc('D', 'A', 'T', 'A'),
            {reinterpret_cast<const std::byte*>(transform.data()), sizeof(transform)});
        std::vector<std::byte> landData;
        std::vector<std::byte> heights;
        append(heights, 10.0f);
        appendZeros(heights, 1089);
        subrecord(landData, fourCc('V', 'H', 'G', 'T'), heights);
        std::vector<std::byte> colors(33 * 33 * 3, std::byte{0xFF});
        colors[0] = std::byte{0x12};
        colors[1] = std::byte{0x34};
        colors[2] = std::byte{0x56};
        subrecord(landData, fourCc('V', 'C', 'L', 'R'), colors);
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
        const auto collectionReference = record(
            fourCc('R', 'E', 'F', 'R'), 0x501, collectionReferenceData);
        const auto combinedCollectionReference = record(
            fourCc('R', 'E', 'F', 'R'), 0x503, combinedCollectionReferenceData);
        const auto lodProxyReference = record(
            fourCc('R', 'E', 'F', 'R'), 0x502, lodProxyReferenceData);
        const auto landscape = record(fourCc('L', 'A', 'N', 'D'), 0x600, landData);
        cellChildren.insert(cellChildren.end(), reference.begin(), reference.end());
        cellChildren.insert(
            cellChildren.end(), collectionReference.begin(), collectionReference.end());
        cellChildren.insert(cellChildren.end(), combinedCollectionReference.begin(),
            combinedCollectionReference.end());
        cellChildren.insert(cellChildren.end(), lodProxyReference.begin(), lodProxyReference.end());
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
        std::vector<std::byte> commonwealthChildren = record(
            fourCc('W', 'R', 'L', 'D'), 0x3C, worldData);
        const auto worldGroup = group(0x3C, 1, worldChildren);
        commonwealthChildren.insert(
            commonwealthChildren.end(), worldGroup.begin(), worldGroup.end());
        auto staticChildren = record(fourCc('S', 'T', 'A', 'T'), baseId, modelData);
        const auto lodProxy = record(fourCc('S', 'T', 'A', 'T'), lodProxyId, lodProxyData);
        staticChildren.insert(staticChildren.end(), lodProxy.begin(), lodProxy.end());
        auto collectionChildren = record(
            fourCc('S', 'C', 'O', 'L'), collectionId, collectionData);
        const auto combinedCollection = record(
            fourCc('S', 'C', 'O', 'L'), combinedCollectionId, combinedCollectionData);
        collectionChildren.insert(
            collectionChildren.end(), combinedCollection.begin(), combinedCollection.end());
        for (const auto& top : {
            group(fourCc('S', 'T', 'A', 'T'), 0, staticChildren),
            group(fourCc('S', 'C', 'O', 'L'), 0, collectionChildren),
            group(fourCc('M', 'S', 'W', 'P'), 0, swapChildren),
            group(fourCc('T', 'X', 'S', 'T'), 0, record(fourCc('T', 'X', 'S', 'T'), txstId, txstData)),
            group(fourCc('L', 'T', 'E', 'X'), 0, record(fourCc('L', 'T', 'E', 'X'), ltexId, ltexData)),
            group(fourCc('W', 'R', 'L', 'D'), 0, commonwealthChildren)})
            file.insert(file.end(), top.begin(), top.end());
        const auto path = std::filesystem::temp_directory_path() / "openf4-esm-world-test.esm";
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        output.write(reinterpret_cast<const char*>(file.data()), static_cast<std::streamsize>(file.size()));
        output.close();
        const auto world = openf4::assets::loadCommonwealth(path);
        const auto selectedWorld = openf4::assets::loadExteriorWorld(path);
        std::filesystem::remove(path);
        require(world.worldFormId == 0x3Cu && selectedWorld.worldFormId == 0x3Cu
                && selectedWorld.landscapes.contains({2, -3}),
            "Generic ESM worldspace selection failed");
        require(world.models.at(baseId) == "architecture\\test.nif", "ESM model failed");
        require(world.models.at(combinedCollectionId)
                == "scol\\fallout4.esm\\cm00000103.nif",
            "ESM combined SCOL model failed");
        require(world.models.contains(lodProxyId), "ESM LOD proxy model was not parsed");
        require(world.modelLods.at(baseId)[3] == "architecture\\test_lod.nif",
            "ESM STAT MNAM LOD failed");
        require(world.baseMaterialSwaps.at(baseId) == baseSwapId, "ESM base MODS failed");
        require(world.references.size() == 4, "ESM reference or SCOL expansion failed");
        require(std::ranges::none_of(world.references, [](const auto& reference) {
            return reference.formId == 0x502;
        }), "Explicit EDID LOD proxy reference was not filtered");
        require(world.references.front().materialSwap == referenceSwapId, "ESM reference XMSP failed");
        require(world.references.front().cell == openf4::assets::CellCoord{2, -3}, "ESM cell failed");
        require(world.references.front().replacesDistantLod,
            "ESM MNAM reference was not marked as a distant replacement");
        const auto& collectionPart = world.references[1];
        require(collectionPart.baseObject == baseId && collectionPart.hasLocalTransform,
            "ESM SCOL part was not expanded");
        require(collectionPart.instanceId == (std::uint64_t{1} << 32u | 0x501u)
                && collectionPart.localPosition[0] == 100.0f
                && collectionPart.localRotation[2] == 0.5f
                && collectionPart.localScale == 2.0f,
            "ESM SCOL part transform failed");
        require(collectionPart.cell == openf4::assets::CellCoord{2, -3}
                && collectionPart.replacesDistantLod,
            "ESM SCOL part replacement cell failed");
        const auto& secondCollectionPart = world.references[2];
        require(secondCollectionPart.instanceId == (std::uint64_t{2} << 32u | 0x501u)
                && secondCollectionPart.localPosition[0] == -100.0f
                && secondCollectionPart.localRotation[2] == -0.75f
                && secondCollectionPart.localScale == 1.5f
                && secondCollectionPart.cell == openf4::assets::CellCoord{1, -4},
            "ESM SCOL placement array was not fully expanded");
        const auto& combinedCollectionReferenceResult = world.references[3];
        require(combinedCollectionReferenceResult.baseObject == combinedCollectionId
                && !combinedCollectionReferenceResult.hasLocalTransform
                && combinedCollectionReferenceResult.replacesDistantLod,
            "ESM combined SCOL was expanded instead of using its canonical model");
        require(world.textureSets.at(txstId).material == "materials\\landscape\\land.bgsm",
            "ESM TXST material path failed");
        require(world.landscapeMaterials.at(ltexId) == "materials\\landscape\\direct.bgsm",
            "ESM LTEX material path failed");
        require(world.defaultLandscapeTexture == 0x000AB07E,
            "Commonwealth default LAND texture changed");
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
        const auto& color = world.landscapes.at({2, -3}).colors[0];
        require(world.landscapes.at({2, -3}).hasColors
                && color.r == 0x12 && color.g == 0x34 && color.b == 0x56,
            "ESM LAND vertex color failed");
    }

    void testEditableLandscape()
    {
        auto editable = openf4::assets::createEditableLandscape();
        for (std::size_t y = 0; y < editable.side; ++y)
        {
            for (std::size_t x = 0; x < editable.side; ++x)
                editable.height(x, y) = static_cast<float>(x + y) * 8.0f;
        }

        const auto path = std::filesystem::temp_directory_path() / "openf4-editable-landscape-test.esm";
        openf4::assets::saveEditableLandscape(path, editable);
        const auto loaded = openf4::assets::loadEditableLandscape(path);
        require(loaded.heights == editable.heights, "Editable ESM VHGT round trip failed");

        const auto center = loaded.cell({0, 0});
        const auto west = loaded.cell({-1, 0});
        const auto south = loaded.cell({0, -1});
        const auto southwest = loaded.cell({-1, -1});
        require(west.heights[32] == center.heights[0],
            "Editable ESM east/west seam diverged");
        require(south.heights[32 * 33] == center.heights[0],
            "Editable ESM north/south seam diverged");
        require(southwest.heights[32 * 33 + 32] == center.heights[0],
            "Editable ESM four-cell corner diverged");

        auto expanded = loaded;
        require(expanded.addCell({2, 0}) && expanded.addCell({3, 0}),
            "Editable landscape did not create missing cells");
        require(expanded.cellForms.at({2, 0})
                == openf4::assets::EditableLandscape::CellForms{0x01000813u, 0x01000814u}
                && expanded.cellForms.at({3, 0})
                == openf4::assets::EditableLandscape::CellForms{0x01000815u, 0x01000816u},
            "Expanded editable CELL/LAND FormIDs were not stable");
        const auto firstAdded = expanded.cell({2, 0});
        const auto secondAdded = expanded.cell({3, 0});
        for (std::size_t y = 0; y < 33; ++y)
        {
            require(firstAdded.heights[y * 33] == loaded.height(96, y + 32),
                "New CELL did not preserve its existing shared edge");
            require(firstAdded.heights[y * 33 + 32] == secondAdded.heights[y * 33],
                "Two dynamically created CELLs diverged at their seam");
        }
        openf4::assets::saveEditableLandscape(path, expanded);
        require(openf4::assets::loadEditableLandscape(path) == expanded,
            "Expanded editable ESM did not reload semantically");

        auto unquantized = editable;
        unquantized.height(20, 20) += 1.0f;
        bool rejected = false;
        try
        {
            openf4::assets::saveEditableLandscape(path, unquantized);
        }
        catch (const std::runtime_error&)
        {
            rejected = true;
        }
        require(rejected, "Editable ESM accepted an unquantized height");

        auto excessiveDelta = editable;
        excessiveDelta.height(20, 20) += 128.0f * 8.0f;
        rejected = false;
        try
        {
            openf4::assets::saveEditableLandscape(path, excessiveDelta);
        }
        catch (const std::runtime_error&)
        {
            rejected = true;
        }
        require(rejected, "Editable ESM accepted an out-of-range VHGT delta");

        std::fstream corrupt(path, std::ios::binary | std::ios::in | std::ios::out);
        corrupt.seekp(-1, std::ios::end);
        corrupt.put('\x7f');
        corrupt.close();
        rejected = false;
        try
        {
            (void)openf4::assets::loadEditableLandscape(path);
        }
        catch (const std::runtime_error&)
        {
            rejected = true;
        }
        std::filesystem::remove(path);
        require(rejected, "Editable ESM accepted corrupt VHGT trailing data");
    }

    void writeDemFixture(const std::filesystem::path& path, int west, int north,
        float elevation)
    {
        constexpr std::uint32_t side = 3612;
        TIFF* image = XTIFFOpen(path.string().c_str(), "w");
        require(image != nullptr, "Could not create DEM GeoTIFF fixture");
        const auto close = [&] { XTIFFClose(image); };
        TIFFSetField(image, TIFFTAG_IMAGEWIDTH, side);
        TIFFSetField(image, TIFFTAG_IMAGELENGTH, side);
        TIFFSetField(image, TIFFTAG_SAMPLESPERPIXEL, 1);
        TIFFSetField(image, TIFFTAG_BITSPERSAMPLE, 32);
        TIFFSetField(image, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);
        TIFFSetField(image, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
        TIFFSetField(image, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
        // One row per strip keeps the fixture compact while allowing the importer to skip
        // directly over the six-pixel USGS collar.
        TIFFSetField(image, TIFFTAG_ROWSPERSTRIP, 1);
        TIFFSetField(image, TIFFTAG_COMPRESSION, COMPRESSION_ADOBE_DEFLATE);
        TIFFSetField(image, TIFFTAG_PREDICTOR, PREDICTOR_FLOATINGPOINT);
        constexpr double step = 1.0 / 3600.0;
        const std::array scale{step, step, 0.0};
        const std::array tie{0.0, 0.0, 0.0,
            static_cast<double>(west) - 6.0 * step,
            static_cast<double>(north) + 6.0 * step, 0.0};
        TIFFSetField(image, TIFFTAG_GEOPIXELSCALE,
            static_cast<std::uint16_t>(scale.size()), scale.data());
        TIFFSetField(image, TIFFTAG_GEOTIEPOINTS,
            static_cast<std::uint16_t>(tie.size()), tie.data());

        GTIF* geo = GTIFNew(image);
        if (!geo)
        {
            close();
            throw std::runtime_error("Could not create DEM GeoTIFF keys");
        }
        GTIFKeySet(geo, GTModelTypeGeoKey, TYPE_SHORT, 1, ModelTypeGeographic);
        GTIFKeySet(geo, GTRasterTypeGeoKey, TYPE_SHORT, 1, RasterPixelIsArea);
        GTIFKeySet(geo, GeographicTypeGeoKey, TYPE_SHORT, 1, 4269);
        const bool wroteKeys = GTIFWriteKeys(geo) != 0;
        GTIFFree(geo);
        if (!wroteKeys)
        {
            close();
            throw std::runtime_error("Could not write DEM GeoTIFF keys");
        }
        std::vector<float> row(side, elevation);
        for (std::uint32_t y = 0; y < side; ++y)
        {
            if (TIFFWriteScanline(image, row.data(), y, 0) < 0)
            {
                close();
                throw std::runtime_error("Could not write DEM GeoTIFF fixture pixels");
            }
        }
        close();
    }

    void testDemImportPlan()
    {
        const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
        const auto directory = std::filesystem::temp_directory_path()
            / ("openf4-dem-import-test-" + std::to_string(nonce));
        std::filesystem::create_directories(directory);
        struct Cleanup
        {
            std::filesystem::path path;
            ~Cleanup()
            {
                std::error_code ignored;
                std::filesystem::remove_all(path, ignored);
            }
        } cleanup{directory};
        const std::array paths{
            directory / "n42w074.tif", directory / "n41w075.tif",
            directory / "n42w075.tif", directory / "n41w074.tif"};
        writeDemFixture(paths[0], -74, 42, 10.0f);
        writeDemFixture(paths[1], -75, 41, 20.0f);
        writeDemFixture(paths[2], -75, 42, 30.0f);
        writeDemFixture(paths[3], -74, 41, 40.0f);

        const auto summary = openf4::assets::inspectDemLandscape(
            directory / "NYC.esm", paths);
        require(summary.cellCount == 10'906'969,
            "DEM true-scale projected CELL count changed");
        require(summary.shards.size() == 12,
            "DEM was not divided into the expected spatial shards");
        require(summary.minimumCell.x <= 0 && summary.maximumCell.x >= 0
                && summary.minimumCell.y <= 0 && summary.maximumCell.y >= 0,
            "Times Square is not inside the imported CELL footprint");
        require(std::ranges::all_of(summary.shards, [](const auto& shard) {
                return shard.cellCount > 0
                    && shard.cellCount <= openf4::assets::maxDemCellsPerEsm;
            }), "DEM terrain shard exceeded 1,048,576 CELLs");
        require(summary.widthMeters > 170'000.0 && summary.widthMeters < 172'000.0
                && summary.heightMeters > 223'000.0 && summary.heightMeters < 225'000.0,
            "DEM true-scale projected dimensions changed");
        require(summary.btrDirectory == directory / "NYC.btr",
            "DEM BTR output directory changed");
    }
}

int main()
try
{
    testMaterialPalette();
    testDdsMips();
    testExteriorWorld();
    testEditableLandscape();
    testDemImportPlan();
    std::cout << "Asset format tests passed\n";
    return 0;
}
catch (const std::exception& error)
{
    std::cerr << error.what() << '\n';
    return 1;
}
