#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace empire::assets
{
    struct CellCoord
    {
        std::int32_t x = 0;
        std::int32_t y = 0;

        auto operator<=>(const CellCoord&) const = default;
    };

    struct MaterialSubstitution
    {
        std::string original;
        std::string replacement;
        std::optional<float> colorRemapIndex;
    };

    struct ResolvedMaterialSwap
    {
        std::string path;
        std::optional<float> colorRemapIndex;
        bool matched = false;
    };

    struct MaterialSwapRecord
    {
        std::string editorId;
        std::vector<MaterialSubstitution> substitutions;

        [[nodiscard]] ResolvedMaterialSwap resolve(std::string_view materialPath) const;
    };

    struct PlacedReference
    {
        std::uint32_t formId = 0;
        std::uint32_t baseObject = 0;
        std::uint32_t materialSwap = 0;
        CellCoord cell;
        float position[3]{};
        float rotation[3]{};
        float scale = 1.0f;
    };

    struct LandscapeCell
    {
        static constexpr std::size_t side = 33;

        struct TextureLayer
        {
            struct OpacityPoint
            {
                std::uint16_t position = 0;
                float opacity = 0.0f;
            };

            std::uint32_t texture = 0;
            std::uint16_t layer = 0;
            std::uint8_t quadrant = 0;
            bool base = false;
            std::vector<OpacityPoint> opacityPoints;
        };

        CellCoord cell;
        std::array<float, side * side> heights{};
        std::array<std::uint32_t, 4> baseTextures{};
        std::vector<TextureLayer> textureLayers;
        bool hasHeights = false;
    };

    struct TextureSetRecord
    {
        std::string diffuse;
        std::string normal;
    };

    struct ExteriorWorld
    {
        std::unordered_map<std::uint32_t, std::string> models;
        std::unordered_map<std::uint32_t, std::uint32_t> baseMaterialSwaps;
        std::unordered_map<std::uint32_t, MaterialSwapRecord> materialSwaps;
        std::unordered_map<std::uint32_t, TextureSetRecord> textureSets;
        std::unordered_map<std::uint32_t, std::uint32_t> landscapeTextures;
        std::map<CellCoord, LandscapeCell> landscapes;
        std::vector<PlacedReference> references;

        [[nodiscard]] CellCoord suggestedCenter() const;
    };

    [[nodiscard]] ExteriorWorld loadCommonwealth(const std::filesystem::path& esmPath);
}
