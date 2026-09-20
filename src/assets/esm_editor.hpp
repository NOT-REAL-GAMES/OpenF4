#pragma once

#include "assets/esm_world.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>

namespace empire::assets
{
    struct EditableLandscape
    {
        static constexpr std::size_t side = 97;
        static constexpr std::size_t cellCount = 9;
        static constexpr std::uint32_t worldFormId = 0x01000800u;
        static constexpr std::uint32_t firstCellFormId = 0x01000801u;
        static constexpr std::uint32_t firstLandFormId = 0x0100080Au;

        struct SampleCoord
        {
            std::int32_t x = 0;
            std::int32_t y = 0;

            auto operator<=>(const SampleCoord&) const = default;
        };

        struct CellForms
        {
            std::uint32_t cell = 0;
            std::uint32_t land = 0;

            bool operator==(const CellForms&) const = default;
        };

        std::array<float, side * side> heights{};
        std::map<CellCoord, CellForms> cellForms;
        std::map<SampleCoord, float> additionalHeights;
        std::uint32_t nextFormId = 0x01000813u;

        [[nodiscard]] float height(std::size_t x, std::size_t y) const;
        float& height(std::size_t x, std::size_t y);
        [[nodiscard]] bool containsCell(CellCoord coord) const noexcept;
        [[nodiscard]] bool hasSample(SampleCoord coord) const noexcept;
        [[nodiscard]] float sample(SampleCoord coord) const;
        void setSample(SampleCoord coord, float value);
        [[nodiscard]] bool addCell(CellCoord coord);
        [[nodiscard]] LandscapeCell cell(CellCoord coord) const;

        bool operator==(const EditableLandscape&) const = default;
    };

    [[nodiscard]] EditableLandscape createEditableLandscape();
    [[nodiscard]] EditableLandscape loadEditableLandscape(const std::filesystem::path& path);
    void saveEditableLandscape(const std::filesystem::path& path, const EditableLandscape& landscape);
}
