#pragma once

#include "assets/esm_document.hpp"

#include <array>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace openf4::assets
{
    // Verified against Fallout4.esm: LCWDefault01Grass02, DefaultClimate,
    // ExtOceanWater and XMarkerHeading, respectively.
    inline constexpr std::uint32_t fo4DefaultLandscapeTexture = 0x000AB07E;
    inline constexpr std::uint32_t fo4DefaultClimate = 0x0000015F;
    inline constexpr std::uint32_t fo4DefaultWater = 0x00000018;
    inline constexpr std::uint32_t fo4ArrivalMarker = 0x00000034;

    struct Fo4WorldCell
    {
        CellCoord coord;
        std::uint32_t cellFormId = 0;
        std::uint32_t landFormId = 0;
        bool hasWater = false;
    };

    struct Fo4WorldModel
    {
        std::uint32_t formId = 0;
        ESMModelKind kind = ESMModelKind::staticObject;
        std::string editorId;
        // Data-relative meshes/...nif; OBND is in Bethesda Z-up units.
        std::string modelPath;
        std::array<std::int16_t, 6> bounds{};
        std::string displayName;
        PapyrusBindings scripts;
    };

    struct Fo4WorldWriteRequest
    {
        std::string editorId;
        std::string displayName;
        std::uint32_t worldFormId = 0;
        std::uint32_t persistentCellFormId = 0;
        std::vector<Fo4WorldCell> cells;
        // Called only for declared cells; at most nine LAND arrays are cached.
        std::function<std::optional<LandscapeCell>(CellCoord)> loadCell;
        std::vector<Fo4WorldModel> models;
        std::vector<ESMReference> references;
        // Persistent XMarkerHeading, with position chosen by the caller.
        ESMReference arrival;
        float defaultLandHeight = -2048.0f;
        float waterHeight = 0.0f;
        std::uint32_t defaultLandscapeTexture = fo4DefaultLandscapeTexture;
        std::uint32_t climate = fo4DefaultClimate;
        std::uint32_t water = fo4DefaultWater;
    };

    // Writes a new, non-master plugin with Fallout4.esm as its only master.
    // Supplied authored IDs use self slot 0x01. Existing files are never replaced.
    // ponytail: small-world geometry gate only; native navigation/LOD are separate
    // build products and must be validated before a complete export is certified.
    void writeFo4WorldPlugin(const std::filesystem::path& path,
        const Fo4WorldWriteRequest& request);
}
