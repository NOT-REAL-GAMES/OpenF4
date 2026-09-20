#include "assets/dem_import.hpp"

#include <NifFile.hpp>
#include <Shaders.hpp>

#include <geotiffio.h>
#include <geovalues.h>
#include <proj.h>
#include <tiffio.h>
#include <xtiffio.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <map>
#include <memory>
#include <optional>
#include <ranges>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <tuple>
#include <set>
#include <utility>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <Windows.h>
#endif

namespace empire::assets
{
    namespace
    {
        using Bytes = std::vector<std::byte>;

        constexpr std::uint32_t fourCc(const char (&text)[5])
        {
            return static_cast<std::uint32_t>(text[0]) |
                (static_cast<std::uint32_t>(text[1]) << 8u) |
                (static_cast<std::uint32_t>(text[2]) << 16u) |
                (static_cast<std::uint32_t>(text[3]) << 24u);
        }

        constexpr std::uint32_t tes4 = fourCc("TES4");
        constexpr std::uint32_t grup = fourCc("GRUP");
        constexpr std::uint32_t wrld = fourCc("WRLD");
        constexpr std::uint32_t cell = fourCc("CELL");
        constexpr std::uint32_t land = fourCc("LAND");
        constexpr std::uint32_t hedr = fourCc("HEDR");
        constexpr std::uint32_t cnam = fourCc("CNAM");
        constexpr std::uint32_t mast = fourCc("MAST");
        constexpr std::uint32_t data = fourCc("DATA");
        constexpr std::uint32_t edid = fourCc("EDID");
        constexpr std::uint32_t xclc = fourCc("XCLC");
        constexpr std::uint32_t xclw = fourCc("XCLW");
        constexpr std::uint32_t vhgt = fourCc("VHGT");
        constexpr std::uint32_t nam4 = fourCc("NAM4");
        constexpr std::uint32_t emgi = fourCc("EMGI");
        constexpr std::size_t recordHeaderSize = 24;
        constexpr std::uint32_t firstLocalFormId = 0x800u;
        constexpr std::uint32_t worldFormId = 0x01000800u;
        constexpr std::uint32_t importedSelfPrefix = 0x02000000u;
        constexpr std::uint32_t sourceSide = 3612;
        constexpr std::uint32_t sourceCollar = 6;
        constexpr std::uint32_t sourceCoreSide = 3600;
        constexpr double sourceStepDegrees = 1.0 / 3600.0;
        constexpr double cellSize = 4096.0;
        constexpr double sampleSpacing = 128.0;
        constexpr double timesSquareLongitude = -73.9855;
        constexpr double timesSquareLatitude = 40.7580;

        template <typename T>
        void append(Bytes& bytes, const T& value)
        {
            const auto* first = reinterpret_cast<const std::byte*>(&value);
            bytes.insert(bytes.end(), first, first + sizeof(T));
        }

        void append(Bytes& bytes, const void* payload, std::size_t size)
        {
            const auto* first = static_cast<const std::byte*>(payload);
            bytes.insert(bytes.end(), first, first + size);
        }

        void appendSubrecord(Bytes& body, std::uint32_t type, const void* payload,
            std::size_t size)
        {
            if (size > std::numeric_limits<std::uint16_t>::max())
                throw std::runtime_error("DEM ESM subrecord is too large");
            append(body, type);
            append(body, static_cast<std::uint16_t>(size));
            append(body, payload, size);
        }

        template <typename T>
        void appendSubrecord(Bytes& body, std::uint32_t type, const T& payload)
        {
            appendSubrecord(body, type, &payload, sizeof(payload));
        }

        void appendStringSubrecord(Bytes& body, std::uint32_t type, const std::string& value)
        {
            appendSubrecord(body, type, value.c_str(), value.size() + 1);
        }

        [[nodiscard]] Bytes makeRecord(std::uint32_t type, std::uint32_t formId,
            const Bytes& body, std::uint32_t flags = 0)
        {
            Bytes result;
            append(result, type);
            append(result, static_cast<std::uint32_t>(body.size()));
            append(result, flags);
            append(result, formId);
            append(result, std::uint32_t{0});
            append(result, std::uint16_t{131});
            append(result, std::uint16_t{0});
            append(result, body.data(), body.size());
            return result;
        }

        [[nodiscard]] Bytes makeGroup(std::uint32_t label, std::uint32_t type,
            const Bytes& body)
        {
            if (body.size() > std::numeric_limits<std::uint32_t>::max() - recordHeaderSize)
                throw std::runtime_error("DEM ESM group exceeds the 4 GiB format limit");
            Bytes result;
            append(result, grup);
            append(result, static_cast<std::uint32_t>(recordHeaderSize + body.size()));
            append(result, label);
            append(result, type);
            append(result, std::uint32_t{0});
            append(result, std::uint32_t{0});
            append(result, body.data(), body.size());
            return result;
        }

        void appendBytes(Bytes& destination, const Bytes& source)
        {
            destination.insert(destination.end(), source.begin(), source.end());
        }

        [[nodiscard]] std::int32_t floorDivide(std::int32_t value, std::int32_t divisor)
        {
            const std::int32_t quotient = value / divisor;
            return quotient - (value % divisor < 0 ? 1 : 0);
        }

        [[nodiscard]] std::uint32_t packedCellCoord(std::int32_t x, std::int32_t y)
        {
            return static_cast<std::uint16_t>(x) |
                (static_cast<std::uint32_t>(static_cast<std::uint16_t>(y)) << 16u);
        }

        struct Projection
        {
            struct ContextDeleter
            {
                void operator()(PJ_CONTEXT* value) const { proj_context_destroy(value); }
            };
            struct ProjectionDeleter
            {
                void operator()(PJ* value) const { proj_destroy(value); }
            };

            std::unique_ptr<PJ_CONTEXT, ContextDeleter> context{proj_context_create()};
            std::unique_ptr<PJ, ProjectionDeleter> operation;

            Projection()
            {
                if (!context)
                    throw std::runtime_error("Could not create a PROJ context");
                operation.reset(proj_create(context.get(),
                    "+proj=utm +zone=18 +ellps=GRS80 +units=m +no_defs"));
                if (!operation)
                    throw std::runtime_error("Could not create NAD83 / UTM zone 18N transform");
            }

            [[nodiscard]] std::pair<double, double> projected(double longitude,
                double latitude) const
            {
                proj_errno_reset(operation.get());
                const PJ_COORD value = proj_trans(
                    operation.get(), PJ_FWD,
                    proj_coord(proj_torad(longitude), proj_torad(latitude), 0.0, 0.0));
                if (proj_errno(operation.get()) != 0 || !std::isfinite(value.xy.x)
                    || !std::isfinite(value.xy.y))
                    throw std::runtime_error("PROJ failed to project a DEM coordinate");
                return {value.xy.x, value.xy.y};
            }

            [[nodiscard]] std::pair<double, double> geographic(double easting,
                double northing) const
            {
                proj_errno_reset(operation.get());
                const PJ_COORD value = proj_trans(
                    operation.get(), PJ_INV, proj_coord(easting, northing, 0.0, 0.0));
                if (proj_errno(operation.get()) != 0 || !std::isfinite(value.lp.lam)
                    || !std::isfinite(value.lp.phi))
                    throw std::runtime_error("PROJ failed to invert a DEM coordinate");
                return {proj_todeg(value.lp.lam), proj_todeg(value.lp.phi)};
            }
        };

        struct TiffCloser
        {
            void operator()(TIFF* value) const { XTIFFClose(value); }
        };

        struct GtifCloser
        {
            void operator()(GTIF* value) const { GTIFFree(value); }
        };

        struct SourceTile
        {
            std::filesystem::path path;
            int west = 0;
            int north = 0;
            std::vector<float> heights;

            [[nodiscard]] float at(std::uint32_t x, std::uint32_t y) const
            {
                return heights[static_cast<std::size_t>(y) * sourceCoreSide + x];
            }
        };

        [[nodiscard]] SourceTile loadTile(const std::filesystem::path& path)
        {
            std::unique_ptr<TIFF, TiffCloser> image(
                XTIFFOpen(path.string().c_str(), "rm"));
            if (!image)
                throw std::runtime_error("Could not open DEM GeoTIFF: " + path.string());
            std::unique_ptr<GTIF, GtifCloser> geo(GTIFNew(image.get()));
            if (!geo)
                throw std::runtime_error("DEM is missing GeoTIFF metadata: " + path.string());

            std::uint32_t width = 0;
            std::uint32_t height = 0;
            std::uint16_t samples = 0;
            std::uint16_t bits = 0;
            std::uint16_t sampleFormat = 0;
            std::uint16_t planar = 0;
            TIFFGetField(image.get(), TIFFTAG_IMAGEWIDTH, &width);
            TIFFGetField(image.get(), TIFFTAG_IMAGELENGTH, &height);
            TIFFGetFieldDefaulted(image.get(), TIFFTAG_SAMPLESPERPIXEL, &samples);
            TIFFGetFieldDefaulted(image.get(), TIFFTAG_BITSPERSAMPLE, &bits);
            TIFFGetFieldDefaulted(image.get(), TIFFTAG_SAMPLEFORMAT, &sampleFormat);
            TIFFGetFieldDefaulted(image.get(), TIFFTAG_PLANARCONFIG, &planar);
            if (width != sourceSide || height != sourceSide || samples != 1 || bits != 32
                || sampleFormat != SAMPLEFORMAT_IEEEFP || planar != PLANARCONFIG_CONTIG)
                throw std::runtime_error(
                    "DEM must be a 3612x3612 single-band Float32 GeoTIFF: " + path.string());

            short geographicType = 0;
            short rasterType = 0;
            if (!GTIFKeyGet(geo.get(), GeographicTypeGeoKey, &geographicType, 0, 1)
                || geographicType != 4269
                || !GTIFKeyGet(geo.get(), GTRasterTypeGeoKey, &rasterType, 0, 1)
                || rasterType != RasterPixelIsArea)
                throw std::runtime_error(
                    "DEM must use NAD83 EPSG:4269 and pixel-is-area raster semantics: "
                    + path.string());

            std::uint16_t scaleCount = 0;
            std::uint16_t tieCount = 0;
            double* scale = nullptr;
            double* tie = nullptr;
            if (!TIFFGetField(image.get(), TIFFTAG_GEOPIXELSCALE, &scaleCount, &scale)
                || scaleCount < 2 || !TIFFGetField(image.get(), TIFFTAG_GEOTIEPOINTS,
                    &tieCount, &tie) || tieCount < 6
                || std::abs(scale[0] - sourceStepDegrees) > 1.0e-9
                || std::abs(scale[1] - sourceStepDegrees) > 1.0e-9
                || std::abs(tie[0]) > 1.0e-6 || std::abs(tie[1]) > 1.0e-6)
                throw std::runtime_error(
                    "DEM must have the expected one-arc-second USGS geotransform: "
                    + path.string());

            SourceTile result;
            result.path = path;
            result.west = static_cast<int>(std::llround(
                tie[3] + sourceCollar * scale[0]));
            result.north = static_cast<int>(std::llround(
                tie[4] - sourceCollar * scale[1]));
            if (std::abs(tie[3] + sourceCollar * scale[0] - result.west) > 1.0e-6
                || std::abs(tie[4] - sourceCollar * scale[1] - result.north) > 1.0e-6)
                throw std::runtime_error("DEM collar does not align to a one-degree tile: "
                    + path.string());
            result.heights.resize(
                static_cast<std::size_t>(sourceCoreSide) * sourceCoreSide);

            if (TIFFIsTiled(image.get()))
            {
                std::uint32_t tileWidth = 0;
                std::uint32_t tileHeight = 0;
                TIFFGetField(image.get(), TIFFTAG_TILEWIDTH, &tileWidth);
                TIFFGetField(image.get(), TIFFTAG_TILELENGTH, &tileHeight);
                const tmsize_t tileBytes = TIFFTileSize(image.get());
                if (tileWidth == 0 || tileHeight == 0 || tileBytes <= 0)
                    throw std::runtime_error("DEM has invalid TIFF tile dimensions");
                std::vector<std::byte> buffer(static_cast<std::size_t>(tileBytes));
                for (std::uint32_t y = 0; y < sourceSide; y += tileHeight)
                {
                    for (std::uint32_t x = 0; x < sourceSide; x += tileWidth)
                    {
                        if (TIFFReadTile(image.get(), buffer.data(), x, y, 0, 0) < 0)
                            throw std::runtime_error("Could not decode DEM TIFF tile: "
                                + path.string());
                        const auto* values = reinterpret_cast<const float*>(buffer.data());
                        const std::uint32_t firstX = std::max(x, sourceCollar);
                        const std::uint32_t firstY = std::max(y, sourceCollar);
                        const std::uint32_t lastX = std::min(x + tileWidth,
                            sourceCollar + sourceCoreSide);
                        const std::uint32_t lastY = std::min(y + tileHeight,
                            sourceCollar + sourceCoreSide);
                        for (std::uint32_t outputY = firstY; outputY < lastY; ++outputY)
                            for (std::uint32_t outputX = firstX; outputX < lastX; ++outputX)
                                result.heights[static_cast<std::size_t>(outputY - sourceCollar)
                                        * sourceCoreSide + outputX - sourceCollar]
                                    = values[static_cast<std::size_t>(outputY - y) * tileWidth
                                        + outputX - x];
                    }
                }
            }
            else
            {
                std::vector<float> row(sourceSide);
                for (std::uint32_t y = sourceCollar;
                    y < sourceCollar + sourceCoreSide; ++y)
                {
                    if (TIFFReadScanline(image.get(), row.data(), y, 0) < 0)
                        throw std::runtime_error("Could not decode DEM TIFF scanline: "
                            + path.string());
                    std::copy_n(row.begin() + sourceCollar, sourceCoreSide,
                        result.heights.begin()
                            + static_cast<std::size_t>(y - sourceCollar) * sourceCoreSide);
                }
            }

            const auto invalid = std::find_if(result.heights.begin(), result.heights.end(),
                [](float value) { return !std::isfinite(value) || value <= -999998.0f; });
            if (invalid != result.heights.end())
                throw std::runtime_error("DEM contains no-data inside its one-degree core: "
                    + path.string());
            return result;
        }

        struct Mosaic
        {
            std::array<SourceTile, 4> tiles;
            int west = 0;
            int east = 0;
            int south = 0;
            int north = 0;

            [[nodiscard]] const SourceTile& tileAt(int longitude, int tileNorth) const
            {
                const auto found = std::find_if(tiles.begin(), tiles.end(),
                    [&](const SourceTile& tile) {
                        return tile.west == longitude && tile.north == tileNorth;
                    });
                if (found == tiles.end())
                    throw std::runtime_error("DEM mosaic has a missing source tile");
                return *found;
            }

            [[nodiscard]] float pixel(int x, int y) const
            {
                const int width = (east - west) * static_cast<int>(sourceCoreSide);
                const int height = (north - south) * static_cast<int>(sourceCoreSide);
                x = std::clamp(x, 0, width - 1);
                y = std::clamp(y, 0, height - 1);
                const int tileX = x / static_cast<int>(sourceCoreSide);
                const int tileY = y / static_cast<int>(sourceCoreSide);
                return tileAt(west + tileX, north - tileY).at(
                    static_cast<std::uint32_t>(x % sourceCoreSide),
                    static_cast<std::uint32_t>(y % sourceCoreSide));
            }

            [[nodiscard]] float sample(double longitude, double latitude) const
            {
                const double x = std::clamp(
                    (longitude - west) / sourceStepDegrees - 0.5,
                    0.0, static_cast<double>((east - west) * sourceCoreSide - 1));
                const double y = std::clamp(
                    (north - latitude) / sourceStepDegrees - 0.5,
                    0.0, static_cast<double>((north - south) * sourceCoreSide - 1));
                const int x0 = static_cast<int>(std::floor(x));
                const int y0 = static_cast<int>(std::floor(y));
                const double tx = x - x0;
                const double ty = y - y0;
                const double a = std::lerp(
                    static_cast<double>(pixel(x0, y0)),
                    static_cast<double>(pixel(x0 + 1, y0)), tx);
                const double b = std::lerp(
                    static_cast<double>(pixel(x0, y0 + 1)),
                    static_cast<double>(pixel(x0 + 1, y0 + 1)), tx);
                return static_cast<float>(std::lerp(a, b, ty));
            }
        };

        [[nodiscard]] Mosaic loadMosaic(
            const std::array<std::filesystem::path, 4>& sources)
        {
            Mosaic result{{loadTile(sources[0]), loadTile(sources[1]),
                loadTile(sources[2]), loadTile(sources[3])}};
            result.west = std::numeric_limits<int>::max();
            result.east = std::numeric_limits<int>::min();
            result.south = std::numeric_limits<int>::max();
            result.north = std::numeric_limits<int>::min();
            std::map<std::pair<int, int>, int> coordinates;
            for (const SourceTile& tile : result.tiles)
            {
                if (!coordinates.emplace(std::pair{tile.west, tile.north}, 1).second)
                    throw std::runtime_error("DEM inputs contain a duplicate one-degree tile");
                result.west = std::min(result.west, tile.west);
                result.east = std::max(result.east, tile.west + 1);
                result.south = std::min(result.south, tile.north - 1);
                result.north = std::max(result.north, tile.north);
            }
            if (result.east - result.west != 2 || result.north - result.south != 2
                || coordinates.size() != 4)
                throw std::runtime_error("DEM inputs must form one contiguous 2x2-degree mosaic");
            for (int y = result.south + 1; y <= result.north; ++y)
                for (int x = result.west; x < result.east; ++x)
                    (void)result.tileAt(x, y);
            return result;
        }

        struct PlannedCell
        {
            CellCoord coord{};
            std::uint32_t localCellForm = 0;
        };

        struct PreparedImport
        {
            Mosaic mosaic;
            Projection projection;
            std::pair<double, double> anchor;
            DemImportSummary summary;
            std::vector<std::vector<PlannedCell>> cells;
        };

        [[nodiscard]] std::string twoDigits(std::size_t value)
        {
            std::ostringstream stream;
            stream << std::setfill('0') << std::setw(2) << value;
            return stream.str();
        }

        [[nodiscard]] PreparedImport prepare(const std::filesystem::path& editorPath,
            const std::array<std::filesystem::path, 4>& sources, bool retainCells)
        {
            PreparedImport result{loadMosaic(sources)};
            result.anchor = result.projection.projected(
                timesSquareLongitude, timesSquareLatitude);
            result.summary.editorPath = std::filesystem::absolute(editorPath);
            const auto parent = result.summary.editorPath.parent_path();
            const std::string stem = result.summary.editorPath.stem().string();
            result.summary.worldPath = parent / (stem + ".world.esm");
            result.summary.btrDirectory = parent / (stem + ".btr");

            double minimumEasting = std::numeric_limits<double>::max();
            double maximumEasting = std::numeric_limits<double>::lowest();
            double minimumNorthing = std::numeric_limits<double>::max();
            double maximumNorthing = std::numeric_limits<double>::lowest();
            for (int edge = 0; edge <= 3600; ++edge)
            {
                const double t = static_cast<double>(edge) / 1800.0;
                for (const auto [longitude, latitude] : std::array{
                         std::pair{result.mosaic.west + t, static_cast<double>(result.mosaic.south)},
                         std::pair{result.mosaic.west + t, static_cast<double>(result.mosaic.north)},
                         std::pair{static_cast<double>(result.mosaic.west), result.mosaic.south + t},
                         std::pair{static_cast<double>(result.mosaic.east), result.mosaic.south + t}})
                {
                    const auto projected = result.projection.projected(longitude, latitude);
                    minimumEasting = std::min(minimumEasting, projected.first);
                    maximumEasting = std::max(maximumEasting, projected.first);
                    minimumNorthing = std::min(minimumNorthing, projected.second);
                    maximumNorthing = std::max(maximumNorthing, projected.second);
                }
            }
            result.summary.widthMeters = maximumEasting - minimumEasting;
            result.summary.heightMeters = maximumNorthing - minimumNorthing;
            const auto cellOf = [&](double easting, double northing) {
                return CellCoord{
                    static_cast<std::int32_t>(std::floor(
                        ((easting - result.anchor.first) * demBethesdaUnitsPerMeter + 2048.0)
                        / cellSize)),
                    static_cast<std::int32_t>(std::floor(
                        ((northing - result.anchor.second) * demBethesdaUnitsPerMeter + 2048.0)
                        / cellSize))};
            };
            result.summary.minimumCell = cellOf(minimumEasting, minimumNorthing);
            result.summary.maximumCell = cellOf(maximumEasting, maximumNorthing);
            const std::int32_t width = result.summary.maximumCell.x
                - result.summary.minimumCell.x + 1;
            const std::int32_t height = result.summary.maximumCell.y
                - result.summary.minimumCell.y + 1;
            const std::size_t columns = static_cast<std::size_t>((width + 1023) / 1024);
            const std::size_t rows = static_cast<std::size_t>((height + 1023) / 1024);
            result.cells.resize(rows * columns);
            result.summary.shards.resize(rows * columns);
            for (std::size_t row = 0; row < rows; ++row)
            {
                for (std::size_t column = 0; column < columns; ++column)
                {
                    auto& shard = result.summary.shards[row * columns + column];
                    shard.path = parent / (stem + ".terrain.r" + twoDigits(row)
                        + ".c" + twoDigits(column) + ".esm");
                    shard.minimum = {result.summary.minimumCell.x
                            + static_cast<std::int32_t>(column * 1024),
                        result.summary.minimumCell.y + static_cast<std::int32_t>(row * 1024)};
                    shard.maximum = {std::min(shard.minimum.x + 1023,
                            result.summary.maximumCell.x),
                        std::min(shard.minimum.y + 1023, result.summary.maximumCell.y)};
                }
            }

            for (std::int32_t y = result.summary.minimumCell.y;
                y <= result.summary.maximumCell.y; ++y)
            {
                const double northing = result.anchor.second
                    + static_cast<double>(y) * cellSize / demBethesdaUnitsPerMeter;
                for (std::int32_t x = result.summary.minimumCell.x;
                    x <= result.summary.maximumCell.x; ++x)
                {
                    const double easting = result.anchor.first
                        + static_cast<double>(x) * cellSize / demBethesdaUnitsPerMeter;
                    const auto geographic = result.projection.geographic(easting, northing);
                    if (geographic.first < result.mosaic.west
                        || geographic.first >= result.mosaic.east
                        || geographic.second < result.mosaic.south
                        || geographic.second >= result.mosaic.north)
                        continue;
                    const std::size_t column = static_cast<std::size_t>(
                        x - result.summary.minimumCell.x) / 1024;
                    const std::size_t row = static_cast<std::size_t>(
                        y - result.summary.minimumCell.y) / 1024;
                    const std::size_t index = row * columns + column;
                    auto& shard = result.summary.shards[index];
                    const std::uint64_t cellIndex = shard.cellCount++;
                    ++result.summary.cellCount;
                    if (retainCells)
                        result.cells[index].push_back({{x, y},
                            firstLocalFormId + static_cast<std::uint32_t>(cellIndex * 2)});
                }
            }
            for (auto& shard : result.summary.shards)
            {
                if (shard.cellCount > maxDemCellsPerEsm)
                    throw std::runtime_error("DEM terrain shard exceeds 1,048,576 cells");
                shard.estimatedBytes = shard.cellCount * 1280u + 256u * 1024u;
                result.summary.estimatedBytes += shard.estimatedBytes;
            }
            return result;
        }

        [[nodiscard]] Bytes makeTes4(std::uint32_t recordCount, std::uint32_t nextObjectId,
            const std::vector<std::string>& masters, const std::string& metadata)
        {
            struct Header
            {
                float version;
                std::uint32_t recordCount;
                std::uint32_t nextObjectId;
            };
            Bytes body;
            appendSubrecord(body, hedr, Header{0.95f, recordCount, nextObjectId});
            appendStringSubrecord(body, cnam, "EMPIRE");
            for (const std::string& master : masters)
            {
                appendStringSubrecord(body, mast, master);
                appendSubrecord(body, data, std::uint64_t{0});
            }
            appendStringSubrecord(body, emgi, metadata);
            return makeRecord(tes4, 0, body, 1u);
        }

        [[nodiscard]] Bytes makeWorldRecord()
        {
            Bytes body;
            appendStringSubrecord(body, edid, "EmpireWorld");
            appendSubrecord(body, data, std::uint8_t{1});
            appendSubrecord(body, nam4, demDefaultWaterHeight);
            return makeRecord(wrld, worldFormId, body);
        }

        void writeAtomic(const std::filesystem::path& path, const Bytes& bytes)
        {
            if (path.has_parent_path())
                std::filesystem::create_directories(path.parent_path());
            std::filesystem::path temporary = path;
            temporary += ".tmp";
            {
                std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
                if (!file)
                    throw std::runtime_error("Could not create DEM ESM: " + temporary.string());
                file.write(reinterpret_cast<const char*>(bytes.data()),
                    static_cast<std::streamsize>(bytes.size()));
                file.flush();
                if (!file)
                    throw std::runtime_error("Could not write DEM ESM: " + temporary.string());
            }
            std::filesystem::rename(temporary, path);
        }

        [[nodiscard]] std::string commonMetadata(const Mosaic& mosaic)
        {
            std::ostringstream text;
            text.imbue(std::locale::classic());
            text << std::setprecision(17)
                 << "EMPIRE_DEM 1\nEPSG 4269 26918\nANCHOR "
                 << timesSquareLongitude << ' ' << timesSquareLatitude
                 << "\nBOUNDS " << mosaic.west << ' ' << mosaic.south << ' '
                 << mosaic.east << ' ' << mosaic.north
                 << "\nUNITS " << demBethesdaUnitsPerMeter << '\n';
            return text.str();
        }

        void writeRootAndOverlay(const PreparedImport& prepared)
        {
            const std::string common = commonMetadata(prepared.mosaic);
            const Bytes rootHeader = makeTes4(1, firstLocalFormId + 1,
                {"Fallout4.esm"}, common + "ROLE WORLD\n");
            Bytes rootBody = makeWorldRecord();
            appendBytes(rootBody, makeGroup(worldFormId, 1, {}));
            Bytes root = rootHeader;
            appendBytes(root, makeGroup(wrld, 0, rootBody));
            writeAtomic(prepared.summary.worldPath, root);

            std::vector<std::string> masters{
                "Fallout4.esm", prepared.summary.worldPath.filename().string()};
            std::ostringstream metadata;
            metadata << common << "ROLE OVERLAY\n";
            for (const auto& shard : prepared.summary.shards)
            {
                masters.push_back(shard.path.filename().string());
                metadata << "SHARD " << shard.path.filename().string() << ' '
                         << shard.minimum.x << ' ' << shard.minimum.y << ' '
                         << shard.maximum.x << ' ' << shard.maximum.y << ' '
                         << shard.cellCount << '\n';
            }
            const Bytes overlayHeader = makeTes4(1, firstLocalFormId,
                masters, metadata.str());
            Bytes overlayBody = makeWorldRecord();
            appendBytes(overlayBody, makeGroup(worldFormId, 1, {}));
            Bytes overlay = overlayHeader;
            appendBytes(overlay, makeGroup(wrld, 0, overlayBody));
            writeAtomic(prepared.summary.editorPath, overlay);
        }

        [[nodiscard]] LandscapeCell sampleCell(const PreparedImport& prepared, CellCoord coord)
        {
            const auto projectedAt = [&](double sampleX, double sampleY) {
                return std::pair{
                    prepared.anchor.first
                        + (coord.x * cellSize + sampleX * sampleSpacing - 2048.0)
                            / demBethesdaUnitsPerMeter,
                    prepared.anchor.second
                        + (coord.y * cellSize + sampleY * sampleSpacing - 2048.0)
                            / demBethesdaUnitsPerMeter};
            };
            const auto lowerLeftProjected = projectedAt(0.0, 0.0);
            const auto lowerRightProjected = projectedAt(32.0, 0.0);
            const auto upperLeftProjected = projectedAt(0.0, 32.0);
            const auto upperRightProjected = projectedAt(32.0, 32.0);
            const auto lowerLeft = prepared.projection.geographic(
                lowerLeftProjected.first, lowerLeftProjected.second);
            const auto lowerRight = prepared.projection.geographic(
                lowerRightProjected.first, lowerRightProjected.second);
            const auto upperLeft = prepared.projection.geographic(
                upperLeftProjected.first, upperLeftProjected.second);
            const auto upperRight = prepared.projection.geographic(
                upperRightProjected.first, upperRightProjected.second);

            LandscapeCell result;
            result.cell = coord;
            result.hasHeights = true;
            for (std::size_t y = 0; y < LandscapeCell::side; ++y)
            {
                const double ty = static_cast<double>(y) / 32.0;
                for (std::size_t x = 0; x < LandscapeCell::side; ++x)
                {
                    const double tx = static_cast<double>(x) / 32.0;
                    const double longitude = std::lerp(
                        std::lerp(lowerLeft.first, lowerRight.first, tx),
                        std::lerp(upperLeft.first, upperRight.first, tx), ty);
                    const double latitude = std::lerp(
                        std::lerp(lowerLeft.second, lowerRight.second, tx),
                        std::lerp(upperLeft.second, upperRight.second, tx), ty);
                    const double units = static_cast<double>(
                        prepared.mosaic.sample(longitude, latitude))
                        * demBethesdaUnitsPerMeter;
                    result.heights[y * LandscapeCell::side + x]
                        = static_cast<float>(std::round(units / 8.0) * 8.0);
                }
            }
            return result;
        }

        [[nodiscard]] Bytes encodeVhgt(const LandscapeCell& landscape)
        {
            std::array<std::int32_t, LandscapeCell::side * LandscapeCell::side> units{};
            for (std::size_t index = 0; index < units.size(); ++index)
                units[index] = static_cast<std::int32_t>(
                    std::lround(landscape.heights[index] / 8.0f));
            Bytes payload;
            append(payload, static_cast<float>(units.front()));
            for (std::size_t y = 0; y < LandscapeCell::side; ++y)
            {
                const std::int32_t rowDelta = y == 0 ? 0
                    : units[y * LandscapeCell::side]
                        - units[(y - 1) * LandscapeCell::side];
                if (rowDelta < -128 || rowDelta > 127)
                    throw std::runtime_error("DEM exceeds VHGT row delta range at CELL "
                        + std::to_string(landscape.cell.x) + ", "
                        + std::to_string(landscape.cell.y));
                append(payload, static_cast<std::int8_t>(rowDelta));
                for (std::size_t x = 1; x < LandscapeCell::side; ++x)
                {
                    const std::int32_t delta = units[y * LandscapeCell::side + x]
                        - units[y * LandscapeCell::side + x - 1];
                    if (delta < -128 || delta > 127)
                        throw std::runtime_error("DEM exceeds VHGT column delta range at CELL "
                            + std::to_string(landscape.cell.x) + ", "
                            + std::to_string(landscape.cell.y));
                    append(payload, static_cast<std::int8_t>(delta));
                }
            }
            append(payload, std::array<std::byte, 3>{});
            return payload;
        }

        [[nodiscard]] Bytes makeCellTree(const PreparedImport& prepared,
            const PlannedCell& planned)
        {
            const std::uint32_t cellForm = importedSelfPrefix | planned.localCellForm;
            const std::uint32_t landForm = importedSelfPrefix
                | (planned.localCellForm + 1);
            Bytes cellBody;
            appendStringSubrecord(cellBody, edid, "EmpireCell_"
                + std::to_string(planned.coord.x) + "_"
                + std::to_string(planned.coord.y));
            constexpr std::uint16_t hasWater = 0x0002;
            appendSubrecord(cellBody, data, hasWater);
            const std::array<std::int32_t, 3> coordinates{
                planned.coord.x, planned.coord.y, 0};
            appendSubrecord(cellBody, xclc, coordinates.data(), sizeof(coordinates));
            appendSubrecord(cellBody, xclw, demDefaultWaterHeight);

            const LandscapeCell heights = sampleCell(prepared, planned.coord);
            const Bytes encoded = encodeVhgt(heights);
            Bytes landBody;
            appendSubrecord(landBody, vhgt, encoded.data(), encoded.size());
            const Bytes temporary = makeGroup(cellForm, 9,
                makeRecord(land, landForm, landBody));
            const Bytes children = makeGroup(cellForm, 6, temporary);
            Bytes result = makeRecord(cell, cellForm, cellBody);
            appendBytes(result, children);
            return result;
        }

        [[nodiscard]] auto groupKey(const PlannedCell& planned)
        {
            return std::tuple{floorDivide(planned.coord.y, 32),
                floorDivide(planned.coord.x, 32), floorDivide(planned.coord.y, 8),
                floorDivide(planned.coord.x, 8), planned.coord.y, planned.coord.x};
        }

        void writeShard(const PreparedImport& prepared, std::size_t shardIndex)
        {
            auto cells = prepared.cells[shardIndex];
            std::ranges::sort(cells, [](const PlannedCell& left, const PlannedCell& right) {
                return groupKey(left) < groupKey(right);
            });
            const auto& description = prepared.summary.shards[shardIndex];
            const std::string metadata = commonMetadata(prepared.mosaic)
                + "ROLE SHARD\nINDEX " + std::to_string(shardIndex)
                + "\nRANGE " + std::to_string(description.minimum.x) + ' '
                + std::to_string(description.minimum.y) + ' '
                + std::to_string(description.maximum.x) + ' '
                + std::to_string(description.maximum.y) + "\n";
            const std::uint32_t nextObjectId = firstLocalFormId
                + static_cast<std::uint32_t>(cells.size() * 2);
            const Bytes header = makeTes4(
                1u + static_cast<std::uint32_t>(cells.size() * 2), nextObjectId,
                {"Fallout4.esm", prepared.summary.worldPath.filename().string()}, metadata);

            std::filesystem::path temporary = description.path;
            temporary += ".tmp";
            std::fstream file(temporary,
                std::ios::binary | std::ios::out | std::ios::trunc);
            if (!file)
                throw std::runtime_error("Could not create DEM shard: " + temporary.string());
            const auto write = [&](const Bytes& bytes) {
                file.write(reinterpret_cast<const char*>(bytes.data()),
                    static_cast<std::streamsize>(bytes.size()));
                if (!file)
                    throw std::runtime_error("Could not write DEM shard: " + temporary.string());
            };
            const auto beginGroup = [&](std::uint32_t label, std::uint32_t type) {
                const std::streampos offset = file.tellp();
                Bytes group;
                append(group, grup);
                append(group, std::uint32_t{0});
                append(group, label);
                append(group, type);
                append(group, std::uint32_t{0});
                append(group, std::uint32_t{0});
                write(group);
                return offset;
            };
            const auto finishGroup = [&](std::streampos offset) {
                const std::streampos end = file.tellp();
                const std::uint64_t size = static_cast<std::uint64_t>(end - offset);
                if (size > std::numeric_limits<std::uint32_t>::max())
                    throw std::runtime_error("DEM shard group exceeds the 4 GiB format limit");
                file.seekp(offset + std::streamoff(4));
                const std::uint32_t groupSize = static_cast<std::uint32_t>(size);
                file.write(reinterpret_cast<const char*>(&groupSize), sizeof(groupSize));
                file.seekp(end);
            };

            write(header);
            const auto top = beginGroup(wrld, 0);
            write(makeWorldRecord());
            const auto worldChildren = beginGroup(worldFormId, 1);
            std::size_t cursor = 0;
            std::uint64_t written = 0;
            while (cursor < cells.size())
            {
                const std::int32_t blockX = floorDivide(cells[cursor].coord.x, 32);
                const std::int32_t blockY = floorDivide(cells[cursor].coord.y, 32);
                Bytes blockBody;
                while (cursor < cells.size()
                    && floorDivide(cells[cursor].coord.x, 32) == blockX
                    && floorDivide(cells[cursor].coord.y, 32) == blockY)
                {
                    const std::int32_t subblockX = floorDivide(cells[cursor].coord.x, 8);
                    const std::int32_t subblockY = floorDivide(cells[cursor].coord.y, 8);
                    Bytes subblockBody;
                    while (cursor < cells.size()
                        && floorDivide(cells[cursor].coord.x, 32) == blockX
                        && floorDivide(cells[cursor].coord.y, 32) == blockY
                        && floorDivide(cells[cursor].coord.x, 8) == subblockX
                        && floorDivide(cells[cursor].coord.y, 8) == subblockY)
                    {
                        appendBytes(subblockBody, makeCellTree(prepared, cells[cursor++]));
                        ++written;
                    }
                    appendBytes(blockBody, makeGroup(
                        packedCellCoord(subblockX, subblockY), 5, subblockBody));
                }
                write(makeGroup(packedCellCoord(blockX, blockY), 4, blockBody));
                if (written % 65536u == 0 || written == cells.size())
                    std::cout << "DEM shard " << shardIndex + 1 << "/"
                              << prepared.summary.shards.size() << ": " << written
                              << "/" << cells.size() << " cells\n";
            }
            finishGroup(worldChildren);
            finishGroup(top);
            file.flush();
            if (!file)
                throw std::runtime_error("Could not finish DEM shard: " + temporary.string());
            file.close();
            std::filesystem::rename(temporary, description.path);
        }

        void ensureOutputsAreNew(const DemImportSummary& summary)
        {
            std::vector<std::filesystem::path> outputs{
                summary.editorPath, summary.worldPath};
            for (const auto& shard : summary.shards)
                outputs.push_back(shard.path);
            for (const auto& path : outputs)
                if (std::filesystem::exists(path))
                    throw std::runtime_error("DEM import refuses to overwrite existing output: "
                        + path.string());
        }

        [[nodiscard]] float sampleAtAssetPosition(const PreparedImport& prepared,
            double assetX, double assetY)
        {
            const double easting = prepared.anchor.first
                + (assetX - 2048.0) / demBethesdaUnitsPerMeter;
            const double northing = prepared.anchor.second
                + (assetY - 2048.0) / demBethesdaUnitsPerMeter;
            const auto geographic = prepared.projection.geographic(easting, northing);
            const double units = static_cast<double>(prepared.mosaic.sample(
                geographic.first, geographic.second)) * demBethesdaUnitsPerMeter;
            return static_cast<float>(std::round(units / 8.0) * 8.0);
        }

        void bakeBtrTile(const PreparedImport& prepared, CellCoord origin,
            const std::filesystem::path& path)
        {
            constexpr std::size_t side = 33;
            constexpr float spacing = static_cast<float>(cellSize);
            // FO4's compact BSTriShape positions overflow above 65,504. Keep the
            // level-32 tile local and restore its 1:1 dimensions with the shape scale.
            constexpr float shapeScale = 8.0f;
            std::array<float, side * side> heights{};
            std::vector<nifly::Vector3> vertices;
            std::vector<nifly::Vector3> normals;
            std::vector<nifly::Vector2> uvs;
            std::vector<nifly::Triangle> triangles;
            vertices.reserve(side * side);
            normals.reserve(side * side);
            uvs.reserve(side * side);
            triangles.reserve(32 * 32 * 2);
            for (std::size_t y = 0; y < side; ++y)
                for (std::size_t x = 0; x < side; ++x)
                    heights[y * side + x] = sampleAtAssetPosition(prepared,
                        (origin.x + static_cast<std::int32_t>(x)) * cellSize,
                        (origin.y + static_cast<std::int32_t>(y)) * cellSize);
            const auto height = [&](int x, int y) {
                return heights[static_cast<std::size_t>(std::clamp(y, 0, 32)) * side
                    + static_cast<std::size_t>(std::clamp(x, 0, 32))];
            };
            for (int y = 0; y < 33; ++y)
            {
                for (int x = 0; x < 33; ++x)
                {
                    vertices.emplace_back(x * spacing / shapeScale,
                        y * spacing / shapeScale, height(x, y) / shapeScale);
                    nifly::Vector3 normal(height(x - 1, y) - height(x + 1, y),
                        height(x, y - 1) - height(x, y + 1), 2.0f * spacing);
                    normal.Normalize();
                    normals.push_back(normal);
                    uvs.emplace_back(static_cast<float>(x) / 32.0f,
                        static_cast<float>(y) / 32.0f);
                }
            }
            for (std::uint16_t y = 0; y < 32; ++y)
            {
                for (std::uint16_t x = 0; x < 32; ++x)
                {
                    const std::uint16_t topLeft = static_cast<std::uint16_t>(y * 33 + x);
                    const std::uint16_t topRight = static_cast<std::uint16_t>(topLeft + 1);
                    const std::uint16_t bottomLeft = static_cast<std::uint16_t>(topLeft + 33);
                    const std::uint16_t bottomRight = static_cast<std::uint16_t>(bottomLeft + 1);
                    triangles.emplace_back(topLeft, topRight, bottomLeft);
                    triangles.emplace_back(topRight, bottomRight, bottomLeft);
                }
            }

            nifly::NifFile file;
            file.Create(nifly::NiVersion::getFO4());
            nifly::NiShape* shape = file.CreateShapeFromData(
                "EmpireWorld Terrain LOD", &vertices, &triangles, &uvs, &normals);
            if (!shape)
                throw std::runtime_error("Could not create BTR terrain shape");
            nifly::MatTransform transform;
            transform.scale = shapeScale;
            shape->SetTransformToParent(transform);
            if (auto* shader = file.GetShader(shape))
            {
                shader->name.get().clear();
                shader->SetWetMaterialName("");
            }
            std::filesystem::create_directories(path.parent_path());
            std::filesystem::path temporary = path;
            temporary += ".tmp";
            if (file.Save(temporary) != 0)
                throw std::runtime_error("Could not save BTR terrain shard: " + path.string());
            std::filesystem::rename(temporary, path);
        }

        std::uint64_t bakeBtrs(PreparedImport& prepared)
        {
            std::map<std::pair<std::int32_t, std::int32_t>, std::size_t> tiles;
            for (std::size_t shard = 0; shard < prepared.cells.size(); ++shard)
            {
                for (const PlannedCell& plannedCell : prepared.cells[shard])
                    tiles.try_emplace({floorDivide(plannedCell.coord.x, 32) * 32,
                        floorDivide(plannedCell.coord.y, 32) * 32}, shard);
            }
            std::uint64_t written = 0;
            for (const auto& [tile, shard] : tiles)
            {
                const std::size_t columns = static_cast<std::size_t>(
                    (prepared.summary.maximumCell.x - prepared.summary.minimumCell.x + 1024)
                    / 1024);
                const std::size_t row = shard / columns;
                const std::size_t column = shard % columns;
                const std::filesystem::path directory = prepared.summary.btrDirectory
                    / ("r" + twoDigits(row) + "c" + twoDigits(column));
                const std::filesystem::path path = directory
                    / ("empireworld.32." + std::to_string(tile.first) + "."
                        + std::to_string(tile.second) + ".btr");
                bakeBtrTile(prepared, {tile.first, tile.second}, path);
                ++written;
                if (written % 256u == 0 || written == tiles.size())
                    std::cout << "BTR bake: " << written << "/" << tiles.size() << " tiles\n";
            }
            prepared.summary.btrCount = written;
            return written;
        }

        template <typename T>
        [[nodiscard]] T readStream(std::ifstream& file, std::uint64_t offset)
        {
            T value{};
            file.seekg(static_cast<std::streamoff>(offset));
            file.read(reinterpret_cast<char*>(&value), sizeof(value));
            if (!file)
                throw std::runtime_error("Truncated DEM landscape ESM");
            return value;
        }

        [[nodiscard]] Bytes readStreamBytes(std::ifstream& file,
            std::uint64_t offset, std::size_t size)
        {
            Bytes result(size);
            file.seekg(static_cast<std::streamoff>(offset));
            file.read(reinterpret_cast<char*>(result.data()),
                static_cast<std::streamsize>(size));
            if (!file)
                throw std::runtime_error("Truncated DEM landscape ESM payload");
            return result;
        }

        template <typename T>
        [[nodiscard]] T readBytes(const Bytes& bytes, std::size_t offset)
        {
            if (offset > bytes.size() || sizeof(T) > bytes.size() - offset)
                throw std::runtime_error("Truncated DEM landscape ESM data");
            T result{};
            std::memcpy(&result, bytes.data() + offset, sizeof(result));
            return result;
        }

        struct EslHeader
        {
            std::vector<std::string> masters;
            std::string metadata;
            std::uint32_t nextObjectId = firstLocalFormId;
            std::uint64_t topOffset = 0;
        };

        [[nodiscard]] EslHeader readEsmHeader(const std::filesystem::path& path)
        {
            std::ifstream file(path, std::ios::binary);
            if (!file || readStream<std::uint32_t>(file, 0) != tes4)
                throw std::runtime_error("DEM project is missing a TES4 header: " + path.string());
            const std::uint32_t size = readStream<std::uint32_t>(file, 4);
            const Bytes body = readStreamBytes(file, recordHeaderSize, size);
            EslHeader result;
            std::size_t cursor = 0;
            while (cursor < body.size())
            {
                if (body.size() - cursor < 6)
                    throw std::runtime_error("DEM project has a truncated TES4 subrecord");
                const std::uint32_t type = readBytes<std::uint32_t>(body, cursor);
                const std::uint16_t subSize = readBytes<std::uint16_t>(body, cursor + 4);
                const std::size_t payload = cursor + 6;
                if (subSize > body.size() - payload)
                    throw std::runtime_error("DEM project TES4 subrecord exceeds its record");
                if (type == hedr && subSize >= 12)
                    result.nextObjectId = readBytes<std::uint32_t>(body, payload + 8);
                else if (type == mast)
                {
                    const char* text = reinterpret_cast<const char*>(body.data() + payload);
                    result.masters.emplace_back(text,
                        std::find(text, text + subSize, '\0'));
                }
                else if (type == emgi)
                {
                    const char* text = reinterpret_cast<const char*>(body.data() + payload);
                    result.metadata.assign(text,
                        std::find(text, text + subSize, '\0'));
                }
                cursor = payload + subSize;
            }
            result.topOffset = recordHeaderSize + size;
            return result;
        }

        [[nodiscard]] std::optional<std::array<std::int32_t, 4>> metadataRange(
            const std::string& metadata)
        {
            std::istringstream lines(metadata);
            std::string line;
            while (std::getline(lines, line))
            {
                std::istringstream values(line);
                std::string name;
                values >> name;
                if (name != "RANGE")
                    continue;
                std::array<std::int32_t, 4> result{};
                if (values >> result[0] >> result[1] >> result[2] >> result[3])
                    return result;
            }
            return std::nullopt;
        }

        [[nodiscard]] LandscapeCell decodeImportedVhgt(CellCoord coord,
            const Bytes& payload)
        {
            if (payload.size() != 1096)
                throw std::runtime_error("Imported LAND has an unsupported VHGT size");
            const float base = readBytes<float>(payload, 0);
            if (!std::isfinite(base))
                throw std::runtime_error("Imported LAND has a malformed VHGT base");
            LandscapeCell result;
            result.cell = coord;
            result.hasHeights = true;
            float row = base;
            std::size_t cursor = 4;
            for (std::size_t y = 0; y < LandscapeCell::side; ++y)
            {
                row += static_cast<float>(readBytes<std::int8_t>(payload, cursor++));
                float column = row;
                result.heights[y * LandscapeCell::side] = column * 8.0f;
                for (std::size_t x = 1; x < LandscapeCell::side; ++x)
                {
                    column += static_cast<float>(
                        readBytes<std::int8_t>(payload, cursor++));
                    result.heights[y * LandscapeCell::side + x] = column * 8.0f;
                }
            }
            return result;
        }

        class ShardReader
        {
        public:
            explicit ShardReader(std::filesystem::path path)
                : mPath(std::move(path)), mHeader(readEsmHeader(mPath)),
                  mFile(mPath, std::ios::binary)
            {
                if (!mFile)
                    throw std::runtime_error("Could not open DEM terrain shard: " + mPath.string());
                indexBlocks();
            }

            [[nodiscard]] const EslHeader& header() const noexcept { return mHeader; }

            [[nodiscard]] std::optional<DemLandscapeCell> find(
                CellCoord coord, std::optional<std::uint8_t> canonicalMaster) const
            {
                const std::uint32_t blockLabel = packedCellCoord(
                    floorDivide(coord.x, 32), floorDivide(coord.y, 32));
                const auto block = mBlocks.find(blockLabel);
                if (block == mBlocks.end())
                    return std::nullopt;
                const std::uint64_t blockEnd = block->second
                    + readStream<std::uint32_t>(mFile, block->second + 4);
                std::uint64_t subblock = block->second + recordHeaderSize;
                const std::uint32_t wantedSubblock = packedCellCoord(
                    floorDivide(coord.x, 8), floorDivide(coord.y, 8));
                while (subblock < blockEnd)
                {
                    const std::uint32_t size = readStream<std::uint32_t>(mFile, subblock + 4);
                    const std::uint32_t label = readStream<std::uint32_t>(mFile, subblock + 8);
                    if (readStream<std::uint32_t>(mFile, subblock) != grup
                        || size < recordHeaderSize || subblock + size > blockEnd)
                        throw std::runtime_error("DEM shard contains a malformed sub-block group");
                    if (label == wantedSubblock)
                    {
                        if (const auto result = findInSubblock(
                                subblock + recordHeaderSize, subblock + size,
                                coord, canonicalMaster))
                            return result;
                    }
                    subblock += size;
                }
                return std::nullopt;
            }

            [[nodiscard]] std::vector<DemLandscapeCell> all(
                std::optional<std::uint8_t> canonicalMaster) const
            {
                std::vector<DemLandscapeCell> result;
                for (const auto& [label, blockOffset] : mBlocks)
                {
                    (void)label;
                    const std::uint64_t blockEnd = blockOffset
                        + readStream<std::uint32_t>(mFile, blockOffset + 4);
                    std::uint64_t subblock = blockOffset + recordHeaderSize;
                    while (subblock < blockEnd)
                    {
                        const std::uint32_t size = readStream<std::uint32_t>(mFile, subblock + 4);
                        scanSubblock(subblock + recordHeaderSize, subblock + size,
                            canonicalMaster, result);
                        subblock += size;
                    }
                }
                return result;
            }

        private:
            std::filesystem::path mPath;
            EslHeader mHeader;
            mutable std::ifstream mFile;
            std::map<std::uint32_t, std::uint64_t> mBlocks;

            void indexBlocks()
            {
                const std::uint64_t top = mHeader.topOffset;
                if (readStream<std::uint32_t>(mFile, top) != grup
                    || readStream<std::uint32_t>(mFile, top + 8) != wrld)
                    throw std::runtime_error("DEM ESM is missing its WRLD top group");
                const std::uint64_t topEnd = top + readStream<std::uint32_t>(mFile, top + 4);
                std::uint64_t cursor = top + recordHeaderSize;
                if (readStream<std::uint32_t>(mFile, cursor) != wrld)
                    throw std::runtime_error("DEM ESM is missing EmpireWorld");
                cursor += recordHeaderSize + readStream<std::uint32_t>(mFile, cursor + 4);
                if (cursor == topEnd)
                    return;
                if (readStream<std::uint32_t>(mFile, cursor) != grup
                    || readStream<std::uint32_t>(mFile, cursor + 12) != 1)
                    throw std::runtime_error("DEM ESM is missing world children");
                const std::uint64_t childrenEnd = cursor
                    + readStream<std::uint32_t>(mFile, cursor + 4);
                cursor += recordHeaderSize;
                while (cursor < childrenEnd)
                {
                    if (readStream<std::uint32_t>(mFile, cursor) != grup
                        || readStream<std::uint32_t>(mFile, cursor + 12) != 4)
                        throw std::runtime_error("DEM ESM has an unexpected world child");
                    const std::uint32_t size = readStream<std::uint32_t>(mFile, cursor + 4);
                    const std::uint32_t label = readStream<std::uint32_t>(mFile, cursor + 8);
                    if (size < recordHeaderSize || cursor + size > childrenEnd
                        || !mBlocks.emplace(label, cursor).second)
                        throw std::runtime_error("DEM ESM has a malformed or duplicate block");
                    cursor += size;
                }
            }

            [[nodiscard]] std::pair<CellCoord, std::uint32_t> readCellRecord(
                std::uint64_t offset) const
            {
                if (readStream<std::uint32_t>(mFile, offset) != cell)
                    throw std::runtime_error("DEM sub-block contains a non-CELL record");
                const std::uint32_t size = readStream<std::uint32_t>(mFile, offset + 4);
                const std::uint32_t form = readStream<std::uint32_t>(mFile, offset + 12);
                const Bytes body = readStreamBytes(mFile, offset + recordHeaderSize, size);
                std::size_t cursor = 0;
                while (cursor < body.size())
                {
                    const std::uint32_t type = readBytes<std::uint32_t>(body, cursor);
                    const std::uint16_t subSize = readBytes<std::uint16_t>(body, cursor + 4);
                    if (type == xclc && subSize >= 8)
                        return {{readBytes<std::int32_t>(body, cursor + 6),
                                    readBytes<std::int32_t>(body, cursor + 10)}, form};
                    cursor += 6 + subSize;
                }
                throw std::runtime_error("Imported CELL is missing XCLC");
            }

            [[nodiscard]] DemLandscapeCell readCellTree(std::uint64_t cellOffset,
                CellCoord coord, std::uint32_t cellForm,
                std::optional<std::uint8_t> canonicalMaster) const
            {
                const std::uint32_t cellSizeBytes = readStream<std::uint32_t>(
                    mFile, cellOffset + 4);
                const std::uint64_t children = cellOffset + recordHeaderSize + cellSizeBytes;
                if (readStream<std::uint32_t>(mFile, children) != grup)
                    throw std::runtime_error("Imported CELL is missing its children group");
                const std::uint64_t temporary = children + recordHeaderSize;
                if (readStream<std::uint32_t>(mFile, temporary) != grup)
                    throw std::runtime_error("Imported CELL is missing temporary children");
                const std::uint64_t landOffset = temporary + recordHeaderSize;
                if (readStream<std::uint32_t>(mFile, landOffset) != land)
                    throw std::runtime_error("Imported CELL is missing LAND");
                const std::uint32_t landSize = readStream<std::uint32_t>(mFile, landOffset + 4);
                const std::uint32_t landForm = readStream<std::uint32_t>(mFile, landOffset + 12);
                const Bytes body = readStreamBytes(
                    mFile, landOffset + recordHeaderSize, landSize);
                std::size_t cursor = 0;
                while (cursor < body.size())
                {
                    const std::uint32_t type = readBytes<std::uint32_t>(body, cursor);
                    const std::uint16_t size = readBytes<std::uint16_t>(body, cursor + 4);
                    if (type == vhgt)
                    {
                        const auto canonical = [&](std::uint32_t form) {
                            return canonicalMaster
                                ? (static_cast<std::uint32_t>(*canonicalMaster) << 24u)
                                    | (form & 0x00FFFFFFu)
                                : form;
                        };
                        Bytes payload(size);
                        std::copy_n(body.begin() + static_cast<std::ptrdiff_t>(cursor + 6),
                            size, payload.begin());
                        return {decodeImportedVhgt(coord, payload),
                            {canonical(cellForm), canonical(landForm)}};
                    }
                    cursor += 6 + size;
                }
                throw std::runtime_error("Imported LAND is missing VHGT");
            }

            [[nodiscard]] std::optional<DemLandscapeCell> findInSubblock(
                std::uint64_t begin, std::uint64_t end, CellCoord wanted,
                std::optional<std::uint8_t> canonicalMaster) const
            {
                std::uint64_t cursor = begin;
                while (cursor < end)
                {
                    const auto [coord, form] = readCellRecord(cursor);
                    const std::uint32_t recordSize = readStream<std::uint32_t>(mFile, cursor + 4);
                    const std::uint64_t children = cursor + recordHeaderSize + recordSize;
                    const std::uint32_t childrenSize = readStream<std::uint32_t>(mFile, children + 4);
                    if (coord == wanted)
                        return readCellTree(cursor, coord, form, canonicalMaster);
                    cursor = children + childrenSize;
                }
                return std::nullopt;
            }

            void scanSubblock(std::uint64_t begin, std::uint64_t end,
                std::optional<std::uint8_t> canonicalMaster,
                std::vector<DemLandscapeCell>& output) const
            {
                std::uint64_t cursor = begin;
                while (cursor < end)
                {
                    const auto [coord, form] = readCellRecord(cursor);
                    output.push_back(readCellTree(cursor, coord, form, canonicalMaster));
                    const std::uint32_t recordSize = readStream<std::uint32_t>(mFile, cursor + 4);
                    const std::uint64_t children = cursor + recordHeaderSize + recordSize;
                    cursor = children + readStream<std::uint32_t>(mFile, children + 4);
                }
            }
        };

        void replaceAtomic(const std::filesystem::path& path, const Bytes& bytes)
        {
            std::filesystem::path temporary = path;
            temporary += ".tmp";
            {
                std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
                if (!file)
                    throw std::runtime_error("Could not create DEM overlay temporary file");
                file.write(reinterpret_cast<const char*>(bytes.data()),
                    static_cast<std::streamsize>(bytes.size()));
                file.flush();
                if (!file)
                    throw std::runtime_error("Could not write DEM overlay temporary file");
            }
#ifdef _WIN32
            if (!MoveFileExW(temporary.c_str(), path.c_str(),
                    MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
            {
                const DWORD error = GetLastError();
                std::error_code ignored;
                std::filesystem::remove(temporary, ignored);
                throw std::runtime_error("Could not atomically replace DEM overlay (Windows error "
                    + std::to_string(error) + ")");
            }
#else
            std::filesystem::rename(temporary, path);
#endif
        }

        [[nodiscard]] Bytes serializeOverlay(const EslHeader& header,
            const std::map<CellCoord, DemLandscapeCell>& cells,
            std::uint32_t nextFormId)
        {
            if (cells.size() > maxDemCellsPerEsm)
                throw std::runtime_error("DEM overlay exceeds 1,048,576 cells");
            // ponytail: the editor overlay is intentionally serialized in memory because it is
            // expected to contain only touched cells. Stream it if real editing sessions ever
            // approach the per-ESM million-cell ceiling.
            const Bytes tesHeader = makeTes4(
                1u + static_cast<std::uint32_t>(cells.size() * 2),
                nextFormId & 0x00FFFFFFu, header.masters, header.metadata);
            std::map<std::uint32_t, std::map<std::uint32_t, Bytes>> hierarchy;
            for (const auto& [coord, entry] : cells)
            {
                Bytes cellBody;
                appendStringSubrecord(cellBody, edid, "EmpireCell_"
                    + std::to_string(coord.x) + "_" + std::to_string(coord.y));
                constexpr std::uint16_t hasWater = 0x0002;
                appendSubrecord(cellBody, data, hasWater);
                const std::array<std::int32_t, 3> coordinates{coord.x, coord.y, 0};
                appendSubrecord(cellBody, xclc, coordinates.data(), sizeof(coordinates));
                appendSubrecord(cellBody, xclw, demDefaultWaterHeight);
                Bytes landBody;
                const Bytes encoded = encodeVhgt(entry.landscape);
                appendSubrecord(landBody, vhgt, encoded.data(), encoded.size());
                const Bytes temporary = makeGroup(entry.forms.cell, 9,
                    makeRecord(land, entry.forms.land, landBody));
                const Bytes children = makeGroup(entry.forms.cell, 6, temporary);
                Bytes tree = makeRecord(cell, entry.forms.cell, cellBody);
                appendBytes(tree, children);
                appendBytes(hierarchy[packedCellCoord(floorDivide(coord.x, 32),
                                floorDivide(coord.y, 32))]
                        [packedCellCoord(floorDivide(coord.x, 8), floorDivide(coord.y, 8))],
                    tree);
            }
            Bytes worldChildren;
            for (const auto& [blockLabel, subblocks] : hierarchy)
            {
                Bytes blockBody;
                for (const auto& [subblockLabel, body] : subblocks)
                    appendBytes(blockBody, makeGroup(subblockLabel, 5, body));
                appendBytes(worldChildren, makeGroup(blockLabel, 4, blockBody));
            }
            Bytes worldBody = makeWorldRecord();
            appendBytes(worldBody, makeGroup(worldFormId, 1, worldChildren));
            Bytes result = tesHeader;
            appendBytes(result, makeGroup(wrld, 0, worldBody));
            return result;
        }
    }

    DemImportSummary inspectDemLandscape(const std::filesystem::path& editorPath,
        const std::array<std::filesystem::path, 4>& sources)
    {
        return prepare(editorPath, sources, false).summary;
    }

    DemImportSummary importDemLandscape(const std::filesystem::path& editorPath,
        const std::array<std::filesystem::path, 4>& sources, DemImportOptions options)
    {
        PreparedImport prepared = prepare(editorPath, sources, true);
        ensureOutputsAreNew(prepared.summary);
        if (options.bakeBtr && std::filesystem::exists(prepared.summary.btrDirectory))
            throw std::runtime_error("DEM import refuses to overwrite existing BTR output: "
                + prepared.summary.btrDirectory.string());
        const auto parent = prepared.summary.editorPath.parent_path();
        if (!parent.empty())
            std::filesystem::create_directories(parent);
        const auto available = std::filesystem::space(parent.empty()
            ? std::filesystem::current_path() : parent).available;
        const std::uint64_t largestShard = std::ranges::max(
            prepared.summary.shards, {}, &DemImportShard::estimatedBytes).estimatedBytes;
        if (available < prepared.summary.estimatedBytes + largestShard)
            throw std::runtime_error("DEM import does not have enough free disk space");

        for (std::size_t index = 0; index < prepared.summary.shards.size(); ++index)
            writeShard(prepared, index);
        writeRootAndOverlay(prepared);
        if (options.bakeBtr)
            (void)bakeBtrs(prepared);
        return prepared.summary;
    }

    struct DemLandscapeProject::Impl
    {
        struct Shard
        {
            CellCoord minimum{};
            CellCoord maximum{};
            std::uint64_t cellCount = 0;
            std::uint8_t masterIndex = 0;
            std::unique_ptr<ShardReader> reader;

            [[nodiscard]] bool covers(CellCoord coord) const noexcept
            {
                return coord.x >= minimum.x && coord.x <= maximum.x
                    && coord.y >= minimum.y && coord.y <= maximum.y;
            }
        };

        std::filesystem::path path;
        EslHeader header;
        std::vector<Shard> shards;
        std::map<CellCoord, DemLandscapeCell> overlay;
        std::uint32_t nextForm = firstLocalFormId;
        std::uint64_t totalCells = 0;

        explicit Impl(const std::filesystem::path& editorPath)
            : path(std::filesystem::absolute(editorPath)), header(readEsmHeader(path))
        {
            if (!header.metadata.starts_with("EMPIRE_DEM 1\n")
                || header.metadata.find("\nROLE OVERLAY\n") == std::string::npos)
                throw std::runtime_error("ESM is not an EMPIRE DEM editor overlay: "
                    + path.string());
            if (header.masters.size() < 3 || header.masters.front() != "Fallout4.esm")
                throw std::runtime_error("DEM overlay has an invalid master list");
            std::istringstream lines(header.metadata);
            std::string line;
            std::size_t shardIndex = 0;
            while (std::getline(lines, line))
            {
                std::istringstream values(line);
                std::string type;
                values >> type;
                if (type != "SHARD")
                    continue;
                std::string filename;
                Shard shard;
                if (!(values >> filename >> shard.minimum.x >> shard.minimum.y
                        >> shard.maximum.x >> shard.maximum.y >> shard.cellCount))
                    throw std::runtime_error("DEM overlay has malformed shard metadata");
                if (shardIndex + 2 >= header.masters.size()
                    || header.masters[shardIndex + 2] != filename
                    || shardIndex + 2 >= 255)
                    throw std::runtime_error("DEM overlay shard metadata disagrees with its masters");
                shard.masterIndex = static_cast<std::uint8_t>(shardIndex + 2);
                shard.reader = std::make_unique<ShardReader>(path.parent_path() / filename);
                totalCells += shard.cellCount;
                shards.push_back(std::move(shard));
                ++shardIndex;
            }
            if (shards.empty() || header.masters.size() != shards.size() + 2)
                throw std::runtime_error("DEM overlay is missing its terrain shards");
            const std::uint32_t selfPrefix = static_cast<std::uint32_t>(
                header.masters.size()) << 24u;
            nextForm = selfPrefix | header.nextObjectId;

            ShardReader overlayReader(path);
            for (auto& entry : overlayReader.all(std::nullopt))
                overlay.emplace(entry.landscape.cell, std::move(entry));
            if (overlay.size() > maxDemCellsPerEsm)
                throw std::runtime_error("DEM overlay exceeds 1,048,576 cells");
        }

        [[nodiscard]] const Shard* shardFor(CellCoord coord) const noexcept
        {
            const auto found = std::find_if(shards.begin(), shards.end(),
                [&](const Shard& shard) { return shard.covers(coord); });
            return found == shards.end() ? nullptr : &*found;
        }

        [[nodiscard]] std::optional<DemLandscapeCell> find(CellCoord coord) const
        {
            if (const auto changed = overlay.find(coord); changed != overlay.end())
                return changed->second;
            const Shard* shard = shardFor(coord);
            if (!shard)
                return std::nullopt;
            return shard->reader->find(coord, shard->masterIndex);
        }

        void save(const EditableLandscape& landscape,
            const std::set<CellCoord>& changedCells)
        {
            for (const CellCoord coord : changedCells)
            {
                const auto forms = landscape.cellForms.find(coord);
                if (forms == landscape.cellForms.end())
                    throw std::runtime_error("Changed DEM cell is missing from the editor document");
                overlay[coord] = {landscape.cell(coord), forms->second};
            }
            if (overlay.size() > maxDemCellsPerEsm)
                throw std::runtime_error("DEM overlay exceeds 1,048,576 cells");
            nextForm = landscape.nextFormId;
            replaceAtomic(path, serializeOverlay(header, overlay, nextForm));
            header.nextObjectId = nextForm & 0x00FFFFFFu;
        }
    };

    DemLandscapeProject::DemLandscapeProject(const std::filesystem::path& editorPath)
        : mImpl(std::make_unique<Impl>(editorPath))
    {
    }

    DemLandscapeProject::~DemLandscapeProject() = default;
    DemLandscapeProject::DemLandscapeProject(DemLandscapeProject&&) noexcept = default;
    DemLandscapeProject& DemLandscapeProject::operator=(DemLandscapeProject&&) noexcept = default;

    bool DemLandscapeProject::recognizes(const std::filesystem::path& editorPath)
    {
        if (!std::filesystem::is_regular_file(editorPath))
            return false;
        try
        {
            const EslHeader header = readEsmHeader(editorPath);
            return header.metadata.starts_with("EMPIRE_DEM 1\n")
                && header.metadata.find("\nROLE OVERLAY\n") != std::string::npos;
        }
        catch (...)
        {
            return false;
        }
    }

    bool DemLandscapeProject::contains(CellCoord coord) const
    {
        return mImpl->find(coord).has_value();
    }

    std::optional<DemLandscapeCell> DemLandscapeProject::cell(CellCoord coord) const
    {
        return mImpl->find(coord);
    }

    std::uint32_t DemLandscapeProject::nextFormId() const noexcept
    {
        return mImpl->nextForm;
    }

    std::uint64_t DemLandscapeProject::importedCellCount() const noexcept
    {
        return mImpl->totalCells;
    }

    std::size_t DemLandscapeProject::overlayCellCount() const noexcept
    {
        return mImpl->overlay.size();
    }

    void DemLandscapeProject::save(const EditableLandscape& landscape,
        const std::set<CellCoord>& changedCells)
    {
        mImpl->save(landscape, changedCells);
    }
}
