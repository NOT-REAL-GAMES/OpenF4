#include "assets/lod_terrain.hpp"

#include "assets/dem_import.hpp"
#include "assets/esm_document.hpp"

#include <NifFile.hpp>
#include <Shaders.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <Windows.h>
#endif

namespace openf4::assets
{
    namespace
    {
        constexpr int gridSide = 33;
        constexpr int marginSide = gridSide + 2;
        constexpr float shapeScale = 8.0f;
        constexpr std::int64_t cellSamples = 32;

        [[nodiscard]] std::int64_t floorDivide(std::int64_t value, std::int64_t divisor)
        {
            return value / divisor - (value % divisor < 0 ? 1 : 0);
        }

        [[nodiscard]] bool validLevel(int level)
        {
            return level == 4 || level == 8 || level == 16 || level == 32;
        }

        void publishBtr(const std::filesystem::path& temporary,
            const std::filesystem::path& output)
        {
#ifdef _WIN32
            if (!MoveFileExW(temporary.c_str(), output.c_str(),
                    MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
            {
                const DWORD error = GetLastError();
                std::error_code ignored;
                std::filesystem::remove(temporary, ignored);
                throw std::runtime_error("Could not publish BTR (Windows error "
                    + std::to_string(error) + "): " + output.string());
            }
#else
            std::filesystem::rename(temporary, output);
#endif
        }
    }

    void writeTerrainBtr(const std::filesystem::path& path, float spacing,
        const std::function<std::optional<float>(int, int)>& height,
        const std::function<bool(int, int)>& covered)
    {
        if (!std::isfinite(spacing) || spacing <= 0.0f)
            throw std::runtime_error("Invalid BTR terrain grid spacing");
        std::array<std::optional<float>, marginSide * marginSide> heights;
        for (int y = -1; y <= gridSide; ++y)
            for (int x = -1; x <= gridSide; ++x)
            {
                auto value = height(x, y);
                if (value && (!std::isfinite(*value) || std::abs(*value / shapeScale) > 65504.0f))
                    throw std::runtime_error("BTR terrain height is non-finite or unbounded");
                heights[static_cast<std::size_t>(y + 1) * marginSide + x + 1] = value;
            }
        const auto at = [&](int x, int y) -> std::optional<float> {
            return heights[static_cast<std::size_t>(y + 1) * marginSide + x + 1];
        };

        std::vector<nifly::Vector3> vertices;
        std::vector<nifly::Vector3> normals;
        std::vector<nifly::Vector2> uvs;
        std::vector<nifly::Triangle> triangles;
        vertices.reserve(gridSide * gridSide);
        normals.reserve(gridSide * gridSide);
        uvs.reserve(gridSide * gridSide);
        triangles.reserve((gridSide - 1) * (gridSide - 1) * 2);
        for (int y = 0; y < gridSide; ++y)
            for (int x = 0; x < gridSide; ++x)
            {
                const float center = at(x, y).value_or(0.0f);
                vertices.emplace_back(x * spacing / shapeScale,
                    y * spacing / shapeScale, center / shapeScale);
                nifly::Vector3 normal(
                    at(x - 1, y).value_or(center) - at(x + 1, y).value_or(center),
                    at(x, y - 1).value_or(center) - at(x, y + 1).value_or(center),
                    2.0f * spacing);
                normal.Normalize();
                normals.push_back(normal);
                uvs.emplace_back(static_cast<float>(x) / (gridSide - 1),
                    static_cast<float>(y) / (gridSide - 1));
            }
        for (std::uint16_t y = 0; y < gridSide - 1; ++y)
            for (std::uint16_t x = 0; x < gridSide - 1; ++x)
            {
                if (!covered(x, y))
                    continue;
                const auto a = static_cast<std::uint16_t>(y * gridSide + x);
                const auto b = static_cast<std::uint16_t>(a + 1);
                const auto c = static_cast<std::uint16_t>(a + gridSide);
                const auto d = static_cast<std::uint16_t>(c + 1);
                if (!at(x, y) || !at(x + 1, y) || !at(x, y + 1) || !at(x + 1, y + 1))
                    throw std::runtime_error("BTR occupied quad has a missing LAND height");
                triangles.emplace_back(a, b, c);
                triangles.emplace_back(b, d, c);
            }
        if (triangles.empty())
            throw std::runtime_error("BTR tile contains no LAND triangles");

        nifly::NifFile file;
        file.Create(nifly::NiVersion::getFO4());
        nifly::NiShape* shape = file.CreateShapeFromData(
            "OpenF4World Terrain LOD", &vertices, &triangles, &uvs, &normals);
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
        if (!path.parent_path().empty())
            std::filesystem::create_directories(path.parent_path());
        std::filesystem::path temporary = path;
        temporary += ".tmp";
        if (file.Save(temporary) != 0)
            throw std::runtime_error("Could not save BTR terrain shard: " + path.string());
        publishBtr(temporary, path);
    }

    struct TerrainLodSource::Impl
    {
        std::filesystem::path path;
        std::unique_ptr<DemLandscapeProject> dem;
        std::unique_ptr<EsmDocument> document;

        explicit Impl(const std::filesystem::path& worldEsm)
            : path(std::filesystem::absolute(worldEsm))
        {
            if (DemLandscapeProject::recognizes(path))
                dem = std::make_unique<DemLandscapeProject>(path);
            else
            {
                document = std::make_unique<EsmDocument>(path);
                if (!document->isOpenF4Owned())
                    throw std::runtime_error("Terrain LOD bake requires an OpenF4-owned world");
            }
        }

        [[nodiscard]] std::optional<LandscapeCell> cell(CellCoord coord) const
        {
            if (dem)
            {
                if (auto result = dem->cell(coord); result && result->landscape.hasHeights)
                    return std::move(result->landscape);
            }
            else if (auto result = document->terrainCell(coord);
                result && result->landscape.hasHeights)
                return std::move(result->landscape);
            return std::nullopt;
        }
    };

    TerrainLodSource::TerrainLodSource(const std::filesystem::path& worldEsm)
        : mImpl(std::make_unique<Impl>(worldEsm))
    {
    }
    TerrainLodSource::~TerrainLodSource() = default;
    TerrainLodSource::TerrainLodSource(TerrainLodSource&&) noexcept = default;
    TerrainLodSource& TerrainLodSource::operator=(TerrainLodSource&&) noexcept = default;

    std::vector<LodTerrainTile> TerrainLodSource::tiles() const
    {
        std::set<LodTerrainTile> result;
        const auto include = [&](CellCoord cell) {
            for (const int level : {4, 8, 16, 32})
                result.insert({level,
                    {static_cast<std::int32_t>(floorDivide(cell.x, level) * level),
                        static_cast<std::int32_t>(floorDivide(cell.y, level) * level)}});
        };
        if (mImpl->dem)
            mImpl->dem->forEachCell(include);
        else
            for (const auto& [coord, cell] : mImpl->document->terrainCells())
                if (cell.landFormId)
                    include(coord);
        return {result.begin(), result.end()};
    }

    std::vector<std::filesystem::path> TerrainLodSource::sourceFiles() const
    {
        return mImpl->dem ? mImpl->dem->sourceFiles()
            : std::vector<std::filesystem::path>{mImpl->path};
    }

    std::optional<std::pair<CellCoord, CellCoord>> TerrainLodSource::shardCellBounds(
        const std::filesystem::path& source) const
    {
        return mImpl->dem ? mImpl->dem->shardCellBounds(source) : std::nullopt;
    }

    void TerrainLodSource::bake(const LodTerrainTile& tile,
        const std::filesystem::path& outputPath) const
    {
        if (!validLevel(tile.level) || tile.origin.x % tile.level != 0
            || tile.origin.y % tile.level != 0)
            throw std::runtime_error("Invalid terrain LOD tile key");

        // ponytail: one tile-local FIFO cache bounds DEM LAND memory to about 6 MiB;
        // shard-aware prefetch is only useful if profiling shows decode cost dominates.
        std::map<CellCoord, std::optional<LandscapeCell>> cache;
        std::deque<CellCoord> insertionOrder;
        const auto cellAt = [&](CellCoord coord) -> const LandscapeCell* {
            auto found = cache.find(coord);
            if (found == cache.end())
            {
                if (cache.size() == 1300)
                {
                    cache.erase(insertionOrder.front());
                    insertionOrder.pop_front();
                }
                found = cache.emplace(coord, mImpl->cell(coord)).first;
                insertionOrder.push_back(coord);
            }
            return found->second ? &*found->second : nullptr;
        };
        const auto height = [&](int gridX, int gridY) -> std::optional<float> {
            const std::int64_t sampleX = static_cast<std::int64_t>(tile.origin.x)
                * cellSamples + static_cast<std::int64_t>(gridX) * tile.level;
            const std::int64_t sampleY = static_cast<std::int64_t>(tile.origin.y)
                * cellSamples + static_cast<std::int64_t>(gridY) * tile.level;
            const auto cellX = floorDivide(sampleX, cellSamples);
            const auto cellY = floorDivide(sampleY, cellSamples);
            for (int dy = 0; dy >= -1; --dy)
                for (int dx = 0; dx >= -1; --dx)
                {
                    if ((dx != 0 && sampleX % cellSamples != 0)
                        || (dy != 0 && sampleY % cellSamples != 0))
                        continue;
                    const auto x = cellX + dx;
                    const auto y = cellY + dy;
                    if (x < std::numeric_limits<std::int32_t>::min()
                        || x > std::numeric_limits<std::int32_t>::max()
                        || y < std::numeric_limits<std::int32_t>::min()
                        || y > std::numeric_limits<std::int32_t>::max())
                        continue;
                    const auto* cell = cellAt({static_cast<std::int32_t>(x),
                        static_cast<std::int32_t>(y)});
                    if (cell)
                    {
                        const auto localX = static_cast<std::size_t>(sampleX - x * cellSamples);
                        const auto localY = static_cast<std::size_t>(sampleY - y * cellSamples);
                        return cell->heights[localY * LandscapeCell::side + localX];
                    }
                }
            return std::nullopt;
        };
        const auto covered = [&](int gridX, int gridY) {
            const std::int64_t cellX = static_cast<std::int64_t>(tile.origin.x)
                + static_cast<std::int64_t>(gridX) * tile.level / 32;
            const std::int64_t cellY = static_cast<std::int64_t>(tile.origin.y)
                + static_cast<std::int64_t>(gridY) * tile.level / 32;
            return cellX >= std::numeric_limits<std::int32_t>::min()
                && cellX <= std::numeric_limits<std::int32_t>::max()
                && cellY >= std::numeric_limits<std::int32_t>::min()
                && cellY <= std::numeric_limits<std::int32_t>::max()
                && cellAt({static_cast<std::int32_t>(cellX),
                    static_cast<std::int32_t>(cellY)}) != nullptr;
        };
        writeTerrainBtr(outputPath, tile.level * 128.0f, height, covered);
    }
}
