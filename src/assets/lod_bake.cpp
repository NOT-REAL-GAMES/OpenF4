#include "assets/lod_bake.hpp"

#include "assets/asset_resolver.hpp"
#include "assets/lod_statics.hpp"
#include "assets/lod_terrain.hpp"

#include <sqlite3.h>

#include <algorithm>
#include <chrono>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

#ifdef _WIN32
#include <windows.h>
#endif

namespace openf4::assets
{
namespace
{
    constexpr int terrainKind = 0;
    constexpr int staticKind = 1;
    // Bump when a baker changes generated geometry or materials.
    constexpr int recipeVersions[] = {1, 1}; // terrain, statics

    struct Key
    {
        int kind = 0;
        int level = 0;
        int x = 0;
        int y = 0;
        auto operator<=>(const Key&) const = default;
    };

    struct Row
    {
        bool dirty = true;
        std::uint64_t bytes = 0;
    };

    struct Stamp
    {
        std::int64_t bytes = -1;
        std::int64_t modified = 0;
        auto operator<=>(const Stamp&) const = default;
    };

    using Stamps = std::map<std::string, Stamp>;

    [[nodiscard]] std::string utf8(const std::filesystem::path& path)
    {
        const auto value = path.generic_u8string();
        return {value.begin(), value.end()};
    }

    [[nodiscard]] Stamp stamp(const std::filesystem::path& path)
    {
        std::error_code error;
        const auto bytes = std::filesystem::file_size(path, error);
        if (error) return {};
        const auto modified = std::filesystem::last_write_time(path, error);
        if (error) return {};
        return {static_cast<std::int64_t>(bytes),
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                modified.time_since_epoch()).count()};
    }

    [[nodiscard]] Stamps sourceStamps(std::vector<std::filesystem::path> paths,
        const std::filesystem::path& world)
    {
        paths.push_back(world);
        Stamps result;
        for (const auto& path : paths)
        {
            const auto absolute = std::filesystem::absolute(path).lexically_normal();
            result[utf8(absolute)] = stamp(absolute);
        }
        return result;
    }

    [[nodiscard]] std::filesystem::path fromUtf8(const std::string& value)
    {
        return std::filesystem::path(std::u8string(value.begin(), value.end()));
    }

    struct Database
    {
        sqlite3* handle = nullptr;

        Database(const std::filesystem::path& path, bool create, bool writable = false)
        {
            if (!create && !std::filesystem::exists(path)) return;
            if (create) std::filesystem::create_directories(path.parent_path());
            const int flags = create ? SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE
                : writable ? SQLITE_OPEN_READWRITE : SQLITE_OPEN_READONLY;
            const int result = sqlite3_open_v2(utf8(path).c_str(), &handle, flags, nullptr);
            if (result != SQLITE_OK)
            {
                const std::string message = handle ? sqlite3_errmsg(handle) : "could not open database";
                if (handle) sqlite3_close(handle);
                handle = nullptr;
                throw std::runtime_error("LOD build index: " + message);
            }
            sqlite3_busy_timeout(handle, 5000);
            if (create)
            {
                exec("CREATE TABLE IF NOT EXISTS tiles("
                    "kind INTEGER NOT NULL, level INTEGER NOT NULL, x INTEGER NOT NULL, y INTEGER NOT NULL,"
                    "dirty INTEGER NOT NULL DEFAULT 1, revision INTEGER NOT NULL DEFAULT 0,"
                    "bytes INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(kind,level,x,y))");
                exec("CREATE TABLE IF NOT EXISTS sources("
                    "kind INTEGER NOT NULL, path TEXT NOT NULL, bytes INTEGER NOT NULL,"
                    "modified INTEGER NOT NULL, PRIMARY KEY(kind,path))");
                exec("CREATE TABLE IF NOT EXISTS members("
                    "level INTEGER NOT NULL, x INTEGER NOT NULL, y INTEGER NOT NULL,"
                    "reference INTEGER NOT NULL, PRIMARY KEY(level,x,y,reference))");
                exec("CREATE TABLE IF NOT EXISTS metadata("
                    "key TEXT PRIMARY KEY, value INTEGER NOT NULL)");
                exec("INSERT OR IGNORE INTO metadata(key,value) VALUES('revision',0)");
            }
        }

        ~Database() { if (handle) sqlite3_close(handle); }
        Database(const Database&) = delete;
        Database& operator=(const Database&) = delete;

        void exec(const char* sql) const
        {
            char* message = nullptr;
            const int result = sqlite3_exec(handle, sql, nullptr, nullptr, &message);
            if (result == SQLITE_OK) return;
            const std::string error = message ? message : sqlite3_errmsg(handle);
            sqlite3_free(message);
            throw std::runtime_error("LOD build index: " + error);
        }
    };

    struct Statement
    {
        sqlite3_stmt* handle = nullptr;
        sqlite3* database = nullptr;

        Statement(const Database& db, const char* sql) : database(db.handle)
        {
            if (sqlite3_prepare_v2(database, sql, -1, &handle, nullptr) != SQLITE_OK)
                throw std::runtime_error("LOD build index: " + std::string(sqlite3_errmsg(database)));
        }
        ~Statement() { sqlite3_finalize(handle); }
        Statement(const Statement&) = delete;
        Statement& operator=(const Statement&) = delete;

        void bind(int index, std::int64_t value)
        {
            if (sqlite3_bind_int64(handle, index, value) != SQLITE_OK)
                throw std::runtime_error("LOD build index: " + std::string(sqlite3_errmsg(database)));
        }
        void bind(int index, const std::string& value)
        {
            if (sqlite3_bind_text(handle, index, value.c_str(),
                    static_cast<int>(value.size()), SQLITE_TRANSIENT) != SQLITE_OK)
                throw std::runtime_error("LOD build index: " + std::string(sqlite3_errmsg(database)));
        }
        [[nodiscard]] int step()
        {
            const int result = sqlite3_step(handle);
            if (result != SQLITE_ROW && result != SQLITE_DONE)
                throw std::runtime_error("LOD build index: " + std::string(sqlite3_errmsg(database)));
            return result;
        }
        void done()
        {
            if (step() != SQLITE_DONE)
                throw std::runtime_error("LOD build index: unexpected query result");
            sqlite3_reset(handle);
            sqlite3_clear_bindings(handle);
        }
    };

    [[nodiscard]] std::uint64_t revision(const Database& db)
    {
        if (!db.handle) return 0;
        Statement query(db, "SELECT value FROM metadata WHERE key='revision'");
        return query.step() == SQLITE_ROW
            ? static_cast<std::uint64_t>(sqlite3_column_int64(query.handle, 0)) : 0;
    }

    [[nodiscard]] std::map<Key, Row> readRows(const Database& db)
    {
        std::map<Key, Row> result;
        if (!db.handle) return result;
        Statement query(db, "SELECT kind,level,x,y,dirty,bytes FROM tiles");
        while (query.step() == SQLITE_ROW)
        {
            const Key key{sqlite3_column_int(query.handle, 0), sqlite3_column_int(query.handle, 1),
                sqlite3_column_int(query.handle, 2), sqlite3_column_int(query.handle, 3)};
            result.emplace(key, Row{sqlite3_column_int(query.handle, 4) != 0,
                static_cast<std::uint64_t>(sqlite3_column_int64(query.handle, 5))});
        }
        return result;
    }

    [[nodiscard]] bool currentRecipe(const Database& db, int kind)
    {
        if (!db.handle) return false;
        Statement query(db, "SELECT value FROM metadata WHERE key=?");
        query.bind(1, std::string(kind == terrainKind ? "terrain_recipe" : "static_recipe"));
        return query.step() == SQLITE_ROW && sqlite3_column_int(query.handle, 0) == recipeVersions[kind];
    }

    [[nodiscard]] Stamps readSources(const Database& db, int kind)
    {
        Stamps result;
        if (!db.handle) return result;
        Statement query(db, "SELECT path,bytes,modified FROM sources WHERE kind=?");
        query.bind(1, kind);
        while (query.step() == SQLITE_ROW)
        {
            const auto* path = reinterpret_cast<const char*>(sqlite3_column_text(query.handle, 0));
            result.emplace(path ? path : "", Stamp{sqlite3_column_int64(query.handle, 1),
                sqlite3_column_int64(query.handle, 2)});
        }
        return result;
    }

    [[nodiscard]] int floorDivide(int value, int divisor)
    {
        const int quotient = value / divisor;
        return quotient - (value % divisor < 0 ? 1 : 0);
    }

    [[nodiscard]] Key terrainKey(const LodTerrainTile& tile)
    {
        return {terrainKind, tile.level, tile.origin.x, tile.origin.y};
    }

    [[nodiscard]] Key staticKey(const LodStaticTile& tile)
    {
        return {staticKind, 4, tile.origin.x, tile.origin.y};
    }

    [[nodiscard]] std::filesystem::path outputPath(const std::filesystem::path& world,
        const Key& key)
    {
        const std::string filename = "openf4world." + std::to_string(key.level) + "."
            + std::to_string(key.x) + "." + std::to_string(key.y);
        if (key.kind == terrainKind)
        {
            auto directory = world;
            directory.replace_extension(".btr");
            return directory / (filename + ".btr");
        }
        return AssetResolver::projectAssetRoot(world) / "lod" / "tiles"
            / (filename + ".glb");
    }

    struct Work
    {
        Key key;
        bool removed = false;
        std::uint64_t estimatedBytes = 0;
    };

    struct Plan
    {
        std::vector<Work> work;
        LodBakePreflight summary;
        Stamps terrainSources;
        Stamps staticSources;
    };

    [[nodiscard]] Plan plan(const std::filesystem::path& world,
        const TerrainLodSource& terrain, const StaticLodSource& statics,
        const LodBakeOptions& options, const Database& db)
    {
        if (!options.terrain && !options.statics)
            throw std::invalid_argument("LOD bake needs terrain or statics enabled");
        const auto rows = readRows(db);
        Plan result;
        std::set<Key> current;
        std::set<Key> invalidated;
        bool terrainFull = false;
        bool staticFull = false;
        if (options.terrain)
        {
            result.terrainSources = sourceStamps(terrain.sourceFiles(), world);
            const auto previous = readSources(db, terrainKind);
            const auto tiles = terrain.tiles();
            for (const auto& tile : tiles) current.insert(terrainKey(tile));
            terrainFull = options.full || previous.empty() || !currentRecipe(db, terrainKind);
            if (!terrainFull)
            {
                // Include the one-cell normal-sampling halo; metadata bounds
                // also retain coverage of CELL records removed from the shard.
                if (previous.size() != result.terrainSources.size()) terrainFull = true;
                for (const auto& [path, old] : previous)
                {
                    const auto found = result.terrainSources.find(path);
                    if (found == result.terrainSources.end())
                    { terrainFull = true; break; }
                    if (found->second == old) continue;
                    const auto bounds = terrain.shardCellBounds(fromUtf8(path));
                    if (!bounds) { terrainFull = true; break; }
                    for (const auto& tile : tiles)
                        if (static_cast<std::int64_t>(tile.origin.x) <= static_cast<std::int64_t>(bounds->second.x) + 1
                            && static_cast<std::int64_t>(tile.origin.x) + tile.level >= bounds->first.x
                            && static_cast<std::int64_t>(tile.origin.y) <= static_cast<std::int64_t>(bounds->second.y) + 1
                            && static_cast<std::int64_t>(tile.origin.y) + tile.level >= bounds->first.y)
                            invalidated.insert(terrainKey(tile));
                }
            }
        }
        if (options.statics)
        {
            result.staticSources = sourceStamps(statics.sourceFiles(), world);
            const auto previous = readSources(db, staticKind);
            const auto tiles = statics.tiles();
            for (const auto& tile : tiles) current.insert(staticKey(tile));
            staticFull = options.full || previous.empty() || !currentRecipe(db, staticKind);
            if (!staticFull)
            {
                std::set<std::string> changedFiles;
                for (const auto& [path, old] : previous)
                {
                    const auto found = result.staticSources.find(path);
                    // A removed dependency may still be used by a tile whose
                    // resolver can no longer find it. Dirty tiles elsewhere do
                    // not account for that external change.
                    if (found == result.staticSources.end())
                    {
                        if (old.bytes >= 0 && stamp(fromUtf8(path)).bytes < 0) staticFull = true;
                    }
                    else if (found->second != old) changedFiles.insert(path);
                }
                for (const auto& [path, value] : result.staticSources)
                {
                    (void)value;
                    if (!previous.contains(path)) changedFiles.insert(path);
                }
                if (changedFiles.contains(utf8(world))) staticFull = true;
                for (const auto& path : changedFiles)
                    if (fromUtf8(path).extension() == ".esm") staticFull = true;
                if (!staticFull && !changedFiles.empty())
                {
                    std::set<std::string> matched;
                    for (const auto& tile : tiles)
                        for (const auto& source : statics.sourceFilesForTile(tile))
                        {
                            const auto name = utf8(std::filesystem::absolute(source).lexically_normal());
                            if (changedFiles.contains(name))
                            {
                                matched.insert(name);
                                invalidated.insert(staticKey(tile));
                            }
                        }
                    if (matched != changedFiles) staticFull = true;
                }
            }
        }
        result.summary.full = terrainFull || staticFull;
        for (const Key& key : current)
        {
            const auto row = rows.find(key);
            const auto output = outputPath(world, key);
            const bool full = key.kind == terrainKind ? terrainFull : staticFull;
            if (!full && !invalidated.contains(key)
                && row != rows.end() && !row->second.dirty
                && (row->second.bytes ? std::filesystem::is_regular_file(output)
                    : key.kind == staticKind && !std::filesystem::exists(output))) continue;
            std::uint64_t estimate = row != rows.end() ? row->second.bytes : 0;
            if (!estimate && std::filesystem::is_regular_file(output))
                estimate = std::filesystem::file_size(output);
            if (!estimate) estimate = key.kind == terrainKind ? 36'000 : 100'000;
            result.work.push_back({key, false, estimate});
            result.summary.estimatedBytes += estimate;
            ++(key.kind == terrainKind ? result.summary.terrainTiles
                : result.summary.staticTiles);
        }
        for (const auto& [key, row] : rows)
        {
            (void)row;
            if ((key.kind == terrainKind && !options.terrain)
                || (key.kind == staticKind && !options.statics)
                || current.contains(key)) continue;
            result.work.push_back({key, true, 0});
            ++(key.kind == terrainKind ? result.summary.terrainTiles
                : result.summary.staticTiles);
        }
        return result;
    }

    void markDirty(const Key& key, Statement& query)
    {
        query.bind(1, key.kind);
        query.bind(2, key.level);
        query.bind(3, key.x);
        query.bind(4, key.y);
        query.done();
    }

    void replaceSources(const Database& db, int kind, const Stamps& stamps)
    {
        Statement clear(db, "DELETE FROM sources WHERE kind=?");
        clear.bind(1, kind);
        clear.done();
        Statement insert(db,
            "INSERT INTO sources(kind,path,bytes,modified) VALUES(?,?,?,?)");
        for (const auto& [path, value] : stamps)
        {
            insert.bind(1, kind);
            insert.bind(2, path);
            insert.bind(3, value.bytes);
            insert.bind(4, value.modified);
            insert.done();
        }
        Statement recipe(db, "INSERT OR REPLACE INTO metadata(key,value) VALUES(?,?)");
        recipe.bind(1, std::string(kind == terrainKind ? "terrain_recipe" : "static_recipe"));
        recipe.bind(2, recipeVersions[kind]);
        recipe.done();
    }

    [[nodiscard]] std::filesystem::path temporaryPath(const std::filesystem::path& output)
    {
        auto temporary = output;
        temporary += ".tmp." + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count());
        return temporary;
    }

    void atomicReplace(const std::filesystem::path& temporary,
        const std::filesystem::path& output)
    {
#ifdef _WIN32
        if (!MoveFileExW(temporary.c_str(), output.c_str(),
                MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
            throw std::filesystem::filesystem_error("Could not publish LOD tile",
                temporary, output, std::error_code(GetLastError(), std::system_category()));
#else
        std::filesystem::rename(temporary, output);
#endif
    }

    void published(const Database& db, const Key& key, std::uint64_t bytes,
        std::span<const std::uint32_t> members)
    {
        db.exec("BEGIN IMMEDIATE");
        try
        {
            db.exec("UPDATE metadata SET value=value+1 WHERE key='revision'");
            Statement update(db, "UPDATE tiles SET dirty=0, revision=(SELECT value FROM metadata "
                "WHERE key='revision'),bytes=? WHERE kind=? AND level=? AND x=? AND y=?");
            update.bind(1, static_cast<std::int64_t>(bytes));
            update.bind(2, key.kind);
            update.bind(3, key.level);
            update.bind(4, key.x);
            update.bind(5, key.y);
            update.done();
            if (key.kind == staticKind)
            {
                Statement clear(db, "DELETE FROM members WHERE level=? AND x=? AND y=?");
                clear.bind(1, key.level); clear.bind(2, key.x); clear.bind(3, key.y);
                clear.done();
                Statement insert(db,
                    "INSERT INTO members(level,x,y,reference) VALUES(?,?,?,?)");
                for (const std::uint32_t id : members)
                {
                    insert.bind(1, key.level); insert.bind(2, key.x);
                    insert.bind(3, key.y); insert.bind(4, id); insert.done();
                }
            }
            db.exec("COMMIT");
        }
        catch (...) { db.exec("ROLLBACK"); throw; }
    }

    void retired(const Database& db, const Key& key)
    {
        db.exec("BEGIN IMMEDIATE");
        try
        {
            db.exec("UPDATE metadata SET value=value+1 WHERE key='revision'");
            Statement erase(db, "DELETE FROM tiles WHERE kind=? AND level=? AND x=? AND y=?");
            erase.bind(1, key.kind); erase.bind(2, key.level);
            erase.bind(3, key.x); erase.bind(4, key.y); erase.done();
            if (key.kind == staticKind)
            {
                Statement clear(db, "DELETE FROM members WHERE level=? AND x=? AND y=?");
                clear.bind(1, key.level); clear.bind(2, key.x); clear.bind(3, key.y);
                clear.done();
            }
            db.exec("COMMIT");
        }
        catch (...) { db.exec("ROLLBACK"); throw; }
    }
}

LodBuildIndex::LodBuildIndex(std::filesystem::path worldEsm)
    : mWorldEsm(std::filesystem::absolute(std::move(worldEsm)).lexically_normal())
{
}

std::filesystem::path LodBuildIndex::indexPath(const std::filesystem::path& worldEsm)
{
    return AssetResolver::projectAssetRoot(worldEsm) / "lod" / "build.sqlite";
}

std::filesystem::path LodBuildIndex::terrainTilePath(const std::filesystem::path& worldEsm,
    const LodTerrainTile& tile)
{
    return outputPath(worldEsm, terrainKey(tile));
}

std::filesystem::path LodBuildIndex::staticTilePath(const std::filesystem::path& worldEsm,
    const LodStaticTile& tile)
{
    return outputPath(worldEsm, staticKey(tile));
}

LodBakePreflight LodBuildIndex::preflight(const TerrainLodSource& terrain,
    const StaticLodSource& statics, const LodBakeOptions& options) const
{
    const Database db(indexPath(mWorldEsm), false);
    return plan(mWorldEsm, terrain, statics, options, db).summary;
}

LodBakeResult LodBuildIndex::bake(const TerrainLodSource& terrain,
    const StaticLodSource& statics, const LodBakeOptions& options) const
{
    const Database db(indexPath(mWorldEsm), true);
    Plan pending = plan(mWorldEsm, terrain, statics, options, db);
    LodBakeResult result;
    db.exec("BEGIN IMMEDIATE");
    try
    {
        Statement dirty(db, "INSERT INTO tiles(kind,level,x,y,dirty) VALUES(?,?,?,?,1) "
            "ON CONFLICT(kind,level,x,y) DO UPDATE SET dirty=1");
        for (const auto& item : pending.work) markDirty(item.key, dirty);
        if (options.terrain) replaceSources(db, terrainKind, pending.terrainSources);
        if (options.statics) replaceSources(db, staticKind, pending.staticSources);
        db.exec("COMMIT");
    }
    catch (...) { db.exec("ROLLBACK"); throw; }

    std::uint64_t completed = 0;
    for (const auto& item : pending.work)
    {
        if (options.cancelled && options.cancelled())
        {
            result.cancelled = true;
            break;
        }
        const auto output = outputPath(mWorldEsm, item.key);
        if (item.removed)
        {
            std::error_code error;
            std::filesystem::remove(output, error);
            if (error) throw std::filesystem::filesystem_error(
                "Could not retire LOD tile", output, error);
            retired(db, item.key);
            ++result.removedTiles;
            result.publishedPaths.push_back(output);
        }
        else
        {
            std::filesystem::create_directories(output.parent_path());
            const auto temporary = temporaryPath(output);
            try
            {
                std::vector<std::uint32_t> members;
                bool hasGeometry = true;
                if (item.key.kind == terrainKind)
                    terrain.bake({item.key.level, {item.key.x, item.key.y}}, temporary);
                else
                {
                    const LodStaticTile tile{{item.key.x, item.key.y}};
                    hasGeometry = statics.bake(tile, temporary);
                    if (hasGeometry) members = statics.members(tile);
                }
                if (hasGeometry)
                {
                    const auto bytes = std::filesystem::file_size(temporary);
                    atomicReplace(temporary, output);
                    published(db, item.key, bytes, members);
                    ++(item.key.kind == terrainKind ? result.terrainBaked : result.staticBaked);
                }
                else
                {
                    // Remember a skipped tile so an unchanged unresolved model
                    // does not get retried on every changed-only bake.
                    if (std::filesystem::remove(output)) ++result.removedTiles;
                    published(db, item.key, 0, {});
                }
            }
            catch (...)
            {
                std::error_code ignored;
                std::filesystem::remove(temporary, ignored);
                throw;
            }
            result.publishedPaths.push_back(output);
        }
        ++completed;
        if (options.progress)
            options.progress({completed, pending.work.size(), output});
    }
    result.revision = ::openf4::assets::revision(db);
    if (options.statics) result.diagnostics = statics.diagnostics();
    return result;
}

void LodBuildIndex::prepareSave() const
{
    if (!std::filesystem::exists(indexPath(mWorldEsm))) return;
    const Database db(indexPath(mWorldEsm), false, true);
    const auto current = stamp(mWorldEsm);
    Statement dirty(db, "UPDATE tiles SET dirty=1 WHERE kind IN "
        "(SELECT kind FROM sources WHERE path=? AND (bytes<>? OR modified<>?))");
    dirty.bind(1, utf8(mWorldEsm));
    dirty.bind(2, current.bytes);
    dirty.bind(3, current.modified);
    dirty.done();
}

void LodBuildIndex::markSavedChanges(std::span<const CellCoord> terrainCells,
    std::span<const CellCoord> staticCells,
    std::span<const std::filesystem::path> sourceFiles) const
{
    if (!std::filesystem::exists(indexPath(mWorldEsm))) return;
    const Database db(indexPath(mWorldEsm), false, true);
    db.exec("BEGIN IMMEDIATE");
    try
    {
        Statement dirty(db, "INSERT INTO tiles(kind,level,x,y,dirty) VALUES(?,?,?,?,1) "
            "ON CONFLICT(kind,level,x,y) DO UPDATE SET dirty=1");
        for (const CellCoord cell : terrainCells)
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx)
                    for (const int level : {4, 8, 16, 32})
                    {
                        const auto nearbyX = static_cast<std::int64_t>(cell.x) + dx;
                        const auto nearbyY = static_cast<std::int64_t>(cell.y) + dy;
                        if (!std::in_range<int>(nearbyX) || !std::in_range<int>(nearbyY)) continue;
                        const auto x = static_cast<int>(nearbyX);
                        const auto y = static_cast<int>(nearbyY);
                        markDirty({terrainKind, level,
                            floorDivide(x, level) * level,
                            floorDivide(y, level) * level}, dirty);
                    }
        for (const CellCoord cell : staticCells)
            markDirty({staticKind, 4,
                floorDivide(cell.x, 4) * 4,
                floorDivide(cell.y, 4) * 4}, dirty);
        Statement update(db, "INSERT INTO sources(kind,path,bytes,modified) VALUES(?,?,?,?) "
            "ON CONFLICT(kind,path) DO UPDATE SET bytes=excluded.bytes,"
            "modified=excluded.modified");
        std::vector<std::filesystem::path> changed(sourceFiles.begin(), sourceFiles.end());
        changed.push_back(mWorldEsm);
        for (const auto& path : changed)
        {
            const auto absolute = std::filesystem::absolute(path).lexically_normal();
            const auto value = stamp(absolute);
            for (const int kind : {terrainKind, staticKind})
            {
                update.bind(1, kind); update.bind(2, utf8(absolute));
                update.bind(3, value.bytes); update.bind(4, value.modified);
                update.done();
            }
        }
        db.exec("COMMIT");
    }
    catch (...) { db.exec("ROLLBACK"); throw; }
}

std::uint64_t LodBuildIndex::revision() const
{
    const Database db(indexPath(mWorldEsm), false);
    return ::openf4::assets::revision(db);
}

std::vector<LodTerrainTile> LodBuildIndex::expectedTerrainTiles() const
{
    const Database db(indexPath(mWorldEsm), false);
    std::vector<LodTerrainTile> result;
    if (!db.handle) return result;
    Statement tiles(db, "SELECT level,x,y FROM tiles WHERE kind=0 ORDER BY level,x,y");
    while (tiles.step() == SQLITE_ROW)
        result.push_back({sqlite3_column_int(tiles.handle, 0),
            {sqlite3_column_int(tiles.handle, 1), sqlite3_column_int(tiles.handle, 2)}});
    return result;
}
}
