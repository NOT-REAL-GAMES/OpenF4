#include "assets/fo4_export.hpp"

#include "assets/asset_resolver.hpp"
#include "assets/fo4_asset_package.hpp"
#include "assets/fo4_world_writer.hpp"
#include "script/papyrus_vm.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>

#ifdef _WIN32
#define NOMINMAX
#include <Windows.h>
#include <bcrypt.h>
#endif

namespace openf4::assets
{
namespace
{
    void checkpoint(const Fo4ExportOptions& options)
    {
        if (options.cancelled && options.cancelled()) throw std::runtime_error("Fallout export cancelled");
    }

    std::string folded(std::string value)
    {
        for (char& c : value) if (c >= 'A' && c <= 'Z') c += 'a' - 'A';
        return value;
    }

    std::string exportName(const std::filesystem::path& path)
    {
        std::string result;
        for (const unsigned char c : path.stem().string())
            result += (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
                || (c >= '0' && c <= '9') || c == '_' ? static_cast<char>(c) : '_';
        if (result.empty() || result.size() > 64) throw std::runtime_error("Export name must contain 1–64 characters");
        return "OF4_" + result;
    }

    void noLinks(const std::filesystem::path& path)
    {
        for (auto p = path; !p.empty(); p = p.parent_path())
        {
            std::error_code error;
            const auto state = std::filesystem::symlink_status(p, error);
            if (error && error != std::errc::no_such_file_or_directory)
                throw std::runtime_error("Cannot inspect export path: " + p.string());
            if (std::filesystem::is_symlink(state)) throw std::runtime_error("Export paths must not traverse symbolic links");
#ifdef _WIN32
            const auto attributes = GetFileAttributesW(p.c_str());
            if (attributes != INVALID_FILE_ATTRIBUTES && (attributes & FILE_ATTRIBUTE_REPARSE_POINT))
                throw std::runtime_error("Export paths must not traverse reparse points: " + p.string());
#endif
            if (p == p.parent_path()) break;
        }
    }

    std::string sha256(const std::filesystem::path& path, const Fo4ExportOptions& options)
    {
        noLinks(path);
        std::ifstream input(path, std::ios::binary);
        if (!input) throw std::runtime_error("Cannot fingerprint export input: " + path.string());
#ifdef _WIN32
        struct Hash
        {
            BCRYPT_ALG_HANDLE algorithm = nullptr;
            BCRYPT_HASH_HANDLE handle = nullptr;
            std::vector<unsigned char> object;
            ~Hash() { if (handle) BCryptDestroyHash(handle); if (algorithm) BCryptCloseAlgorithmProvider(algorithm, 0); }
        } hash;
        const auto check = [](NTSTATUS result) { if (result < 0) throw std::runtime_error("SHA-256 operation failed"); };
        check(BCryptOpenAlgorithmProvider(&hash.algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0));
        ULONG size = 0, count = 0;
        check(BCryptGetProperty(hash.algorithm, BCRYPT_OBJECT_LENGTH,
            reinterpret_cast<PUCHAR>(&size), sizeof(size), &count, 0));
        hash.object.resize(size);
        check(BCryptCreateHash(hash.algorithm, &hash.handle, hash.object.data(), size, nullptr, 0, 0));
        std::array<unsigned char, 65536> buffer{};
        while (input)
        {
            checkpoint(options);
            input.read(reinterpret_cast<char*>(buffer.data()), buffer.size());
            if (input.gcount()) check(BCryptHashData(hash.handle, buffer.data(), static_cast<ULONG>(input.gcount()), 0));
        }
        if (!input.eof()) throw std::runtime_error("Could not fingerprint complete file: " + path.string());
        std::array<unsigned char, 32> digest{};
        check(BCryptFinishHash(hash.handle, digest.data(), static_cast<ULONG>(digest.size()), 0));
        std::ostringstream out; out << std::hex << std::setfill('0');
        for (const auto byte : digest) out << std::setw(2) << static_cast<unsigned>(byte);
        return out.str();
#else
        (void)options;
        throw std::runtime_error("Fallout proof package hashing currently requires Windows");
#endif
    }

    template<class T> T value(std::span<const std::byte> bytes, std::size_t offset)
    {
        if (offset > bytes.size() || sizeof(T) > bytes.size() - offset) throw std::runtime_error("Truncated export source");
        T result; std::memcpy(&result, bytes.data() + offset, sizeof(result)); return result;
    }

    // Reject data the proof writer cannot represent; never silently flatten a foreign world.
    void validateProofSource(const std::filesystem::path& path, float& waterHeight,
        std::set<std::uint32_t>& waterCells, const Fo4ExportOptions& options)
    {
        const std::map<std::string, std::set<std::string>> allowed{
            {"TES4", {"HEDR", "CNAM", "SNAM", "MAST", "DATA"}},
            {"WRLD", {"EDID", "DATA", "NAM4"}},
            {"CELL", {"EDID", "DATA", "XCLC", "XCLW"}},
            {"LAND", {"VHGT"}},
            {"STAT", {"EDID", "OBND", "MODL", "ESRC", "DNAM", "VMAD"}},
            {"MSTT", {"EDID", "OBND", "FULL", "MODL", "ESRC", "DATA", "VMAD"}},
            {"REFR", {"EDID", "NAME", "DATA", "XSCL", "XMSP", "VMAD"}}};
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file || file.tellg() < 24) throw std::runtime_error("Missing export source ESM");
        const auto size = static_cast<std::uint64_t>(file.tellg());
        const auto read = [&](std::uint64_t offset, std::size_t length) {
            std::vector<std::byte> bytes(length);
            file.clear(); file.seekg(static_cast<std::streamoff>(offset));
            if (!file.read(reinterpret_cast<char*>(bytes.data()), length)) throw std::runtime_error("Cannot read export source");
            return bytes;
        };
        struct Context
        {
            int group = -1;
            std::string top;
            std::uint32_t world = 0, cell = 0, block = 0, subblock = 0;
        };
        struct CellState { CellCoord coord; std::uint32_t block, subblock; bool land = false; };
        std::map<std::uint32_t, CellState> cells;
        std::set<CellCoord> coordinates;
        std::set<std::uint32_t> ids;
        std::uint32_t activeWorld = 0;
        bool headerSeen = false, worldChildrenSeen = false;
        const auto packed = [](CellCoord coord, int divisor) {
            const auto divide = [=](std::int32_t n) { return n / divisor - (n % divisor < 0); };
            return std::uint32_t(std::uint16_t(divide(coord.x))) | (std::uint32_t(std::uint16_t(divide(coord.y))) << 16);
        };
        const auto unsupported = [](const std::string& detail) { throw std::runtime_error("World proof cannot preserve " + detail); };
        std::function<void(std::uint64_t, std::uint64_t, unsigned, Context)> scan = [&](auto begin, auto end, unsigned depth, Context context) {
            if (depth > 16) throw std::runtime_error("Unsupported export group depth");
            std::set<std::pair<std::uint32_t, std::uint32_t>> siblingGroups;
            for (auto offset = begin; offset < end;)
            {
                checkpoint(options);
                if (end - offset < 24) throw std::runtime_error("Truncated export record header");
                const auto header = read(offset, 24);
                const std::string type(reinterpret_cast<const char*>(header.data()), 4);
                const auto payloadSize = value<std::uint32_t>(header, 4);
                const auto length = std::uint64_t(payloadSize) + (type == "GRUP" ? 0 : 24);
                if (length < 24 || length > end - offset) throw std::runtime_error("Export record exceeds its parent");
                if (type == "GRUP")
                {
                    const auto label = value<std::uint32_t>(header, 8), group = value<std::uint32_t>(header, 12);
                    if (!siblingGroups.emplace(group, label).second) unsupported("duplicate source groups");
                    auto nested = context; nested.group = static_cast<int>(group);
                    if (group == 0 && context.group == -1 && headerSeen)
                    {
                        nested.top = std::string(reinterpret_cast<const char*>(header.data() + 8), 4);
                        if (nested.top != "WRLD" && nested.top != "STAT" && nested.top != "MSTT") unsupported("source top group " + nested.top);
                    }
                    else if (group == 1 && context.group == 0 && context.top == "WRLD"
                        && activeWorld && label == activeWorld && !worldChildrenSeen)
                    { nested.world = label; worldChildrenSeen = true; }
                    else if (group == 4 && context.group == 1) nested.block = label;
                    else if (group == 5 && context.group == 4) nested.subblock = label;
                    else if (group == 6 && context.group == 5 && cells.contains(label)
                        && cells.at(label).block == context.block && cells.at(label).subblock == context.subblock) nested.cell = label;
                    else if (!(group == 9 && context.group == 6 && label == context.cell)) unsupported("source group nesting or CELL ownership");
                    scan(offset + 24, offset + length, depth + 1, nested);
                }
                else
                {
                    if (!allowed.contains(type)) throw std::runtime_error("World proof cannot preserve record type " + type);
                    const auto id = value<std::uint32_t>(header, 12);
                    if (type == "TES4")
                    {
                        if (context.group != -1 || offset != 0 || headerSeen || id) unsupported("multiple or misplaced TES4 records");
                        headerSeen = true;
                    }
                    else
                    {
                        if ((id >> 24) != 1 || (id & 0xffffff) < 0x800 || !ids.insert(id).second)
                            unsupported("nonlocal or duplicate source FormIDs");
                        if ((type == "WRLD" || type == "STAT" || type == "MSTT")
                            ? context.group != 0 || context.top != type
                            : type == "CELL" ? context.group != 5 || context.world != activeWorld
                            : context.group != 9 || !cells.contains(context.cell)) unsupported("misplaced " + type + " record");
                        if (type == "WRLD")
                        {
                            if (activeWorld) unsupported("multiple WRLD records");
                            activeWorld = id;
                        }
                    }
                    const auto flags = value<std::uint32_t>(header, 8);
                    const auto allowedFlags = type == "TES4" ? 1u : type == "REFR" ? 0x20u : 0u;
                    if (flags & ~allowedFlags) throw std::runtime_error("World proof cannot preserve flags on " + type);
                    if (payloadSize > 1024 * 1024) throw std::runtime_error("World proof record payload exceeds supported size");
                    const auto body = read(offset + 24, payloadSize);
                    std::map<std::string, std::span<const std::byte>> fields;
                    for (std::size_t p = 0; p < body.size();)
                    {
                        if (body.size() - p < 6) throw std::runtime_error("Truncated export subrecord");
                        const std::string field(reinterpret_cast<const char*>(body.data() + p), 4);
                        const auto count = value<std::uint16_t>(body, p + 4); p += 6;
                        if (count > body.size() - p) throw std::runtime_error("Export subrecord exceeds its record");
                        if (!allowed.at(type).contains(field))
                            throw std::runtime_error("World proof cannot preserve " + type + "." + field);
                        if (!fields.emplace(field, std::span<const std::byte>(body).subspan(p, count)).second)
                            unsupported("duplicate " + type + "." + field);
                        p += count;
                    }
                    const auto field = [&](const std::string& name, std::size_t expected) {
                        const auto found = fields.find(name);
                        if (found == fields.end() || found->second.size() != expected) unsupported("missing or malformed " + type + "." + name);
                        return found->second;
                    };
                    const auto text = [&](const std::string& name) {
                        const auto found = fields.find(name);
                        if (found == fields.end() || found->second.empty() || found->second.back() != std::byte{}
                            || std::find(found->second.begin(), found->second.end() - 1, std::byte{}) != found->second.end() - 1)
                            unsupported("missing or malformed " + type + "." + name);
                        return std::string(reinterpret_cast<const char*>(found->second.data()), found->second.size() - 1);
                    };
                    if (fields.contains("VMAD"))
                    {
                        const auto binding = decodePapyrusBindings(fields.at("VMAD"), path.string());
                        if (!binding.editable) unsupported(binding.diagnostic);
                        (void)encodePapyrusBindings(binding, false);
                    }
                    if (type == "TES4")
                    {
                        const auto h = field("HEDR", 12);
                        if (value<float>(h, 0) != .95f || folded(text("MAST")) != "fallout4.esm"
                            || value<std::uint64_t>(field("DATA", 8), 0) != 0) unsupported("unsupported TES4 schema or master data");
                        const auto author = text("CNAM"); if (author != "OpenF4" && author != "EMPIRE") unsupported("foreign source author");
                        if (fields.contains("SNAM")) (void)text("SNAM");
                    }
                    else if (type == "WRLD")
                    {
                        if (text("EDID").empty()) unsupported("empty WRLD editor ID");
                        if (fields.contains("DATA") && value<std::uint8_t>(field("DATA", 1), 0) != 1) unsupported("WRLD.DATA flags");
                        if (fields.contains("NAM4")) waterHeight = value<float>(field("NAM4", 4), 0);
                        if (!std::isfinite(waterHeight)) unsupported("non-finite world water height");
                    }
                    else if (type == "CELL")
                    {
                        if (fields.contains("EDID")) (void)text("EDID");
                        const auto data = value<std::uint16_t>(field("DATA", 2), 0);
                        if (data & ~2u) unsupported("CELL.DATA flags (including interior cells)");
                        const auto location = field("XCLC", 12);
                        const CellCoord coord{value<std::int32_t>(location, 0), value<std::int32_t>(location, 4)};
                        if (value<std::uint32_t>(location, 8) || coord.x < INT16_MIN || coord.x > INT16_MAX
                            || coord.y < INT16_MIN || coord.y > INT16_MAX) unsupported("CELL.XCLC flags or coordinate range");
                        if (!coordinates.insert(coord).second) unsupported("duplicate exterior CELL coordinates");
                        if (packed(coord, 32) != context.block || packed(coord, 8) != context.subblock) unsupported("exterior block coordinate ownership");
                        if (fields.contains("XCLW") && (value<float>(field("XCLW", 4), 0) != waterHeight || !(data & 2u)))
                            unsupported("per-cell water overrides");
                        if (data & 2u) waterCells.insert(id);
                        cells.emplace(id, CellState{coord, context.block, context.subblock});
                    }
                    else if (type == "LAND")
                    {
                        auto& owner = cells.at(context.cell);
                        if (owner.land) unsupported("multiple LAND records in one CELL");
                        owner.land = true;
                        const auto heights = field("VHGT", 1096); const float base = value<float>(heights, 0);
                        if (!std::isfinite(base) || std::round(base) != base
                            || heights[1093] != std::byte{} || heights[1094] != std::byte{} || heights[1095] != std::byte{})
                            unsupported("malformed LAND.VHGT");
                    }
                    else if (type == "STAT" || type == "MSTT")
                    {
                        const auto name = text("EDID"); if (name.empty() || text("MODL").empty()) unsupported("empty native model names");
                        if (fields.contains("ESRC")) (void)text("ESRC");
                        const auto bounds = field("OBND", 12);
                        for (unsigned axis = 0; axis < 3; ++axis)
                            if (value<std::int16_t>(bounds, axis * 2) > value<std::int16_t>(bounds, (axis + 3) * 2)) unsupported("reversed model bounds");
                        if (type == "STAT")
                        {
                            const auto d = field("DNAM", 16);
                            if (value<float>(d, 0) != 90 || value<std::uint32_t>(d, 4) || value<std::uint32_t>(d, 8) || value<std::uint32_t>(d, 12))
                                unsupported("nondefault STAT.DNAM");
                        }
                        else if (value<std::uint8_t>(field("DATA", 1), 0) != 1 || text("FULL") != name)
                            unsupported("MSTT.DATA or custom display name");
                    }
                    else if (type == "REFR")
                    {
                        if (fields.contains("EDID")) (void)text("EDID");
                        if (!value<std::uint32_t>(field("NAME", 4), 0)) unsupported("empty reference base");
                        const auto transform = field("DATA", 24);
                        for (unsigned axis = 0; axis < 6; ++axis)
                            if (!std::isfinite(value<float>(transform, axis * 4))) unsupported("non-finite reference transform");
                        const auto coord = cells.at(context.cell).coord;
                        if (std::floor(double(value<float>(transform, 0)) / 4096) != coord.x
                            || std::floor(double(value<float>(transform, 4)) / 4096) != coord.y) unsupported("reference CELL position");
                        if (fields.contains("XSCL"))
                        {
                            const auto scale = value<float>(field("XSCL", 4), 0);
                            if (!std::isfinite(scale) || scale <= 0) unsupported("reference scale");
                        }
                        if (fields.contains("XMSP") && value<std::uint32_t>(field("XMSP", 4), 0)) unsupported("reference material swaps");
                    }
                }
                offset += length;
            }
        };
        scan(0, size, 0, {});
        if (!headerSeen || !activeWorld || !worldChildrenSeen || cells.empty()) unsupported("incomplete source world hierarchy");
        for (const auto& [id, cell] : cells) if (!cell.land) unsupported("CELL without LAND");
        if (!std::isfinite(waterHeight)) throw std::runtime_error("Source water height is not finite");
    }

    struct Source
    {
        std::string initialDigest;
        EsmDocument document;
        Fo4ExportPreflight summary;
        Fo4WorldWriteRequest request;
        std::map<std::uint32_t, Fo4StagedModel> assets;
        std::map<std::filesystem::path, std::string> fingerprints;
        struct ScriptFile
        {
            std::string path;
            AssetBytes asset;
            script::PexInfo info;
        };
        std::map<std::string, ScriptFile> scripts;

        Source(const std::filesystem::path& world, const Fo4ExportOptions& options)
            : initialDigest(sha256(world, options)), document(world)
        {
            summary.name = exportName(world);
            summary.limitations = {"World geometry proof only: native NPC navigation is not generated.",
                "Native distant terrain/static LOD is not generated.",
                "Existing Fallout overrides, large-world partitioning and game compatibility are not certified."};
            summary.cells = document.terrainCells().size();
            summary.references = std::ranges::count_if(document.references(), [](const auto& r) { return !r.deleted && !r.disabled; });
            summary.estimatedBytes = summary.cells * 8192 + summary.references * 128 + 4096;
            if (!document.isOpenF4Owned() || document.isDemOverlay() || document.masters().size() != 1
                || folded(document.masters()[0]) != "fallout4.esm" || document.worldFormId() >> 24 != 1)
                throw std::runtime_error("World proof currently requires a new OpenF4 world with only Fallout4.esm as master; overlays/DEM require later compatibility gates");
            if (!summary.cells || summary.estimatedBytes >= 2ull * 1024 * 1024 * 1024)
                throw std::runtime_error("World proof requires LAND and cannot exceed the 2 GiB native-group safety budget; partitioned export is not yet available");
            fingerprints.emplace(document.path(), initialDigest);
            if (options.dataDirectory && std::filesystem::is_regular_file(*options.dataDirectory / "Fallout4.esm"))
                fingerprints.emplace(*options.dataDirectory / "Fallout4.esm", sha256(*options.dataDirectory / "Fallout4.esm", options));
            else
                summary.limitations.push_back("Installed Fallout4.esm was not fingerprinted; verify the required master before a game test.");
            request.editorId = summary.name; request.displayName = world.stem().string();
            request.worldFormId = document.worldFormId(); request.waterHeight = -2048;
            std::set<std::uint32_t> waterCells;
            validateProofSource(world, request.waterHeight, waterCells, options);
            auto next = document.nextFormId();
            if ((next >> 24) != 1 || (next & 0xFFFFFF) > 0xFFFFFD) throw std::runtime_error("Insufficient FormIDs for native world ownership and arrival");
            request.persistentCellFormId = next++;
            request.arrival.placement.formId = next;
            request.arrival.placement.baseObject = fo4ArrivalMarker;
            request.arrival.editorId = summary.name + "_Arrival";
            for (const auto& [coord, cell] : document.terrainCells())
            {
                if (!cell.landFormId) throw std::runtime_error("Export CELL is missing LAND");
                request.cells.push_back({coord, cell.cellFormId, cell.landFormId, waterCells.contains(cell.cellFormId)});
            }
            std::map<std::uint32_t, ESMModel> models;
            for (const auto& model : document.models()) models.emplace(model.formId, model);
            std::set<std::uint32_t> needed;
            for (const auto& reference : document.references())
            {
                if (reference.deleted || reference.disabled) continue;
                if (reference.placement.actor)
                    throw std::runtime_error("Actors (ACHR) are not exported by the world exporter yet; "
                        "author them in an ESP over Fallout4.esm instead (" + reference.editorId + ")");
                if (!models.contains(reference.placement.baseObject)) throw std::runtime_error("Reference has a missing or unsupported inherited model base");
                if (!document.terrainCells().contains(reference.placement.cell)) throw std::runtime_error("Reference is outside exported terrain");
                needed.insert(reference.placement.baseObject); request.references.push_back(reference);
                request.references.back().editorId = summary.name + "_Ref_" + std::to_string(reference.placement.formId);
            }
            summary.models = needed.size();
            std::map<std::string, std::vector<std::byte>> assetPaths;
            std::set<std::string> projectPaths;
            for (const auto id : needed)
            {
                checkpoint(options);
                auto model = models.at(id); model.sourceEsm = document.path().string();
                auto info = inspectFo4ModelAsset(model, options.dataDirectory);
                for (const auto& dependency : info.dependencies)
                {
                    if (dependency.resolvedPath.empty())
                        throw std::runtime_error("Export dependency has no source provenance: " + dependency.dataPath);
                    if (!fingerprints.contains(dependency.resolvedPath))
                        fingerprints[dependency.resolvedPath] = sha256(dependency.resolvedPath, options);
                }
                AssetResolver resolver(model.sourceEsm, options.dataDirectory);
                std::vector<std::vector<std::byte>> snapshot;
                snapshot.reserve(info.dependencies.size());
                for (const auto& dependency : info.dependencies)
                {
                    checkpoint(options);
                    auto bytes = resolver.read(dependency.dataPath);
                    if (bytes.source != dependency.source || bytes.resolvedPath != dependency.resolvedPath
                        || bytes.bytes.size() != dependency.bytes)
                        throw std::runtime_error("Export dependency changed during inspection: " + dependency.dataPath);
                    snapshot.push_back(std::move(bytes.bytes));
                }
                // Inspect only after fingerprinting, so metadata and staged bytes share the checked source snapshot.
                auto checked = inspectFo4ModelAsset(model, options.dataDirectory);
                if (checked.dependencies.size() != info.dependencies.size())
                    throw std::runtime_error("Export dependency closure changed during inspection: " + model.path);
                for (std::size_t i = 0; i < checked.dependencies.size(); ++i)
                {
                    checkpoint(options);
                    const auto& dependency = checked.dependencies[i];
                    const auto& previous = info.dependencies[i];
                    if (dependency.dataPath != previous.dataPath || dependency.source != previous.source
                        || dependency.resolvedPath != previous.resolvedPath || dependency.bytes != previous.bytes)
                        throw std::runtime_error("Export dependency changed during inspection: " + dependency.dataPath);
                    auto bytes = resolver.read(dependency.dataPath);
                    if (bytes.source != dependency.source || bytes.resolvedPath != dependency.resolvedPath
                        || bytes.bytes != snapshot[i])
                        throw std::runtime_error("Export dependency bytes changed during inspection: " + dependency.dataPath);
                    const auto key = folded(dependency.dataPath);
                    if (const auto prior = assetPaths.find(key); prior != assetPaths.end() && prior->second != bytes.bytes)
                        throw std::runtime_error("Conflicting Data dependency: " + dependency.dataPath);
                    if (dependency.source == AssetSource::project && projectPaths.insert(key).second)
                        summary.estimatedBytes += dependency.bytes;
                    assetPaths.try_emplace(key, std::move(bytes.bytes));
                }
                info = std::move(checked);
                for (const auto& diagnostic : info.diagnostics)
                    summary.limitations.push_back("Model " + std::to_string(id) + " (" + model.path + "): " + diagnostic);
                request.models.push_back({id, model.kind, summary.name + "_Model_" + std::to_string(id),
                    "meshes/" + info.modelPath, info.bounds, model.editorId, model.scripts});
                assets.emplace(id, std::move(info));
                if (options.progress) options.progress({"Inspecting native assets", assets.size(), needed.size()});
            }
            request.loadCell = [this, &options](CellCoord coord) -> std::optional<LandscapeCell> {
                checkpoint(options);
                const auto cell = document.terrainCell(coord);
                return cell ? std::optional(cell->landscape) : std::nullopt;
            };
            chooseArrival(options.arrivalCell);
            inspectScripts(options);
            verifyInputs(options);
        }

        void inspectScripts(const Fo4ExportOptions& options)
        {
            std::vector<AssetResolver> projects;
            projects.emplace_back(document.path());
            // The current proof gate permits only Fallout4.esm; keep the same winning-sidecar order as runtime.
            for (auto master = document.masters().rbegin(); master != document.masters().rend(); ++master)
            {
                auto path = document.path().parent_path() / *master;
                if (!std::filesystem::is_regular_file(path) && options.dataDirectory) path = *options.dataDirectory / *master;
                projects.emplace_back(path);
            }
            AssetResolver installed(document.path(), options.dataDirectory);
            const auto readScript = [&](const std::string& path) {
                for (const auto& project : projects) if (project.contains(path)) return project.read(path);
                return installed.read(path);
            };
            std::set<std::uint32_t> exported{request.worldFormId, request.persistentCellFormId, request.arrival.placement.formId};
            for (const auto& cell : request.cells) { exported.insert(cell.cellFormId); exported.insert(cell.landFormId); }
            for (const auto& model : request.models) exported.insert(model.formId);
            for (const auto& reference : request.references) exported.insert(reference.placement.formId);
            std::function<const script::PexInfo&(const std::string&)> load = [&](const std::string& name) -> const script::PexInfo& {
                checkpoint(options);
                const auto key = folded(name);
                if (const auto found = scripts.find(key); found != scripts.end()) return found->second.info;
                if (scripts.size() >= 1024) throw std::runtime_error("Script dependency closure exceeds 1024 files");
                const auto path = papyrusScriptPath(name);
                auto asset = readScript(path);
                if (asset.bytes.size() > 16 * 1024 * 1024) throw std::runtime_error("PEX exceeds supported size: " + name);
                auto info = script::inspectPex({reinterpret_cast<const std::uint8_t*>(asset.bytes.data()), asset.bytes.size()});
                if (folded(info.name) != key) throw std::runtime_error("PEX class name differs from its path: " + path);
                if (!fingerprints.contains(asset.resolvedPath)) fingerprints.emplace(asset.resolvedPath, sha256(asset.resolvedPath, options));
                const auto checked = readScript(path);
                if (checked.resolvedPath != asset.resolvedPath || checked.source != asset.source || checked.bytes != asset.bytes)
                    throw std::runtime_error("Script dependency changed during inspection: " + path);
                const bool project = asset.source == AssetSource::project;
                if (project)
                {
                    noLinks(asset.resolvedPath);
                    summary.estimatedBytes += asset.bytes.size();
                }
                auto [found, inserted] = scripts.emplace(key, ScriptFile{path, std::move(asset), std::move(info)});
                // Installed scripts and their dependencies remain supplied by the game's own archives.
                if (project) for (const auto& dependency : found->second.info.dependencies)
                    if (folded(dependency) != key) (void)load(dependency);
                return found->second.info;
            };
            const auto validateBindings = [&](const PapyrusBindings& bindings) {
                if (!bindings.editable) throw std::runtime_error("Cannot export unsupported VMAD: " + bindings.diagnostic);
                for (const auto& binding : bindings.scripts)
                {
                    if (binding.flags & 2u) continue;
                    const auto& info = load(binding.name);
                    std::map<std::string, std::string> properties;
                    std::set<std::string> parents;
                    const auto* current = &info;
                    while (current)
                    {
                        if (!parents.insert(folded(current->name)).second) throw std::runtime_error("PEX inheritance cycle: " + binding.name);
                        for (const auto& [name, type] : current->properties) properties.try_emplace(folded(name), folded(type));
                        if (current->parent.empty()) break;
                        current = &load(current->parent);
                    }
                    for (const auto& property : binding.properties)
                    {
                        const auto found = properties.find(folded(property.name));
                        if (found == properties.end()) throw std::runtime_error("VMAD property is missing from PEX: " + binding.name + "." + property.name);
                        if (property.flags == 3) continue;
                        const auto& expected = found->second;
                        const auto type = property.value.index();
                        const bool scalar = (type == 2 && expected == "string") || (type == 3 && expected == "int")
                            || (type == 4 && expected == "float") || (type == 5 && expected == "bool");
                        const bool object = (type == 0 || type == 1) && expected != "string" && expected != "int"
                            && expected != "float" && expected != "bool" && expected != "none" && expected != "var"
                            && expected.find('[') == std::string::npos && expected.find('#') == std::string::npos;
                        if (!scalar && !object) throw std::runtime_error("VMAD/PEX property type mismatch: " + binding.name + "." + property.name);
                        if (const auto* target = std::get_if<PapyrusObject>(&property.value))
                        {
                            if (target->alias != -1) throw std::runtime_error("Quest alias script properties are not supported by world export");
                            const auto slot = target->formId >> 24;
                            if (slot > 1 || (slot == 1 && !exported.contains(target->formId)))
                                throw std::runtime_error("VMAD object property targets a record outside the export");
                        }
                    }
                }
            };
            for (const auto& model : request.models) validateBindings(model.scripts);
            for (const auto& reference : request.references) validateBindings(reference.scripts);
        }

        void verifyInputs(const Fo4ExportOptions& options) const
        {
            for (const auto& [path, expected] : fingerprints)
                if (sha256(path, options) != expected) throw std::runtime_error("Export source changed during build: " + path.string());
        }

        void chooseArrival(std::optional<CellCoord> selected)
        {
            CellCoord coord = selected.value_or(CellCoord{});
            if (!document.terrainCells().contains(coord))
            {
                if (selected) throw std::runtime_error("Arrival CELL is not part of the exported landscape");
                const auto nearest = std::ranges::min_element(request.cells, {}, [](const auto& c) {
                    return std::int64_t(c.coord.x) * c.coord.x + std::int64_t(c.coord.y) * c.coord.y;
                });
                coord = nearest->coord;
            }
            const auto terrain = document.terrainCell(coord);
            if (!terrain || !terrain->landscape.hasHeights) throw std::runtime_error("Arrival CELL has no terrain");
            const bool hasWater = std::ranges::find(request.cells, coord, &Fo4WorldCell::coord)->hasWater;
            // Conservative spheres avoid treating a rotated object's local OBND as a world AABB.
            for (const int y : {16, 8, 24, 4, 28}) for (const int x : {16, 8, 24, 4, 28})
            {
                const auto& h = terrain->landscape.heights;
                const float z = h[y * 33 + x];
                if (!std::isfinite(z) || (hasWater && z <= request.waterHeight + 32)) continue;
                const float lo = std::min({z, h[(y - 1) * 33 + x], h[(y + 1) * 33 + x], h[y * 33 + x - 1], h[y * 33 + x + 1]});
                const float hi = std::max({z, h[(y - 1) * 33 + x], h[(y + 1) * 33 + x], h[y * 33 + x - 1], h[y * 33 + x + 1]});
                if (hi - lo > 32) continue;
                const glm::vec3 point(float(double(coord.x) * 4096 + x * 128), float(double(coord.y) * 4096 + y * 128), z + 64);
                bool occupied = false;
                for (const auto& reference : request.references)
                {
                    const auto& p = reference.placement;
                    const auto& info = assets.at(p.baseObject);
                    const float radius = glm::length(glm::max(glm::abs(info.bethesdaMinimum), glm::abs(info.bethesdaMaximum))) * p.scale;
                    occupied |= glm::length(point - glm::vec3(p.position[0], p.position[1], p.position[2])) < radius + 160;
                }
                if (occupied) continue;
                request.arrival.placement.cell = coord;
                for (int i = 0; i < 3; ++i) request.arrival.placement.position[i] = point[i];
                return;
            }
            throw std::runtime_error("No dry, level, unobstructed arrival found in the selected CELL; choose another cell");
        }
    };

    void writeText(const std::filesystem::path& path, const std::string& text)
    {
        std::ofstream file(path, std::ios::binary);
        if (!file || !file.write(text.data(), text.size()) || !file.flush())
            throw std::runtime_error("Cannot write export report: " + path.string());
    }
}

Fo4ExportPreflight inspectFo4Export(const std::filesystem::path& world, const Fo4ExportOptions& options)
{
    try { return Source(world, options).summary; }
    catch (const std::exception& error)
    {
        checkpoint(options);
        Fo4ExportPreflight result; result.blockers.push_back(error.what()); return result;
    }
}

Fo4ExportResult exportFo4WorldProof(const std::filesystem::path& world,
    const std::filesystem::path& destination, const Fo4ExportOptions& options)
{
    checkpoint(options);
    if (destination.empty()) throw std::runtime_error("Export destination must be a new package directory");
    const auto target = std::filesystem::absolute(destination).lexically_normal();
    noLinks(target);
    const auto inside = [&](const std::filesystem::path& directory) {
        auto root = folded(std::filesystem::absolute(directory).lexically_normal().generic_string());
        const auto candidate = folded(target.generic_string());
        return candidate == root || candidate.starts_with(root + '/');
    };
    if (inside(AssetResolver::projectAssetRoot(world)) || (options.dataDirectory && inside(*options.dataDirectory)))
        throw std::runtime_error("Export to a separate package directory, not authored assets or Fallout 4 Data");
    if (std::filesystem::exists(target)) throw std::runtime_error("Export destination already exists; choose a new package directory");
    Source source(world, options);
    const auto token = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    const auto temporary = target.parent_path() / (target.filename().string() + ".incomplete-" + token);
    std::filesystem::create_directories(target.parent_path());
    if (!std::filesystem::create_directory(temporary)) throw std::runtime_error("Could not reserve export staging directory");
    const auto data = temporary / "Data";
    try
    {
        std::filesystem::create_directory(data);
        std::size_t done = 0;
        for (const auto& model : source.document.models())
        {
            if (!source.assets.contains(model.formId)) continue;
            checkpoint(options);
            auto owned = model; owned.sourceEsm = source.document.path().string();
            const auto staged = stageFo4ModelAsset(owned, data, options.dataDirectory);
            const auto& expected = source.assets.at(model.formId);
            const bool sameDependencies = std::ranges::equal(staged.dependencies, expected.dependencies,
                [](const Fo4AssetDependency& a, const Fo4AssetDependency& b) {
                    return a.dataPath == b.dataPath && a.resolvedPath == b.resolvedPath
                        && a.source == b.source && a.bytes == b.bytes;
                });
            if (staged.modelPath != expected.modelPath || staged.bounds != expected.bounds
                || staged.bethesdaMinimum != expected.bethesdaMinimum || staged.bethesdaMaximum != expected.bethesdaMaximum
                || staged.copiedFiles != expected.copiedFiles || staged.stockDependencies != expected.stockDependencies
                || staged.diagnostics != expected.diagnostics || !sameDependencies)
                throw std::runtime_error("Model package changed after preflight");
            for (const auto& dependency : expected.dependencies)
                if (dependency.source == AssetSource::project
                    && sha256(data / dependency.dataPath, options) != source.fingerprints.at(dependency.resolvedPath))
                    throw std::runtime_error("Staged asset differs from its inspected source: " + dependency.dataPath);
            if (options.progress) options.progress({"Packaging native assets", ++done, source.assets.size()});
        }
        for (const auto& [name, script] : source.scripts)
        {
            checkpoint(options);
            if (script.asset.source != AssetSource::project) continue;
            const auto targetScript = data / script.path;
            noLinks(targetScript);
            std::filesystem::create_directories(targetScript.parent_path());
            noLinks(targetScript);
            if (std::filesystem::exists(targetScript)) throw std::runtime_error("Conflicting script export path: " + script.path);
            std::ofstream output(targetScript, std::ios::binary);
            output.write(reinterpret_cast<const char*>(script.asset.bytes.data()), script.asset.bytes.size()); output.close();
            if (!output || sha256(targetScript, options) != source.fingerprints.at(script.asset.resolvedPath))
                throw std::runtime_error("Staged script differs from inspected source: " + script.path);
        }
        const auto filename = source.summary.name + ".esp";
        if (options.progress) options.progress({"Writing native world", 0, source.summary.cells});
        writeFo4WorldPlugin(data / filename, source.request);
        checkpoint(options);
        source.verifyInputs(options);
        std::ostringstream report;
        report << "OpenF4 native world geometry proof\nNOT A COMPLETE PLAYABLE-WORLD EXPORT\n\n";
        for (const auto& limitation : source.summary.limitations) report << "- " << limitation << '\n';
        report << "\nSource: " << source.document.path().string() << "\nPlugin: " << filename
            << "\nWorld: " << source.request.editorId << "\nArrival: " << source.request.arrival.editorId
            << "\nCells: " << source.summary.cells << "\nReferences: " << source.summary.references
            << "\nModels: " << source.summary.models << "\n\nRequired master: Fallout4.esm"
            << "\nInstall the Data contents using a separate test profile. No files were installed automatically."
            << "\nFrom a disposable test session: cow " << source.request.editorId << ' '
            << source.request.arrival.placement.cell.x << ' ' << source.request.arrival.placement.cell.y
            << "\nSafe arrival local FormID: " << std::hex << std::setfill('0') << std::setw(6)
            << (source.request.arrival.placement.formId & 0xFFFFFF) << std::dec
            << "\nUse player.moveto XX<local FormID>, replacing XX with this plugin's load-order index. Do not use a main save."
            << "\n\nSource SHA-256:\n";
        for (const auto& [path, digest] : source.fingerprints) report << digest << "  " << path.string() << '\n';
        report << "\nOutput SHA-256:\n";
        std::vector<std::filesystem::path> outputs;
        for (const auto& entry : std::filesystem::recursive_directory_iterator(data))
            if (entry.is_regular_file()) outputs.push_back(entry.path());
        std::ranges::sort(outputs);
        for (const auto& output : outputs)
            report << sha256(output, options) << "  " << output.lexically_relative(temporary).generic_string() << '\n';
        writeText(temporary / "export-report.txt", report.str());
        if (options.progress) options.progress({"Publishing package", source.summary.cells, source.summary.cells});
        checkpoint(options);
        noLinks(target); noLinks(temporary);
        if (std::filesystem::exists(target)) throw std::runtime_error("Export destination appeared during build");
#ifdef _WIN32
        for (unsigned attempt = 0; ; ++attempt)
        {
            checkpoint(options);
            noLinks(target); noLinks(temporary);
            if (std::filesystem::exists(target)) throw std::runtime_error("Export destination appeared during publication");
            if (MoveFileExW(temporary.c_str(), target.c_str(), MOVEFILE_WRITE_THROUGH)) break;
            const auto error = GetLastError();
            if (attempt == 20 || (error != ERROR_ACCESS_DENIED && error != ERROR_SHARING_VIOLATION))
                throw std::runtime_error("Cannot publish export package (Windows error " + std::to_string(error) + ")");
            // File scanners can briefly hold a freshly written directory without delete sharing.
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
#else
        std::filesystem::rename(temporary, target);
#endif
        return {target, target / "Data" / filename, source.request.editorId,
            source.request.arrival.editorId, source.request.arrival.placement.cell};
    }
    catch (const std::exception& error)
    {
        throw std::runtime_error(std::string(error.what()) + "; unpublished staging retained at " + temporary.string());
    }
}
}
