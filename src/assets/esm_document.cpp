#include "assets/esm_document.hpp"
#include "assets/asset_resolver.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <zlib.h>
#ifdef _WIN32
#define NOMINMAX
#include <Windows.h>
#endif

namespace openf4::assets
{
namespace
{
using Bytes = std::vector<std::byte>;
constexpr std::uint32_t cc(const char* s)
{
    return std::uint32_t(s[0]) | std::uint32_t(s[1]) << 8 | std::uint32_t(s[2]) << 16 | std::uint32_t(s[3]) << 24;
}
template<class T> T read(std::span<const std::byte> bytes, std::size_t offset)
{
    if (offset > bytes.size() || sizeof(T) > bytes.size() - offset) throw std::runtime_error("Truncated ESM data");
    T result{}; std::memcpy(&result, bytes.data() + offset, sizeof(T)); return result;
}
template<class T> void put(Bytes& bytes, const T& value)
{
    const auto* p = reinterpret_cast<const std::byte*>(&value); bytes.insert(bytes.end(), p, p + sizeof(T));
}
template<class T> void overwrite(Bytes& bytes, std::size_t offset, const T& value)
{
    if (offset + sizeof(T) > bytes.size()) throw std::runtime_error("Invalid ESM patch");
    std::memcpy(bytes.data() + offset, &value, sizeof(T));
}
void append(Bytes& bytes, const Bytes& value) { bytes.insert(bytes.end(), value.begin(), value.end()); }
std::string string(std::span<const std::byte> bytes)
{
    const auto* p = reinterpret_cast<const char*>(bytes.data());
    return {p, std::find(p, p + bytes.size(), '\0')};
}
struct Sub { std::uint32_t type; Bytes bytes; };
std::vector<Sub> subs(std::span<const std::byte> body)
{
    std::vector<Sub> result;
    for (std::size_t p = 0; p < body.size();)
    {
        auto type = read<std::uint32_t>(body, p);
        std::uint32_t size = read<std::uint16_t>(body, p + 4); p += 6;
        if (type == cc("XXXX"))
        {
            if (size != 4) throw std::runtime_error("Invalid ESM extended subrecord");
            size = read<std::uint32_t>(body, p); p += 4;
            type = read<std::uint32_t>(body, p); (void)read<std::uint16_t>(body, p + 4); p += 6;
        }
        if (size > body.size() - p) throw std::runtime_error("ESM subrecord exceeds record");
        result.push_back({type, Bytes(body.begin() + p, body.begin() + p + size)}); p += size;
    }
    return result;
}
void sub(Bytes& bytes, std::uint32_t type, const Bytes& payload)
{
    if (payload.size() > 65535)
    {
        put(bytes, cc("XXXX")); put(bytes, std::uint16_t{4}); put(bytes, static_cast<std::uint32_t>(payload.size()));
        put(bytes, type); put(bytes, std::uint16_t{0});
    }
    else { put(bytes, type); put(bytes, static_cast<std::uint16_t>(payload.size())); }
    append(bytes, payload);
}
template<class T> void scalar(Bytes& bytes, std::uint32_t type, const T& value) { Bytes b; put(b, value); sub(bytes, type, b); }
void textSub(Bytes& bytes, std::uint32_t type, const std::string& value)
{
    Bytes b(value.size() + 1); std::memcpy(b.data(), value.c_str(), b.size()); sub(bytes, type, b);
}
Bytes record(std::uint32_t type, std::uint32_t form, const Bytes& body, std::uint32_t flags = 0)
{
    Bytes out; put(out, type); put(out, static_cast<std::uint32_t>(body.size())); put(out, flags); put(out, form);
    put(out, std::uint32_t{0}); put(out, std::uint16_t{131}); put(out, std::uint16_t{0}); append(out, body); return out;
}
Bytes group(std::uint32_t label, std::uint32_t type, const Bytes& body)
{
    if (body.size() > UINT32_MAX - 24u) throw std::runtime_error("ESM group exceeds 4 GiB");
    Bytes out; put(out, cc("GRUP")); put(out, static_cast<std::uint32_t>(body.size() + 24)); put(out, label);
    put(out, type); put(out, std::uint64_t{0}); append(out, body); return out;
}
std::uint64_t countRecords(std::span<const std::byte> bytes)
{
    std::uint64_t count = 0;
    for (std::size_t p = 0; p < bytes.size();)
    {
        const bool grouped = read<std::uint32_t>(bytes, p) == cc("GRUP");
        const auto size = std::uint64_t(read<std::uint32_t>(bytes, p + 4)) + (grouped ? 0 : 24);
        if (size < 24 || size > bytes.size() - p) throw std::runtime_error("Invalid encoded ESM record size");
        ++count;
        if (grouped) count += countRecords(bytes.subspan(p + 24, static_cast<std::size_t>(size - 24)));
        p += static_cast<std::size_t>(size);
    }
    return count;
}
Bytes readFile(std::ifstream& file, std::uint64_t offset, std::size_t size)
{
    Bytes b(size); file.clear(); file.seekg(static_cast<std::streamoff>(offset));
    file.read(reinterpret_cast<char*>(b.data()), static_cast<std::streamsize>(size));
    if (!file) throw std::runtime_error("Could not read ESM record"); return b;
}
struct Node
{
    Bytes header;
    std::uint64_t offset = 0, size = 0;
    std::uint32_t type = 0, form = 0, groupType = 0, world = 0, cell = 0, childGroupType = 0;
    std::vector<std::unique_ptr<Node>> children;
};
Bytes body(std::ifstream& file, const Node& n)
{
    auto b = readFile(file, n.offset + 24, static_cast<std::size_t>(n.size - 24));
    if ((read<std::uint32_t>(n.header, 8) & 0x40000u) != 0)
    {
        const auto size = read<std::uint32_t>(b, 0);
        if (size > 128u * 1024u * 1024u) throw std::runtime_error("ESM record decompression exceeds limit");
        Bytes uncompressed(size); uLongf actual = size;
        if (uncompress(reinterpret_cast<Bytef*>(uncompressed.data()), &actual,
            reinterpret_cast<const Bytef*>(b.data() + 4), static_cast<uLong>(b.size() - 4)) != Z_OK || actual != size)
            throw std::runtime_error("Invalid compressed ESM record");
        return uncompressed;
    }
    return b;
}
Bytes patchRecord(std::ifstream& file, const Node* n, std::uint32_t type, std::uint32_t id,
    const Bytes& replacementSubs, const std::set<std::uint32_t>& replace)
{
    Bytes payload;
    if (n) for (const auto& s : subs(body(file, *n))) if (!replace.contains(s.type)) sub(payload, s.type, s.bytes);
    append(payload, replacementSubs);
    if (!n) return record(type, id, payload);
    Bytes result = n->header;
    overwrite(result, 4, static_cast<std::uint32_t>(payload.size()));
    overwrite(result, 8, read<std::uint32_t>(result, 8) & ~0x40000u);
    overwrite(result, 12, id);
    append(result, payload); return result;
}
Bytes vhgt(const LandscapeCell& cell)
{
    std::array<std::int32_t, 1089> u{};
    for (std::size_t i = 0; i < u.size(); ++i)
    {
        double value = cell.heights[i] / 8.0;
        if (!std::isfinite(value) || value < INT32_MIN || value > INT32_MAX || std::abs(value - std::round(value)) > 0.0001)
            throw std::runtime_error("LAND heights must be finite, quantized 8-unit values");
        u[i] = static_cast<std::int32_t>(std::round(value));
    }
    Bytes result; put(result, static_cast<float>(u[0]));
    for (std::size_t y = 0; y < 33; ++y) for (std::size_t x = 0; x < 33; ++x)
    {
        const auto i = y * 33 + x;
        const std::int64_t delta = std::int64_t(u[i]) - u[x ? i - 1 : (y ? i - 33 : i)];
        if (delta < -128 || delta > 127) throw std::runtime_error("LAND height exceeds signed VHGT delta range");
        put(result, static_cast<std::int8_t>(delta));
    }
    put(result, std::array<std::byte, 3>{}); return result;
}
LandscapeCell decodeLand(CellCoord coord, const Bytes& b)
{
    LandscapeCell result; result.cell = coord;
    for (const auto& s : subs(b)) if (s.type == cc("VHGT"))
    {
        if (s.bytes.size() != 1096) throw std::runtime_error("Invalid LAND VHGT length");
        float row = read<float>(s.bytes, 0);
        if (!std::isfinite(row) || std::round(row) != row) throw std::runtime_error("Invalid LAND VHGT height");
        for (std::size_t p = 1093; p < 1096; ++p)
            if (s.bytes[p] != std::byte{0}) throw std::runtime_error("Invalid LAND VHGT trailer");
        for (std::size_t y = 0; y < 33; ++y)
        {
            row += read<std::int8_t>(s.bytes, 4 + y * 33); float value = row;
            for (std::size_t x = 0; x < 33; ++x)
            {
                if (x) value += read<std::int8_t>(s.bytes, 4 + y * 33 + x);
                result.heights[y * 33 + x] = value * 8;
            }
        }
        result.hasHeights = true;
    }
    return result;
}
std::int32_t floorDiv(std::int32_t n, std::int32_t d) { return n / d - (n % d < 0); }
std::uint32_t coordLabel(CellCoord c, std::int32_t divisor)
{
    return std::uint16_t(floorDiv(c.x, divisor)) | (std::uint32_t(std::uint16_t(floorDiv(c.y, divisor))) << 16);
}
void atomicReplace(const std::filesystem::path& target, const std::filesystem::path& temporary)
{
#ifdef _WIN32
    const BOOL ok = std::filesystem::exists(target)
        ? ReplaceFileW(target.c_str(), temporary.c_str(), nullptr, REPLACEFILE_WRITE_THROUGH, nullptr, nullptr)
        : MoveFileExW(temporary.c_str(), target.c_str(), MOVEFILE_WRITE_THROUGH);
    if (!ok) throw std::runtime_error("Could not atomically replace ESM (Windows error " + std::to_string(GetLastError()) + ")");
#else
    std::filesystem::rename(temporary, target);
#endif
}
std::filesystem::path temporaryPath(const std::filesystem::path& path)
{
    static std::atomic<std::uint64_t> sequence{0};
    auto p = path; p += ".openf4-" + std::to_string(++sequence) + ".tmp";
    if (std::filesystem::exists(p)) throw std::runtime_error("ESM temporary path already exists"); return p;
}
std::string folded(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); }); return s;
}
}

struct EsmDocument::Impl
{
    std::filesystem::path source, destination;
    std::vector<std::unique_ptr<Node>> roots;
    std::vector<std::string> masterNames;
    std::uint32_t world = 0, next = 0x800;
    bool owned = false, dem = false;
    std::vector<ESMModel> modelList, originalModels;
    std::vector<ESMReference> refList, originalRefs;
    std::map<CellCoord, ESMTerrainCell> cells, changedTerrain;
    std::unordered_map<std::uint32_t, Node*> records, temporaryGroups, cellGroups;
    std::map<std::pair<std::uint32_t, std::uint32_t>, Node*> childGroups;
    std::map<std::uint32_t, CellCoord> cellCoords;
    Node* worldChildren = nullptr;

    explicit Impl(std::filesystem::path path) : source(std::filesystem::absolute(path)), destination(source)
    {
        std::ifstream file(source, std::ios::binary | std::ios::ate);
        if (!file || file.tellg() < 24) throw std::runtime_error("Could not open ESM: " + source.string());
        const auto size = static_cast<std::uint64_t>(file.tellg());
        scan(file, roots, 0, size, 0, 0, 0);
        if (roots.empty() || roots[0]->type != cc("TES4")) throw std::runtime_error("ESM must start with TES4");
        if (masterNames.size() >= 255) throw std::runtime_error("ESM has too many masters");
        const auto prefix = static_cast<std::uint32_t>(masterNames.size()) << 24;
        next = prefix | std::max(next & 0xFFFFFFu, 0x800u);
        for (const auto& [id, n] : records) if ((id & 0xFF000000u) == prefix && id >= next)
        {
            if ((id & 0xFFFFFFu) == 0xFFFFFFu) throw std::runtime_error("ESM FormID space exhausted"); next = id + 1;
        }
        for (auto& r : refList)
        {
            auto n = records.find(r.placement.formId);
            if (r.childGroupType == 8)
            {
                const auto coordinate = [](float value) {
                    const double cell = std::floor(double(value) / 4096.0);
                    if (!std::isfinite(cell) || cell < INT32_MIN || cell > INT32_MAX)
                        throw std::runtime_error("Persistent reference position is outside supported CELL coordinates");
                    return static_cast<std::int32_t>(cell);
                };
                r.placement.cell = {coordinate(r.placement.position[0]), coordinate(r.placement.position[1])};
            }
            else if (n != records.end() && cellCoords.contains(n->second->cell)) r.placement.cell = cellCoords.at(n->second->cell);
        }
        refList.erase(std::remove_if(refList.begin(), refList.end(), [&](const auto& r) {
            return records.at(r.placement.formId)->world != world;
        }), refList.end());
        originalRefs = refList; originalModels = modelList;
    }

    void scan(std::ifstream& file, std::vector<std::unique_ptr<Node>>& out, std::uint64_t begin,
        std::uint64_t end, std::uint32_t contextWorld, std::uint32_t contextCell, unsigned depth,
        std::uint32_t childGroupType = 0)
    {
        if (depth > 32) throw std::runtime_error("ESM group nesting exceeds limit");
        for (auto p = begin; p < end;)
        {
            if (end - p < 24) throw std::runtime_error("Truncated ESM header");
            auto node = std::make_unique<Node>(); auto& n = *node;
            n.header = readFile(file, p, 24); n.offset = p; n.type = read<std::uint32_t>(n.header, 0);
            auto size = read<std::uint32_t>(n.header, 4);
            n.size = n.type == cc("GRUP") ? size : std::uint64_t(size) + 24;
            if (n.size < 24 || n.size > end - p) throw std::runtime_error("ESM record or group exceeds its parent");
            n.world = contextWorld; n.cell = contextCell; n.childGroupType = childGroupType;
            if (n.type == cc("GRUP"))
            {
                n.form = read<std::uint32_t>(n.header, 8); n.groupType = read<std::uint32_t>(n.header, 12);
                if (n.groupType == 1) n.world = n.form;
                if (n.groupType == 6) { n.cell = n.form; cellGroups[n.form] = &n; }
                if (n.groupType == 9) temporaryGroups[n.form] = &n;
                if (n.groupType == 8 || n.groupType == 9 || n.groupType == 10)
                {
                    n.childGroupType = n.groupType;
                    childGroups[{n.form, n.groupType}] = &n;
                    if (contextCell != n.form) throw std::runtime_error("Reference child group has incorrect CELL ownership");
                }
                if (n.groupType == 1 && n.form == world) worldChildren = &n;
                scan(file, n.children, p + 24, p + n.size, n.world, n.cell, depth + 1, n.childGroupType);
            }
            else
            {
                n.form = read<std::uint32_t>(n.header, 12);
                if (n.form && !records.emplace(n.form, &n).second) throw std::runtime_error("Duplicate ESM record FormID");
                if (n.type == cc("TES4"))
                {
                    for (const auto& s : subs(body(file, n)))
                    {
                        if (s.type == cc("MAST")) masterNames.push_back(string(s.bytes));
                        if (s.type == cc("CNAM")) owned = string(s.bytes) == "OpenF4" || string(s.bytes) == "EMPIRE";
                        if (s.type == cc("HEDR")) next = read<std::uint32_t>(s.bytes, 8);
                        if (s.type == cc("EMGI")) { auto text = string(s.bytes); dem = (text.starts_with("OPENF4_DEM 1\n") || text.starts_with("EMPIRE_DEM 1\n")) && text.find("\nROLE OVERLAY\n") != std::string::npos; }
                    }
                }
                else if (n.type == cc("WRLD"))
                {
                    if (!world) world = n.form;
                    for (const auto& s : subs(body(file, n))) if (s.type == cc("EDID") && folded(string(s.bytes)) == "commonwealth" && world != n.form)
                    {
                        // Other worldspaces may precede Commonwealth in a master.
                        // Their exterior coordinates are unrelated to the active world.
                        world = n.form; cells.clear(); worldChildren = nullptr;
                    }
                }
                else if (n.type == cc("CELL"))
                {
                    for (const auto& s : subs(body(file, n))) if (s.type == cc("XCLC"))
                    {
                        CellCoord c{read<std::int32_t>(s.bytes, 0), read<std::int32_t>(s.bytes, 4)};
                        cellCoords[n.form] = c;
                        if (contextWorld == world)
                        {
                            if (cells.contains(c)) throw std::runtime_error("Duplicate exterior CELL coordinate");
                            auto& entry = cells[c]; entry.cellFormId = n.form; entry.landscape.cell = c;
                            entry.cellSource = {source.string(), n.form};
                        }
                    }
                }
                else if (n.type == cc("LAND"))
                {
                    if (contextWorld == world)
                    {
                        if (!cellCoords.contains(contextCell)) throw std::runtime_error("LAND is outside CELL children");
                        auto& entry = cells.at(cellCoords.at(contextCell));
                        if (entry.landFormId) throw std::runtime_error("Duplicate LAND record in CELL");
                        entry.landFormId = n.form;
                        entry.landSource = {source.string(), n.form};
                    }
                }
                else if (n.type == cc("STAT") || n.type == cc("MSTT"))
                {
                    ESMModel model; model.formId = n.form; model.sourceEsm = source.string();
                    model.kind = n.type == cc("MSTT") ? ESMModelKind::movableStatic : ESMModelKind::staticObject;
                    model.recordSourceEsm = source.string(); model.recordSourceFormId = n.form;
                    model.recordOwnerEsm = source.string();
                    for (const auto& s : subs(body(file, n)))
                    {
                        if (s.type == cc("EDID")) model.editorId = string(s.bytes);
                        if (s.type == cc("MODL")) model.path = string(s.bytes);
                        if (s.type == cc("ESRC")) model.sourcePath = string(s.bytes);
                        if (s.type == cc("VMAD"))
                        {
                            if (!model.scripts.present) model.scripts = decodePapyrusBindings(s.bytes, source.string());
                            else { model.scripts.editable = false; model.scripts.diagnostic = "Duplicate VMAD fields are read-only"; }
                        }
                    }
                    if (!model.path.empty()) modelList.push_back(std::move(model));
                }
                else if (n.type == cc("REFR"))
                {
                    ESMReference ref; ref.placement.formId = n.form; ref.placement.instanceId = n.form;
                    ref.recordSource = {source.string(), n.form};
                    ref.ownerCellFormId = n.cell;
                    ref.childGroupType = n.childGroupType;
                    for (const auto& s : subs(body(file, n)))
                    {
                        if (s.type == cc("EDID")) ref.editorId = string(s.bytes);
                        if (s.type == cc("NAME")) ref.placement.baseObject = read<std::uint32_t>(s.bytes, 0);
                        if (s.type == cc("DATA") && s.bytes.size() >= 24)
                        { std::memcpy(ref.placement.position, s.bytes.data(), 12); std::memcpy(ref.placement.rotation, s.bytes.data() + 12, 12); }
                        if (s.type == cc("XSCL")) ref.placement.scale = read<float>(s.bytes, 0);
                        if (s.type == cc("XMSP")) ref.placement.materialSwap = read<std::uint32_t>(s.bytes, 0);
                        if (s.type == cc("VMAD"))
                        {
                            if (!ref.scripts.present) ref.scripts = decodePapyrusBindings(s.bytes, source.string());
                            else { ref.scripts.editable = false; ref.scripts.diagnostic = "Duplicate VMAD fields are read-only"; }
                        }
                    }
                    ref.deleted = (read<std::uint32_t>(n.header, 8) & 0x20u) != 0;
                    refList.push_back(std::move(ref));
                }
            }
            out.push_back(std::move(node)); p += n.size;
        }
    }

    void save()
    {
        std::ifstream input(source, std::ios::binary);
        if (!input) throw std::runtime_error("Could not reopen source ESM");
        std::unordered_map<const Node*, Bytes> replacements, additions, prefixes;
        std::set<const Node*> removed;
        std::map<std::uint32_t, Bytes> newModels;
        Bytes newTop;
        std::map<std::uint32_t, std::map<std::uint32_t, Bytes>> newRefs;
        std::map<std::uint32_t, ESMRecordSource> persistentOwners;
        std::map<std::uint32_t, ESMModel> oldModels;
        const auto validateScripts = [&](const PapyrusBindings& bindings) {
            for (const auto& script : bindings.scripts) for (const auto& property : script.properties)
                if (const auto* object = std::get_if<PapyrusObject>(&property.value))
                    if ((object->formId >> 24) > masterNames.size())
                        throw std::runtime_error("VMAD object property refers outside the master table");
        };
        const auto patchScripts = [&](std::ifstream& file, const Node* prototype,
            const PapyrusBindings& bindings, Bytes& fields, std::set<std::uint32_t>& replace) {
            std::optional<Bytes> previous;
            if (prototype) for (const auto& field : subs(body(file, *prototype)))
                if (field.type == cc("VMAD")) { previous = Bytes(field.bytes.begin(), field.bytes.end()); break; }
            const auto desired = encodePapyrusBindings(bindings, prototype != nullptr);
            // An undo snapshot may predate the latest save, even when it was never edited itself.
            if (bindings.present == previous.has_value() && (!bindings.present || *previous == desired)) return;
            if (bindings.edited || bindings.originalBytes.empty()) validateScripts(bindings);
            replace.insert(cc("VMAD"));
            if (bindings.present) sub(fields, cc("VMAD"), desired);
        };
        for (const auto& m : originalModels) oldModels.emplace(m.formId, m);
        std::set<std::uint32_t> modelIds;
        std::map<std::string, std::unique_ptr<Impl>> modelSources;
        const auto inheritedRecord = [&](const ESMRecordSource& origin, std::uint32_t type)
            -> std::pair<const Node*, std::unique_ptr<std::ifstream>> {
            if (origin.esm.empty() || !origin.formId) throw std::runtime_error("Inherited record is missing source provenance");
            auto& indexed = modelSources[origin.esm];
            if (!indexed) indexed = std::make_unique<Impl>(origin.esm);
            const auto found = indexed->records.find(origin.formId);
            if (found == indexed->records.end() || found->second->type != type)
                throw std::runtime_error("Original record is missing or has a different type");
            if ((type == cc("CELL") || type == cc("LAND") || type == cc("REFR")) && found->second->world != world)
                throw std::runtime_error("Cannot preserve an inherited record from a different world");
            // Unknown FormID fields remain valid only when every source slot retains its meaning.
            for (std::size_t i = 0; i < indexed->masterNames.size(); ++i)
                if (i >= masterNames.size() || folded(indexed->masterNames[i]) != folded(masterNames[i]))
                    throw std::runtime_error("Cannot preserve an inherited record with reordered master slots");
            if (indexed->source != source
                && (indexed->masterNames.size() >= masterNames.size()
                    || folded(masterNames[indexed->masterNames.size()]) != folded(indexed->source.filename().string())))
                throw std::runtime_error("Cannot preserve an inherited record with a remapped self slot");
            auto file = std::make_unique<std::ifstream>(origin.esm, std::ios::binary);
            if (!*file) throw std::runtime_error("Cannot reopen original record");
            if (indexed->source != source && (read<std::uint32_t>(indexed->roots.front()->header, 8) & 0x80u))
                for (const auto& field : subs(body(*file, *found->second)))
                    if (field.type == cc("FULL"))
                        throw std::runtime_error("Cannot preserve an inherited localized name without its string table");
            return {found->second, std::move(file)};
        };
        for (const auto& m : modelList)
        {
            if (!m.formId || !modelIds.insert(m.formId).second) throw std::runtime_error("Invalid or duplicate model FormID");
            if (auto old = oldModels.find(m.formId); old != oldModels.end() && old->second == m) continue;
            const auto type = m.kind == ESMModelKind::staticObject ? cc("STAT") : cc("MSTT");
            if (m.kind != ESMModelKind::staticObject && m.kind != ESMModelKind::movableStatic)
                throw std::runtime_error("Unsupported model record kind");
            const Node* node = records.contains(m.formId) ? records.at(m.formId) : nullptr;
            if (node && node->type != type) throw std::runtime_error("Changing an existing FormID's record type is not allowed");
            const Node* prototype = node;
            std::unique_ptr<std::ifstream> inherited;
            if (!prototype && !m.recordSourceEsm.empty() && m.recordSourceFormId)
            {
                auto original = inheritedRecord({m.recordSourceEsm, m.recordSourceFormId}, type);
                prototype = original.first; inherited = std::move(original.second);
            }
            auto modelPath = m.path;
            std::ranges::replace(modelPath, '\\', '/');
            if (folded(modelPath).starts_with("meshes/") && folded(modelPath).ends_with(".nif")) modelPath.erase(0, 7);
            Bytes fields; textSub(fields, cc("EDID"), m.editorId); textSub(fields, cc("MODL"), modelPath);
            if (!m.sourcePath.empty()) textSub(fields, cc("ESRC"), m.sourcePath);
            auto& originalFile = inherited ? *inherited : input;
            std::set<std::uint32_t> originalFields;
            if (prototype) for (const auto& field : subs(body(originalFile, *prototype))) originalFields.insert(field.type);
            if (!originalFields.contains(cc("OBND"))) scalar(fields, cc("OBND"), std::array<std::int16_t, 6>{});
            if (m.kind == ESMModelKind::staticObject && !originalFields.contains(cc("DNAM")))
            {
                Bytes direction; put(direction, 90.0f); put(direction, std::array<std::uint32_t, 3>{});
                sub(fields, cc("DNAM"), direction);
            }
            if (m.kind == ESMModelKind::movableStatic)
            {
                if (!originalFields.contains(cc("FULL"))) textSub(fields, cc("FULL"), m.editorId);
                if (!originalFields.contains(cc("DATA"))) scalar(fields, cc("DATA"), std::uint8_t{1});
            }
            std::set<std::uint32_t> replace{cc("EDID"), cc("MODL"), cc("ESRC")};
            patchScripts(originalFile, prototype, m.scripts, fields, replace);
            auto encoded = patchRecord(originalFile, prototype, type, m.formId, fields, replace);
            if (node) replacements[node] = std::move(encoded); else append(newModels[type], encoded);
        }
        for (const auto& m : originalModels) if (!modelIds.contains(m.formId)) removed.insert(records.at(m.formId));
        std::map<std::uint32_t, ESMReference> currentRefs;
        for (const auto& r : refList) if (!currentRefs.emplace(r.placement.formId, r).second) throw std::runtime_error("Duplicate reference FormID");
        for (const auto& r : originalRefs) if (!currentRefs.contains(r.placement.formId)) removed.insert(records.at(r.placement.formId));
        for (const auto& r : refList)
        {
            auto old = std::find_if(originalRefs.begin(), originalRefs.end(), [&](const auto& x) { return x.placement.formId == r.placement.formId; });
            if (old != originalRefs.end() && *old == r) continue;
            const auto& p = r.placement;
            for (float v : p.position) if (!std::isfinite(v)) throw std::runtime_error("Reference position is not finite");
            for (float v : p.rotation) if (!std::isfinite(v)) throw std::runtime_error("Reference rotation is not finite");
            if (!std::isfinite(p.scale) || p.scale <= 0) throw std::runtime_error("Reference scale must be positive and finite");
            const Node* node = records.contains(p.formId) ? records.at(p.formId) : nullptr;
            if (node && node->type != cc("REFR")) throw std::runtime_error("Reference FormID has a different record type");
            const Node* prototype = node;
            std::unique_ptr<std::ifstream> inherited;
            if (!prototype && r.recordSource.esm.empty() && (p.formId >> 24) < masterNames.size())
                throw std::runtime_error("Inherited reference is missing source provenance");
            if (!prototype && !r.recordSource.esm.empty())
            {
                auto original = inheritedRecord(r.recordSource, cc("REFR"));
                prototype = original.first; inherited = std::move(original.second);
            }
            const auto childType = prototype ? prototype->childGroupType : r.childGroupType;
            if (childType != 8 && childType != 9 && childType != 10)
                throw std::runtime_error("Unsupported reference child-group ownership");
            std::uint32_t owner = 0;
            if (childType == 8)
            {
                if (!prototype || !prototype->cell) throw std::runtime_error("Persistent reference is missing original CELL ownership");
                owner = prototype->cell;
                if (!records.contains(owner)) persistentOwners[owner] = {r.recordSource.esm, owner};
            }
            else if (const auto changed = changedTerrain.find(p.cell); changed != changedTerrain.end()) owner = changed->second.cellFormId;
            else if (const auto existing = cells.find(p.cell); existing != cells.end()) owner = existing->second.cellFormId;
            if (!owner) throw std::runtime_error("Reference destination has no CELL ownership");
            Bytes fields; if (!r.editorId.empty()) textSub(fields, cc("EDID"), r.editorId);
            scalar(fields, cc("NAME"), p.baseObject); Bytes transform;
            for (float v : p.position) put(transform, v); for (float v : p.rotation) put(transform, v);
            sub(fields, cc("DATA"), transform); scalar(fields, cc("XSCL"), p.scale);
            if (p.materialSwap) scalar(fields, cc("XMSP"), p.materialSwap);
            std::set<std::uint32_t> replace{cc("EDID"), cc("NAME"), cc("DATA"), cc("XSCL"), cc("XMSP")};
            patchScripts(inherited ? *inherited : input, prototype, r.scripts, fields, replace);
            auto encoded = patchRecord(inherited ? *inherited : input, prototype, cc("REFR"), p.formId, fields, replace);
            overwrite(encoded, 8, (read<std::uint32_t>(encoded, 8) & ~0x20u) | (r.deleted ? 0x20u : 0u));
            if (node && node->cell == owner) replacements[node] = std::move(encoded);
            else { if (node) removed.insert(node); append(newRefs[owner][childType], encoded); }
        }

        std::map<CellCoord, ESMTerrainCell> newCells;
        std::map<std::uint32_t, Bytes> newLand;
        for (const auto& [coord, terrain] : changedTerrain)
        {
            if (!cells.contains(coord)) newCells[coord] = terrain;
            Bytes fields; const auto heights = vhgt(terrain.landscape); sub(fields, cc("VHGT"), heights);
            const Node* node = records.contains(terrain.landFormId) ? records.at(terrain.landFormId) : nullptr;
            if (node && node->type != cc("LAND")) throw std::runtime_error("LAND FormID has a different record type");
            const Node* prototype = node;
            std::unique_ptr<std::ifstream> inherited;
            if (!prototype && !dem && terrain.landSource.esm.empty() && (terrain.landFormId >> 24) < masterNames.size())
                throw std::runtime_error("Inherited LAND is missing source provenance");
            if (!prototype && !terrain.landSource.esm.empty())
            {
                auto original = inheritedRecord(terrain.landSource, cc("LAND"));
                if (terrain.landFormId != original.first->form)
                    throw std::runtime_error("Cannot remap inherited LAND ownership");
                prototype = original.first; inherited = std::move(original.second);
            }
            auto& originalFile = inherited ? *inherited : input;
            // Object placement needs a CELL anchor, not an unchanged LAND override.
            bool unchanged = false;
            if (prototype) for (const auto& s : subs(body(originalFile, *prototype)))
                if (s.type == cc("VHGT")) unchanged = s.bytes == heights;
            if (unchanged) continue;
            auto encoded = patchRecord(originalFile, prototype, cc("LAND"), terrain.landFormId, fields, {cc("VHGT")});
            if (node) replacements[node] = std::move(encoded);
            else newLand[terrain.cellFormId] = std::move(encoded);
        }
        const auto encodeChildren = [&](std::uint32_t id) {
            auto children = std::move(newRefs[id]);
            if (auto land = newLand.find(id); land != newLand.end())
            {
                append(land->second, children[9]); children[9] = std::move(land->second); newLand.erase(land);
            }
            newRefs.erase(id);
            Bytes encoded;
            for (const auto& [type, data] : children) if (!data.empty()) append(encoded, group(id, type, data));
            return group(id, 6, encoded);
        };
        std::map<std::uint32_t, std::map<std::uint32_t, Bytes>> trees;
        for (const auto& [coord, terrain] : newCells)
        {
            Bytes tree;
            if (!terrain.cellSource.esm.empty())
            {
                auto original = inheritedRecord(terrain.cellSource, cc("CELL"));
                if (terrain.cellFormId != original.first->form)
                    throw std::runtime_error("Cannot remap inherited CELL ownership");
                tree = readFile(*original.second, original.first->offset, static_cast<std::size_t>(original.first->size));
            }
            else
            {
                if (!dem && (terrain.cellFormId >> 24) < masterNames.size())
                    throw std::runtime_error("Inherited CELL is missing source provenance");
                Bytes fields; textSub(fields, cc("EDID"), "OpenF4Cell_" + std::to_string(coord.x) + "_" + std::to_string(coord.y));
                scalar(fields, cc("DATA"), static_cast<std::uint16_t>(dem ? 2 : 0));
                scalar(fields, cc("XCLC"), std::array<std::int32_t, 3>{coord.x, coord.y, 0});
                if (dem) scalar(fields, cc("XCLW"), 0.0f);
                tree = record(cc("CELL"), terrain.cellFormId, fields);
            }
            append(tree, encodeChildren(terrain.cellFormId));
            append(trees[coordLabel(coord, 32)][coordLabel(coord, 8)], tree);
        }
        for (const auto& [id, origin] : persistentOwners)
        {
            auto original = inheritedRecord(origin, cc("CELL"));
            if (original.first->world != world || !worldChildren)
                throw std::runtime_error("Persistent reference belongs to a different world");
            if (modelSources.at(origin.esm)->cellCoords.contains(id))
                throw std::runtime_error("Persistent reference CELL must retain its original world-level ownership");
            auto tree = readFile(*original.second, original.first->offset, static_cast<std::size_t>(original.first->size));
            append(tree, encodeChildren(id)); append(additions[worldChildren], tree);
        }
        for (auto& [id, land] : newLand)
        {
            if (temporaryGroups.contains(id)) prefixes[temporaryGroups.at(id)] = std::move(land);
            else { append(land, newRefs[id][9]); newRefs[id][9] = std::move(land); }
        }
        for (const auto& [id, groups] : newRefs)
        {
            if (!records.contains(id) || records.at(id)->type != cc("CELL"))
                throw std::runtime_error("Reference CELL ownership is missing");
            Bytes fresh;
            for (const auto& [type, refs] : groups)
                if (childGroups.contains({id, type})) append(additions[childGroups.at({id, type})], refs);
                else append(fresh, group(id, type, refs));
            if (!fresh.empty())
            {
                if (cellGroups.contains(id)) append(additions[cellGroups.at(id)], fresh);
                else append(additions[records.at(id)], group(id, 6, fresh));
            }
        }
        if (!trees.empty() && !worldChildren) throw std::runtime_error("ESM has no active world children group");
        for (const auto& [block, subblocks] : trees)
        {
            Node* blockNode = nullptr;
            for (const auto& n : worldChildren->children) if (n->type == cc("GRUP") && n->groupType == 4 && n->form == block) blockNode = n.get();
            Bytes freshBlock;
            for (const auto& [subblock, tree] : subblocks)
            {
                Node* subNode = nullptr;
                if (blockNode) for (const auto& n : blockNode->children) if (n->type == cc("GRUP") && n->groupType == 5 && n->form == subblock) subNode = n.get();
                if (subNode) append(additions[subNode], tree); else append(freshBlock, group(subblock, 5, tree));
            }
            if (blockNode) append(additions[blockNode], freshBlock); else append(additions[worldChildren], group(block, 4, freshBlock));
        }
        for (const auto& [type, encoded] : newModels)
        {
            Node* statGroup = nullptr;
            for (const auto& n : roots) if (n->type == cc("GRUP") && n->groupType == 0 && n->form == type) statGroup = n.get();
            if (statGroup) append(additions[statGroup], encoded); else append(newTop, group(type, 0, encoded));
        }
        std::function<std::uint64_t(const Node&)> count = [&](const Node& n) -> std::uint64_t {
            if (removed.contains(&n)) return 0;
            std::uint64_t total = 1 + countRecords(additions[&n]) + countRecords(prefixes[&n]);
            for (const auto& child : n.children) total += count(*child);
            return total;
        };
        auto recordCount = countRecords(newTop);
        for (const auto& n : roots) if (n->type != cc("TES4")) recordCount += count(*n);
        if (recordCount > UINT32_MAX) throw std::runtime_error("ESM record count exceeds HEDR limit");
        Bytes header;
        for (auto s : subs(body(input, *roots.front())))
        {
            if (s.type == cc("HEDR"))
            {
                overwrite(s.bytes, 4, static_cast<std::uint32_t>(recordCount));
                overwrite(s.bytes, 8, next & 0xFFFFFFu);
            }
            sub(header, s.type, s.bytes);
        }
        auto tes4 = roots.front()->header;
        overwrite(tes4, 4, static_cast<std::uint32_t>(header.size())); append(tes4, header);
        replacements[roots.front().get()] = std::move(tes4);
        std::unordered_map<const Node*, std::uint64_t> sizes;
        std::function<std::uint64_t(const Node&)> measure = [&](const Node& n) -> std::uint64_t {
            if (removed.contains(&n)) return sizes[&n] = 0;
            if (auto patch = replacements.find(&n); patch != replacements.end()) return sizes[&n] = patch->second.size() + additions[&n].size();
            if (n.type != cc("GRUP")) return sizes[&n] = n.size + additions[&n].size();
            std::uint64_t size = 24 + additions[&n].size() + prefixes[&n].size();
            for (const auto& child : n.children) size += measure(*child);
            if (size > UINT32_MAX) throw std::runtime_error("ESM group exceeds 4 GiB"); return sizes[&n] = size;
        };
        for (const auto& n : roots) measure(*n);
        if (destination.has_parent_path()) std::filesystem::create_directories(destination.parent_path());
        const auto temporary = temporaryPath(destination);
        try
        {
            std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
            if (!output) throw std::runtime_error("Could not create temporary ESM");
            auto write = [&](const Bytes& b) { output.write(reinterpret_cast<const char*>(b.data()), static_cast<std::streamsize>(b.size())); };
            auto copy = [&](const Node& n) {
                constexpr std::size_t chunk = 1024 * 1024;
                for (std::uint64_t p = 0; p < n.size; p += chunk) write(readFile(input, n.offset + p, static_cast<std::size_t>(std::min<std::uint64_t>(chunk, n.size - p))));
            };
            std::function<void(const Node&)> emit = [&](const Node& n) {
                if (removed.contains(&n)) return;
                if (auto patch = replacements.find(&n); patch != replacements.end())
                {
                    auto b = patch->second;
                    if (n.type == cc("GRUP")) overwrite(b, 4, static_cast<std::uint32_t>(sizes.at(&n)));
                    write(b); write(additions[&n]); return;
                }
                if (n.type != cc("GRUP")) { copy(n); write(additions[&n]); return; }
                auto h = n.header; overwrite(h, 4, static_cast<std::uint32_t>(sizes.at(&n))); write(h);
                write(prefixes[&n]);
                for (const auto& child : n.children) emit(*child); write(additions[&n]);
            };
            for (const auto& n : roots) emit(*n); write(newTop);
            output.flush(); if (!output) throw std::runtime_error("Could not write temporary ESM");
            output.close(); input.close(); atomicReplace(destination, temporary);
        }
        catch (...) { std::error_code ignored; std::filesystem::remove(temporary, ignored); throw; }
    }
};

EsmDocument::EsmDocument(const std::filesystem::path& path) : mImpl(std::make_unique<Impl>(path)) {}
EsmDocument::~EsmDocument() = default;
EsmDocument::EsmDocument(EsmDocument&&) noexcept = default;
EsmDocument& EsmDocument::operator=(EsmDocument&&) noexcept = default;
const std::filesystem::path& EsmDocument::path() const noexcept { return mImpl->destination; }
std::uint32_t EsmDocument::worldFormId() const noexcept { return mImpl->world; }
std::span<const std::string> EsmDocument::masters() const noexcept { return mImpl->masterNames; }
std::uint32_t EsmDocument::nextFormId() const noexcept { return mImpl->next; }
bool EsmDocument::isDemOverlay() const noexcept { return mImpl->dem; }
bool EsmDocument::isOpenF4Owned() const noexcept { return mImpl->owned; }
const std::vector<ESMModel>& EsmDocument::models() const noexcept { return mImpl->modelList; }
const std::vector<ESMReference>& EsmDocument::references() const noexcept { return mImpl->refList; }
const std::map<CellCoord, ESMTerrainCell>& EsmDocument::terrainCells() const noexcept { return mImpl->cells; }
std::optional<ESMTerrainCell> EsmDocument::terrainCell(CellCoord coord) const
{
    if (auto changed = mImpl->changedTerrain.find(coord); changed != mImpl->changedTerrain.end()) return changed->second;
    auto found = mImpl->cells.find(coord); if (found == mImpl->cells.end()) return std::nullopt;
    auto result = found->second;
    if (result.landFormId)
    {
        std::ifstream file(mImpl->source, std::ios::binary);
        result.landscape = decodeLand(coord, body(file, *mImpl->records.at(result.landFormId)));
    }
    return result;
}
std::uint32_t EsmDocument::allocateFormId()
{
    if ((mImpl->next & 0xFFFFFFu) == 0xFFFFFFu) throw std::runtime_error("ESM FormIDs exhausted"); return mImpl->next++;
}
void EsmDocument::setNextFormId(std::uint32_t value)
{
    if ((value >> 24) != mImpl->masterNames.size()) throw std::runtime_error("FormID allocator does not match ESM master count");
    mImpl->next = std::max(mImpl->next, value);
}
void EsmDocument::setModels(std::vector<ESMModel> value) { mImpl->modelList = std::move(value); }
void EsmDocument::setReferences(std::vector<ESMReference> value) { mImpl->refList = std::move(value); }
void EsmDocument::replaceTerrainCells(std::span<const ESMTerrainCell> values)
{
    for (const auto& c : values) mImpl->changedTerrain[c.landscape.cell] = c;
}
void EsmDocument::setPath(std::filesystem::path path) { mImpl->destination = std::filesystem::absolute(path); }
void EsmDocument::save() { mImpl->save(); mImpl = std::make_unique<Impl>(mImpl->destination); }

ESMMasterStack loadEsmMasterStack(const std::filesystem::path& path, std::optional<std::filesystem::path> dataDirectory)
{
    EsmDocument selected(path); ESMMasterStack result; result.worldFormId = selected.worldFormId(); result.basePath = path;
    if (selected.isOpenF4Owned() && !selected.isDemOverlay() && !selected.masters().empty()
        && (selected.worldFormId() >> 24) < selected.masters().size())
    {
        auto candidate = path.parent_path() / selected.masters().back();
        if (!std::filesystem::exists(candidate) && dataDirectory) candidate = *dataDirectory / selected.masters().back();
        if (std::filesystem::exists(candidate)) result.basePath = candidate;
    }
    std::map<std::uint32_t, ESMModel> models; std::map<std::uint32_t, ESMReference> refs;
    std::vector<std::string> canonical(selected.masters().begin(), selected.masters().end());
    canonical.push_back(path.filename().string());
    for (std::size_t fileIndex = 0; fileIndex < canonical.size(); ++fileIndex)
    {
        auto source = fileIndex + 1 == canonical.size() ? path : path.parent_path() / canonical[fileIndex];
        if (!std::filesystem::exists(source) && dataDirectory) source = *dataDirectory / canonical[fileIndex];
        // Base Fallout assets are optional for synthetic OpenF4 fixtures. Terrain shards
        // contain no STAT/REFR, and must not be scanned to display a project overlay.
        if (!std::filesystem::exists(source))
        {
            if (folded(canonical[fileIndex]) == "fallout4.esm") continue;
            throw std::runtime_error("Missing ESM master: " + canonical[fileIndex]);
        }
        const auto name = folded(canonical[fileIndex]);
        if (name.find(".terrain.r") != std::string::npos) continue;
        if (selected.isOpenF4Owned() && selected.worldFormId() != 0x3Cu && name == "fallout4.esm") continue;
        result.sourcePaths.push_back(std::filesystem::absolute(source).lexically_normal());
        EsmDocument document(source);
        auto remap = [&](std::uint32_t form) {
            if (!form) return form;
            const auto local = form >> 24; std::string owner;
            if (local == document.masters().size()) owner = source.filename().string();
            else if (local < document.masters().size()) owner = document.masters()[local];
            else throw std::runtime_error("ESM references a FormID outside its master table");
            const auto match = std::find_if(canonical.begin(), canonical.end(), [&](const auto& n) { return folded(n) == folded(owner); });
            if (match == canonical.end()) throw std::runtime_error("Selected ESM omits a transitive master: " + owner);
            return (static_cast<std::uint32_t>(match - canonical.begin()) << 24) | (form & 0xFFFFFFu);
        };
        for (auto m : document.models())
        {
            m.formId = remap(m.formId); m.sourceEsm = source.string();
            remapPapyrusBindings(m.scripts, remap);
            if (const auto previous = models.find(m.formId); previous != models.end())
            {
                if (previous->second.kind != m.kind) throw std::runtime_error("ESM overlay changes an existing model FormID's record type");
                if (AssetResolver::normalizePath(m.path) == AssetResolver::normalizePath(previous->second.path))
                {
                    AssetResolver project(source);
                    if (!project.resolveLoose(m.path) && !project.resolveLoose("meshes/" + m.path))
                        m.sourceEsm = previous->second.sourceEsm;
                }
            }
            models[m.formId] = std::move(m);
        }
        if (remap(document.worldFormId()) != result.worldFormId) continue;
        for (auto r : document.references())
        {
            r.placement.formId = remap(r.placement.formId); r.placement.instanceId = r.placement.formId;
            r.placement.baseObject = remap(r.placement.baseObject); r.placement.materialSwap = remap(r.placement.materialSwap);
            r.ownerCellFormId = remap(r.ownerCellFormId);
            remapPapyrusBindings(r.scripts, remap);
            if (r.deleted) refs.erase(r.placement.formId);
            else refs[r.placement.formId] = std::move(r);
        }
    }
    for (auto& [id, m] : models) result.models.push_back(std::move(m));
    for (auto& [id, r] : refs) result.references.push_back(std::move(r));
    return result;
}

std::filesystem::path createEditorOverlay(const std::filesystem::path& master)
{
    EsmDocument source(master); auto target = master.parent_path() / (master.stem().string() + ".openf4.esm");
    if (std::filesystem::exists(target)) return target;
    Bytes header; scalar(header, cc("HEDR"), std::array<std::uint32_t, 3>{0x3f733333u, 1u, 0x800u});
    textSub(header, cc("CNAM"), "OpenF4");
    for (const auto& name : source.masters()) { textSub(header, cc("MAST"), name); scalar(header, cc("DATA"), std::uint64_t{0}); }
    textSub(header, cc("MAST"), master.filename().string()); scalar(header, cc("DATA"), std::uint64_t{0});
    textSub(header, cc("SNAM"), "OPENF4_OVERLAY\nBASE " + master.filename().string());
    Bytes bytes = record(cc("TES4"), 0, header, 1);
    const auto original = source.mImpl->records.find(source.worldFormId());
    if (original == source.mImpl->records.end() || original->second->type != cc("WRLD"))
        throw std::runtime_error("Cannot create an overlay without its original WRLD record");
    std::ifstream input(master, std::ios::binary);
    // ponytail: retain localized string IDs for authoring; native export must resolve their source tables first.
    Bytes world = readFile(input, original->second->offset, static_cast<std::size_t>(original->second->size));
    append(world, group(source.worldFormId(), 1, {}));
    append(bytes, group(cc("WRLD"), 0, world));
    auto temporary = temporaryPath(target);
    try
    {
        std::ofstream out(temporary, std::ios::binary); out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        out.flush(); if (!out) throw std::runtime_error("Could not create editor overlay"); out.close(); atomicReplace(target, temporary);
    }
    catch (...) { std::error_code ignored; std::filesystem::remove(temporary, ignored); throw; }
    return target;
}
}
