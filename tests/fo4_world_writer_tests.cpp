#include "assets/fo4_world_writer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    using namespace openf4::assets;
    using Bytes = std::vector<std::byte>;

    void require(bool condition, const char* message)
    {
        if (!condition) throw std::runtime_error(message);
    }

    template<class T> T number(std::span<const std::byte> data, std::size_t at)
    {
        require(at <= data.size() && sizeof(T) <= data.size() - at, "Independent reader: truncated number");
        T value; std::memcpy(&value, data.data() + at, sizeof(value)); return value;
    }

    std::string signature(std::span<const std::byte> data, std::size_t at)
    {
        require(at <= data.size() && 4 <= data.size() - at, "Independent reader: truncated signature");
        return {reinterpret_cast<const char*>(data.data() + at), 4};
    }

    Bytes read(const std::filesystem::path& path)
    {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        require(bool(file) && file.tellg() >= 0, "Could not read test plugin");
        Bytes result(static_cast<std::size_t>(file.tellg())); file.seekg(0);
        require(bool(file.read(reinterpret_cast<char*>(result.data()), result.size())), "Truncated test plugin");
        return result;
    }

    struct Node
    {
        std::string type;
        std::uint32_t id = 0, flags = 0, groupType = 0;
        std::span<const std::byte> body;
        std::vector<Node> children;
    };

    std::vector<Node> parse(std::span<const std::byte> data, unsigned depth = 0)
    {
        require(depth < 12, "Independent reader: excessive hierarchy");
        std::vector<Node> result;
        for (std::size_t p = 0; p < data.size();)
        {
            require(data.size() - p >= 24, "Independent reader: partial header");
            Node node; node.type = signature(data, p);
            const std::size_t storedSize = number<std::uint32_t>(data, p + 4);
            const std::size_t size = node.type == "GRUP" ? storedSize : storedSize + 24;
            require(size >= 24 && size <= data.size() - p, "Independent reader: record/group size exceeds parent");
            node.body = data.subspan(p + 24, size - 24);
            if (node.type == "GRUP")
            {
                node.id = number<std::uint32_t>(data, p + 8);
                node.groupType = number<std::uint32_t>(data, p + 12);
                node.children = parse(node.body, depth + 1);
            }
            else
            {
                node.flags = number<std::uint32_t>(data, p + 8);
                node.id = number<std::uint32_t>(data, p + 12);
                require(number<std::uint16_t>(data, p + 20) == 131, "Wrong FO4 form version");
            }
            result.push_back(std::move(node)); p += size;
        }
        return result;
    }

    using Field = std::pair<std::string, std::span<const std::byte>>;
    std::vector<Field> fields(const Node& node)
    {
        std::vector<Field> result;
        for (std::size_t p = 0; p < node.body.size();)
        {
            require(node.body.size() - p >= 6, "Independent reader: partial field header");
            const auto size = number<std::uint16_t>(node.body, p + 4);
            require(size <= node.body.size() - p - 6, "Independent reader: field exceeds record");
            result.emplace_back(signature(node.body, p), node.body.subspan(p + 6, size)); p += 6 + size;
        }
        return result;
    }

    std::span<const std::byte> field(const Node& node, const std::string& name)
    {
        for (const auto& [key, value] : fields(node)) if (key == name) return value;
        throw std::runtime_error("Missing field: " + name);
    }

    bool hasField(const Node& node, const std::string& name)
    {
        for (const auto& [key, value] : fields(node)) if (key == name) return true;
        return false;
    }

    std::string stringField(const Node& node, const std::string& name)
    {
        const auto bytes = field(node, name);
        require(!bytes.empty() && bytes.back() == std::byte{}, "Unterminated native string");
        return {reinterpret_cast<const char*>(bytes.data()), bytes.size() - 1};
    }

    void flatten(const std::vector<Node>& nodes, std::vector<const Node*>& all)
    {
        for (const auto& node : nodes) { all.push_back(&node); flatten(node.children, all); }
    }

    const Node& byId(const std::vector<const Node*>& nodes, std::uint32_t id)
    {
        for (const auto* node : nodes) if (node->type != "GRUP" && node->id == id) return *node;
        throw std::runtime_error("Missing record ID");
    }

    float height(CellCoord coord, int x, int y)
    {
        return float(8 * (coord.x * 32 + x + 2 * (coord.y * 32 + y)));
    }

    LandscapeCell terrain(CellCoord coord)
    {
        LandscapeCell cell; cell.cell = coord; cell.hasHeights = true; cell.hasColors = coord == CellCoord{0, 0};
        for (int y = 0; y < 33; ++y) for (int x = 0; x < 33; ++x)
        {
            cell.heights[y * 33 + x] = height(coord, x, y);
            cell.colors[y * 33 + x] = {std::uint8_t(x), std::uint8_t(y), 91};
        }
        return cell;
    }

    Fo4WorldWriteRequest fixture()
    {
        Fo4WorldWriteRequest request; request.editorId = "OpenF4NativeWorldProof";
        request.displayName = "Native terrain and objects"; request.worldFormId = 0x01000800;
        request.persistentCellFormId = 0x01000801; request.waterHeight = -2048;
        std::uint32_t id = 0x01000820;
        for (const CellCoord coord : {CellCoord{-1, -1}, {0, -1}, {-1, 0}, {0, 0}, {7, 39}, {-33, 8}})
        { request.cells.push_back({coord, id, id + 1, coord == CellCoord{-1, -1}}); id += 2; }
        request.loadCell = [](CellCoord coord) { return terrain(coord); };
        request.models = {
            {0x01000900, ESMModelKind::staticObject, "ProofDoorway", "meshes/proof/door.nif", {-96, -24, 0, 96, 24, 192}},
            {0x01000901, ESMModelKind::movableStatic, "ProofProp", "meshes/proof/prop.nif", {-32, -32, 0, 32, 32, 64}, "Readable prop name"}};
        ESMReference door; door.editorId = "PlacedProofDoor"; auto& p = door.placement;
        p.formId = 0x01000A00; p.baseObject = 0x01000900; p.cell = {-1, -1};
        p.position[0] = -2048; p.position[1] = -2048; p.position[2] = 96;
        p.rotation[0] = 0.25f; p.rotation[1] = -0.5f; p.rotation[2] = 1.125f; p.scale = 1.5f;
        ESMReference prop; prop.editorId = "PlacedProofProp"; prop.placement.formId = 0x01000A01;
        prop.placement.baseObject = 0x01000901; prop.placement.cell = {0, 0};
        prop.placement.position[0] = 1000; prop.placement.position[1] = 1500; prop.placement.position[2] = 256;
        request.references = {door, prop}; request.arrival.editorId = "ProofArrival";
        request.arrival.placement.formId = 0x01000A02; request.arrival.placement.baseObject = fo4ArrivalMarker;
        request.arrival.placement.cell = {0, 0}; request.arrival.placement.position[0] = 512;
        request.arrival.placement.position[1] = 512; request.arrival.placement.position[2] = 512;
        return request;
    }

    std::array<float, 1089> decodeHeights(const Node& node)
    {
        const auto data = field(node, "VHGT"); require(data.size() == 1096, "Native VHGT length");
        std::array<float, 1089> result{};
        float previousFirst = number<float>(data, 0);
        for (int y = 0; y < 33; ++y)
        {
            float h = previousFirst;
            for (int x = 0; x < 33; ++x)
            {
                h += number<std::int8_t>(data, 4 + y * 33 + x);
                result[y * 33 + x] = h * 8;
                if (x == 0) previousFirst = h;
            }
        }
        require(data[1093] == std::byte{} && data[1094] == std::byte{} && data[1095] == std::byte{}, "VHGT trailer");
        return result;
    }

    void structure(const std::filesystem::path& root)
    {
        const auto request = fixture(); const auto path = root / "native.esp";
        writeFo4WorldPlugin(path, request); const auto bytes = read(path); const auto roots = parse(bytes);
        require(roots.size() == 4 && roots[0].type == "TES4" && roots[0].flags == 0 && roots[0].id == 0,
            "Plugin must be non-master, non-localized TES4 plus STAT/MSTT/WRLD groups");
        require(stringField(roots[0], "MAST") == "Fallout4.esm", "Wrong native master");
        std::vector<const Node*> all; flatten(roots, all);
        require(number<float>(field(roots[0], "HEDR"), 0) == 1.0f, "FO4 HEDR version");
        require(number<std::uint32_t>(field(roots[0], "HEDR"), 4) == all.size() - 1, "HEDR includes records and groups");
        require(number<std::uint32_t>(field(roots[0], "HEDR"), 8) == 0xA03, "HEDR next local object ID");
        std::set<std::uint32_t> seen;
        for (const auto* node : all) if (node->type != "GRUP") require(seen.insert(node->id).second, "Duplicate record IDs");
        const auto& world = byId(all, request.worldFormId);
        require(world.type == "WRLD" && stringField(world, "FULL") == request.displayName, "WRLD identity");
        require(number<float>(field(world, "DNAM"), 0) == -2048 && number<float>(field(world, "DNAM"), 4) == -2048,
            "WRLD land and water defaults are DNAM, not only NAM4");
        require(number<float>(field(world, "NAM4"), 0) == -2048, "LOD water height");
        require(number<std::uint32_t>(field(world, "CNAM"), 0) == 0x15F
            && number<std::uint32_t>(field(world, "NAM2"), 0) == 0x18, "Native stock climate/water");
        require(number<float>(field(world, "ONAM"), 0) == 1 && number<float>(field(world, "NAMA"), 0) == 1,
            "Native world required scale fields");
        require(field(world, "NAM0").size() == 8 && field(world, "NAM9").size() == 8, "Native world bounds");
        const auto& persistentCell = byId(all, request.persistentCellFormId);
        require(persistentCell.flags == 0x400 && !hasField(persistentCell, "XCLC"), "Persistent world CELL is not an exterior grid cell");
        const auto& arrival = byId(all, request.arrival.placement.formId);
        require(arrival.flags == 0x400 && number<std::uint32_t>(field(arrival, "NAME"), 0) == 0x34,
            "Persistent XMarkerHeading arrival");
        const auto& worldGroup = roots[3];
        require(worldGroup.groupType == 0 && signature(std::as_bytes(std::span(&worldGroup.id, 1)), 0) == "WRLD",
            "WRLD top group");
        require(worldGroup.children.size() == 2 && worldGroup.children[1].groupType == 1
            && worldGroup.children[1].id == request.worldFormId, "WRLD children label");
        const auto& worldChildren = worldGroup.children[1].children;
        require(worldChildren[0].id == request.persistentCellFormId && worldChildren[1].groupType == 6
            && worldChildren[1].children[0].groupType == 8
            && worldChildren[1].children[0].children[0].id == request.arrival.placement.formId, "Persistent hierarchy");

        std::map<CellCoord, const Node*> lands;
        std::size_t cellCount = 0, refCount = 0;
        for (std::size_t bi = 2; bi < worldChildren.size(); ++bi)
        {
            const auto& block = worldChildren[bi]; require(block.groupType == 4, "Exterior block");
            for (const auto& subblock : block.children)
            {
                require(subblock.groupType == 5, "Exterior sub-block");
                require(subblock.children.size() % 2 == 0, "Each CELL has one children group");
                for (std::size_t ci = 0; ci < subblock.children.size(); ci += 2)
                {
                    const auto& cell = subblock.children[ci]; require(cell.type == "CELL", "Exterior CELL record");
                    const auto grid = field(cell, "XCLC"); require(grid.size() == 12, "XCLC size");
                    const CellCoord coord{number<std::int32_t>(grid, 0), number<std::int32_t>(grid, 4)};
                    const auto sourceCell = std::ranges::find_if(request.cells, [&](const auto& c) { return c.coord == coord; });
                    require(sourceCell != request.cells.end(), "Undeclared exterior cell");
                    require(number<std::uint16_t>(field(cell, "DATA"), 0) == (sourceCell->hasWater ? 2 : 0)
                        && hasField(cell, "XCLW") == sourceCell->hasWater, "CELL water presence must be preserved");
                    if (sourceCell->hasWater) require(number<float>(field(cell, "XCLW"), 0) == -2048, "Authored water height");
                    if (coord == CellCoord{-33, 8})
                        require(height(coord, 0, 0) < request.waterHeight
                            && number<std::uint16_t>(field(cell, "DATA"), 0) == 0, "Deep dry terrain must not gain water");
                    const auto expectedLabel = [&](int divisor) {
                        return std::uint32_t(std::uint16_t(int(std::floor(double(coord.y) / divisor))))
                            | std::uint32_t(std::uint16_t(int(std::floor(double(coord.x) / divisor)))) << 16;
                    };
                    require(block.id == expectedLabel(32) && subblock.id == expectedLabel(8),
                        "Native group labels must be signed floor Y-low/X-high, including negative coordinates");
                    if (coord == CellCoord{7, 39}) require(block.id == 1 && subblock.id == 4, "Independent native master label example");
                    const auto& children = subblock.children[ci + 1];
                    require(children.groupType == 6 && children.id == cell.id && children.children.size() == 1,
                        "CELL children ownership");
                    const auto& temporary = children.children[0];
                    require(temporary.groupType == 9 && temporary.id == cell.id && temporary.children[0].type == "LAND",
                        "Temporary LAND must precede placed objects");
                    lands[coord] = &temporary.children[0]; ++cellCount;
                    for (std::size_t ri = 1; ri < temporary.children.size(); ++ri)
                    {
                        const auto& ref = temporary.children[ri]; require(ref.type == "REFR" && ref.flags == 0, "Temporary reference flags");
                        const auto expected = std::ranges::find_if(request.references, [&](const auto& r) { return r.placement.formId == ref.id; });
                        require(expected != request.references.end() && expected->placement.cell == coord, "Reference cell ownership");
                        const auto transform = field(ref, "DATA"); require(transform.size() == 24, "REFR transform size");
                        for (int i = 0; i < 3; ++i)
                            require(number<float>(transform, i * 4) == expected->placement.position[i]
                                && number<float>(transform, 12 + i * 4) == expected->placement.rotation[i],
                                "Bethesda reference transforms must not be converted twice");
                        if (expected->placement.scale != 1) require(number<float>(field(ref, "XSCL"), 0) == 1.5f, "Reference scale");
                        ++refCount;
                    }
                }
            }
        }
        require(cellCount == request.cells.size() && refCount == 2, "Missing cells or placed references");
        for (const auto& [coord, land] : lands)
        {
            const auto decoded = decodeHeights(*land); const auto source = terrain(coord);
            require(decoded == source.heights, "VHGT independent semantic round trip");
            const auto normals = field(*land, "VNML"); require(normals.size() == 3267, "VNML size");
            for (std::size_t i = 0; i < 1089; ++i)
                require(number<std::int8_t>(normals, i * 3) == -8 && number<std::int8_t>(normals, i * 3 + 1) == -16
                    && number<std::int8_t>(normals, i * 3 + 2) == 126, "Slope normals including world boundaries");
            require(number<std::uint32_t>(field(*land, "DATA"), 0) == (source.hasColors ? 7u : 5u), "LAND flags match fields");
            require(hasField(*land, "VCLR") == source.hasColors, "Optional LAND colors");
            if (source.hasColors)
            {
                const auto colors = field(*land, "VCLR"); require(colors.size() == 3267, "VCLR size");
                for (std::size_t i = 0; i < 1089; ++i)
                    require(number<std::uint8_t>(colors, i * 3) == i % 33
                        && number<std::uint8_t>(colors, i * 3 + 1) == i / 33
                        && number<std::uint8_t>(colors, i * 3 + 2) == 91, "VCLR values");
            }
            std::set<unsigned> quadrants;
            for (const auto& [name, data] : fields(*land)) if (name == "BTXT")
            {
                require(data.size() == 8 && number<std::uint32_t>(data, 0) == 0xAB07E
                    && number<std::uint8_t>(data, 5) == 0 && number<std::int16_t>(data, 6) == -1, "Native base texture encoding");
                quadrants.insert(number<std::uint8_t>(data, 4));
            }
            require(quadrants == std::set<unsigned>{0, 1, 2, 3}, "Four default material quadrants");
        }
        const auto& stat = byId(all, 0x01000900); const auto& mstt = byId(all, 0x01000901);
        require(stat.type == "STAT" && field(stat, "OBND").size() == 12 && field(stat, "DNAM").size() == 16,
            "STAT required fields");
        require(mstt.type == "MSTT" && field(mstt, "DATA").size() == 1 && hasField(mstt, "FULL"), "MSTT required fields");
        require(stringField(mstt, "FULL") == "Readable prop name", "MSTT display name lost during EDID namespacing");
        require(stringField(stat, "MODL") == "proof\\door.nif", "MODL strips meshes prefix");

        auto reversed = request; std::ranges::reverse(reversed.cells); std::ranges::reverse(reversed.references);
        std::ranges::reverse(reversed.models); writeFo4WorldPlugin(root / "reordered.esp", reversed);
        require(read(root / "reordered.esp") == bytes, "Output order must be deterministic");
    }

    void rejection(const std::filesystem::path& root)
    {
        int serial = 0;
        const auto rejects = [&](auto change, const char* message) {
            auto request = fixture(); change(request); const auto path = root / ("invalid-" + std::to_string(++serial) + ".esp");
            bool rejected = false;
            try { writeFo4WorldPlugin(path, request); } catch (const std::exception&) { rejected = true; }
            require(rejected, message); require(!std::filesystem::exists(path), "Failed write left an incomplete plugin");
        };
        rejects([](auto& r) { r.cells[0].cellFormId = r.worldFormId; }, "Duplicate FormID accepted");
        rejects([](auto& r) { r.worldFormId = 0x02000800; }, "Undeclared self prefix accepted");
        rejects([](auto& r) { r.worldFormId = 0x01FFFFFF; }, "Exhausted FormID accepted");
        rejects([](auto& r) { r.cells.push_back(r.cells.front()); }, "Duplicate CELL accepted");
        rejects([](auto& r) { r.cells[0].coord.x = -32769; }, "Out of range native CELL accepted");
        rejects([](auto& r) { r.references[0].placement.cell = {0, 0}; }, "Wrong reference ownership accepted");
        rejects([](auto& r) { r.references[0].placement.baseObject = 0x01007777; }, "Unresolved base accepted");
        rejects([](auto& r) { r.references[0].placement.scale = 0; }, "Zero scale accepted");
        rejects([](auto& r) { r.references[0].placement.position[2] = std::numeric_limits<float>::quiet_NaN(); }, "NaN transform accepted");
        rejects([](auto& r) { r.references[0].placement.hasLocalTransform = true; }, "Collection-local transform silently flattened");
        rejects([](auto& r) { r.references[0].placement.materialSwap = 1; }, "Material swap silently lost");
        rejects([](auto& r) { r.references[0].deleted = true; }, "Deleted reference exported");
        rejects([](auto& r) { r.models[0].modelPath = "meshes/../outside.nif"; }, "Unsafe asset path accepted");
        rejects([](auto& r) { r.models[0].modelPath = "meshes/source.glb"; }, "Non-native model accepted");
        rejects([](auto& r) { r.models[0].bounds[0] = 100; }, "Reversed bounds accepted");
        rejects([](auto& r) { r.climate = 0x0100015F; }, "Undeclared climate master accepted");
        rejects([](auto& r) { r.editorId = "Bad ID"; }, "Invalid editor ID accepted");
        rejects([](auto& r) { r.models[0].editorId = r.editorId; }, "Duplicate editor ID accepted");
        rejects([](auto& r) { r.arrival.placement.baseObject = 0x3B; }, "Wrong arrival marker accepted");
        rejects([](auto& r) { r.loadCell = [](CellCoord) -> std::optional<LandscapeCell> { return {}; }; }, "Missing LAND accepted");
        rejects([](auto& r) { r.loadCell = [](CellCoord c) { auto land = terrain(c); land.heights[45] += 1; return land; }; }, "Unquantized LAND accepted");
        rejects([](auto& r) { r.loadCell = [](CellCoord c) { auto land = terrain(c); land.heights[45] = std::numeric_limits<float>::infinity(); return land; }; }, "Infinite LAND accepted");
        rejects([](auto& r) { r.loadCell = [](CellCoord c) { auto land = terrain(c); land.heights[45] += 2048; return land; }; }, "VHGT overflow accepted");
        rejects([](auto& r) { r.loadCell = [](CellCoord c) { auto land = terrain(c); if (c == CellCoord{0, 0}) land.heights[0] += 8; return land; }; }, "Four-cell corner seam accepted");
        rejects([](auto& r) { r.loadCell = [](CellCoord c) { auto land = terrain(c); if (c == CellCoord{0, 0}) land.heights[33 * 4] += 8; return land; }; }, "Edge seam accepted");
        rejects([](auto& r) { r.loadCell = [](CellCoord c) { auto land = terrain(c); land.textureLayers.push_back({}); return land; }; }, "Alpha layer silently discarded");
        rejects([](auto& r) { r.loadCell = [](CellCoord c) { auto land = terrain(c); land.baseTextures[0] = 0x01000800; return land; }; }, "Unresolved landscape texture accepted");
        rejects([](auto& r) { r.loadCell = [](CellCoord c) { if (c == CellCoord{7, 39}) throw std::runtime_error("cancelled"); return terrain(c); }; }, "Cancelled stream accepted");

        const auto existing = root / "native.esp"; const auto before = read(existing);
        bool rejected = false; try { writeFo4WorldPlugin(existing, fixture()); } catch (const std::exception&) { rejected = true; }
        require(rejected && read(existing) == before, "Writer overwrote an existing plugin");
    }

    void deltaLimitsAndLazyReads(const std::filesystem::path& root)
    {
        for (const int delta : {-128, 127})
        {
            auto request = fixture(); request.cells = {{{0, 0}, 0x01000820, 0x01000821}}; request.references.clear();
            request.loadCell = [=](CellCoord coord) { auto cell = terrain(coord);
                for (int y = 0; y < 33; ++y) for (int x = 0; x < 33; ++x) cell.heights[y * 33 + x] = float(x * delta * 8);
                return cell; };
            const auto path = root / ("delta" + std::to_string(delta) + ".esp"); writeFo4WorldPlugin(path, request);
            const auto bytes = read(path); const auto nodes = parse(bytes); std::vector<const Node*> all; flatten(nodes, all);
            require(decodeHeights(byId(all, 0x01000821))[32] == float(32 * delta * 8), "Signed delta endpoint lost");
        }
        auto request = fixture(); request.cells.clear(); request.references.clear();
        std::set<CellCoord> coords; std::uint32_t id = 0x01002000;
        for (int y = -4; y < 4; ++y) for (int x = -8; x < 8; ++x)
        { request.cells.push_back({{x, y}, id, id + 1}); id += 2; coords.insert({x, y}); }
        std::size_t calls = 0;
        request.loadCell = [&](CellCoord coord) {
            require(coords.contains(coord), "Lazy reader requested undeclared cell"); ++calls; return terrain(coord);
        };
        writeFo4WorldPlugin(root / "lazy.esp", request);
        require(calls >= request.cells.size() && calls <= request.cells.size() * 9, "Unexpected terrain read amplification");
    }
}

int main()
{
    const auto root = std::filesystem::temp_directory_path()
        / ("openf4-native-world-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    try
    {
        require(std::filesystem::create_directory(root), "Cannot create isolated test directory");
        structure(root); rejection(root); deltaLimitsAndLazyReads(root);
        std::filesystem::remove_all(root);
        std::cout << "Native world writer tests passed (structure only; game/navigation/LOD unverified).\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << "\nRetained fixtures: " << root.string() << '\n'; return 1;
    }
}
