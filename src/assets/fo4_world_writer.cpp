#include "assets/fo4_world_writer.hpp"

#include <algorithm>
#include <cctype>
#include <bit>
#include <cmath>
#include <cstdio>
#include <limits>
#include <map>
#include <set>
#include <span>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <type_traits>

namespace openf4::assets
{
    namespace
    {
        using Bytes = std::vector<std::byte>;
        constexpr std::uint32_t persistent = 0x400;
        constexpr double cellSize = 4096.0;
        constexpr std::size_t side = LandscapeCell::side;
        static_assert(std::endian::native == std::endian::little);

        [[noreturn]] void fail(const std::string& message)
        {
            throw std::runtime_error("Fallout 4 world export: " + message);
        }

        template<class T> void put(Bytes& bytes, const T& value)
        {
            static_assert(std::is_trivially_copyable_v<T>);
            const auto* begin = reinterpret_cast<const std::byte*>(&value);
            bytes.insert(bytes.end(), begin, begin + sizeof(value));
        }

        constexpr std::uint32_t tag(std::string_view value)
        {
            return std::uint32_t(value[0]) | std::uint32_t(value[1]) << 8
                | std::uint32_t(value[2]) << 16 | std::uint32_t(value[3]) << 24;
        }

        void sub(Bytes& bytes, std::string_view name, std::span<const std::byte> data)
        {
            if (data.size() > UINT16_MAX) fail("subrecord exceeds the small-world format gate");
            put(bytes, tag(name)); put(bytes, static_cast<std::uint16_t>(data.size()));
            bytes.insert(bytes.end(), data.begin(), data.end());
        }

        template<class T> void scalar(Bytes& bytes, std::string_view name, const T& value)
        {
            sub(bytes, name, {reinterpret_cast<const std::byte*>(&value), sizeof(value)});
        }

        void text(Bytes& bytes, std::string_view name, std::string_view value)
        {
            if (value.find('\0') != std::string_view::npos || value.size() >= UINT16_MAX)
                fail("invalid string field");
            Bytes data;
            for (const char c : value) data.push_back(std::byte(c));
            data.push_back(std::byte{}); sub(bytes, name, data);
        }

        void editorId(std::string_view value)
        {
            const auto alpha = [](char c) { return (c >= 'A' && c <= 'Z')
                || (c >= 'a' && c <= 'z') || c == '_'; };
            if (value.empty() || value.size() > 128 || !alpha(value.front())
                || !std::ranges::all_of(value, [&](char c) { return alpha(c) || (c >= '0' && c <= '9'); }))
                fail("editor IDs must be 1-128 ASCII letters, digits or underscores and not start with a digit");
        }

        std::string folded(std::string value)
        {
            for (char& c : value) if (c >= 'A' && c <= 'Z') c += 'a' - 'A';
            return value;
        }

        std::string nativeModelPath(std::string value)
        {
            std::ranges::replace(value, '\\', '/');
            const auto lower = folded(value);
            if (!lower.starts_with("meshes/") || !lower.ends_with(".nif") || value.size() > 259
                || std::ranges::any_of(value, [](unsigned char c) { return c < 32 || c > 126 || c == ':'; }))
                fail("model path must be a relative ASCII meshes/...nif path");
            for (std::size_t begin = 0; begin < value.size();)
            {
                const auto end = value.find('/', begin);
                const auto part = value.substr(begin, end == std::string::npos ? end : end - begin);
                if (part.empty() || part == "." || part == "..") fail("unsafe model path");
                if (end == std::string::npos) break;
                begin = end + 1;
            }
            value.erase(0, 7);
            std::ranges::replace(value, '/', '\\');
            return value;
        }

        int floorDivide(int value, int divisor)
        {
            return value / divisor - (value % divisor < 0 ? 1 : 0);
        }

        std::uint32_t groupLabel(CellCoord coord, int divisor)
        {
            // Native exterior group labels store Y first; OpenF4's private writer differs.
            return std::uint16_t(floorDivide(coord.y, divisor))
                | std::uint32_t(std::uint16_t(floorDivide(coord.x, divisor))) << 16;
        }

        auto groupOrder(CellCoord coord)
        {
            return std::tuple{floorDivide(coord.x, 32), floorDivide(coord.y, 32),
                floorDivide(coord.x, 8), floorDivide(coord.y, 8), coord.x, coord.y};
        }

        std::string cellName(std::string_view world, CellCoord coord)
        {
            const auto number = [](int value) { return (value < 0 ? "m" : "p") + std::to_string(std::abs(value)); };
            return std::string(world) + "_Cell_" + number(coord.x) + "_" + number(coord.y);
        }

        class Output
        {
        public:
            explicit Output(const std::filesystem::path& path) : mPath(path)
            {
#ifdef _WIN32
                if (_wfopen_s(&mFile, path.c_str(), L"w+bx") != 0) mFile = nullptr;
#else
                mFile = std::fopen(path.c_str(), "w+bx");
#endif
                if (!mFile) fail("cannot create a new plugin: " + path.string());
            }
            ~Output()
            {
                if (mFile) std::fclose(mFile);
                if (!mFinished) { std::error_code ignored; std::filesystem::remove(mPath, ignored); }
            }
            void write(std::span<const std::byte> bytes)
            {
                if (std::fwrite(bytes.data(), 1, bytes.size(), mFile) != bytes.size()) fail("plugin write failed");
            }
            std::uint64_t position() const
            {
#ifdef _WIN32
                const auto result = _ftelli64(mFile);
#else
                const auto result = ftello(mFile);
#endif
                if (result < 0) fail("cannot read plugin stream position");
                return static_cast<std::uint64_t>(result);
            }
            void patch(std::uint64_t at, std::uint32_t value)
            {
                const auto end = position(); seek(at);
                const auto* bytes = reinterpret_cast<const std::byte*>(&value);
                write({bytes, sizeof(value)}); seek(end);
            }
            void record(std::string_view type, std::uint32_t id, const Bytes& bytes, std::uint32_t flags = 0)
            {
                if (bytes.size() > UINT32_MAX) fail("record exceeds native 32-bit size");
                Bytes header; put(header, tag(type)); put(header, static_cast<std::uint32_t>(bytes.size()));
                put(header, flags); put(header, id); put(header, std::uint32_t{});
                put(header, std::uint16_t{131}); put(header, std::uint16_t{});
                write(header); write(bytes);
                if (type != "TES4") ++count;
            }
            std::uint64_t beginGroup(std::uint32_t label, std::uint32_t type)
            {
                const auto begin = position();
                Bytes header; put(header, tag("GRUP")); put(header, std::uint32_t{});
                put(header, label); put(header, type); put(header, std::uint64_t{});
                write(header); ++count; return begin;
            }
            void endGroup(std::uint64_t begin)
            {
                const auto size = position() - begin;
                if (size > UINT32_MAX) fail("group exceeds native 4 GiB limit; export requires sharding");
                patch(begin + 4, static_cast<std::uint32_t>(size));
            }
            void finish()
            {
                if (count > UINT32_MAX) fail("too many plugin records");
                // TES4 begins with the 12-byte HEDR; count includes GRUPs, not TES4.
                patch(34, static_cast<std::uint32_t>(count));
                if (std::fflush(mFile) != 0) fail("cannot flush plugin");
                auto* file = mFile; mFile = nullptr;
                if (std::fclose(file) != 0) fail("cannot close plugin");
                mFinished = true;
            }
            std::uint64_t count = 0;
        private:
            void seek(std::uint64_t at)
            {
                if (at > std::uint64_t(INT64_MAX)) fail("plugin offset overflow");
#ifdef _WIN32
                if (_fseeki64(mFile, static_cast<std::int64_t>(at), SEEK_SET) != 0)
#else
                if (fseeko(mFile, static_cast<off_t>(at), SEEK_SET) != 0)
#endif
                    fail("cannot seek plugin stream");
            }
            std::filesystem::path mPath;
            std::FILE* mFile = nullptr;
            bool mFinished = false;
        };

        struct Validated
        {
            std::vector<Fo4WorldCell> cells;
            std::map<CellCoord, std::vector<const ESMReference*>> references;
            std::set<CellCoord> coords;
            std::uint32_t nextObjectId = 0x800;
            CellCoord minimum{INT16_MAX, INT16_MAX}, maximum{INT16_MIN, INT16_MIN};
        };

        Validated validate(const Fo4WorldWriteRequest& request)
        {
            if (request.cells.empty() || !request.loadCell) fail("no landscape cells");
            editorId(request.editorId);
            if (request.editorId.size() > 96) fail("world editor ID is too long for generated cell IDs");
            if (!std::isfinite(request.defaultLandHeight) || !std::isfinite(request.waterHeight))
                fail("non-finite world height");
            const auto stock = [](std::uint32_t id) { return id != 0 && (id >> 24) == 0; };
            if (!stock(request.defaultLandscapeTexture) || !stock(request.climate) || !stock(request.water))
                fail("world defaults must reference Fallout4.esm");
            std::set<std::uint32_t> ids, modelIds;
            std::set<std::string> names;
            Validated result;
            const auto addId = [&](std::uint32_t id) {
                if ((id >> 24) != 1 || (id & 0xffffff) < 0x800 || (id & 0xffffff) == 0xffffff
                    || !ids.insert(id).second) fail("invalid, exhausted or duplicate authored FormID");
                result.nextObjectId = std::max(result.nextObjectId, (id & 0xffffff) + 1);
            };
            const auto name = [&](const std::string& id) {
                editorId(id);
                if (!names.insert(folded(id)).second) fail("duplicate editor ID");
            };
            addId(request.worldFormId); name(request.editorId);
            addId(request.persistentCellFormId); name(request.editorId + "_Persistent");
            for (const auto& cell : request.cells)
            {
                if (cell.coord.x < INT16_MIN || cell.coord.x > INT16_MAX
                    || cell.coord.y < INT16_MIN || cell.coord.y > INT16_MAX
                    || !result.coords.insert(cell.coord).second) fail("invalid or duplicate exterior cell coordinate");
                addId(cell.cellFormId); addId(cell.landFormId); name(cellName(request.editorId, cell.coord));
                result.minimum.x = std::min(result.minimum.x, cell.coord.x);
                result.minimum.y = std::min(result.minimum.y, cell.coord.y);
                result.maximum.x = std::max(result.maximum.x, cell.coord.x);
                result.maximum.y = std::max(result.maximum.y, cell.coord.y);
            }
            for (const auto& model : request.models)
            {
                addId(model.formId); modelIds.insert(model.formId); name(model.editorId);
                if (model.kind != ESMModelKind::staticObject && model.kind != ESMModelKind::movableStatic)
                    fail("unsupported model kind");
                for (int i = 0; i < 3; ++i) if (model.bounds[i] > model.bounds[i + 3]) fail("reversed model bounds");
                (void)nativeModelPath(model.modelPath);
            }
            const auto reference = [&](const ESMReference& ref, bool arrival) {
                const auto& p = ref.placement;
                addId(p.formId);
                if (!ref.editorId.empty()) name(ref.editorId);
                if (ref.deleted || p.hasLocalTransform || p.materialSwap)
                    fail("deleted, collection-local or material-swapped references are outside this export gate");
                if (!std::isfinite(p.scale) || p.scale <= 0) fail("invalid reference scale");
                for (int i = 0; i < 3; ++i)
                    if (!std::isfinite(p.position[i]) || !std::isfinite(p.rotation[i])) fail("non-finite reference transform");
                if (!result.coords.contains(p.cell)
                    || std::floor(double(p.position[0]) / cellSize) != p.cell.x
                    || std::floor(double(p.position[1]) / cellSize) != p.cell.y)
                    fail("reference position does not belong to its exterior cell");
                if (arrival)
                {
                    if (p.baseObject != 0 && p.baseObject != fo4ArrivalMarker) fail("arrival must be XMarkerHeading");
                    if (ref.editorId.empty()) name(request.editorId + "_Arrival");
                }
                else if (!modelIds.contains(p.baseObject)) fail("reference has an unresolved base model");
            };
            reference(request.arrival, true);
            for (const auto& ref : request.references)
            {
                reference(ref, false); result.references[ref.placement.cell].push_back(&ref);
            }
            const auto scripts = [&](const PapyrusBindings& bindings) {
                (void)encodePapyrusBindings(bindings, false);
                for (const auto& script : bindings.scripts) for (const auto& property : script.properties)
                    if (const auto* object = std::get_if<PapyrusObject>(&property.value))
                    {
                        if (object->alias != -1) fail("quest-alias script properties are outside this export gate");
                        if ((object->formId >> 24) > 1 || ((object->formId >> 24) == 1 && !ids.contains(object->formId)))
                            fail("script property refers to a record not included in the export");
                    }
            };
            for (const auto& model : request.models) scripts(model.scripts);
            for (const auto& ref : request.references) scripts(ref.scripts);
            scripts(request.arrival.scripts);
            for (auto& [coord, refs] : result.references)
                std::ranges::sort(refs, [](auto* a, auto* b) { return a->placement.formId < b->placement.formId; });
            result.cells = request.cells;
            std::ranges::sort(result.cells, [](const auto& a, const auto& b) { return groupOrder(a.coord) < groupOrder(b.coord); });
            return result;
        }

        class LandscapeWindow
        {
        public:
            LandscapeWindow(const Fo4WorldWriteRequest& request, const std::set<CellCoord>& coords)
                : mRequest(request), mCoords(coords) {}

            const LandscapeCell& load(CellCoord center)
            {
                std::erase_if(mCells, [&](const auto& item) {
                    return std::abs(item.first.x - center.x) > 1 || std::abs(item.first.y - center.y) > 1;
                });
                for (int y = -1; y <= 1; ++y) for (int x = -1; x <= 1; ++x)
                {
                    const CellCoord coord{center.x + x, center.y + y};
                    if (!mCoords.contains(coord) || mCells.contains(coord)) continue;
                    auto source = mRequest.loadCell(coord);
                    if (!source || source->cell != coord || !source->hasHeights) fail("missing or mismatched LAND");
                    for (const float h : source->heights)
                        if (!std::isfinite(h) || std::fmod(h, 8.0f) != 0.0f) fail("LAND heights must be finite 8-unit steps");
                    mCells.emplace(coord, std::move(*source));
                }
                const auto& cell = mCells.at(center);
                for (const auto& [coord, neighbor] : mCells)
                {
                    const int dx = coord.x - center.x, dy = coord.y - center.y;
                    if (!dx && !dy) continue;
                    const int x0 = dx > 0 ? 32 : 0, x1 = dx < 0 ? 0 : 32;
                    const int y0 = dy > 0 ? 32 : 0, y1 = dy < 0 ? 0 : 32;
                    for (int y = y0; y <= y1; ++y) for (int x = x0; x <= x1; ++x)
                        if (cell.heights[y * side + x] != neighbor.heights[(y - dy * 32) * side + x - dx * 32])
                            fail("LAND has divergent shared edges or corners");
                }
                return cell;
            }

            std::optional<double> height(CellCoord coord, int x, int y) const
            {
                const int gx = coord.x * 32 + x, gy = coord.y * 32 + y;
                const int cx = floorDivide(gx, 32), cy = floorDivide(gy, 32);
                const int sx = gx - cx * 32, sy = gy - cy * 32;
                // A boundary sample may belong to either cell, including diagonal-only corners.
                for (int dy = 0; dy >= (sy == 0 ? -1 : 0); --dy)
                    for (int dx = 0; dx >= (sx == 0 ? -1 : 0); --dx)
                        if (const auto found = mCells.find({cx + dx, cy + dy}); found != mCells.end())
                            return found->second.heights[(sy - dy * 32) * side + sx - dx * 32];
                return std::nullopt;
            }
        private:
            const Fo4WorldWriteRequest& mRequest;
            const std::set<CellCoord>& mCoords;
            std::map<CellCoord, LandscapeCell> mCells;
        };

        Bytes landData(const LandscapeCell& cell, const LandscapeWindow& window, std::uint32_t defaultTexture)
        {
            Bytes result, normals, heights;
            scalar(result, "DATA", std::uint32_t(5u | (cell.hasColors ? 2u : 0u)));
            for (int y = 0; y < int(side); ++y) for (int x = 0; x < int(side); ++x)
            {
                const double center = cell.heights[y * side + x];
                const auto left = window.height(cell.cell, x - 1, y), right = window.height(cell.cell, x + 1, y);
                const auto down = window.height(cell.cell, x, y - 1), up = window.height(cell.cell, x, y + 1);
                const double nx = (left.value_or(center) - right.value_or(center)) / (left && right ? 256.0 : 128.0);
                const double ny = (down.value_or(center) - up.value_or(center)) / (down && up ? 256.0 : 128.0);
                const double scale = 127.0 / std::sqrt(nx * nx + ny * ny + 1.0);
                put(normals, static_cast<std::int8_t>(std::lround(nx * scale)));
                put(normals, static_cast<std::int8_t>(std::lround(ny * scale)));
                put(normals, static_cast<std::int8_t>(std::lround(scale)));
            }
            sub(result, "VNML", normals);
            put(heights, cell.heights[0] / 8.0f);
            double previousRow = cell.heights[0] / 8.0;
            for (std::size_t y = 0; y < side; ++y)
            {
                double previous = previousRow;
                for (std::size_t x = 0; x < side; ++x)
                {
                    const double value = cell.heights[y * side + x] / 8.0;
                    const double delta = value - previous;
                    if (delta < -128 || delta > 127) fail("LAND exceeds signed VHGT delta range");
                    put(heights, static_cast<std::int8_t>(delta)); previous = value;
                    if (x == 0) previousRow = value;
                }
            }
            for (int i = 0; i < 3; ++i) put(heights, std::uint8_t{});
            sub(result, "VHGT", heights);
            if (cell.hasColors)
            {
                Bytes colors;
                for (const auto color : cell.colors) { put(colors, color.r); put(colors, color.g); put(colors, color.b); }
                sub(result, "VCLR", colors);
            }
            auto textures = cell.baseTextures;
            std::set<std::uint8_t> quadrants;
            for (const auto& layer : cell.textureLayers)
            {
                // Painting is not yet authored by this gate; never discard alpha layers silently.
                if (!layer.base || layer.quadrant >= 4 || !layer.opacityPoints.empty()
                    || (layer.layer != 0 && layer.layer != UINT16_MAX) || !quadrants.insert(layer.quadrant).second)
                    fail("unsupported LAND texture layers");
                if (textures[layer.quadrant] && textures[layer.quadrant] != layer.texture) fail("conflicting LAND base textures");
                textures[layer.quadrant] = layer.texture;
            }
            for (std::uint8_t quadrant = 0; quadrant < 4; ++quadrant)
            {
                const auto texture = textures[quadrant] ? textures[quadrant] : defaultTexture;
                if (texture >> 24) fail("LAND texture requires an undeclared master");
                Bytes layer; put(layer, texture); put(layer, quadrant); put(layer, std::uint8_t{}); put(layer, std::int16_t{-1});
                sub(result, "BTXT", layer);
            }
            return result;
        }

        void writeReference(Output& output, const ESMReference& reference, std::uint32_t flags,
            std::string_view fallbackId = {})
        {
            Bytes body;
            if (!reference.editorId.empty()) text(body, "EDID", reference.editorId);
            else if (!fallbackId.empty()) text(body, "EDID", fallbackId);
            if (reference.scripts.present) sub(body, "VMAD", encodePapyrusBindings(reference.scripts, false));
            const auto& p = reference.placement;
            scalar(body, "NAME", p.baseObject ? p.baseObject : fo4ArrivalMarker);
            if (p.scale != 1.0f) scalar(body, "XSCL", p.scale);
            const std::array<float, 6> transform{p.position[0], p.position[1], p.position[2],
                p.rotation[0], p.rotation[1], p.rotation[2]};
            scalar(body, "DATA", transform);
            output.record("REFR", p.formId, body, flags);
        }
    }

    void writeFo4WorldPlugin(const std::filesystem::path& path, const Fo4WorldWriteRequest& request)
    {
        const auto plan = validate(request);
        Output output(path);
        Bytes header, hedr;
        put(hedr, 1.0f); put(hedr, std::uint32_t{}); put(hedr, plan.nextObjectId);
        sub(header, "HEDR", hedr); text(header, "CNAM", "OpenF4");
        text(header, "SNAM", "Experimental native world geometry proof; navigation and LOD are not included.");
        text(header, "MAST", "Fallout4.esm"); scalar(header, "DATA", std::uint64_t{});
        output.record("TES4", 0, header);

        for (const auto kind : {ESMModelKind::staticObject, ESMModelKind::movableStatic})
        {
            std::vector<const Fo4WorldModel*> models;
            for (const auto& model : request.models) if (model.kind == kind) models.push_back(&model);
            if (models.empty()) continue;
            std::ranges::sort(models, [](auto* a, auto* b) { return a->formId < b->formId; });
            const auto type = kind == ESMModelKind::staticObject ? "STAT" : "MSTT";
            const auto group = output.beginGroup(tag(type), 0);
            for (const auto* model : models)
            {
                Bytes body; text(body, "EDID", model->editorId);
                if (model->scripts.present) sub(body, "VMAD", encodePapyrusBindings(model->scripts, false));
                scalar(body, "OBND", model->bounds);
                if (kind == ESMModelKind::movableStatic)
                    text(body, "FULL", model->displayName.empty() ? model->editorId : model->displayName);
                text(body, "MODL", nativeModelPath(model->modelPath));
                if (kind == ESMModelKind::movableStatic) scalar(body, "DATA", std::uint8_t{1});
                else scalar(body, "DNAM", std::array<std::uint32_t, 4>{std::bit_cast<std::uint32_t>(90.0f), 0, 0, 0});
                output.record(type, model->formId, body);
            }
            output.endGroup(group);
        }

        const auto top = output.beginGroup(tag("WRLD"), 0);
        Bytes world; text(world, "EDID", request.editorId);
        text(world, "FULL", request.displayName.empty() ? request.editorId : request.displayName);
        scalar(world, "CNAM", request.climate); scalar(world, "NAM2", request.water);
        scalar(world, "NAM3", request.water); scalar(world, "NAM4", request.waterHeight);
        scalar(world, "DNAM", std::array<float, 2>{request.defaultLandHeight, request.waterHeight});
        scalar(world, "ONAM", std::array<float, 4>{1.0f, 0.0f, 0.0f, 0.0f});
        scalar(world, "NAMA", 1.0f); scalar(world, "DATA", std::uint8_t{1});
        scalar(world, "NAM0", std::array<float, 2>{float(plan.minimum.x * cellSize), float(plan.minimum.y * cellSize)});
        scalar(world, "NAM9", std::array<float, 2>{float((plan.maximum.x + 1) * cellSize), float((plan.maximum.y + 1) * cellSize)});
        output.record("WRLD", request.worldFormId, world);
        const auto children = output.beginGroup(request.worldFormId, 1);
        Bytes persistentCell; text(persistentCell, "EDID", request.editorId + "_Persistent");
        scalar(persistentCell, "DATA", std::uint16_t{});
        output.record("CELL", request.persistentCellFormId, persistentCell, persistent);
        const auto persistentChildren = output.beginGroup(request.persistentCellFormId, 6);
        const auto persistentRefs = output.beginGroup(request.persistentCellFormId, 8);
        writeReference(output, request.arrival, persistent, request.editorId + "_Arrival");
        output.endGroup(persistentRefs); output.endGroup(persistentChildren);

        LandscapeWindow landscape(request, plan.coords);
        std::size_t next = 0;
        while (next < plan.cells.size())
        {
            const auto blockLabel = groupLabel(plan.cells[next].coord, 32);
            const auto block = output.beginGroup(blockLabel, 4);
            do
            {
                const auto subblockLabel = groupLabel(plan.cells[next].coord, 8);
                const auto subblock = output.beginGroup(subblockLabel, 5);
                do
                {
                    const auto& cell = plan.cells[next++];
                    Bytes body; text(body, "EDID", cellName(request.editorId, cell.coord));
                    scalar(body, "DATA", std::uint16_t(cell.hasWater ? 2 : 0));
                    scalar(body, "XCLC", std::array<std::int32_t, 3>{cell.coord.x, cell.coord.y, 0});
                    if (cell.hasWater) scalar(body, "XCLW", request.waterHeight);
                    output.record("CELL", cell.cellFormId, body);
                    const auto cellChildren = output.beginGroup(cell.cellFormId, 6);
                    const auto temporary = output.beginGroup(cell.cellFormId, 9);
                    const auto& terrain = landscape.load(cell.coord);
                    output.record("LAND", cell.landFormId, landData(terrain, landscape, request.defaultLandscapeTexture));
                    if (const auto refs = plan.references.find(cell.coord); refs != plan.references.end())
                        for (const auto* ref : refs->second) writeReference(output, *ref, 0);
                    output.endGroup(temporary); output.endGroup(cellChildren);
                } while (next < plan.cells.size() && groupLabel(plan.cells[next].coord, 8) == subblockLabel);
                output.endGroup(subblock);
            } while (next < plan.cells.size() && groupLabel(plan.cells[next].coord, 32) == blockLabel);
            output.endGroup(block);
        }
        output.endGroup(children); output.endGroup(top); output.finish();
    }

    void createFo4OverridePlugin(const std::filesystem::path& path, const std::vector<std::string>& masters,
        const std::string& description)
    {
        if (masters.empty()) throw std::runtime_error("A new plugin needs at least one master");
        if (masters.size() > 254) throw std::runtime_error("Too many masters for one plugin");
        const auto extension = [&] { auto e = path.extension().string(); for (auto& c : e) c = char(std::tolower((unsigned char)c)); return e; }();
        if (extension != ".esp") throw std::runtime_error("New plugins must use the .esp extension");
        if (std::filesystem::exists(path)) throw std::runtime_error("Plugin already exists: " + path.string());
        for (std::size_t i = 0; i < masters.size(); ++i)
        {
            if (masters[i].empty() || masters[i].find_first_of("/\\:") != std::string::npos)
                throw std::runtime_error("Invalid master filename: " + masters[i]);
            for (std::size_t j = 0; j < i; ++j)
            {
                auto a = masters[i], b = masters[j];
                for (auto& c : a) c = char(std::tolower((unsigned char)c));
                for (auto& c : b) c = char(std::tolower((unsigned char)c));
                if (a == b) throw std::runtime_error("Duplicate master: " + masters[i]);
            }
        }
        Output output(path);
        Bytes header, hedr;
        put(hedr, 1.0f); put(hedr, std::uint32_t{}); put(hedr, std::uint32_t{0x800});
        sub(header, "HEDR", hedr); text(header, "CNAM", "OpenF4");
        if (!description.empty()) text(header, "SNAM", description);
        for (const auto& master : masters) { text(header, "MAST", master); scalar(header, "DATA", std::uint64_t{}); }
        output.record("TES4", 0, header);
        output.finish();
    }
}
