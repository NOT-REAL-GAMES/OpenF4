#include "assets/esm_editor.hpp"
#include "assets/esm_document.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <Windows.h>
#endif

namespace openf4::assets
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
        constexpr std::uint32_t vhgt = fourCc("VHGT");
        constexpr std::size_t recordHeaderSize = 24;
        constexpr float heightScale = 8.0f;

        template <typename T>
        void append(Bytes& bytes, const T& value)
        {
            const auto* first = reinterpret_cast<const std::byte*>(&value);
            bytes.insert(bytes.end(), first, first + sizeof(T));
        }

        void append(Bytes& bytes, const void* dataBytes, std::size_t size)
        {
            const auto* first = static_cast<const std::byte*>(dataBytes);
            bytes.insert(bytes.end(), first, first + size);
        }

        template <typename T>
        [[nodiscard]] T read(const Bytes& bytes, std::size_t offset)
        {
            if (offset > bytes.size() || sizeof(T) > bytes.size() - offset)
                throw std::runtime_error("Truncated editable ESM");
            T value{};
            std::memcpy(&value, bytes.data() + offset, sizeof(T));
            return value;
        }

        [[nodiscard]] std::string readString(const Bytes& bytes, std::size_t offset, std::size_t size)
        {
            if (offset > bytes.size() || size > bytes.size() - offset)
                throw std::runtime_error("Truncated editable ESM string");
            const char* text = reinterpret_cast<const char*>(bytes.data() + offset);
            const auto end = std::find(text, text + size, '\0');
            return std::string(text, end);
        }

        void appendSubrecord(Bytes& body, std::uint32_t type, const void* payload, std::size_t size)
        {
            if (size > std::numeric_limits<std::uint16_t>::max())
                throw std::runtime_error("Editable ESM subrecord is too large");
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

        [[nodiscard]] Bytes makeRecord(std::uint32_t type, std::uint32_t formId, const Bytes& body,
            std::uint32_t flags = 0)
        {
            Bytes bytes;
            append(bytes, type);
            append(bytes, static_cast<std::uint32_t>(body.size()));
            append(bytes, flags);
            append(bytes, formId);
            append(bytes, std::uint32_t{0});
            append(bytes, std::uint16_t{131});
            append(bytes, std::uint16_t{0});
            append(bytes, body.data(), body.size());
            return bytes;
        }

        [[nodiscard]] Bytes makeGroup(std::uint32_t label, std::uint32_t groupType, const Bytes& body)
        {
            Bytes bytes;
            append(bytes, grup);
            append(bytes, static_cast<std::uint32_t>(recordHeaderSize + body.size()));
            append(bytes, label);
            append(bytes, groupType);
            append(bytes, std::uint32_t{0});
            append(bytes, std::uint32_t{0});
            append(bytes, body.data(), body.size());
            return bytes;
        }

        void appendBytes(Bytes& destination, const Bytes& source)
        {
            destination.insert(destination.end(), source.begin(), source.end());
        }

        [[nodiscard]] std::optional<std::size_t> initialCellIndex(CellCoord coord)
        {
            if (coord.x < -1 || coord.x > 1 || coord.y < -1 || coord.y > 1)
                return std::nullopt;
            return static_cast<std::size_t>((coord.y + 1) * 3 + coord.x + 1);
        }

        [[nodiscard]] bool validEditorFormId(std::uint32_t formId)
        {
            return (formId & 0xFF000000u) == 0x01000000u
                && (formId & 0x00FFFFFFu) >= 0x00000800u;
        }

        [[nodiscard]] std::uint32_t packedCellCoord(std::int32_t x, std::int32_t y)
        {
            return static_cast<std::uint16_t>(x) |
                (static_cast<std::uint32_t>(static_cast<std::uint16_t>(y)) << 16u);
        }

        [[nodiscard]] std::int32_t floorDivide(std::int32_t value, std::int32_t divisor)
        {
            const std::int32_t quotient = value / divisor;
            const std::int32_t remainder = value % divisor;
            return quotient - (remainder < 0 ? 1 : 0);
        }

        [[nodiscard]] Bytes encodeVhgt(const LandscapeCell& landscape)
        {
            std::array<std::int32_t, LandscapeCell::side * LandscapeCell::side> units{};
            for (std::size_t i = 0; i < units.size(); ++i)
            {
                if (!std::isfinite(landscape.heights[i]))
                    throw std::runtime_error("Editable landscape contains a non-finite height");
                const float quantized = std::round(landscape.heights[i] / heightScale);
                if (std::abs(landscape.heights[i] - quantized * heightScale) > 0.001f)
                    throw std::runtime_error("Editable landscape heights must be quantized to 8 units");
                if (quantized < static_cast<float>(std::numeric_limits<std::int32_t>::min()) ||
                    quantized > static_cast<float>(std::numeric_limits<std::int32_t>::max()))
                    throw std::runtime_error("Editable landscape height is outside the supported range");
                units[i] = static_cast<std::int32_t>(quantized);
            }

            Bytes payload;
            append(payload, static_cast<float>(units[0]));
            for (std::size_t y = 0; y < LandscapeCell::side; ++y)
            {
                const std::int32_t rowDelta = y == 0 ? 0 :
                    units[y * LandscapeCell::side] - units[(y - 1) * LandscapeCell::side];
                if (rowDelta < -128 || rowDelta > 127)
                    throw std::runtime_error("Editable landscape exceeds the signed VHGT row delta range");
                append(payload, static_cast<std::int8_t>(rowDelta));
                for (std::size_t x = 1; x < LandscapeCell::side; ++x)
                {
                    const std::int32_t delta = units[y * LandscapeCell::side + x] -
                        units[y * LandscapeCell::side + x - 1];
                    if (delta < -128 || delta > 127)
                        throw std::runtime_error("Editable landscape exceeds the signed VHGT column delta range");
                    append(payload, static_cast<std::int8_t>(delta));
                }
            }
            append(payload, std::array<std::byte, 3>{});
            return payload;
        }

        [[nodiscard]] LandscapeCell decodeVhgt(CellCoord coord, const Bytes& bytes,
            std::size_t offset, std::size_t size)
        {
            if (size != 1096)
                throw std::runtime_error("Unsupported editable LAND VHGT size");
            const float base = read<float>(bytes, offset);
            if (!std::isfinite(base) || std::round(base) != base)
                throw std::runtime_error("Malformed editable LAND VHGT base height");

            LandscapeCell result;
            result.cell = coord;
            result.hasHeights = true;
            float row = base;
            std::size_t cursor = offset + sizeof(float);
            for (std::size_t y = 0; y < LandscapeCell::side; ++y)
            {
                row += static_cast<float>(read<std::int8_t>(bytes, cursor++));
                float column = row;
                result.heights[y * LandscapeCell::side] = column * heightScale;
                for (std::size_t x = 1; x < LandscapeCell::side; ++x)
                {
                    column += static_cast<float>(read<std::int8_t>(bytes, cursor++));
                    result.heights[y * LandscapeCell::side + x] = column * heightScale;
                }
            }
            if (read<std::uint8_t>(bytes, cursor) != 0 ||
                read<std::uint8_t>(bytes, cursor + 1) != 0 ||
                read<std::uint8_t>(bytes, cursor + 2) != 0)
                throw std::runtime_error("Malformed editable LAND VHGT trailer");
            return result;
        }

        struct Subrecord
        {
            std::uint32_t type = 0;
            std::size_t offset = 0;
            std::size_t size = 0;
        };

        [[nodiscard]] std::vector<Subrecord> subrecords(const Bytes& bytes,
            std::size_t begin, std::size_t end)
        {
            std::vector<Subrecord> result;
            std::size_t cursor = begin;
            while (cursor < end)
            {
                if (end - cursor < 6)
                    throw std::runtime_error("Truncated editable ESM subrecord");
                Subrecord sub;
                sub.type = read<std::uint32_t>(bytes, cursor);
                sub.size = read<std::uint16_t>(bytes, cursor + 4);
                sub.offset = cursor + 6;
                if (sub.offset > end || sub.size > end - sub.offset)
                    throw std::runtime_error("Editable ESM subrecord exceeds its record");
                result.push_back(sub);
                cursor = sub.offset + sub.size;
            }
            return result;
        }

        struct ParsedDocument
        {
            EditableLandscape landscape;
            std::map<std::uint32_t, CellCoord> cells;
            std::map<EditableLandscape::SampleCoord, float> samples;
            std::set<std::uint32_t> formIds{EditableLandscape::worldFormId};
            bool worldSeen = false;
        };

        void parseElements(const Bytes& bytes, std::size_t begin, std::size_t end,
            std::uint32_t currentCell, ParsedDocument& parsed);

        void parseRecord(const Bytes& bytes, std::size_t offset, std::size_t end,
            std::uint32_t currentCell, ParsedDocument& parsed)
        {
            const std::uint32_t type = read<std::uint32_t>(bytes, offset);
            const std::uint32_t size = read<std::uint32_t>(bytes, offset + 4);
            const std::uint32_t flags = read<std::uint32_t>(bytes, offset + 8);
            const std::uint32_t formId = read<std::uint32_t>(bytes, offset + 12);
            if ((flags & 0x00040000u) != 0)
                throw std::runtime_error("Compressed records are unsupported in editable ESM files");
            const std::size_t body = offset + recordHeaderSize;
            if (body > end || size > end - body)
                throw std::runtime_error("Editable ESM record exceeds its group");
            const auto subs = subrecords(bytes, body, body + size);

            if (type == wrld)
            {
                if (formId != EditableLandscape::worldFormId || parsed.worldSeen)
                    throw std::runtime_error("Editable ESM has an unexpected WRLD FormID");
                const auto found = std::find_if(subs.begin(), subs.end(),
                    [](const Subrecord& sub) { return sub.type == edid; });
                if (found == subs.end() || (readString(bytes, found->offset, found->size) != "OpenF4World"
                    && readString(bytes, found->offset, found->size) != "EmpireWorld"))
                    throw std::runtime_error("Editable ESM does not contain OpenF4World");
                parsed.worldSeen = true;
                return;
            }

            if (type == cell)
            {
                const auto found = std::find_if(subs.begin(), subs.end(),
                    [](const Subrecord& sub) { return sub.type == xclc; });
                if (found == subs.end() || found->size < 8)
                    throw std::runtime_error("Editable CELL is missing XCLC");
                const CellCoord coord{read<std::int32_t>(bytes, found->offset),
                    read<std::int32_t>(bytes, found->offset + 4)};
                if (std::abs(static_cast<std::int64_t>(coord.x)) > 1'000'000
                    || std::abs(static_cast<std::int64_t>(coord.y)) > 1'000'000)
                    throw std::runtime_error("Editable CELL coordinate is outside the supported range");
                if (!validEditorFormId(formId) || !parsed.formIds.insert(formId).second)
                    throw std::runtime_error("Editable CELL has an invalid or duplicate FormID");
                if (const auto index = initialCellIndex(coord);
                    index && formId != EditableLandscape::firstCellFormId + *index)
                    throw std::runtime_error("Initial editable CELL has an unstable FormID");
                if (!parsed.cells.emplace(formId, coord).second
                    || parsed.landscape.cellForms.contains(coord))
                    throw std::runtime_error("Editable ESM contains a duplicate CELL coordinate");
                parsed.landscape.cellForms.emplace(coord,
                    EditableLandscape::CellForms{formId, 0});
                return;
            }

            if (type == land)
            {
                const auto cellIt = parsed.cells.find(currentCell);
                if (cellIt == parsed.cells.end())
                    throw std::runtime_error("Editable LAND is outside its CELL children group");
                auto& forms = parsed.landscape.cellForms.at(cellIt->second);
                if (forms.land != 0 || !validEditorFormId(formId)
                    || !parsed.formIds.insert(formId).second)
                    throw std::runtime_error("Editable LAND has an invalid or duplicate FormID");
                if (const auto index = initialCellIndex(cellIt->second);
                    index && formId != EditableLandscape::firstLandFormId + *index)
                    throw std::runtime_error("Initial editable LAND has an unstable FormID");
                const auto found = std::find_if(subs.begin(), subs.end(),
                    [](const Subrecord& sub) { return sub.type == vhgt; });
                if (found == subs.end())
                    throw std::runtime_error("Editable LAND is missing VHGT");
                const LandscapeCell decoded = decodeVhgt(cellIt->second, bytes, found->offset, found->size);
                forms.land = formId;

                for (std::size_t y = 0; y < LandscapeCell::side; ++y)
                {
                    for (std::size_t x = 0; x < LandscapeCell::side; ++x)
                    {
                        const EditableLandscape::SampleCoord global{
                            cellIt->second.x * 32 + static_cast<std::int32_t>(x),
                            cellIt->second.y * 32 + static_cast<std::int32_t>(y)};
                        const float value = decoded.heights[y * LandscapeCell::side + x];
                        const auto [sample, inserted] = parsed.samples.emplace(global, value);
                        if (!inserted && sample->second != value)
                            throw std::runtime_error("Editable LAND cells contain a divergent shared seam");
                    }
                }
                return;
            }

            throw std::runtime_error("Unsupported record in editable WRLD hierarchy");
        }

        void parseGroup(const Bytes& bytes, std::size_t offset, std::size_t end,
            std::uint32_t currentCell, ParsedDocument& parsed)
        {
            const std::uint32_t size = read<std::uint32_t>(bytes, offset + 4);
            const std::uint32_t label = read<std::uint32_t>(bytes, offset + 8);
            const std::uint32_t type = read<std::uint32_t>(bytes, offset + 12);
            if (size < recordHeaderSize || size > end - offset)
                throw std::runtime_error("Editable ESM group has an invalid size");

            std::uint32_t childCell = currentCell;
            switch (type)
            {
            case 0:
                if (label != wrld)
                    throw std::runtime_error("Editable ESM has an unexpected top group");
                break;
            case 1:
                if (label != EditableLandscape::worldFormId)
                    throw std::runtime_error("Editable ESM has an unexpected world children group");
                break;
            case 4:
            case 5:
                break;
            case 6:
                if (!parsed.cells.contains(label))
                    throw std::runtime_error("Editable ESM has CELL children before its CELL record");
                childCell = label;
                break;
            case 9:
                if (currentCell == 0 || label != currentCell)
                    throw std::runtime_error("Editable ESM has an unexpected temporary children group");
                break;
            default:
                throw std::runtime_error("Unsupported group type in editable ESM");
            }
            parseElements(bytes, offset + recordHeaderSize, offset + size, childCell, parsed);
        }

        void parseElements(const Bytes& bytes, std::size_t begin, std::size_t end,
            std::uint32_t currentCell, ParsedDocument& parsed)
        {
            std::size_t cursor = begin;
            while (cursor < end)
            {
                if (end - cursor < recordHeaderSize)
                    throw std::runtime_error("Truncated editable ESM record header");
                const std::uint32_t type = read<std::uint32_t>(bytes, cursor);
                const std::uint32_t size = read<std::uint32_t>(bytes, cursor + 4);
                if (type == grup)
                {
                    parseGroup(bytes, cursor, end, currentCell, parsed);
                    cursor += size;
                }
                else
                {
                    parseRecord(bytes, cursor, end, currentCell, parsed);
                    if (size > end - cursor - recordHeaderSize)
                        throw std::runtime_error("Editable ESM record has an invalid size");
                    cursor += recordHeaderSize + size;
                }
            }
        }

        [[nodiscard]] Bytes serialize(const EditableLandscape& landscape)
        {
            if (landscape.cellForms.size() < EditableLandscape::cellCount)
                throw std::runtime_error("Editable landscape is missing its initial 3x3 cells");
            for (std::int32_t y = -1; y <= 1; ++y)
                for (std::int32_t x = -1; x <= 1; ++x)
                    if (!landscape.containsCell({x, y}))
                        throw std::runtime_error("Editable landscape is missing an initial CELL");
            std::set<std::uint32_t> formIds{EditableLandscape::worldFormId};
            std::uint32_t highestFormId = EditableLandscape::worldFormId;
            for (const auto& [coord, forms] : landscape.cellForms)
            {
                if (!validEditorFormId(forms.cell) || !validEditorFormId(forms.land)
                    || !formIds.insert(forms.cell).second || !formIds.insert(forms.land).second)
                    throw std::runtime_error("Editable landscape has invalid or duplicate FormIDs");
                if (const auto index = initialCellIndex(coord);
                    index && (forms.cell != EditableLandscape::firstCellFormId + *index
                        || forms.land != EditableLandscape::firstLandFormId + *index))
                    throw std::runtime_error("Initial editable landscape FormIDs changed");
                highestFormId = std::max({highestFormId, forms.cell, forms.land});
            }
            if (landscape.nextFormId <= highestFormId)
                throw std::runtime_error("Editable landscape next FormID is not monotonic");
            Bytes tesBody;
            struct Header
            {
                float version;
                std::uint32_t recordCount;
                std::uint32_t nextObjectId;
            };
            const std::uint32_t recordCount = 1u
                + static_cast<std::uint32_t>(landscape.cellForms.size()) * 2u;
            appendSubrecord(tesBody, hedr, Header{0.95f, recordCount,
                landscape.nextFormId & 0x00FFFFFFu});
            appendStringSubrecord(tesBody, cnam, "OpenF4");
            appendStringSubrecord(tesBody, mast, "Fallout4.esm");
            appendSubrecord(tesBody, data, std::uint64_t{0});

            Bytes output = makeRecord(tes4, 0, tesBody, 1u);
            Bytes worldBody;
            appendStringSubrecord(worldBody, edid, "OpenF4World");
            const std::uint8_t worldData = 1;
            appendSubrecord(worldBody, data, worldData);

            std::map<std::uint32_t, std::map<std::uint32_t, Bytes>> hierarchy;
            for (const auto& [coord, forms] : landscape.cellForms)
            {
                const std::uint32_t cellId = forms.cell;
                const std::uint32_t landId = forms.land;

                Bytes cellBody;
                appendStringSubrecord(cellBody, edid,
                    "OpenF4Cell_" + std::to_string(coord.x) + "_" + std::to_string(coord.y));
                appendSubrecord(cellBody, data, std::uint16_t{0});
                const std::array<std::int32_t, 3> coordinates{coord.x, coord.y, 0};
                appendSubrecord(cellBody, xclc, coordinates.data(), sizeof(coordinates));

                Bytes landBody;
                const Bytes encoded = encodeVhgt(landscape.cell(coord));
                appendSubrecord(landBody, vhgt, encoded.data(), encoded.size());
                const Bytes landRecord = makeRecord(land, landId, landBody);
                const Bytes temporary = makeGroup(cellId, 9, landRecord);
                const Bytes children = makeGroup(cellId, 6, temporary);

                Bytes cellTree = makeRecord(cell, cellId, cellBody);
                appendBytes(cellTree, children);
                const std::uint32_t block = packedCellCoord(
                    floorDivide(coord.x, 32), floorDivide(coord.y, 32));
                const std::uint32_t subblock = packedCellCoord(
                    floorDivide(coord.x, 8), floorDivide(coord.y, 8));
                appendBytes(hierarchy[block][subblock], cellTree);
            }

            Bytes worldChildren;
            for (const auto& [blockLabel, subblocks] : hierarchy)
            {
                Bytes blockBody;
                for (const auto& [subblockLabel, cells] : subblocks)
                    appendBytes(blockBody, makeGroup(subblockLabel, 5, cells));
                appendBytes(worldChildren, makeGroup(blockLabel, 4, blockBody));
            }

            Bytes topBody = makeRecord(wrld, EditableLandscape::worldFormId, worldBody);
            appendBytes(topBody, makeGroup(EditableLandscape::worldFormId, 1, worldChildren));
            appendBytes(output, makeGroup(wrld, 0, topBody));
            return output;
        }

        void atomicWrite(const std::filesystem::path& path, const Bytes& bytes)
        {
            if (path.has_parent_path())
                std::filesystem::create_directories(path.parent_path());
            std::filesystem::path temporary = path;
            temporary += ".tmp";
            {
                std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
                if (!file)
                    throw std::runtime_error("Could not create editable ESM temporary file");
                file.write(reinterpret_cast<const char*>(bytes.data()),
                    static_cast<std::streamsize>(bytes.size()));
                file.flush();
                if (!file)
                    throw std::runtime_error("Could not write editable ESM temporary file");
            }

#ifdef _WIN32
            const bool exists = std::filesystem::exists(path);
            const BOOL replaced = exists ?
                ReplaceFileW(path.c_str(), temporary.c_str(), nullptr, REPLACEFILE_WRITE_THROUGH,
                    nullptr, nullptr) :
                MoveFileExW(temporary.c_str(), path.c_str(), MOVEFILE_WRITE_THROUGH);
            if (!replaced)
            {
                const DWORD error = GetLastError();
                std::error_code ignored;
                std::filesystem::remove(temporary, ignored);
                throw std::runtime_error("Could not atomically replace editable ESM (Windows error " +
                    std::to_string(error) + ")");
            }
#else
            std::filesystem::rename(temporary, path);
#endif
        }
    }

    float EditableLandscape::height(std::size_t x, std::size_t y) const
    {
        if (x >= side || y >= side)
            throw std::out_of_range("Editable landscape sample is out of range");
        return heights[y * side + x];
    }

    float& EditableLandscape::height(std::size_t x, std::size_t y)
    {
        if (x >= side || y >= side)
            throw std::out_of_range("Editable landscape sample is out of range");
        return heights[y * side + x];
    }

    bool EditableLandscape::containsCell(CellCoord coord) const noexcept
    {
        return cellForms.contains(coord);
    }

    bool EditableLandscape::hasSample(SampleCoord coord) const noexcept
    {
        if (coord.x >= -32 && coord.x <= 64 && coord.y >= -32 && coord.y <= 64)
            return true;
        return additionalHeights.contains(coord);
    }

    float EditableLandscape::sample(SampleCoord coord) const
    {
        if (coord.x >= -32 && coord.x <= 64 && coord.y >= -32 && coord.y <= 64)
            return height(static_cast<std::size_t>(coord.x + 32),
                static_cast<std::size_t>(coord.y + 32));
        const auto found = additionalHeights.find(coord);
        if (found == additionalHeights.end())
            throw std::out_of_range("Editable landscape global sample is out of range");
        return found->second;
    }

    void EditableLandscape::setSample(SampleCoord coord, float value)
    {
        if (coord.x >= -32 && coord.x <= 64 && coord.y >= -32 && coord.y <= 64)
            height(static_cast<std::size_t>(coord.x + 32),
                static_cast<std::size_t>(coord.y + 32)) = value;
        else
            additionalHeights[coord] = value;
    }

    bool EditableLandscape::addCell(CellCoord coord)
    {
        if (containsCell(coord))
            return false;
        if (std::abs(static_cast<std::int64_t>(coord.x)) > 1'000'000
            || std::abs(static_cast<std::int64_t>(coord.y)) > 1'000'000)
            throw std::runtime_error("Editable CELL coordinate is outside the supported range");
        if ((nextFormId & 0x00FFFFFFu) > 0x00FFFFFEu)
            throw std::runtime_error("Editable ESM has exhausted its FormID range");

        const bool west = containsCell({coord.x - 1, coord.y});
        const bool east = containsCell({coord.x + 1, coord.y});
        const bool south = containsCell({coord.x, coord.y - 1});
        const bool north = containsCell({coord.x, coord.y + 1});
        for (std::int32_t y = 0; y <= 32; ++y)
        {
            for (std::int32_t x = 0; x <= 32; ++x)
            {
                const SampleCoord global{coord.x * 32 + x, coord.y * 32 + y};
                if (hasSample(global))
                    continue;
                float sum = 0.0f;
                int count = 0;
                const auto include = [&](SampleCoord source) {
                    if (hasSample(source))
                    {
                        sum += sample(source);
                        ++count;
                    }
                };
                if (west)
                    include({coord.x * 32, coord.y * 32 + y});
                if (east)
                    include({(coord.x + 1) * 32, coord.y * 32 + y});
                if (south)
                    include({coord.x * 32 + x, coord.y * 32});
                if (north)
                    include({coord.x * 32 + x, (coord.y + 1) * 32});
                const float height = count == 0 ? 0.0f : sum / static_cast<float>(count);
                setSample(global, std::round(height / heightScale) * heightScale);
            }
        }
        cellForms.emplace(coord, CellForms{nextFormId, nextFormId + 1});
        nextFormId += 2;
        return true;
    }

    LandscapeCell EditableLandscape::cell(CellCoord coord) const
    {
        if (!containsCell(coord))
            throw std::out_of_range("Editable landscape CELL does not exist");
        LandscapeCell result;
        result.cell = coord;
        result.hasHeights = true;
        for (std::size_t y = 0; y < LandscapeCell::side; ++y)
        {
            for (std::size_t x = 0; x < LandscapeCell::side; ++x)
            {
                const SampleCoord global{coord.x * 32 + static_cast<std::int32_t>(x),
                    coord.y * 32 + static_cast<std::int32_t>(y)};
                result.heights[y * LandscapeCell::side + x] = sample(global);
            }
        }
        return result;
    }

    EditableLandscape createEditableLandscape()
    {
        EditableLandscape result;
        for (std::int32_t y = -1; y <= 1; ++y)
        {
            for (std::int32_t x = -1; x <= 1; ++x)
            {
                const std::size_t index = *initialCellIndex({x, y});
                result.cellForms.emplace(CellCoord{x, y}, EditableLandscape::CellForms{
                    EditableLandscape::firstCellFormId + static_cast<std::uint32_t>(index),
                    EditableLandscape::firstLandFormId + static_cast<std::uint32_t>(index)});
            }
        }
        return result;
    }

    EditableLandscape loadEditableLandscape(const std::filesystem::path& path)
    {
        std::ifstream file(path, std::ios::binary);
        Bytes bytes(recordHeaderSize);
        file.read(reinterpret_cast<char*>(bytes.data()), recordHeaderSize);
        if (!file) throw std::runtime_error("Editable ESM is truncated");
        if (read<std::uint32_t>(bytes, 0) != tes4 || (read<std::uint32_t>(bytes, 8) & 1u) == 0)
            throw std::runtime_error("Editable ESM must begin with a master TES4 record");
        EsmDocument document(path);
        if (std::find(document.masters().begin(), document.masters().end(), "Fallout4.esm") == document.masters().end())
            throw std::runtime_error("Editable ESM is missing its Fallout4.esm dependency");
        if (document.worldFormId() != EditableLandscape::worldFormId || !document.isOpenF4Owned())
            throw std::runtime_error("Editable ESM does not contain OpenF4World");
        ParsedDocument parsed;
        for (const auto& [coord, entry] : document.terrainCells())
        {
            if (std::abs(static_cast<std::int64_t>(coord.x)) > 1'000'000
                || std::abs(static_cast<std::int64_t>(coord.y)) > 1'000'000)
                throw std::runtime_error("Editable CELL coordinate is outside the supported range");
            if (!validEditorFormId(entry.cellFormId) || !validEditorFormId(entry.landFormId)
                || !parsed.formIds.insert(entry.cellFormId).second || !parsed.formIds.insert(entry.landFormId).second)
                throw std::runtime_error("Editable CELL/LAND has an invalid or duplicate FormID");
            if (const auto initial = initialCellIndex(coord); initial
                && (entry.cellFormId != EditableLandscape::firstCellFormId + *initial
                    || entry.landFormId != EditableLandscape::firstLandFormId + *initial))
                throw std::runtime_error("Initial editable CELL/LAND has an unstable FormID");
            const auto decoded = document.terrainCell(coord);
            if (!decoded || !decoded->landscape.hasHeights)
                throw std::runtime_error("Editable CELL is missing its LAND VHGT record");
            (void)encodeVhgt(decoded->landscape);
            parsed.landscape.cellForms.emplace(coord, EditableLandscape::CellForms{entry.cellFormId, entry.landFormId});
            for (std::int32_t y = 0; y < 33; ++y) for (std::int32_t x = 0; x < 33; ++x)
            {
                const EditableLandscape::SampleCoord global{coord.x * 32 + x, coord.y * 32 + y};
                const float value = decoded->landscape.heights[y * 33 + x];
                const auto [sample, inserted] = parsed.samples.emplace(global, value);
                if (!inserted && sample->second != value)
                    throw std::runtime_error("Editable LAND cells contain a divergent shared seam");
            }
        }
        for (std::int32_t y = -1; y <= 1; ++y)
        {
            for (std::int32_t x = -1; x <= 1; ++x)
                if (!parsed.landscape.cellForms.contains({x, y}))
                    throw std::runtime_error("Editable ESM does not contain the initial 3x3 landscape");
        }
        std::uint32_t highestFormId = EditableLandscape::firstLandFormId;
        for (const auto& [coord, forms] : parsed.landscape.cellForms)
        {
            (void)coord;
            if (forms.land == 0)
                throw std::runtime_error("Editable CELL is missing its LAND record");
            highestFormId = std::max({highestFormId, forms.cell, forms.land});
        }
        for (const auto& [coord, value] : parsed.samples)
            parsed.landscape.setSample(coord, value);
        parsed.landscape.nextFormId = std::max(highestFormId + 1, document.nextFormId());
        return parsed.landscape;
    }

    void saveEditableLandscape(const std::filesystem::path& path, const EditableLandscape& landscape)
    {
        if (!std::filesystem::exists(path))
        {
            atomicWrite(path, serialize(landscape));
            return;
        }
        EsmDocument document(path);
        std::vector<ESMTerrainCell> changes;
        for (const auto& [coord, forms] : landscape.cellForms)
        {
            const auto cell = landscape.cell(coord);
            (void)encodeVhgt(cell);
            const auto existing = document.terrainCell(coord);
            if (!existing || existing->cellFormId != forms.cell || existing->landFormId != forms.land
                || existing->landscape.heights != cell.heights)
                changes.push_back({cell, forms.cell, forms.land});
        }
        document.replaceTerrainCells(changes);
        document.setNextFormId(landscape.nextFormId);
        document.save();
    }
}
