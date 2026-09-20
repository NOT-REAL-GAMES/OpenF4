#include "assets/esm_world.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <span>
#include <stdexcept>
#include <unordered_set>

#include <zlib.h>

namespace empire::assets
{
    namespace
    {
        constexpr std::uint32_t fourCc(char a, char b, char c, char d)
        {
            return static_cast<std::uint32_t>(a)
                | (static_cast<std::uint32_t>(b) << 8u)
                | (static_cast<std::uint32_t>(c) << 16u)
                | (static_cast<std::uint32_t>(d) << 24u);
        }

        constexpr std::uint32_t grup = fourCc('G', 'R', 'U', 'P');
        constexpr std::uint32_t wrld = fourCc('W', 'R', 'L', 'D');
        constexpr std::uint32_t cell = fourCc('C', 'E', 'L', 'L');
        constexpr std::uint32_t refr = fourCc('R', 'E', 'F', 'R');
        constexpr std::uint32_t land = fourCc('L', 'A', 'N', 'D');
        constexpr std::uint32_t txst = fourCc('T', 'X', 'S', 'T');
        constexpr std::uint32_t ltex = fourCc('L', 'T', 'E', 'X');
        constexpr std::uint32_t mswp = fourCc('M', 'S', 'W', 'P');
        constexpr std::uint32_t scol = fourCc('S', 'C', 'O', 'L');
        constexpr std::uint32_t stat = fourCc('S', 'T', 'A', 'T');
        constexpr std::uint32_t compressedFlag = 0x00040000;
        constexpr std::uint32_t deletedFlag = 0x00000020;
        constexpr std::uint32_t disabledFlag = 0x00000800;
        constexpr std::uint32_t commonwealthFormId = 0x0000003C;

        struct RecordHeader
        {
            std::uint32_t type = 0;
            std::uint32_t dataSize = 0;
            std::uint32_t flags = 0;
            std::uint32_t formId = 0;
            std::uint32_t revision = 0;
            std::uint16_t version = 0;
            std::uint16_t unknown = 0;
        };

        template <class T>
        T valueAt(std::span<const std::byte> bytes, std::size_t offset)
        {
            if (offset + sizeof(T) > bytes.size())
                throw std::runtime_error("Truncated ESM record data");
            T value{};
            std::memcpy(&value, bytes.data() + offset, sizeof(value));
            return value;
        }

        class Subrecords
        {
        public:
            struct Value
            {
                std::uint32_t type = 0;
                std::span<const std::byte> data;
            };

            explicit Subrecords(std::span<const std::byte> bytes)
                : mBytes(bytes)
            {
            }

            bool next(Value& result)
            {
                if (mOffset + 6 > mBytes.size())
                    return false;
                std::uint32_t type = valueAt<std::uint32_t>(mBytes, mOffset);
                std::uint32_t size = valueAt<std::uint16_t>(mBytes, mOffset + 4);
                mOffset += 6;
                if (type == fourCc('X', 'X', 'X', 'X'))
                {
                    if (size != 4 || mOffset + 10 > mBytes.size())
                        return false;
                    size = valueAt<std::uint32_t>(mBytes, mOffset);
                    mOffset += 4;
                    type = valueAt<std::uint32_t>(mBytes, mOffset);
                    mOffset += 6;
                }
                if (mOffset + size > mBytes.size())
                    return false;
                result = {type, mBytes.subspan(mOffset, size)};
                mOffset += size;
                return true;
            }

        private:
            std::span<const std::byte> mBytes;
            std::size_t mOffset = 0;
        };

        std::string stringValue(std::span<const std::byte> bytes)
        {
            const char* begin = reinterpret_cast<const char*>(bytes.data());
            std::size_t size = 0;
            while (size < bytes.size() && begin[size] != '\0')
                ++size;
            std::string result(begin, size);
            std::replace(result.begin(), result.end(), '/', '\\');
            std::transform(result.begin(), result.end(), result.begin(), [](unsigned char value) {
                return static_cast<char>(std::tolower(value));
            });
            return result;
        }

        bool isExplicitLodProxy(std::string_view editorId)
        {
            return editorId.ends_with("lod");
        }

        std::string canonicalMaterialPath(std::string_view value)
        {
            std::string result(value);
            std::replace(result.begin(), result.end(), '/', '\\');
            std::transform(result.begin(), result.end(), result.begin(), [](unsigned char character) {
                return static_cast<char>(std::tolower(character));
            });
            const auto embedded = result.rfind("\\materials\\");
            if (embedded != std::string::npos)
                result.erase(0, embedded + 11);
            else if (result.starts_with("materials\\"))
                result.erase(0, 10);
            while (!result.empty() && result.front() == '\\')
                result.erase(result.begin());
            return result;
        }

        class Parser
        {
        public:
            explicit Parser(const std::filesystem::path& path,
                std::uint32_t selectedWorld = 0)
                : mStream(path, std::ios::binary), mSelectedWorld(selectedWorld)
            {
                if (!mStream)
                    throw std::runtime_error("Unable to open ESM: " + path.string());
                mStream.seekg(0, std::ios::end);
                mFileSize = position();
                mStream.seekg(0);
            }

            ExteriorWorld parse()
            {
                mWorld.worldFormId = mSelectedWorld;
                parseEntries(mFileSize, {});
                expandStaticCollections();
                return std::move(mWorld);
            }

        private:
            struct Context
            {
                std::uint32_t topLevel = 0;
                bool selectedWorld = false;
                std::uint32_t cellId = 0;
            };

            struct ParsedCell
            {
                CellCoord coord{};
                std::uint16_t flags = 0;
                float waterHeight = 0.0f;
                bool exterior = false;
            };

            std::ifstream mStream;
            std::uint64_t mFileSize = 0;
            ExteriorWorld mWorld;
            std::uint32_t mSelectedWorld = 0;
            std::unordered_map<std::uint32_t, ParsedCell> mCells;
            std::unordered_set<std::uint32_t> mLodProxyBases;

            std::uint64_t position()
            {
                return static_cast<std::uint64_t>(mStream.tellg());
            }

            template <class T>
            T read()
            {
                T value{};
                mStream.read(reinterpret_cast<char*>(&value), sizeof(value));
                if (!mStream)
                    throw std::runtime_error("Unexpected end of ESM file");
                return value;
            }

            RecordHeader readHeader()
            {
                RecordHeader result;
                result.type = read<std::uint32_t>();
                result.dataSize = read<std::uint32_t>();
                result.flags = read<std::uint32_t>();
                result.formId = read<std::uint32_t>();
                result.revision = read<std::uint32_t>();
                result.version = read<std::uint16_t>();
                result.unknown = read<std::uint16_t>();
                return result;
            }

            void seek(std::uint64_t target)
            {
                mStream.seekg(static_cast<std::streamoff>(target));
                if (!mStream)
                    throw std::runtime_error("Invalid ESM offset");
            }

            static bool modelGroup(std::uint32_t type)
            {
                constexpr std::array types{
                    stat, fourCc('M', 'S', 'T', 'T'),
                    fourCc('A', 'C', 'T', 'I'), fourCc('T', 'R', 'E', 'E'),
                    fourCc('F', 'L', 'O', 'R'), fourCc('F', 'U', 'R', 'N'),
                    fourCc('D', 'O', 'O', 'R'), fourCc('C', 'O', 'N', 'T'),
                    fourCc('T', 'A', 'C', 'T')};
                return std::ranges::find(types, type) != types.end();
            }

            static bool interestingTopLevel(std::uint32_t type)
            {
                return type == wrld || type == txst || type == ltex || type == mswp
                    || type == scol || modelGroup(type);
            }

            void parseEntries(std::uint64_t end, Context context)
            {
                while (position() + sizeof(RecordHeader) <= end)
                {
                    const std::uint64_t start = position();
                    const RecordHeader header = readHeader();
                    if (header.type == grup)
                    {
                        if (header.dataSize < sizeof(RecordHeader))
                            throw std::runtime_error("Invalid ESM group size");
                        const std::uint64_t groupEnd = start + header.dataSize;
                        const std::int32_t groupType = static_cast<std::int32_t>(header.formId);
                        Context child = context;
                        bool enter = true;
                        if (groupType == 0)
                        {
                            child.topLevel = header.flags;
                            enter = interestingTopLevel(child.topLevel);
                        }
                        else if (context.topLevel == wrld && groupType == 1)
                        {
                            child.selectedWorld = header.flags == mSelectedWorld;
                            enter = child.selectedWorld;
                        }
                        else if (context.topLevel == wrld && !context.selectedWorld)
                            enter = false;
                        else if (context.selectedWorld && (groupType == 2 || groupType == 3))
                            enter = false;
                        if (context.selectedWorld && (groupType == 6 || groupType == 8 || groupType == 9))
                            child.cellId = header.flags;
                        if (enter)
                            parseEntries(groupEnd, child);
                        seek(groupEnd);
                        continue;
                    }

                    const std::uint64_t recordEnd = position() + header.dataSize;
                    if (recordEnd > end || recordEnd > mFileSize)
                        throw std::runtime_error("ESM record exceeds its containing group");
                    if (context.topLevel == wrld && header.type == wrld
                        && mSelectedWorld == 0)
                    {
                        mSelectedWorld = header.formId;
                        mWorld.worldFormId = header.formId;
                    }
                    const bool wanted = (modelGroup(context.topLevel) && header.type == context.topLevel)
                        || (context.topLevel == wrld && header.type == wrld
                            && header.formId == mSelectedWorld)
                        || (context.topLevel == scol && header.type == scol)
                        || (context.topLevel == txst && header.type == txst)
                        || (context.topLevel == ltex && header.type == ltex)
                        || (context.topLevel == mswp && header.type == mswp)
                        || (context.selectedWorld && (header.type == cell || header.type == refr
                            || header.type == land));
                    if (wanted)
                        parseRecord(header, context);
                    seek(recordEnd);
                }
            }

            std::vector<std::byte> recordData(const RecordHeader& header)
            {
                std::vector<std::byte> stored(header.dataSize);
                mStream.read(reinterpret_cast<char*>(stored.data()), stored.size());
                if (!mStream)
                    throw std::runtime_error("Truncated ESM record");
                if ((header.flags & compressedFlag) == 0)
                    return stored;
                if (stored.size() < 4)
                    throw std::runtime_error("Compressed ESM record is too small");
                const std::uint32_t unpackedSize = valueAt<std::uint32_t>(stored, 0);
                std::vector<std::byte> unpacked(unpackedSize);
                uLongf outputSize = unpackedSize;
                if (uncompress(reinterpret_cast<Bytef*>(unpacked.data()), &outputSize,
                    reinterpret_cast<const Bytef*>(stored.data() + 4),
                    static_cast<uLong>(stored.size() - 4)) != Z_OK || outputSize != unpackedSize)
                    throw std::runtime_error("ESM record decompression failed");
                return unpacked;
            }

            void parseRecord(const RecordHeader& header, const Context& context)
            {
                const auto bytes = recordData(header);
                Subrecords fields(bytes);
                Subrecords::Value field;
                if (header.type == wrld)
                {
                    while (fields.next(field))
                        if (field.type == fourCc('N', 'A', 'M', '4')
                            && field.data.size() >= sizeof(float))
                            mWorld.defaultWaterHeight = valueAt<float>(field.data, 0);
                    return;
                }
                if (header.type == mswp)
                {
                    MaterialSwapRecord materialSwap;
                    std::string pendingOriginal;
                    while (fields.next(field))
                    {
                        if (field.type == fourCc('E', 'D', 'I', 'D'))
                            materialSwap.editorId = stringValue(field.data);
                        else if (field.type == fourCc('B', 'N', 'A', 'M'))
                            pendingOriginal = canonicalMaterialPath(stringValue(field.data));
                        else if (field.type == fourCc('S', 'N', 'A', 'M')
                            && !pendingOriginal.empty())
                        {
                            materialSwap.substitutions.push_back({
                                std::move(pendingOriginal),
                                canonicalMaterialPath(stringValue(field.data)), std::nullopt});
                            pendingOriginal.clear();
                        }
                        else if (field.type == fourCc('C', 'N', 'A', 'M')
                            && field.data.size() >= sizeof(float)
                            && !materialSwap.substitutions.empty())
                        {
                            materialSwap.substitutions.back().colorRemapIndex
                                = valueAt<float>(field.data, 0);
                        }
                    }
                    mWorld.materialSwaps[header.formId] = std::move(materialSwap);
                    return;
                }
                if (header.type == scol)
                {
                    std::uint32_t pendingBase = 0;
                    auto& parts = mWorld.staticCollections[header.formId];
                    while (fields.next(field))
                    {
                        if (field.type == fourCc('E', 'D', 'I', 'D')
                            && isExplicitLodProxy(stringValue(field.data)))
                            mLodProxyBases.insert(header.formId);
                        else if (field.type == fourCc('M', 'O', 'D', 'L'))
                            mWorld.models[header.formId] = stringValue(field.data);
                        else if (field.type == fourCc('O', 'N', 'A', 'M') && field.data.size() >= 4)
                            pendingBase = valueAt<std::uint32_t>(field.data, 0);
                        else if (field.type == fourCc('D', 'A', 'T', 'A')
                            && field.data.size() >= 28 && pendingBase != 0)
                        {
                            for (std::size_t offset = 0; offset + 28 <= field.data.size(); offset += 28)
                            {
                                StaticCollectionPart part;
                                part.baseObject = pendingBase;
                                for (int index = 0; index < 3; ++index)
                                    part.position[index]
                                        = valueAt<float>(field.data, offset + index * 4);
                                for (int index = 0; index < 3; ++index)
                                    part.rotation[index]
                                        = valueAt<float>(field.data, offset + 12 + index * 4);
                                part.scale = valueAt<float>(field.data, offset + 24);
                                parts.push_back(part);
                            }
                            pendingBase = 0;
                        }
                    }
                    return;
                }
                if (modelGroup(context.topLevel))
                {
                    while (fields.next(field))
                    {
                        if (field.type == fourCc('E', 'D', 'I', 'D')
                            && isExplicitLodProxy(stringValue(field.data)))
                            mLodProxyBases.insert(header.formId);
                        else if (field.type == fourCc('M', 'O', 'D', 'L'))
                            mWorld.models[header.formId] = stringValue(field.data);
                        else if (context.topLevel == stat
                            && field.type == fourCc('M', 'N', 'A', 'M')
                            && field.data.size() >= 4 * 260)
                        {
                            auto& lods = mWorld.modelLods[header.formId];
                            for (std::size_t index = 0; index < lods.size(); ++index)
                                lods[index] = stringValue(field.data.subspan(index * 260, 260));
                        }
                        else if (field.type == fourCc('M', 'O', 'D', 'S')
                            && field.data.size() >= 4)
                            mWorld.baseMaterialSwaps[header.formId]
                                = valueAt<std::uint32_t>(field.data, 0);
                    }
                    return;
                }
                if (header.type == txst)
                {
                    TextureSetRecord texture;
                    while (fields.next(field))
                    {
                        if (field.type == fourCc('T', 'X', '0', '0'))
                            texture.diffuse = stringValue(field.data);
                        else if (field.type == fourCc('T', 'X', '0', '1'))
                            texture.normal = stringValue(field.data);
                        else if (field.type == fourCc('M', 'N', 'A', 'M'))
                            texture.material = stringValue(field.data);
                    }
                    mWorld.textureSets[header.formId] = std::move(texture);
                    return;
                }
                if (header.type == ltex)
                {
                    while (fields.next(field))
                        if (field.type == fourCc('T', 'N', 'A', 'M') && field.data.size() >= 4)
                            mWorld.landscapeTextures[header.formId]
                                = valueAt<std::uint32_t>(field.data, 0);
                        else if (field.type == fourCc('B', 'N', 'A', 'M'))
                            mWorld.landscapeMaterials[header.formId] = stringValue(field.data);
                    return;
                }
                if (header.type == cell)
                {
                    ParsedCell parsed;
                    while (fields.next(field))
                    {
                        if (field.type == fourCc('X', 'C', 'L', 'C') && field.data.size() >= 8)
                        {
                            parsed.coord = {valueAt<std::int32_t>(field.data, 0),
                                valueAt<std::int32_t>(field.data, 4)};
                            parsed.exterior = true;
                        }
                        else if (field.type == fourCc('D', 'A', 'T', 'A')
                            && field.data.size() >= sizeof(std::uint16_t))
                            parsed.flags = valueAt<std::uint16_t>(field.data, 0);
                        else if (field.type == fourCc('X', 'C', 'L', 'W')
                            && field.data.size() >= sizeof(float))
                            parsed.waterHeight = valueAt<float>(field.data, 0);
                    }
                    if (parsed.exterior)
                    {
                        mCells[header.formId] = parsed;
                        constexpr std::uint16_t hasWater = 0x0002;
                        if ((parsed.flags & hasWater) != 0)
                        {
                            const bool explicitHeight = parsed.waterHeight != 0.0f
                                && std::isfinite(parsed.waterHeight)
                                && std::abs(parsed.waterHeight) < 1.0e30f;
                            mWorld.waterHeights[parsed.coord] = explicitHeight
                                ? parsed.waterHeight : mWorld.defaultWaterHeight;
                        }
                    }
                    return;
                }
                const auto cellPosition = mCells.find(context.cellId);
                if (cellPosition == mCells.end())
                    return;
                if (header.type == refr)
                {
                    if (header.flags & (deletedFlag | disabledFlag))
                        return;
                    PlacedReference reference;
                    reference.formId = header.formId;
                    reference.instanceId = header.formId;
                    reference.cell = cellPosition->second.coord;
                    while (fields.next(field))
                    {
                        if (field.type == fourCc('N', 'A', 'M', 'E') && field.data.size() >= 4)
                            reference.baseObject = valueAt<std::uint32_t>(field.data, 0);
                        else if (field.type == fourCc('D', 'A', 'T', 'A') && field.data.size() >= 24)
                        {
                            for (int index = 0; index < 3; ++index)
                                reference.position[index] = valueAt<float>(field.data, index * 4);
                            for (int index = 0; index < 3; ++index)
                                reference.rotation[index] = valueAt<float>(field.data, 12 + index * 4);
                        }
                        else if (field.type == fourCc('X', 'S', 'C', 'L') && field.data.size() >= 4)
                            reference.scale = valueAt<float>(field.data, 0);
                        else if (field.type == fourCc('X', 'M', 'S', 'P') && field.data.size() >= 4)
                            reference.materialSwap = valueAt<std::uint32_t>(field.data, 0);
                    }
                    if (reference.baseObject != 0)
                        mWorld.references.push_back(reference);
                    return;
                }
                if (header.type == land)
                {
                    LandscapeCell landscape;
                    landscape.cell = cellPosition->second.coord;
                    while (fields.next(field))
                    {
                        if (field.type == fourCc('V', 'H', 'G', 'T')
                            && field.data.size() >= 4 + landscape.heights.size())
                        {
                            float row = valueAt<float>(field.data, 0);
                            for (std::size_t y = 0; y < LandscapeCell::side; ++y)
                            {
                                row += valueAt<std::int8_t>(field.data, 4 + y * LandscapeCell::side);
                                float column = row;
                                landscape.heights[y * LandscapeCell::side] = column * 8.0f;
                                for (std::size_t x = 1; x < LandscapeCell::side; ++x)
                                {
                                    column += valueAt<std::int8_t>(
                                        field.data, 4 + y * LandscapeCell::side + x);
                                    landscape.heights[y * LandscapeCell::side + x] = column * 8.0f;
                                }
                            }
                            landscape.hasHeights = true;
                        }
                        else if (field.type == fourCc('V', 'C', 'L', 'R')
                            && field.data.size() >= landscape.colors.size() * 3)
                        {
                            for (std::size_t index = 0; index < landscape.colors.size(); ++index)
                            {
                                landscape.colors[index] = {
                                    valueAt<std::uint8_t>(field.data, index * 3),
                                    valueAt<std::uint8_t>(field.data, index * 3 + 1),
                                    valueAt<std::uint8_t>(field.data, index * 3 + 2)};
                            }
                            landscape.hasColors = true;
                        }
                        else if ((field.type == fourCc('B', 'T', 'X', 'T')
                                     || field.type == fourCc('A', 'T', 'X', 'T'))
                            && field.data.size() >= 8)
                        {
                            LandscapeCell::TextureLayer layer;
                            layer.texture = valueAt<std::uint32_t>(field.data, 0);
                            layer.quadrant = valueAt<std::uint8_t>(field.data, 4);
                            layer.layer = valueAt<std::uint16_t>(field.data, 6);
                            layer.base = field.type == fourCc('B', 'T', 'X', 'T');
                            const std::uint8_t quadrant = valueAt<std::uint8_t>(field.data, 4);
                            if (layer.base && quadrant < landscape.baseTextures.size())
                                landscape.baseTextures[quadrant] = layer.texture;
                            landscape.textureLayers.push_back(std::move(layer));
                        }
                        else if (field.type == fourCc('V', 'T', 'X', 'T')
                            && !landscape.textureLayers.empty())
                        {
                            auto& layer = landscape.textureLayers.back();
                            for (std::size_t offset = 0; offset + 8 <= field.data.size(); offset += 8)
                            {
                                layer.opacityPoints.push_back({
                                    valueAt<std::uint16_t>(field.data, offset),
                                    valueAt<float>(field.data, offset + 4)});
                            }
                        }
                    }
                    if (landscape.hasHeights)
                        mWorld.landscapes[landscape.cell] = std::move(landscape);
                }
            }

            void expandStaticCollections()
            {
                std::vector<PlacedReference> expanded;
                expanded.reserve(mWorld.references.size());
                const auto hasDistantModel = [&](std::uint32_t baseObject) {
                    const auto found = mWorld.modelLods.find(baseObject);
                    return found != mWorld.modelLods.end()
                        && std::ranges::any_of(found->second,
                            [](const std::string& path) { return !path.empty(); });
                };
                for (const auto& reference : mWorld.references)
                {
                    if (mLodProxyBases.contains(reference.baseObject))
                        continue;
                    const auto collection = mWorld.staticCollections.find(reference.baseObject);
                    if (collection == mWorld.staticCollections.end()
                        || mWorld.models.contains(reference.baseObject))
                    {
                        auto placed = reference;
                        placed.replacesDistantLod = collection != mWorld.staticCollections.end()
                            || hasDistantModel(reference.baseObject);
                        expanded.push_back(placed);
                        continue;
                    }
                    std::uint32_t partIndex = 0;
                    for (const auto& part : collection->second)
                    {
                        if (mLodProxyBases.contains(part.baseObject))
                            continue;
                        if (!mWorld.models.contains(part.baseObject)
                            && !mWorld.modelLods.contains(part.baseObject))
                            continue;
                        PlacedReference placed = reference;
                        placed.instanceId = (static_cast<std::uint64_t>(++partIndex) << 32u)
                            | reference.formId;
                        placed.baseObject = part.baseObject;
                        std::ranges::copy(part.position, placed.localPosition);
                        std::ranges::copy(part.rotation, placed.localRotation);
                        placed.localScale = part.scale;
                        placed.hasLocalTransform = true;
                        placed.replacesDistantLod = true;

                        const float cx = std::cos(reference.rotation[0]);
                        const float sx = std::sin(reference.rotation[0]);
                        const float cy = std::cos(reference.rotation[1]);
                        const float sy = std::sin(reference.rotation[1]);
                        const float cz = std::cos(reference.rotation[2]);
                        const float sz = std::sin(reference.rotation[2]);
                        const float x = part.position[0] * reference.scale;
                        const float y = part.position[1] * reference.scale;
                        const float z = part.position[2] * reference.scale;
                        const float worldX = reference.position[0]
                            + cy * cz * x + (sx * sy * cz - cx * sz) * y
                            + (cx * sy * cz + sx * sz) * z;
                        const float worldY = reference.position[1]
                            + cy * sz * x + (sx * sy * sz + cx * cz) * y
                            + (cx * sy * sz - sx * cz) * z;
                        constexpr float exteriorCellSize = 4096.0f;
                        placed.cell = {
                            static_cast<std::int32_t>(std::floor(worldX / exteriorCellSize)),
                            static_cast<std::int32_t>(std::floor(worldY / exteriorCellSize))};
                        expanded.push_back(placed);
                    }
                }
                mWorld.references = std::move(expanded);
            }
        };
    }

    ResolvedMaterialSwap MaterialSwapRecord::resolve(std::string_view materialPath) const
    {
        const std::string original = canonicalMaterialPath(materialPath);
        for (const auto& substitution : substitutions)
        {
            if (substitution.original.empty() || substitution.replacement.empty())
                continue;
            if (original == substitution.original)
                return {substitution.replacement, substitution.colorRemapIndex, true};
            if (substitution.original.ends_with('*'))
            {
                const std::string_view prefix(substitution.original.data(),
                    substitution.original.size() - 1);
                if (original.starts_with(prefix))
                {
                    return {substitution.replacement + original.substr(prefix.size()),
                        substitution.colorRemapIndex, true};
                }
            }
            else if (const auto found = original.find(substitution.original);
                found != std::string::npos)
            {
                std::string replacement = original;
                replacement.replace(found, substitution.original.size(), substitution.replacement);
                return {std::move(replacement), substitution.colorRemapIndex, true};
            }
        }
        return {std::string(materialPath), std::nullopt, false};
    }

    CellCoord ExteriorWorld::suggestedCenter() const
    {
        std::map<CellCoord, std::size_t> counts;
        for (const auto& reference : references)
            if (landscapes.contains(reference.cell))
                ++counts[reference.cell];
        const auto best = std::max_element(counts.begin(), counts.end(),
            [](const auto& left, const auto& right) { return left.second < right.second; });
        return best == counts.end() ? CellCoord{} : best->first;
    }

    ExteriorWorld loadCommonwealth(const std::filesystem::path& esmPath)
    {
        ExteriorWorld world = Parser(esmPath, commonwealthFormId).parse();
        world.defaultLandscapeTexture = 0x000AB07E;
        return world;
    }

    ExteriorWorld loadExteriorWorld(const std::filesystem::path& esmPath)
    {
        ExteriorWorld world = Parser(esmPath).parse();
        world.defaultLandscapeTexture = 0x000AB07E;
        return world;
    }
}
