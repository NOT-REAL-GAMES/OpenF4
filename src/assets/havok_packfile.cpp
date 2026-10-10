#include "assets/havok_packfile.hpp"

#include <NifFile.hpp>
#include <bhk.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <istream>
#include <limits>
#include <stdexcept>
#include <streambuf>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include <glm/ext/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

namespace openf4::assets
{
    namespace
    {
        constexpr std::array<std::byte, 8> magic{
            std::byte{0x57}, std::byte{0xe0}, std::byte{0xe0}, std::byte{0x57},
            std::byte{0x10}, std::byte{0xc0}, std::byte{0xc0}, std::byte{0x10}};
        constexpr std::size_t headerSize = 0x40;
        constexpr std::size_t sectionHeaderSize = 0x40;
        constexpr std::uint32_t invalid = 0xffffffffu;

        class Reader
        {
        public:
            explicit Reader(std::span<const std::byte> bytes) : mBytes(bytes) {}

            void requireRange(std::size_t offset, std::size_t size) const
            {
                if (offset > mBytes.size() || size > mBytes.size() - offset)
                    throw std::runtime_error("Havok packfile offset is out of bounds");
            }

            template <class T>
            [[nodiscard]] T read(std::size_t offset) const
            {
                static_assert(std::is_trivially_copyable_v<T>);
                requireRange(offset, sizeof(T));
                T result{};
                std::memcpy(&result, mBytes.data() + offset, sizeof(T));
                return result;
            }

            [[nodiscard]] std::string string(std::size_t offset, std::size_t end) const
            {
                if (end > mBytes.size() || offset >= end)
                    throw std::runtime_error("Havok packfile string is out of bounds");
                const char* begin = reinterpret_cast<const char*>(mBytes.data() + offset);
                const void* terminator = std::memchr(begin, 0, end - offset);
                if (!terminator)
                    throw std::runtime_error("Havok packfile string is not terminated");
                return {begin, static_cast<const char*>(terminator)};
            }

            [[nodiscard]] std::span<const std::byte> bytes() const noexcept { return mBytes; }

        private:
            std::span<const std::byte> mBytes;
        };

        struct Section
        {
            std::string name;
            std::uint32_t start = 0;
            std::uint32_t localFixups = 0;
            std::uint32_t globalFixups = 0;
            std::uint32_t virtualFixups = 0;
            std::uint32_t exports = 0;
            std::uint32_t imports = 0;
            std::uint32_t end = 0;
        };

        struct Fixups
        {
            std::unordered_map<std::uint32_t, std::uint32_t> local;
            struct Global { std::uint32_t section; std::uint32_t offset; };
            std::unordered_map<std::uint32_t, Global> global;
            std::unordered_map<std::uint32_t, std::string> classes;
        };

        struct Packfile
        {
            explicit Packfile(std::span<const std::byte> bytes) : reader(bytes) {}

            Reader reader;
            std::vector<Section> sections;
            Fixups fixups;
            std::uint32_t dataSectionIndex = 0;
            std::uint32_t rootOffset = 0;
            std::string rootClass;
        };

        class SpanBuffer final : public std::streambuf
        {
        public:
            explicit SpanBuffer(std::span<const std::byte> bytes)
            {
                char* begin = reinterpret_cast<char*>(const_cast<std::byte*>(bytes.data()));
                setg(begin, begin, begin + bytes.size());
            }
        };

        [[nodiscard]] std::size_t absolute(const Section& section, std::uint32_t relative)
        {
            if (relative > section.end)
                throw std::runtime_error("Havok section-relative offset is out of bounds");
            return static_cast<std::size_t>(section.start) + relative;
        }

        [[nodiscard]] std::vector<Section> readSections(const Reader& reader, std::uint32_t count)
        {
            if (count != 3)
                throw std::runtime_error("Unsupported Havok packfile section count");
            std::vector<Section> sections;
            sections.reserve(count);
            for (std::uint32_t index = 0; index < count; ++index)
            {
                const std::size_t offset = headerSize + index * sectionHeaderSize;
                reader.requireRange(offset, sectionHeaderSize);
                Section section;
                section.name = reader.string(offset, offset + 20);
                section.start = reader.read<std::uint32_t>(offset + 20);
                section.localFixups = reader.read<std::uint32_t>(offset + 24);
                section.globalFixups = reader.read<std::uint32_t>(offset + 28);
                section.virtualFixups = reader.read<std::uint32_t>(offset + 32);
                section.exports = reader.read<std::uint32_t>(offset + 36);
                section.imports = reader.read<std::uint32_t>(offset + 40);
                section.end = reader.read<std::uint32_t>(offset + 44);
                if (!(section.localFixups <= section.globalFixups
                    && section.globalFixups <= section.virtualFixups
                    && section.virtualFixups <= section.exports
                    && section.exports <= section.imports
                    && section.imports <= section.end))
                    throw std::runtime_error("Invalid Havok section layout");
                reader.requireRange(section.start, section.end);
                sections.push_back(std::move(section));
            }
            return sections;
        }

        [[nodiscard]] bool allInvalid(const Reader& reader, std::size_t offset, std::size_t size)
        {
            reader.requireRange(offset, size);
            for (std::size_t index = 0; index < size; ++index)
                if (reader.bytes()[offset + index] != std::byte{0xff})
                    return false;
            return true;
        }

        [[nodiscard]] Fixups readFixups(const Reader& reader,
            const std::vector<Section>& sections, std::uint32_t dataSectionIndex,
            std::uint32_t classSectionIndex)
        {
            const Section& data = sections.at(dataSectionIndex);
            const Section& names = sections.at(classSectionIndex);
            Fixups result;
            for (std::uint32_t offset = data.localFixups; offset + 8 <= data.globalFixups; offset += 8)
            {
                const std::size_t entry = absolute(data, offset);
                if (allInvalid(reader, entry, 8))
                    continue;
                const auto source = reader.read<std::uint32_t>(entry);
                const auto destination = reader.read<std::uint32_t>(entry + 4);
                if (source >= data.localFixups || destination >= data.localFixups
                    || !result.local.emplace(source, destination).second)
                    throw std::runtime_error("Invalid Havok local fixup");
            }
            for (std::uint32_t offset = data.globalFixups; offset + 12 <= data.virtualFixups; offset += 12)
            {
                const std::size_t entry = absolute(data, offset);
                if (allInvalid(reader, entry, 12))
                    continue;
                const auto source = reader.read<std::uint32_t>(entry);
                Fixups::Global destination{reader.read<std::uint32_t>(entry + 4),
                    reader.read<std::uint32_t>(entry + 8)};
                if (source >= data.localFixups || destination.section >= sections.size()
                    || destination.offset >= sections[destination.section].localFixups
                    || !result.global.emplace(source, destination).second)
                    throw std::runtime_error("Invalid Havok global fixup");
            }
            for (std::uint32_t offset = data.virtualFixups; offset + 12 <= data.exports; offset += 12)
            {
                const std::size_t entry = absolute(data, offset);
                if (allInvalid(reader, entry, 12))
                    continue;
                const auto object = reader.read<std::uint32_t>(entry);
                const auto section = reader.read<std::uint32_t>(entry + 4);
                const auto name = reader.read<std::uint32_t>(entry + 8);
                if (object >= data.localFixups || section != classSectionIndex || name >= names.localFixups
                    || !result.classes.emplace(object,
                        reader.string(absolute(names, name), absolute(names, names.localFixups))).second)
                    throw std::runtime_error("Invalid Havok virtual fixup");
            }
            return result;
        }

        [[nodiscard]] std::pair<std::uint32_t, std::uint32_t> arrayInfo(
            const Reader& reader, const Section& data, const Fixups& fixups, std::uint32_t slot)
        {
            const std::uint32_t count = reader.read<std::uint32_t>(absolute(data, slot + 8));
            if (count == 0)
                return {0, 0};
            const auto pointer = fixups.local.find(slot);
            if (pointer == fixups.local.end() || pointer->second >= data.localFixups)
                throw std::runtime_error("Havok array has no valid local fixup");
            return {pointer->second, count};
        }

        [[nodiscard]] Packfile readPackfile(std::span<const std::byte> bytes)
        {
            Packfile result(bytes);
            const Reader& reader = result.reader;
            reader.requireRange(0, headerSize);
            if (!std::equal(magic.begin(), magic.end(), bytes.begin()))
                throw std::runtime_error("Not a Havok packfile");
            if (reader.read<std::uint32_t>(12) != 11
                || reader.read<std::uint8_t>(16) != 8
                || reader.read<std::uint8_t>(17) != 1
                || reader.read<std::uint8_t>(18) != 0
                || reader.read<std::uint8_t>(19) != 1)
                throw std::runtime_error("Unsupported Havok packfile layout");
            const std::uint32_t sectionCount = reader.read<std::uint32_t>(20);
            result.dataSectionIndex = reader.read<std::uint32_t>(24);
            result.rootOffset = reader.read<std::uint32_t>(28);
            const std::uint32_t classSectionIndex = reader.read<std::uint32_t>(32);
            if (reader.string(40, 56) != "hk_2014.1.0-r1")
                throw std::runtime_error("Unsupported Havok packfile version");

            result.sections = readSections(reader, sectionCount);
            if (result.dataSectionIndex >= result.sections.size()
                || classSectionIndex >= result.sections.size())
                throw std::runtime_error("Invalid Havok contents section");
            const Section& data = result.sections[result.dataSectionIndex];
            if (result.rootOffset >= data.localFixups)
                throw std::runtime_error("Invalid Havok root object offset");
            result.fixups = readFixups(
                reader, result.sections, result.dataSectionIndex, classSectionIndex);
            const auto rootClass = result.fixups.classes.find(result.rootOffset);
            if (rootClass == result.fixups.classes.end())
                throw std::runtime_error("Havok root has no reflected class");
            result.rootClass = rootClass->second;
            return result;
        }

        [[nodiscard]] std::uint32_t dataPointer(
            const Packfile& packfile, std::uint32_t slot, std::string_view label)
        {
            if (const auto local = packfile.fixups.local.find(slot);
                local != packfile.fixups.local.end())
                return local->second;
            const auto pointer = packfile.fixups.global.find(slot);
            if (pointer == packfile.fixups.global.end()
                || pointer->second.section != packfile.dataSectionIndex)
                throw std::runtime_error(std::string(label) + " has no data fixup");
            return pointer->second.offset;
        }

        [[nodiscard]] std::string pointedString(
            const Packfile& packfile, std::uint32_t slot, std::string_view label)
        {
            const Section& data = packfile.sections[packfile.dataSectionIndex];
            if (const auto pointer = packfile.fixups.local.find(slot);
                pointer != packfile.fixups.local.end())
                return packfile.reader.string(
                    absolute(data, pointer->second), absolute(data, data.localFixups));
            const auto pointer = packfile.fixups.global.find(slot);
            if (pointer == packfile.fixups.global.end())
                throw std::runtime_error(std::string(label) + " has no string fixup");
            const Section& target = packfile.sections[pointer->second.section];
            return packfile.reader.string(
                absolute(target, pointer->second.offset),
                absolute(target, target.localFixups));
        }

        [[nodiscard]] HavokConvexPolytope readConvexShape(const Reader& reader,
            const Section& data, std::uint32_t shapeOffset)
        {
            const std::size_t shape = absolute(data, shapeOffset);
            HavokConvexPolytope result;
            result.convexRadius = reader.read<float>(shape + 0x14);
            const auto vertexCount = reader.read<std::uint16_t>(shape + 0x30);
            const auto vertexDelta = reader.read<std::uint16_t>(shape + 0x32);
            if (!std::isfinite(result.convexRadius) || result.convexRadius < 0.0f
                || vertexCount < 4)
                throw std::runtime_error("Invalid Havok convex shape dimensions");

            const std::size_t vertices = shape + 0x30 + vertexDelta;
            reader.requireRange(vertices, static_cast<std::size_t>(vertexCount) * 16);
            result.vertices.reserve(vertexCount);
            for (std::uint16_t index = 0; index < vertexCount; ++index)
            {
                const std::size_t vertex = vertices + static_cast<std::size_t>(index) * 16;
                const glm::vec3 decoded(reader.read<float>(vertex),
                    reader.read<float>(vertex + 4), reader.read<float>(vertex + 8));
                if (!std::isfinite(decoded.x) || !std::isfinite(decoded.y)
                    || !std::isfinite(decoded.z))
                    throw std::runtime_error("Havok convex shape has an invalid vertex");
                result.vertices.push_back(decoded);
            }
            return result;
        }

        [[nodiscard]] HavokConvexPolytope readConvexPolytope(const Reader& reader,
            const Section& data, std::uint32_t shapeOffset)
        {
            const std::size_t shape = absolute(data, shapeOffset);
            HavokConvexPolytope result = readConvexShape(reader, data, shapeOffset);
            const auto faceCount = reader.read<std::uint16_t>(shape + 0x44);
            const auto faceDelta = reader.read<std::uint16_t>(shape + 0x46);
            const auto faceVertexCount = reader.read<std::uint16_t>(shape + 0x48);
            const auto faceVertexDelta = reader.read<std::uint16_t>(shape + 0x4a);
            if (faceCount == 0 || faceVertexCount == 0)
                throw std::runtime_error("Invalid Havok convex polytope dimensions");

            const std::size_t faces = shape + 0x44 + faceDelta;
            const std::size_t faceVertices = shape + 0x48 + faceVertexDelta;
            reader.requireRange(faces, static_cast<std::size_t>(faceCount) * 4);
            reader.requireRange(faceVertices, faceVertexCount);
            const auto vertexCount = static_cast<std::uint32_t>(result.vertices.size());
            for (std::uint16_t faceIndex = 0; faceIndex < faceCount; ++faceIndex)
            {
                const std::size_t face = faces + static_cast<std::size_t>(faceIndex) * 4;
                const auto first = reader.read<std::uint16_t>(face);
                const auto count = reader.read<std::uint8_t>(face + 2);
                if (count < 3 || first > faceVertexCount || count > faceVertexCount - first)
                    throw std::runtime_error("Invalid Havok convex face");
                const auto root = reader.read<std::uint8_t>(faceVertices + first);
                if (root >= vertexCount)
                    throw std::runtime_error("Havok convex face references an invalid vertex");
                for (std::uint8_t corner = 1; corner + 1 < count; ++corner)
                {
                    const auto second = reader.read<std::uint8_t>(faceVertices + first + corner);
                    const auto third = reader.read<std::uint8_t>(faceVertices + first + corner + 1);
                    if (second >= vertexCount || third >= vertexCount
                        || root == second || root == third || second == third)
                        throw std::runtime_error("Havok convex face is degenerate");
                    result.indices.insert(result.indices.end(), {root, second, third});
                }
            }
            return result;
        }

        [[nodiscard]] HavokCapsule readCapsule(const Reader& reader,
            const Section& data, std::uint32_t shapeOffset)
        {
            const std::size_t shape = absolute(data, shapeOffset);
            HavokCapsule result;
            result.radius = reader.read<float>(shape + 0x14);
            result.pointA = {reader.read<float>(shape + 0x50),
                reader.read<float>(shape + 0x54), reader.read<float>(shape + 0x58)};
            result.pointB = {reader.read<float>(shape + 0x60),
                reader.read<float>(shape + 0x64), reader.read<float>(shape + 0x68)};
            if (!std::isfinite(result.radius) || result.radius <= 0.0f
                || !std::isfinite(result.pointA.x) || !std::isfinite(result.pointA.y)
                || !std::isfinite(result.pointA.z) || !std::isfinite(result.pointB.x)
                || !std::isfinite(result.pointB.y) || !std::isfinite(result.pointB.z))
                throw std::runtime_error("Invalid Havok capsule shape");
            return result;
        }

        [[nodiscard]] bool finite(const glm::vec3& value)
        {
            return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
        }

        void requireArrayRange(const Section& data, std::uint32_t offset,
            std::uint32_t count, std::uint32_t stride, std::string_view name)
        {
            if (stride == 0 || offset > data.localFixups
                || count > (data.localFixups - offset) / stride)
                throw std::runtime_error("Havok " + std::string(name) + " array is out of bounds");
        }

        [[nodiscard]] std::pair<std::uint32_t, std::uint32_t> packedRange(
            std::uint32_t value)
        {
            return {(value >> 8) & 0xffffu, value & 0xffu};
        }

        [[nodiscard]] bool degenerate(const std::vector<glm::vec3>& vertices,
            std::uint32_t a, std::uint32_t b, std::uint32_t c)
        {
            if (a == b || a == c || b == c)
                return true;
            const glm::vec3 cross = glm::cross(vertices[b] - vertices[a],
                vertices[c] - vertices[a]);
            return glm::dot(cross, cross) <= 1.0e-12f;
        }

        [[nodiscard]] HavokTriangleMesh readCompressedMesh(const Reader& reader,
            const Section& data, const Fixups& fixups, std::uint32_t dataSectionIndex,
            std::uint32_t shapeOffset)
        {
            const auto dataFixup = fixups.global.find(shapeOffset + 0x60);
            if (dataFixup == fixups.global.end()
                || dataFixup->second.section != dataSectionIndex)
                throw std::runtime_error("Havok compressed mesh has no shape-data fixup");
            const auto dataClass = fixups.classes.find(dataFixup->second.offset);
            if (dataClass == fixups.classes.end()
                || dataClass->second != "hknpCompressedMeshShapeData")
                throw std::runtime_error("Havok compressed mesh data has an invalid class");

            const std::uint32_t tree = dataFixup->second.offset + 0x10;
            if (tree < dataFixup->second.offset)
                throw std::runtime_error("Havok compressed mesh tree offset overflowed");
            reader.requireRange(absolute(data, tree), 0xa0);

            const auto readVector = [&](std::uint32_t offset) {
                const std::size_t source = absolute(data, offset);
                return glm::vec3(reader.read<float>(source), reader.read<float>(source + 4),
                    reader.read<float>(source + 8));
            };
            const glm::vec3 domainMinimum = readVector(tree + 0x10);
            const glm::vec3 domainMaximum = readVector(tree + 0x20);
            if (!finite(domainMinimum) || !finite(domainMaximum)
                || glm::any(glm::lessThan(domainMaximum, domainMinimum)))
                throw std::runtime_error("Havok compressed mesh has an invalid AABB");

            const auto [sectionsOffset, sectionCount] = arrayInfo(
                reader, data, fixups, tree + 0x40);
            const auto [primitivesOffset, primitiveCount] = arrayInfo(
                reader, data, fixups, tree + 0x50);
            const auto [sharedIndicesOffset, sharedIndexCount] = arrayInfo(
                reader, data, fixups, tree + 0x60);
            const auto [packedVerticesOffset, packedVertexCount] = arrayInfo(
                reader, data, fixups, tree + 0x70);
            const auto [sharedVerticesOffset, sharedVertexCount] = arrayInfo(
                reader, data, fixups, tree + 0x80);
            const auto [runsOffset, runCount] = arrayInfo(
                reader, data, fixups, tree + 0x90);
            if (sectionCount == 0)
                throw std::runtime_error("Havok compressed mesh has no sections");
            requireArrayRange(data, sectionsOffset, sectionCount, 0x60, "mesh section");
            requireArrayRange(data, primitivesOffset, primitiveCount, 4, "mesh primitive");
            requireArrayRange(data, sharedIndicesOffset, sharedIndexCount, 2,
                "mesh shared-index");
            requireArrayRange(data, packedVerticesOffset, packedVertexCount, 4,
                "mesh packed-vertex");
            requireArrayRange(data, sharedVerticesOffset, sharedVertexCount, 8,
                "mesh shared-vertex");
            requireArrayRange(data, runsOffset, runCount, 4, "mesh primitive-run");

            HavokTriangleMesh result;
            result.sectionCount = sectionCount;
            result.vertices.reserve(static_cast<std::size_t>(sharedVertexCount)
                + packedVertexCount);
            result.primitiveDataRuns.reserve(runCount);
            for (std::uint32_t index = 0; index < sharedVertexCount; ++index)
            {
                const std::uint64_t packed = reader.read<std::uint64_t>(
                    absolute(data, sharedVerticesOffset + index * 8));
                const glm::vec3 normalized(
                    static_cast<float>(packed & 0x1fffffu) / 2097151.0f,
                    static_cast<float>((packed >> 21) & 0x1fffffu) / 2097151.0f,
                    static_cast<float>((packed >> 42) & 0x3fffffu) / 4194303.0f);
                const glm::vec3 vertex = domainMinimum
                    + normalized * (domainMaximum - domainMinimum);
                if (!finite(vertex))
                    throw std::runtime_error("Havok compressed shared vertex is invalid");
                result.vertices.push_back(vertex);
            }

            for (std::uint32_t sectionIndex = 0; sectionIndex < sectionCount; ++sectionIndex)
            {
                const std::uint32_t sectionOffset = sectionsOffset + sectionIndex * 0x60;
                const std::size_t section = absolute(data, sectionOffset);
                std::array<float, 6> codec{};
                for (std::size_t index = 0; index < codec.size(); ++index)
                    codec[index] = reader.read<float>(section + 0x30 + index * 4);
                if (std::ranges::any_of(codec,
                        [](float value) { return !std::isfinite(value); })
                    || codec[3] < 0.0f || codec[4] < 0.0f || codec[5] < 0.0f)
                    throw std::runtime_error("Havok compressed mesh codec is invalid");

                const std::uint32_t firstPackedVertex =
                    reader.read<std::uint32_t>(section + 0x48);
                const std::uint32_t sharedOffset = packedRange(
                    reader.read<std::uint32_t>(section + 0x4c)).first;
                const auto [primitiveOffset, sectionPrimitiveCount] = packedRange(
                    reader.read<std::uint32_t>(section + 0x50));
                const auto [runOffset, sectionRunCount] = packedRange(
                    reader.read<std::uint32_t>(section + 0x54));
                const std::uint32_t localVertexCount = reader.read<std::uint8_t>(section + 0x58);
                const std::uint32_t localSharedCount = reader.read<std::uint8_t>(section + 0x59);
                if (firstPackedVertex > packedVertexCount
                    || localVertexCount > packedVertexCount - firstPackedVertex
                    || sharedOffset > sharedIndexCount
                    || localSharedCount > sharedIndexCount - sharedOffset
                    || primitiveOffset > primitiveCount
                    || sectionPrimitiveCount > primitiveCount - primitiveOffset
                    || runOffset > runCount || sectionRunCount > runCount - runOffset)
                    throw std::runtime_error("Havok compressed mesh section range is invalid");

                const std::uint32_t localBase = static_cast<std::uint32_t>(result.vertices.size());
                for (std::uint32_t index = 0; index < localVertexCount; ++index)
                {
                    const std::uint32_t packed = reader.read<std::uint32_t>(absolute(data,
                        packedVerticesOffset + (firstPackedVertex + index) * 4));
                    const glm::vec3 vertex(
                        static_cast<float>(packed & 0x7ffu) * codec[3] + codec[0],
                        static_cast<float>((packed >> 11) & 0x7ffu) * codec[4] + codec[1],
                        static_cast<float>((packed >> 22) & 0x3ffu) * codec[5] + codec[2]);
                    if (!finite(vertex))
                        throw std::runtime_error("Havok compressed local vertex is invalid");
                    result.vertices.push_back(vertex);
                }

                const auto resolveVertex = [&](std::uint8_t index) -> std::uint32_t {
                    if (index < localVertexCount)
                        return localBase + index;
                    const std::uint32_t shared = index - localVertexCount;
                    if (shared >= localSharedCount)
                        throw std::runtime_error(
                            "Havok compressed primitive has an invalid shared vertex");
                    const std::uint32_t resolved = reader.read<std::uint16_t>(absolute(data,
                        sharedIndicesOffset + (sharedOffset + shared) * 2));
                    if (resolved >= sharedVertexCount)
                        throw std::runtime_error(
                            "Havok compressed shared index is out of bounds");
                    return resolved;
                };
                const auto appendTriangle = [&](std::uint32_t a, std::uint32_t b,
                                                std::uint32_t c) {
                    if (a >= result.vertices.size() || b >= result.vertices.size()
                        || c >= result.vertices.size() || degenerate(result.vertices, a, b, c))
                        throw std::runtime_error("Havok compressed primitive is degenerate");
                    result.indices.insert(result.indices.end(), {a, b, c});
                };
                for (std::uint32_t index = 0; index < sectionPrimitiveCount; ++index)
                {
                    const std::size_t primitive = absolute(data,
                        primitivesOffset + (primitiveOffset + index) * 4);
                    const std::uint32_t a = resolveVertex(reader.read<std::uint8_t>(primitive));
                    const std::uint32_t b = resolveVertex(reader.read<std::uint8_t>(primitive + 1));
                    const std::uint32_t c = resolveVertex(reader.read<std::uint8_t>(primitive + 2));
                    const auto rawD = reader.read<std::uint8_t>(primitive + 3);
                    appendTriangle(a, b, c);
                    if (rawD != reader.read<std::uint8_t>(primitive + 2))
                    {
                        const std::uint32_t d = resolveVertex(rawD);
                        appendTriangle(c, d, a);
                    }
                }
                for (std::uint32_t index = 0; index < sectionRunCount; ++index)
                {
                    const std::size_t run = absolute(data,
                        runsOffset + (runOffset + index) * 4);
                    HavokPrimitiveDataRun decoded;
                    decoded.section = sectionIndex;
                    decoded.value = reader.read<std::uint16_t>(run);
                    decoded.index = reader.read<std::uint8_t>(run + 2);
                    decoded.count = reader.read<std::uint8_t>(run + 3);
                    if (decoded.index > sectionPrimitiveCount
                        || decoded.count > sectionPrimitiveCount - decoded.index)
                        throw std::runtime_error(
                            "Havok compressed primitive run is out of bounds");
                    result.primitiveDataRuns.push_back(decoded);
                }
            }
            if (result.indices.empty())
                throw std::runtime_error("Havok compressed mesh has no triangles");
            return result;
        }

        struct CompoundTransform
        {
            glm::mat4 transform{1.0f};
            glm::vec3 scale{1.0f};
        };

        [[nodiscard]] CompoundTransform readCompoundTransform(const Reader& reader,
            const Section& data, std::uint32_t instanceOffset)
        {
            const std::size_t instance = absolute(data, instanceOffset);
            CompoundTransform result;
            for (glm::length_t column = 0; column < 3; ++column)
            {
                const std::size_t source = instance + column * 0x10;
                result.transform[column] = glm::vec4(reader.read<float>(source),
                    reader.read<float>(source + 4), reader.read<float>(source + 8), 0.0f);
            }
            result.transform[3] = glm::vec4(reader.read<float>(instance + 0x30),
                reader.read<float>(instance + 0x34), reader.read<float>(instance + 0x38), 1.0f);
            result.scale = {reader.read<float>(instance + 0x40),
                reader.read<float>(instance + 0x44), reader.read<float>(instance + 0x48)};
            if (!finite(glm::vec3(result.transform[0]))
                || !finite(glm::vec3(result.transform[1]))
                || !finite(glm::vec3(result.transform[2]))
                || !finite(glm::vec3(result.transform[3])) || !finite(result.scale)
                || result.scale.x <= 0.0f || result.scale.y <= 0.0f
                || result.scale.z <= 0.0f)
                throw std::runtime_error("Havok compound child has an invalid transform");
            return result;
        }

        void appendTriangleMesh(HavokRigidBody& body, HavokTriangleMesh child,
            const CompoundTransform& placement)
        {
            if (!body.triangleMesh)
                body.triangleMesh.emplace();
            HavokTriangleMesh& result = *body.triangleMesh;
            if (result.vertices.size() > std::numeric_limits<std::uint32_t>::max()
                - child.vertices.size())
                throw std::runtime_error("Havok compound mesh has too many vertices");
            const std::uint32_t vertexBase =
                static_cast<std::uint32_t>(result.vertices.size());
            for (const glm::vec3& vertex : child.vertices)
                result.vertices.push_back(glm::vec3(placement.transform
                    * glm::vec4(vertex * placement.scale, 1.0f)));
            for (const std::uint32_t childIndex : child.indices)
                result.indices.push_back(vertexBase + childIndex);
            for (auto run : child.primitiveDataRuns)
            {
                run.section += result.sectionCount;
                result.primitiveDataRuns.push_back(run);
            }
            result.sectionCount += child.sectionCount;
        }

        void readCompoundShapes(const Reader& reader, const Section& data,
            const Fixups& fixups, std::uint32_t dataSectionIndex,
            std::uint32_t shapeOffset, HavokRigidBody& body)
        {
            constexpr std::uint32_t instanceStride = 0x80;
            const auto [instancesOffset, instanceCount] = arrayInfo(
                reader, data, fixups, shapeOffset + 0x60);
            if (instanceCount == 0 || instanceCount > 4096
                || instanceCount > (data.localFixups - instancesOffset) / instanceStride)
                throw std::runtime_error("Havok compound instance array is out of bounds");

            body.parts.reserve(body.parts.size() + instanceCount);
            body.capsules.reserve(body.capsules.size() + instanceCount);
            for (std::uint32_t index = 0; index < instanceCount; ++index)
            {
                const std::uint32_t instanceOffset = instancesOffset + index * instanceStride;
                const auto childFixup = fixups.global.find(instanceOffset + 0x50);
                if (childFixup == fixups.global.end()
                    || childFixup->second.section != dataSectionIndex)
                    throw std::runtime_error("Havok compound child has no shape fixup");
                const auto childClass = fixups.classes.find(childFixup->second.offset);
                if (childClass == fixups.classes.end())
                    throw std::runtime_error("Havok compound child has no reflected class");
                const CompoundTransform placement = readCompoundTransform(
                    reader, data, instanceOffset);
                if (childClass->second == "hknpConvexPolytopeShape"
                    || childClass->second == "hknpConvexShape")
                {
                    HavokConvexPart part;
                    part.transform = placement.transform;
                    part.scale = placement.scale;
                    part.polytope = childClass->second == "hknpConvexPolytopeShape"
                        ? readConvexPolytope(reader, data, childFixup->second.offset)
                        : readConvexShape(reader, data, childFixup->second.offset);
                    body.parts.push_back(std::move(part));
                }
                else if (childClass->second == "hknpCapsuleShape")
                {
                    body.capsules.push_back({placement.transform, placement.scale,
                        readCapsule(reader, data, childFixup->second.offset)});
                }
                else if (childClass->second == "hknpCompressedMeshShape")
                {
                    appendTriangleMesh(body, readCompressedMesh(reader, data, fixups,
                        dataSectionIndex, childFixup->second.offset), placement);
                }
                else
                    throw std::runtime_error("Unsupported Havok compound child class: "
                        + childClass->second);
            }
        }

        enum class MaterialProperties { None, Uniform, Unsupported };

        MaterialProperties shapeMaterialProperties(const Packfile& packfile, std::uint32_t shape,
            const HavokRigidBody& body)
        {
            const auto& fixups = packfile.fixups;
            const auto& data = packfile.sections[packfile.dataSectionIndex];
            const auto& reader = packfile.reader;
            if (!fixups.global.contains(shape + 0x20) && !fixups.local.contains(shape + 0x20))
                return MaterialProperties::None;
            const auto properties = dataPointer(packfile, shape + 0x20, "Havok shape properties");
            const auto type = fixups.classes.find(properties);
            if (type == fixups.classes.end() || type->second != "hkRefCountedProperties")
                return MaterialProperties::Unsupported;
            const auto [entries, count] = arrayInfo(reader, data, fixups, properties);
            requireArrayRange(data, entries, count, 16, "shape properties");
            if (count != 1 || reader.read<std::uint16_t>(absolute(data, entries + 8)) != 0xf601
                || reader.read<std::uint16_t>(absolute(data, entries + 10)) != 0)
                return MaterialProperties::Unsupported;
            const auto material = dataPointer(packfile, entries, "Havok shape material properties");
            const auto materialType = fixups.classes.find(material);
            if (materialType == fixups.classes.end() || materialType->second != "hknpBSMaterialProperties")
                return MaterialProperties::Unsupported;
            const auto [materials, materialCount] = arrayInfo(reader, data, fixups, material + 0x10);
            requireArrayRange(data, materials, materialCount, 24, "Bethesda material");
            if (materialCount != 1) return MaterialProperties::Unsupported;
            const auto entry = absolute(data, materials);
            if (reader.read<std::uint64_t>(entry) || reader.read<std::uint64_t>(entry + 8)
                || reader.read<std::uint32_t>(entry + 16) != body.collisionFilterInfo
                || reader.read<std::uint32_t>(entry + 20) != body.materialId)
                return MaterialProperties::Unsupported;
            return MaterialProperties::Uniform;
        }

        [[nodiscard]] HavokPhysicsSystem readPhysicsSystemData(const Packfile& packfile)
        {
            const Reader& reader = packfile.reader;
            const Section& data = packfile.sections[packfile.dataSectionIndex];
            const Fixups& fixups = packfile.fixups;
            const auto [bodiesOffset, bodyCount] = arrayInfo(
                reader, data, fixups, packfile.rootOffset + 0x40);
            const auto [motionOffset, motionCount] = arrayInfo(
                reader, data, fixups, packfile.rootOffset + 0x30);
            const auto [materialsOffset, materialCount] = arrayInfo(
                reader, data, fixups, packfile.rootOffset + 0x10);
            const auto [propertiesOffset, propertiesCount] = arrayInfo(
                reader, data, fixups, packfile.rootOffset + 0x20);
            requireArrayRange(data, materialsOffset, materialCount, 0x50, "material");
            requireArrayRange(data, propertiesOffset, propertiesCount, 0x40, "motion properties");
            if (bodyCount > (data.localFixups - bodiesOffset) / 0x60)
                throw std::runtime_error("Havok body array is out of bounds");
            if (motionCount > bodyCount
                || motionCount > (data.localFixups - motionOffset) / 0x70)
                throw std::runtime_error("Havok motion array is out of bounds");

            HavokPhysicsSystem system;
            if (reader.read<std::uint32_t>(absolute(data, packfile.rootOffset + 0x58)) != 0)
                system.nativeRewriteBlockReason = "Havok constraints must be preserved in their original packfile";
            if (fixups.local.contains(packfile.rootOffset + 0x70)
                || fixups.global.contains(packfile.rootOffset + 0x70))
                system.nativeRewriteBlockReason = "Named Havok systems require raw preservation";
            for (const auto& [offset, name] : fixups.classes)
            {
                if (name != "hknpPhysicsSystemData" && name != "hknpConvexPolytopeShape"
                    && name != "hknpConvexShape" && name != "hknpCapsuleShape"
                    && name != "hknpDynamicCompoundShape" && name != "hknpDynamicCompoundShapeData"
                    && name != "hknpCompressedMeshShape" && name != "hknpCompressedMeshShapeData"
                    && name != "hkRefCountedProperties" && name != "hknpBSMaterialProperties")
                    system.nativeRewriteBlockReason = "Unsupported Havok property objects must be preserved in their original packfile";
            }
            system.bodies.reserve(bodyCount);
            for (std::uint32_t index = 0; index < bodyCount; ++index)
            {
                const std::uint32_t bodyOffset = bodiesOffset + index * 0x60;
                const auto shapeFixup = fixups.global.find(bodyOffset);
                if (shapeFixup == fixups.global.end()
                    || shapeFixup->second.section != packfile.dataSectionIndex)
                    throw std::runtime_error("Havok body has no shape fixup");
                const auto shapeClass = fixups.classes.find(shapeFixup->second.offset);
                if (shapeClass == fixups.classes.end())
                    throw std::runtime_error("Havok body shape has no reflected class");
                const std::size_t body = absolute(data, bodyOffset);
                HavokRigidBody result;
                result.bodyIndex = index;
                result.collisionFilterInfo = reader.read<std::uint32_t>(body + 0x14);
                result.materialId = reader.read<std::uint32_t>(absolute(
                    data, shapeFixup->second.offset + 0x18));
                if (reader.read<std::uint32_t>(absolute(data, shapeFixup->second.offset + 0x1c)) != 0)
                    system.nativeRewriteBlockReason = "Custom Havok shape user data requires raw preservation";
                const auto materialProperties = shapeMaterialProperties(packfile, shapeFixup->second.offset, result);
                bool uniformMeshMaterial = materialProperties == MaterialProperties::Uniform;
                if (materialProperties == MaterialProperties::Unsupported)
                    system.nativeRewriteBlockReason = "Havok shape properties require raw preservation";
                if (reader.read<std::uint32_t>(body + 0x18) != 0
                    || reader.read<std::uint64_t>(body + 0x28) != 0
                    || reader.read<std::uint32_t>(body + 0x1c) != 0
                    || fixups.local.contains(bodyOffset + 0x20) || fixups.global.contains(bodyOffset + 0x20))
                    system.nativeRewriteBlockReason = "Havok body flags, names, or custom data require raw preservation";
                const auto materialIndex = reader.read<std::uint16_t>(body + 0x12);
                if (materialCount != 0)
                {
                    if (materialIndex >= materialCount)
                        throw std::runtime_error("Havok body has an invalid material index");
                    const auto material = absolute(data, materialsOffset + materialIndex * 0x50);
                    if (reader.read<std::uint16_t>(material + 0x12) != reader.read<std::uint16_t>(material + 0x14)
                        || reader.read<std::uint8_t>(material + 0x10) != 0
                        || reader.read<std::uint64_t>(material + 0x40) != 0
                        || fixups.global.contains(materialsOffset + materialIndex * 0x50 + 0x30))
                        system.nativeRewriteBlockReason = "Custom Havok material properties require raw preservation";
                    result.friction = std::bit_cast<float>(
                        std::uint32_t(reader.read<std::uint16_t>(material + 0x12)) << 16);
                    result.restitution = std::bit_cast<float>(
                        std::uint32_t(reader.read<std::uint16_t>(material + 0x16)) << 16);
                    if (!std::isfinite(result.friction) || result.friction < 0.0f
                        || !std::isfinite(result.restitution) || result.restitution < 0.0f)
                        throw std::runtime_error("Havok body has invalid contact properties");
                }
                const std::uint32_t motionId = reader.read<std::uint32_t>(body + 0x0c);
                result.position = {reader.read<float>(body + 0x30),
                    reader.read<float>(body + 0x34), reader.read<float>(body + 0x38)};
                const float x = reader.read<float>(body + 0x40);
                const float y = reader.read<float>(body + 0x44);
                const float z = reader.read<float>(body + 0x48);
                const float w = reader.read<float>(body + 0x4c);
                result.rotation = glm::quat(w, x, y, z);
                if (!finite(result.position) || !std::isfinite(x) || !std::isfinite(y)
                    || !std::isfinite(z) || !std::isfinite(w)
                    || glm::dot(result.rotation, result.rotation)
                        <= std::numeric_limits<float>::epsilon())
                    throw std::runtime_error("Havok body has an invalid transform");
                result.rotation = glm::normalize(result.rotation);
                result.dynamic = motionId < motionCount;
                if (result.dynamic)
                {
                    const std::size_t inertia = absolute(
                        data, motionOffset + motionId * 0x70);
                    const float inverseMass = reader.read<float>(inertia + 4);
                    if (!std::isfinite(inverseMass) || inverseMass < 0.0f)
                        throw std::runtime_error("Havok body has an invalid inverse mass");
                    if (inverseMass > 0.0f)
                        result.mass = 1.0f / inverseMass;
                    if (!std::isfinite(result.mass))
                        throw std::runtime_error("Havok body has an unbounded mass");
                    const glm::vec3 inverseInertia(reader.read<float>(inertia + 0x20),
                        reader.read<float>(inertia + 0x24), reader.read<float>(inertia + 0x28));
                    if (!finite(inverseInertia) || glm::any(glm::lessThan(inverseInertia, glm::vec3(0))))
                        throw std::runtime_error("Havok body has invalid inverse inertia");
                    if (glm::all(glm::greaterThan(inverseInertia, glm::vec3(0))))
                        result.inertiaDiagonal = glm::vec3(1) / inverseInertia;
                    if (!finite(result.inertiaDiagonal))
                        throw std::runtime_error("Havok body has unbounded inertia");
                    const glm::vec3 center(reader.read<float>(inertia + 0x30),
                        reader.read<float>(inertia + 0x34), reader.read<float>(inertia + 0x38));
                    if (!finite(center))
                        throw std::runtime_error("Havok body has invalid center of mass");
                    result.centerOfMass = glm::inverse(result.rotation) * (center - result.position);
                    const glm::quat orientation(reader.read<float>(inertia + 0x4c),
                        reader.read<float>(inertia + 0x40), reader.read<float>(inertia + 0x44),
                        reader.read<float>(inertia + 0x48));
                    if (std::isfinite(glm::dot(orientation, orientation))
                        && glm::dot(orientation, orientation) > 0.0f)
                        result.inertiaOrientation = glm::inverse(result.rotation) * glm::normalize(orientation);
                    const auto propertyIndex = reader.read<std::uint16_t>(inertia);
                    for (std::uint32_t offset = 0x50; offset < 0x70; offset += 4)
                        if (reader.read<std::uint32_t>(inertia + offset) != 0)
                            system.nativeRewriteBlockReason = "Havok initial motion requires raw preservation";
                    if (propertiesCount != 0 && propertyIndex != 0xffff)
                    {
                        if (propertyIndex >= propertiesCount)
                            throw std::runtime_error("Havok body has invalid motion properties index");
                        const auto properties = absolute(data, propertiesOffset + propertyIndex * 0x40);
                        result.gravityFactor = reader.read<float>(properties + 8);
                        result.maxLinearVelocity = reader.read<float>(properties + 0x10);
                        result.maxAngularVelocity = reader.read<float>(properties + 0x14);
                        result.linearDamping = reader.read<float>(properties + 0x18);
                        result.angularDamping = reader.read<float>(properties + 0x1c);
                        for (const float value : {result.gravityFactor, result.maxLinearVelocity,
                                result.maxAngularVelocity, result.linearDamping, result.angularDamping})
                            if (!std::isfinite(value))
                                throw std::runtime_error("Havok body has invalid motion properties");
                        const auto gravityFlag = result.gravityFactor != 1 ? 0x20000000u : 0u;
                        const auto profile = [&](bool legacy) {
                            return reader.read<std::uint32_t>(properties + 4) == (gravityFlag | (legacy ? 2u : 0u))
                                && reader.read<std::uint32_t>(properties + 36) == (legacy ? 0x3efb22d1u : 0x3efb22d2u)
                                && reader.read<std::uint32_t>(properties + 40) == (legacy ? 0x3b23d70au : 0x3b23d70bu)
                                && reader.read<std::uint32_t>(properties + 44) == (legacy ? 0x3b23d70au : 0x3b23d70bu)
                                && reader.read<std::uint32_t>(properties + 52) == (legacy ? 0u : 0x999b67ffu)
                                && reader.read<std::uint32_t>(properties + 56) == (legacy ? 0u : 0x06757304u)
                                && reader.read<std::uint32_t>(properties + 60) == (legacy ? 0u : 0x00000073u);
                        };
                        if (profile(true)) result.motionProfile = HavokMotionProfile::legacyOpenF4;
                        else if (!profile(false))
                            system.nativeRewriteBlockReason = "Unsupported Havok motion/deactivation profile requires raw preservation";
                    }
                }
                if (shapeClass->second == "hknpCompressedMeshShape")
                    result.triangleMesh = readCompressedMesh(reader, data, fixups,
                        packfile.dataSectionIndex, shapeFixup->second.offset);
                else if (shapeClass->second == "hknpDynamicCompoundShape")
                {
                    readCompoundShapes(reader, data, fixups, packfile.dataSectionIndex,
                        shapeFixup->second.offset, result);
                    const auto [instances, instanceCount] = arrayInfo(reader, data, fixups,
                        shapeFixup->second.offset + 0x60);
                    uniformMeshMaterial = true;
                    for (std::uint32_t part = 0; part < instanceCount; ++part)
                    {
                        const auto child = dataPointer(packfile, instances + part * 0x80 + 0x50, "Havok compound child");
                        if (reader.read<std::uint64_t>(absolute(data, child + 0x18)) != result.materialId)
                            system.nativeRewriteBlockReason = "Per-part Havok materials require raw preservation";
                        const auto properties = shapeMaterialProperties(packfile, child, result);
                        if (properties == MaterialProperties::Unsupported)
                            system.nativeRewriteBlockReason = "Havok child shape properties require raw preservation";
                        if (fixups.classes.at(child) == "hknpCompressedMeshShape"
                            && properties != MaterialProperties::Uniform)
                            uniformMeshMaterial = false;
                    }
                }
                else if (shapeClass->second == "hknpConvexPolytopeShape"
                    || shapeClass->second == "hknpConvexShape")
                    result.parts.push_back({glm::mat4(1.0f), glm::vec3(1.0f),
                        shapeClass->second == "hknpConvexPolytopeShape"
                            ? readConvexPolytope(reader, data, shapeFixup->second.offset)
                            : readConvexShape(reader, data, shapeFixup->second.offset)});
                else if (shapeClass->second == "hknpCapsuleShape")
                    result.capsules.push_back({glm::mat4(1.0f), glm::vec3(1.0f),
                        readCapsule(reader, data, shapeFixup->second.offset)});
                else
                    throw std::runtime_error("Unsupported Havok shape class: "
                        + shapeClass->second);
                if (result.parts.empty() && result.capsules.empty() && !result.triangleMesh)
                    throw std::runtime_error("Havok body has an invalid shape representation");
                if (result.triangleMesh && uniformMeshMaterial)
                    for (auto& run : result.triangleMesh->primitiveDataRuns)
                        if (run.value == 0) run.value = 0xffff;
                        else if (run.value != 0xffff)
                            system.nativeRewriteBlockReason = "Havok per-primitive materials require raw preservation";
                system.bodies.push_back(std::move(result));
            }
            return system;
        }

        [[nodiscard]] glm::mat4 readTransform(
            const Reader& reader, const Section& data, std::uint32_t offset)
        {
            glm::mat4 result(1.0f);
            const std::size_t transform = absolute(data, offset);
            for (glm::length_t column = 0; column < 3; ++column)
            {
                const std::size_t source = transform + column * 0x10;
                result[column] = glm::vec4(reader.read<float>(source),
                    reader.read<float>(source + 4), reader.read<float>(source + 8), 0.0f);
            }
            result[3] = glm::vec4(reader.read<float>(transform + 0x30),
                reader.read<float>(transform + 0x34),
                reader.read<float>(transform + 0x38), 1.0f);
            if (!finite(glm::vec3(result[0])) || !finite(glm::vec3(result[1]))
                || !finite(glm::vec3(result[2])) || !finite(glm::vec3(result[3]))
                || std::abs(glm::determinant(glm::mat3(result))) < 0.00001f)
                throw std::runtime_error("Havok constraint has an invalid local frame");
            return result;
        }

        [[nodiscard]] std::vector<HavokSkeletonBone> readSkeleton(
            const Packfile& packfile, std::uint32_t skeletonOffset)
        {
            const Reader& reader = packfile.reader;
            const Section& data = packfile.sections[packfile.dataSectionIndex];
            const Fixups& fixups = packfile.fixups;
            const auto [parentOffset, parentCount] = arrayInfo(
                reader, data, fixups, skeletonOffset + 0x18);
            const auto [boneOffset, boneCount] = arrayInfo(
                reader, data, fixups, skeletonOffset + 0x28);
            const auto [poseOffset, poseCount] = arrayInfo(
                reader, data, fixups, skeletonOffset + 0x38);
            if (boneCount == 0 || boneCount > 256 || parentCount != boneCount
                || poseCount != boneCount
                || parentCount > (data.localFixups - parentOffset) / sizeof(std::int16_t)
                || boneCount > (data.localFixups - boneOffset) / 0x10
                || poseCount > (data.localFixups - poseOffset) / 0x30)
                throw std::runtime_error("Havok skeleton arrays are malformed");

            std::vector<HavokSkeletonBone> result;
            result.reserve(boneCount);
            std::uint32_t rootCount = 0;
            for (std::uint32_t index = 0; index < boneCount; ++index)
            {
                HavokSkeletonBone bone;
                bone.parent = reader.read<std::int16_t>(
                    absolute(data, parentOffset + index * sizeof(std::int16_t)));
                if (bone.parent < 0)
                    ++rootCount;
                else if (bone.parent >= static_cast<std::int16_t>(index))
                    throw std::runtime_error("Havok skeleton hierarchy is cyclic or unordered");

                const std::uint32_t sourceBone = boneOffset + index * 0x10;
                if (fixups.local.contains(sourceBone)
                    || fixups.global.contains(sourceBone))
                    bone.name = pointedString(packfile, sourceBone, "Havok bone name");
                else
                    bone.name = "ragdoll_bone_" + std::to_string(index);
                bone.lockTranslation = reader.read<std::uint8_t>(
                    absolute(data, sourceBone + 8)) != 0;

                const std::size_t pose = absolute(data, poseOffset + index * 0x30);
                const glm::vec3 translation(reader.read<float>(pose),
                    reader.read<float>(pose + 4), reader.read<float>(pose + 8));
                const float x = reader.read<float>(pose + 0x10);
                const float y = reader.read<float>(pose + 0x14);
                const float z = reader.read<float>(pose + 0x18);
                const float w = reader.read<float>(pose + 0x1c);
                const glm::vec3 scale(reader.read<float>(pose + 0x20),
                    reader.read<float>(pose + 0x24), reader.read<float>(pose + 0x28));
                glm::quat rotation(w, x, y, z);
                if (!finite(translation) || !finite(scale)
                    || glm::any(glm::lessThanEqual(scale, glm::vec3(0.0f)))
                    || !std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)
                    || !std::isfinite(w)
                    || glm::dot(rotation, rotation) <= std::numeric_limits<float>::epsilon())
                    throw std::runtime_error("Havok skeleton has an invalid reference pose");
                rotation = glm::normalize(rotation);
                bone.referencePose = glm::scale(glm::translate(
                    glm::mat4(1.0f), translation) * glm::mat4_cast(rotation), scale);
                result.push_back(std::move(bone));
            }
            if (rootCount != 1)
                throw std::runtime_error("Havok ragdoll must have exactly one root bone");
            return result;
        }

        [[nodiscard]] HavokConstraint readConstraint(
            const Packfile& packfile, std::uint32_t cinfoOffset)
        {
            const Reader& reader = packfile.reader;
            const Section& data = packfile.sections[packfile.dataSectionIndex];
            const std::uint32_t constraintOffset = dataPointer(
                packfile, cinfoOffset, "Havok constraint");
            const auto constraintClass = packfile.fixups.classes.find(constraintOffset);
            if (constraintClass == packfile.fixups.classes.end())
                throw std::runtime_error("Havok constraint has no reflected class");

            HavokConstraint result;
            const std::size_t cinfo = absolute(data, cinfoOffset);
            result.bodyA = reader.read<std::uint32_t>(cinfo + 8);
            result.bodyB = reader.read<std::uint32_t>(cinfo + 12);
            result.flags = reader.read<std::uint32_t>(cinfo + 16);
            result.localFrameA = readTransform(reader, data, constraintOffset + 0x30);
            result.localFrameB = readTransform(reader, data, constraintOffset + 0x70);
            const std::size_t constraint = absolute(data, constraintOffset);
            if (constraintClass->second == "hkpRagdollConstraintData")
            {
                result.type = HavokConstraintType::Ragdoll;
                result.maximumFrictionTorque = reader.read<float>(constraint + 0x128);
                result.minimumTwistAngle = reader.read<float>(constraint + 0x138);
                result.maximumTwistAngle = reader.read<float>(constraint + 0x13c);
                result.minimumConeAngle = reader.read<float>(constraint + 0x158);
                result.maximumConeAngle = reader.read<float>(constraint + 0x15c);
                result.minimumPlaneAngle = reader.read<float>(constraint + 0x178);
                result.maximumPlaneAngle = reader.read<float>(constraint + 0x17c);
            }
            else if (constraintClass->second == "hkpLimitedHingeConstraintData")
            {
                result.type = HavokConstraintType::LimitedHinge;
                result.maximumFrictionTorque = reader.read<float>(constraint + 0xf0);
                result.minimumTwistAngle = reader.read<float>(constraint + 0xfc);
                result.maximumTwistAngle = reader.read<float>(constraint + 0x100);
            }
            else
                throw std::runtime_error("Unsupported Havok ragdoll constraint class: "
                    + constraintClass->second);

            const std::array values{result.minimumTwistAngle,
                result.maximumTwistAngle, result.minimumConeAngle,
                result.maximumConeAngle, result.minimumPlaneAngle,
                result.maximumPlaneAngle, result.maximumFrictionTorque};
            if (!std::ranges::all_of(values,
                    [](float value) { return std::isfinite(value); })
                || result.minimumTwistAngle > result.maximumTwistAngle
                || result.minimumConeAngle > result.maximumConeAngle
                || result.minimumPlaneAngle > result.maximumPlaneAngle
                || result.maximumFrictionTorque < 0.0f)
                throw std::runtime_error("Havok ragdoll constraint limits are invalid");
            return result;
        }
    }

    HavokPhysicsSystem loadHavokPhysicsSystem(std::span<const std::byte> bytes)
    {
        const Packfile packfile = readPackfile(bytes);
        if (packfile.rootClass != "hknpPhysicsSystemData")
            throw std::runtime_error("Havok root is not hknpPhysicsSystemData");
        return readPhysicsSystemData(packfile);
    }

    HavokPhysicsSystem loadEditableHavokPhysicsSystem(std::span<const std::byte> bytes)
    {
        const Packfile packfile = readPackfile(bytes);
        if (packfile.rootClass != "hknpPhysicsSystemData")
            throw std::runtime_error("Havok root is not hknpPhysicsSystemData");
        auto system = readPhysicsSystemData(packfile);
        if (!system.nativeRewriteBlockReason.empty())
            throw std::runtime_error("Collision requires untouched raw preservation: " + system.nativeRewriteBlockReason);
        const auto encoded = serializeHavokPhysicsSystem(system);
        const auto fail = [] {
            throw std::runtime_error("Collision is not losslessly editable; preserve the original payload and use Save Native As");
        };
        if (encoded.size() != bytes.size()) fail();
        auto comparable = std::vector<std::byte>(bytes.begin(), bytes.end());
        const Reader rewritten(encoded);
        const auto& data = packfile.sections[packfile.dataSectionIndex];
        const auto [bodies, bodyCount] = arrayInfo(packfile.reader, data, packfile.fixups, packfile.rootOffset + 0x40);
        const auto [motions, motionCount] = arrayInfo(packfile.reader, data, packfile.fixups, packfile.rootOffset + 0x30);
        const auto roundoff = [&](std::uint32_t relative, float absoluteTolerance, float relativeTolerance) {
            const auto offset = absolute(data, relative);
            const float before = packfile.reader.read<float>(offset), after = rewritten.read<float>(offset);
            const float tolerance = std::min(0.0001f, absoluteTolerance
                + relativeTolerance * std::max(std::abs(before), std::abs(after)));
            if (!std::isfinite(before) || !std::isfinite(after) || std::abs(before - after) > tolerance) fail();
            std::memcpy(comparable.data() + offset, encoded.data() + offset, sizeof(float));
        };
        // Only represented floats may differ: converting mass frames and normalizing quaternions rounds.
        // Every other byte (including quality, flags, solver data and shape trees) remains an exact check.
        constexpr float relativeTolerance = 8 * std::numeric_limits<float>::epsilon();
        for (std::uint32_t index = 0; index < bodyCount; ++index)
            for (std::uint32_t component = 0; component < 4; ++component)
                roundoff(bodies + index * 0x60 + 0x40 + component * 4, 0.000002f, 0);
        for (std::uint32_t index = 0; index < motionCount; ++index)
        {
            const auto motion = motions + index * 0x70;
            roundoff(motion + 4, 0, relativeTolerance);
            for (std::uint32_t component = 0; component < 3; ++component)
            {
                roundoff(motion + 0x20 + component * 4, 0, relativeTolerance);
                roundoff(motion + 0x30 + component * 4, 0.000002f, relativeTolerance);
            }
            for (std::uint32_t component = 0; component < 4; ++component)
                roundoff(motion + 0x40 + component * 4, 0.000002f, 0);
        }
        // ponytail: Alternate native layouts stay raw-only until all their metadata can be preserved.
        if (!std::ranges::equal(comparable, encoded)) fail();
        return system;
    }

    HavokRagdoll loadHavokRagdoll(std::span<const std::byte> bytes)
    {
        const Packfile packfile = readPackfile(bytes);
        if (packfile.rootClass != "hknpRagdollData")
            throw std::runtime_error("Havok root is not hknpRagdollData");
        const Reader& reader = packfile.reader;
        const Section& data = packfile.sections[packfile.dataSectionIndex];

        HavokRagdoll result;
        result.bodies = readPhysicsSystemData(packfile).bodies;
        const std::uint32_t skeletonOffset = dataPointer(
            packfile, packfile.rootOffset + 0x78, "Havok ragdoll skeleton");
        const auto skeletonClass = packfile.fixups.classes.find(skeletonOffset);
        if (skeletonClass == packfile.fixups.classes.end()
            || skeletonClass->second != "hkaSkeleton")
            throw std::runtime_error("Havok ragdoll skeleton has the wrong class");
        result.bones = readSkeleton(packfile, skeletonOffset);

        const auto [mapOffset, mapCount] = arrayInfo(
            reader, data, packfile.fixups, packfile.rootOffset + 0x80);
        if (mapCount != result.bones.size()
            || mapCount > (data.localFixups - mapOffset) / sizeof(std::int32_t))
            throw std::runtime_error("Havok ragdoll body map is malformed");
        std::vector<bool> mapped(result.bodies.size(), false);
        result.boneToBody.reserve(mapCount);
        for (std::uint32_t index = 0; index < mapCount; ++index)
        {
            const auto body = reader.read<std::int32_t>(absolute(
                data, mapOffset + index * sizeof(std::int32_t)));
            if (body < 0 || static_cast<std::size_t>(body) >= result.bodies.size()
                || mapped[static_cast<std::size_t>(body)])
                throw std::runtime_error("Havok ragdoll body map is incomplete or duplicated");
            mapped[static_cast<std::size_t>(body)] = true;
            result.boneToBody.push_back(static_cast<std::uint32_t>(body));
        }
        if (result.bodies.size() != result.bones.size()
            || !std::ranges::all_of(mapped, [](bool value) { return value; }))
            throw std::runtime_error("Havok ragdoll requires one body per bone");
        if (std::ranges::any_of(result.bodies, [](const HavokRigidBody& body) {
                return !body.dynamic || !std::isfinite(body.mass) || body.mass <= 0.0f;
            }))
            throw std::runtime_error("Havok ragdoll bodies require positive dynamic mass");

        const auto [constraintOffset, constraintCount] = arrayInfo(
            reader, data, packfile.fixups, packfile.rootOffset + 0x50);
        if (constraintCount + 1 != result.bones.size()
            || constraintCount > (data.localFixups - constraintOffset) / 0x18)
            throw std::runtime_error("Havok ragdoll constraint array is malformed");
        result.constraints.reserve(constraintCount);
        for (std::uint32_t index = 0; index < constraintCount; ++index)
        {
            auto constraint = readConstraint(packfile, constraintOffset + index * 0x18);
            if (constraint.bodyA >= result.bodies.size()
                || constraint.bodyB >= result.bodies.size()
                || constraint.bodyA == constraint.bodyB)
                throw std::runtime_error("Havok ragdoll constraint references an invalid body");
            result.constraints.push_back(std::move(constraint));
        }

        std::vector<bool> used(result.constraints.size(), false);
        for (std::uint32_t bone = 0; bone < result.bones.size(); ++bone)
        {
            const auto parent = result.bones[bone].parent;
            if (parent < 0)
                continue;
            const std::uint32_t body = result.boneToBody[bone];
            const std::uint32_t parentBody = result.boneToBody[parent];
            std::size_t match = result.constraints.size();
            for (std::size_t index = 0; index < result.constraints.size(); ++index)
            {
                const auto& constraint = result.constraints[index];
                if ((constraint.bodyA == body && constraint.bodyB == parentBody)
                    || (constraint.bodyA == parentBody && constraint.bodyB == body))
                {
                    if (match != result.constraints.size())
                        throw std::runtime_error(
                            "Havok ragdoll has duplicate parent constraints");
                    match = index;
                }
            }
            if (match == result.constraints.size())
                throw std::runtime_error("Havok ragdoll is missing a parent constraint");
            used[match] = true;
        }
        if (!std::ranges::all_of(used, [](bool value) { return value; }))
            throw std::runtime_error("Havok ragdoll contains a non-hierarchical constraint");
        return result;
    }

    std::vector<HavokPhysicsSystem> loadNifPhysicsSystems(std::span<const std::byte> bytes)
    {
        SpanBuffer buffer(bytes);
        std::istream stream(&buffer);
        nifly::NifFile file(stream);
        if (!file.IsValid())
            throw std::runtime_error("NIF parser rejected the collision model");

        std::vector<HavokPhysicsSystem> systems;
        const nifly::NiHeader& header = file.GetHeader();
        for (std::uint32_t index = 0; index < header.GetNumBlocks(); ++index)
        {
            const auto* physics = header.GetBlock<nifly::bhkPhysicsSystem>(index);
            if (!physics || physics->data.empty())
                continue;
            const auto* begin = reinterpret_cast<const std::byte*>(physics->data.data());
            systems.push_back(loadHavokPhysicsSystem({begin, physics->data.size()}));
        }
        return systems;
    }

    std::vector<HavokRagdoll> loadNifRagdolls(std::span<const std::byte> bytes)
    {
        SpanBuffer buffer(bytes);
        std::istream stream(&buffer);
        nifly::NifFile file(stream);
        if (!file.IsValid())
            throw std::runtime_error("NIF parser rejected the ragdoll model");

        std::vector<HavokRagdoll> ragdolls;
        const nifly::NiHeader& header = file.GetHeader();
        for (std::uint32_t index = 0; index < header.GetNumBlocks(); ++index)
        {
            const auto* ragdoll = header.GetBlock<nifly::bhkRagdollSystem>(index);
            if (!ragdoll || ragdoll->data.empty())
                continue;
            const auto* begin = reinterpret_cast<const std::byte*>(ragdoll->data.data());
            ragdolls.push_back(loadHavokRagdoll({begin, ragdoll->data.size()}));
        }
        return ragdolls;
    }
}
