#include "assets/ba2_archive.hpp"
#include "assets/esm_world.hpp"
#include "assets/fo4_install.hpp"
#include "assets/havok_packfile.hpp"
#include "assets/nif_model.hpp"
#include "physics/jolt_collision.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <string_view>
#include <vector>

#include <glm/ext/matrix_transform.hpp>

namespace
{
    void require(bool condition, std::string_view message)
    {
        if (!condition)
            throw std::runtime_error(std::string(message));
    }

    template <class T>
    void write(std::vector<std::byte>& bytes, std::size_t offset, T value)
    {
        if (offset + sizeof(T) > bytes.size())
            bytes.resize(offset + sizeof(T));
        std::memcpy(bytes.data() + offset, &value, sizeof(T));
    }

    void writeString(std::vector<std::byte>& bytes, std::size_t offset,
        std::string_view value, std::size_t width)
    {
        require(value.size() < width, "test string does not fit");
        if (offset + width > bytes.size())
            bytes.resize(offset + width);
        std::memcpy(bytes.data() + offset, value.data(), value.size());
        bytes[offset + value.size()] = std::byte{};
    }

    std::size_t align16(std::size_t value) { return (value + 15) & ~std::size_t{15}; }

    glm::quat rotationBetween(glm::vec3 from, glm::vec3 to)
    {
        from = glm::normalize(from);
        to = glm::normalize(to);
        const float cosine = glm::dot(from, to);
        if (cosine < -0.9999f)
        {
            glm::vec3 axis = glm::cross(from, glm::vec3(1.0f, 0.0f, 0.0f));
            if (glm::dot(axis, axis) < 0.0001f)
                axis = glm::cross(from, glm::vec3(0.0f, 0.0f, 1.0f));
            return glm::angleAxis(3.14159265358979323846f, glm::normalize(axis));
        }
        const glm::vec3 axis = glm::cross(from, to);
        return glm::normalize(glm::quat(1.0f + cosine, axis.x, axis.y, axis.z));
    }

    struct ClassTable
    {
        std::vector<std::byte> bytes;
        std::uint32_t systemName = 0;
        std::uint32_t shapeName = 0;
        std::uint32_t convexName = 0;
        std::uint32_t capsuleName = 0;
        std::uint32_t compoundName = 0;
        std::uint32_t compressedName = 0;
        std::uint32_t compressedDataName = 0;
        std::uint32_t ragdollName = 0;
        std::uint32_t skeletonName = 0;
        std::uint32_t ragdollConstraintName = 0;
        std::uint32_t hingeConstraintName = 0;
    };

    ClassTable makeClasses()
    {
        ClassTable result;
        const auto append = [&](std::string_view name, std::uint32_t& nameOffset) {
            const std::size_t entry = result.bytes.size();
            result.bytes.resize(entry + 5 + name.size() + 1);
            write<std::uint32_t>(result.bytes, entry, 0);
            result.bytes[entry + 4] = std::byte{9};
            nameOffset = static_cast<std::uint32_t>(entry + 5);
            std::memcpy(result.bytes.data() + entry + 5, name.data(), name.size());
        };
        append("hknpPhysicsSystemData", result.systemName);
        append("hknpConvexPolytopeShape", result.shapeName);
        append("hknpConvexShape", result.convexName);
        append("hknpCapsuleShape", result.capsuleName);
        append("hknpDynamicCompoundShape", result.compoundName);
        append("hknpCompressedMeshShape", result.compressedName);
        append("hknpCompressedMeshShapeData", result.compressedDataName);
        append("hknpRagdollData", result.ragdollName);
        append("hkaSkeleton", result.skeletonName);
        append("hkpRagdollConstraintData", result.ragdollConstraintName);
        append("hkpLimitedHingeConstraintData", result.hingeConstraintName);
        result.bytes.resize(align16(result.bytes.size()), std::byte{0xff});
        return result;
    }

    void writeSection(std::vector<std::byte>& bytes, std::size_t offset, std::string_view name,
        std::uint32_t start, std::uint32_t local, std::uint32_t global,
        std::uint32_t virtualFixups, std::uint32_t exports, std::uint32_t imports,
        std::uint32_t end)
    {
        writeString(bytes, offset, name, 20);
        write(bytes, offset + 20, start);
        write(bytes, offset + 24, local);
        write(bytes, offset + 28, global);
        write(bytes, offset + 32, virtualFixups);
        write(bytes, offset + 36, exports);
        write(bytes, offset + 40, imports);
        write(bytes, offset + 44, end);
    }

    std::vector<std::byte> makeCubePackfile()
    {
        constexpr std::uint32_t classStart = 0x100;
        constexpr std::uint32_t root = 0;
        constexpr std::uint32_t body = 0x80;
        constexpr std::uint32_t shape = 0xe0;
        constexpr std::uint32_t vertices = shape + 0x50;
        constexpr std::uint32_t faces = vertices + 8 * 16;
        constexpr std::uint32_t faceVertices = faces + 6 * 4;
        constexpr std::uint32_t localFixups = 0x1e0;
        constexpr std::uint32_t globalFixups = localFixups + 0x10;
        constexpr std::uint32_t virtualFixups = globalFixups + 0x10;
        constexpr std::uint32_t end = virtualFixups + 0x20;

        ClassTable classes = makeClasses();
        const auto classSize = static_cast<std::uint32_t>(classes.bytes.size());
        const std::uint32_t dataStart = classStart + classSize;
        std::vector<std::byte> bytes(dataStart + end, std::byte{});
        const std::array<std::byte, 8> magic{
            std::byte{0x57}, std::byte{0xe0}, std::byte{0xe0}, std::byte{0x57},
            std::byte{0x10}, std::byte{0xc0}, std::byte{0xc0}, std::byte{0x10}};
        std::copy(magic.begin(), magic.end(), bytes.begin());
        write<std::uint32_t>(bytes, 12, 11);
        bytes[16] = std::byte{8};
        bytes[17] = std::byte{1};
        bytes[19] = std::byte{1};
        write<std::uint32_t>(bytes, 20, 3);
        write<std::uint32_t>(bytes, 24, 2);
        write<std::uint32_t>(bytes, 28, root);
        write<std::uint32_t>(bytes, 32, 0);
        write<std::uint32_t>(bytes, 36, classes.systemName);
        writeString(bytes, 40, "hk_2014.1.0-r1", 16);
        writeSection(bytes, 0x40, "__classnames__", classStart, classSize, classSize,
            classSize, classSize, classSize, classSize);
        writeSection(bytes, 0x80, "__types__", dataStart, 0, 0, 0, 0, 0, 0);
        writeSection(bytes, 0xc0, "__data__", dataStart, localFixups, globalFixups,
            virtualFixups, end, end, end);
        std::copy(classes.bytes.begin(), classes.bytes.end(), bytes.begin() + classStart);

        const auto data = static_cast<std::size_t>(dataStart);
        write<std::uint32_t>(bytes, data + root + 0x48, 1);
        write<std::uint32_t>(bytes, data + root + 0x4c, 0x80000001u);
        write<float>(bytes, data + body + 0x30, 1.0f);
        write<float>(bytes, data + body + 0x34, 2.0f);
        write<float>(bytes, data + body + 0x38, 3.0f);
        write<float>(bytes, data + body + 0x4c, 1.0f);
        write<float>(bytes, data + shape + 0x14, 0.01f);
        write<std::uint16_t>(bytes, data + shape + 0x30, 8);
        write<std::uint16_t>(bytes, data + shape + 0x32,
            static_cast<std::uint16_t>(vertices - (shape + 0x30)));
        write<std::uint16_t>(bytes, data + shape + 0x44, 6);
        write<std::uint16_t>(bytes, data + shape + 0x46,
            static_cast<std::uint16_t>(faces - (shape + 0x44)));
        write<std::uint16_t>(bytes, data + shape + 0x48, 24);
        write<std::uint16_t>(bytes, data + shape + 0x4a,
            static_cast<std::uint16_t>(faceVertices - (shape + 0x48)));

        constexpr std::array<std::array<float, 3>, 8> points{{
            {{-1, -1, -1}}, {{1, -1, -1}}, {{1, 1, -1}}, {{-1, 1, -1}},
            {{-1, -1, 1}}, {{1, -1, 1}}, {{1, 1, 1}}, {{-1, 1, 1}}}};
        for (std::size_t index = 0; index < points.size(); ++index)
            for (std::size_t axis = 0; axis < 3; ++axis)
                write<float>(bytes, data + vertices + index * 16 + axis * 4, points[index][axis]);
        constexpr std::array<std::array<std::uint8_t, 4>, 6> quads{{
            {{0, 3, 2, 1}}, {{4, 5, 6, 7}}, {{0, 1, 5, 4}},
            {{1, 2, 6, 5}}, {{2, 3, 7, 6}}, {{3, 0, 4, 7}}}};
        for (std::size_t index = 0; index < quads.size(); ++index)
        {
            write<std::uint16_t>(bytes, data + faces + index * 4,
                static_cast<std::uint16_t>(index * 4));
            write<std::uint8_t>(bytes, data + faces + index * 4 + 2, 4);
            std::memcpy(bytes.data() + data + faceVertices + index * 4, quads[index].data(), 4);
        }

        write<std::uint32_t>(bytes, data + localFixups, root + 0x40);
        write<std::uint32_t>(bytes, data + localFixups + 4, body);
        std::fill(bytes.begin() + data + localFixups + 8,
            bytes.begin() + data + globalFixups, std::byte{0xff});
        write<std::uint32_t>(bytes, data + globalFixups, body);
        write<std::uint32_t>(bytes, data + globalFixups + 4, 2);
        write<std::uint32_t>(bytes, data + globalFixups + 8, shape);
        std::fill(bytes.begin() + data + globalFixups + 12,
            bytes.begin() + data + virtualFixups, std::byte{0xff});
        write<std::uint32_t>(bytes, data + virtualFixups, root);
        write<std::uint32_t>(bytes, data + virtualFixups + 4, 0);
        write<std::uint32_t>(bytes, data + virtualFixups + 8, classes.systemName);
        write<std::uint32_t>(bytes, data + virtualFixups + 12, shape);
        write<std::uint32_t>(bytes, data + virtualFixups + 16, 0);
        write<std::uint32_t>(bytes, data + virtualFixups + 20, classes.shapeName);
        std::fill(bytes.begin() + data + virtualFixups + 24, bytes.end(), std::byte{0xff});
        return bytes;
    }

    std::vector<std::byte> makeConvexPackfile()
    {
        auto bytes = makeCubePackfile();
        const ClassTable classes = makeClasses();
        const std::size_t dataStart = 0x100 + classes.bytes.size();
        write<std::uint32_t>(bytes, dataStart + 0x200 + 20, classes.convexName);
        return bytes;
    }

    std::vector<std::byte> makeCapsulePackfile()
    {
        constexpr std::uint32_t shape = 0xe0;
        auto bytes = makeCubePackfile();
        const ClassTable classes = makeClasses();
        const std::size_t dataStart = 0x100 + classes.bytes.size();
        write<std::uint32_t>(bytes, dataStart + 0x200 + 20, classes.capsuleName);
        write<float>(bytes, dataStart + shape + 0x14, 0.25f);
        write<float>(bytes, dataStart + shape + 0x50, 0.0f);
        write<float>(bytes, dataStart + shape + 0x54, -1.0f);
        write<float>(bytes, dataStart + shape + 0x58, 0.0f);
        write<float>(bytes, dataStart + shape + 0x60, 0.0f);
        write<float>(bytes, dataStart + shape + 0x64, 1.0f);
        write<float>(bytes, dataStart + shape + 0x68, 0.0f);
        return bytes;
    }

    std::vector<std::byte> makeCompoundPackfile()
    {
        constexpr std::uint32_t root = 0;
        constexpr std::uint32_t body = 0x80;
        constexpr std::uint32_t convex = 0xe0;
        constexpr std::uint32_t compound = 0x1e0;
        constexpr std::uint32_t instance = 0x270;
        constexpr std::uint32_t localFixups = 0x300;
        constexpr std::uint32_t globalFixups = 0x318;
        constexpr std::uint32_t virtualFixups = 0x348;
        constexpr std::uint32_t end = 0x378;

        auto bytes = makeCubePackfile();
        const ClassTable classes = makeClasses();
        const std::uint32_t dataStart = 0x100
            + static_cast<std::uint32_t>(classes.bytes.size());
        bytes.resize(dataStart + end);
        std::fill(bytes.begin() + dataStart + compound, bytes.end(), std::byte{});
        writeSection(bytes, 0xc0, "__data__", dataStart, localFixups, globalFixups,
            virtualFixups, end, end, end);

        const auto data = static_cast<std::size_t>(dataStart);
        write<std::uint32_t>(bytes, data + compound + 0x68, 1);
        write<std::uint32_t>(bytes, data + compound + 0x6c, 0x80000001u);
        for (std::size_t column = 0; column < 3; ++column)
            write<float>(bytes, data + instance + column * 0x10 + column * 4, 1.0f);
        write<float>(bytes, data + instance + 0x30, 1.0f);
        write<float>(bytes, data + instance + 0x34, 2.0f);
        write<float>(bytes, data + instance + 0x38, 3.0f);
        write<float>(bytes, data + instance + 0x40, 2.0f);
        write<float>(bytes, data + instance + 0x44, 0.5f);
        write<float>(bytes, data + instance + 0x48, 1.5f);

        write<std::uint32_t>(bytes, data + localFixups, root + 0x40);
        write<std::uint32_t>(bytes, data + localFixups + 4, body);
        write<std::uint32_t>(bytes, data + localFixups + 8, compound + 0x60);
        write<std::uint32_t>(bytes, data + localFixups + 12, instance);
        std::fill(bytes.begin() + data + localFixups + 16,
            bytes.begin() + data + globalFixups, std::byte{0xff});

        write<std::uint32_t>(bytes, data + globalFixups, body);
        write<std::uint32_t>(bytes, data + globalFixups + 4, 2);
        write<std::uint32_t>(bytes, data + globalFixups + 8, compound);
        write<std::uint32_t>(bytes, data + globalFixups + 12, instance + 0x50);
        write<std::uint32_t>(bytes, data + globalFixups + 16, 2);
        write<std::uint32_t>(bytes, data + globalFixups + 20, convex);
        std::fill(bytes.begin() + data + globalFixups + 24,
            bytes.begin() + data + virtualFixups, std::byte{0xff});

        write<std::uint32_t>(bytes, data + virtualFixups, root);
        write<std::uint32_t>(bytes, data + virtualFixups + 4, 0);
        write<std::uint32_t>(bytes, data + virtualFixups + 8, classes.systemName);
        write<std::uint32_t>(bytes, data + virtualFixups + 12, convex);
        write<std::uint32_t>(bytes, data + virtualFixups + 16, 0);
        write<std::uint32_t>(bytes, data + virtualFixups + 20, classes.shapeName);
        write<std::uint32_t>(bytes, data + virtualFixups + 24, compound);
        write<std::uint32_t>(bytes, data + virtualFixups + 28, 0);
        write<std::uint32_t>(bytes, data + virtualFixups + 32, classes.compoundName);
        std::fill(bytes.begin() + data + virtualFixups + 36, bytes.end(), std::byte{0xff});
        return bytes;
    }

    std::vector<std::byte> makeCompressedMeshPackfile()
    {
        constexpr std::uint32_t root = 0;
        constexpr std::uint32_t body = 0x80;
        constexpr std::uint32_t shape = 0xe0;
        constexpr std::uint32_t shapeData = 0x190;
        constexpr std::uint32_t tree = shapeData + 0x10;
        constexpr std::uint32_t sections = 0x270;
        constexpr std::uint32_t primitives = 0x2d0;
        constexpr std::uint32_t sharedIndices = 0x2d8;
        constexpr std::uint32_t packedVertices = 0x2dc;
        constexpr std::uint32_t sharedVertices = 0x2e8;
        constexpr std::uint32_t runs = 0x2f0;
        constexpr std::uint32_t localFixups = 0x320;
        constexpr std::uint32_t globalFixups = 0x358;
        constexpr std::uint32_t virtualFixups = 0x370;
        constexpr std::uint32_t end = 0x3a0;

        auto bytes = makeCubePackfile();
        const ClassTable classes = makeClasses();
        const std::uint32_t dataStart = 0x100
            + static_cast<std::uint32_t>(classes.bytes.size());
        bytes.resize(dataStart + end);
        std::fill(bytes.begin() + dataStart, bytes.end(), std::byte{});
        writeSection(bytes, 0xc0, "__data__", dataStart, localFixups, globalFixups,
            virtualFixups, end, end, end);
        const auto data = static_cast<std::size_t>(dataStart);

        write<std::uint32_t>(bytes, data + root + 0x48, 1);
        write<std::uint32_t>(bytes, data + root + 0x4c, 0x80000001u);
        write<float>(bytes, data + body + 0x4c, 1.0f);

        write<float>(bytes, data + tree + 0x10, 0.0f);
        write<float>(bytes, data + tree + 0x14, 0.0f);
        write<float>(bytes, data + tree + 0x18, 0.0f);
        write<float>(bytes, data + tree + 0x20, 1.0f);
        write<float>(bytes, data + tree + 0x24, 1.0f);
        write<float>(bytes, data + tree + 0x28, 1.0f);
        write<std::uint32_t>(bytes, data + tree + 0x48, 1);
        write<std::uint32_t>(bytes, data + tree + 0x58, 2);
        write<std::uint32_t>(bytes, data + tree + 0x68, 1);
        write<std::uint32_t>(bytes, data + tree + 0x78, 3);
        write<std::uint32_t>(bytes, data + tree + 0x88, 1);
        write<std::uint32_t>(bytes, data + tree + 0x98, 2);

        write<float>(bytes, data + sections + 0x30, 0.0f);
        write<float>(bytes, data + sections + 0x34, 0.0f);
        write<float>(bytes, data + sections + 0x38, 0.0f);
        write<float>(bytes, data + sections + 0x3c, 1.0f);
        write<float>(bytes, data + sections + 0x40, 1.0f);
        write<float>(bytes, data + sections + 0x44, 1.0f);
        write<std::uint32_t>(bytes, data + sections + 0x48, 0);
        write<std::uint32_t>(bytes, data + sections + 0x4c, 1);
        write<std::uint32_t>(bytes, data + sections + 0x50, 2);
        write<std::uint32_t>(bytes, data + sections + 0x54, 2);
        write<std::uint8_t>(bytes, data + sections + 0x58, 3);
        write<std::uint8_t>(bytes, data + sections + 0x59, 1);

        const std::array<std::uint8_t, 8> primitiveBytes{
            0, 1, 2, 2,
            0, 1, 3, 2};
        std::memcpy(bytes.data() + data + primitives, primitiveBytes.data(),
            primitiveBytes.size());
        write<std::uint16_t>(bytes, data + sharedIndices, 0);
        write<std::uint32_t>(bytes, data + packedVertices, 0);
        write<std::uint32_t>(bytes, data + packedVertices + 4, 1);
        write<std::uint32_t>(bytes, data + packedVertices + 8, 1u << 11);
        write<std::uint64_t>(bytes, data + sharedVertices, 0x3fffffull << 42);
        write<std::uint16_t>(bytes, data + runs, 7);
        write<std::uint8_t>(bytes, data + runs + 2, 0);
        write<std::uint8_t>(bytes, data + runs + 3, 1);
        write<std::uint16_t>(bytes, data + runs + 4, 9);
        write<std::uint8_t>(bytes, data + runs + 6, 1);
        write<std::uint8_t>(bytes, data + runs + 7, 1);

        constexpr std::array<std::pair<std::uint32_t, std::uint32_t>, 7> locals{{
            {root + 0x40, body},
            {tree + 0x40, sections},
            {tree + 0x50, primitives},
            {tree + 0x60, sharedIndices},
            {tree + 0x70, packedVertices},
            {tree + 0x80, sharedVertices},
            {tree + 0x90, runs}}};
        for (std::size_t index = 0; index < locals.size(); ++index)
        {
            write<std::uint32_t>(bytes, data + localFixups + index * 8, locals[index].first);
            write<std::uint32_t>(bytes, data + localFixups + index * 8 + 4,
                locals[index].second);
        }

        write<std::uint32_t>(bytes, data + globalFixups, body);
        write<std::uint32_t>(bytes, data + globalFixups + 4, 2);
        write<std::uint32_t>(bytes, data + globalFixups + 8, shape);
        write<std::uint32_t>(bytes, data + globalFixups + 12, shape + 0x60);
        write<std::uint32_t>(bytes, data + globalFixups + 16, 2);
        write<std::uint32_t>(bytes, data + globalFixups + 20, shapeData);

        constexpr std::array<std::uint32_t, 3> objects{root, shape, shapeData};
        const std::array<std::uint32_t, 3> names{
            classes.systemName, classes.compressedName, classes.compressedDataName};
        for (std::size_t index = 0; index < objects.size(); ++index)
        {
            write<std::uint32_t>(bytes, data + virtualFixups + index * 12, objects[index]);
            write<std::uint32_t>(bytes, data + virtualFixups + index * 12 + 4, 0);
            write<std::uint32_t>(bytes, data + virtualFixups + index * 12 + 8, names[index]);
        }
        std::fill(bytes.begin() + data + virtualFixups + objects.size() * 12,
            bytes.end(), std::byte{0xff});
        return bytes;
    }

    std::vector<std::byte> makeRagdollPackfile()
    {
        constexpr std::uint32_t root = 0x000;
        constexpr std::uint32_t motions = 0x100;
        constexpr std::uint32_t bodies = 0x250;
        constexpr std::uint32_t capsule = 0x370;
        constexpr std::uint32_t constraints = 0x3e0;
        constexpr std::uint32_t skeleton = 0x410;
        constexpr std::uint32_t parents = 0x460;
        constexpr std::uint32_t bones = 0x470;
        constexpr std::uint32_t poses = 0x4a0;
        constexpr std::uint32_t bodyMap = 0x530;
        constexpr std::uint32_t ragdollConstraint = 0x540;
        constexpr std::uint32_t hingeConstraint = 0x6c0;
        constexpr std::uint32_t localFixups = 0x800;
        constexpr std::uint32_t globalFixups = 0x838;
        constexpr std::uint32_t virtualFixups = 0x880;
        constexpr std::uint32_t end = 0x8e0;

        auto bytes = makeCubePackfile();
        const ClassTable classes = makeClasses();
        const std::uint32_t dataStart = 0x100
            + static_cast<std::uint32_t>(classes.bytes.size());
        bytes.resize(dataStart + end);
        std::fill(bytes.begin() + dataStart, bytes.end(), std::byte{});
        write<std::uint32_t>(bytes, 36, classes.ragdollName);
        writeSection(bytes, 0xc0, "__data__", dataStart, localFixups, globalFixups,
            virtualFixups, end, end, end);
        const auto data = static_cast<std::size_t>(dataStart);

        const auto setArrayCount = [&](std::uint32_t slot, std::uint32_t count) {
            write<std::uint32_t>(bytes, data + slot + 8, count);
            write<std::uint32_t>(bytes, data + slot + 12, 0x80000000u | count);
        };
        setArrayCount(root + 0x30, 3);
        setArrayCount(root + 0x40, 3);
        setArrayCount(root + 0x50, 2);
        setArrayCount(root + 0x80, 3);
        setArrayCount(skeleton + 0x18, 3);
        setArrayCount(skeleton + 0x28, 3);
        setArrayCount(skeleton + 0x38, 3);

        for (std::uint32_t index = 0; index < 3; ++index)
        {
            write<float>(bytes, data + motions + index * 0x70 + 4,
                1.0f / static_cast<float>(index + 1));
            const std::uint32_t body = bodies + index * 0x60;
            write<std::uint32_t>(bytes, data + body + 0x0c, index);
            write<float>(bytes, data + body + 0x38, static_cast<float>(index));
            write<float>(bytes, data + body + 0x4c, 1.0f);
            write<std::int32_t>(bytes, data + bodyMap + index * 4,
                static_cast<std::int32_t>(index));

            write<std::int16_t>(bytes, data + parents + index * 2,
                index == 0 ? std::int16_t{-1}
                           : static_cast<std::int16_t>(index - 1));
            write<std::uint8_t>(bytes, data + bones + index * 0x10 + 8,
                index == 0 ? 1 : 0);
            const std::uint32_t pose = poses + index * 0x30;
            write<float>(bytes, data + pose + 8, static_cast<float>(index));
            write<float>(bytes, data + pose + 0x1c, 1.0f);
            write<float>(bytes, data + pose + 0x20, 1.0f);
            write<float>(bytes, data + pose + 0x24, 1.0f);
            write<float>(bytes, data + pose + 0x28, 1.0f);
        }

        write<float>(bytes, data + capsule + 0x14, 0.2f);
        write<float>(bytes, data + capsule + 0x58, -0.4f);
        write<float>(bytes, data + capsule + 0x68, 0.4f);

        const auto writeFrame = [&](std::uint32_t offset, float z) {
            write<float>(bytes, data + offset, 1.0f);
            write<float>(bytes, data + offset + 0x14, 1.0f);
            write<float>(bytes, data + offset + 0x28, 1.0f);
            write<float>(bytes, data + offset + 0x38, z);
        };
        writeFrame(ragdollConstraint + 0x30, 0.5f);
        writeFrame(ragdollConstraint + 0x70, -0.5f);
        write<float>(bytes, data + ragdollConstraint + 0x128, 2.0f);
        write<float>(bytes, data + ragdollConstraint + 0x138, -0.3f);
        write<float>(bytes, data + ragdollConstraint + 0x13c, 0.4f);
        write<float>(bytes, data + ragdollConstraint + 0x158, -0.5f);
        write<float>(bytes, data + ragdollConstraint + 0x15c, 0.6f);
        write<float>(bytes, data + ragdollConstraint + 0x178, -0.2f);
        write<float>(bytes, data + ragdollConstraint + 0x17c, 0.25f);
        writeFrame(hingeConstraint + 0x30, 0.5f);
        writeFrame(hingeConstraint + 0x70, -0.5f);
        write<float>(bytes, data + hingeConstraint + 0xf0, 1.5f);
        write<float>(bytes, data + hingeConstraint + 0xfc, -0.7f);
        write<float>(bytes, data + hingeConstraint + 0x100, 0.8f);

        write<std::uint32_t>(bytes, data + constraints + 8, 0);
        write<std::uint32_t>(bytes, data + constraints + 12, 1);
        write<std::uint32_t>(bytes, data + constraints + 16, 5);
        write<std::uint32_t>(bytes, data + constraints + 0x18 + 8, 1);
        write<std::uint32_t>(bytes, data + constraints + 0x18 + 12, 2);
        write<std::uint32_t>(bytes, data + constraints + 0x18 + 16, 6);

        constexpr std::array<std::pair<std::uint32_t, std::uint32_t>, 7> locals{{
            {root + 0x30, motions}, {root + 0x40, bodies},
            {root + 0x50, constraints}, {root + 0x80, bodyMap},
            {skeleton + 0x18, parents}, {skeleton + 0x28, bones},
            {skeleton + 0x38, poses}}};
        for (std::size_t index = 0; index < locals.size(); ++index)
        {
            write<std::uint32_t>(bytes, data + localFixups + index * 8,
                locals[index].first);
            write<std::uint32_t>(bytes, data + localFixups + index * 8 + 4,
                locals[index].second);
        }

        constexpr std::array<std::pair<std::uint32_t, std::uint32_t>, 6> globals{{
            {bodies, capsule}, {bodies + 0x60, capsule},
            {bodies + 0xc0, capsule}, {root + 0x78, skeleton},
            {constraints, ragdollConstraint},
            {constraints + 0x18, hingeConstraint}}};
        for (std::size_t index = 0; index < globals.size(); ++index)
        {
            write<std::uint32_t>(bytes, data + globalFixups + index * 12,
                globals[index].first);
            write<std::uint32_t>(bytes, data + globalFixups + index * 12 + 4, 2);
            write<std::uint32_t>(bytes, data + globalFixups + index * 12 + 8,
                globals[index].second);
        }

        constexpr std::array<std::uint32_t, 5> objects{
            root, capsule, skeleton, ragdollConstraint, hingeConstraint};
        const std::array<std::uint32_t, 5> names{classes.ragdollName,
            classes.capsuleName, classes.skeletonName,
            classes.ragdollConstraintName, classes.hingeConstraintName};
        for (std::size_t index = 0; index < objects.size(); ++index)
        {
            write<std::uint32_t>(bytes, data + virtualFixups + index * 12,
                objects[index]);
            write<std::uint32_t>(bytes, data + virtualFixups + index * 12 + 4, 0);
            write<std::uint32_t>(bytes, data + virtualFixups + index * 12 + 8,
                names[index]);
        }
        std::fill(bytes.begin() + data + virtualFixups + objects.size() * 12,
            bytes.end(), std::byte{0xff});
        return bytes;
    }

    void testSyntheticPackfile()
    {
        const auto bytes = makeCubePackfile();
        const auto system = empire::assets::loadHavokPhysicsSystem(bytes);
        require(system.bodies.size() == 1, "cube body was not decoded");
        const auto& body = system.bodies.front();
        require(!body.dynamic, "static cube was marked dynamic");
        require(body.position == glm::vec3(1.0f, 2.0f, 3.0f), "body transform was not decoded");
        require(body.parts.size() == 1, "standalone hull was not flattened to one part");
        const auto& hull = body.parts.front().polytope;
        require(hull.vertices.size() == 8, "cube vertices were not decoded");
        require(hull.indices.size() == 36, "cube faces were not triangulated");
        require(std::abs(hull.convexRadius - 0.01f) < 0.00001f,
            "convex radius was not decoded");
        const auto shape = empire::physics::makeJoltCollisionShape(body);
        require(shape != nullptr, "Jolt hull was not created");

        empire::physics::JoltWorld world;
        constexpr float units = empire::physics::BethesdaUnitsPerMeter;
        (void)world.addStaticBox({5.0f * units, 0.25f * units, 5.0f * units},
            {0.0f, -0.25f * units, 0.0f});
        const auto dynamicBody = world.addDynamicBody(shape, {0.0f, 4.0f * units, 0.0f});
        for (int step = 0; step < 180; ++step)
            world.step(1.0f / 60.0f);
        const float settledHeight = world.bodyTransform(dynamicBody)[3].y;
        require(settledHeight > 0.5f * units && settledHeight < 1.5f * units,
            "Jolt body did not settle on the floor");

        auto corrupt = bytes;
        write<std::uint32_t>(corrupt, 0xc0 + 20, 0xfffffff0u);
        bool rejected = false;
        try { (void)empire::assets::loadHavokPhysicsSystem(corrupt); }
        catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "out-of-range section was accepted");
    }

    void testCompoundAndHeightField()
    {
        auto bytes = makeCompoundPackfile();
        const auto system = empire::assets::loadHavokPhysicsSystem(bytes);
        require(system.bodies.size() == 1 && system.bodies.front().parts.size() == 1,
            "compound child was not flattened");
        const auto& part = system.bodies.front().parts.front();
        require(glm::vec3(part.transform[3]) == glm::vec3(1.0f, 2.0f, 3.0f),
            "compound child translation changed");
        require(part.scale == glm::vec3(2.0f, 0.5f, 1.5f),
            "compound child scale changed");
        require(empire::physics::makeJoltCollisionShape(system.bodies.front()) != nullptr,
            "Jolt rejected the synthetic compound");

        std::uint32_t dataStart = 0;
        std::memcpy(&dataStart, bytes.data() + 0xc0 + 20, sizeof(dataStart));
        write<std::uint32_t>(bytes, dataStart + 0x1e0 + 0x68, 0xffffu);
        bool rejected = false;
        try { (void)empire::assets::loadHavokPhysicsSystem(bytes); }
        catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "malformed compound instance array was accepted");

        const auto cube = empire::assets::loadHavokPhysicsSystem(makeCubePackfile());
        const auto shape = empire::physics::makeJoltCollisionShape(cube.bodies.front());
        constexpr std::uint32_t side = 33;
        constexpr float spacing = 128.0f;
        std::array<float, side * side> heights{};
        for (std::uint32_t y = 0; y < side; ++y)
            for (std::uint32_t x = 0; x < side; ++x)
                heights[y * side + x] = 100.0f + static_cast<float>(x) * 2.0f;
        empire::physics::JoltWorld world;
        const glm::vec3 origin(8192.0f, 0.0f, -4096.0f);
        world.setOrigin(origin);
        auto terrain = world.addStaticHeightField(heights, side, spacing, origin);
        const float surface = heights[16 * side + 16];
        auto falling = world.addDynamicBody(shape,
            origin + glm::vec3(16.0f * spacing, surface + 500.0f, 16.0f * spacing));
        const auto occupiedBeforeRebase = world.dynamicBodyPositions();
        require(occupiedBeforeRebase.size() == 1
                && glm::distance(occupiedBeforeRebase.front(),
                    origin + glm::vec3(16.0f * spacing, surface + 500.0f,
                        16.0f * spacing)) < 0.01f,
            "dynamic body residency position is not in Empire world space");
        for (int step = 0; step < 300; ++step)
            world.step(1.0f / 60.0f);
        const float expected = surface + empire::physics::BethesdaUnitsPerMeter;
        require(std::abs(world.bodyTransform(falling)[3].y - expected) < 12.0f,
            "body did not settle on the non-zero-origin LAND slope");
        const glm::vec3 beforeRebase(world.bodyTransform(falling)[3]);
        world.setOrigin(origin + glm::vec3(4096.0f, 0.0f, 4096.0f));
        const glm::vec3 afterRebase(world.bodyTransform(falling)[3]);
        require(glm::distance(beforeRebase, afterRebase) < 0.01f,
            "floating-origin rebase moved a body in world space");
        require(world.dynamicBodyPositions().size() == 1
                && glm::distance(world.dynamicBodyPositions().front(), afterRebase) < 0.01f,
            "floating-origin rebase moved dynamic body residency");
        world.removeBody(falling);
        require(world.dynamicBodyPositions().empty(),
            "removed dynamic body remained in terrain residency tracking");
        world.removeBody(terrain);
        terrain = world.addStaticHeightField(heights, side, spacing, origin);
        world.removeBody(terrain);
        world.removeBody(terrain);
    }

    void testAdditionalConvexShapes()
    {
        const auto convex = empire::assets::loadHavokPhysicsSystem(makeConvexPackfile());
        require(convex.bodies.size() == 1 && convex.bodies.front().parts.size() == 1,
            "plain convex shape was not decoded");
        require(convex.bodies.front().parts.front().polytope.vertices.size() == 8
                && convex.bodies.front().parts.front().polytope.indices.empty(),
            "plain convex shape used polytope face data");
        require(empire::physics::makeJoltCollisionShape(convex.bodies.front()) != nullptr,
            "Jolt rejected the plain convex shape");

        auto capsuleBytes = makeCapsulePackfile();
        const auto capsule = empire::assets::loadHavokPhysicsSystem(capsuleBytes);
        require(capsule.bodies.size() == 1 && capsule.bodies.front().capsules.size() == 1,
            "capsule shape was not decoded");
        const auto& decoded = capsule.bodies.front().capsules.front().capsule;
        require(decoded.pointA == glm::vec3(0.0f, -1.0f, 0.0f)
                && decoded.pointB == glm::vec3(0.0f, 1.0f, 0.0f)
                && std::abs(decoded.radius - 0.25f) < 0.00001f,
            "capsule dimensions changed");
        require(empire::physics::makeJoltCollisionShape(capsule.bodies.front()) != nullptr,
            "Jolt rejected the capsule shape");

        auto nonUniformCapsule = capsule.bodies.front();
        nonUniformCapsule.capsules.front().scale = {1.0f, 2.0f, 1.0f};
        bool rejectedNonUniform = false;
        try { (void)empire::physics::makeJoltCollisionShape(nonUniformCapsule); }
        catch (const std::runtime_error&) { rejectedNonUniform = true; }
        require(rejectedNonUniform, "non-uniform capsule scale was accepted by Jolt");

        constexpr std::uint32_t shape = 0xe0;
        const ClassTable classes = makeClasses();
        const std::size_t dataStart = 0x100 + classes.bytes.size();
        write<float>(capsuleBytes, dataStart + shape + 0x54,
            std::numeric_limits<float>::quiet_NaN());
        bool rejected = false;
        try { (void)empire::assets::loadHavokPhysicsSystem(capsuleBytes); }
        catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "capsule with a non-finite endpoint was accepted");

        capsuleBytes = makeCapsulePackfile();
        write<float>(capsuleBytes, dataStart + shape + 0x14, -1.0f);
        rejected = false;
        try { (void)empire::assets::loadHavokPhysicsSystem(capsuleBytes); }
        catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "capsule with a negative radius was accepted");

        auto mixed = convex.bodies.front();
        mixed.capsules = capsule.bodies.front().capsules;
        const auto compressed = empire::assets::loadHavokPhysicsSystem(
            makeCompressedMeshPackfile());
        mixed.triangleMesh = compressed.bodies.front().triangleMesh;
        require(empire::physics::makeJoltCollisionShape(mixed) != nullptr,
            "Jolt rejected a mixed mesh, convex, and capsule compound");
    }

    void testCompressedMesh()
    {
        constexpr std::uint32_t shape = 0xe0;
        constexpr std::uint32_t shapeData = 0x190;
        constexpr std::uint32_t tree = shapeData + 0x10;
        constexpr std::uint32_t sections = 0x270;
        constexpr std::uint32_t primitives = 0x2d0;
        constexpr std::uint32_t sharedIndices = 0x2d8;
        constexpr std::uint32_t globalFixups = 0x358;

        const auto bytes = makeCompressedMeshPackfile();
        const auto system = empire::assets::loadHavokPhysicsSystem(bytes);
        require(system.bodies.size() == 1, "compressed mesh body was not decoded");
        const auto& body = system.bodies.front();
        require(body.parts.empty() && body.triangleMesh.has_value(),
            "compressed mesh used the wrong body representation");
        const auto& mesh = *body.triangleMesh;
        require(mesh.sectionCount == 1, "compressed mesh section count changed");
        require(mesh.vertices.size() == 4, "compressed mesh vertices were not reconstructed");
        require(mesh.indices.size() == 9, "compressed mesh triangle and quad were not decoded");
        require(mesh.primitiveDataRuns.size() == 2,
            "compressed mesh primitive runs were not preserved");
        require(mesh.primitiveDataRuns[0].value == 7
                && mesh.primitiveDataRuns[0].index == 0
                && mesh.primitiveDataRuns[0].count == 1
                && mesh.primitiveDataRuns[1].value == 9
                && mesh.primitiveDataRuns[1].index == 1
                && mesh.primitiveDataRuns[1].count == 1,
            "compressed mesh primitive run values changed");
        require(glm::distance(mesh.vertices[0], glm::vec3(0.0f, 0.0f, 1.0f)) < 0.00001f,
            "compressed shared vertex was not reconstructed");
        require(glm::distance(mesh.vertices[2], glm::vec3(1.0f, 0.0f, 0.0f)) < 0.00001f,
            "compressed local vertex was not reconstructed");
        const auto shapeResult = empire::physics::makeJoltCollisionShape(body);
        require(shapeResult != nullptr, "Jolt rejected the synthetic compressed mesh");
        const auto bounds = shapeResult->GetLocalBounds();
        require(bounds.mMax.GetY() > 0.99f && bounds.mMax.GetZ() > 0.99f,
            "compressed mesh axes were not converted for Jolt");

        std::uint32_t dataStart = 0;
        std::memcpy(&dataStart, bytes.data() + 0xc0 + 20, sizeof(dataStart));
        const auto rejected = [&](auto corrupt, std::string_view message) {
            bool didReject = false;
            try { (void)empire::assets::loadHavokPhysicsSystem(corrupt); }
            catch (const std::runtime_error&) { didReject = true; }
            require(didReject, message);
        };

        auto corrupt = bytes;
        std::fill(corrupt.begin() + dataStart + globalFixups + 12,
            corrupt.begin() + dataStart + globalFixups + 24, std::byte{0xff});
        rejected(std::move(corrupt), "compressed mesh without data fixup was accepted");

        corrupt = bytes;
        write<std::uint32_t>(corrupt, dataStart + tree + 0x48, 0xffffffffu);
        rejected(std::move(corrupt), "oversized compressed mesh section array was accepted");

        corrupt = bytes;
        write<std::uint32_t>(corrupt, dataStart + sections + 0x50, 0xffffff02u);
        rejected(std::move(corrupt), "invalid compressed mesh section range was accepted");

        corrupt = bytes;
        write<float>(corrupt, dataStart + sections + 0x3c,
            std::numeric_limits<float>::quiet_NaN());
        rejected(std::move(corrupt), "invalid compressed mesh codec was accepted");

        corrupt = bytes;
        write<std::uint16_t>(corrupt, dataStart + sharedIndices, 10);
        rejected(std::move(corrupt), "invalid compressed mesh shared index was accepted");

        corrupt = bytes;
        write<std::uint8_t>(corrupt, dataStart + primitives, 0);
        write<std::uint8_t>(corrupt, dataStart + primitives + 1, 0);
        write<std::uint8_t>(corrupt, dataStart + primitives + 2, 0);
        rejected(std::move(corrupt), "degenerate compressed mesh primitive was accepted");

        auto dynamicMesh = body;
        dynamicMesh.dynamic = true;
        bool rejectedDynamic = false;
        try { (void)empire::physics::makeJoltCollisionShape(dynamicMesh); }
        catch (const std::runtime_error&) { rejectedDynamic = true; }
        require(rejectedDynamic, "dynamic compressed mesh was accepted by Jolt");

        empire::physics::JoltWorld world;
        auto staticBody = world.addStaticBody(shapeResult);
        require(empire::physics::scaleJoltCollisionShape(
                    shapeResult, glm::vec3(2.0f, 1.0f, 0.5f)) != nullptr,
            "Jolt rejected a valid streamed-reference shape scale");
        bool rejectedScale = false;
        try
        {
            (void)empire::physics::scaleJoltCollisionShape(
                shapeResult, glm::vec3(1.0f, 0.0f, 1.0f));
        }
        catch (const std::runtime_error&)
        {
            rejectedScale = true;
        }
        require(rejectedScale, "Jolt accepted an invalid streamed-reference shape scale");
        empire::assets::HavokRigidBody smallBox;
        smallBox.parts.emplace_back();
        constexpr std::array<float, 2> signs{-0.1f, 0.1f};
        for (const float z : signs)
            for (const float y : signs)
                for (const float x : signs)
                    smallBox.parts.front().polytope.vertices.emplace_back(x, y, z);
        const auto smallBoxShape = empire::physics::makeJoltCollisionShape(smallBox);
        const auto falling = world.addDynamicBody(smallBoxShape,
            glm::vec3(0.25f, 1.0f, 0.25f) * empire::physics::BethesdaUnitsPerMeter);
        for (int step = 0; step < 180; ++step)
            world.step(1.0f / 60.0f);
        require(world.bodyTransform(falling)[3].y > 0.05f
                * empire::physics::BethesdaUnitsPerMeter,
            "dynamic hull fell through synthetic compressed mesh");
        world.removeBody(falling);
        world.removeBody(staticBody);
        world.removeBody(staticBody);
    }

    void testRagdoll()
    {
        constexpr std::uint32_t motions = 0x100;
        constexpr std::uint32_t constraints = 0x3e0;
        constexpr std::uint32_t parents = 0x460;
        constexpr std::uint32_t bodyMap = 0x530;
        constexpr std::uint32_t globalFixups = 0x838;
        constexpr std::uint32_t virtualFixups = 0x880;
        const auto bytes = makeRagdollPackfile();
        const auto ragdoll = empire::assets::loadHavokRagdoll(bytes);
        require(ragdoll.bones.size() == 3 && ragdoll.bodies.size() == 3,
            "synthetic ragdoll body hierarchy changed");
        require(ragdoll.constraints.size() == 2
                && ragdoll.constraints[0].type
                    == empire::assets::HavokConstraintType::Ragdoll
                && ragdoll.constraints[1].type
                    == empire::assets::HavokConstraintType::LimitedHinge,
            "synthetic ragdoll constraint types changed");
        require(ragdoll.bones[0].parent == -1 && ragdoll.bones[1].parent == 0
                && ragdoll.bones[2].parent == 1,
            "synthetic ragdoll parent hierarchy changed");
        require(ragdoll.bones[0].lockTranslation
                && glm::vec3(ragdoll.bones[2].referencePose[3])
                    == glm::vec3(0.0f, 0.0f, 2.0f),
            "synthetic ragdoll reference pose changed");
        require(std::ranges::equal(ragdoll.boneToBody,
                    std::array<std::uint32_t, 3>{0, 1, 2}),
            "synthetic ragdoll body map changed");
        require(std::abs(ragdoll.bodies[0].mass - 1.0f) < 0.00001f
                && std::abs(ragdoll.bodies[1].mass - 2.0f) < 0.00001f
                && std::abs(ragdoll.bodies[2].mass - 3.0f) < 0.00001f,
            "synthetic ragdoll masses changed");
        require(std::abs(ragdoll.constraints[0].minimumTwistAngle + 0.3f) < 0.00001f
                && std::abs(ragdoll.constraints[0].maximumConeAngle - 0.6f) < 0.00001f
                && std::abs(ragdoll.constraints[0].minimumPlaneAngle + 0.2f) < 0.00001f
                && std::abs(ragdoll.constraints[1].maximumTwistAngle - 0.8f) < 0.00001f,
            "synthetic ragdoll limits changed");
        require(glm::vec3(ragdoll.constraints[0].localFrameA[3])
                    == glm::vec3(0.0f, 0.0f, 0.5f)
                && glm::vec3(ragdoll.constraints[0].localFrameB[3])
                    == glm::vec3(0.0f, 0.0f, -0.5f),
            "synthetic ragdoll local frames changed");

        empire::physics::JoltWorld world;
        constexpr float units = empire::physics::BethesdaUnitsPerMeter;
        (void)world.addStaticBox({5.0f * units, 0.25f * units, 5.0f * units},
            {0.0f, -0.25f * units, 0.0f});
        auto handle = world.addRagdoll(ragdoll, {0.0f, 4.0f * units, 0.0f});
        const auto initialTransforms = world.ragdollBodyTransforms(handle);
        require(initialTransforms.size() == 3
                && std::abs(initialTransforms[0][3].y - 4.0f * units) < 0.01f
                && std::abs(initialTransforms[1][3].y - 5.0f * units) < 0.01f
                && std::abs(initialTransforms[2][3].y - 6.0f * units) < 0.01f,
            "synthetic ragdoll body transforms use the wrong coordinate axes");
        for (int step = 0; step < 300; ++step)
            world.step(1.0f / 60.0f);
        const auto transforms = world.ragdollBodyTransforms(handle);
        require(transforms.size() == 3
                && std::ranges::all_of(transforms, [](const glm::mat4& transform) {
                    return std::isfinite(transform[3].x)
                        && std::isfinite(transform[3].y)
                        && std::isfinite(transform[3].z)
                        && transform[3].y > -100.0f;
                }),
            "synthetic Jolt ragdoll did not settle to a finite pose");
        require(glm::distance(glm::vec3(transforms[0][3]), glm::vec3(transforms[1][3]))
                    < 2.0f * units
                && glm::distance(glm::vec3(transforms[1][3]), glm::vec3(transforms[2][3]))
                    < 2.0f * units,
            "synthetic Jolt ragdoll joints separated");
        world.removeRagdoll(handle);
        world.removeRagdoll(handle);
        handle = world.addRagdoll(ragdoll, {0.0f, 4.0f * units, 0.0f});
        world.removeRagdoll(handle);

        std::uint32_t dataStart = 0;
        std::memcpy(&dataStart, bytes.data() + 0xc0 + 20, sizeof(dataStart));
        const auto rejected = [&](auto corrupt, std::string_view message) {
            bool didReject = false;
            try { (void)empire::assets::loadHavokRagdoll(corrupt); }
            catch (const std::runtime_error&) { didReject = true; }
            require(didReject, message);
        };

        auto corrupt = bytes;
        std::fill(corrupt.begin() + dataStart + globalFixups + 36,
            corrupt.begin() + dataStart + globalFixups + 48, std::byte{0xff});
        rejected(std::move(corrupt), "ragdoll without a skeleton fixup was accepted");

        corrupt = bytes;
        write<std::int16_t>(corrupt, dataStart + parents + 4, 2);
        rejected(std::move(corrupt), "cyclic ragdoll skeleton was accepted");

        corrupt = bytes;
        write<std::int32_t>(corrupt, dataStart + bodyMap + 8, 1);
        rejected(std::move(corrupt), "incomplete ragdoll body map was accepted");

        corrupt = bytes;
        write<std::uint32_t>(corrupt, dataStart + constraints + 0x18 + 8, 0);
        write<std::uint32_t>(corrupt, dataStart + constraints + 0x18 + 12, 1);
        rejected(std::move(corrupt), "duplicate ragdoll parent constraint was accepted");

        corrupt = bytes;
        write<std::uint32_t>(corrupt, dataStart + constraints + 12, 99);
        rejected(std::move(corrupt), "out-of-range ragdoll body index was accepted");

        corrupt = bytes;
        const ClassTable classes = makeClasses();
        write<std::uint32_t>(corrupt, dataStart + virtualFixups + 4 * 12 + 8,
            classes.capsuleName);
        rejected(std::move(corrupt), "unsupported ragdoll constraint class was accepted");

        corrupt = bytes;
        write<float>(corrupt, dataStart + motions + 4, 0.0f);
        rejected(std::move(corrupt), "nonpositive ragdoll mass was accepted");

        corrupt = bytes;
        write<float>(corrupt, dataStart + 0x540 + 0x30,
            std::numeric_limits<float>::quiet_NaN());
        rejected(std::move(corrupt), "non-finite ragdoll frame was accepted");
    }

    int testFallout4Asset()
    {
        const auto data = empire::assets::findFallout4Data();
        if (!data)
            return 77;
        const auto archivePath = *data / "Fallout4 - Meshes.ba2";
        if (!std::filesystem::exists(archivePath))
            return 77;
        empire::assets::Ba2Archive archive(archivePath);
        constexpr std::string_view path = "Meshes\\SetDressing\\Ammo\\AmmoBoxNotSkinned.nif";
        if (!archive.contains(path))
            return 77;
        const auto nif = archive.extract(path);
        const auto systems = empire::assets::loadNifPhysicsSystems(nif);
        require(systems.size() == 1 && systems.front().bodies.size() == 1,
            "Fallout 4 ammo box collision body was not decoded");
        const auto& body = systems.front().bodies.front();
        require(!body.dynamic, "Fallout 4 ammo box should have a static body");
        require(std::abs(body.position.z - 0.20356484f) < 0.0001f,
            "Fallout 4 ammo box body transform changed");
        require(body.parts.size() == 1, "Fallout 4 ammo box part count changed");
        const auto& hull = body.parts.front().polytope;
        require(hull.vertices.size() == 8, "Fallout 4 ammo box vertex count changed");
        require(hull.indices.size() == 36, "Fallout 4 ammo box triangle count changed");
        require(std::abs(hull.convexRadius - 0.01f) < 0.0001f,
            "Fallout 4 ammo box radius changed");
        require(empire::physics::makeJoltCollisionShape(body) != nullptr,
            "Jolt rejected the Fallout 4 ammo box hull");
        const auto ammoShape = empire::physics::makeJoltCollisionShape(body);

        const auto testShapeAsset = [&](std::string_view assetPath,
                                        bool expectPlainConvex,
                                        bool expectCapsule,
                                        bool expectMixed) {
            require(archive.contains(assetPath), "Fallout 4 shape fixture is missing");
            std::vector<empire::assets::HavokPhysicsSystem> shapeSystems;
            try
            {
                shapeSystems = empire::assets::loadNifPhysicsSystems(
                    archive.extract(assetPath));
            }
            catch (const std::exception& error)
            {
                throw std::runtime_error(std::string(assetPath) + ": " + error.what());
            }
            require(!shapeSystems.empty(), "Fallout 4 shape fixture has no physics system");
            bool sawPlainConvex = false;
            bool sawCapsule = false;
            bool sawMixed = false;
            std::size_t bodyCount = 0;
            for (const auto& shapeSystem : shapeSystems)
                for (const auto& shapeBody : shapeSystem.bodies)
                {
                    ++bodyCount;
                    sawPlainConvex = sawPlainConvex || std::ranges::any_of(
                        shapeBody.parts, [](const auto& part) {
                            return part.polytope.indices.empty();
                        });
                    sawCapsule = sawCapsule || !shapeBody.capsules.empty();
                    sawMixed = sawMixed || (shapeBody.triangleMesh.has_value()
                        && (!shapeBody.parts.empty() || !shapeBody.capsules.empty()));
                    const auto joltShape = empire::physics::makeJoltCollisionShape(shapeBody);
                    require(joltShape != nullptr,
                        "Jolt rejected a Fallout 4 shape fixture body");
                    if (expectPlainConvex && joltShape->GetMassProperties().mMass <= 0.0f)
                    {
                        empire::physics::JoltWorld fixtureWorld;
                        require(static_cast<bool>(fixtureWorld.addDynamicBody(
                                    joltShape, glm::vec3(0.0f),
                                    glm::quat(1.0f, 0.0f, 0.0f, 0.0f), 10.0f)),
                            "Jolt rejected explicit mass for a zero-volume Fallout 4 fixture");
                        fixtureWorld.step(1.0f / 60.0f);
                    }
                }
            require(bodyCount > 0, "Fallout 4 shape fixture has no rigid bodies");
            require(!expectPlainConvex || sawPlainConvex,
                "Fallout 4 plain convex shape was not decoded");
            require(!expectCapsule || sawCapsule,
                "Fallout 4 capsule shape was not decoded");
            require(!expectMixed || sawMixed,
                "Fallout 4 mixed compound shape was not decoded");
        };

        testShapeAsset("Meshes\\Architecture\\Buildings\\DecoKit\\DecoMainA1x1WinC01.nif",
            true, false, false);
        testShapeAsset("Meshes\\SetDressing\\Greebs\\Greebles01.nif",
            false, true, false);
        testShapeAsset("Meshes\\SetDressing\\StandPipes\\StandPipe02.nif",
            false, true, false);
        for (const std::string_view mixedPath : {
                 "Meshes\\SetDressing\\Greebs\\WallGreeb1x1_01.nif",
                 "Meshes\\SetDressing\\Greebs\\WallGreeb1x1_02.nif",
                 "Meshes\\SetDressing\\Greebs\\WallGreeb1x1_03.nif"})
            testShapeAsset(mixedPath, false, false, true);

        const auto testStaticMesh = [&](std::string_view name,
                                        std::span<const std::byte> nif,
                                        glm::vec3& boundsMinimum,
                                        glm::vec3& boundsMaximum) {
            const auto meshSystems = empire::assets::loadNifPhysicsSystems(nif);
            std::size_t meshBodies = 0;
            std::size_t sections = 0;
            std::size_t vertices = 0;
            std::size_t triangles = 0;
            std::size_t runs = 0;
            glm::vec3 dropPoint{};
            glm::vec3 supportNormal{};
            float bestArea = 0.0f;
            std::size_t bestShape = 0;
            std::vector<JPH::ShapeRefC> shapes;
            boundsMinimum = glm::vec3(std::numeric_limits<float>::max());
            boundsMaximum = glm::vec3(std::numeric_limits<float>::lowest());
            for (const auto& meshSystem : meshSystems)
                for (const auto& meshBody : meshSystem.bodies)
                {
                    if (!meshBody.triangleMesh)
                        continue;
                    ++meshBodies;
                    const auto& decoded = *meshBody.triangleMesh;
                    sections += decoded.sectionCount;
                    vertices += decoded.vertices.size();
                    triangles += decoded.indices.size() / 3;
                    runs += decoded.primitiveDataRuns.size();
                    for (const glm::vec3& vertex : decoded.vertices)
                    {
                        boundsMinimum = glm::min(boundsMinimum, vertex);
                        boundsMaximum = glm::max(boundsMaximum, vertex);
                    }
                    shapes.push_back(empire::physics::makeJoltCollisionShape(meshBody));
                    for (std::size_t index = 0; index < decoded.indices.size(); index += 3)
                    {
                        const auto joltVertex = [&](std::uint32_t vertexIndex) {
                            const glm::vec3& vertex = decoded.vertices[vertexIndex];
                            return glm::vec3(vertex.x, vertex.z, vertex.y)
                                * empire::physics::BethesdaUnitsPerMeter;
                        };
                        const glm::vec3 a = joltVertex(decoded.indices[index]);
                        const glm::vec3 b = joltVertex(decoded.indices[index + 2]);
                        const glm::vec3 c = joltVertex(decoded.indices[index + 1]);
                        const glm::vec3 normal = glm::cross(b - a, c - a);
                        const float area = glm::length(normal);
                        if (area > bestArea)
                        {
                            bestArea = area;
                            supportNormal = normal / area;
                            dropPoint = (a + b + c) / 3.0f;
                            bestShape = shapes.size() - 1;
                        }
                    }
                }
            require(meshBodies > 0 && sections > 0 && vertices > 0 && triangles > 0,
                "Fallout 4 compressed mesh was not decoded");
            require(bestArea > 1.0f, "Fallout 4 compressed mesh has no collision surface");
            std::cout << name << ": bodies=" << meshBodies << " sections=" << sections
                      << " vertices=" << vertices << " triangles=" << triangles
                      << " runs=" << runs << " bounds=(" << boundsMinimum.x << ','
                      << boundsMinimum.y << ',' << boundsMinimum.z << ")-("
                      << boundsMaximum.x << ',' << boundsMaximum.y << ','
                      << boundsMaximum.z << ")\n";
            empire::physics::JoltWorld meshWorld;
            const glm::quat supportRotation = rotationBetween(
                supportNormal, glm::vec3(0.0f, 1.0f, 0.0f));
            dropPoint = supportRotation * dropPoint;
            (void)meshWorld.addStaticBody(shapes[bestShape], glm::vec3(0.0f), supportRotation);
            const auto falling = meshWorld.addDynamicBody(ammoShape,
                dropPoint + glm::vec3(0.0f, 300.0f, 0.0f));
            for (int step = 0; step < 300; ++step)
                meshWorld.step(1.0f / 60.0f);
            const float settled = meshWorld.bodyTransform(falling)[3].y;
            require(settled > dropPoint.y - 50.0f && settled < dropPoint.y + 300.0f,
                "dynamic hull did not settle on Fallout 4 compressed mesh");
            return std::array<std::size_t, 5>{
                meshBodies, sections, vertices, triangles, runs};
        };

        constexpr std::string_view oldStateHousePath =
            "Meshes\\Architecture\\Unique\\OldStateHouse.nif";
        constexpr std::string_view bathHousePath =
            "Meshes\\Architecture\\Unique\\BathHouse\\BathHouseMain01.nif";
        require(archive.contains(oldStateHousePath), "OldStateHouse.nif is missing");
        require(archive.contains(bathHousePath), "BathHouseMain01.nif is missing");
        glm::vec3 oldStateHouseMinimum;
        glm::vec3 oldStateHouseMaximum;
        const auto oldStateHouseCounts = testStaticMesh(
            "OldStateHouse", archive.extract(oldStateHousePath),
            oldStateHouseMinimum, oldStateHouseMaximum);
        glm::vec3 bathHouseMinimum;
        glm::vec3 bathHouseMaximum;
        const auto bathHouseCounts = testStaticMesh(
            "BathHouseMain01", archive.extract(bathHousePath),
            bathHouseMinimum, bathHouseMaximum);
        require(oldStateHouseCounts == std::array<std::size_t, 5>{3, 26, 2514, 3918, 694},
            "OldStateHouse compressed collision counts changed");
        require(glm::distance(oldStateHouseMinimum,
                    glm::vec3(-5.65786f, -16.6306f, -3.6576f)) < 0.001f
                && glm::distance(oldStateHouseMaximum,
                    glm::vec3(5.65786f, 18.5577f, 14.7447f)) < 0.001f,
            "OldStateHouse compressed collision bounds changed");
        require(bathHouseCounts == std::array<std::size_t, 5>{1, 2, 167, 252, 23},
            "BathHouseMain01 compressed collision counts changed");
        require(glm::distance(bathHouseMinimum,
                    glm::vec3(-5.4864f, -0.6858f, 0.0f)) < 0.001f
                && glm::distance(bathHouseMaximum,
                    glm::vec3(5.4864f, 0.6858f, 13.9538f)) < 0.001f,
            "BathHouseMain01 compressed collision bounds changed");

        const auto scolPath = *data / "Meshes" / "SCOL" / "Fallout4.esm" / "CM0005094C.NIF";
        require(std::filesystem::exists(scolPath), "CM0005094C.NIF is missing");
        const auto scolBytes = [&] {
            std::ifstream stream(scolPath, std::ios::binary | std::ios::ate);
            require(stream.good(), "CM0005094C.NIF could not be opened");
            const auto size = stream.tellg();
            require(size > 0, "CM0005094C.NIF is empty");
            std::vector<std::byte> result(static_cast<std::size_t>(size));
            stream.seekg(0);
            stream.read(reinterpret_cast<char*>(result.data()), size);
            require(stream.good(), "CM0005094C.NIF could not be read");
            return result;
        }();
        glm::vec3 scolMinimum;
        glm::vec3 scolMaximum;
        const auto scolCounts = testStaticMesh(
            "CM0005094C", scolBytes, scolMinimum, scolMaximum);
        require(scolCounts == std::array<std::size_t, 5>{1, 1, 21, 28, 5},
            "CM0005094C compressed collision counts changed");
        require(glm::distance(scolMinimum,
                    glm::vec3(-0.976196f, -0.0922583f, -0.428315f)) < 0.001f
                && glm::distance(scolMaximum,
                    glm::vec3(0.976196f, 0.0922583f, 0.428315f)) < 0.001f,
            "CM0005094C compressed collision bounds changed");

        constexpr std::string_view pistolPath =
            "Meshes\\Weapons\\10mmPistol\\10MMPistol.nif";
        require(archive.contains(pistolPath), "Fallout 4 10mm pistol is missing");
        const auto pistolNif = archive.extract(pistolPath);
        const auto pistolSystems = empire::assets::loadNifPhysicsSystems(pistolNif);
        require(pistolSystems.size() == 1 && pistolSystems.front().bodies.size() == 1,
            "Fallout 4 10mm pistol body was not decoded");
        const auto& pistol = pistolSystems.front().bodies.front();
        require(pistol.dynamic && pistol.mass > 0.0f,
            "Fallout 4 10mm pistol dynamics changed");
        require(pistol.parts.size() == 3, "Fallout 4 10mm pistol part count changed");
        require(std::ranges::all_of(pistol.parts,
            [](const auto& part) { return part.polytope.vertices.size() == 8; }),
            "Fallout 4 10mm pistol convex vertex count changed");
        const auto pistolShape = empire::physics::makeJoltCollisionShape(pistol);
        require(pistolShape != nullptr, "Jolt rejected the Fallout 4 10mm pistol compound");

        const auto geometry = empire::assets::loadNifModel(pistolNif);
        require(std::ranges::any_of(geometry.attachmentPoints, [](const auto& point) {
            return point.name == "P-Grip";
        }), "10mm pistol grip attachment point is missing");
        constexpr std::string_view gripPath =
            "Meshes\\Weapons\\10mmPistol\\10mmGrip.nif";
        require(archive.contains(gripPath), "Fallout 4 10mm grip is missing");
        const auto grip = empire::assets::loadNifModel(archive.extract(gripPath));
        require(std::ranges::any_of(grip.attachmentTargets, [](const std::string& target) {
            return target == "C-Grip";
        }), "10mm grip attachment target is missing");
        glm::vec3 renderMinimum(std::numeric_limits<float>::max());
        glm::vec3 renderMaximum(std::numeric_limits<float>::lowest());
        for (const auto& surface : geometry.surfaces)
            for (const auto& vertex : surface.vertices)
            {
                renderMinimum = glm::min(renderMinimum, vertex.position);
                renderMaximum = glm::max(renderMaximum, vertex.position);
            }
        const auto collisionBounds = pistolShape->GetLocalBounds();
        const auto centerOfMass = pistolShape->GetCenterOfMass();
        const glm::vec3 collisionMinimumMeters(collisionBounds.mMin.GetX(),
            collisionBounds.mMin.GetY(), collisionBounds.mMin.GetZ());
        const glm::vec3 collisionMaximumMeters(collisionBounds.mMax.GetX(),
            collisionBounds.mMax.GetY(), collisionBounds.mMax.GetZ());
        const glm::vec3 collisionCenterOfMassMeters(
            centerOfMass.GetX(), centerOfMass.GetY(), centerOfMass.GetZ());
        const glm::mat4 bodyLocal = empire::physics::havokBodyLocalTransform(pistol);
        glm::vec3 collisionMinimum(std::numeric_limits<float>::max());
        glm::vec3 collisionMaximum(std::numeric_limits<float>::lowest());
        for (int corner = 0; corner < 8; ++corner)
        {
            const glm::vec3 local(
                (corner & 1) ? collisionMaximumMeters.x : collisionMinimumMeters.x,
                (corner & 2) ? collisionMaximumMeters.y : collisionMinimumMeters.y,
                (corner & 4) ? collisionMaximumMeters.z : collisionMinimumMeters.z);
            const glm::vec3 transformed = glm::vec3(bodyLocal * glm::vec4(
                (local + collisionCenterOfMassMeters)
                    * empire::physics::BethesdaUnitsPerMeter,
                1.0f));
            collisionMinimum = glm::min(collisionMinimum, transformed);
            collisionMaximum = glm::max(collisionMaximum, transformed);
        }
        const glm::vec3 renderExtent = renderMaximum - renderMinimum;
        const glm::vec3 collisionExtent = collisionMaximum - collisionMinimum;
        const float renderSize = std::max({renderExtent.x, renderExtent.y, renderExtent.z});
        require(glm::all(glm::greaterThan(collisionExtent, renderExtent * 0.65f))
                && glm::all(glm::lessThan(collisionExtent, renderExtent * 1.25f)),
            "10mm pistol collision and render bounds use different units");
        require(glm::distance((collisionMinimum + collisionMaximum) * 0.5f,
                    (renderMinimum + renderMaximum) * 0.5f) < renderSize * 0.1f,
            "10mm pistol body-local transform moved collision away from render geometry");

        constexpr std::string_view skeletonPath =
            "Meshes\\Actors\\Character\\CharacterAssets\\skeleton.nif";
        require(archive.contains(skeletonPath), "Fallout 4 human skeleton is missing");
        const auto ragdolls = empire::assets::loadNifRagdolls(
            archive.extract(skeletonPath));
        require(ragdolls.size() == 1, "Fallout 4 human ragdoll was not decoded");
        const auto& ragdoll = ragdolls.front();
        require(ragdoll.bones.size() == 18 && ragdoll.bodies.size() == 18,
            "Fallout 4 human ragdoll body count changed");
        require(ragdoll.constraints.size() == 17,
            "Fallout 4 human ragdoll constraint count changed");
        require(std::ranges::equal(ragdoll.boneToBody,
                    std::views::iota(std::uint32_t{0}, std::uint32_t{18})),
            "Fallout 4 human ragdoll body map changed");
        require(std::ranges::count_if(ragdoll.constraints, [](const auto& constraint) {
                    return constraint.type == empire::assets::HavokConstraintType::Ragdoll;
                }) == 9
                && std::ranges::count_if(ragdoll.constraints, [](const auto& constraint) {
                    return constraint.type
                        == empire::assets::HavokConstraintType::LimitedHinge;
                }) == 8,
            "Fallout 4 human ragdoll constraint types changed");
        constexpr std::string_view maleBodyPath =
            "Meshes\\Actors\\Character\\CharacterAssets\\MaleBody.nif";
        require(archive.contains(maleBodyPath), "Fallout 4 male body is missing");
        const auto maleBody = empire::assets::loadNifModel(archive.extract(maleBodyPath));
        require(!maleBody.skinBones.empty() && maleBody.skinBones.size() <= 256,
            "Fallout 4 male body skin palette was not decoded");
        for (const auto& surface : maleBody.surfaces)
            for (const auto& vertex : surface.vertices)
            {
                const float sum = vertex.boneWeights.x + vertex.boneWeights.y
                    + vertex.boneWeights.z + vertex.boneWeights.w;
                require(std::abs(sum - 1.0f) < 0.001f,
                    "Fallout 4 male body has an unnormalized skin vertex");
                require(glm::all(glm::lessThan(vertex.boneIndices,
                            glm::uvec4(maleBody.skinBones.size()))),
                    "Fallout 4 male body has an out-of-range skin bone");
            }
        std::vector<glm::vec3> skinBindPositions;
        for (const auto& bone : maleBody.skinBones)
            skinBindPositions.emplace_back(glm::inverse(bone.skinToBone)[3]);
        std::vector<glm::vec3> ragdollBodyPositions;
        for (std::size_t bone = 0; bone < ragdoll.bones.size(); ++bone)
            ragdollBodyPositions.emplace_back(empire::physics::havokBodyLocalTransform(
                ragdoll.bodies[ragdoll.boneToBody[bone]])[3]);
        const auto byX = [](const glm::vec3& left, const glm::vec3& right) {
            return left.x < right.x;
        };
        const auto [skinLeft, skinRight] = std::ranges::minmax_element(
            skinBindPositions, byX);
        const auto [bodyLeft, bodyRight] = std::ranges::minmax_element(
            ragdollBodyPositions, byX);
        const glm::vec3 skinToRagdoll = ((*bodyLeft - *skinLeft)
            + (*bodyRight - *skinRight)) * 0.5f;
        std::vector<bool> mappedRagdollBodies(ragdoll.bones.size(), false);
        for (const glm::vec3& skinPosition : skinBindPositions)
        {
            const glm::vec3 aligned = skinPosition + skinToRagdoll;
            const auto closest = std::ranges::min_element(ragdollBodyPositions,
                [&](const glm::vec3& left, const glm::vec3& right) {
                    const glm::vec3 leftOffset = aligned - left;
                    const glm::vec3 rightOffset = aligned - right;
                    return glm::dot(leftOffset, leftOffset)
                        < glm::dot(rightOffset, rightOffset);
                });
            mappedRagdollBodies[static_cast<std::size_t>(
                closest - ragdollBodyPositions.begin())] = true;
        }
        require(std::ranges::count(mappedRagdollBodies, true) >= 16,
            "Fallout 4 skin bind pose collapsed onto too few ragdoll limbs");
        empire::physics::JoltWorld ragdollWorld;
        constexpr float units = empire::physics::BethesdaUnitsPerMeter;
        (void)ragdollWorld.addStaticBox(
            {10.0f * units, 0.25f * units, 10.0f * units},
            {0.0f, -0.25f * units, 0.0f});
        auto ragdollHandle = ragdollWorld.addRagdoll(
            ragdoll, {0.0f, 5.0f * units, 0.0f});
        auto ragdollPose = ragdollWorld.ragdollBodyTransforms(ragdollHandle);
        require(ragdollPose.size() == 18,
            "Jolt human ragdoll body count changed");
        const auto initialRagdollPose = ragdollPose;
        for (int step = 0; step < 300; ++step)
            ragdollWorld.step(1.0f / 60.0f);
        ragdollPose = ragdollWorld.ragdollBodyTransforms(ragdollHandle);
        require(std::ranges::all_of(ragdollPose, [](const glm::mat4& transform) {
                    return std::isfinite(transform[3].x)
                        && std::isfinite(transform[3].y)
                        && std::isfinite(transform[3].z)
                        && transform[3].y > -1000.0f;
                }),
            "Jolt human ragdoll did not settle to a finite pose");
        require(ragdollPose.front()[3].y
                < initialRagdollPose.front()[3].y - units,
            "Jolt human ragdoll did not fall onto the static floor");
        for (std::size_t bone = 0; bone < ragdoll.bones.size(); ++bone)
        {
            const auto parent = ragdoll.bones[bone].parent;
            if (parent < 0)
                continue;
            const auto body = ragdoll.boneToBody[bone];
            const auto parentBody = ragdoll.boneToBody[static_cast<std::size_t>(parent)];
            require(glm::distance(glm::vec3(ragdollPose[body][3]),
                        glm::vec3(ragdollPose[parentBody][3])) < 5.0f * units,
                "Jolt human ragdoll joint separated while settling");
        }
        const auto poseBeforeShift = ragdollPose;
        ragdollWorld.setOrigin({4096.0f, 0.0f, -4096.0f});
        ragdollPose = ragdollWorld.ragdollBodyTransforms(ragdollHandle);
        for (std::size_t index = 0; index < ragdollPose.size(); ++index)
            require(glm::distance(glm::vec3(poseBeforeShift[index][3]),
                        glm::vec3(ragdollPose[index][3])) < 0.01f,
                "floating-origin rebase moved a ragdoll limb in world space");
        ragdollWorld.removeRagdoll(ragdollHandle);
        ragdollWorld.removeRagdoll(ragdollHandle);
        ragdollHandle = ragdollWorld.addRagdoll(ragdoll, {0.0f, 5.0f * units, 0.0f});
        ragdollWorld.removeRagdoll(ragdollHandle);

        const auto commonwealth = empire::assets::loadCommonwealth(*data / "Fallout4.esm");
        constexpr empire::assets::CellCoord cell{8, -13};
        const auto landscape = commonwealth.landscapes.find(cell);
        require(landscape != commonwealth.landscapes.end(),
            "Fallout 4 LAND cell 8, -13 is missing");
        constexpr float cellSize = 4096.0f;
        constexpr float spacing = 128.0f;
        constexpr std::size_t center = 16 * empire::assets::LandscapeCell::side + 16;
        const float surface = landscape->second.heights[center];
        require(surface >= -3000.0f && surface <= 12470.0f,
            "Fallout 4 LAND height is outside the expected range");
        empire::physics::JoltWorld world;
        const auto terrain = world.addStaticHeightField(landscape->second.heights,
            empire::assets::LandscapeCell::side, spacing,
            {cell.x * cellSize, 0.0f, cell.y * cellSize});
        const glm::vec3 spawn((cell.x + 0.5f) * cellSize,
            surface + 500.0f, (cell.y + 0.5f) * cellSize);
        const glm::mat4 bodyWorld = glm::translate(glm::mat4(1.0f), spawn) * bodyLocal;
        const auto fallingPistol = world.addDynamicBody(pistolShape, glm::vec3(bodyWorld[3]),
            glm::normalize(glm::quat_cast(glm::mat3(bodyWorld))), pistol.mass,
            pistol.friction, pistol.restitution);
        for (int step = 0; step < 300; ++step)
            world.step(1.0f / 60.0f);
        require(world.bodyTransform(fallingPistol)[3].y > surface - 100.0f,
            "Fallout 4 10mm pistol fell through LAND cell 8, -13");
        world.removeBody(fallingPistol);
        world.removeBody(terrain);
        return 0;
    }
}

int main(int argc, char** argv)
{
    try
    {
        if (argc == 2 && std::string_view(argv[1]) == "--fallout4")
            return testFallout4Asset();
        testSyntheticPackfile();
        testCompoundAndHeightField();
        testAdditionalConvexShapes();
        testCompressedMesh();
        testRagdoll();
        std::cout << "Havok packfile tests passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
