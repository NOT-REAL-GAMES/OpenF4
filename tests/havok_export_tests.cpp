#include "assets/havok_packfile.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string_view>

#include <glm/gtc/matrix_transform.hpp>

using namespace openf4::assets;

namespace
{
    void require(bool condition, std::string_view message)
    {
        if (!condition) throw std::runtime_error(std::string(message));
    }

    template<class F> void rejects(F function)
    {
        bool rejected = false;
        try { function(); } catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "Malformed/unsupported collision was not rejected");
    }

    HavokConvexPart box()
    {
        HavokConvexPart result;
        result.polytope.vertices = {{-1,-1,-1},{1,-1,-1},{1,1,-1},{-1,1,-1},
            {-1,-1,1},{1,-1,1},{1,1,1},{-1,1,1}};
        result.polytope.indices = {0,3,2,0,2,1,4,5,6,4,6,7,0,1,5,0,5,4,
            1,2,6,1,6,5,2,3,7,2,7,6,3,0,4,3,4,7};
        result.polytope.convexRadius = 0.01f;
        return result;
    }

    struct Layout
    {
        const std::vector<std::byte>& bytes;
        std::uint32_t data;
        std::map<std::uint32_t, std::uint32_t> pointers;
        std::multimap<std::string,std::uint32_t> objects;

        template<class T> T get(std::size_t offset) const
        {
            require(offset <= bytes.size() && sizeof(T) <= bytes.size() - offset, "Test read outside packfile");
            T result; std::memcpy(&result, bytes.data() + offset, sizeof(T)); return result;
        }
        template<class T> T at(std::uint32_t offset) const { return get<T>(std::size_t(data) + offset); }

        explicit Layout(const std::vector<std::byte>& source) : bytes(source), data(get<std::uint32_t>(0xd4))
        {
            require(get<std::uint32_t>(0) == 0x57e0e057 && get<std::uint32_t>(4) == 0x10c0c010,
                "Wrong native packfile signature");
            require(get<std::uint32_t>(12) == 11 && get<std::uint32_t>(16) == 0x01000108
                && get<std::uint32_t>(20) == 3 && get<std::uint32_t>(24) == 2,
                "Wrong native version, pointer width, endianness, or section count");
            require(std::memcmp(bytes.data() + 40, "hk_2014.1.0-r1", 14) == 0,
                "Wrong native content version");
            const auto names = get<std::uint32_t>(0x54);
            const auto namesSize = get<std::uint32_t>(0x64);
            const auto local = get<std::uint32_t>(0xd8), global = get<std::uint32_t>(0xdc);
            const auto reflected = get<std::uint32_t>(0xe0), end = get<std::uint32_t>(0xe4);
            require(local <= global && global <= reflected && reflected <= end, "Fixup tables out of order");
            require(names >= 0x100 && names <= bytes.size() && namesSize <= bytes.size() - names
                && std::size_t(names) + namesSize <= data && data <= bytes.size()
                && end <= bytes.size() - data && data % 16 == 0 && local % 16 == 0,
                "Packfile section range/alignment is invalid");
            require((global - local) % 8 == 0 && (reflected - global) % 12 == 0
                && (end - reflected) % 12 == 0, "Fixup table contains a partial entry");
            const auto pointer = [&](std::uint32_t from, std::uint32_t to) {
                require(from < local && local - from >= 8 && from % 8 == 0 && to < local,
                    "Fixup addresses a non-pointer slot or leaves object data");
                require(pointers.emplace(from, to).second, "Duplicate pointer fixup");
            };
            for (auto i = local; i + 8 <= global; i += 8)
                if (at<std::uint32_t>(i) != 0xffffffffu) pointer(at<std::uint32_t>(i), at<std::uint32_t>(i + 4));
            for (auto i = global; i + 12 <= reflected; i += 12)
                if (at<std::uint32_t>(i) != 0xffffffffu)
                {
                    require(at<std::uint32_t>(i + 4) == 2, "Writer pointer left the data section");
                    pointer(at<std::uint32_t>(i), at<std::uint32_t>(i + 8));
                }
            for (auto i = reflected; i + 12 <= end; i += 12)
                if (at<std::uint32_t>(i) != 0xffffffffu)
                {
                    const auto nameOffset = at<std::uint32_t>(i + 8);
                    require(at<std::uint32_t>(i) < local && at<std::uint32_t>(i + 4) == 0
                        && nameOffset >= 5 && nameOffset < namesSize, "Reflected class reference is invalid");
                    const auto name = std::size_t(names) + nameOffset;
                    require(get<std::uint32_t>(name - 5) != 0, "Class has no native reflection signature");
                    const auto* begin = reinterpret_cast<const char*>(bytes.data() + name);
                    const auto* terminator = static_cast<const char*>(std::memchr(begin, 0, namesSize - nameOffset));
                    require(terminator != nullptr, "Unterminated reflected class name");
                    objects.emplace(std::string(begin, terminator), at<std::uint32_t>(i));
                }
        }

        std::uint32_t object(std::string name) const
        {
            const auto found = objects.find(name); require(found != objects.end(), "Missing reflected class");
            return found->second;
        }

        void checkTree(std::uint32_t array, bool top, std::size_t leaves) const
        {
            const auto count = at<std::uint32_t>(array + 8);
            require(count == leaves * 2 - 1, "BVH node count is incomplete");
            const auto start = pointers.at(array); std::set<unsigned> seen;
            std::set<unsigned> visited;
            const std::function<void(unsigned)> visit = [&](unsigned node) {
                require(node < count && visited.insert(node).second, "BVH link is cyclic or out of range");
                const auto offset = start + node * (top ? 5 : 4);
                const auto value = top ? unsigned(at<std::uint8_t>(offset + 3)) << 8 | at<std::uint8_t>(offset + 4)
                    : at<std::uint8_t>(offset + 3);
                if (top ? (value & 0x8000) != 0 : (value & 1) != 0)
                { visit(node + 1); visit(node + (top ? (value & 0x7fff) * 2 : value & 0xfe)); }
                else require(seen.insert(top ? value : value / 2).second, "BVH leaf is duplicated");
            };
            visit(0); require(seen.size() == leaves && visited.size() == count, "BVH misses primitives");
            for (unsigned i = 0; i < leaves; ++i) require(seen.contains(i), "BVH leaf identity changed");
        }
    };

    void testStructureNegativeControls()
    {
        HavokPhysicsSystem source;
        source.bodies.emplace_back().parts.push_back(box());
        const auto bytes = serializeHavokPhysicsSystem(source);
        const Layout valid(bytes);
        const auto changed = [&](std::size_t offset, std::uint32_t value) {
            auto corrupt = bytes;
            std::memcpy(corrupt.data() + offset, &value, sizeof(value));
            rejects([&] { (void)Layout(corrupt); });
        };
        changed(12, 10); // Parseable-looking, wrong target version.
        changed(16, 0x01000104); // Wrong native pointer size.
        changed(0xe4, 0xffffffffu);
        changed(0xdc, valid.get<std::uint32_t>(0xd8) + 1);
        const auto local = valid.get<std::uint32_t>(0xd8);
        changed(std::size_t(valid.data) + local + 4, local); // Pointer into fixups, not an object.
        const auto reflected = valid.get<std::uint32_t>(0xe0);
        changed(std::size_t(valid.data) + reflected + 8, 0xffffffffu);
        auto truncated = bytes; truncated.resize(0xdf);
        rejects([&] { (void)Layout(truncated); });
    }

    void testBodies()
    {
        HavokPhysicsSystem source;
        source.bodies.resize(2);
        source.bodies[0].parts.push_back(box());
        auto& body = source.bodies[1]; body.bodyIndex = 1; body.parts.push_back(box());
        body.position = {2,3,4}; body.rotation = glm::angleAxis(0.75f, glm::vec3(0,0,1));
        body.dynamic = true; body.mass = 3; body.inertiaDiagonal = {2,3,4}; body.centerOfMass = {0.1f,0.2f,0.3f};
        body.inertiaOrientation = glm::angleAxis(0.1f, glm::vec3(1,0,0));
        body.friction = 0.75f; body.restitution = 0.25f; body.collisionFilterInfo = 5;
        body.linearDamping = 0.4f; body.angularDamping = 0.2f; body.gravityFactor = 0.8f;
        auto bytes = serializeHavokPhysicsSystem(source);
        const auto decoded = loadHavokPhysicsSystem(bytes);
        require(decoded.bodies.size() == 2 && !decoded.bodies[0].dynamic && decoded.bodies[1].dynamic,
            "Mixed static/dynamic body indexing changed");
        const auto& result = decoded.bodies[1];
        require(glm::length(result.position - body.position) < 0.00001f && result.mass == body.mass,
            "Body transform/mass failed to round trip");
        require(glm::length(result.centerOfMass - body.centerOfMass) < 0.00001f
            && glm::length(result.inertiaDiagonal - body.inertiaDiagonal) < 0.00001f,
            "Mass properties changed coordinate frame");
        require(std::abs(glm::dot(result.inertiaOrientation, body.inertiaOrientation)) > 0.99999f,
            "Principal inertia rotation changed");
        require(result.friction == body.friction && result.restitution == body.restitution
            && result.linearDamping == body.linearDamping && result.angularDamping == body.angularDamping
            && result.gravityFactor == body.gravityFactor && result.collisionFilterInfo == 5
            && result.materialId == body.materialId, "Body material/motion metadata lost");
        Layout layout(bytes);
        const auto array = layout.pointers.at(0x40), refs = layout.pointers.at(0x60);
        require(layout.pointers.at(array) == layout.pointers.at(refs)
            && layout.pointers.at(array + 0x60) == layout.pointers.at(refs + 8),
            "Referenced objects did not use native 8-byte pointer stride");
        const auto convex = layout.object("hknpConvexPolytopeShape");
        require(layout.at<std::uint16_t>(convex + 0x40) >= 6 && layout.at<std::uint16_t>(convex + 0x42) > 0,
            "Convex support planes are missing");
        require(serializeHavokPhysicsSystem(source) == bytes, "Havok export is nondeterministic");
        require(loadEditableHavokPhysicsSystem(bytes).bodies.size() == 2,
            "Authored rotated mass frames were not editable after reload");
        auto changedSolver = bytes;
        const auto solver = layout.data + layout.pointers.at(0x20) + 0x20;
        const float customSolverThreshold = 0.5f;
        std::memcpy(changedSolver.data() + solver, &customSolverThreshold, sizeof(float));
        require(loadHavokPhysicsSystem(changedSolver).bodies.size() == 2,
            "Source solver settings should remain readable");
        rejects([&] { (void)loadEditableHavokPhysicsSystem(changedSolver); });
        auto changedShapeFlags = bytes;
        changedShapeFlags[layout.data + convex + 0x10] ^= std::byte{0x10};
        require(loadHavokPhysicsSystem(changedShapeFlags).bodies.size() == 2,
            "Source shape flags should remain readable");
        rejects([&] { (void)loadEditableHavokPhysicsSystem(changedShapeFlags); });
        source.bodies[1].mass = 100000000.0f;
        require(loadHavokPhysicsSystem(serializeHavokPhysicsSystem(source)).bodies[1].mass > 99900000.0f,
            "A small positive inverse mass was mistaken for infinite mass");
        source.bodies[1].inertiaDiagonal = {};
        rejects([&] { (void)serializeHavokPhysicsSystem(source); });
        source.nativeRewriteBlockReason = "unsupported source metadata";
        rejects([&] { (void)serializeHavokPhysicsSystem(source); });
    }

    void testMotionProfiles()
    {
        HavokPhysicsSystem source;
        auto& body = source.bodies.emplace_back();
        body.parts.push_back(box()); body.dynamic = true; body.mass = 4; body.inertiaDiagonal = {2, 2, 2};
        body.linearDamping = 0.35f; body.angularDamping = 0.6f;
        body.maxLinearVelocity = 80; body.maxAngularVelocity = 12;
        body.friction = 0.75f; body.restitution = 0.25f; body.collisionFilterInfo = 5;
        require(body.motionProfile == HavokMotionProfile::fallout4, "New collision did not use native motion defaults");
        for (const auto profile : {HavokMotionProfile::fallout4, HavokMotionProfile::legacyOpenF4})
            for (const float gravity : {1.0f, 0.8f})
            {
                body.motionProfile = profile; body.gravityFactor = gravity;
                const bool legacy = profile == HavokMotionProfile::legacyOpenF4;
                const auto bytes = serializeHavokPhysicsSystem(source);
                const Layout layout(bytes);
                const auto properties = layout.pointers.at(0x20);
                const auto expectedFlags = (legacy ? 2u : 0u) | (gravity != 1 ? 0x20000000u : 0);
                require(layout.at<std::uint32_t>(properties + 4) == expectedFlags,
                    "Motion flags lost the selected profile or gravity override");
                require(layout.at<std::uint32_t>(properties + 36) == (legacy ? 0x3efb22d1u : 0x3efb22d2u)
                    && layout.at<std::uint32_t>(properties + 40) == (legacy ? 0x3b23d70au : 0x3b23d70bu)
                    && layout.at<std::uint32_t>(properties + 44) == (legacy ? 0x3b23d70au : 0x3b23d70bu)
                    && layout.at<std::uint32_t>(properties + 52) == (legacy ? 0u : 0x999b67ffu)
                    && layout.at<std::uint32_t>(properties + 56) == (legacy ? 0u : 0x06757304u)
                    && layout.at<std::uint32_t>(properties + 60) == (legacy ? 0u : 0x00000073u),
                    "Known motion/deactivation profile bytes changed");
                const auto decoded = loadEditableHavokPhysicsSystem(bytes);
                const auto& restored = decoded.bodies.front();
                require(restored.motionProfile == profile && restored.gravityFactor == gravity
                    && restored.linearDamping == body.linearDamping && restored.angularDamping == body.angularDamping
                    && restored.maxLinearVelocity == body.maxLinearVelocity && restored.maxAngularVelocity == body.maxAngularVelocity
                    && restored.mass == body.mass && restored.friction == body.friction && restored.restitution == body.restitution
                    && restored.collisionFilterInfo == 5, "Motion profile replaced authored body settings");
                require(serializeHavokPhysicsSystem(decoded) == bytes,
                    "Known legacy/native profile was not byte-preserved through editable reload");
                for (const auto offset : {4u, 52u, 56u, 60u})
                {
                    auto unknown = bytes;
                    unknown[layout.data + properties + offset] ^= std::byte{4};
                    require(!loadHavokPhysicsSystem(unknown).nativeRewriteBlockReason.empty(),
                        "Unknown motion metadata was treated as a known profile");
                    rejects([&] { (void)loadEditableHavokPhysicsSystem(unknown); });
                }
            }
        body.motionProfile = static_cast<HavokMotionProfile>(255);
        rejects([&] { (void)serializeHavokPhysicsSystem(source); });
    }

    void testMesh()
    {
        HavokPhysicsSystem source; source.bodies.resize(1); auto& mesh = source.bodies[0].triangleMesh.emplace();
        for (unsigned i = 0; i < 160; ++i)
        {
            const auto base = static_cast<unsigned>(mesh.vertices.size());
            mesh.vertices.insert(mesh.vertices.end(), {{float(i),0,0},{float(i)+0.8f,0,0},{float(i),1,0}});
            mesh.indices.insert(mesh.indices.end(), {base,base+1,base+2});
        }
        const auto bytes = serializeHavokPhysicsSystem(source);
        const auto decoded = loadHavokPhysicsSystem(bytes);
        require(decoded.bodies[0].triangleMesh->indices.size() == mesh.indices.size(), "Mesh lost triangles");
        Layout layout(bytes); const auto shape = layout.object("hknpCompressedMeshShape");
        const auto data = layout.object("hknpCompressedMeshShapeData");
        const auto properties = layout.pointers.at(shape + 0x20);
        require(layout.at<std::uint16_t>(layout.pointers.at(properties) + 8) == 0xf601,
            "Mesh uses the wrong Bethesda material property key");
        const auto material = layout.object("hknpBSMaterialProperties");
        require(layout.at<unsigned>(material + 0x18) == 1
            && layout.at<unsigned>(layout.pointers.at(material + 0x10) + 20) == source.bodies[0].materialId,
            "Mesh surface material was not preserved");
        const auto count = layout.at<unsigned>(data + 0x58);
        require(count > 1, "Oversized mesh was not split into sections");
        layout.checkTree(data + 0x10, true, count);
        const auto sections = layout.pointers.at(data + 0x50);
        for (unsigned i = 0; i < count; ++i)
        {
            const auto section = sections + i * 0x60;
            layout.checkTree(section, false, layout.at<unsigned>(section + 0x50) & 255);
        }
        const auto maxKey = layout.at<unsigned>(data + 0x48);
        require(layout.at<unsigned>(shape + 0x78) == (maxKey + 2) / 2
            && layout.at<unsigned>(shape + 0x90) == maxKey + 1,
            "Mesh query bitfields do not cover section-relative keys");
        require(layout.at<unsigned>(data + 0xc0) == 2, "Mesh SIMD fallback is missing");
        require(layout.at<unsigned>(layout.pointers.at(data + 0xb8) + 112 + 80) == 0xff7fffeeu,
            "Mesh SIMD nodes are truncated or have invalid empty bounds");
        require(!serializeHavokPhysicsSystem(decoded).empty(), "Authored uniform mesh cannot be re-exported");
        source.bodies[0].dynamic = true;
        rejects([&] { (void)serializeHavokPhysicsSystem(source); });
    }

    void testCompound()
    {
        HavokPhysicsSystem source; source.bodies.resize(1); auto& body = source.bodies[0];
        body.parts.push_back(box()); body.parts.push_back(box());
        body.parts[1].transform = glm::translate(glm::mat4(1), glm::vec3(4,0,0));
        body.capsules.push_back({glm::mat4(1), glm::vec3(1), {{0,0,0},{0,0,3},0.2f}});
        const auto bytes = serializeHavokPhysicsSystem(source);
        const auto result = loadHavokPhysicsSystem(bytes);
        require(result.bodies[0].parts.size() == 2 && result.bodies[0].capsules.size() == 1,
            "Compound children failed to round trip");
        Layout layout(bytes); const auto compound = layout.object("hknpDynamicCompoundShape");
        const auto data = layout.object("hknpDynamicCompoundShapeData");
        require(layout.pointers.at(compound + 0xc0) == data && layout.at<unsigned>(data + 0x18) == 7,
            "Native compound BVH was not attached");
        const auto nodes = layout.pointers.at(data + 0x10), instances = layout.pointers.at(compound + 0x60);
        require(layout.at<std::uint16_t>(data + 0x30) == 1 && layout.at<std::uint16_t>(data + 0x20) == 6,
            "Compound root/free nodes are invalid");
        for (unsigned i = 0; i < 3; ++i)
        {
            const auto leaf = layout.at<unsigned>(instances + i * 0x80 + 0x3c) & 0xffff;
            require(leaf > 0 && leaf < 6 && layout.at<std::uint16_t>(nodes + leaf * 32 + 28) == 0
                && layout.at<std::uint16_t>(nodes + leaf * 32 + 30) == i, "Compound instance leaf cache is invalid");
        }
        const auto capsule = layout.object("hknpCapsuleShape");
        require(layout.at<std::uint16_t>(capsule + 0x30) == 8
            && layout.at<std::uint16_t>(capsule + 0x44) == 6,
            "Capsule convex backing is missing");
        body.parts.front().scale = {1,2,1};
        rejects([&] { (void)serializeHavokPhysicsSystem(source); });
        body.parts.front().polytope.convexRadius = 0;
        const auto scaled = loadHavokPhysicsSystem(serializeHavokPhysicsSystem(source));
        glm::vec3 minimum(std::numeric_limits<float>::max()), maximum(std::numeric_limits<float>::lowest());
        for (const auto point : scaled.bodies[0].parts[0].polytope.vertices)
        {
            minimum = glm::min(minimum, point); maximum = glm::max(maximum, point);
        }
        require(glm::length(minimum - glm::vec3(-1,-2,-1)) < 0.0001f
            && glm::length(maximum - glm::vec3(1,2,1)) < 0.0001f,
            "Zero-radius convex nonuniform scale was not baked into native vertices");
        body.parts.front().scale = glm::vec3(1); body.parts.front().polytope.indices.pop_back();
        rejects([&] { (void)serializeHavokPhysicsSystem(source); });
        body.parts.front() = box();
        body.parts.front().polytope.vertices.front().x = std::numeric_limits<float>::infinity();
        rejects([&] { (void)serializeHavokPhysicsSystem(source); });
        body.parts.front() = box(); body.bodyIndex = 2;
        rejects([&] { (void)serializeHavokPhysicsSystem(source); });
    }
}

int main()
{
    try
    {
        testBodies(); testMotionProfiles(); testMesh(); testCompound(); testStructureNegativeControls();
        std::cout << "Havok native serialization tests passed (not a Fallout 4 runtime compatibility claim)\n";
        return 0;
    }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
