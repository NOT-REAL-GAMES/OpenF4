#include "physics/jolt_collision.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

#include <glm/ext/matrix_transform.hpp>

namespace
{
    using namespace openf4;
    void require(bool value, const char* message)
    {
        if (!value) throw std::runtime_error(message);
    }
    template<class F> void rejects(F operation)
    {
        try { operation(); }
        catch (const std::runtime_error&) { return; }
        throw std::runtime_error("Invalid body state was accepted");
    }
    assets::HavokRigidBody cube()
    {
        std::array<glm::vec3, 8> vertices;
        for (unsigned i = 0; i < vertices.size(); ++i)
            vertices[i] = glm::vec3(i & 1 ? 35 : -35, i & 2 ? 35 : -35, i & 4 ? 35 : -35);
        assets::HavokRigidBody body;
        body.parts.push_back({glm::mat4(1), glm::vec3(1), physics::makeHavokConvexHull(vertices)});
        body.dynamic = true;
        body.mass = 4;
        body.restitution = 0;
        physics::updateHavokMassProperties(body);
        return body;
    }
    void scaledBodyLandsOnTerrain()
    {
        physics::JoltWorld world;
        const glm::vec3 origin(-3 * 4096.0f, 0, 7 * 4096.0f);
        world.setOrigin(origin);
        std::array<float, 33 * 33> heights{};
        const auto ground = world.addStaticHeightField(heights, 33, 128, origin);
        auto source = cube();
        const auto body = world.addCollisionBody(source, origin + glm::vec3(2048, 500, 2048),
            glm::quat(1, 0, 0, 0), 2.0f);
        for (int i = 0; i < 240; ++i) world.step(1.0f / 60);
        const auto settled = glm::vec3(world.bodyTransform(body)[3]);
        require(std::abs(settled.y - 70.0f) < 1.5f, "Scaled native body did not rest on LAND");
        const auto beforeShift = world.bodyState(body);
        world.setOrigin(origin + glm::vec3(-32 * 4096.0f, 0, 32 * 4096.0f));
        require(glm::length(glm::vec3(world.bodyTransform(body)[3]) - settled) < 0.02f,
            "Origin shift moved the absolute body pose");
        world.removeBody(body);
        const auto restored = world.addCollisionBody(source, {}, glm::quat(1, 0, 0, 0), 2.0f);
        world.restoreBodyState(restored, beforeShift);
        require(glm::length(glm::vec3(world.bodyTransform(restored)[3]) - settled) < 0.02f,
            "Residency reload lost native body pose");
        world.removeBody(restored);
        world.removeBody(ground);
        // Authoritative metadata is not rewritten by reference scale or simulation.
        require(source.mass == 4 && source.parts.front().scale == glm::vec3(1),
            "Runtime mutated authored native metadata");
    }
    void suspendedVelocitySurvivesReload()
    {
        physics::JoltWorld world;
        auto source = cube();
        source.gravityFactor = source.linearDamping = source.angularDamping = 0;
        auto body = world.addCollisionBody(source);
        auto authored = world.bodyState(body);
        authored.transform = glm::translate(glm::mat4(1), glm::vec3(17, 300, -27))
            * glm::mat4_cast(glm::angleAxis(0.7f, glm::normalize(glm::vec3(1, 2, 3))));
        authored.linearVelocity = {70, 0, -35};
        authored.angularVelocity = {0, 0.2f, 0};
        world.restoreBodyState(body, authored);
        for (int i = 0; i < 10; ++i) world.step(1.0f / 60);
        const auto paused = world.bodyState(body);
        world.removeBody(body);
        world.setOrigin({100000, 0, -100000});
        body = world.addCollisionBody(source);
        world.restoreBodyState(body, paused);
        const auto resumed = world.bodyState(body);
        for (int axis = 0; axis < 3; ++axis)
            require(glm::length(glm::vec3(resumed.transform[axis] - paused.transform[axis])) < 0.0001f,
                "Residency reload lost native body rotation");
        require(glm::length(resumed.linearVelocity - paused.linearVelocity) < 0.001f
            && glm::length(resumed.angularVelocity - paused.angularVelocity) < 0.001f,
            "Residency reload lost native body velocity");
        // Match the runtime: recenter on the resident camera before simulation resumes.
        world.setOrigin({0, 0, 0});
        for (int i = 0; i < 60; ++i) world.step(1.0f / 60);
        const auto moved = world.bodyState(body);
        const auto error = glm::length(glm::vec3(moved.transform[3] - paused.transform[3])
            - paused.linearVelocity);
        if (error >= 0.2f)
            throw std::runtime_error("Reloaded prop movement error after recentering: "
                + std::to_string(error) + " units");
        source.dynamic = false;
        const auto fixed = world.addCollisionBody(source, {500, 1000, 500});
        const auto fixedBefore = world.bodyState(fixed);
        for (int i = 0; i < 60; ++i) world.step(1.0f / 60);
        require(world.bodyState(fixed).transform == fixedBefore.transform,
            "STAT override simulated a native dynamic body");
    }
    void rejectsInvalidState()
    {
        physics::JoltWorld world;
        const auto source = cube();
        const auto body = world.addCollisionBody(source);
        auto state = world.bodyState(body);
        state.transform[0][0] = 2;
        rejects([&] { world.restoreBodyState(body, state); });
        state = world.bodyState(body);
        state.linearVelocity.x = std::numeric_limits<float>::infinity();
        rejects([&] { world.restoreBodyState(body, state); });
        rejects([&] { (void)world.addCollisionBody(source, {}, glm::quat(1, 0, 0, 0), -1); });
        auto triangle = source;
        triangle.triangleMesh.emplace();
        triangle.triangleMesh->vertices = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
        triangle.triangleMesh->indices = {0, 1, 2};
        rejects([&] { (void)world.addCollisionBody(triangle); });
        world.removeBody(body);
        rejects([&] { (void)world.bodyState(body); });
    }
    void motionChangesAndZeroImpulseWake()
    {
        physics::JoltWorld world;
        (void)world.addStaticBox({1000, 35, 1000}, {0, -35, 0});
        const auto body = world.addCollisionBody(cube(), {0, 500, 0});
        require(world.isBodyValid(body) && world.isBodyActive(body), "New dynamic body is not active");
        require(world.setMotionType(body, 0) && !world.isBodyActive(body)
            && world.dynamicBodyPositions().empty(), "Fixed transition did not stop tracking dynamics");
        require(!world.applyHavokImpulse(body, {0, 1, 0}, 1), "Fixed body accepted an impulse");
        require(world.setMotionType(body, 1, false) && !world.isBodyActive(body)
            && world.dynamicBodyPositions().size() == 1, "Inactive dynamic transition woke the body");
        const auto suspended = world.bodyState(body);
        for (int i = 0; i < 30; ++i) world.step(1.0f / 60);
        require(world.bodyState(body).transform == suspended.transform, "Inactive dynamic body moved");
        require(world.applyHavokImpulse(body, {0, 100, 0}, 0) && world.isBodyActive(body),
            "Zero impulse did not wake a dynamic body");
        require(world.bodyState(body).linearVelocity == glm::vec3(0), "Zero impulse added velocity");
        for (int i = 0; i < 60; ++i) world.step(1.0f / 60);
        require(world.bodyTransform(body)[3].y < suspended.transform[3].y - 100,
            "Body did not fall after zero impulse wake");
        require(world.setMotionType(body, 0) && world.setMotionType(body, 1, false)
            && world.setMotionType(body, 1, true) && world.isBodyActive(body),
            "Unchanged dynamic motion type did not honor activation");
        require(world.setMotionType(body, 0) && world.setMotionType(body, 2)
            && world.dynamicBodyPositions().empty(), "Keyframed transition failed");
        const auto held = world.bodyState(body);
        require(!world.applyHavokImpulse(body, {0, 1, 0}, 1), "Keyframed body accepted an impulse");
        for (int i = 0; i < 60; ++i) world.step(1.0f / 60);
        require(world.bodyState(body).transform == held.transform, "Keyframed body fell under gravity");
        require(world.setMotionType(body, 1), "Release from keyframed motion failed");
        for (int i = 0; i < 30; ++i) world.step(1.0f / 60);
        require(world.bodyTransform(body)[3].y < held.transform[3].y - 20,
            "Released keyframed body did not fall");
        for (int i = 0; i < 90; ++i) world.step(1.0f / 60);
        require(std::abs(world.bodyTransform(body)[3].y - 35) < 1.5f,
            "Released body lost collision with the static layer");
    }
    void impulsesPreserveAuthoredMass()
    {
        physics::JoltWorld world;
        auto source = cube();
        source.mass = 7;
        source.inertiaDiagonal = {2, 3, 4};
        source.centerOfMass = {0.1f, 0.2f, 0.3f};
        source.gravityFactor = source.linearDamping = source.angularDamping = 0;
        const auto dynamic = world.addCollisionBody(source);
        require(world.setMotionType(dynamic, 0) && world.setMotionType(dynamic, 2)
            && world.setMotionType(dynamic, 1), "Dynamic round trip failed");
        source.dynamic = false;
        const auto fixed = world.addCollisionBody(source, {1000, 0, 0});
        const auto fixedPose = world.bodyTransform(fixed);
        require(world.setMotionType(fixed, 1), "Valid authored fixed body could not become dynamic");
        require(world.bodyTransform(fixed) == fixedPose, "Promotion changed the authored body frame");
        for (const auto body : {dynamic, fixed})
        {
            require(world.applyHavokImpulse(body, {100, 0, 0}, 14), "Valid impulse was rejected");
            const auto state = world.bodyState(body);
            require(glm::length(state.linearVelocity - glm::vec3(2 * physics::BethesdaUnitsPerMeter, 0, 0))
                < 0.001f, "Impulse normalization, units or original mass was lost");
            require(state.angularVelocity == glm::vec3(0), "Center-of-mass impulse added angular velocity");
        }
    }
    void activationSurvivesResidencyReload()
    {
        physics::JoltWorld world;
        const auto source = cube();
        auto body = world.addCollisionBody(source, {0, 500, 0});
        require(world.setMotionType(body, 0) && world.setMotionType(body, 1, false),
            "Cannot prepare inactive dynamic body");
        auto state = world.bodyState(body);
        require(!state.active, "Captured state lost inactivity");
        world.removeBody(body);
        body = world.addCollisionBody(source);
        world.restoreBodyState(body, state);
        require(!world.isBodyActive(body), "Residency reload activated a sleeping body");
        for (int i = 0; i < 60; ++i) world.step(1.0f / 60);
        require(world.bodyState(body).transform == state.transform, "Sleeping restored body moved");
        require(world.applyHavokImpulse(body, {0, 1, 0}, 0), "Cannot activate restored body");
        world.step(1.0f / 60);
        state = world.bodyState(body);
        require(state.active, "Captured state lost activity");
        world.removeBody(body);
        body = world.addCollisionBody(source);
        world.restoreBodyState(body, state);
        require(world.isBodyActive(body), "Residency reload deactivated an active body");
        for (int i = 0; i < 30; ++i) world.step(1.0f / 60);
        require(world.bodyTransform(body)[3].y < state.transform[3].y - 20,
            "Active restored body did not resume falling");
        require(world.setMotionType(body, 2), "Cannot prepare keyframed reload");
        state = world.bodyState(body);
        world.removeBody(body);
        body = world.addCollisionBody(source);
        require(world.setMotionType(body, 2), "Cannot recreate keyframed motion");
        world.restoreBodyState(body, state);
        require(world.isBodyActive(body) == state.active
            && glm::length(world.bodyState(body).linearVelocity - state.linearVelocity) < 0.001f,
            "Keyframed residency lost activity or velocity");
    }
    void motionCommandsRejectInvalidBodiesAndInputs()
    {
        physics::JoltWorld world;
        const auto body = world.addCollisionBody(cube());
        const auto before = world.bodyState(body);
        require(!world.setMotionType(body, 3) && !world.setMotionType(body, 0xffffffffu),
            "Unsupported motion type was accepted");
        const auto infinity = std::numeric_limits<float>::infinity();
        const auto nan = std::numeric_limits<float>::quiet_NaN();
        require(!world.applyHavokImpulse(body, {}, 0)
            && !world.applyHavokImpulse(body, {1, 0, 0}, -1)
            && !world.applyHavokImpulse(body, {1, 0, 0}, infinity)
            && !world.applyHavokImpulse(body, {1, 0, 0}, nan)
            && !world.applyHavokImpulse(body, {infinity, 0, 0}, 1)
            && !world.applyHavokImpulse(body, {0, nan, 0}, 1), "Invalid impulse was accepted");
        require(world.bodyState(body).transform == before.transform
            && world.bodyState(body).linearVelocity == before.linearVelocity,
            "Rejected command mutated the body");
        world.removeBody(body);
        const auto replacement = world.addCollisionBody(cube());
        require(replacement.value != body.value && world.isBodyValid(replacement),
            "Replacement body reused a stale handle");
        for (const auto stale : {body, physics::BodyHandle{}, physics::BodyHandle{0xfffffffeu}})
            require(!world.isBodyValid(stale) && !world.isBodyActive(stale)
                && !world.setMotionType(stale, 1) && !world.applyHavokImpulse(stale, {0, 1, 0}, 0),
                "Stale or invalid body accepted a command");
        const auto scenery = world.addStaticBox({35, 35, 35}, {1000, 0, 0});
        require(!world.setMotionType(scenery, 1) && !world.setMotionType(scenery, 2),
            "Static scenery without authored mass was promoted");
        auto triangle = cube();
        triangle.dynamic = false;
        triangle.parts.clear();
        triangle.triangleMesh.emplace();
        triangle.triangleMesh->vertices = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
        triangle.triangleMesh->indices = {0, 1, 2};
        const auto mesh = world.addCollisionBody(triangle, {2000, 0, 0});
        require(world.setMotionType(mesh, 0) && !world.setMotionType(mesh, 1)
            && !world.setMotionType(mesh, 2), "Triangle mesh was promoted to moving collision");
        rejects([&] { (void)world.addDynamicBody(physics::makeJoltCollisionShape(triangle), {}); });
    }
    void motionChangesPreserveInertiaAndCenterOfMass()
    {
        physics::JoltWorld world;
        (void)world.addStaticBox({4000, 35, 1000}, {1000, -35, 0});
        auto source = cube();
        source.mass = 7;
        source.inertiaDiagonal = {2, 3, 4};
        source.inertiaOrientation = glm::angleAxis(0.6f, glm::normalize(glm::vec3(1, 2, 3)));
        source.centerOfMass = {0.1f, 0.2f, 0.3f};
        const auto rotation = glm::angleAxis(0.4f, glm::normalize(glm::vec3(3, 2, 1)));
        const auto original = world.addCollisionBody(source, {0, 150, 0}, rotation);
        const auto switched = world.addCollisionBody(source, {1000, 150, 0}, rotation);
        require(world.setMotionType(switched, 0) && world.setMotionType(switched, 1),
            "Inertia comparison motion switch failed");
        source.dynamic = false;
        const auto promoted = world.addCollisionBody(source, {2000, 150, 0}, rotation);
        require(world.setMotionType(promoted, 1), "Inertia comparison promotion failed");
        for (int i = 0; i < 45; ++i) world.step(1.0f / 60);
        const auto expected = world.bodyState(original);
        require(glm::length(expected.angularVelocity) > 0.01f, "Inertia comparison did not rotate the body");
        for (const auto body : {switched, promoted})
        {
            const auto actual = world.bodyState(body);
            require(glm::length(actual.angularVelocity - expected.angularVelocity) < 0.005f
                && glm::length(actual.linearVelocity - expected.linearVelocity) < 0.1f,
                "Motion transition changed authored inertia or center of mass");
        }
    }
}

int main()
{
    try
    {
        scaledBodyLandsOnTerrain();
        suspendedVelocitySurvivesReload();
        rejectsInvalidState();
        motionChangesAndZeroImpulseWake();
        impulsesPreserveAuthoredMass();
        activationSurvivesResidencyReload();
        motionCommandsRejectInvalidBodiesAndInputs();
        motionChangesPreserveInertiaAndCenterOfMass();
        std::cout << "Project object physics lifecycle passed\n";
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
