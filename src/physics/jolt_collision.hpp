#pragma once

#include "assets/havok_packfile.hpp"
#include "assets/coordinates.hpp"

#include <Jolt/Jolt.h>
#include <Jolt/Physics/Collision/Shape/Shape.h>

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include <glm/mat4x4.hpp>

namespace openf4::physics
{
    inline constexpr float BethesdaUnitsPerMeter = assets::bethesdaUnitsPerMeter;

    // Registers Jolt's allocator/factory once; needed before using Jolt geometry helpers directly.
    void ensureJoltRuntime();

    // Engine-space input in Bethesda units; result is Havok Z-up meters.
    [[nodiscard]] assets::HavokConvexPolytope makeHavokConvexHull(
        std::span<const glm::vec3> positions, float convexRadiusUnits = 0.0f);
    // Only geometry is captured; Fallout-specific body metadata comes from the authoring document.
    [[nodiscard]] assets::HavokRigidBody captureJoltCollisionShape(
        const JPH::Shape& shape, const assets::HavokRigidBody& metadata);
    void updateHavokMassProperties(assets::HavokRigidBody& body);

    [[nodiscard]] JPH::ShapeRefC makeJoltConvexHull(
        const assets::HavokConvexPolytope& polytope);
    [[nodiscard]] JPH::ShapeRefC makeJoltCollisionShape(
        const assets::HavokRigidBody& body);
    [[nodiscard]] JPH::ShapeRefC scaleJoltCollisionShape(
        const JPH::ShapeRefC& shape, const glm::vec3& scale);
    [[nodiscard]] glm::mat4 havokBodyLocalTransform(
        const assets::HavokRigidBody& body);

    struct BodyHandle
    {
        std::uint32_t value = 0xffffffffu;

        [[nodiscard]] explicit operator bool() const noexcept
        {
            return value != 0xffffffffu;
        }
    };

    struct RagdollHandle
    {
        std::uint32_t value = 0xffffffffu;

        [[nodiscard]] explicit operator bool() const noexcept
        {
            return value != 0xffffffffu;
        }
    };

    struct BodyState
    {
        glm::mat4 transform{1.0f};
        glm::vec3 linearVelocity{}; // Engine units per second; transforms are absolute.
        glm::vec3 angularVelocity{};
        bool active = true;
    };

    class JoltWorld
    {
    public:
        JoltWorld();
        ~JoltWorld();
        JoltWorld(const JoltWorld&) = delete;
        JoltWorld& operator=(const JoltWorld&) = delete;

        [[nodiscard]] BodyHandle addStaticBox(
            const glm::vec3& halfExtent, const glm::vec3& position);
        [[nodiscard]] BodyHandle addStaticHeightField(std::span<const float> heights,
            std::uint32_t side, float sampleSpacing, const glm::vec3& origin);
        [[nodiscard]] BodyHandle addStaticBody(const JPH::ShapeRefC& shape,
            const glm::vec3& position = glm::vec3(0.0f),
            const glm::quat& rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f));
        [[nodiscard]] BodyHandle addDynamicBody(const JPH::ShapeRefC& shape,
            const glm::vec3& position,
            const glm::quat& rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f),
            float mass = 0.0f, float friction = 0.5f, float restitution = 0.4f);
        // Position/rotation place the body frame; source shape coordinates remain local.
        [[nodiscard]] BodyHandle addCollisionBody(const assets::HavokRigidBody& body,
            const glm::vec3& position = glm::vec3(0.0f),
            const glm::quat& rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f),
            float scale = 1.0f);
        [[nodiscard]] RagdollHandle addRagdoll(const assets::HavokRagdoll& ragdoll,
            const glm::vec3& position,
            const glm::quat& rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f));
        void setOrigin(const glm::vec3& origin);
        void removeBody(BodyHandle body);
        void removeRagdoll(RagdollHandle ragdoll);
        void step(float deltaSeconds);
        [[nodiscard]] glm::mat4 bodyTransform(BodyHandle body) const;
        [[nodiscard]] BodyState bodyState(BodyHandle body) const;
        void restoreBodyState(BodyHandle body, const BodyState& state);
        [[nodiscard]] bool isBodyValid(BodyHandle body) const;
        [[nodiscard]] bool isBodyActive(BodyHandle body) const;
        // Fallout motion types: 0 fixed, 1 dynamic, 2 keyframed. Invalid/stale handles
        // and unsupported promotions return false. Switching preserves mass/inertia.
        [[nodiscard]] bool setMotionType(BodyHandle body, std::uint32_t motionType,
            bool allowActivate = true);
        // Engine-axis direction is normalized; magnitude is an impulse in kg*m/s.
        // A zero magnitude with a nonzero direction still wakes a dynamic body.
        [[nodiscard]] bool applyHavokImpulse(BodyHandle body,
            const glm::vec3& direction, float magnitude);
        [[nodiscard]] std::vector<glm::vec3> dynamicBodyPositions() const;
        [[nodiscard]] std::vector<glm::mat4> ragdollBodyTransforms(
            RagdollHandle ragdoll) const;

    private:
        struct Impl;
        std::unique_ptr<Impl> mImpl;
    };
}
