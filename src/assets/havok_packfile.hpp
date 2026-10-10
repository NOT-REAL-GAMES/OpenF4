#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <glm/mat4x4.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/vec3.hpp>

namespace openf4::assets
{
    struct HavokConvexPolytope
    {
        std::vector<glm::vec3> vertices;
        std::vector<std::uint32_t> indices;
        float convexRadius = 0.0f;
    };

    struct HavokConvexPart
    {
        glm::mat4 transform{1.0f};
        glm::vec3 scale{1.0f};
        HavokConvexPolytope polytope;
    };

    struct HavokCapsule
    {
        glm::vec3 pointA{};
        glm::vec3 pointB{};
        float radius = 0.0f;
    };

    struct HavokCapsulePart
    {
        glm::mat4 transform{1.0f};
        glm::vec3 scale{1.0f};
        HavokCapsule capsule;
    };

    struct HavokPrimitiveDataRun
    {
        std::uint32_t section = 0;
        std::uint16_t value = 0;
        std::uint8_t index = 0;
        std::uint8_t count = 0;
    };

    struct HavokTriangleMesh
    {
        std::vector<glm::vec3> vertices;
        std::vector<std::uint32_t> indices;
        std::vector<HavokPrimitiveDataRun> primitiveDataRuns;
        std::uint32_t sectionCount = 0;
    };

    enum class HavokMotionProfile : std::uint8_t
    {
        fallout4,
        legacyOpenF4
    };

    struct HavokRigidBody
    {
        std::uint32_t bodyIndex = 0;
        glm::vec3 position{};
        glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f};
        bool dynamic = false;
        float mass = 0.0f;
        float friction = 0.5f;
        float restitution = 0.4f;
        std::vector<HavokConvexPart> parts;
        std::vector<HavokCapsulePart> capsules;
        std::optional<HavokTriangleMesh> triangleMesh;
        std::uint32_t collisionFilterInfo = 1;
        std::uint32_t materialId = 0x064003d4;
        // Mass properties are body-local, in Havok meters and kg*m^2, not NIF units.
        glm::vec3 centerOfMass{};
        glm::vec3 inertiaDiagonal{};
        glm::quat inertiaOrientation{1.0f, 0.0f, 0.0f, 0.0f};
        float linearDamping = 0.1f;
        float angularDamping = 0.05f;
        float gravityFactor = 1.0f;
        float maxLinearVelocity = 104.375f;
        float maxAngularVelocity = 31.57f;
        // Keep earlier authored packages losslessly editable; new bodies use the verified native profile.
        HavokMotionProfile motionProfile = HavokMotionProfile::fallout4;
    };

    struct HavokPhysicsSystem
    {
        std::vector<HavokRigidBody> bodies;
        std::string nativeRewriteBlockReason;
    };

    struct HavokSkeletonBone
    {
        std::string name;
        std::int16_t parent = -1;
        glm::mat4 referencePose{1.0f};
        bool lockTranslation = false;
    };

    enum class HavokConstraintType
    {
        Ragdoll,
        LimitedHinge
    };

    struct HavokConstraint
    {
        HavokConstraintType type = HavokConstraintType::Ragdoll;
        std::uint32_t bodyA = 0;
        std::uint32_t bodyB = 0;
        std::uint32_t flags = 0;
        glm::mat4 localFrameA{1.0f};
        glm::mat4 localFrameB{1.0f};
        float minimumTwistAngle = 0.0f;
        float maximumTwistAngle = 0.0f;
        float minimumConeAngle = 0.0f;
        float maximumConeAngle = 0.0f;
        float minimumPlaneAngle = 0.0f;
        float maximumPlaneAngle = 0.0f;
        float maximumFrictionTorque = 0.0f;
    };

    struct HavokRagdoll
    {
        std::vector<HavokRigidBody> bodies;
        std::vector<HavokSkeletonBone> bones;
        std::vector<std::uint32_t> boneToBody;
        std::vector<HavokConstraint> constraints;
    };

    [[nodiscard]] HavokPhysicsSystem loadHavokPhysicsSystem(
        std::span<const std::byte> bytes);
    // Rejects any source bytes not reproduced by the writer, except bounded mass-frame float roundoff.
    [[nodiscard]] HavokPhysicsSystem loadEditableHavokPhysicsSystem(
        std::span<const std::byte> bytes);
    [[nodiscard]] std::vector<std::byte> serializeHavokPhysicsSystem(
        const HavokPhysicsSystem& system);
    [[nodiscard]] std::vector<HavokPhysicsSystem> loadNifPhysicsSystems(
        std::span<const std::byte> bytes);
    [[nodiscard]] HavokRagdoll loadHavokRagdoll(
        std::span<const std::byte> bytes);
    [[nodiscard]] std::vector<HavokRagdoll> loadNifRagdolls(
        std::span<const std::byte> bytes);
}
