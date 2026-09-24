#pragma once

#include "assets/havok_packfile.hpp"

#include <Jolt/Jolt.h>
#include <Jolt/Physics/Collision/Shape/Shape.h>

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include <glm/mat4x4.hpp>

namespace openf4::physics
{
    inline constexpr float BethesdaUnitsPerMeter = 69.99125f;

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
        [[nodiscard]] RagdollHandle addRagdoll(const assets::HavokRagdoll& ragdoll,
            const glm::vec3& position,
            const glm::quat& rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f));
        void setOrigin(const glm::vec3& origin);
        void removeBody(BodyHandle body);
        void removeRagdoll(RagdollHandle ragdoll);
        void step(float deltaSeconds);
        [[nodiscard]] glm::mat4 bodyTransform(BodyHandle body) const;
        [[nodiscard]] std::vector<glm::vec3> dynamicBodyPositions() const;
        [[nodiscard]] std::vector<glm::mat4> ragdollBodyTransforms(
            RagdollHandle ragdoll) const;

    private:
        struct Impl;
        std::unique_ptr<Impl> mImpl;
    };
}
