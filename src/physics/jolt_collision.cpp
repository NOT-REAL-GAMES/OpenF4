#include "physics/jolt_collision.hpp"

#include <Jolt/Jolt.h>
#include <Jolt/Core/Factory.h>
#include <Jolt/Core/JobSystemThreadPool.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Body/BodyLock.h>
#include <Jolt/Physics/Body/MassProperties.h>
#include <Jolt/Physics/Collision/BroadPhase/BroadPhaseLayerInterfaceTable.h>
#include <Jolt/Physics/Collision/BroadPhase/ObjectVsBroadPhaseLayerFilterTable.h>
#include <Jolt/Physics/Collision/ObjectLayerPairFilterTable.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/CapsuleShape.h>
#include <Jolt/Physics/Collision/Shape/HeightFieldShape.h>
#include <Jolt/Physics/Collision/Shape/ConvexHullShape.h>
#include <Jolt/Physics/Collision/Shape/MeshShape.h>
#include <Jolt/Physics/Collision/Shape/OffsetCenterOfMassShape.h>
#include <Jolt/Physics/Collision/Shape/ScaledShape.h>
#include <Jolt/Physics/Collision/Shape/StaticCompoundShape.h>
#include <Jolt/Physics/Constraints/HingeConstraint.h>
#include <Jolt/Physics/Constraints/SwingTwistConstraint.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/Ragdoll/Ragdoll.h>
#include <Jolt/RegisterTypes.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <glm/ext/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/quaternion.hpp>

namespace openf4::physics
{
    namespace
    {
        constexpr JPH::ObjectLayer staticLayer = 0;
        constexpr JPH::ObjectLayer movingLayer = 1;

        struct Runtime
        {
            Runtime()
            {
                JPH::RegisterDefaultAllocator();
                JPH::Factory::sInstance = new JPH::Factory();
                JPH::RegisterTypes();
            }

            ~Runtime()
            {
                JPH::UnregisterTypes();
                delete JPH::Factory::sInstance;
                JPH::Factory::sInstance = nullptr;
            }
        };

        Runtime& runtime()
        {
            static Runtime value;
            return value;
        }

        JPH::RVec3 toJoltPosition(const glm::vec3& value)
        {
            return {value.x / BethesdaUnitsPerMeter, value.y / BethesdaUnitsPerMeter,
                value.z / BethesdaUnitsPerMeter};
        }

        JPH::Quat toJolt(const glm::quat& value)
        {
            return {value.x, value.y, value.z, value.w};
        }

        glm::mat4 toEngineAxes(const glm::mat4& value)
        {
            return assets::swapBethesdaAxes(value);
        }

        glm::vec3 toEngineAxes(const glm::vec3& value)
        {
            return assets::swapBethesdaAxes(value);
        }

        glm::quat toEngineAxes(const glm::quat& value)
        {
            const glm::mat4 converted = toEngineAxes(glm::mat4_cast(glm::normalize(value)));
            return glm::normalize(glm::quat_cast(glm::mat3(converted)));
        }

        JPH::Vec3 toJoltDirection(const glm::vec3& value)
        {
            const float length = glm::length(value);
            if (!std::isfinite(length) || length <= 0.00001f)
                throw std::runtime_error("Havok constraint has a degenerate axis");
            const glm::vec3 normalized = value / length;
            return {normalized.x, normalized.y, normalized.z};
        }

        JPH::RVec3 toJoltMeters(const glm::vec3& value)
        {
            return {value.x, value.y, value.z};
        }

        glm::mat4 bodyTransformMeters(const assets::HavokRigidBody& body,
            const glm::vec3& spawnPositionMeters, const glm::quat& spawnRotation)
        {
            const glm::vec3 localPosition = toEngineAxes(body.position);
            const glm::quat localRotation = toEngineAxes(body.rotation);
            return glm::translate(glm::mat4(1.0f),
                       spawnPositionMeters + spawnRotation * localPosition)
                * glm::mat4_cast(glm::normalize(spawnRotation * localRotation));
        }

        void applyMass(JPH::BodyCreationSettings& settings,
            const JPH::Shape* shape, float mass)
        {
            if (!std::isfinite(mass) || mass <= 0.0f)
                throw std::runtime_error("Havok ragdoll body has invalid mass");
            const JPH::MassProperties inferred = shape->GetMassProperties();
            if (std::isfinite(inferred.mMass) && inferred.mMass > 0.0f)
            {
                settings.mOverrideMassProperties =
                    JPH::EOverrideMassProperties::CalculateInertia;
                settings.mMassPropertiesOverride.mMass = mass;
                return;
            }
            JPH::MassProperties fallback;
            fallback.SetMassAndInertiaOfSolidBox(JPH::Vec3::sMax(
                shape->GetLocalBounds().GetSize(), JPH::Vec3::sReplicate(0.01f)), 1.0f);
            fallback.ScaleToMass(mass);
            settings.mOverrideMassProperties =
                JPH::EOverrideMassProperties::MassAndInertiaProvided;
            settings.mMassPropertiesOverride = fallback;
        }

        bool hasAuthoredMassProperties(const assets::HavokRigidBody& body)
        {
            return std::isfinite(body.mass) && body.mass > 0.0f
                && std::isfinite(body.inertiaDiagonal.x) && body.inertiaDiagonal.x > 0.0f
                && std::isfinite(body.inertiaDiagonal.y) && body.inertiaDiagonal.y > 0.0f
                && std::isfinite(body.inertiaDiagonal.z) && body.inertiaDiagonal.z > 0.0f
                && std::isfinite(glm::dot(body.inertiaOrientation, body.inertiaOrientation))
                && glm::dot(body.inertiaOrientation, body.inertiaOrientation) >= 1e-8f;
        }

        bool identity(const glm::mat4& value)
        {
            for (glm::length_t column = 0; column < 4; ++column)
                for (glm::length_t row = 0; row < 4; ++row)
                    if (std::abs(value[column][row] - (column == row ? 1.0f : 0.0f)) > 0.00001f)
                        return false;
            return true;
        }

        void validateChildTransform(const glm::mat4& transform, const glm::vec3& scale)
        {
            for (glm::length_t column = 0; column < 4; ++column)
                for (glm::length_t row = 0; row < 4; ++row)
                    if (!std::isfinite(transform[column][row]))
                        throw std::runtime_error("Havok child transform is not finite");
            for (glm::length_t axis = 0; axis < 3; ++axis)
                if (!std::isfinite(scale[axis]) || scale[axis] <= 0.0f
                    || std::abs(transform[axis][3]) > 0.0001f)
                    throw std::runtime_error("Havok child scale or affine transform is invalid");
            const glm::mat3 basis(transform);
            const glm::mat3 gram = glm::transpose(basis) * basis;
            for (glm::length_t column = 0; column < 3; ++column)
                for (glm::length_t row = 0; row < 3; ++row)
                    if (std::abs(gram[column][row] - (column == row ? 1.0f : 0.0f)) > 0.001f)
                        throw std::runtime_error("Havok child transform contains scale or shear");
            if (std::abs(transform[3][3] - 1.0f) > 0.0001f || glm::determinant(basis) <= 0.0f)
                throw std::runtime_error("Havok child transform is reflected or non-affine");
        }

        JPH::ShapeRefC makeJoltTriangleMesh(const assets::HavokTriangleMesh& mesh)
        {
            if (mesh.vertices.empty() || mesh.indices.empty() || mesh.indices.size() % 3 != 0)
                throw std::runtime_error("Havok triangle mesh is empty or malformed");

            JPH::VertexList vertices;
            vertices.reserve(mesh.vertices.size());
            for (const glm::vec3& vertex : mesh.vertices)
            {
                if (!std::isfinite(vertex.x) || !std::isfinite(vertex.y)
                    || !std::isfinite(vertex.z))
                    throw std::runtime_error("Invalid Havok triangle mesh vertex");
                vertices.emplace_back(vertex.x, vertex.z, vertex.y);
            }
            JPH::IndexedTriangleList triangles;
            triangles.reserve(mesh.indices.size() / 3);
            for (std::size_t index = 0; index < mesh.indices.size(); index += 3)
            {
                const auto a = mesh.indices[index];
                const auto b = mesh.indices[index + 1];
                const auto c = mesh.indices[index + 2];
                if (a >= vertices.size() || b >= vertices.size() || c >= vertices.size())
                    throw std::runtime_error("Havok triangle mesh index is out of bounds");
                // Swapping Bethesda Y/Z reflects the mesh, so reverse winding for Jolt.
                triangles.emplace_back(a, c, b, 0);
            }
            const auto result = JPH::MeshShapeSettings(
                std::move(vertices), std::move(triangles)).Create();
            if (!result.IsValid())
                throw std::runtime_error(std::string("Jolt rejected Havok triangle mesh: ")
                    + result.GetError().c_str());
            return result.Get();
        }
    }

    void ensureJoltRuntime() { (void)runtime(); }

    JPH::ShapeRefC makeJoltConvexHull(const assets::HavokConvexPolytope& polytope)
    {
        (void)runtime();
        if (polytope.vertices.size() < 4 || !std::isfinite(polytope.convexRadius)
            || polytope.convexRadius < 0.0f)
            throw std::runtime_error("Invalid Havok convex hull");

        JPH::Array<JPH::Vec3> points;
        points.reserve(polytope.vertices.size());
        for (const auto& point : polytope.vertices)
        {
            if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z))
                throw std::runtime_error("Invalid Havok convex hull vertex");
            points.emplace_back(point.x, point.z, point.y);
        }
        const auto result = JPH::ConvexHullShapeSettings(points, polytope.convexRadius).Create();
        if (!result.IsValid())
            throw std::runtime_error(std::string("Jolt rejected Havok convex hull: ")
                + result.GetError().c_str());
        return result.Get();
    }

    JPH::ShapeRefC makeJoltCollisionShape(const assets::HavokRigidBody& body)
    {
        (void)runtime();
        const std::size_t shapeCount = body.parts.size() + body.capsules.size()
            + (body.triangleMesh ? 1u : 0u);
        if (shapeCount == 0)
            throw std::runtime_error("Havok body has no collision shapes");
        if (body.dynamic && body.triangleMesh)
            throw std::runtime_error("Dynamic Havok triangle meshes are unsupported");
        for (const auto& part : body.parts)
            validateChildTransform(part.transform, part.scale);
        for (const auto& part : body.capsules)
            validateChildTransform(part.transform, part.scale);
        if (shapeCount == 1 && body.triangleMesh)
            return makeJoltTriangleMesh(*body.triangleMesh);
        if (shapeCount == 1 && body.parts.size() == 1
            && identity(body.parts.front().transform)
            && glm::all(glm::epsilonEqual(body.parts.front().scale, glm::vec3(1.0f), 0.00001f)))
            return makeJoltConvexHull(body.parts.front().polytope);

        JPH::StaticCompoundShapeSettings compound;
        JPH::ShapeRefC meshChild;
        if (body.triangleMesh)
        {
            meshChild = makeJoltTriangleMesh(*body.triangleMesh);
            compound.AddShape(JPH::Vec3::sZero(), JPH::Quat::sIdentity(), meshChild.GetPtr());
        }
        for (const auto& part : body.parts)
        {
            JPH::ShapeRefC child = makeJoltConvexHull(part.polytope);
            const glm::vec3 scale(part.scale.x, part.scale.z, part.scale.y);
            if (!glm::all(glm::epsilonEqual(scale, glm::vec3(1.0f), 0.00001f)))
            {
                const auto scaled = JPH::ScaledShapeSettings(
                    child.GetPtr(), JPH::Vec3(scale.x, scale.y, scale.z)).Create();
                if (!scaled.IsValid())
                    throw std::runtime_error(std::string("Jolt rejected Havok child scale: ")
                        + scaled.GetError().c_str());
                child = scaled.Get();
            }
            const glm::mat4 transform = toEngineAxes(part.transform);
            const glm::vec3 position(transform[3]);
            const glm::quat rotation = glm::normalize(glm::quat_cast(glm::mat3(transform)));
            compound.AddShape(JPH::Vec3(position.x, position.y, position.z),
                toJolt(rotation), child.GetPtr());
        }
        for (const auto& part : body.capsules)
        {
            if (!std::isfinite(part.capsule.radius) || part.capsule.radius <= 0.0f)
                throw std::runtime_error("Havok capsule radius must be finite and positive");
            if (!std::isfinite(part.scale.x) || !std::isfinite(part.scale.y)
                || !std::isfinite(part.scale.z)
                || !glm::all(glm::greaterThan(part.scale, glm::vec3(0.0f))))
                throw std::runtime_error("Havok capsule scale must be finite and positive");
            if (!glm::all(glm::epsilonEqual(
                    part.scale, glm::vec3(part.scale.x), 0.00001f)))
                throw std::runtime_error("Non-uniform Havok capsule scale is unsupported");

            const glm::vec3 pointA = toEngineAxes(part.capsule.pointA) * part.scale.x;
            const glm::vec3 pointB = toEngineAxes(part.capsule.pointB) * part.scale.x;
            const glm::vec3 axis = pointB - pointA;
            const float length = glm::length(axis);
            if (!std::isfinite(length))
                throw std::runtime_error("Havok capsule has invalid endpoints");
            const auto capsuleResult = JPH::CapsuleShapeSettings(
                length * 0.5f, part.capsule.radius * part.scale.x).Create();
            if (!capsuleResult.IsValid())
                throw std::runtime_error(std::string("Jolt rejected Havok capsule: ")
                    + capsuleResult.GetError().c_str());

            const glm::mat4 transform = toEngineAxes(part.transform);
            const glm::vec3 position = glm::vec3(transform
                * glm::vec4((pointA + pointB) * 0.5f, 1.0f));
            const glm::quat parentRotation = glm::normalize(
                glm::quat_cast(glm::mat3(transform)));
            const JPH::Quat capsuleRotation = length > 0.00001f
                ? JPH::Quat::sFromTo(JPH::Vec3::sAxisY(),
                    JPH::Vec3(axis.x, axis.y, axis.z).Normalized())
                : JPH::Quat::sIdentity();
            compound.AddShape(JPH::Vec3(position.x, position.y, position.z),
                toJolt(parentRotation) * capsuleRotation, capsuleResult.Get().GetPtr());
        }
        const auto result = compound.Create();
        if (!result.IsValid())
            throw std::runtime_error(std::string("Jolt rejected Havok compound shape: ")
                + result.GetError().c_str());
        return result.Get();
    }

    JPH::ShapeRefC scaleJoltCollisionShape(
        const JPH::ShapeRefC& shape, const glm::vec3& scale)
    {
        if (shape == nullptr || !std::isfinite(scale.x) || !std::isfinite(scale.y)
            || !std::isfinite(scale.z) || glm::any(glm::lessThanEqual(scale, glm::vec3(0.0f))))
            throw std::runtime_error("Jolt shape scale must be finite and positive");
        if (glm::all(glm::epsilonEqual(scale, glm::vec3(1.0f), 0.00001f)))
            return shape;
        const auto result = JPH::ScaledShapeSettings(
            shape.GetPtr(), JPH::Vec3(scale.x, scale.y, scale.z)).Create();
        if (!result.IsValid())
            throw std::runtime_error(std::string("Jolt rejected reference scale: ")
                + result.GetError().c_str());
        return result.Get();
    }

    glm::mat4 havokBodyLocalTransform(const assets::HavokRigidBody& body)
    {
        if (!std::isfinite(body.position.x) || !std::isfinite(body.position.y)
            || !std::isfinite(body.position.z) || !std::isfinite(glm::dot(body.rotation, body.rotation))
            || glm::dot(body.rotation, body.rotation) < 1e-8f)
            throw std::runtime_error("Havok body transform is invalid");
        const glm::quat rotation = toEngineAxes(body.rotation);
        const glm::vec3 position = toEngineAxes(body.position);
        return glm::translate(glm::mat4(1.0f), position * BethesdaUnitsPerMeter)
            * glm::mat4_cast(rotation);
    }

    struct JoltWorld::Impl
    {
        JPH::BroadPhaseLayerInterfaceTable broadPhase{2, 2};
        JPH::ObjectLayerPairFilterTable layerPairs{2};
        std::unique_ptr<JPH::ObjectVsBroadPhaseLayerFilterTable> objectVsBroadPhase;
        JPH::TempAllocatorImpl temporaryMemory{32 * 1024 * 1024};
        JPH::JobSystemThreadPool jobs{JPH::cMaxPhysicsJobs, JPH::cMaxPhysicsBarriers,
            std::max(1, static_cast<int>(std::thread::hardware_concurrency()) - 1)};
        JPH::PhysicsSystem system;
        std::unordered_set<std::uint32_t> bodies;
        std::unordered_set<std::uint32_t> dynamicBodies;
        std::unordered_map<std::uint32_t, JPH::Ref<JPH::Ragdoll>> ragdolls;
        std::uint32_t nextRagdollHandle = 1;
        JPH::CollisionGroup::GroupID nextCollisionGroup = 1;
        std::thread::id ownerThread = std::this_thread::get_id();
        glm::vec3 origin{};
        float accumulator = 0.0f;

        Impl()
        {
            broadPhase.MapObjectToBroadPhaseLayer(staticLayer, JPH::BroadPhaseLayer(0));
            broadPhase.MapObjectToBroadPhaseLayer(movingLayer, JPH::BroadPhaseLayer(1));
            layerPairs.EnableCollision(staticLayer, movingLayer);
            layerPairs.EnableCollision(movingLayer, movingLayer);
            objectVsBroadPhase = std::make_unique<JPH::ObjectVsBroadPhaseLayerFilterTable>(
                broadPhase, 2, layerPairs, 2);
            system.Init(65'536, 0, 65'536, 16'384,
                broadPhase, *objectVsBroadPhase, layerPairs);
            system.SetGravity({0.0f, -9.81f, 0.0f});
        }

        ~Impl()
        {
            for (auto& [handle, ragdoll] : ragdolls)
                ragdoll->RemoveFromPhysicsSystem();
            ragdolls.clear();
            auto& bodyInterface = system.GetBodyInterface();
            for (const std::uint32_t value : bodies)
            {
                const JPH::BodyID body(value);
                bodyInterface.RemoveBody(body);
                bodyInterface.DestroyBody(body);
            }
        }

        void requireOwnerThread() const
        {
            if (std::this_thread::get_id() != ownerThread)
                throw std::runtime_error("JoltWorld must be accessed from its owning thread");
        }

        BodyHandle addBody(const JPH::Shape* shape, const glm::vec3& position,
            const glm::quat& rotation, JPH::EMotionType motion, JPH::ObjectLayer layer,
            float mass = 0.0f, float friction = 0.7f, float restitution = 0.0f,
            const assets::HavokRigidBody* source = nullptr)
        {
            if (motion != JPH::EMotionType::Static && shape->MustBeStatic())
                throw std::runtime_error("Jolt moving body requires a dynamic-compatible shape");
            JPH::BodyCreationSettings settings(
                shape, toJoltPosition(position - origin), toJolt(rotation), motion, layer);
            settings.mFriction = friction;
            settings.mRestitution = restitution;
            // Keep authored properties available when an explicitly fixed native prop
            // is later made dynamic. Ordinary scenery stays static-only.
            settings.mAllowDynamicOrKinematic = !shape->MustBeStatic()
                && source && hasAuthoredMassProperties(*source);
            if (settings.HasMassProperties())
            {
                if (shape->GetInnerRadius() > 0.0f)
                    settings.mMotionQuality = JPH::EMotionQuality::LinearCast;
                if (mass > 0.0f)
                    applyMass(settings, shape, mass);
                if (source)
                {
                    settings.mLinearDamping = source->linearDamping;
                    settings.mAngularDamping = source->angularDamping;
                    settings.mGravityFactor = source->gravityFactor;
                    settings.mMaxLinearVelocity = source->maxLinearVelocity;
                    settings.mMaxAngularVelocity = source->maxAngularVelocity;
                    if (glm::all(glm::greaterThan(source->inertiaDiagonal, glm::vec3(0))))
                    {
                        const auto diagonal = toEngineAxes(source->inertiaDiagonal);
                        const auto basis = JPH::Mat44::sRotation(toJolt(toEngineAxes(source->inertiaOrientation)));
                        settings.mOverrideMassProperties = JPH::EOverrideMassProperties::MassAndInertiaProvided;
                        settings.mMassPropertiesOverride.mMass = mass;
                        settings.mMassPropertiesOverride.mInertia = basis * JPH::Mat44::sScale(
                            {diagonal.x, diagonal.y, diagonal.z}) * basis.Transposed3x3();
                    }
                }
            }
            const JPH::BodyID body = system.GetBodyInterface().CreateAndAddBody(
                settings, motion == JPH::EMotionType::Dynamic
                    ? JPH::EActivation::Activate : JPH::EActivation::DontActivate);
            if (body.IsInvalid())
                throw std::runtime_error("Jolt could not create a body");
            const std::uint32_t handle = body.GetIndexAndSequenceNumber();
            bodies.insert(handle);
            if (motion == JPH::EMotionType::Dynamic)
                dynamicBodies.insert(handle);
            return {handle};
        }
    };

    JoltWorld::JoltWorld()
    {
        (void)runtime();
        mImpl = std::make_unique<Impl>();
    }

    JoltWorld::~JoltWorld() = default;

    BodyHandle JoltWorld::addStaticBox(
        const glm::vec3& halfExtent, const glm::vec3& position)
    {
        mImpl->requireOwnerThread();
        if (halfExtent.x <= 0.0f || halfExtent.y <= 0.0f || halfExtent.z <= 0.0f)
            throw std::runtime_error("Jolt box half extent must be positive");
        const glm::vec3 meters = halfExtent / BethesdaUnitsPerMeter;
        const auto result = JPH::BoxShapeSettings(
            {meters.x, meters.y, meters.z}).Create();
        if (!result.IsValid())
            throw std::runtime_error(std::string("Jolt rejected box: ") + result.GetError().c_str());
        return mImpl->addBody(result.Get(), position, glm::quat(1.0f, 0.0f, 0.0f, 0.0f),
            JPH::EMotionType::Static, staticLayer);
    }

    BodyHandle JoltWorld::addStaticHeightField(std::span<const float> heights,
        std::uint32_t side, float sampleSpacing, const glm::vec3& origin)
    {
        mImpl->requireOwnerThread();
        if (side < 2 || heights.size() != static_cast<std::size_t>(side) * side
            || !std::isfinite(sampleSpacing) || sampleSpacing <= 0.0f
            || std::ranges::any_of(heights, [](float height) { return !std::isfinite(height); }))
            throw std::runtime_error("Jolt heightfield dimensions are invalid");
        const JPH::HeightFieldShapeSettings settings(heights.data(), JPH::Vec3::sZero(),
            JPH::Vec3(sampleSpacing / BethesdaUnitsPerMeter,
                1.0f / BethesdaUnitsPerMeter, sampleSpacing / BethesdaUnitsPerMeter), side);
        const auto result = settings.Create();
        if (!result.IsValid())
            throw std::runtime_error(std::string("Jolt rejected LAND heightfield: ")
                + result.GetError().c_str());
        return mImpl->addBody(result.Get(), origin, glm::quat(1.0f, 0.0f, 0.0f, 0.0f),
            JPH::EMotionType::Static, staticLayer);
    }

    BodyHandle JoltWorld::addStaticBody(const JPH::ShapeRefC& shape,
        const glm::vec3& position, const glm::quat& rotation)
    {
        mImpl->requireOwnerThread();
        if (shape == nullptr)
            throw std::runtime_error("Jolt static body needs a shape");
        return mImpl->addBody(shape.GetPtr(), position, rotation,
            JPH::EMotionType::Static, staticLayer);
    }

    BodyHandle JoltWorld::addDynamicBody(const JPH::ShapeRefC& shape,
        const glm::vec3& position, const glm::quat& rotation,
        float mass, float friction, float restitution)
    {
        mImpl->requireOwnerThread();
        if (shape == nullptr)
            throw std::runtime_error("Jolt dynamic body needs a shape");
        if (!std::isfinite(mass) || mass < 0.0f || !std::isfinite(friction)
            || friction < 0.0f || !std::isfinite(restitution) || restitution < 0.0f)
            throw std::runtime_error("Jolt dynamic body material or mass is invalid");
        return mImpl->addBody(shape.GetPtr(), position, rotation,
            JPH::EMotionType::Dynamic, movingLayer, mass, friction, restitution);
    }

    BodyHandle JoltWorld::addCollisionBody(const assets::HavokRigidBody& source,
        const glm::vec3& position, const glm::quat& rotation, float scale)
    {
        mImpl->requireOwnerThread();
        auto body = source;
        const auto finite = [](glm::vec3 v) {
            return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
        };
        if (!std::isfinite(scale) || scale <= 0
            || !finite(position) || !finite(body.centerOfMass) || !finite(body.inertiaDiagonal)
            || !std::isfinite(glm::dot(rotation, rotation)) || glm::dot(rotation, rotation) < 1e-8f
            || !std::isfinite(body.friction) || body.friction < 0
            || !std::isfinite(body.restitution) || body.restitution < 0 || body.restitution > 1
            || !std::isfinite(body.linearDamping) || body.linearDamping < 0
            || !std::isfinite(body.angularDamping) || body.angularDamping < 0
            || !std::isfinite(body.gravityFactor)
            || !std::isfinite(body.maxLinearVelocity) || body.maxLinearVelocity <= 0
            || !std::isfinite(body.maxAngularVelocity) || body.maxAngularVelocity <= 0)
            throw std::runtime_error("Native collision body has invalid properties");
        if (body.dynamic)
        {
            if (body.triangleMesh)
                throw std::runtime_error("Dynamic native collision requires convex or capsule shapes");
            if (!std::isfinite(body.mass) || body.mass <= 0)
                throw std::runtime_error("Dynamic native collision requires positive mass");
            if (body.inertiaDiagonal == glm::vec3(0)) updateHavokMassProperties(body);
            if (glm::any(glm::lessThanEqual(body.inertiaDiagonal, glm::vec3(0)))
                || !std::isfinite(glm::dot(body.inertiaOrientation, body.inertiaOrientation))
                || glm::dot(body.inertiaOrientation, body.inertiaOrientation) < 1e-8f)
                throw std::runtime_error("Native collision body has invalid inertia");
        }
        auto shape = scaleJoltCollisionShape(makeJoltCollisionShape(body), glm::vec3(scale));
        body.centerOfMass *= scale;
        body.inertiaDiagonal *= scale * scale; // Reference scaling preserves authored mass.
        if (!finite(body.centerOfMass) || !finite(body.inertiaDiagonal))
            throw std::runtime_error("Native collision scale overflows its mass properties");
        if (body.dynamic || (!shape->MustBeStatic() && hasAuthoredMassProperties(body)))
        {
            const auto offset = toEngineAxes(body.centerOfMass)
                - glm::vec3(shape->GetCenterOfMass().GetX(), shape->GetCenterOfMass().GetY(),
                    shape->GetCenterOfMass().GetZ());
            if (glm::length(offset) > 1e-6f)
            {
                const auto adjusted = JPH::OffsetCenterOfMassShapeSettings(
                    {offset.x, offset.y, offset.z}, shape.GetPtr()).Create();
                if (!adjusted.IsValid())
                    throw std::runtime_error("Jolt rejected native center of mass");
                shape = adjusted.Get();
            }
        }
        return mImpl->addBody(shape.GetPtr(), position, glm::normalize(rotation),
            body.dynamic ? JPH::EMotionType::Dynamic : JPH::EMotionType::Static,
            body.dynamic ? movingLayer : staticLayer, body.mass, body.friction,
            body.restitution, &body);
    }

    RagdollHandle JoltWorld::addRagdoll(const assets::HavokRagdoll& source,
        const glm::vec3& position, const glm::quat& rotation)
    {
        mImpl->requireOwnerThread();
        if (source.bones.empty() || source.bones.size() != source.bodies.size()
            || source.boneToBody.size() != source.bones.size()
            || source.constraints.size() + 1 != source.bones.size()
            || !std::isfinite(position.x) || !std::isfinite(position.y)
            || !std::isfinite(position.z)
            || !std::isfinite(rotation.x) || !std::isfinite(rotation.y)
            || !std::isfinite(rotation.z) || !std::isfinite(rotation.w)
            || glm::dot(rotation, rotation) <= std::numeric_limits<float>::epsilon())
            throw std::runtime_error("Havok ragdoll dimensions or transform are invalid");

        const glm::quat spawnRotation = glm::normalize(rotation);
        const JPH::RVec3 spawnPosition = toJoltPosition(position - mImpl->origin);
        const glm::vec3 spawnPositionMeters(
            static_cast<float>(spawnPosition.GetX()),
            static_cast<float>(spawnPosition.GetY()),
            static_cast<float>(spawnPosition.GetZ()));
        JPH::Ref<JPH::RagdollSettings> settings = new JPH::RagdollSettings();
        settings->mSkeleton = new JPH::Skeleton();
        settings->mParts.resize(source.bones.size());

        std::vector<std::uint32_t> bodyToBone(source.bodies.size(), 0xffffffffu);
        std::vector<glm::mat4> bodyWorld(source.bodies.size());
        for (std::uint32_t bone = 0; bone < source.bones.size(); ++bone)
        {
            const auto parent = source.bones[bone].parent;
            if (parent >= static_cast<std::int16_t>(bone))
                throw std::runtime_error("Havok ragdoll skeleton is not parent ordered");
            settings->mSkeleton->AddJoint(source.bones[bone].name, parent);

            const std::uint32_t bodyIndex = source.boneToBody[bone];
            if (bodyIndex >= source.bodies.size() || bodyToBone[bodyIndex] != 0xffffffffu)
                throw std::runtime_error("Havok ragdoll body map is invalid");
            bodyToBone[bodyIndex] = bone;
            const auto& body = source.bodies[bodyIndex];
            if (!body.dynamic)
                throw std::runtime_error("Havok ragdoll contains a static body");
            const JPH::ShapeRefC shape = makeJoltCollisionShape(body);
            bodyWorld[bodyIndex] = bodyTransformMeters(
                body, spawnPositionMeters, spawnRotation);
            const glm::vec3 bodyPosition(bodyWorld[bodyIndex][3]);
            const glm::quat bodyRotation = glm::normalize(
                glm::quat_cast(glm::mat3(bodyWorld[bodyIndex])));
            JPH::BodyCreationSettings bodySettings(shape.GetPtr(),
                toJoltMeters(bodyPosition), toJolt(bodyRotation),
                JPH::EMotionType::Dynamic, movingLayer);
            bodySettings.mFriction = body.friction;
            bodySettings.mRestitution = body.restitution;
            if (shape->GetInnerRadius() > 0.0f)
                bodySettings.mMotionQuality = JPH::EMotionQuality::LinearCast;
            applyMass(bodySettings, shape.GetPtr(), body.mass);
            static_cast<JPH::BodyCreationSettings&>(settings->mParts[bone]) =
                std::move(bodySettings);
        }

        const auto convertedFrame = [](const glm::mat4& frame) {
            return toEngineAxes(frame);
        };
        const auto worldFrame = [&](std::uint32_t body, const glm::mat4& frame) {
            return bodyWorld[body] * convertedFrame(frame);
        };
        const auto setWorldFrame = [](const glm::mat4& frame,
                                      JPH::RVec3& positionValue,
                                      JPH::Vec3& primaryAxis,
                                      JPH::Vec3& secondaryAxis) {
            positionValue = toJoltMeters(glm::vec3(frame[3]));
            primaryAxis = toJoltDirection(glm::vec3(frame[0]));
            secondaryAxis = toJoltDirection(glm::vec3(frame[1]));
        };

        for (std::uint32_t bone = 0; bone < source.bones.size(); ++bone)
        {
            const auto parent = source.bones[bone].parent;
            if (parent < 0)
                continue;
            const std::uint32_t body = source.boneToBody[bone];
            const std::uint32_t parentBody = source.boneToBody[parent];
            const auto found = std::ranges::find_if(source.constraints,
                [&](const assets::HavokConstraint& constraint) {
                    return (constraint.bodyA == body && constraint.bodyB == parentBody)
                        || (constraint.bodyA == parentBody && constraint.bodyB == body);
                });
            if (found == source.constraints.end())
                throw std::runtime_error("Havok ragdoll is missing a parent constraint");
            const bool parentIsA = found->bodyA == parentBody;
            const glm::mat4 parentFrame = worldFrame(parentBody,
                parentIsA ? found->localFrameA : found->localFrameB);
            const glm::mat4 childFrame = worldFrame(body,
                parentIsA ? found->localFrameB : found->localFrameA);

            if (found->type == assets::HavokConstraintType::Ragdoll)
            {
                JPH::Ref<JPH::SwingTwistConstraintSettings> joint =
                    new JPH::SwingTwistConstraintSettings();
                joint->mSpace = JPH::EConstraintSpace::WorldSpace;
                setWorldFrame(parentFrame, joint->mPosition1,
                    joint->mTwistAxis1, joint->mPlaneAxis1);
                setWorldFrame(childFrame, joint->mPosition2,
                    joint->mTwistAxis2, joint->mPlaneAxis2);
                joint->mTwistMinAngle = std::clamp(
                    found->minimumTwistAngle, -std::numbers::pi_v<float>,
                    std::numbers::pi_v<float>);
                joint->mTwistMaxAngle = std::clamp(
                    found->maximumTwistAngle, -std::numbers::pi_v<float>,
                    std::numbers::pi_v<float>);
                joint->mNormalHalfConeAngle = std::clamp(std::max(
                    std::abs(found->minimumConeAngle),
                    std::abs(found->maximumConeAngle)), 0.0f,
                    std::numbers::pi_v<float>);
                joint->mPlaneHalfConeAngle = std::clamp(std::max(
                    std::abs(found->minimumPlaneAngle),
                    std::abs(found->maximumPlaneAngle)), 0.0f,
                    std::numbers::pi_v<float>);
                joint->mMaxFrictionTorque = found->maximumFrictionTorque;
                settings->mParts[bone].mToParent = joint;
            }
            else
            {
                JPH::Ref<JPH::HingeConstraintSettings> joint =
                    new JPH::HingeConstraintSettings();
                joint->mSpace = JPH::EConstraintSpace::WorldSpace;
                setWorldFrame(parentFrame, joint->mPoint1,
                    joint->mHingeAxis1, joint->mNormalAxis1);
                setWorldFrame(childFrame, joint->mPoint2,
                    joint->mHingeAxis2, joint->mNormalAxis2);
                joint->mLimitsMin = std::clamp(
                    found->minimumTwistAngle, -std::numbers::pi_v<float>, 0.0f);
                joint->mLimitsMax = std::clamp(
                    found->maximumTwistAngle, 0.0f, std::numbers::pi_v<float>);
                joint->mMaxFrictionTorque = found->maximumFrictionTorque;
                settings->mParts[bone].mToParent = joint;
            }
        }

        settings->DisableParentChildCollisions();
        if (!settings->Stabilize())
            throw std::runtime_error("Jolt could not stabilize the Havok ragdoll");
        settings->CalculateConstraintPriorities();
        JPH::Ragdoll* created = settings->CreateRagdoll(
            mImpl->nextCollisionGroup++, 0, &mImpl->system);
        if (!created)
            throw std::runtime_error("Jolt could not create the Havok ragdoll");
        JPH::Ref<JPH::Ragdoll> ragdoll = created;
        ragdoll->AddToPhysicsSystem(JPH::EActivation::Activate);
        const std::uint32_t handle = mImpl->nextRagdollHandle++;
        mImpl->ragdolls.emplace(handle, std::move(ragdoll));
        return {handle};
    }

    void JoltWorld::setOrigin(const glm::vec3& origin)
    {
        mImpl->requireOwnerThread();
        if (!std::isfinite(origin.x) || !std::isfinite(origin.y) || !std::isfinite(origin.z))
            throw std::runtime_error("Jolt world origin must be finite");
        const JPH::RVec3 shift = toJoltPosition(mImpl->origin - origin);
        if (!shift.IsNearZero())
        {
            auto& bodyInterface = mImpl->system.GetBodyInterface();
            for (const std::uint32_t value : mImpl->bodies)
            {
                const JPH::BodyID body(value);
                bodyInterface.SetPosition(body, bodyInterface.GetPosition(body) + shift,
                    JPH::EActivation::DontActivate);
            }
            for (const auto& [handle, ragdoll] : mImpl->ragdolls)
                for (const JPH::BodyID body : ragdoll->GetBodyIDs())
                    bodyInterface.SetPosition(body,
                        bodyInterface.GetPosition(body) + shift,
                        JPH::EActivation::DontActivate);
        }
        mImpl->origin = origin;
    }

    void JoltWorld::removeBody(BodyHandle body)
    {
        mImpl->requireOwnerThread();
        const JPH::BodyID id(body.value);
        const auto found = mImpl->bodies.find(body.value);
        if (found == mImpl->bodies.end())
            return;
        auto& bodyInterface = mImpl->system.GetBodyInterface();
        bodyInterface.RemoveBody(id);
        bodyInterface.DestroyBody(id);
        mImpl->bodies.erase(found);
        mImpl->dynamicBodies.erase(body.value);
    }

    void JoltWorld::removeRagdoll(RagdollHandle ragdoll)
    {
        mImpl->requireOwnerThread();
        const auto found = mImpl->ragdolls.find(ragdoll.value);
        if (found == mImpl->ragdolls.end())
            return;
        found->second->RemoveFromPhysicsSystem();
        mImpl->ragdolls.erase(found);
    }

    void JoltWorld::step(float deltaSeconds)
    {
        mImpl->requireOwnerThread();
        constexpr float fixedStep = 1.0f / 60.0f;
        mImpl->accumulator = std::min(mImpl->accumulator + std::max(deltaSeconds, 0.0f),
            fixedStep * 4.0f);
        while (mImpl->accumulator >= fixedStep)
        {
            const auto error = mImpl->system.Update(
                fixedStep, 1, &mImpl->temporaryMemory, &mImpl->jobs);
            if (error != JPH::EPhysicsUpdateError::None)
                throw std::runtime_error("Jolt physics update exhausted its working buffers");
            mImpl->accumulator -= fixedStep;
        }
    }

    glm::mat4 JoltWorld::bodyTransform(BodyHandle body) const
    {
        mImpl->requireOwnerThread();
        if (!mImpl->bodies.contains(body.value))
            throw std::runtime_error("Jolt body handle is invalid");
        JPH::RVec3 position;
        JPH::Quat rotation;
        mImpl->system.GetBodyInterface().GetPositionAndRotation(
            JPH::BodyID(body.value), position, rotation);
        const glm::quat orientation(
            rotation.GetW(), rotation.GetX(), rotation.GetY(), rotation.GetZ());
        return glm::translate(glm::mat4(1.0f), mImpl->origin + glm::vec3(
            static_cast<float>(position.GetX()), static_cast<float>(position.GetY()),
            static_cast<float>(position.GetZ())) * BethesdaUnitsPerMeter)
            * glm::mat4_cast(orientation);
    }

    BodyState JoltWorld::bodyState(BodyHandle body) const
    {
        BodyState state;
        state.transform = bodyTransform(body);
        const auto& interface = mImpl->system.GetBodyInterface();
        const auto linear = interface.GetLinearVelocity(JPH::BodyID(body.value));
        const auto angular = interface.GetAngularVelocity(JPH::BodyID(body.value));
        state.linearVelocity = glm::vec3(linear.GetX(), linear.GetY(), linear.GetZ())
            * BethesdaUnitsPerMeter;
        state.angularVelocity = {angular.GetX(), angular.GetY(), angular.GetZ()};
        state.active = interface.IsActive(JPH::BodyID(body.value));
        return state;
    }

    void JoltWorld::restoreBodyState(BodyHandle body, const BodyState& state)
    {
        mImpl->requireOwnerThread();
        if (!mImpl->bodies.contains(body.value))
            throw std::runtime_error("Jolt body handle is invalid");
        for (int column = 0; column < 4; ++column)
            for (int row = 0; row < 4; ++row)
                if (!std::isfinite(state.transform[column][row]))
                    throw std::runtime_error("Jolt body state must be finite");
        const glm::mat3 basis(state.transform);
        const auto gram = glm::transpose(basis) * basis;
        for (int column = 0; column < 3; ++column)
        {
            if (!std::isfinite(state.linearVelocity[column])
                || !std::isfinite(state.angularVelocity[column]))
                throw std::runtime_error("Jolt body velocity must be finite");
            for (int row = 0; row < 3; ++row)
                if (std::abs(gram[column][row] - (column == row ? 1.0f : 0.0f)) > 1e-4f)
                    throw std::runtime_error("Jolt body state must have a rigid transform");
        }
        if (glm::determinant(basis) <= 0 || state.transform[0][3] != 0
            || state.transform[1][3] != 0 || state.transform[2][3] != 0
            || state.transform[3][3] != 1)
            throw std::runtime_error("Jolt body state must have a rigid transform");
        auto& interface = mImpl->system.GetBodyInterface();
        interface.SetPositionAndRotation(JPH::BodyID(body.value),
            toJoltPosition(glm::vec3(state.transform[3]) - mImpl->origin),
            toJolt(glm::normalize(glm::quat_cast(basis))), JPH::EActivation::DontActivate);
        if (interface.GetMotionType(JPH::BodyID(body.value)) != JPH::EMotionType::Static)
        {
            const auto velocity = state.linearVelocity / BethesdaUnitsPerMeter;
            interface.SetLinearAndAngularVelocity(JPH::BodyID(body.value),
                {velocity.x, velocity.y, velocity.z},
                {state.angularVelocity.x, state.angularVelocity.y, state.angularVelocity.z});
            if (state.active) interface.ActivateBody(JPH::BodyID(body.value));
            else interface.DeactivateBody(JPH::BodyID(body.value));
        }
    }

    bool JoltWorld::isBodyValid(BodyHandle body) const
    {
        mImpl->requireOwnerThread();
        if (!mImpl->bodies.contains(body.value)) return false;
        const JPH::BodyLockRead lock(mImpl->system.GetBodyLockInterface(),
            JPH::BodyID(body.value));
        return lock.SucceededAndIsInBroadPhase();
    }

    bool JoltWorld::isBodyActive(BodyHandle body) const
    {
        mImpl->requireOwnerThread();
        if (!mImpl->bodies.contains(body.value)) return false;
        const JPH::BodyLockRead lock(mImpl->system.GetBodyLockInterface(),
            JPH::BodyID(body.value));
        return lock.SucceededAndIsInBroadPhase() && lock.GetBody().IsActive();
    }

    bool JoltWorld::setMotionType(BodyHandle body, std::uint32_t motionType,
        bool allowActivate)
    {
        mImpl->requireOwnerThread();
        if (motionType > 2 || !mImpl->bodies.contains(body.value)) return false;
        const auto motion = motionType == 0 ? JPH::EMotionType::Static
            : motionType == 1 ? JPH::EMotionType::Dynamic : JPH::EMotionType::Kinematic;
        const JPH::BodyID id(body.value);
        {
            const JPH::BodyLockRead lock(mImpl->system.GetBodyLockInterface(), id);
            if (!lock.SucceededAndIsInBroadPhase()) return false;
            if (motion != JPH::EMotionType::Static
                && (!lock.GetBody().CanBeKinematicOrDynamic()
                    || lock.GetBody().GetShape()->MustBeStatic())) return false;
        }
        auto& interface = mImpl->system.GetBodyInterface();
        interface.SetMotionType(id, motion, allowActivate
            ? JPH::EActivation::Activate : JPH::EActivation::DontActivate);
        interface.SetObjectLayer(id, motion == JPH::EMotionType::Static ? staticLayer : movingLayer);
        if (motion == JPH::EMotionType::Dynamic) mImpl->dynamicBodies.insert(body.value);
        else mImpl->dynamicBodies.erase(body.value);
        // Also wake an unchanged motion type: callers may use this as an activation request.
        if (allowActivate && motion != JPH::EMotionType::Static) interface.ActivateBody(id);
        return true;
    }

    bool JoltWorld::applyHavokImpulse(BodyHandle body, const glm::vec3& direction,
        float magnitude)
    {
        mImpl->requireOwnerThread();
        if (!mImpl->bodies.contains(body.value) || !std::isfinite(magnitude) || magnitude < 0
            || !std::isfinite(direction.x) || !std::isfinite(direction.y)
            || !std::isfinite(direction.z)) return false;
        // Double precision keeps normalization finite for every finite float direction.
        const double length = glm::length(glm::dvec3(direction));
        if (length == 0) return false;
        const JPH::BodyID id(body.value);
        {
            const JPH::BodyLockRead lock(mImpl->system.GetBodyLockInterface(), id);
            if (!lock.SucceededAndIsInBroadPhase() || !lock.GetBody().IsDynamic()) return false;
        }
        auto& interface = mImpl->system.GetBodyInterface();
        interface.ActivateBody(id);
        if (magnitude > 0)
        {
            const glm::vec3 impulse(glm::dvec3(direction) * (magnitude / length));
            interface.AddImpulse(id, {impulse.x, impulse.y, impulse.z});
        }
        return true;
    }

    std::vector<glm::vec3> JoltWorld::dynamicBodyPositions() const
    {
        mImpl->requireOwnerThread();
        std::vector<glm::vec3> result;
        result.reserve(mImpl->dynamicBodies.size() + mImpl->ragdolls.size() * 18);
        const auto& bodyInterface = mImpl->system.GetBodyInterface();
        const auto append = [&](JPH::BodyID body) {
            const JPH::RVec3 position = bodyInterface.GetPosition(body);
            result.push_back(mImpl->origin + glm::vec3(
                static_cast<float>(position.GetX()), static_cast<float>(position.GetY()),
                static_cast<float>(position.GetZ())) * BethesdaUnitsPerMeter);
        };
        for (const std::uint32_t body : mImpl->dynamicBodies)
            append(JPH::BodyID(body));
        for (const auto& [handle, ragdoll] : mImpl->ragdolls)
        {
            (void)handle;
            for (const JPH::BodyID body : ragdoll->GetBodyIDs())
                append(body);
        }
        return result;
    }

    std::vector<glm::mat4> JoltWorld::ragdollBodyTransforms(
        RagdollHandle ragdoll) const
    {
        mImpl->requireOwnerThread();
        const auto found = mImpl->ragdolls.find(ragdoll.value);
        if (found == mImpl->ragdolls.end())
            throw std::runtime_error("Jolt ragdoll handle is invalid");
        std::vector<glm::mat4> result;
        result.reserve(found->second->GetBodyCount());
        const auto& bodyInterface = mImpl->system.GetBodyInterface();
        for (const JPH::BodyID body : found->second->GetBodyIDs())
        {
            JPH::RVec3 positionValue;
            JPH::Quat rotationValue;
            bodyInterface.GetPositionAndRotation(body, positionValue, rotationValue);
            const glm::quat orientation(rotationValue.GetW(), rotationValue.GetX(),
                rotationValue.GetY(), rotationValue.GetZ());
            result.push_back(glm::translate(glm::mat4(1.0f), mImpl->origin + glm::vec3(
                    static_cast<float>(positionValue.GetX()),
                    static_cast<float>(positionValue.GetY()),
                    static_cast<float>(positionValue.GetZ())) * BethesdaUnitsPerMeter)
                * glm::mat4_cast(orientation));
        }
        return result;
    }
}
