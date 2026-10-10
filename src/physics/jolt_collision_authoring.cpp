#include "physics/jolt_collision.hpp"

#include <Jolt/Physics/Body/MassProperties.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/CapsuleShape.h>
#include <Jolt/Physics/Collision/Shape/CompoundShape.h>
#include <Jolt/Physics/Collision/Shape/ConvexHullShape.h>
#include <Jolt/Physics/Collision/Shape/OffsetCenterOfMassShape.h>
#include <Jolt/Physics/Collision/Shape/RotatedTranslatedShape.h>
#include <Jolt/Physics/Collision/Shape/ScaledShape.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

#include <glm/ext/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

namespace openf4::physics
{
    namespace
    {
        glm::vec3 vector(JPH::Vec3 value) { return {value.GetX(), value.GetY(), value.GetZ()}; }
        glm::quat quaternion(JPH::Quat value)
        {
            return {value.GetW(), value.GetX(), value.GetY(), value.GetZ()};
        }

        bool finite(glm::vec3 value)
        {
            return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
        }

        float roundShapeScale(const glm::mat4& transform)
        {
            const float scale = glm::length(glm::vec3(transform[0]));
            const glm::mat3 basis(transform);
            const auto gram = glm::transpose(basis) * basis;
            for (int c = 0; c < 3; ++c)
                for (int r = 0; r < 3; ++r)
                    if (std::abs(gram[c][r] - (c == r ? scale * scale : 0.f))
                        > std::max(1.0f, scale * scale) * 1e-5f)
                        throw std::runtime_error("Non-uniform or sheared rounded collision cannot be exported");
            if (!std::isfinite(scale) || scale <= 0 || glm::determinant(basis) <= 0)
                throw std::runtime_error("Invalid or reflected collision scale");
            return scale;
        }

        assets::HavokConvexPolytope captureHull(const JPH::ConvexHullShape& hull,
            const glm::mat4& transform)
        {
            // ponytail: Jolt exposes outer hull points, not its rounded core; reject until core extraction is exact.
            if (hull.GetConvexRadius() > 0.0f)
                throw std::runtime_error("Rounded Jolt hull capture is unsupported; author a zero-radius hull or preserve native collision");
            assets::HavokConvexPolytope result;
            const auto center = vector(hull.GetCenterOfMass());
            for (JPH::uint i = 0; i < hull.GetNumPoints(); ++i)
                result.vertices.push_back(assets::swapBethesdaAxes(glm::vec3(
                    transform * glm::vec4(center + vector(hull.GetPoint(i)), 1))));
            for (JPH::uint face = 0; face < hull.GetNumFaces(); ++face)
            {
                std::vector<JPH::uint> indices(hull.GetNumVerticesInFace(face));
                hull.GetFaceVertices(face, static_cast<JPH::uint>(indices.size()), indices.data());
                for (std::size_t i = 1; i + 1 < indices.size(); ++i)
                    result.indices.insert(result.indices.end(), {indices[0], indices[i + 1], indices[i]});
            }
            return result;
        }

        void capture(const JPH::Shape& shape, const glm::mat4& transform,
            assets::HavokRigidBody& body, unsigned depth)
        {
            if (depth > 32 || !std::isfinite(glm::determinant(transform))
                || glm::determinant(glm::mat3(transform)) <= 0)
                throw std::runtime_error("Invalid, reflected, or too deeply nested collision shape");
            switch (shape.GetSubType())
            {
            case JPH::EShapeSubType::ConvexHull:
                body.parts.push_back({glm::mat4(1), glm::vec3(1),
                    captureHull(static_cast<const JPH::ConvexHullShape&>(shape), transform)});
                break;
            case JPH::EShapeSubType::Box:
            {
                const auto& box = static_cast<const JPH::BoxShape&>(shape);
                if (box.GetConvexRadius() > 0.0f)
                    throw std::runtime_error("Rounded Jolt box capture is unsupported; author a zero-radius box");
                const auto half = vector(box.GetHalfExtent());
                assets::HavokConvexPolytope poly;
                for (unsigned i = 0; i < 8; ++i)
                {
                    const glm::vec3 point = half * glm::vec3(i & 1 ? 1 : -1,
                        i & 2 ? 1 : -1, i & 4 ? 1 : -1);
                    poly.vertices.push_back(assets::swapBethesdaAxes(point));
                }
                poly.convexRadius = box.GetConvexRadius();
                const auto hull = makeJoltConvexHull(poly);
                body.parts.push_back({glm::mat4(1), glm::vec3(1),
                    captureHull(static_cast<const JPH::ConvexHullShape&>(*hull), transform)});
                break;
            }
            case JPH::EShapeSubType::Capsule:
            {
                const auto& capsule = static_cast<const JPH::CapsuleShape&>(shape);
                const float scale = roundShapeScale(transform);
                const auto endpoint = [&](float sign) {
                    return assets::swapBethesdaAxes(glm::vec3(transform * glm::vec4(
                        0, sign * capsule.GetHalfHeightOfCylinder(), 0, 1)));
                };
                body.capsules.push_back({glm::mat4(1), glm::vec3(1),
                    {endpoint(-1), endpoint(1), capsule.GetRadius() * scale}});
                break;
            }
            case JPH::EShapeSubType::StaticCompound:
            case JPH::EShapeSubType::MutableCompound:
            {
                const auto& compound = static_cast<const JPH::CompoundShape&>(shape);
                for (const auto& child : compound.GetSubShapes())
                {
                    const auto rotation = quaternion(child.GetRotation());
                    const auto origin = vector(compound.GetCenterOfMass()) + vector(child.GetPositionCOM())
                        - rotation * vector(child.mShape->GetCenterOfMass());
                    capture(*child.mShape, transform * glm::translate(glm::mat4(1), origin)
                        * glm::mat4_cast(rotation), body, depth + 1);
                }
                break;
            }
            case JPH::EShapeSubType::RotatedTranslated:
            {
                const auto& decorated = static_cast<const JPH::RotatedTranslatedShape&>(shape);
                capture(*decorated.GetInnerShape(), transform
                    * glm::translate(glm::mat4(1), vector(decorated.GetPosition()))
                    * glm::mat4_cast(quaternion(decorated.GetRotation())), body, depth + 1);
                break;
            }
            case JPH::EShapeSubType::Scaled:
            {
                const auto& scaled = static_cast<const JPH::ScaledShape&>(shape);
                const auto scale = vector(scaled.GetScale());
                if (!finite(scale) || glm::any(glm::lessThanEqual(scale, glm::vec3(0))))
                    throw std::runtime_error("Collision scale must be finite and positive");
                capture(*scaled.GetInnerShape(), glm::scale(transform, scale), body, depth + 1);
                break;
            }
            case JPH::EShapeSubType::OffsetCenterOfMass:
                capture(*static_cast<const JPH::OffsetCenterOfMassShape&>(shape).GetInnerShape(),
                    transform, body, depth + 1);
                break;
            case JPH::EShapeSubType::Mesh:
            {
                if (body.dynamic)
                    throw std::runtime_error("Dynamic triangle collision cannot be exported");
                if (!body.triangleMesh) body.triangleMesh.emplace();
                auto& mesh = *body.triangleMesh;
                JPH::Shape::GetTrianglesContext context;
                shape.GetTrianglesStart(context, JPH::AABox::sBiggest(), shape.GetCenterOfMass(),
                    JPH::Quat::sIdentity(), JPH::Vec3::sReplicate(1));
                std::array<JPH::Float3, 96> points;
                for (;;)
                {
                    const int triangles = shape.GetTrianglesNext(context, 32, points.data());
                    if (!triangles) break;
                    if (mesh.vertices.size() > 3'000'000 - static_cast<std::size_t>(triangles) * 3)
                        throw std::runtime_error("Captured collision exceeds the triangle limit");
                    for (int i = 0; i < triangles; ++i)
                    {
                        const auto first = static_cast<std::uint32_t>(mesh.vertices.size());
                        for (int j = 0; j < 3; ++j)
                        {
                            const auto& p = points[static_cast<std::size_t>(i) * 3 + j];
                            mesh.vertices.push_back(assets::swapBethesdaAxes(glm::vec3(
                                transform * glm::vec4(p.x, p.y, p.z, 1))));
                        }
                        mesh.indices.insert(mesh.indices.end(), {first, first + 2, first + 1});
                    }
                }
                break;
            }
            default:
                throw std::runtime_error("Jolt shape is outside the native collision export subset");
            }
        }
    }

    assets::HavokConvexPolytope makeHavokConvexHull(
        std::span<const glm::vec3> positions, float convexRadiusUnits)
    {
        if (convexRadiusUnits != 0.0f)
            throw std::runtime_error("Native hull authoring currently requires zero convex radius");
        assets::HavokConvexPolytope input;
        input.convexRadius = convexRadiusUnits / BethesdaUnitsPerMeter;
        for (const auto point : positions)
            input.vertices.push_back(assets::engineToHavokPosition(point));
        const auto shape = makeJoltConvexHull(input);
        return captureHull(static_cast<const JPH::ConvexHullShape&>(*shape), glm::mat4(1));
    }

    assets::HavokRigidBody captureJoltCollisionShape(
        const JPH::Shape& shape, const assets::HavokRigidBody& metadata)
    {
        if (metadata.triangleMesh && !metadata.triangleMesh->primitiveDataRuns.empty())
            throw std::runtime_error("Per-triangle Havok metadata requires the authored mesh, not a cooked Jolt mesh");
        assets::HavokRigidBody body = metadata;
        body.parts.clear();
        body.capsules.clear();
        body.triangleMesh.reset();
        capture(shape, glm::mat4(1), body, 0);
        (void)makeJoltCollisionShape(body);
        return body;
    }

    void updateHavokMassProperties(assets::HavokRigidBody& body)
    {
        if (!body.dynamic)
        {
            body.mass = 0;
            body.centerOfMass = {};
            body.inertiaDiagonal = {};
            body.inertiaOrientation = {1, 0, 0, 0};
            return;
        }
        if (!std::isfinite(body.mass) || body.mass <= 0)
            throw std::runtime_error("Dynamic native collision requires a positive mass");
        const auto shape = makeJoltCollisionShape(body);
        auto properties = shape->GetMassProperties();
        if (!std::isfinite(properties.mMass) || properties.mMass <= 0)
            throw std::runtime_error("Collision has no valid solid volume for mass properties");
        properties.ScaleToMass(body.mass);
        JPH::Mat44 orientation;
        JPH::Vec3 diagonal;
        if (!properties.DecomposePrincipalMomentsOfInertia(orientation, diagonal))
            throw std::runtime_error("Collision inertia cannot be decomposed");
        glm::mat4 engineOrientation(1);
        for (int i = 0; i < 3; ++i) engineOrientation[i] = glm::vec4(vector(orientation.GetColumn3(i)), 0);
        body.centerOfMass = assets::swapBethesdaAxes(vector(shape->GetCenterOfMass()));
        body.inertiaDiagonal = assets::swapBethesdaAxes(vector(diagonal));
        body.inertiaOrientation = glm::normalize(glm::quat_cast(
            assets::swapBethesdaAxes(engineOrientation)));
        if (!finite(body.inertiaDiagonal)
            || glm::any(glm::lessThanEqual(body.inertiaDiagonal, glm::vec3(0))))
            throw std::runtime_error("Collision inertia is not positive and finite");
    }
}
