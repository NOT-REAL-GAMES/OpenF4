#include "physics/jolt_collision.hpp"

#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/CapsuleShape.h>
#include <Jolt/Physics/Collision/Shape/ScaledShape.h>
#include <Jolt/Physics/Collision/Shape/StaticCompoundShape.h>

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
    template<class F> void rejects(F operation, const char* message)
    {
        try { operation(); }
        catch (const std::runtime_error&) { return; }
        throw std::runtime_error(message);
    }
    void near(glm::vec3 a, glm::vec3 b, const char* message)
    {
        require(glm::length(a - b) < 0.001f, message);
    }
    glm::vec3 vector(JPH::Vec3 v) { return {v.GetX(), v.GetY(), v.GetZ()}; }
    void sameBounds(const JPH::Shape& a, const JPH::Shape& b)
    {
        near(vector(a.GetCenterOfMass() + a.GetLocalBounds().mMin),
            vector(b.GetCenterOfMass() + b.GetLocalBounds().mMin), "Collision minimum bounds changed");
        near(vector(a.GetCenterOfMass() + a.GetLocalBounds().mMax),
            vector(b.GetCenterOfMass() + b.GetLocalBounds().mMax), "Collision maximum bounds changed");
    }

    assets::HavokRigidBody cube()
    {
        std::array<glm::vec3, 8> points;
        for (unsigned i = 0; i < points.size(); ++i)
            points[i] = glm::vec3(i & 1 ? 1 : -1, i & 2 ? 2 : -2, i & 4 ? 3 : -3)
                * physics::BethesdaUnitsPerMeter + glm::vec3(123, -57, 218);
        assets::HavokRigidBody body;
        body.parts.push_back({glm::mat4(1), glm::vec3(1), physics::makeHavokConvexHull(points)});
        return body;
    }

    void testHullAndAxes()
    {
        near(assets::havokToEnginePosition(assets::engineToHavokPosition({31, -27, 59})),
            {31, -27, 59}, "Position units/axes did not round trip");
        auto body = cube();
        const auto& hull = body.parts[0].polytope;
        require(hull.vertices.size() == 8 && hull.indices.size() == 36,
            "Generated cube hull is not closed and triangulated");
        glm::vec3 center(0);
        for (const auto p : hull.vertices) center += p;
        center /= static_cast<float>(hull.vertices.size());
        for (std::size_t i = 0; i < hull.indices.size(); i += 3)
        {
            const auto a = hull.vertices[hull.indices[i]];
            const auto b = hull.vertices[hull.indices[i + 1]];
            const auto c = hull.vertices[hull.indices[i + 2]];
            require(glm::dot(glm::cross(b - a, c - a), a - center) > 0,
                "Havok hull face winding is inverted");
        }
        body.materialId = 123456;
        body.collisionFilterInfo = 42;
        body.position = {1, 2, 3};
        const auto shape = physics::makeJoltCollisionShape(body);
        const auto captured = physics::captureJoltCollisionShape(*shape, body);
        require(captured.materialId == body.materialId && captured.collisionFilterInfo == 42
            && captured.position == body.position, "Capture replaced retained Fallout metadata");
        sameBounds(*shape, *physics::makeJoltCollisionShape(captured));
        const std::array bad{glm::vec3(0), glm::vec3(1), glm::vec3(2),
            glm::vec3(std::numeric_limits<float>::quiet_NaN())};
        rejects([&] { (void)physics::makeHavokConvexHull(bad); }, "Non-finite hull was accepted");
    }

    void testCompound()
    {
        const auto source = physics::makeJoltCollisionShape(cube());
        const auto capsule = JPH::CapsuleShapeSettings(1.2f, 0.35f).Create();
        const auto box = JPH::BoxShapeSettings({0.5f, 1, 2}, 0).Create();
        require(capsule.IsValid() && box.IsValid(), "Primitive construction failed");
        JPH::StaticCompoundShapeSettings settings;
        settings.AddShape({2, -3, 4}, JPH::Quat::sIdentity(), source.GetPtr());
        settings.AddShape({-4, 2, 1}, JPH::Quat::sRotation(JPH::Vec3::sAxisZ(), 0.7f), capsule.Get().GetPtr());
        settings.AddShape({1, 6, -2}, JPH::Quat::sIdentity(), box.Get().GetPtr());
        const auto compound = settings.Create();
        require(compound.IsValid(), "Compound construction failed");
        const auto captured = physics::captureJoltCollisionShape(*compound.Get(), {});
        require(captured.parts.size() == 2 && captured.capsules.size() == 1,
            "Compound capture lost a child");
        sameBounds(*compound.Get(), *physics::makeJoltCollisionShape(captured));
        const auto scaled = JPH::ScaledShapeSettings(capsule.Get().GetPtr(), {1, 2, 1}).Create();
        if (scaled.IsValid())
            rejects([&] { (void)physics::captureJoltCollisionShape(*scaled.Get(), {}); },
                "Non-uniform capsule silently changed shape during export");
    }

    void testNativeTransforms()
    {
        auto body = cube();
        body.position = {1, -2, 3};
        body.rotation = glm::angleAxis(0.7f, glm::normalize(glm::vec3(2, 3, -1)));
        const glm::vec3 havokPoint{4, -5, 6};
        const auto transformed = glm::vec3(physics::havokBodyLocalTransform(body)
            * glm::vec4(assets::havokToEnginePosition(havokPoint), 1));
        near(transformed, assets::havokToEnginePosition(body.position + body.rotation * havokPoint),
            "Body rotation did not conjugate the reflected Bethesda basis");
        auto scaledQuaternion = body;
        scaledQuaternion.rotation *= 2.0f;
        near(glm::vec3(physics::havokBodyLocalTransform(scaledQuaternion)
                * glm::vec4(assets::havokToEnginePosition(havokPoint), 1)), transformed,
            "Body quaternion normalization changed rotation");
        body.rotation = {0, 0, 0, 0};
        rejects([&] { (void)physics::havokBodyLocalTransform(body); },
            "Zero body quaternion was accepted");

        const auto invalidPart = [&](glm::mat4 transform, glm::vec3 scale) {
            auto malformed = cube();
            malformed.parts.front().transform = transform;
            malformed.parts.front().scale = scale;
            rejects([&] { (void)physics::makeJoltCollisionShape(malformed); },
                "Malformed collision child transform was accepted");
        };
        glm::mat4 transform(1);
        transform[0][0] = std::numeric_limits<float>::quiet_NaN();
        invalidPart(transform, glm::vec3(1));
        transform = glm::mat4(1);
        transform[1][0] = 0.25f;
        invalidPart(transform, glm::vec3(1));
        invalidPart(glm::scale(glm::mat4(1), glm::vec3(-1, 1, 1)), glm::vec3(1));
        invalidPart(glm::scale(glm::mat4(1), glm::vec3(2)), glm::vec3(1));
        transform = glm::mat4(1);
        transform[0][3] = 0.1f;
        invalidPart(transform, glm::vec3(1));
        invalidPart(glm::mat4(1), {0, 1, 1});
        invalidPart(glm::mat4(1), {1, std::numeric_limits<float>::infinity(), 1});

        assets::HavokRigidBody capsule;
        capsule.capsules.push_back({glm::mat4(1), glm::vec3(1), {{0, 0, 0}, {0, 0, 1}, 0.2f}});
        capsule.capsules.front().transform[3][0] = std::numeric_limits<float>::quiet_NaN();
        rejects([&] { (void)physics::makeJoltCollisionShape(capsule); },
            "Malformed capsule placement was accepted");
        capsule.capsules.front().transform = glm::mat4(1);
        capsule.capsules.front().capsule.radius = std::numeric_limits<float>::quiet_NaN();
        rejects([&] { (void)physics::makeJoltCollisionShape(capsule); },
            "Non-finite capsule radius was accepted");
    }

    void testRoundedCaptureRejection()
    {
        auto body = cube();
        body.parts.front().polytope.convexRadius = 0.1f;
        const auto roundedHull = physics::makeJoltCollisionShape(body);
        rejects([&] { (void)physics::captureJoltCollisionShape(*roundedHull, body); },
            "Rounded Jolt hull capture silently changed native collision dimensions");
        const auto box = JPH::BoxShapeSettings({1, 2, 3}, 0.1f).Create();
        require(box.IsValid(), "Rounded box construction failed");
        rejects([&] { (void)physics::captureJoltCollisionShape(*box.Get(), {}); },
            "Rounded Jolt box capture silently changed native collision dimensions");
        std::array<glm::vec3, 8> points;
        for (unsigned i = 0; i < points.size(); ++i)
            points[i] = glm::vec3(i & 1 ? 100 : -100, i & 2 ? 100 : -100, i & 4 ? 100 : -100);
        rejects([&] { (void)physics::makeHavokConvexHull(points, 7); },
            "Rounded native hull authoring silently changed collision dimensions");
    }

    void testMesh()
    {
        assets::HavokRigidBody body;
        body.triangleMesh.emplace();
        body.triangleMesh->vertices = {{0, 0, 0}, {2, 0, 0}, {0, 3, 0}};
        body.triangleMesh->indices = {0, 1, 2};
        const auto shape = physics::makeJoltCollisionShape(body);
        const auto captured = physics::captureJoltCollisionShape(*shape, body);
        require(captured.triangleMesh && captured.triangleMesh->indices.size() == 3,
            "Mesh capture lost its triangle");
        sameBounds(*shape, *physics::makeJoltCollisionShape(captured));
        body.dynamic = true;
        rejects([&] { (void)physics::captureJoltCollisionShape(*shape, body); },
            "Dynamic mesh export was accepted");
        body.dynamic = false;
        body.triangleMesh->primitiveDataRuns.push_back({0, 17, 0, 1});
        rejects([&] { (void)physics::captureJoltCollisionShape(*shape, body); },
            "Capture silently discarded triangle material metadata");
    }

    void testMassAndPreview()
    {
        auto body = cube();
        body.dynamic = true;
        body.mass = 3;
        physics::updateHavokMassProperties(body);
        near(assets::havokToEnginePosition(body.centerOfMass), {123, -57, 218},
            "Authored center of mass changed axis or scale");
        const glm::mat3 rotation = glm::mat3_cast(body.inertiaOrientation);
        glm::mat3 diagonal(0);
        for (int i = 0; i < 3; ++i) diagonal[i][i] = body.inertiaDiagonal[i];
        const auto tensor = rotation * diagonal * glm::transpose(rotation);
        // Half extents in Havok axes: (1, 3, 2) meters, mass 3 kg.
        near({tensor[0][0], tensor[1][1], tensor[2][2]}, {13, 5, 10},
            "Native inertia units or basis are incorrect");
        physics::JoltWorld world;
        body.gravityFactor = 0;
        const glm::vec3 start{100, 200, 300};
        const auto handle = world.addCollisionBody(body, start);
        for (int i = 0; i < 120; ++i) world.step(1.f / 60);
        near(glm::vec3(world.bodyTransform(handle)[3]), start,
            "Native gravity factor or center-of-mass frame was not respected");
        world.removeBody(handle);
        body.mass = -1;
        rejects([&] { physics::updateHavokMassProperties(body); }, "Negative authored mass was accepted");
    }
}

int main()
{
    try
    {
        testHullAndAxes();
        testCompound();
        testNativeTransforms();
        testRoundedCaptureRejection();
        testMesh();
        testMassAndPreview();
        std::cout << "Collision authoring tests passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
