#include "editor/editor_manipulation.hpp"
#include "world/world_scene.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace openf4::editor
{
    namespace
    {
        void validate(const glm::mat4& transform)
        {
            assets::PlacedReference reference;
            world::setBethesdaReferenceTransform(reference, transform);
            if (glm::length(glm::vec3(transform[0][3], transform[1][3], transform[2][3])) > 0.0001f
                || std::abs(transform[3][3] - 1.0f) > 0.0001f)
                throw std::runtime_error("Manipulation requires an affine transform");
        }

        glm::mat3 rotation(const glm::mat4& transform)
        {
            return glm::mat3(transform) / glm::length(glm::vec3(transform[0]));
        }

        float snap(float value, float step)
        {
            return step > 0.0f ? std::round(value / step) * step : value;
        }
    }

    glm::mat4 makeManipulationPivot(const glm::vec3& position,
        const glm::mat4& activeTransform, ManipulationSpace space)
    {
        validate(activeTransform);
        glm::mat4 result(space == ManipulationSpace::local ? rotation(activeTransform) : glm::mat3(1.0f));
        result[3] = glm::vec4(position, 1.0f);
        validate(result);
        return result;
    }

    glm::mat4 snapManipulationPivot(const glm::mat4& startPivot,
        const glm::mat4& manipulatedPivot, ManipulationOperation operation,
        ManipulationSpace space, float snapStep)
    {
        validate(startPivot);
        if (!std::isfinite(snapStep) || snapStep < 0.0f)
            throw std::runtime_error("Manipulation snap step must be finite and nonnegative");
        glm::mat4 result = startPivot;
        const auto startRotation = rotation(startPivot);
        if (operation == ManipulationOperation::scale)
        {
            float factor = 1.0f;
            float largestChange = 0.0f;
            for (int axis = 0; axis < 3; ++axis)
            {
                const float ratio = glm::length(glm::vec3(manipulatedPivot[axis]))
                    / glm::length(glm::vec3(startPivot[axis]));
                if (!std::isfinite(ratio) || ratio <= 0.0f
                    || glm::dot(glm::vec3(manipulatedPivot[axis]), glm::vec3(startPivot[axis])) <= 0.0f)
                    throw std::runtime_error("Manipulation requires positive scale");
                const float change = std::abs(std::log(ratio));
                if (change > largestChange) { factor = ratio; largestChange = change; }
            }
            for (int axis = 0; axis < 3; ++axis) result[axis] *= factor;
        }
        else
        {
            validate(manipulatedPivot);
            if (operation == ManipulationOperation::translate)
            {
                const auto basis = space == ManipulationSpace::local ? startRotation : glm::mat3(1.0f);
                auto delta = glm::transpose(basis) * glm::vec3(manipulatedPivot[3] - startPivot[3]);
                for (int axis = 0; axis < 3; ++axis) delta[axis] = snap(delta[axis], snapStep);
                result[3] += glm::vec4(basis * delta, 0.0f);
            }
            else
            {
                const auto currentRotation = rotation(manipulatedPivot);
                const auto relative = space == ManipulationSpace::local
                    ? glm::transpose(startRotation) * currentRotation
                    : currentRotation * glm::transpose(startRotation);
                auto delta = glm::normalize(glm::quat_cast(relative));
                if (delta.w < 0.0f) delta = -delta;
                const float angle = 2.0f * std::acos(std::clamp(delta.w, -1.0f, 1.0f));
                glm::mat3 snapped(1.0f);
                const glm::vec3 vector(delta.x, delta.y, delta.z);
                if (glm::length(vector) > 0.000001f)
                    snapped = glm::mat3_cast(glm::angleAxis(
                        glm::radians(snap(glm::degrees(angle), snapStep)), glm::normalize(vector)));
                const auto orientation = space == ManipulationSpace::local
                    ? startRotation * snapped : snapped * startRotation;
                const float scale = glm::length(glm::vec3(startPivot[0]));
                for (int axis = 0; axis < 3; ++axis) result[axis] = glm::vec4(orientation[axis] * scale, 0.0f);
            }
        }
        validate(result);
        return result;
    }

    glm::mat4 applyManipulationDelta(const glm::mat4& startTransform,
        const glm::mat4& startPivot, const glm::mat4& currentPivot)
    {
        validate(startTransform);
        validate(startPivot);
        validate(currentPivot);
        const auto result = currentPivot * glm::inverse(startPivot) * startTransform;
        validate(result);
        return result;
    }

    glm::mat4 alignToTerrainNormal(const glm::mat4& transform, const glm::vec3& normal)
    {
        validate(transform);
        const float length = glm::length(normal);
        if (!std::isfinite(length) || length < 0.000001f || normal.y <= 0.0f)
            throw std::runtime_error("Terrain alignment requires an upward finite normal");
        const glm::vec3 up = normal / length;
        glm::vec3 heading(transform[2].x, 0.0f, transform[2].z);
        if (glm::length(heading) < 0.000001f)
            heading = glm::vec3(-transform[0].z, 0.0f, transform[0].x);
        heading = glm::normalize(heading);
        // Solve dot(forward, up) = 0 while keeping forward's XZ azimuth.
        const auto forward = glm::normalize(glm::vec3(heading.x * up.y,
            -glm::dot(heading, up), heading.z * up.y));
        const auto right = glm::normalize(glm::cross(up, forward));
        const float scale = glm::length(glm::vec3(transform[0]));
        auto result = transform;
        result[0] = glm::vec4(right * scale, 0.0f);
        result[1] = glm::vec4(up * scale, 0.0f);
        result[2] = glm::vec4(forward * scale, 0.0f);
        validate(result);
        return result;
    }
}
