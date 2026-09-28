#pragma once

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

namespace openf4::editor
{
    enum class ManipulationSpace { world, local };
    enum class ManipulationOperation { translate, rotate, scale };

    // All arguments use engine coordinates, before the renderer's X reflection.
    [[nodiscard]] glm::mat4 makeManipulationPivot(const glm::vec3& position,
        const glm::mat4& activeTransform, ManipulationSpace space);

    // Both matrices are measured from the same captured drag start. Translation
    // snapStep is in engine units; rotation snapStep is in degrees. Zero disables
    // snapping. Axis scale handles are converted to positive uniform scaling.
    [[nodiscard]] glm::mat4 snapManipulationPivot(const glm::mat4& startPivot,
        const glm::mat4& manipulatedPivot, ManipulationOperation operation,
        ManipulationSpace space, float snapStep = 0.0f);

    [[nodiscard]] glm::mat4 applyManipulationDelta(const glm::mat4& startTransform,
        const glm::mat4& startPivot, const glm::mat4& currentPivot);

    // Set local up to an upward terrain normal while preserving the object's
    // horizontal forward heading, translation, and positive uniform scale.
    [[nodiscard]] glm::mat4 alignToTerrainNormal(const glm::mat4& transform,
        const glm::vec3& normal);
}
