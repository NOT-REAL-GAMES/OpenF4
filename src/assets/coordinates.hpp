#pragma once

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

namespace openf4::assets
{
    inline constexpr float bethesdaUnitsPerMeter = 69.99125f;

    // Bethesda/Havok are Z-up; OpenF4 is Y-up. This reflection is its own inverse.
    [[nodiscard]] inline glm::vec3 swapBethesdaAxes(glm::vec3 value) noexcept
    {
        return {value.x, value.z, value.y};
    }

    [[nodiscard]] inline glm::mat4 swapBethesdaAxes(const glm::mat4& value) noexcept
    {
        glm::mat4 axes(1.0f);
        axes[1] = {0, 0, 1, 0};
        axes[2] = {0, 1, 0, 0};
        return axes * value * axes;
    }

    [[nodiscard]] inline glm::vec3 engineToHavokPosition(glm::vec3 units) noexcept
    {
        return swapBethesdaAxes(units) / bethesdaUnitsPerMeter;
    }

    [[nodiscard]] inline glm::vec3 havokToEnginePosition(glm::vec3 meters) noexcept
    {
        return swapBethesdaAxes(meters) * bethesdaUnitsPerMeter;
    }
}
