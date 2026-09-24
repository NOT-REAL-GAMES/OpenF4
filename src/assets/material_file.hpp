#pragma once

#include <cstddef>
#include <span>
#include <string>

#include <glm/glm.hpp>

namespace openf4::assets
{
    struct MaterialInfo
    {
        std::string diffuseTexture;
        std::string normalTexture;
        std::string grayscaleTexture;
        glm::vec4 baseColor{1.0f};
        float grayscaleToPaletteScale = 0.5f;
        float alphaCutoff = 0.0f;
        bool grayscaleToPalette = false;
        bool twoSided = false;
        bool decal = false;
    };

    [[nodiscard]] MaterialInfo loadMaterial(std::span<const std::byte> bytes);
}
