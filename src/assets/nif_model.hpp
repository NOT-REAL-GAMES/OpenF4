#pragma once

#include "render/mesh_builder.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace empire::assets
{
    struct ModelSurface
    {
        std::vector<render::SourceVertex> vertices;
        std::vector<std::uint32_t> indices;
        std::string materialPath;
        std::string diffuseTexture;
        std::string normalTexture;
        float alphaCutoff = 0.0f;
    };

    struct ModelGeometry
    {
        std::vector<ModelSurface> surfaces;
    };

    [[nodiscard]] ModelGeometry loadNifModel(std::span<const std::byte> bytes);
}
