#pragma once

#include "render/mesh_builder.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include <glm/mat4x4.hpp>

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
        bool decal = false;
    };

    struct ModelSkinBone
    {
        std::string name;
        glm::mat4 skinToBone{1.0f};
    };

    struct ModelGeometry
    {
        std::vector<ModelSurface> surfaces;
        std::vector<ModelSkinBone> skinBones;
        struct ConnectionPoint
        {
            std::string name;
            glm::mat4 transform{1.0f};
        };
        std::vector<ConnectionPoint> attachmentPoints;
        std::vector<std::string> attachmentTargets;
    };

    [[nodiscard]] ModelGeometry loadNifModel(std::span<const std::byte> bytes);
}
