#pragma once

#include "assets/nif_model.hpp"

#include <limits>
#include <optional>

namespace openf4::world
{
    struct GeometryBounds
    {
        glm::dvec3 minimum{};
        glm::dvec3 maximum{};
    };

    namespace detail
    {
        // Immutable, shared per-model geometry; constructed in the model worker.
        class PickingGeometry
        {
        public:
            explicit PickingGeometry(const assets::ModelGeometry& geometry);
            [[nodiscard]] std::optional<double> raycast(const glm::dvec3& origin,
                const glm::dvec3& unitDirection, const glm::mat4& transform,
                double maxDistance = std::numeric_limits<double>::infinity()) const;
            [[nodiscard]] GeometryBounds bounds(const glm::mat4& transform) const;

        private:
            struct Node
            {
                GeometryBounds bounds;
                std::uint32_t first = 0;
                std::uint32_t count = 0;
                std::uint32_t left = 0;
                std::uint32_t right = 0;
            };
            std::vector<glm::vec3> mVertices;
            std::vector<glm::uvec3> mTriangles;
            std::vector<Node> mNodes;
            glm::dvec4 mSphere{};
            std::uint32_t build(std::uint32_t first, std::uint32_t count);
        };
    }
}
