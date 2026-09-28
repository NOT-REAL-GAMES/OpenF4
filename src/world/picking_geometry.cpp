#include "world/picking_geometry.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace openf4::world::detail
{
    namespace
    {
        GeometryBounds emptyBounds()
        {
            return {glm::dvec3(std::numeric_limits<double>::infinity()),
                glm::dvec3(-std::numeric_limits<double>::infinity())};
        }

        void include(GeometryBounds& bounds, const glm::dvec3& point)
        {
            bounds.minimum = glm::min(bounds.minimum, point);
            bounds.maximum = glm::max(bounds.maximum, point);
        }

        bool rayBounds(const GeometryBounds& bounds, const glm::dvec3& origin,
            const glm::dvec3& direction, double limit)
        {
            double near = 0.0;
            for (int axis = 0; axis < 3; ++axis)
            {
                if (std::abs(direction[axis]) < 1.0e-15)
                {
                    if (origin[axis] < bounds.minimum[axis] || origin[axis] > bounds.maximum[axis])
                        return false;
                    continue;
                }
                double first = (bounds.minimum[axis] - origin[axis]) / direction[axis];
                double last = (bounds.maximum[axis] - origin[axis]) / direction[axis];
                if (first > last)
                    std::swap(first, last);
                near = std::max(near, first);
                limit = std::min(limit, last);
                if (near > limit)
                    return false;
            }
            return true;
        }

        std::optional<double> rayTriangle(const glm::dvec3& origin,
            const glm::dvec3& direction, const glm::dvec3& a,
            const glm::dvec3& b, const glm::dvec3& c)
        {
            const auto edge1 = b - a;
            const auto edge2 = c - a;
            const auto cross = glm::cross(direction, edge2);
            const double determinant = glm::dot(edge1, cross);
            const double epsilon = 1.0e-12 * glm::length(edge1) * glm::length(cross);
            if (std::abs(determinant) <= epsilon)
                return std::nullopt;
            const auto offset = origin - a;
            const double u = glm::dot(offset, cross) / determinant;
            if (u < 0.0 || u > 1.0)
                return std::nullopt;
            const auto q = glm::cross(offset, edge1);
            const double v = glm::dot(direction, q) / determinant;
            if (v < 0.0 || u + v > 1.0)
                return std::nullopt;
            const double distance = glm::dot(edge2, q) / determinant;
            return distance >= 0.0 ? std::optional(distance) : std::nullopt;
        }
    }

    PickingGeometry::PickingGeometry(const assets::ModelGeometry& geometry)
    {
        for (const auto& surface : geometry.surfaces)
        {
            if (surface.indices.empty())
                continue;
            if (surface.indices.size() % 3 != 0
                || surface.vertices.size() + mVertices.size() > std::numeric_limits<std::uint32_t>::max())
                throw std::runtime_error("Invalid picking geometry topology");
            const auto offset = static_cast<std::uint32_t>(mVertices.size());
            for (const auto& vertex : surface.vertices)
            {
                if (!std::isfinite(vertex.position.x) || !std::isfinite(vertex.position.y)
                    || !std::isfinite(vertex.position.z))
                    throw std::runtime_error("Invalid picking geometry vertex");
                mVertices.push_back(vertex.position);
            }
            for (std::size_t i = 0; i < surface.indices.size(); i += 3)
            {
                glm::uvec3 triangle;
                for (int corner = 0; corner < 3; ++corner)
                {
                    const auto index = surface.indices[i + corner];
                    if (index >= surface.vertices.size())
                        throw std::runtime_error("Invalid picking geometry index");
                    triangle[corner] = offset + index;
                }
                mTriangles.push_back(triangle);
            }
        }
        if (mTriangles.empty())
            throw std::runtime_error("Model contains no pickable triangles");
        build(0, static_cast<std::uint32_t>(mTriangles.size()));
        const auto center = (mNodes.front().bounds.minimum + mNodes.front().bounds.maximum) * 0.5;
        double radius = 0.0;
        for (const auto& triangle : mTriangles)
            for (int corner = 0; corner < 3; ++corner)
                radius = std::max(radius, glm::length(glm::dvec3(mVertices[triangle[corner]]) - center));
        mSphere = glm::dvec4(center, radius);
    }

    std::uint32_t PickingGeometry::build(std::uint32_t first, std::uint32_t count)
    {
        const auto index = static_cast<std::uint32_t>(mNodes.size());
        Node node{emptyBounds(), first, count};
        for (std::uint32_t i = first; i < first + count; ++i)
            for (int corner = 0; corner < 3; ++corner)
                include(node.bounds, glm::dvec3(mVertices[mTriangles[i][corner]]));
        mNodes.push_back(node);
        if (count <= 8)
            return index;
        const auto extent = node.bounds.maximum - node.bounds.minimum;
        const int axis = extent.x >= extent.y && extent.x >= extent.z ? 0
            : extent.y >= extent.z ? 1 : 2;
        const auto midpoint = first + count / 2;
        std::nth_element(mTriangles.begin() + first, mTriangles.begin() + midpoint,
            mTriangles.begin() + first + count, [&](glm::uvec3 a, glm::uvec3 b) {
                const auto centroid = [&](glm::uvec3 t) {
                    return static_cast<double>(mVertices[t.x][axis])
                        + mVertices[t.y][axis] + mVertices[t.z][axis];
                };
                return centroid(a) < centroid(b);
            });
        const auto left = build(first, midpoint - first);
        const auto right = build(midpoint, first + count - midpoint);
        mNodes[index].left = left;
        mNodes[index].right = right;
        mNodes[index].count = 0;
        return index;
    }

    std::optional<double> PickingGeometry::raycast(const glm::dvec3& origin,
        const glm::dvec3& unitDirection, const glm::mat4& transform, double maxDistance) const
    {
        if (!(maxDistance >= 0.0))
            return std::nullopt;
        const glm::dmat4 matrix(transform);
        if (!std::isfinite(glm::determinant(matrix)) || std::abs(glm::determinant(matrix)) < 1.0e-18)
            return std::nullopt;
        const auto inverse = glm::inverse(matrix);
        const auto localOrigin = glm::dvec3(inverse * glm::dvec4(origin, 1.0));
        // Do not normalize after inverse-transforming: t remains world-space distance.
        const auto direction = glm::dvec3(inverse * glm::dvec4(unitDirection, 0.0));
        const double directionLength = glm::dot(direction, direction);
        if (!(directionLength > 0.0) || !std::isfinite(directionLength))
            return std::nullopt;
        const auto offset = localOrigin - glm::dvec3(mSphere);
        const double b = glm::dot(offset, direction);
        const double c = glm::dot(offset, offset) - mSphere.w * mSphere.w;
        const double discriminant = b * b - directionLength * c;
        if (discriminant < 0.0 || (-b + std::sqrt(discriminant)) / directionLength < 0.0)
            return std::nullopt;
        std::optional<double> nearest;
        const auto visit = [&](auto&& self, std::uint32_t index) -> void {
            const auto& node = mNodes[index];
            if (!rayBounds(node.bounds, localOrigin, direction, maxDistance))
                return;
            if (node.count == 0)
            {
                self(self, node.left);
                self(self, node.right);
                return;
            }
            for (std::uint32_t i = node.first; i < node.first + node.count; ++i)
            {
                const auto& triangle = mTriangles[i];
                const auto distance = rayTriangle(localOrigin, direction,
                    mVertices[triangle.x], mVertices[triangle.y], mVertices[triangle.z]);
                if (distance && *distance <= maxDistance)
                {
                    maxDistance = *distance;
                    nearest = distance;
                }
            }
        };
        visit(visit, 0);
        return nearest;
    }

    GeometryBounds PickingGeometry::bounds(const glm::mat4& transform) const
    {
        GeometryBounds result = emptyBounds();
        const glm::dmat4 matrix(transform);
        for (const auto& triangle : mTriangles)
            for (int corner = 0; corner < 3; ++corner)
                include(result, glm::dvec3(matrix * glm::dvec4(mVertices[triangle[corner]], 1.0)));
        return result;
    }
}
