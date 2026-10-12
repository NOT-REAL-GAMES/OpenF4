#include "editor/mesh_edit.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>

namespace openf4::editor::mesh
{
    assets::ModelSurface makeCube(glm::vec3 center, float edgeLength)
    {
        if (!(edgeLength > 0) || !std::isfinite(edgeLength)) throw std::runtime_error("Cube edge must be positive");
        const float h = edgeLength * 0.5f;
        // Outward normal n with u x v = n; quads p0..p3 are counter-clockwise seen from
        // outside, emitted clockwise (engine convention) as (0,2,1) and (0,3,2).
        struct Face { glm::vec3 n, u, v; };
        const std::array<Face, 6> faces{{
            {{ 1,0,0}, {0,1,0}, {0,0,1}}, {{-1,0,0}, {0,0,1}, {0,1,0}},
            {{0, 1,0}, {0,0,1}, {1,0,0}}, {{0,-1,0}, {1,0,0}, {0,0,1}},
            {{0,0, 1}, {1,0,0}, {0,1,0}}, {{0,0,-1}, {0,1,0}, {1,0,0}},
        }};
        constexpr std::array<glm::vec2, 4> corners{{{-1,-1}, {1,-1}, {1,1}, {-1,1}}};
        assets::ModelSurface surface;
        surface.metallicFactor = 0; surface.roughnessFactor = 1;
        for (const auto& face : faces)
        {
            const auto base = static_cast<std::uint32_t>(surface.vertices.size());
            for (const auto corner : corners)
            {
                render::SourceVertex vertex;
                vertex.position = center + h * (face.n + corner.x * face.u + corner.y * face.v);
                vertex.normal = face.n;
                vertex.uv = (corner + 1.0f) * 0.5f;
                surface.vertices.push_back(vertex);
            }
            for (auto index : {0u, 2u, 1u, 0u, 3u, 2u}) surface.indices.push_back(base + index);
        }
        return surface;
    }

    Topology buildTopology(const assets::ModelSurface& surface)
    {
        Topology result;
        std::map<std::tuple<float, float, float>, std::uint32_t> weld;
        result.vertexPoint.reserve(surface.vertices.size());
        for (const auto& vertex : surface.vertices)
        {
            const auto key = std::make_tuple(vertex.position.x, vertex.position.y, vertex.position.z);
            const auto [found, inserted] = weld.try_emplace(key, static_cast<std::uint32_t>(result.points.size()));
            if (inserted) result.points.push_back(vertex.position);
            result.vertexPoint.push_back(found->second);
        }
        std::set<std::pair<std::uint32_t, std::uint32_t>> edges;
        for (std::size_t i = 0; i + 2 < surface.indices.size(); i += 3)
        {
            std::array<std::uint32_t, 3> p{};
            for (int k = 0; k < 3; ++k)
            {
                const auto index = surface.indices[i + k];
                if (index >= result.vertexPoint.size()) throw std::runtime_error("Surface has invalid triangle indices");
                p[k] = result.vertexPoint[index];
            }
            result.triangles.emplace_back(p[0], p[1], p[2]);
            for (int k = 0; k < 3; ++k)
            {
                const auto a = p[k], b = p[(k + 1) % 3];
                if (a != b) edges.emplace(std::min(a, b), std::max(a, b));
            }
        }
        for (const auto& [a, b] : edges) result.edges.emplace_back(a, b);
        return result;
    }

    void refreshPoints(Topology& topology, const assets::ModelSurface& surface)
    {
        if (surface.vertices.size() != topology.vertexPoint.size()) throw std::runtime_error("Topology is stale");
        for (std::size_t i = 0; i < surface.vertices.size(); ++i)
            topology.points[topology.vertexPoint[i]] = surface.vertices[i].position;
    }

    std::size_t elementCount(const Topology& topology, Element element)
    {
        switch (element)
        {
        case Element::vertex: return topology.points.size();
        case Element::edge: return topology.edges.size();
        default: return topology.triangles.size();
        }
    }

    std::vector<std::uint32_t> selectedPoints(const Topology& topology, Element element,
        std::span<const std::uint32_t> elements)
    {
        std::set<std::uint32_t> points;
        for (const auto e : elements)
        {
            if (e >= elementCount(topology, element)) continue;
            if (element == Element::vertex) points.insert(e);
            else if (element == Element::edge) { points.insert(topology.edges[e].x); points.insert(topology.edges[e].y); }
            else for (int k = 0; k < 3; ++k) points.insert(topology.triangles[e][k]);
        }
        return {points.begin(), points.end()};
    }

    glm::vec3 centroid(const Topology& topology, std::span<const std::uint32_t> points)
    {
        if (points.empty()) return {};
        glm::dvec3 sum{};
        for (const auto p : points) sum += glm::dvec3(topology.points.at(p));
        return glm::vec3(sum / static_cast<double>(points.size()));
    }

    std::vector<glm::vec3> transformPoints(const Topology& topology, std::span<const glm::vec3> start,
        std::span<const std::uint32_t> points, const glm::mat4& transform)
    {
        if (start.size() != topology.vertexPoint.size()) throw std::runtime_error("Topology is stale");
        std::vector<char> moved(topology.points.size(), 0);
        for (const auto p : points) moved.at(p) = 1;
        std::vector<glm::vec3> result(start.begin(), start.end());
        for (std::size_t i = 0; i < result.size(); ++i)
            if (moved[topology.vertexPoint[i]]) result[i] = glm::vec3(transform * glm::vec4(start[i], 1));
        return result;
    }

    void applyPositions(assets::ModelSurface& surface, std::span<const glm::vec3> positions)
    {
        if (positions.size() != surface.vertices.size()) throw std::runtime_error("Vertex count mismatch");
        std::vector<char> changed(positions.size(), 0);
        bool any = false;
        for (std::size_t i = 0; i < positions.size(); ++i)
        {
            if (surface.vertices[i].position == positions[i]) continue;
            surface.vertices[i].position = positions[i]; changed[i] = 1; any = true;
        }
        if (!any) return;
        // Recompute normals for vertices of every triangle touching a changed vertex,
        // leaving untouched (e.g. authored smooth) normals elsewhere intact.
        std::vector<char> dirty(positions.size(), 0);
        const auto& indices = surface.indices;
        for (std::size_t i = 0; i + 2 < indices.size(); i += 3)
            if (changed[indices[i]] || changed[indices[i + 1]] || changed[indices[i + 2]])
                dirty[indices[i]] = dirty[indices[i + 1]] = dirty[indices[i + 2]] = 1;
        std::vector<glm::vec3> sums(positions.size(), glm::vec3(0));
        for (std::size_t i = 0; i + 2 < indices.size(); i += 3)
        {
            const auto a = indices[i], b = indices[i + 1], c = indices[i + 2];
            if (!dirty[a] && !dirty[b] && !dirty[c]) continue;
            const auto& pa = surface.vertices[a].position;
            const auto n = glm::cross(surface.vertices[c].position - pa, surface.vertices[b].position - pa);
            sums[a] += n; sums[b] += n; sums[c] += n;
        }
        for (std::size_t i = 0; i < sums.size(); ++i)
            if (dirty[i] && glm::length(sums[i]) > 1e-12f) surface.vertices[i].normal = glm::normalize(sums[i]);
    }

    namespace
    {
        float segmentDistance(glm::vec2 p, glm::vec2 a, glm::vec2 b)
        {
            const auto ab = b - a; const float length = glm::dot(ab, ab);
            const float t = length > 0 ? std::clamp(glm::dot(p - a, ab) / length, 0.0f, 1.0f) : 0.0f;
            return glm::length(p - (a + t * ab));
        }
        bool inside(glm::vec2 p, glm::vec2 a, glm::vec2 b, glm::vec2 c)
        {
            const auto side = [](glm::vec2 o, glm::vec2 u, glm::vec2 v) { return (u.x - o.x) * (v.y - o.y) - (u.y - o.y) * (v.x - o.x); };
            if (std::abs(side(a, b, c)) < 1e-3f) return false; // edge-on in screen space
            const float d1 = side(p, a, b), d2 = side(p, b, c), d3 = side(p, c, a);
            const bool negative = d1 < 0 || d2 < 0 || d3 < 0, positive = d1 > 0 || d2 > 0 || d3 > 0;
            return !(negative && positive);
        }
    }

    std::optional<std::uint32_t> pickElement(const Topology& topology, Element element,
        std::span<const ScreenPoint> screen, glm::vec2 mouse, float radius)
    {
        if (screen.size() != topology.points.size()) return std::nullopt;
        std::optional<std::uint32_t> best;
        float bestScore = std::numeric_limits<float>::infinity();
        const auto consider = [&](std::uint32_t index, float score) { if (score < bestScore) { bestScore = score; best = index; } };
        if (element == Element::vertex)
        {
            for (std::uint32_t i = 0; i < screen.size(); ++i)
            {
                if (!screen[i].visible) continue;
                const float d = glm::length(screen[i].position - mouse);
                // Prefer nearer-to-camera points among those within the pick radius.
                if (d <= radius) consider(i, d + screen[i].depth * 1e-6f);
            }
        }
        else if (element == Element::edge)
        {
            for (std::uint32_t i = 0; i < topology.edges.size(); ++i)
            {
                const auto& a = screen[topology.edges[i].x]; const auto& b = screen[topology.edges[i].y];
                if (!a.visible || !b.visible) continue;
                const float d = segmentDistance(mouse, a.position, b.position);
                if (d <= radius) consider(i, d + (a.depth + b.depth) * 5e-7f);
            }
        }
        else
        {
            for (std::uint32_t i = 0; i < topology.triangles.size(); ++i)
            {
                const auto& t = topology.triangles[i];
                const auto& a = screen[t.x]; const auto& b = screen[t.y]; const auto& c = screen[t.z];
                if (!a.visible || !b.visible || !c.visible) continue;
                if (inside(mouse, a.position, b.position, c.position)) consider(i, (a.depth + b.depth + c.depth) / 3);
            }
        }
        return best;
    }
}
