#pragma once

// PROTOTYPE 0.1 - minimal mesh editing for the asset workspace.
// Pure, renderer-free helpers: cube generation, welded topology for
// vertex/edge/triangle selection, screen-space picking, and position edits.
// All positions are engine space (Y-up, Bethesda units, before the renderer's
// X reflection). Triangles follow the engine convention used by imports:
// outward normal = cross(c - a, b - a).

#include "assets/nif_model.hpp"

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include <glm/glm.hpp>

namespace openf4::editor::mesh
{
    enum class Element { vertex, edge, triangle };

    // Coincident source vertices (UV/normal seams) are welded into one point so a
    // selected "vertex" moves every split copy and the surface never tears.
    struct Topology
    {
        std::vector<glm::vec3> points;
        std::vector<std::uint32_t> vertexPoint;     // source vertex -> point
        std::vector<glm::uvec2> edges;              // unique point pairs, x < y
        std::vector<glm::uvec3> triangles;          // point triples, source order
    };

    struct ScreenPoint
    {
        glm::vec2 position{};
        float depth = 0;      // clip w; smaller is closer
        bool visible = false;
    };

    // Axis-aligned cube, 24 vertices (flat normals, per-face UVs), 12 triangles.
    [[nodiscard]] assets::ModelSurface makeCube(glm::vec3 center, float edgeLength);

    [[nodiscard]] Topology buildTopology(const assets::ModelSurface& surface);
    // Re-reads point positions after vertex edits without re-welding.
    void refreshPoints(Topology& topology, const assets::ModelSurface& surface);

    [[nodiscard]] std::size_t elementCount(const Topology& topology, Element element);
    // Unique point indices touched by the selected elements.
    [[nodiscard]] std::vector<std::uint32_t> selectedPoints(const Topology& topology,
        Element element, std::span<const std::uint32_t> elements);
    [[nodiscard]] glm::vec3 centroid(const Topology& topology, std::span<const std::uint32_t> points);

    // Returns new positions for every source vertex: those welded to `points`
    // are transformed by `transform`, the rest keep `start`.
    [[nodiscard]] std::vector<glm::vec3> transformPoints(const Topology& topology,
        std::span<const glm::vec3> start, std::span<const std::uint32_t> points,
        const glm::mat4& transform);

    // Writes positions and recomputes normals only around changed vertices.
    void applyPositions(assets::ModelSurface& surface, std::span<const glm::vec3> positions);

    // Nearest element under `mouse` within `radius` pixels. Points are indexed by
    // topology point. Triangles use an inside test and the closest average depth.
    [[nodiscard]] std::optional<std::uint32_t> pickElement(const Topology& topology,
        Element element, std::span<const ScreenPoint> screen, glm::vec2 mouse, float radius);
}
