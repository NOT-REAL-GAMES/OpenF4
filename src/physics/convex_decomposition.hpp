#pragma once

// PROTOTYPE collision authoring: convexity checks and approximate convex decomposition.
//
// Everything here is space-agnostic (any consistent units/axes); the editor feeds engine
// units. Triangle winding is ignored: solids are recovered by voxelization + flood fill.
//
// Decomposition algorithm (documented in convex_decomposition.cpp): hierarchical
// voxel-based approximate convex decomposition in the spirit of V-HACD 2 - voxelize the
// surface, flood-fill the interior, then repeatedly cut the most concave cell along the
// axis-aligned voxel plane that minimises the summed concavity (convex-hull volume minus
// solid volume) of the two halves. Leaves are re-fitted to the exact input surface.

#include <cstdint>
#include <span>
#include <vector>

#include <glm/vec3.hpp>

namespace openf4::physics
{
    struct ConvexityReport
    {
        bool convex = true;
        // Deepest surface sample (vertex/edge midpoint/triangle centroid) inside the convex
        // hull of the input, in input units. Zero for convex input.
        float depth = 0;
        float hullVolume = 0;
        // Voxel-estimated enclosed volume; 0 for open (non-watertight) input.
        float solidVolume = 0;
    };

    struct ConvexDecompositionSettings
    {
        // Voxels along the longest bounding-box axis (clamped to [8, 96]).
        unsigned resolution = 40;
        // Hard cap on the number of output pieces.
        unsigned maxPieces = 16;
        // Stop splitting a cell once its concavity (beyond voxel staircase error) is below
        // this fraction of the total solid volume.
        float concavityTolerance = 0.02f;
    };

    // Sample-depth convexity test for an arbitrary triangle soup. `relativeTolerance` is a
    // fraction of the bounding-box diagonal; samples shallower than that count as on-hull.
    [[nodiscard]] ConvexityReport measureConvexity(std::span<const glm::vec3> vertices,
        std::span<const std::uint32_t> indices, float relativeTolerance = 0.005f);

    // Exact polytope test matching the Havok writer (Havok winding: outward normal =
    // cross(b - a, c - a)): every triangle's plane has every vertex on or behind it within
    // max(1, diagonal) * 1e-5, no degenerate triangles, positive volume. Empty index lists
    // (point clouds) are convex by definition.
    [[nodiscard]] bool polytopeIsConvex(std::span<const glm::vec3> vertices,
        std::span<const std::uint32_t> indices);

    // Returns point clouds; the convex hull of each cloud is one output piece. Pieces never
    // contain fewer than 4 non-coplanar points. Convex input yields exactly one piece.
    [[nodiscard]] std::vector<std::vector<glm::vec3>> decomposeConvex(
        std::span<const glm::vec3> vertices, std::span<const std::uint32_t> indices,
        const ConvexDecompositionSettings& settings = {});

    // Convex hull volume of a point cloud (0 when degenerate).
    [[nodiscard]] float convexHullVolume(std::span<const glm::vec3> points);
}
