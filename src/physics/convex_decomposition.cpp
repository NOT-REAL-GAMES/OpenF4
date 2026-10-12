// PROTOTYPE collision authoring: convexity checks and approximate convex decomposition.
//
// Split algorithm ("Split optimally" in the asset editor). Exact minimum convex
// decomposition is NP-hard, so this is an approximate convex decomposition in the spirit
// of V-HACD 2 (Mamou), implemented natively on top of Jolt's ConvexHullBuilder:
//   1. Voxelize the triangles with an exact triangle/box overlap test (Akenine-Moller SAT)
//      on a grid with `resolution` voxels along the longest axis plus two padding layers.
//   2. Flood-fill the outside from the padding; everything not reached is solid. Open
//      meshes that leak are retried once with the shell dilated by one voxel.
//   3. Every leaf is re-fitted to the exact input: its points are the input triangles
//      clipped to the leaf's box plus corners of fully interior voxels, so pieces hug the
//      authored surface and neighbours share their cut planes. Its concavity is that hull's
//      volume minus the voxel solid volume (surface voxels count half).
//   4. Repeatedly take the leaf with the largest concavity above `concavityTolerance` x
//      total volume and cut it with the axis-aligned voxel plane minimising the voxel-hull
//      concavity of the two halves (coarse sampling on all three axes, then exhaustive
//      refinement around the best plane). Stop at `maxPieces` or when all leaves are within
//      tolerance.
// Convex input (by sample depth) short-circuits to a single hull of the input vertices.
#include "physics/convex_decomposition.hpp"

#include "physics/jolt_collision.hpp"

#include <Jolt/Jolt.h>
#include <Jolt/Geometry/ConvexHullBuilder.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>

#include <glm/geometric.hpp>
#include <glm/vec4.hpp>
#include <glm/common.hpp>
#include <glm/vector_relational.hpp>

namespace openf4::physics
{
    namespace
    {
        struct Hull
        {
            bool valid = false;
            float volume = 0;
            float area = 0;
            std::vector<glm::vec4> planes; // outward unit normal, w = -dot(n, point on plane)
        };

        Hull buildHull(std::span<const glm::vec3> points, float tolerance, bool wantPlanes)
        {
            Hull result;
            if (points.size() < 4) return result;
            ensureJoltRuntime();
            JPH::ConvexHullBuilder::Positions positions;
            positions.reserve(points.size());
            for (const auto p : points) positions.push_back(JPH::Vec3(p.x, p.y, p.z));
            JPH::ConvexHullBuilder builder(positions);
            const char* error = nullptr;
            const auto status = builder.Initialize(std::numeric_limits<int>::max(), tolerance, error);
            if (status != JPH::ConvexHullBuilder::EResult::Success
                && status != JPH::ConvexHullBuilder::EResult::MaxVerticesReached) return result;
            JPH::Vec3 center;
            builder.GetCenterOfMassAndVolume(center, result.volume);
            if (!(result.volume > 0) || !std::isfinite(result.volume)) { result.volume = 0; return result; }
            for (const auto* face : builder.GetFaces())
            {
                if (face->mRemoved) continue;
                const float length = face->mNormal.Length();
                if (!(length > 0)) continue;
                result.area += length * 0.5f;
                if (wantPlanes)
                {
                    const auto n = face->mNormal / length;
                    result.planes.emplace_back(n.GetX(), n.GetY(), n.GetZ(), -n.Dot(face->mCentroid));
                }
            }
            result.valid = true;
            return result;
        }

        bool finite(glm::vec3 v) { return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z); }

        void bounds(std::span<const glm::vec3> points, glm::vec3& lo, glm::vec3& hi)
        {
            lo = glm::vec3(std::numeric_limits<float>::max()); hi = -lo;
            for (const auto p : points) { lo = glm::min(lo, p); hi = glm::max(hi, p); }
        }

        // Akenine-Moller triangle/AABB separating-axis test; box centred at the origin.
        bool triangleBoxOverlap(glm::vec3 half, glm::vec3 v0, glm::vec3 v1, glm::vec3 v2)
        {
            for (int q = 0; q < 3; ++q)
                if (std::min({v0[q], v1[q], v2[q]}) > half[q] || std::max({v0[q], v1[q], v2[q]}) < -half[q]) return false;
            const std::array<glm::vec3, 3> edges{v1 - v0, v2 - v1, v0 - v2};
            for (const auto& e : edges)
                for (int i = 0; i < 3; ++i)
                {
                    glm::vec3 unit(0); unit[i] = 1;
                    const auto axis = glm::cross(unit, e);
                    const float p0 = glm::dot(axis, v0), p1 = glm::dot(axis, v1), p2 = glm::dot(axis, v2);
                    const float r = half.x * std::abs(axis.x) + half.y * std::abs(axis.y) + half.z * std::abs(axis.z);
                    if (std::min({p0, p1, p2}) > r || std::max({p0, p1, p2}) < -r) return false;
                }
            const auto normal = glm::cross(edges[0], edges[1]);
            const float r = half.x * std::abs(normal.x) + half.y * std::abs(normal.y) + half.z * std::abs(normal.z);
            return std::abs(glm::dot(normal, v0)) <= r;
        }

        void clipAxis(std::vector<glm::vec3>& polygon, int axis, float bound, bool keepAbove)
        {
            if (polygon.empty()) return;
            std::vector<glm::vec3> out;
            const auto inside = [&](const glm::vec3& p) { return keepAbove ? p[axis] >= bound : p[axis] <= bound; };
            for (std::size_t i = 0; i < polygon.size(); ++i)
            {
                const auto& a = polygon[i]; const auto& b = polygon[(i + 1) % polygon.size()];
                const bool ia = inside(a), ib = inside(b);
                if (ia) out.push_back(a);
                if (ia != ib)
                {
                    const float t = (bound - a[axis]) / (b[axis] - a[axis]);
                    auto p = a + (b - a) * t; p[axis] = bound; out.push_back(p);
                }
            }
            polygon = std::move(out);
        }

        enum : std::uint8_t { unknown = 0, surface = 1, exterior = 2, interior = 3 };

        struct Grid
        {
            glm::ivec3 dims{};
            glm::vec3 origin{};
            float size = 1;
            std::vector<std::uint8_t> label;
            [[nodiscard]] std::size_t index(int x, int y, int z) const
            { return static_cast<std::size_t>(x) + static_cast<std::size_t>(dims.x) * (static_cast<std::size_t>(y) + static_cast<std::size_t>(dims.y) * static_cast<std::size_t>(z)); }
        };

        bool floodOutside(Grid& grid)
        {
            for (auto& l : grid.label) if (l == exterior || l == interior) l = unknown;
            std::vector<glm::ivec3> stack{{0, 0, 0}};
            grid.label[0] = exterior;
            constexpr std::array<glm::ivec3, 6> steps{{{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}}};
            while (!stack.empty())
            {
                const auto v = stack.back(); stack.pop_back();
                for (const auto s : steps)
                {
                    const auto n = v + s;
                    if (glm::any(glm::lessThan(n, glm::ivec3(0))) || glm::any(glm::greaterThanEqual(n, grid.dims))) continue;
                    auto& l = grid.label[grid.index(n.x, n.y, n.z)];
                    if (l != unknown) continue;
                    l = exterior; stack.push_back(n);
                }
            }
            bool any = false;
            for (auto& l : grid.label) if (l == unknown) { l = interior; any = true; }
            return any;
        }

        struct Cell
        {
            glm::ivec3 lo{}, hi{};          // half-open voxel box
            std::vector<std::uint32_t> voxels; // indices into the occupied list
            float concavity = 0;               // voxel estimate, used to rank cut planes
            float excess = 0;                  // exact-hull estimate, used to stop splitting
            std::vector<glm::vec3> points;     // re-fitted hull points (input units)
        };

        // Hull of a voxel set equals the hull of the end caps of its x-rows.
        std::vector<glm::vec3> rowCorners(const std::vector<glm::ivec3>& occupied,
            const std::vector<std::uint32_t>& voxels, glm::ivec3 lo, glm::ivec3 hi)
        {
            const int ny = hi.y - lo.y, nz = hi.z - lo.z;
            std::vector<glm::ivec2> rows(static_cast<std::size_t>(ny) * static_cast<std::size_t>(nz),
                glm::ivec2(std::numeric_limits<int>::max(), std::numeric_limits<int>::min()));
            for (const auto i : voxels)
            {
                const auto v = occupied[i];
                auto& row = rows[static_cast<std::size_t>(v.y - lo.y) + static_cast<std::size_t>(ny) * static_cast<std::size_t>(v.z - lo.z)];
                row.x = std::min(row.x, v.x); row.y = std::max(row.y, v.x);
            }
            std::vector<glm::vec3> points;
            for (int z = 0; z < nz; ++z) for (int y = 0; y < ny; ++y)
            {
                const auto row = rows[static_cast<std::size_t>(y) + static_cast<std::size_t>(ny) * static_cast<std::size_t>(z)];
                if (row.x > row.y) continue;
                for (const float x : {float(row.x), float(row.y + 1)})
                    for (int k = 0; k < 4; ++k)
                        points.emplace_back(x, float(y + lo.y + (k & 1)), float(z + lo.z + (k >> 1)));
            }
            return points;
        }

        void measure(Cell& cell, const std::vector<glm::ivec3>& occupied)
        {
            const auto hull = buildHull(rowCorners(occupied, cell.voxels, cell.lo, cell.hi), 1e-3f, false);
            const float solid = static_cast<float>(cell.voxels.size());
            cell.concavity = hull.valid ? std::max(0.0f, hull.volume - solid) : 0.0f;
        }

        bool splitCell(const Cell& cell, const std::vector<glm::ivec3>& occupied, Cell& left, Cell& right)
        {
            glm::ivec3 mn(std::numeric_limits<int>::max()), mx(std::numeric_limits<int>::min());
            for (const auto i : cell.voxels) { mn = glm::min(mn, occupied[i]); mx = glm::max(mx, occupied[i]); }
            struct Candidate { int axis = -1, plane = 0; float cost = std::numeric_limits<float>::max(); };
            Candidate best;
            const auto evaluate = [&](int axis, int plane) {
                Cell a, b; a.lo = b.lo = cell.lo; a.hi = b.hi = cell.hi; a.hi[axis] = plane; b.lo[axis] = plane;
                for (const auto i : cell.voxels) (occupied[i][axis] < plane ? a : b).voxels.push_back(i);
                if (a.voxels.empty() || b.voxels.empty()) return false;
                measure(a, occupied); measure(b, occupied);
                // Tiny balance term breaks ties toward even cuts.
                const float balance = std::abs(float(a.voxels.size()) - float(b.voxels.size())) * 1e-4f;
                const float cost = a.concavity + b.concavity + balance;
                if (cost >= best.cost) return false;
                best = {axis, plane, cost}; left = std::move(a); right = std::move(b);
                return true;
            };
            constexpr int coarse = 10;
            int bestStep = 1;
            for (int axis = 0; axis < 3; ++axis)
            {
                const int first = mn[axis] + 1, last = mx[axis]; // plane p: left holds coord < p
                if (last < first) continue;
                const int step = std::max(1, (last - first + coarse) / coarse);
                for (int p = first; p <= last; p += step) if (evaluate(axis, p)) bestStep = step;
            }
            if (best.axis < 0) return false;
            if (bestStep > 1)
            {
                const int axis = best.axis, centre = best.plane;
                const int from = std::max(mn[axis] + 1, centre - bestStep + 1), to = std::min(mx[axis], centre + bestStep - 1);
                for (int p = from; p <= to; ++p) if (p != centre) (void)evaluate(axis, p);
            }
            return true;
        }
    }

    float convexHullVolume(std::span<const glm::vec3> points)
    {
        if (points.size() < 4) return 0;
        glm::vec3 lo, hi; bounds(points, lo, hi);
        const auto hull = buildHull(points, std::max(glm::length(hi - lo) * 1e-5f, 1e-7f), false);
        return hull.valid ? hull.volume : 0.0f;
    }

    ConvexityReport measureConvexity(std::span<const glm::vec3> vertices,
        std::span<const std::uint32_t> indices, float relativeTolerance)
    {
        ConvexityReport report;
        if (vertices.size() < 4) return report;
        for (const auto v : vertices) if (!finite(v)) throw std::runtime_error("Convexity input has non-finite vertices");
        glm::vec3 lo, hi; bounds(vertices, lo, hi);
        const float diagonal = glm::length(hi - lo);
        const auto hull = buildHull(vertices, std::max(diagonal * 1e-5f, 1e-7f), true);
        if (!hull.valid) return report; // flat/degenerate: no interior to be concave about
        report.hullVolume = hull.volume;
        const auto depthOf = [&](glm::vec3 p) {
            float depth = std::numeric_limits<float>::max();
            for (const auto& plane : hull.planes) depth = std::min(depth, -(glm::dot(glm::vec3(plane), p) + plane.w));
            return std::max(depth, 0.0f);
        };
        float deepest = 0;
        for (const auto v : vertices) deepest = std::max(deepest, depthOf(v));
        for (std::size_t i = 0; i + 2 < indices.size(); i += 3)
        {
            if (indices[i] >= vertices.size() || indices[i + 1] >= vertices.size() || indices[i + 2] >= vertices.size())
                throw std::runtime_error("Convexity input has invalid triangle indices");
            const auto a = vertices[indices[i]], b = vertices[indices[i + 1]], c = vertices[indices[i + 2]];
            deepest = std::max({deepest, depthOf((a + b + c) / 3.0f), depthOf((a + b) * .5f),
                depthOf((b + c) * .5f), depthOf((c + a) * .5f)});
        }
        report.depth = deepest;
        report.convex = deepest <= relativeTolerance * diagonal;
        return report;
    }

    bool polytopeIsConvex(std::span<const glm::vec3> vertices, std::span<const std::uint32_t> indices)
    {
        if (indices.empty()) return true;
        glm::vec3 lo, hi; bounds(vertices, lo, hi);
        const float tolerance = std::max(1.0f, glm::length(hi - lo)) * 0.00001f;
        double volume = 0;
        for (std::size_t i = 0; i + 2 < indices.size(); i += 3)
        {
            const auto ia = indices[i], ib = indices[i + 1], ic = indices[i + 2];
            if (ia >= vertices.size() || ib >= vertices.size() || ic >= vertices.size()) return false;
            const auto a = vertices[ia], b = vertices[ib], c = vertices[ic];
            auto normal = glm::cross(b - a, c - a);
            if (!(glm::length(normal) > 1.0e-8f)) return false;
            normal = glm::normalize(normal);
            const float w = -glm::dot(normal, a);
            for (const auto v : vertices) if (glm::dot(normal, v) + w > tolerance) return false;
            volume += glm::dot(glm::dvec3(a), glm::cross(glm::dvec3(b), glm::dvec3(c))) / 6;
        }
        return volume > 1.0e-12;
    }

    std::vector<std::vector<glm::vec3>> decomposeConvex(std::span<const glm::vec3> vertices,
        std::span<const std::uint32_t> indices, const ConvexDecompositionSettings& settings)
    {
        if (vertices.size() < 4 || indices.size() < 3 || indices.size() % 3)
            throw std::runtime_error("Convex decomposition needs an indexed triangle mesh with at least 4 vertices");
        for (const auto index : indices) if (index >= vertices.size()) throw std::runtime_error("Convex decomposition has invalid triangle indices");
        for (const auto v : vertices) if (!finite(v)) throw std::runtime_error("Convex decomposition input is not finite");
        if (measureConvexity(vertices, indices).convex)
        {
            if (convexHullVolume(vertices) <= 0) throw std::runtime_error("Input is flat; it has no convex volume to split");
            return {std::vector<glm::vec3>(vertices.begin(), vertices.end())};
        }

        glm::vec3 lo, hi; bounds(vertices, lo, hi);
        const auto extent = hi - lo;
        const float longest = std::max({extent.x, extent.y, extent.z});
        if (!(longest > 0)) throw std::runtime_error("Convex decomposition input has no extent");
        const unsigned resolution = std::clamp(settings.resolution, 8u, 96u);
        Grid grid;
        grid.size = longest / static_cast<float>(resolution);
        // Two padding layers: faces flush with a voxel boundary also mark the voxel beyond it,
        // so the outermost layer must stay untouched for the outside flood fill to start.
        grid.origin = lo - glm::vec3(2 * grid.size);
        for (int q = 0; q < 3; ++q)
            grid.dims[q] = std::max(1, static_cast<int>(std::ceil(extent[q] / grid.size - 1e-4f))) + 4;
        grid.label.assign(static_cast<std::size_t>(grid.dims.x) * grid.dims.y * grid.dims.z, unknown);

        const auto toGrid = [&](glm::vec3 p) { return (p - grid.origin) / grid.size; };
        const glm::vec3 half(0.5f + 1e-4f);
        for (std::size_t i = 0; i < indices.size(); i += 3)
        {
            const auto a = toGrid(vertices[indices[i]]), b = toGrid(vertices[indices[i + 1]]), c = toGrid(vertices[indices[i + 2]]);
            const auto tmin = glm::max(glm::ivec3(glm::floor(glm::min(a, glm::min(b, c)) - 1e-3f)), glm::ivec3(0));
            const auto tmax = glm::min(glm::ivec3(glm::floor(glm::max(a, glm::max(b, c)) + 1e-3f)), grid.dims - 1);
            for (int z = tmin.z; z <= tmax.z; ++z) for (int y = tmin.y; y <= tmax.y; ++y) for (int x = tmin.x; x <= tmax.x; ++x)
            {
                auto& l = grid.label[grid.index(x, y, z)];
                if (l == surface) continue;
                const glm::vec3 centre(x + .5f, y + .5f, z + .5f);
                if (triangleBoxOverlap(half, a - centre, b - centre, c - centre)) l = surface;
            }
        }
        if (!floodOutside(grid))
        {
            // Leaky (open) mesh: close pinholes by dilating the shell one voxel and retry.
            auto dilated = grid.label;
            for (int z = 0; z < grid.dims.z; ++z) for (int y = 0; y < grid.dims.y; ++y) for (int x = 0; x < grid.dims.x; ++x)
            {
                if (grid.label[grid.index(x, y, z)] != surface) continue;
                for (int dz = -1; dz <= 1; ++dz) for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx)
                {
                    const glm::ivec3 n(x + dx, y + dy, z + dz);
                    if (glm::any(glm::lessThan(n, glm::ivec3(1))) || glm::any(glm::greaterThanEqual(n, grid.dims - 1))) continue;
                    dilated[grid.index(n.x, n.y, n.z)] = surface;
                }
            }
            grid.label = std::move(dilated);
            (void)floodOutside(grid);
        }

        std::vector<glm::ivec3> occupied;
        for (int z = 0; z < grid.dims.z; ++z) for (int y = 0; y < grid.dims.y; ++y) for (int x = 0; x < grid.dims.x; ++x)
        {
            const auto l = grid.label[grid.index(x, y, z)];
            if (l == surface || l == interior) occupied.emplace_back(x, y, z);
        }
        if (occupied.empty()) throw std::runtime_error("Convex decomposition found no solid voxels");
        const float voxelVolume = grid.size * grid.size * grid.size;
        // Solid estimate: interior voxels count fully, surface voxels half (the surface runs
        // through them), which is unbiased for faces both flush with and across voxels.
        const auto solidVolume = [&](const std::vector<std::uint32_t>& voxels) {
            double count = 0;
            for (const auto i : voxels)
            {
                const auto v = occupied[i];
                count += grid.label[grid.index(v.x, v.y, v.z)] == interior ? 1.0 : 0.5;
            }
            return static_cast<float>(count) * voxelVolume;
        };
        // Re-fit a leaf to the exact input: input triangles clipped to the leaf's cut planes
        // (uncut sides are the grid bounds) plus corners of fully interior voxels. Polygons
        // lying entirely in a cut plane belong to the neighbouring solid and are dropped;
        // otherwise a face flush with the cut would stretch this piece across a notch.
        const auto refit = [&](Cell& leaf) {
            std::vector<std::uint32_t> solid;
            for (const auto i : leaf.voxels)
            {
                const auto v = occupied[i];
                if (grid.label[grid.index(v.x, v.y, v.z)] == interior) solid.push_back(i);
            }
            const auto boxLo = grid.origin + glm::vec3(leaf.lo) * grid.size;
            const auto boxHi = grid.origin + glm::vec3(leaf.hi) * grid.size;
            const float eps = grid.size * 1e-3f;
            leaf.points.clear();
            for (std::size_t i = 0; i < indices.size(); i += 3)
            {
                std::vector<glm::vec3> polygon{vertices[indices[i]], vertices[indices[i + 1]], vertices[indices[i + 2]]};
                glm::vec3 tlo, thi; bounds(polygon, tlo, thi);
                if (glm::any(glm::lessThan(thi, boxLo - eps)) || glm::any(glm::greaterThan(tlo, boxHi + eps))) continue;
                for (int q = 0; q < 3 && !polygon.empty(); ++q) { clipAxis(polygon, q, boxLo[q], true); clipAxis(polygon, q, boxHi[q], false); }
                if (polygon.empty()) continue;
                bool inCutPlane = false;
                for (int q = 0; q < 3 && !inCutPlane; ++q)
                    for (const auto [cut, plane] : {std::pair{leaf.lo[q] > 0, boxLo[q]}, std::pair{leaf.hi[q] < grid.dims[q], boxHi[q]}})
                        if (cut && std::ranges::all_of(polygon, [&](const glm::vec3& p) { return std::abs(p[q] - plane) <= eps; }))
                            inCutPlane = true;
                if (!inCutPlane) leaf.points.insert(leaf.points.end(), polygon.begin(), polygon.end());
            }
            for (const auto corner : rowCorners(occupied, solid, leaf.lo, leaf.hi))
                leaf.points.push_back(grid.origin + corner * grid.size);
            const float hull = convexHullVolume(leaf.points);
            leaf.excess = hull > 0 ? hull - solidVolume(leaf.voxels) : 0.0f;
        };

        std::vector<Cell> leaves(1);
        leaves[0].lo = glm::ivec3(0); leaves[0].hi = grid.dims;
        leaves[0].voxels.resize(occupied.size());
        for (std::uint32_t i = 0; i < occupied.size(); ++i) leaves[0].voxels[i] = i;
        const float tolerance = std::max(0.0f, settings.concavityTolerance) * solidVolume(leaves[0].voxels);
        const unsigned maxPieces = std::max(1u, settings.maxPieces);
        measure(leaves[0], occupied); refit(leaves[0]);
        std::vector<char> frozen(1, 0);
        while (leaves.size() < maxPieces)
        {
            std::size_t worst = leaves.size(); float worstExcess = tolerance;
            for (std::size_t i = 0; i < leaves.size(); ++i)
                if (!frozen[i] && leaves[i].excess > worstExcess) { worstExcess = leaves[i].excess; worst = i; }
            if (worst == leaves.size()) break;
            Cell left, right;
            if (!splitCell(leaves[worst], occupied, left, right)) { frozen[worst] = 1; continue; }
            refit(left); refit(right);
            leaves[worst] = std::move(left); frozen[worst] = 0;
            leaves.push_back(std::move(right)); frozen.push_back(0);
        }

        std::vector<std::vector<glm::vec3>> pieces;
        for (auto& leaf : leaves)
            if (convexHullVolume(leaf.points) > 0) pieces.push_back(std::move(leaf.points));
        if (pieces.empty()) pieces.emplace_back(vertices.begin(), vertices.end());
        return pieces;
    }
}
