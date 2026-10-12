// PROTOTYPE convex collision editing: element transforms on convex parts, non-convex
// detection, and approximate convex decomposition ("Split optimally").
#include "editor/asset_document.hpp"
#include "editor/mesh_edit.hpp"
#include "physics/convex_decomposition.hpp"
#include "physics/jolt_collision.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <glm/ext/matrix_transform.hpp>

namespace
{
    using namespace openf4;
    using editor::mesh::Element;
    void require(bool value, const std::string& message) { if (!value) throw std::runtime_error(message); }
    bool near(glm::vec3 a, glm::vec3 b, float tolerance = 1e-2f) { return glm::length(a - b) < tolerance; }

    struct Mesh { std::vector<glm::vec3> vertices; std::vector<std::uint32_t> indices; };

    // Closed mesh of the boundary faces of unit cells scaled by `size`.
    Mesh blocks(const std::vector<glm::ivec3>& cells, float size)
    {
        std::set<std::tuple<int, int, int>> filled;
        for (const auto c : cells) filled.emplace(c.x, c.y, c.z);
        Mesh mesh;
        for (const auto c : cells)
            for (int axis = 0; axis < 3; ++axis)
                for (int sign : {-1, 1})
                {
                    auto n = c; n[axis] += sign;
                    if (filled.contains({n.x, n.y, n.z})) continue;
                    const int u = (axis + 1) % 3, v = (axis + 2) % 3;
                    const auto base = static_cast<std::uint32_t>(mesh.vertices.size());
                    for (const auto [du, dv] : std::array<std::pair<int, int>, 4>{{{0, 0}, {1, 0}, {1, 1}, {0, 1}}})
                    {
                        glm::vec3 p(c);
                        p[axis] += sign > 0 ? 1.0f : 0.0f; p[u] += float(du); p[v] += float(dv);
                        mesh.vertices.push_back(p * size);
                    }
                    for (auto i : {0u, 1u, 2u, 0u, 2u, 3u}) mesh.indices.push_back(base + i);
                }
        return mesh;
    }

    float meshVolume(const std::vector<glm::vec3>& vertices, const std::vector<std::uint32_t>& indices)
    {
        double volume = 0;
        for (std::size_t i = 0; i + 2 < indices.size(); i += 3)
            volume += glm::dot(glm::dvec3(vertices[indices[i]]),
                glm::cross(glm::dvec3(vertices[indices[i + 1]]), glm::dvec3(vertices[indices[i + 2]]))) / 6;
        return static_cast<float>(std::abs(volume));
    }

    bool insideHull(const std::vector<glm::vec3>& hull, glm::vec3 point)
    {
        auto with = hull; with.push_back(point);
        const float before = physics::convexHullVolume(hull);
        return physics::convexHullVolume(with) <= before * 1.001f;
    }

    assets::ModelSurface surfaceOf(const Mesh& mesh)
    {
        assets::ModelSurface surface;
        for (const auto p : mesh.vertices) { render::SourceVertex v; v.position = p; surface.vertices.push_back(v); }
        surface.indices = mesh.indices;
        return surface;
    }

    void testConvexityDetection()
    {
        const auto cube = blocks({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}, {0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}}, 25);
        const auto cubeReport = physics::measureConvexity(cube.vertices, cube.indices);
        require(cubeReport.convex && cubeReport.depth < 0.01f, "Cube reported as non-convex");
        require(std::abs(cubeReport.hullVolume - 125000.0f) < 1.0f, "Cube hull volume wrong");
        const auto ell = blocks({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}, 50);
        const auto ellReport = physics::measureConvexity(ell.vertices, ell.indices);
        require(!ellReport.convex && ellReport.depth > 10.0f, "L-shape not detected as non-convex");
        const auto pieces = physics::decomposeConvex(cube.vertices, cube.indices);
        require(pieces.size() == 1, "Convex cube was split");
    }

    void checkPieces(const std::vector<std::vector<glm::vec3>>& pieces, float expectedVolume, glm::vec3 notch, const char* name)
    {
        require(pieces.size() >= 2, std::string(name) + ": expected at least two pieces, got " + std::to_string(pieces.size()));
        float total = 0;
        for (const auto& piece : pieces)
        {
            const auto hull = physics::makeHavokConvexHull(piece);
            require(physics::polytopeIsConvex(hull.vertices, hull.indices), std::string(name) + ": piece is not convex");
            const float volume = physics::convexHullVolume(piece);
            require(volume > 0, std::string(name) + ": empty piece");
            total += volume;
            require(!insideHull(piece, notch), std::string(name) + ": a piece fills the concavity");
        }
        require(std::abs(total - expectedVolume) < expectedVolume * 0.15f, std::string(name) + ": pieces volume "
            + std::to_string(total) + " vs solid " + std::to_string(expectedVolume));
    }

    void testDecomposeL()
    {
        const auto ell = blocks({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}, 50);
        const auto pieces = physics::decomposeConvex(ell.vertices, ell.indices);
        checkPieces(pieces, 3 * 50.0f * 50 * 50, {75, 75, 25}, "L");
        // A U-shape needs at least three pieces.
        const auto u = blocks({{0, 0, 0}, {1, 0, 0}, {2, 0, 0}, {0, 1, 0}, {2, 1, 0}}, 40);
        const auto uPieces = physics::decomposeConvex(u.vertices, u.indices);
        checkPieces(uPieces, 5 * 40.0f * 40 * 40, {60, 60, 20}, "U");
        require(uPieces.size() >= 3, "U-shape split into fewer than three pieces");
        std::cout << "  L -> " << pieces.size() << " pieces, U -> " << uPieces.size() << " pieces\n";
        physics::ConvexDecompositionSettings capped; capped.maxPieces = 2;
        require(physics::decomposeConvex(u.vertices, u.indices, capped).size() <= 2, "maxPieces ignored");
    }

    void testConvexElementTransforms()
    {
        auto asset = editor::AssetDocument::createEmpty();
        asset.addBox({0, 0, 0}, {35, 35, 35});
        auto positions = asset.convexPartPositions(0);
        require(positions.size() == 8, "Box part should have 8 vertices");
        for (const auto p : positions) require(near(glm::abs(p), glm::vec3(35)), "Box vertex not at a corner");
        const auto surface = [&] {
            assets::ModelSurface s;
            for (const auto p : asset.convexPartPositions(0)) { render::SourceVertex v; v.position = p; s.vertices.push_back(v); }
            s.indices = asset.convexPartIndices(0);
            return s;
        };
        auto topology = editor::mesh::buildTopology(surface());
        require(topology.points.size() == 8 && topology.triangles.size() == 12 && topology.edges.size() == 18, "Convex box topology wrong");

        // Triangle mode: the two top triangles drive the four top vertices.
        std::vector<std::uint32_t> top;
        for (std::uint32_t t = 0; t < topology.triangles.size(); ++t)
        {
            const auto& tri = topology.triangles[t];
            if (topology.points[tri.x].y > 34 && topology.points[tri.y].y > 34 && topology.points[tri.z].y > 34) top.push_back(t);
        }
        require(top.size() == 2, "Expected two top triangles");
        const auto topPoints = editor::mesh::selectedPoints(topology, Element::triangle, top);
        require(topPoints.size() == 4, "Top triangles should select four points");
        asset.setConvexPartPositions(0, editor::mesh::transformPoints(topology, positions, topPoints,
            glm::translate(glm::mat4(1), {0, 20, 0})));
        float maxY = -1e9f;
        for (const auto p : asset.convexPartPositions(0)) maxY = std::max(maxY, p.y);
        require(std::abs(maxY - 55) < 1e-2f && asset.convexPartIsConvex(0) && asset.collisionConvex(), "Top face move failed");
        require(asset.undo(), "Convex vertex move not undoable");
        for (const auto p : asset.convexPartPositions(0)) require(p.y < 35.01f, "Undo did not restore convex vertices");
        require(asset.redo(), "Convex vertex move redo failed");

        // Vertex mode: scale all about the centroid, then rotate them.
        positions = asset.convexPartPositions(0);
        topology = editor::mesh::buildTopology(surface());
        std::vector<std::uint32_t> all(topology.points.size());
        for (std::uint32_t i = 0; i < all.size(); ++i) all[i] = i;
        const auto c = editor::mesh::centroid(topology, all);
        const auto scale = glm::translate(glm::mat4(1), c) * glm::scale(glm::mat4(1), glm::vec3(2)) * glm::translate(glm::mat4(1), -c);
        asset.setConvexPartPositions(0, editor::mesh::transformPoints(topology, positions, all, scale));
        const auto scaled = asset.convexPartPositions(0);
        for (std::size_t i = 0; i < scaled.size(); ++i)
            require(near(scaled[i], glm::vec3(scale * glm::vec4(positions[i], 1)), 0.05f), "Scaled convex vertex wrong");
        const auto rotate = glm::rotate(glm::mat4(1), glm::radians(30.0f), glm::vec3(0, 1, 0));
        asset.setConvexPartPositions(0, editor::mesh::transformPoints(topology, scaled, all, rotate));
        require(near(asset.convexPartPositions(0)[0], glm::vec3(rotate * glm::vec4(scaled[0], 1)), 0.05f)
            && asset.convexPartIsConvex(0), "Rotated convex vertex wrong");

        // Edge mode: move one edge outward along its normal-ish direction keeps the part valid.
        positions = asset.convexPartPositions(0);
        topology = editor::mesh::buildTopology(surface());
        require(editor::mesh::selectedPoints(topology, Element::edge, std::array<std::uint32_t, 1>{0}).size() == 2, "Edge selects two points");
    }

    void testNonConvexEditAndSplit()
    {
        auto asset = editor::AssetDocument::createEmpty();
        asset.addBox({0, 0, 0}, {35, 35, 35});
        auto positions = asset.convexPartPositions(0);
        std::size_t corner = 0;
        for (std::size_t i = 0; i < positions.size(); ++i) if (positions[i].x > 0 && positions[i].y > 0 && positions[i].z > 0) corner = i;
        positions[corner] = {0, 0, 0}; // push the corner into the centre: a dented box
        asset.setConvexPartPositions(0, positions);
        require(!asset.convexPartIsConvex(0) && !asset.collisionConvex(), "Dented box still reported convex");
        const auto report = asset.convexPartConvexity(0);
        require(!report.convex && report.depth > 1.0f, "Dent depth not reported");
        try { (void)asset.saveNative("unused.esm", 0x800); throw std::logic_error("saved"); }
        catch (const std::logic_error&) { throw std::runtime_error("Non-convex part was saved"); }
        catch (const std::runtime_error& error)
        { require(std::string(error.what()).find("not convex") != std::string::npos, std::string("Unexpected save error: ") + error.what()); }
        const float dentedVolume = meshVolume(asset.convexPartPositions(0), asset.convexPartIndices(0));
        require(dentedVolume < 70.0f * 70 * 70 * 0.97f, "Dent did not remove volume");

        const auto count = asset.splitConvexPart(0);
        const auto& parts = asset.collisionSystems()[0].physics.bodies[0].parts;
        require(count >= 2 && parts.size() == count, "Split did not produce multiple parts");
        float total = 0;
        for (std::size_t i = 0; i < parts.size(); ++i)
        {
            require(asset.convexPartIsConvex(i), "Split produced a non-convex piece");
            total += meshVolume(asset.convexPartPositions(i), asset.convexPartIndices(i));
        }
        require(asset.collisionConvex(), "Collision still non-convex after split");
        std::cout << "  dented box -> " << count << " pieces, volume " << total << " vs " << dentedVolume << '\n';
        require(std::abs(total - dentedVolume) < dentedVolume * 0.15f,
            "Split pieces volume " + std::to_string(total) + " vs dented " + std::to_string(dentedVolume));
        require(asset.undo() && asset.collisionSystems()[0].physics.bodies[0].parts.size() == 1 && !asset.convexPartIsConvex(0),
            "Split not undoable");
        asset.rehullConvexPart(0);
        require(asset.convexPartIsConvex(0) && asset.convexPartPositions(0).size() == 7, "Re-hull should drop the dented vertex");
    }

    void testSurfaceDecomposition()
    {
        const auto ell = blocks({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}, 50);
        assets::ModelGeometry geometry; geometry.surfaces.push_back(surfaceOf(ell));
        editor::AssetDocument asset(std::move(geometry));
        const std::array<std::size_t, 1> selected{0};
        require(!asset.surfaceConvexity(selected).convex, "L surface not flagged non-convex");
        const auto count = asset.generateConvexDecomposition(selected);
        const auto& parts = asset.collisionSystems()[0].physics.bodies[0].parts;
        require(count >= 2 && parts.size() == count, "Surface decomposition produced one piece");
        for (std::size_t i = 0; i < count; ++i) require(asset.convexPartIsConvex(i), "Surface decomposition piece not convex");
        std::cout << "  L surface -> " << count << " convex parts\n";
        asset.setMovable(true, 10);
        require(asset.movable() && asset.collisionConvex(), "Decomposed collision cannot become movable");
    }
}

int main()
{
    try
    {
        testConvexityDetection();
        testDecomposeL();
        testConvexElementTransforms();
        testNonConvexEditAndSplit();
        testSurfaceDecomposition();
        std::cout << "Collision edit tests passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
