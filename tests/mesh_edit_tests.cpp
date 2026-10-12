// PROTOTYPE 0.1 mesh editing: pure topology/transform checks plus AssetDocument undo.
#include "editor/asset_document.hpp"
#include "editor/mesh_edit.hpp"
#include "editor/editor_document.hpp"

#include <array>
#include <chrono>
#include <filesystem>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <glm/ext/matrix_transform.hpp>

namespace
{
    using namespace openf4;
    using editor::mesh::Element;
    void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
    bool near(glm::vec3 a, glm::vec3 b) { return glm::length(a - b) < 1e-3f; }
    template<class F> void rejects(F operation, const char* message)
    {
        try { operation(); } catch (const std::exception&) { return; }
        throw std::runtime_error(message);
    }

    void testCubeTopology()
    {
        const auto cube = editor::mesh::makeCube({10, 20, 30}, 2);
        require(cube.vertices.size() == 24 && cube.indices.size() == 36, "Cube has wrong vertex/index count");
        const auto topology = editor::mesh::buildTopology(cube);
        require(topology.points.size() == 8, "Cube corners were not welded");
        require(topology.edges.size() == 18 && topology.triangles.size() == 12, "Cube edges/triangles wrong");
        // Engine winding: outward normal = cross(c - a, b - a), and must match stored normals.
        for (std::size_t i = 0; i < cube.indices.size(); i += 3)
        {
            const auto& a = cube.vertices[cube.indices[i]]; const auto& b = cube.vertices[cube.indices[i + 1]];
            const auto& c = cube.vertices[cube.indices[i + 2]];
            const auto n = glm::normalize(glm::cross(c.position - a.position, b.position - a.position));
            require(glm::dot(n, a.normal) > 0.99f, "Cube triangle winding disagrees with its normal");
            require(glm::dot(a.position - glm::vec3(10, 20, 30), a.normal) > 0, "Cube normal points inward");
        }
        require(near(editor::mesh::centroid(topology, std::vector<std::uint32_t>{0,1,2,3,4,5,6,7}), {10, 20, 30}),
            "Cube centroid wrong");
        const std::vector<std::uint32_t> edge{0};
        require(editor::mesh::selectedPoints(topology, Element::edge, edge).size() == 2, "Edge selects two points");
        const std::vector<std::uint32_t> triangle{0};
        require(editor::mesh::selectedPoints(topology, Element::triangle, triangle).size() == 3, "Triangle selects three points");
    }

    void testTransformMovesWeldedCopies()
    {
        auto cube = editor::mesh::makeCube({}, 2);
        const auto topology = editor::mesh::buildTopology(cube);
        std::vector<glm::vec3> start;
        for (const auto& v : cube.vertices) start.push_back(v.position);
        const std::vector<std::uint32_t> point{topology.vertexPoint[0]};
        const auto moved = editor::mesh::transformPoints(topology, start, point,
            glm::translate(glm::mat4(1), glm::vec3(0, 5, 0)));
        int copies = 0;
        for (std::size_t i = 0; i < moved.size(); ++i)
        {
            if (topology.vertexPoint[i] == point[0]) { ++copies; require(near(moved[i], start[i] + glm::vec3(0, 5, 0)), "Copy not moved"); }
            else require(moved[i] == start[i], "Unselected vertex moved");
        }
        require(copies == 3, "Cube corner should have three split copies");
        editor::mesh::applyPositions(cube, moved);
        for (const auto& v : cube.vertices) require(std::abs(glm::length(v.normal) - 1) < 1e-4f, "Normals not unit length");

        // Scale about a pivot: whole cube doubled about its centre.
        std::vector<std::uint32_t> all; for (std::uint32_t i = 0; i < 8; ++i) all.push_back(i);
        const auto scaled = editor::mesh::transformPoints(topology, start, all, glm::scale(glm::mat4(1), glm::vec3(2)));
        for (std::size_t i = 0; i < scaled.size(); ++i) require(near(scaled[i], start[i] * 2.0f), "Scale wrong");
    }

    void testPicking()
    {
        const auto cube = editor::mesh::makeCube({}, 2);
        const auto topology = editor::mesh::buildTopology(cube);
        // Orthographic top-down "screen": x,z -> pixels, depth = -y (top is closer).
        std::vector<editor::mesh::ScreenPoint> screen;
        for (const auto& p : topology.points) screen.push_back({{p.x * 100 + 200, p.z * 100 + 200}, 10 - p.y, true});
        const auto vertex = editor::mesh::pickElement(topology, Element::vertex, screen, {302, 301}, 8);
        require(vertex && topology.points[*vertex].y > 0, "Vertex pick should prefer the nearer (top) corner");
        require(!editor::mesh::pickElement(topology, Element::vertex, screen, {250, 250}, 8), "Vertex pick too generous");
        const auto edge = editor::mesh::pickElement(topology, Element::edge, screen, {200, 101}, 8);
        require(edge.has_value(), "Edge pick missed");
        const auto triangle = editor::mesh::pickElement(topology, Element::triangle, screen, {230, 260}, 0);
        require(triangle.has_value(), "Triangle pick missed");
        for (int k = 0; k < 3; ++k) require(topology.points[topology.triangles[*triangle][k]].y > 0, "Triangle pick should hit the top face");
    }

    void testDocumentUndo()
    {
        assets::ModelGeometry geometry; geometry.surfaces.push_back(editor::mesh::makeCube({}, 140));
        editor::AssetDocument asset(std::move(geometry));
        require(asset.meshEditable(), "New geometry should be editable");
        const auto original = asset.geometryHandle();
        editor::AssetMaterial material; material.baseColor = {0.5f, 0.5f, 0.5f, 1};
        const auto index = asset.addSurface(editor::mesh::makeCube({200, 0, 0}, 70), material);
        require(index == 1 && asset.materials().size() == 2 && asset.sourceGeometry().surfaces.size() == 2, "Cube not added");
        const auto topology = editor::mesh::buildTopology(asset.sourceGeometry().surfaces[1]);
        std::vector<glm::vec3> start; for (const auto& v : asset.sourceGeometry().surfaces[1].vertices) start.push_back(v.position);
        asset.beginTransaction();
        for (int step = 1; step <= 3; ++step) // a gizmo drag commits every frame inside one transaction
            asset.setSurfacePositions(1, editor::mesh::transformPoints(topology, start,
                std::vector<std::uint32_t>{0}, glm::translate(glm::mat4(1), glm::vec3(0, float(step), 0))));
        asset.endTransaction();
        require(asset.sourceGeometry().surfaces[1].vertices[0].position != start[0], "Vertex was not moved");
        require(asset.undo() && asset.sourceGeometry().surfaces[1].vertices[0].position == start[0], "Drag is not one undo step");
        require(asset.undo() && asset.geometryHandle() == original && asset.materials().size() == 1, "Cube add not undone");
        require(asset.redo() && asset.redo() && asset.sourceGeometry().surfaces.size() == 2, "Redo failed");
        const auto exported = asset.geometryForExport();
        require(exported.surfaces.size() == 2 && exported.surfaces[1].baseColor == material.baseColor, "Export lost the cube");
    }

    void testEmptyAssetAndSurfaceDelete()
    {
        auto blank = editor::AssetDocument::createEmpty();
        require(blank.sourceGeometry().surfaces.empty() && blank.materials().empty() && blank.meshEditable() && !blank.dirty(),
            "Empty asset should start blank, editable and clean");
        require(blank.collisionSystems().size() == 1 && blank.collisionSystems()[0].physics.bodies.size() == 1 && !blank.movable(),
            "Empty asset should carry one empty static body");
        rejects([] { (void)editor::AssetDocument(assets::ModelGeometry{}); }, "Empty geometry accepted without allowEmpty");
        const auto root = std::filesystem::temp_directory_path()
            / ("openf4-mesh-edit-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(root);
        editor::EditorDocument world(root / "blank.esm"); world.setStartAwakeByDefault(true);
        rejects([&] { (void)blank.saveNative(world.path(), world.nextFormId()); }, "Saved an asset with no surfaces");
        editor::AssetMaterial grey, red; grey.baseColor = {0.5f, 0.5f, 0.5f, 1}; red.baseColor = {1, 0, 0, 1};
        require(blank.addSurface(editor::mesh::makeCube({}, 70), grey) == 0
            && blank.addSurface(editor::mesh::makeCube({200, 0, 0}, 70), red) == 1 && blank.dirty(), "Cubes not added to blank asset");
        blank.eraseSurface(0);
        require(blank.sourceGeometry().surfaces.size() == 1 && blank.materials().size() == 1 && blank.materials()[0] == red
            && near(blank.sourceGeometry().surfaces[0].vertices[0].position, editor::mesh::makeCube({200, 0, 0}, 70).vertices[0].position),
            "Surface delete removed the wrong surface/material");
        require(blank.undo() && blank.sourceGeometry().surfaces.size() == 2 && blank.materials()[0] == grey, "Surface delete not undoable");
        require(blank.redo() && blank.sourceGeometry().surfaces.size() == 1, "Surface delete redo failed");
        rejects([&] { blank.eraseSurface(4); }, "Deleted a missing surface");
        blank.eraseSurface(0);
        require(blank.sourceGeometry().surfaces.empty(), "Last surface was not deletable");
        require(blank.undo(), "Undo of last-surface delete failed");
        // First save of a blank asset publishes a package and adds a new world base (MSTT -> awake).
        blank.generateCollision(std::array<std::size_t, 1>{0}, true); blank.setMovable(true, 5);
        const auto expected = world.nextFormId();
        const auto package = blank.saveNative(world.path(), expected);
        const auto id = world.addNativeModel(package.modelPath, assets::ESMModelKind::movableStatic);
        require(id == expected && world.model(id) && world.model(id)->kind == assets::ESMModelKind::movableStatic
            && world.model(id)->path == package.modelPath, "Blank asset did not become a new MSTT base");
        require(assets::papyrusScriptActive(world.model(id)->scripts, {}, assets::wakeOnLoadScriptName), "New MSTT from blank asset is not awake");
        rejects([&] { (void)world.addNativeModel("meshes/missing.nif", assets::ESMModelKind::staticObject); }, "Unpublished package accepted");
        std::error_code ignored; std::filesystem::remove_all(root, ignored);
    }
}

int main()
{
    try
    {
        testCubeTopology();
        testTransformMovesWeldedCopies();
        testPicking();
        testDocumentUndo();
        testEmptyAssetAndSurfaceDelete();
        std::cout << "Mesh edit tests passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
