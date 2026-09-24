#include "assets/esm_document.hpp"
#include "assets/esm_editor.hpp"
#include "assets/dem_import.hpp"
#include "editor/editor_document.hpp"
#include "world/world_scene.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <span>
#include <stdexcept>
#include <vector>

namespace
{
using Bytes = std::vector<std::byte>;
using namespace openf4::assets;
void require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
template<class F> void rejects(F fn, const char* message)
{
    bool rejected = false;
    try { fn(); } catch (const std::exception&) { rejected = true; }
    require(rejected, message);
}
constexpr std::uint32_t cc(const char* p)
{
    return std::uint32_t(p[0]) | std::uint32_t(p[1]) << 8 | std::uint32_t(p[2]) << 16 | std::uint32_t(p[3]) << 24;
}
template<class T> T get(std::span<const std::byte> bytes, std::size_t p)
{
    if (p + sizeof(T) > bytes.size()) throw std::runtime_error("Broken test ESM");
    T value; std::memcpy(&value, bytes.data() + p, sizeof(T)); return value;
}
template<class T> void put(Bytes& bytes, T value)
{
    auto* p = reinterpret_cast<const std::byte*>(&value); bytes.insert(bytes.end(), p, p + sizeof(T));
}
template<class T> void patch(Bytes& bytes, std::size_t p, T value) { std::memcpy(bytes.data() + p, &value, sizeof(T)); }
void append(Bytes& bytes, const Bytes& part) { bytes.insert(bytes.end(), part.begin(), part.end()); }
Bytes text(const char* value)
{
    const auto* p = reinterpret_cast<const std::byte*>(value); return {p, p + std::strlen(value) + 1};
}
Bytes sub(const char* type, const Bytes& payload)
{
    Bytes bytes; put(bytes, cc(type)); put(bytes, std::uint16_t(payload.size())); append(bytes, payload); return bytes;
}
Bytes record(const char* type, std::uint32_t id, const Bytes& body)
{
    Bytes bytes; put(bytes, cc(type)); put(bytes, std::uint32_t(body.size())); put(bytes, std::uint32_t{0}); put(bytes, id);
    put(bytes, std::uint32_t{0}); put(bytes, std::uint16_t{131}); put(bytes, std::uint16_t{0}); append(bytes, body); return bytes;
}
Bytes group(std::uint32_t label, std::uint32_t type, const Bytes& body)
{
    auto bytes = record("GRUP", type, body); patch(bytes, 4, std::uint32_t(bytes.size())); patch(bytes, 8, label); return bytes;
}
Bytes readFile(const std::filesystem::path& path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate); require(bool(file), "Cannot read fixture");
    Bytes bytes(static_cast<std::size_t>(file.tellg())); file.seekg(0); file.read(reinterpret_cast<char*>(bytes.data()), bytes.size()); return bytes;
}
void writeFile(const std::filesystem::path& path, const Bytes& bytes)
{
    std::ofstream file(path, std::ios::binary | std::ios::trunc); file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    require(bool(file), "Cannot write fixture");
}
using Edit = std::function<void(Bytes&)>;
Bytes rewrite(std::span<const std::byte> bytes, const Edit& edit)
{
    Bytes output;
    for (std::size_t p = 0; p < bytes.size();)
    {
        const bool isGroup = get<std::uint32_t>(bytes, p) == cc("GRUP");
        const std::size_t size = get<std::uint32_t>(bytes, p + 4) + (isGroup ? 0u : 24u);
        require(size >= 24 && size <= bytes.size() - p, "Invalid group size");
        Bytes current(bytes.begin() + p, bytes.begin() + p + 24);
        if (isGroup) append(current, rewrite(bytes.subspan(p + 24, size - 24), edit));
        else { current.insert(current.end(), bytes.begin() + p + 24, bytes.begin() + p + size); edit(current); }
        patch(current, 4, std::uint32_t(current.size() - (isGroup ? 0u : 24u))); append(output, current); p += size;
    }
    return output;
}
bool contains(const Bytes& bytes, const Bytes& needle)
{
    return std::search(bytes.begin(), bytes.end(), needle.begin(), needle.end()) != bytes.end();
}
glm::mat4 at(float x, float y = 0, float z = 0)
{
    PlacedReference reference; reference.position[0] = x; reference.position[1] = y; reference.position[2] = z;
    return openf4::world::bethesdaReferenceTransform(reference);
}
void testEditing(const std::filesystem::path& root)
{
    const auto path = root / "edit.esm";
    openf4::editor::EditorDocument editor(path);
    require(!editor.dirty(), "New document begins dirty");
    const auto model = editor.addModel("models/cube/source.glb", "original.glb");
    const auto id = editor.place(model, at(512, 512, 128), "Cube");
    require(editor.dirty(), "Placement did not dirty document");
    editor.save(); const auto initialBytes = readFile(path);
    require(!editor.dirty(), "Save did not clear dirty state");
    editor.beginTransaction(); editor.setTransform(id, at(9000, 512, 256)); editor.setTransform(id, at(13000, -5000, 256)); editor.endTransaction();
    require(editor.references()[0].placement.cell == CellCoord{3, -2}, "Movement did not migrate CELL");
    require(editor.undo() && editor.references()[0].placement.cell == CellCoord{0, 0}, "Transform transaction undo failed");
    require(!editor.dirty(), "Undo to saved state did not clear cached dirty flag");
    require(editor.redo() && editor.references()[0].placement.cell == CellCoord{3, -2}, "Transform transaction redo failed");
    require(editor.dirty(), "Redo did not invalidate cached dirty flag");
    const auto duplicate = editor.duplicate(id); editor.erase(id);
    require(editor.references().size() == 1 && editor.references()[0].placement.formId == duplicate, "Delete/duplicate failed");
    require(editor.undo() && editor.references().size() == 2, "Delete undo failed");
    editor.save();
    openf4::editor::EditorDocument reloaded(path);
    require(reloaded.references().size() == 2 && reloaded.models().contains(model), "STAT/REFR semantic reload failed");
    require(reloaded.references()[0].placement.formId == id, "Reference FormID changed");
    EsmDocument raw(path);
    require(raw.terrainCell({3, -2}).has_value(), "Reference migration did not create CELL/LAND ownership");
    require(raw.nextFormId() > duplicate && raw.nextFormId() == reloaded.nextFormId(), "FormID allocator regressed");
    auto terrain = loadEditableLandscape(path);
    terrain.height(48, 48) = 64;
    saveEditableLandscape(path, terrain);
    require(EsmDocument(path).references().size() == 2, "Terrain save erased references");
    require(loadEditableLandscape(path).height(48, 48) == 64, "Terrain edit did not persist");
    auto resident = loadEditableLandscape(path);
    reloaded.synchronizeTerrain(resident, false);
    require(!reloaded.dirty(), "Resident terrain adoption dirtied object document");
    resident.nextFormId = reloaded.nextFormId(); (void)resident.addCell({5, 5});
    reloaded.synchronizeTerrain(resident, false);
    require(reloaded.dirty(), "New camera-created CELL was incorrectly adopted as saved");
    reloaded.save();
    require(EsmDocument(path).terrainCell({5, 5}).has_value(), "New resident CELL did not save");
    auto sculpted = *reloaded.terrainSnapshot(); const auto beforeSculpt = sculpted.sample({0, 0});
    sculpted.setSample({0, 0}, beforeSculpt + 8); reloaded.synchronizeTerrain(sculpted);
    require(reloaded.undo() && reloaded.terrainSnapshot()->sample({0, 0}) == beforeSculpt, "Terrain transaction undo failed");
    require(reloaded.redo() && reloaded.terrainSnapshot()->sample({0, 0}) == beforeSculpt + 8, "Terrain transaction redo failed");
    reloaded.save();
    auto seam = loadEditableLandscape(path);
    require(seam.cell({-1, -1}).heights.back() == seam.cell({0, 0}).heights.front(), "Four-CELL corner diverged after document save");
}
void testRawPreservation(const std::filesystem::path& root)
{
    const auto path = root / "raw.esm";
    openf4::editor::EditorDocument editor(path);
    const auto model = editor.addModel("models/raw.glb"); const auto id = editor.place(model, at(128)); editor.save();
    const auto unknownSub = sub("ZZZZ", text("preserve unknown authored-record payload"));
    auto bytes = rewrite(readFile(path), [&](Bytes& record) {
        if (get<std::uint32_t>(record, 12) == id || get<std::uint32_t>(record, 12) == model
            || get<std::uint32_t>(record, 0) == cc("LAND")) append(record, unknownSub);
    });
    const auto unknownRecord = record("TEST", 0x01009000, sub("ABCD", text("opaque untouched record")));
    append(bytes, group(cc("TEST"), 0, unknownRecord)); writeFile(path, bytes);
    EsmDocument document(path); auto refs = document.references(); refs[0].placement.position[2] = 512;
    document.setReferences(refs); auto models = document.models(); models[0].path = "models/renamed.glb";
    document.setModels(models); document.save();
    auto edited = readFile(path);
    require(contains(edited, unknownRecord), "Unknown record was not preserved byte-for-byte");
    unsigned count = 0;
    (void)rewrite(edited, [&](Bytes& rec) { if (contains(rec, unknownSub)) ++count; });
    require(count == 11, "Unknown subrecords were lost on changed STAT/REFR/LAND records");
    auto terrain = loadEditableLandscape(path); terrain.height(12, 12) = 32; saveEditableLandscape(path, terrain);
    require(contains(readFile(path), unknownRecord), "Terrain save removed unknown top group");
    require(EsmDocument(path).references()[0].placement.position[2] == 512, "Terrain save reverted reference edit");
    auto next = EsmDocument(path).nextFormId(); require(next > 0x01009000, "Allocator ignored unknown local record IDs");
    EsmDocument failed(path); auto before = readFile(path); auto invalid = *failed.terrainCell({0, 0}); invalid.landscape.heights[500] = 1;
    failed.replaceTerrainCells(std::span(&invalid, 1));
    rejects([&] { failed.save(); }, "Unquantized terrain was accepted");
    require(readFile(path) == before, "Failed save changed original file");
    EsmDocument destinationFailure(path); destinationFailure.setPath(root); rejects([&] { destinationFailure.save(); }, "Directory destination was accepted");
    require(readFile(path) == before, "Atomic replacement failure changed source");
    for (const auto& entry : std::filesystem::directory_iterator(root.parent_path()))
        require(!entry.path().filename().string().starts_with(root.filename().string() + ".openf4-"), "Failed save left a temporary file");
}
void testMasterOverlay(const std::filesystem::path& root)
{
    const auto base = root / "protected.esm";
    saveEditableLandscape(base, createEditableLandscape());
    auto bytes = rewrite(readFile(base), [&](Bytes& rec) {
        if (get<std::uint32_t>(rec, 0) != cc("TES4")) return;
        auto marker = text("OpenF4"); auto found = std::search(rec.begin(), rec.end(), marker.begin(), marker.end());
        require(found != rec.end(), "Missing ownership marker"); *found = std::byte{'X'};
    }); writeFile(base, bytes);
    openf4::editor::EditorDocument editor(base);
    require(editor.path().filename() == "protected.openf4.esm", "Protected master did not use overlay");
    const auto model = editor.addModel("models/overlay.glb"); require((model >> 24) == 2, "Overlay allocated wrong self prefix");
    const auto ref = editor.place(model, at(128)); editor.save();
    require(readFile(base) == bytes, "Overlay changed protected master");
    const auto stack = loadEsmMasterStack(editor.path());
    require(std::filesystem::equivalent(stack.basePath, base), "Saved overlay lost base master selection");
    require(stack.references.size() == 1 && stack.references[0].placement.formId == ref, "Overlay stack did not load placed object");
    openf4::editor::EditorDocument reopened(editor.path());
    require(reopened.references().size() == 1 && !reopened.hasEditableTerrain(), "Overlay reload changed terrain ownership");
}
void testInheritedReferences(const std::filesystem::path& root)
{
    const auto base = root / "inherited.esm";
    openf4::editor::EditorDocument source(base);
    const auto model = source.addModel("models/inherited.glb"); const auto ref = source.place(model, at(128)); source.save();
    auto bytes = rewrite(readFile(base), [&](Bytes& rec) {
        if (get<std::uint32_t>(rec, 0) != cc("TES4")) return;
        const auto marker = text("OpenF4"); auto found = std::search(rec.begin(), rec.end(), marker.begin(), marker.end());
        require(found != rec.end(), "Missing ownership marker"); *found = std::byte{'X'};
    }); writeFile(base, bytes);
    openf4::editor::EditorDocument editor(base);
    require(editor.references().size() == 1, "Master reference was not inherited");
    editor.setTransform(ref, at(17000)); editor.save();
    auto stack = loadEsmMasterStack(editor.path());
    require(stack.references.size() == 1 && stack.references[0].placement.cell == CellCoord{4, 0}, "Overlay did not migrate inherited reference");
    editor.erase(ref); editor.save();
    require(loadEsmMasterStack(editor.path()).references.empty(), "Inherited reference deletion did not persist");
    require(readFile(base) == bytes, "Inherited reference edit changed protected master");
    require(editor.undo(), "Inherited deletion undo failed"); editor.save();
    require(loadEsmMasterStack(editor.path()).references.size() == 1, "Undo did not replace deletion tombstone");
}
void testSaveAs(const std::filesystem::path& root)
{
    const auto source = root / "save-as.esm"; const auto target = root / "saved-copy.esm";
    openf4::editor::EditorDocument editor(source);
    const auto model = editor.addModel("models/cube.glb"); (void)editor.place(model, at(128)); editor.save();
    const auto original = readFile(source);
    std::filesystem::create_directories(editor.assetRoot() / "models");
    writeFile(editor.assetRoot() / "models/cube.glb", text("asset bytes"));
    std::filesystem::create_directory(root / "save-as.btr"); writeFile(root / "save-as.btr/tile.btr", text("derived BTR bytes"));
    editor.saveAs(target);
    require(readFile(source) == original, "Save As changed original ESM");
    require(readFile(root / "saved-copy.assets/models/cube.glb") == text("asset bytes"), "Save As omitted asset sidecar");
    require(readFile(root / "saved-copy.btr/tile.btr") == text("derived BTR bytes"), "Save As omitted BTR sidecar");
    openf4::editor::EditorDocument reopened(target);
    require(reopened.references().size() == 1 && reopened.sourceEsm(model) == target.string(), "Save As lost source-relative asset ownership");
    rejects([&] { editor.saveAs(source); }, "Save As silently overwrote an existing ESM");
    require(editor.undo(), "Save As discarded undo history");
    require(editor.sourceEsm(model) == target.string(), "Save As undo restored old source ESM path");
    openf4::editor::EditorDocument overlay(root / "protected.esm");
    rejects([&] { overlay.saveAs(root / "relocated" / "overlay.esm"); }, "Save As silently broke local master dependencies");
}
void testDemHeader(const std::filesystem::path& root)
{
    const auto path = root / "dem-header.esm"; saveEditableLandscape(path, createEditableLandscape());
    auto bytes = rewrite(readFile(path), [&](Bytes& rec) {
        if (get<std::uint32_t>(rec, 0) != cc("TES4")) return;
        append(rec, sub("MAST", text("fixture.world.esm"))); append(rec, sub("DATA", Bytes(8)));
        append(rec, sub("MAST", text("fixture.terrain.r00.c00.esm"))); append(rec, sub("DATA", Bytes(8)));
        append(rec, sub("EMGI", text("OPENF4_DEM 1\nROLE OVERLAY\n")));
    }); writeFile(path, bytes);
    EsmDocument document(path);
    require(document.isDemOverlay(), "DEM metadata must be read from EMGI, not SNAM");
    require(DemLandscapeProject::recognizes(path), "New OpenF4 DEM metadata was not recognized");
    const auto id = document.allocateFormId(); require((id >> 24) == 3, "DEM allocation ignored master-table self prefix");
    document.save(); require(EsmDocument(path).nextFormId() == id + 1, "DEM allocator did not persist HEDR nextObjectId");
    bytes = readFile(path);
    const auto modern = text("OPENF4_DEM 1\nROLE OVERLAY\n");
    const auto legacy = text("EMPIRE_DEM 1\nROLE OVERLAY\n");
    const auto marker = std::search(bytes.begin(), bytes.end(), modern.begin(), modern.end());
    require(marker != bytes.end() && modern.size() == legacy.size(), "Missing modern DEM marker");
    std::copy(legacy.begin(), legacy.end(), marker); writeFile(path, bytes);
    require(DemLandscapeProject::recognizes(path) && EsmDocument(path).isDemOverlay(),
        "Legacy DEM metadata was not recognized");
    EsmDocument old(path); old.save();
    require(contains(readFile(path), legacy), "Saving a legacy DEM rewrote its format marker");
}

void testBrandingCompatibility(const std::filesystem::path& root)
{
    const auto path = root / "legacy-world.esm";
    saveEditableLandscape(path, createEditableLandscape());
    auto bytes = readFile(path);
    require(contains(bytes, text("OpenF4")) && contains(bytes, text("OpenF4World"))
        && contains(bytes, text("OpenF4Cell_0_0")), "New worlds did not use OpenF4 identifiers");
    for (const auto& [current, legacy] : std::array<std::pair<std::string, std::string>, 3>{{
        {"OpenF4", "EMPIRE"}, {"OpenF4World", "EmpireWorld"}, {"OpenF4Cell_", "EmpireCell_"}}})
    {
        // Equal-length legacy names let the fixture retain its exact record/group layout.
        require(current.size() == legacy.size(), "Branding fixture replacement changes sizes");
        const auto* first = reinterpret_cast<const std::byte*>(current.data());
        auto cursor = bytes.begin();
        while ((cursor = std::search(cursor, bytes.end(), first, first + current.size())) != bytes.end())
        {
            // Author replacement must not consume the prefix of World/Cell identifiers.
            if (current == "OpenF4" && cursor[current.size()] != std::byte{0}) { cursor += current.size(); continue; }
            std::memcpy(&*cursor, legacy.data(), legacy.size()); cursor += legacy.size();
        }
    }
    writeFile(path, bytes);
    require(EsmDocument(path).isOpenF4Owned(), "Legacy author marker lost ownership");
    auto terrain = loadEditableLandscape(path); terrain.setSample({0, 0}, 8);
    saveEditableLandscape(path, terrain);
    openf4::editor::EditorDocument editor(path);
    require(editor.path() == std::filesystem::absolute(path), "Legacy world incorrectly became protected");
    const auto model = editor.addModel("models/legacy.glb"); (void)editor.place(model, at(128)); editor.save();
    const auto saved = readFile(path);
    require(contains(saved, text("EMPIRE")) && contains(saved, text("EmpireWorld"))
        && contains(saved, text("EmpireCell_0_0")), "Ordinary save migrated legacy identifiers");
    require(loadEditableLandscape(path).sample({0, 0}) == 8
        && EsmDocument(path).references().size() == 1, "Legacy terrain/object round-trip failed");
}
}

int main()
try
{
    const auto root = std::filesystem::temp_directory_path() / ("openf4-editor-doc-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directory(root);
    testEditing(root); testRawPreservation(root); testMasterOverlay(root); testInheritedReferences(root); testSaveAs(root); testDemHeader(root); testBrandingCompatibility(root);
    std::filesystem::remove_all(root);
    std::cout << "Editor document tests passed\n"; return 0;
}
catch (const std::exception& error)
{
    std::cerr << error.what() << '\n'; return 1;
}
