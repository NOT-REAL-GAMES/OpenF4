// Actors groundwork: NPC_ indexing, ACHR round-trip through EsmDocument, ACHR overrides and
// new ACHRs in an active plugin with stable FormIDs, OpenF4-world persistence, placeholder mesh.
#include "assets/esm_document.hpp"
#include "assets/fo4_world_writer.hpp"
#include "assets/load_order.hpp"
#include "editor/editor_document.hpp"
#include "world/world_scene.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
using Bytes = std::vector<std::byte>;
using namespace openf4::assets;
using openf4::editor::EditorDocument;
using openf4::editor::EditorOpenOptions;

void require(bool condition, const std::string& message) { if (!condition) throw std::runtime_error(message); }
template<class F> void rejects(F fn, const std::string& message)
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
    if (p + sizeof(T) > bytes.size()) throw std::runtime_error("Broken test plugin");
    T value; std::memcpy(&value, bytes.data() + p, sizeof(T)); return value;
}
template<class T> void put(Bytes& bytes, T value)
{
    auto* p = reinterpret_cast<const std::byte*>(&value); bytes.insert(bytes.end(), p, p + sizeof(T));
}
void append(Bytes& bytes, const Bytes& part) { bytes.insert(bytes.end(), part.begin(), part.end()); }
Bytes text(const std::string& value)
{
    const auto* p = reinterpret_cast<const std::byte*>(value.c_str()); return {p, p + value.size() + 1};
}
Bytes sub(const char* type, const Bytes& payload)
{
    Bytes bytes; put(bytes, cc(type)); put(bytes, std::uint16_t(payload.size())); append(bytes, payload); return bytes;
}
Bytes record(const char* type, std::uint32_t id, const Bytes& body, std::uint32_t flags = 0)
{
    Bytes bytes; put(bytes, cc(type)); put(bytes, std::uint32_t(body.size())); put(bytes, flags); put(bytes, id);
    put(bytes, std::uint32_t{0x0BADF00D}); put(bytes, std::uint16_t{131}); put(bytes, std::uint16_t{7}); append(bytes, body); return bytes;
}
Bytes group(std::uint32_t label, std::uint32_t type, const Bytes& body)
{
    Bytes bytes; put(bytes, cc("GRUP")); put(bytes, std::uint32_t(body.size() + 24)); put(bytes, label); put(bytes, type);
    put(bytes, std::uint64_t{0}); append(bytes, body); return bytes;
}
Bytes readFile(const std::filesystem::path& path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate); require(bool(file), "Cannot read " + path.string());
    Bytes bytes(static_cast<std::size_t>(file.tellg())); file.seekg(0);
    file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size())); return bytes;
}
void writeFile(const std::filesystem::path& path, const Bytes& bytes)
{
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    require(bool(file), "Cannot write fixture");
}
// Every record (not group) in a file, by FormID, as raw bytes including the header.
void eachRecord(std::span<const std::byte> bytes, const std::function<void(std::span<const std::byte>)>& visit)
{
    for (std::size_t p = 0; p < bytes.size();)
    {
        const bool isGroup = get<std::uint32_t>(bytes, p) == cc("GRUP");
        const std::size_t size = get<std::uint32_t>(bytes, p + 4) + (isGroup ? 0u : 24u);
        require(size >= 24 && size <= bytes.size() - p, "Invalid size in test plugin");
        if (isGroup) eachRecord(bytes.subspan(p + 24, size - 24), visit); else visit(bytes.subspan(p, size));
        p += size;
    }
}
Bytes findRecord(const Bytes& bytes, std::uint32_t id, std::uint32_t type = 0)
{
    Bytes result;
    eachRecord(bytes, [&](std::span<const std::byte> r) {
        if (get<std::uint32_t>(r, 12) == id && (!type || get<std::uint32_t>(r, 0) == type)) result.assign(r.begin(), r.end()); });
    return result;
}
bool contains(const Bytes& bytes, const Bytes& needle)
{
    return std::search(bytes.begin(), bytes.end(), needle.begin(), needle.end()) != bytes.end();
}
Bytes tes4(std::uint32_t flags, const std::vector<std::string>& masters, const std::string& author, std::uint32_t next = 0x1000)
{
    Bytes body, hedr; put(hedr, 1.0f); put(hedr, std::uint32_t{0}); put(hedr, next);
    append(body, sub("HEDR", hedr)); append(body, sub("CNAM", text(author)));
    for (const auto& m : masters) { append(body, sub("MAST", text(m))); append(body, sub("DATA", Bytes(8))); }
    return record("TES4", 0, body, flags);
}
Bytes refr(std::uint32_t id, std::uint32_t base, float x, float y)
{
    Bytes body, data; append(body, sub("NAME", [&] { Bytes b; put(b, base); return b; }()));
    put(data, x); put(data, y); put(data, 0.0f); put(data, 0.0f); put(data, 0.0f); put(data, 0.0f);
    append(body, sub("DATA", data)); append(body, sub("XZZZ", text("unknown reference payload")));
    return record("REFR", id, body);
}
Bytes cellTree(std::uint32_t cell, std::uint32_t land, const Bytes& references, const char* name)
{
    Bytes cellBody; append(cellBody, sub("EDID", text(name)));
    Bytes xclc; put(xclc, std::int32_t{0}); put(xclc, std::int32_t{0}); put(xclc, std::uint32_t{0});
    append(cellBody, sub("XCLC", xclc)); append(cellBody, sub("ZCEL", text("unknown cell data")));
    Bytes children;
    if (land) append(children, record("LAND", land, sub("VHGT", Bytes(1096))));
    append(children, references);
    Bytes tree = record("CELL", cell, cellBody);
    append(tree, group(cell, 6, group(cell, 9, children)));
    return group(0, 4, group(0, 5, tree));
}
const Bytes achrUnknown = sub("XZZZ", text("unknown actor payload"));
Bytes u32(std::uint32_t value) { Bytes b; put(b, value); return b; }
Bytes achr(std::uint32_t id, std::uint32_t base, float x, float y)
{
    Bytes body, data; append(body, sub("EDID", text("SettlerRef")));
    append(body, sub("NAME", u32(base)));
    put(data, x); put(data, y); put(data, 0.0f); put(data, 0.0f); put(data, 0.0f); put(data, 1.5f);
    append(body, sub("DATA", data)); append(body, achrUnknown);
    return record("ACHR", id, body);
}
// Master.esm: RACE 0x920, NPC_ 0x910, STAT 0x900; Commonwealth cell with REFR 0xB00 + ACHR 0xB10.
std::filesystem::path makeActorMaster(const std::filesystem::path& root)
{
    Bytes file = tes4(0x1, {}, "Bethesda");
    Bytes statBody; append(statBody, sub("EDID", text("Rock01"))); append(statBody, sub("MODL", text("rocks\\rock01.nif")));
    append(file, group(cc("STAT"), 0, record("STAT", 0x900, statBody)));
    append(file, group(cc("RACE"), 0, record("RACE", 0x920, sub("EDID", text("HumanRace")))));
    Bytes npc; append(npc, sub("EDID", text("Settler"))); append(npc, sub("FULL", text("Settler")));
    append(npc, sub("RNAM", u32(0x920))); append(npc, sub("ZNPC", text("unknown npc data")));
    append(file, group(cc("NPC_"), 0, record("NPC_", 0x910, npc)));
    Bytes worlds;
    Bytes wrld; append(wrld, sub("EDID", text("Commonwealth")));
    append(worlds, record("WRLD", 0x3C, wrld));
    Bytes refs; append(refs, refr(0xB00, 0x900, 100, 100)); append(refs, achr(0xB10, 0x910, 200, 200));
    append(worlds, group(0x3C, 1, cellTree(0xA00, 0xA01, refs, "CommonwealthCell")));
    append(file, group(cc("WRLD"), 0, worlds));
    const auto path = root / "Master.esm"; writeFile(path, file); return path;
}
glm::mat4 at(float x, float y = 0, float z = 0)
{
    PlacedReference reference; reference.position[0] = x; reference.position[1] = y; reference.position[2] = z;
    return openf4::world::bethesdaReferenceTransform(reference);
}
EditorOpenOptions direct(const LoadOrder& order, std::uint32_t world)
{
    EditorOpenOptions options; options.directEdit = true; options.worldFormId = world;
    for (const auto& e : order.entries()) options.resolvedFiles.push_back(e.header.path);
    std::string reason; options.writable = order.canWrite(order.activeEntry()->header.path, &reason); options.readOnlyReason = reason;
    return options;
}
const openf4::editor::EditorReference* findRef(const EditorDocument& document, std::uint32_t id)
{
    const auto found = std::ranges::find_if(document.references(), [&](const auto& r) { return r.placement.formId == id; });
    return found == document.references().end() ? nullptr : &*found;
}
const ESMReference* findRef(const std::vector<ESMReference>& refs, std::uint32_t id)
{
    const auto found = std::ranges::find_if(refs, [&](const auto& r) { return r.placement.formId == id; });
    return found == refs.end() ? nullptr : &*found;
}
std::uint32_t subU32(const Bytes& rec, const char* type)
{
    for (std::size_t p = 24; p + 6 <= rec.size();)
    {
        const auto t = get<std::uint32_t>(rec, p); const auto size = get<std::uint16_t>(rec, p + 4);
        if (t == cc(type)) return get<std::uint32_t>(rec, p + 6);
        p += 6 + size;
    }
    throw std::runtime_error(std::string("Missing subrecord ") + type);
}

void testDocumentRoundTrip(const std::filesystem::path& root)
{
    const auto master = makeActorMaster(root);
    const auto copy = root / "Copy.esm"; std::filesystem::copy_file(master, copy);
    const auto originalAchr = findRecord(readFile(copy), 0xB10, cc("ACHR"));
    {
        EsmDocument document(copy, 0x3C);
        const auto* actor = findRef(document.references(), 0xB10);
        require(actor && actor->placement.actor && actor->placement.baseObject == 0x910
            && std::abs(actor->placement.position[0] - 200.0f) < 0.01f && std::abs(actor->placement.rotation[2] - 1.5f) < 0.001f
            && actor->editorId == "SettlerRef", "ACHR was not indexed as an actor reference");
        const auto* object = findRef(document.references(), 0xB00);
        require(object && !object->placement.actor, "REFR was indexed as an actor");
        require(document.actorBases().size() == 1 && document.actorBases()[0].formId == 0x910
            && document.actorBases()[0].editorId == "Settler" && document.actorBases()[0].name == "Settler"
            && document.actorBases()[0].race == 0x920, "NPC_ base was not indexed");
        document.save();  // nothing changed
    }
    require(findRecord(readFile(copy), 0xB10, cc("ACHR")) == originalAchr, "Unchanged ACHR was rewritten");
    std::uint32_t fresh = 0;
    {
        EsmDocument document(copy, 0x3C);
        auto refs = document.references();
        for (auto& r : refs) if (r.placement.formId == 0xB10) { r.placement.position[0] = 333.0f; r.placement.scale = 1.25f; }
        ESMReference added; added.placement.formId = fresh = document.allocateFormId(); added.placement.instanceId = fresh;
        added.placement.baseObject = 0x910; added.placement.actor = true; added.placement.cell = {0, 0};
        added.placement.position[0] = 64.0f; added.placement.position[1] = 32.0f; added.editorId = "NewActor";
        refs.push_back(added);
        document.setReferences(refs);
        document.save();
    }
    const auto saved = readFile(copy);
    const auto moved = findRecord(saved, 0xB10, cc("ACHR"));
    require(!moved.empty() && findRecord(saved, 0xB10, cc("REFR")).empty(), "Edited ACHR changed record type");
    require(contains(moved, achrUnknown), "Edited ACHR lost unknown subrecords");
    require(get<std::uint32_t>(moved, 16) == 0x0BADF00Du, "Edited ACHR lost header metadata");
    const auto added = findRecord(saved, fresh, cc("ACHR"));
    require(!added.empty() && subU32(added, "NAME") == 0x910, "New ACHR was not written with its NPC_ base");
    EsmDocument reopened(copy, 0x3C);
    const auto* a = findRef(reopened.references(), 0xB10);
    const auto* b = findRef(reopened.references(), fresh);
    require(a && a->placement.actor && std::abs(a->placement.position[0] - 333.0f) < 0.01f && std::abs(a->placement.scale - 1.25f) < 0.0001f,
        "Edited ACHR did not round-trip");
    require(b && b->placement.actor && b->editorId == "NewActor" && b->placement.cell == CellCoord{0, 0}, "New ACHR did not round-trip");
    require(findRef(reopened.references(), 0xB00) && !findRef(reopened.references(), 0xB00)->placement.actor, "REFR changed type");
}

void testPluginActors(const std::filesystem::path& root)
{
    const auto master = makeActorMaster(root);
    const auto masterBytes = readFile(master);
    const auto plugin = root / "Mod.esp";
    createFo4OverridePlugin(plugin, {"Master.esm"}, "actor test");
    LoadOrder order(root);
    (void)order.add(master); (void)order.add(plugin);
    order.makeActive(1);
    std::uint32_t fresh = 0;
    {
        EditorDocument editor(plugin, direct(order, 0x3C));
        const auto* actor = findRef(editor, 0xB10);
        require(actor && actor->placement.actor && actor->placement.baseObject == 0x910, "Plugin did not inherit the master ACHR");
        require(editor.actorBase(0x910) && editor.actorBase(0x910)->race == 0x920 && editor.actorBase(0x910)->editorId == "Settler",
            "NPC_ bases are not visible through the load order");
        auto p = editor.referenceProvenance(0xB10);
        require(p.definingFile == "Master.esm" && p.editCreatesOverride && !p.definedByActive && p.writable, "ACHR provenance is wrong");

        rejects([&] { (void)editor.placeActor(0x900, at(10)); }, "STAT accepted as an actor base");
        rejects([&] { (void)editor.placeActor(0x00000999, at(10)); }, "Unknown base from an indexed master accepted");
        rejects([&] { (void)editor.placeActor(0x01000999, at(10)); }, "Unknown own-slot base accepted");
        rejects([&] { (void)editor.placeActor(0x05000910, at(10)); }, "Base outside the master list accepted");
        rejects([&] { (void)editor.place(0x910, at(10)); }, "NPC_ placed as a static model");
        rejects([&] { editor.retarget(0xB10, 0x900); }, "Actor retargeted to a STAT");
        rejects([&] { editor.setActorBase(0xB00, 0x910); }, "REFR accepted an NPC_ base");

        editor.setTransform(0xB10, at(500));
        fresh = editor.placeActor(0x910, at(600));
        require((fresh >> 24) == 1 && (fresh & 0xFFFFFFu) >= 0x800, "New ACHR did not use the plugin's self slot");
        require(findRef(editor, fresh)->placement.actor && findRef(editor, fresh)->name.starts_with("Settler_"), "New ACHR state is wrong");
        require(editor.referenceProvenance(fresh).definedByActive, "New ACHR provenance is wrong");
        const auto copy = editor.duplicate(fresh);
        require(findRef(editor, copy)->placement.actor, "Duplicate lost the actor flag");
        editor.erase(copy);
        require(editor.undo() && findRef(editor, copy) && editor.undo() && !findRef(editor, copy), "Undo of actor edits failed");
        editor.save();
        require(readFile(master) == masterBytes, "Saving actor overrides rewrote the master");
        const auto saved = readFile(plugin);
        const auto moved = findRecord(saved, 0xB10, cc("ACHR"));
        require(!moved.empty() && findRecord(saved, 0xB10, cc("REFR")).empty(), "ACHR override missing or written as REFR");
        require(contains(moved, achrUnknown) && subU32(moved, "NAME") == 0x910, "ACHR override lost master data");
        require(findRecord(saved, 0xB10 | 0x01000000u).empty(), "ACHR override was re-identified");
        const auto added = findRecord(saved, fresh, cc("ACHR"));
        require(!added.empty() && subU32(added, "NAME") == 0x910, "New ACHR missing from plugin");
        require(findRecord(saved, 0x910).empty(), "NPC_ base was copied into the plugin");
        p = editor.referenceProvenance(0xB10);
        require(p.activeHasOverride && p.winningFile == "Mod.esp" && !p.pendingChange, "Saved ACHR provenance is wrong");

        // Plain Delete on the master ACHR disables it through the override.
        require(editor.deleteReference(0xB10, false) == openf4::editor::ReferenceDeleteAction::disable, "Delete did not disable the actor");
        editor.save();
        require(get<std::uint32_t>(findRecord(readFile(plugin), 0xB10, cc("ACHR")), 8) & 0x800u, "Disabled ACHR override missing flag");
    }
    {
        ESMStackOptions options; options.worldFormId = 0x3C; options.resolvedFiles = {master, plugin};
        const auto stack = loadEsmMasterStack(plugin, options);
        const auto* moved = findRef(stack.references, 0xB10);
        const auto* added = findRef(stack.references, fresh);
        require(moved && moved->placement.actor && moved->disabled && moved->recordSource.esm.find("Mod.esp") != std::string::npos,
            "ACHR override does not win after reload");
        require(added && added->placement.actor && added->placement.baseObject == 0x910, "New ACHR lost after reload");
        require(std::ranges::any_of(stack.actorBases, [](const auto& a) { return a.formId == 0x910 && a.race == 0x920; }),
            "Stack lost NPC_ bases");
    }
    {
        // Revert to load order drops the plugin's ACHR override entirely.
        EditorDocument editor(plugin, direct(order, 0x3C));
        editor.revertToLoadOrder(0xB10);
        editor.save();
        require(findRecord(readFile(plugin), 0xB10).empty(), "Revert left an ACHR override behind");
        require(!findRecord(readFile(plugin), fresh, cc("ACHR")).empty(), "Revert removed the plugin's own ACHR");
        require(readFile(master) == masterBytes, "Revert rewrote the master");
    }
}

void testOpenF4World(const std::filesystem::path& root)
{
    const auto master = makeActorMaster(root);
    std::filesystem::path overlay;
    std::uint32_t fresh = 0;
    {
        EditorDocument editor(master);
        overlay = editor.path();
        require(overlay != master && editor.isOpenF4Owned(), "OpenF4 overlay was not created");
        require(editor.actorBase(0x910), "OpenF4 world does not see master NPC_ bases");
        fresh = editor.placeActor(0x910, at(700, 0, 20));
        editor.save();
    }
    {
        EditorDocument reopened(overlay);
        const auto* actor = findRef(reopened, fresh);
        require(actor && actor->placement.actor && actor->placement.baseObject == 0x910
            && std::abs(actor->placement.position[0] - 700.0f) < 0.05f, "ACHR did not persist in the OpenF4 world");
    }
    // What openf4.exe feeds ObjectScene: the stack reference keeps the actor flag.
    const auto stack = loadEsmMasterStack(overlay, root);
    const auto* runtime = findRef(stack.references, fresh);
    require(runtime && runtime->placement.actor, "Runtime stack lost the actor flag");
}

void testPlaceholderGeometry()
{
    const auto geometry = openf4::world::detail::actorPlaceholderGeometry();
    require(!geometry.surfaces.empty(), "Actor placeholder has no surfaces");
    float top = -1e9f, bottom = 1e9f;
    for (const auto& surface : geometry.surfaces)
    {
        require(!surface.vertices.empty() && !surface.indices.empty() && surface.indices.size() % 3 == 0, "Placeholder surface is empty");
        for (auto index : surface.indices) require(index < surface.vertices.size(), "Placeholder index out of range");
        for (const auto& v : surface.vertices) { top = std::max(top, v.position.y); bottom = std::min(bottom, v.position.y); }
    }
    require(std::abs(bottom) < 0.01f && std::abs(top - 128.0f) < 0.01f, "Placeholder is not a 128-unit Y-up figure");
}
}

int main()
{
    try
    {
        const auto root = std::filesystem::temp_directory_path() / "openf4-actor-tests";
        std::filesystem::remove_all(root);
        for (const char* name : {"roundtrip", "plugin", "world"}) std::filesystem::create_directories(root / name);
        testDocumentRoundTrip(root / "roundtrip");
        testPluginActors(root / "plugin");
        testOpenF4World(root / "world");
        testPlaceholderGeometry();
        std::filesystem::remove_all(root);
        std::cout << "Actor tests passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
