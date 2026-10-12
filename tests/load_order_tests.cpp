// ESP-on-master loading: overrides, deletion overrides, stable FormIDs, dependency-aware
// load order, master write protection, provenance and raw record preservation.
#include "assets/esm_document.hpp"
#include "assets/fo4_world_writer.hpp"
#include "assets/load_order.hpp"
#include "editor/editor_document.hpp"
#include "world/world_scene.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
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
const Bytes refUnknown = sub("XZZZ", text("unknown reference payload"));

// Master.esm: a non-OpenF4 master with two worldspaces and a STAT.
std::filesystem::path makeMaster(const std::filesystem::path& root)
{
    Bytes file = tes4(0x1, {}, "Bethesda");
    Bytes statBody; append(statBody, sub("EDID", text("Rock01"))); append(statBody, sub("MODL", text("rocks\\rock01.nif")));
    append(statBody, sub("ZSTA", text("unknown static data")));
    append(file, group(cc("STAT"), 0, record("STAT", 0x900, statBody)));
    Bytes worlds;
    Bytes wrld; append(wrld, sub("EDID", text("Commonwealth"))); append(wrld, sub("ZWLD", text("unknown world data")));
    append(worlds, record("WRLD", 0x3C, wrld));
    Bytes refs; append(refs, refr(0xB00, 0x900, 100, 100)); append(refs, refr(0xB01, 0x900, 300, 300));
    append(worlds, group(0x3C, 1, cellTree(0xA00, 0xA01, refs, "CommonwealthCell")));
    Bytes other; append(other, sub("EDID", text("OtherWorld")));
    append(worlds, record("WRLD", 0x50, other));
    append(worlds, group(0x50, 1, cellTree(0xA10, 0xA11, refr(0xC00, 0x900, 50, 50), "OtherCell")));
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
    const auto* active = order.activeEntry();
    std::string reason; options.writable = order.canWrite(active->header.path, &reason); options.readOnlyReason = reason;
    return options;
}
bool hasReference(const EditorDocument& document, std::uint32_t id)
{
    return std::ranges::any_of(document.references(), [&](const auto& r) { return r.placement.formId == id; });
}

void testDependencyAwareLoading(const std::filesystem::path& root)
{
    makeMaster(root);
    createFo4OverridePlugin(root / "Dependent.esp", {"Master.esm"});
    LoadOrder order(root);
    const auto added = order.add(root / "Dependent.esp");
    require(added.missingMasters == std::vector<std::string>{"Master.esm"}, "Missing master was not reported");
    require(added.resolvable.size() == 1 && added.resolvable[0].filename() == "Master.esm", "Missing master was not offered from Data");
    rejects([&] { order.makeActive(0); }, "Plugin with a missing master was activated");
    require(order.addMissingMasters(0) == 1, "Offered master was not added");
    require(order.entries()[0].header.name == "Master.esm" && order.entries()[1].header.name == "Dependent.esp",
        "Master was not inserted before its dependent");
    require(order.validate().empty(), "Valid load order reported problems");

    const auto before = order.entries();
    rejects([&] { order.move(1, 0); }, "Reorder that loads a plugin before its master was accepted");
    require(order.entries()[0].header.name == "Master.esm", "Refused reorder still changed the order");
    std::string reason; require(!order.canMove(0, 1, &reason) && !reason.empty(), "canMove allowed a master-order violation");
    rejects([&] { order.remove(0); }, "Removed a master that a loaded plugin requires");
    rejects([&] { (void)order.add(root / "Master.esm"); }, "Duplicate file was added");

    // A master added after its dependent master is inserted before it.
    Bytes second = tes4(0x1, {"Master.esm"}, "Bethesda"); writeFile(root / "Second.esm", second);
    LoadOrder reverse(root);
    (void)reverse.add(root / "Second.esm"); (void)reverse.add(root / "Master.esm");
    require(reverse.entries()[0].header.name == "Master.esm", "Late master was not placed before its dependent");
    // A plugin that omits a transitive master cannot become active.
    createFo4OverridePlugin(root / "Omits.esp", {"Second.esm"});
    (void)reverse.add(root / "Omits.esp");
    rejects([&] { reverse.makeActive(*reverse.find("Omits.esp")); }, "Plugin omitting a transitive master was activated");
    // Masters always precede plugins.
    rejects([&] { reverse.move(*reverse.find("Omits.esp"), 0); }, "Plugin moved ahead of masters");
}

void testMasterProtectionAndOverrides(const std::filesystem::path& root)
{
    const auto master = makeMaster(root);
    const auto masterBytes = readFile(master);
    const auto plugin = root / "Mod.esp";
    createFo4OverridePlugin(plugin, {"Master.esm"}, "load order test");
    LoadOrder order(root);
    (void)order.add(master); (void)order.add(plugin);

    // --- Make Active on a master is read-only browse.
    order.makeActive(0);
    require(order.isProtected(0) && !order.canWrite(master), "Master became writable through Make Active");
    {
        EditorDocument browse(master, direct(order, 0x3C));
        require(!browse.writable() && hasReference(browse, 0xB00), "Read-only master browse did not load");
        rejects([&] { browse.setTransform(0xB00, at(900)); }, "Read-only master accepted a move");
        rejects([&] { browse.erase(0xB00); }, "Read-only master accepted a delete");
        rejects([&] { (void)browse.place(0x900, at(10)); }, "Read-only master accepted a placement");
        rejects([&] { browse.save(); }, "Read-only master accepted save");
        rejects([&] { browse.saveAs(root / "copy.esm"); }, "Read-only master accepted save as");
        const auto provenance = browse.referenceProvenance(0xB00);
        require(!provenance.writable && !provenance.blockedReason.empty(), "Provenance hid the read-only gate");
    }
    require(readFile(master) == masterBytes, "Read-only browse changed the master");

    // --- Enable Master Editing requires the exact filename and makes a verified backup.
    rejects([&] { (void)order.enableMasterEditing("master.esm"); }, "Wrong-case confirmation enabled master editing");
    rejects([&] { (void)order.enableMasterEditing("yes"); }, "Arbitrary confirmation enabled master editing");
    require(!order.masterEditingEnabled() && !order.canWrite(master), "Failed confirmation left master editing on");
    for (const auto& e : std::filesystem::directory_iterator(root))
        require(e.path().extension() != ".bak", "Failed confirmation created a backup");
    const auto backup = order.enableMasterEditing("Master.esm");
    require(order.masterEditingEnabled() && order.canWrite(master), "Exact confirmation did not enable master editing");
    require(std::filesystem::exists(backup) && readFile(backup) == masterBytes
        && backup.filename().string().starts_with("Master.esm.") && backup.extension() == ".bak", "Master backup is missing or differs");
    rejects([&] { (void)order.enableMasterEditing("Mod.esp"); }, "Confirmation accepted a different filename");
    // Ordinary selection change drops the permission; a new session never has it.
    order.makeActive(1);
    require(!order.masterEditingEnabled(), "Master editing survived a change of active file");
    order.makeActive(0);
    require(!order.canWrite(master), "Re-selecting the master restored master editing");
    require(!LoadOrder(root).masterEditingEnabled(), "Master editing is remembered across sessions");
    std::filesystem::remove(backup);

    // --- Edits of master-owned records become overrides in the active plugin.
    order.makeActive(1);
    require(order.canWrite(plugin), "Ordinary plugin is not writable");
    {
        EditorDocument editor(plugin, direct(order, 0x3C));
        require(editor.writable() && hasReference(editor, 0xB00) && hasReference(editor, 0xB01), "Plugin did not inherit master references");
        require(!hasReference(editor, 0xC00), "Other worldspace leaked into Commonwealth");
        auto p = editor.referenceProvenance(0xB00);
        require(p.definingFile == "Master.esm" && p.winningFile == "Master.esm" && p.activeFile == "Mod.esp"
            && p.editCreatesOverride && !p.activeHasOverride && !p.definedByActive && p.writable && !p.pendingChange,
            "Initial override provenance is wrong");
        editor.setTransform(0xB00, at(200, 0, 10));
        require(editor.referenceProvenance(0xB00).pendingChange, "Pending edit not reported");
        editor.erase(0xB01);
        const auto fresh = editor.place(0x900, at(250));
        require((fresh >> 24) == 1 && (fresh & 0xFFFFFFu) >= 0x800, "New record did not use the plugin's self slot");
        require(editor.referenceProvenance(fresh).definedByActive && !editor.referenceProvenance(fresh).editCreatesOverride,
            "New record provenance is wrong");
        editor.save();
        require(readFile(master) == masterBytes, "Saving overrides rewrote the master");
        const auto saved = readFile(plugin);
        const auto moved = findRecord(saved, 0xB00, cc("REFR"));
        require(!moved.empty(), "Override was not written with the master's FormID");
        require(contains(moved, refUnknown), "Override lost unknown subrecords from the master record");
        require(get<std::uint32_t>(moved, 16) == 0x0BADF00Du && get<std::uint16_t>(moved, 22) == 7,
            "Override lost the source record header metadata");
        const auto tomb = findRecord(saved, 0xB01, cc("REFR"));
        require(!tomb.empty() && (get<std::uint32_t>(tomb, 8) & 0x20u), "Deletion override was not written with the Deleted flag");
        require(!findRecord(saved, 0xA00, cc("CELL")).empty() && !findRecord(saved, 0x3C, cc("WRLD")).empty(),
            "Parent CELL/WRLD overrides missing");
        require(findRecord(saved, 0x3C, cc("WRLD")) == findRecord(masterBytes, 0x3C, cc("WRLD")), "Parent WRLD override changed master bytes");
        require(findRecord(saved, 0xA01, cc("LAND")).empty(), "Unchanged LAND was overridden");
        require(findRecord(saved, 0xB00 | 0x01000000u).empty(), "Override was re-identified into the plugin's slot");
        require(!findRecord(masterBytes, 0xB01).empty() && !(get<std::uint32_t>(findRecord(readFile(master), 0xB01), 8) & 0x20u),
            "Master reference was erased or flagged");
        p = editor.referenceProvenance(0xB00);
        require(p.activeHasOverride && !p.editCreatesOverride && p.winningFile == "Mod.esp" && p.definingFile == "Master.esm" && !p.pendingChange,
            "Saved override provenance is wrong");

        // Stable allocation: later records continue above existing ones.
        const auto later = editor.place(0x900, at(260));
        require(later > fresh && (later >> 24) == 1, "FormID allocation reused or regressed");
        editor.save();
    }
    {
        // Reloaded: deletion hides the master ref, override wins, unknown data survives.
        ESMStackOptions options; options.worldFormId = 0x3C; options.directPlugin = false;
        options.resolvedFiles = {master, plugin};
        const auto stack = loadEsmMasterStack(plugin, options);
        require(stack.canonicalNames == std::vector<std::string>{"Master.esm", "Mod.esp"}, "Canonical master mapping is wrong");
        require(std::ranges::none_of(stack.references, [](const auto& r) { return r.placement.formId == 0xB01; }), "Deletion override did not hide the reference");
        const auto moved = std::ranges::find_if(stack.references, [](const auto& r) { return r.placement.formId == 0xB00; });
        require(moved != stack.references.end() && std::abs(moved->placement.position[0] - 200.0f) < 0.05f
            && moved->recordSource.esm.find("Mod.esp") != std::string::npos, "Override does not win after reload");
    }

    // --- Unknown records in the plugin are preserved byte-for-byte across later saves.
    Bytes unknownRecord = record("KYWD", 0x01000F00u, sub("EDID", text("ModKeyword")));
    {
        auto bytes = readFile(plugin);
        Bytes extra = record("ZZUN", 0x01000F01u, sub("ZZZZ", text("an unknown record type")), 0x00000800u);
        append(bytes, group(cc("KYWD"), 0, unknownRecord)); append(bytes, group(cc("ZZUN"), 0, extra));
        writeFile(plugin, bytes);
        EditorDocument editor(plugin, direct(order, 0x3C));
        editor.setTransform(0xB00, at(210));
        editor.save();
        const auto saved = readFile(plugin);
        require(findRecord(saved, 0x01000F00u) == unknownRecord && findRecord(saved, 0x01000F01u) == extra,
            "Unknown plugin records were not preserved byte-for-byte");
        require(readFile(master) == masterBytes, "Second save rewrote the master");
        const auto next = editor.place(0x900, at(270));
        require(next > 0x01000F01u, "Allocator reused an ID owned by an unknown record");
    }

    // --- Worldspace selection.
    {
        const auto worlds = order.activeWorldspaces();
        require(worlds.size() == 2 && worlds[0].formId == 0x3C && worlds[0].editorId == "Commonwealth"
            && worlds[1].formId == 0x50 && worlds[1].definingFile == "Master.esm", "Worldspace list is wrong");
        EditorDocument other(plugin, direct(order, 0x50));
        require(hasReference(other, 0xC00) && !hasReference(other, 0xB00), "Worldspace selection loaded the wrong world");
        other.setTransform(0xC00, at(60));
        other.save();
        const auto saved = readFile(plugin);
        require(!findRecord(saved, 0xC00, cc("REFR")).empty() && !findRecord(saved, 0x50, cc("WRLD")).empty()
            && !findRecord(saved, 0x3C, cc("WRLD")).empty(), "Second-world override missing or replaced the first world");
        require(readFile(master) == masterBytes, "Second-world save rewrote the master");
        EditorDocument commonwealth(plugin, direct(order, 0x3C));
        require(hasReference(commonwealth, 0xB00) && !hasReference(commonwealth, 0xB01), "Second world save broke Commonwealth overrides");
    }

    // --- Master editing actually opens the gate only after the explicit action.
    {
        order.makeActive(0);
        EditorDocument locked(master, direct(order, 0x3C));
        rejects([&] { locked.setTransform(0xB00, at(5)); }, "Master accepted edits before Enable Master Editing");
        const auto masterBackup = order.enableMasterEditing("Master.esm");
        std::string reason;
        locked.setWritePolicy(order.canWrite(master, &reason), reason);
        locked.setTransform(0xB00, at(5)); locked.save();
        require(readFile(master) != masterBytes && readFile(masterBackup) == masterBytes, "Enabled master edit/backup mismatch");
        writeFile(master, masterBytes); std::filesystem::remove(masterBackup);
        order.disableMasterEditing();
        require(!order.canWrite(master), "Disable Master Editing did not close the gate");
    }
}

const openf4::editor::EditorReference& referenceOf(const EditorDocument& document, std::uint32_t id)
{
    const auto found = std::ranges::find_if(document.references(), [&](const auto& r) { return r.placement.formId == id; });
    require(found != document.references().end(), "Reference missing");
    return *found;
}

// Delete disables in a plugin; Ctrl+Shift+Delete (hard) needs explicit confirmation.
void testDeletePolicy(const std::filesystem::path& root)
{
    using openf4::editor::ReferenceDeleteAction;
    const auto master = makeMaster(root);
    const auto masterBytes = readFile(master);
    const auto plugin = root / "Mod.esp";
    createFo4OverridePlugin(plugin, {"Master.esm"}, "delete policy test");
    LoadOrder order(root);
    (void)order.add(master); (void)order.add(plugin);

    // Read-only master: both Delete and a confirmed hard delete are blocked.
    order.makeActive(0);
    {
        EditorDocument browse(master, direct(order, 0x3C));
        require(browse.planReferenceDelete(0xB00, false).action == ReferenceDeleteAction::blocked
            && browse.planReferenceDelete(0xB00, true).action == ReferenceDeleteAction::blocked, "Read-only master offered a delete");
        rejects([&] { (void)browse.deleteReference(0xB00, false); }, "Read-only master accepted Delete");
        rejects([&] { (void)browse.deleteReference(0xB00, true, true); }, "Read-only master accepted a confirmed hard delete");
        rejects([&] { browse.setDisabled(0xB00, true); }, "Read-only master accepted a disable");
        require(hasReference(browse, 0xB00) && !referenceOf(browse, 0xB00).disabled, "Blocked delete changed the reference");
    }
    require(readFile(master) == masterBytes, "Blocked delete touched the master");

    order.makeActive(1);
    std::uint32_t fresh = 0;
    {
        EditorDocument editor(plugin, direct(order, 0x3C));
        require(editor.isPluginFile(), "Mod.esp not recognised as a plugin");

        // Plain Delete of a master-owned reference: disable through a new same-FormID override.
        const auto plan = editor.planReferenceDelete(0xB00, false);
        require(plan.action == ReferenceDeleteAction::disable && !plan.requiresConfirmation && !plan.crashRisk
            && plan.provenance.editCreatesOverride, "Plain Delete of a master reference is not a disable");
        require(editor.deleteReference(0xB00, false) == ReferenceDeleteAction::disable, "Delete did not disable");
        require(hasReference(editor, 0xB00) && referenceOf(editor, 0xB00).disabled, "Delete erased instead of disabling");
        require(editor.planReferenceDelete(0xB00, false).action == ReferenceDeleteAction::alreadyDisabled
            && editor.deleteReference(0xB00, false) == ReferenceDeleteAction::alreadyDisabled
            && hasReference(editor, 0xB00), "Second Delete escalated beyond disable");
        require(editor.undo() && !referenceOf(editor, 0xB00).disabled, "Undo did not re-enable");
        require(editor.redo() && referenceOf(editor, 0xB00).disabled, "Redo did not disable again");

        // Plain Delete never erases in a plugin, even the plugin's own new record.
        fresh = editor.place(0x900, at(250));
        require(editor.planReferenceDelete(fresh, false).action == ReferenceDeleteAction::disable
            && editor.deleteReference(fresh, false) == ReferenceDeleteAction::disable
            && hasReference(editor, fresh) && referenceOf(editor, fresh).disabled, "Delete erased a plugin record");

        // Hard delete: always needs confirmation; master-owned = Deleted-flag override with crash warning.
        const auto hard = editor.planReferenceDelete(0xB01, true);
        require(hard.action == ReferenceDeleteAction::deletedOverride && hard.hard && hard.requiresConfirmation
            && hard.crashRisk && !hard.erasesFromMaster && !hard.warning.empty(), "Hard delete plan of a master reference is wrong");
        rejects([&] { (void)editor.deleteReference(0xB01, true); }, "Unconfirmed hard delete was applied");
        rejects([&] { (void)editor.deleteReference(0xB01, true, false); }, "Unconfirmed hard delete was applied");
        require(hasReference(editor, 0xB01) && !referenceOf(editor, 0xB01).disabled, "Refused hard delete changed the reference");
        const auto own = editor.planReferenceDelete(fresh, true);
        require(own.action == ReferenceDeleteAction::removeRecord && own.requiresConfirmation && !own.crashRisk
            && !own.erasesFromMaster, "Hard delete plan of a plugin record is wrong");

        editor.save();
        auto saved = readFile(plugin);
        const auto disabled = findRecord(saved, 0xB00, cc("REFR"));
        require(!disabled.empty() && (get<std::uint32_t>(disabled, 8) & 0x800u) && !(get<std::uint32_t>(disabled, 8) & 0x20u),
            "Disable override not written with Initially Disabled (and without Deleted)");
        require(contains(disabled, refUnknown) && get<std::uint32_t>(disabled, 16) == 0x0BADF00Du,
            "Disable override lost master record data");
        require(findRecord(saved, 0xB00 | 0x01000000u).empty(), "Disable re-identified the master record");
        require(findRecord(saved, 0xB01, cc("REFR")).empty(), "Refused hard delete still wrote an override");
        const auto freshRecord = findRecord(saved, fresh, cc("REFR"));
        require(!freshRecord.empty() && (get<std::uint32_t>(freshRecord, 8) & 0x800u), "Disabled plugin record not kept");
        require(readFile(master) == masterBytes, "Disable rewrote the master");

        // Confirmed hard delete.
        require(editor.deleteReference(0xB01, true, true) == ReferenceDeleteAction::deletedOverride
            && !hasReference(editor, 0xB01), "Confirmed hard delete of a master reference failed");
        require(editor.deleteReference(fresh, true, true) == ReferenceDeleteAction::removeRecord
            && !hasReference(editor, fresh), "Confirmed hard delete of a plugin record failed");
        editor.save();
        saved = readFile(plugin);
        const auto tomb = findRecord(saved, 0xB01, cc("REFR"));
        require(!tomb.empty() && (get<std::uint32_t>(tomb, 8) & 0x20u), "Hard delete did not write a Deleted-flag override");
        require(findRecord(saved, fresh).empty(), "Hard-deleted plugin record survived");
        require(readFile(master) == masterBytes, "Hard delete rewrote the master");

        // Re-enable through the Inspector toggle clears the flag but keeps the override.
        editor.setDisabled(0xB00, false); editor.save();
        const auto enabled = findRecord(readFile(plugin), 0xB00, cc("REFR"));
        require(!enabled.empty() && !(get<std::uint32_t>(enabled, 8) & 0x820u), "Re-enable left a flag behind");
        editor.setDisabled(0xB00, true); editor.save();
    }
    {
        // Reloaded: the disabled override stays visible to the editor as disabled.
        EditorDocument reopened(plugin, direct(order, 0x3C));
        require(hasReference(reopened, 0xB00) && referenceOf(reopened, 0xB00).disabled, "Disabled override did not reload as disabled");
        require(!hasReference(reopened, 0xB01), "Deleted override did not hide the reference");
    }

    // Master editing: plain Delete still only disables; a hard delete flags erasing from the master.
    {
        order.makeActive(0);
        const auto backup = order.enableMasterEditing("Master.esm");
        EditorDocument editor(master, direct(order, 0x3C));
        require(!editor.isPluginFile() && editor.writable(), "Master editing did not open the master");
        require(editor.planReferenceDelete(0xB00, false).action == ReferenceDeleteAction::disable,
            "Plain Delete would erase from an enabled master");
        const auto hard = editor.planReferenceDelete(0xB00, true);
        require(hard.action == ReferenceDeleteAction::removeRecord && hard.requiresConfirmation && hard.erasesFromMaster,
            "Hard delete in master editing does not flag erasing from the master");
        rejects([&] { (void)editor.deleteReference(0xB00, true); }, "Unconfirmed hard delete in master editing applied");
        require(readFile(master) == masterBytes, "Planning deletes rewrote the master");
        order.disableMasterEditing(); std::filesystem::remove(backup);
    }
}

bool approx(float a, float b) { return std::abs(a - b) < 0.01f; }

// Revert to load order: dropping the active plugin's version lets the prior load order win.
void testRevertToLoadOrder(const std::filesystem::path& root)
{
    using openf4::editor::ReferenceDeleteAction;
    const auto master = makeMaster(root);
    const auto masterBytes = readFile(master);
    const auto plugin = root / "Revert.esp";
    createFo4OverridePlugin(plugin, {"Master.esm"}, "revert test");
    LoadOrder order(root);
    (void)order.add(master); (void)order.add(plugin);

    // Read-only master: revert is blocked.
    order.makeActive(0);
    {
        EditorDocument browse(master, direct(order, 0x3C));
        rejects([&] { browse.revertToLoadOrder(0xB00); }, "Read-only master accepted a revert");
    }
    order.makeActive(1);
    std::uint32_t fresh = 0;
    {
        EditorDocument editor(plugin, direct(order, 0x3C));
        // Nothing to revert yet.
        require(!editor.referenceProvenance(0xB00).canRevertToLoadOrder, "Untouched master reference offered a revert");
        rejects([&] { editor.revertToLoadOrder(0xB00); }, "Revert of an untouched reference was accepted");
        // Unsaved edit only: revert restores the master version and nothing is written.
        editor.setTransform(0xB00, at(500));
        auto p = editor.referenceProvenance(0xB00);
        require(p.canRevertToLoadOrder && p.revertWinner == "Master.esm", "Pending edit not revertible to Master.esm");
        editor.revertToLoadOrder(0xB00);
        require(approx(referenceOf(editor, 0xB00).placement.position[0], 100.0f), "Revert did not restore the master position");
        require(!editor.referenceProvenance(0xB00).canRevertToLoadOrder, "Reverted reference still revertible");
        // Real overrides: move B00, disable B01, plus a plugin-owned record.
        editor.setTransform(0xB00, at(200, 0, 10));
        require(editor.deleteReference(0xB01, false) == ReferenceDeleteAction::disable, "Delete did not disable");
        fresh = editor.place(0x900, at(250));
        rejects([&] { editor.revertToLoadOrder(fresh); }, "Revert accepted a record the plugin itself defines");
        editor.save();
        const auto saved = readFile(plugin);
        require(!findRecord(saved, 0xB00, cc("REFR")).empty() && !findRecord(saved, 0xB01, cc("REFR")).empty(), "Overrides not written");
        require(readFile(master) == masterBytes, "Override save touched the master");

        // Same session, after save: overrides are now stored in the plugin.
        p = editor.referenceProvenance(0xB00);
        require(p.activeHasOverride && p.canRevertToLoadOrder && p.revertWinner == "Master.esm", "Saved override not revertible");
        editor.revertToLoadOrder(0xB01);
        require(!referenceOf(editor, 0xB01).disabled && editor.referenceProvenance(0xB01).revertPending, "Revert did not re-enable B01");
        require(editor.undo() && referenceOf(editor, 0xB01).disabled, "Undo of revert failed");
        require(editor.redo() && !referenceOf(editor, 0xB01).disabled, "Redo of revert failed");
        editor.save();
        const auto after = readFile(plugin);
        require(findRecord(after, 0xB01, cc("REFR")).empty(), "Reverted override still in the plugin");
        require(!findRecord(after, 0xB00, cc("REFR")).empty(), "Unrelated override was dropped");
        require(!findRecord(after, fresh, cc("REFR")).empty(), "Plugin's own record was dropped");
        require(readFile(master) == masterBytes, "Revert rewrote the master");
        require(!editor.referenceProvenance(0xB01).canRevertToLoadOrder && !editor.referenceProvenance(0xB01).activeHasOverride,
            "Reverted reference still reports an override after save");
    }
    {
        // Reopened: revert a stored move override; locked references are refused.
        EditorDocument editor(plugin, direct(order, 0x3C));
        require(approx(referenceOf(editor, 0xB00).placement.position[0], 200.0f), "Override did not reload");
        require(approx(referenceOf(editor, 0xB01).placement.position[0], 300.0f) && !referenceOf(editor, 0xB01).disabled,
            "Reverted reference did not reload from the master");
        editor.setLocked(0xB00, true);
        rejects([&] { editor.revertToLoadOrder(0xB00); }, "Revert accepted a locked reference");
        editor.setLocked(0xB00, false);
        editor.revertToLoadOrder(0xB00);
        require(approx(referenceOf(editor, 0xB00).placement.position[0], 100.0f), "Stored override revert did not restore master data");
        // Hard delete B01 (Deleted-flag override), save, then restore it.
        require(editor.deleteReference(0xB01, true, true) == ReferenceDeleteAction::deletedOverride, "Hard delete failed");
        editor.save();
        const auto saved = readFile(plugin);
        require(findRecord(saved, 0xB00, cc("REFR")).empty(), "Reverted move override survived save");
        const auto tomb = findRecord(saved, 0xB01, cc("REFR"));
        require(!tomb.empty() && (get<std::uint32_t>(tomb, 8) & 0x20u), "Deleted-flag override not written");
        require(readFile(master) == masterBytes, "Hard delete rewrote the master");
    }
    {
        EditorDocument editor(plugin, direct(order, 0x3C));
        require(!hasReference(editor, 0xB01), "Deleted override did not hide the reference");
        const auto deletions = editor.revertibleDeletions();
        require(std::ranges::any_of(deletions, [](const auto& d) { return d.first == 0xB01u; }), "Deleted override not offered for restore");
        editor.revertToLoadOrder(0xB01);
        require(hasReference(editor, 0xB01) && approx(referenceOf(editor, 0xB01).placement.position[0], 300.0f), "Restore did not bring back B01");
        require(editor.revertibleDeletions().empty(), "Restored reference still listed as deleted");
        editor.save();
        require(findRecord(readFile(plugin), 0xB01, cc("REFR")).empty(), "Deleted-flag override survived restore");
        require(readFile(master) == masterBytes, "Restore rewrote the master");
    }
    {
        EditorDocument editor(plugin, direct(order, 0x3C));
        require(hasReference(editor, 0xB00) && hasReference(editor, 0xB01) && hasReference(editor, fresh), "Reload lost references");
        require(approx(referenceOf(editor, 0xB00).placement.position[0], 100.0f) && !referenceOf(editor, 0xB01).disabled,
            "Master version does not win after revert");
        require(!editor.referenceProvenance(0xB00).activeHasOverride && !editor.referenceProvenance(0xB00).canRevertToLoadOrder,
            "Plugin still claims a version of B00");
        rejects([&] { editor.revertToLoadOrder(0xB00); }, "Revert without an override was accepted");
    }
    require(readFile(master) == masterBytes, "Master bytes changed");
}

void testPluginWriter(const std::filesystem::path& root)
{
    makeMaster(root);
    const auto path = root / "Header.esp";
    createFo4OverridePlugin(path, {"Master.esm"});
    rejects([&] { createFo4OverridePlugin(path, {"Master.esm"}); }, "New plugin replaced an existing file");
    rejects([&] { createFo4OverridePlugin(root / "Bad.esm", {"Master.esm"}); }, "New plugin accepted a master extension");
    rejects([&] { createFo4OverridePlugin(root / "Dup.esp", {"Master.esm", "master.esm"}); }, "Duplicate masters accepted");
    const auto header = readPluginHeader(path);
    require(header.masters == std::vector<std::string>{"Master.esm"} && header.openf4Owned && !header.isMasterFile()
        && (header.flags & 0x1u) == 0, "New plugin header is wrong");
    require(EsmDocument(path).nextFormId() == 0x01000800u, "New plugin allocator does not start at the self slot");
}

// Bethesda masters keep each world's persistent CELL directly in World Children and give it
// XCLC 0,0 (Fallout4.esm: Commonwealth 00018AA2 beside exterior 0,0 0000E400). It is not a
// grid cell, so switching the active file between the master and a plugin must not report
// a duplicate exterior coordinate, and persistent references must keep their owner.
void testPersistentWorldCell(const std::filesystem::path& root)
{
    Bytes file = tes4(0x1, {}, "Bethesda");
    Bytes statBody; append(statBody, sub("EDID", text("Rock01"))); append(statBody, sub("MODL", text("rocks\\rock01.nif")));
    append(file, group(cc("STAT"), 0, record("STAT", 0x900, statBody)));
    Bytes wrld; append(wrld, sub("EDID", text("Commonwealth")));
    Bytes persistentBody; append(persistentBody, sub("EDID", text("CommonwealthPersistent")));
    Bytes xclc; put(xclc, std::int32_t{0}); put(xclc, std::int32_t{0}); put(xclc, std::uint32_t{0});
    append(persistentBody, sub("XCLC", xclc));
    Bytes children = record("CELL", 0xA20, persistentBody, 0x400);
    append(children, group(0xA20, 6, group(0xA20, 8, refr(0xB20, 0x900, 5000, -5000))));
    append(children, cellTree(0xA00, 0xA01, refr(0xB00, 0x900, 100, 100), "CommonwealthCell"));
    Bytes worlds = record("WRLD", 0x3C, wrld); append(worlds, group(0x3C, 1, children));
    append(file, group(cc("WRLD"), 0, worlds));
    const auto master = root / "Fallout4.esm"; writeFile(master, file);
    const auto masterBytes = readFile(master);

    const auto grid = EsmDocument(master, 0x3C).terrainCell({0, 0});
    require(grid && grid->cellFormId == 0xA00, "Persistent world CELL claimed the exterior 0,0 grid slot");
    require(EsmDocument(master).terrainCells().size() == 1, "Persistent world CELL was indexed as terrain");

    const auto plugin = root / "Mod.esp";
    createFo4OverridePlugin(plugin, {"Fallout4.esm"}, "persistent cell test");
    LoadOrder order(root);
    (void)order.add(master); (void)order.add(plugin);
    for (int pass = 0; pass < 2; ++pass)
    {
        order.makeActive(0);
        {
            EditorDocument browse(master, direct(order, 0x3C));
            require(hasReference(browse, 0xB00) && hasReference(browse, 0xB20), "Master browse lost references");
        }
        order.makeActive(1);
        EditorDocument editor(plugin, direct(order, 0x3C));
        require(hasReference(editor, 0xB00) && hasReference(editor, 0xB20), "Plugin did not inherit persistent references");
        if (pass == 0)
        {
            editor.setTransform(0xB20, at(5100, -5000));
            editor.setTransform(0xB00, at(150, 100));
            editor.save();
        }
    }
    require(readFile(master) == masterBytes, "Persistent-cell override rewrote the master");
    const auto saved = readFile(plugin);
    require(!findRecord(saved, 0xA20, cc("CELL")).empty() && !findRecord(saved, 0xB20, cc("REFR")).empty()
        && !findRecord(saved, 0xA00, cc("CELL")).empty() && !findRecord(saved, 0xB00, cc("REFR")).empty(),
        "Persistent or exterior override was not written under its original CELL");
    require(EsmDocument(plugin, 0x3C).terrainCell({0, 0})->cellFormId == 0xA00, "Plugin re-indexed the persistent CELL as 0,0");
}
// Fallout 4 GRUP type 10 is Quest Children (label = QUST FormID, holding DIAL/SCEN), not a CELL
// reference group. Fallout4.esm has ~1000 of them under the top-level QUST group; treating them as
// CELL-owned made Make Active fail with "Reference child group has incorrect CELL ownership".
void testQuestChildrenGroups(const std::filesystem::path& root)
{
    Bytes file = tes4(0x1, {}, "Bethesda");
    Bytes statBody; append(statBody, sub("EDID", text("Rock01"))); append(statBody, sub("MODL", text("rocks\\rock01.nif")));
    append(file, group(cc("STAT"), 0, record("STAT", 0x900, statBody)));
    Bytes wrld; append(wrld, sub("EDID", text("Commonwealth")));
    Bytes worlds = record("WRLD", 0x3C, wrld);
    append(worlds, group(0x3C, 1, cellTree(0xA00, 0xA01, refr(0xB00, 0x900, 100, 100), "CommonwealthCell")));
    append(file, group(cc("WRLD"), 0, worlds));
    Bytes quests = record("QUST", 0x3648, sub("EDID", text("MQ101")));
    Bytes questChildren = record("DIAL", 0x3649, sub("EDID", text("MQ101Topic")));
    append(questChildren, group(0x3649, 7, record("INFO", 0x364A, sub("ZINF", text("unknown info")))));
    append(quests, group(0x3648, 10, questChildren));
    append(file, group(cc("QUST"), 0, quests));
    const auto master = root / "Fallout4.esm"; writeFile(master, file);
    const auto masterBytes = readFile(master);
    require(EsmDocument(master, 0x3C).terrainCell({0, 0})->cellFormId == 0xA00, "Quest Children group broke master load");

    const auto plugin = root / "Mod.esp";
    createFo4OverridePlugin(plugin, {"Fallout4.esm"}, "quest children test");
    LoadOrder order(root);
    (void)order.add(master); (void)order.add(plugin);
    order.makeActive(0);
    { EditorDocument browse(master, direct(order, 0x3C)); require(hasReference(browse, 0xB00), "Master browse lost references"); }
    order.makeActive(1);
    {
        EditorDocument editor(plugin, direct(order, 0x3C));
        require(hasReference(editor, 0xB00), "Plugin did not inherit references");
        editor.setTransform(0xB00, at(150, 100)); editor.save();
    }
    order.makeActive(0);
    { EditorDocument browse(master, direct(order, 0x3C)); require(hasReference(browse, 0xB00), "Master reopen failed"); }
    require(readFile(master) == masterBytes, "Quest children test rewrote the master");
    require(!findRecord(readFile(plugin), 0xB00, cc("REFR")).empty(), "Override was not saved");

    // A genuine CELL reference group (type 9) labelled with the wrong CELL is still rejected.
    Bytes broken = tes4(0x1, {}, "Bethesda");
    Bytes cellBody; append(cellBody, sub("EDID", text("Cell")));
    Bytes xclc; put(xclc, std::int32_t{0}); put(xclc, std::int32_t{0}); put(xclc, std::uint32_t{0}); append(cellBody, sub("XCLC", xclc));
    Bytes tree = record("CELL", 0xA00, cellBody); append(tree, group(0xA00, 6, group(0xA77, 9, refr(0xB00, 0x900, 1, 1))));
    Bytes brokenWorlds = record("WRLD", 0x3C, wrld); append(brokenWorlds, group(0x3C, 1, group(0, 4, group(0, 5, tree))));
    append(broken, group(cc("WRLD"), 0, brokenWorlds));
    const auto brokenPath = root / "Broken.esm"; writeFile(brokenPath, broken);
    rejects([&] { EsmDocument document(brokenPath); }, "Mis-owned CELL reference group was accepted");
}

// Optional smoke test against a real install: set OPENF4_FALLOUT4_ESM to Fallout4.esm.
void testRealFallout4Master()
{
    const char* path = std::getenv("OPENF4_FALLOUT4_ESM");
    if (!path || !*path) return;
    EsmDocument document(path);
    require(!document.terrainCells().empty(), "Real Fallout4.esm loaded no exterior cells");
    std::cout << "Real Fallout4.esm: " << document.terrainCells().size() << " exterior cells\n";
}
}
int main()
{
    try
    {
        const auto root = std::filesystem::temp_directory_path() / "openf4-load-order-tests";
        std::filesystem::remove_all(root);
        for (const char* name : {"dependency", "protection", "writer", "delete", "revert", "persistent", "quest"}) std::filesystem::create_directories(root / name);
        testDependencyAwareLoading(root / "dependency");
        testMasterProtectionAndOverrides(root / "protection");
        testPluginWriter(root / "writer");
        testDeletePolicy(root / "delete");
        testRevertToLoadOrder(root / "revert");
        testPersistentWorldCell(root / "persistent");
        testQuestChildrenGroups(root / "quest");
        testRealFallout4Master();
        std::filesystem::remove_all(root);
        std::cout << "Load order tests passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
