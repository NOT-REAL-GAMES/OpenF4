#include "editor/editor_document.hpp"
#include "assets/asset_resolver.hpp"
#include "assets/dem_import.hpp"
#include "assets/load_order.hpp"
#include "assets/lod_bake.hpp"
#include "assets/model_import.hpp"
#include "assets/papyrus_binding.hpp"
#include "world/world_scene.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cmath>
#include <iostream>
#include <set>
#include <stdexcept>
#include <unordered_map>

namespace openf4::editor
{
namespace
{
struct State
{
    std::vector<EditorReference> references;
    std::map<std::uint32_t, assets::ESMModel> models;
    std::optional<assets::EditableLandscape> terrain;
    // Records reverted to the load order this session: their active-file override is dropped on save.
    std::set<std::uint32_t> reverted;
    bool operator==(const State&) const = default;
};
void includeCell(assets::EditableLandscape& terrain, const assets::ESMTerrainCell& cell, bool preserveResident = false)
{
    for (std::int32_t y = 0; y < 33; ++y) for (std::int32_t x = 0; x < 33; ++x)
    {
        const auto c = cell.landscape.cell;
        bool resident = terrain.containsCell(c);
        if (preserveResident)
            for (int dy = y == 0 ? -1 : 0; dy <= (y == 32 ? 1 : 0); ++dy)
                for (int dx = x == 0 ? -1 : 0; dx <= (x == 32 ? 1 : 0); ++dx)
                    resident = resident || terrain.containsCell({c.x + dx, c.y + dy});
        if (!preserveResident || !resident)
            terrain.setSample({c.x * 32 + x, c.y * 32 + y}, cell.landscape.heights[y * 33 + x]);
    }
    terrain.cellForms[cell.landscape.cell] = {cell.cellFormId, cell.landFormId};
}
bool sameTerrainCell(const assets::EditableLandscape& a, const assets::EditableLandscape& b, assets::CellCoord c)
{
    if (!b.containsCell(c) || a.cellForms.at(c) != b.cellForms.at(c)) return false;
    return a.cell(c).heights == b.cell(c).heights;
}
assets::ESMReference authoredReference(const EditorReference& reference)
{
    return {reference.placement, reference.name, false, reference.recordSource,
        reference.ownerCellFormId, reference.childGroupType, reference.scripts, reference.disabled};
}
EditorReference editorReference(const assets::ESMReference& r)
{
    EditorReference reference{r.placement, r.editorId};
    reference.recordSource = r.recordSource; reference.ownerCellFormId = r.ownerCellFormId;
    reference.childGroupType = r.childGroupType; reference.scripts = r.scripts; reference.disabled = r.disabled;
    return reference;
}
void validateModelKind(assets::ESMModelKind kind)
{
    if (kind != assets::ESMModelKind::staticObject && kind != assets::ESMModelKind::movableStatic)
        throw std::runtime_error("Unsupported model record kind");
}
std::string nativePackagePath(const std::filesystem::path& root, std::string value)
{
    value = assets::AssetResolver::normalizePath(value);
    std::ranges::replace(value, '\\', '/');
    const std::filesystem::path relative(value);
    if (value.find('\0') != std::string::npos || value.find(':') != std::string::npos
        || relative.has_root_path() || relative.extension() != ".nif"
        || std::any_of(relative.begin(), relative.end(), [](const auto& part) { return part == ".."; })
        || !std::filesystem::is_regular_file(root / relative))
        throw std::runtime_error("Native package must be published in the project's sidecar before assignment");
    const auto resolved = std::filesystem::canonical(root / relative).lexically_relative(std::filesystem::canonical(root));
    if (resolved.empty() || std::any_of(resolved.begin(), resolved.end(), [](const auto& part) { return part == ".."; }))
        throw std::runtime_error("Native package resolves outside the project's sidecar");
    return relative.lexically_normal().generic_string();
}
}

struct EditorDocument::Impl
{
    std::filesystem::path path, base;
    std::optional<std::filesystem::path> dataDirectory;
    std::unique_ptr<assets::EsmDocument> document;
    std::unique_ptr<assets::EsmDocument> baseDocument;
    std::unique_ptr<assets::DemLandscapeProject> dem;
    State state, saved;
    mutable std::optional<bool> dirtyCache;
    std::vector<State> undo, redo;
    std::optional<State> transaction;
    std::uint32_t transactionNext = 0;
    unsigned transactionDepth = 0;
    std::map<std::uint32_t, std::string> modelPaths;
    std::map<std::uint32_t, assets::ESMReference> baselineRefs;
    // Load-order version (without the active file) of records the active file overrides/deletes.
    std::map<std::uint32_t, assets::ESMReference> shadowedRefs;
    std::map<std::uint32_t, assets::ESMActorBase> actorBases;
    std::set<std::string> indexedFiles;  // folded filenames whose records were scanned this session
    bool editableTerrain = false;
    bool startAwake = false;
    bool writable = true, direct = false;
    bool pluginFile = false;
    std::string readOnlyReason;
    std::optional<std::uint32_t> worldPreset, baseWorld;
    std::optional<assets::ESMRecordSource> worldSource;
    std::vector<std::string> canonical;
    std::uint64_t wakeAttachments = 0;
    std::uint32_t next = 0;
    // Editor caches (outliner, object projection) key on this; FormID lookups use the index.
    std::uint64_t revision = nextRevision();
    mutable std::unordered_map<std::uint32_t, std::size_t> referenceIndex;
    mutable std::size_t indexedCount = 0;
    mutable bool referenceIndexStale = true;

    static std::uint64_t nextRevision()
    {
        static std::atomic<std::uint64_t> counter{0};
        return ++counter;
    }
    void touch() { revision = nextRevision(); referenceIndexStale = true; }
    // Every hit is verified against the vector, so a stale index can only cause a rebuild,
    // never a wrong answer; a miss on a fresh index falls back to a linear scan.
    const EditorReference* findIndexed(std::uint32_t id) const
    {
        const auto lookup = [&]() -> const EditorReference* {
            const auto found = referenceIndex.find(id);
            if (found == referenceIndex.end() || found->second >= state.references.size()) return nullptr;
            const auto& candidate = state.references[found->second];
            return candidate.placement.formId == id ? &candidate : nullptr;
        };
        if (const auto* hit = lookup()) return hit;
        const auto rebuild = [&] {
            referenceIndex.clear(); referenceIndex.reserve(state.references.size());
            for (std::size_t i = 0; i < state.references.size(); ++i)
                referenceIndex.emplace(state.references[i].placement.formId, i);
            indexedCount = state.references.size(); referenceIndexStale = false;
        };
        if (referenceIndexStale || indexedCount != state.references.size()) { rebuild(); return lookup(); }
        const auto found = std::find_if(state.references.begin(), state.references.end(),
            [&](const auto& r) { return r.placement.formId == id; });
        if (found == state.references.end()) return nullptr;
        rebuild();
        return &*found;
    }

    Impl(std::filesystem::path p, editor::EditorOpenOptions options)
        : path(std::filesystem::absolute(p)), dataDirectory(std::move(options.dataDirectory))
    {
        writable = options.writable; readOnlyReason = std::move(options.readOnlyReason); direct = options.directEdit;
        if (options.worldFormId && *options.worldFormId) worldPreset = options.worldFormId;
        if (direct)
        {
            // Load Order targets already exist; never create or replace them on open.
            if (!std::filesystem::is_regular_file(path)) throw std::runtime_error("Active file does not exist: " + path.string());
        }
        else if (!std::filesystem::exists(path)) assets::saveEditableLandscape(path, assets::createEditableLandscape());
        document = std::make_unique<assets::EsmDocument>(path, worldPreset);
        if (!direct && !document->isOpenF4Owned())
        {
            path = assets::createEditorOverlay(path);
            document = std::make_unique<assets::EsmDocument>(path, worldPreset);
        }
        try { pluginFile = !assets::readPluginHeader(path).isMasterFile(); }
        catch (const std::exception&)
        {
            auto extension = path.extension().string();
            for (auto& c : extension) c = char(std::tolower((unsigned char)c));
            pluginFile = extension == ".esp";
        }
        assets::ESMStackOptions stackOptions; stackOptions.dataDirectory = dataDirectory;
        stackOptions.worldFormId = worldPreset; stackOptions.resolvedFiles = options.resolvedFiles;
        stackOptions.directPlugin = direct && !document->isOpenF4Owned();
        const auto stack = assets::loadEsmMasterStack(path, stackOptions); base = std::filesystem::absolute(stack.basePath);
        canonical = stack.canonicalNames;
        // Direct edits of non-OpenF4 files (plugins, or a master after Enable Master Editing)
        // never get the OpenF4 heightfield editor; their LAND stays native.
        editableTerrain = base == path && (!direct || document->isOpenF4Owned());
        const auto world = document->worldFormId();
        if (worldPreset && base != path)
        {
            // The base master owns the world; it shares the low FormID slots with this file.
            const auto baseHeader = assets::readPluginHeader(base);
            baseWorld = (static_cast<std::uint32_t>(baseHeader.masters.size()) << 24) | (world & 0xFFFFFFu);
        }
        else if (worldPreset) baseWorld = worldPreset;
        if (world && !document->hasWorldRecord() && (world >> 24) + 1 < canonical.size() && base != path)
        {
            worldSource = assets::ESMRecordSource{base.string(), baseWorld.value_or(world)};
            document->setWorldSource(*worldSource);
        }
        if (document->isDemOverlay()) dem = std::make_unique<assets::DemLandscapeProject>(path);
        for (const auto& m : stack.models) state.models.emplace(m.formId, m);
        for (const auto& a : stack.actorBases) actorBases.emplace(a.formId, a);
        for (const auto& file : stack.sourcePaths) indexedFiles.insert(assets::foldedName(file.filename().string()));
        for (const auto& r : stack.references)
        {
            EditorReference reference{r.placement, r.editorId};
            reference.recordSource = r.recordSource; reference.ownerCellFormId = r.ownerCellFormId;
            reference.childGroupType = r.childGroupType;
            reference.scripts = r.scripts;
            reference.disabled = r.disabled;
            state.references.push_back(std::move(reference)); baselineRefs[r.placement.formId] = r;
        }
        for (const auto& r : stack.shadowedReferences) shadowedRefs[r.placement.formId] = authoredReference(editorReference(r));
        next = document->nextFormId();
        if (editableTerrain && !dem)
        {
            state.terrain.emplace();
            for (const auto& [coord, cell] : document->terrainCells())
                if (auto value = document->terrainCell(coord); value && value->landscape.hasHeights) includeCell(*state.terrain, *value);
            state.terrain->nextFormId = next;
        }
        saved = state; refreshPaths();
    }
    void refreshPaths()
    {
        modelPaths.clear(); for (const auto& [id, model] : state.models) modelPaths[id] = model.path;
    }
    std::uint32_t allocate()
    {
        if ((next & 0xFFFFFFu) >= 0xFFFFFFu) throw std::runtime_error("Editor FormID space exhausted");
        return next++;
    }
    EditorReference& reference(std::uint32_t id)
    {
        const auto* found = findIndexed(id);
        if (!found) throw std::runtime_error("Editor reference no longer exists");
        return const_cast<EditorReference&>(*found);
    }
    // The record as the remaining load order (everything but the active file) defines it.
    std::optional<assets::ESMReference> loadOrderVersion(std::uint32_t id) const
    {
        if ((id >> 24) + 1 == canonical.size()) return std::nullopt;
        if (const auto found = shadowedRefs.find(id); found != shadowedRefs.end()) return found->second;
        if (!document->containsRecord(id))
            if (const auto found = baselineRefs.find(id); found != baselineRefs.end()) return found->second;
        return std::nullopt;
    }
    void requireWritable() const
    {
        if (!writable) throw std::runtime_error(readOnlyReason.empty() ? "This file is read-only in the current session" : readOnlyReason);
    }
    std::unique_ptr<assets::EsmDocument> openBase() const
    {
        return std::make_unique<assets::EsmDocument>(base, baseWorld);
    }
    template<class F> void change(F operation)
    {
        requireWritable();
        State before = state;
        const auto previousNext = next;
        referenceIndexStale = true;
        try { operation(); }
        catch (...) { state = std::move(before); next = previousNext; refreshPaths(); touch(); throw; }
        dirtyCache.reset();
        touch();
        if (!transactionDepth && !(before == state))
        {
            undo.push_back(std::move(before)); if (undo.size() > 64) undo.erase(undo.begin()); redo.clear();
        }
        refreshPaths();
    }
    void ensureCell(assets::CellCoord coord)
    {
        if (!state.terrain) state.terrain.emplace();
        auto& terrain = *state.terrain;
        if (terrain.containsCell(coord)) return;
        if (auto existing = document->terrainCell(coord); existing && existing->landscape.hasHeights)
        { includeCell(terrain, *existing, true); return; }
        if (dem)
        {
            if (auto existing = dem->cell(coord))
            { includeCell(terrain, {existing->landscape, existing->forms.cell, existing->forms.land}, true); return; }
            for (const auto nearby : {assets::CellCoord{coord.x - 1, coord.y}, {coord.x + 1, coord.y}, {coord.x, coord.y - 1}, {coord.x, coord.y + 1}})
                if (!terrain.containsCell(nearby)) if (auto existing = dem->cell(nearby))
                    includeCell(terrain, {existing->landscape, existing->forms.cell, existing->forms.land}, true);
        }
        else if (!editableTerrain)
        {
            if (!baseDocument) baseDocument = openBase();
            if (auto existing = baseDocument->terrainCell(coord); existing && existing->landscape.hasHeights)
            { includeCell(terrain, *existing, true); return; }
        }
        terrain.nextFormId = next; (void)terrain.addCell(coord); next = terrain.nextFormId;
    }
    // Attaches the wake script to an MSTT base and drops now-redundant plain per-placement
    // copies (the old per-reference Start Awake) so FO4 does not bind the script twice.
    bool attachWake(std::uint32_t id)
    {
        auto& model = state.models.at(id);
        if (model.kind != assets::ESMModelKind::movableStatic || !model.scripts.editable) return false;
        const auto name = assets::wakeOnLoadScriptName;
        if (assets::papyrusScriptActive(model.scripts, {}, name)) return false;
        auto scripts = model.scripts;
        assets::setPapyrusScriptActive(scripts, {}, name, true);
        scripts.present = true; scripts.edited = true; scripts.sourceEsm = path.string();
        // A base whose existing VMAD cannot be re-encoded keeps its bytes; Start Awake stays per-placement.
        try { (void)assets::encodePapyrusBindings(scripts, false); } catch (const std::exception&) { return false; }
        model.scripts = std::move(scripts); model.recordOwnerEsm = path.string();
        const auto key = [](std::string value) { for (auto& c : value) c = char(std::tolower((unsigned char)c)); return value; };
        const auto wanted = key(std::string(name));
        for (auto& ref : state.references)
        {
            if (ref.placement.baseObject != id || !ref.scripts.editable || ref.locked) continue;
            const auto removed = std::erase_if(ref.scripts.scripts, [&](const assets::PapyrusScript& s) {
                return key(s.name) == wanted && s.flags == 0 && s.properties.empty(); });
            if (!removed) continue;
            ref.scripts.present = !ref.scripts.scripts.empty(); ref.scripts.edited = true; ref.scripts.sourceEsm = path.string();
            if (ref.childGroupType != 8) ensureCell(ref.placement.cell);
        }
        ++wakeAttachments;
        return true;
    }
    void defaultWake(std::uint32_t id) { if (startAwake) (void)attachWake(id); }
    std::uint32_t place(std::uint32_t model, const glm::mat4& transform, std::string name)
    {
        if (!state.models.contains(model))
            throw std::runtime_error(actorBases.contains(model) ? "NPC_ bases are placed as actors (placeActor)" : "Cannot place an unknown model");
        EditorReference ref; ref.placement.formId = allocate(); ref.placement.instanceId = ref.placement.formId;
        ref.placement.baseObject = model;
        world::setBethesdaReferenceTransform(ref.placement, transform);
        ensureCell(ref.placement.cell); ref.name = name.empty() ? "Object_" + std::to_string(ref.placement.formId) : std::move(name);
        const auto id = ref.placement.formId; state.references.push_back(std::move(ref)); return id;
    }
    void validateActorBase(std::uint32_t npcBase) const
    {
        if (!npcBase) throw std::runtime_error("Actor base FormID is empty");
        if (state.models.contains(npcBase)) throw std::runtime_error("Actor base must be an NPC_, not a STAT/MSTT");
        if (actorBases.contains(npcBase)) return;
        const auto slot = npcBase >> 24;
        if (slot >= canonical.size()) throw std::runtime_error("Actor base FormID is outside the active file's master list");
        if (slot + 1 == canonical.size())
            throw std::runtime_error("Actor base is not an NPC_ record of " + path.filename().string());
        // A master this session did not index (OpenF4 worlds skip Fallout4.esm): trust the ID.
        if (indexedFiles.contains(assets::foldedName(canonical[slot])))
            throw std::runtime_error("Actor base is not an NPC_ record of " + canonical[slot]);
    }
    std::uint32_t placeActor(std::uint32_t npcBase, const glm::mat4& transform, std::string name)
    {
        validateActorBase(npcBase);
        EditorReference ref; ref.placement.formId = allocate(); ref.placement.instanceId = ref.placement.formId;
        ref.placement.baseObject = npcBase; ref.placement.actor = true;
        world::setBethesdaReferenceTransform(ref.placement, transform);
        ensureCell(ref.placement.cell);
        if (name.empty())
        {
            const auto found = actorBases.find(npcBase);
            name = (found != actorBases.end() && !found->second.editorId.empty() ? found->second.editorId : std::string("Actor"))
                + "_" + std::to_string(ref.placement.formId & 0xFFFFFFu);
        }
        ref.name = std::move(name);
        const auto id = ref.placement.formId; state.references.push_back(std::move(ref)); return id;
    }
    void save(std::filesystem::path target)
    {
        requireWritable();
        std::set<assets::CellCoord> changedStaticCells;
        std::set<std::uint32_t> changedModels;
        for (const auto& [id, model] : state.models)
        {
            const auto previous = saved.models.find(id);
            if (previous == saved.models.end() || previous->second != model)
                changedModels.insert(id);
        }
        for (const auto& [id, model] : saved.models)
            if (!state.models.contains(id)) changedModels.insert(id);
        std::map<std::uint32_t, assets::ESMReference> oldRefs, newRefs;
        for (const auto& reference : saved.references)
            oldRefs.emplace(reference.placement.formId,
                authoredReference(reference));
        for (const auto& reference : state.references)
            newRefs.emplace(reference.placement.formId,
                authoredReference(reference));
        for (const auto& [id, reference] : oldRefs)
        {
            const auto current = newRefs.find(id);
            if (current == newRefs.end() || current->second != reference
                || changedModels.contains(reference.placement.baseObject))
                changedStaticCells.insert(reference.placement.cell);
        }
        for (const auto& [id, reference] : newRefs)
        {
            const auto previous = oldRefs.find(id);
            if (previous == oldRefs.end() || previous->second != reference
                || changedModels.contains(reference.placement.baseObject))
                changedStaticCells.insert(reference.placement.cell);
        }
        // Terrain tools may have saved since this document was opened. Re-index the
        // current source and apply only authored changes instead of stale file bytes.
        assets::EsmDocument output(path, worldPreset);
        if (worldSource) output.setWorldSource(*worldSource);
        output.setNextFormId(std::max(next, output.nextFormId()));
        std::vector<assets::ESMModel> models;
        // Preserve externally added bases that this authoring session never owned.
        for (const auto& model : output.models())
            if (!saved.models.contains(model.formId) && !state.models.contains(model.formId)) models.push_back(model);
        for (const auto& [id, m] : state.models)
        {
            if (m.sourceEsm != path.string() && m.recordOwnerEsm != path.string()) continue;
            models.push_back(m);
        }
        output.setModels(std::move(models));
        std::vector<assets::ESMReference> references;
        std::set<std::uint32_t> current;
        std::set<std::uint32_t> local;
        for (const auto& r : output.references()) local.insert(r.placement.formId);
        for (const auto& r : state.references)
        {
            current.insert(r.placement.formId);
            auto authored = authoredReference(r);
            // Reverted to the load order and untouched since: the active file keeps no version.
            if (state.reverted.contains(r.placement.formId))
                if (const auto prior = loadOrderVersion(r.placement.formId); prior && *prior == authored) continue;
            auto original = baselineRefs.find(r.placement.formId);
            if (local.contains(r.placement.formId) || original == baselineRefs.end() || original->second != authored) references.push_back(std::move(authored));
        }
        for (const auto& [id, r] : baselineRefs) if (!current.contains(id) && (id >> 24) != (next >> 24))
        { auto deleted = r; deleted.deleted = true; references.push_back(std::move(deleted)); }
        // Retain existing tombstones, even though they are absent from merged authoring state.
        for (const auto& r : output.references()) if (r.deleted && !current.contains(r.placement.formId)) references.push_back(r);
        // First-time overrides/tombstones: remember the version they shadow for Revert to load order.
        for (const auto& reference : references)
        {
            const auto id = reference.placement.formId;
            if ((id >> 24) + 1 == canonical.size() || shadowedRefs.contains(id) || document->containsRecord(id)) continue;
            if (const auto found = baselineRefs.find(id); found != baselineRefs.end()) shadowedRefs[id] = found->second;
        }
        for (const auto& reference : references)
            if (reference.childGroupType != 8) ensureCell(reference.placement.cell);
        output.setReferences(references);
        std::vector<assets::ESMTerrainCell> changes;
        if (state.terrain)
        {
            std::set<assets::CellCoord> referenceCells;
            for (const auto& r : references) referenceCells.insert(r.placement.cell);
            for (const auto& [coord, forms] : state.terrain->cellForms)
            {
                const bool modified = !saved.terrain || !sameTerrainCell(*state.terrain, *saved.terrain, coord);
                const bool needsOwnership = referenceCells.contains(coord) && !output.terrainCells().contains(coord);
                if (modified || needsOwnership)
                {
                    assets::ESMTerrainCell change{state.terrain->cell(coord), forms.cell, forms.land};
                    // The editable height grid does not own native CELL/LAND metadata.
                    auto original = output.terrainCell(coord);
                    if ((!original || !original->landFormId) && !editableTerrain && !dem)
                    {
                        if (!baseDocument) baseDocument = openBase();
                        const auto inherited = baseDocument->terrainCell(coord);
                        if (inherited)
                        {
                            if (!original) original = inherited;
                            else { original->landFormId = inherited->landFormId; original->landSource = inherited->landSource; }
                        }
                    }
                    if (original && original->cellFormId == change.cellFormId)
                    {
                        change.cellSource = original->cellSource;
                        if (original->landFormId == change.landFormId) change.landSource = original->landSource;
                    }
                    changes.push_back(std::move(change));
                }
            }
        }
        output.replaceTerrainCells(changes); output.setPath(target);
        bool lodIndexPrepared = false;
        try
        {
            assets::LodBuildIndex(target).prepareSave();
            lodIndexPrepared = true;
        }
        catch (const std::exception& error)
        {
            std::cerr << "LOD build index preparation failed: " << error.what() << '\n';
        }
        output.save();
        std::vector<assets::CellCoord> changedTerrain;
        changedTerrain.reserve(changes.size());
        for (const auto& change : changes)
            changedTerrain.push_back(change.landscape.cell);
        const std::vector<assets::CellCoord> changedStatics(
            changedStaticCells.begin(), changedStaticCells.end());
        try
        {
            if (lodIndexPrepared)
                assets::LodBuildIndex(target).markSavedChanges(changedTerrain, changedStatics);
        }
        catch (const std::exception& error)
        {
            // The saved ESM remains authoritative; stale derived products will
            // be invalidated by their source fingerprint on the next bake.
            std::cerr << "LOD build index invalidation failed: " << error.what() << '\n';
        }
        const auto oldPath = path; path = std::filesystem::absolute(target);
        if (base == oldPath) base = path;
        const auto relocate = [&](State& snapshot) {
            for (auto& [id, m] : snapshot.models)
            {
                if (m.sourceEsm == oldPath.string()) m.sourceEsm = path.string();
                if (m.recordSourceEsm == oldPath.string()) m.recordSourceEsm = path.string();
                if (m.recordOwnerEsm == oldPath.string()) m.recordOwnerEsm = path.string();
                if (m.scripts.sourceEsm == oldPath.string()) m.scripts.sourceEsm = path.string();
            }
            for (auto& reference : snapshot.references)
            {
                if (reference.recordSource.esm == oldPath.string()) reference.recordSource.esm = path.string();
                if (reference.scripts.sourceEsm == oldPath.string()) reference.scripts.sourceEsm = path.string();
            }
        };
        relocate(state); for (auto& snapshot : undo) relocate(snapshot); for (auto& snapshot : redo) relocate(snapshot);
        if (transaction) relocate(*transaction);
        document = std::make_unique<assets::EsmDocument>(path, worldPreset); next = std::max(next, document->nextFormId());
        if (worldSource && !document->hasWorldRecord()) document->setWorldSource(*worldSource);
        if (dem && path != oldPath) dem = std::make_unique<assets::DemLandscapeProject>(path);
        state.reverted.clear();
        saved = state; dirtyCache = false;
        baselineRefs.clear(); for (const auto& r : state.references) baselineRefs[r.placement.formId] = authoredReference(r);
        touch();
    }
};

EditorDocument::EditorDocument(std::filesystem::path path, std::optional<std::filesystem::path> dataDirectory)
    : mImpl(std::make_unique<Impl>(std::move(path), EditorOpenOptions{std::move(dataDirectory)})) {}
EditorDocument::EditorDocument(std::filesystem::path path, EditorOpenOptions options)
    : mImpl(std::make_unique<Impl>(std::move(path), std::move(options))) {}
void EditorDocument::setWritePolicy(bool writable, std::string reason)
{
    mImpl->writable = writable; mImpl->readOnlyReason = writable ? std::string() : std::move(reason);
}
bool EditorDocument::writable() const noexcept { return mImpl->writable; }
const std::string& EditorDocument::readOnlyReason() const noexcept { return mImpl->readOnlyReason; }
bool EditorDocument::isOpenF4Owned() const noexcept { return mImpl->document->isOpenF4Owned(); }
bool EditorDocument::isPluginFile() const noexcept { return mImpl->pluginFile; }
std::optional<std::uint32_t> EditorDocument::baseWorldFormId() const noexcept { return mImpl->baseWorld; }
const std::vector<std::string>& EditorDocument::loadedFiles() const noexcept { return mImpl->canonical; }
namespace
{
std::string slotName(const std::vector<std::string>& canonical, std::uint32_t id)
{
    const auto slot = id >> 24;
    return slot < canonical.size() ? canonical[slot] : "(slot " + std::to_string(slot) + " outside the master list)";
}
}
RecordProvenance EditorDocument::referenceProvenance(std::uint32_t id) const
{
    const auto& impl = *mImpl;
    const auto* found = impl.findIndexed(id);
    if (!found) throw std::runtime_error("Editor reference no longer exists");
    RecordProvenance result; result.formId = id;
    result.activeFile = impl.path.filename().string();
    result.definingFile = slotName(impl.canonical, id);
    result.definedByActive = (id >> 24) + 1 == impl.canonical.size();
    result.activeHasOverride = !result.definedByActive && impl.document->containsRecord(id);
    result.editCreatesOverride = !result.definedByActive && !result.activeHasOverride;
    if (result.definedByActive || result.activeHasOverride) result.winningFile = result.activeFile;
    else if (!found->recordSource.esm.empty()) result.winningFile = std::filesystem::path(found->recordSource.esm).filename().string();
    else result.winningFile = result.definingFile;
    const auto baseline = impl.baselineRefs.find(id);
    result.pendingChange = baseline == impl.baselineRefs.end() || baseline->second != authoredReference(*found);
    result.writable = impl.writable && !found->locked;
    result.blockedReason = !impl.writable ? impl.readOnlyReason : found->locked ? "Reference is locked" : std::string();
    if (!result.definedByActive)
        if (const auto prior = impl.loadOrderVersion(id))
        {
            result.revertWinner = prior->recordSource.esm.empty() ? result.definingFile
                : std::filesystem::path(prior->recordSource.esm).filename().string();
            result.revertPending = impl.state.reverted.contains(id) && *prior == authoredReference(*found);
            result.canRevertToLoadOrder = !result.revertPending && (result.activeHasOverride || result.pendingChange);
            if (result.revertPending) result.winningFile = result.revertWinner + " (after save)";
        }
    return result;
}
void EditorDocument::revertToLoadOrder(std::uint32_t id)
{
    auto& impl = *mImpl;
    impl.requireWritable();
    if ((id >> 24) + 1 == impl.canonical.size())
        throw std::runtime_error("This record is defined by " + impl.path.filename().string()
            + " itself; there is no earlier load-order version to revert to (use Delete instead)");
    const auto prior = impl.loadOrderVersion(id);
    if (!prior) throw std::runtime_error("No load-order version of this record is known");
    const bool stored = impl.document->containsRecord(id);
    const auto found = std::find_if(impl.state.references.begin(), impl.state.references.end(),
        [&](const auto& r) { return r.placement.formId == id; });
    if (found != impl.state.references.end())
    {
        if (found->locked) throw std::runtime_error("Cannot revert a locked reference");
        if (authoredReference(*found) == *prior && (!stored || impl.state.reverted.contains(id)))
            throw std::runtime_error(impl.path.filename().string() + " has no version of this record to drop");
    }
    else if (!stored && !impl.baselineRefs.contains(id))
        throw std::runtime_error("Reference is not deleted by " + impl.path.filename().string());
    impl.change([&] {
        auto restored = editorReference(*prior);
        const auto existing = std::find_if(impl.state.references.begin(), impl.state.references.end(),
            [&](const auto& r) { return r.placement.formId == id; });
        if (existing != impl.state.references.end())
        {
            restored.visible = existing->visible; restored.locked = existing->locked; restored.bounds = existing->bounds;
            *existing = std::move(restored);
        }
        else impl.state.references.push_back(std::move(restored));
        if (stored) impl.state.reverted.insert(id);
    });
}
std::vector<std::pair<std::uint32_t, std::string>> EditorDocument::revertibleDeletions() const
{
    const auto& impl = *mImpl;
    std::set<std::uint32_t> present;
    for (const auto& r : impl.state.references) present.insert(r.placement.formId);
    std::vector<std::pair<std::uint32_t, std::string>> result;
    for (const auto& [id, r] : impl.shadowedRefs)
        if (!present.contains(id)) result.emplace_back(id, r.editorId);
    // Pending (unsaved) hard deletes of records the active file has no version of yet.
    for (const auto& [id, r] : impl.baselineRefs)
        if (!present.contains(id) && (id >> 24) + 1 != impl.canonical.size() && !impl.shadowedRefs.contains(id)
            && !impl.document->containsRecord(id))
            result.emplace_back(id, r.editorId);
    return result;
}
RecordProvenance EditorDocument::modelProvenance(std::uint32_t id) const
{
    const auto& impl = *mImpl;
    const auto found = impl.state.models.find(id);
    if (found == impl.state.models.end()) throw std::runtime_error("Unknown model");
    RecordProvenance result; result.formId = id;
    result.activeFile = impl.path.filename().string();
    result.definingFile = slotName(impl.canonical, id);
    result.definedByActive = (id >> 24) + 1 == impl.canonical.size();
    result.activeHasOverride = !result.definedByActive && impl.document->containsRecord(id);
    result.editCreatesOverride = !result.definedByActive && !result.activeHasOverride;
    const auto& origin = found->second.recordSourceEsm.empty() ? found->second.sourceEsm : found->second.recordSourceEsm;
    result.winningFile = result.definedByActive || result.activeHasOverride ? result.activeFile
        : origin.empty() ? result.definingFile : std::filesystem::path(origin).filename().string();
    const auto saved = impl.saved.models.find(id);
    result.pendingChange = saved == impl.saved.models.end() || saved->second != found->second;
    result.writable = impl.writable; result.blockedReason = impl.writable ? std::string() : impl.readOnlyReason;
    return result;
}
EditorDocument::~EditorDocument() = default;
EditorDocument::EditorDocument(EditorDocument&&) noexcept = default;
EditorDocument& EditorDocument::operator=(EditorDocument&&) noexcept = default;
const std::filesystem::path& EditorDocument::path() const noexcept { return mImpl->path; }
const std::filesystem::path& EditorDocument::basePath() const noexcept { return mImpl->base; }
bool EditorDocument::hasEditableTerrain() const noexcept { return mImpl->editableTerrain; }
std::filesystem::path EditorDocument::assetRoot() const { return assets::AssetResolver::projectAssetRoot(path()); }
std::uint32_t EditorDocument::worldFormId() const noexcept { return mImpl->document->worldFormId(); }
const std::vector<EditorReference>& EditorDocument::references() const noexcept { return mImpl->state.references; }
const EditorReference* EditorDocument::findReference(std::uint32_t id) const noexcept
{
    try { return mImpl->findIndexed(id); } catch (...) { return nullptr; }
}
std::uint64_t EditorDocument::revision() const noexcept { return mImpl->revision; }
const std::map<std::uint32_t, std::string>& EditorDocument::models() const noexcept { return mImpl->modelPaths; }
const assets::ESMModel* EditorDocument::model(std::uint32_t id) const noexcept
{
    const auto found = mImpl->state.models.find(id);
    return found == mImpl->state.models.end() ? nullptr : &found->second;
}
std::optional<std::string> EditorDocument::sourcePath(std::uint32_t id) const
{
    auto found = mImpl->state.models.find(id); if (found == mImpl->state.models.end() || found->second.sourcePath.empty()) return std::nullopt;
    return found->second.sourcePath;
}
std::optional<std::string> EditorDocument::sourceEsm(std::uint32_t id) const
{
    auto found = mImpl->state.models.find(id); if (found == mImpl->state.models.end()) return std::nullopt; return found->second.sourceEsm;
}
glm::mat4 EditorDocument::engineTransform(std::uint32_t id) const
{
    const auto* found = mImpl->findIndexed(id);
    if (!found) throw std::runtime_error("Editor reference no longer exists");
    return world::bethesdaReferenceTransform(found->placement);
}
std::uint32_t EditorDocument::nextFormId() const noexcept { return mImpl->next; }
void EditorDocument::setNextFormId(std::uint32_t next)
{
    if ((next >> 24) != (mImpl->next >> 24)) throw std::runtime_error("Terrain allocator uses a different ESM prefix");
    mImpl->next = std::max(mImpl->next, next);
}
void EditorDocument::synchronizeTerrain(const assets::EditableLandscape& terrain, bool recordUndo)
{
    mImpl->dirtyCache.reset();
    setNextFormId(terrain.nextFormId);
    if (!recordUndo)
    {
        std::set<assets::CellCoord> resident;
        for (const auto& [coord, forms] : terrain.cellForms)
            if ((mImpl->saved.terrain && mImpl->saved.terrain->containsCell(coord))
                || mImpl->document->terrainCells().contains(coord)
                || (mImpl->dem && mImpl->dem->contains(coord))) resident.insert(coord);
        const auto absorb = [&](State& state) {
            if (!state.terrain) state.terrain.emplace();
            for (const auto& [coord, forms] : terrain.cellForms)
                if (resident.contains(coord) && !state.terrain->containsCell(coord))
                    includeCell(*state.terrain, {terrain.cell(coord), forms.cell, forms.land}, true);
            state.terrain->nextFormId = mImpl->next;
        };
        absorb(mImpl->state); absorb(mImpl->saved);
        for (auto& state : mImpl->undo) absorb(state);
        for (auto& state : mImpl->redo) absorb(state);
        if (mImpl->transaction) absorb(*mImpl->transaction);
        for (const auto& [coord, forms] : terrain.cellForms)
            if (!mImpl->state.terrain->containsCell(coord))
                includeCell(*mImpl->state.terrain, {terrain.cell(coord), forms.cell, forms.land}, true);
        return;
    }
    mImpl->change([&] {
        if (!mImpl->state.terrain) mImpl->state.terrain = terrain;
        else
        {
            for (const auto& [coord, forms] : terrain.cellForms) includeCell(*mImpl->state.terrain, {terrain.cell(coord), forms.cell, forms.land});
            mImpl->state.terrain->nextFormId = mImpl->next;
        }
    });
}
const std::optional<assets::EditableLandscape>& EditorDocument::terrainSnapshot() const noexcept { return mImpl->state.terrain; }
std::uint32_t EditorDocument::reserveModelId() { mImpl->requireWritable(); return mImpl->allocate(); }
std::uint32_t EditorDocument::commitImportedModel(std::uint32_t id, std::string modelPath, std::string source, const glm::mat4& transform)
{
    std::uint32_t placed = 0;
    if ((id >> 24) != (mImpl->next >> 24) || id >= mImpl->next) throw std::runtime_error("Import FormID was not reserved");
    mImpl->change([&] {
        if (mImpl->state.models.contains(id)) throw std::runtime_error("Import FormID already exists");
        mImpl->state.models[id] = {id, "OpenF4Model_" + std::to_string(id), std::move(modelPath), std::move(source), path().string()};
        placed = mImpl->place(id, transform, {});
    }); return placed;
}
std::uint32_t EditorDocument::importModel(const std::filesystem::path& source)
{
    const auto id = reserveModelId(); auto imported = assets::importAndCopyModelAsset(source, path(), id);
    mImpl->change([&] { mImpl->state.models[id] = {id, "OpenF4Model_" + std::to_string(id), imported.modelPath, source.string(), path().string()}; }); return id;
}
std::uint32_t EditorDocument::addModel(std::string modelPath, std::string source, assets::ESMModelKind kind)
{
    validateModelKind(kind);
    const auto id = reserveModelId();
    mImpl->change([&] {
        mImpl->state.models[id] = {id, "OpenF4Model_" + std::to_string(id), std::move(modelPath), std::move(source), path().string(), kind};
        mImpl->defaultWake(id);
    }); return id;
}
std::uint32_t EditorDocument::addNativeModel(std::string modelPath, assets::ESMModelKind kind)
{
    validateModelKind(kind);
    return addModel(nativePackagePath(assetRoot(), std::move(modelPath)), {}, kind);
}
std::uint32_t EditorDocument::assignNativeModel(std::uint32_t id, std::string modelPath, assets::ESMModelKind kind)
{
    validateModelKind(kind);
    modelPath = nativePackagePath(assetRoot(), std::move(modelPath));
    std::uint32_t result = id;
    mImpl->change([&] {
        const auto found = mImpl->state.models.find(id);
        if (found == mImpl->state.models.end()) throw std::runtime_error("Cannot assign an unknown model");
        auto replacement = found->second;
        const bool becomesMovable = kind == assets::ESMModelKind::movableStatic && replacement.kind != kind;
        if (replacement.kind != kind)
        {
            result = mImpl->allocate(); replacement.formId = result;
            replacement.editorId = "OpenF4Model_" + std::to_string(result);
            replacement.recordSourceEsm.clear(); replacement.recordSourceFormId = 0;
            for (auto& ref : mImpl->state.references)
                if (ref.placement.baseObject == id)
                {
                    ref.placement.baseObject = result;
                    if (ref.childGroupType != 8) mImpl->ensureCell(ref.placement.cell);
                }
        }
        replacement.path = std::move(modelPath); replacement.kind = kind; replacement.sourceEsm = path().string();
        mImpl->state.models[result] = std::move(replacement);
        if (becomesMovable) mImpl->defaultWake(result);
    });
    return result;
}
std::uint32_t EditorDocument::makeUnique(std::uint32_t reference, std::string modelPath, assets::ESMModelKind kind)
{
    validateModelKind(kind);
    modelPath = nativePackagePath(assetRoot(), std::move(modelPath));
    std::uint32_t result = 0;
    mImpl->change([&] {
        auto& ref = mImpl->reference(reference);
        if (ref.locked) throw std::runtime_error("Cannot make a locked reference unique");
        if (ref.placement.actor) throw std::runtime_error("Actors (ACHR) have no model to make unique");
        auto replacement = mImpl->state.models.at(ref.placement.baseObject);
        result = mImpl->allocate(); replacement.formId = result;
        replacement.editorId = "OpenF4Model_" + std::to_string(result);
        if (replacement.kind != kind) { replacement.recordSourceEsm.clear(); replacement.recordSourceFormId = 0; }
        replacement.path = std::move(modelPath); replacement.kind = kind; replacement.sourceEsm = path().string();
        mImpl->state.models[result] = std::move(replacement); ref.placement.baseObject = result;
        if (ref.childGroupType != 8) mImpl->ensureCell(ref.placement.cell);
        mImpl->defaultWake(result);
    });
    return result;
}
void EditorDocument::retarget(std::uint32_t id, std::uint32_t model)
{
    mImpl->change([&] {
        if (!mImpl->state.models.contains(model)) throw std::runtime_error("Cannot retarget to an unknown model");
        auto& ref = mImpl->reference(id);
        if (ref.locked) throw std::runtime_error("Cannot retarget a locked reference");
        if (ref.placement.actor) throw std::runtime_error("Actors (ACHR) need an NPC_ base; use setActorBase");
        ref.placement.baseObject = model;
        if (ref.childGroupType != 8) mImpl->ensureCell(ref.placement.cell);
    });
}
void EditorDocument::setModelScripts(std::uint32_t id, assets::PapyrusBindings scripts)
{
    mImpl->change([&] {
        auto found = mImpl->state.models.find(id);
        if (found == mImpl->state.models.end()) throw std::runtime_error("Cannot attach scripts to an unknown model");
        if (!found->second.scripts.editable) throw std::runtime_error("Existing model VMAD is read-only: " + found->second.scripts.diagnostic);
        scripts.present = !scripts.scripts.empty(); scripts.edited = true; scripts.sourceEsm = path().string();
        (void)assets::encodePapyrusBindings(scripts, false);
        if (scripts.present == found->second.scripts.present && scripts.scripts == found->second.scripts.scripts) return;
        found->second.scripts = std::move(scripts); found->second.recordOwnerEsm = path().string();
    });
}
void EditorDocument::setStartAwakeByDefault(bool enabled) noexcept { mImpl->startAwake = enabled; }
bool EditorDocument::startAwakeByDefault() const noexcept { return mImpl->startAwake; }
std::uint64_t EditorDocument::wakeAttachments() const noexcept { return mImpl->wakeAttachments; }
std::size_t EditorDocument::migrateStartAwake()
{
    std::size_t count = 0;
    // Never auto-edit third-party plugins or read-only targets.
    if (!mImpl->writable || !mImpl->document->isOpenF4Owned()) return 0;
    const auto prefix = mImpl->next >> 24;
    const auto own = path().string();
    mImpl->change([&] {
        for (auto& [id, model] : mImpl->state.models)
        {
            if ((id >> 24) != prefix || model.sourceEsm != own) continue;
            if (model.kind != assets::ESMModelKind::movableStatic) continue;
            if (mImpl->attachWake(id)) ++count;
        }
    });
    return count;
}
void EditorDocument::setReferenceScripts(std::uint32_t id, assets::PapyrusBindings scripts)
{
    mImpl->change([&] {
        auto& ref = mImpl->reference(id);
        if (ref.locked) throw std::runtime_error("Cannot edit scripts on a locked reference");
        if (!ref.scripts.editable) throw std::runtime_error("Existing reference VMAD is read-only: " + ref.scripts.diagnostic);
        scripts.present = !scripts.scripts.empty(); scripts.edited = true; scripts.sourceEsm = path().string();
        (void)assets::encodePapyrusBindings(scripts, false);
        if (scripts.present == ref.scripts.present && scripts.scripts == ref.scripts.scripts) return;
        ref.scripts = std::move(scripts);
        if (ref.childGroupType != 8) mImpl->ensureCell(ref.placement.cell);
    });
}
std::uint32_t EditorDocument::place(std::uint32_t model, const glm::mat4& transform, std::string name)
{
    std::uint32_t result = 0; mImpl->change([&] { result = mImpl->place(model, transform, std::move(name)); }); return result;
}
const std::map<std::uint32_t, assets::ESMActorBase>& EditorDocument::actorBases() const noexcept { return mImpl->actorBases; }
const assets::ESMActorBase* EditorDocument::actorBase(std::uint32_t id) const noexcept
{
    const auto found = mImpl->actorBases.find(id);
    return found == mImpl->actorBases.end() ? nullptr : &found->second;
}
std::uint32_t EditorDocument::placeActor(std::uint32_t base, const glm::mat4& transform, std::string name)
{
    std::uint32_t result = 0; mImpl->change([&] { result = mImpl->placeActor(base, transform, std::move(name)); }); return result;
}
void EditorDocument::setActorBase(std::uint32_t id, std::uint32_t base)
{
    mImpl->change([&] {
        auto& ref = mImpl->reference(id);
        if (!ref.placement.actor) throw std::runtime_error("Only actors (ACHR) take an NPC_ base");
        if (ref.locked) throw std::runtime_error("Cannot change the base of a locked reference");
        mImpl->validateActorBase(base);
        if (ref.placement.baseObject == base) return;
        ref.placement.baseObject = base;
        if (ref.childGroupType != 8) mImpl->ensureCell(ref.placement.cell);
    });
}
void EditorDocument::setTransform(std::uint32_t id, const glm::mat4& transform)
{
    // Gizmo drags call this every frame inside a transaction, which already holds the rollback
    // and undo snapshot. When the move needs no new terrain cell, skip change()'s full-state
    // copy (every reference of the worldspace) and edit the one placement in place.
    if (auto& impl = *mImpl; impl.transactionDepth)
    {
        impl.requireWritable();
        auto& ref = impl.reference(id);
        if (ref.locked) return;
        auto placement = ref.placement;
        world::setBethesdaReferenceTransform(placement, transform);
        if (ref.childGroupType == 8 || (impl.state.terrain && impl.state.terrain->containsCell(placement.cell)))
        {
            if (placement != ref.placement)
            {
                ref.placement = placement;
                impl.dirtyCache = true;  // recomputed exactly at endTransaction
                impl.revision = Impl::nextRevision();
            }
            return;
        }
    }
    mImpl->change([&] {
        auto& ref = mImpl->reference(id); if (ref.locked) return;
        world::setBethesdaReferenceTransform(ref.placement, transform);
        if (ref.childGroupType != 8) mImpl->ensureCell(ref.placement.cell);
    });
}
void EditorDocument::erase(std::uint32_t id)
{
    mImpl->change([&] { if (mImpl->reference(id).locked) return;
        std::erase_if(mImpl->state.references, [&](const auto& r) { return r.placement.formId == id; }); });
}
namespace
{
std::string hexId(std::uint32_t id)
{
    static constexpr char digits[] = "0123456789ABCDEF";
    std::string out(8, '0');
    for (int i = 7; i >= 0; --i) { out[std::size_t(i)] = digits[id & 0xFu]; id >>= 4; }
    return out;
}
}
ReferenceDeletePlan EditorDocument::planReferenceDelete(std::uint32_t id, bool hard) const
{
    const auto& impl = *mImpl;
    const auto* found = impl.findIndexed(id);
    if (!found) throw std::runtime_error("Editor reference no longer exists");
    ReferenceDeletePlan plan; plan.hard = hard; plan.provenance = referenceProvenance(id);
    const auto& p = plan.provenance;
    const auto name = (found->name.empty() ? std::string("reference") : found->name) + " [" + hexId(id) + "]";
    if (!p.writable)
    {
        plan.action = ReferenceDeleteAction::blocked;
        plan.blockedReason = p.blockedReason.empty() ? std::string("This reference cannot be edited") : p.blockedReason;
        plan.summary = "Delete blocked: " + plan.blockedReason;
        return plan;
    }
    // Only an OpenF4-authored world ESM (not a plugin, not a third-party master) treats its
    // own placements as scratch objects that plain Delete may remove outright.
    const bool nativeWorld = !impl.pluginFile && impl.document->isOpenF4Owned();
    if (!hard)
    {
        if (p.definedByActive && nativeWorld)
        {
            plan.action = ReferenceDeleteAction::removeRecord;
            plan.summary = "Remove " + name + " from " + p.activeFile + " (an OpenF4 world's own object).";
            return plan;
        }
        if (found->disabled)
        {
            plan.action = ReferenceDeleteAction::alreadyDisabled;
            plan.summary = name + " is already disabled. Ctrl+Shift+Delete deletes it permanently.";
            return plan;
        }
        plan.action = ReferenceDeleteAction::disable;
        plan.summary = p.definedByActive
            ? "Disable " + name + " in " + p.activeFile + " (Initially Disabled; the record is kept)."
            : p.activeHasOverride
                ? "Disable " + name + " through the existing override in " + p.activeFile + "; " + p.definingFile + " is not modified."
                : "Disable " + name + " through a NEW override in " + p.activeFile + " (same FormID); " + p.definingFile + " is not modified.";
        return plan;
    }
    plan.requiresConfirmation = true;
    if (p.definedByActive)
    {
        plan.action = ReferenceDeleteAction::removeRecord;
        plan.erasesFromMaster = !impl.pluginFile && !impl.document->isOpenF4Owned();
        plan.summary = "Permanently remove " + name + " from " + p.activeFile + ".";
        plan.warning = plan.erasesFromMaster
            ? "THIS ERASES THE RECORD FROM THE MASTER FILE " + p.activeFile + ". Every plugin that references this FormID "
              "will point at nothing; this is usually unrecoverable without the backup."
            : "The record is erased from " + p.activeFile + ". Any other plugin that references this FormID will point at nothing.";
        return plan;
    }
    plan.action = ReferenceDeleteAction::deletedOverride;
    plan.crashRisk = true;
    plan.summary = "Write a Deleted-flag override of " + name + " (defined in " + p.definingFile + ") into "
        + p.activeFile + ". " + p.definingFile + " itself is not modified.";
    plan.warning = "Deleting a reference that another file defines can CRASH Fallout 4: navmeshes, enable parents, "
        "linked refs, scripts, quests and precombined/previs data in " + p.definingFile + " or other plugins may still "
        "point at it. Delete (disable) is the safe choice unless you are certain nothing uses this reference.";
    return plan;
}
ReferenceDeleteAction EditorDocument::deleteReference(std::uint32_t id, bool hard, bool confirmed)
{
    mImpl->requireWritable();
    const auto plan = planReferenceDelete(id, hard);
    switch (plan.action)
    {
    case ReferenceDeleteAction::blocked:
        throw std::runtime_error(plan.summary);
    case ReferenceDeleteAction::alreadyDisabled:
        break;
    case ReferenceDeleteAction::disable:
        setDisabled(id, true);
        break;
    case ReferenceDeleteAction::removeRecord:
    case ReferenceDeleteAction::deletedOverride:
        if (plan.requiresConfirmation && !confirmed)
            throw std::runtime_error("Permanent delete requires explicit confirmation (Ctrl+Shift+Delete)");
        mImpl->change([&] {
            if (mImpl->reference(id).locked) throw std::runtime_error("Cannot delete a locked reference");
            std::erase_if(mImpl->state.references, [&](const auto& r) { return r.placement.formId == id; });
        });
        break;
    }
    return plan.action;
}
void EditorDocument::setDisabled(std::uint32_t id, bool disabled)
{
    mImpl->change([&] {
        auto& ref = mImpl->reference(id);
        if (ref.locked) throw std::runtime_error("Cannot enable/disable a locked reference");
        if (ref.disabled == disabled) return;
        ref.disabled = disabled;
        if (ref.childGroupType != 8) mImpl->ensureCell(ref.placement.cell);
    });
}
std::uint32_t EditorDocument::duplicate(std::uint32_t id)
{
    std::uint32_t result = 0; mImpl->change([&] {
        auto copy = mImpl->reference(id); copy.placement.formId = mImpl->allocate(); copy.placement.instanceId = copy.placement.formId;
        copy.name += " copy"; copy.recordSource = {}; copy.ownerCellFormId = 0; copy.childGroupType = 9; copy.disabled = false;
        mImpl->ensureCell(copy.placement.cell);
        result = copy.placement.formId; mImpl->state.references.push_back(copy);
    }); return result;
}
void EditorDocument::setVisible(std::uint32_t id, bool visible) { mImpl->change([&] { mImpl->reference(id).visible = visible; }); }
void EditorDocument::setLocked(std::uint32_t id, bool locked) { mImpl->change([&] { mImpl->reference(id).locked = locked; }); }
void EditorDocument::beginTransaction()
{
    if (mImpl->transactionDepth++ == 0)
    {
        mImpl->transaction = mImpl->state;
        mImpl->transactionNext = mImpl->next;
    }
}
void EditorDocument::endTransaction()
{
    if (!mImpl->transactionDepth) return;
    if (--mImpl->transactionDepth == 0)
    {
        mImpl->dirtyCache.reset(); mImpl->touch();
        if (*mImpl->transaction != mImpl->state)
        {
            mImpl->undo.push_back(std::move(*mImpl->transaction)); if (mImpl->undo.size() > 64) mImpl->undo.erase(mImpl->undo.begin()); mImpl->redo.clear();
        }
        mImpl->transaction.reset();
    }
}
void EditorDocument::cancelTransaction()
{
    if (!mImpl->transactionDepth) return;
    mImpl->state = std::move(*mImpl->transaction);
    mImpl->next = mImpl->transactionNext;
    mImpl->transaction.reset();
    mImpl->transactionDepth = 0;
    mImpl->dirtyCache.reset();
    mImpl->refreshPaths();
    mImpl->touch();
}
bool EditorDocument::undo()
{
    if (mImpl->transactionDepth || mImpl->undo.empty()) return false;
    mImpl->requireWritable();
    mImpl->dirtyCache.reset();
    mImpl->redo.push_back(mImpl->state); mImpl->state = std::move(mImpl->undo.back()); mImpl->undo.pop_back(); mImpl->refreshPaths(); mImpl->touch(); return true;
}
bool EditorDocument::redo()
{
    if (mImpl->transactionDepth || mImpl->redo.empty()) return false;
    mImpl->requireWritable();
    mImpl->dirtyCache.reset();
    mImpl->undo.push_back(mImpl->state); mImpl->state = std::move(mImpl->redo.back()); mImpl->redo.pop_back(); mImpl->refreshPaths(); mImpl->touch(); return true;
}
bool EditorDocument::dirty() const noexcept
{
    if (!mImpl->dirtyCache) mImpl->dirtyCache = mImpl->state != mImpl->saved;
    return *mImpl->dirtyCache;
}
void EditorDocument::save() { mImpl->save(path()); }
void EditorDocument::saveAs(std::filesystem::path target)
{
    mImpl->requireWritable();
    target = std::filesystem::absolute(target).lexically_normal();
    if (target == path()) { save(); return; }
    if (std::filesystem::exists(target)) throw std::runtime_error("Save As destination already exists; choose a new ESM filename");
    if (target.parent_path() != path().parent_path())
        for (const auto& master : mImpl->document->masters())
            if (std::filesystem::exists(path().parent_path() / master))
                throw std::runtime_error("Save As must stay beside this project's local ESM masters. Choose a sibling filename; relocating master/shard dependencies is not supported yet.");
    const auto sourceBtr = path().parent_path() / (path().stem().string() + ".btr");
    const auto targetBtr = target.parent_path() / (target.stem().string() + ".btr");
    const std::array<std::pair<std::filesystem::path, std::filesystem::path>, 2> sidecars{{
        {assetRoot(), assets::AssetResolver::projectAssetRoot(target)}, {sourceBtr, targetBtr}}};
    for (const auto& [source, destination] : sidecars)
        if (std::filesystem::exists(destination)) throw std::runtime_error("Save As sidecar directory already exists: " + destination.string());
    std::vector<std::filesystem::path> created;
    try
    {
        if (target.has_parent_path()) std::filesystem::create_directories(target.parent_path());
        for (const auto& [source, destination] : sidecars) if (std::filesystem::exists(source))
        {
            // Only these newly created, exact sidecar targets are removed on failure.
            created.push_back(destination);
            std::filesystem::copy(source, destination, std::filesystem::copy_options::recursive);
        }
        mImpl->save(std::move(target));
    }
    catch (...)
    {
        for (const auto& destination : created) { std::error_code ignored; std::filesystem::remove_all(destination, ignored); }
        throw;
    }
}
}
