#include "editor/editor_document.hpp"
#include "assets/asset_resolver.hpp"
#include "assets/dem_import.hpp"
#include "assets/model_import.hpp"
#include "world/world_scene.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <set>
#include <stdexcept>

namespace openf4::editor
{
namespace
{
struct State
{
    std::vector<EditorReference> references;
    std::map<std::uint32_t, assets::ESMModel> models;
    std::optional<assets::EditableLandscape> terrain;
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
    unsigned transactionDepth = 0;
    std::map<std::uint32_t, std::string> modelPaths;
    std::map<std::uint32_t, assets::ESMReference> baselineRefs;
    bool editableTerrain = false;
    std::uint32_t next = 0;

    Impl(std::filesystem::path p, std::optional<std::filesystem::path> data)
        : path(std::filesystem::absolute(p)), dataDirectory(std::move(data))
    {
        if (!std::filesystem::exists(path)) assets::saveEditableLandscape(path, assets::createEditableLandscape());
        document = std::make_unique<assets::EsmDocument>(path);
        if (!document->isOpenF4Owned())
        {
            path = assets::createEditorOverlay(path);
            document = std::make_unique<assets::EsmDocument>(path);
        }
        const auto stack = assets::loadEsmMasterStack(path, dataDirectory); base = std::filesystem::absolute(stack.basePath);
        editableTerrain = base == path;
        if (document->isDemOverlay()) dem = std::make_unique<assets::DemLandscapeProject>(path);
        for (const auto& m : stack.models) state.models.emplace(m.formId, m);
        for (const auto& r : stack.references)
        {
            state.references.push_back({r.placement, r.editorId}); baselineRefs[r.placement.formId] = r;
        }
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
        auto found = std::find_if(state.references.begin(), state.references.end(), [&](const auto& r) { return r.placement.formId == id; });
        if (found == state.references.end()) throw std::runtime_error("Editor reference no longer exists"); return *found;
    }
    template<class F> void change(F operation)
    {
        State before = state;
        try { operation(); }
        catch (...) { state = std::move(before); refreshPaths(); throw; }
        dirtyCache.reset();
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
            if (!baseDocument) baseDocument = std::make_unique<assets::EsmDocument>(base);
            if (auto existing = baseDocument->terrainCell(coord); existing && existing->landscape.hasHeights)
            { includeCell(terrain, *existing, true); return; }
        }
        terrain.nextFormId = next; (void)terrain.addCell(coord); next = terrain.nextFormId;
    }
    std::uint32_t place(std::uint32_t model, const glm::mat4& transform, std::string name)
    {
        if (!state.models.contains(model)) throw std::runtime_error("Cannot place an unknown model");
        EditorReference ref; ref.placement.formId = allocate(); ref.placement.instanceId = ref.placement.formId;
        ref.placement.baseObject = model;
        world::setBethesdaReferenceTransform(ref.placement, transform);
        ensureCell(ref.placement.cell); ref.name = name.empty() ? "Object_" + std::to_string(ref.placement.formId) : std::move(name);
        const auto id = ref.placement.formId; state.references.push_back(std::move(ref)); return id;
    }
    void save(std::filesystem::path target)
    {
        // Terrain tools may have saved since this document was opened. Re-index the
        // current source and apply only authored changes instead of stale file bytes.
        assets::EsmDocument output(path);
        output.setNextFormId(std::max(next, output.nextFormId()));
        std::vector<assets::ESMModel> models = output.models();
        for (const auto& [id, m] : state.models)
        {
            if (m.sourceEsm != path.string()) continue;
            auto found = std::find_if(models.begin(), models.end(), [&](const auto& v) { return v.formId == id; });
            if (found == models.end()) models.push_back(m); else *found = m;
        }
        output.setModels(std::move(models));
        std::vector<assets::ESMReference> references;
        std::set<std::uint32_t> current;
        std::set<std::uint32_t> local;
        for (const auto& r : output.references()) local.insert(r.placement.formId);
        for (const auto& r : state.references)
        {
            current.insert(r.placement.formId);
            assets::ESMReference authored{r.placement, r.name};
            auto original = baselineRefs.find(r.placement.formId);
            if (local.contains(r.placement.formId) || original == baselineRefs.end() || original->second != authored) references.push_back(std::move(authored));
        }
        for (const auto& [id, r] : baselineRefs) if (!current.contains(id) && (id >> 24) != (next >> 24))
        { auto deleted = r; deleted.deleted = true; references.push_back(std::move(deleted)); }
        // Retain existing tombstones, even though they are absent from merged authoring state.
        for (const auto& r : output.references()) if (r.deleted && !current.contains(r.placement.formId)) references.push_back(r);
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
                if (modified || needsOwnership) changes.push_back({state.terrain->cell(coord), forms.cell, forms.land});
            }
        }
        output.replaceTerrainCells(changes); output.setPath(target); output.save();
        const auto oldPath = path; path = std::filesystem::absolute(target);
        if (base == oldPath) base = path;
        const auto relocate = [&](State& snapshot) {
            for (auto& [id, m] : snapshot.models) if (m.sourceEsm == oldPath.string()) m.sourceEsm = path.string();
        };
        relocate(state); for (auto& snapshot : undo) relocate(snapshot); for (auto& snapshot : redo) relocate(snapshot);
        if (transaction) relocate(*transaction);
        document = std::make_unique<assets::EsmDocument>(path); next = std::max(next, document->nextFormId());
        if (dem && path != oldPath) dem = std::make_unique<assets::DemLandscapeProject>(path);
        saved = state; dirtyCache = false;
        baselineRefs.clear(); for (const auto& r : state.references) baselineRefs[r.placement.formId] = {r.placement, r.name};
    }
};

EditorDocument::EditorDocument(std::filesystem::path path, std::optional<std::filesystem::path> dataDirectory)
    : mImpl(std::make_unique<Impl>(std::move(path), std::move(dataDirectory))) {}
EditorDocument::~EditorDocument() = default;
EditorDocument::EditorDocument(EditorDocument&&) noexcept = default;
EditorDocument& EditorDocument::operator=(EditorDocument&&) noexcept = default;
const std::filesystem::path& EditorDocument::path() const noexcept { return mImpl->path; }
const std::filesystem::path& EditorDocument::basePath() const noexcept { return mImpl->base; }
bool EditorDocument::hasEditableTerrain() const noexcept { return mImpl->editableTerrain; }
std::filesystem::path EditorDocument::assetRoot() const { return assets::AssetResolver::projectAssetRoot(path()); }
std::uint32_t EditorDocument::worldFormId() const noexcept { return mImpl->document->worldFormId(); }
const std::vector<EditorReference>& EditorDocument::references() const noexcept { return mImpl->state.references; }
const std::map<std::uint32_t, std::string>& EditorDocument::models() const noexcept { return mImpl->modelPaths; }
std::optional<std::string> EditorDocument::sourcePath(std::uint32_t id) const
{
    auto found = mImpl->state.models.find(id); if (found == mImpl->state.models.end() || found->second.sourcePath.empty()) return std::nullopt;
    return found->second.sourcePath;
}
std::optional<std::string> EditorDocument::sourceEsm(std::uint32_t id) const
{
    auto found = mImpl->state.models.find(id); if (found == mImpl->state.models.end()) return std::nullopt; return found->second.sourceEsm;
}
glm::mat4 EditorDocument::engineTransform(std::uint32_t id) const { return world::bethesdaReferenceTransform(mImpl->reference(id).placement); }
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
std::uint32_t EditorDocument::reserveModelId() { return mImpl->allocate(); }
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
std::uint32_t EditorDocument::addModel(std::string modelPath, std::string source)
{
    const auto id = reserveModelId();
    mImpl->change([&] { mImpl->state.models[id] = {id, "OpenF4Model_" + std::to_string(id), std::move(modelPath), std::move(source), path().string()}; }); return id;
}
std::uint32_t EditorDocument::place(std::uint32_t model, const glm::mat4& transform, std::string name)
{
    std::uint32_t result = 0; mImpl->change([&] { result = mImpl->place(model, transform, std::move(name)); }); return result;
}
void EditorDocument::setTransform(std::uint32_t id, const glm::mat4& transform)
{
    mImpl->change([&] {
        auto& ref = mImpl->reference(id); if (ref.locked) return;
        world::setBethesdaReferenceTransform(ref.placement, transform); const auto coord = ref.placement.cell; mImpl->ensureCell(coord);
    });
}
void EditorDocument::erase(std::uint32_t id)
{
    mImpl->change([&] { if (mImpl->reference(id).locked) return;
        std::erase_if(mImpl->state.references, [&](const auto& r) { return r.placement.formId == id; }); });
}
std::uint32_t EditorDocument::duplicate(std::uint32_t id)
{
    std::uint32_t result = 0; mImpl->change([&] {
        auto copy = mImpl->reference(id); copy.placement.formId = mImpl->allocate(); copy.placement.instanceId = copy.placement.formId;
        copy.name += " copy"; result = copy.placement.formId; mImpl->state.references.push_back(copy);
    }); return result;
}
void EditorDocument::setVisible(std::uint32_t id, bool visible) { mImpl->change([&] { mImpl->reference(id).visible = visible; }); }
void EditorDocument::setLocked(std::uint32_t id, bool locked) { mImpl->change([&] { mImpl->reference(id).locked = locked; }); }
void EditorDocument::beginTransaction()
{
    if (mImpl->transactionDepth++ == 0) mImpl->transaction = mImpl->state;
}
void EditorDocument::endTransaction()
{
    if (!mImpl->transactionDepth) return;
    if (--mImpl->transactionDepth == 0)
    {
        if (*mImpl->transaction != mImpl->state)
        {
            mImpl->undo.push_back(std::move(*mImpl->transaction)); if (mImpl->undo.size() > 64) mImpl->undo.erase(mImpl->undo.begin()); mImpl->redo.clear();
        }
        mImpl->transaction.reset();
    }
}
bool EditorDocument::undo()
{
    if (mImpl->transactionDepth || mImpl->undo.empty()) return false;
    mImpl->dirtyCache.reset();
    mImpl->redo.push_back(mImpl->state); mImpl->state = std::move(mImpl->undo.back()); mImpl->undo.pop_back(); mImpl->refreshPaths(); return true;
}
bool EditorDocument::redo()
{
    if (mImpl->transactionDepth || mImpl->redo.empty()) return false;
    mImpl->dirtyCache.reset();
    mImpl->undo.push_back(mImpl->state); mImpl->state = std::move(mImpl->redo.back()); mImpl->redo.pop_back(); mImpl->refreshPaths(); return true;
}
bool EditorDocument::dirty() const noexcept
{
    if (!mImpl->dirtyCache) mImpl->dirtyCache = mImpl->state != mImpl->saved;
    return *mImpl->dirtyCache;
}
void EditorDocument::save() { mImpl->save(path()); }
void EditorDocument::saveAs(std::filesystem::path target)
{
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
