// Editor hot-path benchmark on a real load: whar-style plugin over Fallout4.esm/Commonwealth.
// Compares the per-frame document work the editor used to do (full scans/copies) with the
// cached/indexed paths. Never saves: the document is only read and edited in memory.
//   openf4-editor-perf-bench <plugin.esm|esp> [worldFormIdHex=3C]
#include "assets/fo4_install.hpp"
#include "editor/editor_document.hpp"
#include <algorithm>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace
{
using Clock = std::chrono::steady_clock;
template<class F> double timeMs(int iterations, F&& f)
{
    const auto start = Clock::now();
    for (int i = 0; i < iterations; ++i) f();
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count() / iterations;
}
volatile std::size_t sink = 0;
}

int main(int argc, char** argv)
try
{
    using namespace openf4;
    if (argc < 2) { std::cerr << "usage: openf4-editor-perf-bench <plugin> [worldFormIdHex]\n"; return 2; }
    const auto data = assets::findFallout4Data();
    if (!data) { std::cerr << "Fallout 4 data not found\n"; return 77; }
    editor::EditorOpenOptions options;
    options.dataDirectory = *data; options.directEdit = true;
    options.worldFormId = static_cast<std::uint32_t>(std::stoul(argc > 2 ? argv[2] : "3C", nullptr, 16));
    options.resolvedFiles = {*data / "Fallout4.esm"};
    const auto openStart = Clock::now();
    editor::EditorDocument document(argv[1], options);
    std::printf("open: %.0f ms\n", std::chrono::duration<double, std::milli>(Clock::now() - openStart).count());
    const auto& refs = document.references();
    std::printf("references: %zu, models: %zu\n", refs.size(), document.models().size());
    if (refs.empty()) return 0;
    const auto selected = refs[refs.size() / 2].placement.formId;

    // --- old per-frame work ---
    const double oldOutliner = timeMs(5, [&] {
        for (const auto& r : refs)
        {
            const std::string label = (r.name.empty() ? "Object " + std::to_string(r.placement.formId) : r.name)
                + (r.visible ? "" : " [hidden]") + (r.locked ? " [locked]" : "")
                + (r.disabled ? " [disabled]" : "") + (r.placement.actor ? " [ACHR]" : "");
            sink += label.size();
        }
    });
    const double oldProjection = timeMs(5, [&] {
        std::vector<assets::PlacedReference> references; std::vector<std::uint32_t> hidden;
        references.reserve(refs.size());
        for (const auto& r : refs) { references.push_back(r.placement); if (!r.visible || r.disabled) hidden.push_back(r.placement.formId); }
        std::unordered_map<std::uint32_t, std::filesystem::path> sources;
        for (const auto& [id, path] : document.models()) if (auto s = document.sourceEsm(id)) sources.emplace(id, *s);
        sink += references.size() + sources.size();
    });
    const double oldLookup = timeMs(20, [&] {
        for (int k = 0; k < 4; ++k)  // reconcile x2, selectedReference, provenance
        {
            const auto it = std::find_if(refs.begin(), refs.end(), [&](const auto& r) { return r.placement.formId == selected; });
            sink += it != refs.end();
        }
    });
    const double oldModelRows = timeMs(5, [&] {
        for (const auto& [id, path] : document.models())
        {
            const auto* base = document.model(id);
            const std::string label = path + (base && base->kind == assets::ESMModelKind::movableStatic ? "  [MSTT]" : "  [STAT]") + "###model";
            sink += label.size();
        }
    });
    // --- new per-frame work ---
    (void)document.findReference(selected);  // build index
    const double newLookup = timeMs(2000, [&] { for (int k = 0; k < 4; ++k) sink += document.findReference(selected) != nullptr; });

    // --- drag: setTransform per frame inside a transaction ---
    auto transform = document.engineTransform(selected);
    document.beginTransaction();
    document.setTransform(selected, transform);  // the first move may load the target LAND cell
    const double dragFast = timeMs(60, [&] { transform[3].y += 1.0f; document.setTransform(selected, transform); });
    document.cancelTransaction();
    const double dragChange = timeMs(3, [&] { transform[3].y += 1.0f; document.setTransform(selected, transform); });  // change(): full-state copy

    std::printf("old per-frame (ms): outliner labels %.2f + projection copy %.2f + linear lookups %.3f + model rows %.2f = %.2f\n",
        oldOutliner, oldProjection, oldLookup, oldModelRows, oldOutliner + oldProjection + oldLookup + oldModelRows);
    std::printf("new per-frame (ms): indexed lookups %.5f (caches reused while revision %llu is unchanged)\n",
        newLookup, static_cast<unsigned long long>(document.revision()));
    std::printf("drag setTransform (ms/frame): transaction fast path %.4f vs full change() copy %.2f\n", dragFast, dragChange);
    return 0;
}
catch (const std::exception& error)
{
    std::cerr << "bench failed: " << error.what() << '\n';
    return 1;
}
