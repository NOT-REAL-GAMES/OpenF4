# OpenF4

OpenF4 is a clean SDL3 and Vulkan rewrite of the Fallout: Empire renderer. It does not link
VulkanSceneGraph or inherit the previous engine's frame graph and synchronization behavior.
OpenF4 names the runtime, editor, and shared engine; EMPIRE is the mod built on it.

## Current Core

- SDL3 window, Vulkan surface, high-DPI sizing, events, and input
- Raw Vulkan 1.3 renderer using `VK_EXT_mesh_shader`
- Two frames in flight with frame fences and swapchain-image-owned present semaphores
- Per-swapchain-image depth targets
- Growable device-local scene buffers with fence-backed append uploads
- Compute frustum/LOD culling into one indirect mesh-task draw
- Conservative dithered LOD overlap ranges
- Persistent directional shadow map using the compacted meshlet list
- GPU timestamps, CPU timings, and visible/source meshlet diagnostics
- Fallout 4 installation discovery and GNRL BA2 extraction
- 64-bit BA2 file and chunk offsets
- zlib and LZ4 decompression, including version-3 DX10 texture chunks
- DDS/DX10 reconstruction and Fallout 4 NIF geometry loading
- Commonwealth ESM exterior cells, blended LAND texture layers, and placed references
- Per-reference `XMSP` and base-object `MODS` material swaps through `MSWP` records
- BGSM/BGEM diffuse, normal, and grayscale-to-palette material binding
- BGSM and NIF alpha-mask thresholds for foliage and cutout materials
- Demand-grown bindless texture descriptors, doubling from 16 to a 4096-texture ceiling
- Sparse 2D image residency for large scene textures, including resident mip tails
- Distance-ordered world streaming that publishes one placed object per frame
- Bethesda references use the original scalar ZYX conversion, `(X,Z,Y)` intermediate space,
  Y/Z quaternion swap, and the final world-X shader flip
- Editor marker, collision, occlusion, previs, precombined, and helper-mesh filtering
- OpenF4-owned 3x3 ESM landscape creation, sculpting, undo/redo, and atomic save

The default scene opens the Commonwealth at a dense exterior cell, renders LAND first,
then streams nearby placed objects. Models and textures are cached and reused across
references.

## Build

```powershell
cmake -S . -B build -G "Visual Studio 18 2026" -A x64 `
  -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build build --config RelWithDebInfo --parallel
ctest --test-dir build -C RelWithDebInfo --output-on-failure
```

The vcpkg manifest installs the rendering, physics, GIS, and model-import dependencies,
including Assimp and stb. Dear ImGui docking and ImGuizmo are linked only by the editor.
Nifly is pinned to a specific revision through CMake FetchContent.

## Run

```powershell
build/bin/RelWithDebInfo/openf4.exe
build/bin/RelWithDebInfo/openf4.exe --benchmark 300
build/bin/RelWithDebInfo/openf4.exe --cubes --benchmark 300
build/bin/RelWithDebInfo/openf4.exe --cubes --resize-test --benchmark 60
build/bin/RelWithDebInfo/openf4.exe --model "meshes\architecture\example.nif"
build/bin/RelWithDebInfo/openf4.exe --cell 1 1 --radius 0
build/bin/RelWithDebInfo/openf4.exe --cell 1 1 --max-objects 100 --benchmark 130
build/bin/RelWithDebInfo/openf4.exe --land-only
build/bin/RelWithDebInfo/openf4.exe --editor "mods\OpenF4World.esm"
build/bin/RelWithDebInfo/openf4.exe "C:\Games\Fallout 4\Data\Fallout4.esm"
build/bin/RelWithDebInfo/openf4.exe --import-dem "mods\NYC.esm" `
  "USGS_1_n41w075_20221115.tif" "USGS_1_n41w074_20260820.tif" `
  "USGS_1_n42w074_20241010.tif" "USGS_1_n42w075_20240925.tif"
```

Passing a `.esm` path selects that master in read-only world mode. OpenF4 DEM
overlays stream nearby LAND and their baked level-32 BTR tiles; use `--editor` explicitly
to sculpt. Other masters stream the first exterior worldspace contained in that file.
OpenF4 editor overlays resolve their master stack and project-owned placed models.

## OpenF4 editor

```powershell
build/bin/Release/openf4-editor.exe "build/mods/NYC.esm"
build/bin/Release/openf4.exe "build/mods/NYC.esm"
```

The separate editor has a docked viewport, Outliner, Inspector, Asset Browser, and
Diagnostics. Use File/New for a small flat world, or File/Open for an existing ESM.
OpenF4 worlds save in place; protected masters use a sibling `.openf4.esm` overlay.
Untouched records are stream-copied during atomic saves, including unknown data.
Imported models are stored under `<world>.assets/models/<form-id>/`; their STAT and
REFR records are also loaded by the normal runtime. GLB/GLTF is the tested import path;
other static Assimp formats are accepted when usable. One glTF meter is 69.99125 units.

Hold RMB in the viewport to fly with WASD and Q/E; Shift accelerates. Drop a model file
into the window, use File/Import, or drag an imported Asset Browser entry into the view.
Click to select, Shift-click to toggle selection, F to frame, W/E/R to change gizmos,
Delete to remove, and Ctrl+D to duplicate. The Inspector exposes transforms, hide, and
lock. Ctrl+Z/Y undo/redo complete commands; Ctrl+S saves. Landscape mode reuses the
terrain tools described below, with the same document history and reference-safe save.

Object picking uses loaded mesh triangles (not just bounding spheres), with LAND
occlusion. The active object has an amber visible-only outline; other selected objects
have cyan outlines. Outliner and viewport selection share the same active reference.
The viewport toolbar selects World/Local axes and Active Object/Selection Center
pivots. Move and rotation snapping are off initially, with 128-unit and 15-degree
steps; snapping is relative to the drag's start. Scaling is always uniform, and
numeric Inspector entry remains exact. Escape cancels a drag and its cell ownership
changes; each group operation is a single undo step. Hidden/locked references are
excluded from transforms, duplication, and grounding.

Use **Build → Bake Changed LOD** after saving to update affected terrain and static
tiles, or **Bake All LOD** to rebuild the world. The first bake is always full.
The editor shows progress and allows cancellation; completed tiles refresh in the
open viewport. The headless equivalent is:

```powershell
build/bin/Release/openf4-editor.exe "build/mods/NYC.esm" --bake-lod --yes
build/bin/Release/openf4-editor.exe "build/mods/NYC.esm" --bake-lod --full --terrain-only --yes
```

`--statics-only` selects static tiles instead. Preflight prints the tile count and
estimated size; jobs over 10,000 tiles require editor confirmation or CLI `--yes`.
Terrain BTRs are written to `<world>.btr/` at levels 4, 8, 16, and 32. Simplified
static glTF tiles and the SQLite change index live in `<world>.assets/lod/`.
These are derived products: the ESM and its terrain shards remain authoritative.

`End` or **Drop to Ground** places each selected object's lowest geometry point at
the loaded LAND height beneath its origin. Optional **Align to terrain** tilts local
up to the terrain normal while keeping heading. Water, BTRs, and other objects are
not grounding surfaces; unavailable LAND/models are skipped with a diagnostic.
Placement preferences are session-local. Transparent holes are still geometric for
picking; texture-alpha-aware selection is not implemented.

Basic imported materials support base color, PNG/JPEG textures, and alpha masks.
Blend materials currently fall back to alpha cutout; double-sided materials still use
back-face culling. Diagnostics report these fallbacks. Tangent/PBR metadata is retained
for the later material pass. Advanced material editing and external
Bethesda-tool compatibility remain follow-on work. Save As of a sharded project stays
beside its local terrain masters; relocating those masters is not automatic.

Place `--bake-btr` immediately after `--import-dem` to bake sharded level-32 distant
terrain after the ESM masters finish. Imported exterior cells use a zero-meter sea-level
water plane by default. The DEM editor keeps a 9-by-9 CELL window resident around the
camera and displays that water plane while editing.

Use `--inspect-dem` with the same arguments to validate inputs and print shard sizes
without writing the multi-gigabyte import.

Set `FALLOUT4_PATH` when Fallout 4 is outside the usual Steam folders. It may point to the
game root or directly to `Data`.

Controls are `WASD`, `Space`/`C`, mouse look, left Shift for fast movement, and Escape to
release or recapture the mouse.

Editor controls are `1` raise, `2` lower, `3` smooth, `4` flatten, left-drag to sculpt,
mouse wheel for radius, Shift+wheel for strength, `Ctrl+Z`/`Ctrl+Y` for history, and
`Ctrl+S` to save. Entering an exterior cell not yet present in the editor ESM creates and
persists it automatically. Editor ESM files are intentionally OpenF4-only for now.
The DEM importer accepts the four contiguous USGS one-arc-second NAD83 tiles in any order,
uses true Fallout scale, and writes spatial terrain masters capped at 1,048,576 cells each.

## Synchronization Policy

There is no `vkQueueWaitIdle`. `vkDeviceWaitIdle` is limited to swapchain recreation and
shutdown. Waiting elsewhere is limited to the one startup upload fence and the normal
per-frame/resource fence before reusing or publishing frame-owned resources. Set
`OPENF4_DISABLE_SPARSE=1` only when comparing sparse and conventional image allocation.

## Legacy compatibility

Use `OPENF4_BUILD_TESTS` for the CMake test option. The old `EMPIRE_BUILD_TESTS` option
is accepted as a deprecated fallback when `OPENF4_BUILD_TESTS` has not been set.
`EMPIRE_DISABLE_SPARSE` remains an alias for `OPENF4_DISABLE_SPARSE`; either variable's
presence disables sparse allocation, preserving the previous behavior.

Existing EMPIRE-authored ESMs, `EMPIRE_DEM 1` metadata, and `empireworld.32.*.btr` files
remain readable without conversion or rebaking. Ordinary saves preserve their existing
identifiers; newly generated worlds and terrain use OpenF4 names. When both BTR name
prefixes exist for one tile, the OpenF4 file takes precedence. User world filenames,
master dependencies, FormIDs, `.assets` directories, and `.openf4.esm` overlays are unchanged.
The old executable/target names and C++ namespace are not aliases; update scripts to
`openf4.exe` and `openf4-editor.exe`. Old binaries left in an existing build directory
are stale artifacts, not compatibility launchers.
