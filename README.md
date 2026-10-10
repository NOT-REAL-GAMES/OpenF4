# OpenF4

OpenF4 is a clean-room reimplementation of Fallout 4,
meant to expand its capabilities beyond some of its more
dated design decisions from 2015.

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
overlays stream nearby LAND and their baked BTR hierarchy (including older level-32-only
products); use `--editor` explicitly
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
Static tiles target 25% of source triangles within a 1% bounds-relative error limit;
models that cannot reach that target retain more triangles. Animated models bake
their imported static pose. Each distant reference disappears when its full model
is installed, so loaded doorways do not retain coarse facades.

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

## Native asset workspace

Select a placed object and choose **Edit Asset** in the Inspector, or right-click an
Asset Browser entry and choose **Edit Asset**. Double-clicking a browser entry still
places it in the world. The **Asset** workspace button also opens the selected
object when no asset is open yet; its tooltip explains the selection requirement.
The **World** and **Asset** workspace buttons keep independent
cameras and selection; unsaved world edits survive switching, and world streaming
pauses while the asset workspace is active.

In Asset mode, RMB orbits, MMB pans, the wheel zooms, and F frames the model. Select
surfaces in the viewport or Outliner (Ctrl adds surfaces). Edit tint, alpha cutoff,
double-sided rendering, and diffuse/normal PNG, JPEG, or DDS textures in the Inspector;
an empty texture path clears that texture. Asset Ctrl+Z/Y uses its own 64-entry history,
separate from world commands. Escape cancels an active edit gesture.

Generate a convex hull or static triangle collision from selected surfaces, or add
boxes and capsules. Select collision parts to move/rotate/resize them with W/E/R and
the Inspector. Body controls expose mass, friction, restitution, damping, and gravity.
**Play / Pause / Reset** tests a single rigid body against an isolated Jolt ground
plane. Editing resets the simulation; simulated poses are never authored or saved.
Existing multi-body native assets can be inspected and, when supported, edited, but
the isolated physics preview currently requires one body.

**Save Native NIF** (Ctrl+S in Asset mode) publishes a new immutable NIF/BGSM/DDS
package under `<world>.assets/meshes|materials|textures/openf4/<form-id>/<version>/`
and changes the shared model assignment in one world undo step. All its placements
use the new asset after returning to World mode. The imported source and previous
packages remain intact; undo never deletes published files. The ESM stays dirty until
**Save World** (Ctrl+S in World mode); normal shutdown also saves dirty world edits.
Opening another asset/world or quitting prompts **Save / Discard / Cancel** when the
asset has unsaved edits. Asset save errors leave those edits available to retry.

Use **Make Unique** on one placed object's Inspector to publish an independent base
model and retarget only that placement. **Movable prop (MSTT)** selects an authored
dynamic body; ordinary scenery remains STAT. Changing between those record types
allocates a new FormID instead of changing an existing record's type. Project-world
runtime loads native collision without the physics-demo option, while world editing
itself stays non-simulating. Movable props are excluded from static LOD baking.

Native export supports convex hulls, capsules, compounds, and static triangle meshes
with uniform collision materials. Dynamic bodies retain mass, principal inertia,
center of mass, damping, and contact settings. Existing third-party NIFs use a guarded
collision-only path that preserves other blocks byte-for-byte and keeps attachment
identities and motion flags. Missing project dependencies are copied; conflicting
files are rejected, never overwritten. Unsupported files remain inspectable with a
capability reason instead of being flattened into a lossy export.

The asset preview is not a full Fallout material renderer: alpha blending and advanced
PBR/palette appearance still have limitations. Animated/skinned imports, render-mesh
topology editing, ragdolls, and constraints are not native authoring targets yet.
Rounded Jolt convex-hull capture is rejected until its core/radius conversion can
preserve size. Native collision rewriting rejects unrepresented source changes; only
bounded roundoff in represented mass-frame floats is accepted. Inherited ESM model
overrides also reject localized display-name records and reordered master slots when
their opaque data cannot be preserved safely.

The cube, doorway, and compound export fixtures have been exercised in Fallout 4;
this is not a general Fallout 4 or Creation Kit compatibility guarantee for authored
worlds. Generate disposable native-game fixtures into a **new** directory with:

```powershell
build/bin/Release/openf4-native-model-export-tests.exe --export-fixtures build/native-proof
```

The command writes three models, their dependencies, and a test-only
`OpenF4NativeCollisionProof.esp`; it does not install or activate anything. Use an
isolated game session/save directory for validation, never an existing playthrough.
The normal OpenF4 ESM authoring format remains separate from this proof plugin.

### Native compatibility evidence

Compatibility is reported per capability and build, not as general Havok/Jolt or
Fallout 4 parity. Generated native assets, guarded collision-only preservation,
and unsupported-file rejection have separate results. The historical doorway
proof uses static triangle collision; it is not evidence for convex compounds.

The asset-document tests include independent asymmetric geometry, doorway probes,
mass/inertia oracles, and fresh-document native save/reload generations E0 through
E10. Preserve the fixture packages and reports with:

```powershell
build/bin/Release/openf4-asset-document-tests.exe --compatibility-fixtures build/compatibility-fixtures
```

For source/build provenance, SHA-256 hashes, every CTest target, and fresh E0/E10
runtime checks, use PowerShell 7.4 or newer with a **new** output directory:

```powershell
./tools/compatibility.ps1 -OutputDirectory build/compatibility/run-01
```

`-SkipRuntime` leaves the runtime gate explicitly untested. `-Mode baseline`
archives the known local proof packages and logs without modifying them; later
source snapshots are not represented as historical build provenance. Failed runs
remain available for diagnosis. Neither mode installs game files, changes game
settings, nor touches saves.

The opt-in `OPENF4_COMPATIBILITY_CAPTURE_DIR` environment variable makes the normal
runtime capture fixture-sidecar reads and installed render/physics state in a new
directory. The driver compares the consumed NIF/BGSM/DDS bytes with the published
packages. Capture is disabled for normal runs and never copies game archive data.

Reports distinguish **Not run**, **Reported pass**, **Pass**, **Fail**, **Blocked**,
and **N/A with reason**. Automated success alone does not mark a profile Proven:
the applicable Fallout 4 checks, evidence, and reviewer approval are still required.
Native packages for those checks are generated but must be staged separately for
an isolated game session, with settings restored afterward. Do not load malformed
negative-control fixtures into Fallout 4.

After explicitly choosing to run the isolated game check, with Fallout 4 closed:

```powershell
./tools/compatibility-game.ps1 -RunDirectory build/compatibility/run-01
# Quit the game without saving before restoring:
./tools/compatibility-game.ps1 -Action restore -RunDirectory build/compatibility/run-01
```

This stages only runtime-verified native dependencies and two disposable ESPs,
backs up the settings, disables autosaves, and selects a separate save directory.
Restoration verifies hashes and recovers installed test files rather than deleting
them. Interrupted copies and settings writes are recoverable from the journal.
The E0/E10 ESMs are OpenF4 test worlds, **not** Fallout 4-compatible worlds. The
ESP fixtures contain base assets only: use `coc qasmoke`, `help OF4CompatE0 4`,
`help OF4CompatE10 4`, and `player.placeatme` with the resolved FormIDs. The four
numbered assets are an asymmetric movable box, a static doorway, an unequal
compound, and a compound with deliberately nonuniform mass distribution. Bright
test textures are intentional orientation markers.

Reopening, saving, and editing contact/damping/gravity settings preserve authored
center of mass and inertia. Changing mass or collision geometry intentionally
recalculates a uniform-density distribution. Undo and redo restore the original
authored values, including custom distributions.

### Native world export: first acceptance gate

The new native world writer is an experimental **geometry proof**, not the completed
playable-world exporter. It writes a separate ESP with native exterior groups, LAND
heights/normals/default texture, authored water flags, native STAT/MSTT placements,
and a persistent arrival marker. The asset packager checks collision and copies
project NIF/BGSM/DDS dependencies without changing their bytes. Base-game dependencies
remain references to `Fallout4.esm` and its archives.
Animated, skinned, behavior-graph, and connection-point assets are rejected by this
proof profile until their external dependency closure is supported; native asset
inspection and preservation elsewhere are unchanged.

For a small, new OpenF4 world whose placed assets have already been saved as native
NIF packages, choose **File → Export to Fallout 4 (.esp)…**. Choose a **new package
folder** (the folder picker selects its parent), then **Save World and Check Export**.
The compatibility summary shows blockers and limitations before **Export package**
is enabled. Export runs in the background with cancellation; conflicting editing,
imports, saves, and compilation are blocked while the dialog is open. Canceling
does not undo the explicit Save World step. The finished dialog displays the actual
`Data/<world>.esp` path and arrival command without changing the open project.
**Save As** remains an OpenF4 `.esm` authoring save, not a Fallout conversion.

The equivalent test-tool entry point exports the already-saved source:

```powershell
build/bin/Release/openf4-fo4-export-tests.exe --export-proof mods/Proof.esm build/world-proof-01
```

`export-report.txt` contains capabilities, source/output SHA-256 hashes, the world
editor ID, and test arrival instructions. This command neither saves the source nor
installs files. It refuses existing output directories, authored sidecars, Fallout
Data destinations, unsupported records, dependency conflicts, and changed inputs.
Failed or cancelled publication retains an explicitly incomplete staging directory.

Acceptance still requires entering the new world in Fallout 4 and checking terrain
and object collision. **Navigation, native distant LOD, existing-world overrides,
and NYC-scale partitioning are not implemented by this proof.** The editor command
exposes this experimental proof exporter, not a finished playable-world export.
The E0/E10 compatibility ESMs can be proof inputs; the source ESMs themselves remain
OpenF4 authoring files, not drop-in Fallout worlds. Do not use a main save for testing.

After explicitly choosing an isolated game check, with Fallout 4 closed:

```powershell
./tools/compatibility-game.ps1 -RunDirectory build/world-check-01 -WorldPackage build/world-proof-01
# Quit without saving, then restore all original settings:
./tools/compatibility-game.ps1 -Action restore -RunDirectory build/world-check-01
```

World-proof staging verifies the report's complete Data-file hashes and refuses
conflicts before changing settings. It is a manual acceptance test, not an automated
compatibility pass. The same journal-backed restore path is used as for asset proofs.

## Project Papyrus scripts

The editor's reference Inspector has a **Papyrus scripts** section. Attach to the
selected reference, or enable **Shared base** to affect every placement of that
STAT/MSTT. Attachments and scalar/object property edits participate in world
undo/redo and **Save World**. Unknown VMAD data remains preserved and read-only.

For a first test, select a movable prop, use **Create wake source**, **Compile**,
**Attach / enable**, then **Save World**. The generated `OnLoad` script requests
dynamic motion and a zero-magnitude wake impulse. This is opt-in, not an automatic
script attached to every exported object. **Open source externally** uses your
file association; no separate editor or compiler dependency is bundled.

Compilation uses the locally installed Fallout 4 Creation Kit compiler and base
sources. Project source lives in `<world>.assets/scripts/source/`, and compiled
PEX in `<world>.assets/scripts/`; namespaces become subdirectories. Project/master
sidecar sources precede installed source paths. Compilation runs in the background
with a 60-second timeout and publishes a checked PEX atomically. Errors retain the
previous compiled file. Source/PEX saves are separate from world undo history.

Normal project-world runtime loading executes project-attached scripts through a
standalone, MIT-licensed derivative of SkyMP's Papyrus VM (see
`third_party/papyrus-vm/UPSTREAM.md`). Native parent metadata is supplied by OpenF4;
it does not start the vanilla game's script population. The initial runtime supports
scalar/reference values, automatic property overrides, functions, inheritance,
states, control flow, `OnInit`, `OnLoad`, `OnUnload`, and `Utility.Wait`.

Implemented native calls are `Form.GetFormID`, `ObjectReference.Is3DLoaded`,
`GetPositionX/Y/Z`, `SetMotionType`, `ApplyHavokImpulse`, and `Debug.Trace`.
Physics mutation currently requires a resident single-body MSTT. Papyrus coordinates
use Bethesda axes; the bridge converts to OpenF4/Jolt internally. `OnLoad` waits
for installed collision and, for movable props, surrounding terrain/static collision.
Variables, latent waits, and motion state survive cell residency changes in the
current session. Script execution never dirties the authored world; the editor does
not simulate scripts while authoring.

The ESP exporter writes supported VMAD and copies the compiled project dependency
closure unchanged into `Data/scripts`. Installed Bethesda scripts remain external
dependencies, never redistributed. Compilation/export and OpenF4 execution have
different compatibility gates: arrays, structs, Var values, unknown natives, quests,
actors, and Fallout savegame VM persistence are not supported by this first runtime.
Unsupported execution produces diagnostics. Execution, recursion, event, and wait
limits prevent runaway scripts; a stack exceeding its instruction allowance is
stopped rather than replayed. This is not full Fallout 4 Papyrus compatibility.

CTest includes checked PEX/VMAD fixtures, document/export preservation, and Jolt
motion tests. `papyrus_integration` additionally compiles a fresh script with the
installed CK and tests save/reload, load/unload, latent calls, physics wake/collision,
and failed-compile rollback (skipped when CK is unavailable).

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
