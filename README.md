# EMPIRE

EMPIRE is a clean SDL3 and Vulkan rewrite of the Fallout: Empire renderer. It does not link
VulkanSceneGraph or inherit the previous engine's frame graph and synchronization behavior.

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
- EMPIRE-owned 3x3 ESM landscape creation, sculpting, undo/redo, and atomic save

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

The vcpkg manifest installs SDL3 with Vulkan support, shaderc, VMA, glm,
LZ4, and zlib. Nifly is pinned to a specific revision through CMake FetchContent.

## Run

```powershell
build/bin/RelWithDebInfo/empire.exe
build/bin/RelWithDebInfo/empire.exe --benchmark 300
build/bin/RelWithDebInfo/empire.exe --cubes --benchmark 300
build/bin/RelWithDebInfo/empire.exe --cubes --resize-test --benchmark 60
build/bin/RelWithDebInfo/empire.exe --model "meshes\architecture\example.nif"
build/bin/RelWithDebInfo/empire.exe --cell 1 1 --radius 0
build/bin/RelWithDebInfo/empire.exe --cell 1 1 --max-objects 100 --benchmark 130
build/bin/RelWithDebInfo/empire.exe --land-only
build/bin/RelWithDebInfo/empire.exe --editor "mods\EmpireWorld.esm"
build/bin/RelWithDebInfo/empire.exe "C:\Games\Fallout 4\Data\Fallout4.esm"
build/bin/RelWithDebInfo/empire.exe --import-dem "mods\NYC.esm" `
  "USGS_1_n41w075_20221115.tif" "USGS_1_n41w074_20260820.tif" `
  "USGS_1_n42w074_20241010.tif" "USGS_1_n42w075_20240925.tif"
```

Passing exactly one `.esm` path selects that master in read-only world mode. EMPIRE DEM
overlays stream nearby LAND and their baked level-32 BTR tiles; use `--editor` explicitly
to sculpt. Other masters stream the first exterior worldspace contained in that file.
Master dependency merging is not performed yet.

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
persists it automatically. Editor ESM files are intentionally EMPIRE-only for now.
The DEM importer accepts the four contiguous USGS one-arc-second NAD83 tiles in any order,
uses true Fallout scale, and writes spatial terrain masters capped at 1,048,576 cells each.

## Synchronization Policy

There is no `vkQueueWaitIdle`. `vkDeviceWaitIdle` is limited to swapchain recreation and
shutdown. Waiting elsewhere is limited to the one startup upload fence and the normal
per-frame/resource fence before reusing or publishing frame-owned resources. Set
`EMPIRE_DISABLE_SPARSE=1` only when comparing sparse and conventional image allocation.
