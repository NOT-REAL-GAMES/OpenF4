cmake_minimum_required(VERSION 3.25)

# MinGW vcpkg builds need a patched oneTBB port (see cmake/vcpkg-overlay-ports/tbb).
# Must be set before project() so the vcpkg toolchain picks it up; MSVC is unaffected.
if(VCPKG_TARGET_TRIPLET MATCHES "mingw")
    list(APPEND VCPKG_OVERLAY_PORTS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/vcpkg-overlay-ports")
    # Pins the UCRT64 compiler for vcpkg-built deps (stock triplet can pick up mingw64/MSVCRT).
    list(APPEND VCPKG_OVERLAY_TRIPLETS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/vcpkg-overlay-triplets")
endif()
project(OpenF4 VERSION 0.1.0 LANGUAGES CXX)

if(NOT DEFINED OPENF4_BUILD_TESTS AND DEFINED EMPIRE_BUILD_TESTS)
    message(DEPRECATION "EMPIRE_BUILD_TESTS is deprecated; use OPENF4_BUILD_TESTS")
    set(OPENF4_BUILD_TESTS "${EMPIRE_BUILD_TESTS}" CACHE BOOL "Build focused unit tests")
endif()
option(OPENF4_BUILD_TESTS "Build focused unit tests" ON)

# OpenF4 builds as C++26. CMake 4.3 knows -std=c++26 for Clang/GCC but has no
# C++26 mapping for MSVC yet; VS 2026 (MSVC 19.5x) exposes the C++26 working
# draft through /std:c++latest, so teach CMake that mapping when it is missing.
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" AND NOT DEFINED CMAKE_CXX26_STANDARD_COMPILE_OPTION)
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 19.50)
        message(FATAL_ERROR "OpenF4 requires C++26: use Visual Studio 2026 (MSVC 19.50+) or newer; found ${CMAKE_CXX_COMPILER_VERSION}")
    endif()
    set(CMAKE_CXX26_STANDARD_COMPILE_OPTION "-std:c++latest")
    set(CMAKE_CXX26_EXTENSION_COMPILE_OPTION "-std:c++latest")
    list(APPEND CMAKE_CXX_COMPILE_FEATURES cxx_std_26)
    set(CMAKE_CXX26_COMPILE_FEATURES cxx_std_26)
endif()
set(OPENF4_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD ${OPENF4_CXX_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
# MSYS2 / MinGW-w64 GCC build (e.g. UCRT64 GCC 16). Used for C++26 features MSVC
# does not ship yet (static reflection, std::inplace_vector). See README "MSYS2".
if(MINGW)
    # world_scene/vulkan_renderer exceed the default COFF section count.
    add_compile_options(-Wa,-mbig-obj)
    # Aggregate init with omitted trailing members is idiomatic here (Vulkan structs).
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wno-missing-field-initializers>)
    # Avoid shipping libgcc/libstdc++/winpthread DLLs next to every exe.
    add_link_options(-static-libgcc -static-libstdc++)
    # vcpkg's DLLs still link libstdc++-6/libgcc_s/libwinpthread dynamically. Ship the
    # copies from this compiler's bin dir next to the exes so a mingw64 (MSVCRT) runtime
    # earlier on PATH is never picked up by a UCRT build.
    get_filename_component(_openf4_mingw_bin "${CMAKE_CXX_COMPILER}" DIRECTORY)
    foreach(_rt libstdc++-6.dll libgcc_s_seh-1.dll libwinpthread-1.dll libgomp-1.dll)
        if(EXISTS "${_openf4_mingw_bin}/${_rt}")
            file(COPY "${_openf4_mingw_bin}/${_rt}" DESTINATION "${CMAKE_BINARY_DIR}/bin")
        endif()
    endforeach()
endif()
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 16)
    option(OPENF4_ENABLE_REFLECTION "Enable C++26 static reflection (-freflection) on GCC 16+" ON)
    if(OPENF4_ENABLE_REFLECTION)
        # Only for targets actually built as C++26 (nifly stays C++17).
        add_compile_options("$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<VERSION_GREATER_EQUAL:$<TARGET_PROPERTY:CXX_STANDARD>,26>>:-freflection>")
    endif()
endif()

find_package(Vulkan REQUIRED)
find_package(SDL3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(Jolt CONFIG REQUIRED)
find_package(GeoTIFF CONFIG REQUIRED)
find_package(PROJ CONFIG REQUIRED)
find_package(VulkanMemoryAllocator CONFIG REQUIRED)
find_package(unofficial-shaderc CONFIG REQUIRED)
find_package(ZLIB REQUIRED)
find_package(assimp CONFIG REQUIRED)
find_package(directxtex CONFIG REQUIRED)
find_package(meshoptimizer CONFIG REQUIRED)
find_package(unofficial-sqlite3 CONFIG REQUIRED)
find_package(Stb REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(imguizmo CONFIG REQUIRED)

# MinGW/GCC: libstdc++'s parallel algorithms (std::execution::par) only run in parallel
# when oneTBB is visible (_GLIBCXX_USE_TBB_PAR_BACKEND); without it they fall back to
# serial. stdexec supplies P2300 senders (std::execution) which libstdc++ 16 lacks.
# Both are mingw-only vcpkg deps; MSVC uses its own STL parallel backend.
set(OPENF4_HAS_STDEXEC OFF)
if(MINGW)
    find_package(TBB CONFIG REQUIRED)
    find_package(stdexec CONFIG QUIET)
    if(TARGET STDEXEC::stdexec)
        set(OPENF4_HAS_STDEXEC ON)
    endif()
    message(STATUS "OpenF4 MinGW: TBB ${TBB_VERSION} (par backend), stdexec=${OPENF4_HAS_STDEXEC}")
endif()

# openf4-parallel: one place for the std::execution backend. Every first-party library
# links it PUBLIC, so any TU that includes <execution> (and therefore <tbb/tbb.h> under
# libstdc++) gets the TBB headers, and every executable/test links libtbb12 no matter
# which subset of libraries it pulls in. Empty on MSVC.
add_library(openf4-parallel INTERFACE)
if(MINGW)
    target_link_libraries(openf4-parallel INTERFACE TBB::tbb)
    if(OPENF4_HAS_STDEXEC)
        target_link_libraries(openf4-parallel INTERFACE STDEXEC::stdexec)
        target_compile_definitions(openf4-parallel INTERFACE OPENF4_HAS_STDEXEC=1)
    endif()
endif()

add_subdirectory(third_party/papyrus-vm)
add_library(openf4-script src/script/papyrus_vm.cpp)
target_include_directories(openf4-script PUBLIC src)
target_link_libraries(openf4-script PUBLIC openf4-parallel PRIVATE openf4-papyrus-vm)

include(FetchContent)
FetchContent_Declare(nifly
    GIT_REPOSITORY https://github.com/ousnius/nifly.git
    GIT_TAG 134124c2b6e6e369897abb3dc62e16e74629014e
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(nifly)
set(CMAKE_CXX_STANDARD ${OPENF4_CXX_STANDARD})

find_package(lz4 CONFIG QUIET)
if(NOT TARGET lz4::lz4)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(LZ4 REQUIRED IMPORTED_TARGET liblz4)
    add_library(lz4::lz4 ALIAS PkgConfig::LZ4)
endif()

add_library(openf4-assets
    src/assets/asset_resolver.cpp
    src/assets/ba2_archive.cpp
    src/assets/dds_texture.cpp
    src/assets/dem_import.cpp
    src/assets/dx10_archive.cpp
    src/assets/esm_editor.cpp
    src/assets/esm_document.cpp
    src/assets/load_order.cpp
    src/assets/esm_world.cpp
    src/assets/fo4_install.cpp
    src/assets/fo4_export.cpp
    src/assets/fo4_asset_package.cpp
    src/assets/fo4_world_writer.cpp
    src/assets/havok_packfile.cpp
    src/assets/havok_write.cpp
    src/assets/lod_bake.cpp
    src/assets/lod_statics.cpp
    src/assets/lod_terrain.cpp
    src/assets/material_file.cpp
    src/assets/model_import.cpp
    src/assets/nif_model.cpp
    src/assets/native_model_export.cpp
    src/assets/papyrus_binding.cpp
)
target_include_directories(openf4-assets PUBLIC src PRIVATE ${GEOTIFF_INCLUDE_DIR} ${Stb_INCLUDE_DIR})
target_link_libraries(openf4-assets PUBLIC openf4-parallel glm::glm lz4::lz4 ZLIB::ZLIB
    PRIVATE nifly ${GEOTIFF_LIBRARIES} PROJ::proj assimp::assimp Microsoft::DirectXTex
        meshoptimizer::meshoptimizer unofficial::sqlite3::sqlite3 openf4-script)
if(WIN32)
    target_link_libraries(openf4-assets PRIVATE bcrypt)
endif()

add_library(openf4-physics src/physics/jolt_collision.cpp src/physics/jolt_collision_authoring.cpp src/physics/convex_decomposition.cpp)
target_include_directories(openf4-physics PUBLIC src)
target_link_libraries(openf4-physics PUBLIC openf4-assets Jolt::Jolt)

add_library(openf4-renderer
    src/render/mesh_builder.cpp
    src/render/vulkan_renderer.cpp
)
target_include_directories(openf4-renderer PUBLIC src)
target_link_libraries(openf4-renderer
    PUBLIC
        SDL3::SDL3
        Vulkan::Vulkan
        glm::glm
        GPUOpen::VulkanMemoryAllocator
        openf4-parallel
    PRIVATE
        unofficial::shaderc::shaderc
)

if(MSVC)
    target_compile_options(nifly PRIVATE /FS)
    target_compile_options(openf4-assets PRIVATE /W4 /permissive- /FS)
    target_compile_options(openf4-physics PRIVATE /W4 /permissive- /FS)
    target_compile_options(openf4-renderer PRIVATE /W4 /permissive- /FS)
else()
    target_compile_options(openf4-assets PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(openf4-physics PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(openf4-renderer PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_library(openf4-world src/world/world_scene.cpp src/world/picking_geometry.cpp src/script/object_scripts.cpp)
target_include_directories(openf4-world PUBLIC src)
target_link_libraries(openf4-world PUBLIC openf4-assets openf4-renderer openf4-physics openf4-script)
# TBB/stdexec (MinGW) arrive PUBLIC via openf4-parallel on every base library.
target_link_libraries(openf4-world PUBLIC openf4-parallel)
if(MSVC)
    target_compile_options(openf4-world PRIVATE /W4 /permissive- /FS)
else()
    target_compile_options(openf4-world PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(openf4 src/main.cpp)
target_link_libraries(openf4 PRIVATE openf4-world)

add_library(openf4-editor-document src/editor/editor_document.cpp src/editor/editor_manipulation.cpp src/editor/asset_document.cpp src/editor/mesh_edit.cpp)
target_include_directories(openf4-editor-document PUBLIC src)
target_link_libraries(openf4-editor-document PUBLIC openf4-world)
add_library(openf4-script-compiler src/script/papyrus_compiler.cpp)
target_include_directories(openf4-script-compiler PUBLIC src)
target_link_libraries(openf4-script-compiler PUBLIC openf4-assets openf4-script SDL3::SDL3)
add_executable(openf4-editor src/editor/main.cpp src/editor/editor_app.cpp src/editor/asset_workspace.cpp src/editor/scripts_panel.cpp)
target_link_libraries(openf4-editor PRIVATE openf4-editor-document openf4-script-compiler imgui::imgui imguizmo::imguizmo)
if(MSVC)
    target_compile_options(openf4-editor-document PRIVATE /W4 /permissive- /FS)
    target_compile_options(openf4-editor PRIVATE /W4 /permissive- /FS)
endif()

add_executable(openf4-physics-demo apps/physics_demo.cpp)
target_link_libraries(openf4-physics-demo PRIVATE openf4-physics openf4-renderer)

if(OPENF4_BUILD_TESTS)
    enable_testing()
    add_executable(openf4-papyrus-vm-tests tests/papyrus_vm_tests.cpp)
    target_link_libraries(openf4-papyrus-vm-tests PRIVATE openf4-script)
    add_test(NAME papyrus_vm COMMAND openf4-papyrus-vm-tests)
    add_executable(openf4-papyrus-binding-tests tests/papyrus_binding_tests.cpp)
    target_link_libraries(openf4-papyrus-binding-tests PRIVATE openf4-editor-document)
    add_test(NAME papyrus_binding COMMAND openf4-papyrus-binding-tests)
    add_executable(openf4-papyrus-integration-tests tests/papyrus_integration_tests.cpp)
    target_link_libraries(openf4-papyrus-integration-tests PRIVATE openf4-editor-document openf4-script-compiler)
    add_test(NAME papyrus_integration COMMAND openf4-papyrus-integration-tests)
    set_tests_properties(papyrus_integration PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 150)
    add_executable(openf4-fo4-world-writer-tests tests/fo4_world_writer_tests.cpp)
    target_link_libraries(openf4-fo4-world-writer-tests PRIVATE openf4-assets)
    add_test(NAME fo4_world_writer COMMAND openf4-fo4-world-writer-tests)
    add_executable(openf4-fo4-asset-package-tests tests/fo4_asset_package_tests.cpp)
    target_link_libraries(openf4-fo4-asset-package-tests PRIVATE openf4-assets nifly)
    add_test(NAME fo4_asset_package COMMAND openf4-fo4-asset-package-tests)
    add_executable(openf4-fo4-export-tests tests/fo4_export_tests.cpp)
    target_link_libraries(openf4-fo4-export-tests PRIVATE openf4-assets)
    add_test(NAME fo4_export COMMAND openf4-fo4-export-tests)
    set_tests_properties(fo4_export PROPERTIES SKIP_RETURN_CODE 77)
    add_executable(openf4-fo4-export-source-tests tests/fo4_export_source_tests.cpp)
    target_link_libraries(openf4-fo4-export-source-tests PRIVATE openf4-assets)
    add_test(NAME fo4_export_source COMMAND openf4-fo4-export-source-tests)
    set_tests_properties(fo4_export_source PROPERTIES SKIP_RETURN_CODE 77)
    add_executable(openf4-editor-document-tests tests/editor_document_tests.cpp)
    target_link_libraries(openf4-editor-document-tests PRIVATE openf4-editor-document)
    add_test(NAME editor_document COMMAND openf4-editor-document-tests)
    add_executable(openf4-asset-document-tests tests/asset_document_tests.cpp tests/compatibility_fixtures.cpp)
    target_link_libraries(openf4-asset-document-tests PRIVATE openf4-editor-document nifly)
    add_test(NAME asset_document COMMAND openf4-asset-document-tests)
    add_executable(openf4-mesh-edit-tests tests/mesh_edit_tests.cpp)
    target_link_libraries(openf4-mesh-edit-tests PRIVATE openf4-editor-document)
    add_test(NAME mesh_edit COMMAND openf4-mesh-edit-tests)
    add_executable(openf4-collision-edit-tests tests/collision_edit_tests.cpp)
    target_link_libraries(openf4-collision-edit-tests PRIVATE openf4-editor-document)
    add_test(NAME collision_edit COMMAND openf4-collision-edit-tests)
    add_executable(openf4-load-order-tests tests/load_order_tests.cpp)
    target_link_libraries(openf4-load-order-tests PRIVATE openf4-editor-document)
    add_test(NAME load_order COMMAND openf4-load-order-tests)
    add_executable(openf4-actor-tests tests/actor_tests.cpp)
    target_link_libraries(openf4-actor-tests PRIVATE openf4-editor-document)
    add_test(NAME actors COMMAND openf4-actor-tests)
    add_executable(openf4-asset-capture-tests tests/asset_capture_tests.cpp)
    target_link_libraries(openf4-asset-capture-tests PRIVATE openf4-assets)
    add_test(NAME asset_capture COMMAND openf4-asset-capture-tests)
    if(WIN32)
        find_program(OPENF4_POWERSHELL pwsh)
        if(OPENF4_POWERSHELL)
            add_test(NAME compatibility_evidence_driver COMMAND "${OPENF4_POWERSHELL}"
                -NoProfile -File "${CMAKE_CURRENT_SOURCE_DIR}/tests/compatibility_driver_tests.ps1")
            add_test(NAME compatibility_game_setup COMMAND "${OPENF4_POWERSHELL}"
                -NoProfile -File "${CMAKE_CURRENT_SOURCE_DIR}/tests/compatibility_game_tests.ps1")
        endif()
    endif()
    add_executable(openf4-object-physics-tests tests/object_physics_tests.cpp)
    target_link_libraries(openf4-object-physics-tests PRIVATE openf4-physics)
    add_test(NAME object_physics COMMAND openf4-object-physics-tests)
    add_executable(openf4-model-import-tests tests/model_import_tests.cpp)
    target_link_libraries(openf4-model-import-tests PRIVATE openf4-assets)
    add_test(NAME model_import COMMAND openf4-model-import-tests)
    add_executable(openf4-ba2-tests tests/ba2_archive_tests.cpp)
    target_link_libraries(openf4-ba2-tests PRIVATE openf4-assets)
    add_test(NAME ba2_archive COMMAND openf4-ba2-tests)
    add_executable(openf4-asset-format-tests tests/asset_format_tests.cpp)
    target_include_directories(openf4-asset-format-tests PRIVATE ${GEOTIFF_INCLUDE_DIR})
    target_link_libraries(openf4-asset-format-tests PRIVATE openf4-assets ${GEOTIFF_LIBRARIES})
    add_test(NAME asset_formats COMMAND openf4-asset-format-tests)
    add_executable(openf4-mesh-builder-tests tests/mesh_builder_tests.cpp)
    target_link_libraries(openf4-mesh-builder-tests PRIVATE openf4-world)
    add_test(NAME mesh_builder COMMAND openf4-mesh-builder-tests)
    add_executable(openf4-havok-tests tests/havok_packfile_tests.cpp)
    target_link_libraries(openf4-havok-tests PRIVATE openf4-physics)
    add_test(NAME havok_packfile COMMAND openf4-havok-tests)
    add_executable(openf4-collision-authoring-tests tests/collision_authoring_tests.cpp)
    target_link_libraries(openf4-collision-authoring-tests PRIVATE openf4-physics)
    add_test(NAME collision_authoring COMMAND openf4-collision-authoring-tests)
    add_executable(openf4-native-model-export-tests tests/native_model_export_tests.cpp)
    target_link_libraries(openf4-native-model-export-tests PRIVATE openf4-assets nifly)
    add_test(NAME native_model_export COMMAND openf4-native-model-export-tests)
    add_executable(openf4-havok-export-tests tests/havok_export_tests.cpp)
    target_link_libraries(openf4-havok-export-tests PRIVATE openf4-assets)
    add_test(NAME havok_export COMMAND openf4-havok-export-tests)
    add_test(NAME havok_fallout4_asset COMMAND openf4-havok-tests --fallout4)
    set_tests_properties(havok_fallout4_asset PROPERTIES SKIP_RETURN_CODE 77)
    add_executable(openf4-lod-terrain-tests tests/lod_terrain_tests.cpp)
    target_link_libraries(openf4-lod-terrain-tests PRIVATE openf4-assets)
    add_test(NAME lod_terrain COMMAND openf4-lod-terrain-tests)
    add_executable(openf4-lod-statics-tests tests/lod_statics_tests.cpp)
    target_link_libraries(openf4-lod-statics-tests PRIVATE openf4-assets nifly Microsoft::DirectXTex)
    add_test(NAME lod_statics COMMAND openf4-lod-statics-tests)
    add_executable(openf4-lod-bake-tests tests/lod_bake_tests.cpp)
    target_link_libraries(openf4-lod-bake-tests PRIVATE openf4-assets unofficial::sqlite3::sqlite3)
    add_test(NAME lod_bake COMMAND openf4-lod-bake-tests)

    if(MINGW)
        # std::execution smoke test: TBB-backed par algorithms + stdexec senders.
        add_executable(openf4-execution-tests tests/execution_tests.cpp)
        target_link_libraries(openf4-execution-tests PRIVATE openf4-parallel)
        add_test(NAME execution COMMAND openf4-execution-tests)
    endif()
endif()

# Editor hot-path timing on a real Fallout4.esm load (manual tool, not a ctest).
add_executable(openf4-editor-perf-bench tools/editor_perf_bench.cpp)
target_link_libraries(openf4-editor-perf-bench PRIVATE openf4-editor-document)

