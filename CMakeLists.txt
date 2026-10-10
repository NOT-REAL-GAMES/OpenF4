cmake_minimum_required(VERSION 3.25)

project(OpenF4 VERSION 0.1.0 LANGUAGES CXX)

if(NOT DEFINED OPENF4_BUILD_TESTS AND DEFINED EMPIRE_BUILD_TESTS)
    message(DEPRECATION "EMPIRE_BUILD_TESTS is deprecated; use OPENF4_BUILD_TESTS")
    set(OPENF4_BUILD_TESTS "${EMPIRE_BUILD_TESTS}" CACHE BOOL "Build focused unit tests")
endif()
option(OPENF4_BUILD_TESTS "Build focused unit tests" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

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

add_subdirectory(third_party/papyrus-vm)
add_library(openf4-script src/script/papyrus_vm.cpp)
target_include_directories(openf4-script PUBLIC src)
target_link_libraries(openf4-script PRIVATE openf4-papyrus-vm)

include(FetchContent)
FetchContent_Declare(nifly
    GIT_REPOSITORY https://github.com/ousnius/nifly.git
    GIT_TAG 134124c2b6e6e369897abb3dc62e16e74629014e
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(nifly)
set(CMAKE_CXX_STANDARD 20)

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
target_link_libraries(openf4-assets PUBLIC glm::glm lz4::lz4 ZLIB::ZLIB
    PRIVATE nifly ${GEOTIFF_LIBRARIES} PROJ::proj assimp::assimp Microsoft::DirectXTex
        meshoptimizer::meshoptimizer unofficial::sqlite3::sqlite3 openf4-script)
if(WIN32)
    target_link_libraries(openf4-assets PRIVATE bcrypt)
endif()

add_library(openf4-physics src/physics/jolt_collision.cpp src/physics/jolt_collision_authoring.cpp)
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
if(MSVC)
    target_compile_options(openf4-world PRIVATE /W4 /permissive- /FS)
else()
    target_compile_options(openf4-world PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(openf4 src/main.cpp)
target_link_libraries(openf4 PRIVATE openf4-world)

add_library(openf4-editor-document src/editor/editor_document.cpp src/editor/editor_manipulation.cpp src/editor/asset_document.cpp)
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
endif()
