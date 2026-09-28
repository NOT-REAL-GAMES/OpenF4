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
find_package(meshoptimizer CONFIG REQUIRED)
find_package(unofficial-sqlite3 CONFIG REQUIRED)
find_package(Stb REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(imguizmo CONFIG REQUIRED)

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
    src/assets/havok_packfile.cpp
    src/assets/lod_bake.cpp
    src/assets/lod_statics.cpp
    src/assets/lod_terrain.cpp
    src/assets/material_file.cpp
    src/assets/model_import.cpp
    src/assets/nif_model.cpp
)
target_include_directories(openf4-assets PUBLIC src PRIVATE ${GEOTIFF_INCLUDE_DIR} ${Stb_INCLUDE_DIR})
target_link_libraries(openf4-assets PUBLIC glm::glm lz4::lz4 ZLIB::ZLIB
    PRIVATE nifly ${GEOTIFF_LIBRARIES} PROJ::proj assimp::assimp
        meshoptimizer::meshoptimizer unofficial::sqlite3::sqlite3)

add_library(openf4-physics src/physics/jolt_collision.cpp)
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

add_library(openf4-world src/world/world_scene.cpp src/world/picking_geometry.cpp)
target_include_directories(openf4-world PUBLIC src)
target_link_libraries(openf4-world PUBLIC openf4-assets openf4-renderer openf4-physics)
if(MSVC)
    target_compile_options(openf4-world PRIVATE /W4 /permissive- /FS)
else()
    target_compile_options(openf4-world PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(openf4 src/main.cpp)
target_link_libraries(openf4 PRIVATE openf4-world)

add_library(openf4-editor-document src/editor/editor_document.cpp src/editor/editor_manipulation.cpp)
target_include_directories(openf4-editor-document PUBLIC src)
target_link_libraries(openf4-editor-document PUBLIC openf4-world)
add_executable(openf4-editor src/editor/main.cpp src/editor/editor_app.cpp)
target_link_libraries(openf4-editor PRIVATE openf4-editor-document imgui::imgui imguizmo::imguizmo)
if(MSVC)
    target_compile_options(openf4-editor-document PRIVATE /W4 /permissive- /FS)
    target_compile_options(openf4-editor PRIVATE /W4 /permissive- /FS)
endif()

add_executable(openf4-physics-demo apps/physics_demo.cpp)
target_link_libraries(openf4-physics-demo PRIVATE openf4-physics openf4-renderer)

if(OPENF4_BUILD_TESTS)
    enable_testing()
    add_executable(openf4-editor-document-tests tests/editor_document_tests.cpp)
    target_link_libraries(openf4-editor-document-tests PRIVATE openf4-editor-document)
    add_test(NAME editor_document COMMAND openf4-editor-document-tests)
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
    add_test(NAME havok_fallout4_asset COMMAND openf4-havok-tests --fallout4)
    set_tests_properties(havok_fallout4_asset PROPERTIES SKIP_RETURN_CODE 77)
    add_executable(openf4-lod-terrain-tests tests/lod_terrain_tests.cpp)
    target_link_libraries(openf4-lod-terrain-tests PRIVATE openf4-assets)
    add_test(NAME lod_terrain COMMAND openf4-lod-terrain-tests)
    add_executable(openf4-lod-statics-tests tests/lod_statics_tests.cpp)
    target_link_libraries(openf4-lod-statics-tests PRIVATE openf4-assets)
    add_test(NAME lod_statics COMMAND openf4-lod-statics-tests)
    add_executable(openf4-lod-bake-tests tests/lod_bake_tests.cpp)
    target_link_libraries(openf4-lod-bake-tests PRIVATE openf4-assets)
    add_test(NAME lod_bake COMMAND openf4-lod-bake-tests)
endif()
