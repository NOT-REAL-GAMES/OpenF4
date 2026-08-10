cmake_minimum_required(VERSION 3.25)

project(EMPIRE VERSION 0.1.0 LANGUAGES CXX)

option(EMPIRE_BUILD_TESTS "Build focused unit tests" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

find_package(Vulkan REQUIRED)
find_package(SDL3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(VulkanMemoryAllocator CONFIG REQUIRED)
find_package(unofficial-shaderc CONFIG REQUIRED)
find_package(ZLIB REQUIRED)

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

add_library(empire-assets
    src/assets/ba2_archive.cpp
    src/assets/dds_texture.cpp
    src/assets/dx10_archive.cpp
    src/assets/esm_world.cpp
    src/assets/fo4_install.cpp
    src/assets/material_file.cpp
    src/assets/nif_model.cpp
)
target_include_directories(empire-assets PUBLIC src)
target_link_libraries(empire-assets PUBLIC glm::glm lz4::lz4 ZLIB::ZLIB PRIVATE nifly)

add_library(empire-renderer
    src/render/mesh_builder.cpp
    src/render/vulkan_renderer.cpp
)
target_include_directories(empire-renderer PUBLIC src)
target_link_libraries(empire-renderer
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
    target_compile_options(empire-assets PRIVATE /W4 /permissive- /FS)
    target_compile_options(empire-renderer PRIVATE /W4 /permissive- /FS)
else()
    target_compile_options(empire-assets PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(empire-renderer PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_library(empire-world src/world/world_scene.cpp)
target_include_directories(empire-world PUBLIC src)
target_link_libraries(empire-world PUBLIC empire-assets empire-renderer)
if(MSVC)
    target_compile_options(empire-world PRIVATE /W4 /permissive- /FS)
else()
    target_compile_options(empire-world PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(empire src/main.cpp)
target_link_libraries(empire PRIVATE empire-world)

if(EMPIRE_BUILD_TESTS)
    enable_testing()
    add_executable(empire-ba2-tests tests/ba2_archive_tests.cpp)
    target_link_libraries(empire-ba2-tests PRIVATE empire-assets)
    add_test(NAME ba2_archive COMMAND empire-ba2-tests)
    add_executable(empire-asset-format-tests tests/asset_format_tests.cpp)
    target_link_libraries(empire-asset-format-tests PRIVATE empire-assets)
    add_test(NAME asset_formats COMMAND empire-asset-format-tests)
    add_executable(empire-mesh-builder-tests tests/mesh_builder_tests.cpp)
    target_link_libraries(empire-mesh-builder-tests PRIVATE empire-world)
    add_test(NAME mesh_builder COMMAND empire-mesh-builder-tests)
endif()
