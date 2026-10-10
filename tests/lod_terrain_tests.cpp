#include "assets/esm_editor.hpp"
#include "assets/lod_terrain.hpp"
#include "assets/nif_model.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <ranges>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace
{
    using namespace openf4::assets;

    void require(bool condition, const char* message)
    {
        if (!condition)
            throw std::runtime_error(message);
    }

    [[nodiscard]] ModelSurface readSurface(const std::filesystem::path& path)
    {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        require(static_cast<bool>(file), "BTR output was not written");
        std::vector<std::byte> bytes(static_cast<std::size_t>(file.tellg()));
        file.seekg(0);
        file.read(reinterpret_cast<char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
        require(static_cast<bool>(file), "Could not read BTR output");
        auto model = loadNifModel(bytes);
        require(model.surfaces.size() == 1, "BTR has no terrain surface");
        return std::move(model.surfaces.front());
    }

    [[nodiscard]] bool close(float a, float b)
    {
        return std::abs(a - b) < 0.1f;
    }
}

int main(int argc, char** argv)
{
    if (argc == 4 && std::string_view(argv[1]) == "--dem")
    {
        try
        {
            const auto output = std::filesystem::absolute(argv[3]);
            const auto world = std::filesystem::absolute(argv[2]);
            const auto sidecar = world.parent_path() / (world.stem().string() + ".btr");
            require(!std::filesystem::exists(output), "DEM test output must be a new directory");
            require(output.lexically_normal() != sidecar.lexically_normal(),
                "DEM test may not overwrite world products");
            TerrainLodSource source(world);
            const auto tiles = source.tiles();
            std::cout << "DEM terrain preflight: " << tiles.size() << " tiles\n";
            for (const int level : {4, 8, 16, 32})
            {
                const LodTerrainTile key{level, {0, 0}};
                require(std::ranges::find(tiles, key) != tiles.end(),
                    "DEM fixture lacks origin terrain");
                const auto tile = output / ("openf4world." + std::to_string(level) + ".0.0.btr");
                source.bake(key, tile);
                require(!readSurface(tile).indices.empty(), "DEM BTR could not be reimported");
                std::cout << "Baked level " << level << " origin tile\n";
            }
            return 0;
        }
        catch (const std::exception& error)
        {
            std::cerr << error.what() << '\n';
            return 1;
        }
    }
    const auto root = std::filesystem::temp_directory_path()
        / ("openf4-lod-terrain-" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(root);
    try
    {
        const auto world = root / "small.esm";
        EditableLandscape landscape = createEditableLandscape();
        for (int x = 2; x <= 4; ++x)
            require(landscape.addCell({x, 0}), "Could not extend test LAND");
        for (const auto& [coord, form] : landscape.cellForms)
        {
            (void)form;
            for (int y = 0; y <= 32; ++y)
                for (int x = 0; x <= 32; ++x)
                {
                    const int sampleX = coord.x * 32 + x;
                    const int sampleY = coord.y * 32 + y;
                    landscape.setSample({sampleX, sampleY},
                        static_cast<float>((sampleX + sampleY) * 8));
                }
        }
        saveEditableLandscape(world, landscape);

        TerrainLodSource source(world);
        const auto tiles = source.tiles();
        for (const int level : {4, 8, 16, 32})
        {
            require(std::ranges::find(tiles, LodTerrainTile{level, {0, 0}})
                != tiles.end(), "Missing level from small-world tile enumeration");
            require(std::ranges::find(tiles, LodTerrainTile{level, {-level, -level}})
                != tiles.end(), "Negative CELL mapped to wrong tile origin");
        }
        require(source.sourceFiles() == std::vector<std::filesystem::path>{world},
            "Small-world source fingerprint list is incorrect");

        const auto leftPath = root / "left.btr";
        const auto rightPath = root / "right.btr";
        source.bake({4, {0, 0}}, leftPath);
        source.bake({4, {4, 0}}, rightPath);
        const auto left = readSurface(leftPath);
        const auto right = readSurface(rightPath);
        require(left.indices.size() == 6u * 8u * 8u * 6u,
            "Level-4 BTR did not clip absent LAND cells");
        require(right.indices.size() == 8u * 8u * 6u,
            "Sparse level-4 BTR contains missing cells");
        for (int y = 0; y <= 8; ++y)
        {
            const auto& a = left.vertices[static_cast<std::size_t>(y) * 33 + 32];
            const auto& b = right.vertices[static_cast<std::size_t>(y) * 33];
            require(close(a.position.y, b.position.y),
                "Adjacent BTR tiles disagree at a shared LAND edge");
            require(glm::length(a.normal - b.normal) < 0.01f,
                "Adjacent BTR tiles disagree on shared-edge normals");
        }
        for (const int level : {8, 16, 32})
        {
            const auto tile = root / (std::to_string(level) + ".btr");
            source.bake({level, {0, 0}}, tile);
            const auto mesh = readSurface(tile);
            require(mesh.indices.size() == 7u * (32u / level) * (32u / level) * 6u,
                "Coarser BTR did not clip missing cells");
            for (std::size_t index = 0; index < mesh.indices.size(); index += 3)
            {
                const auto center = (mesh.vertices[mesh.indices[index]].position
                    + mesh.vertices[mesh.indices[index + 1]].position
                    + mesh.vertices[mesh.indices[index + 2]].position) / 3.0f;
                require(landscape.containsCell({static_cast<int>(std::floor(center.x / 4096.0f)),
                    static_cast<int>(std::floor(center.z / 4096.0f))}),
                    "BTR triangle covers an absent LAND cell");
            }
        }

        const auto seam = EditableLandscape::SampleCoord{4 * 32, 0};
        landscape.setSample(seam, landscape.sample(seam) + 80.0f);
        saveEditableLandscape(world, landscape);
        TerrainLodSource refreshed(world);
        refreshed.bake({4, {0, 0}}, leftPath);
        refreshed.bake({4, {4, 0}}, rightPath);
        const auto leftAfter = readSurface(leftPath);
        const auto rightAfter = readSurface(rightPath);
        require(close(leftAfter.vertices[32].position.y,
                rightAfter.vertices[0].position.y),
            "Sculpted seam diverged after atomic rebake");
        require(close(leftAfter.vertices[32].position.y,
                left.vertices[32].position.y + 80.0f),
            "Rebake did not read the saved LAND edit");

        std::filesystem::remove_all(root);
        return 0;
    }
    catch (const std::exception& error)
    {
        std::filesystem::remove_all(root);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
