#include "assets/asset_resolver.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <vector>

namespace
{
    void require(bool condition, const char* message)
    {
        if (!condition) throw std::runtime_error(message);
    }
    void write(const std::filesystem::path& path, std::string_view bytes)
    {
        std::filesystem::create_directories(path.parent_path());
        std::ofstream output(path, std::ios::binary);
        output << bytes;
        require(bool(output), "Cannot write capture fixture");
    }
    std::string read(const std::filesystem::path& path)
    {
        std::ifstream input(path, std::ios::binary);
        require(bool(input), "Cannot read capture fixture");
        return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    }
}

int main()
{
    using namespace openf4::assets;
    const auto root = std::filesystem::temp_directory_path() / ("openf4-asset-capture-"
        + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    try
    {
        const auto world = root / "proof.esm";
        const auto project = AssetResolver::projectAssetRoot(world);
        const auto data = root / "Data";
        const auto evidence = root / "evidence";
        write(world, "fixture");
        write(project / "models/source.nif", "first bytes");
        write(data / "textures/loose.dds", "do not copy game data");
        write(data / "textures/openf4/missing.dds", "wrong native dependency");
        write(root / "other.assets/models/source.nif", "another project");
        require(compatibilityJsonString("a\"b\\c\n\t") == "\"a\\\"b\\\\c\\u000a\\u0009\"",
            "Compatibility JSON does not escape control characters");
        {
            AssetResolver ordinary(world);
            (void)ordinary.extract("models/source.nif");
            require(!std::filesystem::exists(evidence), "Normal asset reads created evidence");
        }
        std::filesystem::create_directories(root / "existing");
        bool rejected = false;
        try { beginCompatibilityAssetCapture(world, root / "existing"); }
        catch (const std::exception&) { rejected = true; }
        require(rejected, "Capture overwrote an existing evidence directory");
        rejected = false;
        try { beginCompatibilityAssetCapture(world, project / "capture"); }
        catch (const std::exception&) { rejected = true; }
        require(rejected, "Capture allowed evidence inside its source tree");
        beginCompatibilityAssetCapture(world, evidence);
        {
            AssetResolver resolver(world, data);
            (void)resolver.extract("models/source.nif");
            write(project / "models/source.nif", "second bytes");
            (void)resolver.extract("models/source.nif");
            (void)resolver.extractTexture("textures/loose.dds");
            AssetResolver other(root / "other.esm");
            (void)other.extract("models/source.nif");
            std::vector<std::future<void>> readers;
            for (int i = 0; i < 4; ++i)
                readers.push_back(std::async(std::launch::async, [&] {
                    AssetResolver concurrent(world);
                    for (int j = 0; j < 4; ++j) (void)concurrent.extract("models/source.nif");
                }));
            for (auto& reader : readers) reader.get();
            rejected = false;
            try { (void)resolver.extractTexture("textures/openf4/missing.dds"); }
            catch (const std::exception&) { rejected = true; }
            require(rejected, "Generated fixture dependency silently resolved from game Data");
        }
        endCompatibilityAssetCapture();
        require(read(evidence / "consumed/000001.bin") == "first bytes"
            && read(evidence / "consumed/000002.bin") == "second bytes",
            "Capture did not preserve separate exact consumed versions");
        require(!std::filesystem::exists(evidence / "consumed/000003.bin"), "Capture copied loose game data");
        std::size_t captures = 0;
        for (const auto& entry : std::filesystem::directory_iterator(evidence / "consumed"))
        {
            ++captures;
            if (entry.path().filename() != "000001.bin")
                require(read(entry.path()) == "second bytes", "Concurrent capture changed bytes");
        }
        require(captures == 18, "Capture lost concurrent reads or copied another project's assets");
        const auto provenance = read(evidence / "assets.jsonl");
        require(provenance.find("\"source\":\"looseData\"") != std::string::npos
            && provenance.find("\"capturedPath\":null") != std::string::npos,
            "Excluded dependency provenance was not reported");
        std::filesystem::remove_all(root);
        std::cout << "Compatibility asset capture tests passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        endCompatibilityAssetCapture();
        std::cerr << error.what() << " (fixtures: " << root.string() << ")\n";
        return 1;
    }
}
