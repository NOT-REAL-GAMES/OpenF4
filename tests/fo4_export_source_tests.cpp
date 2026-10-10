#include "assets/esm_document.hpp"
#include "assets/esm_editor.hpp"
#include "assets/fo4_export.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

namespace
{
    using namespace openf4::assets;
    using Bytes = std::vector<std::byte>;
    void require(bool value, const std::string& error) { if (!value) throw std::runtime_error(error); }
    template<class T> T get(std::span<const std::byte> data, std::size_t offset)
    {
        require(offset <= data.size() && sizeof(T) <= data.size() - offset, "Malformed test fixture");
        T result; std::memcpy(&result, data.data() + offset, sizeof(result)); return result;
    }
    template<class T> void patch(Bytes& data, std::size_t offset, T value)
    {
        require(offset <= data.size() && sizeof(T) <= data.size() - offset, "Invalid test fixture patch");
        std::memcpy(data.data() + offset, &value, sizeof(value));
    }
    template<class T> Bytes scalar(T value)
    {
        Bytes result(sizeof(T)); std::memcpy(result.data(), &value, sizeof(T)); return result;
    }
    void append(Bytes& result, const Bytes& part) { result.insert(result.end(), part.begin(), part.end()); }
    std::string tag(std::span<const std::byte> bytes, std::size_t offset = 0)
    {
        require(offset + 4 <= bytes.size(), "Missing test record tag");
        return {reinterpret_cast<const char*>(bytes.data() + offset), 4};
    }
    Bytes text(const char* text)
    {
        const auto first = reinterpret_cast<const std::byte*>(text); return {first, first + std::strlen(text) + 1};
    }
    Bytes sub(const char* type, const Bytes& payload)
    {
        Bytes result(6); std::memcpy(result.data(), type, 4); patch(result, 4, std::uint16_t(payload.size())); append(result, payload); return result;
    }
    Bytes record(const char* type, std::uint32_t id, const Bytes& payload)
    {
        Bytes result(24); std::memcpy(result.data(), type, 4); patch(result, 4, std::uint32_t(payload.size()));
        patch(result, 12, id); patch(result, 20, std::uint16_t{131}); append(result, payload); return result;
    }
    Bytes read(const std::filesystem::path& path)
    {
        std::ifstream input(path, std::ios::binary | std::ios::ate); require(bool(input), "Cannot read test fixture");
        Bytes result(static_cast<std::size_t>(input.tellg())); input.seekg(0);
        require(bool(input.read(reinterpret_cast<char*>(result.data()), result.size())), "Cannot read complete fixture"); return result;
    }
    void write(const std::filesystem::path& path, const Bytes& bytes)
    {
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        require(bool(output.write(reinterpret_cast<const char*>(bytes.data()), bytes.size())), "Cannot write fixture");
    }
    Bytes rewrite(std::span<const std::byte> bytes, const std::function<void(Bytes&)>& operation)
    {
        Bytes result;
        for (std::size_t p = 0; p < bytes.size();)
        {
            const bool grouped = tag(bytes, p) == "GRUP";
            const auto size = std::size_t(get<std::uint32_t>(bytes, p + 4)) + (grouped ? 0 : 24);
            require(size >= 24 && size <= bytes.size() - p, "Invalid fixture group length");
            Bytes node(bytes.begin() + p, bytes.begin() + p + 24);
            if (grouped) append(node, rewrite(bytes.subspan(p + 24, size - 24), operation));
            else node.insert(node.end(), bytes.begin() + p + 24, bytes.begin() + p + size);
            operation(node); patch(node, 4, std::uint32_t(node.size() - (grouped ? 0 : 24))); append(result, node); p += size;
        }
        return result;
    }
    void field(Bytes& record, const std::string& name, const std::optional<Bytes>& replacement)
    {
        Bytes result(record.begin(), record.begin() + 24); bool found = false;
        for (std::size_t p = 24; p < record.size();)
        {
            const auto size = 6 + std::size_t(get<std::uint16_t>(record, p + 4));
            require(size <= record.size() - p, "Invalid fixture field");
            if (tag(record, p) == name)
            {
                found = true; if (replacement) append(result, sub(name.c_str(), *replacement));
            }
            else result.insert(result.end(), record.begin() + p, record.begin() + p + size);
            p += size;
        }
        require(found, "Fixture field not found: " + name); record = std::move(result);
    }
    void tests(const std::filesystem::path& root)
    {
        const auto world = root / "source.esm";
        saveEditableLandscape(world, createEditableLandscape());
        EsmDocument document(world);
        const auto stat = document.allocateFormId(), mstt = document.allocateFormId();
        document.setModels({{stat, "TestStatic", "unused.nif"},
            {mstt, "TestMovable", "unused.nif", {}, {}, ESMModelKind::movableStatic}});
        document.save();
        const auto valid = read(world);
        require(inspectFo4Export(world).canWriteWorldProof(), "Generated OpenF4 source was rejected");
        const auto rejects = [&](const std::string& name, const std::function<void(Bytes&)>& operation) {
            const auto corrupt = rewrite(valid, operation); require(corrupt != valid, "Mutation did not run: " + name);
            write(world, corrupt);
            const auto result = inspectFo4Export(world);
            require(!result.canWriteWorldProof() && !result.blockers.empty(), "Accepted unsupported source: " + name);
            require(read(world) == corrupt, "Preflight changed source: " + name);
        };
        rejects("additional world", [&](Bytes& node) {
            if (tag(node) == "GRUP" && get<std::uint32_t>(node, 12) == 0 && tag(node, 8) == "WRLD")
            {
                auto body = sub("EDID", text("IgnoredWorld")); append(body, sub("DATA", scalar<std::uint8_t>(1)));
                append(node, record("WRLD", 0x01009900u, body));
            }
        });
        rejects("outside-world reference", [&](Bytes& node) {
            if (tag(node) == "GRUP" && get<std::uint32_t>(node, 12) == 0 && tag(node, 8) == "STAT")
            {
                auto body = sub("NAME", scalar(stat)); append(body, sub("DATA", Bytes(24)));
                append(node, record("REFR", 0x01009901u, body));
            }
        });
        rejects("missing CELL coordinates", [](Bytes& node) { if (tag(node) == "CELL") field(node, "XCLC", std::nullopt); });
        rejects("interior flag", [](Bytes& node) { if (tag(node) == "CELL") field(node, "DATA", scalar<std::uint16_t>(1)); });
        rejects("world flags", [](Bytes& node) { if (tag(node) == "WRLD") field(node, "DATA", scalar<std::uint8_t>(3)); });
        rejects("land flags", [](Bytes& node) { if (tag(node) == "CELL") {
            for (std::size_t p = 24; p < node.size(); p += 6 + get<std::uint16_t>(node, p + 4))
                if (tag(node, p) == "XCLC") patch(node, p + 14, std::uint32_t{1});
        } });
        rejects("block ownership", [](Bytes& node) {
            if (tag(node) == "GRUP" && get<std::uint32_t>(node, 12) == 4) patch(node, 8, get<std::uint32_t>(node, 8) + 20u);
        });
        rejects("persistent child group", [](Bytes& node) {
            if (tag(node) == "GRUP" && get<std::uint32_t>(node, 12) == 9) patch(node, 12, std::uint32_t{8});
        });
        rejects("duplicate known field", [](Bytes& node) { if (tag(node) == "CELL") append(node, sub("DATA", scalar<std::uint16_t>(0))); });
        rejects("STAT material semantics", [](Bytes& node) {
            if (tag(node) == "STAT") { Bytes data(16); patch(data, 0, 45.0f); field(node, "DNAM", data); }
        });
        rejects("MSTT behavior flags", [](Bytes& node) { if (tag(node) == "MSTT") field(node, "DATA", scalar<std::uint8_t>(0)); });
        rejects("MSTT custom display name", [](Bytes& node) { if (tag(node) == "MSTT") field(node, "FULL", text("Authored name must not disappear")); });
        rejects("malformed known field size", [](Bytes& node) { if (tag(node) == "WRLD") field(node, "DATA", Bytes(2)); });
        rejects("dry water override", [](Bytes& node) { if (tag(node) == "CELL") append(node, sub("XCLW", scalar(-2048.0f))); });
        rejects("different water plane", [](Bytes& node) {
            if (tag(node) == "CELL") { field(node, "DATA", scalar<std::uint16_t>(2)); append(node, sub("XCLW", scalar(25.0f))); }
        });
        const auto wet = rewrite(valid, [](Bytes& node) {
            if (tag(node) == "CELL") { field(node, "DATA", scalar<std::uint16_t>(2)); append(node, sub("XCLW", scalar(-2048.0f))); }
        });
        write(world, wet); require(inspectFo4Export(world).canWriteWorldProof(), "Supported explicit water cells were rejected");
        require(read(world) == wet, "Water preflight changed source");
        write(world, valid); require(inspectFo4Export(world).canWriteWorldProof(), "Restored source no longer passes");
        require(std::distance(std::filesystem::directory_iterator(root), std::filesystem::directory_iterator{}) == 1,
            "Read-only preflight created output files");
    }
}

int main()
{
#ifndef _WIN32
    return 77;
#endif
    std::filesystem::path root;
    try
    {
        root = std::filesystem::temp_directory_path() / ("openf4-fo4-export-source-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directory(root); tests(root); std::filesystem::remove_all(root);
        std::cout << "Fallout source preflight tests passed\n"; return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << "\nRetained fixture: " << root.string() << '\n'; return 1;
    }
}
