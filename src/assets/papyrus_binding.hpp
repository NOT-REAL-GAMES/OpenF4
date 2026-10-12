#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace openf4::assets
{
    // Script that MSTT "Start Awake" attaches; FO4 keeps ESP-loaded rigid bodies asleep otherwise.
    inline constexpr std::string_view wakeOnLoadScriptName = "OpenF4WakeOnLoad";

    struct PapyrusObject
    {
        std::uint32_t formId = 0;
        std::int16_t alias = -1;
        std::uint16_t unused = 0;
        bool operator==(const PapyrusObject&) const = default;
    };
    using PapyrusValue = std::variant<std::monostate, PapyrusObject, std::string, std::int32_t, float, bool>;
    struct PapyrusProperty
    {
        std::string name;
        std::uint8_t flags = 1;
        PapyrusValue value;
        bool operator==(const PapyrusProperty&) const = default;
    };
    struct PapyrusScript
    {
        std::string name;
        std::uint8_t flags = 0;
        std::vector<PapyrusProperty> properties;
        bool operator==(const PapyrusScript&) const = default;
    };
    struct PapyrusBindings
    {
        bool present = false;
        bool editable = true;
        std::string diagnostic;
        std::string sourceEsm;
        std::vector<std::byte> originalBytes;
        std::vector<PapyrusScript> scripts;
        bool edited = false;
        bool operator==(const PapyrusBindings&) const = default;
    };

    // Unsupported VMAD remains opaque so unrelated world edits cannot destroy it.
    [[nodiscard]] PapyrusBindings decodePapyrusBindings(std::span<const std::byte> bytes,
        std::string sourceEsm = {});
    [[nodiscard]] std::vector<std::byte> encodePapyrusBindings(const PapyrusBindings& bindings,
        bool preserveOriginal = true);
    void remapPapyrusBindings(PapyrusBindings& bindings,
        const std::function<std::uint32_t(std::uint32_t)>& remap);
    [[nodiscard]] std::vector<PapyrusScript> effectivePapyrusScripts(
        const PapyrusBindings& base, const PapyrusBindings& reference);
    // True when the named script survives base/reference layering (case-insensitive).
    [[nodiscard]] bool papyrusScriptActive(const PapyrusBindings& base,
        const PapyrusBindings& reference, std::string_view name);
    // Enables or disables one script on a local layer, leaving every other attachment untouched.
    // Inherited scripts are disabled with a flag-2 marker and restored by dropping that marker.
    void setPapyrusScriptActive(PapyrusBindings& local, const PapyrusBindings& inherited,
        std::string_view name, bool active);
    [[nodiscard]] std::string papyrusScriptPath(std::string_view name,
        std::string_view extension = ".pex");
}
