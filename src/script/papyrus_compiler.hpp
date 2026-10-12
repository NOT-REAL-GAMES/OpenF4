#pragma once

#include "assets/papyrus_binding.hpp"

#include <filesystem>
#include <string>
#include <string_view>

namespace openf4::script
{
    struct CompileResult
    {
        bool success = false;
        std::string diagnostics;
        std::filesystem::path output;
    };

    [[nodiscard]] std::filesystem::path scriptSourcePath(
        const std::filesystem::path& world, std::string_view name);
    // Runs the installed Creation Kit compiler. Failed builds never replace a working PEX.
    [[nodiscard]] CompileResult compilePapyrus(const std::filesystem::path& world,
        const std::filesystem::path& data, std::string_view name);
    void createWakeScript(const std::filesystem::path& world, std::string_view name);
    // Script that MSTT "Start Awake" attaches; FO4 keeps ESP-loaded rigid bodies asleep otherwise.
    inline constexpr std::string_view wakeScriptName = assets::wakeOnLoadScriptName;
}
