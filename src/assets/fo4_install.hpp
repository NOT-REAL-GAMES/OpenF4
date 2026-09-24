#pragma once

#include <filesystem>
#include <optional>

namespace openf4::assets
{
    [[nodiscard]] std::optional<std::filesystem::path> findFallout4Data();
}
