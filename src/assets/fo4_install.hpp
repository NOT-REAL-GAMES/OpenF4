#pragma once

#include <filesystem>
#include <optional>

namespace empire::assets
{
    [[nodiscard]] std::optional<std::filesystem::path> findFallout4Data();
}
