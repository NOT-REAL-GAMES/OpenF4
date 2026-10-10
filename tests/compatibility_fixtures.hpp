#pragma once

#include <filesystem>

namespace openf4::tests
{
    // Requires a new output directory. Failed runs retain their partial evidence.
    void runCompatibilityTests(const std::filesystem::path& outputRoot);
}
