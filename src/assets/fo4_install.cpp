#include "assets/fo4_install.hpp"

#include <array>
#include <cstdlib>

namespace empire::assets
{
    namespace
    {
        std::optional<std::filesystem::path> dataDirectory(const std::filesystem::path& root)
        {
            const auto data = root.filename() == "Data" ? root : root / "Data";
            if (std::filesystem::is_directory(data))
                return std::filesystem::canonical(data);
            return std::nullopt;
        }
    }

    std::optional<std::filesystem::path> findFallout4Data()
    {
#ifdef _WIN32
        char* configured = nullptr;
        std::size_t configuredSize = 0;
        _dupenv_s(&configured, &configuredSize, "FALLOUT4_PATH");
        if (configured)
        {
            if (auto found = dataDirectory(configured))
            {
                std::free(configured);
                return found;
            }
            std::free(configured);
        }
#else
        if (const char* configured = std::getenv("FALLOUT4_PATH"))
            if (auto found = dataDirectory(configured))
                return found;
#endif

        constexpr std::array candidates{
            R"(C:\Program Files (x86)\Steam\steamapps\common\Fallout 4)",
            R"(C:\Program Files\Steam\steamapps\common\Fallout 4)"
        };
        for (const auto* candidate : candidates)
        {
            if (auto found = dataDirectory(candidate))
                return found;
        }
        return std::nullopt;
    }
}
