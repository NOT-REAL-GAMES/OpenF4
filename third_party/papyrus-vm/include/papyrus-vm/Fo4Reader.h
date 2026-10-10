#pragma once
#include "PexScript.h"
#include <map>
#include <span>

struct Fo4Script
{
    std::shared_ptr<PexScript> script;
    std::vector<std::string> dependencies;
    std::map<std::string, std::string> properties;
    std::vector<std::string> limitations;
};
Fo4Script ReadFo4Pex(std::span<const std::uint8_t> bytes);
