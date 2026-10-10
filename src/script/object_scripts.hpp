#pragma once

#include "assets/esm_document.hpp"
#include "script/papyrus_vm.hpp"
#include <functional>
#include <memory>
#include <glm/vec3.hpp>

namespace openf4::script
{
    struct ObjectScriptHost
    {
        // Absolute engine coordinates, never renderer-reflected or floating-origin relative.
        std::function<glm::vec3(std::uint32_t)> position;
        std::function<bool(std::uint32_t)> loaded;
        std::function<void(std::uint32_t, std::uint32_t, bool)> motion;
        std::function<void(std::uint32_t, glm::vec3, float)> impulse;
        std::function<void(std::string_view)> diagnostic;
    };

    // Session script state outlives cell residency; the editor never creates this runtime.
    class ObjectScripts
    {
    public:
        ObjectScripts(const assets::ESMMasterStack& records, const std::filesystem::path& world,
            const std::filesystem::path& data, ObjectScriptHost host);
        ~ObjectScripts();
        void update(float seconds);
    private:
        struct Impl;
        std::unique_ptr<Impl> mImpl;
    };
}
