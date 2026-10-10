#include "script/object_scripts.hpp"
#include "assets/asset_resolver.hpp"
#include "assets/coordinates.hpp"

#include <algorithm>
#include <cstring>
#include <map>
#include <set>
#include <stdexcept>

namespace openf4::script
{
    namespace
    {
        float number(const Value& value)
        {
            if (const auto* f = std::get_if<float>(&value)) return *f;
            if (const auto* i = std::get_if<std::int32_t>(&value)) return static_cast<float>(*i);
            throw std::runtime_error("Papyrus native expected a numeric argument");
        }
        void arguments(std::span<const Value> args, std::size_t count)
        {
            if (args.size() != count) throw std::runtime_error("Papyrus native argument count mismatch");
        }
        Value property(const assets::PapyrusValue& value)
        {
            return std::visit([](const auto& v) -> Value {
                using T = std::decay_t<decltype(v)>;
                if constexpr (std::is_same_v<T, assets::PapyrusObject>)
                {
                    if (v.alias != -1) throw std::runtime_error("Papyrus alias properties are not supported by the project runtime");
                    return v.formId ? Value(Reference{v.formId}) : Value(std::monostate{});
                }
                else return v;
            }, value);
        }
    }

    struct ObjectScripts::Impl
    {
        ObjectScriptHost host;
        std::vector<assets::AssetResolver> projects;
        assets::AssetResolver installed;
        struct Object
        {
            std::vector<ScriptAttachment> attachments;
            bool attached = false, loaded = false, failed = false;
        };
        std::map<std::uint32_t, Object> objects;
        PapyrusVm vm;

        std::optional<std::vector<std::uint8_t>> load(std::string_view name)
        {
            const auto path = assets::papyrusScriptPath(name);
            for (auto& resolver : projects)
                if (resolver.contains(path)) return bytes(resolver.read(path));
            if (installed.contains(path)) return bytes(installed.read(path));
            return std::nullopt;
        }
        static std::vector<std::uint8_t> bytes(const assets::AssetBytes& input)
        {
            if (input.bytes.size() > 16 * 1024 * 1024) throw std::runtime_error("Papyrus script exceeds the 16 MiB limit");
            std::vector<std::uint8_t> result(input.bytes.size());
            std::memcpy(result.data(), input.bytes.data(), input.bytes.size());
            return result;
        }

        Impl(const assets::ESMMasterStack& records, const std::filesystem::path& world,
            const std::filesystem::path& data, ObjectScriptHost callbacks)
            : host(std::move(callbacks)), installed(world, data),
              vm([this](std::string_view name) { return load(name); },
                 [this](std::string_view text) { host.diagnostic(text); })
        {
            projects.emplace_back(world);
            for (auto it = records.sourcePaths.rbegin(); it != records.sourcePaths.rend(); ++it)
                if (*it != world) projects.emplace_back(*it);
            std::map<std::uint32_t, const assets::ESMModel*> models;
            for (const auto& model : records.models) models.emplace(model.formId, &model);
            for (const auto& ref : records.references)
            {
                if (ref.deleted) continue;
                const auto model = models.find(ref.placement.baseObject);
                const auto& base = model != models.end() ? model->second->scripts : assets::PapyrusBindings{};
                if (!base.present && !ref.scripts.present) continue;
                try
                {
                    Object object;
                    for (const auto& binding : assets::effectivePapyrusScripts(base, ref.scripts))
                    {
                        const auto path = assets::papyrusScriptPath(binding.name);
                        // Only project-authored instances execute; installed scripts supply native parents.
                        if (!std::ranges::any_of(projects, [&](const auto& resolver) { return resolver.contains(path); }))
                        {
                            host.diagnostic("Skipping non-project script: " + binding.name);
                            continue;
                        }
                        ScriptAttachment attachment;
                        attachment.name = binding.name;
                        for (const auto& p : binding.properties)
                            attachment.properties.emplace(p.name, property(p.value));
                        object.attachments.push_back(std::move(attachment));
                    }
                    if (!object.attachments.empty()) objects.emplace(ref.placement.formId, std::move(object));
                }
                catch (const std::exception& error) { host.diagnostic(error.what()); }
            }
            vm.registerNative("Form", "GetFormID", false, [](std::uint64_t self, std::span<const Value> args) -> Value {
                arguments(args, 0); return static_cast<std::int32_t>(static_cast<std::uint32_t>(self));
            });
            vm.registerNative("ObjectReference", "Is3DLoaded", false,
                [this](std::uint64_t self, std::span<const Value> args) -> Value {
                    arguments(args, 0); return host.loaded(static_cast<std::uint32_t>(self));
                });
            for (int axis = 0; axis < 3; ++axis)
                vm.registerNative("ObjectReference", std::string("GetPosition") + "XYZ"[axis], false,
                    [this, axis](std::uint64_t self, std::span<const Value> args) -> Value {
                        arguments(args, 0);
                        return assets::swapBethesdaAxes(host.position(static_cast<std::uint32_t>(self)))[axis];
                    });
            vm.registerNative("ObjectReference", "SetMotionType", false,
                [this](std::uint64_t self, std::span<const Value> args) -> Value {
                    arguments(args, 2);
                    const auto type = std::get<std::int32_t>(args[0]);
                    if (type < 0 || type > 2) throw std::runtime_error("Fallout 4 motion type must be 0, 1, or 2");
                    host.motion(static_cast<std::uint32_t>(self), static_cast<std::uint32_t>(type), std::get<bool>(args[1]));
                    return {};
                });
            vm.registerNative("ObjectReference", "ApplyHavokImpulse", false,
                [this](std::uint64_t self, std::span<const Value> args) -> Value {
                    arguments(args, 4);
                    host.impulse(static_cast<std::uint32_t>(self),
                        assets::swapBethesdaAxes(glm::vec3(number(args[0]), number(args[1]), number(args[2]))), number(args[3]));
                    return {};
                });
            vm.registerNative("Debug", "Trace", true,
                [this](std::uint64_t, std::span<const Value> args) -> Value {
                    arguments(args, 2); host.diagnostic("Papyrus: " + std::get<std::string>(args[0])); return {};
                });
        }

        void update(float seconds)
        {
            for (auto& [form, object] : objects)
            {
                if (object.failed) continue;
                const bool loaded = host.loaded(form);
                if (!loaded && !object.attached) continue;
                try
                {
                    if (!object.attached)
                    {
                        vm.attach(form, object.attachments);
                        object.attached = true;
                        vm.sendEvent(form, "OnInit");
                    }
                    if (loaded != object.loaded)
                    {
                        vm.sendEvent(form, loaded ? "OnLoad" : "OnUnload");
                        object.loaded = loaded;
                    }
                }
                catch (const std::exception& error)
                {
                    object.failed = true;
                    host.diagnostic("Papyrus reference " + std::to_string(form) + ": " + error.what());
                }
            }
            vm.update(seconds);
        }
    };
    ObjectScripts::ObjectScripts(const assets::ESMMasterStack& records, const std::filesystem::path& world,
        const std::filesystem::path& data, ObjectScriptHost host)
        : mImpl(std::make_unique<Impl>(records, world, data, std::move(host))) {}
    ObjectScripts::~ObjectScripts() = default;
    void ObjectScripts::update(float seconds) { mImpl->update(seconds); }
}
