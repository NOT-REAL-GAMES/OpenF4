#include "papyrus_vm.hpp"

#include <papyrus-vm/Fo4Reader.h>
#include <papyrus-vm/VirtualMachine.h>

#include <algorithm>
#include <cmath>
#include <deque>
#include <set>
#include <stdexcept>
#include <unordered_map>

namespace openf4::script
{
    namespace
    {
        std::string key(std::string_view name)
        {
            std::string result(name);
            for (auto& c : result) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            return result;
        }
        void checkName(std::string_view name)
        {
            if (name.empty() || name.size() > 256 || name.front() == ':' || name.back() == ':')
                throw std::runtime_error("Invalid Papyrus script name");
            for (auto c : name)
                if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_' && c != ':')
                    throw std::runtime_error("Invalid Papyrus script name: " + std::string(name));
        }
        bool builtin(std::string_view name)
        {
            const auto n = key(name);
            return n == "scriptobject" || n == "form" || n == "objectreference" || n == "utility" || n == "debug";
        }
        PexInfo information(const Fo4Script& parsed)
        {
            return {parsed.script->source, parsed.script->objectTable.front().parentClassName,
                parsed.dependencies, parsed.properties, parsed.limitations.empty(), parsed.limitations};
        }
    }

    PexInfo inspectPex(std::span<const std::uint8_t> bytes)
    {
        return information(ReadFo4Pex(bytes));
    }

    struct PapyrusVm::Impl
    {
        struct Object final : IGameObject
        {
            std::uint64_t id = 0;
            std::string label;
            bool attached = false, enabled = true;
            std::vector<std::shared_ptr<ActivePexInstance>> instances;
            const char* GetStringID() override { return label.c_str(); }
            const char* GetParentNativeScript() override { return "ObjectReference"; }
            bool EqualsByValue(const IGameObject& rhs) const override
            {
                const auto* other = dynamic_cast<const Object*>(&rhs);
                return other && other->id == id;
            }
            const std::vector<std::shared_ptr<ActivePexInstance>>& ListActivePexInstances() const override { return instances; }
            void AddScript(std::shared_ptr<ActivePexInstance> instance) noexcept override { instances.push_back(std::move(instance)); }
        };
        struct Variables final : IVariablesHolder
        {
            std::map<std::string, VarValue> values;
            VarValue state{std::string{}};
            VarValue* GetVariableByName(const char* name, const PexScript& pex) override
            {
                const auto n = key(name);
                if (n == "::state" || n == "__state") return &state;
                auto it = values.find(key(pex.source) + ":" + n);
                return it == values.end() ? nullptr : &it->second;
            }
        };
        struct Event { std::uint64_t id; std::string name; std::vector<Value> arguments; };
        struct Wait { std::uint64_t owner; double due; Viet::Promise<VarValue> promise; };
        Loader loader;
        Diagnostic diagnostic;
        std::map<std::string, Fo4Script> scripts;
        std::map<std::uint64_t, std::shared_ptr<Object>> objects;
        std::map<std::pair<std::uint64_t, std::string>, std::shared_ptr<Variables>> variables;
        VirtualMachine vm{std::vector<PexScript::Lazy>{}};
        std::deque<Event> events;
        std::vector<Wait> waits;
        std::set<std::string> loading;
        double clock = 0;
        std::uint32_t remaining = 0;
        std::uint64_t executing = 0;

        Impl(Loader source, Diagnostic log) : loader(std::move(source)), diagnostic(std::move(log))
        {
            if (!loader) throw std::invalid_argument("Papyrus VM requires a script loader");
            vm.beforeInstruction = [this] {
                if (!remaining) throw std::runtime_error("Papyrus instruction budget exceeded; stack stopped");
                --remaining;
            };
            for (const auto name : {"ScriptObject", "Form", "ObjectReference", "Utility", "Debug"}) {
                auto pex = std::make_shared<PexScript>();
                pex->source = name;
                ::Object type;
                type.NameIndex = name;
                if (std::string_view(name) == "ObjectReference") type.parentClassName = "Form";
                if (std::string_view(name) == "Form") type.parentClassName = "ScriptObject";
                pex->objectTable.push_back(std::move(type));
                vm.AddScript({name, [pex] { return pex; }});
                scripts.emplace(key(name), Fo4Script{pex});
            }
            vm.SetMissingScriptHandler([this](std::string name) -> std::optional<PexScript::Lazy> {
                auto pex = load(name);
                return PexScript::Lazy{pex->source, [pex] { return pex; }};
            });
            vm.RegisterFunction("Utility", "Wait", FunctionType::GlobalFunction,
                [this](VarValue, std::vector<VarValue> args) {
                    if (args.size() != 1 || (args[0].GetType() != VarValue::kType_Float && args[0].GetType() != VarValue::kType_Integer))
                        throw std::runtime_error("Utility.Wait expects seconds");
                    const auto seconds = static_cast<double>(args[0]);
                    if (!std::isfinite(seconds) || seconds < 0 || seconds > 86400)
                        throw std::runtime_error("Utility.Wait duration outside supported range");
                    if (waits.size() >= 1024) throw std::runtime_error("Papyrus suspended stack limit exceeded");
                    Viet::Promise<VarValue> promise;
                    waits.push_back({executing, clock + seconds, promise});
                    return VarValue(promise);
                });
        }
        ~Impl()
        {
            // Continuations hold upstream instance pointers; release them before destroying instances.
            waits.clear(); events.clear();
        }
        void report(std::uint64_t owner, std::string_view error)
        {
            if (auto it = objects.find(owner); it != objects.end()) it->second->enabled = false;
            if (diagnostic) diagnostic("Papyrus reference " + std::to_string(owner) + ": " + std::string(error));
        }
        std::shared_ptr<Object> object(std::uint64_t id)
        {
            if (!id) return {};
            auto& ptr = objects[id];
            if (!ptr) {
                if (objects.size() > 65536) throw std::runtime_error("Papyrus object limit exceeded");
                ptr = std::make_shared<Object>(); ptr->id = id; ptr->label = "Reference " + std::to_string(id);
            }
            return ptr;
        }
        VarValue encode(const Value& value)
        {
            return std::visit([this](const auto& v) -> VarValue {
                using T = std::decay_t<decltype(v)>;
                if constexpr (std::is_same_v<T, std::monostate>) return VarValue::None();
                // The objects map owns all handles for the VM lifetime. A
                // strong pointer here creates self/cyclic-property leaks.
                else if constexpr (std::is_same_v<T, Reference>) return v.id ? VarValue(object(v.id).get()) : VarValue::None();
                else if constexpr (std::is_same_v<T, float>) {
                    if (!std::isfinite(v)) throw std::runtime_error("Non-finite Papyrus float");
                    return VarValue(double(v));
                } else return VarValue(v);
            }, value);
        }
        static Value decode(const VarValue& v)
        {
            switch (v.GetType()) {
                case VarValue::kType_Object: {
                    auto* object = dynamic_cast<Object*>(static_cast<IGameObject*>(v));
                    return object ? Value(Reference{object->id}) : Value(std::monostate{});
                }
                case VarValue::kType_Bool: return static_cast<bool>(v);
                case VarValue::kType_Integer: return std::int32_t(static_cast<int>(v));
                case VarValue::kType_Float: return float(static_cast<double>(v));
                case VarValue::kType_String: return std::string(static_cast<const char*>(v));
                default: throw std::runtime_error("Unsupported Papyrus value at native boundary");
            }
        }
        std::shared_ptr<PexScript> load(std::string_view name)
        {
            checkName(name);
            const auto n = key(name);
            if (auto it = scripts.find(n); it != scripts.end()) return it->second.script;
            if (loading.size() >= 32 || !loading.insert(n).second)
                throw std::runtime_error("Papyrus inheritance/dependency cycle: " + std::string(name));
            try {
                const auto bytes = loader(name);
                if (!bytes) throw std::runtime_error("Papyrus script missing: " + std::string(name));
                auto parsed = ReadFo4Pex(*bytes);
                if (key(parsed.script->source) != n) throw std::runtime_error("Papyrus file/class name mismatch");
                if (!parsed.limitations.empty()) throw std::runtime_error("Papyrus runtime unsupported " + parsed.limitations.front() + " in " + std::string(name));
                auto pex = parsed.script;
                const auto& parent = pex->objectTable.front().parentClassName;
                if (!parent.empty()) load(parent);
                // Load static-call targets lazily: mutually referring ordinary scripts are valid.
                scripts.emplace(n, std::move(parsed));
                vm.AddScript({pex->source, [pex] { return pex; }});
                loading.erase(n);
                return pex;
            } catch (...) { loading.erase(n); throw; }
        }
        void initialize(Variables& vars, const std::shared_ptr<PexScript>& pex,
            const std::map<std::string, Value>& properties, std::set<std::string>& used)
        {
            const auto& model = pex->objectTable.front();
            if (!model.parentClassName.empty()) initialize(vars, load(model.parentClassName), properties, used);
            // VarValue assignment intentionally preserves the destination's
            // declared type, so new storage must copy-construct that metadata.
            for (const auto& v : model.variables) vars.values.try_emplace(key(pex->source) + ":" + key(v.name), v.value);
            for (const auto& prop : model.properties) {
                const auto input = std::find_if(properties.begin(), properties.end(), [&](const auto& p) { return key(p.first) == key(prop.name); });
                if (input == properties.end()) continue;
                if (!(prop.flags & 4)) throw std::runtime_error("VMAD property must have automatic storage: " + prop.name);
                auto* target = vars.GetVariableByName(prop.autoVarName.c_str(), *pex);
                if (!target) throw std::runtime_error("Automatic property storage missing: " + prop.name);
                auto converted = encode(input->second);
                const auto expected = ActivePexInstance::GetTypeByName(prop.type);
                if (converted.GetType() != expected)
                    throw std::runtime_error("VMAD property type mismatch: " + prop.name);
                *target = converted;
                target->objectType = prop.type;
                used.insert(key(input->first));
            }
        }
    };

    PapyrusVm::PapyrusVm(Loader loader, Diagnostic diagnostic) : mImpl(std::make_unique<Impl>(std::move(loader), std::move(diagnostic))) {}
    PapyrusVm::~PapyrusVm() = default;
    void PapyrusVm::registerNative(std::string className, std::string functionName, bool global, Native native)
    {
        if (!native) throw std::invalid_argument("Empty Papyrus native handler");
        mImpl->vm.RegisterFunction(className, functionName, global ? FunctionType::GlobalFunction : FunctionType::Method,
            [impl = mImpl.get(), native = std::move(native)](VarValue self, std::vector<VarValue> args) {
                std::vector<Value> values;
                for (const auto& value : args) values.push_back(Impl::decode(value));
                auto* owner = dynamic_cast<Impl::Object*>(static_cast<IGameObject*>(self));
                return impl->encode(native(owner ? owner->id : 0, values));
            });
    }
    void PapyrusVm::validate(std::string_view scriptName) { mImpl->load(scriptName); }
    void PapyrusVm::attach(std::uint64_t reference, std::span<const ScriptAttachment> attachments)
    {
        if (!reference || attachments.size() > 32) throw std::runtime_error("Invalid Papyrus attachment target/count");
        auto object = mImpl->object(reference);
        if (object->attached) throw std::runtime_error("Papyrus object is already attached");
        std::vector<VirtualMachine::ScriptInfo> info;
        std::map<std::string, std::shared_ptr<Impl::Variables>> prepared;
        for (const auto& attachment : attachments) {
            if (prepared.contains(key(attachment.name))) throw std::runtime_error("Duplicate Papyrus script attachment");
            auto pex = mImpl->load(attachment.name);
            if (builtin(attachment.name)) throw std::runtime_error("Attach a project script, not a native base class");
            auto vars = std::make_shared<Impl::Variables>();
            std::set<std::string> used;
            mImpl->initialize(*vars, pex, attachment.properties, used);
            if (used.size() != attachment.properties.size()) throw std::runtime_error("VMAD property is not declared in PEX");
            vars->state = VarValue(pex->objectTable.front().autoStateName);
            prepared.emplace(key(attachment.name), vars);
            info.push_back({pex->source, vars});
        }
        mImpl->vm.AddObject(object, info);
        for (auto& [name, vars] : prepared) mImpl->variables[{reference, name}] = std::move(vars);
        object->attached = true;
    }
    void PapyrusVm::detach(std::uint64_t reference)
    {
        if (auto it = mImpl->objects.find(reference); it != mImpl->objects.end()) {
            it->second->enabled = false;
            mImpl->vm.RemoveObject(it->second);
        }
        std::erase_if(mImpl->events, [&](const auto& event) { return event.id == reference; });
        std::erase_if(mImpl->waits, [&](const auto& wait) { return wait.owner == reference; });
    }
    void PapyrusVm::sendEvent(std::uint64_t reference, std::string_view event, std::span<const Value> arguments)
    {
        if (mImpl->events.size() >= 4096 || event.size() > 128 || arguments.size() > 128)
            throw std::runtime_error("Papyrus event limit exceeded");
        const auto it = mImpl->objects.find(reference);
        if (it == mImpl->objects.end() || !it->second->attached || !it->second->enabled) return;
        mImpl->events.push_back({reference, std::string(event), {arguments.begin(), arguments.end()}});
    }
    void PapyrusVm::update(float deltaSeconds, std::uint32_t instructionBudget)
    {
        if (!std::isfinite(deltaSeconds) || deltaSeconds < 0) throw std::invalid_argument("Invalid Papyrus delta time");
        mImpl->clock += deltaSeconds;
        const auto slice = std::min(instructionBudget, 100000u);
        // Every dispatch gets a complete slice; earlier handlers cannot make
        // an otherwise finite handler fault. Cap frame dispatches separately.
        size_t dispatches = slice ? 32 : 0;
        auto ready = std::move(mImpl->waits);
        mImpl->waits.clear();
        for (auto& wait : ready) {
            auto it = mImpl->objects.find(wait.owner);
            if (it == mImpl->objects.end() || !it->second->enabled) continue;
            if (wait.due > mImpl->clock || !dispatches) { mImpl->waits.push_back(std::move(wait)); continue; }
            --dispatches; mImpl->remaining = slice;
            mImpl->executing = wait.owner;
            try { wait.promise.Resolve(VarValue::None()); }
            catch (const std::exception& e) { mImpl->report(wait.owner, e.what()); }
        }
        const auto count = mImpl->events.size();
        for (size_t i = 0; i < count && dispatches; ++i) {
            auto event = std::move(mImpl->events.front()); mImpl->events.pop_front();
            const auto object = mImpl->objects.at(event.id);
            if (!object->enabled) continue;
            --dispatches; mImpl->remaining = slice;
            mImpl->executing = event.id;
            try {
                std::vector<VarValue> args;
                for (const auto& value : event.arguments) args.push_back(mImpl->encode(value));
                mImpl->vm.SendEvent(object, event.name.c_str(), args);
            } catch (const std::exception& e) { mImpl->report(event.id, e.what()); }
        }
        mImpl->executing = 0;
    }
    Value PapyrusVm::value(std::uint64_t reference, std::string_view script, std::string_view variable) const
    {
        const auto vars = mImpl->variables.at({reference, key(script)});
        const auto pex = mImpl->scripts.at(key(script)).script;
        auto* result = vars->GetVariableByName(std::string(variable).c_str(), *pex);
        if (!result) throw std::runtime_error("Papyrus variable not found");
        return Impl::decode(*result);
    }
    std::size_t PapyrusVm::pendingWork() const { return mImpl->events.size() + mImpl->waits.size(); }
}
