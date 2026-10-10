#include "script/papyrus_vm.hpp"
#include "fixtures/papyrus/compiled.hpp"

#include <algorithm>
#include <array>
#include <iostream>
#include <stdexcept>

using namespace openf4::script;
namespace fixture = openf4::test::papyrus;
namespace
{
    void check(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
    template<class F> void rejects(F&& fn, const char* message)
    {
        try { fn(); } catch (const std::exception&) { return; }
        throw std::runtime_error(message);
    }
    std::optional<std::vector<std::uint8_t>> load(std::string_view name)
    {
        std::string k(name);
        for (auto& c : k) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (k == "openf4vmbase") return std::vector(fixture::OpenF4VmBase.begin(), fixture::OpenF4VmBase.end());
        if (k == "openf4vmfixture") return std::vector(fixture::OpenF4VmFixture.begin(), fixture::OpenF4VmFixture.end());
        if (k == "openf4vmarray") return std::vector(fixture::OpenF4VmArray.begin(), fixture::OpenF4VmArray.end());
        return std::nullopt;
    }
    int counter(const PapyrusVm& vm, std::uint64_t id)
    {
        return std::get<std::int32_t>(vm.value(id, "OpenF4VmFixture", "::Counter_var"));
    }
}
int main()
{
    try {
        const auto info = inspectPex(fixture::OpenF4VmFixture);
        check(info.name == "OpenF4VmFixture" && info.parent == "OpenF4VmBase", "PEX identity");
        check(info.runtimeSupported && info.properties.at("Counter") == "Int", "PEX property metadata");
        check(std::find(info.dependencies.begin(), info.dependencies.end(), "utility") != info.dependencies.end(), "PEX native dependency");
        check(!inspectPex(fixture::OpenF4VmArray).runtimeSupported, "array runtime limitations");
        for (size_t size = 0; size < fixture::OpenF4VmBase.size(); ++size)
            rejects([&] { (void)inspectPex(std::span(fixture::OpenF4VmBase).first(size)); }, "truncation accepted");
        auto corrupt = *load("OpenF4VmBase");
        corrupt[6] = 1;
        rejects([&] { (void)inspectPex(corrupt); }, "Skyrim bytecode accepted as FO4");
        corrupt = *load("OpenF4VmBase"); corrupt.push_back(0);
        rejects([&] { (void)inspectPex(corrupt); }, "trailing data accepted");

        std::vector<std::string> errors;
        std::vector<std::uint64_t> motions, impulses;
        PapyrusVm vm(load, [&](std::string_view error) { errors.emplace_back(error); });
        vm.registerNative("ObjectReference", "SetMotionType", false, [&](std::uint64_t id, std::span<const Value> args) -> Value {
            check(args.size() == 2 && std::get<int32_t>(args[0]) == 1 && std::get<bool>(args[1]), "motion native arguments");
            motions.push_back(id); return {};
        });
        vm.registerNative("ObjectReference", "ApplyHavokImpulse", false, [&](std::uint64_t id, std::span<const Value> args) -> Value {
            check(args.size() == 4 && std::get<float>(args[2]) == 1 && std::get<float>(args[3]) == 2.5f, "impulse native arguments");
            impulses.push_back(id); return {};
        });
        rejects([&] { vm.validate("OpenF4VmArray"); }, "unsupported execution accepted");
        const std::array attachments{ScriptAttachment{"OpenF4VmFixture", {{"Impulse", 2.5f}, {"Target", Reference{42}}}}};
        vm.attach(42, attachments);
        vm.attach(43, attachments);
        vm.sendEvent(42, "OnInit"); vm.sendEvent(42, "OnLoad");
        vm.update(0);
        if (!errors.empty()) throw std::runtime_error(errors.back());
        check(counter(vm, 42) == 6 && counter(vm, 43) == 0, "instances, parent function and initial events");
        check(motions == std::vector<std::uint64_t>{42} && impulses == motions, "native calls");
        check(vm.pendingWork() == 1, "latent wait queued");
        vm.update(0.05f); check(counter(vm, 42) == 6, "wait resumed too early");
        vm.update(0.06f);
        if (!errors.empty()) throw std::runtime_error(errors.back());
        check(counter(vm, 42) == 16, "latent continuation");
        vm.sendEvent(42, "OnLoad"); vm.update(0);
        check(counter(vm, 42) == 116, "state event dispatch");
        vm.sendEvent(42, "SetComputed", std::array<Value, 1>{std::int32_t{24}}); vm.update(0);
        if (!errors.empty()) throw std::runtime_error(errors.back());
        check(counter(vm, 42) == 12, "custom property setter");
        vm.sendEvent(42, "CheckReference"); vm.update(0);
        if (!errors.empty()) throw std::runtime_error(errors.back());
        check(counter(vm, 42) == 17, "reference equality/cast/Is");
        vm.sendEvent(42, "OnUnload"); vm.sendEvent(42, "OnLoad"); vm.update(0);
        check(counter(vm, 42) == 117, "state retained through unload/reload");
        vm.sendEvent(42, "CheckParent"); vm.update(0);
        check(counter(vm, 42) == 42, "two-argument parent call");
        vm.sendEvent(42, "CheckLatentSetter"); vm.update(0); vm.update(0);
        if (!errors.empty()) throw std::runtime_error(errors.back());
        check(counter(vm, 42) == 42, "latent setter overwrote caller input");
        vm.sendEvent(42, "WaitLoop"); vm.update(0);
        for (int i = 0; i < 4096; ++i) vm.update(0);
        if (!errors.empty()) throw std::runtime_error(errors.back());
        check(counter(vm, 42) == 4096 && vm.pendingWork() == 0, "repeated Wait continuation completion");
        vm.sendEvent(42, "ArithmeticEdges"); vm.update(0);
        if (!errors.empty()) throw std::runtime_error(errors.back());
        check(counter(vm, 42) == 1, "safe integer edge and float arithmetic");
        for (int i = 0; i < 32; ++i) vm.sendEvent(42, "CheckParent");
        vm.update(0, 64);
        if (!errors.empty()) throw std::runtime_error(errors.back());
        check(counter(vm, 42) == 42 && vm.pendingWork() == 0, "earlier events starved later handler budget");

        vm.sendEvent(43, "Spin"); vm.update(0, 64);
        check(!errors.empty() && errors.back().find("budget") != std::string::npos, "runaway script bounded and diagnosed");
        const auto stopped = counter(vm, 43);
        vm.sendEvent(43, "OnLoad"); vm.update(0);
        check(counter(vm, 43) == stopped, "faulted object remained active");
        vm.attach(44, attachments); vm.sendEvent(44, "MissingNative"); vm.update(0);
        check(errors.back().find("GetPositionX") != std::string::npos && counter(vm, 44) == 0, "unknown native silently continued");
        vm.attach(45, attachments); vm.sendEvent(45, "OnLoad"); vm.update(0);
        vm.detach(45); vm.update(1);
        check(counter(vm, 45) == 4, "detached continuation executed");
        vm.attach(46, attachments); vm.sendEvent(46, "GrowString"); vm.update(0);
        check(errors.back().find("string size") != std::string::npos, "exponential string growth unbounded");
        vm.attach(47, attachments); vm.sendEvent(47, "DivideByZero"); vm.update(0);
        check(errors.back().find("division by zero") != std::string::npos, "division by zero not diagnosed");
        vm.attach(48, attachments); vm.sendEvent(48, "LatentRecursion"); vm.update(0);
        for (int i = 0; i < 150; ++i) vm.update(0);
        check(errors.back().find("latent call stack") != std::string::npos, "latent recursion unbounded");
        // Change the compiled integer divide opcode to float divide while
        // preserving valid PEX framing and its integer operands.
        auto mistyped = *load("OpenF4VmFixture");
        size_t changed = 0;
        for (size_t i = 0; i + 12 < mistyped.size(); ++i) {
            if (mistyped[i] == 7 && mistyped[i + 1] == 1 &&
                mistyped[i + 4] == 3 && mistyped[i + 5] == 1 &&
                mistyped[i + 6] == 0 && mistyped[i + 7] == 0 && mistyped[i + 8] == 0 && mistyped[i + 9] == 1) {
                mistyped[i] = 8; ++changed;
            }
        }
        check(changed == 1, "malformed fixture mutation did not identify integer division");
        PapyrusVm unsafe([&](std::string_view name) { return name == "OpenF4VmFixture" ? std::optional(mistyped) : load(name); },
            [&](std::string_view error) { errors.emplace_back(error); });
        unsafe.attach(99, attachments); unsafe.sendEvent(99, "DivideByZero"); unsafe.update(0);
        check(errors.back().find("operand type mismatch") != std::string::npos, "mistyped PEX opcode was not rejected safely");
        std::cout << "Papyrus VM tests passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Papyrus VM test failed: " << e.what() << '\n'; return 1;
    }
}
