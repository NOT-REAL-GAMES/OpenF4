#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace openf4::script
{
    struct Reference { std::uint64_t id = 0; bool operator==(const Reference&) const = default; };
    using Value = std::variant<std::monostate, bool, std::int32_t, float, std::string, Reference>;
    struct ScriptAttachment
    {
        std::string name;
        std::map<std::string, Value> properties;
    };
    struct PexInfo
    {
        std::string name, parent;
        std::vector<std::string> dependencies;
        std::map<std::string, std::string> properties;
        bool runtimeSupported = true;
        std::vector<std::string> limitations;
    };
    // Malformed or non-FO4 bytecode throws. Valid unsupported features remain exportable.
    [[nodiscard]] PexInfo inspectPex(std::span<const std::uint8_t> bytes);

    class PapyrusVm
    {
    public:
        using Loader = std::function<std::optional<std::vector<std::uint8_t>>(std::string_view)>;
        using Diagnostic = std::function<void(std::string_view)>;
        using Native = std::function<Value(std::uint64_t, std::span<const Value>)>;
        explicit PapyrusVm(Loader loader, Diagnostic diagnostic = {});
        ~PapyrusVm();
        PapyrusVm(const PapyrusVm&) = delete;
        PapyrusVm& operator=(const PapyrusVm&) = delete;

        void registerNative(std::string className, std::string functionName, bool global, Native native);
        void validate(std::string_view scriptName);
        // Attach once per session; streaming OnUnload does not discard variables or pending waits.
        void attach(std::uint64_t reference, std::span<const ScriptAttachment> scripts);
        void detach(std::uint64_t reference);
        void sendEvent(std::uint64_t reference, std::string_view event, std::span<const Value> arguments = {});
        // At most 32 dispatches per tick. Each receives its own instruction
        // budget (hard-capped at 100000); exhaustion faults that owner visibly.
        void update(float unpausedDeltaSeconds, std::uint32_t instructionBudget = 10000);
        [[nodiscard]] Value value(std::uint64_t reference, std::string_view script, std::string_view variable) const;
        [[nodiscard]] std::size_t pendingWork() const;
    private:
        struct Impl;
        std::unique_ptr<Impl> mImpl;
    };
}
