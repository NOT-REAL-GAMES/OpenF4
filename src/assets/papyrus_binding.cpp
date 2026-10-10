#include "assets/papyrus_binding.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <set>
#include <stdexcept>

namespace openf4::assets
{
namespace
{
    using Bytes = std::vector<std::byte>;
    std::string lower(std::string s)
    {
        std::ranges::transform(s, s.begin(), [](unsigned char c) { return char(std::tolower(c)); }); return s;
    }
    struct Reader
    {
        std::span<const std::byte> bytes;
        std::size_t position = 0;
        template<class T> T get()
        {
            if (sizeof(T) > bytes.size() - position) throw std::runtime_error("Truncated VMAD");
            T result; std::memcpy(&result, bytes.data() + position, sizeof(T)); position += sizeof(T); return result;
        }
        std::string text()
        {
            const auto count = get<std::uint16_t>();
            if (count > bytes.size() - position) throw std::runtime_error("Truncated VMAD string");
            std::string value(reinterpret_cast<const char*>(bytes.data() + position), count); position += count;
            if (value.find('\0') != std::string::npos) throw std::runtime_error("VMAD string contains null");
            return value;
        }
    };
    template<class T> void put(Bytes& bytes, T value)
    {
        const auto* data = reinterpret_cast<const std::byte*>(&value); bytes.insert(bytes.end(), data, data + sizeof(T));
    }
    void text(Bytes& bytes, const std::string& value)
    {
        if (value.size() > UINT16_MAX || value.find('\0') != std::string::npos) throw std::runtime_error("Invalid VMAD string");
        put(bytes, static_cast<std::uint16_t>(value.size()));
        const auto* data = reinterpret_cast<const std::byte*>(value.data()); bytes.insert(bytes.end(), data, data + value.size());
    }
    void flags(std::uint8_t value) { if (value > 3) throw std::runtime_error("Unsupported VMAD flags"); }
}

std::string papyrusScriptPath(std::string_view name, std::string_view extension)
{
    if (name.empty() || name.size() > 240 || (extension != ".pex" && extension != ".psc"))
        throw std::runtime_error("Invalid Papyrus script name or extension");
    std::string path = "scripts/"; bool first = true;
    for (const unsigned char c : name)
    {
        if (c == ':') { if (first) throw std::runtime_error("Invalid Papyrus namespace"); path += '/'; first = true; }
        else
        {
            if (!(std::isalpha(c) || c == '_' || (!first && std::isdigit(c))) || c > 127)
                throw std::runtime_error("Invalid Papyrus identifier");
            path += static_cast<char>(c); first = false;
        }
    }
    if (first) throw std::runtime_error("Empty Papyrus name component");
    path += extension; return path;
}

PapyrusBindings decodePapyrusBindings(std::span<const std::byte> bytes, std::string sourceEsm)
{
    PapyrusBindings result; result.present = true; result.sourceEsm = std::move(sourceEsm);
    result.originalBytes.assign(bytes.begin(), bytes.end());
    try
    {
        if (bytes.size() > 8 * 1024 * 1024) throw std::runtime_error("VMAD exceeds supported size");
        Reader input{bytes};
        if (input.get<std::uint16_t>() != 6 || input.get<std::uint16_t>() != 2)
            throw std::runtime_error("Only FO4 VMAD version 6, object format 2 is editable");
        const auto count = input.get<std::uint16_t>();
        if (count > 1024) throw std::runtime_error("VMAD script count exceeds supported limit");
        std::set<std::string> names;
        for (unsigned i = 0; i < count; ++i)
        {
            PapyrusScript script; script.name = input.text(); (void)papyrusScriptPath(script.name);
            if (!names.insert(lower(script.name)).second) throw std::runtime_error("Duplicate VMAD script");
            script.flags = input.get<std::uint8_t>(); flags(script.flags);
            const auto properties = input.get<std::uint16_t>();
            if (properties > 4096) throw std::runtime_error("VMAD property count exceeds supported limit");
            std::set<std::string> propertyNames;
            for (unsigned p = 0; p < properties; ++p)
            {
                PapyrusProperty property; property.name = input.text();
                if (property.name.empty() || !propertyNames.insert(lower(property.name)).second)
                    throw std::runtime_error("Empty or duplicate VMAD property");
                const auto type = input.get<std::uint8_t>(); property.flags = input.get<std::uint8_t>(); flags(property.flags);
                switch (type)
                {
                case 0: break;
                case 1:
                {
                    PapyrusObject object; object.unused = input.get<std::uint16_t>(); object.alias = input.get<std::int16_t>();
                    object.formId = input.get<std::uint32_t>(); property.value = object; break;
                }
                case 2: property.value = input.text(); break;
                case 3: property.value = input.get<std::int32_t>(); break;
                case 4:
                {
                    const auto value = input.get<float>();
                    if (!std::isfinite(value)) throw std::runtime_error("Non-finite VMAD float"); property.value = value; break;
                }
                case 5:
                {
                    const auto value = input.get<std::uint8_t>();
                    if (value > 1) throw std::runtime_error("Invalid VMAD bool"); property.value = value != 0; break;
                }
                default: throw std::runtime_error("VMAD arrays, structs, variables and unknown property types are read-only");
                }
                script.properties.push_back(std::move(property));
            }
            result.scripts.push_back(std::move(script));
        }
        if (input.position != bytes.size()) throw std::runtime_error("VMAD trailing/fragment data is read-only");
    }
    catch (const std::exception& error) { result.editable = false; result.diagnostic = error.what(); result.scripts.clear(); }
    return result;
}

std::vector<std::byte> encodePapyrusBindings(const PapyrusBindings& bindings, bool preserveOriginal)
{
    if (!bindings.present) { if (!bindings.scripts.empty()) throw std::runtime_error("Absent VMAD has scripts"); return {}; }
    if (preserveOriginal && !bindings.edited && (!bindings.editable || !bindings.originalBytes.empty())) return bindings.originalBytes;
    if (!bindings.editable) throw std::runtime_error("Cannot edit or remap unsupported VMAD: " + bindings.diagnostic);
    if (bindings.scripts.size() > 1024) throw std::runtime_error("Too many VMAD scripts");
    Bytes bytes; put(bytes, std::uint16_t{6}); put(bytes, std::uint16_t{2}); put(bytes, static_cast<std::uint16_t>(bindings.scripts.size()));
    std::set<std::string> names;
    for (const auto& script : bindings.scripts)
    {
        (void)papyrusScriptPath(script.name);
        if (!names.insert(lower(script.name)).second) throw std::runtime_error("Duplicate VMAD script");
        text(bytes, script.name); flags(script.flags); put(bytes, script.flags);
        if (script.properties.size() > 4096) throw std::runtime_error("Too many VMAD properties");
        put(bytes, static_cast<std::uint16_t>(script.properties.size())); std::set<std::string> properties;
        for (const auto& property : script.properties)
        {
            if (property.name.empty() || !properties.insert(lower(property.name)).second) throw std::runtime_error("Empty or duplicate VMAD property");
            text(bytes, property.name); put(bytes, static_cast<std::uint8_t>(property.value.index())); flags(property.flags); put(bytes, property.flags);
            std::visit([&](const auto& value) {
                using T = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<T, PapyrusObject>) { put(bytes, value.unused); put(bytes, value.alias); put(bytes, value.formId); }
                else if constexpr (std::is_same_v<T, std::string>) text(bytes, value);
                else if constexpr (std::is_same_v<T, bool>) put(bytes, std::uint8_t(value));
                else if constexpr (!std::is_same_v<T, std::monostate>)
                {
                    if constexpr (std::is_same_v<T, float>) if (!std::isfinite(value)) throw std::runtime_error("Non-finite VMAD float");
                    put(bytes, value);
                }
            }, property.value);
        }
    }
    if (bytes.size() > 8 * 1024 * 1024) throw std::runtime_error("VMAD exceeds supported size");
    return bytes;
}

void remapPapyrusBindings(PapyrusBindings& bindings, const std::function<std::uint32_t(std::uint32_t)>& remap)
{
    if (!bindings.editable) return;
    for (auto& script : bindings.scripts) for (auto& property : script.properties)
        if (auto* object = std::get_if<PapyrusObject>(&property.value)) object->formId = remap(object->formId);
}

std::vector<PapyrusScript> effectivePapyrusScripts(const PapyrusBindings& base, const PapyrusBindings& reference)
{
    if (!base.editable || !reference.editable) throw std::runtime_error("Unsupported VMAD cannot execute");
    std::vector<PapyrusScript> scripts;
    for (const auto* binding : {&base, &reference}) for (const auto& script : binding->scripts)
    {
        const auto found = std::ranges::find_if(scripts, [&](const auto& value) { return lower(value.name) == lower(script.name); });
        if (script.flags & 2u) { if (found != scripts.end()) scripts.erase(found); continue; }
        if (found == scripts.end()) scripts.push_back(script);
        else for (const auto& property : script.properties)
        {
            auto existing = std::ranges::find_if(found->properties, [&](const auto& value) { return lower(value.name) == lower(property.name); });
            if (property.flags == 3) { if (existing != found->properties.end()) found->properties.erase(existing); }
            else if (existing != found->properties.end()) *existing = property;
            else found->properties.push_back(property);
        }
    }
    for (auto& script : scripts) std::erase_if(script.properties, [](const auto& p) { return p.flags == 3; });
    return scripts;
}
}
