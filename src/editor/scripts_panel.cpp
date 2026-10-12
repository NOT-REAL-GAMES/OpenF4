#include "editor/scripts_panel.hpp"
#include "assets/asset_resolver.hpp"
#include <imgui.h>
#include <SDL3/SDL.h>
#include <algorithm>
#include <cctype>
#include <fstream>

namespace openf4::editor
{
    void ScriptsPanel::poll()
    {
        if (mCompile.valid() && mCompile.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
        {
            try
            {
                const auto result = mCompile.get();
                mDiagnostic = (result.success ? "Compiled: " + result.output.string() : "Compilation failed")
                    + "\n" + result.diagnostics;
            }
            catch (const std::exception& error) { mDiagnostic = error.what(); }
            if (mAwakeCompile) mAwakeStatus = mDiagnostic;
            mAwakeCompile = false;
        }
    }

    void ScriptsPanel::compileWake(const EditorDocument& document, const std::filesystem::path& data)
    {
        const auto name = script::wakeScriptName;
        const auto root = assets::AssetResolver::projectAssetRoot(document.path());
        if (std::filesystem::is_regular_file(root / assets::papyrusScriptPath(name))) return;
        if (!std::filesystem::is_regular_file(script::scriptSourcePath(document.path(), name)))
            script::createWakeScript(document.path(), name);
        if (!busy())
        {
            const auto path = document.path(); const std::string scriptName(name);
            mCompile = std::async(std::launch::async, [path, data, scriptName] { return script::compilePapyrus(path, data, scriptName); });
            mAwakeCompile = true;
            mAwakeStatus = "Compiling " + std::string(name) + "...";
        }
        else mAwakeStatus = "Attached; compile " + std::string(name) + " in Papyrus scripts once the current build finishes.";
    }

    void ScriptsPanel::ensureWakeScript(const EditorDocument& document, const std::filesystem::path& data)
    {
        const bool reopened = mWakeSeenPath != document.path();
        if ((!reopened && mWakeSeen == document.wakeAttachments()) || busy()) return;
        mWakeSeenPath = document.path(); mWakeSeen = document.wakeAttachments();
        bool used = !reopened;
        if (!used)
        {
            const auto name = script::wakeScriptName;
            for (const auto& ref : document.references())
            {
                const auto* model = document.model(ref.placement.baseObject);
                if (model && model->kind == assets::ESMModelKind::movableStatic
                    && assets::papyrusScriptActive(model->scripts, ref.scripts, name)) { used = true; break; }
            }
        }
        if (!used) return;
        try { compileWake(document, data); }
        catch (const std::exception& error) { mAwakeStatus = error.what(); }
    }

    void ScriptsPanel::drawStartAwake(EditorDocument& document, std::uint32_t referenceId, const std::filesystem::path& data)
    {
        poll();
        const auto* ref = document.findReference(referenceId);
        if (!ref) return;
        const auto* model = document.model(ref->placement.baseObject);
        if (!model || model->kind != assets::ESMModelKind::movableStatic) return;
        const auto name = script::wakeScriptName;
        const bool editable = model->scripts.editable && ref->scripts.editable;
        bool awake = editable && assets::papyrusScriptActive(model->scripts, ref->scripts, name);
        ImGui::BeginDisabled(!editable);
        if (ImGui::Checkbox("Start Awake", &awake))
        {
            try
            {
                auto bindings = ref->scripts;
                assets::setPapyrusScriptActive(bindings, model->scripts, name, awake);
                bindings.edited = true;
                document.setReferenceScripts(referenceId, bindings);
                mAwakeStatus.clear();
                if (awake) compileWake(document, data);
            }
            catch (const std::exception& error) { mAwakeStatus = error.what(); }
        }
        ImGui::EndDisabled();
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
            ImGui::SetTooltip(editable ? "Attaches %s (OnLoad: SetMotionType dynamic + tiny impulse) so this rigid body\n"
                "wakes when Fallout 4 loads it from the ESP. New MSTTs start awake by default (the script\n"
                "lives on the MSTT base); unchecking disables it for this placement only."
                : "This object's VMAD is preserved read-only; Start Awake is unavailable.", name.data());
        if (!mAwakeStatus.empty()) ImGui::TextWrapped("%s", mAwakeStatus.c_str());
    }

    void ScriptsPanel::draw(EditorDocument& document, std::uint32_t referenceId, const std::filesystem::path& data)
    {
        poll();
        if (!ImGui::CollapsingHeader("Papyrus scripts")) return;
        const auto* ref = document.findReference(referenceId);
        if (!ref) return;
        const auto baseId = ref->placement.baseObject;
        const auto* model = document.model(baseId);
        if (!model) return;
        const auto inheritedBindings = model->scripts;
        ImGui::Checkbox("Shared base (all placements)", &mShared);
        auto bindings = mShared ? model->scripts : ref->scripts;
        const auto commit = [&] {
            bindings.present = true;
            bindings.edited = true;
            if (mShared) document.setModelScripts(baseId, bindings);
            else document.setReferenceScripts(referenceId, bindings);
        };
        const auto attempt = [&](auto&& operation) {
            try { operation(); }
            catch (const std::exception& error) { mDiagnostic = error.what(); }
        };
        if (!bindings.editable)
        {
            ImGui::TextWrapped("Script data is preserved read-only: %s", bindings.diagnostic.c_str());
            return;
        }
        ImGui::TextWrapped("Attachments/properties are undoable world edits. Source and compiled PEX files save separately. Scripts run only in the runtime, not while authoring.");
        for (std::size_t index = 0; index < bindings.scripts.size(); ++index)
        {
            const auto binding = bindings.scripts[index];
            ImGui::PushID(static_cast<int>(index));
            if (ImGui::Selectable((binding.name + ((binding.flags & 2) ? " [disabled]" : "")).c_str()))
                SDL_strlcpy(mName.data(), binding.name.c_str(), mName.size());
            ImGui::SameLine();
            if (ImGui::SmallButton("Remove local binding"))
                attempt([&] { bindings.scripts.erase(bindings.scripts.begin() + index); commit(); });
            for (const auto& property : binding.properties)
            {
                const auto text = std::visit([](const auto& value) -> std::string {
                    using T = std::decay_t<decltype(value)>;
                    if constexpr (std::is_same_v<T, std::monostate>) return "None";
                    else if constexpr (std::is_same_v<T, assets::PapyrusObject>) return "Form " + std::to_string(value.formId);
                    else if constexpr (std::is_same_v<T, std::string>) return value;
                    else if constexpr (std::is_same_v<T, bool>) return value ? "true" : "false";
                    else return std::to_string(value);
                }, property.value);
                ImGui::TextWrapped("  %s = %s", property.name.c_str(), text.c_str());
            }
            ImGui::PopID();
        }
        if (!mShared && inheritedBindings.editable && !inheritedBindings.scripts.empty())
        {
            ImGui::TextUnformatted("Base bindings (inherited unless overridden):");
            for (const auto& inherited : inheritedBindings.scripts)
            {
                ImGui::PushID(inherited.name.c_str());
                ImGui::TextUnformatted(inherited.name.c_str());
                ImGui::SameLine();
                if (ImGui::SmallButton("Disable here")) attempt([&] {
                    std::erase_if(bindings.scripts, [&](const auto& b) { return assets::AssetResolver::normalizePath(b.name)
                        == assets::AssetResolver::normalizePath(inherited.name); });
                    bindings.scripts.push_back({inherited.name, 2, {}}); commit();
                });
                ImGui::PopID();
            }
        }
        ImGui::InputText("Script name", mName.data(), mName.size());
        if (ImGui::Button("Attach / enable")) attempt([&] {
            (void)assets::papyrusScriptPath(mName.data());
            auto found = std::ranges::find_if(bindings.scripts, [&](const auto& b) {
                return assets::AssetResolver::normalizePath(b.name) == assets::AssetResolver::normalizePath(mName.data()); });
            if (found == bindings.scripts.end()) bindings.scripts.push_back({mName.data(), 0, {}});
            else found->flags = 0;
            commit();
        });
        ImGui::SameLine();
        if (ImGui::Button("Create wake source")) attempt([&] {
            script::createWakeScript(document.path(), mName.data());
            mDiagnostic = "Created source; compile and attach it, then Save World.";
        });
        if (ImGui::Button("Open source externally")) attempt([&] {
            const auto path = std::filesystem::absolute(script::scriptSourcePath(document.path(), mName.data()));
            if (!std::filesystem::is_regular_file(path)) throw std::runtime_error("Create or copy the script source first");
            const auto utf8 = path.generic_u8string();
            std::string url = "file:///";
            constexpr char hex[] = "0123456789ABCDEF";
            for (const auto byte : utf8)
            {
                const auto c = static_cast<unsigned char>(byte);
                if (std::isalnum(c) || c == '/' || c == ':' || c == '-' || c == '_' || c == '.') url += static_cast<char>(c);
                else { url += '%'; url += hex[c >> 4]; url += hex[c & 15]; }
            }
            if (!SDL_OpenURL(url.c_str())) throw std::runtime_error(SDL_GetError());
        });
        ImGui::SameLine();
        ImGui::BeginDisabled(busy());
        if (ImGui::Button("Compile"))
        {
            const auto path = document.path(); const std::string name(mName.data());
            mCompile = std::async(std::launch::async, [path, data, name] { return script::compilePapyrus(path, data, name); });
        }
        ImGui::EndDisabled();
        ImGui::InputText("Property", mPropertyName.data(), mPropertyName.size());
        ImGui::Combo("Type", &mPropertyType, "Bool\0Int\0Float\0String\0Object FormID\0");
        switch (mPropertyType)
        {
        case 0: ImGui::Checkbox("Value", &mBool); break;
        case 1: ImGui::InputInt("Value", &mInteger); break;
        case 2: ImGui::InputFloat("Value", &mFloat); break;
        case 3: ImGui::InputText("Value", mString.data(), mString.size()); break;
        case 4: ImGui::InputScalar("Value (hex)", ImGuiDataType_U32, &mForm, nullptr, nullptr, "%08X", ImGuiInputTextFlags_CharsHexadecimal); break;
        }
        if (ImGui::Button("Apply property")) attempt([&] {
            auto found = std::ranges::find_if(bindings.scripts, [&](const auto& b) {
                return assets::AssetResolver::normalizePath(b.name) == assets::AssetResolver::normalizePath(mName.data()); });
            if (found == bindings.scripts.end()) throw std::runtime_error("Attach the script locally before overriding its properties");
            assets::PapyrusValue value;
            switch (mPropertyType)
            {
            case 0: value = mBool; break;
            case 1: value = static_cast<std::int32_t>(mInteger); break;
            case 2: value = mFloat; break;
            case 3: value = std::string(mString.data()); break;
            case 4: value = assets::PapyrusObject{mForm}; break;
            }
            std::erase_if(found->properties, [&](const auto& p) { return assets::AssetResolver::normalizePath(p.name)
                == assets::AssetResolver::normalizePath(mPropertyName.data()); });
            found->properties.push_back({mPropertyName.data(), 1, value}); commit();
        });
        if (busy()) ImGui::TextUnformatted("Compiling in background...");
        if (!mDiagnostic.empty()) ImGui::TextWrapped("%s", mDiagnostic.c_str());
    }
}
