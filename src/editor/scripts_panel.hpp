#pragma once
#include "editor/editor_document.hpp"
#include "script/papyrus_compiler.hpp"
#include <array>
#include <future>

namespace openf4::editor
{
    class ScriptsPanel
    {
    public:
        void draw(EditorDocument& document, std::uint32_t reference, const std::filesystem::path& data);
        // MSTT-only "Start Awake" option: attaches/detaches the wake script on this placement.
        void drawStartAwake(EditorDocument& document, std::uint32_t reference, const std::filesystem::path& data);
        // Start Awake default support: after the document auto-attaches the wake script (or a
        // world using it opens), writes its source and compiles the PEX once if it is missing.
        void ensureWakeScript(const EditorDocument& document, const std::filesystem::path& data);
        void poll();
        [[nodiscard]] bool busy() const { return mCompile.valid(); }
    private:
        std::future<script::CompileResult> mCompile;
        std::string mDiagnostic, mAwakeStatus;
        bool mAwakeCompile = false;
        std::filesystem::path mWakeSeenPath;
        std::uint64_t mWakeSeen = 0;
        void compileWake(const EditorDocument& document, const std::filesystem::path& data);
        bool mShared = false;
        std::array<char, 256> mName{"OpenF4WakeOnLoad"};
        std::array<char, 256> mPropertyName{};
        std::array<char, 1024> mString{};
        int mPropertyType = 0, mInteger = 0;
        std::uint32_t mForm = 0;
        float mFloat = 0;
        bool mBool = false;
    };
}
