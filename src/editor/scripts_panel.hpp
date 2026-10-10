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
        void poll();
        [[nodiscard]] bool busy() const { return mCompile.valid(); }
    private:
        std::future<script::CompileResult> mCompile;
        std::string mDiagnostic;
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
