#pragma once
#include "editor/asset_document.hpp"
#include "editor/editor_document.hpp"
#include "render/vulkan_renderer.hpp"
#include <memory>
#include <imgui.h>

namespace openf4::editor
{
    // One asset session; the world document and renderer remain owned by the shell.
    class AssetWorkspace
    {
    public:
        AssetWorkspace(EditorDocument& world, std::uint32_t model,
            const std::filesystem::path& dataDirectory);
        ~AssetWorkspace();
        AssetWorkspace(const AssetWorkspace&) = delete;
        AssetWorkspace& operator=(const AssetWorkspace&) = delete;
        void draw(render::VulkanRenderer& renderer, float delta, ImVec2& position, ImVec2& size);
        void suspend(render::VulkanRenderer& renderer);
        void save();
        void makeUnique(std::uint32_t reference);
        void history(bool redo);
        void cancelGesture();
        [[nodiscard]] bool dirty() const;
        [[nodiscard]] bool renderable() const;
        [[nodiscard]] glm::mat4 view() const;
        [[nodiscard]] glm::mat4 projection() const;
        [[nodiscard]] glm::vec3 cameraPosition() const;
        [[nodiscard]] AssetDocument& document();
        [[nodiscard]] std::uint32_t modelId() const;
        // Deterministic integration exercise, using the same authoring/save/preview commands.
        void smokeEdit();
        void smokeVerifyPhysics();
    private:
        struct Impl;
        std::unique_ptr<Impl> mImpl;
    };
}
