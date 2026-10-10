#include "editor/editor_app.hpp"
#include "editor/editor_document.hpp"
#include "editor/editor_manipulation.hpp"
#include "editor/asset_workspace.hpp"
#include "editor/scripts_panel.hpp"
#include "assets/fo4_install.hpp"
#include "assets/fo4_export.hpp"
#include "assets/lod_bake.hpp"
#include "assets/lod_statics.hpp"
#include "assets/lod_terrain.hpp"
#include "assets/model_import.hpp"
#include "world/world_scene.hpp"

#include <SDL3/SDL.h>
#include <imgui.h>
#include <imgui_internal.h>
#include <imgui_impl_sdl3.h>
#include <imgui_impl_vulkan.h>
#include <ImGuizmo.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <deque>
#include <filesystem>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace openf4::editor
{
    namespace
    {
        constexpr float cellSize = 4096.0f;

        void checkVk(VkResult result)
        {
            if (result != VK_SUCCESS)
                throw std::runtime_error("Editor Vulkan operation failed: " + std::to_string(result));
        }

        struct SdlLifetime
        {
            SdlLifetime()
            {
                if (!SDL_Init(SDL_INIT_VIDEO))
                    throw std::runtime_error(SDL_GetError());
            }
            ~SdlLifetime() { SDL_Quit(); }
        };

        struct WindowDeleter
        {
            void operator()(SDL_Window* window) const { SDL_DestroyWindow(window); }
        };

        class EditorUi
        {
        public:
            explicit EditorUi(SDL_Window* window) : mWindow(window)
            {
                IMGUI_CHECKVERSION();
                ImGui::CreateContext();
                ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_DockingEnable;
                ImGui::GetIO().ConfigWindowsMoveFromTitleBarOnly = true;
                ImGui::StyleColorsDark();
                auto& style = ImGui::GetStyle();
                style.WindowRounding = 3.0f;
                style.FrameRounding = 3.0f;
                style.GrabRounding = 3.0f;
                style.Colors[ImGuiCol_WindowBg] = {0.075f, 0.083f, 0.10f, 1.0f};
                style.Colors[ImGuiCol_Header] = {0.14f, 0.25f, 0.32f, 1.0f};
                if (!ImGui_ImplSDL3_InitForVulkan(window))
                    throw std::runtime_error("Unable to initialize ImGui SDL3 backend");
                mPlatformAttached = true;
            }

            ~EditorUi()
            {
                detach();
                ImGui::DestroyContext();
            }

            void attach(render::UiContext context)
            {
                if (mAttached && context.device == mContext.device
                    && context.swapchainGeneration == mContext.swapchainGeneration)
                    return;
                detach();
                // Vulkan shutdown destroys ImGui platform-window data too, including
                // the main SDL window ID used to route events. Rebind both backends.
                if (!ImGui_ImplSDL3_InitForVulkan(mWindow))
                    throw std::runtime_error("Unable to reinitialize ImGui SDL3 backend");
                mPlatformAttached = true;
                mContext = context;
                const std::array poolSizes{
                    VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 256},
                    VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 256},
                    VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_SAMPLER, 32}};
                VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
                pool.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
                pool.maxSets = 544;
                pool.poolSizeCount = static_cast<std::uint32_t>(poolSizes.size());
                pool.pPoolSizes = poolSizes.data();
                checkVk(vkCreateDescriptorPool(context.device, &pool, nullptr, &mPool));

                ImGui_ImplVulkan_InitInfo init{};
                init.ApiVersion = VK_API_VERSION_1_3;
                init.Instance = context.instance;
                init.PhysicalDevice = context.physicalDevice;
                init.Device = context.device;
                init.QueueFamily = context.queueFamily;
                init.Queue = context.queue;
                init.DescriptorPool = mPool;
                init.MinImageCount = context.minImageCount;
                init.ImageCount = context.imageCount;
                init.UseDynamicRendering = true;
                init.CheckVkResultFn = checkVk;
                init.MinAllocationSize = 1024 * 1024;
                auto& pipeline = init.PipelineInfoMain;
                pipeline.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
                pipeline.PipelineRenderingCreateInfo.sType =
                    VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR;
                pipeline.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
                pipeline.PipelineRenderingCreateInfo.pColorAttachmentFormats = &mContext.colorFormat;
                pipeline.PipelineRenderingCreateInfo.depthAttachmentFormat = context.depthFormat;
                if (!ImGui_ImplVulkan_Init(&init))
                    throw std::runtime_error("Unable to initialize ImGui Vulkan backend");
                mAttached = true;
            }

            void detach()
            {
                if (mContext.device)
                    vkDeviceWaitIdle(mContext.device);
                if (mAttached)
                    ImGui_ImplVulkan_Shutdown();
                if (mPlatformAttached)
                    ImGui_ImplSDL3_Shutdown();
                if (mPool)
                    vkDestroyDescriptorPool(mContext.device, mPool, nullptr);
                mPool = VK_NULL_HANDLE;
                mAttached = false;
                mPlatformAttached = false;
                mContext = {};
            }

        private:
            SDL_Window* mWindow = nullptr;
            render::UiContext mContext{};
            VkDescriptorPool mPool = VK_NULL_HANDLE;
            bool mAttached = false;
            bool mPlatformAttached = false;
        };

        enum class FileAction { open, create, saveAs, import, exportFolder };
        struct DialogResult
        {
            FileAction action;
            std::filesystem::path path;
            std::string error;
        };
        struct DialogMailbox
        {
            std::mutex mutex;
            std::deque<DialogResult> results;
            bool pending = false;
        };
        struct DialogRequest
        {
            std::shared_ptr<DialogMailbox> mailbox;
            FileAction action;
        };

        void SDLCALL fileDialogResult(void* userdata, const char* const* files, int)
        {
            std::unique_ptr<DialogRequest> request(static_cast<DialogRequest*>(userdata));
            std::scoped_lock lock(request->mailbox->mutex);
            request->mailbox->pending = false;
            if (!files)
                request->mailbox->results.push_back({request->action, {}, SDL_GetError()});
            else if (*files)
                request->mailbox->results.push_back({request->action,
                    std::filesystem::u8path(*files), {}});
        }

        glm::mat4 reflection()
        {
            glm::mat4 result(1.0f);
            result[0][0] = -1.0f;
            return result;
        }

        glm::dvec3 renderOrigin(glm::dvec3 position)
        {
            return {std::floor(position.x / cellSize) * cellSize, 0.0,
                std::floor(position.z / cellSize) * cellSize};
        }

        struct PendingImport
        {
            std::uint32_t formId;
            glm::mat4 transform;
            std::future<assets::ImportedModelAsset> future;
        };

        struct BakeProgressState
        {
            std::atomic<bool> cancel{false};
            std::atomic<std::uint64_t> completed{0}, total{0};
        };

        struct ExportProgressState
        {
            std::atomic<bool> cancel{false};
            std::mutex mutex;
            assets::Fo4ExportProgress progress;
        };

        assets::LodBakePreflight preflightLod(const std::filesystem::path& world,
            const std::optional<std::filesystem::path>& data, const assets::LodBakeOptions& options)
        {
            assets::TerrainLodSource terrain(world);
            assets::StaticLodSource statics(world, data);
            return assets::LodBuildIndex(world).preflight(terrain, statics, options);
        }

        assets::LodBakeResult bakeLod(const std::filesystem::path& world,
            const std::optional<std::filesystem::path>& data, const assets::LodBakeOptions& options)
        {
            assets::TerrainLodSource terrain(world);
            assets::StaticLodSource statics(world, data);
            return assets::LodBuildIndex(world).bake(terrain, statics, options);
        }

        class EditorApp
        {
        public:
            EditorApp(std::filesystem::path data, std::filesystem::path path,
                int frameLimit, std::filesystem::path smokeImport, bool smokeTest, bool smokeBake, bool smokeAsset,
                std::filesystem::path smokeExport)
                : mData(std::move(data)), mFrameLimit(frameLimit),
                  mSmokeExport(std::move(smokeExport)), mSmokeImport(std::move(smokeImport)),
                  mSmokeTest(smokeTest), mSmokeBake(smokeBake), mSmokeAsset(smokeAsset)
            {
                mWindow.reset(SDL_CreateWindow("OpenF4 Editor", 1600, 1000,
                    SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY));
                if (!mWindow)
                    throw std::runtime_error(SDL_GetError());
                mUi = std::make_unique<EditorUi>(mWindow.get());
                openWorld(path);
            }

            ~EditorApp()
            {
                stopExport();
                // The UI backend owns Vulkan resources and must be destroyed first.
                mUi.reset();
            }

            int run()
            {
                std::cout << "OpenF4 editor: " << mDocument->path().string() << '\n'
                    << "RMB + WASD: fly; Q/E: down/up; Shift: faster; F: frame selection\n"
                    << "W/E/R: move/rotate/scale; Delete: delete; Ctrl+D: duplicate; "
                       "End: drop to LAND; Escape: cancel drag; Ctrl+Z/Y: undo/redo; Ctrl+S: save\n";
                auto previous = std::chrono::steady_clock::now();
                while (mRunning)
                {
                    const auto now = std::chrono::steady_clock::now();
                    const float delta = std::clamp(std::chrono::duration<float>(now - previous).count(),
                        0.0001f, 0.1f);
                    previous = now;
                    if (mNextFrameAction) { auto action = std::move(mNextFrameAction); mNextFrameAction = {}; attempt(action); }
                    events();
                    handleDialogs();
                    if (!mAssetMode) finishImport();
                    if (mSmokeReopen)
                    {
                        mSmokeReopen = false;
                        attempt([&] {
                            const auto path = mDocument->path();
                            openWorld(path);
                            if (mDocument->references().size() != mSmokeExpectedReferences)
                                throw std::runtime_error("Editor smoke reopen lost placed references");
                            if (mSmokeReference)
                            {
                                const auto actual = mDocument->engineTransform(*mSmokeReference);
                                for (int column = 0; column < 4; ++column)
                                    for (int row = 0; row < 4; ++row)
                                        if (std::abs(actual[column][row] - mSmokeTransform[column][row]) > 0.01f)
                                            throw std::runtime_error("Editor smoke reopen changed the object transform");
                                select(*mSmokeReference);
                            }
                            if (!mSmokeBlockTransforms.empty()) select(std::nullopt);
                            for (const auto& [id, expected] : mSmokeBlockTransforms)
                            {
                                const auto actual = mDocument->engineTransform(id);
                                for (int column = 0; column < 4; ++column)
                                    if (glm::length(actual[column] - expected[column]) > 0.01f)
                                        throw std::runtime_error("Editor smoke block changed after reopen");
                                select(id, true);
                            }
                            if (mTerrain && mSmokeTerrainCell)
                            {
                                const auto reloaded = mTerrain->snapshot();
                                if (!reloaded.containsCell(*mSmokeTerrainCell)
                                    || reloaded.cell(*mSmokeTerrainCell).heights != mSmokeTerrainHeights)
                                    throw std::runtime_error("Editor smoke reopen changed sculpted LAND");
                            }
                            mSmokeReopened = true;
                            mSmokeReopenFrame = mFrames;
                            std::cout << "Editor smoke reopen verified\n";
                        });
                    }
                    if (!mRunning)
                        break;
                    int width = 0, height = 0;
                    SDL_GetWindowSizeInPixels(mWindow.get(), &width, &height);
                    if (width <= 0 || height <= 0)
                    {
                        SDL_Delay(16);
                        continue;
                    }
                    mUi->attach(mRenderer->uiContext());
                    ImGui_ImplVulkan_NewFrame();
                    ImGui_ImplSDL3_NewFrame();
                    ImGui::NewFrame();
                    ImGuizmo::BeginFrame();
                    mScripts.poll();
                    if (!mAssetMode) pollBake();
                    pollExport();
                    shell();
                    ImGui::BeginDisabled(mExportDialog);
                    if (mAssetMode && mAsset)
                    {
                        // Asset shortcuts include saves; keep its preview frozen behind the export modal.
                        if (!mExportDialog) mAsset->draw(*mRenderer, delta, mViewportPosition, mViewportSize);
                    }
                    else
                    {
                        if (!mExportDialog) navigate(delta);
                        reconcileSelection();
                        viewport(delta);
                        inspector();
                        outliner();
                    }
                    assetsPanel();
                    diagnostics();
                    ImGui::EndDisabled();
                    if (!mAssetMode) { shortcuts(); projectObjects(); }
                    smokeStep();
                    ImGui::Render();
                    const glm::dvec3 origin = mAssetMode ? glm::dvec3(0) : renderOrigin(mPosition);
                    const glm::vec3 local = mAssetMode ? mAsset->cameraPosition() : glm::vec3(mPosition - origin);
                    const glm::mat4 view = mAssetMode ? mAsset->view() : glm::lookAtRH(local, local + forward(), {0.0f, 1.0f, 0.0f});
                    auto projection = render::reverseZInfinitePerspectiveRH_ZO(glm::radians(70.0f),
                        std::max(mViewportSize.x, 1.0f) / std::max(mViewportSize.y, 1.0f), 0.1f);
                    projection[1][1] *= -1.0f;
                    if (mAssetMode) projection = mAsset->projection();
                    const auto scale = ImGui::GetIO().DisplayFramebufferScale;
                    render::RenderViewport rectangle{
                        static_cast<std::int32_t>(mViewportPosition.x * scale.x),
                        static_cast<std::int32_t>(mViewportPosition.y * scale.y),
                        static_cast<std::uint32_t>(std::max(mViewportSize.x, 0.0f) * scale.x),
                        static_cast<std::uint32_t>(std::max(mViewportSize.y, 0.0f) * scale.y)};
                    if (mAssetMode && !mAsset->renderable()) rectangle.width = rectangle.height = 0;
                    mRenderer->render(view, projection, local, glm::vec3(origin), rectangle,
                        [](VkCommandBuffer commands) {
                            ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), commands);
                        });
                    ++mFrames;
                    if (mFrameLimit > 0 && mFrames >= mFrameLimit)
                        mRunning = false;
                }
                finishStroke();
                if (mAsset) mAsset->suspend(*mRenderer);
                finishManipulation(true);
                // Do not abandon a source copy on shutdown: finish and commit its result.
                if (mImport)
                {
                    mImport->future.wait();
                    finishImport();
                }
                // Compilation publishes into this world's sidecar; finish before final save/teardown.
                while (mScripts.busy()) { mScripts.poll(); if (mScripts.busy()) SDL_Delay(10); }
                stopExport();
                if (mBakeProgress) mBakeProgress->cancel.store(true);
                if (mBakePreview.valid()) mBakePreview.wait();
                if (mBakeJob.valid()) mBakeJob.wait();
                mBakePreview = {};
                mBakeJob = {};
                mBakeAwaitingConfirmation = false;
                mBakeProgress.reset();
                if (mDocument->dirty())
                    save();
                std::cout << "Editor finished: " << mFrames << " frames, "
                    << mDocument->references().size() << " references, "
                    << mObjects->loaded() << " loaded, " << mObjects->pending() << " pending, "
                    << (mTerrain ? mTerrain->pendingWork() : 0) << " terrain jobs pending\n";
                if (mSmokeTest && (mSmokeMenuChecks != 3 || !mSmokeCommands || !mSmokeReopened
                    || mObjects->pending() != 0 || (mTerrain && mTerrain->pendingWork() != 0)
                    || (mSmokeBake && !mSmokeBakeCompleted)
                    || (!mSmokeImport.empty() && mObjects->loaded() == 0)))
                {
                    std::cerr << "Editor smoke did not complete menu input, imports, commands, reopen, and streaming drain\n";
                    mSmokeFailed = true;
                }
                if (mSmokeAsset && mAssetSmokeStage != 6) { std::cerr << "Asset smoke did not finish\n"; return 1; }
                if (!mSmokeExport.empty() && (!mSmokeExportDone || mSmokeFailed))
                { std::cerr << "Editor export smoke did not finish\n"; return 1; }
                if ((mSmokeTest || mSmokeAsset) && mSmokeFailed) return 1;
                return 0;
            }

        private:
            void recordError(const std::exception& error)
            {
                mMessages.push_back(error.what());
                std::cerr << "Editor: " << error.what() << '\n';
                mSmokeFailed |= mSmokeTest || mSmokeAsset || !mSmokeExport.empty();
            }

            template<class Operation>
            void attempt(Operation&& operation)
            {
                try { operation(); }
                catch (const std::exception& error) { recordError(error); }
            }

            [[nodiscard]] bool bakeBusy() const noexcept
            {
                return mBakePreview.valid() || mBakeJob.valid() || mBakeAwaitingConfirmation;
            }

            bool exportBusy() const { return mExportPreview.valid() || mExportJob.valid(); }

            void stopExport()
            {
                if (mExportProgress) mExportProgress->cancel.store(true);
                if (mExportPreview.valid()) mExportPreview.wait();
                if (mExportJob.valid()) mExportJob.wait();
                pollExport();
                mExportPreview = {}; mExportJob = {};
                mExportDialog = false;
            }

            void requestExport()
            {
                if (mExportDialog) return;
                if (bakeBusy() || mImport || mScripts.busy())
                    throw std::runtime_error("Finish the current bake, import, or script compilation before exporting");
                { std::scoped_lock lock(mDialogMailbox->mutex);
                  if (mDialogMailbox->pending) throw std::runtime_error("Finish the current file dialog before exporting"); }
                if (mAsset && mAsset->dirty())
                { guardAsset([this] { requestExport(); }); return; }
                finishManipulation(true); finishStroke();
                mNavigating = false;
                SDL_SetWindowRelativeMouseMode(mWindow.get(), false);
                const auto destination = mDocument->path().parent_path()
                    / (mDocument->path().stem().string() + "-Fallout4");
                setExportDestination(destination);
                mExportPreflight.reset(); mExportResult.reset(); mExportError.clear();
                mExportProgress = std::make_shared<ExportProgressState>();
                mExportDialog = mExportOpenPopup = true;
            }

            void setExportDestination(const std::filesystem::path& path)
            {
                const auto utf8 = path.u8string();
                if (utf8.size() >= mExportDestination.size()) throw std::runtime_error("Export path is too long");
                SDL_strlcpy(mExportDestination.data(), reinterpret_cast<const char*>(utf8.c_str()), mExportDestination.size());
            }

            assets::Fo4ExportOptions exportOptions() const
            {
                assets::Fo4ExportOptions options;
                options.dataDirectory = mData;
                options.cancelled = [progress = mExportProgress] { return progress->cancel.load(); };
                options.progress = [progress = mExportProgress](const assets::Fo4ExportProgress& update) {
                    std::scoped_lock lock(progress->mutex); progress->progress = update;
                };
                return options;
            }

            void inspectExport()
            {
                save();
                mExportWorld = mDocument->path();
                mExportError.clear(); mExportPreflight.reset();
                mExportProgress->cancel.store(false);
                mExportPreview = std::async(std::launch::async,
                    [world = mExportWorld, options = exportOptions()] { return assets::inspectFo4Export(world, options); });
            }

            void launchExport()
            {
                const auto destination = std::filesystem::u8path(mExportDestination.data());
                if (destination.empty() || std::filesystem::exists(destination))
                    throw std::runtime_error("Choose a new, nonempty package directory; existing directories are never overwritten");
                if (!mExportPreflight || !mExportPreflight->canWriteWorldProof())
                    throw std::runtime_error("Export preflight must pass first");
                mExportProgress->cancel.store(false);
                mExportError.clear();
                mExportJob = std::async(std::launch::async,
                    [world = mExportWorld, destination, options = exportOptions()] {
                        return assets::exportFo4WorldProof(world, destination, options);
                    });
            }

            void pollExport()
            {
                try
                {
                    if (mExportPreview.valid() && mExportPreview.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
                        mExportPreflight = mExportPreview.get();
                    if (mExportJob.valid() && mExportJob.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
                    {
                        mExportResult = mExportJob.get();
                        mMessages.push_back("Fallout 4 package exported: " + mExportResult->pluginPath.string());
                        std::cout << mMessages.back() << '\n';
                    }
                }
                catch (const std::exception& error)
                {
                    mExportError = error.what();
                    mMessages.push_back("Fallout 4 export: " + mExportError);
                    std::cerr << mMessages.back() << '\n';
                }
            }

            void exportPanel()
            {
                if (mExportOpenPopup) { ImGui::OpenPopup("Export to Fallout 4 (.esp)"); mExportOpenPopup = false; }
                ImGui::SetNextWindowSize({680, 0}, ImGuiCond_Appearing);
                if (!ImGui::BeginPopupModal("Export to Fallout 4 (.esp)", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) return;
                ImGui::TextWrapped("Experimental world geometry export, not a complete playable-world export. Navigation and native distant LOD are not generated. DEM/overlay worlds are not supported yet.");
                ImGui::TextWrapped("Your .esm remains the authoring project. Export creates a new package containing Data/<world>.esp, native assets, scripts, and export-report.txt. Nothing is installed into Fallout 4.");
                ImGui::Separator();
                const bool busy = exportBusy();
                ImGui::BeginDisabled(busy || mExportResult.has_value());
                ImGui::InputText("New package folder", mExportDestination.data(), mExportDestination.size());
                if (ImGui::Button("Choose parent folder...")) showDialog(FileAction::exportFolder);
                ImGui::EndDisabled();
                if (mExportResult)
                {
                    ImGui::TextWrapped("Export complete: %s", mExportResult->pluginPath.string().c_str());
                    ImGui::TextWrapped("Read export-report.txt before testing in a separate Fallout 4 profile. Your editor world is still %s.", mDocument->path().filename().string().c_str());
                    ImGui::Text("Arrival: cow %s %d %d", mExportResult->worldEditorId.c_str(),
                        mExportResult->arrivalCell.x, mExportResult->arrivalCell.y);
                }
                else if (busy)
                {
                    assets::Fo4ExportProgress progress;
                    { std::scoped_lock lock(mExportProgress->mutex); progress = mExportProgress->progress; }
                    ImGui::TextWrapped("%s", mExportProgress->cancel.load() ? "Canceling safely..."
                        : progress.phase.empty() ? "Checking world and dependencies..." : progress.phase.c_str());
                    if (progress.total) ImGui::ProgressBar(static_cast<float>(progress.completed) / progress.total);
                    ImGui::BeginDisabled(mExportProgress->cancel.load());
                    if (ImGui::Button("Cancel export")) mExportProgress->cancel.store(true);
                    ImGui::EndDisabled();
                }
                else if (mExportPreflight)
                {
                    const auto& preflight = *mExportPreflight;
                    ImGui::Text("%llu cells, %llu references, %llu models; estimated %.1f MiB",
                        static_cast<unsigned long long>(preflight.cells), static_cast<unsigned long long>(preflight.references),
                        static_cast<unsigned long long>(preflight.models), preflight.estimatedBytes / 1048576.0);
                    for (const auto& limitation : preflight.limitations) ImGui::TextWrapped("- %s", limitation.c_str());
                    for (const auto& blocker : preflight.blockers) ImGui::TextWrapped("Cannot export: %s", blocker.c_str());
                    ImGui::BeginDisabled(!preflight.canWriteWorldProof());
                    if (ImGui::Button("Export package"))
                        try { launchExport(); } catch (const std::exception& error) { mExportError = error.what(); }
                    ImGui::EndDisabled();
                }
                else
                {
                    ImGui::TextWrapped("The next step saves current world edits, then checks export compatibility. Canceling export does not undo that save.");
                    if (ImGui::Button("Save World and Check Export"))
                        try { inspectExport(); } catch (const std::exception& error) { mExportError = error.what(); }
                }
                if (!mExportError.empty()) ImGui::TextWrapped("%s", mExportError.c_str());
                bool dialogPending;
                { std::scoped_lock lock(mDialogMailbox->mutex); dialogPending = mDialogMailbox->pending; }
                ImGui::BeginDisabled(busy || dialogPending);
                if (ImGui::Button("Close")) { mExportDialog = false; ImGui::CloseCurrentPopup(); }
                ImGui::EndDisabled();
                ImGui::EndPopup();
            }

            void requestBake(bool full)
            {
                if (mExportDialog) throw std::runtime_error("Close the export dialog before baking");
                if (bakeBusy()) return;
                save();
                mBakeWorld = mDocument->path();
                mBakeOptions = {};
                mBakeOptions.full = full;
                mBakeProgress = std::make_shared<BakeProgressState>();
                mBakeOptions.cancelled = [progress = mBakeProgress] {
                    return progress->cancel.load();
                };
                mBakePreview = std::async(std::launch::async,
                    [world = mBakeWorld, data = mData, options = mBakeOptions] {
                        return preflightLod(world, data, options);
                    });
                mMessages.push_back(full ? "Preflighting full LOD bake..."
                    : "Preflighting changed LOD bake...");
            }

            void launchBake()
            {
                mBakeAwaitingConfirmation = false;
                mBakeProgress->completed.store(0);
                mBakeProgress->total.store(0);
                auto options = mBakeOptions;
                options.cancelled = [progress = mBakeProgress] { return progress->cancel.load(); };
                options.progress = [progress = mBakeProgress](const assets::LodBakeProgress& update) {
                    progress->completed.store(update.completed);
                    progress->total.store(update.total);
                };
                mBakeJob = std::async(std::launch::async,
                    [world = mBakeWorld, data = mData, options = std::move(options)] {
                        return bakeLod(world, data, options);
                    });
            }

            void pollBake()
            {
                if (mBakePreview.valid() && mBakePreview.wait_for(std::chrono::seconds(0))
                    == std::future_status::ready)
                {
                    attempt([&] {
                        mBakePreflight = mBakePreview.get();
                        if (mBakeProgress && mBakeProgress->cancel.load())
                        {
                            mBakePreflight.reset();
                            mBakeProgress.reset();
                            mMessages.push_back("LOD bake canceled before starting");
                            return;
                        }
                        const auto& count = *mBakePreflight;
                        mMessages.push_back("LOD preflight: " + std::to_string(count.terrainTiles)
                            + " terrain, " + std::to_string(count.staticTiles) + " static tiles; about "
                            + std::to_string(count.estimatedBytes / 1048576) + " MiB"
                            + (count.full ? " (full)" : " (changed)"));
                        if (count.totalTiles() > 10000)
                            mBakeAwaitingConfirmation = true;
                        else
                            launchBake();
                    });
                }
                if (mBakeJob.valid() && mBakeJob.wait_for(std::chrono::seconds(0))
                    == std::future_status::ready)
                {
                    attempt([&] {
                        auto result = mBakeJob.get();
                        mMessages.insert(mMessages.end(), result.diagnostics.begin(), result.diagnostics.end());
                        mMessages.push_back(std::string(result.cancelled ? "LOD bake canceled: "
                            : "LOD bake complete: ") + std::to_string(result.terrainBaked)
                            + " terrain, " + std::to_string(result.staticBaked) + " static tiles");
                        if (!result.publishedPaths.empty() && mDocument->path() == mBakeWorld)
                        {
                            if (mTerrain) mTerrain->refreshDerivedLod(*mRenderer,
                                result.publishedPaths);
                            mObjects->refreshDerivedLod(*mRenderer, result.publishedPaths);
                        }
                        if (mSmokeBakeStarted)
                        {
                            if (result.cancelled || mPosition != mSmokeBakeCamera
                                || mSelection != mSmokeBakeSelection
                                || (mSmokeBakeReference && (!mDocument->dirty()
                                    || mDocument->engineTransform(*mSmokeBakeReference) != mSmokeBakeTransform)))
                                throw std::runtime_error("Editor LOD refresh changed camera, selection, or unsaved edits");
                            mSmokeBakeCompleted = true;
                            std::cout << "Editor smoke LOD bake/refresh preserved camera, selection and unsaved edits\n";
                        }
                        mBakeProgress.reset();
                        mBakePreflight.reset();
                    });
                }
                if (!bakeBusy())
                {
                    mBakeProgress.reset();
                    mBakePreflight.reset();
                }
            }

            void openWorld(const std::filesystem::path& path)
            {
                if (mExportDialog) throw std::runtime_error("Close the export dialog before opening a world");
                if (mScripts.busy()) throw std::runtime_error("Wait for script compilation before opening a world");
                if (bakeBusy())
                    throw std::runtime_error("Wait for or cancel the current LOD bake before opening a world");
                if (mDocument) finishManipulation(true);
                if (mDocument && mDocument->dirty())
                    save();
                if (mImport)
                    throw std::runtime_error("Wait for the current model import before opening a world");
                if (mAsset)
                {
                    if (mAsset->dirty()) throw std::runtime_error("Save or discard the asset before opening a world");
                    mAsset->suspend(*mRenderer); mAsset.reset(); mAssetMode = false;
                }
                auto next = std::make_unique<EditorDocument>(path, mData);
                std::unique_ptr<world::LandscapeEditor> terrain;
                std::unique_ptr<world::WorldStreamer> streamer;
                world::WorldScene scene;
                if (next->hasEditableTerrain())
                {
                    terrain = std::make_unique<world::LandscapeEditor>(mData, next->path());
                    terrain->setExternalSave(true);
                    terrain->setBrushEnabled(false);
                    scene = terrain->takeInitialScene();
                }
                else
                {
                    streamer = std::make_unique<world::WorldStreamer>(mData, std::nullopt, 2,
                        false, 0, next->basePath());
                    scene = streamer->takeInitialScene();
                }
                mUi->detach();
                mRenderer.reset();
                mDocument = std::move(next);
                mTerrain = std::move(terrain);
                mStreamer = std::move(streamer);
                mObjects = std::make_unique<world::ObjectScene>(mData, mDocument->path());
                mPosition = scene.cameraPosition;
                lookAt(scene.cameraTarget);
                mRenderer = std::make_unique<render::VulkanRenderer>(mWindow.get(), scene.mesh,
                    scene.instances, scene.materials, scene.textures);
                mUi->attach(mRenderer->uiContext());
                mSelection.clear();
                mSelectionOrder.clear();
                mActiveReference.reset();
                mTerrainMode = false;
                mStroke = false;
                mGizmoDragging = false;
                mLastCell = cameraCell();
                if (mTerrain)
                {
                    mTerrain->setNextFormId(std::max(mTerrain->nextFormId(), mDocument->nextFormId()));
                    mDocument->setNextFormId(mTerrain->nextFormId());
                    mDocument->synchronizeTerrain(mTerrain->snapshot(), false);
                    mResidentTerrainCells = mTerrain->residentCells();
                }
                mMessages.push_back("Opened " + mDocument->path().string());
            }

            int smokeMenuPhase() const
            {
                if (!mSmokeTest) return 0;
                if (mFrames >= 5 && mFrames <= 7) return mFrames - 4;
                if (mFrames >= 17 && mFrames <= 19) return mFrames - 16;
                if (mSmokeReopened && mFrames >= mSmokeReopenFrame + 4
                    && mFrames <= mSmokeReopenFrame + 6) return mFrames - mSmokeReopenFrame - 3;
                return 0;
            }

            void events()
            {
                const int menuPhase = smokeMenuPhase();
                if (menuPhase == 1 || menuPhase == 2)
                {
                    SDL_WarpMouseInWindow(mWindow.get(), mFileMenuCenter.x, mFileMenuCenter.y);
                    SDL_Event click{};
                    click.type = menuPhase == 1 ? SDL_EVENT_MOUSE_BUTTON_DOWN : SDL_EVENT_MOUSE_BUTTON_UP;
                    click.button.windowID = SDL_GetWindowID(mWindow.get());
                    click.button.button = SDL_BUTTON_LEFT;
                    click.button.down = menuPhase == 1;
                    click.button.x = mFileMenuCenter.x;
                    click.button.y = mFileMenuCenter.y;
                    SDL_PushEvent(&click);
                }
                SDL_Event event;
                while (SDL_PollEvent(&event))
                {
                    ImGui_ImplSDL3_ProcessEvent(&event);
                    if (mExportDialog && event.type != SDL_EVENT_QUIT && event.type != SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED)
                        continue;
                    switch (event.type)
                    {
                    case SDL_EVENT_QUIT: guardAsset([this] { mRunning = false; }); break;
                    case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED: mRenderer->resize(); break;
                    case SDL_EVENT_WINDOW_FOCUS_LOST:
                        if (mAsset) mAsset->cancelGesture();
                        finishManipulation(true);
                        mNavigating = false;
                        SDL_SetWindowRelativeMouseMode(mWindow.get(), false);
                        break;
                    case SDL_EVENT_MOUSE_BUTTON_UP:
                        if (event.button.button == SDL_BUTTON_RIGHT)
                        {
                            mNavigating = false;
                            SDL_SetWindowRelativeMouseMode(mWindow.get(), false);
                        }
                        break;
                    case SDL_EVENT_MOUSE_MOTION:
                        if (mNavigating)
                        {
                            mYaw -= event.motion.xrel * 0.003f;
                            mPitch = std::clamp(mPitch - event.motion.yrel * 0.003f, -1.55f, 1.55f);
                        }
                        break;
                    case SDL_EVENT_MOUSE_WHEEL:
                        if (mAssetMode) break;
                        if (mViewportHovered && mTerrainMode && mTerrain)
                        {
                            if (SDL_GetModState() & SDL_KMOD_SHIFT)
                                mTerrain->adjustStrength(event.wheel.y);
                            else
                                mTerrain->adjustRadius(event.wheel.y);
                        }
                        else if (mViewportHovered)
                            mMoveSpeed = std::clamp(mMoveSpeed * std::pow(1.25f, event.wheel.y), 10.0f, 1.0e6f);
                        break;
                    case SDL_EVENT_DROP_FILE:
                        if (event.drop.data && !mAssetMode)
                            attempt([&] { startImport(std::filesystem::u8path(event.drop.data)); });
                        break;
                    default: break;
                    }
                }
            }

            void showDialog(FileAction action)
            {
                if (mExportDialog && action != FileAction::exportFolder)
                { mMessages.push_back("Close the export dialog before changing files"); return; }
                if (mScripts.busy() && action != FileAction::import)
                { mMessages.push_back("Wait for script compilation before changing worlds"); return; }
                if (mAsset && mAsset->dirty() && action != FileAction::import)
                { guardAsset([this, action] { showDialog(action); }); return; }
                if (bakeBusy() && action != FileAction::import)
                {
                    mMessages.push_back("Finish or cancel the LOD bake before opening or saving another world");
                    return;
                }
                finishManipulation(true);
                std::scoped_lock lock(mDialogMailbox->mutex);
                if (mDialogMailbox->pending || mImport)
                    return;
                mDialogMailbox->pending = true;
                auto* request = new DialogRequest{mDialogMailbox, action};
                static constexpr SDL_DialogFileFilter esmFilter[]{ {"Bethesda master", "esm"} };
                static constexpr SDL_DialogFileFilter modelFilter[]{
                    {"3D model", "glb;gltf;obj;fbx;dae;ply;stl;blend;3ds"}, {"All files", "*"} };
                if (action == FileAction::exportFolder)
                    SDL_ShowOpenFolderDialog(fileDialogResult, request, mWindow.get(), nullptr, false);
                else if (action == FileAction::saveAs || action == FileAction::create)
                    SDL_ShowSaveFileDialog(fileDialogResult, request, mWindow.get(), esmFilter, 1,
                        nullptr);
                else
                    SDL_ShowOpenFileDialog(fileDialogResult, request, mWindow.get(),
                        action == FileAction::import ? modelFilter : esmFilter,
                        action == FileAction::import ? 2 : 1, nullptr, false);
            }

            void handleDialogs()
            {
                std::deque<DialogResult> results;
                {
                    std::scoped_lock lock(mDialogMailbox->mutex);
                    results.swap(mDialogMailbox->results);
                }
                for (const auto& result : results)
                    attempt([&] {
                        if (!result.error.empty())
                            throw std::runtime_error(result.error);
                        if (result.action == FileAction::exportFolder)
                        {
                            if (mExportDialog && !exportBusy())
                                setExportDestination(result.path / std::filesystem::u8path(mExportDestination.data()).filename());
                            return;
                        }
                        if (mExportDialog) throw std::runtime_error("Close the export dialog before changing files");
                        if (mScripts.busy() && result.action != FileAction::import)
                            throw std::runtime_error("Wait for script compilation before changing worlds");
                        if (bakeBusy() && result.action != FileAction::import)
                            throw std::runtime_error("Finish or cancel the LOD bake before opening or saving another world");
                        if (result.action == FileAction::import)
                            startImport(result.path);
                        else if (result.action == FileAction::saveAs)
                        {
                            finishStroke();
                            synchronizeTerrain();
                            mDocument->saveAs(result.path);
                            openWorld(mDocument->path());
                        }
                        else
                        {
                            if (result.action == FileAction::create && std::filesystem::exists(result.path))
                                throw std::runtime_error("New world path already exists; use Open instead");
                            openWorld(result.path);
                        }
                    });
            }

            void shell()
            {
                ImGuiViewport* main = ImGui::GetMainViewport();
                ImGui::SetNextWindowPos(main->WorkPos);
                ImGui::SetNextWindowSize(main->WorkSize);
                ImGui::SetNextWindowViewport(main->ID);
                ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
                ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
                ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
                constexpr ImGuiWindowFlags flags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoDocking
                    | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize
                    | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus
                    | ImGuiWindowFlags_NoNavFocus | ImGuiWindowFlags_NoBackground;
                ImGui::Begin("OpenF4 workspace", nullptr, flags);
                ImGui::PopStyleVar(3);
                if (ImGui::BeginMenuBar())
                {
                    const auto menuPosition = ImGui::GetCursorScreenPos();
                    const auto menuSize = ImGui::CalcTextSize("File");
                    mFileMenuCenter = {menuPosition.x + menuSize.x * 0.5f,
                        menuPosition.y + menuSize.y * 0.5f};
                    const bool fileMenuOpen = ImGui::BeginMenu("File");
                    if (smokeMenuPhase() == 3)
                    {
                        if (!fileMenuOpen)
                            recordError(std::runtime_error("Editor smoke SDL menu click did not open File"));
                        else
                            std::cout << "Editor smoke SDL menu click verified (" << ++mSmokeMenuChecks << "/3)\n";
                    }
                    if (fileMenuOpen)
                    {
                        if (ImGui::MenuItem("New world...", "Ctrl+N")) showDialog(FileAction::create);
                        if (ImGui::MenuItem("Open world...", "Ctrl+O")) showDialog(FileAction::open);
                        if (ImGui::MenuItem("Save World", mAssetMode ? nullptr : "Ctrl+S")) attempt([&] { save(); });
                        if (mAsset && ImGui::MenuItem("Save Native NIF", mAssetMode ? "Ctrl+S" : nullptr)) attempt([&] { mAsset->save(); });
                        if (ImGui::MenuItem("Save as...")) showDialog(FileAction::saveAs);
                        if (ImGui::MenuItem("Export to Fallout 4 (.esp)...")) attempt([&] { requestExport(); });
                        ImGui::Separator();
                        if (ImGui::MenuItem("Import model...")) showDialog(FileAction::import);
                        ImGui::Separator();
                        if (ImGui::MenuItem("Exit")) guardAsset([this] { mRunning = false; });
                        if (smokeMenuPhase() == 3) ImGui::CloseCurrentPopup();
                        ImGui::EndMenu();
                    }
                    if (ImGui::BeginMenu("Edit"))
                    {
                        if (ImGui::MenuItem("Undo", "Ctrl+Z")) { if(mAssetMode) mAsset->history(false); else history(false); }
                        if (ImGui::MenuItem("Redo", "Ctrl+Y")) { if(mAssetMode) mAsset->history(true); else history(true); }
                        ImGui::Separator();
                        if (ImGui::MenuItem("Duplicate", "Ctrl+D", false, !mAssetMode && !mSelection.empty())) duplicate();
                        if (ImGui::MenuItem("Delete", "Delete", false, !mAssetMode && !mSelection.empty())) erase();
                        ImGui::EndMenu();
                    }
                    if (ImGui::BeginMenu("Build"))
                    {
                        ImGui::BeginDisabled(bakeBusy() || mAssetMode);
                        if (ImGui::MenuItem("Bake Changed LOD"))
                            attempt([&] { requestBake(false); });
                        if (ImGui::MenuItem("Bake All LOD"))
                            attempt([&] { requestBake(true); });
                        ImGui::EndDisabled();
                        if (bakeBusy() && ImGui::MenuItem("Cancel Bake"))
                        {
                            if (mBakeProgress) mBakeProgress->cancel.store(true);
                            if (mBakeAwaitingConfirmation)
                            {
                                mBakeAwaitingConfirmation = false;
                                mBakePreflight.reset();
                                mBakeProgress.reset();
                            }
                        }
                        ImGui::EndMenu();
                    }
                    ImGui::Separator();
                    if (ImGui::MenuItem("World", nullptr, !mAssetMode)) switchWorkspace(false);
                    if (ImGui::MenuItem("Asset", nullptr, mAssetMode, mAsset != nullptr || selectedReference() != nullptr)) activateAssetWorkspace();
                    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
                        ImGui::SetTooltip("Open the selected object's asset, or return to the open asset.\nSelect an object first, or right-click a model in Asset Browser > Edit Asset.");
                    ImGui::BeginDisabled(mAssetMode);
                    if (ImGui::MenuItem("Objects", nullptr, !mTerrainMode))
                    { finishManipulation(true); finishStroke(); mTerrainMode = false; }
                    if (ImGui::MenuItem("Landscape", nullptr, mTerrainMode, mTerrain != nullptr))
                    { finishManipulation(true); mTerrainMode = true; }
                    ImGui::EndDisabled();
                    ImGui::Separator();
                    ImGui::TextUnformatted(mDocument->path().filename().string().c_str());
                    if (mDocument->dirty()) { ImGui::SameLine(); ImGui::TextUnformatted("*"); }
                    if (mImport) { ImGui::SameLine(); ImGui::TextUnformatted("Importing..."); }
                    ImGui::EndMenuBar();
                }
                if (mAssetPending) ImGui::OpenPopup("Unsaved asset");
                if (ImGui::BeginPopupModal("Unsaved asset", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
                {
                    ImGui::TextUnformatted("Save native asset changes before continuing?\nThe world assignment remains dirty until Save World.");
                    if (ImGui::Button("Save")) attempt([&] {
                        mAsset->save(); auto action = std::move(mAssetPending); mAssetPending = {};
                        ImGui::CloseCurrentPopup(); action();
                    });
                    ImGui::SameLine();
                    if (ImGui::Button("Discard"))
                    {
                        mAsset->suspend(*mRenderer); mAsset.reset(); mAssetMode = false;
                        auto action = std::move(mAssetPending); mAssetPending = {};
                        ImGui::CloseCurrentPopup(); attempt(action);
                    }
                    ImGui::SameLine();
                    if (ImGui::Button("Cancel")) { mAssetPending = {}; ImGui::CloseCurrentPopup(); }
                    ImGui::EndPopup();
                }
                if (mBakeAwaitingConfirmation)
                    ImGui::OpenPopup("Confirm large LOD bake");
                if (ImGui::BeginPopupModal("Confirm large LOD bake", nullptr,
                        ImGuiWindowFlags_AlwaysAutoResize))
                {
                    if (mBakePreflight)
                        ImGui::Text("Bake %llu tiles (about %.1f GiB output)?",
                            static_cast<unsigned long long>(mBakePreflight->totalTiles()),
                            mBakePreflight->estimatedBytes / 1073741824.0);
                    if (ImGui::Button("Bake"))
                    {
                        launchBake();
                        ImGui::CloseCurrentPopup();
                    }
                    ImGui::SameLine();
                    if (ImGui::Button("Cancel"))
                    {
                        mBakeAwaitingConfirmation = false;
                        mBakePreflight.reset();
                        mBakeProgress.reset();
                        ImGui::CloseCurrentPopup();
                    }
                    ImGui::EndPopup();
                }
                exportPanel();
                const ImGuiID dock = ImGui::GetID("EditorDockspace");
                if (!ImGui::DockBuilderGetNode(dock))
                {
                    ImGui::DockBuilderAddNode(dock, ImGuiDockNodeFlags_DockSpace);
                    ImGui::DockBuilderSetNodeSize(dock, main->WorkSize);
                    ImGuiID center = dock;
                    ImGuiID left = ImGui::DockBuilderSplitNode(center, ImGuiDir_Left, 0.17f, nullptr, &center);
                    ImGuiID right = ImGui::DockBuilderSplitNode(center, ImGuiDir_Right, 0.24f, nullptr, &center);
                    ImGuiID bottom = ImGui::DockBuilderSplitNode(center, ImGuiDir_Down, 0.21f, nullptr, &center);
                    ImGui::DockBuilderDockWindow("Outliner", left);
                    ImGui::DockBuilderDockWindow("Inspector", right);
                    ImGui::DockBuilderDockWindow("Asset Browser", bottom);
                    ImGui::DockBuilderDockWindow("Diagnostics", bottom);
                    ImGui::DockBuilderDockWindow("Viewport", center);
                    ImGui::DockBuilderFinish(dock);
                }
                // PassthruCentralNode paints an opaque backdrop when the central node
                // contains a window. Our docked Viewport is transparent itself, so the
                // dock host must not add that backdrop over the world render.
                ImGui::DockSpace(dock);
                ImGui::End();
            }

            glm::vec3 forward() const
            {
                return {std::cos(mPitch) * std::sin(mYaw), std::sin(mPitch),
                    std::cos(mPitch) * std::cos(mYaw)};
            }

            void lookAt(glm::dvec3 target)
            {
                const glm::dvec3 direction = glm::normalize(target - mPosition);
                mPitch = static_cast<float>(std::asin(std::clamp(direction.y, -1.0, 1.0)));
                mYaw = static_cast<float>(std::atan2(direction.x, direction.z));
            }

            assets::CellCoord cameraCell() const
            {
                return {static_cast<std::int32_t>(std::floor(-mPosition.x / cellSize)),
                    static_cast<std::int32_t>(std::floor(mPosition.z / cellSize))};
            }

            void navigate(float delta)
            {
                if (!mNavigating || ImGui::GetIO().WantTextInput)
                    return;
                const bool* keys = SDL_GetKeyboardState(nullptr);
                const glm::vec3 direction = forward();
                const glm::vec3 right = glm::normalize(glm::cross(direction, glm::vec3(0, 1, 0)));
                const double speed = mMoveSpeed * delta * (keys[SDL_SCANCODE_LSHIFT] ? 4.0 : 1.0);
                mPosition += glm::dvec3(direction) * speed
                    * static_cast<double>(keys[SDL_SCANCODE_W] - keys[SDL_SCANCODE_S]);
                mPosition += glm::dvec3(right) * speed
                    * static_cast<double>(keys[SDL_SCANCODE_D] - keys[SDL_SCANCODE_A]);
                mPosition.y += speed * static_cast<double>(keys[SDL_SCANCODE_E] - keys[SDL_SCANCODE_Q]);
            }

            glm::vec3 cursorRay() const
            {
                const ImVec2 mouse = ImGui::GetIO().MousePos;
                const float aspect = std::max(mViewportSize.x, 1.0f) / std::max(mViewportSize.y, 1.0f);
                const float x = ((mouse.x - mViewportPosition.x) / std::max(mViewportSize.x, 1.0f) * 2.0f - 1.0f)
                    * std::tan(glm::radians(35.0f)) * aspect;
                const float y = (1.0f - (mouse.y - mViewportPosition.y) / std::max(mViewportSize.y, 1.0f) * 2.0f)
                    * std::tan(glm::radians(35.0f));
                const glm::vec3 direction = forward();
                const glm::vec3 right = glm::normalize(glm::cross(direction, glm::vec3(0, 1, 0)));
                return glm::normalize(direction + x * right + y * glm::cross(right, direction));
            }

            glm::mat4 placementTransform() const
            {
                // Raycast already returns engine coordinates. Only the camera-space
                // fallback needs to undo the renderer's X reflection.
                if (mTerrain)
                    if (auto hit = mTerrain->raycast(mPosition, mViewportHovered ? cursorRay() : forward()))
                        return glm::translate(glm::mat4(1.0f), *hit);
                glm::vec3 point(mPosition + glm::dvec3(forward()) * 1024.0);
                point.x = -point.x;
                return glm::translate(glm::mat4(1.0f), point);
            }

            void placementToolbar()
            {
                ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8, 4));
                ImGui::BeginChild("Placement controls", ImVec2(0, ImGui::GetFrameHeightWithSpacing() * 2 + 8),
                    ImGuiChildFlags_AlwaysUseWindowPadding);
                ImGui::BeginDisabled(mTerrainMode || mGizmoDragging || mInspectorEditing);
                if (ImGui::RadioButton("Move", mOperation == ImGuizmo::TRANSLATE)) mOperation = ImGuizmo::TRANSLATE;
                ImGui::SameLine();
                if (ImGui::RadioButton("Rotate", mOperation == ImGuizmo::ROTATE)) mOperation = ImGuizmo::ROTATE;
                ImGui::SameLine();
                if (ImGui::RadioButton("Scale", mOperation == ImGuizmo::SCALEU)) mOperation = ImGuizmo::SCALEU;
                ImGui::SameLine();
                ImGui::SetNextItemWidth(85);
                int space = mManipulationSpace == ManipulationSpace::world ? 0 : 1;
                if (ImGui::Combo("##Orientation", &space, "World\0Local\0"))
                    mManipulationSpace = space == 0 ? ManipulationSpace::world : ManipulationSpace::local;
                ImGui::SameLine();
                ImGui::SetNextItemWidth(145);
                int pivot = mCenterPivot ? 1 : 0;
                if (ImGui::Combo("##Pivot", &pivot, "Active object\0Selection center\0")) mCenterPivot = pivot == 1;
                ImGui::SameLine();
                ImGui::BeginDisabled(mTerrain == nullptr || editableSelection().empty());
                if (ImGui::Button("Drop to Ground")) dropToGround();
                ImGui::EndDisabled();
                ImGui::Checkbox("Move snap", &mSnapTranslation);
                ImGui::SameLine(); ImGui::SetNextItemWidth(80);
                ImGui::InputFloat("##Move step", &mTranslationStep, 0, 0, "%.1f");
                if (!std::isfinite(mTranslationStep) || mTranslationStep <= 0) mTranslationStep = 128;
                ImGui::SameLine(); ImGui::Checkbox("Rotate snap", &mSnapRotation);
                ImGui::SameLine(); ImGui::SetNextItemWidth(65);
                ImGui::InputFloat("##Rotation step", &mRotationStep, 0, 0, "%.1f");
                if (!std::isfinite(mRotationStep) || mRotationStep <= 0) mRotationStep = 15;
                ImGui::SameLine(); ImGui::Checkbox("Align to terrain", &mAlignTerrain);
                ImGui::EndDisabled();
                ImGui::EndChild();
                ImGui::PopStyleVar();
            }

            void dropToGround()
            {
                if (mGizmoDragging || mInspectorEditing || mTerrainMode) return;
                if (!mTerrain) { mMessages.push_back("Drop to Ground requires editable LAND."); return; }
                // Resolve every query before opening a transaction: missing streamed
                // geometry is a skipped reference, never a request to create LAND.
                std::vector<std::pair<std::uint32_t, glm::mat4>> changes;
                for (auto id : editableSelection())
                {
                    auto transform = mDocument->engineTransform(id);
                    const auto surface = mTerrain->surfaceAt({transform[3].x, transform[3].z});
                    if (!surface)
                    {
                        mMessages.push_back("Drop to Ground skipped " + std::to_string(id) + ": LAND is not loaded.");
                        continue;
                    }
                    if (mAlignTerrain) transform = alignToTerrainNormal(transform, surface->normal);
                    const auto bounds = mObjects->geometryBounds(id, transform);
                    if (!bounds)
                    {
                        mMessages.push_back("Drop to Ground skipped " + std::to_string(id) + ": model is not loaded.");
                        continue;
                    }
                    transform[3].y += static_cast<float>(surface->height - bounds->minimum.y);
                    changes.emplace_back(id, transform);
                }
                if (changes.empty()) return;
                synchronizeTerrain();
                mDocument->beginTransaction();
                try
                {
                    for (const auto& [id, transform] : changes) mDocument->setTransform(id, transform);
                    mDocument->endTransaction();
                    restoreTerrain();
                }
                catch (const std::exception& error) { mDocument->cancelTransaction(); restoreTerrain(); recordError(error); }
            }

            void viewport(float delta)
            {
                ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
                ImGui::Begin("Viewport", nullptr, ImGuiWindowFlags_NoBackground
                    | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
                placementToolbar();
                mViewportPosition = ImGui::GetCursorScreenPos();
                mViewportSize = ImGui::GetContentRegionAvail();
                mViewportSize.x = std::max(mViewportSize.x, 1.0f);
                mViewportSize.y = std::max(mViewportSize.y, 1.0f);
                // Reserve the viewport without claiming an active/hovered item ID:
                // ImGuizmo deliberately refuses drags while another widget owns it.
                ImGui::Dummy(mViewportSize);
                mViewportHovered = !mExportDialog && ImGui::IsItemHovered();
                mViewportFocused = !mExportDialog && ImGui::IsWindowFocused();
                if (mExportDialog)
                {
                    // ImGuizmo/raw mouse handlers do not honor ImGui's disabled widget flag.
                    ImGui::End(); ImGui::PopStyleVar(); return;
                }
                // Decide after ImGui has processed this frame's popup/capture state,
                // not from the previous frame's hover position in the SDL event loop.
                if (mViewportHovered && !mGizmoDragging && !mInspectorEditing && !ImGui::GetIO().WantTextInput
                    && ImGui::IsMouseClicked(ImGuiMouseButton_Right))
                {
                    mNavigating = true;
                    SDL_SetWindowRelativeMouseMode(mWindow.get(), true);
                }
                if (ImGui::BeginDragDropTarget())
                {
                    if (const auto* payload = ImGui::AcceptDragDropPayload("OPENF4_MODEL"))
                    {
                        const auto model = *static_cast<const std::uint32_t*>(payload->Data);
                        attempt([&] { placeModel(model); });
                    }
                    ImGui::EndDragDropTarget();
                }
                // Keep streamed terrain/its allocator stable during a gesture. The
                // camera is stationary, and publication resumes on commit/cancel.
                if (mTerrain && !mGizmoDragging && !mInspectorEditing)
                {
                    mTerrain->setBrushEnabled(mTerrainMode && mViewportHovered && !mNavigating);
                    mTerrain->setNextFormId(std::max(mTerrain->nextFormId(), mDocument->nextFormId()));
                    const bool sculpt = mTerrainMode && mViewportHovered && !mNavigating
                        && ImGui::IsMouseDown(ImGuiMouseButton_Left);
                    if (sculpt && !mStroke)
                    {
                        synchronizeTerrain();
                        mDocument->beginTransaction();
                        mStroke = true;
                    }
                    mTerrain->update(*mRenderer, mPosition,
                        mTerrainMode && mViewportHovered ? cursorRay() : forward(), delta, sculpt);
                    mDocument->setNextFormId(std::max(mDocument->nextFormId(), mTerrain->nextFormId()));
                    if (!sculpt && mStroke)
                        finishStroke();
                    if (!sculpt && mResidentTerrainCells != mTerrain->residentCells())
                    {
                        mDocument->synchronizeTerrain(mTerrain->snapshot(), false);
                        mResidentTerrainCells = mTerrain->residentCells();
                    }
                    if (mLastCell != cameraCell())
                    {
                        synchronizeTerrain();
                        mLastCell = cameraCell();
                    }
                }
                if (mStreamer)
                {
                    mStreamer->update(*mRenderer, mPosition);
                    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(2);
                    while (mStreamer->streamNext(*mRenderer)
                        && std::chrono::steady_clock::now() < deadline) {}
                }
                if (!mTerrainMode)
                {
                    gizmo();
                    if (mViewportHovered && !mNavigating && !mSuppressGizmo && !ImGuizmo::IsOver()
                        && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
                        pick();
                }
                const std::string overlay = mTerrainMode && mTerrain ? mTerrain->status()
                    : "RMB + WASD to fly  |  Drop a model to place  |  F frame selection";
                ImGui::GetWindowDrawList()->AddText(ImVec2(mViewportPosition.x + 12, mViewportPosition.y + 10),
                    IM_COL32(235, 240, 245, 255), overlay.c_str());
                ImGui::End();
                ImGui::PopStyleVar();
            }

            void pick()
            {
                glm::dvec3 origin = mPosition;
                glm::dvec3 direction(cursorRay());
                origin.x = -origin.x;
                direction.x = -direction.x;
                double limit = std::numeric_limits<double>::infinity();
                if (mTerrain)
                    if (auto terrainHit = mTerrain->raycast(mPosition, cursorRay()))
                        limit = glm::length(glm::dvec3(*terrainHit) - origin) + 0.01;
                std::vector<std::uint32_t> excluded;
                for (const auto& reference : mDocument->references())
                    if (!reference.visible || reference.locked) excluded.push_back(reference.placement.formId);
                const auto hit = mObjects->pick(origin, glm::normalize(direction), excluded, limit);
                select(hit ? std::optional(hit->formId) : std::nullopt, ImGui::GetIO().KeyShift);
            }

            const EditorReference* reference(std::uint32_t id) const
            {
                const auto& references = mDocument->references();
                const auto it = std::find_if(references.begin(), references.end(),
                    [id](const EditorReference& ref) { return ref.placement.formId == id; });
                return it == references.end() ? nullptr : &*it;
            }

            const EditorReference* selectedReference() const
            {
                return mActiveReference ? reference(*mActiveReference) : nullptr;
            }

            void reconcileSelection()
            {
                std::erase_if(mSelection, [&](auto id) { return reference(id) == nullptr; });
                std::erase_if(mSelectionOrder, [&](auto id) { return !mSelection.contains(id); });
                for (auto id : mSelection)
                    if (std::find(mSelectionOrder.begin(), mSelectionOrder.end(), id) == mSelectionOrder.end())
                        mSelectionOrder.push_back(id);
                if (!mActiveReference || !mSelection.contains(*mActiveReference))
                    mActiveReference = mSelectionOrder.empty() ? std::nullopt : std::optional(mSelectionOrder.back());
            }

            void select(std::optional<std::uint32_t> id, bool toggle = false)
            {
                if (mGizmoDragging || mInspectorEditing) return;
                if (!toggle) { mSelection.clear(); mSelectionOrder.clear(); mActiveReference.reset(); }
                if (id)
                {
                    if (toggle && mSelection.contains(*id)) mSelection.erase(*id);
                    else
                    {
                        mSelection.insert(*id);
                        std::erase(mSelectionOrder, *id);
                        mSelectionOrder.push_back(*id);
                        mActiveReference = *id;
                    }
                }
                reconcileSelection();
            }

            std::vector<std::uint32_t> editableSelection() const
            {
                std::vector<std::uint32_t> result;
                for (auto id : mSelection)
                    if (auto ref = reference(id); ref && ref->visible && !ref->locked) result.push_back(id);
                return result;
            }

            glm::mat4 selectionPivot(const std::vector<std::uint32_t>& ids) const
            {
                const auto active = mActiveReference
                    && std::find(ids.begin(), ids.end(), *mActiveReference) != ids.end()
                    ? *mActiveReference : ids.back();
                const auto transform = mDocument->engineTransform(active);
                glm::vec3 center(transform[3]);
                if (mCenterPivot)
                {
                    glm::dvec3 minimum(std::numeric_limits<double>::max()), maximum(-minimum);
                    for (auto id : ids)
                    {
                        const auto bounds = mObjects->geometryBounds(id, mDocument->engineTransform(id));
                        const glm::dvec3 position(mDocument->engineTransform(id)[3]);
                        minimum = glm::min(minimum, bounds ? bounds->minimum : position);
                        maximum = glm::max(maximum, bounds ? bounds->maximum : position);
                    }
                    center = glm::vec3((minimum + maximum) * 0.5);
                }
                return makeManipulationPivot(center, transform, mManipulationSpace);
            }

            void finishManipulation(bool cancel)
            {
                if (!mGizmoDragging && !mInspectorEditing) return;
                if (cancel) mDocument->cancelTransaction();
                else mDocument->endTransaction();
                mSuppressGizmo = mGizmoDragging;
                mGizmoDragging = mInspectorEditing = mEndInspectorTransaction = false;
                mInspectorItem = 0;
                mDragTransforms.clear();
                restoreTerrain();
            }

            void gizmo()
            {
                if (ImGui::IsKeyPressed(ImGuiKey_Escape)) finishManipulation(true);
                if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)) mSuppressGizmo = false;
                ImGuizmo::Enable(!mSuppressGizmo && !mInspectorEditing);
                const auto ids = editableSelection();
                if (ids.empty() || mSuppressGizmo || mInspectorEditing) return;
                const glm::dvec3 origin = renderOrigin(mPosition);
                const glm::vec3 local(mPosition - origin);
                const auto view = glm::lookAtRH(local, local + forward(), glm::vec3(0, 1, 0));
                const auto projection = glm::perspectiveRH_NO(glm::radians(70.0f),
                    mViewportSize.x / mViewportSize.y, 0.1f, 10000000.0f);
                const glm::mat4 mirror = reflection();
                const glm::mat4 before = mGizmoDragging ? mDragPivot : selectionPivot(ids);
                glm::mat4 matrix = mirror * before * mirror;
                matrix[3] -= glm::vec4(glm::vec3(origin), 0.0f);
                ImGuizmo::SetDrawlist();
                ImGuizmo::SetOrthographic(false);
                ImGuizmo::SetRect(mViewportPosition.x, mViewportPosition.y, mViewportSize.x, mViewportSize.y);
                const bool changed = ImGuizmo::Manipulate(glm::value_ptr(view), glm::value_ptr(projection),
                    mOperation, mManipulationSpace == ManipulationSpace::world ? ImGuizmo::WORLD : ImGuizmo::LOCAL,
                    glm::value_ptr(matrix));
                if (ImGuizmo::IsUsing() && !mGizmoDragging)
                {
                    synchronizeTerrain();
                    mDocument->beginTransaction();
                    mGizmoDragging = true;
                    mStartPivot = before;
                    mDragPivot = before;
                    mDragTransforms.clear();
                    for (auto id : ids) mDragTransforms.emplace_back(id, mDocument->engineTransform(id));
                }
                if (changed && mGizmoDragging)
                {
                    try
                    {
                        matrix[3] += glm::vec4(glm::vec3(origin), 0.0f);
                        mDragPivot = mirror * matrix * mirror;
                        const auto operation = mOperation == ImGuizmo::TRANSLATE ? ManipulationOperation::translate
                            : mOperation == ImGuizmo::ROTATE ? ManipulationOperation::rotate : ManipulationOperation::scale;
                        const float snap = operation == ManipulationOperation::translate && mSnapTranslation ? mTranslationStep
                            : operation == ManipulationOperation::rotate && mSnapRotation ? mRotationStep : 0.0f;
                        const auto after = snapManipulationPivot(mStartPivot, mDragPivot, operation, mManipulationSpace, snap);
                        for (const auto& [id, transform] : mDragTransforms)
                            mDocument->setTransform(id, applyManipulationDelta(transform, mStartPivot, after));
                    }
                    catch (const std::exception& error) { finishManipulation(true); recordError(error); }
                }
                if (!ImGuizmo::IsUsing() && mGizmoDragging)
                    finishManipulation(false);
            }

            void inspector()
            {
                ImGui::Begin("Inspector");
                if (mTerrainMode && mTerrain)
                {
                    ImGui::TextUnformatted("Landscape sculpting");
                    static constexpr const char* tools[]{"Raise", "Lower", "Smooth", "Flatten"};
                    for (int i = 0; i < 4; ++i)
                        if (ImGui::RadioButton(tools[i], mLandscapeTool == i))
                        {
                            mLandscapeTool = i;
                            mTerrain->setTool(static_cast<world::LandscapeTool>(i));
                        }
                    ImGui::Separator();
                    ImGui::TextWrapped("Left-drag to sculpt. Wheel changes radius; Shift + wheel changes strength. "
                        "Each stroke is one undo transaction.");
                    ImGui::TextWrapped("%s", mTerrain->status().c_str());
                }
                else if (const auto* reference = selectedReference())
                {
                    const auto id = reference->placement.formId;
                    const auto baseObject = reference->placement.baseObject;
                    const bool locked = reference->locked;
                    ImGui::Text("%s", reference->name.empty() ? "Placed object" : reference->name.c_str());
                    const auto* base = mDocument->model(reference->placement.baseObject);
                    ImGui::Text("REFR %08X  /  %s %08X", id,
                        base && base->kind == assets::ESMModelKind::movableStatic ? "MSTT" : "STAT", reference->placement.baseObject);
                    ImGui::Text("Cell (%d, %d)", reference->placement.cell.x, reference->placement.cell.y);
                    if (mSelection.size() > 1)
                        ImGui::Text("%zu selected; gizmo moves the selection", mSelection.size());
                    bool visible = reference->visible;
                    bool lockedValue = locked;
                    ImGui::BeginDisabled(mGizmoDragging || mInspectorEditing);
                    if (ImGui::Checkbox("Visible", &visible)) mDocument->setVisible(id, visible);
                    ImGui::SameLine();
                    if (ImGui::Checkbox("Locked", &lockedValue)) mDocument->setLocked(id, lockedValue);
                    ImGui::EndDisabled();
                    ImGui::Separator();
                    ImGui::BeginDisabled(mGizmoDragging || mInspectorEditing);
                    if (ImGui::RadioButton("Move", mOperation == ImGuizmo::TRANSLATE)) mOperation = ImGuizmo::TRANSLATE;
                    ImGui::SameLine();
                    if (ImGui::RadioButton("Rotate", mOperation == ImGuizmo::ROTATE)) mOperation = ImGuizmo::ROTATE;
                    ImGui::SameLine();
                    if (ImGui::RadioButton("Scale", mOperation == ImGuizmo::SCALEU)) mOperation = ImGuizmo::SCALEU;
                    ImGui::EndDisabled();
                    float translation[3], rotation[3], scale[3];
                    glm::mat4 transform = mDocument->engineTransform(id);
                    ImGuizmo::DecomposeMatrixToComponents(glm::value_ptr(transform), translation, rotation, scale);
                    ImGui::BeginDisabled(locked || !visible || mGizmoDragging);
                    bool changed = false;
                    const auto trackEdit = [&] {
                        const auto item = ImGui::GetItemID();
                        if (ImGui::IsItemActivated())
                        {
                            if (mInspectorEditing && mInspectorItem != item) finishManipulation(false);
                            if (!mInspectorEditing)
                            {
                                synchronizeTerrain();
                                mDocument->beginTransaction();
                                mInspectorEditing = true;
                                mInspectorItem = item;
                            }
                        }
                        if (mInspectorItem == item && ImGui::IsItemDeactivated()) mEndInspectorTransaction = true;
                    };
                    auto component = [&](const char* label, float* values, float speed) {
                        const bool edited = ImGui::DragFloat3(label, values, speed);
                        trackEdit();
                        return edited;
                    };
                    changed |= component("Position", translation, 4.0f);
                    changed |= component("Rotation", rotation, 0.5f);
                    float uniformScale = scale[0];
                    changed |= ImGui::DragFloat("Uniform scale", &uniformScale, 0.01f, 0.001f, 10000.0f);
                    trackEdit();
                    if (changed)
                        attempt([&] {
                            scale[0] = scale[1] = scale[2] = std::max(uniformScale, 0.001f);
                            ImGuizmo::RecomposeMatrixFromComponents(translation, rotation, scale, glm::value_ptr(transform));
                            mDocument->setTransform(id, transform);
                        });
                    if (ImGui::IsKeyPressed(ImGuiKey_Escape)) finishManipulation(true);
                    else if (mEndInspectorTransaction) finishManipulation(false);
                    ImGui::EndDisabled();
                    if (ImGui::Button("Frame selected")) frameSelection();
                    ImGui::SameLine();
                    if (ImGui::Button("Duplicate")) duplicate();
                    if (ImGui::Button("Edit Asset")) requestAsset(baseObject);
                    ImGui::SameLine();
                    if (ImGui::Button("Make Unique")) mNextFrameAction = [this, id] { guardAsset([this, id] {
                        const auto found = std::ranges::find_if(mDocument->references(), [id](const auto& ref) { return ref.placement.formId == id; });
                        if (found == mDocument->references().end()) throw std::runtime_error("Selected reference no longer exists");
                        auto next = std::make_unique<AssetWorkspace>(*mDocument, found->placement.baseObject, mData);
                        next->makeUnique(id);
                        if (mAsset) mAsset->suspend(*mRenderer);
                        mAsset = std::move(next); mAssetMode = true;
                    }); };
                    if (auto source = mDocument->sourcePath(baseObject))
                        ImGui::TextWrapped("Source: %s", source->c_str());
                    ImGui::BeginDisabled(mGizmoDragging || mInspectorEditing || locked);
                    mScripts.draw(*mDocument, id, mData);
                    ImGui::EndDisabled();
                }
                else
                    ImGui::TextWrapped("Select an object in the viewport or outliner. Drop a GLB file into the viewport to import and place it.");
                ImGui::End();
            }

            void outliner()
            {
                ImGui::Begin("Outliner");
                ImGui::Text("%zu references", mDocument->references().size());
                static ImGuiTextFilter filter;
                filter.Draw("Filter");
                for (const auto& reference : mDocument->references())
                {
                    const auto id = reference.placement.formId;
                    const std::string label = (reference.name.empty() ? "Object " + std::to_string(id) : reference.name)
                        + (mActiveReference == id ? " [active]" : "")
                        + (reference.visible ? "" : " [hidden]") + (reference.locked ? " [locked]" : "");
                    if (!filter.PassFilter(label.c_str())) continue;
                    ImGui::PushID(static_cast<int>(id));
                    if (ImGui::Selectable(label.c_str(), mSelection.contains(id), ImGuiSelectableFlags_AllowDoubleClick))
                    {
                        select(id, ImGui::GetIO().KeyShift);
                        if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) frameSelection();
                    }
                    ImGui::PopID();
                }
                ImGui::End();
            }

            void assetsPanel()
            {
                ImGui::Begin("Asset Browser");
                ImGui::BeginDisabled(mImport.has_value() || mAssetMode);
                if (ImGui::Button("Import model...")) showDialog(FileAction::import);
                ImGui::EndDisabled();
                ImGui::SameLine();
                ImGui::TextDisabled("%s", mDocument->assetRoot().string().c_str());
                ImGui::Separator();
                for (const auto& [id, path] : mDocument->models())
                {
                    ImGui::PushID(static_cast<int>(id));
                    if (ImGui::Selectable(path.c_str(), false, ImGuiSelectableFlags_AllowDoubleClick)
                        && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left))
                        if (!mAssetMode) attempt([&] { placeModel(id); });
                    if (ImGui::BeginPopupContextItem())
                    {
                        if (ImGui::MenuItem("Edit Asset")) requestAsset(id);
                        ImGui::EndPopup();
                    }
                    if (ImGui::BeginDragDropSource())
                    {
                        ImGui::SetDragDropPayload("OPENF4_MODEL", &id, sizeof(id));
                        ImGui::Text("Place %s", path.c_str());
                        ImGui::EndDragDropSource();
                    }
                    ImGui::PopID();
                }
                ImGui::End();
            }

            void guardAsset(std::function<void()> action)
            {
                if (mAsset) mAsset->cancelGesture();
                if (mAsset && mAsset->dirty()) mAssetPending = std::move(action);
                else attempt(action);
            }

            void switchWorkspace(bool asset)
            {
                if (mExportDialog) return;
                if (asset == mAssetMode || (asset && !mAsset)) return;
                finishManipulation(true); finishStroke();
                mNavigating = false; SDL_SetWindowRelativeMouseMode(mWindow.get(), false);
                if (!asset) mAsset->suspend(*mRenderer);
                mAssetMode = asset;
            }

            void activateAssetWorkspace()
            {
                if (mAsset) switchWorkspace(true);
                else if (const auto* reference = selectedReference()) requestAsset(reference->placement.baseObject);
            }

            void requestAsset(std::uint32_t id)
            {
                mNextFrameAction = [this, id] { openAssetNow(id); };
            }

            void openAssetNow(std::uint32_t id)
            {
                if (mExportDialog) return;
                if (mImport || bakeBusy()) { mMessages.push_back("Finish the import or LOD bake before editing an asset"); return; }
                if (mAsset && mAsset->modelId() == id) { switchWorkspace(true); return; }
                guardAsset([this,id] {
                    auto next = std::make_unique<AssetWorkspace>(*mDocument, id, mData);
                    if (mAsset) mAsset->suspend(*mRenderer);
                    finishManipulation(true); finishStroke();
                    mNavigating = false; SDL_SetWindowRelativeMouseMode(mWindow.get(), false);
                    mAsset = std::move(next); mAssetMode = true;
                });
            }

            void diagnostics()
            {
                ImGui::Begin("Diagnostics");
                const auto& stats = mRenderer->stats();
                ImGui::Text("%s", mRenderer->deviceName().c_str());
                ImGui::Text("CPU %.2f ms | GPU %.2f ms | %u visible meshlets",
                    stats.cpuFrameMs, stats.gpuFrameMs, stats.visibleMeshlets);
                ImGui::Text("Textures %u (%u sparse) | resident %.1f MiB / virtual %.1f MiB",
                    stats.residentTextures, stats.sparseTextures,
                    stats.textureResidentBytes / 1048576.0, stats.textureVirtualBytes / 1048576.0);
                ImGui::Text("Objects %zu loaded / %zu pending | camera %.0f, %.0f, %.0f | speed %.0f",
                    mObjects->loaded(), mObjects->pending(), mPosition.x, mPosition.y, mPosition.z, mMoveSpeed);
                if (mTerrain) ImGui::Text("Terrain %zu resident cells / %zu pending jobs",
                    mTerrain->residentCells(), mTerrain->pendingWork());
                if (mBakePreview.valid()) ImGui::TextUnformatted("LOD bake: scanning saved sources...");
                if (mBakeAwaitingConfirmation) ImGui::TextUnformatted("LOD bake: awaiting confirmation");
                if (mBakeJob.valid() && mBakeProgress)
                {
                    const auto done = mBakeProgress->completed.load();
                    const auto total = mBakeProgress->total.load();
                    ImGui::Text("LOD bake: %llu / %llu tiles",
                        static_cast<unsigned long long>(done),
                        static_cast<unsigned long long>(total));
                    ImGui::ProgressBar(total ? static_cast<float>(done) / static_cast<float>(total) : 0.0f);
                    if (ImGui::Button("Cancel bake")) mBakeProgress->cancel.store(true);
                }
                for (const auto& message : mMessages) ImGui::TextWrapped("%s", message.c_str());
                for (const auto& message : mObjects->diagnostics()) ImGui::TextWrapped("%s", message.c_str());
                ImGui::End();
            }

            void shortcuts()
            {
                if (mExportDialog) return;
                if (ImGui::IsKeyPressed(ImGuiKey_Escape)) { finishManipulation(true); return; }
                if (mGizmoDragging || mInspectorEditing) return;
                if (ImGui::GetIO().WantTextInput || mNavigating) return;
                const bool control = ImGui::GetIO().KeyCtrl;
                if (control)
                {
                    if (ImGui::IsKeyPressed(ImGuiKey_S)) attempt([&] { save(); });
                    if (ImGui::IsKeyPressed(ImGuiKey_O)) showDialog(FileAction::open);
                    if (ImGui::IsKeyPressed(ImGuiKey_N)) showDialog(FileAction::create);
                    if (ImGui::IsKeyPressed(ImGuiKey_Z)) history(false);
                    if (ImGui::IsKeyPressed(ImGuiKey_Y)) history(true);
                    if (ImGui::IsKeyPressed(ImGuiKey_D)) duplicate();
                }
                else
                {
                    if (ImGui::IsKeyPressed(ImGuiKey_F)) frameSelection();
                    if (ImGui::IsKeyPressed(ImGuiKey_Delete)) erase();
                    if (ImGui::IsKeyPressed(ImGuiKey_End)) dropToGround();
                    if (ImGui::IsKeyPressed(ImGuiKey_W)) mOperation = ImGuizmo::TRANSLATE;
                    if (ImGui::IsKeyPressed(ImGuiKey_E)) mOperation = ImGuizmo::ROTATE;
                    if (ImGui::IsKeyPressed(ImGuiKey_R)) mOperation = ImGuizmo::SCALEU;
                    if (mTerrainMode && mTerrain)
                        for (int i = 0; i < 4; ++i)
                            if (ImGui::IsKeyPressed(static_cast<ImGuiKey>(ImGuiKey_1 + i)))
                            {
                                mLandscapeTool = i;
                                mTerrain->setTool(static_cast<world::LandscapeTool>(i));
                            }
                }
            }

            void startImport(const std::filesystem::path& source)
            {
                if (mExportDialog) throw std::runtime_error("Close the export dialog before importing");
                finishManipulation(true);
                if (mImport) throw std::runtime_error("Another import is still running");
                if (!std::filesystem::is_regular_file(source))
                    throw std::runtime_error("Model source does not exist: " + source.string());
                finishStroke();
                if (mTerrain)
                    mDocument->setNextFormId(std::max(mDocument->nextFormId(), mTerrain->nextFormId()));
                const auto id = mDocument->reserveModelId();
                const auto path = mDocument->path();
                mImport.emplace(PendingImport{id, placementTransform(), std::async(std::launch::async,
                    [source, path, id] { return assets::importAndCopyModelAsset(source, path, id); })});
            }

            void finishImport()
            {
                if (mGizmoDragging || mInspectorEditing || mStroke) return;
                if (!mImport || mImport->future.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
                    return;
                auto pending = std::move(*mImport);
                mImport.reset();
                attempt([&] {
                    auto imported = pending.future.get();
                    const auto reference = mDocument->commitImportedModel(pending.formId,
                        imported.modelPath, imported.imported.sourcePath.string(), pending.transform);
                    select(reference);
                    restoreTerrain();
                    for (const auto& warning : imported.imported.warnings) mMessages.push_back(warning);
                    mMessages.push_back("Imported " + imported.modelPath);
                    std::cout << "Editor imported: " << imported.modelPath << " REFR " << reference << '\n';
                    mSmokeImported = true;
                });
            }

            void placeModel(std::uint32_t model)
            {
                if (mGizmoDragging || mInspectorEditing) return;
                finishStroke();
                const auto reference = mDocument->place(model, placementTransform());
                select(reference);
                restoreTerrain();
            }

            void projectObjects()
            {
                std::vector<assets::PlacedReference> references;
                std::vector<std::uint32_t> hidden;
                references.reserve(mDocument->references().size());
                for (const auto& reference : mDocument->references())
                {
                    references.push_back(reference.placement);
                    if (!reference.visible) hidden.push_back(reference.placement.formId);
                }
                std::unordered_map<std::uint32_t, std::filesystem::path> sources;
                for (const auto& [id, path] : mDocument->models())
                    if (auto source = mDocument->sourceEsm(id)) sources.emplace(id, *source);
                mObjects->update(*mRenderer, references, mDocument->models(), sources, mPosition, hidden);
                reconcileSelection();
                std::vector<std::uint32_t> selectedInstances;
                std::optional<std::uint32_t> activeInstance;
                for (auto id : mSelection)
                    if (auto instance = mObjects->rendererInstance(id))
                    {
                        selectedInstances.push_back(*instance);
                        if (mActiveReference == id) activeInstance = *instance;
                    }
                mRenderer->setSelection(selectedInstances, activeInstance);
            }

            void frameSelection()
            {
                if (mSelection.empty()) return;
                glm::vec3 minimum(std::numeric_limits<float>::max());
                glm::vec3 maximum(-std::numeric_limits<float>::max());
                for (auto id : mSelection)
                {
                    const auto bounds = mObjects->bounds(id);
                    const glm::vec3 center = bounds ? glm::vec3(*bounds)
                        : glm::vec3(mDocument->engineTransform(id)[3]);
                    const float radius = bounds ? std::max(bounds->w, 10.0f) : 128.0f;
                    minimum = glm::min(minimum, center - radius);
                    maximum = glm::max(maximum, center + radius);
                }
                glm::vec3 center = (minimum + maximum) * 0.5f;
                center.x = -center.x;
                const float radius = std::max(glm::length(maximum - minimum) * 0.5f, 30.0f);
                mPosition = glm::dvec3(center - forward() * radius * 2.5f);
            }

            void erase()
            {
                if (mGizmoDragging || mInspectorEditing) return;
                finishStroke();
                mDocument->beginTransaction();
                try
                {
                    for (const auto id : editableSelection()) mDocument->erase(id);
                    mDocument->endTransaction();
                    reconcileSelection();
                }
                catch (const std::exception& error) { mDocument->cancelTransaction(); recordError(error); }
            }

            void duplicate()
            {
                if (mGizmoDragging || mInspectorEditing) return;
                finishStroke();
                const auto ids = editableSelection();
                if (ids.empty()) return;
                mDocument->beginTransaction();
                try
                {
                    std::set<std::uint32_t> copies;
                    std::optional<std::uint32_t> activeCopy;
                    for (const auto id : ids)
                    {
                        const auto copy = mDocument->duplicate(id);
                        copies.insert(copy);
                        if (mActiveReference == id) activeCopy = copy;
                    }
                    mDocument->endTransaction();
                    mSelection = std::move(copies);
                    mSelectionOrder.clear();
                    mActiveReference = activeCopy;
                    reconcileSelection();
                    restoreTerrain();
                }
                catch (const std::exception& error) { mDocument->cancelTransaction(); restoreTerrain(); recordError(error); }
            }

            void synchronizeTerrain()
            {
                if (mTerrain) mDocument->synchronizeTerrain(mTerrain->snapshot());
            }

            void restoreTerrain()
            {
                if (mTerrain && mDocument->terrainSnapshot())
                    mTerrain->applySnapshot(*mRenderer, *mDocument->terrainSnapshot());
                if (mTerrain)
                    mTerrain->setNextFormId(std::max(mTerrain->nextFormId(), mDocument->nextFormId()));
            }

            void finishStroke()
            {
                if (!mStroke || !mTerrain) return;
                mTerrain->update(*mRenderer, mPosition, forward(), 0.0f, false);
                synchronizeTerrain();
                mDocument->endTransaction();
                mStroke = false;
            }

            void history(bool redo)
            {
                attempt([&] {
                    finishManipulation(true);
                    finishStroke();
                    if (redo ? mDocument->redo() : mDocument->undo())
                    {
                        restoreTerrain();
                        reconcileSelection();
                    }
                });
            }

            void save()
            {
                if (exportBusy()) throw std::runtime_error("Export is using the saved source; cancel or finish it before saving");
                if (bakeBusy())
                    throw std::runtime_error("LOD bake is using the saved source; cancel or finish it before saving");
                finishManipulation(false);
                finishStroke();
                synchronizeTerrain();
                mDocument->save();
                mMessages.push_back("Saved " + mDocument->path().string());
                std::cout << "Editor saved: " << mDocument->path().string() << '\n';
            }

            void smokeStep()
            {
                if (!mSmokeExport.empty()) { exportSmokeStep(); return; }
                if (!mSmokeImport.empty() && mFrames == 3)
                    attempt([&] { startImport(mSmokeImport); });
                if (mSmokeAsset)
                {
                    assetSmokeStep();
                    return;
                }
                if (!mSmokeTest) return;
                if (mFrames == 8) SDL_SetWindowSize(mWindow.get(), 1280, 800);
                if (mFrames == 15) SDL_SetWindowSize(mWindow.get(), 1440, 900);
                if (!mSmokeCommands && (mSmokeImport.empty() || mSmokeImported)
                    && mFrames >= 20 && mObjects->pending() == 0)
                {
                    attempt([&] {
                        if (!mSmokeImport.empty() && mObjects->loaded() == 0)
                            throw std::runtime_error("Editor smoke imported object was not uploaded");
                        auto referenceCount = mDocument->references().size();
                        if (mSelection.empty() && referenceCount != 0)
                            select(mDocument->references().front().placement.formId);
                        if (!mSelection.empty())
                        {
                            const auto id = *mSelection.begin();
                            auto transform = mDocument->engineTransform(id);
                            transform[3].x += cellSize + 128.0f;
                            mDocument->setTransform(id, transform);
                            if (!mDocument->undo() || !mDocument->redo())
                                throw std::runtime_error("Editor smoke transform undo/redo failed");
                            duplicate();
                            erase();
                            history(false);
                            history(true);
                            select(id);
                            mSmokeReference = id;
                            mSmokeTransform = mDocument->engineTransform(id);
                            frameSelection();
                            // Exercise the same selection, duplication, grounding and
                            // cancellation commands used by the interactive controls.
                            select(id);
                            const auto baseline = mDocument->engineTransform(id);
                            const auto terrainBefore = mDocument->terrainSnapshot();
                            const auto allocatorBefore = mDocument->nextFormId();
                            mDocument->beginTransaction();
                            mGizmoDragging = true;
                            auto far = baseline; far[3].x += cellSize * 12.0f;
                            mDocument->setTransform(id, far);
                            finishManipulation(true);
                            if (mDocument->engineTransform(id) != baseline
                                || mDocument->terrainSnapshot() != terrainBefore
                                || mDocument->nextFormId() != allocatorBefore)
                                throw std::runtime_error("Editor smoke canceled drag retained transform/cell/allocator edits");

                            mDocument->setLocked(id, true);
                            duplicate();
                            if (mDocument->references().size() != referenceCount)
                                throw std::runtime_error("Editor smoke duplicated a locked reference");
                            mDocument->setLocked(id, false);
                            mDocument->setVisible(id, false);
                            duplicate();
                            if (mDocument->references().size() != referenceCount)
                                throw std::runtime_error("Editor smoke duplicated a hidden reference");
                            mDocument->setVisible(id, true);

                            select(id); duplicate();
                            const auto second = *mActiveReference;
                            duplicate();
                            const auto third = *mActiveReference;
                            referenceCount += 2;
                            const std::array block{id, second, third};
                            mDocument->beginTransaction();
                            for (std::size_t i = 0; i < block.size(); ++i)
                            {
                                auto placed = baseline;
                                placed[3].x = 3900.0f + 256.0f * static_cast<float>(i);
                                placed[3].z = 1024.0f;
                                mDocument->setTransform(block[i], placed);
                            }
                            mDocument->endTransaction();
                            restoreTerrain();
                            select(id); select(second, true); select(third, true);
                            if (mActiveReference != third || mSelection.size() != 3)
                                throw std::runtime_error("Editor smoke active selection ordering failed");
                            mCenterPivot = true;
                            const auto pivot = selectionPivot(editableSelection());
                            auto movedPivot = pivot; movedPivot[3].x += 130.0f;
                            movedPivot = snapManipulationPivot(pivot, movedPivot,
                                ManipulationOperation::translate, ManipulationSpace::world, 128.0f);
                            mDocument->beginTransaction();
                            for (auto object : block)
                                mDocument->setTransform(object, applyManipulationDelta(
                                    mDocument->engineTransform(object), pivot, movedPivot));
                            mDocument->endTransaction();
                            if (!mDocument->undo() || !mDocument->redo())
                                throw std::runtime_error("Editor smoke group undo/redo failed");
                            projectObjects();
                            dropToGround();
                            projectObjects();
                            for (auto object : block)
                            {
                                const auto placed = mDocument->engineTransform(object);
                                const auto bounds = mObjects->geometryBounds(object, placed);
                                const auto ground = mTerrain ? mTerrain->surfaceAt({placed[3].x, placed[3].z}) : std::nullopt;
                                if (ground && (!bounds || std::abs(bounds->minimum.y - ground->height) > 0.01))
                                    throw std::runtime_error("Editor smoke Drop to Ground missed LAND");
                                if (bounds)
                                {
                                    auto above = (bounds->minimum + bounds->maximum) * 0.5;
                                    above.y = bounds->maximum.y + 128.0;
                                    const auto hit = mObjects->pick(above, {0, -1, 0});
                                    if (!hit || hit->formId != object)
                                        throw std::runtime_error("Editor smoke exact triangle selection missed placed cube");
                                }
                                mSmokeBlockTransforms.emplace_back(object, placed);
                            }
                            mSmokeTransform = mDocument->engineTransform(id);
                            frameSelection();
                            std::cout << "Editor smoke precise placement: three references across CELL boundary, group snap/undo, grounding and cancel verified\n";
                        }
                        if (mDocument->references().size() != referenceCount)
                            throw std::runtime_error("Editor smoke duplicate/delete history changed reference count");
                        if (mTerrain)
                        {
                            synchronizeTerrain();
                            const auto before = mTerrain->snapshot();
                            if (before.cellForms.empty())
                                throw std::runtime_error("Editor smoke has no LAND to sculpt");
                            const auto cell = before.containsCell(cameraCell())
                                ? cameraCell() : before.cellForms.begin()->first;
                            const auto heights = before.cell(cell).heights;
                            const glm::dvec3 above(-(cell.x + 0.5) * cellSize,
                                heights[16 * assets::LandscapeCell::side + 16] + 1024.0,
                                (cell.y + 0.5) * cellSize);
                            mDocument->beginTransaction();
                            mTerrain->setTool(world::LandscapeTool::raise);
                            mTerrain->setBrushEnabled(true);
                            mTerrain->update(*mRenderer, above, {0.0f, -1.0f, 0.0f}, 0.125f, true);
                            mTerrain->update(*mRenderer, above, {0.0f, -1.0f, 0.0f}, 0.0f, false);
                            mTerrain->setBrushEnabled(false);
                            synchronizeTerrain();
                            mDocument->endTransaction();
                            const auto raised = mTerrain->snapshot().cell(cell).heights;
                            if (raised == heights)
                                throw std::runtime_error("Editor smoke raise stroke did not change LAND");
                            if (!mDocument->undo()) throw std::runtime_error("Editor smoke terrain undo failed");
                            restoreTerrain();
                            if (mTerrain->snapshot().cell(cell).heights != heights)
                                throw std::runtime_error("Editor smoke terrain undo changed baseline heights");
                            if (!mDocument->redo()) throw std::runtime_error("Editor smoke terrain redo failed");
                            restoreTerrain();
                            if (mTerrain->snapshot().cell(cell).heights != raised)
                                throw std::runtime_error("Editor smoke terrain redo changed sculpted heights");
                            mSmokeTerrainCell = cell;
                            mSmokeTerrainHeights = raised;
                        }
                        mSmokeExpectedReferences = referenceCount;
                        mPosition.x += cellSize * 2.0;
                        mSmokeCrossingFrame = mFrames;
                        std::cout << "Editor smoke object and terrain commands complete; crossing cells\n";
                    });
                    mSmokeCommands = true;
                }
                // Outrun LAND preparation, including a level-32 BTR boundary, to
                // exercise concurrent camera-cell checks and background shard reads.
                if (mSmokeCommands && mTerrain && !mSmokeFailed && !mSmokeReopened
                    && mFrames > mSmokeCrossingFrame && mFrames <= mSmokeCrossingFrame + 16)
                    mPosition.z += cellSize * 2.0;
                if (mSmokeCommands && !mSmokeFailed && !mSmokeReopened && !mSmokeReopen
                    && mFrames >= mSmokeCrossingFrame + 20 && mObjects->pending() == 0
                    && (!mTerrain || mTerrain->pendingWork() == 0))
                    attempt([&] {
                        save();
                        mSmokeReopen = true;
                    });
                if (mSmokeBake && mSmokeReopened && !mSmokeBakeStarted
                    && mFrames >= mSmokeReopenFrame + 10 && mObjects->pending() == 0
                    && (!mTerrain || mTerrain->pendingWork() == 0))
                    attempt([&] {
                        requestBake(true);
                        mSmokeBakeStarted = true;
                        mSmokeBakeCamera = mPosition;
                        mSmokeBakeSelection = mSelection;
                        if (mActiveReference)
                        {
                            mSmokeBakeReference = mActiveReference;
                            mSmokeBakeTransform = mDocument->engineTransform(*mActiveReference);
                            mSmokeBakeTransform[3].x += 64.0f;
                            mDocument->setTransform(*mActiveReference, mSmokeBakeTransform);
                        }
                    });
            }

            void assetSmokeStep()
            {
                if (mFrames < mAssetSmokeFrame + 15 || mSmokeFailed || mAssetSmokeStage >= 6) return;
                attempt([&] {
                    if (mAssetSmokeStage == 0)
                    {
                        if (!mSmokeImported || mObjects->pending() || !mObjects->loaded()) return;
                        mSmokeReference = mDocument->references().back().placement.formId;
                        mSmokeBakeCamera = mPosition; mSmokeBakeSelection = mSelection;
                        mSmokeTransform = mDocument->engineTransform(*mSmokeReference);
                        mSmokeTransform[3].x += 16;
                        mDocument->setTransform(*mSmokeReference, mSmokeTransform);
                        if(mTerrain)
                        {
                            const auto before=mTerrain->snapshot();
                            const auto cell=before.cellForms.begin()->first;
                            const auto heights=before.cell(cell).heights;
                            const glm::dvec3 above(-(cell.x+.5)*cellSize,heights[16*assets::LandscapeCell::side+16]+1024,(cell.y+.5)*cellSize);
                            mDocument->beginTransaction();
                            mTerrain->setTool(world::LandscapeTool::raise);mTerrain->setBrushEnabled(true);
                            mTerrain->update(*mRenderer,above,{0,-1,0},.125f,true);
                            mTerrain->update(*mRenderer,above,{0,-1,0},0,false);mTerrain->setBrushEnabled(false);
                            synchronizeTerrain();mDocument->endTransaction();
                            mSmokeTerrainCell=cell;mSmokeTerrainHeights=mTerrain->snapshot().cell(cell).heights;
                            if(mSmokeTerrainHeights==heights) throw std::runtime_error("Asset smoke did not sculpt LAND");
                        }
                        if (mAsset || !selectedReference()) throw std::runtime_error("Asset smoke requires a selected object without an open asset");
                        activateAssetWorkspace();
                    }
                    else if (mAssetSmokeStage == 1)
                    {
                        if (!mAsset || !mAssetMode || !mAsset->renderable()) throw std::runtime_error("Asset workspace command did not open the selected object");
                        std::cout << "Asset workspace command opened selected object without a prior asset session\n";
                        mAsset->smokeEdit();
                        SDL_SetWindowSize(mWindow.get(), 1280, 800);
                    }
                    else if (mAssetSmokeStage == 2)
                    {
                        mAsset->smokeVerifyPhysics();
                        const auto cameras = mPosition == mSmokeBakeCamera && mSelection == mSmokeBakeSelection;
                        if (!cameras || mDocument->engineTransform(*mSmokeReference) != mSmokeTransform || !mDocument->dirty())
                            throw std::runtime_error("Asset preview changed world state");
                        if(mTerrain && mSmokeTerrainCell && mTerrain->snapshot().cell(*mSmokeTerrainCell).heights!=mSmokeTerrainHeights)
                            throw std::runtime_error("Asset preview lost unsaved LAND");
                        mAsset->save();
                        if (mAsset->dirty() || !mDocument->dirty()) throw std::runtime_error("Asset save dirty boundary failed");
                        const auto* model = mDocument->model(mAsset->modelId());
                        if (!model || model->kind != assets::ESMModelKind::movableStatic) throw std::runtime_error("Native asset did not assign MSTT");
                        switchWorkspace(false);
                    }
                    else if (mAssetSmokeStage == 3)
                    {
                        if (mObjects->pending()) return;
                        if (!mDocument->undo() || !mDocument->redo()) throw std::runtime_error("Native assignment world undo failed");
                        save();
                        const auto id=mAsset->modelId();
                        mAsset.reset(); openAssetNow(id);
                        if (!mAsset->document().movable()) throw std::runtime_error("Native asset reload lost movable collision");
                    }
                    else if (mAssetSmokeStage == 4)
                    {
                        // Draw between replacements so the check exercises uploaded resources and fences.
                        const auto current=mRenderer->stats();
                        if(mAssetSmokePreviewStats && (current.residentTextures!=mAssetSmokePreviewStats->residentTextures
                            || current.textureResidentBytes!=mAssetSmokePreviewStats->textureResidentBytes
                            || current.sourceCandidates!=mAssetSmokePreviewStats->sourceCandidates))
                            throw std::runtime_error("Preview resources accumulated");
                        mAssetSmokePreviewStats=current;
                        if(mAssetSmokeReplacements++<4)
                        {switchWorkspace(false);switchWorkspace(true);mAssetSmokeFrame=mFrames;return;}
                        mAsset->makeUnique(*mSmokeReference);
                        mAsset->save();switchWorkspace(false);save();
                    }
                    else if (mAssetSmokeStage == 5)
                    {
                        if(mObjects->pending()) return;
                        const auto path=mDocument->path();openWorld(path);
                        if(mDocument->engineTransform(*mSmokeReference)!=mSmokeTransform) throw std::runtime_error("Asset smoke reopen changed placement");
                        if(mTerrain && mSmokeTerrainCell && mTerrain->snapshot().cell(*mSmokeTerrainCell).heights!=mSmokeTerrainHeights)
                            throw std::runtime_error("Native asset save/reopen lost sculpted LAND");
                        std::cout<<"Asset smoke verified: edit, physics preview, undo, native save, MSTT assignment, unique, reopen, viewport resize and world preservation\n";
                    }
                    ++mAssetSmokeStage;mAssetSmokeFrame=mFrames;
                });
            }

            void exportSmokeStep()
            {
                if (mFrames < 20 || mSmokeFailed || mSmokeExportDone || mObjects->pending()) return;
                attempt([&] {
                    if (!mExportDialog)
                    {
                        if (!mDocument->references().empty())
                        {
                            const auto id = mDocument->references().front().placement.formId;
                            select(id);
                            auto transform = mDocument->engineTransform(id);
                            transform[3].y += 16.0f;
                            mDocument->setTransform(id, transform);
                            if (!mDocument->dirty()) throw std::runtime_error("Export smoke did not create unsaved work");
                        }
                        mSmokeExportCamera = mPosition; mSmokeExportSelection = mSelection;
                        mSmokeExportWorld = mDocument->path();
                        requestExport(); setExportDestination(mSmokeExport);
                    }
                    else if (!mExportError.empty()) throw std::runtime_error(mExportError);
                    else if (!exportBusy())
                    {
                        if (mExportResult)
                        {
                            if (!std::filesystem::is_regular_file(mExportResult->pluginPath)
                                || !std::filesystem::is_regular_file(mExportResult->packageDirectory / "export-report.txt")
                                || mDocument->path() != mSmokeExportWorld || mPosition != mSmokeExportCamera
                                || mSelection != mSmokeExportSelection || mDocument->dirty())
                                throw std::runtime_error("Export smoke changed editor state or failed to publish package");
                            mSmokeExportDone = true;
                            std::cout << "Editor export smoke verified: save, preflight, native package, camera/selection/project preserved\n";
                        }
                        else if (mExportPreflight) launchExport();
                        else inspectExport();
                    }
                });
            }

            SdlLifetime mSdl;
            std::filesystem::path mData;
            std::unique_ptr<SDL_Window, WindowDeleter> mWindow;
            std::unique_ptr<render::VulkanRenderer> mRenderer;
            std::unique_ptr<EditorUi> mUi;
            std::unique_ptr<EditorDocument> mDocument;
            std::unique_ptr<AssetWorkspace> mAsset;
            bool mAssetMode = false;
            unsigned mAssetSmokeReplacements = 0;
            std::optional<render::RendererStats> mAssetSmokePreviewStats;
            std::function<void()> mAssetPending;
            std::function<void()> mNextFrameAction;
            std::unique_ptr<world::LandscapeEditor> mTerrain;
            std::unique_ptr<world::WorldStreamer> mStreamer;
            std::unique_ptr<world::ObjectScene> mObjects;
            std::shared_ptr<DialogMailbox> mDialogMailbox = std::make_shared<DialogMailbox>();
            std::optional<PendingImport> mImport;
            std::future<assets::LodBakePreflight> mBakePreview;
            std::future<assets::LodBakeResult> mBakeJob;
            std::optional<assets::LodBakePreflight> mBakePreflight;
            assets::LodBakeOptions mBakeOptions;
            std::shared_ptr<BakeProgressState> mBakeProgress;
            std::filesystem::path mBakeWorld;
            bool mBakeAwaitingConfirmation = false;
            bool mExportDialog = false, mExportOpenPopup = false;
            std::array<char, 4096> mExportDestination{};
            std::filesystem::path mExportWorld;
            std::shared_ptr<ExportProgressState> mExportProgress;
            std::future<assets::Fo4ExportPreflight> mExportPreview;
            std::future<assets::Fo4ExportResult> mExportJob;
            std::optional<assets::Fo4ExportPreflight> mExportPreflight;
            std::optional<assets::Fo4ExportResult> mExportResult;
            std::string mExportError;
            std::filesystem::path mSmokeExport, mSmokeExportWorld;
            glm::dvec3 mSmokeExportCamera{};
            std::set<std::uint32_t> mSmokeExportSelection;
            bool mSmokeExportDone = false;
            std::set<std::uint32_t> mSelection;
            std::vector<std::uint32_t> mSelectionOrder;
            std::optional<std::uint32_t> mActiveReference;
            std::vector<std::string> mMessages;
            ScriptsPanel mScripts;
            glm::dvec3 mPosition{};
            float mYaw = 0.0f, mPitch = 0.0f;
            float mMoveSpeed = 4096.0f;
            ImVec2 mViewportPosition{}, mViewportSize{1, 1};
            ImVec2 mFileMenuCenter{};
            assets::CellCoord mLastCell{};
            std::size_t mResidentTerrainCells = 0;
            ImGuizmo::OPERATION mOperation = ImGuizmo::TRANSLATE;
            ManipulationSpace mManipulationSpace = ManipulationSpace::world;
            bool mCenterPivot = false;
            bool mSnapTranslation = false, mSnapRotation = false, mAlignTerrain = false;
            float mTranslationStep = 128.0f, mRotationStep = 15.0f;
            glm::mat4 mStartPivot{1.0f}, mDragPivot{1.0f};
            std::vector<std::pair<std::uint32_t, glm::mat4>> mDragTransforms;
            bool mRunning = true;
            bool mNavigating = false;
            bool mViewportHovered = false;
            bool mViewportFocused = false;
            bool mTerrainMode = false;
            bool mStroke = false;
            bool mGizmoDragging = false;
            bool mSuppressGizmo = false;
            bool mInspectorEditing = false;
            ImGuiID mInspectorItem = 0;
            bool mEndInspectorTransaction = false;
            int mLandscapeTool = 0;
            int mFrames = 0, mFrameLimit = 0;
            std::filesystem::path mSmokeImport;
            bool mSmokeTest = false;
            bool mSmokeAsset = false;
            int mAssetSmokeStage = 0, mAssetSmokeFrame = 0;
            bool mSmokeBake = false, mSmokeBakeStarted = false, mSmokeBakeCompleted = false;
            glm::dvec3 mSmokeBakeCamera{};
            std::set<std::uint32_t> mSmokeBakeSelection;
            std::optional<std::uint32_t> mSmokeBakeReference;
            glm::mat4 mSmokeBakeTransform{1.0f};
            bool mSmokeImported = false;
            bool mSmokeCommands = false;
            bool mSmokeFailed = false;
            int mSmokeMenuChecks = 0;
            int mSmokeReopenFrame = 0;
            bool mSmokeReopen = false;
            bool mSmokeReopened = false;
            int mSmokeCrossingFrame = 0;
            std::size_t mSmokeExpectedReferences = 0;
            std::optional<std::uint32_t> mSmokeReference;
            glm::mat4 mSmokeTransform{1.0f};
            std::vector<std::pair<std::uint32_t, glm::mat4>> mSmokeBlockTransforms;
            std::optional<assets::CellCoord> mSmokeTerrainCell;
            std::array<float, assets::LandscapeCell::side * assets::LandscapeCell::side> mSmokeTerrainHeights{};
        };
    }

    int runEditor(int argc, char** argv)
    {
        std::filesystem::path path = "mods/OpenF4World.esm";
        std::filesystem::path smokeImport;
        std::filesystem::path smokeExport;
        int frameLimit = 0;
        bool pathSet = false, smokeTest = false, smokeBake = false, smokeAsset = false;
        bool headlessBake = false, bakeFull = false, bakeYes = false;
        bool terrainOnly = false, staticsOnly = false;
        for (int index = 1; index < argc; ++index)
        {
            const std::string argument = argv[index];
            if (argument == "--frames")
            {
                if (++index >= argc) throw std::runtime_error("--frames expects a positive count");
                frameLimit = std::stoi(argv[index]);
                if (frameLimit <= 0) throw std::runtime_error("--frames expects a positive count");
            }
            else if (argument == "--smoke-import")
            {
                if (++index >= argc) throw std::runtime_error("--smoke-import expects a model path");
                smokeImport = argv[index];
            }
            else if (argument == "--smoke-test") smokeTest = true;
            else if (argument == "--smoke-export")
            {
                if (++index >= argc) throw std::runtime_error("--smoke-export expects a new package directory");
                smokeExport = argv[index];
            }
            else if (argument == "--smoke-asset") smokeAsset = true;
            else if (argument == "--smoke-bake") { smokeTest = true; smokeBake = true; }
            else if (argument == "--bake-lod") headlessBake = true;
            else if (argument == "--full") bakeFull = true;
            else if (argument == "--yes") bakeYes = true;
            else if (argument == "--terrain-only") terrainOnly = true;
            else if (argument == "--statics-only") staticsOnly = true;
            else if (argument == "--help")
            {
                std::cout << "openf4-editor [world.esm] [--frames N] [--smoke-import model.glb] [--smoke-test|--smoke-asset]\n"
                    << "openf4-editor world.esm --smoke-export new-package-directory [--frames N]\n"
                    << "openf4-editor world.esm --bake-lod [--full] [--terrain-only|--statics-only] [--yes]\n";
                return 0;
            }
            else if (!pathSet && !argument.starts_with("--"))
            {
                path = argument;
                pathSet = true;
            }
            else throw std::runtime_error("Unknown editor argument: " + argument);
        }
        if (terrainOnly && staticsOnly)
            throw std::runtime_error("Choose either --terrain-only or --statics-only");
        if (!headlessBake && (bakeFull || bakeYes || terrainOnly || staticsOnly))
            throw std::runtime_error("LOD bake options require --bake-lod");
        const auto data = assets::findFallout4Data();
        if (headlessBake)
        {
            if (!pathSet || frameLimit || smokeTest || smokeAsset || !smokeImport.empty() || !smokeExport.empty())
                throw std::runtime_error("Headless bake expects a world.esm and bake options only");
            assets::LodBakeOptions options;
            options.full = bakeFull;
            options.terrain = !staticsOnly;
            options.statics = !terrainOnly;
            const auto preflight = preflightLod(path, data, options);
            std::cout << "LOD preflight: " << preflight.terrainTiles << " terrain + "
                << preflight.staticTiles << " static tiles; estimated "
                << preflight.estimatedBytes / 1048576 << " MiB"
                << (preflight.full ? " (full)" : " (changed)") << '\n';
            if (preflight.totalTiles() > 10000 && !bakeYes)
                throw std::runtime_error("More than 10,000 tiles; rerun with --yes to confirm the bake");
            std::uint64_t lastReport = 0;
            options.progress = [&lastReport](const assets::LodBakeProgress& progress) {
                if (progress.completed == progress.total || progress.completed >= lastReport + 100)
                {
                    std::cout << "LOD bake: " << progress.completed << " / " << progress.total
                        << " tiles\n";
                    lastReport = progress.completed;
                }
            };
            const auto result = bakeLod(path, data, options);
            for (const auto& diagnostic : result.diagnostics)
                std::cout << diagnostic << '\n';
            std::cout << "LOD bake complete: " << result.terrainBaked << " terrain, "
                << result.staticBaked << " static tiles, " << result.removedTiles
                << " removed; revision " << result.revision << '\n';
            return result.cancelled ? 1 : 0;
        }
        if (!data) throw std::runtime_error("Editor requires Fallout 4 data; set FALLOUT4_PATH");
        if (!smokeExport.empty() && (!pathSet || smokeTest || smokeAsset || !smokeImport.empty()))
            throw std::runtime_error("Export smoke requires only a proof world, output directory, and optional --frames");
        if (frameLimit == 0 && !smokeExport.empty()) frameLimit = 360;
        if ((smokeTest || smokeAsset) && frameLimit == 0) frameLimit = 240;
        if (smokeAsset && (smokeImport.empty() || smokeTest)) throw std::runtime_error("--smoke-asset requires --smoke-import and is separate from --smoke-test");
        EditorApp app(*data, path, frameLimit, smokeImport, smokeTest, smokeBake, smokeAsset, smokeExport);
        return app.run();
    }
}
