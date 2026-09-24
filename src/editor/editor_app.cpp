#include "editor/editor_app.hpp"
#include "editor/editor_document.hpp"
#include "assets/fo4_install.hpp"
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

        enum class FileAction { open, create, saveAs, import };
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

        class EditorApp
        {
        public:
            EditorApp(std::filesystem::path data, std::filesystem::path path,
                int frameLimit, std::filesystem::path smokeImport, bool smokeTest)
                : mData(std::move(data)), mFrameLimit(frameLimit),
                  mSmokeImport(std::move(smokeImport)), mSmokeTest(smokeTest)
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
                // The UI backend owns Vulkan resources and must be destroyed first.
                mUi.reset();
            }

            int run()
            {
                std::cout << "OpenF4 editor: " << mDocument->path().string() << '\n'
                    << "RMB + WASD: fly; Q/E: down/up; Shift: faster; F: frame selection\n"
                    << "W/E/R: move/rotate/scale; Delete: delete; Ctrl+D: duplicate; "
                       "Ctrl+Z/Y: undo/redo; Ctrl+S: save\n";
                auto previous = std::chrono::steady_clock::now();
                while (mRunning)
                {
                    const auto now = std::chrono::steady_clock::now();
                    const float delta = std::clamp(std::chrono::duration<float>(now - previous).count(),
                        0.0001f, 0.1f);
                    previous = now;
                    events();
                    handleDialogs();
                    finishImport();
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
                                mSelection = {*mSmokeReference};
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
                    shell();
                    navigate(delta);
                    viewport(delta);
                    inspector();
                    outliner();
                    assetsPanel();
                    diagnostics();
                    shortcuts();
                    projectObjects();
                    smokeStep();
                    ImGui::Render();
                    const glm::dvec3 origin = renderOrigin(mPosition);
                    const glm::vec3 local(mPosition - origin);
                    const glm::mat4 view = glm::lookAtRH(local, local + forward(), {0.0f, 1.0f, 0.0f});
                    auto projection = render::reverseZInfinitePerspectiveRH_ZO(glm::radians(70.0f),
                        std::max(mViewportSize.x, 1.0f) / std::max(mViewportSize.y, 1.0f), 0.1f);
                    projection[1][1] *= -1.0f;
                    const auto scale = ImGui::GetIO().DisplayFramebufferScale;
                    const render::RenderViewport rectangle{
                        static_cast<std::int32_t>(mViewportPosition.x * scale.x),
                        static_cast<std::int32_t>(mViewportPosition.y * scale.y),
                        static_cast<std::uint32_t>(std::max(mViewportSize.x, 0.0f) * scale.x),
                        static_cast<std::uint32_t>(std::max(mViewportSize.y, 0.0f) * scale.y)};
                    mRenderer->render(view, projection, local, glm::vec3(origin), rectangle,
                        [](VkCommandBuffer commands) {
                            ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), commands);
                        });
                    ++mFrames;
                    if (mFrameLimit > 0 && mFrames >= mFrameLimit)
                        mRunning = false;
                }
                finishStroke();
                // Do not abandon a source copy on shutdown: finish and commit its result.
                if (mImport)
                {
                    mImport->future.wait();
                    finishImport();
                }
                if (mDocument->dirty())
                    save();
                std::cout << "Editor finished: " << mFrames << " frames, "
                    << mDocument->references().size() << " references, "
                    << mObjects->loaded() << " loaded, " << mObjects->pending() << " pending, "
                    << (mTerrain ? mTerrain->pendingWork() : 0) << " terrain jobs pending\n";
                if (mSmokeTest && (mSmokeMenuChecks != 3 || !mSmokeCommands || !mSmokeReopened
                    || mObjects->pending() != 0 || (mTerrain && mTerrain->pendingWork() != 0)
                    || (!mSmokeImport.empty() && mObjects->loaded() == 0)))
                {
                    std::cerr << "Editor smoke did not complete menu input, imports, commands, reopen, and streaming drain\n";
                    mSmokeFailed = true;
                }
                if (mSmokeTest && mSmokeFailed) return 1;
                return 0;
            }

        private:
            void recordError(const std::exception& error)
            {
                mMessages.push_back(error.what());
                std::cerr << "Editor: " << error.what() << '\n';
                mSmokeFailed |= mSmokeTest;
            }

            template<class Operation>
            void attempt(Operation&& operation)
            {
                try { operation(); }
                catch (const std::exception& error) { recordError(error); }
            }

            void openWorld(const std::filesystem::path& path)
            {
                if (mDocument && mDocument->dirty())
                    save();
                if (mImport)
                    throw std::runtime_error("Wait for the current model import before opening a world");
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
                    switch (event.type)
                    {
                    case SDL_EVENT_QUIT: mRunning = false; break;
                    case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED: mRenderer->resize(); break;
                    case SDL_EVENT_WINDOW_FOCUS_LOST:
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
                        if (event.drop.data)
                            attempt([&] { startImport(std::filesystem::u8path(event.drop.data)); });
                        break;
                    default: break;
                    }
                }
            }

            void showDialog(FileAction action)
            {
                std::scoped_lock lock(mDialogMailbox->mutex);
                if (mDialogMailbox->pending || mImport)
                    return;
                mDialogMailbox->pending = true;
                auto* request = new DialogRequest{mDialogMailbox, action};
                static constexpr SDL_DialogFileFilter esmFilter[]{ {"Bethesda master", "esm"} };
                static constexpr SDL_DialogFileFilter modelFilter[]{
                    {"3D model", "glb;gltf;obj;fbx;dae;ply;stl;blend;3ds"}, {"All files", "*"} };
                if (action == FileAction::saveAs || action == FileAction::create)
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
                        if (ImGui::MenuItem("Save", "Ctrl+S")) attempt([&] { save(); });
                        if (ImGui::MenuItem("Save as...")) showDialog(FileAction::saveAs);
                        ImGui::Separator();
                        if (ImGui::MenuItem("Import model...")) showDialog(FileAction::import);
                        ImGui::Separator();
                        if (ImGui::MenuItem("Exit")) mRunning = false;
                        if (smokeMenuPhase() == 3) ImGui::CloseCurrentPopup();
                        ImGui::EndMenu();
                    }
                    if (ImGui::BeginMenu("Edit"))
                    {
                        if (ImGui::MenuItem("Undo", "Ctrl+Z")) history(false);
                        if (ImGui::MenuItem("Redo", "Ctrl+Y")) history(true);
                        ImGui::Separator();
                        if (ImGui::MenuItem("Duplicate", "Ctrl+D", false, !mSelection.empty())) duplicate();
                        if (ImGui::MenuItem("Delete", "Delete", false, !mSelection.empty())) erase();
                        ImGui::EndMenu();
                    }
                    ImGui::Separator();
                    if (ImGui::MenuItem("Objects", nullptr, !mTerrainMode)) mTerrainMode = false;
                    if (ImGui::MenuItem("Landscape", nullptr, mTerrainMode, mTerrain != nullptr)) mTerrainMode = true;
                    ImGui::Separator();
                    ImGui::TextUnformatted(mDocument->path().filename().string().c_str());
                    if (mDocument->dirty()) { ImGui::SameLine(); ImGui::TextUnformatted("*"); }
                    if (mImport) { ImGui::SameLine(); ImGui::TextUnformatted("Importing..."); }
                    ImGui::EndMenuBar();
                }
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

            void viewport(float delta)
            {
                ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
                ImGui::Begin("Viewport", nullptr, ImGuiWindowFlags_NoBackground
                    | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
                mViewportPosition = ImGui::GetCursorScreenPos();
                mViewportSize = ImGui::GetContentRegionAvail();
                mViewportSize.x = std::max(mViewportSize.x, 1.0f);
                mViewportSize.y = std::max(mViewportSize.y, 1.0f);
                // Reserve the viewport without claiming an active/hovered item ID:
                // ImGuizmo deliberately refuses drags while another widget owns it.
                ImGui::Dummy(mViewportSize);
                mViewportHovered = ImGui::IsItemHovered();
                mViewportFocused = ImGui::IsWindowFocused();
                // Decide after ImGui has processed this frame's popup/capture state,
                // not from the previous frame's hover position in the SDL event loop.
                if (mViewportHovered && !ImGui::GetIO().WantTextInput
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
                if (mTerrain)
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
                    if (mViewportHovered && !mNavigating && !ImGuizmo::IsOver()
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
                double nearest = std::numeric_limits<double>::infinity();
                std::optional<std::uint32_t> hit;
                for (const auto& reference : mDocument->references())
                {
                    if (!reference.visible || reference.locked) continue;
                    const auto bounds = mObjects->bounds(reference.placement.formId);
                    if (!bounds) continue;
                    const glm::dvec3 offset = origin - glm::dvec3(*bounds);
                    const double b = glm::dot(offset, direction);
                    const double discriminant = b * b - glm::dot(offset, offset)
                        + static_cast<double>(bounds->w) * bounds->w;
                    if (discriminant < 0.0) continue;
                    const double entry = -b - std::sqrt(discriminant);
                    const double exit = -b + std::sqrt(discriminant);
                    const double distance = entry >= 0.0 ? entry : exit;
                    if (distance >= 0.0 && distance < nearest)
                    {
                        nearest = distance;
                        hit = reference.placement.formId;
                    }
                }
                if (!ImGui::GetIO().KeyShift) mSelection.clear();
                if (hit)
                {
                    if (mSelection.contains(*hit)) mSelection.erase(*hit);
                    else mSelection.insert(*hit);
                }
            }

            const EditorReference* selectedReference() const
            {
                if (mSelection.empty()) return nullptr;
                const auto id = *mSelection.begin();
                const auto& references = mDocument->references();
                const auto it = std::find_if(references.begin(), references.end(),
                    [id](const EditorReference& ref) { return ref.placement.formId == id; });
                return it == references.end() ? nullptr : &*it;
            }

            void gizmo()
            {
                const auto* selected = selectedReference();
                if (!selected || selected->locked || !selected->visible)
                    return;
                const auto id = selected->placement.formId;
                const glm::dvec3 origin = renderOrigin(mPosition);
                const glm::vec3 local(mPosition - origin);
                const auto view = glm::lookAtRH(local, local + forward(), glm::vec3(0, 1, 0));
                const auto projection = glm::perspectiveRH_NO(glm::radians(70.0f),
                    mViewportSize.x / mViewportSize.y, 0.1f, 10000000.0f);
                const glm::mat4 mirror = reflection();
                const glm::mat4 before = mDocument->engineTransform(id);
                glm::mat4 matrix = mirror * before * mirror;
                matrix[3] -= glm::vec4(glm::vec3(origin), 0.0f);
                ImGuizmo::SetDrawlist();
                ImGuizmo::SetOrthographic(false);
                ImGuizmo::SetRect(mViewportPosition.x, mViewportPosition.y, mViewportSize.x, mViewportSize.y);
                const bool changed = ImGuizmo::Manipulate(glm::value_ptr(view), glm::value_ptr(projection),
                    mOperation, ImGuizmo::LOCAL, glm::value_ptr(matrix));
                if (ImGuizmo::IsUsing() && !mGizmoDragging)
                {
                    mDocument->beginTransaction();
                    mGizmoDragging = true;
                }
                if (changed)
                    attempt([&] {
                        matrix[3] += glm::vec4(glm::vec3(origin), 0.0f);
                        const glm::mat4 after = mirror * matrix * mirror;
                        const glm::mat4 delta = after * glm::inverse(before);
                        for (const auto selectedId : mSelection)
                        {
                            const auto& refs = mDocument->references();
                            const auto ref = std::find_if(refs.begin(), refs.end(), [&](const auto& value) {
                                return value.placement.formId == selectedId;
                            });
                            if (ref != refs.end() && !ref->locked)
                                mDocument->setTransform(selectedId, delta * mDocument->engineTransform(selectedId));
                        }
                    });
                if (!ImGuizmo::IsUsing() && mGizmoDragging)
                {
                    mDocument->endTransaction();
                    mGizmoDragging = false;
                    restoreTerrain();
                }
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
                    ImGui::Text("REFR %08X  /  STAT %08X", id, reference->placement.baseObject);
                    ImGui::Text("Cell (%d, %d)", reference->placement.cell.x, reference->placement.cell.y);
                    if (mSelection.size() > 1)
                        ImGui::Text("%zu selected; gizmo moves the selection", mSelection.size());
                    bool visible = reference->visible;
                    bool lockedValue = locked;
                    if (ImGui::Checkbox("Visible", &visible)) mDocument->setVisible(id, visible);
                    ImGui::SameLine();
                    if (ImGui::Checkbox("Locked", &lockedValue)) mDocument->setLocked(id, lockedValue);
                    ImGui::Separator();
                    if (ImGui::RadioButton("Move", mOperation == ImGuizmo::TRANSLATE)) mOperation = ImGuizmo::TRANSLATE;
                    ImGui::SameLine();
                    if (ImGui::RadioButton("Rotate", mOperation == ImGuizmo::ROTATE)) mOperation = ImGuizmo::ROTATE;
                    ImGui::SameLine();
                    if (ImGui::RadioButton("Scale", mOperation == ImGuizmo::SCALEU)) mOperation = ImGuizmo::SCALEU;
                    float translation[3], rotation[3], scale[3];
                    glm::mat4 transform = mDocument->engineTransform(id);
                    ImGuizmo::DecomposeMatrixToComponents(glm::value_ptr(transform), translation, rotation, scale);
                    ImGui::BeginDisabled(locked);
                    bool changed = false;
                    auto component = [&](const char* label, float* values, float speed) {
                        const bool edited = ImGui::DragFloat3(label, values, speed);
                        if (ImGui::IsItemActivated()) mDocument->beginTransaction();
                        if (ImGui::IsItemDeactivated()) mEndInspectorTransaction = true;
                        return edited;
                    };
                    changed |= component("Position", translation, 4.0f);
                    changed |= component("Rotation", rotation, 0.5f);
                    float uniformScale = scale[0];
                    changed |= ImGui::DragFloat("Uniform scale", &uniformScale, 0.01f, 0.001f, 10000.0f);
                    if (ImGui::IsItemActivated()) mDocument->beginTransaction();
                    if (ImGui::IsItemDeactivated()) mEndInspectorTransaction = true;
                    if (changed)
                        attempt([&] {
                            scale[0] = scale[1] = scale[2] = std::max(uniformScale, 0.001f);
                            ImGuizmo::RecomposeMatrixFromComponents(translation, rotation, scale, glm::value_ptr(transform));
                            mDocument->setTransform(id, transform);
                        });
                    if (mEndInspectorTransaction)
                    {
                        mDocument->endTransaction();
                        mEndInspectorTransaction = false;
                        restoreTerrain();
                    }
                    ImGui::EndDisabled();
                    if (ImGui::Button("Frame selected")) frameSelection();
                    ImGui::SameLine();
                    if (ImGui::Button("Duplicate")) duplicate();
                    if (auto source = mDocument->sourcePath(baseObject))
                        ImGui::TextWrapped("Source: %s", source->c_str());
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
                        + (reference.visible ? "" : " [hidden]") + (reference.locked ? " [locked]" : "");
                    if (!filter.PassFilter(label.c_str())) continue;
                    ImGui::PushID(static_cast<int>(id));
                    if (ImGui::Selectable(label.c_str(), mSelection.contains(id)))
                    {
                        if (!ImGui::GetIO().KeyShift) mSelection.clear();
                        if (mSelection.contains(id)) mSelection.erase(id);
                        else mSelection.insert(id);
                        if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) frameSelection();
                    }
                    ImGui::PopID();
                }
                ImGui::End();
            }

            void assetsPanel()
            {
                ImGui::Begin("Asset Browser");
                ImGui::BeginDisabled(mImport.has_value());
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
                        attempt([&] { placeModel(id); });
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
                for (const auto& message : mMessages) ImGui::TextWrapped("%s", message.c_str());
                for (const auto& message : mObjects->diagnostics()) ImGui::TextWrapped("%s", message.c_str());
                ImGui::End();
            }

            void shortcuts()
            {
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
                if (!mImport || mImport->future.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
                    return;
                auto pending = std::move(*mImport);
                mImport.reset();
                attempt([&] {
                    auto imported = pending.future.get();
                    const auto reference = mDocument->commitImportedModel(pending.formId,
                        imported.modelPath, imported.imported.sourcePath.string(), pending.transform);
                    mSelection = {reference};
                    restoreTerrain();
                    for (const auto& warning : imported.imported.warnings) mMessages.push_back(warning);
                    mMessages.push_back("Imported " + imported.modelPath);
                    std::cout << "Editor imported: " << imported.modelPath << " REFR " << reference << '\n';
                    mSmokeImported = true;
                });
            }

            void placeModel(std::uint32_t model)
            {
                finishStroke();
                const auto reference = mDocument->place(model, placementTransform());
                mSelection = {reference};
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
                attempt([&] {
                    mDocument->beginTransaction();
                    for (const auto id : mSelection)
                    {
                        const auto& refs = mDocument->references();
                        const auto ref = std::find_if(refs.begin(), refs.end(), [&](const auto& value) {
                            return value.placement.formId == id;
                        });
                        if (ref != refs.end() && !ref->locked) mDocument->erase(id);
                    }
                    mDocument->endTransaction();
                    mSelection.clear();
                });
            }

            void duplicate()
            {
                attempt([&] {
                    mDocument->beginTransaction();
                    std::set<std::uint32_t> copies;
                    for (const auto id : mSelection) copies.insert(mDocument->duplicate(id));
                    mDocument->endTransaction();
                    mSelection = std::move(copies);
                    restoreTerrain();
                });
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
                    finishStroke();
                    if (redo ? mDocument->redo() : mDocument->undo())
                    {
                        restoreTerrain();
                        mSelection.clear();
                    }
                });
            }

            void save()
            {
                finishStroke();
                synchronizeTerrain();
                mDocument->save();
                mMessages.push_back("Saved " + mDocument->path().string());
                std::cout << "Editor saved: " << mDocument->path().string() << '\n';
            }

            void smokeStep()
            {
                if (!mSmokeImport.empty() && mFrames == 3)
                    attempt([&] { startImport(mSmokeImport); });
                if (!mSmokeTest) return;
                if (mFrames == 8) SDL_SetWindowSize(mWindow.get(), 1280, 800);
                if (mFrames == 15) SDL_SetWindowSize(mWindow.get(), 1440, 900);
                if (!mSmokeCommands && (mSmokeImport.empty() || mSmokeImported)
                    && mFrames >= 20 && mObjects->pending() == 0)
                {
                    attempt([&] {
                        if (!mSmokeImport.empty() && mObjects->loaded() == 0)
                            throw std::runtime_error("Editor smoke imported object was not uploaded");
                        const auto referenceCount = mDocument->references().size();
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
                            mSelection = {id};
                            mSmokeReference = id;
                            mSmokeTransform = mDocument->engineTransform(id);
                            frameSelection();
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
                if (mSmokeCommands && !mSmokeFailed && !mSmokeReopened && !mSmokeReopen
                    && mFrames >= mSmokeCrossingFrame + 20 && mObjects->pending() == 0
                    && (!mTerrain || mTerrain->pendingWork() == 0))
                    attempt([&] {
                        save();
                        mSmokeReopen = true;
                    });
            }

            SdlLifetime mSdl;
            std::filesystem::path mData;
            std::unique_ptr<SDL_Window, WindowDeleter> mWindow;
            std::unique_ptr<render::VulkanRenderer> mRenderer;
            std::unique_ptr<EditorUi> mUi;
            std::unique_ptr<EditorDocument> mDocument;
            std::unique_ptr<world::LandscapeEditor> mTerrain;
            std::unique_ptr<world::WorldStreamer> mStreamer;
            std::unique_ptr<world::ObjectScene> mObjects;
            std::shared_ptr<DialogMailbox> mDialogMailbox = std::make_shared<DialogMailbox>();
            std::optional<PendingImport> mImport;
            std::set<std::uint32_t> mSelection;
            std::vector<std::string> mMessages;
            glm::dvec3 mPosition{};
            float mYaw = 0.0f, mPitch = 0.0f;
            float mMoveSpeed = 4096.0f;
            ImVec2 mViewportPosition{}, mViewportSize{1, 1};
            ImVec2 mFileMenuCenter{};
            assets::CellCoord mLastCell{};
            std::size_t mResidentTerrainCells = 0;
            ImGuizmo::OPERATION mOperation = ImGuizmo::TRANSLATE;
            bool mRunning = true;
            bool mNavigating = false;
            bool mViewportHovered = false;
            bool mViewportFocused = false;
            bool mTerrainMode = false;
            bool mStroke = false;
            bool mGizmoDragging = false;
            bool mEndInspectorTransaction = false;
            int mLandscapeTool = 0;
            int mFrames = 0, mFrameLimit = 0;
            std::filesystem::path mSmokeImport;
            bool mSmokeTest = false;
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
            std::optional<assets::CellCoord> mSmokeTerrainCell;
            std::array<float, assets::LandscapeCell::side * assets::LandscapeCell::side> mSmokeTerrainHeights{};
        };
    }

    int runEditor(int argc, char** argv)
    {
        std::filesystem::path path = "mods/OpenF4World.esm";
        std::filesystem::path smokeImport;
        int frameLimit = 0;
        bool pathSet = false, smokeTest = false;
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
            else if (argument == "--help")
            {
                std::cout << "openf4-editor [world.esm] [--frames N] [--smoke-import model.glb] [--smoke-test]\n";
                return 0;
            }
            else if (!pathSet && !argument.starts_with("--"))
            {
                path = argument;
                pathSet = true;
            }
            else throw std::runtime_error("Unknown editor argument: " + argument);
        }
        const auto data = assets::findFallout4Data();
        if (!data) throw std::runtime_error("Editor requires Fallout 4 data; set FALLOUT4_PATH");
        if (smokeTest && frameLimit == 0) frameLimit = 180;
        EditorApp app(*data, path, frameLimit, smokeImport, smokeTest);
        return app.run();
    }
}
