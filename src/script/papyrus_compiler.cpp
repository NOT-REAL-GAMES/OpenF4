#include "script/papyrus_compiler.hpp"
#include "script/papyrus_vm.hpp"
#include "assets/asset_resolver.hpp"
#include "assets/papyrus_binding.hpp"
#include "assets/esm_document.hpp"

#include <SDL3/SDL.h>
#include <algorithm>
#include <atomic>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <vector>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

namespace openf4::script
{
    namespace
    {
        std::string utf8(const std::filesystem::path& path)
        {
            const auto text = path.u8string();
            return {text.begin(), text.end()};
        }

        void publish(const std::filesystem::path& from, const std::filesystem::path& to)
        {
#ifdef _WIN32
            if (!MoveFileExW(from.c_str(), to.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
                throw std::runtime_error("Cannot atomically publish compiled Papyrus script");
#else
            std::filesystem::rename(from, to);
#endif
        }
    }

    std::filesystem::path scriptSourcePath(const std::filesystem::path& world, std::string_view name)
    {
        const auto relative = std::filesystem::path(assets::papyrusScriptPath(name, ".psc"));
        return assets::AssetResolver::projectAssetRoot(world) / "scripts/source"
            / relative.lexically_relative("scripts");
    }

    CompileResult compilePapyrus(const std::filesystem::path& world,
        const std::filesystem::path& data, std::string_view name)
    {
        CompileResult result;
        try
        {
            const auto source = std::filesystem::absolute(scriptSourcePath(world, name));
            const auto compiler = data.parent_path() / "Papyrus Compiler/PapyrusCompiler.exe";
            const auto base = data / "Scripts/Source/Base";
            if (!std::filesystem::is_regular_file(compiler))
                throw std::runtime_error("Install Fallout 4 Creation Kit: PapyrusCompiler.exe was not found");
            if (!std::filesystem::is_regular_file(source))
                throw std::runtime_error("Script source not found: " + source.string());
            if (std::filesystem::file_size(source) > 1024 * 1024)
                throw std::runtime_error("Script source exceeds the 1 MiB compiler limit");
            const auto root = std::filesystem::absolute(assets::AssetResolver::projectAssetRoot(world));
            static std::atomic<unsigned> serial{0};
            const auto staging = root / "scripts" / (".compile-" + std::to_string(SDL_GetTicksNS())
                + "-" + std::to_string(++serial));
            if (!std::filesystem::create_directories(staging))
                throw std::runtime_error("Cannot create isolated Papyrus compiler output directory");
            struct Cleanup
            {
                std::filesystem::path path;
                ~Cleanup() { std::error_code error; std::filesystem::remove_all(path, error); }
            } cleanup{staging};
            std::string imports = utf8(root / "scripts/source");
            if (std::filesystem::is_regular_file(world))
            {
                const auto stack = assets::loadEsmMasterStack(world, data);
                for (auto it = stack.sourcePaths.rbegin(); it != stack.sourcePaths.rend(); ++it)
                {
                    const auto sourceRoot = assets::AssetResolver::projectAssetRoot(*it) / "scripts/source";
                    if (std::filesystem::absolute(*it) != std::filesystem::absolute(world)
                        && std::filesystem::is_directory(sourceRoot))
                        imports += ";" + utf8(sourceRoot);
                }
            }
            imports += ";" + utf8(data / "Scripts/Source/User") + ";" + utf8(base);
            const std::vector<std::string> arguments{utf8(compiler), utf8(source),
                "-i=" + imports,
                "-f=" + utf8(base / "Institute_Papyrus_Flags.flg"), "-o=" + utf8(staging)};
            std::vector<const char*> argv;
            for (const auto& argument : arguments) argv.push_back(argument.c_str());
            argv.push_back(nullptr);
            const auto properties = SDL_CreateProperties();
            SDL_SetPointerProperty(properties, SDL_PROP_PROCESS_CREATE_ARGS_POINTER, argv.data());
            SDL_SetNumberProperty(properties, SDL_PROP_PROCESS_CREATE_STDOUT_NUMBER, SDL_PROCESS_STDIO_APP);
            SDL_SetBooleanProperty(properties, SDL_PROP_PROCESS_CREATE_STDERR_TO_STDOUT_BOOLEAN, true);
            SDL_SetStringProperty(properties, SDL_PROP_PROCESS_CREATE_WORKING_DIRECTORY_STRING,
                utf8(compiler.parent_path()).c_str());
            std::unique_ptr<SDL_Process, decltype(&SDL_DestroyProcess)> process(
                SDL_CreateProcessWithProperties(properties), SDL_DestroyProcess);
            SDL_DestroyProperties(properties);
            if (!process) throw std::runtime_error(SDL_GetError());
            int exitCode = -1;
            auto* output = SDL_GetProcessOutput(process.get());
            const auto started = SDL_GetTicks();
            const auto drain = [&] {
                char buffer[4096];
                for (size_t read; output && (read = SDL_ReadIO(output, buffer, sizeof(buffer))) != 0;)
                    if (result.diagnostics.size() < 1024 * 1024)
                        result.diagnostics.append(buffer, std::min(read, 1024 * 1024 - result.diagnostics.size()));
            };
            while (!SDL_WaitProcess(process.get(), false, &exitCode))
            {
                drain();
                if (SDL_GetTicks() - started > 60000)
                {
                    SDL_KillProcess(process.get(), true);
                    SDL_WaitProcess(process.get(), true, &exitCode);
                    throw std::runtime_error("Papyrus compiler timed out after 60 seconds");
                }
                SDL_Delay(10);
            }
            drain();
            if (exitCode != 0) throw std::runtime_error("Papyrus compilation failed (exit " + std::to_string(exitCode) + ")");
            const auto relative = std::filesystem::path(assets::papyrusScriptPath(name));
            const auto compiled = staging / relative.lexically_relative("scripts");
            if (!std::filesystem::is_regular_file(compiled))
                throw std::runtime_error("Compiler produced no matching PEX; check Scriptname and filename");
            // Publication requires a real FO4 PEX, even when its functions exceed OpenF4's runtime subset.
            std::ifstream input(compiled, std::ios::binary);
            std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(input)), {});
            input.close();
            (void)inspectPex(bytes);
            result.output = root / relative;
            std::filesystem::create_directories(result.output.parent_path());
            publish(compiled, result.output);
            result.success = true;
        }
        catch (const std::exception& error) { result.diagnostics += std::string("\n") + error.what(); }
        return result;
    }

    void createWakeScript(const std::filesystem::path& world, std::string_view name)
    {
        const auto path = scriptSourcePath(world, name);
        if (std::filesystem::exists(path)) throw std::runtime_error("Script source already exists; it was not overwritten");
        std::filesystem::create_directories(path.parent_path());
        std::ofstream output(path, std::ios::binary);
        output << "Scriptname " << name << " extends ObjectReference\n\n"
            "Event OnLoad()\n"
            "    SetMotionType(1, true)\n"
            "    ApplyHavokImpulse(0.0, 0.0, 1.0, 0.0)\n"
            "EndEvent\n";
        output.close();
        if (!output) throw std::runtime_error("Could not write Papyrus source");
    }
}
