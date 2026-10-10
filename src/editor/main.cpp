#include "editor/editor_app.hpp"

#include <SDL3/SDL.h>
#include <exception>
#include <iostream>
#include <string_view>

int main(int argc, char** argv)
{
    std::cout << std::unitbuf;
    std::cerr << std::unitbuf;
    try
    {
        return openf4::editor::runEditor(argc, argv);
    }
    catch (const std::exception& error)
    {
        std::cerr << "OpenF4 editor: " << error.what() << '\n';
        bool unattended = false;
        for (int i = 1; i < argc; ++i)
            unattended |= std::string_view(argv[i]) == "--frames"
                || std::string_view(argv[i]) == "--smoke-test"
                || std::string_view(argv[i]) == "--smoke-bake"
                || std::string_view(argv[i]) == "--bake-lod";
        if (!unattended)
            SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "OpenF4 editor", error.what(), nullptr);
        return 1;
    }
}
