# Standalone MIT leaf of SkyMP; no server, Skyrim SDK, or game-owned fixtures.
find_package(spdlog CONFIG REQUIRED)
add_library(openf4-papyrus-vm STATIC
    src/papyrus-vm-lib/ActivePexInstance.cpp
    src/papyrus-vm-lib/IGameObject.cpp
    src/papyrus-vm-lib/OpcodesImplementation.cpp
    src/papyrus-vm-lib/Utils.cpp
    src/papyrus-vm-lib/VarValue.cpp
    src/papyrus-vm-lib/VirtualMachine.cpp
    src/papyrus-vm-lib/Fo4Reader.cpp
    support/viet/src/Promise.cpp)
target_include_directories(openf4-papyrus-vm PUBLIC include
    support/viet/include support/overlay_ports/makeid)
target_link_libraries(openf4-papyrus-vm PRIVATE spdlog::spdlog)
target_compile_features(openf4-papyrus-vm PUBLIC cxx_std_20)
if(MSVC)
    target_compile_options(openf4-papyrus-vm PRIVATE /FS)
endif()
