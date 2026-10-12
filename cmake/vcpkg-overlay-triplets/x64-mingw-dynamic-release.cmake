# OpenF4 overlay of vcpkg's community x64-mingw-dynamic-release triplet.
# The stock triplet lets vcpkg's MinGW toolchain find_program() the compiler from PATH,
# so with C:\msys64\mingw64\bin ahead of ucrt64\bin packages silently build against
# MSVCRT (e.g. libpugixml.dll importing msvcrt.dll -> "Entry Point Not Found:
# _ZNSi5seekgESt4fposI9_MbstatetE" next to a UCRT libstdc++). Pin the UCRT64 toolchain.
set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE dynamic)
set(VCPKG_ENV_PASSTHROUGH PATH OPENF4_MINGW_BIN)

set(VCPKG_CMAKE_SYSTEM_NAME MinGW)
set(VCPKG_POLICY_DLLS_WITHOUT_LIBS enabled)

set(VCPKG_BUILD_TYPE release)

if(DEFINED ENV{OPENF4_MINGW_BIN})
    file(TO_CMAKE_PATH "$ENV{OPENF4_MINGW_BIN}" _openf4_mingw_bin)
else()
    set(_openf4_mingw_bin "C:/msys64/ucrt64/bin")
endif()
if(EXISTS "${_openf4_mingw_bin}/x86_64-w64-mingw32-g++.exe")
    file(TO_NATIVE_PATH "${_openf4_mingw_bin}" _openf4_mingw_bin_native)
    set(ENV{PATH} "${_openf4_mingw_bin_native};$ENV{PATH}")
    list(APPEND VCPKG_CMAKE_CONFIGURE_OPTIONS
        "-DCMAKE_C_COMPILER=${_openf4_mingw_bin}/x86_64-w64-mingw32-gcc.exe"
        "-DCMAKE_CXX_COMPILER=${_openf4_mingw_bin}/x86_64-w64-mingw32-g++.exe"
        "-DCMAKE_RC_COMPILER=${_openf4_mingw_bin}/windres.exe")
endif()