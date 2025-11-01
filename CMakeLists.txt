# CortexLLM - GGUF Model Inference Engine
# Build system for LLM inference using GGUF format models
# ============================================================

cmake_minimum_required(VERSION 3.14) # for add_link_options and implicit target directories.
project(cortexLLM VERSION 1.0.0 LANGUAGES C CXX)
include(CheckIncludeFileCXX)

# Build settings
set(CMAKE_WARN_UNUSED_CLI YES)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set default build type
if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

message("CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")

# ============================================================
# Module paths and directories
# ============================================================

# Add path to modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(CORTEX_STANDALONE ON)
else()
    set(CORTEX_STANDALONE OFF)
endif()

# Load unified build information
include(build-info)

# ============================================================
# GGML Configuration
# ============================================================



if (MINGW)
    set(BUILD_SHARED_LIBS_DEFAULT OFF)
else()
    set(BUILD_SHARED_LIBS_DEFAULT ON)
endif()

option(BUILD_SHARED_LIBS "build shared libraries" ${BUILD_SHARED_LIBS_DEFAULT})

# ============================================================
# Platform-specific settings
# ============================================================

if (WIN32)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

if (MSVC)
    add_compile_options("$<$<COMPILE_LANGUAGE:C>:/utf-8>")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/utf-8>")
    add_compile_options("$<$<COMPILE_LANGUAGE:C>:/bigobj>")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/bigobj>")
endif()

# iOS support removed - not needed for cortexLLM

# ============================================================
# Build Options
# ============================================================

# Compiler warnings
option(CORTEX_ALL_WARNINGS "cortexLLM: enable all compiler warnings" ON)
option(CORTEX_FATAL_WARNINGS "cortexLLM: enable -Werror flag" OFF)

# Sanitizers
option(CORTEX_SANITIZE_THREAD    "cortexLLM: enable thread sanitizer"    OFF)
option(CORTEX_SANITIZE_ADDRESS   "cortexLLM: enable address sanitizer"   OFF)
option(CORTEX_SANITIZE_UNDEFINED "cortexLLM: enable undefined sanitizer" OFF)

# Build components
option(CORTEX_BUILD "cortexLLM: build executable" ${CORTEX_STANDALONE})

# CortexLLM specific options
option(CORTEX_BUILD_UTILS "cortexLLM: build utils library (logging)" ON)
option(CORTEX_BUILD_TESTS "cortexLLM: build tests" OFF)

# ============================================================
# CMake Modules and Build Configuration
# ============================================================

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/common.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/CortexOptions.cmake)

if (NOT DEFINED CORTEX_BUILD_NUMBER)
    set(CORTEX_BUILD_NUMBER ${BUILD_NUMBER})
endif()
if (NOT DEFINED CORTEX_BUILD_COMMIT)
    set(CORTEX_BUILD_COMMIT ${BUILD_COMMIT})
endif()
set(CORTEX_INSTALL_VERSION ${CMAKE_PROJECT_VERSION})

# Configure GGML options
set(GGML_ALL_WARNINGS   ${CORTEX_ALL_WARNINGS})
set(GGML_FATAL_WARNINGS ${CORTEX_FATAL_WARNINGS})

# ============================================================
# Sanitizers (if enabled)
# ============================================================

if (NOT MSVC)
    if (CORTEX_SANITIZE_THREAD)
        message(STATUS "Using -fsanitize=thread")

        add_compile_options(-fsanitize=thread)
        link_libraries     (-fsanitize=thread)
    endif()

    if (CORTEX_SANITIZE_ADDRESS)
        message(STATUS "Using -fsanitize=address")

        add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
        link_libraries     (-fsanitize=address)
    endif()

    if (CORTEX_SANITIZE_UNDEFINED)
        message(STATUS "Using -fsanitize=undefined")

        add_compile_options(-fsanitize=undefined)
        link_libraries     (-fsanitize=undefined)
    endif()
endif()

# ============================================================
# Dependencies
# ============================================================

set(GGML_BUILD_NUMBER ${CORTEX_BUILD_NUMBER})
set(GGML_BUILD_COMMIT ${CORTEX_BUILD_COMMIT})
add_subdirectory(pkg/ggml)

if (MINGW)
    # Target Windows 8 for PrefetchVirtualMemory
    add_compile_definitions(_WIN32_WINNT=${GGML_WIN_VER})
endif()

# ============================================================
# CortexLLM Components
# ============================================================

# Build utils library
if (CORTEX_BUILD_UTILS)
    add_subdirectory(utils)
endif()

# Build runq executable
if (CORTEX_BUILD)
    add_subdirectory(src)
endif()

# ============================================================
# Installation
# ============================================================

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(CORTEX_INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR} CACHE PATH "Location of header  files")
set(CORTEX_LIB_INSTALL_DIR     ${CMAKE_INSTALL_LIBDIR}     CACHE PATH "Location of library files")
set(CORTEX_BIN_INSTALL_DIR     ${CMAKE_INSTALL_BINDIR}     CACHE PATH "Location of binary  files")

configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/cortex-config.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/cortex-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cortex
    PATH_VARS CORTEX_INCLUDE_INSTALL_DIR
              CORTEX_LIB_INSTALL_DIR
              CORTEX_BIN_INSTALL_DIR)

write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/cortex-version.cmake
    VERSION ${CORTEX_INSTALL_VERSION}
    COMPATIBILITY SameMajorVersion)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/cortex-config.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/cortex-version.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cortex)

configure_file(cmake/cortex.pc.in
        "${CMAKE_CURRENT_BINARY_DIR}/cortex.pc"
        @ONLY)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/cortex.pc"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
