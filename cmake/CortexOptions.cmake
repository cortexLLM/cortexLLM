# CortexOptions.cmake
# Build options configuration for CortexLLM
# =========================================

# Summary of available build options:
# 
# CORTEX_BUILD_UTILS: Build the utils library (logging) [default: ON]
# CORTEX_BUILD_TESTS: Build test programs [default: OFF]
#
# CORTEX_BUILD: Build the cortexllm inference executable [default: ${CORTEX_STANDALONE}]
#
# CORTEX_ALL_WARNINGS: Enable all compiler warnings [default: ON]
# CORTEX_FATAL_WARNINGS: Treat warnings as errors [default: OFF]
#
# Sanitizers:
#   CORTEX_SANITIZE_THREAD: Enable thread sanitizer [default: OFF]
#   CORTEX_SANITIZE_ADDRESS: Enable address sanitizer [default: OFF]
#   CORTEX_SANITIZE_UNDEFINED: Enable undefined behavior sanitizer [default: OFF]

# Print build configuration summary
if(CORTEX_STANDALONE)
    message(STATUS "CortexLLM Build Configuration:")
    message(STATUS "  Version: ${CMAKE_PROJECT_VERSION}")
    message(STATUS "  Build Number: ${BUILD_NUMBER}")
    message(STATUS "  Commit: ${BUILD_COMMIT}")
    message(STATUS "  Compiler: ${BUILD_COMPILER}")
    message(STATUS "  Target: ${BUILD_TARGET}")
    message(STATUS "  Build Utils: ${CORTEX_BUILD_UTILS}")
    message(STATUS "  Build Tests: ${CORTEX_BUILD_TESTS}")
    message(STATUS "  Build cortexllm: ${CORTEX_BUILD}")
endif()

