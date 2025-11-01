# CortexLLM GGUF Inference Executable
# Main executable for running GGUF format model inference

set(TARGET_RUNQ cortexllm)

add_executable(${TARGET_RUNQ} 
    main.cpp 
    gguf-loader.cpp
    gguf_token.cpp
)

# Set include directories
target_include_directories(${TARGET_RUNQ} PRIVATE
    ${CMAKE_SOURCE_DIR}/utils
    ${CMAKE_SOURCE_DIR}/include
)

# Link libraries
target_link_libraries(${TARGET_RUNQ} PRIVATE 
    ggml 
    cortex_utils
    m
)

# C++ standard
target_compile_features(${TARGET_RUNQ} PRIVATE cxx_std_17)

# Debug symbols
target_compile_options(${TARGET_RUNQ} PRIVATE -g)

# Install target
install(TARGETS ${TARGET_RUNQ} RUNTIME)
