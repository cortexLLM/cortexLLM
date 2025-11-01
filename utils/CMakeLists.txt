# CortexLLM Utils Library
# Logging and utility functions

add_library(cortex_utils STATIC log.cpp chat_template.cpp)

target_include_directories(cortex_utils
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../pkg
)

# Link against ggml for log level enums
target_link_libraries(cortex_utils PUBLIC ggml)

# Set C++ standard (need C++17 for structured bindings in chat_template.cpp)
target_compile_features(cortex_utils PRIVATE cxx_std_17)

# Add debug symbols
target_compile_options(cortex_utils PRIVATE -g)

# Export target for use in parent projects
set_target_properties(cortex_utils PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

