cmake_minimum_required(VERSION 3.22.1)

project("native_llm")

set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)

add_subdirectory(llama.cpp)

add_library(
        native_llm
        SHARED
        bridge.cpp
)

find_library(log-lib log)

target_link_libraries(
        native_llm
        PRIVATE
        llama
        ${log-lib}
)