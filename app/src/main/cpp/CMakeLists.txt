cmake_minimum_required(VERSION 3.22.1)

project("native_llm")

set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)

# Android 是交叉编译，llama.cpp 默认只会生成保守的 ARM 基线代码。
# 当前实验真机为 Snapdragon 8 Elite（SM8750），已确认支持 DotProd 与 i8mm；
# Q4_K_M 的量化矩阵计算可利用这两组指令。后续发布给不同设备时，改用
# 多 variant 的 CPU backend，而不要把这组针对真机的指令当作通用 APK 配置。
set(GGML_CPU_ARM_ARCH "armv8.2-a+dotprod+i8mm" CACHE STRING "" FORCE)

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
