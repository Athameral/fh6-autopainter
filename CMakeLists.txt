cmake_minimum_required(VERSION 3.18)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
project(my_taichi_app CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. 强制控制子模块 Taichi 编译为静态库，且只编译 C-API
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(TI_WITH_C_API ON CACHE BOOL "" FORCE)
set(TI_WITH_PYTHON OFF CACHE BOOL "" FORCE) # 关掉 Python 绑定以加快编译
set(TI_WITH_LLVM OFF CACHE BOOL "" FORCE) # 关掉 Taichi 自带的 LLVM 编译器，使用系统自带的
set(TI_WITH_CUDA OFF CACHE BOOL "" FORCE)
set(TI_WITH_METAL OFF CACHE BOOL "" FORCE)
set(TI_WITH_OPENGL OFF CACHE BOOL "" FORCE)

set(TI_WITH_VULKAN ON CACHE BOOL "" FORCE)
set(TI_WITH_LTO ON CACHE BOOL "" FORCE)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
add_compile_options(-ffunction-sections -fdata-sections) # 减小体积
# 2. 引入 3rd/taichi 子目录
add_subdirectory(3rd/taichi)

# # 3. 将前面用 llvm-objcopy 转好的 AOT 字节码对象文件打成一个静态资源库
# add_library(aot_assets STATIC metadata.o patch0.o)
# set_target_properties(aot_assets PROPERTIES LINKER_LANGUAGE CXX)

# 4. 主程序 Target（stb_impl.cpp 提供 stb_image/stb_image_write 实现）
add_executable(my_app src/app/main.cpp)
add_executable(test_app src/app/test.cpp)


# # 5. 包含 Taichi C/C++ API 的头文件路径（external/include 提供 stb 单头文件库）
target_include_directories(my_app PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/3rd/taichi/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/3rd/taichi/c_api/include
    ${CMAKE_CURRENT_SOURCE_DIR}/3rd/taichi/external/include
)

# # 6. 链接 Taichi 静态库、AOT 资源以及系统的图形驱动库
target_link_libraries(my_app PRIVATE
    taichi_c_api
)
target_link_options(my_app PRIVATE -static -Wl,--gc-sections -s -Wl,--icf=safe)