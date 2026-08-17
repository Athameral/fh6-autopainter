cmake_minimum_required(VERSION 3.18)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
project(my_taichi_app CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 在顶层 CMakeLists.txt 同级查找 AOT 模块，并把路径传给 C++ 汇编代码。
file(GLOB GRAPH_TCM_FILES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/graphs.tcm"
)
if(NOT GRAPH_TCM_FILES)
    message(FATAL_ERROR "graphs.tcm not found next to the top-level CMakeLists.txt, try export it first.")
endif()
list(GET GRAPH_TCM_FILES 0 GRAPH_TCM_FILE)
file(TO_CMAKE_PATH "${GRAPH_TCM_FILE}" GRAPH_TCM_FILE)

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

# Taichi 自带的 spdlog：应用层日志使用 header-only 模式，避免额外链接库。
add_library(app_spdlog INTERFACE)
target_include_directories(app_spdlog INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/3rd/taichi/external/spdlog/include
)
target_compile_definitions(app_spdlog INTERFACE SPDLOG_USE_STD_FORMAT)

# # 3. 将前面用 llvm-objcopy 转好的 AOT 字节码对象文件打成一个静态资源库
# add_library(aot_assets STATIC metadata.o patch0.o)
# set_target_properties(aot_assets PROPERTIES LINKER_LANGUAGE CXX)

# 4. 主程序 Target（stb_impl.cpp 提供 stb_image/stb_image_write 实现）
add_executable(my_app src/app/main.cpp)
add_executable(test_app src/app/test.cpp)
target_compile_definitions(my_app PRIVATE "GRAPH_TCM_PATH=\"${GRAPH_TCM_FILE}\"")

# imgui
set(IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rd/imgui)

set(IMGUI_SOURCES
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    # backends
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp
)
target_sources(my_app PRIVATE
    ${IMGUI_SOURCES}
    src/app/appui.cpp
    src/app/displaytexture.cpp
    src/app/gpuworker.cpp
    src/app/injector.cpp
)
target_include_directories(my_app PRIVATE
                           ${IMGUI_DIR}
                           ${IMGUI_DIR}/backends
                           # glfw
                           3rd/taichi/external/glfw/include
                           # vulkan 头文件（Taichi 自带，注意目录名是 Vulkan-Headers）
                           3rd/taichi/external/Vulkan-Headers/include
                           3rd/taichi/external/VulkanMemoryAllocator/include
                           # volk：Vulkan 函数指针声明 + 动态加载器
                           3rd/taichi/external/volk
                           # spdlog
                           3rd/taichi/external/spdlog/include
                           # cimg
                           # 3rd/CImg
                           )
# imgui 官方例程（src/app/main.cpp）自带 Volk 支持：
#   定义 IMGUI_IMPL_VULKAN_USE_VOLK 后，imgui_impl_vulkan.h 会 include <volk.h>，
#   volk.h 把 vk* 全部声明为全局函数指针变量（VK_NO_PROTOTYPES），
#   main.cpp 里的 #define VOLK_IMPLEMENTATION 提供这些指针的定义，
#   运行时 volkInitialize()/volkLoadInstance() 加载 vulkan-1.dll。
#   → 不需要安装 Vulkan SDK，也不需要链接 vulkan-1.lib。
# 注意：不要用 IMGUI_IMPL_VULKAN_NO_PROTOTYPES（那只对 imgui 后端生效，
#   main.cpp 自身的 vkCreateInstance 等直接调用仍会报未定义符号）。
target_compile_definitions(my_app PRIVATE IMGUI_IMPL_VULKAN_USE_VOLK)
target_link_libraries(my_app PRIVATE glfw )
target_link_libraries(my_app PRIVATE app_spdlog)


# # 5. 包含 Taichi C/C++ API 的头文件路径（external/include 提供 stb 单头文件库）
target_include_directories(my_app PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/3rd/taichi/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/3rd/taichi/c_api/include
    ${CMAKE_CURRENT_SOURCE_DIR}/3rd/taichi/external/include
)

# # 6. 链接 Taichi 静态库、AOT 资源以及系统的图形驱动库
#    注意：Vulkan 不需要显式链接！Taichi 与 GLFW 都在运行时动态加载 vulkan-1.dll
#    （Taichi 通过 volk，GLFW 在 Windows 上默认行为）。
target_link_libraries(my_app PRIVATE
    taichi_c_api
)
target_link_options(my_app PRIVATE -static -Wl,--gc-sections -s -Wl,--icf=safe
    # main.cpp 与 taichi 静态库内部都编译了 volk（VOLK_IMPLEMENTATION），
    # 两处定义语义相同（全局函数指针，运行时由 volkInitialize 填充），
    # 允许重复定义并取第一个（main.o 在前），否则链接会报 multiple definition。
    -Wl,--allow-multiple-definition)