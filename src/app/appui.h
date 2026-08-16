#pragma once
#include "displaytexture.h"
#include "gpuworker.h"
#include "structures.h"
#include <imgui.h>
#include <taichi/cpp/taichi.hpp>
#include <GLFW/glfw3.h>
#include <volk.h> // Vulkan 类型 + 全局函数指针（VK_NO_PROTOTYPES），保证本头文件自包含

#include <atomic>
#include <string>

void file_dragin_callback(GLFWwindow *window, int count, const char **paths);

class App
{
  public:
    App(GLFWwindow *window, ti::Runtime &runtime, const PainterParams &params, const ti::AotModule &aot_module,
        VkPhysicalDevice physical_device, VkDevice device, VkQueue queue, uint32_t queue_family);
    ~App();

    void renderUI();
    void setTargetImagePath(const char *path);

  private:
    GLFWwindow *window;
    ti::Runtime &runtime;
    const ti::AotModule &aot_module;
    PainterParams params;

    // 显示纹理（方案 B）：canvas 每步由 worker 线程 queueUpload；target 加载后主线程 uploadAndRegister。
    // 注意声明顺序：canvas_tex 先于 gpu_worker 声明 → 析构时 gpu_worker（join 线程）先析构，纹理后释放。
    DisplayTexture canvas_tex;
    DisplayTexture target_tex;

    GPUWorker gpu_worker;
    std::string target_image_path, old_target_image_path;

    // Vulkan 句柄（与 main.cpp 共享同一个 device/queue）
    VkPhysicalDevice vk_physical_device = VK_NULL_HANDLE;
    VkDevice vk_device = VK_NULL_HANDLE;
    VkQueue vk_queue = VK_NULL_HANDLE;
    uint32_t vk_queue_family = 0;

    std::atomic<bool> target_image_ready{false}; // async 加载线程置位，主线程消费

    void renderControlPanel();
    void renderTargetPanel();
    // 主线程：按需创建/重建两张显示纹理（图片加载/尺寸变化时调用一次）
    void ensureDisplayTextures();
};
