#pragma once
#include "displaytexture.h"
#include "gpuworker.h"
#include "injector.h"
#include "structures.h"

#include "stb_image.h"
#include <imgui.h>
#include <taichi/cpp/taichi.hpp>
#include <GLFW/glfw3.h>
#include <vector>
#include <volk.h> // Vulkan 类型 + 全局函数指针（VK_NO_PROTOTYPES），保证本头文件自包含

#include <atomic>
#include <string>

void file_dragin_callback(GLFWwindow *window, int count, const char **paths);

class App
{
  public:
    App(GLFWwindow *window, ti::Runtime &runtime, PainterParams &params, const ti::AotModule &aot_module,
        VkPhysicalDevice physical_device, VkDevice device, VkQueue queue, uint32_t queue_family);
    ~App();

    void renderUI();
    void setTargetImagePath(const char *path);
    void step();

  private:
    // 重置 PainterStatus（计数归零）+ 重分配 canvas buffer + 销毁 canvas 显示纹理
    // （下帧 ensureDisplayTextures 重建+register，canvas_ready 时 upload）。
    // 不动 target_tex（target 内容/尺寸未变）。调用方须保证 worker 已停下。
    void resetStatus();
    GLFWwindow *window;
    ti::Runtime &runtime;
    const ti::AotModule &aot_module;
    PainterParams &params;
    struct PainterStatus
    {
        uint32_t n_shapes_drawn = 0;
        std::vector<struct Ellipse> ellipses;
    } status;

    std::vector<float> target_rgb;
    std::vector<int32_t> target_alpha_mask;

    // 显示纹理：canvas 与 target 均走 uploadAndRegister（主线程同步上传，target 同款路径）。
    // canvas 由 worker 每步置 canvas_ready 通知后主线程消费上传；target 加载后上传一次。
    // 注意声明顺序：canvas_tex 先于 gpu_worker 声明 → 析构时 gpu_worker（join 线程）先析构，纹理后释放。
    DisplayTexture canvas_tex;
    DisplayTexture target_tex;

    GPUWorker gpu_worker;

    // 外部内存注入器：App 长期持有，跨请求复用进程句柄与定位缓存。
    // inject_busy_ 防重入：注入/重定位进行中时忽略新的按钮点击。
    fh6::injector::Injector injector_;
    std::atomic<bool> inject_busy_{false};

    std::string target_image_path, old_target_image_path;
    // always in utf-8.

    // Vulkan 句柄（与 main.cpp 共享同一个 device/queue）
    VkPhysicalDevice vk_physical_device = VK_NULL_HANDLE;
    VkDevice vk_device = VK_NULL_HANDLE;
    VkQueue vk_queue = VK_NULL_HANDLE;
    uint32_t vk_queue_family = 0;

    std::atomic<bool> target_image_ready{false}; // async 加载线程置位，主线程消费

    void renderControlPanel();
    void renderTargetPanel();
    void renderCanvasPanel();
    void renderDebugPanel();
    // 主线程：按需创建/重建两张显示纹理（图片加载/尺寸变化时调用一次）
    void ensureDisplayTextures();
    // 新建纹理的首次 upload：唯一目的是完成 UNDEFINED→SHADER_READ_ONLY 布局转换，
    // 消除"已注册但从未上传"的中间状态（该状态下被采样是无效用法，严格驱动直接 fault）。
    void uploadBlank(DisplayTexture &tex);
    void copyVectorToTarget();

    static float *stbi_loadf_utf8(const char *filename, int *x, int *y, int *channels_in_file,
                                  int desired_channels);
    static void drawImage(const DisplayTexture &texture, bool fit);
};
