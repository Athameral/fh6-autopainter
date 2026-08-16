#include "appui.h"
#include "imgui.h"
#include "taichi/cpp/taichi.hpp"

#include "stb_image.h"

#include <future>
#include <iostream>


void App::renderUI()
{
    renderControlPanel();
    renderTargetPanel();

    // 【关键】Taichi 内部资源清理（submitted_cmdbuffers_ 的 fence/semaphore 只有
    // wait_idle 才释放）。worker 线程不能 wait（会被主线程渲染命令拖住），
    // 所以由主线程每帧 wait：此时 worker 阻塞在 gui_ack.wait 上不会提交新命令，
    // 队列稳定，wait 只等 worker 上一步的命令完成（主线程本来就在等 vsync，无感）。
    // 顺带保证 canvas 纹理内容已更新（拷贝完成）后才注册/渲染。
    // runtime.wait();

    // 通知 worker：本帧 UI 已处理完结果（gui_ack 握手）。
    // worker 的 gui_ack.wait(false) 遇 true 立即通过，不阻塞生成循环。
    gpu_worker.gui_ack = true;
    gpu_worker.gui_ack.notify_one();
}

void App::renderControlPanel()
{
    ImGui::Begin("Control Panel");
    ImGui::Text("GPU Worker Status: should_exit %s", gpu_worker.should_exit ? "true" : "false");
    ImGui::Text("generate_interrupted: %s", gpu_worker.generate_interrupted ? "true" : "false");
    ImGui::Text("launch_graph: %s", gpu_worker.launch_graph ? "true" : "false");
    ImGui::Text("gui_ack: %s", gpu_worker.gui_ack ? "true" : "false");
    ImGui::Text("target_image_ready: %s", target_image_ready ? "true" : "false");
    if (ImGui::Button("Start Worker"))
    {
        gpu_worker.should_exit = false;
        gpu_worker.launch_graph = true;
        gpu_worker.launch_graph.notify_one();
    }
    if (ImGui::Button("Stop Worker"))
    {
        // launch_graph 必须置 true + notify：worker 可能正沉睡在 launch_graph.wait(false)，
        // 设 false 无法唤醒它（wait 谓词要求值 != false）。置 true 唤醒后，循环顶部
        // 检查 should_exit 退出；run() 末尾会复位 launch_graph。
        gpu_worker.should_exit = true;
        gpu_worker.launch_graph = true;
        gpu_worker.generate_interrupted = true;
        gpu_worker.launch_graph.notify_all();
        gpu_worker.gui_ack.notify_all(); // 万一 worker 卡在 gui_ack.wait，也唤醒它退出
    }
    ImGui::End();
}

void App::renderTargetPanel()
{
    ImGui::Begin("Target Panel");
    ImGui::Text("Target Image: %s", target_image_path.c_str());
    if (target_image_path != old_target_image_path)
    {
        target_image_ready = false;
        old_target_image_path = target_image_path;
        // load image（async 线程只做 CPU 数据准备 + 写 Taichi buffer）
        int img_w, img_h;
        auto fut = std::async(std::launch::async, [&]() {
            auto *data = stbi_loadf(target_image_path.c_str(), &img_w, &img_h, nullptr, 4);
            gpu_worker.params.canvas_h = img_h;
            gpu_worker.params.canvas_w = img_w;
            gpu_worker.remakeBuffer();
            std::vector<float> target_rgb(img_w * img_h * 3);
            std::vector<int32_t> target_alpha(img_w * img_h);
            if (data)
            {
                for (int i = 0; i < img_w * img_h; ++i)
                {
                    target_rgb[i * 3 + 0] = data[i * 4 + 0];
                    target_rgb[i * 3 + 1] = data[i * 4 + 1];
                    target_rgb[i * 3 + 2] = data[i * 4 + 2];
                    target_alpha[i] = data[i * 4 + 3] > 1e-2 ? 1 : 0;
                }
                gpu_worker.gpu_buffer.target_origin.write(target_rgb.data(), target_rgb.size());
                gpu_worker.gpu_buffer.valid_mask.write(target_alpha.data(), target_alpha.size());
                stbi_image_free(data);
            }
            else
            {
                std::cerr << "Failed to load image: " << target_image_path << std::endl;
            }
            // end
            target_image_ready = true;
        });
    }

    // 主线程：图片加载完成后，创建/重建显示纹理并上传 target（一次性）
    if (target_image_ready.exchange(false))
    {
        ensureDisplayTextures();
        target_tex.uploadAndRegister(gpu_worker.gpu_buffer.target_origin, runtime);
        std::cerr << "[main] target uploaded, target_desc=" << (void *)target_tex.descriptor()
                  << " canvas_desc=" << (void *)canvas_tex.descriptor()
                  << " canvas_valid=" << canvas_tex.is_valid() << std::endl;
    }

    if (target_tex.descriptor() != VK_NULL_HANDLE)
    {
        ImGui::Image(ImTextureRef((ImTextureID)(uintptr_t)target_tex.descriptor()),
                     ImVec2((float)target_tex.width(), (float)target_tex.height()));
    }
    if (canvas_tex.descriptor() != VK_NULL_HANDLE)
    {
        ImGui::Image(ImTextureRef((ImTextureID)(uintptr_t)canvas_tex.descriptor()),
                     ImVec2((float)canvas_tex.width(), (float)canvas_tex.height()));
    }
    ImGui::End();
}

// 主线程：按需创建/重建两张显示纹理（图片加载完成、尺寸可能变化时调用）。
// 尺寸未变时保留原纹理（descriptor 继续有效）；变了则 create() 内部先 destroy 再重建。
void App::ensureDisplayTextures()
{
    const uint32_t w = gpu_worker.params.canvas_w;
    const uint32_t h = gpu_worker.params.canvas_h;
    if (w == 0 || h == 0)
        return;

    const VkFormat fmt = VK_FORMAT_R32G32B32_SFLOAT; // 与 float[3] RGB ndarray 布局一致
    if (!canvas_tex.is_valid() || canvas_tex.width() != w || canvas_tex.height() != h)
    {
        canvas_tex.create(vk_physical_device, vk_device, vk_queue, vk_queue_family, w, h, fmt);
        canvas_tex.registerTexture(); // canvas 只注册不上传：内容由 worker 每步 queueUpload 刷新
    }
    if (!target_tex.is_valid() || target_tex.width() != w || target_tex.height() != h)
    {
        target_tex.create(vk_physical_device, vk_device, vk_queue, vk_queue_family, w, h, fmt);
    }
}

void App::setTargetImagePath(const char *path)
{
    target_image_path = path;
}

App::App(GLFWwindow *window, ti::Runtime &runtime, const PainterParams &params, const ti::AotModule &aot_module,
         VkPhysicalDevice physical_device, VkDevice device, VkQueue queue, uint32_t queue_family)
    : window(window), runtime(runtime), aot_module(aot_module), params(params),
      gpu_worker(runtime, params, aot_module), vk_physical_device(physical_device), vk_device(device), vk_queue(queue),
      vk_queue_family(queue_family)
{
    // 让 worker 线程每步把最新 canvas 刷到显示纹理（worker 此刻阻塞在 launch_graph.wait，
    // 首次读取发生在用户点 Start 之后；换图必须先停 worker，指针此后不再变化）
    gpu_worker.canvas_display = &canvas_tex;

    std::cout << "App initialized with GPUWorker." << std::endl;
    glfwSetWindowUserPointer(window, this);
    glfwSetDropCallback(window, file_dragin_callback);
}

App::~App()
{
    // 唤醒 worker 线程并等待其退出（gpu_worker 析构会 join；这里先保证它不被阻塞）
    gpu_worker.should_exit = true;
    gpu_worker.launch_graph = true;
    gpu_worker.gui_ack = true;
    gpu_worker.launch_graph.notify_all();
    gpu_worker.gui_ack.notify_all();
    // canvas_tex / target_tex 在成员析构阶段（gpu_worker 之后）由 DisplayTexture::~DisplayTexture
    // 自行 vkDeviceWaitIdle + 释放，无需在此处理。
}

void file_dragin_callback(GLFWwindow *window, int count, const char **paths)
{
    assert(window != nullptr);
    if (count > 1)
    {
        std::cerr << "Only one file can be dragged in at a time." << std::endl;
        return;
    }
    assert(count == 1);
    auto *app = (App *)glfwGetWindowUserPointer(window);
    assert(app != nullptr);
    app->setTargetImagePath(paths[0]);
}