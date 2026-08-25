#include "appui.h"
#include "imgui.h"
#include "taichi/cpp/taichi.hpp"

#include "stb_image.h"

#include <spdlog/spdlog.h>
#include <future>
#include <thread>

#include <filesystem>
#include <fstream>

void App::renderUI()
{
    renderControlPanel();
    renderTargetPanel();
    renderCanvasPanel();

    // 【关键】Taichi 内部资源清理（submitted_cmdbuffers_ 的 fence/semaphore 只有
    // wait_idle 才释放）。双队列后由 worker 自己每步 runtime.wait() 负责（q1 独占，
    // 不被主线程 q0 渲染拖住），主线程无需再 wait。
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
        gpu_worker.setParams(params);
        copyVectorToTarget();
        // 重置计数 + 重分配 canvas + 销毁旧 canvas 显示纹理，让新一轮从干净状态开始。
        // 假设：用户点 Start 前 worker 已 Stop（无在飞 g1/g2/g3 引用旧 canvas）。
        resetStatus();
        gpu_worker.should_exit = false;
        gpu_worker.launch_graph = true;
        gpu_worker.generate_interrupted = false;
        gpu_worker.launch_graph.notify_one();
    }
    if (ImGui::Button("Stop Worker"))
    {
        // launch_graph 必须置 true + notify：worker 可能正沉睡在 launch_graph.wait(false)，
        // 设 false 无法唤醒它（wait 谓词要求值 != false）。置 true 唤醒后，循环顶部
        // 检查 should_exit 退出；run() 末尾会复位 launch_graph。
        gpu_worker.should_exit = false;
        gpu_worker.launch_graph = false;
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
        spdlog::debug(
                  "[image] change detected, before loading: path='{}', old_path='{}', thread={}",
                      target_image_path, old_target_image_path,
                      std::hash<std::thread::id>{}(std::this_thread::get_id()));
        target_image_ready = false;
        old_target_image_path = target_image_path;
        // load image（async 线程只做 CPU 数据准备 + 写 Taichi buffer）
        int img_w = 0, img_h = 0;
        auto fut = std::async(std::launch::async, [&]() {
            spdlog::debug("[image] load begin: path='{}', thread={}", target_image_path,
                          std::hash<std::thread::id>{}(std::this_thread::get_id()));
            auto *data = stbi_loadf_utf8(target_image_path.c_str(), &img_w, &img_h, nullptr, 4);
            spdlog::debug(
                              "[image] stbi_loadf returned: data={}, width={}, height={}, failure_reason='{}'",
                          static_cast<const void *>(data), img_w, img_h,
                          stbi_failure_reason() ? stbi_failure_reason() : "none");

            if (!data || img_w <= 0 || img_h <= 0)
            {
                spdlog::error(
                                  "[image] load failed: path='{}', data={}, width={}, height={}",
                              target_image_path, static_cast<const void *>(data), img_w, img_h);
                if (data)
                    stbi_image_free(data);
                target_image_ready = false;
                return;
            }

            spdlog::debug(
                              "[image] load succeeded, before GPU buffer rebuild: width={}, height={}",
                          img_w, img_h);
            params.canvas_h = img_h;
            params.canvas_w = img_w;
            // gpu_worker.remakeBuffer();
            gpu_worker.setParams(params);
            spdlog::debug("[image] GPU buffers rebuilt");
            target_rgb.resize(img_w * img_h * 3);
            target_alpha_mask.resize(img_w * img_h);
            if (data)
            {
                for (int i = 0; i < img_w * img_h; ++i)
                {
                    target_rgb[i * 3 + 0] = data[i * 4 + 0];
                    target_rgb[i * 3 + 1] = data[i * 4 + 1];
                    target_rgb[i * 3 + 2] = data[i * 4 + 2];
                    target_alpha_mask[i] = data[i * 4 + 3] > 1e-2 ? 1 : 0;
                }
                copyVectorToTarget();
                spdlog::debug("[image] target data copied to GPU buffer");
                stbi_image_free(data);
                spdlog::debug("[image] decoded image memory freed");
            }
            else
            {
                spdlog::error("Failed to load image: {}", target_image_path);
            }
            // end
            target_image_ready = true;
            spdlog::debug("[image] load pipeline completed: target_image_ready=true");
        });
    }

    // 主线程：图片加载完成后，创建/重建显示纹理并上传 target（一次性）
    if (target_image_ready.exchange(false))
    {
        spdlog::debug(
                  "[image] main thread observed ready=true, before display texture upload, thread={}",
                      std::hash<std::thread::id>{}(std::this_thread::get_id()));
        ensureDisplayTextures();
        spdlog::debug(
                  "[image] display textures ensured: canvas={}x{}, target={}x{}",
                      canvas_tex.width(), canvas_tex.height(), target_tex.width(), target_tex.height());
        target_tex.uploadAndRegister(gpu_worker.gpu_buffer.target_origin, runtime);
        spdlog::debug("[image] target display texture upload completed");
        spdlog::debug(
                  "[main] target uploaded, target_desc={}, canvas_desc={}, canvas_valid={}",
                      static_cast<void *>(target_tex.descriptor()), static_cast<void *>(canvas_tex.descriptor()),
                      canvas_tex.is_valid());
    }

    if (target_tex.descriptor() != VK_NULL_HANDLE)
    {
        ImGui::Image(ImTextureRef((ImTextureID)(uintptr_t)target_tex.descriptor()),
                     ImVec2((float)target_tex.width(), (float)target_tex.height()));
    }
    ImGui::End();
}

// canvas 独立窗口：worker 每画完一个形状置位 canvas_ready，这里同步上传最新 canvas
// 到显示纹理（target 同款路径：uploadAndRegister = runtime.wait + 同步拷贝 + fence 等待）。
// exchange(false) 保证一帧只上传一次；上传完成后 renderUI 底部才置 gui_ack，
// 因此 worker 的下一轮 g1 必然在本轮上传完成之后启动，无 g3 写 canvas 竞争。
void App::renderCanvasPanel()
{
    ImGui::Begin("Canvas Panel");
    static uint32_t interval = 0;
    if (gpu_worker.canvas_ready.exchange(false))
    {
        if (interval++ % 100 == 0)
        {
            spdlog::debug("[main] canvas_ready, uploading to display texture...");
                canvas_tex.upload(gpu_worker.gpu_buffer.canvas, runtime);
            }
        }
    }

    if (canvas_tex.descriptor() != VK_NULL_HANDLE)
    {
        const float tex_w = (float)canvas_tex.width();
        const float tex_h = (float)canvas_tex.height();
        // 缩放适配窗口（保持宽高比），方便大图全览
        const float avail_w = ImGui::GetContentRegionAvail().x;
        const float avail_h = ImGui::GetContentRegionAvail().y;
        float scale = 1.0f;
        if (avail_w > 0.0f && avail_h > 0.0f)
        {
            const float sx = avail_w / tex_w;
            const float sy = avail_h / tex_h;
            scale = sx < sy ? sx : sy;
        }
        ImGui::Image(ImTextureRef((ImTextureID)(uintptr_t)canvas_tex.descriptor()),
                     ImVec2(tex_w * scale, tex_h * scale));
        ImGui::Text("canvas %ux%u (%.0f%%)", canvas_tex.width(), canvas_tex.height(), scale * 100.0f);
    }
    else
    {
        ImGui::Text("Canvas not ready. Drag in a target image and press Start Worker.");
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
        canvas_tex.registerTexture(); // canvas 只注册不上传：内容由主线程每步 uploadAndRegister 刷新
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

App::App(GLFWwindow *window, ti::Runtime &runtime, PainterParams &params, const ti::AotModule &aot_module,
         VkPhysicalDevice physical_device, VkDevice device, VkQueue queue, uint32_t queue_family)
    : window(window), runtime(runtime), aot_module(aot_module), params(params),
      gpu_worker(runtime, params, aot_module), vk_physical_device(physical_device), vk_device(device), vk_queue(queue),
      vk_queue_family(queue_family)
{
    // canvas 的显示走 target 同款路径：worker 每步置 canvas_ready，主线程
    // renderUI 消费并 uploadAndRegister 同步上传（见 renderTargetPanel），
    // 不再需要 worker 直接持有显示纹理指针。

    spdlog::info("App initialized with GPUWorker.");
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
        spdlog::warn("Only one file can be dragged in at a time.");
        return;
    }
    assert(count == 1);
    auto *app = (App *)glfwGetWindowUserPointer(window);
    assert(app != nullptr);
    app->setTargetImagePath(paths[0]);
}

float *App::stbi_loadf_utf8(const char *filename, int *x, int *y, int *channels_in_file, int desired_channels)
{
    // filename should be in utf-8, use only on windows.
    std::filesystem::path filePath = std::filesystem::u8path(filename);
    std::vector<uint8_t> buffer;
    std::ifstream file(filePath, std::ios::binary | std::ios::ate);
    if (!file.is_open())
    {
        spdlog::error("failed to open file: {}", filename);
        return nullptr;
    }
    buffer.resize(file.tellg());
    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char *>(buffer.data()), buffer.size());
    file.close();
    return stbi_loadf_from_memory(buffer.data(), buffer.size(), x, y, channels_in_file, desired_channels);
}

void App::copyVectorToTarget() // should be called after resetting params.
{
    auto img_h = gpu_worker.params.canvas_h, img_w = gpu_worker.params.canvas_w;
    // target/valid_mask/target_origin 都是 GPU 高频访问 buffer，使用
    // 一次性 host-visible staging 上传，避免把它们分配到 HOST_VISIBLE 内存。
    auto target_staging = runtime.allocate_ndarray<float>({(uint32_t)img_h, (uint32_t)img_w}, {3}, true);
    auto mask_staging = runtime.allocate_ndarray<int32_t>({(uint32_t)img_h, (uint32_t)img_w}, {}, true);

    target_staging.write(target_rgb);
    target_staging.copy_to(gpu_worker.gpu_buffer.target_origin);
    target_staging.copy_to(gpu_worker.gpu_buffer.target);
    mask_staging.write(target_alpha_mask);
    mask_staging.copy_to(gpu_worker.gpu_buffer.valid_mask);

    // copy_to 是异步提交；staging 是局部对象，必须等 GPU 完成后才能析构。
    spdlog::debug("[image] target buffers uploaded, before runtime.wait()");
    runtime.wait();
}

void App::resetStatus()
{
    // 计数归零（PainterStatus 默认值：n_shapes_drawn=0）
    status = PainterStatus{};
    // 重分配 canvas ndarray + 重绑 g1/g2/g3（调用方须保证 worker 已停下）
    gpu_worker.resetCanvas();
    // 销毁 canvas 显示纹理后立即重建（create+register）。
    canvas_tex.destroy();
    ensureDisplayTextures();
    // 新 VkImage 的 initialLayout=UNDEFINED，内容未定义（驱动常复用显存 → 显示旧 canvas）。
    // 上传全零 staging ndarray 让显示先变黑，直到 worker 第一轮 g3+upload 刷新真实内容。
    if (canvas_tex.is_valid())
    {
        const uint32_t w = gpu_worker.params.canvas_w, h = gpu_worker.params.canvas_h;
        auto zeros = runtime.allocate_ndarray<float>({h, w}, {3}, true);
        std::vector<float> zero_vec((size_t)w * h * 3, 0.0f);
        zeros.write(zero_vec);
        runtime.wait(); // 等 staging write 完成，upload 才能读到零
        canvas_tex.upload(zeros, runtime);
    }
}