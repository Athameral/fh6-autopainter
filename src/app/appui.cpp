#include "appui.h"
#include "imgui.h"
#include "injector.h"
#include "spdlog/common.h"
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
    renderDebugPanel();

    // 【关键】Taichi 内部资源清理（submitted_cmdbuffers_ 的 fence/semaphore 只有
    // wait_idle 才释放）。双队列后由 worker 自己每步 runtime.wait() 负责（q1 独占，
    // 不被主线程 q0 渲染拖住），主线程无需再 wait。
    // 顺带保证 canvas 纹理内容已更新（拷贝完成）后才注册/渲染。
    // runtime.wait();

    // 通知 worker：本帧 UI 已处理完结果（gui_ack 握手）。
    // worker 的 gui_ack.wait(false) 遇 true 立即通过，不阻塞生成循环。

}

void App::step()
{
    renderUI();
    if (gpu_worker.data_ready) {
        gpu_worker.data_ready = false;
        status.ellipses.push_back(gpu_worker.best_ellipse_cpu);
    }
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
        gpu_worker.launch_graph.notify_one();
        gpu_worker.gui_ack.notify_one(); // 万一 worker 卡在 gui_ack.wait，也唤醒它退出
    }
    if (ImGui::Button("Inject"))
    {
        if (inject_busy_.exchange(true))
        {
            spdlog::warn("[inject-ui] an injection is already running; ignoring the click");
        }
        else
        {
            std::thread([this]() {
                spdlog::info("[inject-ui] worker started: ellipses={} canvas={}x{} template_layers={}",
                             status.ellipses.size(), params.canvas_w, params.canvas_h, params.total_shapes);
                auto result = injector_.writeEllipses(status.ellipses,
                                                      params.canvas_w,
                                                      params.canvas_h,
                                                      params.total_shapes);
                spdlog::info(
                    "[inject-ui] result: success={} pid={} process='{}' locator='{}' requested={} converted={} "
                    "skipped={} written={} cleared={} mesh_paths={} cached_group={} cached_vtable={}",
                    result.success,
                    result.pid,
                    result.process_name,
                    result.locator,
                    result.requested_shapes,
                    result.converted_shapes,
                    result.skipped_shapes,
                    result.written_layers,
                    result.cleared_layers,
                    result.mesh_paths_updated,
                    result.used_cached_group,
                    result.used_cached_vtable);
                if (!result.error.empty())
                    spdlog::error("[inject-ui] injection failed: {}", result.error);
                else
                    spdlog::info("[inject-ui] injection succeeded");
                inject_busy_ = false;
            }).detach();
        }
    }
    if (ImGui::Button("Force Relocate"))
    {
        if (inject_busy_.exchange(true))
        {
            spdlog::warn("[inject-ui] an injection is already running; ignoring the click");
        }
        else
        {
            std::thread([this]() {
                spdlog::info("[inject-ui] force relocate started: template_layers={}", params.total_shapes);
                const bool ok = injector_.Locate(params.total_shapes);
                const auto loc = injector_.location();
                spdlog::info("[inject-ui] force relocate: success={} locator='{}' group=0x{:x} vtable=0x{:x}",
                             ok,
                             loc.locator,
                             static_cast<unsigned long long>(loc.group),
                             static_cast<unsigned long long>(loc.vtable));
                inject_busy_ = false;
            }).detach();
        }
    }
    if (ImGui::CollapsingHeader("Parameters"))
    {
        ImGui::SliderFloat("min_radius", &params.min_radius, 0.1f, 100.0f);
        ImGui::SliderFloat("max_radius", &params.max_radius, 0.001f, 0.2f);
        ImGui::SliderFloat("min_alpha", &params.min_alpha, 0.0f, 1.0f);
        ImGui::SliderFloat("max_alpha", &params.max_alpha, 0.0f, 1.0f);
        ImGui::SliderInt("random_samples", &params.random_samples, 1000, 120000);
        ImGui::SliderInt("sample_bins", &params.sample_bins, 16, 8192);
        ImGui::SliderFloat("sample_leak_ratio", &params.sample_leak_ratio, 0.0f, 1.0f);
        ImGui::SliderInt("blur_size", &params.blur_size, 1, 15);
        // ImGui::SliderInt("canvas_w", reinterpret_cast<int *>(&params.canvas_w), 64, 2048);
        ImGui::Text("canvas_w: %d", params.canvas_w);
        ImGui::Text("canvas_h: %d", params.canvas_h);
        // ImGui::SliderInt("canvas_h", reinterpret_cast<int *>(&params.canvas_h), 64, 2048);
        ImGui::SliderFloat("sharpen_intensity", &params.sharpen_intensity, 0.0f, 5.0f);
        ImGui::SliderInt("mutations_per_round", &params.mutations_per_round, 1000, 20000);
        ImGui::SliderFloat("move_step", &params.move_step, 1.0f, 20.0f);
        ImGui::SliderFloat("radius_step", &params.radius_step, 1.0f, 20.0f);
        ImGui::SliderFloat("theta_step_rad", &params.theta_step_rad, 0.01f, 3.14f);
        ImGui::SliderFloat("alpha_step", &params.alpha_step, 0.01f, 1.0f);
        ImGui::SliderFloat("sample_step_deno", &params.sample_step_deno, 1.f, 10.f);
        ImGui::SliderInt("hill_climb_rounds", &params.hill_climb_rounds, 1, 50);
        ImGui::SliderInt("total_shapes", reinterpret_cast<int *>(&params.total_shapes), 100, 10000);
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
        // ensure 已 register target_tex；这里只 upload 内容。
        target_tex.upload(gpu_worker.gpu_buffer.target_origin, runtime);
        spdlog::debug("[image] target display texture upload completed");
        spdlog::debug(
                  "[main] target uploaded, target_desc={}, canvas_desc={}, canvas_valid={}",
                      static_cast<void *>(target_tex.descriptor()), static_cast<void *>(canvas_tex.descriptor()),
                      canvas_tex.is_valid());
    }

    if (target_tex.descriptor() != VK_NULL_HANDLE)
    {
        static bool fit = true;
        ImGui::Checkbox("Shrink to fit window", &fit);
        drawImage(target_tex, fit);
        ImGui::Text("target %ux%u", target_tex.width(), target_tex.height());
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
        status.n_shapes_drawn++;
        if (interval++ % 3 == 0)
        {
            // canvas 纹理已在 ensureDisplayTextures（拖图时）create+register；
            // 这里只 upload 最新内容。Start 前未拖图时 is_valid()=false → 跳过。
            if (canvas_tex.is_valid())
        {
            spdlog::debug("[main] canvas_ready, uploading to display texture...");
                canvas_tex.upload(gpu_worker.gpu_buffer.canvas, runtime);
            }
        }
    }

    if (canvas_tex.descriptor() != VK_NULL_HANDLE)
    {
        static bool fit = true;
        ImGui::Checkbox("Shrink to fit window", &fit);
        drawImage(canvas_tex, fit);
        ImGui::Text("canvas %ux%u", canvas_tex.width(), canvas_tex.height());
    }
    else
    {
        ImGui::Text("Canvas not ready. Drag in a target image and press Start Worker.");
    }
    ImGui::ProgressBar(float(status.n_shapes_drawn) / float(gpu_worker.params.total_shapes));
    ImGui::End();
}

// 主线程：按需创建/重建两张显示纹理（图片加载完成、尺寸可能变化时调用）。
// 尺寸未变时保留原纹理（descriptor 继续有效）；变了则 create() 内部先 destroy 再重建。
// canvas 与 target 均在此 create + registerTexture；后续内容刷新由 upload 负责。
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
        canvas_tex.registerTexture(); // create 后立即 register；内容由 upload 刷新
        // 立即 upload 一次：register 声明的是 SHADER_READ_ONLY，但新 image 停在
        // UNDEFINED——不先转布局，本帧 renderCanvasPanel 就会采样一张
        // "已注册但从未上传"的纹理（无效用法，严格驱动直接 fault → 闪退）。
        uploadBlank(canvas_tex);
    }
    if (!target_tex.is_valid() || target_tex.width() != w || target_tex.height() != h)
    {
        target_tex.create(vk_physical_device, vk_device, vk_queue, vk_queue_family, w, h, fmt);
        target_tex.registerTexture(); // 与 canvas 统一：ensure 里 register，upload 只拷贝
        // 同上：Start（未拖图）路径下 target 永远等不到内容 upload，必须在此转布局。
        uploadBlank(target_tex);
    }
}

// 新建纹理的首次 upload：staging 不写内容（新分配的 host-visible 内存即初始画面），
// 只为借 upload 内部的 recordCopy 完成 UNDEFINED→TRANSFER_DST→SHADER_READ_ONLY 转换。
void App::uploadBlank(DisplayTexture &tex)
{
    const uint32_t w = gpu_worker.params.canvas_w, h = gpu_worker.params.canvas_h;
    auto staging = runtime.allocate_ndarray<float>({h, w}, {3}, true);
    staging.write(std::vector<float>(w * h * 3, 0.0f));
    if (!staging.is_valid())
    {
        spdlog::error("[display] initial upload: staging allocation failed ({}x{})", w, h);
        return;
    }
    tex.upload(staging, runtime);
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

void App::drawImage(const DisplayTexture &texture, bool fit = false)
{
    float scale = 1.0f;
    const float tex_w = (float)texture.width();
    const float tex_h = (float)texture.height();
    // fit to parent window
    if (fit)
    {
        const float avail_w = ImGui::GetContentRegionAvail().x;
        const float avail_h = ImGui::GetContentRegionAvail().y;
        if (avail_w > 0.0f && avail_h > 0.0f)
        {
            const float sx = avail_w / tex_w;
            const float sy = avail_h / tex_h;
            scale = sx < sy ? sx : sy;
        }
    }
    else
    {
        scale = 1.0f;
    }
    ImGui::Image(ImTextureRef((ImTextureID)(uintptr_t)texture.descriptor()), ImVec2(tex_w * scale, tex_h * scale));
}

void App::renderDebugPanel()
{
    static spdlog::level::level_enum old_log_level = spdlog::level::debug;
    static spdlog::level::level_enum log_level = spdlog::get_level();

    ImGui::Begin("Debug Panel");
    ImGui::RadioButton("Debug", (int*)&log_level, (int)spdlog::level::debug);
    ImGui::RadioButton("Info", (int*)&log_level, (int)spdlog::level::info);
    ImGui::RadioButton("Warning", (int*)&log_level, (int)spdlog::level::warn);
    ImGui::RadioButton("Error", (int*)&log_level, (int)spdlog::level::err);

    if (log_level != old_log_level)
    {
        spdlog::set_level(log_level);
        old_log_level = log_level;
    }

    ImGui::End();
}