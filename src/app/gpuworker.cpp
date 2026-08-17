#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

#include <spdlog/spdlog.h>
#include <taichi/cpp/taichi.hpp>
#include <utility>

#include "gpuworker.h"



PainterGPUBuffer::PainterGPUBuffer(ti::Runtime &runtime, const PainterParams &params)
    : runtime(runtime)
{
    canvas = runtime.allocate_ndarray<float>({params.canvas_h, params.canvas_w}, {3}, false);
    target = runtime.allocate_ndarray<float>({params.canvas_h, params.canvas_w}, {3}, false);
    target_origin = runtime.allocate_ndarray<float>({params.canvas_h, params.canvas_w}, {3}, false);
    error_field = runtime.allocate_ndarray<float>({params.canvas_h, params.canvas_w}, {}, false);
    error_field_buffer = runtime.allocate_ndarray<float>({params.canvas_h, params.canvas_w}, {}, false);
    sampled_pixels = runtime.allocate_ndarray<float>({(uint32_t)params.random_samples}, {2}, false);
    hist_buffer = runtime.allocate_ndarray<int32_t>({(uint32_t)params.sample_bins}, {}, false);
    valid_mask = runtime.allocate_ndarray<int32_t>({params.canvas_h, params.canvas_w}, {}, false);
    valid_pixels = runtime.allocate_ndarray<int32_t>({params.canvas_h * params.canvas_w}, {2}, true);
    valid_n_pixels = runtime.allocate_ndarray<int32_t>({1}, {}, true);
    best_ellipse = runtime.allocate_ndarray<float>({1}, {6}, true);
    best_ycbcr = runtime.allocate_ndarray<float>({1}, {3}, true);
    best_score = runtime.allocate_ndarray<float>({1}, {}, true);
}

// move assignment operator
PainterGPUBuffer &PainterGPUBuffer::operator=(PainterGPUBuffer &&other) noexcept
{
    if (this != &other)
    {
        // runtime is a reference member and cannot be rebound
        canvas = std::move(other.canvas);
        target = std::move(other.target);
        target_origin = std::move(other.target_origin);
        error_field = std::move(other.error_field);
        error_field_buffer = std::move(other.error_field_buffer);
        sampled_pixels = std::move(other.sampled_pixels);
        hist_buffer = std::move(other.hist_buffer);
        valid_mask = std::move(other.valid_mask);
        valid_pixels = std::move(other.valid_pixels);
        valid_n_pixels = std::move(other.valid_n_pixels);
        best_ellipse = std::move(other.best_ellipse);
        best_ycbcr = std::move(other.best_ycbcr);
        best_score = std::move(other.best_score);
    }
    return *this;
}

GPUWorker::GPUWorker(ti::Runtime &runtime, const PainterParams &params, const ti::AotModule &aot_module)
    : runtime(runtime), params(params), gpu_buffer(runtime, params), should_exit(false),
      generate_interrupted(false), launch_graph(false), gui_ack(false)
{
    // g0 = aot_module.get_compute_graph("g0");
    g_m = aot_module.get_compute_graph("g_m");
    g1 = aot_module.get_compute_graph("g1");
    g2 = aot_module.get_compute_graph("g2");
    g3 = aot_module.get_compute_graph("g3");

    worker_thread = std::thread(&GPUWorker::run, this);
}

GPUWorker::~GPUWorker()
{
    should_exit = true;
    if (worker_thread.joinable())
        worker_thread.join();
}

// 绑定全部 graph 参数（与 src/export_graph.py 的 g_m/g0~g3 一一对应）
void GPUWorker::bind_graph_args()
    {
        // g_m: 一次性构建 valid 像素列表（target 不变 → 只执行一次）
        g_m["valid_mask"] = gpu_buffer.valid_mask;
        g_m["valid_pixels"] = gpu_buffer.valid_pixels;
        g_m["valid_n_pixels"] = gpu_buffer.valid_n_pixels;

        // g0: 锐化目标图（主循环前一次）
        g0["target_origin"] = gpu_buffer.target_origin;
        g0["target"] = gpu_buffer.target;
        g0["SHARPEN_INTENSITY"] = params.sharpen_intensity;

        // g1: 误差场 → topk 采样 → 随机搜索取最优
        g1["canvas"] = gpu_buffer.canvas;
        g1["target"] = gpu_buffer.target;
        g1["valid_mask"] = gpu_buffer.valid_mask;
        g1["error_field"] = gpu_buffer.error_field;
        g1["error_field_buffer"] = gpu_buffer.error_field_buffer;
        g1["sampled_pixels"] = gpu_buffer.sampled_pixels;
        g1["hist_buffer"] = gpu_buffer.hist_buffer;
        g1["valid_pixels"] = gpu_buffer.valid_pixels;
        g1["valid_n_pixels"] = gpu_buffer.valid_n_pixels;
        g1["best_ellipse"] = gpu_buffer.best_ellipse;
        g1["best_ycbcr"] = gpu_buffer.best_ycbcr;
        g1["best_score"] = gpu_buffer.best_score;

        g1["MIN_RADIUS"] = params.min_radius;
        g1["MAX_RADIUS"] = params.max_radius;
        g1["MIN_ALPHA"] = params.min_alpha;
        g1["MAX_ALPHA"] = params.max_alpha;
        g1["RANDOM_SAMPLES"] = params.random_samples;
        g1["SAMPLE_BINS"] = params.sample_bins;
        g1["SAMPLE_LEAK_RATIO"] = params.sample_leak_ratio;
        g1["BLUR_SIZE"] = params.blur_size;
        g1["SAMPLE_STEP"] = params.sample_step;

        // g2: 爬山变异（每轮原地更新 best_ellipse，多轮复用同一 ndarray）
        g2["best_ellipse"] = gpu_buffer.best_ellipse;
        g2["canvas"] = gpu_buffer.canvas;
        g2["target"] = gpu_buffer.target;
        g2["valid_mask"] = gpu_buffer.valid_mask;
        g2["best_ycbcr"] = gpu_buffer.best_ycbcr;
        g2["best_score"] = gpu_buffer.best_score;

        g2["MUTATIONS_PER_ROUND"] = params.mutations_per_round;
        g2["MOVE_STEP"] = params.move_step;
        g2["RADIUS_STEP"] = params.radius_step;
        g2["THETA_STEP_RAD"] = params.theta_step_rad;
        g2["ALPHA_STEP"] = params.alpha_step;
        g2["SAMPLE_STEP"] = params.sample_step;

        // g3: 把最优椭圆画到 canvas
        g3["best_ellipse"] = gpu_buffer.best_ellipse;
        g3["best_ycbcr"] = gpu_buffer.best_ycbcr;
        g3["canvas"] = gpu_buffer.canvas;
}

void GPUWorker::run()
{
    while (!should_exit)
    {
        launch_graph.wait(false); // Wait until launch_graph is set to true
        spdlog::debug("GPUWorker: Starting generation loop...");

        // set our params and buffers
        bind_graph_args();

        // g_m：构建 valid 像素列表（valid_pixels/valid_n_pixels）。
        // target 不变 → 列表不变，每次 Start 执行一次即可（remakeBuffer 重建 buffer
        // 后需要重新构建，故放在这里而非构造函数）。同队列 FIFO，先于 g1 执行。
        g_m.launch();
        // runtime.wait(); // 调试：等 g_m 完成以便读回验证

        // 【调试探针】读回 valid 相关 buffer，验证 g_m 是否填充了 valid_pixels。
        if (false)
        {
            int32_t n_valid = 0;
            gpu_buffer.valid_n_pixels.read(&n_valid, 1);
            std::vector<int32_t> vp(6);
            gpu_buffer.valid_pixels.read(vp.data(), vp.size());
            spdlog::debug(
                          "[dbg] g_m: valid_n_pixels={} first_px=({}, {}) ({}, {}) ({}, {})",
                          n_valid, vp[0], vp[1], vp[2], vp[3], vp[4], vp[5]);
        }

        // step 0. sharpen image, optional, can be done on CPU
        for (int step = 0; step < (int)params.total_shapes; ++step)
        {
            spdlog::debug("GPUWorker: Generating shape {} of {}", step + 1, params.total_shapes);
            if (generate_interrupted || should_exit)
            {
                generate_interrupted = false;
                break;
            }

            // step 1. generate random ellipses, evaluate, and pick the best one
            // i.e. launch graph 1
            g1.launch();

            // step 2. mutation hill climing
            // i.e. launch graph 2, for multiple rounds
            for (int r = 0; r < params.hill_climb_rounds; ++r)
            {
                g2.launch();
            }

            // step 3. draw the best ellipse on canvas
            // and write results to buffer, wait for reading.
            g3.launch();

            // 【双队列】worker 独占 q1（compute queue）：runtime.wait() 只排空
            // 自己队列的命令（kernel + staging 拷贝），不被主线程 q0 的渲染拖住；
            // 同时回收 Taichi 内部 submitted_cmdbuffers_（fence/semaphore 泄漏根治）。
            // wait 返回 = q1 全部完成 = canvas 写入对后续 host 提交可见（fence 语义），
            // 主线程拿到 canvas_ready 后在 q0 拷贝/渲染即安全，无需额外 GPU 同步。
            runtime.wait();

            // 【调试探针】每步读回全部关键 buffer，确认数据链路哪一环断了。
            // 所有 ndarray 已临时设 host accessible，runtime.wait() 后可直接 read。
            if (false)
            {
                const uint32_t W = params.canvas_w, H = params.canvas_h;
                std::vector<float> canvas_cpu(gpu_buffer.canvas.scalar_count());
                gpu_buffer.canvas.read(canvas_cpu);
                auto px = [&](uint32_t x, uint32_t y) {
                    const size_t i = ((size_t)y * W + x) * 3;
                    return canvas_cpu[i];
                };
                float score = 0.0f;
                gpu_buffer.best_score.read(&score, 1);
                std::vector<float> ell(6), ycc(3), sp(6), ef(16);
                gpu_buffer.best_ellipse.read(ell.data(), 6);
                gpu_buffer.best_ycbcr.read(ycc.data(), 3);
                gpu_buffer.sampled_pixels.read(sp.data(), 6);
                gpu_buffer.error_field.read(ef.data(), 16);
                float ef_min = 1e30f, ef_max = -1e30f;
                for (float v : ef) {
                    if (v < ef_min) ef_min = v;
                    if (v > ef_max) ef_max = v;
                }
                spdlog::debug(
                        "[dbg] step {} score={} ell=({}, {}, {}, {}, {}, {}) ycc=({}, {}, {}) sp0=({}, {}) "
                        "sp1=({}, {}) ef[0..15]min={} max={} center=({}, {}, {}) corners=({}, {}, {}, {})",
                    step, score, ell[0], ell[1], ell[2], ell[3], ell[4], ell[5], ycc[0], ycc[1], ycc[2], sp[0],
                    sp[1], sp[2], sp[3], ef_min, ef_max, px(W / 2, H / 2), px(W / 2 + 1, H / 2),
                    px(W / 2, H / 2 + 1), px(0, 0), px(W - 1, 0), px(0, H - 1), px(W - 1, H - 1));
            }
            gui_ack = false;

            // 通知主线程：canvas 已更新。主线程（renderUI）消费后会用
            // uploadAndRegister 同步上传到显示纹理（target 同款路径，见 appui.cpp）。
            // 随后主线程置 gui_ack 才放行下一轮——因此下一次 g1 必然发生在
            // 本次上传完成之后，g3 写 canvas 与上传拷贝不存在竞争。
            canvas_ready = true;
            canvas_ready.notify_all();

            // TODO: write ellipse to buffer and let gui read
            //（best_* read 已移到主线程 wait 之后，避免 host/device 竞争）
            gui_ack.wait(false); // Wait for GUI to acknowledge that it has processed the results
        }

        // 一轮完成：复位 launch_graph，等用户再次 Start（否则 wait(false) 立即通过、无限重跑）
        launch_graph = false;
    }
    return;
}

void GPUWorker::setParams(const PainterParams &new_params)
{
    params = new_params;
    remakeBuffer();
}

void GPUWorker::remakeBuffer()
{
    gpu_buffer = PainterGPUBuffer(runtime, params);
    bind_graph_args();
}