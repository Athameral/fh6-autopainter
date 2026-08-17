#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

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

// 绑定全部 graph 参数（与 src/export_graph.py 的 g0~g3 一一对应）
void GPUWorker::bind_graph_args()
    {
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
        std::cerr << "GPUWorker: Starting generation loop..." << std::endl;

        // set our params and buffers
        bind_graph_args();

        // 首次运行（或 buffer 重建后）：清空 canvas。
        // write() 走 staging 异步提交，拷贝命令在同队列 FIFO 排在 g1 之前，天然正确。
        if (!canvas_cleared)
        {
            std::vector<float> zeros((size_t)params.canvas_w * params.canvas_h * 3, 0.0f);
            gpu_buffer.canvas.write(zeros.data(), zeros.size());
            canvas_cleared = true;
            std::cerr << "GPUWorker: canvas cleared (" << params.canvas_w << "x" << params.canvas_h << ")" << std::endl;
        }

        // step 0. sharpen image, optional, can be done on CPU
        for (int step = 0; step < (int)params.total_shapes; ++step)
        {
            std::cerr << "GPUWorker: Generating shape " << step + 1 << " of " << params.total_shapes << std::endl;
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

            // 注意：worker 线程【绝不能】调 runtime.wait()（= vkQueueWaitIdle 等整条队列排空）。
            // 主线程每帧都向同一条队列提交 ImGui 渲染命令，队列永远排不空 → worker 被卡死。
            // 清理 Taichi 内部 submitted_cmdbuffers_（fence/semaphore 泄漏）的责任由主线程承担：
            // 主线程每帧渲染前调 runtime.wait()，它本来就在等 vsync，多等 worker 命令无感，
            // 且 worker 此时阻塞在 gui_ack.wait 上（不会提交新命令），队列稳定，wait 不卡。
            //（原来每 50 步 wait 的方案废弃：仍会被主线程渲染拖住）

            // 把最新 canvas 拷到显示纹理：同一队列，拷贝排在 kernel 之后、主线程
            // ImGui 渲染之前，无额外同步开销（同队列 FIFO 天然有序）。
            if (canvas_display)
            {
                canvas_display->queueUpload(gpu_buffer.canvas, runtime);
            }
            else
            {
                std::cerr << "GPUWorker: canvas_display is null, skipping upload" << std::endl;
            }

            // TODO: write ellipse to buffer and let gui read
            //（best_* read 已移到主线程 wait 之后，避免 host/device 竞争）
            gui_ack = false;
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
    canvas_cleared = false; // 新 buffer 需要重新清零
}