#pragma once
#include <atomic>
#include <thread>

#include <taichi/cpp/taichi.hpp>
#include "structures.h"

class PainterGPUBuffer
{
  public:
    ti::Runtime &runtime;
    ti::NdArray<float> canvas, target, target_origin, error_field, error_field_buffer, sampled_pixels;
    ti::NdArray<int32_t> valid_mask, valid_pixels, valid_n_pixels, hist_buffer;
    ti::NdArray<float> best_ellipse, best_ycbcr, best_score;
    PainterGPUBuffer(ti::Runtime &runtime, const PainterParams &params);
    PainterGPUBuffer &operator=(PainterGPUBuffer &&other) noexcept;
};

class GPUWorker
{
  public:
    ti::Runtime &runtime;
    PainterParams params; // a snapshot only.
    PainterGPUBuffer gpu_buffer;
    struct Ellipse best_ellipse_cpu;

    ti::ComputeGraph g0, g_m, g1, g2, g3;

    std::atomic<bool> should_exit = false;
    std::atomic<bool> generate_interrupted = false;
    std::atomic<bool> launch_graph = false;
    std::atomic<bool> gui_ack = false;

    // worker 每画完一个形状置位（g3.launch() 之后）；主线程消费并
    // 用 uploadAndRegister 同步上传 canvas 到显示纹理（target 同款路径）。
    // 消费端 exchange(false) 复位，保证一帧只上传一次最新结果。
    std::atomic<bool> canvas_ready = false;
    std::atomic<bool> data_ready = false; // worker 生成完毕，主线程可读 best_ellipse_cpu

    std::thread worker_thread;

    GPUWorker(ti::Runtime &runtime, const PainterParams &params, const ti::AotModule &aot_module);
    void setParams(const PainterParams &new_params);
    void remakeBuffer();
    // 重分配 canvas ndarray（清零=重新 allocate，不写 zeros；显示路径有 canvas_ready
    // 门控，只 upload g3 画过的内容）并重绑 g1/g2/g3 的 canvas 引用。
    // 调用方须保证 worker 已停下（无在飞 g1/g2/g3 引用旧 canvas buffer）。
    void resetCanvas();
    ~GPUWorker();

  private:
    void bind_graph_args();
    void run();
};