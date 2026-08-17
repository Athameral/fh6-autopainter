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
    PainterParams params;
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

    std::thread worker_thread;

    GPUWorker(ti::Runtime &runtime, const PainterParams &params, const ti::AotModule &aot_module);
    void setParams(const PainterParams &new_params);
    void remakeBuffer();
    ~GPUWorker();

  private:
    void bind_graph_args();
    void run();
};