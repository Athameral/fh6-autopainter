#pragma once
#include <atomic>
#include <thread>

#include <taichi/cpp/taichi.hpp>
#include "displaytexture.h"
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
    Ellipse best_ellipse_cpu;

    ti::ComputeGraph g0, g1, g2, g3;

    // canvas 显示纹理（由 App 构造后赋值；worker 线程每步 queueUpload 刷新）。
    // 非 atomic：换图必须先把 worker 停下来（remakeBuffer 会销毁 buffer），
    // 指针在 App 构造后不再变化，worker 首次读取发生在 Start 之后，无并发访问。
    DisplayTexture *canvas_display = nullptr;

    std::atomic<bool> should_exit = false;
    std::atomic<bool> generate_interrupted = false;
    std::atomic<bool> launch_graph = false;
    std::atomic<bool> gui_ack = false;

    // canvas 首次运行前需要清零（allocate_ndarray 不保证初始化为 0，
    // 未初始化显存可能是 NaN/垃圾值 → 显示黑屏或噪声）。
    // remakeBuffer() 重建 buffer 后需重新清零，故在 remakeBuffer 里重置。
    bool canvas_cleared = false;

    std::thread worker_thread;

    GPUWorker(ti::Runtime &runtime, const PainterParams &params, const ti::AotModule &aot_module);
    void setParams(const PainterParams &new_params);
    void remakeBuffer();
    ~GPUWorker();

  private:
    void bind_graph_args();
    void run();
};