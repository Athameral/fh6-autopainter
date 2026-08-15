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
    Ellipse best_ellipse_cpu;

    ti::ComputeGraph g0, g1, g2, g3;

    std::atomic<bool> should_exit;
    std::atomic<bool> generate_interrupted;
    std::atomic<bool> launch_graph;
    std::atomic<bool> gui_ack;

    std::thread worker_thread;

    GPUWorker(ti::Runtime &runtime, const PainterParams &params, const ti::AotModule &aot_module);
    void setParams(const PainterParams &new_params);
    void remakeBuffer();
    ~GPUWorker();

  private:
    void bind_graph_args();
    void run();
};