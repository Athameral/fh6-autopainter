// main.cpp — 用 Taichi C++ API 加载 graphs.tcm，复现 tests/test_05.py 的
// "Geometrize" 流程：误差场 → 重要性采样 → 随机生成+挑选最优椭圆 (g1)
// → 多轮爬山变异 (g2) → 叠加到画布 (g3)。
//
// 用法（需在项目根目录运行，或使用绝对路径）：
//   my_app --shapes 300 --image tests/pics/01.png
//
// 说明：
//   - PNG 解码用 stb_image（3rd/taichi/external/include，子模块自带）
//   - PIL LANCZOS 缩放近似为双线性缩放（stb 不提供 resize）
//   - sharpen_kernel 未导出进 graph，因此在 CPU 端实现相同的 unsharp mask
#include <taichi/cpp/taichi.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <spdlog/spdlog.h>
#include "stb_image.h"
#include "stb_image_write.h"
#include <windows.h>

// ── 配置（与 tests/test_05.py 保持一致）────────────────────────────────
constexpr int W = 1024;
constexpr int H = 662;
constexpr int RANDOM_SAMPLES = 120000;
constexpr int HILL_CLIMB_ROUNDS = 36;
constexpr int MUTATIONS_PER_ROUND = 8000;
constexpr float THETA_STEP_RAD = 0.523599f;  // ≈ 30°
constexpr float ALPHA_STEP = 0.15f;
constexpr float MIN_RADIUS = 1.0f;
constexpr float MAX_RADIUS_FRAC = 0.05f;
constexpr float MIN_ALPHA = 0.3f;
constexpr float MAX_ALPHA = 1.0f;
constexpr int SAVE_EVERY = 50;
constexpr int MAX_ATTEMPTS = 12;  // rejection sampling 重试次数

// ── 小工具 ──────────────────────────────────────────────────────────────

// 双线性采样（近似 PIL LANCZOS 缩放）。img 为 (h, w, 3) 行优先 RGB，
// 坐标 x∈[0,w-1], y∈[0,h-1]。
static void sample_bilinear(const float *img, int w, int h, float x, float y,
                            float out[3]) {
  int x0 = (int)std::floor(x);
  int y0 = (int)std::floor(y);
  int x1 = std::min(x0 + 1, w - 1);
  int y1 = std::min(y0 + 1, h - 1);
  float fx = x - x0;
  float fy = y - y0;
  const float *p00 = img + (size_t)(y0 * w + x0) * 3;
  const float *p10 = img + (size_t)(y0 * w + x1) * 3;
  const float *p01 = img + (size_t)(y1 * w + x0) * 3;
  const float *p11 = img + (size_t)(y1 * w + x1) * 3;
  for (int c = 0; c < 3; ++c) {
    float top = p00[c] * (1.0f - fx) + p10[c] * fx;
    float bot = p01[c] * (1.0f - fx) + p11[c] * fx;
    out[c] = top * (1.0f - fy) + bot * fy;
  }
}

// 读取 PNG → 缩放到 (W, H) → float RGB [0,1]，写入 out（(H,W,3) 行优先）。
static bool load_and_resize_image(const std::string &path, int W, int H,
                                  std::vector<float> &out) {
  int src_w = 0, src_h = 0, n = 0;
  unsigned char *data = stbi_load(path.c_str(), &src_w, &src_h, &n, 3);
  if (data == nullptr) {
    spdlog::error("无法读取图片: {} ({})", path, stbi_failure_reason());
    return false;
  }

  out.assign((size_t)H * W * 3, 0.0f);
  std::vector<float> src((size_t)src_h * src_w * 3);
  for (size_t i = 0; i < src.size(); ++i) {
    src[i] = data[i] / 255.0f;
  }
  stbi_image_free(data);

  for (int y = 0; y < H; ++y) {
    // 中心对齐的源坐标
    float sy = (y + 0.5f) * (float)src_h / (float)H - 0.5f;
    sy = std::max(0.0f, std::min((float)(src_h - 1), sy));
    for (int x = 0; x < W; ++x) {
      float sx = (x + 0.5f) * (float)src_w / (float)W - 0.5f;
      sx = std::max(0.0f, std::min((float)(src_w - 1), sx));
      sample_bilinear(src.data(), src_w, src_h, sx, sy,
                      &out[((size_t)y * W + x) * 3]);
    }
  }
  return true;
}

// 背景色 = 四边像素平均（对应 load_target 里的 edge.mean()）
static void compute_edge_bg(const float *rgb, int W, int H, float bg[3]) {
  double sum[3] = {0, 0, 0};
  int count = 0;
  auto add_pixel = [&](int y, int x) {
    const float *p = rgb + ((size_t)y * W + x) * 3;
    for (int c = 0; c < 3; ++c) sum[c] += p[c];
    ++count;
  };
  for (int x = 0; x < W; ++x) {
    add_pixel(0, x);
    add_pixel(H - 1, x);
  }
  for (int y = 0; y < H; ++y) {
    add_pixel(y, 0);
    add_pixel(y, W - 1);
  }
  for (int c = 0; c < 3; ++c) bg[c] = (float)(sum[c] / count);
}

// Unsharp mask（复现 src/kernels/sharpening.py 的 sharpen_kernel，
// 3×3 box blur + intensity 增益，逐 RGB 通道）
static void sharpen_unsharp(const std::vector<float> &in,
                            std::vector<float> &out, int W, int H,
                            float intensity) {
  out.assign((size_t)H * W * 3, 0.0f);
  for (int y = 0; y < H; ++y) {
    int ym = std::max(y - 1, 0);
    int yp = std::min(y + 1, H - 1);
    for (int x = 0; x < W; ++x) {
      int xm = std::max(x - 1, 0);
      int xp = std::min(x + 1, W - 1);
      for (int c = 0; c < 3; ++c) {
        float center = in[((size_t)y * W + x) * 3 + c];
        float blur = (in[((size_t)ym * W + xm) * 3 + c] +
                      in[((size_t)ym * W + x) * 3 + c] +
                      in[((size_t)ym * W + xp) * 3 + c] +
                      in[((size_t)y * W + xm) * 3 + c] + center +
                      in[((size_t)y * W + xp) * 3 + c] +
                      in[((size_t)yp * W + xm) * 3 + c] +
                      in[((size_t)yp * W + x) * 3 + c] +
                      in[((size_t)yp * W + xp) * 3 + c]) /
                     9.0f;
        float v = center + intensity * (center - blur);
        out[((size_t)y * W + x) * 3 + c] = std::min(1.0f, std::max(0.0f, v));
      }
    }
  }
}

// 保存预览 PNG（canvas 为 (H,W,3) 行优先 RGB，y=0 在顶部，与 stb 一致）
static void save_preview(const std::string &path, const float *canvas, int W,
                         int H) {
  std::vector<unsigned char> rgb8((size_t)H * W * 3);
  for (size_t i = 0; i < rgb8.size(); ++i) {
    rgb8[i] = (unsigned char)(std::min(1.0f, std::max(0.0f, canvas[i])) * 255.0f +
                              0.5f);
  }
  if (!stbi_write_png(path.c_str(), W, H, 3, rgb8.data(), W * 3)) {
    spdlog::error("保存 PNG 失败: {}", path);
  } else {
    spdlog::info("预览已保存: {}", path);
  }
}

// ── 主流程 ──────────────────────────────────────────────────────────────
int main(int argc, char **argv) {
    
  int max_shapes = 2500;
  std::string image_path = "C:/Users/athameral/Downloads/Mushoku_Tensei_bd1.jpg";
  SetConsoleOutputCP(CP_UTF8);
  SetConsoleCP(CP_UTF8);
  // 简易 CLI 解析
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--shapes") == 0 && i + 1 < argc) {
      max_shapes = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--image") == 0 && i + 1 < argc) {
      image_path = argv[++i];
    } else if (std::strcmp(argv[i], "--help") == 0) {
      spdlog::info("用法: my_app [--shapes N] [--image path.png]");
      return 0;
    }
  }
  if (max_shapes <= 0) {
    spdlog::error("--shapes 必须为正整数");
    return 1;
  }

  const float diag = std::sqrt((float)(W * W) + (float)(H * H));

  // ── 1. 加载目标图，初始化画布 ──
  std::vector<float> target_origin;  // (H,W,3) RGB [0,1]
  if (!load_and_resize_image(image_path, W, H, target_origin)) {
    return 1;
  }

  float bg[3];
  compute_edge_bg(target_origin.data(), W, H, bg);
  spdlog::info("画布底色: ({}, {}, {}) (RGB)", bg[0], bg[1], bg[2]);
  spdlog::info("目标图: {} ({}x{})", image_path, W, H);

  // 锐化目标图（Python 端在 main() 里调用
  // sharpen_kernel(target_origin, target, 1.0)）
  std::vector<float> target;  // 锐化后的目标
  sharpen_unsharp(target_origin, target, W, H, 1.0f);

  std::vector<float> canvas_init((size_t)H * W * 3);
  for (size_t i = 0; i < (size_t)H * W; ++i) {
    canvas_init[i * 3 + 0] = bg[0];
    canvas_init[i * 3 + 1] = bg[1];
    canvas_init[i * 3 + 2] = bg[2];
  }

  // ── 2. Runtime / AOT 加载 ──
  ti::Runtime runtime(TI_ARCH_VULKAN);
  ti::AotModule aot = runtime.load_aot_module("graphs.tcm");
  if (!aot.is_valid()) {
    spdlog::error("加载 graphs.tcm 失败（请确认运行目录下存在该文件）");
    return 1;
  }
  ti::ComputeGraph g1 = aot.get_compute_graph("g1");
  ti::ComputeGraph g2 = aot.get_compute_graph("g2");
  ti::ComputeGraph g3 = aot.get_compute_graph("g3");
  spdlog::info("成功加载 AOT 模块和计算图 g1/g2/g3");

  // ── 3. 分配 ndarray（一次性，复用）──
  // 对应 Python:
  //   canvas       = ti.ndarray(ti.math.vec3, shape=(H, W))
  //   error_field  = ti.ndarray(ti.f32,       shape=(H, W))
  //   sampled_px   = ti.ndarray(ti.math.vec2, shape=(RANDOM_SAMPLES,))
  //   best_ellipse = ti.ndarray(ti.types.vector(6, ti.f32), shape=(1,))
  //   best_ycbcr   = ti.ndarray(ti.math.vec3, shape=(1,))
  //   best_score   = ti.ndarray(ti.f32,       shape=(1,))
  //
  // 内存布局关键点（性能！）：
  //   - kernel 高频读写的画布/目标/误差场/采样点必须放 device-local（VRAM），
  //     host_access=true 会强制 HOST_VISIBLE，独立显卡上等于放系统内存，
  //     GPU 每次访问都走 PCIe，kernel 明显变慢（且 GPU 一直"忙"在等内存）。
  //   - 需要 host 读写的只有：初始化用的 staging、每步读回的小结果缓冲。
  ti::NdArray<float> canvas = runtime.allocate_ndarray<float>(
      {(uint32_t)H, (uint32_t)W}, {3}, /*host_access=*/false);
  ti::NdArray<float> target_np = runtime.allocate_ndarray<float>(
      {(uint32_t)H, (uint32_t)W}, {3}, false);
  ti::NdArray<float> error_field = runtime.allocate_ndarray<float>(
      {(uint32_t)H, (uint32_t)W}, {}, false);
  ti::NdArray<float> sampled_pixels = runtime.allocate_ndarray<float>(
      {(uint32_t)RANDOM_SAMPLES}, {2}, false);
  ti::NdArray<float> best_ellipse =
      runtime.allocate_ndarray<float>({1}, {6}, true);
  ti::NdArray<float> best_ycbcr =
      runtime.allocate_ndarray<float>({1}, {3}, true);
  ti::NdArray<float> best_score =
      runtime.allocate_ndarray<float>({1}, {}, true);
  // host 可见的 staging 缓冲：初始化上传 / 每步读回画布
  ti::NdArray<float> staging =
      runtime.allocate_ndarray<float>({(uint32_t)H, (uint32_t)W}, {3}, true);

  // 初始化：host → staging（host 可见）→ device-to-device 拷进 VRAM。
  // 注意 buffer_copy 是排队异步的：复用 staging 写入下一份数据之前，
  // 必须先 runtime.wait() 等上一次拷贝真正执行完，否则会把新数据拷进 canvas。
  staging.write(canvas_init.data(), canvas_init.size());
  staging.copy_to(canvas);
  runtime.wait();  // 等 staging→canvas 完成，才能复用 staging
  staging.write(target.data(), target.size());
  staging.copy_to(target_np);
  runtime.wait();

  // ── 4. 主循环：每步生成一个最优椭圆并画上去 ──
  auto total_start = std::chrono::steady_clock::now();

  std::vector<float> ell(6), ycc(3), score(1);
  std::vector<float> canvas_cpu((size_t)H * W * 3);

  for (int shape_i = 0; shape_i < max_shapes; ++shape_i) {
    auto t0 = std::chrono::steady_clock::now();

    // 动态最大半径 & 采样步长（与 Python 的 one_shape 一致）
    float progress = (float)shape_i / (float)max_shapes;
    float max_radius =
        diag * (MAX_RADIUS_FRAC - 0.20f * std::pow(progress, 1.5f));
    if (max_radius < 4.0f) max_radius = 4.0f;
    int sample_step = std::max(1, (int)(max_radius / 8.0f));

    // ── g1: 误差场 → 重要性采样 → 随机生成并挑选最优 ──
    g1["canvas"] = canvas;
    g1["target"] = target_np;
    g1["error_field"] = error_field;
    g1["sampled_pixels"] = sampled_pixels;
    g1["best_ellipse"] = best_ellipse;
    g1["best_ycbcr"] = best_ycbcr;
    g1["best_score"] = best_score;
    g1["MIN_RADIUS"] = MIN_RADIUS;
    g1["MAX_RADIUS"] = max_radius;
    g1["MIN_ALPHA"] = MIN_ALPHA;
    g1["MAX_ALPHA"] = MAX_ALPHA;
    g1["RANDOM_SAMPLES"] = RANDOM_SAMPLES;
    g1["REJECTION_MAX_ATTEMPTS"] = MAX_ATTEMPTS;
    g1["SAMPLE_STEP"] = sample_step;
    g1.launch();

    // ── g2: 爬山变异 R 轮 ──
    float move_step = std::max(2.0f, diag * 0.012f);
    float radius_step = std::max(2.0f, diag * 0.010f);
    for (int r = 0; r < HILL_CLIMB_ROUNDS; ++r) {
      g2["best_ellipse"] = best_ellipse;  // 输入输出同一 ndarray，原地更新
      g2["MUTATIONS_PER_ROUND"] = MUTATIONS_PER_ROUND;
      g2["MOVE_STEP"] = move_step;
      g2["RADIUS_STEP"] = radius_step;
      g2["THETA_STEP_RAD"] = THETA_STEP_RAD;
      g2["ALPHA_STEP"] = ALPHA_STEP;
      g2["canvas"] = canvas;
      g2["target"] = target_np;
      g2["SAMPLE_STEP"] = sample_step;
      g2["best_ycbcr"] = best_ycbcr;
      g2["best_score"] = best_score;
      g2.launch();
    }

    // ── g3: 把最优椭圆叠加到画布 ──
    g3["best_ellipse"] = best_ellipse;
    g3["best_ycbcr"] = best_ycbcr;
    g3["canvas"] = canvas;
    g3.launch();

    // 必须同步：read() 是 map+memcpy，不隐式等待 GPU；
    // g2 对 best_* 是原地更新，不 wait 会读到上一轮的旧值。
    runtime.wait();

    // ── 读回结果，打印日志 ──
    best_ellipse.read(ell.data(), ell.size());
    best_ycbcr.read(ycc.data(), ycc.size());
    best_score.read(score.data(), score.size());

    float x = ell[0], y = ell[1], rx = ell[2], ry = ell[3];
    float alpha = ell[4], theta_rad = ell[5];
    float Y = ycc[0], Cb = ycc[1], Cr = ycc[2];

    // YCbCr → RGB 仅供日志
    float rgb_r = Y + 1.402f * Cr;
    float rgb_g = Y - 0.344136f * Cb - 0.714136f * Cr;
    float rgb_b = Y + 1.772f * Cb;

    float angle_deg =
        std::fmod(std::fabs(theta_rad) * 180.0f / 3.14159265f, 360.0f);

    auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0)
            .count() /
        1000.0;

    spdlog::info(
      "[{}/{}] Δ={:+.4f} @({:.0f},{:.0f}) r=({:.1f},{:.1f}) θ={:.0f}° "
      "α={:.2f} RGB=({:.2f},{:.2f},{:.2f}) {:.3f}s",
      shape_i + 1, max_shapes, score[0], x, y, rx, ry, angle_deg, alpha,
      rgb_r, rgb_g, rgb_b, elapsed);

    // MSE 与整张画布读回是纯诊断，改成每 SAVE_EVERY 步才做一次。
    // （原来每步 8MB 读回 + 200 万次串行 double 循环，是 C++ 版
    //   明显慢于 test_06 且随图元数累积的主因。）
    if ((shape_i + 1) % SAVE_EVERY == 0) {
      canvas.copy_to(staging);
      runtime.wait();  // 等 D2D 拷贝完成，staging 里才是新数据
      staging.read(canvas_cpu.data(), canvas_cpu.size());

      double mse = 0.0;
      for (size_t i = 0; i < canvas_cpu.size(); ++i) {
        double d = (double)canvas_cpu[i] - target[i];
        mse += d * d;
      }
      mse /= (double)(H * W * 3);
      spdlog::info("        MSE={:.6f}", mse);

      char path[128];
      std::snprintf(path, sizeof(path), "preview_05_%04d.png", shape_i + 1);
      save_preview(path, canvas_cpu.data(), W, H);
    }
  }

  // 最后再存一张
  char final_path[128];
  std::snprintf(final_path, sizeof(final_path), "preview_05_%04d.png",
                max_shapes);
  canvas.copy_to(staging);
  runtime.wait();
  staging.read(canvas_cpu.data(), canvas_cpu.size());
  save_preview(final_path, canvas_cpu.data(), W, H);

  auto total_elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                    total_start)
          .count();
  spdlog::info("完成！{} 个图元，总耗时 {:.1f}s，平均 {:.2f}s/个", max_shapes,
               total_elapsed, total_elapsed / max_shapes);
  return 0;
}