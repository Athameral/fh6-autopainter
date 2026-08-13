"""test_05.py — Geometrize via utils.py（纯 Taichi kernel，canvas 全程 RGB）

与 test_04 的逻辑完全相同，但所有 GPU 计算都委托给 src/utils.py 里的
Taichi kernel，不在 test 文件里写 inline kernel。

用法：
  python -m tests.test_05 --arch vulkan --shapes 300   # 无窗口
  python -m tests.test_05 --arch vulkan                 # GGUI 实时显示
"""
import argparse
import os
import time

import numpy as np
import taichi as ti
from PIL import Image

from src.kernels.colors import rgb_to_ycbcr as _rgb_to_ycbcr  # unused directly, for docs
from src.kernels.sampling import compute_error_field, sample_from_error, sample_from_error_topk
from src.kernels.evaluate import generate_and_pick_best, mutate_and_pick_best
from src.kernels.render import apply_ellipse
from src.kernels.sharpening import sharpen_kernel

# ── CLI ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--arch", choices=["cpu", "vulkan", "cuda"], default="vulkan")
parser.add_argument("--shapes", type=int, default=None,
                    help="无窗口模式：生成固定数量图元后退出（缺省打开 GGUI 窗口）")
parser.add_argument("--image", type=str, default=None,
                    help="目标图片路径（默认 tests/pics/01.png）")
ARGS = parser.parse_args()

ti.init(arch=getattr(ti, ARGS.arch))
print(f"后端: {ARGS.arch}")

# ── 配置 ─────────────────────────────────────────────────────────────────
W = 1024
H = 662
MAX_SHAPES = ARGS.shapes or 2500
RANDOM_SAMPLES = 60000
HILL_CLIMB_ROUNDS = 48
MUTATIONS_PER_ROUND = 8000
MOVE_STEP = 8.0
RADIUS_STEP = 6.0
THETA_STEP_RAD = 0.523599   # ≈ 30° in radians (mutate_and_pick_best takes rad)
ALPHA_STEP = 0.15
MIN_RADIUS = 1.0
MAX_RADIUS_FRAC = 0.03
MIN_ALPHA = 0.3
MAX_ALPHA = 1.0
SAVE_EVERY = 50
# MAX_ATTEMPTS = 12          # rejection sampling 重试次数

# ── 加载目标图 ───────────────────────────────────────────────────────────
IMG_PATH = ARGS.image or os.path.join(os.path.dirname(__file__), "pics", "01.png")


def load_target():
    img = Image.open(IMG_PATH).convert("RGB").resize((W, H), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # (H, W, 3) RGB
    target = arr.copy()

    # 背景色 = 边缘平均（RGB）
    edge = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]], axis=0)
    bg = edge.mean(axis=0, keepdims=True).astype(np.float32)
    canvas = np.tile(bg, (H, W, 1)).astype(np.float32)

    print(f"画布底色: {np.round(np.clip(bg[0], 0, 1), 2)} (RGB)")
    print(f"目标图: {IMG_PATH}  ({W}x{H})")
    return canvas, target


canvas, target_origin = load_target()
target = target_origin.copy()

# ── Taichi ndarray 缓冲区（一次性分配，复用）────────────────────────────
# 注意 ti.types.vector(6) 映射为 (1, 6) 的 float32 ndarray
error_field = ti.ndarray(ti.f32, shape=(H, W))
error_field_buffer = ti.ndarray(ti.f32, shape=(H, W))
sampled_pixels = ti.ndarray(ti.math.vec2, shape=(RANDOM_SAMPLES, ))
best_ellipse = ti.ndarray(ti.types.vector(6, ti.f32), shape=(1, ))
best_ycbcr = ti.ndarray(ti.math.vec3, shape=(1, ))
best_score = ti.ndarray(ti.f32, shape=(1, ))
hist_buffer = ti.ndarray(ti.i32, shape=(1024, ))  # sample_from_error_topk 用于直方图

diag = np.sqrt(W * W + H * H).astype(np.float32)


# ── 单步：生成一个椭圆并画上去 ──────────────────────────────────────────
def one_shape(shape_i: int, canvas: np.ndarray) -> np.ndarray:
    """生成最优椭圆、画到 canvas 上，返回更新后的 canvas（原地修改）。"""
    t0 = time.perf_counter()

    # 动态最大半径 & 采样步长
    progress = shape_i / max(MAX_SHAPES, 1)
    max_radius = diag * (MAX_RADIUS_FRAC - 0.20 * progress ** 1.5)
    max_radius = max(max_radius, 4.0)
    sample_step = max(1, int(max_radius / 8.0))
    # sample_step = 1

    # 1. 误差场
    compute_error_field(canvas, target, error_field, error_field_buffer, blur_size=5)
    # ti.sync()

    # 2. 重要性采样
    # sample_from_error(RANDOM_SAMPLES, MAX_ATTEMPTS, error_field, sampled_pixels)
    # n_samples_actual[0] = RANDOM_SAMPLES
    # or below
    sample_from_error_topk(RANDOM_SAMPLES, 1024, 0.1, hist_buffer, error_field, sampled_pixels)
    # ti.sync()

    # 3. 随机搜索
    generate_and_pick_best(
        MIN_RADIUS, max_radius, MIN_ALPHA, MAX_ALPHA,
        RANDOM_SAMPLES, sampled_pixels,
        canvas, target, sample_step,
        best_ellipse, best_ycbcr, best_score,
    )
    # ti.sync()

    # 4. 爬山 R 轮
    move_step = max(2.0, diag * 0.012)
    radius_step = max(2.0, diag * 0.010)
    for _ in range(HILL_CLIMB_ROUNDS):
        mutate_and_pick_best(
            best_ellipse,
            MUTATIONS_PER_ROUND,
            move_step, radius_step, THETA_STEP_RAD, ALPHA_STEP,
            canvas, target, sample_step,
            best_ellipse, best_ycbcr, best_score,
        )
    # ti.sync()

    # 5. 画到画布
    apply_ellipse(best_ellipse, best_ycbcr, canvas)
    # ti.sync()

    elapsed = time.perf_counter() - t0

    # 日志
    x = float(best_ellipse[0][0])
    y = float(best_ellipse[0][1])
    rx = float(best_ellipse[0][2])
    ry = float(best_ellipse[0][3])
    alpha = float(best_ellipse[0][4])
    theta_rad = float(best_ellipse[0][5])
    y_cc = best_ycbcr[0]
    score = float(best_score[0])
    angle_deg = np.rad2deg(theta_rad) % 360

    # YCbCr → RGB 仅供日志
    rgb_r = y_cc[0] + 1.402 * y_cc[2]
    rgb_g = y_cc[0] - 0.344136 * y_cc[1] - 0.714136 * y_cc[2]
    rgb_b = y_cc[0] + 1.772 * y_cc[1]

    # 总 MSE（诊断）
    mse = float(np.mean((canvas - target) ** 2))

    print(
        f"[{shape_i + 1:4d}/{MAX_SHAPES}] "
        f"Δ={score:+.4f}  MSE={mse:.6f}  "
        f"@({x:5.0f},{y:5.0f})  "
        f"r=({rx:4.1f},{ry:4.1f})  "
        f"θ={angle_deg:5.0f}°  α={alpha:.2f}  "
        f"RGB=({rgb_r:.2f},{rgb_g:.2f},{rgb_b:.2f})  "
        f"{elapsed:.3f}s"
    )
    return canvas


# ── 保存预览 ────────────────────────────────────────────────────────────
def save_preview(shape_count: int, canvas: np.ndarray):
    out_path = os.path.join(os.path.dirname(__file__),
                            f"preview_05_{shape_count:04d}.png")
    rgb = np.clip(canvas, 0, 1)
    img = Image.fromarray((rgb * 255).astype(np.uint8), "RGB")
    img.save(out_path)
    print(f"  预览已保存: {out_path}")


# ── 主循环 ───────────────────────────────────────────────────────────────
def main():
    sharpen_kernel(target_origin, target, 1.0)  # 先锐化目标图，减少噪点干扰
    total_start = time.perf_counter()

    if ARGS.shapes is not None:
        # ── 无窗口模式 ──
        for shape_i in range(MAX_SHAPES):
            one_shape(shape_i, canvas)
            if (shape_i + 1) % SAVE_EVERY == 0:
                save_preview(shape_i + 1, canvas)

        save_preview(MAX_SHAPES, canvas)
        total_elapsed = time.perf_counter() - total_start
        print(f"\n完成！{MAX_SHAPES} 个图元，总耗时 {total_elapsed:.1f}s，"
              f"平均 {total_elapsed / MAX_SHAPES:.2f}s/个")
        return

    # ── GGUI 实时显示 ──
    # 三区域：渲染 | 目标 | 误差图（叠加采样点），每两区域间 2px 分隔线
    DISPLAY_W = 3 * W + 4
    display = ti.Vector.field(3, ti.f32, shape=(DISPLAY_W, H))

    @ti.kernel
    def compose(canvas_np: ti.types.ndarray(dtype=ti.math.vec3, ndim=2),
                target_np: ti.types.ndarray(dtype=ti.math.vec3, ndim=2),
                error_np: ti.types.ndarray(dtype=ti.f32, ndim=2),
                samples_np: ti.types.ndarray(dtype=ti.math.vec2, ndim=1),
                n_samples: ti.i32):
        # 三张图：canvas / target / 误差灰度图（值 clamp 到 [0,1] 保证归一化显示）
        for i, j in ti.ndrange(H, W):
            display[j, H - 1 - i] = canvas_np[i, j]
            display[W + 2 + j, H - 1 - i] = target_np[i, j]
            g = ti.min(1.0, ti.max(0.0, error_np[i, j]))   # 归一化
            display[2 * W + 4 + j, H - 1 - i] = ti.math.vec3(g)
        # 分隔线（白色，4 列）
        for i in range(H):
            for off in range(2):
                display[W + off, i] = ti.math.vec3(1.0)
                display[2 * W + 2 + off, i] = ti.math.vec3(1.0)
        # 本轮实际采样点（红色）叠加在误差图上，观察空间分布是否均匀
        for k in range(n_samples):
            p = samples_np[k]
            sx = ti.cast(p[0], ti.i32)
            sy = ti.cast(p[1], ti.i32)
            if 0 <= sx < W and 0 <= sy < H:
                display[2 * W + 4 + sx, H - 1 - sy] = ti.math.vec3(1.0, 0.0, 0.0)

    window = ti.ui.Window(
        "Autopainter v5 — Geometrize | 左: 渲染  中: 目标  右: 误差+采样点(红)",
        (DISPLAY_W, H), vsync=True,
    )
    win_canvas = window.get_canvas()

    # 先快速放 5 个椭圆初始化画面
    for i in range(5):
        one_shape(i, canvas)

    shape_i = 5
    while window.running:
        for _ in range(1):  # 每帧 1 个
            if shape_i >= MAX_SHAPES:
                break
            one_shape(shape_i, canvas)
            shape_i += 1
        compose(canvas, target, error_field, sampled_pixels, RANDOM_SAMPLES)
        win_canvas.set_image(display)
        window.show()

    # 窗口关闭后打印总计时
    total_elapsed = time.perf_counter() - total_start
    print(f"\n窗口已关闭：已绘制 {shape_i} 个图元，总耗时 {total_elapsed:.1f}s，"
          f"平均 {total_elapsed / max(shape_i, 1):.2f}s/个")


if __name__ == "__main__":
    main()
