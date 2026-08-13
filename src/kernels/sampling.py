"""Error Field Computing and Sampling"""
import taichi as ti
import taichi.math as tm

from .colors import rgb_to_ycbcr


# TODO: 解决透明度的问题
# 首先不在透明度的位置采样
# 其次生成时拒绝明显踩了透明位置的椭圆
# 最后按踩了透明点的数量惩罚分数


@ti.kernel
def build_valid_pixels_and_mask(
    alpha: ti.types.ndarray(dtype=ti.f32, ndim=2),
    out_valid_pixels: ti.types.ndarray(dtype=ti.types.vector(2, ti.i32), ndim=1),
    out_n_valid_pixels: ti.types.ndarray(dtype=ti.i32, ndim=1),
    out_mask: ti.types.ndarray(dtype=ti.i32, ndim=2),
):
    """Build a mask of valid pixels (1 for valid, 0 for invalid) based on alpha channel."""
    for I in ti.grouped(alpha):
        out_mask[I] = 1 if alpha[I] > 0.01 else 0

    valid_count = 0
    for I in ti.grouped(out_mask):
        if out_mask[I] == 1:
            idx = ti.atomic_add(valid_count, 1)
            out_valid_pixels[idx][0] = I[1]  # x
            out_valid_pixels[idx][1] = I[0]  # y
    out_n_valid_pixels[0] = valid_count

@ti.kernel
def compute_error_field(canvas: ti.types.ndarray(dtype=tm.vec3, ndim=2),
                        target: ti.types.ndarray(dtype=tm.vec3, ndim=2),
                        valid_mask: ti.types.ndarray(dtype=ti.i32, ndim=2),
                        out_error_field: ti.types.ndarray(dtype=ti.f32, ndim=2),
                        error_field_buffer: ti.types.ndarray(dtype=ti.f32, ndim=2),
                        blur_size: ti.i32):
    """Per-pixel error in YCbCr space (L1 sum over 3 channels), normalized to [0,1].
    canvas and target are both RGB; conversion to YCbCr happens on the fly."""
    for I in ti.grouped(error_field_buffer):
        error_field_buffer[I] = 0.0
    
    max_error = 0.0
    for I in ti.grouped(canvas):
        if valid_mask[I] == 1:
            c1 = rgb_to_ycbcr(canvas[I])
            c2 = rgb_to_ycbcr(target[I])
            e = ti.abs(c1[0] - c2[0]) + ti.abs(c1[1] - c2[1]) + ti.abs(c1[2] - c2[2])
            error_field_buffer[I] = e
            ti.atomic_max(max_error, e)
    for I in ti.grouped(error_field_buffer):
        error_field_buffer[I] = error_field_buffer[I] / max_error
    H, W = error_field_buffer.shape[0], error_field_buffer.shape[1]
    for i, j in error_field_buffer:
        sum_val = 0.0
        count = 0
        for di in range(-blur_size // 2, blur_size // 2 + 1):
            for dj in range(-blur_size // 2, blur_size // 2 + 1):
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W:
                    sum_val += error_field_buffer[ni, nj]
                    count += 1
        out_error_field[i, j] = sum_val / count


@ti.kernel
def sample_from_error(
    n_samples: ti.i32,
    max_attempts: ti.i32,
    error_field: ti.types.ndarray(dtype=ti.f32, ndim=2),
    out_sampled_pixels: ti.types.ndarray(dtype=ti.math.vec2, ndim=1)
):
    """Sample n_samples pixels from the error field,
    with probability proportional to the error value.
    Taichi neither provides a built-in weighted sampling function,
    nor a simple parallel cumsum kernel. So we implement a
    rejection sampling here."""
    for i in range(n_samples):
        for _ in range(max_attempts):
            x = ti.random(ti.i32) % error_field.shape[1]
            y = ti.random(ti.i32) % error_field.shape[0]
            p = ti.random(ti.f32)
            out_sampled_pixels[i] = ti.math.vec2(x, y)
            if p < error_field[y, x]:
                break


@ti.kernel
def sample_from_error_topk(
    n_samples: ti.i32,
    n_bins: ti.i32,
    leak_ratio: ti.f32,
    # how likely to sample from the pixels below the threshold (0.0 ~ 1.0)
    # since it's not always good to ignore the low-error pixels,
    # we allow a small leak ratio, 0.1 is good
    valid_pixels: ti.types.ndarray(dtype=ti.types.vector(2, ti.i32), ndim=1),
    n_valid_pixels: ti.types.ndarray(dtype=ti.i32, ndim=1),
    hist_buffer: ti.types.ndarray(dtype=ti.i32, ndim=1),
    error_field: ti.types.ndarray(dtype=ti.f32, ndim=2),
    out_sampled_pixels: ti.types.ndarray(dtype=ti.math.vec2, ndim=1),
):
    """Sample n_samples pixels from the error field.
    This is a top-k sampling method: we first divide the image into histogram of n_bins,
    then get the approximate threshold of top-k error values,
    and finally sample from the pixels whose error is above the threshold.
    minmax_buffer 是长度 2 的临时 buffer（[min, max]），用于归约：局部变量上的
    ti.atomic_max/min 在并行循环里不可靠（实测 CPU/Vulkan 都停在初始值），
    写外部 ndarray 的原子操作才跨线程有效。hist_buffer 由本 kernel 负责清零。
    out_n_samples_actual 是长度 1 的 buffer，传出实际写入的采样数：
    阈值取 bin 上边界后合格像素数 < n_samples，调用方按此值消费，
    不会读到残留的旧采样点。"""

    n_samples_topk = ti.cast(n_samples * (1 - leak_ratio), ti.i32)
    n_samples_leak = n_samples - n_samples_topk

    # 0. 重置归约 buffer 并清零直方图（hist_buffer 跨调用复用，不清零会累积）
    min_error = 999.0
    max_error = 0.0
    for i in range(n_bins):
        hist_buffer[i] = 0

    # 1. 归约 min/max（ndarray 上的原子操作跨线程有效）
    for I in ti.grouped(error_field):
        val = error_field[I]
        ti.atomic_min(min_error, val)
        ti.atomic_max(max_error, val)

    # 2. Compute histogram of error values
    for I in ti.grouped(error_field):
        val = error_field[I]
        bin_idx = ti.cast((val - min_error) / (max_error - min_error + 1e-6) * n_bins, ti.i32)
        ti.atomic_add(hist_buffer[bin_idx], 1)

    # 3. Find threshold of top-k error values.
    # 阈值取 bin b 的【上边界】 (b+1)/n_bins（不是 b/n_bins，也不是 b + 1/n_bins——
    # 后者运算符优先级是 b + (1/n_bins)，b 是整数索引会算出远超 max 的阈值）：
    # 这样只保留 bin b+1 及以上的像素，数量 < n_samples，所有合格像素都能入选
    # （公平，没有“并行遍历先到先得、idx == n_samples 后截断”的偏向）。
    threshold = min_error
    accum = 0
    ti.loop_config(serialize=True)
    for i in range(n_bins):
        b = n_bins - 1 - i
        accum += hist_buffer[b]
        if accum >= n_samples_topk:
            threshold = min_error + ((b + 1) / n_bins) * (max_error - min_error)
            break
    
    # 4. Sample from pixels whose error is above the threshold.
    # 只用 ti.atomic_add 拿唯一索引——切勿再对 topk_count 做普通读写（如 += 1），
    # 局部变量的原子操作会被降级为 load+add+store，与普通 RMW 混用会产生竞争。
    # idx < n_samples 是防御性保护（极端情况如全场同色时合格像素仍可能超过 n_samples）。
    topk_count = 0
    for I in ti.grouped(error_field):
        if error_field[I] >= threshold:
            idx = ti.atomic_add(topk_count, 1)
            if idx < n_samples_topk:
                out_sampled_pixels[idx] = ti.math.vec2(I[1], I[0])  # (x, y)

    for i in range(n_samples_leak):
        idx = ti.atomic_add(topk_count, 1)
        if idx < n_samples:
            out_sampled_pixels[idx] = valid_pixels[ti.random(ti.i32) % n_valid_pixels[0]]
