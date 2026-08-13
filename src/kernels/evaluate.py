"""Evaluation and Picing Engine.
Including candidate evaluation, parallel searching and mutation."""
import taichi as ti
import taichi.math as tm

from .colors import rgb_to_ycbcr


@ti.func
def compute_optimal_color_and_score(
    ellipse: ti.types.vector(6),
    canvas: ti.types.ndarray(dtype=tm.vec3, ndim=2),
    target: ti.types.ndarray(dtype=tm.vec3, ndim=2),
    valid_mask: ti.types.ndarray(dtype=ti.i32, ndim=2),
    sample_step: ti.i32,
    out_ycbcr: ti.template(),
    out_score: ti.template(),
):
    # bbox
    cx = ellipse[0]
    cy = ellipse[1]
    rx = ellipse[2]
    ry = ellipse[3]
    alpha = ellipse[4]
    theta = ellipse[5]

    cos_t = ti.cos(theta)
    sin_t = ti.sin(theta)
    ex = ti.sqrt(rx * rx * cos_t * cos_t + ry * ry * sin_t * sin_t)
    ey = ti.sqrt(rx * rx * sin_t * sin_t + ry * ry * cos_t * cos_t)

    x_min = ti.max(0, ti.cast(ti.floor(cx - ex - 1.0), ti.i32))
    x_max = ti.min(canvas.shape[1] - 1, ti.cast(ti.ceil(cx + ex + 1.0), ti.i32))
    y_min = ti.max(0, ti.cast(ti.floor(cy - ey - 1.0), ti.i32))
    y_max = ti.min(canvas.shape[0] - 1, ti.cast(ti.ceil(cy + ey + 1.0), ti.i32))

    # 钳制输入防止除零
    rx = ti.max(rx, 1.0)
    ry = ti.max(ry, 1.0)
    alpha = ti.min(1.0, ti.max(1e-6, alpha))
    ss = ti.max(sample_step, 1)

    inv_rx2 = 1.0 / (rx * rx)
    inv_ry2 = 1.0 / (ry * ry)

    # 统计量（YCbCr 三通道各自累积）
    N = 0
    sY_sum = 0.0;  sCb_sum = 0.0;  sCr_sum = 0.0   # Σ current
    tY_sum = 0.0;  tCb_sum = 0.0;  tCr_sum = 0.0   # Σ target
    sY2 = 0.0;     sCb2 = 0.0;     sCr2 = 0.0      # Σ current²
    stY = 0.0;     stCb = 0.0;     stCr = 0.0      # Σ current*target

    broken_alpha = 0

    for y in range(y_min, y_max + 1):
        dy = ti.cast(y, ti.f32) + 0.5 - cy
        for x in range(x_min, x_max + 1):
            # 采样步长：跳像素加速大椭圆评估
            if (x - x_min) % ss != 0 or (y - y_min) % ss != 0:
                continue
            dx = ti.cast(x, ti.f32) + 0.5 - cx

            # 旋转到椭圆局部坐标系
            xr = dx * cos_t + dy * sin_t
            yr = -dx * sin_t + dy * cos_t

            if xr * xr * inv_rx2 + yr * yr * inv_ry2 <= 1.0:
                N += 1
                if valid_mask[y, x] == 0:
                    broken_alpha += 1
                sc = rgb_to_ycbcr(canvas[y, x])   # current (RGB→YCbCr on the fly)
                tc = rgb_to_ycbcr(target[y, x])   # target (RGB→YCbCr on the fly)
                # Y 通道
                sY_sum += sc[0]; tY_sum += tc[0]
                sY2 += sc[0] * sc[0]; stY += sc[0] * tc[0]
                # Cb 通道
                sCb_sum += sc[1]; tCb_sum += tc[1]
                sCb2 += sc[1] * sc[1]; stCb += sc[1] * tc[1]
                # Cr 通道
                sCr_sum += sc[2]; tCr_sum += tc[2]
                sCr2 += sc[2] * sc[2]; stCr += sc[2] * tc[2]

    # 解析求最优颜色 & ΔMSE
    if N > 0 and broken_alpha < N * 1e-2:
        invN = 1.0 / ti.cast(N, ti.f32)
        invA = 1.0 - alpha

        # 最优颜色（未 clamp）：c_opt = (avg_target - avg_current*(1-a)) / a
        oY_raw = (tY_sum * invN - sY_sum * invN * invA) / alpha
        oCb_raw = (tCb_sum * invN - sCb_sum * invN * invA) / alpha
        oCr_raw = (tCr_sum * invN - sCr_sum * invN * invA) / alpha

        # clamp 到各通道合法范围：Y∈[0,1], Cb/Cr∈[-0.5, 0.5]
        oY = ti.min(1.0, ti.max(0.0, oY_raw))
        oCb = ti.min(0.5, ti.max(-0.5, oCb_raw))
        oCr = ti.min(0.5, ti.max(-0.5, oCr_raw))

        # Δ = a²(N·c² − 2c·Σs + Σs²) − 2a(c·Σt − c·Σs − Σst + Σs²)
        Nf = ti.cast(N, ti.f32)
        a2 = alpha * alpha
        two_a = 2.0 * alpha

        dY = a2 * (Nf * oY * oY - 2.0 * oY * sY_sum + sY2) \
           - two_a * (oY * tY_sum - oY * sY_sum - stY + sY2)
        dCb = a2 * (Nf * oCb * oCb - 2.0 * oCb * sCb_sum + sCb2) \
           - two_a * (oCb * tCb_sum - oCb * sCb_sum - stCb + sCb2)
        dCr = a2 * (Nf * oCr * oCr - 2.0 * oCr * sCr_sum + sCr2) \
           - two_a * (oCr * tCr_sum - oCr * sCr_sum - stCr + sCr2)

        delta = dY + dCb + dCr
        # 乘回采样步长²（跳过的像素近似恢复）
        delta *= ti.cast(ss * ss, ti.f32)

        out_ycbcr = tm.vec3(oY, oCb, oCr)
        # ΔMSE ≤ 0 表示改善（负值越小越好），取负号后 score ≥ 0 表示改善程度
        out_score = -delta
    else:
        out_ycbcr = tm.vec3(0.0, 0.0, 0.0)
        out_score = 0.0


@ti.kernel
def generate_and_pick_best(
    min_radius: ti.f32,
    max_radius: ti.f32,
    min_alpha: ti.f32,
    max_alpha: ti.f32,
    n_samples: ti.i32,
    sampled_pixels: ti.types.ndarray(dtype=ti.math.vec2, ndim=1),
    canvas: ti.types.ndarray(dtype=tm.vec3, ndim=2),
    target: ti.types.ndarray(dtype=tm.vec3, ndim=2),
    valid_mask: ti.types.ndarray(dtype=ti.i32, ndim=2),
    sample_step: ti.i32,
    out_best_ellipse: ti.types.ndarray(dtype=ti.types.vector(6), ndim=1),
    # out_best_ellipse[0] = (x, y, rx, ry, alpha, theta)
    out_best_ycbcr: ti.types.ndarray(dtype=tm.vec3, ndim=1),
    out_best_score: ti.types.ndarray(dtype=ti.f32, ndim=1),
):
    """Generate random ellipses (centers from sampled_pixels[0..n_samples_actual[0])),
    evaluate each via compute_optimal_color_and_score, and output the single
    best ellipse (highest score).
    n_samples_actual 是长度 1 的 buffer：采样 kernel 写入的实际样本数
    （topk 时合格像素可能少于请求数）。"""
    best_score = -1e30
    best_ellipse = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_ycbcr = tm.vec3(0.0)

    tmp_ycbcr = tm.vec3(0.0)
    tmp_score = ti.f32(0.0)

    for i in range(n_samples):
        # ---- generate one random ellipse ----
        ell = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        ell[0] = sampled_pixels[i][0]                                       # x
        ell[1] = sampled_pixels[i][1]                                       # y
        ell[2] = ti.random(ti.f32) * (max_radius - min_radius) + min_radius  # rx
        ell[3] = ti.random(ti.f32) * (max_radius - min_radius) + min_radius  # ry
        ell[4] = ti.random(ti.f32) * (max_alpha - min_alpha) + min_alpha     # alpha
        ell[5] = ti.random(ti.f32) * 2.0 * tm.pi                             # theta (rad)

        # ---- evaluate ----
        compute_optimal_color_and_score(ell, canvas, target, valid_mask,
                                        sample_step, tmp_ycbcr, tmp_score)

        # ---- track best ----
        
        if tmp_score > best_score:
            best_score = tmp_score
            best_ellipse = ell
            best_ycbcr = tmp_ycbcr

    out_best_ellipse[0] = best_ellipse
    out_best_ycbcr[0] = best_ycbcr
    out_best_score[0] = best_score


@ti.kernel
def mutate_and_pick_best(
    base_ellipse: ti.types.ndarray(dtype=ti.types.vector(6), ndim=1),
    n_mutations: ti.i32,
    move_step: ti.f32,
    radius_step: ti.f32,
    theta_step: ti.f32,
    alpha_step: ti.f32,
    canvas: ti.types.ndarray(dtype=tm.vec3, ndim=2),
    target: ti.types.ndarray(dtype=tm.vec3, ndim=2),
    valid_mask: ti.types.ndarray(dtype=ti.i32, ndim=2),
    sample_step: ti.i32,
    out_best_ellipse: ti.types.ndarray(dtype=ti.types.vector(6), ndim=1),
    out_best_ycbcr: ti.types.ndarray(dtype=tm.vec3, ndim=1),
    out_best_score: ti.types.ndarray(dtype=ti.f32, ndim=1),
):
    """Hill-climb one round: mutate base_ellipse n_mutations times,
    evaluate each variant, and output the single best (highest score).
    If no mutation beats the base, the base itself is returned."""
    bx = base_ellipse[0][0]
    by = base_ellipse[0][1]
    brx = base_ellipse[0][2]
    bry = base_ellipse[0][3]
    balpha = base_ellipse[0][4]
    btheta = base_ellipse[0][5]

    tmp_ycbcr = tm.vec3(0.0)
    tmp_score = ti.f32(0.0)

    # evaluate the base first
    compute_optimal_color_and_score(
        base_ellipse[0], canvas, target, valid_mask,
        sample_step, tmp_ycbcr, tmp_score)

    best_score = tmp_score
    best_ellipse = base_ellipse[0]
    best_ycbcr = tmp_ycbcr

    w = ti.cast(canvas.shape[1], ti.f32)
    h = ti.cast(canvas.shape[0], ti.f32)

    for _ in range(n_mutations):
        # ---- perturb each parameter independently ----
        x = bx + (ti.random(ti.f32) * 2.0 - 1.0) * move_step
        y = by + (ti.random(ti.f32) * 2.0 - 1.0) * move_step
        rx = brx + (ti.random(ti.f32) * 2.0 - 1.0) * radius_step
        ry = bry + (ti.random(ti.f32) * 2.0 - 1.0) * radius_step
        alpha = balpha + (ti.random(ti.f32) * 2.0 - 1.0) * alpha_step
        theta = btheta + (ti.random(ti.f32) * 2.0 - 1.0) * theta_step

        # ---- clamp to valid ranges ----
        x = ti.min(w - 1.0, ti.max(0.0, x))
        y = ti.min(h - 1.0, ti.max(0.0, y))
        rx = ti.max(1.0, rx)
        ry = ti.max(1.0, ry)
        alpha = ti.min(1.0, ti.max(1e-6, alpha))

        ell = ti.Vector([x, y, rx, ry, alpha, theta])

        compute_optimal_color_and_score(ell, canvas, target, valid_mask,
                                        sample_step, tmp_ycbcr, tmp_score)
        score = tmp_score
        ycbcr = tmp_ycbcr

        if score > best_score:
            best_score = score
            best_ellipse = ell
            best_ycbcr = ycbcr

    out_best_ellipse[0] = best_ellipse
    out_best_ycbcr[0] = best_ycbcr
    out_best_score[0] = best_score
