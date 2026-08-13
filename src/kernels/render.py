"""Rendering Engine — Taichi Kernels for Drawing."""
import taichi as ti
import taichi.math as tm


@ti.kernel
def apply_ellipse(
    ellipse: ti.types.ndarray(dtype=ti.types.vector(6), ndim=1),
    ycbcr_color: ti.types.ndarray(dtype=tm.vec3, ndim=1),
    canvas: ti.types.ndarray(dtype=tm.vec3, ndim=2),
):
    """Alpha-blend one ellipse onto the canvas (RGB, over operator).

    Coordinate convention (OpenCV / numpy / PIL):
      x → right,  y → down,  origin = top‑left
      canvas[y, x]  (dimension 0 = row = y, dimension 1 = col = x)

    ellipse[0] = (cx, cy, rx, ry, alpha, theta_rad)  — in pixel coords
    ycbcr_color[0] = (Y, Cb, Cr)  — converted to RGB on the fly before blending
    canvas is RGB in / RGB out.
    """
    cx = ellipse[0][0]
    cy = ellipse[0][1]
    rx = ti.max(ellipse[0][2], 1.0)
    ry = ti.max(ellipse[0][3], 1.0)
    alpha = ti.min(1.0, ti.max(1e-6, ellipse[0][4]))
    theta = ellipse[0][5]

    # YCbCr → RGB
    ycc = ycbcr_color[0]
    cr = ycc[0] + 1.402 * ycc[2]
    cg = ycc[0] - 0.344136 * ycc[1] - 0.714136 * ycc[2]
    cb = ycc[0] + 1.772 * ycc[1]
    color = tm.vec3(
        ti.min(1.0, ti.max(0.0, cr)),
        ti.min(1.0, ti.max(0.0, cg)),
        ti.min(1.0, ti.max(0.0, cb)),
    )
    inv_alpha = 1.0 - alpha

    cos_t = ti.cos(theta)
    sin_t = ti.sin(theta)
    inv_rx2 = 1.0 / (rx * rx)
    inv_ry2 = 1.0 / (ry * ry)

    ex = ti.sqrt(rx * rx * cos_t * cos_t + ry * ry * sin_t * sin_t)
    ey = ti.sqrt(rx * rx * sin_t * sin_t + ry * ry * cos_t * cos_t)

    W = canvas.shape[1]
    H = canvas.shape[0]

    x_min = ti.max(0, ti.cast(ti.floor(cx - ex - 1.0), ti.i32))
    x_max = ti.min(W - 1, ti.cast(ti.ceil(cx + ex + 1.0), ti.i32))
    y_min = ti.max(0, ti.cast(ti.floor(cy - ey - 1.0), ti.i32))
    y_max = ti.min(H - 1, ti.cast(ti.ceil(cy + ey + 1.0), ti.i32))

    for y in range(y_min, y_max + 1):
        dy = ti.cast(y, ti.f32) + 0.5 - cy
        for x in range(x_min, x_max + 1):
            dx = ti.cast(x, ti.f32) + 0.5 - cx
            xr = dx * cos_t + dy * sin_t
            yr = -dx * sin_t + dy * cos_t
            if xr * xr * inv_rx2 + yr * yr * inv_ry2 <= 1.0:
                cur = canvas[y, x]
                canvas[y, x] = cur * inv_alpha + color * alpha


@ti.kernel
def render_to_rgb(
    canvas_ycbcr: ti.types.ndarray(dtype=tm.vec3, ndim=2),
    out_rgb: ti.types.ndarray(dtype=tm.vec3, ndim=2),
):
    """Convert YCbCr canvas to RGB for display / saving."""
    for y, x in ti.ndrange(canvas_ycbcr.shape[0], canvas_ycbcr.shape[1]):
        c = canvas_ycbcr[y, x]
        r = c[0] + 1.402 * c[2]
        g = c[0] - 0.344136 * c[1] - 0.714136 * c[2]
        b = c[0] + 1.772 * c[1]
        out_rgb[y, x] = tm.vec3(
            ti.min(1.0, ti.max(0.0, r)),
            ti.min(1.0, ti.max(0.0, g)),
            ti.min(1.0, ti.max(0.0, b)),
        )
