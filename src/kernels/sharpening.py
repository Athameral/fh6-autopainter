import taichi as ti
import taichi.math as tm


@ti.kernel
def sharpen_kernel(canvas: ti.types.ndarray(tm.vec3, ndim=2),
                   dest: ti.types.ndarray(tm.vec3, ndim=2),
                   intensity: ti.f32):
    h, w = canvas.shape[0], canvas.shape[1]
    for i, j in ti.ndrange(h, w):
        ip = ti.min(i + 1, h - 1)
        im = ti.max(i - 1, 0)
        jp = ti.min(j + 1, w - 1)
        jm = ti.max(j - 1, 0)

        center = canvas[i, j]
        blur = (canvas[im, jm] + canvas[im, j] + canvas[im, jp] +
                canvas[i, jm] + center + canvas[i, jp] +
                canvas[ip, jm] + canvas[ip, j] + canvas[ip, jp]) / 9.0

        # Unsharp mask: original + intensity * (original - blur)
        dest[i, j] = ti.max(0.0, ti.min(1.0, center + intensity * (center - blur)))

