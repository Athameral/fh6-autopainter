"""Color Space Conversion"""
import taichi as ti
import taichi.math as tm


@ti.func
def rgb_to_ycbcr(c: tm.vec3) -> tm.vec3:
    y = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
    cb = -0.168736 * c[0] - 0.331264 * c[1] + 0.5 * c[2]
    cr = 0.5 * c[0] - 0.418688 * c[1] - 0.081312 * c[2]
    return tm.vec3(y, cb, cr)


@ti.func
def ycbcr_to_rgb(c: tm.vec3) -> tm.vec3:
    r = c[0] + 1.402 * c[2]
    g = c[0] - 0.344136 * c[1] - 0.714136 * c[2]
    b = c[0] + 1.772 * c[1]
    r = tm.clamp(r, 0.0, 1.0)
    g = tm.clamp(g, 0.0, 1.0)
    b = tm.clamp(b, 0.0, 1.0)
    return tm.vec3(r, g, b)
