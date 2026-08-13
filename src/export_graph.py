import taichi as ti

from kernels.sampling import compute_error_field, sample_from_error_topk
from kernels.evaluate import generate_and_pick_best, mutate_and_pick_best
from kernels.render import apply_ellipse
from kernels.sharpening import sharpen_kernel

# buffers
canvas = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "canvas", dtype=ti.math.vec3, ndim=2)
target = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "target", dtype=ti.math.vec3, ndim=2)
error_field = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "error_field", dtype=ti.f32, ndim=2)
sampled_pixels = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "sampled_pixels", dtype=ti.math.vec2, ndim=1)
hist_buffer = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "hist_buffer", dtype=ti.i32, ndim=1)
best_ellipse = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "best_ellipse", dtype=ti.types.vector(6, dtype=ti.f32), ndim=1)
best_ycbcr = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "best_ycbcr", dtype=ti.math.vec3, ndim=1)
best_score = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "best_score", dtype=ti.f32, ndim=1)

# params
## sampling ellipse
MIN_RADIUS = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "MIN_RADIUS", dtype=ti.f32)
MAX_RADIUS = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "MAX_RADIUS", dtype=ti.f32)
MIN_ALPHA = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "MIN_ALPHA", dtype=ti.f32)
MAX_ALPHA = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "MAX_ALPHA", dtype=ti.f32)
RANDOM_SAMPLES = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "RANDOM_SAMPLES", dtype=ti.i32)
SAMPLE_BINS = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "SAMPLE_BINS", dtype=ti.i32)
SAMPLE_LEAK_RATIO = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "SAMPLE_LEAK_RATIO", dtype=ti.f32)
# REJECTION_MAX_ATTEMPTS = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "REJECTION_MAX_ATTEMPTS", dtype=ti.i32)

## hill climbing mutation
MUTATIONS_PER_ROUND = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "MUTATIONS_PER_ROUND", dtype=ti.i32)
MOVE_STEP = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "MOVE_STEP", dtype=ti.f32)
RADIUS_STEP = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "RADIUS_STEP", dtype=ti.f32)
THETA_STEP_RAD = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "THETA_STEP_RAD", dtype=ti.f32)
ALPHA_STEP = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "ALPHA_STEP", dtype=ti.f32)

## evaluation
SAMPLE_STEP = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "SAMPLE_STEP", dtype=ti.i32)

ti.init(arch=ti.vulkan)

gb1 = ti.graph.GraphBuilder()
gb1.dispatch(compute_error_field, canvas, target, error_field)
gb1.dispatch(sample_from_error_topk, RANDOM_SAMPLES, SAMPLE_BINS, SAMPLE_LEAK_RATIO, hist_buffer, error_field, sampled_pixels)
gb1.dispatch(generate_and_pick_best, MIN_RADIUS, MAX_RADIUS, MIN_ALPHA, MAX_ALPHA,
            RANDOM_SAMPLES, sampled_pixels, canvas, target, SAMPLE_STEP,
            best_ellipse, best_ycbcr, best_score)

# we have to cut the graph into 3 parts, since mutate_and_pick_best
# will be called multiple times in a loop, and we cannot put a loop inside the graph.
gb2 = ti.graph.GraphBuilder()
gb2.dispatch(mutate_and_pick_best, best_ellipse, MUTATIONS_PER_ROUND,
                     MOVE_STEP, RADIUS_STEP, THETA_STEP_RAD, ALPHA_STEP,
                     canvas, target, SAMPLE_STEP,
                     best_ellipse, best_ycbcr, best_score)

gb3 = ti.graph.GraphBuilder()
gb3.dispatch(apply_ellipse, best_ellipse, best_ycbcr, canvas)

g1 = gb1.compile()
g2 = gb2.compile()
g3 = gb3.compile()

mod = ti.aot.Module(ti.vulkan)
mod.add_graph("g1", g1)
mod.add_graph("g2", g2)
mod.add_graph("g3", g3)
mod.archive("graphs.tcm")