#pragma once
#include <cstdint>

struct Ellipse
{
    struct
    {
        float x;
        float y;
        float rx;
        float ry;
        float alpha;
        float theta_rad;
    } Shape;

    struct
    {
        float Y;
        float Cb;
        float Cr;
    } Color;
};

struct PainterParams
{
    // gpu
    // sampling ellipse
    float min_radius;
    float max_radius;
    float min_alpha;
    float max_alpha;
    int32_t random_samples;
    int32_t sample_bins;
    float sample_leak_ratio;
    int32_t blur_size;

    // canvas
    uint32_t canvas_w;
    uint32_t canvas_h;

    // sharpen
    float sharpen_intensity;

    // hill climbing mutation
    int32_t mutations_per_round;
    float move_step;
    float radius_step;
    float theta_step_rad;
    float alpha_step;

    // evaluation
    float sample_step_deno;

    // cpu
    int32_t hill_climb_rounds;
    uint32_t total_shapes;
};