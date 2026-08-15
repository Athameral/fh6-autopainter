#pragma once
#include "gpuworker.h"
#include "structures.h"
#include <imgui.h>
#include <taichi/cpp/taichi.hpp>
#include <GLFW/glfw3.h>

void file_dragin_callback(GLFWwindow *window, int count, const char **paths);

class App
{
  public:
    App(GLFWwindow *window, ti::Runtime &runtime, const PainterParams &params, const ti::AotModule &aot_module);

    void renderUI();
    void setTargetImagePath(const char *path);

  private:
    GLFWwindow *window;
    ti::Runtime &runtime;
    const ti::AotModule &aot_module;
    ti::Texture canvas_display_texture, target_display_texture;
    PainterParams params;
    GPUWorker gpu_worker;
    std::string target_image_path, old_target_image_path;
    void renderControlPanel();
    void renderTargetPanel();
};
