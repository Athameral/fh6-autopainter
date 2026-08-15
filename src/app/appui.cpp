#include "appui.h"
#include "imgui.h"
#include "taichi/cpp/taichi.hpp"

#include "stb_image.h"

#include <future>
#include <iostream>


void App::renderUI()
{
    renderControlPanel();
    renderTargetPanel();
}

void App::renderControlPanel()
{
    ImGui::Begin("Control Panel");
    ImGui::Text("GPU Worker Status: %s", gpu_worker.should_exit ? "Stopped" : "Running");
    if (ImGui::Button("Start Worker"))
    {
        gpu_worker.should_exit = false;
        gpu_worker.launch_graph = true;
    }
    if (ImGui::Button("Stop Worker"))
    {
        gpu_worker.should_exit = true;
        gpu_worker.launch_graph = true;
        gpu_worker.generate_interrupted = true;
    }
    ImGui::End();
}

void App::renderTargetPanel()
{
    static std::atomic<bool> image_ready = false;
    ImGui::Begin("Target Panel");
    ImGui::Text("Target Image: %s", target_image_path.c_str());
    if (target_image_path != old_target_image_path)
    {
        image_ready = false;
        old_target_image_path = target_image_path;
        // load image
        int img_w, img_h;
        // read as rgba8
        auto fut = std::async(std::launch::async, [&]() {
            auto *data = stbi_loadf(target_image_path.c_str(), &img_w, &img_h, nullptr, 4);
            gpu_worker.params.canvas_h = img_h;
            gpu_worker.params.canvas_w = img_w;
            gpu_worker.remakeBuffer();
            std::vector<float> target_rgb(img_w * img_h * 3);
            std::vector<int32_t> target_alpha(img_w * img_h);
            if (data)
            {
                for (int i = 0; i < img_w * img_h; ++i)
                {
                    target_rgb[i * 3 + 0] = data[i * 4 + 0];
                    target_rgb[i * 3 + 1] = data[i * 4 + 1];
                    target_rgb[i * 3 + 2] = data[i * 4 + 2];
                    target_alpha[i] = data[i * 4 + 3] > 1e-2 ? 1 : 0;
                }
                gpu_worker.gpu_buffer.target_origin.write(target_rgb.data(), target_rgb.size());
                gpu_worker.gpu_buffer.valid_mask.write(target_alpha.data(), target_alpha.size());
                stbi_image_free(data);
            }
            else
            {
                std::cerr << "Failed to load image: " << target_image_path << std::endl;
            }
            // end
            image_ready = true;
        });
    }
    ImGui::Text("Image Ready: %s", image_ready ? "Yes" : "No");
    // ImGui::
    ImGui::End();
}

void App::setTargetImagePath(const char *path)
{
    target_image_path = path;
}

App::App(GLFWwindow *window, ti::Runtime &runtime, const PainterParams &params, const ti::AotModule &aot_module)
    : window(window), runtime(runtime), aot_module(aot_module), params(params), gpu_worker(runtime, params, aot_module)
{
    std::cout << "App initialized with GPUWorker." << std::endl;
    glfwSetWindowUserPointer(window, this);
    glfwSetDropCallback(window, file_dragin_callback);
}

void file_dragin_callback(GLFWwindow *window, int count, const char **paths)
{
    assert(window != nullptr);
    if (count > 1)
    {
        std::cerr << "Only one file can be dragged in at a time." << std::endl;
        return;
    }
    assert(count == 1);
    auto *app = (App *)glfwGetWindowUserPointer(window);
    assert(app != nullptr);
    app->setTargetImagePath(paths[0]);
}