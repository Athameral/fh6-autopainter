#pragma once
#include <cstdint>
#include <volk.h> // Vulkan 类型 + 全局函数指针（VK_NO_PROTOTYPES）
#include <taichi/cpp/taichi.hpp>

// 显示纹理（方案 B）：把 Taichi ndarray（VkBuffer）通过 vkCmdCopyBufferToImage
// 拷进 VkImage，并注册成 ImGui 可采样的 VkDescriptorSet。
//
// 线程约定：
//  - create() / destroy() / registerTexture() / upload() 必须在主线程调用
//    （涉及 ImGui_ImplVulkan_AddTexture 与 descriptor 分配）。
//  - queueUpload() 设计为 worker 线程调用：只录制命令 + vkQueueSubmit（无 fence），
//    依赖"与 Taichi 共享同一队列 + 调用方每步 runtime.wait()"保证拷贝先于下一次复用完成。
//  - 同一实例不要混用 upload() 与 queueUpload()（两者共用 command pool）。
class DisplayTexture
{
  public:
    DisplayTexture() = default;
    ~DisplayTexture();
    DisplayTexture(const DisplayTexture &) = delete;
    DisplayTexture &operator=(const DisplayTexture &) = delete;

    // 创建 VkImage + 显存 + view + command pool（不注册 descriptor）。
    bool create(VkPhysicalDevice physical_device, VkDevice device, VkQueue queue, uint32_t queue_family,
                uint32_t w, uint32_t h, VkFormat format);
    void destroy(); // 逆序释放：descriptor → view → image → memory → pool

    // 主线程：把 src 全量拷入并注册 ImGui 纹理（内部 runtime.wait() + fence 同步）。
    // 适合静态图（target）：图片加载完成后调用一次。
    bool uploadAndRegister(const ti::NdArray<float> &src, ti::Runtime &runtime);

    // 主线程：仅注册 ImGui 纹理（不拷贝数据）。适合动态图（canvas）——
    // create() 后调用一次，内容由 queueUpload 持续刷新。
    bool registerTexture();

    // worker 线程：录制拷贝命令并提交。内部用持久 fence 等待上一次拷贝完成后
    // 才 reset 命令池（防止 reset 仍在执行的命令缓冲 → DEVICE_LOST），
    // 因此可以安全地每步调用，无需调用方先 runtime.wait()。
    void queueUpload(const ti::NdArray<float> &src, ti::Runtime &runtime);

    VkDescriptorSet descriptor() const { return descriptor_; }
    uint32_t width() const { return w_; }
    uint32_t height() const { return h_; }
    bool is_valid() const { return image_ != VK_NULL_HANDLE; }

  private:
    uint32_t findMemoryType(uint32_t type_bits, VkMemoryPropertyFlags props) const;
    void recordCopy(VkCommandBuffer cmd, VkBuffer src);

    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    uint32_t queue_family_ = 0;

    VkImage image_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    VkImageView view_ = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_ = VK_NULL_HANDLE;
    VkCommandPool pool_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE; // queueUpload 同步：等上次拷贝完成后再 reset pool

    uint32_t w_ = 0, h_ = 0;
    VkFormat format_ = VK_FORMAT_UNDEFINED;
};
