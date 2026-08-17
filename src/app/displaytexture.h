#pragma once
#include <cstdint>
#include <volk.h> // Vulkan 类型 + 全局函数指针（VK_NO_PROTOTYPES）
#include <taichi/cpp/taichi.hpp>

// 显示纹理（方案 B）：把 Taichi ndarray（VkBuffer）通过 vkCmdCopyBufferToImage
// 拷进 VkImage，并注册成 ImGui 可采样的 VkDescriptorSet。
//
// 线程约定：
//  - create() / destroy() / registerTexture() / uploadAndRegister() 必须在主线程调用
//    （涉及 ImGui_ImplVulkan_AddTexture 与 descriptor 分配）。
//  - uploadAndRegister() 是当前唯一上传路径（target 与 canvas 共用）：内部
//    runtime.wait() + 同步拷贝 + fence 等待，适合静态图（target）与主线程驱动的
//    动态图（canvas 每步由 worker 置 canvas_ready、主线程消费后调用）。
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

    // 主线程：仅注册 ImGui 纹理（不拷贝数据）。create() 后调用一次，
    // 内容由 uploadAndRegister 持续刷新（canvas 的动态更新路径）。
    bool registerTexture();

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

    // The image is sampled between uploads, so subsequent copies must transition
    // from SHADER_READ_ONLY_OPTIMAL rather than always pretending it is UNDEFINED.
    VkImageLayout current_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;

    uint32_t w_ = 0, h_ = 0;
    VkFormat format_ = VK_FORMAT_UNDEFINED;
};
