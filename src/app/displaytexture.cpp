#include "displaytexture.h"

#include "imgui_impl_vulkan.h" // ImGui_ImplVulkan_AddTexture / RemoveTexture
#include <taichi/taichi_vulkan.h> // ti_export_vulkan_memory

#include <spdlog/spdlog.h>

DisplayTexture::~DisplayTexture()
{
    destroy();
}

bool DisplayTexture::create(VkPhysicalDevice physical_device, VkDevice device, VkQueue queue, uint32_t queue_family,
                            uint32_t w, uint32_t h, VkFormat format)
{
    destroy();
    if (w == 0 || h == 0)
        return false;

    physical_device_ = physical_device;
    device_ = device;
    queue_ = queue;
    queue_family_ = queue_family;
    current_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    w_ = w;
    h_ = h;
    format_ = format;

    // R32G32B32_SFLOAT 的采样支持是 Vulkan 可选特性：不支持的设备上继续
    VkFormatProperties fp{};
    vkGetPhysicalDeviceFormatProperties(physical_device_, format_, &fp);
    if ((fp.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) == 0)
    {
        spdlog::error("DisplayTexture: format {} not sampleable on this device "
                      "(optimalTilingFeatures = {:#x}), may crash",
                      static_cast<int>(format_), fp.optimalTilingFeatures);
        // return false;
    }

    // 1. VkImage：TRANSFER_DST（接收拷贝）+ SAMPLED（ImGui 采样）
    VkImageCreateInfo ici = {};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = format_;
    ici.extent = {w_, h_, 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vkCreateImage(device_, &ici, nullptr, &image_) != VK_SUCCESS)
    {
        spdlog::error("DisplayTexture: vkCreateImage failed");
        destroy();
        return false;
    }

    // 2. 显存（DEVICE_LOCAL）
    VkMemoryRequirements mem_req;
    vkGetImageMemoryRequirements(device_, image_, &mem_req);
    VkMemoryAllocateInfo mai = {};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = mem_req.size;
    mai.memoryTypeIndex = findMemoryType(mem_req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (mai.memoryTypeIndex == (uint32_t)-1 || vkAllocateMemory(device_, &mai, nullptr, &memory_) != VK_SUCCESS)
    {
        spdlog::error("DisplayTexture: vkAllocateMemory failed");
        destroy();
        return false;
    }
    vkBindImageMemory(device_, image_, memory_, 0);

    // 3. VkImageView
    VkImageViewCreateInfo vci = {};
    vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image = image_;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = format_;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    if (vkCreateImageView(device_, &vci, nullptr, &view_) != VK_SUCCESS)
    {
        spdlog::error("DisplayTexture: vkCreateImageView failed");
        destroy();
        return false;
    }

    // 4. 命令池（TRANSIENT：一次性命令；uploadAndRegister 复用）
    VkCommandPoolCreateInfo cpci = {};
    cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    cpci.queueFamilyIndex = queue_family_;
    if (vkCreateCommandPool(device_, &cpci, nullptr, &pool_) != VK_SUCCESS)
    {
        spdlog::error("DisplayTexture: vkCreateCommandPool failed");
        destroy();
        return false;
    }
    return true;
}

void DisplayTexture::destroy()
{
    if (device_ == VK_NULL_HANDLE)
        return;

    // 确保 GPU 不再引用我们的资源
    vkDeviceWaitIdle(device_);

    if (descriptor_ != VK_NULL_HANDLE)
    {
        ImGui_ImplVulkan_RemoveTexture(descriptor_);
        descriptor_ = VK_NULL_HANDLE;
    }
    if (pool_ != VK_NULL_HANDLE)
    {
        vkDestroyCommandPool(device_, pool_, nullptr);
        pool_ = VK_NULL_HANDLE;
    }
    if (view_ != VK_NULL_HANDLE)
    {
        vkDestroyImageView(device_, view_, nullptr);
        view_ = VK_NULL_HANDLE;
    }
    if (image_ != VK_NULL_HANDLE)
    {
        vkDestroyImage(device_, image_, nullptr);
        image_ = VK_NULL_HANDLE;
    }
    if (memory_ != VK_NULL_HANDLE)
    {
        vkFreeMemory(device_, memory_, nullptr);
        memory_ = VK_NULL_HANDLE;
    }
    w_ = h_ = 0;
    format_ = VK_FORMAT_UNDEFINED;
    current_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
}

bool DisplayTexture::upload(const ti::NdArray<float> &src, ti::Runtime &runtime)
{
    if (image_ == VK_NULL_HANDLE || device_ == VK_NULL_HANDLE)
        return false;

    // 形状校验：src 必须是 (h_, w_) 且 elem_shape=(3,)，否则 vkCmdCopyBufferToImage
    // 会越界读 src buffer（DisplayTexture 用 w_/h_ 作 imageExtent/bufferRowLength）。
    const TiNdShape &s = src.shape();
    const TiNdShape &es = src.elem_shape();
    assert(s.dim_count == 2 && s.dims[0] == h_ && s.dims[1] == w_ &&
           es.dim_count == 1 && es.dims[0] == 3 &&
           "DisplayTexture::upload: src shape mismatch with texture w_/h_");

    // 导出 Taichi buffer 的底层 VkBuffer（同一 device，直接可用）
    TiVulkanMemoryInteropInfo mem_info = {};
    ti_export_vulkan_memory(runtime, src.memory().memory(), &mem_info);
    // 分配失败的 ndarray（TiMemory=null）会让 export 提前返回、mem_info 保持全零。
    // 不检查就把 null buffer 交给 vkCmdCopyBufferToImage，在多数驱动上直接闪退
    // （host 侧 null 句柄解引用 AV，或 GPU 侧 fault → DEVICE_LOST → abort）。
    if (mem_info.buffer == VK_NULL_HANDLE)
    {
        spdlog::error("DisplayTexture::upload: src ndarray has no VkBuffer "
                      "(allocation failed / OOM? shape mismatch? {}x{})",
                      w_, h_);
        return false;
    }

    // 录制单次命令：UNDEFINED/SHADER_READ_ONLY→TRANSFER_DST→拷贝→SHADER_READ_ONLY
    vkResetCommandPool(device_, pool_, 0);
    VkCommandBufferAllocateInfo cbai = {};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = pool_;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(device_, &cbai, &cmd);

    VkCommandBufferBeginInfo cbbi = {};
    cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &cbbi);
    recordCopy(cmd, mem_info.buffer);
    vkEndCommandBuffer(cmd);

    // 提交 + 等 fence（注册 ImGui 前必须确保拷贝完成）
    VkFenceCreateInfo fci = {};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence = VK_NULL_HANDLE;
    vkCreateFence(device_, &fci, nullptr, &fence);
    VkSubmitInfo si = {};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    // 双队列：kernel 在 q1（compute），拷贝提交到 q0（graphics）。
    // 跨队列可见性由调用方保证（worker runtime.wait() 排空 q1 / copyVectorToTarget
    // 末尾 wait）——fence wait 的第二同步范围包含 wait 之后 host 的 vkQueueSubmit，
    // 因此本拷贝必然看见 kernel 的全部写入，无需 semaphore。
    VkResult result = vkQueueSubmit(queue_, 1, &si, fence);
    if (result != VK_SUCCESS)
    {
        vkDestroyFence(device_, fence, nullptr);
        vkFreeCommandBuffers(device_, pool_, 1, &cmd);
        return false;
    }

    result = vkWaitForFences(device_, 1, &fence, VK_TRUE, UINT64_MAX);
    vkDestroyFence(device_, fence, nullptr);
    // 释放本次 command buffer（避免反复 upload 在 pool 里累积 cmd 对象）
    vkFreeCommandBuffers(device_, pool_, 1, &cmd);
    if (result != VK_SUCCESS)
        return false;

    current_layout_ = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    return true;
}

bool DisplayTexture::uploadAndRegister(const ti::NdArray<float> &src, ti::Runtime &runtime)
{
    if (!upload(src, runtime))
        return false;
    return registerTexture();
}

bool DisplayTexture::registerTexture()
{
    if (descriptor_ != VK_NULL_HANDLE)
        return true; // 已注册
    if (view_ == VK_NULL_HANDLE)
        return false;

    descriptor_ = ImGui_ImplVulkan_AddTexture(view_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    if (descriptor_ == VK_NULL_HANDLE)
    {
        spdlog::error("DisplayTexture: ImGui_ImplVulkan_AddTexture failed");
        return false;
    }
    return true;
}

void DisplayTexture::recordCopy(VkCommandBuffer cmd, VkBuffer src)
{
    VkImageSubresourceRange range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    // 1. UNDEFINED → TRANSFER_DST（内容不保留，整图覆盖）
    VkImageMemoryBarrier to_dst = {};
    to_dst.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    to_dst.oldLayout = current_layout_;
    to_dst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    to_dst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_dst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_dst.image = image_;
    to_dst.subresourceRange = range;
    to_dst.srcAccessMask = current_layout_ == VK_IMAGE_LAYOUT_UNDEFINED ? 0 : VK_ACCESS_SHADER_READ_BIT;
    to_dst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    const VkPipelineStageFlags src_stage =
        current_layout_ == VK_IMAGE_LAYOUT_UNDEFINED ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
                                                      : VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    vkCmdPipelineBarrier(cmd, src_stage, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &to_dst);

    // 2. 拷贝（buffer 是紧密 float[3] = 12B/px；image 是 RGBA32F = 16B/px。
    //    bufferRowLength = w 让驱动按 12B stride 读 buffer、写 16B stride image）
    VkBufferImageCopy region = {};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {w_, h_, 1};
    region.bufferRowLength = w_;
    region.bufferImageHeight = h_;
    vkCmdCopyBufferToImage(cmd, src, image_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    // 3. TRANSFER_DST → SHADER_READ_ONLY（ImGui 采样）
    VkImageMemoryBarrier to_read = {};
    to_read.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    to_read.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    to_read.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    to_read.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_read.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_read.image = image_;
    to_read.subresourceRange = range;
    to_read.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    to_read.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &to_read);
}

uint32_t DisplayTexture::findMemoryType(uint32_t type_bits, VkMemoryPropertyFlags props) const
{
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(physical_device_, &mem_props);
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i)
    {
        if ((type_bits & (1u << i)) && (mem_props.memoryTypes[i].propertyFlags & props) == props)
            return i;
    }
    return (uint32_t)-1;
}
