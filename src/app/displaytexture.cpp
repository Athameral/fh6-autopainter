#include "displaytexture.h"

#include "imgui_impl_vulkan.h" // ImGui_ImplVulkan_AddTexture / RemoveTexture
#include <taichi/taichi_vulkan.h> // ti_export_vulkan_memory

#include <cstdio>

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
    w_ = w;
    h_ = h;
    format_ = format;

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
    if (fence_ != VK_NULL_HANDLE)
    {
        vkDestroyFence(device_, fence_, nullptr);
        fence_ = VK_NULL_HANDLE;
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
}

bool DisplayTexture::uploadAndRegister(const ti::NdArray<float> &src, ti::Runtime &runtime)
{
    if (image_ == VK_NULL_HANDLE || device_ == VK_NULL_HANDLE)
        return false;

    // NdArray.write() 是 ti_map_memory + memcpy（走 staging，异步提交），
    // 必须等队列执行完，GPU 侧 buffer 内容才可见。
    runtime.wait();

    // 导出 Taichi buffer 的底层 VkBuffer（同一 device，直接可用）
    TiVulkanMemoryInteropInfo mem_info = {};
    ti_export_vulkan_memory(runtime, src.memory().memory(), &mem_info);

    // 录制单次命令：UNDEFINED→TRANSFER_DST→拷贝→SHADER_READ_ONLY
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
    // 与 Taichi 同一队列：kernel 已由 runtime.wait() 同步完成，拷贝按序执行
    vkQueueSubmit(queue_, 1, &si, fence);
    vkWaitForFences(device_, 1, &fence, VK_TRUE, UINT64_MAX);
    vkDestroyFence(device_, fence, nullptr);

    // 注册到 ImGui（消耗 1 个 SAMPLED_IMAGE + 1 个 SAMPLER descriptor）
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

void DisplayTexture::queueUpload(const ti::NdArray<float> &src, ti::Runtime &runtime)
{
    if (image_ == VK_NULL_HANDLE || device_ == VK_NULL_HANDLE)
    {
        std::fprintf(stderr, "DisplayTexture::queueUpload: texture not created, skipping\n");
        return;
    }

    // 等上一次拷贝完成再复位 pool：避免 reset 仍在 GPU 上执行的命令缓冲
    // （VK_ERROR_DEVICE_LOST 根源）。同队列 FIFO → 也保证此前的 kernel 全部完成。
    vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);
    vkResetFences(device_, 1, &fence_);

    TiVulkanMemoryInteropInfo mem_info = {};
    ti_export_vulkan_memory(runtime, src.memory().memory(), &mem_info);

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

    // 挂上传 fence：下次 queueUpload 先等它，保证本拷贝完成后才 reset pool。
    // 同一队列上，本拷贝排在 Taichi kernel 之后、主线程 ImGui 渲染之前，
    // GPU 按提交顺序执行，天然正确。
    VkSubmitInfo si = {};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    vkQueueSubmit(queue_, 1, &si, fence_);
}

void DisplayTexture::recordCopy(VkCommandBuffer cmd, VkBuffer src)
{
    VkImageSubresourceRange range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    // 1. UNDEFINED → TRANSFER_DST（内容不保留，整图覆盖）
    VkImageMemoryBarrier to_dst = {};
    to_dst.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    to_dst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    to_dst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    to_dst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_dst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_dst.image = image_;
    to_dst.subresourceRange = range;
    to_dst.srcAccessMask = 0;
    to_dst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &to_dst);

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
