"""vk_queues.py — 用 ctypes 直接枚举本机 Vulkan 物理设备与队列族布局。

用途：定位「Python test_05 无窗口走 Compute_0、C++ my_app 走 3D」的差异，
先确认显卡到底暴露了哪些队列族（哪些是 graphics/compute/纯 compute）。
运行：
  ..\\.venv\\Scripts\\python.exe scripts/vk_queues.py
"""
import ctypes

vk = ctypes.WinDLL("vulkan-1.dll")

VK_STRUCTURE_TYPE_APPLICATION_INFO = 0
VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1

VK_QUEUE_GRAPHICS_BIT = 0x00000001
VK_QUEUE_COMPUTE_BIT = 0x00000002
VK_QUEUE_TRANSFER_BIT = 0x00000004


class VkApplicationInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("pApplicationName", ctypes.c_char_p),
        ("applicationVersion", ctypes.c_uint32),
        ("pEngineName", ctypes.c_char_p),
        ("engineVersion", ctypes.c_uint32),
        ("apiVersion", ctypes.c_uint32),
    ]


class VkInstanceCreateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("pApplicationInfo", ctypes.POINTER(VkApplicationInfo)),
        ("enabledLayerCount", ctypes.c_uint32),
        ("ppEnabledLayerNames", ctypes.POINTER(ctypes.c_char_p)),
        ("enabledExtensionCount", ctypes.c_uint32),
        ("ppEnabledExtensionNames", ctypes.POINTER(ctypes.c_char_p)),
    ]


class VkPhysicalDeviceProperties(ctypes.Structure):
    _fields_ = [
        ("apiVersion", ctypes.c_uint32),
        ("driverVersion", ctypes.c_uint32),
        ("vendorID", ctypes.c_uint32),
        ("deviceID", ctypes.c_uint32),
        ("deviceType", ctypes.c_uint32),
        ("deviceName", ctypes.c_char * 256),
        ("pipelineCacheUUID", ctypes.c_uint8 * 16),
    ]


class VkQueueFamilyProperties(ctypes.Structure):
    _fields_ = [
        ("queueFlags", ctypes.c_uint32),
        ("queueCount", ctypes.c_uint32),
        ("timestampValidBits", ctypes.c_uint32),
        ("minImageTransferGranularity", ctypes.c_uint32 * 3),
    ]


vk.vkCreateInstance.restype = ctypes.c_int32
vk.vkCreateInstance.argtypes = [ctypes.POINTER(VkInstanceCreateInfo),
                               ctypes.c_void_p,
                               ctypes.POINTER(ctypes.c_void_p)]
vk.vkEnumeratePhysicalDevices.restype = ctypes.c_int32
vk.vkEnumeratePhysicalDevices.argtypes = [ctypes.c_void_p,
                                          ctypes.POINTER(ctypes.c_uint32),
                                          ctypes.POINTER(ctypes.c_void_p)]
vk.vkGetPhysicalDeviceQueueFamilyProperties.restype = None
vk.vkGetPhysicalDeviceQueueFamilyProperties.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32),
    ctypes.POINTER(VkQueueFamilyProperties)]
vk.vkGetPhysicalDeviceProperties.restype = None
vk.vkGetPhysicalDeviceProperties.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(VkPhysicalDeviceProperties)]

app = VkApplicationInfo(VK_STRUCTURE_TYPE_APPLICATION_INFO, None, b"probe", 1,
                        b"probe", 1, 0x00400000)  # apiVersion 1.0.0
inst_ci = VkInstanceCreateInfo(VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, None, 0,
                               ctypes.pointer(app), 0, None, 0, None)
inst = ctypes.c_void_p()
res = vk.vkCreateInstance(ctypes.byref(inst_ci), None, ctypes.byref(inst))
print("vkCreateInstance:", res)
if res != 0:
    raise SystemExit(1)

count = ctypes.c_uint32(0)
vk.vkEnumeratePhysicalDevices(inst, ctypes.byref(count), None)
print("physical devices:", count.value)
devs = (ctypes.c_void_p * count.value)()
vk.vkEnumeratePhysicalDevices(inst, ctypes.byref(count), devs)

for di in range(count.value):
    props = VkPhysicalDeviceProperties()
    vk.vkGetPhysicalDeviceProperties(devs[di], ctypes.byref(props))
    print(f"\nDevice {di}: {props.deviceName.decode()} type={props.deviceType}")
    qcount = ctypes.c_uint32(0)
    vk.vkGetPhysicalDeviceQueueFamilyProperties(devs[di], ctypes.byref(qcount),
                                                None)
    qprops = (VkQueueFamilyProperties * qcount.value)()
    vk.vkGetPhysicalDeviceQueueFamilyProperties(devs[di], ctypes.byref(qcount),
                                                qprops)
    for qi in range(qcount.value):
        f = qprops[qi].queueFlags
        tags = []
        if f & VK_QUEUE_GRAPHICS_BIT:
            tags.append("GRAPHICS")
        if f & VK_QUEUE_COMPUTE_BIT:
            tags.append("COMPUTE")
        if f & VK_QUEUE_TRANSFER_BIT:
            tags.append("TRANSFER")
        pure_compute = (f & VK_QUEUE_COMPUTE_BIT) and not (
            f & VK_QUEUE_GRAPHICS_BIT)
        print(f"  queue family {qi}: flags=0x{f:x} [{'+'.join(tags)}]"
              f"{'  <-- 纯 compute' if pure_compute else ''} "
              f"count={qprops[qi].queueCount}")
