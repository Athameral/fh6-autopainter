#include "injector.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <spdlog/spdlog.h>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#define NOGDI
#include <windows.h>
#include <tlhelp32.h>
#endif

namespace fh6::injector
{
namespace detail
{

// Diagnostic sink: forwards to the Options callback when set, spdlog otherwise,
// and optionally accumulates into a WriteResult's diagnostics string.
class Log
{
  public:
	explicit Log(const Injector::Options &options, std::string *diagnostics)
		: log_(options.log), diagnostics_(diagnostics)
	{
	}

	void operator()(std::string_view message) const
	{
		write(spdlog::level::info, message);
	}
	void warning(std::string_view message) const
	{
		write(spdlog::level::warn, message);
	}
	void error(std::string_view message) const
	{
		write(spdlog::level::err, message);
	}

  private:
	void write(spdlog::level::level_enum level, std::string_view message) const
	{
		if (diagnostics_ != nullptr)
		{
			if (!diagnostics_->empty())
				diagnostics_->push_back('\n');
			diagnostics_->append(message.data(), message.size());
		}
		if (log_)
			log_(message);
		else
			spdlog::log(level, "[inject] {}", std::string(message));
	}

	std::function<void(std::string_view)> log_;
	std::string *diagnostics_;
};

#ifndef _WIN32
// Placeholder so the header's unique_ptr member stays destructible off Windows.
class RemoteProcess
{
};
#endif

} // namespace detail

namespace
{

// ---- build-specific reverse-engineered offsets (single source of truth) ----

struct GameProfile
{
	std::uint32_t livery_count_offset;  // CLiveryGroup: uint16 layer count
	std::uint32_t layer_table_offset;   // CLiveryGroup: vector<layer*> begin/end/capacity
	std::uint32_t pos_offset;           // layer: float[2]
	std::uint32_t scale_offset;         // layer: float[2]
	std::uint32_t rotation_offset;      // layer: float
	std::uint32_t skew_offset;          // layer: float
	std::uint32_t color_offset;         // layer: RGBA bytes
	std::uint32_t mask_offset;          // layer: 1 byte (NOT the group's table at 0x78)
	std::uint32_t shape_id_offset;      // layer: uint16 shape word
	std::uint32_t mesh_data_offset;     // layer: std::string data pointer
	std::uint32_t mesh_size_offset;     // layer: std::string size
	std::uint32_t mesh_capacity_offset; // layer: std::string capacity
};

constexpr GameProfile kFH6Profile{
	0x5A, // CLiveryGroup::layer count
	0x78, // CLiveryGroup::vector<layer*> begin/end/capacity
	0x18, // layer position (float[2])
	0x28, // layer scale (float[2])
	0x50, // layer rotation (float)
	0x70, // layer skew (float)
	0x74, // layer color (RGBA bytes)
	0x78, // layer mask (byte; distinct object from the group table)
	0x7A, // layer shape word (uint16)
	0x80, // std::string data pointer
	0x90, // std::string size
	0x98, // std::string capacity
};

constexpr std::uint16_t kWordCircle = 0x0066; // ellipse = circle + non-uniform scale (0x88 renders as a crescent)
constexpr float kEditorHalfX = 1984.0f;       // canvas half-width in editor position units
constexpr float kEditorHalfY = 1141.0f;       // canvas half-height in editor position units
constexpr float kScaleBase = 64.0f;           // editor units per scale-1.0
constexpr float kPi = 3.14159265358979323846f;

constexpr std::uintptr_t kMinimumUserAddress = 0x10000u;
constexpr std::size_t kScanChunkBytes = 64u * 1024u * 1024u; // heap scans read in 64 MB chunks
constexpr std::size_t kMaxRegionRead = 256u * 1024u * 1024u; // directed vtable scan skips larger regions
constexpr int kMinCountScanCount = 16;                       // uint16 count matches below 16 are noise
constexpr float kStrictScaleEpsilon = 0.01f;                 // "scale not both ~0" threshold

// ---- pixel-space -> editor-space conversion (ShapeToWord.md) ----

struct LayerWrite
{
	float x = 0.0f;
	float y = 0.0f;
	float sx = 0.0f;
	float sy = 0.0f;
	float rotation = 0.0f;
	float skew = 0.0f;
	std::array<std::uint8_t, 4> color{};
	std::uint16_t word = 0;
};

struct CanvasMap
{
	float width = 1.0f;
	float height = 1.0f;
	float k = 1.0f;
};

template <typename... Args>
std::string joinDiagnostic(Args &&...args)
{
	std::ostringstream stream;
	(stream << ... << std::forward<Args>(args));
	return stream.str();
}

std::string hexAddress(std::uintptr_t address)
{
	std::ostringstream stream;
	stream << "0x" << std::hex << address;
	return stream.str();
}

bool isFinite(float value)
{
	return std::isfinite(value) != 0;
}

bool isFiniteLayer(const LayerWrite &layer)
{
	return isFinite(layer.x) && isFinite(layer.y) && isFinite(layer.sx) && isFinite(layer.sy) &&
		   isFinite(layer.rotation) && isFinite(layer.skew);
}

std::uint8_t toByte(float value)
{
	value = std::clamp(value, 0.0f, 1.0f);
	const long rounded = std::lround(static_cast<double>(value) * 255.0);
	return static_cast<std::uint8_t>(std::clamp(rounded, 0L, 255L));
}

// FH6's editor reserves a tiny non-zero alpha floor (ShapeToWord.md); the clear
// writer deliberately bypasses this remap.
std::uint8_t fh6Alpha(float alpha)
{
	alpha = std::clamp(alpha, 0.0f, 1.0f);
	constexpr double minPercent = 0.78;
	const double percent = minPercent + static_cast<double>(alpha) * (100.0 - minPercent);
	const long rounded = std::lround(percent / 100.0 * 255.0);
	return static_cast<std::uint8_t>(std::clamp(rounded, 0L, 255L));
}

CanvasMap makeCanvasMap(std::uint32_t width, std::uint32_t height)
{
	const float w = static_cast<float>(width);
	const float h = static_cast<float>(height);
	const float kx = 2.0f * kEditorHalfX / w;
	const float ky = 2.0f * kEditorHalfY / h;
	return CanvasMap{w, h, std::min(kx, ky)};
}

// windows.h's GDI declares a global Ellipse function; qualify the struct tag
// the same way the previous implementation did.
bool ellipseToLayer(const struct ::Ellipse &ellipse, const CanvasMap &canvas, LayerWrite &out, std::string &reason)
{
	const float cx = ellipse.Shape.x;
	const float cy = ellipse.Shape.y;
	const float rx = ellipse.Shape.rx;
	const float ry = ellipse.Shape.ry;
	const float alpha = ellipse.Shape.alpha;
	const float theta = ellipse.Shape.theta_rad;

	if (!isFinite(cx) || !isFinite(cy) || !isFinite(rx) || !isFinite(ry) || !isFinite(alpha) || !isFinite(theta))
	{
		reason = "contains NaN or infinity";
		return false;
	}
	if (rx <= 0.0f || ry <= 0.0f)
	{
		reason = "has a non-positive radius";
		return false;
	}

	out.x = (cx - canvas.width * 0.5f) * canvas.k;
	out.y = (canvas.height * 0.5f - cy) * canvas.k; // editor Y is up
	out.sx = rx * canvas.k / kScaleBase;
	out.sy = ry * canvas.k / kScaleBase;
	out.rotation = -theta * 180.0f / kPi; // Y-flip reverses rotation sense
	out.skew = 0.0f;
	out.word = kWordCircle;

	// The Taichi pipeline stores the optimized color as YCbCr plus a separate
	// alpha; convert to the raw RGBA bytes FH6 expects.
	const float y = ellipse.Color.Y;
	const float cb = ellipse.Color.Cb;
	const float cr = ellipse.Color.Cr;
	if (!isFinite(y) || !isFinite(cb) || !isFinite(cr))
	{
		reason = "color contains NaN or infinity";
		return false;
	}
	const float red = y + 1.402f * cr;
	const float green = y - 0.344136f * cb - 0.714136f * cr;
	const float blue = y + 1.772f * cb;
	out.color = {toByte(red), toByte(green), toByte(blue), fh6Alpha(alpha)};

	if (!isFiniteLayer(out) || std::abs(out.x) > 10000.0f || std::abs(out.y) > 10000.0f ||
		std::abs(out.sx) > 10000.0f || std::abs(out.sy) > 10000.0f || std::abs(out.rotation) > 100000.0f)
	{
		reason = "maps outside the conservative FH6 field ranges";
		return false;
	}
	return true;
}

} // namespace

#ifdef _WIN32

namespace
{

// ---- Win32 plumbing ----

std::string win32Error(DWORD error)
{
	if (error == ERROR_SUCCESS)
		return "unknown Win32 error";

	wchar_t buffer[512]{};
	const DWORD length = FormatMessageW(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
										nullptr,
										error,
										0,
										buffer,
										static_cast<DWORD>(std::size(buffer)),
										nullptr);
	std::wstring message(buffer, buffer + length);
	while (!message.empty() && (message.back() == L'\r' || message.back() == L'\n' || message.back() == L' '))
		message.pop_back();
	if (message.empty())
		return "Win32 error " + std::to_string(error);

	int utf8Length = WideCharToMultiByte(CP_UTF8, 0, message.data(), static_cast<int>(message.size()), nullptr, 0,
										 nullptr, nullptr);
	if (utf8Length <= 0)
		return "Win32 error " + std::to_string(error);
	std::string utf8(static_cast<std::size_t>(utf8Length), '\0');
	WideCharToMultiByte(CP_UTF8,
						0,
						message.data(),
						static_cast<int>(message.size()),
						utf8.data(),
						utf8Length,
						nullptr,
						nullptr);
	return utf8 + " (code " + std::to_string(error) + ")";
}

std::string wideToUtf8(std::wstring_view value)
{
	if (value.empty())
		return {};
	const int length = WideCharToMultiByte(CP_UTF8,
										   0,
										   value.data(),
										   static_cast<int>(value.size()),
										   nullptr,
										   0,
										   nullptr,
										   nullptr);
	if (length <= 0)
		return {};
	std::string result(static_cast<std::size_t>(length), '\0');
	WideCharToMultiByte(CP_UTF8,
						0,
						value.data(),
						static_cast<int>(value.size()),
						result.data(),
						length,
						nullptr,
						nullptr);
	return result;
}

bool equalsIgnoreCase(std::wstring_view lhs, std::wstring_view rhs)
{
	if (lhs.size() != rhs.size())
		return false;
	for (std::size_t i = 0; i < lhs.size(); ++i)
	{
		const wchar_t left = (lhs[i] >= L'A' && lhs[i] <= L'Z') ? lhs[i] + (L'a' - L'A') : lhs[i];
		const wchar_t right = (rhs[i] >= L'A' && rhs[i] <= L'Z') ? rhs[i] + (L'a' - L'A') : rhs[i];
		if (left != right)
			return false;
	}
	return true;
}

bool equalsIgnoreCaseAscii(std::string_view lhs, std::string_view rhs)
{
	if (lhs.size() != rhs.size())
		return false;
	for (std::size_t i = 0; i < lhs.size(); ++i)
	{
		const char left = (lhs[i] >= 'A' && lhs[i] <= 'Z') ? static_cast<char>(lhs[i] + ('a' - 'A')) : lhs[i];
		const char right = (rhs[i] >= 'A' && rhs[i] <= 'Z') ? static_cast<char>(rhs[i] + ('a' - 'A')) : rhs[i];
		if (left != right)
			return false;
	}
	return true;
}

struct ProcessInfo
{
	std::uint32_t pid = 0;
	std::string name;
};

bool findFH6Process(ProcessInfo &out, std::string &error, const detail::Log &log)
{
	HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
	if (snapshot == INVALID_HANDLE_VALUE)
	{
		error = "CreateToolhelp32Snapshot failed: " + win32Error(GetLastError());
		return false;
	}

	PROCESSENTRY32W entry{};
	entry.dwSize = sizeof(entry);
	BOOL ok = Process32FirstW(snapshot, &entry);
	while (ok != FALSE)
	{
		if (equalsIgnoreCase(entry.szExeFile, L"ForzaHorizon6.exe"))
		{
			out.pid = entry.th32ProcessID;
			out.name = wideToUtf8(entry.szExeFile);
			CloseHandle(snapshot);
			log(joinDiagnostic("target process found: pid=", out.pid));
			return true;
		}
		ok = Process32NextW(snapshot, &entry);
	}

	CloseHandle(snapshot);
	error = "ForzaHorizon6.exe was not found; start FH6 and open its Vinyl Group Editor first";
	return false;
}

struct Region
{
	std::uintptr_t base = 0;
	std::size_t size = 0;
	DWORD state = 0;
	DWORD type = 0;
	DWORD protect = 0;
};

} // namespace

namespace detail
{

// Read/write/query wrapper around one opened game process. The region snapshot
// is taken lazily on the first scan; isPrivateWritable() answers from it (fast
// path for the scoring loops) while isPrivateWritableLive() always re-queries
// for the final write gate.
class RemoteProcess
{
  public:
	RemoteProcess(HANDLE handle, std::uint32_t pid, std::string name)
		: handle_(handle), pid_(pid), name_(std::move(name))
	{
		SYSTEM_INFO systemInfo{};
		GetSystemInfo(&systemInfo);
		allocationGranularity_ =
			systemInfo.dwAllocationGranularity != 0 ? systemInfo.dwAllocationGranularity : 0x10000u;
		maxUserAddress_ = reinterpret_cast<std::uintptr_t>(systemInfo.lpMaximumApplicationAddress);
		if (maxUserAddress_ < kMinimumUserAddress)
			maxUserAddress_ = std::numeric_limits<std::uintptr_t>::max();
	}

	RemoteProcess(const RemoteProcess &) = delete;
	RemoteProcess &operator=(const RemoteProcess &) = delete;

	~RemoteProcess()
	{
		close();
	}

	bool valid() const
	{
		return handle_ != nullptr;
	}

	// Cheap liveness probe: a terminated process keeps its handle, but the
	// handle becomes signaled (requires SYNCHRONIZE access).
	bool alive() const
	{
		return handle_ != nullptr && WaitForSingleObject(handle_, 0) == WAIT_TIMEOUT;
	}

	std::uint32_t pid() const
	{
		return pid_;
	}
	const std::string &name() const
	{
		return name_;
	}
	std::uintptr_t maxUserAddress() const
	{
		return maxUserAddress_;
	}

	bool isUserPointer(std::uintptr_t address) const
	{
		return address >= kMinimumUserAddress && address <= maxUserAddress_;
	}

	bool readBytes(std::uintptr_t address, void *buffer, std::size_t size) const
	{
		if (size == 0)
			return true;
		if (!isSafeRange(address, size))
			return false;
		SIZE_T bytesRead = 0;
		const BOOL ok = ReadProcessMemory(handle_,
										  reinterpret_cast<LPCVOID>(address),
										  buffer,
										  static_cast<SIZE_T>(size),
										  &bytesRead);
		return ok != FALSE && bytesRead == size;
	}

	bool writeBytes(std::uintptr_t address, const void *buffer, std::size_t size) const
	{
		if (size == 0)
			return true;
		if (!isSafeRange(address, size))
			return false;
		SIZE_T bytesWritten = 0;
		const BOOL ok = WriteProcessMemory(handle_,
										   reinterpret_cast<LPVOID>(address),
										   buffer,
										   static_cast<SIZE_T>(size),
										   &bytesWritten);
		return ok != FALSE && bytesWritten == size;
	}

	template <typename T>
	bool readValue(std::uintptr_t address, T &value) const
	{
		static_assert(std::is_trivially_copyable_v<T>);
		return readBytes(address, &value, sizeof(T));
	}

	template <typename T>
	bool writeValue(std::uintptr_t address, const T &value) const
	{
		static_assert(std::is_trivially_copyable_v<T>);
		return writeBytes(address, &value, sizeof(T));
	}

	bool query(std::uintptr_t address, MEMORY_BASIC_INFORMATION &info) const
	{
		std::memset(&info, 0, sizeof(info));
		const SIZE_T queried = VirtualQueryEx(handle_, reinterpret_cast<LPCVOID>(address), &info, sizeof(info));
		return queried != 0;
	}

	bool isPrivateWritable(std::uintptr_t address, std::size_t size = 1) const
	{
		return isPrivateWritableImpl(address, size, false);
	}

	// Live VirtualQueryEx for the final write gate: the snapshot may be stale
	// relative to editor allocations that happened after it was taken.
	bool isPrivateWritableLive(std::uintptr_t address, std::size_t size = 1) const
	{
		return isPrivateWritableImpl(address, size, true);
	}

	bool isReadable(std::uintptr_t address, std::size_t size = 1) const
	{
		MEMORY_BASIC_INFORMATION info{};
		if (!isSafeRange(address, size) || !query(address, info))
			return false;
		if (info.State != MEM_COMMIT || !isReadableProtection(info.Protect))
			return false;
		return contains(info, address, size);
	}

	std::uintptr_t moduleBase() const
	{
		HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32, pid_);
		if (snapshot == INVALID_HANDLE_VALUE)
			return 0;

		MODULEENTRY32W module{};
		module.dwSize = sizeof(module);
		std::uintptr_t result = 0;
		if (Module32FirstW(snapshot, &module) != FALSE)
			result = reinterpret_cast<std::uintptr_t>(module.modBaseAddr);
		CloseHandle(snapshot);
		return result; // first module is the .exe
	}

	void forEachRegion(DWORD typeFilter, bool writableOnly, const std::function<bool(const Region &)> &callback) const
	{
		ensureRegionCache();
		for (const Region &region : regionCache_)
		{
			const bool typeMatches = typeFilter == 0 || region.type == typeFilter;
			const bool protectionMatches =
				writableOnly ? isWritableProtection(region.protect) : isReadableProtection(region.protect);
			if (region.state != MEM_COMMIT || !typeMatches || !protectionMatches)
				continue;
			if (!callback(region))
				return;
		}
	}

	// Chunked pattern scan over committed regions matching the filter. Regions
	// larger than maxRegionSize (when non-zero) are skipped: directed scans do
	// not need the huge mapped files. Returns false when onMatch stopped early.
	bool scanPattern(DWORD typeFilter,
					 bool writableOnly,
					 const std::vector<std::uint8_t> &pattern,
					 const std::function<bool(std::uintptr_t)> &onMatch,
					 const std::function<void(std::uint64_t)> &onProgress = {},
					 std::size_t maxRegionSize = 0) const
	{
		if (pattern.empty())
			return true;

		bool keepScanning = true;
		std::uint64_t scanned = 0;
		forEachRegion(typeFilter, writableOnly, [&](const Region &region) {
			if (maxRegionSize != 0 && region.size > maxRegionSize)
				return true;
			if (!scanRegion(region, pattern, onMatch, scanned, onProgress))
			{
				keepScanning = false;
				return false;
			}
			return true;
		});
		return keepScanning;
	}

	// Drop the region snapshot so the next scan rebuilds it from live queries.
	void refreshRegions() const
	{
		regionCacheInitialized_ = false;
		regionCache_.clear();
	}

  private:
	HANDLE handle_ = nullptr;
	std::uint32_t pid_ = 0;
	std::string name_;
	std::uintptr_t allocationGranularity_ = 0x10000u;
	std::uintptr_t maxUserAddress_ = std::numeric_limits<std::uintptr_t>::max();
	mutable bool regionCacheInitialized_ = false;
	mutable std::vector<Region> regionCache_;

	void close()
	{
		if (handle_ != nullptr)
		{
			CloseHandle(handle_);
			handle_ = nullptr;
		}
	}

	bool isSafeRange(std::uintptr_t address, std::size_t size) const
	{
		if (size == 0)
			return address <= maxUserAddress_;
		if (!isUserPointer(address))
			return false;
		return size - 1 <= maxUserAddress_ - address;
	}

	static bool contains(const MEMORY_BASIC_INFORMATION &info, std::uintptr_t address, std::size_t size)
	{
		const auto base = reinterpret_cast<std::uintptr_t>(info.BaseAddress);
		const auto regionSize = static_cast<std::size_t>(info.RegionSize);
		if (address < base || address - base > regionSize)
			return false;
		return size <= regionSize - (address - base);
	}

	static bool contains(const Region &region, std::uintptr_t address, std::size_t size)
	{
		if (address < region.base || address - region.base > region.size)
			return false;
		return size <= region.size - (address - region.base);
	}

	const Region *cachedRegion(std::uintptr_t address, std::size_t size) const
	{
		if (!regionCacheInitialized_ || regionCache_.empty())
			return nullptr;
		const auto it = std::upper_bound(
			regionCache_.begin(), regionCache_.end(), address, [](std::uintptr_t value, const Region &region) {
				return value < region.base;
			});
		if (it == regionCache_.begin())
			return nullptr;
		const Region &region = *std::prev(it);
		return contains(region, address, size) ? &region : nullptr;
	}

	bool isPrivateWritableImpl(std::uintptr_t address, std::size_t size, bool live) const
	{
		if (!isSafeRange(address, size))
			return false;
		if (!live && regionCacheInitialized_)
		{
			if (const Region *region = cachedRegion(address, size))
				return region->state == MEM_COMMIT && region->type == MEM_PRIVATE &&
					   isWritableProtection(region->protect);
			return false;
		}

		MEMORY_BASIC_INFORMATION info{};
		if (!query(address, info))
			return false;
		if (info.State != MEM_COMMIT || info.Type != MEM_PRIVATE || !isWritableProtection(info.Protect))
			return false;
		return contains(info, address, size);
	}

	void ensureRegionCache() const
	{
		if (regionCacheInitialized_)
			return;
		regionCacheInitialized_ = true;
		std::uintptr_t address = kMinimumUserAddress;
		while (address <= maxUserAddress_)
		{
			MEMORY_BASIC_INFORMATION info{};
			if (!query(address, info))
			{
				const std::uintptr_t step = std::min<std::uintptr_t>(allocationGranularity_, maxUserAddress_);
				if (address > maxUserAddress_ - step)
					break;
				address += step;
				continue;
			}

			const auto base = reinterpret_cast<std::uintptr_t>(info.BaseAddress);
			const std::size_t size = static_cast<std::size_t>(info.RegionSize);
			if (size == 0 || base > maxUserAddress_)
				break;
			regionCache_.push_back(Region{base, size, info.State, info.Type, info.Protect});

			if (size > maxUserAddress_ - base)
				break;
			const std::uintptr_t next = base + size;
			if (next <= address)
				break;
			address = next;
		}
	}

	static DWORD baseProtection(DWORD protection)
	{
		return protection & 0xFFu;
	}

	static bool isReadableProtection(DWORD protection)
	{
		if ((protection & PAGE_GUARD) != 0 || baseProtection(protection) == PAGE_NOACCESS)
			return false;
		switch (baseProtection(protection))
		{
		case PAGE_READONLY:
		case PAGE_READWRITE:
		case PAGE_WRITECOPY:
		case PAGE_EXECUTE_READ:
		case PAGE_EXECUTE_READWRITE:
		case PAGE_EXECUTE_WRITECOPY:
			return true;
		default:
			return false;
		}
	}

	static bool isWritableProtection(DWORD protection)
	{
		if ((protection & PAGE_GUARD) != 0 || baseProtection(protection) == PAGE_NOACCESS)
			return false;
		switch (baseProtection(protection))
		{
		case PAGE_READWRITE:
		case PAGE_WRITECOPY:
		case PAGE_EXECUTE_READWRITE:
		case PAGE_EXECUTE_WRITECOPY:
			return true;
		default:
			return false;
		}
	}

	bool scanRegion(const Region &region,
					const std::vector<std::uint8_t> &pattern,
					const std::function<bool(std::uintptr_t)> &onMatch,
					std::uint64_t &scanned,
					const std::function<void(std::uint64_t)> &onProgress) const
	{
		if (region.size < pattern.size())
		{
			scanned += region.size;
			if (onProgress)
				onProgress(scanned);
			return true;
		}

		// One buffer for all chunks in this region; the count locator's
		// two-byte pattern gets a memchr fast path instead of std::search.
		std::vector<std::uint8_t> bytes;
		bytes.reserve(kScanChunkBytes + pattern.size() - 1u);
		for (std::size_t offset = 0; offset < region.size;)
		{
			const std::size_t remaining = region.size - offset;
			const std::size_t chunk = std::min(kScanChunkBytes, remaining);
			const bool hasNextChunk = chunk < remaining;
			const std::size_t overlap = hasNextChunk ? std::min(pattern.size() - 1u, remaining - chunk) : 0;
			const std::size_t readLength = chunk + overlap;
			bytes.resize(readLength);

			const std::uintptr_t readAddress = region.base + offset;
			if (readBytes(readAddress, bytes.data(), bytes.size()))
			{
				std::size_t searchOffset = 0;
				const std::size_t searchEnd = hasNextChunk ? chunk + overlap : bytes.size();
				if (pattern.size() == 2u)
				{
					const std::uint8_t first = pattern[0];
					const std::uint8_t second = pattern[1];
					while (searchOffset + 1u < searchEnd)
					{
						const auto *found = static_cast<const std::uint8_t *>(
							std::memchr(bytes.data() + searchOffset, first, searchEnd - searchOffset - 1u));
						if (found == nullptr)
							break;
						const std::size_t matchOffset = static_cast<std::size_t>(found - bytes.data());
						if (bytes[matchOffset + 1u] == second && !onMatch(readAddress + matchOffset))
							return false;
						searchOffset = matchOffset + 1u;
					}
				}
				else
				{
					while (searchOffset + pattern.size() <= searchEnd)
					{
						const auto begin = bytes.begin() + static_cast<std::ptrdiff_t>(searchOffset);
						const auto end = bytes.begin() + static_cast<std::ptrdiff_t>(searchEnd);
						const auto match = std::search(begin, end, pattern.begin(), pattern.end());
						if (match == end)
							break;
						const std::size_t matchOffset = static_cast<std::size_t>(match - bytes.begin());
						if (!onMatch(readAddress + matchOffset))
							return false;
						searchOffset = matchOffset + 1u;
					}
				}
			}

			offset += chunk;
			scanned += chunk;
			if (onProgress)
				onProgress(scanned);
		}
		return true;
	}
};

static_assert(sizeof(std::uintptr_t) == 8, "FH6 memory layout requires a 64-bit build");

} // namespace detail

namespace
{

using detail::RemoteProcess;

// ---- helpers ----

std::unique_ptr<RemoteProcess> openRemoteProcess(const ProcessInfo &info, std::string &error)
{
	// No privilege escalation is requested or needed: the game runs as the same
	// user at the same integrity level. The only real failure mode is the
	// sandboxed Microsoft Store / Game Pass build. SYNCHRONIZE lets alive()
	// detect a closed game via the process handle.
	constexpr DWORD access =
		PROCESS_QUERY_INFORMATION | PROCESS_VM_READ | PROCESS_VM_OPERATION | PROCESS_VM_WRITE | SYNCHRONIZE;
	HANDLE handle = OpenProcess(access, FALSE, static_cast<DWORD>(info.pid));
	if (handle == nullptr)
	{
		error = "OpenProcess(" + std::to_string(info.pid) + ") failed: " + win32Error(GetLastError()) +
				"; Microsoft Store/Game Pass builds may reject external handles";
		return nullptr;
	}
	return std::make_unique<RemoteProcess>(handle, info.pid, info.name);
}

std::vector<std::uint8_t> littleEndianBytes(std::uint64_t value, std::size_t byteCount)
{
	std::vector<std::uint8_t> result(byteCount);
	for (std::size_t i = 0; i < byteCount; ++i)
		result[i] = static_cast<std::uint8_t>((value >> (i * 8u)) & 0xFFu);
	return result;
}

bool addAddress(const RemoteProcess &process, std::uintptr_t base, std::uintptr_t offset, std::uintptr_t &result)
{
	if (base > process.maxUserAddress() || offset > process.maxUserAddress() - base)
		return false;
	result = base + offset;
	return true;
}

bool addProfileAddress(const RemoteProcess &process, std::uintptr_t base, std::uint32_t offset, std::uintptr_t &result)
{
	return addAddress(process, base, static_cast<std::uintptr_t>(offset), result);
}

bool readPointer(const RemoteProcess &process, std::uintptr_t address, std::uintptr_t &value)
{
	std::uint64_t raw = 0;
	if (!process.readValue(address, raw))
		return false;
	value = static_cast<std::uintptr_t>(raw);
	return true;
}

bool readFloatPair(const RemoteProcess &process, std::uintptr_t address, std::array<float, 2> &value)
{
	return process.readBytes(address, value.data(), sizeof(value));
}

// ---- candidate scoring: heap scans produce plenty of false positives, so no
// candidate is trusted until its table both looks like layers and covers the
// expected count (TODO.md's locator quality gates) ----

bool plausibleShapeWord(const RemoteProcess &process, std::uintptr_t layer)
{
	std::uintptr_t address = 0;
	std::uint16_t word = 0;
	return addProfileAddress(process, layer, kFH6Profile.shape_id_offset, address) &&
		   process.readValue(address, word) && word != 0 && word < 0x2000u;
}

bool pairInRange(const std::array<float, 2> &value, float minimum, float maximum)
{
	return isFinite(value[0]) && isFinite(value[1]) && value[0] >= minimum && value[0] <= maximum &&
		   value[1] >= minimum && value[1] <= maximum;
}

bool pairAbsInRange(const std::array<float, 2> &value, float minimum, float maximum)
{
	return isFinite(value[0]) && isFinite(value[1]) && std::abs(value[0]) >= minimum &&
		   std::abs(value[0]) <= maximum && std::abs(value[1]) >= minimum && std::abs(value[1]) <= maximum;
}

// 0..5: pos in range, scale in range, color readable, plausible shape word,
// mask byte 0/1. One point per passing check.
int scoreLayerPointer(const RemoteProcess &process, std::uintptr_t layer)
{
	if (!process.isUserPointer(layer) || !process.isPrivateWritable(layer))
		return 0;

	int score = 0;
	std::uintptr_t address = 0;
	std::array<float, 2> pair{};
	if (addProfileAddress(process, layer, kFH6Profile.pos_offset, address) && readFloatPair(process, address, pair) &&
		pairInRange(pair, -10000.0f, 10000.0f))
		++score;
	if (addProfileAddress(process, layer, kFH6Profile.scale_offset, address) && readFloatPair(process, address, pair) &&
		pairAbsInRange(pair, 0.0f, 10000.0f))
		++score;

	std::array<std::uint8_t, 4> color{};
	if (addProfileAddress(process, layer, kFH6Profile.color_offset, address) &&
		process.readBytes(address, color.data(), color.size()))
		++score;
	if (plausibleShapeWord(process, layer))
		++score;

	std::uint8_t mask = 0;
	if (addProfileAddress(process, layer, kFH6Profile.mask_offset, address) && process.readValue(address, mask) &&
		(mask == 0 || mask == 1))
		++score;
	return score;
}

// All of scoreLayerPointer's checks plus: scale must not be ~0 on both axes
// (cleared layers sit at 0.001 and must not count as strict). Alpha is
// unrestricted: the editor uses it for common semi-transparent layers.
bool strictLayerPointer(const RemoteProcess &process, std::uintptr_t layer)
{
	if (!process.isUserPointer(layer) || !process.isPrivateWritable(layer))
		return false;

	std::uintptr_t address = 0;
	std::array<float, 2> position{};
	std::array<float, 2> scale{};
	if (!addProfileAddress(process, layer, kFH6Profile.pos_offset, address) ||
		!readFloatPair(process, address, position) || !pairInRange(position, -10000.0f, 10000.0f))
		return false;
	if (!addProfileAddress(process, layer, kFH6Profile.scale_offset, address) ||
		!readFloatPair(process, address, scale) || !pairAbsInRange(scale, 0.0f, 10000.0f) ||
		(std::abs(scale[0]) < kStrictScaleEpsilon && std::abs(scale[1]) < kStrictScaleEpsilon))
		return false;

	std::array<std::uint8_t, 4> color{};
	if (!addProfileAddress(process, layer, kFH6Profile.color_offset, address) ||
		!process.readBytes(address, color.data(), color.size()))
		return false;
	if (!plausibleShapeWord(process, layer))
		return false;

	std::uint8_t mask = 0;
	return addProfileAddress(process, layer, kFH6Profile.mask_offset, address) && process.readValue(address, mask) &&
		   (mask == 0 || mask == 1);
}

// std::vector<layer*> invariant at group+0x78: begin/end/capacity with
// end == begin + count*8 and capacity >= end.
bool tableVectorOK(const RemoteProcess &process, std::uintptr_t group, int count, std::uintptr_t &table)
{
	if (count <= 0 || !process.isPrivateWritable(group))
		return false;
	const auto countSize = static_cast<std::uintptr_t>(count) * sizeof(std::uintptr_t);
	if (countSize / sizeof(std::uintptr_t) != static_cast<std::uintptr_t>(count))
		return false;

	std::uintptr_t tableSlot = 0;
	if (!addProfileAddress(process, group, kFH6Profile.layer_table_offset, tableSlot))
		return false;

	std::uintptr_t begin = 0;
	std::uintptr_t end = 0;
	std::uintptr_t capacity = 0;
	if (!readPointer(process, tableSlot, begin) || !readPointer(process, tableSlot + sizeof(std::uintptr_t), end) ||
		!readPointer(process, tableSlot + sizeof(std::uintptr_t) * 2u, capacity))
		return false;
	if (!process.isUserPointer(begin) || !process.isPrivateWritable(begin))
		return false;
	if (begin > process.maxUserAddress() - countSize || end != begin + countSize || capacity < end)
		return false;

	table = begin;
	return true;
}

// Sample up to 64 entries: with >= 16 samples require distinct >= max(8, 75%)
// and layer-like (score >= 3) >= max(8, 50%); otherwise the table scores 0.
int scoreTable(const RemoteProcess &process, std::uintptr_t table, int sampleCount)
{
	if (sampleCount <= 0 || !process.isPrivateWritable(table))
		return 0;
	sampleCount = std::min(sampleCount, 64);

	int total = 0;
	int layerLike = 0;
	std::unordered_set<std::uintptr_t> distinct;
	for (int index = 0; index < sampleCount; ++index)
	{
		std::uintptr_t slot = 0;
		if (!addAddress(process, table, static_cast<std::uintptr_t>(index) * sizeof(std::uintptr_t), slot))
			return 0;
		std::uintptr_t layer = 0;
		if (!readPointer(process, slot, layer) || !process.isPrivateWritable(layer))
			return 0;
		distinct.insert(layer);
		const int score = scoreLayerPointer(process, layer);
		total += score;
		if (score >= 3)
			++layerLike;
	}

	if (sampleCount >= 16)
	{
		const int requiredDistinct = std::max(8, sampleCount * 3 / 4);
		const int requiredLayerLike = std::max(8, sampleCount / 2);
		if (static_cast<int>(distinct.size()) < requiredDistinct || layerLike < requiredLayerLike)
			return 0;
	}
	return total + layerLike;
}

// Scan up to min(count*2, 3000) slots: valid (score >= 3, deduplicated) entries
// must reach min(count, 3000), of which strict entries >= max(32, 25%).
bool validateCoverage(const RemoteProcess &process, std::uintptr_t table, int count)
{
	if (count <= 0 || !process.isPrivateWritable(table))
		return false;

	const int required = std::min(count, 3000);
	const int scanLimit = std::min(required * 2, 3000);
	const int strictRequired = std::min(std::max(32, required / 4), required);

	int valid = 0;
	int strict = 0;
	std::unordered_set<std::uintptr_t> seen;
	for (int index = 0; index < scanLimit; ++index)
	{
		std::uintptr_t slot = 0;
		if (!addAddress(process, table, static_cast<std::uintptr_t>(index) * sizeof(std::uintptr_t), slot))
			break;
		std::uintptr_t layer = 0;
		if (!readPointer(process, slot, layer) || seen.contains(layer))
			continue;
		if (!process.isPrivateWritable(layer) || scoreLayerPointer(process, layer) < 3)
			continue;

		seen.insert(layer);
		++valid;
		if (strictLayerPointer(process, layer))
			++strict;
		if (valid >= required)
			return strict >= strictRequired;
	}
	return false;
}

// The one gate every locator path funnels through: vtable (expected or read
// from the group), exact layer count, vector invariant, table score, coverage.
// No candidate from any path is trusted without all of them.
bool validateGroup(const RemoteProcess &process,
				   std::uintptr_t group,
				   std::uintptr_t expectedVtable,
				   int count,
				   std::uintptr_t &table,
				   std::uintptr_t &vtable)
{
	if (!process.isUserPointer(group) || !process.isPrivateWritable(group))
		return false;
	if (!readPointer(process, group, vtable) || !process.isUserPointer(vtable))
		return false;
	if (expectedVtable != 0 && vtable != expectedVtable)
		return false;

	std::uintptr_t address = 0;
	std::uint16_t layerCount = 0;
	if (!addProfileAddress(process, group, kFH6Profile.livery_count_offset, address) ||
		!process.readValue(address, layerCount) || static_cast<int>(layerCount) != count)
		return false;
	if (!tableVectorOK(process, group, count, table))
		return false;
	return scoreTable(process, table, std::min(count, 64)) > 0 && validateCoverage(process, table, count);
}

// ---- writes: the 7 importer fields; the per-layer geometry resource at +0xA8
// is NEVER written (aliasing it across layers corrupts ownership and crashes
// the game on free) ----

bool writeLayerFields(const RemoteProcess &process, std::uintptr_t layer, const LayerWrite &value, std::string &error)
{
	if (!process.isPrivateWritableLive(layer) || !isFiniteLayer(value))
	{
		error = "layer pointer or converted values are invalid";
		return false;
	}

	std::uintptr_t address = 0;
	const std::array<float, 2> position{value.x, value.y};
	const std::array<float, 2> scale{value.sx, value.sy};
	const std::uint8_t mask = 0; // generated shapes never set the mask flag (LayerWrite.md)

	if (!addProfileAddress(process, layer, kFH6Profile.pos_offset, address) ||
		!process.writeBytes(address, position.data(), sizeof(position)))
	{
		error = "position";
		return false;
	}
	if (!addProfileAddress(process, layer, kFH6Profile.scale_offset, address) ||
		!process.writeBytes(address, scale.data(), sizeof(scale)))
	{
		error = "scale";
		return false;
	}
	if (!addProfileAddress(process, layer, kFH6Profile.rotation_offset, address) ||
		!process.writeValue(address, value.rotation))
	{
		error = "rotation";
		return false;
	}
	if (!addProfileAddress(process, layer, kFH6Profile.skew_offset, address) || !process.writeValue(address, value.skew))
	{
		error = "skew";
		return false;
	}
	if (!addProfileAddress(process, layer, kFH6Profile.color_offset, address) ||
		!process.writeBytes(address, value.color.data(), value.color.size()))
	{
		error = "color";
		return false;
	}
	if (!addProfileAddress(process, layer, kFH6Profile.mask_offset, address) || !process.writeValue(address, mask))
	{
		error = "mask";
		return false;
	}
	if (!addProfileAddress(process, layer, kFH6Profile.shape_id_offset, address) ||
		!process.writeValue(address, value.word))
	{
		error = "shape word";
		return false;
	}
	return true;
}

// Mesh-path hot refresh: rewriting the std::string at +0x80 makes the game
// rebuild the mesh on the next redraw. Strictly within capacity; the resource
// pointer at +0xA8 is never touched. Best-effort and quiet by design.
std::optional<std::string> meshPathForWord(std::uint16_t word)
{
	// The app currently emits only the circle word; a generic word->mesh table
	// needs the generated mapping from the reference project (ShapeToWord.md).
	if (word == kWordCircle)
		return std::string{"GAME:\\Media\\Livery\\Vinyls\\A_02.modelbin"};
	return std::nullopt;
}

bool writeLayerPath(const RemoteProcess &process, std::uintptr_t layer, std::uint16_t word)
{
	const auto path = meshPathForWord(word);
	if (!path)
		return false;

	std::uintptr_t address = 0;
	std::uintptr_t data = 0;
	std::uint64_t capacity = 0;
	if (!addProfileAddress(process, layer, kFH6Profile.mesh_data_offset, address) ||
		!readPointer(process, address, data) || !process.isUserPointer(data) ||
		!process.isPrivateWritable(data, path->size() + 1u) ||
		!addProfileAddress(process, layer, kFH6Profile.mesh_capacity_offset, address) ||
		!process.readValue(address, capacity) || path->size() + 1u > capacity)
		return false;

	std::vector<std::uint8_t> bytes(path->begin(), path->end());
	bytes.push_back(0);
	if (!process.writeBytes(data, bytes.data(), bytes.size()))
		return false;

	const std::uint64_t size = static_cast<std::uint64_t>(path->size());
	return addProfileAddress(process, layer, kFH6Profile.mesh_size_offset, address) && process.writeValue(address, size);
}

// ClearWrites.md: blank an unused template slot — zero position, 0.001 scale,
// zero rotation/color/mask. Skew and the shape word are deliberately NOT
// touched: a cleared layer keeps its mesh but becomes visually negligible.
bool writeClearedLayer(const RemoteProcess &process, std::uintptr_t layer, std::string &error)
{
	if (!process.isPrivateWritableLive(layer))
	{
		error = "layer pointer is not private writable";
		return false;
	}

	std::uintptr_t address = 0;
	const std::array<float, 2> position{0.0f, 0.0f};
	const std::array<float, 2> scale{0.001f, 0.001f};
	const std::array<std::uint8_t, 4> color{0, 0, 0, 0};
	const float rotation = 0.0f;
	const std::uint8_t mask = 0;

	if (!addProfileAddress(process, layer, kFH6Profile.pos_offset, address) ||
		!process.writeBytes(address, position.data(), sizeof(position)))
	{
		error = "position";
		return false;
	}
	if (!addProfileAddress(process, layer, kFH6Profile.scale_offset, address) ||
		!process.writeBytes(address, scale.data(), sizeof(scale)))
	{
		error = "scale";
		return false;
	}
	if (!addProfileAddress(process, layer, kFH6Profile.rotation_offset, address) ||
		!process.writeValue(address, rotation))
	{
		error = "rotation";
		return false;
	}
	if (!addProfileAddress(process, layer, kFH6Profile.color_offset, address) ||
		!process.writeBytes(address, color.data(), color.size()))
	{
		error = "color";
		return false;
	}
	if (!addProfileAddress(process, layer, kFH6Profile.mask_offset, address) || !process.writeValue(address, mask))
	{
		error = "mask";
		return false;
	}
	return true;
}

} // namespace

#endif // _WIN32

// ---- Injector: platform-independent shell ----

Injector::Injector() = default;

Injector::Injector(Options options) : options_(std::move(options))
{
}

Injector::~Injector() = default;

Injector::Location Injector::location() const
{
	std::lock_guard<std::mutex> lock(mutex_);
	return location_;
}

void Injector::invalidateCache()
{
	std::lock_guard<std::mutex> lock(mutex_);
	dropCache();
}

void Injector::dropCache()
{
	cached_pid_ = 0;
	cached_group_ = 0;
	cached_vtable_ = 0;
	cached_vtable_rva_ = 0;
	cached_count_ = 0;
}

#ifndef _WIN32

// Windows-only feature: clean failures everywhere else.
bool Injector::Locate(std::uint32_t)
{
	return false;
}

bool Injector::ensureValid(std::uint32_t)
{
	return false;
}

Injector::WriteResult Injector::writeEllipses(std::span<const struct ::Ellipse>, std::uint32_t, std::uint32_t,
											   std::uint32_t)
{
	WriteResult result;
	result.error = "FH6 external-memory injection is only supported on Windows";
	return result;
}

#else

bool Injector::Locate(std::uint32_t template_layer_count)
{
	std::lock_guard<std::mutex> lock(mutex_);
	detail::Log log(options_, nullptr);
	std::string error;
	if (!attachProcess(error, log))
		return false;
	dropCache(); // force a full relocation from scratch
	return locateLocked(template_layer_count, error, log);
}

bool Injector::ensureValid(std::uint32_t template_layer_count)
{
	std::lock_guard<std::mutex> lock(mutex_);
	detail::Log log(options_, nullptr);
	std::string error;
	if (!attachProcess(error, log))
		return false;
	return locateLocked(template_layer_count, error, log);
}

Injector::WriteResult Injector::writeEllipses(std::span<const struct ::Ellipse> ellipses,
											  std::uint32_t canvas_width,
											  std::uint32_t canvas_height,
											  std::uint32_t template_layer_count)
{
	WriteResult result;
	result.requested_shapes = ellipses.size();
	detail::Log log(options_, &result.diagnostics);

	const auto fail = [&](std::string message) {
		result.error = std::move(message);
		log.error("request failed: " + result.error);
		return result;
	};

	if (template_layer_count == 0 || template_layer_count > 0xFFFFu)
		return fail("invalid template layer count; expected the exact FH6 count in the range 1..65535");
	if (canvas_width == 0 || canvas_height == 0)
		return fail("canvas dimensions must be positive");
	log(joinDiagnostic("request started: shapes=", result.requested_shapes, " canvas=", canvas_width, "x", canvas_height,
					   " template_layer_count=", template_layer_count));

	// Conversion is pure computation; finish it before touching the game.
	const CanvasMap canvas = makeCanvasMap(canvas_width, canvas_height);
	if (!isFinite(canvas.k) || canvas.k <= 0.0f)
		return fail("canvas dimensions produce an invalid FH6 coordinate map");

	std::vector<LayerWrite> layers;
	layers.reserve(ellipses.size());
	for (std::size_t index = 0; index < ellipses.size(); ++index)
	{
		LayerWrite layer;
		std::string reason;
		if (!ellipseToLayer(ellipses[index], canvas, layer, reason))
		{
			++result.skipped_shapes;
			log(joinDiagnostic("skipping ellipse ", index + 1, ": ", reason));
			continue;
		}
		layers.push_back(layer);
	}
	result.converted_shapes = layers.size();

	std::lock_guard<std::mutex> lock(mutex_);
	std::string error;
	if (!attachProcess(error, log))
		return fail(error);
	result.pid = process_->pid();
	result.process_name = process_->name();

	// Offsets are calibrated for FH6 only; never write into another binary even
	// if process discovery is broadened later.
	if (!equalsIgnoreCaseAscii(result.process_name, "ForzaHorizon6.exe"))
		return fail("refusing to write to an unexpected process name: " + result.process_name);

	if (!locateLocked(template_layer_count, error, log))
		return fail(error);
	result.locator = location_.locator;
	result.used_cached_group = location_.cached_group;
	result.used_cached_vtable = location_.cached_vtable;

	const std::size_t planned = std::min(layers.size(), static_cast<std::size_t>(template_layer_count));
	if (layers.size() > planned)
		log.warning(joinDiagnostic("template capacity reached; ", layers.size() - planned,
								   " converted shape(s) will not be written"));

	// Preflight every intended slot, then re-check each pointer immediately
	// before its write: the editor heap can change between validation and the
	// loop (group edits, undo, editor teardown). A stale slot aborts the whole
	// write rather than leaving a half-written group.
	std::vector<std::uintptr_t> preflight(planned);
	for (std::size_t index = 0; index < planned; ++index)
	{
		std::uintptr_t slot = 0;
		if (!addAddress(*process_, location_.table, index * sizeof(std::uint64_t), slot) ||
			!readPointer(*process_, slot, preflight[index]) || !process_->isPrivateWritableLive(preflight[index]))
			return fail("layer slot " + std::to_string(index + 1) +
						" is not a valid private writable pointer; aborting before writes");
	}

	for (std::size_t index = 0; index < planned; ++index)
	{
		std::uintptr_t slot = 0;
		std::uintptr_t layerPointer = 0;
		if (!addAddress(*process_, location_.table, index * sizeof(std::uint64_t), slot) ||
			!readPointer(*process_, slot, layerPointer) || layerPointer != preflight[index] ||
			!process_->isPrivateWritable(layerPointer))
			return fail("layer slot " + std::to_string(index + 1) +
						" changed before its write; stopped to avoid a stale-table write");

		std::string writeError;
		if (!writeLayerFields(*process_, layerPointer, layers[index], writeError))
			return fail("write of layer " + std::to_string(index + 1) + " failed: " + writeError +
						"; layers written before failure: " + std::to_string(result.written_layers));
		++result.written_layers;

		if (options_.update_mesh_path && writeLayerPath(*process_, layerPointer, layers[index].word))
			++result.mesh_paths_updated;

		if (result.written_layers == 1 || result.written_layers % 100 == 0)
			log(joinDiagnostic("wrote layer ", result.written_layers, "/", planned));
	}
	if (options_.update_mesh_path && result.written_layers > 0 && result.mesh_paths_updated == 0)
		log.warning("mesh path refresh failed for every written layer (string capacity too small?)");

	if (options_.clear_unused)
	{
		for (std::size_t index = planned; index < template_layer_count; ++index)
		{
			std::uintptr_t slot = 0;
			std::uintptr_t layerPointer = 0;
			if (!addAddress(*process_, location_.table, index * sizeof(std::uint64_t), slot) ||
				!readPointer(*process_, slot, layerPointer) || !process_->isPrivateWritableLive(layerPointer))
				continue; // best-effort: leave leftover layers alone
			std::string clearError;
			if (writeClearedLayer(*process_, layerPointer, clearError))
				++result.cleared_layers;
		}
		log(joinDiagnostic("clear: cleared=", result.cleared_layers, " requested=", template_layer_count - planned));
	}

	result.success = true;
	log(joinDiagnostic("request completed: written=", result.written_layers, " cleared=", result.cleared_layers,
					   " mesh_paths=", result.mesh_paths_updated, " skipped=", result.skipped_shapes));
	return result;
}

// ---- private: process + locator cascade (mutex held) ----

bool Injector::attachProcess(std::string &error, const detail::Log &log)
{
	ProcessInfo info;
	if (!findFH6Process(info, error, log))
	{
		process_.reset();
		return false;
	}
	if (process_ && process_->alive() && process_->pid() == info.pid)
		return true; // same game session: keep the handle

	process_ = openRemoteProcess(info, error);
	if (!process_)
		return false;
	log(joinDiagnostic("process handle opened: pid=", info.pid));

	if (cached_pid_ != 0 && cached_pid_ != info.pid)
	{
		log("game process restarted; dropping the cached location");
		dropCache();
	}
	return true;
}

// The locator cascade shared by Locate/ensureValid/writeEllipses, cheapest
// path first. Every path funnels candidates through validateGroup; the winner
// is cached by cacheLocation on the way out.
bool Injector::locateLocked(std::uint32_t template_layer_count, std::string &error, const detail::Log &log)
{
	if (template_layer_count == 0 || template_layer_count > 0xFFFFu)
	{
		error = "invalid template layer count; expected 1..65535";
		return false;
	}
	const int count = static_cast<int>(template_layer_count);

	location_ = Location{}; // drop any stale location; module_base is refreshed below
	location_.module_base = process_->moduleBase();
	if (location_.module_base == 0)
	{
		error = "could not determine the ForzaHorizon6.exe module base";
		log.error(error);
		return false;
	}
	process_->refreshRegions(); // scans and scoring share one fresh heap snapshot

	if (tryCachedGroup(count, log) || tryCachedVtable(count, log) || tryRtti(count, log) || tryCountScan(count, log))
	{
		cacheLocation();
		return true;
	}

	error = "no valid FH6 layer table found for " + std::to_string(template_layer_count) +
			" layers; confirm FH6 is in the Vinyl Group Editor and the template count is exact";
	log.error(error);
	return false;
}

// Level 1 — cached group: the group address is still valid if its first qword
// still equals the cached vtable, the layer count still matches, and the table
// still passes scoring. A handful of reads; ideal for repeat injections into
// the same open editor group.
bool Injector::tryCachedGroup(int count, const detail::Log &log)
{
	if (cached_group_ == 0 || cached_vtable_ == 0)
		return false;

	std::uintptr_t table = 0;
	std::uintptr_t vtable = 0;
	if (!validateGroup(*process_, cached_group_, cached_vtable_, count, table, vtable))
	{
		log("cached group no longer validates; trying the cached vtable");
		return false;
	}

	location_.group = cached_group_;
	location_.table = table;
	location_.vtable = vtable;
	location_.layer_count = static_cast<std::uint32_t>(count);
	location_.locator = "cached group";
	location_.cached_group = true;
	location_.cached_vtable = false;
	log(joinDiagnostic("locator: cached group group=", hexAddress(location_.group), " table=", hexAddress(table)));
	return true;
}

// Level 2 — cached vtable: the group object moved (editor reopened) but the
// class vtable is stable within the process, so scan the private writable heap
// for the cached vtable value and validate each hit as a group candidate.
bool Injector::tryCachedVtable(int count, const detail::Log &log)
{
	if (cached_vtable_rva_ == 0 || location_.module_base == 0 ||
		cached_vtable_rva_ > process_->maxUserAddress() - location_.module_base)
		return false;
	const std::uintptr_t vtable = location_.module_base + cached_vtable_rva_;
	if (!process_->isUserPointer(vtable))
		return false;

	log(joinDiagnostic("cached vtable scan: vtable=", hexAddress(vtable)));
	const auto pattern = littleEndianBytes(vtable, sizeof(std::uint64_t));
	bool found = false;
	process_->scanPattern(
		MEM_PRIVATE,
		true,
		pattern,
		[&](std::uintptr_t candidate) {
			std::uintptr_t table = 0;
			std::uintptr_t candidateVtable = 0;
			if (!validateGroup(*process_, candidate, vtable, count, table, candidateVtable))
				return true;
			location_.group = candidate;
			location_.table = table;
			location_.vtable = candidateVtable;
			location_.layer_count = static_cast<std::uint32_t>(count);
			location_.locator = "cached vtable";
			location_.cached_group = false;
			location_.cached_vtable = true;
			found = true;
			return false;
		},
		{},
		kMaxRegionRead); // directed scan: skip huge regions

	if (found)
		log(joinDiagnostic("locator: cached vtable group=", hexAddress(location_.group),
						   " table=", hexAddress(location_.table)));
	else
		log("cached vtable found no valid group; falling back to the count scan");
	return found;
}

// Level 3 — RTTI: the MSVC chain (".?AVCLiveryGroup@@" -> TypeDescriptor ->
// CompleteObjectLocator -> vtable) needs the mangled type name in the image,
// and the shipped FH6 build strips it. Stubbed; restore the RTTI scan from git
// history if a future build keeps its type names.
bool Injector::tryRtti(int /*count*/, const detail::Log &log)
{
	log("RTTI locator skipped: type names are stripped in this build");
	return false;
}

// Level 4 — count scan: always-available fallback. Scan the whole private
// writable heap for a uint16 equal to the template layer count; each hit minus
// 0x5A is a candidate group, and validateGroup filters the false positives.
bool Injector::tryCountScan(int count, const detail::Log &log)
{
	if (count < kMinCountScanCount)
	{
		log.error(joinDiagnostic("count scan needs at least ", kMinCountScanCount,
								 " template layers to be reliable; got ", count));
		return false;
	}

	log(joinDiagnostic("count scan: searching the private writable heap for layer count ", count));
	const auto pattern = littleEndianBytes(static_cast<std::uint16_t>(count), sizeof(std::uint16_t));
	const auto countOffset = static_cast<std::uintptr_t>(kFH6Profile.livery_count_offset);
	std::uint64_t nextReport = 1ull << 30;
	bool found = false;
	process_->scanPattern(
		MEM_PRIVATE,
		true,
		pattern,
		[&](std::uintptr_t countAddress) {
			if (countAddress < countOffset)
				return true;
			const std::uintptr_t candidate = countAddress - countOffset;
			std::uintptr_t table = 0;
			std::uintptr_t vtable = 0;
			if (!validateGroup(*process_, candidate, 0, count, table, vtable))
				return true;
			location_.group = candidate;
			location_.table = table;
			location_.vtable = vtable;
			location_.layer_count = static_cast<std::uint32_t>(count);
			location_.locator = "count scan";
			location_.cached_group = false;
			location_.cached_vtable = false;
			found = true;
			return false;
		},
		[&](std::uint64_t scanned) {
			if (scanned >= nextReport)
			{
				log(joinDiagnostic("count scan: ", scanned >> 20, " MB searched"));
				nextReport += 1ull << 30;
			}
		});

	if (found)
		log(joinDiagnostic("locator: count scan group=", hexAddress(location_.group),
						   " table=", hexAddress(location_.table), " vtable=", hexAddress(location_.vtable)));
	else
		log("count scan found no valid layer table");
	return found;
}

// Remember the winning location for the next ensureValid(). The RVA (not the
// absolute address) is what stays meaningful across ASLR restarts.
void Injector::cacheLocation()
{
	cached_pid_ = process_ ? process_->pid() : 0;
	cached_group_ = location_.group;
	cached_vtable_ = location_.vtable;
	cached_vtable_rva_ = location_.vtable > location_.module_base ? location_.vtable - location_.module_base : 0;
	cached_count_ = location_.layer_count;
}

#endif // _WIN32

} // namespace fh6::injector
