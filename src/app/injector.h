#pragma once

#include "structures.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>

namespace fh6::injector
{

namespace detail
{
class Log;           // diagnostic sink; defined in injector.cpp
class RemoteProcess; // Win32 process-handle wrapper; defined in injector.cpp
}

// External-memory FH6 layer injector.
//
// It never injects a DLL, creates a remote thread, patches code, or writes the
// game's per-layer resource pointer (layer + 0xA8). It reads the game's heap to
// find the CLiveryGroup layer table, funnels every candidate through the same
// scoring/coverage validation, and only then writes the documented layer fields.
//
// Locate() forces a full relocation from scratch. ensureValid() tries the
// multi-level cache first and is near-free while the same editor group is open.
// Both refresh the in-process cache on success; nothing is persisted to disk.
class Injector
{
  public:
    struct Options
    {
        // Rewrite the std::string mesh path at layer + 0x80 when its capacity
        // allows. Best-effort: failure never invalidates the geometry write.
        bool update_mesh_path = true;

        // Blank unused template slots after the written shapes. Off by default:
        // the caller must opt into touching unused layers.
        bool clear_unused = false;

        // Diagnostic sink, called synchronously on the injecting thread. The
        // view is only valid during the call; when empty, messages go to spdlog.
        std::function<void(std::string_view)> log;
    };

    // A validated CLiveryGroup location. Absolute addresses are only valid
    // within the attached process; vtable_rva is relative to the main module
    // base so it survives ASLR across game restarts.
    struct Location
    {
        std::uintptr_t group = 0;       // CLiveryGroup instance
        std::uintptr_t table = 0;       // vector<layer*> begin
        std::uintptr_t vtable = 0;      // group's vtable (first qword)
        std::uintptr_t module_base = 0; // ForzaHorizon6.exe base
        std::uintptr_t vtable_rva = 0;  // vtable - module_base
        std::uint32_t layer_count = 0;
        const char *locator = "none"; // which path produced this location
        bool cached_group = false;
        bool cached_vtable = false;
    };

    struct WriteResult
    {
        bool success = false;
        std::uint32_t pid = 0;
        std::string process_name;
        std::string locator;
        std::string error;

        std::size_t requested_shapes = 0;
        std::size_t converted_shapes = 0;
        std::size_t skipped_shapes = 0;
        std::size_t written_layers = 0;
        std::size_t cleared_layers = 0;
        std::size_t mesh_paths_updated = 0;

        bool used_cached_group = false;
        bool used_cached_vtable = false;

        // Copy of the diagnostic stream, useful when no log callback is set.
        std::string diagnostics;
    };

    // No default argument on the Options constructor: it would need Options'
    // NSDMIs inside the class definition, which is ill-formed (CWG 2335). The
    // default constructor is likewise only declared here and defined in the
    // .cpp, where RemoteProcess is a complete type.
    Injector(); // default options
    explicit Injector(Options options);
    ~Injector();

    Injector(const Injector &) = delete;
    Injector &operator=(const Injector &) = delete;

    // Force a complete relocation: drop the cache and locate from scratch
    // (count scan). Returns true and refreshes location() on success.
    bool Locate(std::uint32_t template_layer_count);

    // Make sure location() is valid for this template layer count, running the
    // locator cascade from cheapest to most expensive:
    //   1. cached group address — a handful of reads while the group is alive
    //   2. cached vtable        — directed heap scan after the group moved
    //   3. RTTI                 — stubbed: type names are stripped in this build
    //   4. count scan           — always-available full-heap fallback
    bool ensureValid(std::uint32_t template_layer_count);

    // Snapshot of the current location (thread-safe).
    Location location() const;

    // Drop the cached group/vtable so the next ensureValid() relocates from
    // scratch.
    void invalidateCache();

    // ensureValid() + write the pixel-space ellipses into the template's layer
    // table. template_layer_count must be the exact layer count of the open,
    // ungrouped FH6 template; it is the count-scan key and is intentionally
    // independent of ellipses.size().
    WriteResult writeEllipses(std::span<const struct ::Ellipse> ellipses,
                              std::uint32_t canvas_width,
                              std::uint32_t canvas_height,
                              std::uint32_t template_layer_count);

  private:
    // (Re)attach to ForzaHorizon6.exe; drops the cache when the PID changed.
    // Mutex must be held.
    bool attachProcess(std::string &error, const detail::Log &log);
    // The locator cascade shared by Locate/ensureValid/writeEllipses.
    // Mutex must be held.
    bool locateLocked(std::uint32_t template_layer_count, std::string &error, const detail::Log &log);
    bool tryCachedGroup(int count, const detail::Log &log);
    bool tryCachedVtable(int count, const detail::Log &log);
    bool tryRtti(int count, const detail::Log &log);
    bool tryCountScan(int count, const detail::Log &log);
    // Remember location_'s group/vtable/RVA for the next ensureValid().
    void cacheLocation();
    void dropCache();

    Options options_;
    Location location_;

    // In-process cache. No disk persistence: a new PID simply re-runs the count
    // scan; the vtable RVA stays meaningful because the binary does not change.
    std::uint32_t cached_pid_ = 0;
    std::uintptr_t cached_group_ = 0;
    std::uintptr_t cached_vtable_ = 0;
    std::uintptr_t cached_vtable_rva_ = 0;
    std::uint32_t cached_count_ = 0;

    std::unique_ptr<detail::RemoteProcess> process_;
    mutable std::mutex mutex_; // one locate/injection at a time
};

} // namespace fh6::injector
