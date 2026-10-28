/**
 * @file TestCudaDeviceBuffer.cu
 * @brief Owning CUDA buffers (CudaDeviceBuffer / CudaPinnedBuffer) and the objects
 *        built on them.
 *
 * 1. The buffer class: allocation, lending, moves (also over a buffer that holds
 *    memory), release, containers, the destructor, pinned memory, a failed
 *    allocation (its message names the caller and the size).
 * 2. Every kind of CUDA object (propagator computations of all solvers, the CrysFFT
 *    engines, non-periodic transforms, complex fields, the Anderson mixers), built
 *    with 4 streams (reduce-memory computations use one), returns everything when
 *    destroyed. Checked exactly with the library's live counters: CudaBuffer and
 *    the backend's propagator and checkpoint arrays (count and bytes), cuFFT
 *    plans, streams and events (other processes and cuFFT's own retained memory
 *    do not affect them); a live computation is checked to hold its streams,
 *    events and plans first. No release may fail (cuda_release_failures(); a
 *    double free would count).
 * 3. Failure-injection sweep with 2 streams: the test hook
 *    cuda_buffer_fail_countdown() makes the N-th counted allocation throw; N runs
 *    over every allocation of each construction (and its first run), and after
 *    each failure the counters must be back where they started; the run in which
 *    no allocation fails must succeed. Failures of cuFFT plan, stream and event
 *    creation are not injected; section 4 may reach them for real.
 * 4. A small construction that really runs out of device memory releases what it
 *    took (the counters, and device memory end to end: this process's usage from
 *    NVML, or the device's free memory where NVML cannot tell): a ballast holds
 *    the rest of the free memory for a moment, with several leftovers so the
 *    failure falls at different stages. The 64^3 computation used to size them
 *    must release everything too, and may fail only for lack of memory. When the
 *    device memory cannot be held, or the driver backs device allocations with
 *    system memory once it is full (the default sysmem fallback of WDDM drivers on
 *    Windows; "Prefer No Sysmem Fallback" in the NVIDIA control panel turns it off),
 *    it exits with code 77 (a ctest skip). Section 4 runs only with --out-of-memory,
 *    as the separate, serial ctest entry TestCudaDeviceBufferOutOfMemory, so its skip
 *    does not hide sections 1-3 (tests/CMakeLists.txt).
 */
#include <algorithm>
#include <complex>
#include <concepts>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <cuda_runtime.h>
#ifdef POLYMERFTS_TEST_HAS_NVML
#include <nvml.h>
#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif
#endif
#include "CudaCommon.h"
#include "AbstractFactory.h"
#include "AndersonMixing.h"
#include "Molecules.h"
#include "PlatformSelector.h"
#include "Polymer.h"
#include "PropagatorComputationOptimizer.h"
#include "SpaceGroup.h"

static long long live_device() { return CudaDeviceMemory::counters().live_count.load(); }
static long long live_pinned() { return CudaPinnedMemory::counters().live_count.load(); }
static long long live_device_bytes() { return CudaDeviceMemory::counters().live_bytes.load(); }

/// Everything the library counts: live device and pinned allocations, cuFFT
/// plans, streams and events.
struct Live
{
    long long device = 0, pinned = 0, device_bytes = 0, pinned_bytes = 0, plans = 0, streams = 0, events = 0;
    static Live now()
    {
        const CudaHandleCounters& h = cuda_handle_counters();
        return {live_device(), live_pinned(), CudaDeviceMemory::counters().live_bytes.load(),
                CudaPinnedMemory::counters().live_bytes.load(), h.cufft_plans.load(), h.streams.load(),
                h.events.load()};
    }
    bool operator==(const Live&) const = default;
    std::string minus(const Live& before) const
    {
        return std::to_string(device - before.device) + " device and " + std::to_string(pinned - before.pinned)
            + " pinned allocations (" + std::to_string(device_bytes - before.device_bytes) + " and "
            + std::to_string(pinned_bytes - before.pinned_bytes) + " bytes), " + std::to_string(plans - before.plans)
            + " cuFFT plans, " + std::to_string(streams - before.streams) + " streams, "
            + std::to_string(events - before.events) + " events";
    }
};

/// Device memory used by this process according to NVML, which other processes
/// on the GPU do not change; -1 when NVML cannot tell (e.g. no per-process
/// accounting in a container).
static long long process_device_bytes()
{
#ifndef POLYMERFTS_TEST_HAS_NVML
    return -1;
#else
    static const bool nvml_ready = (nvmlInit_v2() == NVML_SUCCESS);
    if (!nvml_ready)
        return -1;
    int device = 0;
    char bus_id[32];
    nvmlDevice_t handle;
    if (cudaGetDevice(&device) != cudaSuccess || cudaDeviceGetPCIBusId(bus_id, sizeof(bus_id), device) != cudaSuccess
        || nvmlDeviceGetHandleByPciBusId_v2(bus_id, &handle) != NVML_SUCCESS)
    {
        (void)cudaGetLastError();
        return -1;
    }
    unsigned int count = 64;
    std::vector<nvmlProcessInfo_t> processes(count);
    if (nvmlDeviceGetComputeRunningProcesses(handle, &count, processes.data()) != NVML_SUCCESS)
        return -1;
    for (unsigned int i = 0; i < count; ++i)
        if (processes[i].pid == static_cast<unsigned int>(getpid()))
            return processes[i].usedGpuMemory == NVML_VALUE_NOT_AVAILABLE
                ? -1 : static_cast<long long>(processes[i].usedGpuMemory);
    return -1;
#endif
}

/// The message of an exception on one line (cuFFT and nested messages span several)
static std::string one_line(std::string text)
{
    std::replace(text.begin(), text.end(), '\n', ' ');
    return text;
}

#ifdef _WIN32
// The CRT environment that getenv reads; an empty value removes the variable
static void set_env(const char* name, const char* value) { _putenv_s(name, value); }
static void unset_env(const char* name) { _putenv_s(name, ""); }
#else
static void set_env(const char* name, const char* value) { setenv(name, value, 1); }
static void unset_env(const char* name) { unsetenv(name); }
#endif

/// Sets OMP_NUM_THREADS (the backends create one stream per thread, at most 4)
/// for a scope and restores the caller's value afterwards.
struct ScopedStreams
{
    explicit ScopedStreams(const char* n)
    {
        const char* value = std::getenv("OMP_NUM_THREADS");
        had = (value != nullptr);
        if (had)
            saved = value;
        set_env("OMP_NUM_THREADS", n);
    }
    ~ScopedStreams()
    {
        if (had)
            set_env("OMP_NUM_THREADS", saved.c_str());
        else
            unset_env("OMP_NUM_THREADS");
    }
    bool had = false;
    std::string saved;
};

static size_t free_bytes()
{
    size_t f = 0, t = 0;
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemGetInfo(&f, &t));
    return f;
}

/// Hold up to `bytes` of device memory, in chunks: one allocation of all the
/// free memory usually fails, so the chunk size halves down to 1 MiB on failure.
static std::vector<CudaDeviceBuffer<char>> hold_device_memory(size_t bytes)
{
    std::vector<CudaDeviceBuffer<char>> chunks;
    size_t chunk = bytes, held = 0;
    while (held < bytes && chunk >= (size_t(1) << 20))
    {
        const size_t want = std::min(chunk, bytes - held);
        try
        {
            chunks.emplace_back(want);
            held += want;
        }
        catch (const std::exception&)
        {
            chunk /= 2;
        }
    }
    return chunks;
}

static bool check(bool cond, const std::string& what)
{
    printf("  [%s] %s\n", cond ? "PASS" : "FAIL", what.c_str());
    return cond;
}

/// No release call has failed (ignore_release_error() counts a failed cudaFree,
/// e.g. a double free, instead of leaving it in the error slot) and no error is
/// pending in the slot. Reading the slot clears it, so call this before
/// combining its result with other conditions.
static bool no_cuda_error(const std::string& where)
{
    const long long failed_releases = cuda_release_failures().load();
    const cudaError_t e = cudaGetLastError();
    if (failed_releases != 0)
        printf("  %lld failed release call(s) by %s\n", failed_releases, where.c_str());
    if (e != cudaSuccess)
        printf("  CUDA error pending after %s: %s\n", where.c_str(), cudaGetErrorString(e));
    return failed_releases == 0 && e == cudaSuccess;
}

/// The buffer lends its pointer implicitly, also to void* parameters (cudaMemcpy).
/// (GCC and Clang reject `delete buf`: the two conversions make it ambiguous. A
/// requires-expression cannot test that: GCC reports the ambiguity as a hard
/// error, and nvcc's front end treats the expression as valid.)
static_assert(std::convertible_to<CudaDeviceBuffer<double>, double*>);
static_assert(std::convertible_to<CudaDeviceBuffer<double>, const double*>);
static_assert(std::convertible_to<CudaDeviceBuffer<double>, void*>);
static_assert(std::convertible_to<CudaDeviceBuffer<double>, const void*>);

//------------------------------------------------------------------------------
// One construction and a short run of each kind of object
//------------------------------------------------------------------------------
template <typename T>
static std::unique_ptr<AbstractFactory<T>> make_factory(bool reduce_memory)
{
    if constexpr (std::is_same_v<T, double>)
        return PlatformSelector::create_factory_real("cuda", reduce_memory);
    else
        return PlatformSelector::create_factory_complex("cuda", reduce_memory);
}

struct ComputationSpec
{
    std::string chain_model = "continuous";
    std::string method = "rqm4";
    bool reduce_memory = false;
    std::vector<int> nx = {16, 16, 16};
    std::vector<double> lx = {1.9, 1.9, 1.9};
    std::vector<std::string> bc;           ///< empty: periodic
    std::vector<double> angles;            ///< empty: orthogonal
    std::string sg_symbol;                 ///< empty: no space group
    int sg_hall = -1;
    std::string basis;                     ///< "", "m3", "pmmm" or "z-mirror"
    bool stress = true;
};

template <typename T>
static void run_computation(const ComputationSpec& c)
{
    std::unique_ptr<AbstractFactory<T>> factory(make_factory<T>(c.reduce_memory));
    std::unique_ptr<SpaceGroup> sg;
    if (!c.sg_symbol.empty())
    {
        sg = std::make_unique<SpaceGroup>(c.nx, c.sg_symbol, c.sg_hall);
        if (c.basis == "m3")       sg->enable_m3_physical_basis();
        if (c.basis == "pmmm")     sg->enable_pmmm_physical_basis();
        if (c.basis == "z-mirror") sg->enable_z_mirror_physical_basis();
    }
    std::unique_ptr<ComputationBox<T>> cb(c.angles.empty()
        ? factory->create_computation_box(c.nx, c.lx, c.bc)
        : factory->create_computation_box(c.nx, c.lx, c.bc, c.angles));
    std::unique_ptr<Molecules> molecules(factory->create_molecules_information(c.chain_model, 0.05, {{"A", 1.0}, {"B", 1.0}}));
    molecules->add_polymer(1.0, {{"A", 0.4, 0, 1}, {"B", 0.6, 1, 2}});
    std::unique_ptr<PropagatorComputationOptimizer> optimizer(factory->create_propagator_computation_optimizer(molecules.get(), true));
    std::unique_ptr<PropagatorComputation<T>> computation(
        factory->create_propagator_computation(cb.get(), molecules.get(), optimizer.get(), c.method, sg.get()));
    std::vector<T> w(cb->get_n_basis(), T(0.1));
    computation->compute_propagators({{"A", w.data()}, {"B", w.data()}});
    computation->compute_concentrations();
    if (c.stress)
        computation->compute_stress();   // the stress buffers and CrysFFT multipliers are built lazily
}

static void run_anderson_mixing(bool reduce_memory)
{
    const int n_var = 2 * 4096;
    std::unique_ptr<AbstractFactory<double>> factory(make_factory<double>(reduce_memory));
    std::unique_ptr<AndersonMixing<double>> am(factory->create_anderson_mixing(n_var, 5, 1e3, 0.1, 0.1));
    std::vector<double> w(n_var), w_new(n_var), w_deriv(n_var);
    for (int it = 0; it < 4; ++it)
    {
        for (int i = 0; i < n_var; ++i)
        {
            w[i] = 0.01 * ((i * 37 + it * 11) % 101);
            w_deriv[i] = 0.001 * ((i * 53 + it * 7) % 97) - 0.05;
        }
        am->calculate_new_fields(w_new.data(), w.data(), w_deriv.data(), 1.0, 0.5);
    }
}

struct Case
{
    std::string name;
    std::function<void()> run;
};

static std::vector<Case> cases()
{
    std::vector<Case> list;
    auto add = [&](const std::string& name, const ComputationSpec& spec) {
        list.push_back({name, [spec]() { run_computation<double>(spec); }});
    };
    ComputationSpec s;
    add("continuous rqm4, periodic", s);
    s = ComputationSpec(); s.method = "rk2";
    add("continuous rk2, periodic", s);
    s = ComputationSpec(); s.chain_model = "discrete";
    add("discrete, periodic", s);
    s = ComputationSpec(); s.method = "cn-adi2"; s.nx = {12, 12, 12}; s.stress = false;
    add("continuous cn-adi2", s);
    s = ComputationSpec(); s.reduce_memory = true;
    add("continuous rqm4, reduce_memory", s);
    s = ComputationSpec(); s.chain_model = "discrete"; s.reduce_memory = true;
    add("discrete, reduce_memory", s);
    s = ComputationSpec(); s.sg_symbol = "Im-3m"; s.sg_hall = 529; s.basis = "m3";
    add("continuous rqm4, Im-3m m3 basis (recursive 3m CrysFFT)", s);
    s = ComputationSpec(); s.chain_model = "discrete"; s.sg_symbol = "Pmmm"; s.basis = "pmmm"; s.lx = {2.0, 1.8, 1.6};
    add("discrete, Pmmm basis (Pmmm CrysFFT)", s);
    s = ComputationSpec(); s.method = "rk2"; s.nx = {12, 12, 16}; s.lx = {1.8, 1.8, 2.4}; s.angles = {90.0, 90.0, 120.0};
    s.sg_symbol = "P6_3/mmc"; s.sg_hall = 488; s.basis = "z-mirror";
    add("continuous rk2, P6_3/mmc z-mirror basis (ObliqueZ CrysFFT)", s);
    s = ComputationSpec(); s.sg_symbol = "Im-3m"; s.sg_hall = 529;
    add("continuous rqm4, Im-3m irreducible basis", s);
    s = ComputationSpec(); s.nx = {16, 12, 8}; s.lx = {2.0, 1.5, 1.0};
    s.bc = {"reflecting", "reflecting", "absorbing", "absorbing", "reflecting", "reflecting"};
    add("continuous rqm4, 3D reflecting/absorbing (DCT/DST)", s);
    s = ComputationSpec(); s.chain_model = "discrete"; s.nx = {16, 12}; s.lx = {2.0, 1.5};
    s.bc = {"reflecting", "reflecting", "absorbing", "absorbing"};
    add("discrete, 2D reflecting/absorbing (DCT/DST)", s);
    s = ComputationSpec(); s.nx = {32}; s.lx = {3.0}; s.bc = {"reflecting", "reflecting"};
    add("continuous rqm4, 1D reflecting (DCT)", s);
    s = ComputationSpec();
    list.push_back({"complex fields, continuous rqm4, periodic", [s]() { run_computation<std::complex<double>>(s); }});
    s = ComputationSpec(); s.chain_model = "discrete";
    list.push_back({"complex fields, discrete, periodic", [s]() { run_computation<std::complex<double>>(s); }});
    list.push_back({"Anderson mixing", []() { run_anderson_mixing(false); }});
    list.push_back({"Anderson mixing (reduce memory, pinned history)", []() { run_anderson_mixing(true); }});
    return list;
}

//------------------------------------------------------------------------------
// 4. A construction that really runs out of device memory (--out-of-memory)
//------------------------------------------------------------------------------
static bool out_of_memory_check(bool& oom_skipped)
{
    bool ok = true;
    printf("--- out of device memory ---\n");
    {
        ComputationSpec spec;
        spec.nx = {64, 64, 64};
        spec.lx = {4.0, 4.0, 4.0};

        // What one construction and run take, measured once while the objects
        // are alive: at least the counted bytes, and by device free memory also
        // cuFFT's plan work areas (unless another process changed it meanwhile)
        size_t need = 0;
        bool measured = false;
        const Live live_before_measurement = Live::now();
        const size_t free_before = free_bytes();
        try
        {
            const long long counted_before = live_device_bytes();
            std::unique_ptr<AbstractFactory<double>> factory(make_factory<double>(false));
            std::unique_ptr<ComputationBox<double>> cb(factory->create_computation_box(spec.nx, spec.lx, {}));
            std::unique_ptr<Molecules> molecules(factory->create_molecules_information("continuous", 0.05, {{"A", 1.0}, {"B", 1.0}}));
            molecules->add_polymer(1.0, {{"A", 0.4, 0, 1}, {"B", 0.6, 1, 2}});
            std::unique_ptr<PropagatorComputationOptimizer> optimizer(factory->create_propagator_computation_optimizer(molecules.get(), true));
            std::unique_ptr<PropagatorComputation<double>> computation(
                factory->create_propagator_computation(cb.get(), molecules.get(), optimizer.get(), "rqm4"));
            std::vector<double> w(cb->get_n_basis(), 0.1);
            computation->compute_propagators({{"A", w.data()}, {"B", w.data()}});
            computation->compute_concentrations();
            computation->compute_stress();
            const size_t free_during = free_bytes();
            const size_t counted = static_cast<size_t>(live_device_bytes() - counted_before);
            need = std::max(counted, free_before > free_during ? free_before - free_during : size_t(0));
            measured = true;
        }
        catch (const std::exception& e)
        {
            // Only a lack of device memory makes the section impossible; any
            // other failure of this ordinary computation is a test failure
            const std::string message = one_line(e.what());
            const bool memory = message.find("out of memory") != std::string::npos
                             || message.find("cuFFT error code 2") != std::string::npos   // CUFFT_ALLOC_FAILED
                             || free_before < (size_t(1) << 30);
            printf("  the 64^3 computation failed: %s\n", message.c_str());
            ok &= check(memory, "the 64^3 computation fails only for lack of device memory");
        }
        ok &= check(no_cuda_error("the measurement"), "the measurement leaves no release failure or pending error");
        const Live live_after_measurement = Live::now();
        if (!(live_after_measurement == live_before_measurement))
            printf("  the measurement left %s\n", live_after_measurement.minus(live_before_measurement).c_str());
        ok &= check(live_after_measurement == live_before_measurement, "the 64^3 computation releases everything");

        // Device memory after the measurement (modules loaded, cuFFT's retained
        // per-plan memory included), for the end-to-end check below: this
        // process's usage from NVML when it is available, else the device's
        // free memory (which other processes also change)
        const long long process_start = process_device_bytes();
        const size_t free_start = free_bytes();
        const Live live_start = Live::now();

        // Leave a fraction of that free; the rest of the free memory is held by a
        // ballast for the moment of the construction. The last case leaves (almost)
        // nothing, so one construction runs out even if another process changed the
        // free memory while `need` was measured.
        int ran_out = 0, tried = 0;
        if (measured)
        {
            printf("  a 64^3 computation takes about %.1f MiB of device memory\n", need / 1048576.0);
            for (double fraction : {0.9, 0.6, 0.3, 0.1, 0.02, 0.0})
            {
                const size_t leftover = static_cast<size_t>(fraction * need);
                const size_t free_now = free_bytes();
                if (free_now <= leftover)
                    continue;
                const Live before = Live::now();
                std::vector<CudaDeviceBuffer<char>> ballast = hold_device_memory(free_now - leftover);
                const size_t free_left = free_bytes();
                if (free_left > leftover + need)
                {
                    printf("  [SKIP] leftover %.1f MiB: could hold only part of the device memory (%.1f MiB left free)\n",
                           leftover / 1048576.0, free_left / 1048576.0);
                    continue;
                }
                // A driver that backs device allocations with system memory once the
                // device memory is full (the default sysmem fallback of WDDM drivers on
                // Windows) never runs out: an allocation larger than the free memory
                // must fail for the construction to be able to fail
                bool oversubscribed = false;
                try
                {
                    CudaDeviceBuffer<char> probe(free_left + need);
                    oversubscribed = true;
                }
                catch (const std::exception&) {}
                if (oversubscribed)
                {
                    printf("  [SKIP] leftover %.1f MiB: %.1f MiB could be allocated with %.1f MiB free "
                           "(device memory backed by system memory)\n",
                           leftover / 1048576.0, (free_left + need) / 1048576.0, free_left / 1048576.0);
                    continue;
                }
                ++tried;
                std::string message;
                bool threw = false;
                try { run_computation<double>(spec); }
                catch (const std::exception& e) { threw = true; message = e.what(); }
                const bool no_error = no_cuda_error("an out-of-memory construction");
                ballast.clear();
                if (threw)
                    ++ran_out;
                // The innermost failure is at the end of the message (outer frames prepend theirs)
                message = one_line(message);
                if (message.size() > 240)
                    message = "..." + message.substr(message.size() - 240);
                const Live after = Live::now();
                if (!(after == before))
                    printf("  left %s\n", after.minus(before).c_str());
                ok &= check(no_error && after == before,
                            "free " + std::to_string(free_left >> 20) + " MiB: "
                            + (threw ? "ran out (" + message + ")" : "fit") + "; everything released");
            }
        }
        if (tried > 0)
        {
            ok &= check(ran_out > 0, "at least one construction ran out of device memory");
            // End to end, for memory the counters do not see (cuFFT's internal
            // allocations; the plan counter catches a leaked plan). 8 MiB covers the
            // 2 MiB allocation pages and cuFFT's retained per-plan memory.
            const long long process_end = process_device_bytes();
            const size_t free_end = free_bytes();
            const bool per_process = (process_start >= 0 && process_end >= 0);
            const double drift_mib = per_process
                ? static_cast<double>(process_end - process_start) / 1048576.0
                : (static_cast<double>(free_start) - static_cast<double>(free_end)) / 1048576.0;
            ok &= check(Live::now() == live_start && drift_mib <= 8.0,
                        std::string("the device memory ") + (per_process ? "of this process (NVML)" : "free on the device")
                        + " returns to its level before these constructions (" + std::to_string(drift_mib)
                        + " MiB more in use; 8 MiB allowed)");
        }
        else
        {
            printf("  [SKIP] the out-of-memory check could not run (device memory could not be held, "
                   "or the driver backs it with system memory)\n");
            oom_skipped = true;
        }
    }
    return ok;
}

int main(int argc, char** argv)
{
    // --out-of-memory runs only section 4, as its own ctest entry (run serially,
    // skipped with code 77 where it cannot run), so that a skip there does not
    // hide the result of sections 1-3
    if (argc > 1 && std::string(argv[1]) == "--out-of-memory")
    {
        bool oom_skipped = false;
        const bool ok = out_of_memory_check(oom_skipped);
        printf("TestCudaDeviceBuffer --out-of-memory: %s\n", ok ? (oom_skipped ? "SKIPPED" : "PASSED") : "FAILED");
        if (!ok)
            return 1;
        return oom_skipped ? 77 : 0;   // 77: SKIP_RETURN_CODE in tests/CMakeLists.txt
    }

    bool ok = true;

    //--------------------------------------------------------------------------
    // 1. The buffer class
    //--------------------------------------------------------------------------
    printf("--- buffer class ---\n");
    const size_t n = size_t(1) << 22;               // 32 MiB of doubles, well above the allocation granule
    const long long d_start = live_device(), p_start = live_pinned();
    {
        CudaDeviceBuffer<double> a(n);
        ok &= check(a.get() != nullptr && a.size() == n && a.bytes() == n * sizeof(double) && !a.empty(), "allocation and sizes");
        ok &= check(live_device() == d_start + 1, "the counter sees the allocation");
        CUDA_CHECK(cudaMemset(a, 0, a.bytes()));   // lent to the runtime
        double* raw = a;                                 // lent, not transferred
        ok &= check(raw == a.get(), "implicit conversion lends the same pointer");

        CudaDeviceBuffer<double> b(std::move(a));
        ok &= check(a.empty() && a.size() == 0 && b.get() == raw && b.size() == n, "move constructor transfers ownership");
        CudaDeviceBuffer<double> c;
        c = std::move(b);
        ok &= check(b.empty() && c.get() == raw && c.size() == n && live_device() == d_start + 1, "move assignment transfers ownership");

        CudaDeviceBuffer<double> d(n);                  // move over a buffer that holds memory
        d = std::move(c);
        ok &= check(d.get() == raw && live_device() == d_start + 1, "move assignment over a full buffer frees the old allocation");

        const long long bytes_start = live_device_bytes() - static_cast<long long>(n * sizeof(double));
        d.allocate(16);
        ok &= check(d.size() == 16 && live_device() == d_start + 1 && live_device_bytes() == bytes_start + 16 * 8,
                    "allocate() frees the previous 32 MiB allocation");
        d.allocate(0);
        ok &= check(d.empty() && d.size() == 0 && live_device() == d_start && live_device_bytes() == bytes_start,
                    "allocate(0) frees the allocation and leaves the buffer empty");
        d.allocate(16);
        d.release();
        ok &= check(d.empty() && d.size() == 0 && live_device() == d_start, "release() empties the buffer");
        d.release();
        ok &= check(d.empty() && live_device() == d_start, "release() of an empty buffer does nothing");

        std::vector<CudaDeviceBuffer<int>> v(3);
        for (auto& e : v)
            e.allocate(1024);
        v.erase(v.begin());                              // moves the remaining elements
        ok &= check(v.size() == 2 && v[0].size() == 1024 && v[1].size() == 1024 && live_device() == d_start + 2,
                    "containers of buffers move their elements");
    }
    ok &= check(live_device() == d_start, "the destructor of the vector's buffers frees them");
    {
        const long long bytes_before = live_device_bytes();
        {
            CudaDeviceBuffer<double> only_destroyed(n);
            ok &= check(live_device_bytes() == bytes_before + static_cast<long long>(n * sizeof(double)),
                        "the counters see a 32 MiB allocation");
        }
        ok &= check(live_device_bytes() == bytes_before && live_device() == d_start,
                    "the destructor alone releases it");
    }
    {
        CudaPinnedBuffer<std::complex<double>> p(1000);
        ok &= check(p.get() != nullptr && p.size() == 1000 && live_pinned() == p_start + 1, "pinned allocation");
        p.get()[999] = std::complex<double>(1.0, 2.0);
        CudaDeviceBuffer<cuDoubleComplex> dev(1000);
        CUDA_CHECK(cudaMemcpy(dev, p, p.bytes(), cudaMemcpyHostToDevice));
        std::vector<std::complex<double>> h(1000);
        CUDA_CHECK(cudaMemcpy(h.data(), dev, dev.bytes(), cudaMemcpyDeviceToHost));
        ok &= check(h[999] == std::complex<double>(1.0, 2.0), "pinned -> device -> host round trip");
    }
    ok &= check(live_pinned() == p_start && live_device() == d_start, "pinned and device buffers released");
    ok &= check(no_cuda_error("the buffer checks"), "no release failed and no CUDA error is pending");

    bool threw = false;
    std::string message;
    try
    {
        CudaDeviceBuffer<double> huge(size_t(1) << 50);   // 8 PiB: cannot succeed
    }
    catch (const std::exception& e)
    {
        threw = true;
        message = e.what();
    }
    printf("  message: %s\n", message.c_str());
    ok &= check(threw && live_device() == d_start, "an impossible allocation throws and is not counted");
    ok &= check(message.find("9007199254740992 bytes of device memory") != std::string::npos
                && message.find("TestCudaDeviceBuffer.cu") != std::string::npos
                && message.find("main") != std::string::npos,
                "its message gives the size and the caller's file and function");
    ok &= check(no_cuda_error("the failed allocation"), "the failed allocation leaves no error pending");

    cuda_buffer_fail_countdown().store(0);
    message.clear();
    try
    {
        CudaPinnedBuffer<int> injected(4);
    }
    catch (const std::exception& e)
    {
        message = e.what();
    }
    ok &= check(message.find("test hook") != std::string::npos && message.find("16 bytes of pinned host memory") != std::string::npos
                && cuda_buffer_fail_countdown().load() < 0 && live_pinned() == p_start,
                "the failure hook throws once, with a message that says so");
    cuda_buffer_fail_countdown().store(-1);

    //--------------------------------------------------------------------------
    // 2. Every kind of object returns everything it took
    //--------------------------------------------------------------------------
    printf("--- objects release their buffers, plans, streams and events (4 streams) ---\n");
    const std::vector<Case> all = cases();
    {
        ScopedStreams streams("4");
        // The counters see what a live computation holds: 4 streams with a kernel
        // and a copy stream and two events each (reduce-memory computations use
        // one stream), and its plans and buffers
        {
            const Live before = Live::now();
            std::unique_ptr<AbstractFactory<double>> factory(make_factory<double>(false));
            std::unique_ptr<ComputationBox<double>> cb(factory->create_computation_box({16, 16, 16}, {1.9, 1.9, 1.9}, {}));
            std::unique_ptr<Molecules> molecules(factory->create_molecules_information("continuous", 0.05, {{"A", 1.0}, {"B", 1.0}}));
            molecules->add_polymer(1.0, {{"A", 0.4, 0, 1}, {"B", 0.6, 1, 2}});
            std::unique_ptr<PropagatorComputationOptimizer> optimizer(factory->create_propagator_computation_optimizer(molecules.get(), true));
            std::unique_ptr<PropagatorComputation<double>> computation(
                factory->create_propagator_computation(cb.get(), molecules.get(), optimizer.get(), "rqm4"));
            const Live alive = Live::now();
            ok &= check(alive.streams - before.streams == 8 && alive.events - before.events == 8
                        && alive.plans > before.plans && alive.device > before.device,
                        "a live 4-stream computation holds 8 streams, 8 events, " + std::to_string(alive.plans - before.plans)
                        + " cuFFT plans and " + std::to_string(alive.device - before.device) + " device allocations");
        }
        for (const Case& c : all)
        {
            const Live before = Live::now();
            bool ran = true;
            try { c.run(); }
            catch (const std::exception& e) { ran = false; printf("  %s threw: %s\n", c.name.c_str(), e.what()); }
            const bool no_error = no_cuda_error(c.name);
            const Live after = Live::now();
            if (!(after == before))
                printf("  %s left %s\n", c.name.c_str(), after.minus(before).c_str());
            ok &= check(ran && no_error && after == before, c.name);
        }
    }

    //--------------------------------------------------------------------------
    // 3. Failure-injection sweep (2 streams: the per-stream code runs for a
    //    stream other than the first, with fewer allocations than 4 streams;
    //    reduce-memory computations use one stream)
    //--------------------------------------------------------------------------
    printf("--- every allocation of every construction fails once (2 streams) ---\n");
    {
        ScopedStreams streams("2");
        for (const Case& c : all)
        {
            const Live before = Live::now();
            long long points = 0;
            bool clean = true;
            for (long long k = 0; k < 100000; ++k)
            {
                cuda_buffer_fail_countdown().store(k);
                bool threw = false;
                std::string message;
                try { c.run(); }
                catch (const std::exception& e) { threw = true; message = e.what(); }
                const bool fired = cuda_buffer_fail_countdown().load() < 0;
                cuda_buffer_fail_countdown().store(-1);
                const bool no_error = no_cuda_error(c.name);
                const Live after = Live::now();
                if (!no_error || !(after == before))
                {
                    printf("  %s: allocation %lld failed and left %s\n", c.name.c_str(), k, after.minus(before).c_str());
                    clean = false;
                    break;
                }
                if (!fired)                              // the run made fewer than k+1 allocations
                {
                    if (threw)                           // so nothing was injected: it must succeed
                    {
                        printf("  %s threw without an injected failure: %s\n", c.name.c_str(), message.c_str());
                        clean = false;
                    }
                    points = k;
                    break;
                }
            }
            ok &= check(clean && points > 0, c.name + ": " + std::to_string(points) + " failure points, nothing left behind");
        }
    }

    printf("TestCudaDeviceBuffer: %s\n", ok ? "PASSED" : "FAILED");
    return ok ? 0 : 1;
}
