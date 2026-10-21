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
 *    with 4 streams, returns everything when destroyed. Checked exactly with the
 *    library's live counters: CudaBuffer and the backend's propagator and
 *    checkpoint arrays, cuFFT plans, streams and events (other processes and
 *    cuFFT's own retained memory do not affect them). No release may fail
 *    (cuda_release_failures(); a double free would count).
 * 3. Failure-injection sweep with 2 streams: the test hook
 *    cuda_buffer_fail_countdown() makes the N-th counted allocation throw; N runs
 *    over every allocation of each construction (and its first run), and after
 *    each failure the counters must be back where they started; the run in which
 *    no allocation fails must succeed. Failures of cuFFT plan, stream and event
 *    creation are not injected; section 4 may reach them for real.
 * 4. A small construction that really runs out of device memory releases what it
 *    took (the counters, and device free memory end to end): a ballast holds the
 *    rest of the free memory for a moment, with several leftovers so the failure
 *    falls at different stages. When the device memory cannot be held, the test
 *    exits with code 77 (a ctest skip). It runs serially (tests/CMakeLists.txt).
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
    long long device = 0, pinned = 0, plans = 0, streams = 0, events = 0;
    static Live now()
    {
        const CudaHandleCounters& h = cuda_handle_counters();
        return {live_device(), live_pinned(), h.cufft_plans.load(), h.streams.load(), h.events.load()};
    }
    bool operator==(const Live&) const = default;
    std::string minus(const Live& before) const
    {
        return std::to_string(device - before.device) + " device and " + std::to_string(pinned - before.pinned)
            + " pinned allocations, " + std::to_string(plans - before.plans) + " cuFFT plans, "
            + std::to_string(streams - before.streams) + " streams, " + std::to_string(events - before.events) + " events";
    }
};

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
        setenv("OMP_NUM_THREADS", n, 1);
    }
    ~ScopedStreams()
    {
        if (had)
            setenv("OMP_NUM_THREADS", saved.c_str(), 1);
        else
            unsetenv("OMP_NUM_THREADS");
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
static AbstractFactory<T>* make_factory(bool reduce_memory)
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

int main()
{
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
    //    stream other than the first, with fewer allocations than 4 streams)
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

    //--------------------------------------------------------------------------
    // 4. A construction that really runs out of device memory
    //--------------------------------------------------------------------------
    printf("--- out of device memory ---\n");
    bool oom_skipped = false;
    {
        ComputationSpec spec;
        spec.nx = {64, 64, 64};
        spec.lx = {4.0, 4.0, 4.0};

        // What one construction and run take, measured once while the objects
        // are alive: at least the counted bytes, and by device free memory also
        // cuFFT's plan work areas (unless another process changed it meanwhile)
        size_t need = 0;
        bool measured = false;
        try
        {
            const size_t free_before = free_bytes();
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
            printf("  the 64^3 computation itself did not fit: %s\n", e.what());
        }
        ok &= check(no_cuda_error("the measurement"), "the measurement leaves no release failure or pending error");

        // Free memory after the measurement (modules loaded, cuFFT's retained
        // per-plan memory included), for the end-to-end check below
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
            // End to end, for memory the counters do not see (cuFFT's own); a leaked
            // 64^3 plan holds several MiB
            const size_t free_end = free_bytes();
            const double drift_mib = (static_cast<double>(free_start) - static_cast<double>(free_end)) / 1048576.0;
            ok &= check(Live::now() == live_start && free_end + (size_t(8) << 20) >= free_start,
                        "the device memory returns to its level before these constructions ("
                        + std::to_string(drift_mib) + " MiB less; 8 MiB allowed for the 2 MiB pages and cuFFT's retained memory)");
        }
        else
        {
            printf("  [SKIP] the out-of-memory check could not run (device memory could not be held)\n");
            oom_skipped = true;
        }
    }

    printf("TestCudaDeviceBuffer: %s\n", ok ? (oom_skipped ? "PASSED (out-of-memory check skipped)" : "PASSED") : "FAILED");
    if (!ok)
        return 1;
    return oom_skipped ? 77 : 0;   // 77: SKIP_RETURN_CODE in tests/CMakeLists.txt
}
