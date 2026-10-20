/**
 * @file TestCudaDeviceBuffer.cu
 * @brief Owning CUDA buffers (CudaDeviceBuffer / CudaPinnedBuffer) and the objects
 *        built on them.
 *
 * 1. The buffer class: allocation, lending, moves (also over a buffer that holds
 *    memory), release, containers, the destructor, pinned memory, a failed allocation.
 * 2. Every kind of CUDA object (propagator computations of all solvers, the CrysFFT
 *    engines, non-periodic transforms, complex fields, the Anderson mixers) returns
 *    all its buffers when destroyed. Checked exactly with the live-allocation
 *    counters of the buffer class, which other processes and cuFFT's own plan
 *    memory do not affect.
 * 3. Failure-injection sweep: the test hook cuda_buffer_fail_countdown() makes the
 *    N-th buffer allocation throw; N runs over every allocation of each construction
 *    (and its first run), and after each failure the counters must be back where
 *    they started. This covers every partial-construction path.
 * 4. A construction that really runs out of device memory releases what it took.
 *    That check fills the GPU on purpose (the test runs serially, see
 *    tests/CMakeLists.txt).
 */
#include <algorithm>
#include <complex>
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

static size_t free_bytes()
{
    size_t f = 0, t = 0;
    gpu_error_check(cudaDeviceSynchronize());
    gpu_error_check(cudaMemGetInfo(&f, &t));
    return f;
}

static bool check(bool cond, const std::string& what)
{
    printf("  [%s] %s\n", cond ? "PASS" : "FAIL", what.c_str());
    return cond;
}

/// The release calls ignore errors on purpose; a failed cudaFree (e.g. a double
/// free) still leaves its error in the last-error slot, which this reads.
static bool no_cuda_error(const std::string& where)
{
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
        printf("  CUDA error after %s: %s\n", where.c_str(), cudaGetErrorString(e));
    return e == cudaSuccess;
}

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
        gpu_error_check(cudaMemset(a, 0, a.bytes()));   // lent to the runtime
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

        const size_t before_allocate = free_bytes();
        d.allocate(16);
        ok &= check(d.size() == 16 && live_device() == d_start + 1 && free_bytes() > before_allocate + n * sizeof(double) / 2,
                    "allocate() frees the previous 32 MiB allocation");
        d.release();
        ok &= check(d.empty() && d.size() == 0 && live_device() == d_start, "release() empties the buffer");
        d.allocate(0);
        ok &= check(d.empty() && live_device() == d_start, "allocate(0) leaves the buffer empty");

        std::vector<CudaDeviceBuffer<int>> v(3);
        for (auto& e : v)
            e.allocate(1024);
        v.erase(v.begin());                              // moves the remaining elements
        ok &= check(v.size() == 2 && v[0].size() == 1024 && v[1].size() == 1024 && live_device() == d_start + 2,
                    "containers of buffers move their elements");
    }
    ok &= check(live_device() == d_start, "the destructor of the vector's buffers frees them");
    {
        const size_t before = free_bytes();
        {
            CudaDeviceBuffer<double> only_destroyed(n);
        }
        ok &= check(free_bytes() + n * sizeof(double) / 2 > before && live_device() == d_start,
                    "the destructor alone returns a 32 MiB allocation to the device");
    }
    {
        CudaPinnedBuffer<std::complex<double>> p(1000);
        ok &= check(p.get() != nullptr && p.size() == 1000 && live_pinned() == p_start + 1, "pinned allocation");
        p.get()[999] = std::complex<double>(1.0, 2.0);
        CudaDeviceBuffer<cuDoubleComplex> dev(1000);
        gpu_error_check(cudaMemcpy(dev, p, p.bytes(), cudaMemcpyHostToDevice));
        std::vector<std::complex<double>> h(1000);
        gpu_error_check(cudaMemcpy(h.data(), dev, dev.bytes(), cudaMemcpyDeviceToHost));
        ok &= check(h[999] == std::complex<double>(1.0, 2.0), "pinned -> device -> host round trip");
    }
    ok &= check(live_pinned() == p_start && live_device() == d_start, "pinned and device buffers released");
    ok &= check(no_cuda_error("the buffer checks"), "no CUDA error left by the releases");

    bool threw = false;
    try
    {
        CudaDeviceBuffer<double> huge(size_t(1) << 50);   // 8 PiB: cannot succeed
    }
    catch (const std::exception&)
    {
        threw = true;
    }
    cudaGetLastError();                                  // clear the allocation error
    ok &= check(threw && live_device() == d_start, "an impossible allocation throws and is not counted");

    //--------------------------------------------------------------------------
    // 2. Every kind of object returns all its buffers
    //--------------------------------------------------------------------------
    printf("--- objects release their buffers (4 streams) ---\n");
    const std::vector<Case> all = cases();
    for (const Case& c : all)
    {
        const long long d0 = live_device(), p0 = live_pinned();
        bool ran = true;
        try { c.run(); }
        catch (const std::exception& e) { ran = false; printf("  %s threw: %s\n", c.name.c_str(), e.what()); }
        ok &= check(ran && live_device() == d0 && live_pinned() == p0 && no_cuda_error(c.name), c.name);
    }

    //--------------------------------------------------------------------------
    // 3. Failure-injection sweep (one stream: fewer allocations, same code)
    //--------------------------------------------------------------------------
    printf("--- every allocation of every construction fails once ---\n");
    setenv("OMP_NUM_THREADS", "1", 1);                   // the backends size their streams from it
    for (const Case& c : all)
    {
        const long long d0 = live_device(), p0 = live_pinned();
        long long points = 0;
        bool clean = true;
        for (long long k = 0; k < 100000; ++k)
        {
            cuda_buffer_fail_countdown().store(k);
            try { c.run(); }
            catch (const std::exception&) {}
            const bool fired = cuda_buffer_fail_countdown().load() < 0;
            cuda_buffer_fail_countdown().store(-1);
            if (live_device() != d0 || live_pinned() != p0 || !no_cuda_error(c.name))
            {
                printf("  %s: allocation %lld failed and left %lld device / %lld pinned buffers\n",
                       c.name.c_str(), k, live_device() - d0, live_pinned() - p0);
                clean = false;
                break;
            }
            if (!fired)                                  // the run made fewer than k+1 allocations
            {
                points = k;
                break;
            }
        }
        ok &= check(clean && points > 0, c.name + ": " + std::to_string(points) + " failure points, nothing left behind");
    }
    unsetenv("OMP_NUM_THREADS");

    //--------------------------------------------------------------------------
    // 4. A construction that really runs out of device memory
    //--------------------------------------------------------------------------
    printf("--- out of device memory ---\n");
    {
        const long long d0 = live_device(), p0 = live_pinned();
        const size_t before_oom = free_bytes();
        // The box takes 3 fields of m doubles; the computation then runs out partway
        const size_t m = std::min(before_oom / (8 * 5), size_t(1) << 28);
        const int ny = 64, nz = 64;
        const int nxx = static_cast<int>(m / (ny * nz));
        if (nxx < 2)
        {
            printf("  [SKIP] not enough free device memory for the out-of-memory check\n");
        }
        else
        {
            bool built = false, failed = false;
            {
                std::unique_ptr<AbstractFactory<double>> factory(make_factory<double>(false));
                std::unique_ptr<ComputationBox<double>> cb(factory->create_computation_box({nxx, ny, nz}, {4.0, 2.0, 2.0}, {}));
                std::unique_ptr<Molecules> molecules(factory->create_molecules_information("continuous", 0.05, {{"A", 1.0}, {"B", 1.0}}));
                molecules->add_polymer(1.0, {{"A", 0.4, 0, 1}, {"B", 0.6, 1, 2}});
                std::unique_ptr<PropagatorComputationOptimizer> optimizer(factory->create_propagator_computation_optimizer(molecules.get(), true));
                try
                {
                    std::unique_ptr<PropagatorComputation<double>> computation(
                        factory->create_propagator_computation(cb.get(), molecules.get(), optimizer.get(), "rqm4"));
                    built = true;
                }
                catch (const std::exception&)
                {
                    failed = true;
                }
                cudaGetLastError();                      // clear the out-of-memory error
            }
            const size_t after_oom = free_bytes();
            printf("  grid %d x %d x %d: construction %s; device free before %zu, after %zu\n",
                   nxx, ny, nz, failed ? "ran out of memory" : (built ? "succeeded" : "?"), before_oom, after_oom);
            if (built)
                printf("  [SKIP] the device had room for the whole computation; failure path not exercised\n");
            // The propagator handles come from the backend's own allocator, not
            // CudaBuffer; the free-memory comparison covers them (64 MiB margin
            // for cuFFT's per-plan memory and the 2 MiB pages; one field here is GiBs)
            ok &= check(live_device() == d0 && live_pinned() == p0 && after_oom + (size_t(64) << 20) >= before_oom,
                        "a construction that runs out of memory releases what it acquired");
        }
    }

    printf("TestCudaDeviceBuffer: %s\n", ok ? "PASSED" : "FAILED");
    return ok ? 0 : 1;
}
