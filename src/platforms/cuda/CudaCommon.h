/**
 * @file CudaCommon.h
 * @brief CUDA utilities, kernel declarations, and common GPU infrastructure.
 *
 * This header provides essential CUDA infrastructure including:
 *
 * - **CudaCommon singleton**: GPU configuration (blocks, threads)
 * - **Type mappings**: std::complex<double> ↔ cuDoubleComplex
 * - **Error handling**: GPU error checking macros
 * - **CUDA kernels**: Element-wise operations on GPU arrays
 *
 * **Multi-Stream Architecture:**
 *
 * The library supports up to MAX_STREAMS (4) parallel CUDA streams for
 * concurrent propagator computation. Each stream has two sub-streams:
 * one for kernel execution and one for memory transfers, enabling
 * overlap of computation and data movement.
 *
 * **Type System:**
 *
 * The template type T is mapped to CUDA types via CuDeviceData<T>:
 * - double → double (real fields)
 * - std::complex<double> → cuDoubleComplex (complex fields)
 *
 * @see CudaFactory for creating CUDA-based simulation objects
 */

#ifndef CUDA_COMMON_H_
#define CUDA_COMMON_H_

#include <atomic>
#include <complex>
#include <map>
#include <source_location>
#include <string>
#include <cuda_runtime.h>
#include <cufft.h>

#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>
#include "Exception.h"
#include "GridLimits.h"

// // The maximum of GPUs
// #define MAX_GPUS 2

/**
 * @def MAX_STREAMS
 * @brief Maximum number of parallel CUDA streams for propagator computation.
 *
 * Each stream enables independent propagator computations to run
 * concurrently on the GPU. More streams can hide memory latency
 * but increase memory usage.
 */
#define MAX_STREAMS 4

/**
 * @typedef ftsComplex
 * @brief Alias for cuDoubleComplex (CUDA double-precision complex).
 */
typedef cuDoubleComplex ftsComplex;

/**
 * @brief Type mapping from C++ types to CUDA device types.
 *
 * Maps template parameter T to appropriate CUDA type:
 * - double → double
 * - std::complex<double> → cuDoubleComplex
 *
 * @tparam T C++ numeric type (double or std::complex<double>)
 *
 * @example
 * @code
 * CuDeviceData<double>* d_real;              // double*
 * CuDeviceData<std::complex<double>>* d_cplx; // cuDoubleComplex*
 * @endcode
 */
template<typename T>
using CuDeviceData = std::conditional_t<std::is_same_v<T, double>,               double,
                     std::conditional_t<std::is_same_v<T, std::complex<double>>, cuDoubleComplex, void>
>;

// Complex buffers are copied with sizeof(T)*n bytes and their pointers are
// reinterpreted between the two types (reinterpret_map, raw casts), so the
// host and device complex types must have the same size.
static_assert(sizeof(std::complex<double>) == sizeof(cuDoubleComplex),
              "std::complex<double> and cuDoubleComplex must have the same size");

/**
 * @struct ComplexSumOp
 * @brief Custom reduction operator for cuDoubleComplex summation.
 *
 * Used with CUB or Thrust reduction algorithms for summing complex arrays.
 */
struct ComplexSumOp {
    __device__ __forceinline__
    cuDoubleComplex operator()(const cuDoubleComplex& a, const cuDoubleComplex& b) const {
        cuDoubleComplex result;
        result.x = a.x + b.x;
        result.y = a.y + b.y;
        return result;
    }
};

/**
 * @brief Convert cuDoubleComplex to std::complex<double>.
 * @param z CUDA complex number
 * @return Equivalent std::complex<double>
 */
std::complex<double> cuDoubleToStdComplex(cuDoubleComplex z);

/**
 * @brief Convert std::complex<double> to cuDoubleComplex.
 * @param z Standard library complex number
 * @return Equivalent cuDoubleComplex
 */
cuDoubleComplex stdToCuDoubleComplex(const std::complex<double>& z);

/**
 * @brief Reinterpret a map of pointers from one type to another.
 * @tparam From Source pointer type
 * @tparam To Target pointer type
 * @param input Map with From* values
 * @return Map with To* values (reinterpret_cast)
 */
template<typename From, typename To>
std::map<std::string, const To*> reinterpret_map(const std::map<std::string, const From*>& input);

/**
 * @class CudaCommon
 * @brief Singleton class for CUDA GPU configuration.
 *
 * Manages the launch configuration of the grid-stride kernels:
 * - Number of thread blocks (default 1024, LFTS_GPU_NUM_BLOCKS)
 * - Number of threads per block (default 256, LFTS_GPU_NUM_THREADS)
 *
 * The kernels run on the CUDA runtime's current device; select the GPU with
 * CUDA_VISIBLE_DEVICES.
 *
 * Uses Scott Meyer's singleton pattern for thread-safe initialization.
 *
 * @example
 * @code
 * CudaCommon& cuda = CudaCommon::get_instance();
 * my_kernel<<<cuda.get_n_blocks(), cuda.get_n_threads()>>>(...);
 * @endcode
 */
class CudaCommon
{
private:
    int n_blocks;
    int n_threads;

    /// Throw unless n_threads <= 1024, 1 <= n_blocks * n_threads <= MAX_GRID_THREADS and
    /// both are within the limits of the current device
    static void check_launch(int n_blocks, int n_threads);

    CudaCommon();
    ~CudaCommon();
    // Disable copy constructor
    CudaCommon(const CudaCommon &) = delete;
    CudaCommon& operator= (const CudaCommon &) = delete;
public:

    /**
     * @brief Get singleton instance.
     * @return Reference to the CudaCommon singleton
     */
    static CudaCommon& get_instance()
    {
        try{
            static CudaCommon* instance = new CudaCommon();
            return *instance;
        }
        catch(std::exception& exc)
        {
            throw_without_line_number(exc.what());
        }
    };

    /// Get number of thread blocks.
    int get_n_blocks();

    /// Get threads per block.
    int get_n_threads();

    /// Set number of thread blocks.
    void set_n_blocks(int n_blocks);

    /// Set threads per block.
    void set_n_threads(int n_threads);
};

/**
 * @def CUDA_CHECK
 * @brief Macro for CUDA error checking with source location.
 *
 * Wraps CUDA API calls and throws an exception on error. Every call that
 * returns an error code is checked, and every kernel launch is followed by
 * CUDA_CHECK_LAUNCH(), as the CUDA C++ Best Practices Guide (2.3)
 * recommends: the check is a host-side read that does not synchronize,
 * and cudaGetLastError() clears the slot, so an error is reported at the
 * launch that caused it and not again by a later, unrelated check. Errors of
 * a running kernel (e.g. an illegal address) are not seen by the launch check:
 * they surface at the first CUDA call made after the fault, at the latest at
 * the next synchronizing call, which names its own line; run with
 * CUDA_LAUNCH_BLOCKING=1 (or compute-sanitizer) to see them at their launch.
 * Release calls are not checked: they run in destructors and release() paths,
 * which must not throw. cudaFree/cudaFreeHost go through ignore_release_error()
 * so a failed one is not reported by a later check; plans, streams and events
 * are created and destroyed through the counting helpers (count_cufft_plan(),
 * release_cufft_plan(), ...).
 *
 * @example
 * @code
 * CUDA_CHECK(count_cuda_stream(cudaStreamCreate(&stream)));   // counted handle
 * CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
 * ker_scale<<<blocks, threads, 0, stream>>>(d_x, 2.0, n);
 * CUDA_CHECK_LAUNCH();
 * @endcode
 */
#define CUDA_CHECK(code) throw_on_cuda_error((code), __FILE__, __LINE__, __func__)

/**
 * @def CUDA_CHECK_LAUNCH
 * @brief Check the kernel launch just issued: CUDA_CHECK(cudaGetLastError()).
 *
 * Kernel launches return void; a bad launch (configuration, shared memory)
 * is only recorded in the error slot. Put this right after every launch.
 */
#define CUDA_CHECK_LAUNCH() CUDA_CHECK(cudaGetLastError())

/// Throw the exception for a failed CUDA call (out of line: only the error path pays).
[[noreturn]] void report_cuda_error(cudaError_t code, const char *file, int line, const char *func);

/// Throw the exception for a failed (or, with `injected`, a test-hook) allocation of
/// `bytes` bytes of `kind` memory, naming the code that asked for it.
[[noreturn]] void report_cuda_allocation_error(cudaError_t code, size_t bytes, const char *kind,
                                               const std::source_location& where, bool injected = false);

/**
 * @brief Throw exception on CUDA error.
 * @param code CUDA error code
 * @param file Source file name
 * @param line Line number
 * @param func Function name
 * @throws Exception if code != cudaSuccess
 */
inline void throw_on_cuda_error(cudaError_t code, const char *file, int line, const char *func)
{
    if (code != cudaSuccess)
        report_cuda_error(code, file, line, func);
}

/**
 * @def CUFFT_CHECK
 * @brief Macro for cuFFT error checking with source location.
 *
 * Wraps cuFFT API calls (plan creation, cufftSetStream, cufftExec*) and
 * throws an exception on any result other than CUFFT_SUCCESS.
 *
 * @example
 * @code
 * CUFFT_CHECK(count_cufft_plan(cufftPlanMany(&plan, ...)));   // counted handle
 * CUFFT_CHECK(cufftExecD2Z(plan, in, out));
 * @endcode
 */
#define CUFFT_CHECK(code) throw_on_cufft_error((code), __FILE__, __LINE__, __func__)

/// Blocks for a grid-stride kernel over n elements: enough to cover n, capped at
/// MAX_GRID_THREADS threads in total (the kernels loop over the rest), so that
/// `i += blockDim.x * gridDim.x` stays within int (see GridLimits.h).
inline int grid_blocks(long long n, int block_size)
{
    const long long blocks = (n + block_size - 1) / block_size;
    const long long max_blocks = MAX_GRID_THREADS / block_size;
    return static_cast<int>(blocks < 1 ? 1 : (blocks > max_blocks ? max_blocks : blocks));
}

/**
 * @brief Throw exception on cuFFT error.
 * @param code cuFFT result code
 * @param file Source file name
 * @param line Line number
 * @param func Function name
 * @throws Exception if code != CUFFT_SUCCESS
 */
inline void throw_on_cufft_error(cufftResult code, const char *file, int line, const char *func)
{
    if (code != CUFFT_SUCCESS)
        throw Exception("cuFFT error code " + std::to_string(static_cast<int>(code)), file, line, func);
}

/// Release calls that failed since the process started (a double free, or a
/// release on a broken context); the tests check that it does not grow.
inline std::atomic<long long>& cuda_release_failures()
{
    static std::atomic<long long> count{0};
    return count;
}

/**
 * @brief Take the result of a release call (cudaFree, cudaFreeHost) without
 *        throwing; streams, events and cuFFT plans go through the counting
 *        release_cuda_stream(), release_cuda_event() and release_cufft_plan().
 *
 * Releases run in destructors and release() paths, which must not throw, and a
 * failed release cannot be retried. A failed runtime call (cudaFree, ...) still
 * left its error in the per-thread slot, where the next CUDA_CHECK_LAUNCH()
 * would report it against an unrelated launch; this clears the slot and counts
 * the failure in cuda_release_failures(). A sticky error of a broken context
 * stays and is reported by the next checked call, as it must. (cuFFT plans
 * are destroyed by release_cufft_plan(): cuFFT has its own runtime and error
 * state.)
 *
 * @code
 * ignore_release_error(cudaFree(ptr));
 * release_cuda_stream(stream);       // counted handles have their own release
 * release_cufft_plan(plan);
 * @endcode
 */
inline void ignore_release_error(cudaError_t code) noexcept
{
    if (code != cudaSuccess)
    {
        (void)cudaGetLastError();
        cuda_release_failures().fetch_add(1, std::memory_order_relaxed);
    }
}
/// Live cuFFT plans, CUDA streams and CUDA events created by the library. Like
/// the allocation counters below they are exact, so the tests can check that
/// destroyed objects released all of them (device free memory shows a leaked
/// plan's work area only roughly).
struct CudaHandleCounters
{
    std::atomic<long long> cufft_plans{0};
    std::atomic<long long> streams{0};
    std::atomic<long long> events{0};
};
inline CudaHandleCounters& cuda_handle_counters()
{
    static CudaHandleCounters counters;
    return counters;
}

/// Pass the result of a call that creates a cuFFT plan handle (cufftCreate,
/// cufftPlan1d/2d/3d, cufftPlanMany) through this; a successful one is counted.
/// @code
/// CUFFT_CHECK(count_cufft_plan(cufftPlanMany(&plan, ...)));
/// @endcode
inline cufftResult count_cufft_plan(cufftResult code) noexcept
{
    if (code == CUFFT_SUCCESS)
        cuda_handle_counters().cufft_plans.fetch_add(1, std::memory_order_relaxed);
    return code;
}
/// Destroy a plan counted by count_cufft_plan(), without throwing. Like every
/// release helper, it takes the plan off the live count whether or not the
/// destroy call succeeded (the object no longer holds it) and counts a failure
/// in cuda_release_failures(). cuFFT carries its own CUDA runtime (libcufft does
/// not link libcudart), so a failure leaves nothing in this program's error slot.
inline void release_cufft_plan(cufftHandle plan) noexcept
{
    cuda_handle_counters().cufft_plans.fetch_sub(1, std::memory_order_relaxed);
    if (cufftDestroy(plan) != CUFFT_SUCCESS)
        cuda_release_failures().fetch_add(1, std::memory_order_relaxed);
}
/// Count a created stream (pass the result of cudaStreamCreate through this).
inline cudaError_t count_cuda_stream(cudaError_t code) noexcept
{
    if (code == cudaSuccess)
        cuda_handle_counters().streams.fetch_add(1, std::memory_order_relaxed);
    return code;
}
/// Destroy a stream counted by count_cuda_stream(), without throwing.
inline void release_cuda_stream(cudaStream_t stream) noexcept
{
    cuda_handle_counters().streams.fetch_sub(1, std::memory_order_relaxed);
    ignore_release_error(cudaStreamDestroy(stream));
}
/// Count a created event (pass the result of cudaEventCreate* through this).
inline cudaError_t count_cuda_event(cudaError_t code) noexcept
{
    if (code == cudaSuccess)
        cuda_handle_counters().events.fetch_add(1, std::memory_order_relaxed);
    return code;
}
/// Destroy an event counted by count_cuda_event(), without throwing.
inline void release_cuda_event(cudaEvent_t event) noexcept
{
    cuda_handle_counters().events.fetch_sub(1, std::memory_order_relaxed);
    ignore_release_error(cudaEventDestroy(event));
}

/**
 * @brief Owning CUDA event, destroyed with the object.
 *
 * The profiling paths time their steps with these, so an exception between
 * the create and the elapsed-time read does not leak the events.
 *
 * @code
 * std::vector<CudaEvent> ev(profile ? 2 : 0);   // created only when profiling
 * if (profile) CUDA_CHECK(cudaEventRecord(ev[0], stream));
 * @endcode
 */
class CudaEvent
{
public:
    CudaEvent() { CUDA_CHECK(count_cuda_event(cudaEventCreate(&event_))); }
    ~CudaEvent() { release_cuda_event(event_); }
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
    operator cudaEvent_t() const noexcept { return event_; }   ///< lends the handle
private:
    cudaEvent_t event_ = nullptr;
};

__global__ void ker_copy_data_with_idx(cuDoubleComplex* dst, const cuDoubleComplex* src, const int* negative_k_idx, const int M);

__global__ void ker_linear_scaling(double* dst, const double* src, double a, double b, const int M);
__global__ void ker_linear_scaling(cuDoubleComplex* dst, const cuDoubleComplex* src, double a, cuDoubleComplex b, const int M);
__global__ void ker_linear_scaling(cuDoubleComplex* dst, const cuDoubleComplex* src, cuDoubleComplex a, double b, const int M);

__global__ void ker_exp(double* dst, const double* src, double a, double exp_b, const int M);
__global__ void ker_exp(cuDoubleComplex* dst, const cuDoubleComplex* src, double a, double exp_b, const int M);

__global__ void ker_multi(double* dst, const double* src1, const double* src2, double a, const int M);
__global__ void ker_multi(cuDoubleComplex* dst, const cuDoubleComplex* src1, const double* src2, double a, const int M);
__global__ void ker_multi(cuDoubleComplex* dst, const cuDoubleComplex* src1, const cuDoubleComplex* src2, double a, const int M);
__global__ void ker_multi(cuDoubleComplex* dst, const cuDoubleComplex* src1, const cuDoubleComplex* src2, cuDoubleComplex a, const int M);
// Combined stress kernels for non-orthogonal boxes

__global__ void ker_mutiple_multi(int n_comp, double* dst, const double* src1, const double* src2, double  a, const int M);
__global__ void ker_mutiple_multi(int n_comp, cuDoubleComplex* dst, const cuDoubleComplex* src1, const cuDoubleComplex* src2, double  a, const int M);

__global__ void ker_multi_complex_conjugate(double* dst, const cuDoubleComplex* src1, const cuDoubleComplex* src2, const int M);
__global__ void ker_multi_complex_conjugate(cuDoubleComplex* dst, const cuDoubleComplex* src1, const cuDoubleComplex* src2, const int M);

__global__ void ker_divide(double* dst, const double* src1, const double* src2, double a, const int M);
__global__ void ker_divide(cuDoubleComplex* dst, const cuDoubleComplex* src1, const cuDoubleComplex* src2, double a, const int M);

__global__ void ker_add_multi(double* dst, const double* src1, const double* src2, double a, const int M);
__global__ void ker_add_multi(cuDoubleComplex* dst, const cuDoubleComplex* src1, const cuDoubleComplex* src2, double a, const int M);
__global__ void ker_add_multi(cuDoubleComplex* dst, const cuDoubleComplex* src1, const cuDoubleComplex* src2, cuDoubleComplex a, const int M);

__global__ void ker_lin_comb(double* dst, double a, const double* src1, double b, const double* src2, const int M);
__global__ void ker_lin_comb(cuDoubleComplex* dst, double a, const cuDoubleComplex* src1, double b, const cuDoubleComplex* src2, const int M);
__global__ void ker_lin_comb(cuDoubleComplex* dst, cuDoubleComplex a, const cuDoubleComplex* src1, double b, const cuDoubleComplex* src2, const int M);
/// dst = a*(scale*src1*w1) + b*src2: ker_multi(src1, w1, scale) followed by
/// ker_lin_comb in one pass, with the same arithmetic (RQM4 Richardson step).
__global__ void ker_multi_lin_comb(double* dst, double a, const double* src1, const double* w1, double scale,
                                   double b, const double* src2, const int M);
__global__ void ker_multi_lin_comb(cuDoubleComplex* dst, double a, const cuDoubleComplex* src1, const cuDoubleComplex* w1, double scale,
                                   double b, const cuDoubleComplex* src2, const int M);
__global__ void ker_lin_comb_exp(
    double* dst,
    const double* src1, const double* exp1, double a,
    const double* src2, const double* exp2, double b,
    const double* mask, const int M);

__global__ void ker_add_lin_comb(double* dst, double a, const double* src1, double b, const double* src2, const int M);
__global__ void ker_add_lin_comb(cuDoubleComplex* dst, double a, const cuDoubleComplex* src1, double b, const cuDoubleComplex* src2, const int M);
__global__ void ker_add_lin_comb(cuDoubleComplex* dst, cuDoubleComplex a, const cuDoubleComplex* src1, cuDoubleComplex b, const cuDoubleComplex* src2, const int M);

__global__ void ker_multi_complex_real(cuDoubleComplex* dst, const double* src, double a, const int M);
__global__ void ker_multi_complex_real(cuDoubleComplex* dst, const cuDoubleComplex* src1, const double* src2, double a, const int M);

__global__ void ker_multi_exp_dw_two(
    double* dst1, const double* src1, const double* exp_dw1,
    double* dst2, const double* src2, const double* exp_dw2,
    double a, const int M);

__global__ void ker_multi_exp_dw_two(
    cuDoubleComplex* dst1, const cuDoubleComplex* src1, const cuDoubleComplex* exp_dw1,
    cuDoubleComplex* dst2, const cuDoubleComplex* src2, const cuDoubleComplex* exp_dw2,
    double a, const int M);

__global__ void ker_multi_exp_dw_four(
    double* dst1, const double* src1, const double* exp_dw1,
    double* dst2, const double* src2, const double* exp_dw2,
    double* dst3, const double* src3, const double* exp_dw3,
    double* dst4, const double* src4, const double* exp_dw4,
    double a, const int M);

__global__ void ker_multi_exp_dw_four(
    cuDoubleComplex* dst1, const cuDoubleComplex* src1, const cuDoubleComplex* exp_dw1,
    cuDoubleComplex* dst2, const cuDoubleComplex* src2, const cuDoubleComplex* exp_dw2,
    cuDoubleComplex* dst3, const cuDoubleComplex* src3, const cuDoubleComplex* exp_dw3,
    cuDoubleComplex* dst4, const cuDoubleComplex* src4, const cuDoubleComplex* exp_dw4,
    double a, const int M);

__global__ void ker_complex_real_multi_bond_two(
    cuDoubleComplex* dst1, const double* boltz_bond1,
    cuDoubleComplex* dst2, const double* boltz_bond2,
    const int M);

__global__ void ker_complex_real_multi_bond_four(
    cuDoubleComplex* dst1, const double* boltz_bond1,
    cuDoubleComplex* dst2, const double* boltz_bond2,
    cuDoubleComplex* dst3, const double* boltz_bond3,
    cuDoubleComplex* dst4, const double* boltz_bond4,
    const int M);

// ============================================================
// Space group reduced basis kernels
// ============================================================

/**
 * @brief Expand reduced basis to full grid.
 *
 * Broadcasts values from reduced-basis points to total_grid points
 * using the full_to_reduced_map lookup table.
 *
 * @param dst Output full grid array (size: total_grid)
 * @param src Input reduced basis array (size: n_basis)
 * @param full_to_reduced_map Map from full grid index to reduced basis index
 * @param total_grid Number of full grid points
 */
__global__ void ker_expand_reduced_basis(
    double* dst, const double* src, const int* full_to_reduced_map, const int total_grid);

__global__ void ker_expand_reduced_basis(
    cuDoubleComplex* dst, const cuDoubleComplex* src, const int* full_to_reduced_map, const int total_grid);

/**
 * @brief Expand reduced basis and apply reduced multiplier in one pass.
 *
 * dst[i] = src[phys_to_reduced_map[i]] * reduced_multiplier[phys_to_reduced_map[i]]
 *
 * @param dst Output physical grid array (size: n_phys)
 * @param src Input reduced basis array (size: n_basis)
 * @param reduced_multiplier Reduced basis multiplier array (size: n_basis)
 * @param phys_to_reduced_map Map from physical index to reduced basis index
 * @param n_phys Number of physical grid points
 */
__global__ void ker_expand_reduced_basis_multi(
    double* dst, const double* src, const double* reduced_multiplier,
    const int* phys_to_reduced_map, const int n_phys);

/**
 * @brief Reduce full grid to reduced basis.
 *
 * Extracts values at reduced-basis points from full grid.
 *
 * @param dst Output reduced basis array (size: n_basis)
 * @param src Input full grid array (size: total_grid)
 * @param reduced_basis_indices Flat indices of reduced-basis points
 * @param n_basis Number of reduced-basis points
 */
__global__ void ker_reduce_to_basis(
    double* dst, const double* src, const int* reduced_basis_indices, const int n_basis);

__global__ void ker_reduce_to_basis(
    cuDoubleComplex* dst, const cuDoubleComplex* src, const int* reduced_basis_indices, const int n_basis);

/**
 * @brief Multiply physical-grid array by reduced-basis multiplier using mapping.
 *
 * @param dst Physical grid array (size: n_phys)
 * @param reduced_multiplier Reduced basis array (size: n_basis)
 * @param phys_to_reduced_map Map from physical index to reduced basis index
 * @param n_phys Number of physical grid points
 */
__global__ void ker_multi_map(
    double* dst, const double* reduced_multiplier, const int* phys_to_reduced_map, const int n_phys);

/**
 * @brief Gather from full grid and compute product: dst[i] = src[indices[i]]^2 * norm
 *
 * Used for solvent concentration computation in reduced basis.
 * More efficient than computing on full grid then reducing (O(N) vs O(M+N)).
 *
 * @param dst Output reduced basis array (size: n_basis)
 * @param src Input full grid array (size: total_grid)
 * @param reduced_basis_indices Flat indices of reduced-basis points
 * @param norm Normalization factor
 * @param n_basis Number of reduced-basis points
 */
__global__ void ker_multi_gather(
    double* dst, const double* src, const int* reduced_basis_indices, double norm, const int n_basis);

__global__ void ker_multi_gather(
    cuDoubleComplex* dst, const cuDoubleComplex* src, const int* reduced_basis_indices, cuDoubleComplex norm, const int n_basis);

/**
 * @brief Multiply array by integer weights: dst[i] = src[i] * weights[i]
 *
 * Used for orbit-weighted integration in reduced basis.
 *
 * @param dst Output array
 * @param src Input array
 * @param weights Integer weight array (e.g., orbit_counts)
 * @param n Number of elements
 */
__global__ void ker_multi_weight(
    double* dst, const double* src, const int* weights, const int n);

__global__ void ker_multi_weight(
    cuDoubleComplex* dst, const cuDoubleComplex* src, const int* weights, const int n);

//==============================================================================
// Owning CUDA allocations
//==============================================================================

/// Live counted allocations of one memory kind (CudaBuffer and the backend's
/// propagator and checkpoint arrays). Exact (unlike device-wide free memory,
/// which other processes and cuFFT's own plan memory change), so the tests can
/// check that destroyed objects released everything.
struct CudaBufferCounters
{
    std::atomic<long long> live_count{0};
    std::atomic<long long> live_bytes{0};
};

/// Test hook: when set to N >= 0, the N-th following counted allocation
/// (0-based; CudaBuffer and the backends' propagator and checkpoint arrays)
/// throws instead of allocating, once; -1 (the default) disables it.
/// tests/TestCudaDeviceBuffer.cu sweeps N to fail every allocation of a
/// construction in turn and checks that the counters return to their start.
inline std::atomic<long long>& cuda_buffer_fail_countdown()
{
    static std::atomic<long long> countdown{-1};
    return countdown;
}

/// cudaMalloc / cudaFree policy of CudaBuffer
struct CudaDeviceMemory
{
    static constexpr const char* kind = "device";
    static cudaError_t allocate(void** p, size_t bytes) { return cudaMalloc(p, bytes); }
    static cudaError_t release(void* p) { return cudaFree(p); }
    static CudaBufferCounters& counters() { static CudaBufferCounters c; return c; }
};

/// cudaMallocHost / cudaFreeHost (pinned host memory) policy of CudaBuffer
struct CudaPinnedMemory
{
    static constexpr const char* kind = "pinned host";
    static cudaError_t allocate(void** p, size_t bytes) { return cudaMallocHost(p, bytes); }
    static cudaError_t release(void* p) { return cudaFreeHost(p); }
    static CudaBufferCounters& counters() { static CudaBufferCounters c; return c; }
};

/**
 * @brief Allocate `bytes` bytes with a memory policy and count the allocation.
 *
 * Every owned CUDA allocation goes through this pair: CudaBuffer, and the
 * backend's propagator (allocate()) and checkpoint (allocate_checkpoint())
 * arrays, which are handed out as raw handles. So the live counters and the
 * failure hook cover all of them. A failure throws with the size and the
 * caller's file, line and function (`where`, the caller by default).
 */
template <typename Memory>
void* counted_cuda_allocate(size_t bytes, const std::source_location& where = std::source_location::current())
{
    if (bytes == 0)            // nothing to allocate, count or fail
        return nullptr;
    std::atomic<long long>& countdown = cuda_buffer_fail_countdown();
    if (countdown.load(std::memory_order_relaxed) >= 0 && countdown.fetch_sub(1) == 0)
        report_cuda_allocation_error(cudaErrorMemoryAllocation, bytes, Memory::kind, where, /*injected=*/true);
    void* p = nullptr;
    const cudaError_t code = Memory::allocate(&p, bytes);
    if (code != cudaSuccess)
        report_cuda_allocation_error(code, bytes, Memory::kind, where);
    Memory::counters().live_count.fetch_add(1, std::memory_order_relaxed);
    Memory::counters().live_bytes.fetch_add(static_cast<long long>(bytes), std::memory_order_relaxed);
    return p;
}

/// Release an allocation of counted_cuda_allocate() (no-op for nullptr); `bytes` as allocated.
template <typename Memory>
void counted_cuda_release(void* p, size_t bytes) noexcept
{
    if (p == nullptr)
        return;
    ignore_release_error(Memory::release(p));
    Memory::counters().live_count.fetch_sub(1, std::memory_order_relaxed);
    Memory::counters().live_bytes.fetch_sub(static_cast<long long>(bytes), std::memory_order_relaxed);
}

/**
 * @brief Owning device (or pinned host) allocation of `count` elements of T.
 *
 * Move-only: the allocation is released by the destructor, so a class that
 * keeps its device arrays in CudaBuffer members needs no cleanup code and
 * leaks nothing when its constructor throws. The raw pointer is lent through
 * get() or the implicit conversion, so kernel launches, cudaMemcpy and cuFFT
 * calls read as before; template functions that deduce the pointer type
 * (CUB, thrust) take get(), and so do tests for null (`buf.empty()` or
 * `buf.get() == nullptr`), because the second conversion (to void*), with
 * which GCC and Clang reject `delete buf`, makes `if (buf)` ambiguous too. A failed
 * allocation throws with its size and the caller's location. Errors of the
 * release call do not throw (see ignore_release_error()): a destructor must
 * not throw, and at process exit the runtime may already be gone.
 *
 * @code
 * CudaDeviceBuffer<double> d_q(M);            // allocated
 * ker_scale<<<B, T>>>(d_q, 2.0, M);           // lent as double*
 * cub::DeviceReduce::Sum(d_tmp.get(), bytes, d_q.get(), d_out.get(), M);
 * CudaPinnedBuffer<double> h_q(M);            // page-locked host memory
 * @endcode
 */
template <typename T, typename Memory = CudaDeviceMemory>
class CudaBuffer
{
public:
    CudaBuffer() = default;
    explicit CudaBuffer(size_t count, std::source_location where = std::source_location::current())
    {
        allocate(count, where);
    }
    ~CudaBuffer() { release(); }

    CudaBuffer(CudaBuffer&& other) noexcept : ptr_(other.ptr_), count_(other.count_)
    {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }
    CudaBuffer& operator=(CudaBuffer&& other) noexcept
    {
        if (this != &other)
        {
            release();
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    /// Allocate `count` elements, releasing a previous allocation first (count 0 only releases).
    /// A failure names the caller (`where`) in the exception.
    void allocate(size_t count, std::source_location where = std::source_location::current())
    {
        release();
        if (count == 0)
            return;
        ptr_ = static_cast<T*>(counted_cuda_allocate<Memory>(sizeof(T) * count, where));
        count_ = count;
    }
    /// Free the allocation (no-op when empty).
    void release() noexcept
    {
        counted_cuda_release<Memory>(ptr_, bytes());
        ptr_ = nullptr;
        count_ = 0;
    }

    T* get() const noexcept { return ptr_; }
    operator T*() const noexcept { return ptr_; }   ///< lends the pointer (no ownership transfer)
    /// Second pointer conversion, with which GCC and Clang (also as nvcc's host
    /// compiler) reject `delete buf` as ambiguous; that delete would free memory
    /// the buffer frees again. (The standard counts only conversions to object
    /// pointer types for delete, so a stricter compiler may accept it.) Passing
    /// a buffer to a void* parameter (cudaMemcpy, cudaMemset) picks this one.
    operator void*() const noexcept { return ptr_; }
    size_t size() const noexcept { return count_; }   ///< number of elements
    size_t bytes() const noexcept { return sizeof(T) * count_; }
    bool empty() const noexcept { return ptr_ == nullptr; }

private:
    T* ptr_ = nullptr;
    size_t count_ = 0;
};

template <typename T> using CudaDeviceBuffer = CudaBuffer<T, CudaDeviceMemory>;
template <typename T> using CudaPinnedBuffer = CudaBuffer<T, CudaPinnedMemory>;

// The implicit pointer conversion would let a buffer be freed by hand, which
// the buffer then frees again; these overloads make that a compile error.
template <typename T, typename M> cudaError_t cudaFree(const CudaBuffer<T, M>&) = delete;
template <typename T, typename M> cudaError_t cudaFreeHost(const CudaBuffer<T, M>&) = delete;
template <typename T, typename M> cudaError_t cudaFreeAsync(const CudaBuffer<T, M>&, cudaStream_t) = delete;

#endif
