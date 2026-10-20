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
#include <string>
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
 * Manages GPU settings including:
 * - Number of thread blocks for kernel launches
 * - Number of threads per block
 * - GPU device selection
 *
 * Uses Scott Meyer's singleton pattern for thread-safe initialization.
 *
 * **Typical Configuration:**
 *
 * - n_blocks: ceil(n_grid / n_threads)
 * - n_threads: 256-512 (depends on GPU architecture)
 *
 * @example
 * @code
 * CudaCommon& cuda = CudaCommon::get_instance();
 * cuda.set(128, 256, 0);  // 128 blocks, 256 threads, GPU 0
 *
 * // Launch kernel
 * my_kernel<<<cuda.get_n_blocks(), cuda.get_n_threads()>>>(...);
 * @endcode
 */
class CudaCommon
{
private:
    int n_blocks;
    int n_threads;

    /// Throw unless n_threads <= 1024 and 1 <= n_blocks * n_threads <= MAX_GRID_THREADS
    static void check_launch(int n_blocks, int n_threads);

    // int n_gpus;

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

    /**
     * @brief Configure GPU settings.
     * @param n_blocks    Number of thread blocks
     * @param n_threads   Threads per block
     * @param process_idx GPU device index
     */
    void set(int n_blocks, int n_threads, int process_idx);

    /// Get number of thread blocks.
    int get_n_blocks();

    /// Get threads per block.
    int get_n_threads();
    // int get_n_gpus();

    /// Set number of thread blocks.
    void set_n_blocks(int n_blocks);

    /// Set threads per block.
    void set_n_threads(int n_threads);

    /// Set GPU device index.
    void set_idx(int process_idx);
};

/**
 * @def gpu_error_check
 * @brief Macro for CUDA error checking with source location.
 *
 * Wraps CUDA API calls and throws an exception on error. Every call that
 * returns an error code is checked, and every kernel launch is followed by
 * gpu_error_check(cudaGetLastError()), as the CUDA C++ Best Practices Guide
 * (2.3) recommends: the check is a host-side read that does not synchronize,
 * and cudaGetLastError() clears the slot, so an error is reported at the
 * launch that caused it and not again by a later, unrelated check. Errors of
 * a running kernel (e.g. an illegal address) surface at the next synchronizing
 * call; run with CUDA_LAUNCH_BLOCKING=1 (or compute-sanitizer) to see them at
 * their launch. Release calls (cudaFree, *Destroy) are not checked: they run
 * in destructors, which must not throw.
 *
 * @example
 * @code
 * gpu_error_check(cudaMalloc(&ptr, size));
 * gpu_error_check(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
 * ker_scale<<<blocks, threads, 0, stream>>>(d_x, 2.0, n);
 * gpu_error_check(cudaGetLastError());
 * @endcode
 */
#define gpu_error_check(code) throw_on_cuda_error((code), __FILE__, __LINE__, __func__)

/// Throw the exception for a failed CUDA call (out of line: only the error path pays).
[[noreturn]] void report_cuda_error(cudaError_t code, const char *file, int line, const char *func);

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
 * @def cufft_error_check
 * @brief Macro for cuFFT error checking with source location.
 *
 * Wraps cuFFT API calls (plan creation, cufftSetStream, cufftExec*) and
 * throws an exception on any result other than CUFFT_SUCCESS.
 *
 * @example
 * @code
 * cufft_error_check(cufftPlanMany(&plan, ...));
 * cufft_error_check(cufftExecD2Z(plan, in, out));
 * @endcode
 */
#define cufft_error_check(code) throw_on_cufft_error((code), __FILE__, __LINE__, __func__)

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

/// Live CudaBuffer allocations of one memory kind. Exact (unlike device-wide
/// free memory, which other processes and cuFFT's own plan memory change), so
/// the tests can check that destroyed objects released everything.
struct CudaBufferCounters
{
    std::atomic<long long> live_count{0};
    std::atomic<long long> live_bytes{0};
};

/// Test hook: when set to N >= 0, the N-th following CudaBuffer allocation
/// (0-based) throws instead of allocating, once; -1 (the default) disables it.
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
    static cudaError_t allocate(void** p, size_t bytes) { return cudaMalloc(p, bytes); }
    static void release(void* p) { cudaFree(p); }
    static CudaBufferCounters& counters() { static CudaBufferCounters c; return c; }
};

/// cudaMallocHost / cudaFreeHost (pinned host memory) policy of CudaBuffer
struct CudaPinnedMemory
{
    static cudaError_t allocate(void** p, size_t bytes) { return cudaMallocHost(p, bytes); }
    static void release(void* p) { cudaFreeHost(p); }
    static CudaBufferCounters& counters() { static CudaBufferCounters c; return c; }
};

/**
 * @brief Owning device (or pinned host) allocation of `count` elements of T.
 *
 * Move-only: the allocation is released by the destructor, so a class that
 * keeps its device arrays in CudaBuffer members needs no cleanup code and
 * leaks nothing when its constructor throws. The raw pointer is lent through
 * get() or the implicit conversion, so kernel launches, cudaMemcpy and cuFFT
 * calls read as before; template functions that deduce the pointer type
 * (CUB, thrust) take get(). Errors of the release call are ignored on
 * purpose: a destructor must not throw, and at process exit the runtime may
 * already be gone.
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
    explicit CudaBuffer(size_t count) { allocate(count); }
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
    void allocate(size_t count)
    {
        release();
        if (count == 0)
            return;
        std::atomic<long long>& countdown = cuda_buffer_fail_countdown();
        if (countdown.load(std::memory_order_relaxed) >= 0 && countdown.fetch_sub(1) == 0)
            throw_with_line_number("CudaBuffer: injected allocation failure (test hook)");
        void* p = nullptr;
        gpu_error_check(Memory::allocate(&p, sizeof(T) * count));
        ptr_ = static_cast<T*>(p);
        count_ = count;
        Memory::counters().live_count.fetch_add(1, std::memory_order_relaxed);
        Memory::counters().live_bytes.fetch_add(static_cast<long long>(bytes()), std::memory_order_relaxed);
    }
    /// Free the allocation (no-op when empty).
    void release() noexcept
    {
        if (ptr_ != nullptr)
        {
            Memory::release(ptr_);
            Memory::counters().live_count.fetch_sub(1, std::memory_order_relaxed);
            Memory::counters().live_bytes.fetch_sub(static_cast<long long>(bytes()), std::memory_order_relaxed);
        }
        ptr_ = nullptr;
        count_ = 0;
    }

    T* get() const noexcept { return ptr_; }
    operator T*() const noexcept { return ptr_; }   ///< lends the pointer (no ownership transfer)
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

#endif
