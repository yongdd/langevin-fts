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

#include <complex>
#include <map>
#include <cufft.h>

#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>
#include "Exception.h"

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

    /**
     * @brief Get number of thread blocks.
     */
    int get_n_blocks();

    /**
     * @brief Get threads per block.
     */
    int get_n_threads();
    // int get_n_gpus();

    /**
     * @brief Set number of thread blocks.
     */
    void set_n_blocks(int n_blocks);

    /**
     * @brief Set threads per block.
     */
    void set_n_threads(int n_threads);

    /**
     * @brief Set GPU device index.
     */
    void set_idx(int process_idx);
};

/**
 * @def gpu_error_check
 * @brief Macro for CUDA error checking with source location.
 *
 * Wraps CUDA API calls and throws an exception on error.
 *
 * @example
 * @code
 * gpu_error_check(cudaMalloc(&ptr, size));
 * gpu_error_check(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
 * @endcode
 */
#define gpu_error_check(code) throw_on_cuda_error((code), __FILE__, __LINE__, __func__);

/**
 * @brief Throw exception on CUDA error.
 * @param code CUDA error code
 * @param file Source file name
 * @param line Line number
 * @param func Function name
 * @throws Exception if code != cudaSuccess
 */
void throw_on_cuda_error(cudaError_t code, const char *file, int line, const char *func);

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
__global__ void ker_multi_stress_combined(double* dst, const double* src, const double* basis1, const double* basis2, double factor, double a, const int M);
__global__ void ker_multi_stress_combined_3d(double* dst, const double* src, const double* basis1, const double* cross1, const double* cross2, double factor1, double factor2, double a, const int M);

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

// ETDRK4 kernels for complex coefficients (periodic BC)
__global__ void ker_etdrk4_stage_a(
    cuDoubleComplex* dst, const cuDoubleComplex* q_hat, const cuDoubleComplex* N_hat,
    const double* E2, const double* alpha, const int M);

__global__ void ker_etdrk4_stage_c(
    cuDoubleComplex* dst, const cuDoubleComplex* a_hat, const cuDoubleComplex* N_b_hat,
    const cuDoubleComplex* N_n_hat, const double* E2, const double* alpha, const int M);

__global__ void ker_etdrk4_final(
    cuDoubleComplex* dst, const cuDoubleComplex* q_hat,
    const cuDoubleComplex* N_n_hat, const cuDoubleComplex* N_a_hat,
    const cuDoubleComplex* N_b_hat, const cuDoubleComplex* N_c_hat,
    const double* E, const double* f1, const double* f2, const double* f3, const int M);

// ETDRK4 kernels for real coefficients (non-periodic BC - DCT/DST)
__global__ void ker_etdrk4_stage_a_real(
    double* dst, const double* q_hat, const double* N_hat,
    const double* E2, const double* alpha, const int M);

__global__ void ker_etdrk4_stage_c_real(
    double* dst, const double* a_hat, const double* N_b_hat,
    const double* N_n_hat, const double* E2, const double* alpha, const int M);

__global__ void ker_etdrk4_final_real(
    double* dst, const double* q_hat,
    const double* N_n_hat, const double* N_a_hat,
    const double* N_b_hat, const double* N_c_hat,
    const double* E, const double* f1, const double* f2, const double* f3, const int M);

// Krogstad ETDRK4 kernels for complex coefficients (periodic BC)
__global__ void ker_etdrk4_krogstad_stage_b(
    cuDoubleComplex* dst, const cuDoubleComplex* a_hat,
    const cuDoubleComplex* N_a_hat, const cuDoubleComplex* N_n_hat,
    const double* phi2_half, const int M);

__global__ void ker_etdrk4_krogstad_stage_c(
    cuDoubleComplex* dst, const cuDoubleComplex* q_hat,
    const cuDoubleComplex* N_n_hat, const cuDoubleComplex* N_b_hat,
    const double* E, const double* phi1, const double* phi2, const int M);

__global__ void ker_etdrk4_krogstad_final(
    cuDoubleComplex* dst, const cuDoubleComplex* c_hat,
    const cuDoubleComplex* N_n_hat, const cuDoubleComplex* N_a_hat,
    const cuDoubleComplex* N_b_hat, const cuDoubleComplex* N_c_hat,
    const double* phi2, const double* phi3, const int M);

// Krogstad ETDRK4 kernels for real coefficients (non-periodic BC - DCT/DST)
__global__ void ker_etdrk4_krogstad_stage_b_real(
    double* dst, const double* a_hat,
    const double* N_a_hat, const double* N_n_hat,
    const double* phi2_half, const int M);

__global__ void ker_etdrk4_krogstad_stage_c_real(
    double* dst, const double* q_hat,
    const double* N_n_hat, const double* N_b_hat,
    const double* E, const double* phi1, const double* phi2, const int M);

__global__ void ker_etdrk4_krogstad_final_real(
    double* dst, const double* c_hat,
    const double* N_n_hat, const double* N_a_hat,
    const double* N_b_hat, const double* N_c_hat,
    const double* phi2, const double* phi3, const int M);

// ============================================================
// Space group reduced basis kernels
// ============================================================

/**
 * @brief Expand reduced basis to full grid.
 *
 * Broadcasts values from n_irreducible points to total_grid points
 * using the full_to_reduced_map lookup table.
 *
 * @param dst Output full grid array (size: total_grid)
 * @param src Input reduced basis array (size: n_irreducible)
 * @param full_to_reduced_map Map from full grid index to reduced basis index
 * @param total_grid Number of full grid points
 */
__global__ void ker_expand_reduced_basis(
    double* dst, const double* src, const int* full_to_reduced_map, const int total_grid);

__global__ void ker_expand_reduced_basis(
    cuDoubleComplex* dst, const cuDoubleComplex* src, const int* full_to_reduced_map, const int total_grid);

/**
 * @brief Reduce full grid to reduced basis.
 *
 * Extracts values at irreducible mesh points from full grid.
 *
 * @param dst Output reduced basis array (size: n_irreducible)
 * @param src Input full grid array (size: total_grid)
 * @param reduced_basis_indices Flat indices of irreducible points
 * @param n_irreducible Number of irreducible points
 */
__global__ void ker_reduce_to_basis(
    double* dst, const double* src, const int* reduced_basis_indices, const int n_irreducible);

__global__ void ker_reduce_to_basis(
    cuDoubleComplex* dst, const cuDoubleComplex* src, const int* reduced_basis_indices, const int n_irreducible);

/**
 * @brief Gather from full grid and compute product: dst[i] = src[indices[i]]^2 * norm
 *
 * Used for solvent concentration computation in reduced basis.
 * More efficient than computing on full grid then reducing (O(N) vs O(M+N)).
 *
 * @param dst Output reduced basis array (size: n_irreducible)
 * @param src Input full grid array (size: total_grid)
 * @param reduced_basis_indices Flat indices of irreducible points
 * @param norm Normalization factor
 * @param n_irreducible Number of irreducible points
 */
__global__ void ker_multi_gather(
    double* dst, const double* src, const int* reduced_basis_indices, double norm, const int n_irreducible);

__global__ void ker_multi_gather(
    cuDoubleComplex* dst, const cuDoubleComplex* src, const int* reduced_basis_indices, cuDoubleComplex norm, const int n_irreducible);

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

#endif
