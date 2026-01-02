/**
 * @file CudaAndersonMixing.h
 * @brief GPU implementation of Anderson Mixing algorithm.
 *
 * This header provides CudaAndersonMixing, the GPU-specific implementation
 * of Anderson Mixing for accelerating SCFT convergence. All field history
 * and computations are performed on the GPU.
 *
 * **GPU Optimizations:**
 *
 * - Field history stored in GPU device memory (CudaCircularBuffer)
 * - Dot products computed via CUB parallel reduction
 * - Two CUDA streams for overlapping kernel execution and memory copies
 *
 * **Memory Usage:**
 *
 * GPU memory: max_hist × n_var × sizeof(T) × 2 (for w_hist and w_deriv_hist)
 * For large grids, this can be significant.
 *
 * @see AndersonMixing for the algorithm description
 * @see CudaAndersonMixingReduceMemory for memory-saving version
 * @see CpuAndersonMixing for CPU implementation
 */

#ifndef CUDA_ANDERSON_MIXING_H_
#define CUDA_ANDERSON_MIXING_H_

#include "CircularBuffer.h"
#include "AndersonMixing.h"
#include "CudaCommon.h"
#include "CudaCircularBuffer.h"

/**
 * @class CudaAndersonMixing
 * @brief GPU-accelerated Anderson Mixing for SCFT iteration.
 *
 * Implements Anderson mixing entirely on the GPU for fast convergence
 * acceleration. Uses circular buffers in device memory for history.
 *
 * @tparam T Numeric type (double or std::complex<double>)
 *
 * **GPU Memory Layout:**
 *
 * - d_cb_w_hist: History of field values in device memory
 * - d_cb_w_deriv_hist: History of residuals in device memory
 * - cb_w_deriv_dots: Cached dot products (on CPU for small size)
 *
 * **CUDA Streams:**
 *
 * Uses two streams for overlapping:
 * - streams[0]: Kernel execution
 * - streams[1]: Memory transfers
 *
 * @example
 * @code
 * CudaAndersonMixing<double> am(n_grid, 20, 1e-1, 0.01, 0.1);
 *
 * // In SCFT loop
 * am.calculate_new_fields(w_new, w_current, w_deriv, old_err, err);
 * @endcode
 */
template <typename T>
class CudaAndersonMixing : public AndersonMixing<T>
{
private:
    cudaStream_t streams[2];  ///< CUDA streams [kernel, memcpy]

    CudaCircularBuffer<T> *d_cb_w_hist;        ///< Field history on GPU
    CudaCircularBuffer<T> *d_cb_w_deriv_hist;  ///< Residual history on GPU
    CircularBuffer<T> *cb_w_deriv_dots;        ///< Cached dot products (CPU)
    T *w_deriv_dots;                            ///< Current residual dots

    /// @name Least-squares solve arrays (on CPU)
    /// @{
    T **u_nm;   ///< Matrix for normal equations
    T *v_n;     ///< RHS vector
    T *a_n;     ///< Solution coefficients
    /// @}

    /// @name Temporary GPU arrays
    /// @{
    CuDeviceData<T> *d_w_current;  ///< Current field on GPU
    CuDeviceData<T> *d_w_new;      ///< New field on GPU
    CuDeviceData<T> *d_w_deriv;    ///< Residual on GPU
    CuDeviceData<T> *d_sum;        ///< Reduction temp
    /// @}

    /// @name CUB reduction storage
    /// @{
    size_t temp_storage_bytes = 0;
    CuDeviceData<T> *d_temp_storage = nullptr;
    CuDeviceData<T> *d_sum_out;
    /// @}

    void print_array(int n, T *a);  ///< Debug helper

public:
    /**
     * @brief Construct GPU Anderson mixing optimizer.
     *
     * @param n_var      Number of field variables (n_grid)
     * @param max_hist   Maximum history length
     * @param start_error Error threshold for switching to Anderson mixing
     * @param mix_min    Minimum mixing parameter
     * @param mix_init   Initial mixing parameter
     */
    CudaAndersonMixing(int n_var, int max_hist,
        double start_error, double mix_min, double mix_init);

    /**
     * @brief Destructor. Frees GPU memory and streams.
     */
    ~CudaAndersonMixing();

    /**
     * @brief Reset history and iteration counter.
     */
    void reset_count() override;

    /**
     * @brief Compute next iterate using Anderson mixing.
     *
     * @param w_new         Output: new field values (device pointer)
     * @param w_current     Input: current field values (device pointer)
     * @param w_deriv       Input: residual (device pointer)
     * @param old_error_level Previous error level
     * @param error_level   Current error level
     */
    void calculate_new_fields(
        T *w_new, T *w_current, T *w_deriv,
        double old_error_level, double error_level) override;

};
#endif
