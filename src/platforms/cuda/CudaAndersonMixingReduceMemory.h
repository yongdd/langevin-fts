/**
 * @file CudaAndersonMixingReduceMemory.h
 * @brief Memory-efficient GPU Anderson Mixing using pinned host memory.
 *
 * This header provides CudaAndersonMixingReduceMemory, a variant of
 * CudaAndersonMixing that stores field history in pinned host memory
 * instead of GPU device memory to reduce GPU memory usage.
 *
 * **Memory Saving Strategy:**
 *
 * - History buffers stored in pinned (page-locked) host memory
 * - Only 2 GPU buffers for current processing
 * - Async transfers hide memory copy latency
 *
 * **Trade-offs:**
 *
 * - Lower GPU memory usage (good for large grids)
 * - Slightly slower due to PCIe transfers
 * - Same convergence behavior as CudaAndersonMixing
 *
 * @see AndersonMixing for algorithm description
 * @see CudaAndersonMixing for full GPU version
 * @see PinnedCircularBuffer for pinned memory buffer
 */

#ifndef CUDA_ANDERSON_MIXING_REDUCE_MEMORY_H_
#define CUDA_ANDERSON_MIXING_REDUCE_MEMORY_H_

#include "CircularBuffer.h"
#include "ComputationBox.h"
#include "AndersonMixing.h"
#include "CudaCommon.h"
#include "PinnedCircularBuffer.h"

/**
 * @class CudaAndersonMixingReduceMemory
 * @brief Memory-efficient Anderson Mixing using pinned host memory.
 *
 * Trades GPU memory for PCIe bandwidth by storing history in pinned
 * host memory. Uses async transfers to minimize performance impact.
 *
 * @tparam T Numeric type (double or std::complex<double>)
 *
 * **Memory Layout:**
 *
 * - Host (pinned): Field and residual history (max_hist × n_var each)
 * - Device: Only 4 temporary arrays (d_w_hist1, d_w_hist2, etc.)
 *
 * **Data Flow:**
 *
 * 1. Copy needed history entries from pinned to GPU
 * 2. Compute dot products on GPU
 * 3. Build and solve normal equations on CPU
 * 4. Compute new field on GPU
 * 5. Store new values in pinned memory
 *
 * @note Use when GPU memory is limited and grid size is large.
 */
template <typename T>
class CudaAndersonMixingReduceMemory : public AndersonMixing<T>
{
private:
    /// @name Pinned Memory History
    /// @{
    PinnedCircularBuffer<T> *pinned_cb_w_hist;        ///< Field history (pinned)
    PinnedCircularBuffer<T> *pinned_cb_w_deriv_hist;  ///< Residual history (pinned)
    /// @}

    CircularBuffer<T> *cb_w_deriv_dots;  ///< Cached dot products
    T *w_deriv_dots;

    /// @name Least-squares arrays
    /// @{
    T **u_nm;
    T *v_n;
    T *a_n;
    /// @}

    /// @name Temporary GPU arrays
    /// @{
    CuDeviceData<T> *d_w_new;
    CuDeviceData<T> *d_w_deriv;
    CuDeviceData<T> *d_sum;
    /// @}

    /// @name GPU buffers for history access (2 pairs)
    /// @{
    CuDeviceData<T> *d_w_hist1;        ///< First history entry on GPU
    CuDeviceData<T> *d_w_hist2;        ///< Second history entry on GPU
    CuDeviceData<T> *d_w_deriv_hist1;  ///< First residual history on GPU
    CuDeviceData<T> *d_w_deriv_hist2;  ///< Second residual history on GPU
    /// @}

    /// @name CUB reduction storage
    /// @{
    size_t temp_storage_bytes = 0;
    CuDeviceData<T> *d_temp_storage = nullptr;
    CuDeviceData<T> *d_sum_out;
    /// @}

    void print_array(int n, T *a);

public:
    /**
     * @brief Construct memory-efficient Anderson mixing.
     *
     * @param n_var      Number of field variables
     * @param max_hist   Maximum history length
     * @param start_error Error threshold
     * @param mix_min    Minimum mixing parameter
     * @param mix_init   Initial mixing parameter
     */
    CudaAndersonMixingReduceMemory(int n_var, int max_hist,
        double start_error, double mix_min, double mix_init);

    /**
     * @brief Destructor. Frees GPU and pinned memory.
     */
    ~CudaAndersonMixingReduceMemory();

    /**
     * @brief Reset history.
     */
    void reset_count() override;

    /**
     * @brief Compute new field using Anderson mixing.
     *
     * @param w_new         Output: new field (device pointer)
     * @param w_current     Input: current field (device pointer)
     * @param w_deriv       Input: residual (device pointer)
     * @param old_error_level Previous error
     * @param error_level   Current error
     */
    void calculate_new_fields(
        T *w_new, T *w_current, T *w_deriv,
        double old_error_level, double error_level) override;

};
#endif
