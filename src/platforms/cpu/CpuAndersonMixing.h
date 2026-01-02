/**
 * @file CpuAndersonMixing.h
 * @brief CPU implementation of Anderson Mixing algorithm.
 *
 * This header provides CpuAndersonMixing, the CPU-specific implementation
 * of the AndersonMixing iteration accelerator. It uses circular buffers
 * to store field history and performs the least-squares optimization
 * on the CPU.
 *
 * @see AndersonMixing for the interface and algorithm description
 * @see CudaAndersonMixing for GPU implementation
 */

#ifndef CPU_ANDERSON_MIXING_H_
#define CPU_ANDERSON_MIXING_H_

#include "ComputationBox.h"
#include "CircularBuffer.h"
#include "AndersonMixing.h"

/**
 * @class CpuAndersonMixing
 * @brief CPU-specific Anderson Mixing implementation.
 *
 * Implements the Anderson mixing algorithm using CPU memory and operations.
 * Uses CircularBuffer to efficiently manage the history of field values
 * and residuals.
 *
 * @tparam T Numeric type (double or std::complex<double>)
 *
 * **Memory Layout:**
 *
 * - cb_w_hist: History of field values w_k
 * - cb_w_deriv_hist: History of residuals (functional derivatives)
 * - u_nm, v_n, a_n: Temporary arrays for least-squares solve
 *
 * @note Uses OpenMP for parallel dot products when available.
 */
template <typename T>
class CpuAndersonMixing : public AndersonMixing<T>
{
private:
    CircularBuffer<T> *cb_w_hist;        ///< History of field values
    CircularBuffer<T> *cb_w_deriv_hist;  ///< History of residuals
    CircularBuffer<T> *cb_w_deriv_dots;  ///< Cached dot products
    T *w_deriv_dots;                     ///< Current residual dot products

    T **u_nm;  ///< Matrix for normal equations (max_hist x max_hist)
    T *v_n;    ///< RHS vector for normal equations
    T *a_n;    ///< Solution coefficients

    /**
     * @brief Compute dot product of two vectors.
     * @param a First vector (length n_var)
     * @param b Second vector (length n_var)
     * @return Dot product sum_i a[i] * b[i]
     */
    T dot_product(T *a, T *b);

    /**
     * @brief Debug helper to print array contents.
     * @param n Array length
     * @param a Array pointer
     */
    void print_array(int n, T *a);

public:
    /**
     * @brief Construct CPU Anderson mixing optimizer.
     *
     * @param n_var      Number of field variables
     * @param max_hist   Maximum history length
     * @param start_error Error threshold for switching to Anderson mixing
     * @param mix_min    Minimum mixing parameter
     * @param mix_init   Initial mixing parameter
     */
    CpuAndersonMixing(int n_var, int max_hist,
        double start_error, double mix_min, double mix_init);

    /**
     * @brief Destructor. Frees all allocated memory.
     */
    ~CpuAndersonMixing();

    /**
     * @brief Reset history and iteration counter.
     *
     * Clears circular buffers and resets mixing parameter to initial value.
     */
    void reset_count() override;

    /**
     * @brief Compute next iterate using Anderson mixing.
     *
     * @param w_new         Output: new field values
     * @param w_current     Input: current field values
     * @param w_deriv       Input: functional derivative
     * @param old_error_level Previous error level
     * @param error_level   Current error level
     */
    void calculate_new_fields(
        T *w_new, T *w_current, T *w_deriv,
        double old_error_level, double error_level) override;
};
#endif
