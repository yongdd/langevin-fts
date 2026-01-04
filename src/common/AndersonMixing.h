/**
 * @file AndersonMixing.h
 * @brief Anderson Mixing algorithm for accelerating iterative SCFT convergence.
 *
 * This header provides the AndersonMixing class, which implements the Anderson
 * acceleration (Anderson mixing) algorithm for accelerating the convergence of
 * fixed-point iterations in Self-Consistent Field Theory (SCFT) calculations.
 *
 * Anderson mixing uses a history of previous iterations to extrapolate toward
 * the fixed point, significantly reducing the number of iterations required
 * for convergence compared to simple mixing schemes.
 *
 * **Algorithm Overview:**
 *
 * Given the current iterate w_k and its residual r_k = G(w_k) - w_k, Anderson
 * mixing constructs the next iterate as a linear combination of previous
 * iterates that minimizes the residual norm:
 *
 * w_{k+1} = sum_i alpha_i * (w_{k-i} + r_{k-i})
 *
 * where the coefficients alpha_i are determined by a least-squares problem.
 *
 * @see CpuAndersonMixing, CudaAndersonMixing for platform-specific implementations
 *
 * @note The implementation follows the formulation in D. G. Anderson,
 *       "Iterative Procedures for Nonlinear Integral Equations",
 *       J. ACM 12, 547 (1965).
 *
 * @example
 * @code
 * // Create Anderson Mixing with 20 history vectors
 * int n_var = 32 * 32 * 32;  // Number of field variables
 * int max_hist = 20;          // Maximum history length
 * double start_error = 1e-1;  // Use simple mixing until error < start_error
 * double mix_min = 0.1;       // Minimum mixing parameter
 * double mix_init = 0.1;      // Initial mixing parameter
 *
 * AndersonMixing<double>* am = factory->create_anderson_mixing(
 *     n_var, max_hist, start_error, mix_min, mix_init);
 *
 * // In SCFT iteration loop
 * while (error > tolerance) {
 *     // Compute functional derivative
 *     compute_concentrations(w, phi);
 *     compute_residual(w, phi, w_deriv);  // w_deriv = -dH/dw
 *
 *     // Anderson mixing update
 *     am->calculate_new_fields(w_new, w, w_deriv, old_error, error);
 *
 *     std::swap(w, w_new);
 *     old_error = error;
 *     error = compute_error(w_deriv);
 * }
 * @endcode
 */

#ifndef ANDERSON_MIXING_H_
#define ANDERSON_MIXING_H_

#include <cassert>
#include <iostream>

#include "Exception.h"

/**
 * @class AndersonMixing
 * @brief Abstract base class for Anderson Mixing iteration acceleration.
 *
 * Anderson mixing is a quasi-Newton method that accelerates the convergence
 * of fixed-point iterations by using information from previous iterations.
 * It is particularly effective for SCFT calculations where simple iteration
 * converges slowly or not at all.
 *
 * @tparam T Numeric type for field values (double or std::complex<double>)
 *
 * **Mixing Strategy:**
 *
 * - When error > start_error: Uses simple mixing with parameter mix_init
 * - When error < start_error: Uses Anderson mixing with history
 * - The mixing parameter is adaptively adjusted based on error reduction
 * - If error increases, mixing parameter is reduced toward mix_min
 *
 * **History Management:**
 *
 * The algorithm stores up to max_hist previous field/residual pairs.
 * When history is full, the oldest entry is discarded (circular buffer).
 * Call reset_count() to clear history when starting a new SCFT run.
 *
 * **Performance Notes:**
 *
 * - Typical max_hist values: 10-30 (higher for difficult convergence)
 * - Memory usage: O(max_hist * n_var) for storing history
 * - Each iteration requires solving a small (max_hist x max_hist) linear system
 *
 * @see SCFT::run for usage in SCFT iterations
 */
template <typename T>
class AndersonMixing
{
protected:
    int n_var;        ///< Number of field variables (total grid points * field components)
    int max_hist;     ///< Maximum number of history vectors to store
    int n_anderson;   ///< Current number of stored history vectors

    double start_error;  ///< Error threshold to switch from simple to Anderson mixing
    double mix_min;      ///< Minimum allowed mixing parameter (for stability)
    double mix;          ///< Current mixing parameter
    double mix_init;     ///< Initial mixing parameter

    /**
     * @brief Solve the least-squares problem for Anderson coefficients.
     *
     * Finds coefficients a that minimize ||U*a - v||^2 where U is the matrix
     * of residual differences and v is the current residual.
     *
     * @param u Matrix of residual differences, size (n x n_var), stored as u[i][j]
     * @param v Current residual vector, length n_var
     * @param a Output: optimal coefficients, length n
     * @param n Number of history vectors to use
     *
     * @note Uses normal equations: (U^T U) a = U^T v
     */
    void find_an(T **u, T *v, T *a, int n);

public:
    /**
     * @brief Construct an AndersonMixing instance.
     *
     * @param n_var      Number of field variables to optimize
     * @param max_hist   Maximum history length (typical: 10-30)
     * @param start_error Error threshold for Anderson vs simple mixing (typical: 1e-1)
     * @param mix_min    Minimum mixing parameter (typical: 0.01-0.1)
     * @param mix_init   Initial mixing parameter (typical: 0.1-0.5)
     *
     * @note mix_min <= mix_init is required for correct behavior
     *
     * @example
     * @code
     * // Conservative mixing for difficult problems
     * AndersonMixing<double> am(n_var, 30, 1e-2, 0.01, 0.01);
     *
     * // Aggressive mixing for well-behaved problems
     * AndersonMixing<double> am(n_var, 20, 1e-1, 0.1, 0.1);
     * @endcode
     */
    AndersonMixing(int n_var, int max_hist, double start_error, double mix_min, double mix_init);

    /**
     * @brief Virtual destructor.
     */
    virtual ~AndersonMixing(){};

    /**
     * @brief Reset iteration counter and clear history.
     *
     * Call this method before starting a new SCFT run to clear the history
     * from previous runs. Also resets the mixing parameter to mix_init.
     *
     * @example
     * @code
     * // Before each new SCFT calculation
     * am->reset_count();
     * for (int iter = 0; iter < max_iter; iter++) {
     *     // ... SCFT iteration ...
     * }
     * @endcode
     */
    virtual void reset_count(){};

    /**
     * @brief Get number of field variables.
     * @return n_var value passed to constructor
     */
    int get_n_var() const { return n_var;};

    /**
     * @brief Compute the next iterate using Anderson mixing.
     *
     * Updates w_new based on the current fields w_current and the functional
     * derivative w_deriv. The update formula depends on the current error level:
     *
     * - Simple mixing: w_new = w_current + mix * w_deriv
     * - Anderson mixing: Linear combination of history minimizing residual
     *
     * @param w_new         Output: new field values, length n_var
     * @param w_current     Input: current field values, length n_var
     * @param w_deriv       Input: functional derivative -dH/dw, length n_var
     * @param old_error_level Previous iteration's error level
     * @param error_level   Current iteration's error level
     *
     * @note w_deriv should be the negative functional derivative so that
     *       gradient descent would be w_new = w_current + step * w_deriv.
     *
     * @note The method adaptively adjusts the mixing parameter based on
     *       error reduction: if error increases, mixing is reduced.
     *
     * @example
     * @code
     * // Typical SCFT iteration
     * double error = 1.0, old_error = 1.0;
     * while (error > 1e-8) {
     *     compute_concentrations(w_current, phi);
     *     compute_residual(w_current, phi, w_deriv);
     *     error = compute_error_norm(w_deriv);
     *
     *     am->calculate_new_fields(w_new, w_current, w_deriv, old_error, error);
     *
     *     std::swap(w_current, w_new);
     *     old_error = error;
     * }
     * @endcode
     */
    virtual void calculate_new_fields(
        T *w_new, T *w_current, T *w_deriv,
        double old_error_level, double error_level)=0;
};
#endif
