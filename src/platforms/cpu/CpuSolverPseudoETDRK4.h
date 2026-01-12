/**
 * @file CpuSolverPseudoETDRK4.h
 * @brief ETDRK4 pseudo-spectral solver for continuous chain model on CPU.
 *
 * This header provides CpuSolverPseudoETDRK4, which implements the ETDRK4
 * (Exponential Time Differencing Runge-Kutta 4th order) method for solving
 * the modified diffusion equation with continuous Gaussian chains.
 *
 * @warning **Reduced Convergence Order for Polymer MDE**
 *
 * ETDRK4 achieves only O(ds) convergence (1st-order) instead of O(ds^4) when
 * applied to the polymer modified diffusion equation. This is because N(q) = -w*q
 * is linear in q, so intermediate stage evaluations provide no additional
 * information for the Runge-Kutta weighting to exploit.
 *
 * **Recommendation:** Use CpuSolverPseudoRQM4 instead. RQM4 is 2x faster and
 * achieves true 4th-order convergence.
 *
 * **ETDRK4 Algorithm (Cox & Matthews 2002):**
 *
 * For the equation: dq/ds = L*q + N(q) where:
 * - L = (b^2/6)*nabla^2 (linear diffusion, eigenvalue c = -k^2*b^2/6)
 * - N(q) = -w*q (nonlinear/potential term)
 *
 * Stages:
 *   a = E2*q_hat + alpha*N_hat_n
 *   b = E2*q_hat + alpha*N_hat_a
 *   c = E2*a_hat + alpha*(2*N_hat_b - N_hat_n)
 *   q_hat_{n+1} = E*q_hat + f1*N_hat_n + f2*(N_hat_a + N_hat_b) + f3*N_hat_c
 *
 * **Numerical Stability:**
 *
 * Uses Kassam-Trefethen (2005) contour integral method for coefficient
 * computation, avoiding catastrophic cancellation for small eigenvalues.
 *
 * @see ETDRK4Coefficients for coefficient computation
 * @see CpuSolverPseudoRQM4 for RQM4 alternative
 */

#ifndef CPU_SOLVER_PSEUDO_ETDRK4_H_
#define CPU_SOLVER_PSEUDO_ETDRK4_H_

#include <string>
#include <vector>
#include <map>
#include <memory>

#include "Exception.h"
#include "Molecules.h"
#include "ComputationBox.h"
#include "CpuSolverPseudoBase.h"
#include "ETDRK4Coefficients.h"

/**
 * @class CpuSolverPseudoETDRK4
 * @brief CPU pseudo-spectral solver using ETDRK4 time integration.
 *
 * Implements the ETDRK4 method for advancing chain propagators with
 * 4th-order accuracy and L-stability. Supports all boundary conditions
 * (periodic, reflecting, absorbing).
 *
 * @tparam T Numeric type (double or std::complex<double>)
 *
 * **Memory Usage:**
 *
 * Per monomer type:
 * - w_field: Raw potential field (n_grid)
 * - ETDRK4 coefficients: E, E2, alpha, f1, f2, f3 (n_complex each)
 *
 * **Performance:**
 *
 * Each propagator step requires:
 * - 8 FFTs (4 stages x 2 transforms)
 * - O(n_grid) element-wise operations
 *
 * @example
 * @code
 * // Create solver
 * CpuSolverPseudoETDRK4<double> solver(cb, molecules);
 *
 * // Update for new fields
 * solver.update_dw(w_fields);
 *
 * // Advance propagator
 * solver.advance_propagator(q_in, q_out, "A", nullptr);
 * @endcode
 */
template <typename T>
class CpuSolverPseudoETDRK4 : public CpuSolverPseudoBase<T>
{
protected:
    /**
     * @brief Raw potential field for ETDRK4.
     *
     * Stores the original w field (per monomer type) for computing
     * the nonlinear term N(q) = -w*q in ETDRK4.
     */
    std::map<std::string, std::vector<T>> w_field;

    /**
     * @brief ETDRK4 coefficient arrays.
     *
     * Created in constructor. Uses unique_ptr for automatic cleanup.
     */
    std::unique_ptr<ETDRK4Coefficients<T>> etdrk4_coefficients_;

    /**
     * @brief Get Boltzmann bond factor for stress computation.
     *
     * For ETDRK4 continuous chains, stress computation does not include
     * the Boltzmann bond factor.
     *
     * @return nullptr (ETDRK4 doesn't use boltz_bond in stress)
     */
    const double* get_stress_boltz_bond(
        std::string monomer_type, bool is_half_bond_length) const override;

public:
    /**
     * @brief Construct ETDRK4 pseudo-spectral solver.
     *
     * Initializes FFT objects, allocates field arrays, and computes
     * ETDRK4 coefficients for each monomer type.
     *
     * @param cb        Computation box defining the grid and BCs
     * @param molecules Molecules container with monomer types
     *
     * @note ETDRK4 coefficients are computed during construction.
     */
    CpuSolverPseudoETDRK4(ComputationBox<T>* cb, Molecules *molecules);

    /**
     * @brief Destructor. Frees allocated arrays.
     */
    ~CpuSolverPseudoETDRK4();

    /**
     * @brief Update potential fields.
     *
     * Stores raw w field and computes Boltzmann factors for each monomer type.
     *
     * @param w_input Map of potential fields by monomer type
     */
    void update_dw(std::map<std::string, const T*> w_input) override;

    /**
     * @brief Advance propagator by one contour step ds using ETDRK4.
     *
     * Uses 4th-order Exponential Time Differencing Runge-Kutta with
     * Kassam-Trefethen coefficient computation for L-stability.
     *
     * @param q_in        Input propagator q(r,s)
     * @param q_out       Output propagator q(r,s+ds)
     * @param monomer_type Monomer type for coefficients
     * @param q_mask      Optional mask (set q=0 in masked regions)
     */
    void advance_propagator(T *q_in, T *q_out, std::string monomer_type, const double *q_mask) override;

    /**
     * @brief Half-bond step (not used for ETDRK4/continuous chains).
     *
     * Empty implementation - continuous chains don't use half-bond steps.
     */
    void advance_propagator_half_bond_step(T *, T *, std::string) override {};
};
#endif
