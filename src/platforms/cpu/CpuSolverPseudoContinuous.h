/**
 * @file CpuSolverPseudoContinuous.h
 * @brief Pseudo-spectral solver for continuous chain model on CPU.
 *
 * This header provides CpuSolverPseudoContinuous, which implements the
 * pseudo-spectral method for solving the modified diffusion equation
 * with continuous Gaussian chains.
 *
 * **Numerical Method:**
 *
 * Uses 4th-order Richardson extrapolation for high accuracy:
 *
 *     q(s+ds) = (4/3) q^(ds/2,ds/2) - (1/3) q^(ds)
 *
 * where q^(ds/2,ds/2) is computed with two half-steps and q^(ds) is
 * computed with one full step. Each step uses operator splitting:
 *
 *     q(s+h) = exp(-w·h/2) · FFT⁻¹[ exp(-k²b²h/6) · FFT[ exp(-w·h/2) · q(s) ] ]
 *
 * **Accuracy:**
 *
 * - 4th-order accurate in ds (contour discretization)
 * - Spectral accuracy in spatial discretization
 *
 * @see CpuSolver for the abstract interface
 * @see CpuSolverPseudoDiscrete for discrete chain version
 * @see Pseudo for the pseudo-spectral implementation details
 */

#ifndef CPU_SOLVER_PSEUDO_CONTINUOUS_H_
#define CPU_SOLVER_PSEUDO_CONTINUOUS_H_

#include <string>
#include <vector>
#include <map>

#include "Exception.h"
#include "Molecules.h"
#include "ComputationBox.h"
#include "CpuSolver.h"
#include "Pseudo.h"
#include "FFT.h"

/**
 * @class CpuSolverPseudoContinuous
 * @brief CPU pseudo-spectral solver for continuous Gaussian chains.
 *
 * Implements operator splitting with 4th-order Richardson extrapolation
 * for solving the continuous chain diffusion equation.
 *
 * @tparam T Numeric type (double or std::complex<double>)
 *
 * **Memory Usage:**
 *
 * Per monomer type:
 * - exp_dw, exp_dw_half: Boltzmann factors (n_grid each)
 * - exp_k2_1, exp_k2_2: Fourier space diffusion operators
 * - Temporary arrays for FFT workspace
 *
 * **Performance:**
 *
 * Each propagator step requires:
 * - 6 FFTs (3 for full step, 3 for each half-step) with Richardson
 * - O(n_grid) element-wise operations
 *
 * @example
 * @code
 * // Create solver
 * CpuSolverPseudoContinuous<double> solver(cb, molecules);
 *
 * // Update for new fields
 * solver.update_dw(w_fields);
 *
 * // Advance propagator
 * solver.advance_propagator(q_in, q_out, "A", nullptr);
 * @endcode
 */
template <typename T>
class CpuSolverPseudoContinuous : public CpuSolver<T>
{
private:
    ComputationBox<T>* cb;      ///< Computation box for grid info
    Molecules *molecules;        ///< Molecules container
    std::string chain_model;     ///< Chain model identifier ("continuous")

    FFT<T> *fft;                 ///< FFT object for transforms
    Pseudo<T> *pseudo;           ///< Pseudo-spectral operator helper

public:
    /**
     * @brief Construct pseudo-spectral solver for continuous chains.
     *
     * Initializes FFT objects and allocates Boltzmann factor arrays
     * for each monomer type defined in molecules.
     *
     * @param cb        Computation box defining the grid
     * @param molecules Molecules container with monomer types
     *
     * @note Uses MklFFT for the FFT implementation.
     */
    CpuSolverPseudoContinuous(ComputationBox<T>* cb, Molecules *molecules);

    /**
     * @brief Destructor. Frees FFT and Pseudo objects.
     */
    ~CpuSolverPseudoContinuous();

    /**
     * @brief Update Fourier-space diffusion operators.
     *
     * Recomputes exp(-k²b²ds/6) for each monomer type when box
     * dimensions change. Required after box size updates.
     */
    void update_laplacian_operator() override;

    /**
     * @brief Update Boltzmann factors from potential fields.
     *
     * Computes exp(-w·ds/2) for each monomer type from input fields.
     *
     * @param w_input Map of potential fields by monomer type
     */
    void update_dw(std::map<std::string, const T*> w_input) override;

    /**
     * @brief Advance propagator by one contour step ds.
     *
     * Uses 4th-order Richardson extrapolation combining full and
     * half-step results for high accuracy.
     *
     * @param q_in        Input propagator q(r,s)
     * @param q_out       Output propagator q(r,s+ds)
     * @param monomer_type Monomer type for Boltzmann factors
     * @param q_mask      Optional mask (set q=0 in masked regions)
     */
    void advance_propagator(T *q_in, T *q_out, std::string monomer_type, const double *q_mask) override;

    /**
     * @brief Half-bond step (not used for continuous chains).
     *
     * Empty implementation - continuous chains don't use half-bond steps.
     *
     * @param q_in        Ignored
     * @param q_out       Ignored
     * @param monomer_type Ignored
     */
    void advance_propagator_half_bond_step(T *, T *, std::string) override {};

    /**
     * @brief Compute stress contribution from one segment.
     *
     * Calculates ∂H/∂ε_αβ contribution from correlating forward and
     * backward propagators.
     *
     * @param q_1                Forward propagator
     * @param q_2                Backward propagator
     * @param monomer_type       Monomer type for segment length
     * @param is_half_bond_length Ignored (always false for continuous)
     *
     * @return Stress components [σ_xx, σ_yy, σ_zz, σ_xy, σ_xz, σ_yz]
     */
    std::vector<T> compute_single_segment_stress(
                T *q_1, T *q_2, std::string monomer_type, bool is_half_bond_length) override;
};
#endif
