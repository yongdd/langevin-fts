/**
 * @file CpuSolverPseudoRK2.h
 * @brief Pseudo-spectral solver for continuous chain model on CPU using RK2.
 *
 * This header provides CpuSolverPseudoRK2, which implements the
 * pseudo-spectral method for solving the modified diffusion equation
 * with continuous Gaussian chains using RK2 (Rasmussen-Kalosakas 2nd-order).
 *
 * **Boundary Conditions:**
 *
 * Supports all boundary conditions via the unified MklFFT class:
 * - PERIODIC: Standard FFT (complex coefficients)
 * - REFLECTING: DCT-II/III (Neumann BC, zero flux)
 * - ABSORBING: DST-II/III (Dirichlet BC, zero value)
 *
 * **Numerical Method (RK2):**
 *
 * Uses simple operator splitting (Strang splitting) without Richardson extrapolation:
 *
 *     q(s+ds) = exp(-w·ds/2) · FFT⁻¹[ exp(-k²b²ds/6) · FFT[ exp(-w·ds/2) · q(s) ] ]
 *
 * **Accuracy:**
 *
 * - 2nd-order accurate in ds (contour discretization)
 * - Spectral accuracy in spatial discretization
 *
 * **References:**
 * - Rasmussen & Kalosakas, J. Polym. Sci. B 40, 1777 (2002)
 * - Stasiak & Matsen, Eur. Phys. J. E 34, 110 (2011)
 *
 * @see CpuSolverPseudoBase for shared functionality
 * @see CpuSolverPseudoRQM4 for 4th-order Richardson extrapolation version
 * @see CpuSolverPseudoETDRK4 for ETDRK4 alternative
 */

#ifndef CPU_SOLVER_PSEUDO_RK2_H_
#define CPU_SOLVER_PSEUDO_RK2_H_

#include <string>
#include <vector>
#include <map>

#include "Exception.h"
#include "Molecules.h"
#include "ComputationBox.h"
#include "CpuSolverPseudoBase.h"

/**
 * @class CpuSolverPseudoRK2
 * @brief CPU pseudo-spectral solver for continuous Gaussian chains using RK2.
 *
 * Implements operator splitting with RK2 (2nd-order Rasmussen-Kalosakas method)
 * for solving the continuous chain diffusion equation. Supports all
 * boundary conditions (periodic, reflecting, absorbing).
 *
 * @tparam T Numeric type (double or std::complex<double>)
 *
 * **Memory Usage:**
 *
 * Per monomer type:
 * - exp_dw: Boltzmann factors (n_grid each)
 * - Temporary arrays for FFT workspace
 *
 * **Performance:**
 *
 * Each propagator step requires:
 * - 2 FFTs (1 forward, 1 backward)
 * - O(n_grid) element-wise operations
 *
 * This is faster than RQM4 (which requires 6 FFTs) but 2nd-order instead of 4th-order.
 *
 * @example
 * @code
 * // Create solver
 * CpuSolverPseudoRK2<double> solver(cb, molecules);
 *
 * // Update for new fields
 * solver.update_dw(w_fields);
 *
 * // Advance propagator
 * solver.advance_propagator(q_in, q_out, "A", nullptr);
 * @endcode
 */
template <typename T>
class CpuSolverPseudoRK2 : public CpuSolverPseudoBase<T>
{
protected:
    /**
     * @brief Get Boltzmann bond factor for stress computation.
     *
     * For continuous chains, stress computation does not include the
     * Boltzmann bond factor (it's absorbed into the RK2 scheme).
     *
     * @return nullptr (continuous chains don't use boltz_bond in stress)
     */
    const double* get_stress_boltz_bond(
        std::string monomer_type, bool is_half_bond_length) const override;

public:
    /**
     * @brief Construct pseudo-spectral solver for continuous chains.
     *
     * Initializes FFT objects and allocates Boltzmann factor arrays
     * for each monomer type defined in molecules.
     *
     * @param cb        Computation box defining the grid and BCs
     * @param molecules Molecules container with monomer types
     * @param backend   FFT backend to use (MKL or FFTW, default: MKL)
     *
     * @note Supports both MKL and FFTW FFT implementations.
     */
    CpuSolverPseudoRK2(ComputationBox<T>* cb, Molecules *molecules, FFTBackend backend = FFTBackend::MKL);

    /**
     * @brief Destructor. Frees exp_dw arrays.
     */
    ~CpuSolverPseudoRK2();

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
     * Uses RK2 (2nd-order operator splitting) with a single full step.
     *
     * @param q_in        Input propagator q(r,s)
     * @param q_out       Output propagator q(r,s+ds)
     * @param monomer_type Monomer type for Boltzmann factors
     * @param q_mask      Optional mask (set q=0 in masked regions)
     * @param ds_index    Index for the ds value (1-based, default: 1 for global ds)
     */
    void advance_propagator(T *q_in, T *q_out, std::string monomer_type, const double *q_mask, int ds_index) override;

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
     * @brief Update Laplacian operator and re-register local ds values.
     *
     * Overrides base class to ensure local_ds values are re-registered
     * when the grid changes (e.g., during box size optimization).
     */
    void update_laplacian_operator() override;
};
#endif
