/**
 * @file CpuSolverPseudoDiscrete.h
 * @brief Pseudo-spectral solver for discrete chain model on CPU.
 *
 * This header provides CpuSolverPseudoDiscrete, which implements the
 * pseudo-spectral method for discrete chain propagators using the
 * Chapman-Kolmogorov integral equation.
 *
 * **Boundary Conditions:**
 *
 * Supports all boundary conditions via the unified FftwFFT class:
 * - PERIODIC: Standard FFT (complex coefficients)
 * - REFLECTING: DCT-II/III (Neumann BC, zero flux)
 * - ABSORBING: DST-II/III (Dirichlet BC, zero value)
 *
 * **Discrete Chain Model (Chapman-Kolmogorov Equation):**
 *
 * Unlike continuous chains that solve the modified diffusion equation,
 * discrete chains use the Chapman-Kolmogorov integral equation:
 *
 *     q(r, n+1) = exp(-w(r)*ds) * integral g(r-r') q(r', n) dr'
 *
 * where g(r) is the bond function. For the bead-spring (Gaussian) model:
 *
 *     g(r) = (3/2πa²)^(3/2) exp(-3|r|²/2a²)
 *
 * with Fourier transform ĝ(k) = exp(-a²|k|²/6).
 *
 * See Park et al. J. Chem. Phys. 150, 234901 (2019) for details.
 *
 * **Numerical Method (N-1 Bond Model):**
 *
 * For each propagator step from segment i to i+1:
 *
 *     q(i+1) = exp(-w*ds) * FFT^-1[ ĝ(k) * FFT[q(i)] ]
 *
 * where:
 * - ĝ(k) = exp(-b²|k|²ds/6) is the full bond function
 * - exp(-w*ds) is the full-segment Boltzmann factor
 *
 * The initial condition is q(1) = exp(-w*ds).
 *
 * Half-bond steps (ĝ^(1/2)(k) = exp(-b²|k|²ds/12)) are used only at
 * chain ends and junction points for proper boundary treatment.
 *
 * **Advantages of Discrete Model:**
 *
 * - No contour discretization error (exact for discrete chains)
 * - More natural for short chains
 * - Compatible with discrete interaction models
 *
 * @see CpuSolverPseudoBase for shared functionality
 * @see CpuSolverPseudoRQM4 for continuous chain version
 */

#ifndef CPU_SOLVER_PSEUDO_DISCRETE_H_
#define CPU_SOLVER_PSEUDO_DISCRETE_H_

#include <string>
#include <vector>
#include <map>

#include "Exception.h"
#include "Molecules.h"
#include "ComputationBox.h"
#include "CpuSolverPseudoBase.h"

/**
 * @class CpuSolverPseudoDiscrete
 * @brief CPU pseudo-spectral solver for discrete chain model.
 *
 * Implements the discrete chain propagator update using the Chapman-Kolmogorov
 * integral equation. Supports all boundary conditions (periodic, reflecting,
 * absorbing).
 *
 * @tparam T Numeric type (double or std::complex<double>)
 *
 * **N-1 Bond Model:**
 *
 * Full segment step (bond convolution + full-segment Boltzmann weight):
 *     q(i+1) = exp(-w*ds) * FFT^-1[ ĝ(k) * FFT[q(i)] ]
 *
 * Half-bond step only (for chain ends and junctions):
 *     q' = FFT^-1[ ĝ^(1/2)(k) * FFT[q] ]
 *
 * where:
 * - ĝ(k) = exp(-b²|k|²ds/6) is the full bond function
 * - ĝ^(1/2)(k) = exp(-b²|k|²ds/12) is the half-bond function
 *
 * **Memory per Monomer Type:**
 *
 * - exp_dw: Full-segment Boltzmann factor exp(-w*ds)
 * - boltz_bond: Full bond function ĝ(k)
 * - boltz_bond_half: Half-bond function ĝ^(1/2)(k)
 *
 * @example
 * @code
 * CpuSolverPseudoDiscrete<double> solver(cb, molecules);
 * solver.update_dw(w_fields);
 *
 * // Full segment step
 * solver.advance_propagator(q_i, q_i1, "A", nullptr);
 *
 * // Half-bond step for chain ends
 * solver.advance_propagator_half_bond_step(q, q_half, "A");
 * @endcode
 */
template <typename T>
class CpuSolverPseudoDiscrete : public CpuSolverPseudoBase<T>
{
protected:
    /**
     * @brief Get bond function for stress computation.
     *
     * For discrete chains, stress computation includes the bond function.
     * Returns ĝ(k) = exp(-b²|k|²ds/6) for full bond or
     * ĝ^(1/2)(k) = exp(-b²|k|²ds/12) for half-bond.
     *
     * @param monomer_type Monomer type
     * @param is_half_bond_length Whether using half bond length
     * @return Pointer to appropriate bond function array
     */
    const double* get_stress_boltz_bond(
        std::string monomer_type, bool is_half_bond_length) const override;

public:
    /**
     * @brief Construct pseudo-spectral solver for discrete chains.
     *
     * Initializes FFT objects and Boltzmann factor arrays for the
     * discrete chain model.
     *
     * @param cb        Computation box defining the grid and BCs
     * @param molecules Molecules container with monomer types
     * @param backend   FFT backend to use (FFTW, default: FFTW)
     */
    CpuSolverPseudoDiscrete(ComputationBox<T>* cb, Molecules *molecules, FFTBackend backend = FFTBackend::FFTW);

    /**
     * @brief Destructor. Frees exp_dw arrays.
     */
    ~CpuSolverPseudoDiscrete();

    /**
     * @brief Update Boltzmann factors from potential fields.
     *
     * Computes exp(-w) for each monomer type.
     *
     * @param w_input Map of potential fields by monomer type
     */
    void update_dw(std::map<std::string, const T*> w_input) override;

    /**
     * @brief Advance propagator by one full segment step.
     *
     * Computes: q(i+1) = exp(-w*ds) * FFT^-1[ ĝ(k) * FFT[q(i)] ]
     * where ĝ(k) = exp(-b²|k|²ds/6) is the full bond function.
     *
     * @param q_in        Input propagator q(i)
     * @param q_out       Output propagator q(i+1)
     * @param monomer_type Monomer type for bond function and segment weight
     * @param q_mask      Optional mask (set q=0 in masked regions)
     * @param ds_index    Index for per-block local ds value
     */
    void advance_propagator(T *q_in, T *q_out, std::string monomer_type, const double* q_mask, int ds_index) override;

    /**
     * @brief Advance propagator by half bond step only.
     *
     * Computes: q' = FFT^-1[ ĝ^(1/2)(k) * FFT[q] ]
     * where ĝ^(1/2)(k) = exp(-b²|k|²ds/12) is the half-bond function.
     *
     * Used at chain ends and junction points.
     *
     * @param q_in        Input propagator
     * @param q_out       Output propagator after half-bond convolution
     * @param monomer_type Monomer type for bond function
     */
    void advance_propagator_half_bond_step(T *q_in, T *q_out, std::string monomer_type) override;

    /**
     * @brief Update Laplacian operator and re-register local ds values.
     *
     * Overrides base class to register per-block local_ds values with Pseudo,
     * ensuring correct bond functions for each block.
     */
    void update_laplacian_operator() override;
};
#endif
