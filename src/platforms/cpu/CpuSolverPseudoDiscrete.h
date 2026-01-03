/**
 * @file CpuSolverPseudoDiscrete.h
 * @brief Pseudo-spectral solver for discrete chain model on CPU.
 *
 * This header provides CpuSolverPseudoDiscrete, which implements the
 * pseudo-spectral method for solving the discrete chain propagator
 * recurrence relation.
 *
 * **Discrete Chain Model:**
 *
 * Unlike continuous chains that solve a differential equation, discrete
 * chains use a recurrence relation:
 *
 *     q(r, n+1) = exp(-w(r)) · ∫ G(r-r') q(r', n) dr'
 *
 * where G(r) is the bond distribution (Gaussian for Gaussian chains).
 *
 * **Numerical Method:**
 *
 * Uses operator splitting with half-bond steps for accuracy:
 *
 *     q(n+1) = B^(1/2) · A · B^(1/2) · q(n)
 *
 * where:
 * - A = exp(-w) is the Boltzmann factor
 * - B^(1/2) = FFT⁻¹[ exp(-k²b²/12) · FFT[·] ] is half-bond diffusion
 *
 * **Advantages of Discrete Model:**
 *
 * - No contour discretization error (exact for discrete chains)
 * - More natural for short chains
 * - Compatible with discrete interaction models
 *
 * @see CpuSolver for the abstract interface
 * @see CpuSolverPseudoContinuous for continuous chain version
 */

#ifndef CPU_SOLVER_PSEUDO_DISCRETE_H_
#define CPU_SOLVER_PSEUDO_DISCRETE_H_

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
 * @class CpuSolverPseudoDiscrete
 * @brief CPU pseudo-spectral solver for discrete chain model.
 *
 * Implements the discrete chain propagator update using operator splitting
 * with half-bond steps for improved accuracy at chain ends.
 *
 * @tparam T Numeric type (double or std::complex<double>)
 *
 * **Operator Splitting Scheme:**
 *
 * Full segment step:
 *     q(n+1) = B^(1/2) · exp(-w) · B^(1/2) · q(n)
 *
 * Half-bond step only:
 *     q'(n+1/2) = B^(1/2) · q(n)
 *
 * where B^(1/2) is diffusion by half a bond length.
 *
 * **Memory per Monomer Type:**
 *
 * - exp_dw: Full Boltzmann factor exp(-w)
 * - exp_dw_half: Not used (discrete uses full factor)
 * - exp_k2: Half-bond diffusion exp(-k²b²/12)
 *
 * @example
 * @code
 * CpuSolverPseudoDiscrete<double> solver(cb, molecules);
 * solver.update_dw(w_fields);
 *
 * // Full segment step
 * solver.advance_propagator(q_n, q_n1, "A", nullptr);
 *
 * // Half-bond step for chain ends
 * solver.advance_propagator_half_bond_step(q, q_half, "A");
 * @endcode
 */
template <typename T>
class CpuSolverPseudoDiscrete : public CpuSolver<T>
{
private:
    ComputationBox<T>* cb;      ///< Computation box for grid info
    Molecules *molecules;        ///< Molecules container
    std::string chain_model;     ///< Chain model identifier ("discrete")

    FFT<T> *fft;                 ///< FFT object for transforms
    Pseudo<T> *pseudo;           ///< Pseudo-spectral operator helper

public:
    /**
     * @brief Construct pseudo-spectral solver for discrete chains.
     *
     * Initializes FFT objects and Boltzmann factor arrays for the
     * discrete chain model.
     *
     * @param cb        Computation box defining the grid
     * @param molecules Molecules container with monomer types
     */
    CpuSolverPseudoDiscrete(ComputationBox<T>* cb, Molecules *molecules);

    /**
     * @brief Destructor. Frees FFT and Pseudo objects.
     */
    ~CpuSolverPseudoDiscrete();

    /**
     * @brief Update half-bond diffusion operator.
     *
     * Recomputes exp(-k²b²/12) when box dimensions change.
     */
    void update_laplacian_operator() override;

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
     * Computes: q(n+1) = B^(1/2) · exp(-w) · B^(1/2) · q(n)
     *
     * @param q_in        Input propagator q(n)
     * @param q_out       Output propagator q(n+1)
     * @param monomer_type Monomer type for Boltzmann factor and bond length
     * @param q_mask      Optional mask (set q=0 in masked regions)
     */
    void advance_propagator(T *q_in, T *q_out, std::string monomer_type, const double* q_mask) override;

    /**
     * @brief Advance propagator by half bond step only.
     *
     * Computes: q' = B^(1/2) · q = FFT⁻¹[ exp(-k²b²/12) · FFT[q] ]
     *
     * Used at chain ends to properly handle the half-bond contribution.
     *
     * @param q_in        Input propagator
     * @param q_out       Output propagator after half-bond diffusion
     * @param monomer_type Monomer type for bond length
     */
    void advance_propagator_half_bond_step(T *q_in, T *q_out, std::string monomer_type) override;

    /**
     * @brief Compute stress from one segment.
     *
     * @param q_1                Forward propagator
     * @param q_2                Backward propagator
     * @param monomer_type       Monomer type
     * @param is_half_bond_length True for half-bond contribution at ends
     *
     * @return Stress components [σ_xx, σ_yy, σ_zz, σ_xy, σ_xz, σ_yz]
     */
    std::vector<T> compute_single_segment_stress(
                T *q_1, T *q_2, std::string monomer_type, bool is_half_bond_length) override;
};
#endif
