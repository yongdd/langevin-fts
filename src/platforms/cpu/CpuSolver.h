/**
 * @file CpuSolver.h
 * @brief Abstract interface for CPU propagator solvers.
 *
 * This header defines the CpuSolver class, which provides a common interface
 * for different propagator solving methods on CPU. The two main implementations
 * are:
 *
 * - **Pseudo-spectral method**: Uses FFT for diffusion (continuous) or bond
 *   convolution (discrete) in Fourier space
 * - **Real-space method**: Uses finite differences with Crank-Nicolson scheme
 *
 * **Chain Models:**
 *
 * - **Continuous chains**: Solve the modified diffusion equation
 *       ∂q/∂s = (b²/6) ∇²q - w(r) q
 *
 * - **Discrete chains**: Solve the Chapman-Kolmogorov integral equation
 *       q(r, n+1) = exp(-w(r)*ds) * ∫ g(r-r') q(r', n) dr'
 *   where g(r) is the bond function. See Park et al. J. Chem. Phys. 150, 234901 (2019).
 *
 * @see CpuSolverPseudoRQM4 for continuous chain pseudo-spectral
 * @see CpuSolverPseudoDiscrete for discrete chain pseudo-spectral
 * @see CpuSolverCNADI for finite difference method
 */

#ifndef CPU_SOLVER_H_
#define CPU_SOLVER_H_

#include <string>
#include <vector>
#include <map>

#include "Exception.h"
#include "Molecules.h"
#include "ComputationBox.h"
#include "FFT.h"

class SpaceGroup;

/**
 * @class CpuSolver
 * @brief Abstract base class for CPU-based propagator solvers.
 *
 * Defines the interface for advancing chain propagators and computing
 * stress. Derived classes implement specific numerical methods.
 *
 * @tparam T Numeric type (double or std::complex<double>)
 *
 * **Operator Splitting (Continuous Chains):**
 *
 * For continuous chains (modified diffusion equation):
 *     q(s+ds) = exp(-w·ds/2) · FFT⁻¹[ exp(-b²|k|²ds/6) · FFT[ exp(-w·ds/2) · q(s) ] ]
 *
 * **Operator Splitting (Discrete Chains):**
 *
 * For discrete chains (Chapman-Kolmogorov equation):
 *     q(n+1) = B^(1/2) · exp(-w·ds) · B^(1/2) · q(n)
 *
 * where B^(1/2) is half-bond convolution with ĝ^(1/2)(k) = exp(-b²|k|²ds/12).
 *
 * **Common Usage Pattern:**
 *
 * @code
 * // Create solver (via derived class)
 * CpuSolver<double>* solver = new CpuSolverPseudoRQM4<double>(cb, molecules);
 *
 * // Update when fields change
 * solver->update_dw(w_fields);
 *
 * // Advance propagator
 * solver->advance_propagator(q_in, q_out, "A", mask);
 * @endcode
 */
template <typename T>
class CpuSolver
{
public:
    /**
     * @brief Boltzmann factor for full segment: exp(-w(r)·ds) per ds_index.
     *
     * Key: (ds_index, monomer_type)
     * Value: Vector of size n_grid containing exp(-w[i]*ds)
     *
     * ds_index is 1-based (from ContourLengthMapping).
     * Used in the real-space method and discrete chain pseudo-spectral.
     */
    std::map<int, std::map<std::string, std::vector<T>>> exp_dw;

    /**
     * @brief Boltzmann factor for half segment: exp(-w(r)·ds/2) per ds_index.
     *
     * Key: (ds_index, monomer_type)
     * Value: Vector of size n_grid containing exp(-w[i]*ds/2)
     *
     * ds_index is 1-based (from ContourLengthMapping).
     * Used in operator splitting schemes for symmetric time stepping.
     */
    std::map<int, std::map<std::string, std::vector<T>>> exp_dw_half;

    /**
     * @brief Boltzmann factor for quarter segment: exp(-w(r)·ds/4) per ds_index.
     *
     * Key: (ds_index, monomer_type)
     * Value: Vector of size n_grid containing exp(-w[i]*ds/4)
     *
     * ds_index is 1-based (from ContourLengthMapping).
     * Used in RQM4 Richardson extrapolation.
     */
    std::map<int, std::map<std::string, std::vector<T>>> exp_dw_quarter;

    /**
     * @brief Virtual destructor.
     */
    virtual ~CpuSolver() {};

    /**
     * @brief Set space group for reduced basis operations.
     *
     * When set, propagator input/output uses reduced basis and the solver
     * internally handles expand/reduce around FFT operations.
     *
     * Default implementation does nothing (for solvers that don't support space groups).
     *
     * @param sg Space group pointer (nullptr to disable reduced basis)
     */
    virtual void set_space_group(SpaceGroup* /*sg*/) {}

    /**
     * @brief Reset internal solver state for a new propagator.
     *
     * Called when starting a new propagator computation to reset any
     * internal state (e.g., for Global Richardson method which maintains
     * independent full-step and half-step evolutions).
     *
     * Default implementation does nothing. Override in derived classes
     * that maintain internal state between advance_propagator() calls.
     */
    virtual void reset_internal_state() {}

    /**
     * @brief Update Laplacian operator for changed box dimensions.
     *
     * Called when box size changes (e.g., during stress relaxation).
     * Recomputes FFT wavenumbers k² or finite difference coefficients.
     *
     * For pseudo-spectral: Updates exp(-b²|k|²ds/6) in Fourier space
     *     - Continuous chains: diffusion propagator
     *     - Discrete chains: bond function ĝ(k)
     * For real-space: Updates tridiagonal matrix coefficients
     */
    virtual void update_laplacian_operator() = 0;

    /**
     * @brief Update Boltzmann factors from new potential fields.
     *
     * Computes exp_dw and exp_dw_half from input potential fields.
     * Must be called whenever w fields change (each SCFT iteration).
     *
     * @param w_input Map of potential fields by monomer type
     *                Key: monomer type (e.g., "A", "B")
     *                Value: Pointer to potential field array (size n_grid)
     *
     * @example
     * @code
     * std::map<std::string, const double*> w_fields;
     * w_fields["A"] = w_A;
     * w_fields["B"] = w_B;
     * solver->update_dw(w_fields);
     * @endcode
     */
    virtual void update_dw(std::map<std::string, const T*> w_input) = 0;

    /**
     * @brief Advance propagator by one contour step.
     *
     * Solves the modified diffusion equation for one step ds:
     *
     *     q(r, s+ds) = Propagator[ q(r, s), w(r) ]
     *
     * When space_group is set on the solver, input/output are in reduced basis
     * and the solver handles expand/reduce around FFT operations internally.
     *
     * @param q_in        Input propagator at contour s (size n_grid or n_basis)
     * @param q_out       Output propagator at contour s+ds (size n_grid or n_basis)
     * @param monomer_type Monomer type determining segment length and field
     * @param q_mask      Optional mask for impenetrable regions (nullptr if none)
     * @param ds_index    Index for the ds value to use (1-based, default: 1 for global ds)
     *
     * @note For continuous chains, ds is the contour discretization parameter.
     *       For discrete chains, this advances by one segment.
     *
     * @example
     * @code
     * // Advance propagator for type A monomer with global ds
     * solver->advance_propagator(q_current, q_next, "A", nullptr, 1);
     * @endcode
     */
    virtual void advance_propagator(
        T *q_in, T *q_out, std::string monomer_type, const double *q_mask, int ds_index) = 0;

    /**
     * @brief Advance propagator by half bond step (discrete chain model).
     *
     * Used in discrete chain model where the full propagator step is split:
     *
     *     q(n+1) = B^(1/2) · A · B^(1/2) · q(n)
     *
     * where B^(1/2) is the half-bond convolution with ĝ^(1/2)(k) = exp(-b²|k|²ds/12).
     *
     * When space_group is set on the solver, input/output are in reduced basis
     * and the solver handles expand/reduce around FFT operations internally.
     *
     * @param q_in        Input propagator (size n_grid or n_basis)
     * @param q_out       Output propagator (size n_grid or n_basis)
     * @param monomer_type Monomer type for segment length
     *
     * @note Only meaningful for discrete chain model. Empty implementation
     *       for continuous chain solvers.
     */
    virtual void advance_propagator_half_bond_step(T *q_in, T *q_out, std::string monomer_type) = 0;

    /**
     * @brief Compute stress contribution from a single segment.
     *
     * Calculates the stress tensor contribution from correlating two
     * propagators (forward and backward) at a single contour point:
     *
     *     σ_αβ = ∫ q₁(r) · (∂/∂ε_αβ)[Propagator] · q₂(r) dr
     *
     * @param q_1                First propagator (typically forward)
     * @param q_2                Second propagator (typically backward)
     * @param monomer_type       Monomer type for segment length
     * @param is_half_bond_length Whether using half bond length (discrete model)
     *
     * @return Vector of stress components [σ_xx, σ_yy, σ_zz, σ_xy, σ_xz, σ_yz]
     *         Full 6-component stress tensor for non-orthogonal systems.
     *
     * @note Used in box size optimization to find stress-free unit cell.
     */
    virtual std::vector<T> compute_single_segment_stress(
        T *q_1, T *q_2, std::string monomer_type, bool is_half_bond_length) = 0;

};
#endif
