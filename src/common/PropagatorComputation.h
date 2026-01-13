/**
 * @file PropagatorComputation.h
 * @brief Abstract interface for computing chain propagators in polymer field theory.
 *
 * This header defines the PropagatorComputation class, which is the central
 * computational engine for polymer field theory simulations. It computes
 * chain propagators q(r,s) that describe the statistical weight of polymer
 * chain configurations, and from these derives partition functions and
 * monomer concentrations.
 *
 * **Chain Propagators:**
 *
 * For a continuous Gaussian chain, the propagator q(r,s) satisfies the
 * modified diffusion equation:
 *
 *   dq/ds = (a²/6) ∇²q - w(r) q
 *
 * where:
 * - a is the statistical segment length
 * - w(r) is the external potential field
 * - s is the contour variable (0 to N)
 *
 * Initial condition: q(r,0) = 1 for free ends, or specified for grafted chains.
 *
 * **Partition Functions:**
 *
 * The single-chain partition function is:
 *   Q = (1/V) ∫ q(r,N) dr
 *
 * **Concentration Calculation:**
 *
 * Monomer concentrations are computed by integrating propagator products:
 *   φ(r) = (φ_total/Q) ∫₀ᴺ q(r,s) q†(r,N-s) ds
 *
 * where q† is the complementary propagator from the other chain end.
 *
 * @see Pseudo for the pseudo-spectral solver implementation
 * @see PropagatorComputationOptimizer for computation scheduling
 *
 * @example
 * @code
 * // Create solver
 * PropagatorComputation<double>* solver = factory->create_propagator_computation(
 *     cb, molecules, optimizer, "rqm4");
 *
 * // Compute propagators with given potential fields
 * std::map<std::string, const double*> w = {{"A", w_A}, {"B", w_B}};
 * solver->compute_propagators(w);
 *
 * // Compute concentrations
 * solver->compute_concentrations();
 *
 * // Get concentration field for monomer type A
 * double* phi_A = new double[total_grid];
 * solver->get_total_concentration("A", phi_A);
 *
 * // Get partition function
 * double Q = solver->get_total_partition(0);
 * @endcode
 */

#ifndef PROPAGATOR_COMPUTATION_H_
#define PROPAGATOR_COMPUTATION_H_

#include <iostream>
#include <cassert>
#include <cstdio>
#include <tuple>
#include <map>

#include "ComputationBox.h"
#include "Molecules.h"
#include "Polymer.h"
#include "PropagatorCode.h"
#include "PropagatorComputationOptimizer.h"
#include "Exception.h"

/// Tolerance for partition function consistency check in check_total_partition()
constexpr double PARTITION_TOLERANCE = 1e-7;

/**
 * @class PropagatorComputation
 * @brief Abstract base class for chain propagator computation.
 *
 * This class defines the interface for computing chain propagators and
 * derived quantities (partition functions, concentrations, stress).
 * Platform-specific implementations (CPU, CUDA) inherit from this class.
 *
 * @tparam T Numeric type (double or std::complex<double>)
 *
 * **Computation Pipeline:**
 *
 * 1. compute_propagators(w): Solve diffusion equations for all propagators
 * 2. compute_concentrations(): Integrate propagator products to get φ(r)
 * 3. get_total_concentration(): Retrieve computed concentration fields
 *
 * Or use compute_statistics() which combines steps 1-2.
 *
 * **Memory Management:**
 *
 * The class allocates internal storage for propagators and concentrations.
 * Memory can be reduced using the reduce_memory_usage option,
 * which stores only checkpoints and recomputes on demand.
 *
 * **Parallel Computation:**
 *
 * For branched polymers, independent propagators are computed in parallel
 * using the schedule from PropagatorComputationOptimizer. On CUDA, this
 * uses multiple streams for concurrent kernel execution.
 *
 * @see CpuComputationContinuous, CudaComputationContinuous for implementations
 */
template <typename T>
class PropagatorComputation
{
protected:
    ComputationBox<T>* cb;                                      ///< Computation box (grid and integration)
    Molecules *molecules;                                       ///< Polymer and solvent definitions
    PropagatorComputationOptimizer *propagator_computation_optimizer;  ///< Computation schedule

    std::vector<T> single_polymer_partitions;  ///< Partition functions Q[p] for each polymer type p
    std::vector<T> single_solvent_partitions;  ///< Partition functions Q_s[s] for each solvent type s

    /**
     * @brief Stress contributions from each polymer.
     *
     * dq_dl[p][i] = d(ln Q_p)/d(ε_ij) where ε_ij is strain component.
     * Layout: [xx, yy, zz, xy, xz, yz] for full stress tensor.
     * Used for box relaxation in SCFT.
     */
    std::vector<std::array<T,6>> dq_dl;

public:
    /**
     * @brief Construct a PropagatorComputation solver.
     *
     * @param cb           Computation box managing grid and FFT
     * @param molecules    Polymer/solvent definitions
     * @param propagator_computation_optimizer Computation schedule optimizer
     */
    PropagatorComputation(ComputationBox<T>* cb, Molecules *molecules, PropagatorComputationOptimizer* propagator_computation_optimizer);

    /**
     * @brief Virtual destructor.
     */
    virtual ~PropagatorComputation();

    /**
     * @brief Get total number of grid points.
     * @return M = Nx * Ny * Nz
     */
    int get_total_grid() const {return this->cb->get_total_grid();};

    /**
     * @brief Get number of blocks in a polymer.
     * @param polymer Polymer index
     * @return Number of blocks (edges) in polymer graph
     */
    int get_n_blocks(int polymer) const { Polymer& pc = this->molecules->get_polymer(polymer); return pc.get_n_blocks();};

    /**
     * @brief Update Laplacian operator after box size change.
     *
     * Must be called after ComputationBox::set_lx() or set_lattice_parameters()
     * to update the Fourier-space Laplacian for the new box dimensions.
     */
    virtual void update_laplacian_operator() = 0;

    /**
     * @brief Compute all chain propagators.
     *
     * Solves the modified diffusion equation for all propagators needed
     * to compute partition functions and concentrations. Uses dynamic
     * programming to avoid redundant computations for branched polymers.
     *
     * @param w_block Map from monomer type to potential field w(r)
     * @param q_init  Optional: map from label to initial condition q(r,0)
     *                for grafted chain ends
     *
     * @note w_block must contain entries for all monomer types.
     * @note For free ends, q_init is empty and q(r,0) = 1 is used.
     *
     * @example
     * @code
     * std::map<std::string, const double*> w = {{"A", w_A}, {"B", w_B}};
     * solver->compute_propagators(w);
     *
     * // For grafted chain with "surface" initial condition
     * std::map<std::string, const double*> q_init = {{"surface", grafting_density}};
     * solver->compute_propagators(w, q_init);
     * @endcode
     */
    virtual void compute_propagators(
        std::map<std::string, const T*> w_block,
        std::map<std::string, const T*> q_init = {}) = 0;

    /**
     * @brief Advance a single propagator step.
     *
     * Computes q_out = exp(-ds*L/2) exp(-ds*w) exp(-ds*L/2) q_init
     * where L is the Laplacian operator.
     *
     * @param q_init      Input propagator at contour position s
     * @param q_out       Output propagator at contour position s+ds
     * @param monomer_type Monomer type determining segment length
     *
     * @note Used internally by compute_propagators() and for testing.
     */
    virtual void advance_propagator_single_segment(T* q_init, T *q_out, std::string monomer_type) = 0;

    /**
     * @brief Compute monomer concentrations from propagators.
     *
     * Integrates propagator products along the chain contour to compute
     * concentration fields φ(r) for each monomer type.
     *
     * @pre compute_propagators() must be called first.
     *
     * @note Uses Simpson's rule for accurate contour integration.
     */
    virtual void compute_concentrations() = 0;

    /**
     * @brief Compute propagators and concentrations in one call.
     *
     * Equivalent to compute_propagators(w, q_init) + compute_concentrations().
     *
     * @param w_block Map from monomer type to potential field
     * @param q_init  Optional: initial conditions for grafted chains
     */
    virtual void compute_statistics(
        std::map<std::string, const T*> w_block,
        std::map<std::string, const T*> q_init = {}) = 0;

    /**
     * @brief Compute stress tensor from propagators.
     *
     * Computes d(ln Q)/d(L) for box relaxation optimization.
     * Results stored internally; retrieve with get_stress().
     *
     * @pre compute_propagators() must be called first.
     */
    virtual void compute_stress() = 0;

    /**
     * @brief Get single-chain partition function.
     *
     * @param polymer Polymer index (0 to n_polymer_types-1)
     * @return Q = (1/V) ∫ q(r,N) dr
     *
     * @pre compute_propagators() must be called first.
     */
    virtual T get_total_partition(int polymer) = 0;

    /**
     * @brief Get chain propagator at specific contour position.
     *
     * @param q_out   Output array of length total_grid
     * @param polymer Polymer index
     * @param v       Starting vertex of edge
     * @param u       Ending vertex of edge
     * @param n       Contour index (0 to n_segment)
     *
     * @note Returns q(r,s) propagating from vertex v toward u at step n.
     */
    virtual void get_chain_propagator(T *q_out, int polymer, int v, int u, int n) = 0;

    /**
     * @brief Get total concentration for a monomer type (all polymers).
     *
     * @param monomer_type Monomer type label (e.g., "A")
     * @param phi         Output array of length total_grid
     *
     * @pre compute_concentrations() must be called first.
     */
    virtual void get_total_concentration(std::string monomer_type, T *phi) = 0;

    /**
     * @brief Get concentration for a monomer type from specific polymer.
     *
     * @param polymer     Polymer index
     * @param monomer_type Monomer type label
     * @param phi         Output array of length total_grid
     */
    virtual void get_total_concentration(int polymer, std::string monomer_type, T *phi) = 0;

    /**
     * @brief Get block-resolved concentration for a polymer.
     *
     * Returns φ for each block concatenated: [φ_block0, φ_block1, ...]
     *
     * @param polymer Polymer index
     * @param phi     Output array of length (n_blocks * total_grid)
     */
    virtual void get_block_concentration(int polymer, T *phi) = 0;

    /**
     * @brief Get solvent partition function.
     * @param s Solvent index
     * @return Q_s = (1/V) ∫ exp(-w(r)) dr
     */
    virtual T get_solvent_partition(int s) = 0;

    /**
     * @brief Get solvent concentration field.
     * @param s   Solvent index
     * @param phi Output array of length total_grid
     */
    virtual void get_solvent_concentration(int s, T *phi) = 0;

    /**
     * @brief Get stress vector for all polymers combined.
     *
     * @return Vector of {d(ln Q)/dLx, d(ln Q)/dLy, d(ln Q)/dLz}
     *
     * @pre compute_stress() must be called first.
     */
    virtual std::vector<T> get_stress();

    /**
     * @brief Get concentration for grand canonical ensemble.
     *
     * Scales concentration by fugacity for open systems.
     *
     * @param fugacity    Chemical fugacity z = exp(μ/kT)
     * @param polymer     Polymer index
     * @param monomer_type Monomer type label
     * @param phi         Output concentration field
     */
    virtual void get_total_concentration_gce  (double fugacity, int polymer, std::string monomer_type, T *phi) = 0;

    /**
     * @brief Get stress for grand canonical ensemble.
     *
     * @param fugacities Vector of fugacities for each polymer type
     * @return Stress vector weighted by fugacities
     */
    virtual std::vector<T> get_stress_gce(std::vector<double> fugacities);

    /**
     * @brief Verify partition function consistency.
     *
     * Checks that Q = ∫ q(r,s) q†(r,N-s) dr is constant for all s.
     * This is a mathematical identity that should hold for correct propagators.
     *
     * @return true if partition function is consistent, false otherwise
     *
     * @note Useful for debugging and validation. Small deviations are normal
     *       due to numerical discretization.
     */
    virtual bool check_total_partition() = 0;

};
#endif
