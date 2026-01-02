/**
 * @file CpuComputationContinuous.h
 * @brief CPU propagator computation for continuous chain model.
 *
 * This header provides CpuComputationContinuous, which implements the full
 * propagator computation pipeline for continuous Gaussian chains on CPU.
 * It orchestrates propagator solving, concentration calculation, and
 * stress computation.
 *
 * **Computation Pipeline:**
 *
 * 1. **compute_propagators()**: Solve modified diffusion equation for all chains
 * 2. **compute_concentrations()**: Calculate segment densities from propagators
 * 3. **compute_stress()**: Calculate stress tensor for box relaxation
 *
 * **Propagator Management:**
 *
 * Propagators are stored with keys based on their dependency codes from the
 * PropagatorComputationOptimizer. This enables efficient reuse of shared
 * propagator segments across different polymer chains.
 *
 * **Concentration Calculation:**
 *
 * For each block, concentration is computed as:
 *
 *     φ(r) = ∫₀^N q_forward(r,s) · q_backward(r,N-s) ds
 *
 * using Simpson's rule for the contour integration.
 *
 * @see PropagatorComputation for the abstract interface
 * @see CpuComputationDiscrete for discrete chain version
 * @see CpuComputationReduceMemoryContinuous for memory-saving version
 */

#ifndef CPU_PSEUDO_CONTINUOUS_H_
#define CPU_PSEUDO_CONTINUOUS_H_

#include <string>
#include <vector>
#include <map>

#include "ComputationBox.h"
#include "Polymer.h"
#include "Molecules.h"
#include "PropagatorComputationOptimizer.h"
#include "PropagatorComputation.h"
#include "CpuSolver.h"
#include "Scheduler.h"

/**
 * @class CpuComputationContinuous
 * @brief CPU propagator computation for continuous Gaussian chains.
 *
 * Manages the complete propagator computation workflow including:
 * - Propagator storage and scheduling
 * - Forward/backward propagator computation
 * - Concentration field calculation
 * - Partition function and stress computation
 *
 * @tparam T Numeric type (double or std::complex<double>)
 *
 * **Memory Layout:**
 *
 * Propagators are stored as 2D arrays:
 *     propagator[key][s][grid_point]
 *
 * where key encodes the dependency structure and monomer type.
 *
 * **Optimizations:**
 *
 * - Propagator aggregation: Identical propagators computed once
 * - Scheduled computation: Ordered to maximize parallelism
 * - Simpson's rule: O(N) contour integration with O(ds²) accuracy
 *
 * @example
 * @code
 * CpuComputationContinuous<double> comp(cb, molecules, optimizer, "pseudospectral");
 *
 * // Compute everything for given fields
 * comp.compute_statistics(w_fields);
 *
 * // Get results
 * double Q = comp.get_total_partition(0);
 * comp.get_total_concentration("A", phi_A);
 * @endcode
 */
template <typename T>
class CpuComputationContinuous : public PropagatorComputation<T>
{
private:
    CpuSolver<T> *propagator_solver;  ///< Solver for diffusion equation
    std::string method;                ///< Solver method ("pseudospectral" or "realspace")

    Scheduler *sc;                     ///< Propagator computation scheduler
    int n_streams;                     ///< Number of parallel computation streams

    /**
     * @brief Storage for computed propagators.
     *
     * Key: dependency code + monomer type (e.g., "v0u1_A")
     * Value: 2D array [contour_step][grid_point]
     */
    std::map<std::string, T **> propagator;

    /**
     * @brief Size of each propagator (number of contour steps).
     *
     * Used for proper deallocation.
     */
    std::map<std::string, int> propagator_size;

    #ifndef NDEBUG
    /**
     * @brief Debug: track which propagator steps are computed.
     */
    std::map<std::string, bool *> propagator_finished;
    #endif

    /**
     * @brief Segment pairs for partition function calculation.
     *
     * Each tuple: (polymer_id, q_forward_ptr, q_backward_ptr, n_repeated)
     * Stores one segment per chain for computing Q = ∫ q_f · q_b dr
     */
    std::vector<std::tuple<int, T *, T *, int>> single_partition_segment;

    /**
     * @brief Block concentration fields.
     *
     * Key: (polymer_id, key_left, key_right) with key_left <= key_right
     * Value: Concentration array (size n_grid)
     */
    std::map<std::tuple<int, std::string, std::string>, T *> phi_block;

    /**
     * @brief Solvent concentration fields.
     *
     * One array per solvent species.
     */
    std::vector<T *> phi_solvent;

    /**
     * @brief Calculate concentration for one polymer block.
     *
     * Integrates product of forward and backward propagators:
     *     φ(r) = ∫ q_1(r,s) · q_2(r,N-s) ds
     *
     * @param phi      Output concentration array
     * @param q_1      Forward propagator array [step][point]
     * @param q_2      Backward propagator array [step][point]
     * @param N_LEFT   Starting contour index
     * @param N_RIGHT  Ending contour index
     */
    void calculate_phi_one_block(T *phi, T **q_1, T **q_2, const int N_LEFT, const int N_RIGHT);

public:
    /**
     * @brief Construct CPU computation for continuous chains.
     *
     * @param cb                              Computation box
     * @param molecules                       Molecules container
     * @param propagator_computation_optimizer Propagator optimizer with dependency info
     * @param method                          Solver method ("pseudospectral" or "realspace")
     */
    CpuComputationContinuous(ComputationBox<T>* cb, Molecules *molecules, PropagatorComputationOptimizer* propagator_computation_optimizer, std::string method);

    /**
     * @brief Destructor. Frees all propagator and concentration arrays.
     */
    ~CpuComputationContinuous();

    /**
     * @brief Update solver for new box dimensions.
     */
    void update_laplacian_operator() override;

    /**
     * @brief Compute all propagators from potential fields.
     *
     * @param w_block Map of potential fields by monomer type
     * @param q_init  Optional initial conditions for propagators
     */
    void compute_propagators(
        std::map<std::string, const T*> w_block,
        std::map<std::string, const T*> q_init = {}) override;

    /**
     * @brief Advance propagator by a single contour step.
     *
     * @param q_init      Input propagator
     * @param q_out       Output propagator
     * @param monomer_type Monomer type for field
     */
    void advance_propagator_single_segment(T* q_init, T *q_out, std::string monomer_type) override;

    /**
     * @brief Calculate all concentration fields from propagators.
     */
    void compute_concentrations() override;

    /**
     * @brief Compute propagators, concentrations, and partition functions.
     *
     * Convenience function that calls compute_propagators() and
     * compute_concentrations() in sequence.
     *
     * @param w_block Potential fields
     * @param q_init  Optional initial conditions
     */
    void compute_statistics(
        std::map<std::string, const T*> w_block,
        std::map<std::string, const T*> q_init = {}) override;

    /**
     * @brief Compute stress tensor components.
     */
    void compute_stress() override;

    /**
     * @brief Get total partition function for a polymer.
     *
     * @param polymer Polymer index
     * @return Q = V⁻¹ ∫ q_forward · q_backward dr
     */
    T get_total_partition(int polymer) override;

    /**
     * @brief Get propagator values at a specific point.
     *
     * @param q_out   Output array (size n_grid)
     * @param polymer Polymer index
     * @param v       Source vertex
     * @param u       Target vertex
     * @param n       Contour step index
     */
    void get_chain_propagator(T *q_out, int polymer, int v, int u, int n) override;

    /// @name Canonical Ensemble Methods
    /// @{

    /**
     * @brief Get total concentration of a monomer type.
     *
     * @param monomer_type Monomer type (e.g., "A")
     * @param phi          Output concentration array
     */
    void get_total_concentration(std::string monomer_type, T *phi) override;

    /**
     * @brief Get concentration of monomer type from a specific polymer.
     *
     * @param polymer      Polymer index
     * @param monomer_type Monomer type
     * @param phi          Output concentration array
     */
    void get_total_concentration(int polymer, std::string monomer_type, T *phi) override;

    /**
     * @brief Get block concentration for a polymer.
     *
     * @param polymer Polymer index
     * @param phi     Output array (size n_grid * n_blocks)
     */
    void get_block_concentration(int polymer, T *phi) override;

    /// @}

    /**
     * @brief Get solvent partition function.
     *
     * @param s Solvent index
     * @return Solvent partition function
     */
    T get_solvent_partition(int s) override;

    /**
     * @brief Get solvent concentration field.
     *
     * @param s   Solvent index
     * @param phi Output concentration array
     */
    void get_solvent_concentration(int s, T *phi) override;

    /// @name Grand Canonical Ensemble Methods
    /// @{

    /**
     * @brief Get concentration with fugacity weighting.
     *
     * @param fugacity     Chemical activity
     * @param polymer      Polymer index
     * @param monomer_type Monomer type
     * @param phi          Output concentration array
     */
    void get_total_concentration_gce(double fugacity, int polymer, std::string monomer_type, T *phi) override;

    /// @}

    /**
     * @brief Validate partition function calculation (for testing).
     *
     * @return True if partition function is consistent across segments
     */
    bool check_total_partition() override;
};
#endif
