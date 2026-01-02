/**
 * @file CpuComputationReduceMemoryContinuous.h
 * @brief Memory-efficient CPU propagator computation for continuous chains.
 *
 * This header provides CpuComputationReduceMemoryContinuous, a variant of
 * CpuComputationContinuous that trades computation time for reduced memory
 * usage through checkpointing.
 *
 * **Checkpointing Strategy:**
 *
 * Instead of storing propagators at every contour step, this implementation:
 *
 * 1. Stores propagators only at checkpoint intervals (e.g., every √N steps)
 * 2. Recomputes intermediate steps on-the-fly during concentration calculation
 *
 * **Memory vs Time Tradeoff:**
 *
 * - Standard: O(N × M) memory, O(N × M) computation
 * - Checkpointing: O(√N × M) memory, O(N × √N × M) computation
 *
 * where N is contour steps and M is grid points.
 *
 * **When to Use:**
 *
 * - Large 3D grids where propagator storage exceeds available memory
 * - Long polymer chains (large N)
 * - Systems with many polymer species
 *
 * **Tradeoff Example:**
 *
 * For N=1000, M=64³:
 * - Standard: ~2 GB per propagator
 * - Checkpointing (√N ≈ 32 checkpoints): ~65 MB per propagator
 * - Extra computation: ~32× more propagator steps
 *
 * @see CpuComputationContinuous for standard version
 * @see PropagatorComputation for the abstract interface
 */

#ifndef CPU_PSEUDO_REDUCE_MEMORY_CONTINUOUS_H_
#define CPU_PSEUDO_REDUCE_MEMORY_CONTINUOUS_H_

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
 * @class CpuComputationReduceMemoryContinuous
 * @brief Memory-efficient propagator computation using checkpointing.
 *
 * Reduces memory usage by storing only checkpoint propagators and
 * recomputing intermediate values during concentration calculation.
 *
 * @tparam T Numeric type (double or std::complex<double>)
 *
 * **Checkpoint Placement:**
 *
 * Checkpoints are placed at approximately √N intervals to minimize
 * the product of storage and recomputation cost.
 *
 * **Recomputation Process:**
 *
 * During concentration calculation:
 * 1. Load propagator from nearest checkpoint
 * 2. Recompute forward to required contour position
 * 3. Use for concentration contribution
 * 4. Repeat for next position
 *
 * @note This version is slower but uses significantly less memory.
 *       Use for large-scale 3D simulations.
 *
 * @example
 * @code
 * // Enable via reduce_memory_usage parameter in factory
 * MklFactory<double> factory(true);  // reduce_memory_usage = true
 *
 * // Usage is identical to standard version
 * auto* comp = factory.create_pseudospectral_solver(cb, molecules, optimizer);
 * comp->compute_statistics(w_fields);
 * @endcode
 */
template <typename T>
class CpuComputationReduceMemoryContinuous : public PropagatorComputation<T>
{
private:
    CpuSolver<T> *propagator_solver;  ///< Diffusion equation solver
    std::string method;                ///< Solver method

    Scheduler *sc;                     ///< Propagator scheduler
    int n_streams;                     ///< Parallel streams

    /**
     * @brief Maximum segment count across all propagators.
     *
     * Determines size of recomputation workspace.
     */
    int total_max_n_segment;

    /**
     * @brief Workspace for propagator recomputation.
     *
     * Size: total_max_n_segment + 1
     * Used to store recomputed propagator steps between checkpoints.
     */
    std::vector<T*> q_recal;

    /**
     * @brief Ping-pong buffers for propagator advancement.
     *
     * Two buffers alternated during sequential propagator computation.
     */
    std::array<T*,2> q_pair;

    /**
     * @brief Checkpointed propagator values.
     *
     * Key: (propagator_key, checkpoint_index)
     * Value: Propagator array at checkpoint (size n_grid)
     *
     * Stores propagators only at checkpoint positions.
     */
    std::map<std::tuple<std::string, int>, T *> propagator_at_check_point;

    #ifndef NDEBUG
    /**
     * @brief Debug: track computed propagators.
     */
    std::map<std::string, bool *> propagator_finished;
    #endif

    /**
     * @brief Segment pairs for partition function.
     *
     * Tuple: (polymer_id, q_forward_ptr, q_backward_ptr, n_repeated)
     */
    std::vector<std::tuple<int, T *, T *, int>> single_partition_segment;

    /**
     * @brief Block concentration fields.
     */
    std::map<std::tuple<int, std::string, std::string>, T *> phi_block;

    /**
     * @brief Solvent concentrations.
     */
    std::vector<T *> phi_solvent;

    /**
     * @brief Calculate concentration for one block with recomputation.
     *
     * Recomputes propagators from checkpoints as needed.
     *
     * @param phi          Output concentration
     * @param key_left     Left propagator key
     * @param key_right    Right propagator key
     * @param N_LEFT       Left contour index
     * @param N_RIGHT      Right contour index
     * @param monomer_type Monomer type
     */
    void calculate_phi_one_block(T *phi, std::string key_left, std::string key_right, const int N_LEFT, const int N_RIGHT, std::string monomer_type);

    /**
     * @brief Recompute propagator from checkpoint.
     *
     * Starting from nearest checkpoint, advances propagator to
     * the required contour positions.
     *
     * @param key          Propagator key
     * @param N_START      Starting contour (checkpoint position)
     * @param N_RIGHT      Ending contour position
     * @param monomer_type Monomer type
     *
     * @return Vector of propagator pointers for requested range
     */
    std::vector<T*> recalcaulte_propagator(std::string key, const int N_START, const int N_RIGHT, std::string monomer_type);

public:
    /**
     * @brief Construct memory-efficient computation.
     *
     * @param cb                              Computation box
     * @param molecules                       Molecules container
     * @param propagator_computation_optimizer Propagator optimizer
     * @param method                          Solver method
     */
    CpuComputationReduceMemoryContinuous(ComputationBox<T>* cb, Molecules *molecules, PropagatorComputationOptimizer* propagator_computation_optimizer, std::string method);

    /**
     * @brief Destructor. Frees checkpoints and workspace.
     */
    ~CpuComputationReduceMemoryContinuous();

    /**
     * @brief Update solver for new box dimensions.
     */
    void update_laplacian_operator() override;

    /**
     * @brief Compute propagators and store checkpoints.
     *
     * Only checkpoint values are stored, not full propagator history.
     *
     * @param w_block Potential fields
     * @param q_init  Optional initial conditions
     */
    void compute_propagators(
        std::map<std::string, const T*> w_block,
        std::map<std::string, const T*> q_init = {}) override;

    /**
     * @brief Advance propagator by one step.
     */
    void advance_propagator_single_segment(T* q_init, T *q_out, std::string monomer_type) override;

    /**
     * @brief Calculate concentrations with on-the-fly recomputation.
     *
     * Recomputes propagator values from checkpoints as needed.
     */
    void compute_concentrations() override;

    /**
     * @brief Compute all statistics.
     */
    void compute_statistics(
        std::map<std::string, const T*> w_block,
        std::map<std::string, const T*> q_init = {}) override;

    /**
     * @brief Compute stress (requires recomputation).
     */
    void compute_stress() override;

    /**
     * @brief Get partition function.
     */
    T get_total_partition(int polymer) override;

    /**
     * @brief Get propagator (requires recomputation).
     */
    void get_chain_propagator(T *q_out, int polymer, int v, int u, int n) override;

    /// @name Canonical Ensemble Methods
    /// @{
    void get_total_concentration(std::string monomer_type, T *phi) override;
    void get_total_concentration(int polymer, std::string monomer_type, T *phi) override;
    void get_block_concentration(int polymer, T *phi) override;
    /// @}

    T get_solvent_partition(int s) override;
    void get_solvent_concentration(int s, T *phi) override;

    /// @name Grand Canonical Ensemble Methods
    /// @{
    void get_total_concentration_gce(double fugacity, int polymer, std::string monomer_type, T *phi) override;
    /// @}

    /**
     * @brief Validate partition function.
     */
    bool check_total_partition() override;
};
#endif
