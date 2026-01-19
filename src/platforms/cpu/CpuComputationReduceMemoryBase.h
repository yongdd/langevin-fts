/**
 * @file CpuComputationReduceMemoryBase.h
 * @brief Common base class for memory-efficient CPU propagator computation.
 *
 * This header provides CpuComputationReduceMemoryBase, a common base class
 * that consolidates shared functionality between CpuComputationReduceMemoryContinuous
 * and CpuComputationReduceMemoryDiscrete, including:
 *
 * - Common member variable declarations (solver, scheduler, checkpoints, workspace)
 * - Shared concentration query methods
 * - Partition function accessors
 * - Laplacian operator updates
 *
 * **Design Decision:**
 *
 * The `single_partition_segment` has different tuple structures between
 * Continuous (4 elements) and Discrete (5 elements with monomer_type),
 * so `get_total_partition()` remains virtual and is implemented in derived classes.
 * Similarly, methods with significantly different logic stay in derived classes:
 * - Constructor/destructor (different checkpoint allocation)
 * - compute_propagators() (discrete has half-bond steps)
 * - compute_concentrations() (different integration formulas)
 * - compute_stress() (different stress computation)
 * - calculate_phi_one_block() (different formulas)
 *
 * @see CpuComputationReduceMemoryContinuous for continuous chain implementation
 * @see CpuComputationReduceMemoryDiscrete for discrete chain implementation
 */

#ifndef CPU_COMPUTATION_REDUCE_MEMORY_BASE_H_
#define CPU_COMPUTATION_REDUCE_MEMORY_BASE_H_

#include <string>
#include <vector>
#include <map>
#include <array>

#include "ComputationBox.h"
#include "Polymer.h"
#include "Molecules.h"
#include "PropagatorComputationOptimizer.h"
#include "PropagatorComputation.h"
#include "CpuSolver.h"
#include "Scheduler.h"

/**
 * @class CpuComputationReduceMemoryBase
 * @brief Common base class for memory-efficient CPU propagator computation.
 *
 * Consolidates shared code between continuous and discrete chain
 * memory-efficient computation classes, including concentration queries
 * and shared workspace management.
 *
 * @tparam T Numeric type (double or std::complex<double>)
 *
 * **Shared Functionality:**
 *
 * - Laplacian operator updates
 * - Total concentration queries (by monomer type, by polymer)
 * - Block concentration retrieval
 * - Solvent partition and concentration accessors
 * - Grand canonical ensemble concentration
 *
 * **Virtual Methods:**
 *
 * Derived classes must implement:
 * - get_total_partition(): Different tuple structure for partition segment
 * - compute_propagators(): Different propagator algorithms
 * - compute_concentrations(): Different integration formulas
 * - compute_stress(): Different stress computation logic
 * - get_chain_propagator(): Different range checks and recomputation
 * - check_total_partition(): Different validation logic
 */
template <typename T>
class CpuComputationReduceMemoryBase : public PropagatorComputation<T>
{
protected:
    CpuSolver<T> *propagator_solver;  ///< Solver for diffusion equation
    std::string method;                ///< Solver method ("pseudospectral" or "realspace")
    Scheduler *sc;                     ///< Propagator computation scheduler
    int n_streams;                     ///< Number of parallel streams (always 1 for reduce memory)

    /**
     * @brief Total sum of segments across all propagators.
     *
     * Used to determine checkpoint_interval = ceil(2*sqrt(total_max_n_segment)).
     */
    int total_max_n_segment;

    /**
     * @brief Interval between checkpoints.
     *
     * Set to ceil(2*sqrt(total_max_n_segment)) to minimize the product
     * of storage and recomputation cost.
     */
    int checkpoint_interval;

    /**
     * @brief Workspace for propagator recomputation.
     *
     * Size: checkpoint_interval = O(sqrt(N))
     * Storage for block values in calculate_phi_one_block().
     */
    std::vector<T*> q_recal;

    /**
     * @brief Ping-pong buffers for q_right propagator advancement.
     *
     * Two buffers alternated during sequential propagator computation.
     */
    std::array<T*,2> q_pair;

    /**
     * @brief Ping-pong buffers for skip phase in calculate_phi_one_block().
     *
     * Used to advance q_left from checkpoint to block start without storing.
     */
    std::array<T*,2> q_skip;

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
     * @brief Debug: track which propagator steps are computed.
     */
    std::map<std::string, bool *> propagator_finished;
    #endif

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

public:
    /**
     * @brief Construct CPU reduce-memory computation base.
     *
     * @param cb                              Computation box
     * @param molecules                       Molecules container
     * @param propagator_computation_optimizer Propagator optimizer with dependency info
     */
    CpuComputationReduceMemoryBase(ComputationBox<T>* cb, Molecules *molecules,
                                   PropagatorComputationOptimizer* propagator_computation_optimizer);

    /**
     * @brief Virtual destructor.
     */
    virtual ~CpuComputationReduceMemoryBase() {}

    /**
     * @brief Update solver for new box dimensions.
     */
    void update_laplacian_operator() override;

    /**
     * @brief Compute all statistics.
     *
     * Convenience function that calls compute_propagators() and
     * compute_concentrations() in sequence.
     */
    void compute_statistics(
        std::map<std::string, const T*> w_block,
        std::map<std::string, const T*> q_init = {}) override;

    /**
     * @brief Advance propagator by one step.
     *
     * @param q_init Input propagator
     * @param q_out  Output propagator
     * @param p      Polymer index
     * @param v      Starting vertex of the block
     * @param u      Ending vertex of the block
     */
    void advance_propagator_single_segment(T* q_init, T *q_out, int p, int v, int u) override;

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
     * @brief Get concentration with fugacity weighting.
     *
     * @param fugacity     Chemical activity
     * @param polymer      Polymer index
     * @param monomer_type Monomer type
     * @param phi          Output concentration array
     */
    void get_total_concentration_gce(double fugacity, int polymer, std::string monomer_type, T *phi) override;

    /**
     * @brief Get block concentration for a polymer.
     *
     * @param polymer Polymer index
     * @param phi     Output array (size n_grid * n_blocks)
     */
    void get_block_concentration(int polymer, T *phi) override;

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
};

#endif
