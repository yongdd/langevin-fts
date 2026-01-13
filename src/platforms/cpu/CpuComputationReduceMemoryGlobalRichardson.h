/**
 * @file CpuComputationReduceMemoryGlobalRichardson.h
 * @brief Memory-efficient CPU propagator computation with Global Richardson.
 *
 * This class implements memory-efficient Global Richardson extrapolation
 * using checkpointing to reduce memory footprint while maintaining 4th-order
 * accuracy.
 *
 * **Checkpointing Strategy:**
 *
 * Instead of storing all propagators, this implementation:
 * 1. Stores propagators at checkpoint intervals (every sqrt(N) steps)
 * 2. Maintains checkpoints for BOTH full-step and half-step chains
 * 3. Recomputes intermediate values on-the-fly during concentration calc
 * 4. Applies Richardson extrapolation during recomputation
 *
 * **Memory vs Time Tradeoff:**
 *
 * - Standard GlobalRichardson: O(3N × M) memory (full + half + richardson)
 * - Checkpointing: O(3√N × M) memory
 * - Extra computation: O(√N) more propagator steps during phi calculation
 *
 * @see CpuComputationGlobalRichardson for full-storage version
 * @see CpuComputationReduceMemoryContinuous for similar pattern
 */

#ifndef CPU_COMPUTATION_REDUCE_MEMORY_GLOBAL_RICHARDSON_H_
#define CPU_COMPUTATION_REDUCE_MEMORY_GLOBAL_RICHARDSON_H_

#include <string>
#include <vector>
#include <map>
#include <array>

#include "ComputationBox.h"
#include "Polymer.h"
#include "Molecules.h"
#include "PropagatorComputation.h"
#include "PropagatorComputationOptimizer.h"
#include "CpuSolverGlobalRichardsonBase.h"
#include "Scheduler.h"

/**
 * @class CpuComputationReduceMemoryGlobalRichardson
 * @brief Memory-efficient Global Richardson computation using checkpointing.
 */
class CpuComputationReduceMemoryGlobalRichardson : public PropagatorComputation<double>
{
private:
    CpuSolverGlobalRichardsonBase* solver;  ///< Base CN-ADI2 solver
    Scheduler* sc;                           ///< Computation scheduler
    int n_streams;                           ///< Number of parallel threads

    /**
     * @brief Sum of max segments across all propagators.
     * Used to determine checkpoint_interval = ceil(sqrt(total_max_n_segment)).
     */
    int total_max_n_segment;

    /**
     * @brief Interval between checkpoints.
     * Set to ceil(sqrt(total_max_n_segment)).
     */
    int checkpoint_interval;

    /// @name Checkpointed propagators
    /// Key: (propagator_key, checkpoint_index)
    /// @{
    std::map<std::tuple<std::string, int>, double*> propagator_full_at_check_point;
    std::map<std::tuple<std::string, int>, double*> propagator_half_at_check_point;
    /// @}

    /// @name Workspace for recomputation
    /// @{
    std::vector<double*> q_full_recal;   ///< Recomputed full-step propagators
    std::vector<double*> q_half_recal;   ///< Recomputed half-step propagators
    std::array<double*, 2> q_full_pair;  ///< Ping-pong for full-step advancement
    std::array<double*, 2> q_half_pair;  ///< Ping-pong for half-step advancement
    double* q_half_temp;                 ///< Intermediate half-step storage
    std::array<double*, 2> q_full_skip;  ///< Skip buffers for full-step
    std::array<double*, 2> q_half_skip;  ///< Skip buffers for half-step
    /// @}

    /// @name Concentration fields
    /// @{
    std::map<std::tuple<int, std::string, std::string>, double*> phi_block;
    std::vector<double*> phi_solvent;
    /// @}

    /**
     * @brief Segment pairs for partition function calculation.
     * Tuple: (polymer_idx, key_left, key_right, n_segment_left, n_aggregated)
     */
    std::vector<std::tuple<int, std::string, std::string, int, int>> partition_segment_info;

    #ifndef NDEBUG
    std::map<std::string, bool*> propagator_finished;
    #endif

    /**
     * @brief Calculate concentration with on-the-fly Richardson extrapolation.
     *
     * Recomputes propagators from checkpoints and applies Richardson
     * extrapolation during concentration calculation.
     */
    void calculate_phi_one_block(
        double* phi,
        std::string key_left,
        std::string key_right,
        const int N_LEFT,
        const int N_RIGHT,
        std::string monomer_type
    );

public:
    /**
     * @brief Construct reduce-memory Global Richardson computation.
     */
    CpuComputationReduceMemoryGlobalRichardson(
        ComputationBox<double>* cb,
        Molecules* molecules,
        PropagatorComputationOptimizer* propagator_computation_optimizer);

    ~CpuComputationReduceMemoryGlobalRichardson();

    void update_laplacian_operator() override;

    /**
     * @brief Compute propagators, storing only checkpoints.
     */
    void compute_propagators(
        std::map<std::string, const double*> w_block,
        std::map<std::string, const double*> q_init = {}) override;

    void advance_propagator_single_segment(double*, double*, std::string) override
    {
        throw_with_line_number("advance_propagator_single_segment not supported for Global Richardson.");
    }

    /**
     * @brief Compute concentrations with checkpoint recomputation.
     */
    void compute_concentrations() override;

    void compute_statistics(
        std::map<std::string, const double*> w_block,
        std::map<std::string, const double*> q_init = {}) override;

    void compute_stress() override;

    void get_chain_propagator(double* q_out, int polymer, int v, int u, int n) override;

    double get_total_partition(int polymer) override
    {
        return single_polymer_partitions[polymer];
    }

    double get_solvent_partition(int s) override
    {
        return single_solvent_partitions[s];
    }

    void get_total_concentration(std::string monomer_type, double* phi) override;
    void get_total_concentration(int polymer, std::string monomer_type, double* phi) override;

    void get_block_concentration(int, double*) override
    {
        throw_with_line_number("get_block_concentration not implemented for Global Richardson.");
    }

    void get_solvent_concentration(int s, double* phi) override;
    void get_total_concentration_gce(double fugacity, int polymer, std::string monomer_type, double* phi) override;

    bool check_total_partition() override;
};

#endif
