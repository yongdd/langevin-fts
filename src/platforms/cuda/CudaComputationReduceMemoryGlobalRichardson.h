/**
 * @file CudaComputationReduceMemoryGlobalRichardson.h
 * @brief Memory-efficient GPU propagator computation with Global Richardson.
 *
 * This class implements memory-efficient Global Richardson extrapolation
 * using checkpointing to reduce GPU memory footprint.
 *
 * @see CudaComputationGlobalRichardson for full-storage version
 * @see CpuComputationReduceMemoryGlobalRichardson for CPU version
 */

#ifndef CUDA_COMPUTATION_REDUCE_MEMORY_GLOBAL_RICHARDSON_H_
#define CUDA_COMPUTATION_REDUCE_MEMORY_GLOBAL_RICHARDSON_H_

#include <string>
#include <vector>
#include <map>
#include <array>

#include "ComputationBox.h"
#include "Polymer.h"
#include "Molecules.h"
#include "PropagatorComputation.h"
#include "PropagatorComputationOptimizer.h"
#include "CudaSolverGlobalRichardsonBase.h"
#include "CudaCommon.h"
#include "Scheduler.h"

/**
 * @class CudaComputationReduceMemoryGlobalRichardson
 * @brief Memory-efficient GPU Global Richardson computation using checkpointing.
 */
class CudaComputationReduceMemoryGlobalRichardson : public PropagatorComputation<double>
{
private:
    ComputationBox<double>* cb;
    Molecules* molecules;
    PropagatorComputationOptimizer* propagator_computation_optimizer;

    CudaSolverGlobalRichardsonBase* solver;
    Scheduler* sc;
    int n_streams;

    cudaStream_t stream;  ///< Single stream for reduce-memory mode

    int total_max_n_segment;
    int checkpoint_interval;

    /// @name Checkpointed propagators on device
    /// @{
    std::map<std::tuple<std::string, int>, double*> d_propagator_full_at_check_point;
    std::map<std::tuple<std::string, int>, double*> d_propagator_half_at_check_point;
    /// @}

    /// @name Workspace for recomputation on device
    /// @{
    std::vector<double*> d_q_full_recal;
    std::vector<double*> d_q_half_recal;
    std::array<double*, 2> d_q_full_pair;
    std::array<double*, 2> d_q_half_pair;
    double* d_q_half_temp;
    std::array<double*, 2> d_q_full_skip;
    std::array<double*, 2> d_q_half_skip;
    /// @}

    /// @name Concentration fields on device
    /// @{
    std::map<std::tuple<int, std::string, std::string>, double*> d_phi_block;
    std::vector<double*> d_phi_solvent;
    /// @}

    /// @name Working arrays
    /// @{
    double* d_q_unity;
    double* d_q_mask;
    double* d_phi;
    /// @}

    std::vector<std::tuple<int, std::string, std::string, int, int>> partition_segment_info;

    #ifndef NDEBUG
    std::map<std::string, bool*> propagator_finished;
    #endif

    void calculate_phi_one_block(
        double* d_phi,
        std::string key_left,
        std::string key_right,
        const int N_LEFT,
        const int N_RIGHT,
        std::string monomer_type
    );

public:
    CudaComputationReduceMemoryGlobalRichardson(
        ComputationBox<double>* cb,
        Molecules* molecules,
        PropagatorComputationOptimizer* propagator_computation_optimizer);

    ~CudaComputationReduceMemoryGlobalRichardson();

    void update_laplacian_operator() override;

    void compute_propagators(
        std::map<std::string, const double*> w_block,
        std::map<std::string, const double*> q_init = {}) override;

    void advance_propagator_single_segment(double*, double*, std::string) override
    {
        throw_with_line_number("advance_propagator_single_segment not supported for Global Richardson.");
    }

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
