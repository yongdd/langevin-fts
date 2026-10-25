/**
 * @file ComputationReduceMemoryContinuous.h
 * @brief Propagator computation for continuous chains, reduced memory.
 *
 * Same physics and arithmetic as ComputationContinuous, but q(r, s) is kept
 * only at checkpoints: s = 0, every checkpoint_interval segments, and the
 * segments other propagators start from. Every other segment is recomputed
 * from the last checkpoint before it when the concentrations, the stress,
 * or a propagator are needed. The solvers are deterministic, so the
 * recomputed segments equal the stored ones bit for bit, and the sums over
 * segment pairs run in the same order as in full storage.
 *
 * Memory: the checkpoints (backend checkpoint memory, pinned host memory on
 * CUDA) plus checkpoint_interval + 4 work fields
 * (see ComputationReduceMemoryBase).
 *
 * @see ComputationContinuous for the full-storage computation
 */

#ifndef COMPUTATION_REDUCE_MEMORY_CONTINUOUS_H_
#define COMPUTATION_REDUCE_MEMORY_CONTINUOUS_H_

#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "ComputationReduceMemoryBase.h"

template <typename T>
class ComputationReduceMemoryContinuous : public ComputationReduceMemoryBase<T>
{
private:
    std::string method;  ///< "pseudospectral" or "realspace"

    /// One segment per polymer for its partition function: (polymer, key, n, key†, n_aggregated)
    std::vector<std::tuple<int, std::string, int, std::string, int>> single_partition_segment;

protected:
    int propagator_ds_index(const std::string& key) const override;

public:
    ComputationReduceMemoryContinuous(ComputationBox<T>* cb, Molecules* molecules,
                                      PropagatorComputationOptimizer* propagator_computation_optimizer,
                                      std::string method, SpaceGroup* space_group,
                                      const std::function<std::unique_ptr<ComputationBackend<T>>()>& create_backend);

    void compute_propagators(std::map<std::string, const T*> w_input,
                             std::map<std::string, const T*> q_init = {}) override;
    void advance_propagator_single_segment(T* q_init, T* q_out, int p, int v, int u) override;
    void compute_concentrations() override;
    void compute_stress() override;
    void get_chain_propagator(T* q_out, int polymer, int v, int u, int n) override;
    std::vector<double> compute_partition_errors() override;
};

#endif
