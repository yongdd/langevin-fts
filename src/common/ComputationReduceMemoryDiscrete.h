/**
 * @file ComputationReduceMemoryDiscrete.h
 * @brief Propagator computation for discrete chains, reduced memory.
 *
 * Same physics and arithmetic as ComputationDiscrete, but q(r, n) is kept
 * only at checkpoints (see ComputationReduceMemoryBase) and recomputed from
 * the last checkpoint before it when needed; the sums over segment pairs run
 * in the same order as in full storage.
 *
 * The half-bond states q(r, n+1/2) that junctions need are always kept, as
 * half_checkpoint(key, n): the junction value a propagator starts from
 * (n = 0) and the states half a bond past the segments in junction_ends.
 *
 * @see ComputationDiscrete for the full-storage computation
 */

#ifndef COMPUTATION_REDUCE_MEMORY_DISCRETE_H_
#define COMPUTATION_REDUCE_MEMORY_DISCRETE_H_

#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "ComputationReduceMemoryBase.h"

template <typename T>
class ComputationReduceMemoryDiscrete : public ComputationReduceMemoryBase<T>
{
private:
    /// q(r, n+1/2) of propagator key, for n = 0 and the junction ends
    std::map<std::tuple<std::string, int>, CheckpointArray<T>*> half_checkpoint;

    /// One segment per polymer for its partition function: (polymer, key, n, key†, monomer type, n_aggregated)
    std::vector<std::tuple<int, std::string, int, std::string, std::string, int>> single_partition_segment;

    void add_half_checkpoint(const std::string& key, int n);
    void free_half_checkpoints();
    CheckpointArray<T>* half_checkpoint_at(const std::string& key, int n);

protected:
    int propagator_ds_index(const std::string&) const override { return 0; }

public:
    ComputationReduceMemoryDiscrete(ComputationBox<T>* cb, Molecules* molecules,
                                    PropagatorComputationOptimizer* propagator_computation_optimizer,
                                    SpaceGroup* space_group,
                                    const std::function<std::unique_ptr<ComputationBackend<T>>()>& create_backend);
    ~ComputationReduceMemoryDiscrete();

    void compute_propagators(std::map<std::string, const T*> w_input,
                             std::map<std::string, const T*> q_init = {}) override;
    void advance_propagator_single_segment(T* q_init, T* q_out, int p, int v, int u) override;
    void compute_concentrations() override;
    void compute_stress() override;
    void get_chain_propagator(T* q_out, int polymer, int v, int u, int n) override;
    bool check_total_partition() override;
};

#endif
