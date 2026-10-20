/**
 * @file ComputationReduceMemoryBase.h
 * @brief Checkpoint storage shared by the reduce-memory propagator computations.
 *
 * The reduce-memory computations keep a propagator q(r, n) only at
 * checkpoints and recompute the other segments from the last checkpoint
 * before them. The solvers are deterministic, so a recomputed segment equals
 * the one the propagator computation produced, bit for bit.
 *
 * Checkpoints: the start of every propagator, the segments other propagators
 * start from, the start of every scheduled job, every checkpoint_interval
 * segments, and the segments of the partition functions. The derived class
 * adds the ones specific to its chain model. The interval is
 * ⌈2√(total segments)⌉, which balances the checkpoints against the
 * recomputation workspace.
 *
 * Work fields: q_recal (up to checkpoint_interval segments of one
 * propagator, no more than the longest block needs), q_work and q_right
 * (two each).
 *
 * Propagators are computed one at a time on stream 0.
 *
 * @see ComputationReduceMemoryContinuous, ComputationReduceMemoryDiscrete
 */

#ifndef COMPUTATION_REDUCE_MEMORY_BASE_H_
#define COMPUTATION_REDUCE_MEMORY_BASE_H_

#include <functional>
#include <memory>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#include "ComputationCommon.h"

template <typename T>
class ComputationReduceMemoryBase : public ComputationCommon<T>
{
protected:
    /// First stored segment of a propagator: 0 for continuous chains, 1 for discrete chains
    const int first_segment;
    int checkpoint_interval;  ///< Segments between regular checkpoints

    /// q(r, n) of propagator key at the checkpointed segments n
    std::map<std::tuple<std::string, int>, CheckpointArray<T>*> checkpoint;

    std::vector<BackendArray<T>*> q_recal;           ///< Recomputed consecutive segments of one propagator
    BackendArray<T>* q_work[2]  = {nullptr, nullptr}; ///< Propagator being advanced (ping-pong)
    BackendArray<T>* q_right[2] = {nullptr, nullptr}; ///< The other propagator of a block, advanced along it

    /// Free the checkpoints and the work fields (also on a failed construction).
    void free_storage();

    /**
     * @param first_segment 0 for continuous chains, 1 for discrete chains
     * @param create_backend See ComputationCommon
     */
    ComputationReduceMemoryBase(ComputationBox<T>* cb, Molecules* molecules,
                                PropagatorComputationOptimizer* propagator_computation_optimizer,
                                SpaceGroup* space_group, int first_segment,
                                const std::function<std::unique_ptr<ComputationBackend<T>>()>& create_backend);

    /// Contour step index used to advance propagator key (discrete chains: 0).
    virtual int propagator_ds_index(const std::string& key) const = 0;

    /// Allocate the checkpoint (key, n) if it does not exist yet.
    void add_checkpoint(const std::string& key, int n);
    CheckpointArray<T>* checkpoint_at(const std::string& key, int n);

    /// Add the regular checkpoints; call after the model-specific ones.
    void add_regular_checkpoints();

    /// q_recal[i] = q(r, n_first + i) of propagator key for i = 0..count-1.
    void recompute_segments(const std::string& key, int n_first, int count);

    /**
     * @brief Walk the segment pairs of a block in order.
     *
     * Calls visit(n, q(left_offset - n), q†(n)) for n = n_first..n_last, where
     * q is propagator key_left and q† is key_right. q† is advanced from its
     * checkpoint at n_first; q is recomputed from checkpoints in chunks of
     * checkpoint_interval segments. A segment of q below first_segment is
     * passed as nullptr.
     */
    void for_each_segment_pair(const std::string& key_left, const std::string& key_right,
                               int left_offset, int n_first, int n_last,
                               const std::function<void(int, BackendArray<T>*, BackendArray<T>*)>& visit);

public:
    virtual ~ComputationReduceMemoryBase();
};

#endif
