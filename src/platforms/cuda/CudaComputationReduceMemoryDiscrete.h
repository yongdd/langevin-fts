/**
 * @file CudaComputationReduceMemoryDiscrete.h
 * @brief Memory-efficient GPU propagator computation for discrete chains.
 *
 * This header provides CudaComputationReduceMemoryDiscrete, a GPU
 * implementation that minimizes GPU memory usage by storing propagators
 * only at checkpoint positions in pinned host memory.
 *
 * **Checkpointing Strategy:**
 *
 * Instead of storing propagators at every segment:
 * 1. Store propagators only at checkpoint intervals (dependency points)
 * 2. Recompute intermediate steps on-the-fly during concentration calculation
 *
 * **Memory vs Time Tradeoff:**
 *
 * - Standard: O(N × M) memory, O(N × M) computation
 * - Checkpointing: O(√N × M) memory, O(N × √N × M) computation
 *
 * where N is segment count and M is grid points.
 *
 * **Discrete Chain Specifics:**
 *
 * - Half-bond steps at junction points
 * - exp(-w) weighting in concentration calculation
 * - Discrete sum instead of Simpson's rule
 *
 * @see PropagatorComputation for the abstract interface
 * @see CudaComputationDiscrete for full GPU memory version
 * @see CpuComputationReduceMemoryDiscrete for CPU version with same strategy
 */

#ifndef CUDA_PSEUDO_REDUCE_MEMORY_DISCRETE_H_
#define CUDA_PSEUDO_REDUCE_MEMORY_DISCRETE_H_

#include <array>
#include <cufft.h>

#include "ComputationBox.h"
#include "Polymer.h"
#include "Molecules.h"
#include "CudaComputationReduceMemoryBase.h"
#include "CudaCommon.h"
#include "CudaSolver.h"
#include "Scheduler.h"
#include "SpaceGroup.h"

/**
 * @class CudaComputationReduceMemoryDiscrete
 * @brief Memory-efficient GPU computation for discrete chains using checkpointing.
 *
 * Reduces memory usage by storing only checkpoint propagators and
 * recomputing intermediate values during concentration calculation.
 *
 * @tparam T Numeric type (double or std::complex<double>)
 *
 * **Checkpoint Placement:**
 *
 * Checkpoints are placed at dependency points (junctions, chain ends)
 * to minimize storage while enabling recomputation.
 *
 * **Recomputation Process:**
 *
 * During concentration calculation:
 * 1. Load propagator from nearest checkpoint
 * 2. Recompute forward to required segment position
 * 3. Use for concentration contribution
 * 4. Repeat for next position
 *
 * @note This version is slower but uses significantly less memory.
 *       Use for large-scale 3D simulations with discrete chains.
 */
template <typename T>
class CudaComputationReduceMemoryDiscrete : public CudaComputationReduceMemoryBase<T>
{
private:
    std::array<T*,2> q_pair;  ///< Ping-pong buffers for propagator advancement

    /**
     * @brief Half-bond step propagators at checkpoints.
     *
     * Key: (propagator_key, segment_index)
     * Value: Half-bond propagator at junction (size n_grid)
     *
     * Stores half-bond propagators at junction points.
     */
    std::map<std::tuple<std::string, int>, T *> propagator_half_steps_at_check_point;

    #ifndef NDEBUG
    std::map<std::string, std::map<int, bool>> propagator_half_steps_finished;
    #endif

    /**
     * @brief Partition function data in pinned host memory.
     *
     * Tuple: (polymer_id, q_forward, q_backward, monomer_type, n_repeated)
     * Includes monomer_type for exp(-w) factor in discrete model.
     */
    std::vector<std::tuple<int, T *, T *, std::string, int>> single_partition_segment;

    // ============== Reduced basis support ==============
    /**
     * @brief Device array for full_to_reduced_map.
     * Maps each full grid point to its irreducible index.
     */
    int* d_full_to_reduced_map_;

    /**
     * @brief Device array for reduced_basis_indices.
     * Flat indices of irreducible mesh points.
     */
    int* d_reduced_basis_indices_;

    /**
     * @brief Temporary device buffers for full grid propagator.
     * d_q_full_[0] = input, d_q_full_[1] = output
     * Only allocated when space group is set.
     */
    CuDeviceData<T>* d_q_full_[2];

    /**
     * @brief Buffer for expanding concentration to full grid (device).
     */
    CuDeviceData<T>* d_phi_full_buffer_;

    /**
     * @brief Compute concentration for one block with recomputation.
     *
     * Recomputes propagators from checkpoints as needed.
     * Uses discrete sum instead of integral.
     *
     * @param phi          Output concentration (pinned host)
     * @param key_left     Left propagator key
     * @param key_right    Right propagator key
     * @param N_LEFT       Left contour index
     * @param N_RIGHT      Right contour index
     * @param monomer_type Monomer type
     */
    void calculate_phi_one_block(T *phi, std::string key_left, std::string key_right,
        const int N_LEFT, const int N_RIGHT, std::string monomer_type);

    /**
     * @brief Recalculate propagator from checkpoint.
     *
     * Starting from nearest checkpoint, advances propagator to
     * the required segment positions.
     *
     * @param key          Propagator key
     * @param N_START      Starting segment (checkpoint position)
     * @param N_RIGHT      Ending segment position
     * @param monomer_type Monomer type
     *
     * @return Vector of propagator pointers for requested range
     */
    std::vector<T*> recalculate_propagator(std::string key, const int N_START, const int N_RIGHT, std::string monomer_type);

public:
    /**
     * @brief Construct memory-efficient GPU computation for discrete chains.
     *
     * Allocates minimal GPU workspace and checkpoint buffers in pinned host memory.
     *
     * @param cb          Computation box
     * @param molecules   Molecules container
     * @param propagator_computation_optimizer Optimization strategy
     */
    CudaComputationReduceMemoryDiscrete(ComputationBox<T>* cb, Molecules *molecules,
        PropagatorComputationOptimizer *propagator_computation_optimizer, SpaceGroup* space_group = nullptr);

    /** @brief Destructor. Frees GPU and pinned memory. */
    ~CudaComputationReduceMemoryDiscrete();

    /**
     * @brief Compute propagators storing only checkpoints.
     *
     * Includes half-bond propagators at junctions.
     *
     * @param w_block Potential fields (device)
     * @param q_init  Optional initial conditions
     */
    void compute_propagators(
        std::map<std::string, const T*> w_block,
        std::map<std::string, const T*> q_init = {}) override;

    /** @brief Compute concentrations with on-the-fly recomputation. */
    void compute_concentrations() override;

    /** @brief Compute stress (requires recomputation). */
    void compute_stress() override;

    /**
     * @brief Get partition function with exp(-w) normalization.
     * @param polymer Polymer index
     * @return log(Q) normalized by volume
     */
    T get_total_partition(int polymer) override;

    /**
     * @brief Extract chain propagator (requires stored checkpoint).
     *
     * @param q_out   Output array (host)
     * @param polymer Polymer index
     * @param v, u    Block indices
     * @param n       Contour step
     */
    void get_chain_propagator(T *q_out, int polymer, int v, int u, int n) override;

    /** @brief Verify partition function consistency. */
    bool check_total_partition() override;
};

#endif
