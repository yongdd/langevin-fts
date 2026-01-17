/**
 * @file CudaComputationReduceMemoryContinuous.h
 * @brief Memory-efficient GPU propagator computation for continuous chains.
 *
 * This header provides CudaComputationReduceMemoryContinuous, a GPU
 * implementation that minimizes GPU memory usage by storing propagators
 * in pinned host memory and using checkpointing for recalculation.
 *
 * **Memory-Saving Strategy:**
 *
 * Instead of storing all propagator values in GPU memory:
 * - Store checkpoints at intervals in pinned host memory
 * - Recalculate intermediate values as needed
 * - Use async transfers to overlap computation and memory copy
 *
 * **Async Transfer Optimization:**
 *
 * Two CUDA streams per propagator:
 * - streams[i][0]: Kernel execution
 * - streams[i][1]: Host-device memory transfers
 *
 * Overlapping hides PCIe transfer latency. See supporting information
 * of Macromolecules 2021, 54, 24, 11304 for details.
 *
 * **Trade-offs:**
 *
 * - GPU memory: O(1) per propagator instead of O(N_segments)
 * - Computation: 2-4× slower due to recalculation
 * - Suitable for large grids or long chains where GPU memory is limited
 *
 * @see PropagatorComputation for the abstract interface
 * @see CudaComputationContinuous for full GPU memory version
 * @see CpuComputationReduceMemoryContinuous for CPU version
 */

#ifndef CUDA_PSEUDO_REDUCE_MEMORY_CONTINUOUS_H_
#define CUDA_PSEUDO_REDUCE_MEMORY_CONTINUOUS_H_

#include <array>
#include <cufft.h>

#include "ComputationBox.h"
#include "Polymer.h"
#include "Molecules.h"
#include "CudaComputationReduceMemoryBase.h"
#include "CudaCommon.h"
#include "CudaSolver.h"
#include "Scheduler.h"

/**
 * @class CudaComputationReduceMemoryContinuous
 * @brief Memory-efficient GPU computation for continuous chains.
 *
 * Uses checkpointing and recalculation to minimize GPU memory footprint.
 * Propagators stored in pinned host memory with async transfers.
 *
 * @tparam T Numeric type (double or std::complex<double>)
 *
 * **Checkpointing:**
 *
 * Stores propagator values at regular intervals (checkpoints):
 * - propagator_at_check_point[(key, n)]: Checkpoint values
 * - q_recal: Workspace for recalculating from checkpoints
 *
 * **Memory Layout:**
 *
 * Device memory (minimal):
 * - d_q_one[stream][2]: Current and next propagator values
 * - d_q_block_v/u[2]: Concentration computation workspace
 *
 * Host memory (pinned):
 * - phi_block: Concentrations
 * - single_partition_segment: Partition function data
 */
template <typename T>
class CudaComputationReduceMemoryContinuous : public CudaComputationReduceMemoryBase<T>
{
private:
    std::string method;  ///< Solver method ("pseudospectral" or "realspace")

    /**
     * @brief Partition function data in pinned host memory.
     *
     * Tuple: (polymer_id, q_forward, q_backward, n_repeated)
     */
    std::vector<std::tuple<int, T *, T *, int>> single_partition_segment;

    /**
     * @brief Compute concentration for one block.
     *
     * Uses recalculated propagators from checkpoints.
     *
     * @param phi         Output concentration (pinned host)
     * @param key_left    Left propagator key
     * @param key_right   Right propagator key
     * @param N_LEFT      Left segment count
     * @param N_RIGHT     Right segment count
     * @param monomer_type Monomer type
     * @param NORM        Normalization factor
     */
    void calculate_phi_one_block(T *phi, std::string key_left, std::string key_right,
        const int N_LEFT, const int N_RIGHT, std::string monomer_type, const T NORM);

    /**
     * @brief Recalculate propagator from nearest checkpoint.
     *
     * @param key          Propagator key
     * @param N_START      Starting segment (from checkpoint)
     * @param N_RIGHT      Ending segment
     * @param monomer_type Monomer type
     * @return Vector of recalculated propagator pointers
     */
    std::vector<T*> recalcaulte_propagator(std::string key,
        const int N_START, const int N_RIGHT, std::string monomer_type);

public:
    /**
     * @brief Construct memory-efficient GPU computation.
     *
     * Allocates minimal GPU workspace and pinned host buffers.
     *
     * @param cb          Computation box
     * @param pc          Molecules container
     * @param propagator_computation_optimizer Optimization strategy
     * @param method      Solver method ("pseudospectral" or "realspace")
     * @param numerical_method Numerical algorithm:
     *                         - For pseudospectral: "rqm4" or "etdrk4"
     *                         - For realspace: "cn-adi2" or "cn-adi4-lr"
     */
    CudaComputationReduceMemoryContinuous(ComputationBox<T>* cb, Molecules *pc,
        PropagatorComputationOptimizer *propagator_computation_optimizer, std::string method, std::string numerical_method = "");

    /** @brief Destructor. Frees GPU and pinned memory. */
    ~CudaComputationReduceMemoryContinuous();

    /**
     * @brief Compute propagators with checkpointing.
     *
     * Stores only checkpoints to pinned memory, not full history.
     *
     * @param w_block Potential fields (device)
     * @param q_init  Optional initial conditions
     */
    void compute_propagators(
        std::map<std::string, const T*> w_block,
        std::map<std::string, const T*> q_init = {}) override;

    /** @brief Compute concentrations using recalculation. */
    void compute_concentrations() override;

    /** @brief Compute stress using recalculated propagators. */
    void compute_stress() override;

    /**
     * @brief Get partition function.
     * @param polymer Polymer index
     * @return log(Q) normalized by volume
     */
    T get_total_partition(int polymer) override;

    /**
     * @brief Extract chain propagator (requires recalculation).
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
