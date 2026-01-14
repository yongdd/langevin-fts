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
#include "PropagatorComputation.h"
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
class CudaComputationReduceMemoryContinuous : public PropagatorComputation<T>
{
private:
    CudaSolver<T> *propagator_solver;  ///< PDE solver
    std::string method;                 ///< Solver method ("pseudo" or "real")

    int n_streams;  ///< Number of parallel streams

    cudaStream_t streams[MAX_STREAMS][2];  ///< CUDA streams [kernel, memcpy]

    /// @name Minimal GPU Workspace
    /// @{
    CuDeviceData<T> *d_q_one[MAX_STREAMS][2];              ///< Current/next propagator
    CuDeviceData<T> *d_propagator_sub_dep[MAX_STREAMS][2]; ///< Sub-dependency workspace
    /// @}

    /// @name Recalculation Workspace
    /// @{
    int total_max_n_segment;    ///< Total sum of segments across all propagators
    int checkpoint_interval;    ///< Checkpoint interval (sqrt(total_N) for optimal memory-computation tradeoff)
    std::vector<T*> q_recal;    ///< Recalculation temporary (size: checkpoint_interval+3 for block-based computation)
    /// @}

    /**
     * @brief Checkpoint storage in pinned host memory.
     *
     * Key: (propagator_key, segment_index)
     * Value: Propagator at checkpoint (pinned host)
     */
    std::map<std::tuple<std::string, int>, T *> propagator_at_check_point;

    CuDeviceData<T> *d_q_unity;  ///< Unity array for initialization

    double *d_q_mask;  ///< Mask for impenetrable regions

    /**
     * @brief Shared GPU workspace buffers (2×M each).
     *
     * Memory layout for d_workspace[0] (2×M contiguous):
     *   [0, M):    d_propagator_sub_dep[0][0], used as q_left in concentration
     *   [M, 2M):   d_propagator_sub_dep[0][1], used as q_right in concentration
     *
     * Memory layout for d_workspace[1] (2×M contiguous):
     *   [0, M):    q_right ping-pong buffer in concentration
     *   [M, 2M):   d_phi (concentration output)
     *
     * In stress computation, d_workspace[0] and d_workspace[1] are used
     * as contiguous 2×M buffers for batch FFT.
     */
    CuDeviceData<T> *d_workspace[2];
    CuDeviceData<T> *d_phi;  ///< = d_workspace[1]+M, concentration output

    Scheduler *sc;  ///< Propagator execution scheduler

    #ifndef NDEBUG
    std::map<std::string, bool *> propagator_finished;  ///< Debug tracking
    #endif

    /**
     * @brief Partition function data in pinned host memory.
     *
     * Tuple: (polymer_id, q_forward, q_backward, n_repeated)
     */
    std::vector<std::tuple<int, T *, T *, int>> single_partition_segment;

    /**
     * @brief Block concentrations in pinned host memory.
     *
     * Key: (polymer_id, key_left, key_right)
     */
    std::map<std::tuple<int, std::string, std::string>, T *> phi_block;

    std::vector<T *> phi_solvent;  ///< Solvent concentrations (pinned)

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

    /** @brief Update operators for new box dimensions. */
    void update_laplacian_operator() override;

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

    /**
     * @brief Advance single segment (utility).
     *
     * @param q_init       Input propagator
     * @param q_out        Output propagator
     * @param monomer_type Monomer type
     */
    void advance_propagator_single_segment(T* q_init, T *q_out, std::string monomer_type) override;

    /** @brief Compute concentrations using recalculation. */
    void compute_concentrations() override;

    /**
     * @brief Complete SCFT statistics with memory efficiency.
     *
     * @param w_input Potential fields
     * @param q_init  Optional initial conditions
     */
    void compute_statistics(
        std::map<std::string, const T*> w_input,
        std::map<std::string, const T*> q_init = {}) override;

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

    /// @name Canonical Ensemble Concentrations
    /// @{
    void get_total_concentration(std::string monomer_type, T *phi) override;
    void get_total_concentration(int polymer, std::string monomer_type, T *phi) override;
    void get_block_concentration(int polymer, T *phi) override;
    /// @}

    /// @name Solvent Methods
    /// @{
    T get_solvent_partition(int s) override;
    void get_solvent_concentration(int s, T *phi) override;
    /// @}

    /** @brief Grand canonical concentration. */
    void get_total_concentration_gce(double fugacity, int polymer,
        std::string monomer_type, T *phi) override;

    /** @brief Verify partition function consistency. */
    bool check_total_partition() override;

};

#endif
