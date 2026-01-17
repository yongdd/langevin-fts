/**
 * @file CudaComputationReduceMemoryBase.h
 * @brief Common base class for memory-efficient GPU propagator computation.
 *
 * This header provides CudaComputationReduceMemoryBase, a common base class
 * that consolidates shared functionality between CudaComputationReduceMemoryContinuous
 * and CudaComputationReduceMemoryDiscrete, including:
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
 * @see CudaComputationReduceMemoryContinuous for continuous chain implementation
 * @see CudaComputationReduceMemoryDiscrete for discrete chain implementation
 */

#ifndef CUDA_COMPUTATION_REDUCE_MEMORY_BASE_H_
#define CUDA_COMPUTATION_REDUCE_MEMORY_BASE_H_

#include <string>
#include <vector>
#include <map>
#include <array>

#include "ComputationBox.h"
#include "Polymer.h"
#include "Molecules.h"
#include "PropagatorComputationOptimizer.h"
#include "PropagatorComputation.h"
#include "CudaCommon.h"
#include "CudaSolver.h"
#include "Scheduler.h"

/**
 * @class CudaComputationReduceMemoryBase
 * @brief Common base class for memory-efficient GPU propagator computation.
 *
 * Consolidates shared code between continuous and discrete chain
 * memory-efficient GPU computation classes, including concentration queries
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
class CudaComputationReduceMemoryBase : public PropagatorComputation<T>
{
protected:
    CudaSolver<T> *propagator_solver;  ///< PDE solver
    Scheduler *sc;                      ///< Propagator execution scheduler
    int n_streams;                      ///< Number of parallel streams

    cudaStream_t streams[MAX_STREAMS][2];  ///< CUDA streams [kernel, memcpy]

    /// @name Minimal GPU Workspace
    /// @{
    CuDeviceData<T> *d_q_one[MAX_STREAMS][2];              ///< Current/next propagator
    CuDeviceData<T> *d_propagator_sub_dep[MAX_STREAMS][2]; ///< Sub-dependency workspace
    CuDeviceData<T> *d_q_unity;                            ///< Unity array for initialization
    double *d_q_mask;                                       ///< Mask for impenetrable regions
    /// @}

    /// @name Recalculation Workspace
    /// @{
    int total_max_n_segment;    ///< Total sum of segments across all propagators
    int checkpoint_interval;    ///< Checkpoint interval (sqrt(total_N) for optimal memory-computation tradeoff)
    std::vector<T*> q_recal;    ///< Recalculation temporary (size: checkpoint_interval+3)
    /// @}

    /// @name Checkpoint Storage Mode
    /// @{
    /**
     * @brief If true, store checkpoints in GPU global memory; otherwise use pinned host memory.
     *
     * When use_device_checkpoint_memory is true:
     * - Checkpoints are stored in GPU global memory (cudaMalloc)
     * - Faster access from GPU kernels, no host-device transfers needed
     * - Uses more GPU memory
     *
     * When use_device_checkpoint_memory is false (default):
     * - Checkpoints are stored in pinned host memory (cudaMallocHost)
     * - Enables async transfers to overlap with computation
     * - Uses less GPU memory at the cost of PCIe transfer overhead
     */
    bool use_device_checkpoint_memory;

    /**
     * @brief Allocate checkpoint memory (device or pinned based on use_device_checkpoint_memory).
     *
     * @param[out] ptr   Pointer to receive allocated memory
     * @param[in]  count Number of elements to allocate
     */
    void alloc_checkpoint_memory(T** ptr, size_t count);

    /**
     * @brief Free checkpoint memory (device or pinned based on use_device_checkpoint_memory).
     *
     * @param[in] ptr Pointer to free
     */
    void free_checkpoint_memory(T* ptr);
    /// @}

    /**
     * @brief Shared GPU workspace buffers (2×M each).
     *
     * Memory layout for d_workspace[0] (2×M contiguous):
     *   [0, M):    d_propagator_sub_dep[0][0], used as q_left in concentration
     *   [M, 2M):   d_propagator_sub_dep[0][1], used as q_right in concentration
     *
     * Memory layout for d_workspace[1] (2×M contiguous):
     *   [0, M):    d_phi (concentration output)
     *   [M, 2M):   unused or q_right ping-pong buffer
     */
    CuDeviceData<T> *d_workspace[2];
    CuDeviceData<T> *d_phi;  ///< Concentration output pointer

    /**
     * @brief Checkpointed propagator values in pinned host memory.
     *
     * Key: (propagator_key, checkpoint_index)
     * Value: Propagator array at checkpoint (size n_grid)
     */
    std::map<std::tuple<std::string, int>, T *> propagator_at_check_point;

    #ifndef NDEBUG
    /**
     * @brief Debug: track which propagator steps are computed.
     */
    std::map<std::string, bool *> propagator_finished;
    #endif

    /**
     * @brief Block concentration fields in pinned host memory.
     *
     * Key: (polymer_id, key_left, key_right) with key_left <= key_right
     * Value: Concentration array (size n_grid)
     */
    std::map<std::tuple<int, std::string, std::string>, T *> phi_block;

    /**
     * @brief Solvent concentration fields in pinned host memory.
     *
     * One array per solvent species.
     */
    std::vector<T *> phi_solvent;

public:
    /**
     * @brief Construct GPU reduce-memory computation base.
     *
     * @param cb                              Computation box
     * @param molecules                       Molecules container
     * @param propagator_computation_optimizer Propagator optimizer with dependency info
     * @param use_device_checkpoint_memory    If true, store checkpoints in GPU global memory;
     *                                        otherwise use pinned host memory (default: false)
     */
    CudaComputationReduceMemoryBase(ComputationBox<T>* cb, Molecules *molecules,
                                    PropagatorComputationOptimizer* propagator_computation_optimizer,
                                    bool use_device_checkpoint_memory = false);

    /**
     * @brief Virtual destructor.
     */
    virtual ~CudaComputationReduceMemoryBase() {}

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
     */
    void advance_propagator_single_segment(T* q_init, T *q_out, std::string monomer_type) override;

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
