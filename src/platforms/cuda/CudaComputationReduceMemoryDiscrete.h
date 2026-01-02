/**
 * @file CudaComputationReduceMemoryDiscrete.h
 * @brief Memory-efficient GPU propagator computation for discrete chains.
 *
 * This header provides CudaComputationReduceMemoryDiscrete, a GPU
 * implementation that minimizes GPU memory by storing propagators
 * in pinned host memory for discrete chain models.
 *
 * **Memory-Saving Strategy:**
 *
 * Similar to CudaComputationReduceMemoryContinuous but handles:
 * - Half-bond propagators at junctions
 * - exp(-w) weighting in concentration calculation
 * - Half-bond stress contributions
 *
 * **Storage Location:**
 *
 * Host (pinned) memory:
 * - propagator: Full propagator history
 * - propagator_half_steps: Half-bond values at junctions
 * - phi_block: Concentrations
 *
 * Device memory (minimal):
 * - d_q_one, d_q_block_v/u: Workspace for current computation
 *
 * **Trade-offs:**
 *
 * - Lower GPU memory usage
 * - PCIe transfer overhead for each propagator access
 * - Uses async streams to hide transfer latency
 *
 * @see PropagatorComputation for the abstract interface
 * @see CudaComputationDiscrete for full GPU memory version
 * @see CudaComputationReduceMemoryContinuous for continuous chain version
 */

#ifndef CUDA_PSEUDO_REDUCE_MEMORY_DISCRETE_H_
#define CUDA_PSEUDO_REDUCE_MEMORY_DISCRETE_H_

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
 * @class CudaComputationReduceMemoryDiscrete
 * @brief Memory-efficient GPU computation for discrete chains.
 *
 * Stores propagators in pinned host memory with minimal GPU workspace.
 * Handles half-bond steps and exp(-w) weighting for discrete model.
 *
 * @tparam T Numeric type (double or std::complex<double>)
 *
 * **Propagator Storage:**
 *
 * All propagators stored in pinned host memory:
 * - propagator[key]: Host pointers to full-step values
 * - propagator_half_steps[key]: Host pointers to half-bond values
 *
 * **Stress Computation:**
 *
 * Uses block_stress_computation_plan to track which segments
 * have half-bond vs full-bond contributions.
 */
template <typename T>
class CudaComputationReduceMemoryDiscrete : public PropagatorComputation<T>
{
private:
    CudaSolver<T> *propagator_solver;  ///< Pseudo-spectral PDE solver

    int n_streams;  ///< Number of parallel streams

    cudaStream_t streams[MAX_STREAMS][2];  ///< CUDA streams [kernel, memcpy]

    CuDeviceData<T> *d_q_unity;  ///< Unity array for initialization

    double *d_q_mask;  ///< Mask for impenetrable regions

    CuDeviceData<T> *d_q_pair[MAX_STREAMS][2];  ///< Stress workspace

    /// @name Minimal GPU Workspace
    /// @{
    CuDeviceData<T> *d_q_one[MAX_STREAMS][2];              ///< Current/next propagator
    CuDeviceData<T> *d_propagator_sub_dep[MAX_STREAMS][2]; ///< Sub-dependency workspace
    /// @}

    /// @name Concentration Computation Workspace
    /// @{
    CuDeviceData<T> *d_q_block_v[2];  ///< Forward propagator [prev, next]
    CuDeviceData<T> *d_q_block_u[2];  ///< Backward propagator [prev, next]
    CuDeviceData<T> *d_phi;           ///< Temporary concentration
    /// @}

    Scheduler *sc;  ///< Propagator execution scheduler

    /// @name Propagator Storage (Pinned Host Memory)
    /// @{
    /** @brief Full-step propagators q(r,n) in pinned host memory. */
    std::map<std::string, T **> propagator;

    /** @brief Half-bond propagators q(r,n+1/2) in pinned host memory. */
    std::map<std::string, T **> propagator_half_steps;

    /** @brief Propagator array sizes for deallocation. */
    std::map<std::string, int> propagator_size;
    /// @}

    #ifndef NDEBUG
    std::map<std::string, bool *> propagator_finished;
    std::map<std::string, std::map<int, bool>> propagator_half_steps_finished;
    #endif

    /**
     * @brief Partition function data in pinned host memory.
     *
     * Tuple: (polymer_id, q_forward, q_backward, monomer_type, n_repeated)
     */
    std::vector<std::tuple<int, T *, T *, std::string, int>> single_partition_segment;

    /** @brief Block concentrations in pinned host memory. */
    std::map<std::tuple<int, std::string, std::string>, T *> phi_block;

    /**
     * @brief Stress computation plan for discrete chains.
     *
     * Key: (polymer_id, key_left, key_right)
     * Value: vector of (q_forward, q_backward, is_half_bond_length)
     */
    std::map<std::tuple<int, std::string, std::string>,
        std::vector<std::tuple<T *, T *, bool>>> block_stress_computation_plan;

    std::vector<T *> phi_solvent;  ///< Solvent concentrations (pinned)

    /**
     * @brief Compute concentration for one block (discrete model).
     *
     * Includes exp(-w) weighting in the integration.
     *
     * @param phi       Output concentration (pinned host)
     * @param q_1       Forward propagator array (pinned host)
     * @param q_2       Backward propagator array (pinned host)
     * @param d_exp_dw  Boltzmann factor (device)
     * @param N_LEFT    Left segment count
     * @param N_RIGHT   Right segment count
     * @param NORM      Normalization factor
     */
    void calculate_phi_one_block(T *phi, T **q_1, T **q_2,
        CuDeviceData<T> *d_exp_dw, const int N_LEFT, const int N_RIGHT, const T NORM);

public:
    /**
     * @brief Construct memory-efficient GPU computation for discrete chains.
     *
     * Allocates minimal GPU workspace and pinned host buffers.
     *
     * @param cb          Computation box
     * @param molecules   Molecules container
     * @param propagator_computation_optimizer Optimization strategy
     */
    CudaComputationReduceMemoryDiscrete(ComputationBox<T>* cb, Molecules *molecules,
        PropagatorComputationOptimizer *propagator_computation_optimizer);

    /** @brief Destructor. Frees GPU and pinned memory. */
    ~CudaComputationReduceMemoryDiscrete();

    /** @brief Update half-bond diffusion operators. */
    void update_laplacian_operator() override;

    /**
     * @brief Compute propagators storing to pinned host memory.
     *
     * Includes half-bond propagators at junctions.
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

    /** @brief Compute concentrations with exp(-w) weighting. */
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

    /** @brief Compute stress with half-bond contributions. */
    void compute_stress() override;

    /**
     * @brief Get partition function with exp(-w) normalization.
     * @param polymer Polymer index
     * @return log(Q) normalized by volume
     */
    T get_total_partition(int polymer) override;

    /**
     * @brief Extract chain propagator from pinned host memory.
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
