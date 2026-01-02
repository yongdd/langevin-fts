/**
 * @file CudaComputationDiscrete.h
 * @brief GPU propagator computation for discrete chain model.
 *
 * This header provides CudaComputationDiscrete, the GPU implementation
 * of propagator computation for discrete chain models. Stores all
 * propagator values including half-bond steps in GPU device memory.
 *
 * **Discrete Chain Model:**
 *
 * Full segment step:
 *     q(n+1) = B^(1/2) · exp(-w) · B^(1/2) · q(n)
 *
 * Half-bond step (for junction matching):
 *     q'(n+1/2) = B^(1/2) · q(n)
 *
 * **Additional Storage for Discrete Model:**
 *
 * - d_propagator_half_steps: Half-bond propagators at block junctions
 * - block_stress_computation_plan: Tracks half-bond contributions to stress
 *
 * **Concentration Formula:**
 *
 * For discrete chains:
 *     φ(r) = Σ_n q_forward(n) · exp(-w) · q_backward(N-n-1)
 *
 * (differs from continuous model by exp(-w) factor)
 *
 * @see PropagatorComputation for the abstract interface
 * @see CudaComputationContinuous for continuous chain version
 * @see CudaComputationReduceMemoryDiscrete for memory-efficient version
 */

#ifndef CUDA_COMPUTATION_DISCRETE_H_
#define CUDA_COMPUTATION_DISCRETE_H_

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
 * @class CudaComputationDiscrete
 * @brief GPU propagator computation for discrete chains.
 *
 * Implements discrete chain propagator updates with half-bond steps
 * for proper junction matching at branch points.
 *
 * @tparam T Numeric type (double or std::complex<double>)
 *
 * **Half-Bond Storage:**
 *
 * In addition to full-step propagators, stores half-bond values:
 * - d_propagator_half_steps[key][junction_idx]
 * - Required for matching at polymer branch points
 *
 * **Stress Computation:**
 *
 * Uses block_stress_computation_plan to track which segments
 * contribute half-bond vs full-bond stress terms.
 */
template <typename T>
class CudaComputationDiscrete : public PropagatorComputation<T>
{
private:
    CudaSolver<T> *propagator_solver;  ///< Pseudo-spectral PDE solver

    int n_streams;  ///< Number of parallel streams

    cudaStream_t streams[MAX_STREAMS][2];  ///< CUDA streams [kernel, memcpy]

    CuDeviceData<T> *d_q_unity;  ///< Unity array for initialization

    double *d_q_mask;  ///< Mask for impenetrable regions

    CuDeviceData<T> *d_q_pair[MAX_STREAMS][2];  ///< Workspace [prev, next]

    Scheduler *sc;  ///< Propagator execution scheduler

    /// @name Propagator Storage
    /// @{
    /** @brief Full-step propagators q(r,n). */
    std::map<std::string, CuDeviceData<T> **> d_propagator;

    /** @brief Half-bond propagators q(r,n+1/2) at junctions. */
    std::map<std::string, CuDeviceData<T> **> d_propagator_half_steps;

    /** @brief Propagator array sizes for deallocation. */
    std::map<std::string, int> propagator_size;
    /// @}

    #ifndef NDEBUG
    /** @brief Debug: track full-step completion. */
    std::map<std::string, bool *> propagator_finished;
    /** @brief Debug: track half-step completion. */
    std::map<std::string, std::map<int, bool>> propagator_half_steps_finished;
    #endif

    /**
     * @brief Single segment per polymer for partition function.
     *
     * Tuple: (polymer_id, q_forward, q_backward, monomer_type, n_repeated)
     * Includes monomer_type for exp(-w) factor in discrete model.
     */
    std::vector<std::tuple<int, CuDeviceData<T> *, CuDeviceData<T> *, std::string, int>> single_partition_segment;

    /// @name Concentration Storage
    /// @{
    /** @brief Block concentrations on device. */
    std::map<std::tuple<int, std::string, std::string>, CuDeviceData<T> *> d_phi_block;

    CuDeviceData<T> *d_phi;  ///< Temporary for computation
    /// @}

    /**
     * @brief Stress computation plan for discrete chains.
     *
     * Key: (polymer_id, key_left, key_right)
     * Value: vector of (q_forward, q_backward, is_half_bond_length)
     */
    std::map<std::tuple<int, std::string, std::string>,
        std::vector<std::tuple<CuDeviceData<T> *, CuDeviceData<T> *, bool>>> block_stress_computation_plan;

    std::vector<CuDeviceData<T> *> d_phi_solvent;  ///< Solvent concentrations

    /**
     * @brief Compute concentration for one block (discrete model).
     *
     * Includes exp(-w) factor in the integration.
     *
     * @param d_phi     Output concentration (device)
     * @param d_q_1     Forward propagator array
     * @param d_q_2     Backward propagator array
     * @param d_exp_dw  Boltzmann factor exp(-w·ds)
     * @param N_LEFT    Left segment count
     * @param N_RIGHT   Right segment count
     */
    void calculate_phi_one_block(CuDeviceData<T> *d_phi,
        CuDeviceData<T> **d_q_1, CuDeviceData<T> **d_q_2,
        CuDeviceData<T> *d_exp_dw, const int N_LEFT, const int N_RIGHT);

public:
    /**
     * @brief Construct GPU propagator computation for discrete chains.
     *
     * @param cb          Computation box
     * @param molecules   Molecules container
     * @param propagator_computation_optimizer Optimization strategy
     */
    CudaComputationDiscrete(ComputationBox<T>* cb, Molecules *molecules,
        PropagatorComputationOptimizer *propagator_computation_optimizer);

    /** @brief Destructor. Frees GPU resources. */
    ~CudaComputationDiscrete();

    /** @brief Update half-bond diffusion operators. */
    void update_laplacian_operator() override;

    /**
     * @brief Compute all propagators including half-bond steps.
     *
     * @param w_block Potential fields by monomer type (device)
     * @param q_init  Optional initial conditions (device)
     */
    void compute_propagators(
        std::map<std::string, const T*> w_block,
        std::map<std::string, const T*> q_init = {}) override;

    /**
     * @brief Advance single segment (utility).
     *
     * @param q_init       Input propagator (device)
     * @param q_out        Output propagator (device)
     * @param monomer_type Monomer type
     */
    void advance_propagator_single_segment(T* q_init, T *q_out, std::string monomer_type) override;

    /** @brief Compute concentrations with exp(-w) weighting. */
    void compute_concentrations() override;

    /**
     * @brief Complete SCFT statistics computation.
     *
     * @param w_input Potential fields (device)
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
     * @brief Extract chain propagator to host.
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
