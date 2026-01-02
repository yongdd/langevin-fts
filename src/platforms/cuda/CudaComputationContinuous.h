/**
 * @file CudaComputationContinuous.h
 * @brief GPU propagator computation for continuous Gaussian chain model.
 *
 * This header provides CudaComputationContinuous, the full GPU implementation
 * of propagator computation for continuous chains. Stores all propagator
 * values in GPU device memory for maximum performance.
 *
 * **Continuous Chain Model:**
 *
 * Solves the modified diffusion equation:
 *     ∂q/∂s = (b²/6)∇²q - w(r)q
 *
 * using pseudo-spectral or real-space methods with 4th-order accuracy.
 *
 * **Multi-Stream Parallelism:**
 *
 * Uses up to MAX_STREAMS parallel CUDA streams to compute independent
 * propagators concurrently. The Scheduler determines optimal execution order.
 *
 * **Memory Layout:**
 *
 * All propagators stored in GPU device memory:
 * - d_propagator[key]: Array of device pointers for each contour step
 * - d_phi_block[key]: Block concentrations in device memory
 *
 * @see PropagatorComputation for the abstract interface
 * @see CudaComputationDiscrete for discrete chain version
 * @see CudaComputationReduceMemoryContinuous for memory-efficient version
 */

#ifndef CUDA_COMPUTATION_CONTINUOUS_H_
#define CUDA_COMPUTATION_CONTINUOUS_H_

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
 * @class CudaComputationContinuous
 * @brief GPU propagator computation for continuous chains.
 *
 * Implements full propagator storage on GPU with multi-stream parallelism.
 * Provides maximum performance when GPU memory is sufficient.
 *
 * @tparam T Numeric type (double or std::complex<double>)
 *
 * **Propagator Storage:**
 *
 * Each propagator q(r,s) stored as array of device pointers:
 * - d_propagator[key][n]: Propagator at contour step n
 * - Key format: dependency_code + monomer_type
 *
 * **Stream Organization:**
 *
 * Two streams per propagator computation:
 * - streams[i][0]: Kernel execution
 * - streams[i][1]: Async memory transfers
 *
 * **Partition Function:**
 *
 * Computed from single_partition_segment which stores one segment
 * per polymer for efficient Q calculation.
 */
template <typename T>
class CudaComputationContinuous : public PropagatorComputation<T>
{
private:
    CudaSolver<T> *propagator_solver;  ///< PDE solver (pseudo-spectral or real-space)
    std::string method;                 ///< Solver method ("pseudo" or "real")

    int n_streams;  ///< Number of parallel streams

    cudaStream_t streams[MAX_STREAMS][2];  ///< CUDA streams [kernel, memcpy]

    CuDeviceData<T> *d_q_unity;  ///< Unity array for propagator initialization

    double *d_q_mask;  ///< Mask for impenetrable regions (nanoparticles)

    CuDeviceData<T> *d_q_pair[MAX_STREAMS][2];  ///< Workspace [prev, next] per stream

    Scheduler *sc;  ///< Execution scheduler for propagator dependencies

    /// @name Propagator Storage
    /// @{
    /**
     * @brief Propagator arrays on device.
     *
     * Key: dependency_code + monomer_type
     * Value: Array of device pointers [n_segments+1]
     */
    std::map<std::string, CuDeviceData<T> **> d_propagator;

    /** @brief Size of each propagator array for deallocation. */
    std::map<std::string, int> propagator_size;

    #ifndef NDEBUG
    /** @brief Debug: track propagator computation completion. */
    std::map<std::string, bool *> propagator_finished;
    #endif
    /// @}

    /**
     * @brief Single segment per polymer for partition function.
     *
     * Tuple: (polymer_id, q_forward, q_backward, n_repeated)
     */
    std::vector<std::tuple<int, CuDeviceData<T> *, CuDeviceData<T> *, int>> single_partition_segment;

    /// @name Concentration Storage
    /// @{
    /**
     * @brief Block concentrations on device.
     *
     * Key: (polymer_id, key_left, key_right) with key_left <= key_right
     */
    std::map<std::tuple<int, std::string, std::string>, CuDeviceData<T> *> d_phi_block;

    CuDeviceData<T> *d_phi;  ///< Temporary for concentration computation
    /// @}

    std::vector<CuDeviceData<T> *> d_phi_solvent;  ///< Solvent concentrations

    /**
     * @brief Compute concentration for one polymer block.
     *
     * Integrates q_forward × q_backward over contour.
     *
     * @param d_phi   Output concentration (device)
     * @param d_q_1   Forward propagator array
     * @param d_q_2   Backward propagator array
     * @param N_LEFT  Left segment count
     * @param N_RIGHT Right segment count
     */
    void calculate_phi_one_block(CuDeviceData<T> *d_phi,
        CuDeviceData<T> **d_q_1, CuDeviceData<T> **d_q_2,
        const int N_LEFT, const int N_RIGHT);

public:
    /**
     * @brief Construct GPU propagator computation for continuous chains.
     *
     * @param cb          Computation box
     * @param pc          Molecules container
     * @param propagator_computation_optimizer Optimization strategy
     * @param method      Solver method ("pseudo" or "real")
     */
    CudaComputationContinuous(ComputationBox<T>* cb, Molecules *pc,
        PropagatorComputationOptimizer *propagator_computation_optimizer, std::string method);

    /** @brief Destructor. Frees GPU resources. */
    ~CudaComputationContinuous();

    /** @brief Update operators for new box dimensions. */
    void update_laplacian_operator() override;

    /**
     * @brief Compute all propagators on GPU.
     *
     * @param w_block Potential fields by monomer type (device)
     * @param q_init  Optional initial conditions (device)
     */
    void compute_propagators(
        std::map<std::string, const T*> w_block,
        std::map<std::string, const T*> q_init = {}) override;

    /**
     * @brief Advance single segment (utility for external use).
     *
     * @param q_init       Input propagator (device)
     * @param q_out        Output propagator (device)
     * @param monomer_type Monomer type
     */
    void advance_propagator_single_segment(T* q_init, T *q_out, std::string monomer_type) override;

    /** @brief Compute all concentrations from propagators. */
    void compute_concentrations() override;

    /**
     * @brief Complete SCFT statistics computation.
     *
     * Calls compute_propagators and compute_concentrations.
     *
     * @param w_input Potential fields (device)
     * @param q_init  Optional initial conditions
     */
    void compute_statistics(
        std::map<std::string, const T*> w_input,
        std::map<std::string, const T*> q_init = {}) override;

    /** @brief Compute stress tensor for box relaxation. */
    void compute_stress() override;

    /**
     * @brief Get total partition function for polymer.
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
    /** @brief Get total concentration by monomer type. */
    void get_total_concentration(std::string monomer_type, T *phi) override;

    /** @brief Get concentration for specific polymer and monomer. */
    void get_total_concentration(int polymer, std::string monomer_type, T *phi) override;

    /** @brief Get block-by-block concentration. */
    void get_block_concentration(int polymer, T *phi) override;
    /// @}

    /// @name Solvent Methods
    /// @{
    T get_solvent_partition(int s) override;
    void get_solvent_concentration(int s, T *phi) override;
    /// @}

    /**
     * @brief Grand canonical concentration.
     *
     * @param fugacity Polymer fugacity
     * @param polymer  Polymer index
     * @param monomer_type Monomer type
     * @param phi      Output concentration (host)
     */
    void get_total_concentration_gce(double fugacity, int polymer,
        std::string monomer_type, T *phi) override;

    /** @brief Verify partition function consistency. */
    bool check_total_partition() override;
};

#endif
