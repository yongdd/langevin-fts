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
#include "CudaComputationBase.h"
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
class CudaComputationContinuous : public CudaComputationBase<T>
{
private:
    std::string method;                 ///< Solver method ("pseudo" or "real")

    /**
     * @brief Single segment per polymer for partition function.
     *
     * Tuple: (polymer_id, q_forward, q_backward, n_repeated)
     */
    std::vector<std::tuple<int, CuDeviceData<T> *, CuDeviceData<T> *, int>> single_partition_segment;

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
     * @param method      Solver method ("pseudospectral" or "realspace")
     * @param numerical_method Numerical algorithm:
     *                         - For pseudospectral: "rqm4" (default) or "etdrk4"
     *                         - For realspace: "cn-adi2" (default) or "cn-adi4"
     */
    CudaComputationContinuous(ComputationBox<T>* cb, Molecules *pc,
        PropagatorComputationOptimizer *propagator_computation_optimizer, std::string method, std::string numerical_method = "");

    /** @brief Destructor. Frees GPU resources. */
    ~CudaComputationContinuous();

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
     * @brief Extract chain propagator to host.
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
