/**
 * @file CudaSolverPseudoDiscrete.h
 * @brief GPU pseudo-spectral solver for discrete chain model.
 *
 * This header provides CudaSolverPseudoDiscrete, the CUDA implementation
 * of the pseudo-spectral method for discrete chain propagators using
 * the Chapman-Kolmogorov integral equation.
 *
 * **Chapman-Kolmogorov Equation (N-1 Bond Model):**
 *
 * For discrete chains, the propagator satisfies:
 *     q(r, i+1) = exp(-w(r)*ds) * integral g(r-r') q(r', i) dr'
 *
 * In Fourier space:
 *     q(i+1) = exp(-w*ds) * FFT^-1[ ĝ(k) * FFT[q(i)] ]
 *
 * where ĝ(k) = exp(-b²|k|²ds/6) is the full bond function.
 * See Park et al. J. Chem. Phys. 150, 234901 (2019).
 *
 * Half-bond steps (ĝ^(1/2)(k) = exp(-b²|k|²ds/12)) are used only at
 * chain ends and junction points.
 *
 * **GPU Resources:**
 *
 * Same stream and cuFFT organization as continuous solver,
 * but uses bond convolution operators instead of diffusion.
 *
 * @see CudaSolver for the abstract interface
 * @see CudaSolverPseudoRQM4 for continuous chain version
 * @see CpuSolverPseudoDiscrete for CPU version
 */

#ifndef CUDA_SOLVER_PSEUDO_DISCRETE_H_
#define CUDA_SOLVER_PSEUDO_DISCRETE_H_

#include <string>
#include <vector>
#include <map>

#include "Exception.h"
#include "Molecules.h"
#include "ComputationBox.h"
#include "Pseudo.h"
#include "CudaSolver.h"
#include "CudaCommon.h"

/**
 * @class CudaSolverPseudoDiscrete
 * @brief GPU pseudo-spectral solver for discrete chain model.
 *
 * Implements discrete chain propagator updates using the Chapman-Kolmogorov
 * integral equation on the GPU.
 *
 * @tparam T Numeric type (double or std::complex<double>)
 *
 * **N-1 Bond Model:**
 *
 * Full segment step: q(i+1) = exp(-w*ds) * FFT^-1[ ĝ(k) * FFT[q(i)] ]
 * - ĝ(k) = exp(-b²|k|²ds/6) is the full bond function
 * - exp(-w*ds) is the full-segment Boltzmann factor
 */
template <typename T>
class CudaSolverPseudoDiscrete : public CudaSolver<T>
{
private:
    ComputationBox<T>* cb;       ///< Computation box
    Molecules *molecules;         ///< Molecules container
    Pseudo<T> *pseudo;            ///< Pseudo-spectral utilities
    std::string chain_model;      ///< Chain model ("discrete")

    int n_streams;                ///< Number of parallel streams

    cudaStream_t streams[MAX_STREAMS][2];  ///< CUDA streams

    /// @name cuFFT Resources
    /// @{
    T *d_q_unity;                                      ///< Unity array
    cufftHandle plan_for_one[MAX_STREAMS];             ///< Single forward FFT
    cufftHandle plan_bak_one[MAX_STREAMS];             ///< Single backward FFT
    cufftHandle plan_for_two[MAX_STREAMS];             ///< Batched forward FFT
    cufftHandle plan_bak_two[MAX_STREAMS];             ///< Batched backward FFT
    /// @}

    /// @name Workspace Arrays
    /// @{
    T *d_q_step_1_one[MAX_STREAMS];
    T *d_q_step_2_one[MAX_STREAMS];
    T *d_q_step_1_two[MAX_STREAMS];
    T *d_q_step_2_two[MAX_STREAMS];
    /// @}

    /// @name Fourier Space Buffers
    /// @{
    cuDoubleComplex *d_qk_in_1_one[MAX_STREAMS];
    cuDoubleComplex *d_qk_in_2_one[MAX_STREAMS];
    cuDoubleComplex *d_qk_in_1_two[MAX_STREAMS];
    cuDoubleComplex *d_qk_in_2_two[MAX_STREAMS];
    /// @}

    /// @name CUB Reduction Storage
    /// @{
    size_t temp_storage_bytes[MAX_STREAMS];
    CuDeviceData<T> *d_temp_storage[MAX_STREAMS];
    CuDeviceData<T> *d_stress_sum[MAX_STREAMS];
    CuDeviceData<T> *d_stress_sum_out[MAX_STREAMS];
    CuDeviceData<T> *d_q_multi[MAX_STREAMS];
    /// @}

public:
    /**
     * @brief Construct GPU pseudo-spectral solver for discrete chains.
     *
     * @param cb                  Computation box
     * @param molecules           Molecules container
     * @param n_streams           Number of parallel streams
     * @param streams             Pre-created CUDA streams
     * @param use_checkpointing   Checkpointing mode
     */
    CudaSolverPseudoDiscrete(ComputationBox<T>* cb, Molecules *molecules, int n_streams, cudaStream_t streams[MAX_STREAMS][2], bool use_checkpointing);

    /**
     * @brief Destructor. Frees GPU resources.
     */
    ~CudaSolverPseudoDiscrete();

    /** @brief Update half-bond diffusion operators. */
    void update_laplacian_operator() override;

    /** @brief Update Boltzmann factors. */
    void update_dw(std::string device, std::map<std::string, const T*> w_input) override;

    /**
     * @brief Advance propagator by one full segment.
     *
     * Computes: q(i+1) = exp(-w*ds) * FFT^-1[ ĝ(k) * FFT[q(i)] ]
     * where ĝ(k) = exp(-b²|k|²ds/6) is the full bond function.
     *
     * @param STREAM      Stream index
     * @param d_q_in      Input propagator q(i)
     * @param d_q_out     Output propagator q(i+1)
     * @param monomer_type Monomer type
     * @param d_q_mask    Optional mask
     * @param ds_index    Index for ds value (discrete chains use ds_index=1)
     */
    void advance_propagator(
        const int STREAM,
        CuDeviceData<T> *d_q_in, CuDeviceData<T> *d_q_out,
        std::string monomer_type, double *d_q_mask, int ds_index = 1) override;

    /**
     * @brief Advance by half bond step only.
     *
     * Computes: q' = FFT^-1[ ĝ^(1/2)(k) * FFT[q] ]
     * where ĝ^(1/2)(k) = exp(-b²|k|²ds/12) is the half-bond function.
     *
     * Used at chain ends and junction points.
     *
     * @param STREAM      Stream index
     * @param d_q_in      Input propagator
     * @param d_q_out     Output after half-bond convolution
     * @param monomer_type Monomer type
     */
    void advance_propagator_half_bond_step(
        const int STREAM,
        CuDeviceData<T> *d_q_in, CuDeviceData<T> *d_q_out,
        std::string monomer_type) override;

    /**
     * @brief Compute stress from one segment.
     *
     * @param STREAM           Stream index
     * @param d_q_pair         Propagator product
     * @param d_segment_stress Output stress
     * @param monomer_type     Monomer type
     * @param is_half_bond_length True for half-bond contribution
     */
    void compute_single_segment_stress(
        const int STREAM,
        CuDeviceData<T> *d_q_pair, CuDeviceData<T> *d_segment_stress,
        std::string monomer_type, bool is_half_bond_length) override;
};
#endif
