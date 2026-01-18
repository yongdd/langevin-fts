/**
 * @file CudaSolverPseudoRQM4.h
 * @brief GPU pseudo-spectral solver for continuous chain model using RQM4.
 *
 * This header provides CudaSolverPseudoRQM4, the CUDA implementation
 * of the pseudo-spectral method with RQM4 (Ranjan-Qin-Morse 4th-order using
 * Richardson extrapolation) for continuous Gaussian chains.
 *
 * **GPU Architecture:**
 *
 * - Multiple CUDA streams for concurrent propagator computation
 * - cuFFT for GPU-accelerated FFT operations
 * - Custom CUDA kernels for element-wise operations
 *
 * **Stream Organization:**
 *
 * For each stream index, two CUDA streams are used:
 * - streams[STREAM][0]: Kernel execution
 * - streams[STREAM][1]: Memory transfers (async)
 *
 * **cuFFT Plans:**
 *
 * Pre-created FFT plans for efficiency:
 * - plan_for_one: Single r2c transform
 * - plan_bak_one: Single c2r transform
 * - plan_for_two: Batched r2c (for half-step pairs)
 * - plan_bak_two: Batched c2r (for half-step pairs)
 *
 * **Stress Array Convention (Voigt Notation):**
 *
 * - Index 0-2: Diagonal components (σ₁, σ₂, σ₃) for length optimization
 * - Index 3-5: Off-diagonal components (σ₁₂, σ₁₃, σ₂₃) for angle optimization
 *
 * @see CudaSolver for the abstract interface
 * @see CudaSolverPseudoDiscrete for discrete chain version
 * @see CpuSolverPseudoRQM4 for CPU version
 * @see docs/StressTensorCalculation.md for detailed derivation
 */

#ifndef CUDA_SOLVER_PSEUDO_RQM4_H_
#define CUDA_SOLVER_PSEUDO_RQM4_H_

#include <string>
#include <vector>
#include <map>

#include "Exception.h"
#include "Molecules.h"
#include "ComputationBox.h"
#include "Pseudo.h"
#include "CudaSolver.h"
#include "CudaCommon.h"
#include "CudaFFT.h"
#include "FFT.h"

/**
 * @class CudaSolverPseudoRQM4
 * @brief GPU pseudo-spectral solver for continuous Gaussian chains using RQM4.
 *
 * Implements operator splitting with RQM4 (4th-order Richardson extrapolation)
 * on GPU using cuFFT and custom CUDA kernels.
 *
 * @tparam T Numeric type (double or std::complex<double>)
 *
 * **RQM4 (Richardson Extrapolation):**
 *
 * Same algorithm as CPU version:
 *     q(s+ds) = (4/3) q^(ds/2,ds/2) - (1/3) q^(ds)
 *
 * But both terms computed in parallel using batched FFTs.
 *
 * **Memory per Stream:**
 *
 * - d_q_step_1_one, d_q_step_2_one: Full step workspace
 * - d_q_step_1_two, d_q_step_2_two: Half step workspace
 * - d_qk_in_*: Fourier space buffers
 */
template <typename T>
class CudaSolverPseudoRQM4 : public CudaSolver<T>
{
private:
    ComputationBox<T>* cb;       ///< Computation box
    Molecules *molecules;         ///< Molecules container
    Pseudo<T> *pseudo;            ///< Pseudo-spectral utilities
    std::string chain_model;      ///< Chain model ("continuous")

    int n_streams;                ///< Number of parallel streams
    int dim_;                     ///< Number of dimensions (1, 2, or 3)
    bool is_periodic_;            ///< True if all BCs are periodic

    cudaStream_t streams[MAX_STREAMS][2];  ///< CUDA streams [kernel, memcpy] per index

    /// @name Spectral Transform Resources
    /// @{
    CuDeviceData<T> *d_q_unity;                        ///< Unity array for initialization
    FFT<T>* fft_[MAX_STREAMS];                         ///< Spectral transform objects (per-stream, for non-periodic BC)
    cufftHandle plan_for_one[MAX_STREAMS];             ///< Single forward FFT plans (periodic BC)
    cufftHandle plan_bak_one[MAX_STREAMS];             ///< Single backward FFT plans (periodic BC)
    cufftHandle plan_for_two[MAX_STREAMS];             ///< Batched forward FFT plans (periodic BC)
    cufftHandle plan_bak_two[MAX_STREAMS];             ///< Batched backward FFT plans (periodic BC)
    /// @}

    /// @name Workspace Arrays (per stream)
    /// @{
    CuDeviceData<T> *d_q_step_1_one[MAX_STREAMS];      ///< Full step workspace 1
    CuDeviceData<T> *d_q_step_2_one[MAX_STREAMS];      ///< Full step workspace 2
    CuDeviceData<T> *d_q_step_1_two[MAX_STREAMS];      ///< Half step workspace 1
    CuDeviceData<T> *d_q_step_2_two[MAX_STREAMS];      ///< Half step workspace 2
    /// @}

    /// @name Fourier Space Buffers (periodic BC - complex coefficients)
    /// @{
    cuDoubleComplex *d_qk_in_1_one[MAX_STREAMS];
    cuDoubleComplex *d_qk_in_2_one[MAX_STREAMS];
    cuDoubleComplex *d_qk_in_1_two[MAX_STREAMS];
    cuDoubleComplex *d_qk_in_2_two[MAX_STREAMS];
    /// @}

    /// @name Spectral Space Buffers (non-periodic BC - real coefficients)
    /// @{
    double *d_rk_in_1_one[MAX_STREAMS];    ///< Real coefficients buffer 1
    double *d_rk_in_2_one[MAX_STREAMS];    ///< Real coefficients buffer 2
    /// @}

    /// @name CUB Reduction Storage (per stream)
    /// @{
    size_t temp_storage_bytes[MAX_STREAMS];
    CuDeviceData<T> *d_temp_storage[MAX_STREAMS];
    CuDeviceData<T> *d_stress_sum[MAX_STREAMS];
    CuDeviceData<T> *d_stress_sum_out[MAX_STREAMS];
    CuDeviceData<T> *d_q_multi[MAX_STREAMS];
    /// @}

public:
    /**
     * @brief Construct GPU pseudo-spectral solver for continuous chains.
     *
     * @param cb                  Computation box
     * @param molecules           Molecules container
     * @param n_streams           Number of parallel streams
     * @param streams             Pre-created CUDA streams
     * @param reduce_memory   Checkpointing mode (affects workspace allocation)
     */
    CudaSolverPseudoRQM4(ComputationBox<T>* cb, Molecules *molecules, int n_streams, cudaStream_t streams[MAX_STREAMS][2], bool reduce_memory);

    /**
     * @brief Destructor. Frees GPU memory and cuFFT plans.
     */
    ~CudaSolverPseudoRQM4();

    /** @brief Update Fourier-space operators for new box dimensions. */
    void update_laplacian_operator() override;

    /**
     * @brief Update Boltzmann factors.
     * @param device  "device" or "host" for w_input location
     * @param w_input Potential fields
     */
    void update_dw(std::string device, std::map<std::string, const T*> w_input) override;

    /**
     * @brief Advance propagator by one step using RQM4.
     *
     * @param STREAM      Stream index for concurrent execution
     * @param d_q_in      Input propagator (device)
     * @param d_q_out     Output propagator (device)
     * @param monomer_type Monomer type
     * @param d_q_mask    Optional mask (device, nullptr if none)
     * @param ds_index    Index for ds value (1-based, for per-block ds support)
     */
    void advance_propagator(
        const int STREAM,
        CuDeviceData<T> *d_q_in, CuDeviceData<T> *d_q_out,
        std::string monomer_type, double *d_q_mask, int ds_index = 1) override;

    /** @brief Half-bond step (empty for continuous chains). */
    void advance_propagator_half_bond_step(
        const int, CuDeviceData<T> *, CuDeviceData<T> *, std::string) override {};

    /**
     * @brief Compute stress contribution from one segment.
     *
     * Computes ∂ln(Q)/∂θ in Fourier space by multiplying forward and backward
     * propagators with weighted basis functions. Cross-term corrections for
     * non-orthogonal boxes are included.
     *
     * For continuous chains: Φ(k) = 1 (bond factor already in propagator).
     *
     * @param STREAM           Stream index for concurrent execution
     * @param d_q_pair         Forward and backward propagators (device)
     * @param d_segment_stress Output: stress in Voigt notation
     *                         [σ₁, σ₂, σ₃, σ₁₂, σ₁₃, σ₂₃] (device, 6 components)
     * @param monomer_type     Monomer type for segment length
     * @param is_half_bond_length Ignored for continuous chains
     */
    void compute_single_segment_stress(
        const int STREAM,
        CuDeviceData<T> *d_q_pair, CuDeviceData<T> *d_segment_stress,
        std::string monomer_type, bool is_half_bond_length) override;
};
#endif
