/**
 * @file CudaSolverPseudoRK2.h
 * @brief GPU pseudo-spectral solver for continuous chain model using RK2.
 *
 * This header provides CudaSolverPseudoRK2, the CUDA implementation
 * of the pseudo-spectral method with RK2 (Rasmussen-Kalosakas 2nd-order)
 * for continuous Gaussian chains.
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
 * **Comparison to RQM4:**
 *
 * RK2 uses only a single full step (2 FFTs) while RQM4 combines full and half
 * steps via Richardson extrapolation (6 FFTs). RK2 is faster but only
 * 2nd-order accurate (vs 4th-order for RQM4).
 *
 * @see CudaSolver for the abstract interface
 * @see CudaSolverPseudoRQM4 for 4th-order version
 * @see CpuSolverPseudoRK2 for CPU version
 */

#ifndef CUDA_SOLVER_PSEUDO_RK2_H_
#define CUDA_SOLVER_PSEUDO_RK2_H_

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
 * @class CudaSolverPseudoRK2
 * @brief GPU pseudo-spectral solver for continuous Gaussian chains using RK2.
 *
 * Implements operator splitting with RK2 (2nd-order Rasmussen-Kalosakas method)
 * on GPU using cuFFT and custom CUDA kernels.
 *
 * @tparam T Numeric type (double or std::complex<double>)
 *
 * **RK2 (Operator Splitting):**
 *
 *     q(s+ds) = exp(-w·ds/2) · FFT⁻¹[ exp(-k²b²ds/6) · FFT[ exp(-w·ds/2) · q(s) ] ]
 *
 * **Memory per Stream:**
 *
 * - d_q_step_1, d_q_step_2: Workspace arrays
 * - d_qk_in_*: Fourier space buffers
 */
template <typename T>
class CudaSolverPseudoRK2 : public CudaSolver<T>
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
    /// @}

    /// @name Workspace Arrays (per stream)
    /// @{
    CuDeviceData<T> *d_q_step_1[MAX_STREAMS];          ///< Workspace 1
    CuDeviceData<T> *d_q_step_2[MAX_STREAMS];          ///< Workspace 2
    /// @}

    /// @name Fourier Space Buffers (periodic BC - complex coefficients)
    /// @{
    cuDoubleComplex *d_qk_in_1[MAX_STREAMS];           ///< Complex coefficients buffer 1
    cuDoubleComplex *d_qk_in_2[MAX_STREAMS];           ///< Complex coefficients buffer 2
    /// @}

    /// @name Spectral Space Buffers (non-periodic BC - real coefficients)
    /// @{
    double *d_rk_in_1[MAX_STREAMS];                    ///< Real coefficients buffer 1
    double *d_rk_in_2[MAX_STREAMS];                    ///< Real coefficients buffer 2
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
     * @param reduce_memory_usage Memory saving mode (affects workspace allocation)
     */
    CudaSolverPseudoRK2(ComputationBox<T>* cb, Molecules *molecules, int n_streams, cudaStream_t streams[MAX_STREAMS][2], bool reduce_memory_usage);

    /**
     * @brief Destructor. Frees GPU memory and cuFFT plans.
     */
    ~CudaSolverPseudoRK2();

    /** @brief Update Fourier-space operators for new box dimensions. */
    void update_laplacian_operator() override;

    /**
     * @brief Update Boltzmann factors.
     * @param device  "device" or "host" for w_input location
     * @param w_input Potential fields
     */
    void update_dw(std::string device, std::map<std::string, const T*> w_input) override;

    /**
     * @brief Advance propagator by one step using RK2.
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
     * @brief Compute stress from one segment.
     *
     * @param STREAM           Stream index
     * @param d_q_pair         Product of forward and backward propagators
     * @param d_segment_stress Output stress contribution
     * @param monomer_type     Monomer type
     * @param is_half_bond_length Ignored for continuous
     */
    void compute_single_segment_stress(
        const int STREAM,
        CuDeviceData<T> *d_q_pair, CuDeviceData<T> *d_segment_stress,
        std::string monomer_type, bool is_half_bond_length) override;

    /**
     * @brief Enable or disable cell-averaged bond function.
     *
     * @param enabled True to enable cell-averaging, false for standard bond function
     */
    void set_cell_averaged_bond(bool enabled) override;

    /**
     * @brief Set the number of aliased momentum terms for cell-averaging.
     *
     * @param n Number of aliased copies in each direction (n = 0, 1, 2, ...)
     */
    void set_cell_average_momentum(int n) override;
};
#endif
