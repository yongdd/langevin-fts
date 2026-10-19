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
#include "CudaSolverPseudoBase.h"
#include "CudaCommon.h"
#include "CudaFFT.h"
#include "CudaCrysFFT.h"
#include "FFT.h"
#include "SpaceGroup.h"

class CudaCrysFFTBase;

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
class CudaSolverPseudoRK2 : public CudaSolverPseudoBase<T>
{
private:
    using Base = CudaSolverPseudoBase<T>;
    using Base::cb; using Base::molecules; using Base::pseudo; using Base::chain_model;
    using Base::n_streams; using Base::dim_; using Base::is_periodic_; using Base::streams;
    using Base::temp_storage_bytes; using Base::d_temp_storage;
    using Base::d_stress_sum; using Base::d_stress_sum_out; using Base::d_q_multi;
    using Base::space_group_; using Base::d_reduced_basis_indices_; using Base::d_full_to_reduced_map_;
    using Base::n_basis_; using Base::d_q_full_in_; using Base::d_q_full_out_;
    using Base::use_crysfft_; using Base::crysfft_mode_; using Base::crysfft_identity_map_;
    using Base::crysfft_physical_size_; using Base::crysfft_reduced_size_;
    using Base::d_crysfft_phys_to_reduced_; using Base::d_crysfft_reduced_to_phys_; using Base::d_crysfft_orbit_counts_;
    using Base::crysfft_; using Base::d_crysfft_in_; using Base::d_crysfft_out_;
    using Base::fft_; using Base::fft_forward; using Base::fft_backward;

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
    CudaSolverPseudoRK2(ComputationBox<T>* cb, Molecules *molecules, int n_streams, cudaStream_t streams[MAX_STREAMS][2], bool reduce_memory);

    /// Destructor. Frees GPU memory and cuFFT plans.
    ~CudaSolverPseudoRK2();

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
        std::string monomer_type, double *d_q_mask, int ds_index) override;

    /// Half-bond step (empty for continuous chains).
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
};
#endif
