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
 * - cuFFT plans (batch 1 for the full step, batch 2 for the half-step pair)
 *   and DCT/DST objects are owned by CudaSolverPseudoBase
 *
 * **Stress Array Convention (Voigt Notation):**
 *
 * - Index 0-2: Diagonal components (σ₁, σ₂, σ₃) for length optimization
 * - Index 3-5: Off-diagonal components (σ₁₂, σ₁₃, σ₂₃) for angle optimization
 *
 * @see CudaSolver for the abstract interface
 * @see CudaSolverPseudoDiscrete for discrete chain version
 * @see CpuSolverPseudoRQM4 for CPU version
 * @see docs/theory/StressTensor.md for detailed derivation
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
#include "CudaSolverPseudoBase.h"
#include "CudaCommon.h"
#include "CudaFFT.h"
#include "CudaCrysFFT.h"
#include "FFT.h"

class SpaceGroup;  // Forward declaration
class CudaCrysFFTBase;  // Forward declaration

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
 * - d_q_step_1_two: Half step workspace (2*M; also lent as the propagator pair of the stress)
 * - d_qk_in_2_one, d_qk_in_1_two: Fourier space buffers
 */
template <typename T>
class CudaSolverPseudoRQM4 : public CudaSolverPseudoBase<T>
{
private:
    using Base = CudaSolverPseudoBase<T>;
    using Base::cb; using Base::molecules; using Base::pseudo;
    using Base::n_streams; using Base::is_periodic_; using Base::streams;

    using Base::space_group_; using Base::d_reduced_basis_indices_; using Base::d_full_to_reduced_map_;
    using Base::n_basis_; using Base::d_q_full_in_; using Base::d_q_full_out_;
    using Base::use_crysfft_; using Base::crysfft_identity_map_;
    using Base::crysfft_physical_size_;
    using Base::d_crysfft_phys_to_reduced_; using Base::d_crysfft_reduced_to_phys_;
    using Base::crysfft_; using Base::d_crysfft_in_; using Base::d_crysfft_out_;
    using Base::fft_; using Base::fft_forward; using Base::fft_backward;

    /// @name Workspace Arrays (per stream)
    /// @{
    CuDeviceData<T> *d_q_step_1_one[MAX_STREAMS];      ///< Full step workspace 1
    CuDeviceData<T> *d_q_step_2_one[MAX_STREAMS];      ///< Full step workspace 2
    CuDeviceData<T> *d_q_step_1_two[MAX_STREAMS];      ///< Half step workspace 1
    /// @}

    /// @name Fourier Space Buffers (periodic BC - complex coefficients)
    /// @{
    cuDoubleComplex *d_qk_in_2_one[MAX_STREAMS];
    cuDoubleComplex *d_qk_in_1_two[MAX_STREAMS];
    /// @}

    /// @name Spectral Space Buffers (non-periodic BC - real coefficients)
    /// @{
    double *d_rk_in_1_one[MAX_STREAMS];    ///< Real coefficients buffer 1
    double *d_rk_in_2_one[MAX_STREAMS];    ///< Real coefficients buffer 2
    /// @}

public:
    /**
     * @brief Construct GPU pseudo-spectral solver for continuous chains.
     *
     * @param cb                  Computation box
     * @param molecules           Molecules container
     * @param n_streams           Number of parallel streams
     * @param streams             Pre-created CUDA streams
     * @param reduce_memory   Unused (kept for the factory signature)
     */
    CudaSolverPseudoRQM4(ComputationBox<T>* cb, Molecules *molecules, int n_streams, cudaStream_t streams[MAX_STREAMS][2], bool reduce_memory);

    /// Destructor. Frees the workspaces (the base class destroys the cuFFT plans).
    ~CudaSolverPseudoRQM4();

    /**
     * @brief Update Boltzmann factors.
     * @param device  "gpu" or "cpu" for w_input location
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
     * @param ds_index    Index for ds value (0-based, for per-block ds support)
     */
    void advance_propagator(
        const int STREAM,
        CuDeviceData<T> *d_q_in, CuDeviceData<T> *d_q_out,
        std::string monomer_type, double *d_q_mask, int ds_index) override;

    /// Half-bond step (empty for continuous chains).
    void advance_propagator_half_bond_step(
        const int, CuDeviceData<T> *, CuDeviceData<T> *, std::string) override {};

    /// Scratch buffers lent to CudaSolverPseudoBase::compute_single_segment_stress()
    typename Base::StressScratch stress_scratch(int STREAM) override;
};
#endif
