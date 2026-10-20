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
#include "CudaSolverPseudoBase.h"
#include "CudaCommon.h"
#include "CudaFFT.h"
#include "CudaCrysFFT.h"

class SpaceGroup;  // Forward declaration
class CudaCrysFFTBase;  // Forward declaration

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
class CudaSolverPseudoDiscrete : public CudaSolverPseudoBase<T>
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


    /// @name Fourier Space Buffers
    /// @{
    cuDoubleComplex *d_qk_in_1_one[MAX_STREAMS];
    cuDoubleComplex *d_qk_in_1_two[MAX_STREAMS];
    double *d_rk_in_1_one[MAX_STREAMS];  ///< Real-valued FFT coefficients for non-periodic BC (DCT/DST); nullptr when periodic
    double *d_rk_in_2_one[MAX_STREAMS];  ///< Real-valued FFT coefficients for non-periodic BC (DCT/DST); nullptr when periodic
    /// @}

public:
    /**
     * @brief Construct GPU pseudo-spectral solver for discrete chains.
     *
     * @param cb                  Computation box
     * @param molecules           Molecules container
     * @param n_streams           Number of parallel streams
     * @param streams             Pre-created CUDA streams
     * @param reduce_memory   Unused (kept for the factory signature)
     */
    CudaSolverPseudoDiscrete(ComputationBox<T>* cb, Molecules *molecules, int n_streams, cudaStream_t streams[MAX_STREAMS][2], bool reduce_memory);

    /// Destructor. Frees GPU resources.
    ~CudaSolverPseudoDiscrete();

    /**
     * @brief True when d_exp_dw is stored in the reduced basis (size n_basis).
     *
     * In CrysFFT mode the solver keeps exp(-w*ds) in the reduced basis;
     * otherwise (standard FFT with space group) it is stored on the full grid.
     * Callers that consume exp_dw in the reduced basis must convert when
     * this returns false.
     */
    bool exp_dw_stored_reduced() const { return use_crysfft_ && space_group_ != nullptr; }

    /// Update Boltzmann factors.
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
     * @param ds_index    Index for ds value (discrete chains use ds_index=0)
     */
    void advance_propagator(
        const int STREAM,
        CuDeviceData<T> *d_q_in, CuDeviceData<T> *d_q_out,
        std::string monomer_type, double *d_q_mask, int ds_index) override;

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

    /// Scratch buffers lent to CudaSolverPseudoBase::compute_single_segment_stress()
    typename Base::StressScratch stress_scratch(int STREAM) override;
    /// Boltzmann bond factor of the stress integrand (see CudaSolverPseudoBase)
    const double* get_stress_boltz_bond(
        const std::string& monomer_type, bool is_half_bond_length) const override;
};
#endif
