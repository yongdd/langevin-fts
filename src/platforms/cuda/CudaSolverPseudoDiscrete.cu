/**
 * @file CudaSolverPseudoDiscrete.cu
 * @brief CUDA pseudo-spectral solver for discrete chain model.
 *
 * Implements propagator advancement for discrete chains using the
 * Chapman-Kolmogorov integral equation via cuFFT.
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
 * **Methods:**
 *
 * - advance_propagator(): Full segment step (bond convolution + full-segment weight)
 * - advance_propagator_half_bond_step(): Half bond convolution at chain ends/junctions
 * - compute_single_segment_stress(): Stress contribution per segment
 *
 * **cuFFT Plans:**
 *
 * - cuFFT plans and DCT/DST objects are owned by CudaSolverPseudoBase
 *   (fft_forward / fft_backward / fft_)
 * - Stream-associated for concurrent execution
 *
 * **Template Instantiations:**
 *
 * - CudaSolverPseudoDiscrete<double>: Real field solver
 * - CudaSolverPseudoDiscrete<std::complex<double>>: Complex field solver
 *
 * @see CudaComputationDiscrete for discrete chain orchestration
 */

#include <iostream>
#include <cmath>
#include <numbers>
#include <array>
#include <thrust/reduce.h>

#include "CudaPseudo.h"
#include "CudaTypeDispatch.h"
#include "CudaSolverPseudoDiscrete.h"
#include "CudaCrysFFT.h"
#include "CudaCrysFFTRecursive3m.h"
#include "CudaCrysFFTObliqueZ.h"
#include "CudaCrysFFTStress.h"
#include "CrysFFTSelector.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include "SpaceGroup.h"

template <typename T>
CudaSolverPseudoDiscrete<T>::CudaSolverPseudoDiscrete(
    ComputationBox<T>* cb,
    Molecules *molecules,
    int n_streams,
    cudaStream_t streams[MAX_STREAMS][2],
    [[maybe_unused]] bool reduce_memory)
    : CudaSolverPseudoBase<T>(cb, molecules, n_streams, streams, /*max_fft_batch=*/2,
                              /*q_full_in_factor=*/2, /*with_exp_dw_half=*/false,
                              /*complex_requires_periodic=*/false)
{
    try{
        const int M_COMPLEX = pseudo->get_total_complex_grid();
        for(int i=0; i<n_streams; i++)
        {
            d_rk_in_1_one[i] = nullptr;
            d_rk_in_2_one[i] = nullptr;
        }

        // Allocate memory for pseudo-spectral: advance_propagator()
        for(int i=0; i<n_streams; i++)
        {
            gpu_error_check(cudaMalloc((void**)&d_qk_in_1_one[i], sizeof(cuDoubleComplex)*M_COMPLEX));
            gpu_error_check(cudaMalloc((void**)&d_qk_in_1_two[i], sizeof(cuDoubleComplex)*2*M_COMPLEX));
            // Real-valued buffers for non-periodic BC stress calculation (DCT/DST)
            // For complex fields the DCT/DST coefficients of the real and
            // imaginary parts are stored interleaved (complex pairs).
            {
                const int rk_size = std::is_same<T, std::complex<double>>::value
                                    ? 2 * M_COMPLEX : M_COMPLEX;
                gpu_error_check(cudaMalloc((void**)&d_rk_in_1_one[i], sizeof(double)*rk_size));
                gpu_error_check(cudaMalloc((void**)&d_rk_in_2_one[i], sizeof(double)*rk_size));
            }
        }

        this->update_laplacian_operator();
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
CudaSolverPseudoDiscrete<T>::~CudaSolverPseudoDiscrete()
{

    // For pseudo-spectral: advance_propagator()
    for(int i=0; i<n_streams; i++)
    {
        cudaFree(d_qk_in_1_one[i]);
        cudaFree(d_qk_in_1_two[i]);
        cudaFree(d_rk_in_1_one[i]);
        cudaFree(d_rk_in_2_one[i]);
    }

}

template <typename T>
void CudaSolverPseudoDiscrete<T>::update_dw(std::string device, std::map<std::string, const T*> w_input)
{
    // exp(-w ds): one full segment per discrete bond step
    this->update_dw_impl(device, w_input, false, 1.0, 0.0);
}
template <typename T>
void CudaSolverPseudoDiscrete<T>::advance_propagator(
    const int STREAM,
    CuDeviceData<T> *d_q_in, CuDeviceData<T> *d_q_out,
    std::string monomer_type,
    double *d_q_mask, int ds_index)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = cb->get_total_grid();
        const int M_COMPLEX = pseudo->get_total_complex_grid();

        // Get Boltzmann factors for the correct ds_index
        CuDeviceData<T> *_d_exp_dw = this->d_exp_dw[ds_index][monomer_type];
        const double* _d_boltz_bond = pseudo->get_boltz_bond(monomer_type, ds_index);

        const bool use_crysfft = (use_crysfft_ && space_group_ != nullptr);
        if (use_crysfft)
        {
            // [real-only feature] Space-group reduced basis, CrysFFT and non-periodic stress are implemented for real fields; complex fields take the generic path or raise.
            if constexpr (!std::is_same_v<T, double>)
            {
                throw_with_line_number("CrysFFT path is only supported for real fields.");
            }
            else
            {
                const ContourLengthMapping& mapping = this->molecules->get_contour_length_mapping();
                double local_ds = mapping.get_ds_from_index(ds_index);
                auto bond_lengths = this->molecules->get_bond_lengths();
                double bond_length_sq = bond_lengths[monomer_type] * bond_lengths[monomer_type];
                double coeff_full = bond_length_sq * local_ds / 6.0;

                const int M_phys = crysfft_physical_size_;
                if (crysfft_identity_map_)
                {
                    gpu_error_check(cudaMemcpyAsync(
                        d_crysfft_in_[STREAM], d_q_in,
                        sizeof(double) * M_phys, cudaMemcpyDeviceToDevice, streams[STREAM][0]));

                    crysfft_[STREAM]->set_contour_step(coeff_full);
                    crysfft_[STREAM]->diffusion(d_crysfft_in_[STREAM], d_crysfft_out_[STREAM], streams[STREAM][0]);

                    ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                        d_q_out, d_crysfft_out_[STREAM], _d_exp_dw, 1.0, M_phys);
                    gpu_error_check(cudaPeekAtLastError());

                    if (d_q_mask != nullptr)
                    {
                        ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                            d_q_out, d_q_out, d_q_mask, 1.0, M_phys);
                        gpu_error_check(cudaPeekAtLastError());
                    }
                    return;
                }
                ker_expand_reduced_basis<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                    d_crysfft_in_[STREAM], d_q_in, d_crysfft_phys_to_reduced_, M_phys);
                gpu_error_check(cudaPeekAtLastError());

                crysfft_[STREAM]->set_contour_step(coeff_full);
                crysfft_[STREAM]->diffusion(d_crysfft_in_[STREAM], d_crysfft_out_[STREAM], streams[STREAM][0]);

                ker_multi_map<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                    d_crysfft_out_[STREAM], _d_exp_dw, d_crysfft_phys_to_reduced_, M_phys);
                gpu_error_check(cudaPeekAtLastError());

                ker_reduce_to_basis<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                    d_q_out, d_crysfft_out_[STREAM], d_crysfft_reduced_to_phys_, n_basis_);
                gpu_error_check(cudaPeekAtLastError());

                if (d_q_mask != nullptr)
                {
                    ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                        d_q_out, d_q_out, d_q_mask, 1.0, n_basis_);
                    gpu_error_check(cudaPeekAtLastError());
                }
                return;
            }
        }

        // Determine input/output pointers based on space group
        CuDeviceData<T> *fft_in = d_q_in;
        CuDeviceData<T> *fft_out = d_q_out;

        if (space_group_ != nullptr)
        {
            // [real-only feature] Space-group reduced basis, CrysFFT and non-periodic stress are implemented for real fields; complex fields take the generic path or raise.
            if constexpr (std::is_same_v<T, double>)
            {
                // Expand reduced basis → full grid
                ker_expand_reduced_basis<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                    d_q_full_in_[STREAM], d_q_in, d_full_to_reduced_map_, M);
                gpu_error_check(cudaPeekAtLastError());
                fft_in = d_q_full_in_[STREAM];
                fft_out = d_q_full_out_[STREAM];
            }
            else
            {
                throw_with_line_number("Space group reduced basis is only supported for real fields.");
            }
        }

        if (this->is_periodic_)
        {
            // Execute a forward FFT
            fft_forward(STREAM, fft_in, d_qk_in_1_one[STREAM], 1);

            // Multiply exp(-k^2 ds/6) in fourier space
            ker_multi_complex_real<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_qk_in_1_one[STREAM], _d_boltz_bond, 1.0, M_COMPLEX);

            // Execute a backward FFT
            fft_backward(STREAM, d_qk_in_1_one[STREAM], fft_out, 1);

            // Evaluate exp(-w*ds) in real space (1/M normalization from unnormalized cuFFT)
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(fft_out, fft_out, _d_exp_dw, 1.0/static_cast<double>(M), M);
        }
        else
        {
            // Non-periodic BC (reflecting/absorbing): use CudaFFT (DCT/DST) with real
            // coefficients. forward() copies the input into its own work buffer, so
            // fft_in is preserved. Uses the default stream throughout, matching CudaFFT.
            fft_[STREAM]->forward(reinterpret_cast<T*>(fft_in), d_rk_in_1_one[STREAM]);

            // Multiply bond Boltzmann factor (real factor; coefficients are
            // real for real fields, interleaved complex pairs for complex)
            // [type-specific algorithm] DCT/DST coefficients are real for real fields and interleaved complex pairs for complex fields.
            if constexpr (std::is_same<T, double>::value)
                ker_multi<<<N_BLOCKS, N_THREADS>>>(d_rk_in_1_one[STREAM], d_rk_in_1_one[STREAM], _d_boltz_bond, 1.0, M_COMPLEX);
            else
                ker_multi_complex_real<<<N_BLOCKS, N_THREADS>>>(reinterpret_cast<cuDoubleComplex*>(d_rk_in_1_one[STREAM]), _d_boltz_bond, 1.0, M_COMPLEX);
            gpu_error_check(cudaPeekAtLastError());

            fft_[STREAM]->backward(d_rk_in_1_one[STREAM], reinterpret_cast<T*>(fft_out));

            // Evaluate exp(-w*ds); DCT/DST backward is already normalized (no 1/M).
            ker_multi<<<N_BLOCKS, N_THREADS>>>(fft_out, fft_out, _d_exp_dw, 1.0, M);
            gpu_error_check(cudaPeekAtLastError());
        }

        // Multiply mask (on full grid)
        if (d_q_mask != nullptr)
        {
            double* d_q_mask_full = d_q_mask;
            if (space_group_ != nullptr)
            {
                // [real-only feature] Space-group reduced basis, CrysFFT and non-periodic stress are implemented for real fields; complex fields take the generic path or raise.
                if constexpr (std::is_same_v<T, double>)
                {
                    ker_expand_reduced_basis<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                        d_q_full_in_[STREAM], d_q_mask, d_full_to_reduced_map_, M);
                    gpu_error_check(cudaPeekAtLastError());
                    d_q_mask_full = d_q_full_in_[STREAM];
                }
                else
                {
                    throw_with_line_number("Space group reduced basis is only supported for real fields.");
                }
            }
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(fft_out, fft_out, d_q_mask_full, 1.0, M);
        }

        // Reduce full grid → reduced basis
        if (space_group_ != nullptr)
        {
            ker_reduce_to_basis<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                d_q_out, fft_out, d_reduced_basis_indices_, n_basis_);
            gpu_error_check(cudaPeekAtLastError());
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
void CudaSolverPseudoDiscrete<T>::advance_propagator_half_bond_step(
    const int STREAM,
    CuDeviceData<T> *d_q_in, CuDeviceData<T> *d_q_out, std::string monomer_type)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = cb->get_total_grid();
        const int M_COMPLEX = pseudo->get_total_complex_grid();
        // Discrete chains always use ds_index=0 (global ds)
        const double* _d_boltz_bond_half = pseudo->get_boltz_bond_half(monomer_type, 0);

        const bool use_crysfft = (use_crysfft_ && space_group_ != nullptr);
        if (use_crysfft)
        {
            // [real-only feature] Space-group reduced basis, CrysFFT and non-periodic stress are implemented for real fields; complex fields take the generic path or raise.
            if constexpr (!std::is_same_v<T, double>)
            {
                throw_with_line_number("CrysFFT path is only supported for real fields.");
            }
            else
            {
                const ContourLengthMapping& mapping = this->molecules->get_contour_length_mapping();
                double local_ds = mapping.get_ds_from_index(0);
                auto bond_lengths = this->molecules->get_bond_lengths();
                double bond_length_sq = bond_lengths[monomer_type] * bond_lengths[monomer_type];
                double coeff_half = bond_length_sq * local_ds / 12.0;

                const int M_phys = crysfft_physical_size_;
                if (crysfft_identity_map_)
                {
                    gpu_error_check(cudaMemcpyAsync(
                        d_crysfft_in_[STREAM], d_q_in,
                        sizeof(double) * M_phys, cudaMemcpyDeviceToDevice, streams[STREAM][0]));

                    crysfft_[STREAM]->set_contour_step(coeff_half);
                    crysfft_[STREAM]->diffusion(d_crysfft_in_[STREAM], d_crysfft_out_[STREAM], streams[STREAM][0]);

                    gpu_error_check(cudaMemcpyAsync(
                        d_q_out, d_crysfft_out_[STREAM],
                        sizeof(double) * M_phys, cudaMemcpyDeviceToDevice, streams[STREAM][0]));
                    return;
                }
                ker_expand_reduced_basis<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                    d_crysfft_in_[STREAM], d_q_in, d_crysfft_phys_to_reduced_, M_phys);
                gpu_error_check(cudaPeekAtLastError());

                crysfft_[STREAM]->set_contour_step(coeff_half);
                crysfft_[STREAM]->diffusion(d_crysfft_in_[STREAM], d_crysfft_out_[STREAM], streams[STREAM][0]);

                ker_reduce_to_basis<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                    d_q_out, d_crysfft_out_[STREAM], d_crysfft_reduced_to_phys_, n_basis_);
                gpu_error_check(cudaPeekAtLastError());
                return;
            }
        }

        // Determine input/output pointers based on space group
        CuDeviceData<T> *fft_in = d_q_in;
        CuDeviceData<T> *fft_out = d_q_out;

        if (space_group_ != nullptr)
        {
            // [real-only feature] Space-group reduced basis, CrysFFT and non-periodic stress are implemented for real fields; complex fields take the generic path or raise.
            if constexpr (std::is_same_v<T, double>)
            {
                // Expand reduced basis → full grid
                ker_expand_reduced_basis<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                    d_q_full_in_[STREAM], d_q_in, d_full_to_reduced_map_, M);
                gpu_error_check(cudaPeekAtLastError());
                fft_in = d_q_full_in_[STREAM];
                fft_out = d_q_full_out_[STREAM];
            }
            else
            {
                throw_with_line_number("Space group reduced basis is only supported for real fields.");
            }
        }

        if (this->is_periodic_)
        {
            // 3D fourier discrete transform, forward
            fft_forward(STREAM, fft_in, d_qk_in_1_one[STREAM], 1);
            gpu_error_check(cudaPeekAtLastError());

            // Multiply exp(-k^2 ds/12) in fourier space, in all 3 directions
            ker_multi_complex_real<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_qk_in_1_one[STREAM], _d_boltz_bond_half, 1.0/static_cast<double>(M), M_COMPLEX);
            gpu_error_check(cudaPeekAtLastError());

            // 3D fourier discrete transform, backward
            fft_backward(STREAM, d_qk_in_1_one[STREAM], fft_out, 1);
            gpu_error_check(cudaPeekAtLastError());
        }
        else
        {
            // Non-periodic BC (reflecting/absorbing): CudaFFT (DCT/DST), real
            // coefficients, default stream. backward() includes normalization (no 1/M).
            fft_[STREAM]->forward(reinterpret_cast<T*>(fft_in), d_rk_in_1_one[STREAM]);
            // [type-specific algorithm] DCT/DST coefficients are real for real fields and interleaved complex pairs for complex fields.
            if constexpr (std::is_same<T, double>::value)
                ker_multi<<<N_BLOCKS, N_THREADS>>>(d_rk_in_1_one[STREAM], d_rk_in_1_one[STREAM], _d_boltz_bond_half, 1.0, M_COMPLEX);
            else
                ker_multi_complex_real<<<N_BLOCKS, N_THREADS>>>(reinterpret_cast<cuDoubleComplex*>(d_rk_in_1_one[STREAM]), _d_boltz_bond_half, 1.0, M_COMPLEX);
            gpu_error_check(cudaPeekAtLastError());
            fft_[STREAM]->backward(d_rk_in_1_one[STREAM], reinterpret_cast<T*>(fft_out));
        }

        // Reduce full grid → reduced basis
        if (space_group_ != nullptr)
        {
            ker_reduce_to_basis<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                d_q_out, fft_out, d_reduced_basis_indices_, n_basis_);
            gpu_error_check(cudaPeekAtLastError());
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
void CudaSolverPseudoDiscrete<T>::compute_single_segment_stress(
    const int STREAM,
    CuDeviceData<T> *d_q_pair, CuDeviceData<T> *d_segment_stress,
    std::string monomer_type, bool is_half_bond_length)
{
    try{
        // [real-only feature] Space-group reduced basis, CrysFFT and non-periodic stress are implemented for real fields; complex fields take the generic path or raise.
        if constexpr (std::is_same<T, std::complex<double>>::value)
        {
            if (!this->is_periodic_)
                throw_with_line_number(
                    "Stress computation for complex fields with non-periodic "
                    "boundary conditions is not implemented.");
        }
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int DIM = this->cb->get_dim();
        // const int M   = total_grid;
        const int M_COMPLEX = pseudo->get_total_complex_grid();

        auto bond_lengths = this->molecules->get_bond_lengths();
        double bond_length_sq;
        double *_d_boltz_bond;

        const double* _d_fourier_basis_x = pseudo->get_fourier_basis_x();
        const double* _d_fourier_basis_y = pseudo->get_fourier_basis_y();
        const double* _d_fourier_basis_z = pseudo->get_fourier_basis_z();
        const double* _d_fourier_basis_xy = pseudo->get_fourier_basis_xy();
        const double* _d_fourier_basis_xz = pseudo->get_fourier_basis_xz();
        const double* _d_fourier_basis_yz = pseudo->get_fourier_basis_yz();
        const int* _d_negative_k_idx = pseudo->get_negative_frequency_mapping();

        // Discrete chains always use ds_index=0 (global ds)
        if (is_half_bond_length)
        {
            bond_length_sq = 0.5*bond_lengths[monomer_type]*bond_lengths[monomer_type];
            _d_boltz_bond = pseudo->get_boltz_bond_half(monomer_type, 0);
        }
        else
        {
            bond_length_sq = bond_lengths[monomer_type]*bond_lengths[monomer_type];
            _d_boltz_bond = pseudo->get_boltz_bond(monomer_type, 0);
        }

        const int M = this->cb->get_total_grid();

        // =====================================================================
        // CrysFFT stress fast path (reduced grid, no full-grid FFT)
        // Mirrors CpuSolverPseudoBase::compute_single_segment_stress.
        // Discrete chains include the Boltzmann bond factor exp(-k²·b²·ds/6)
        // in the per-axis multiplier (b² already halved for half-bond steps).
        // ObliqueZ and all other cases fall through to the standard path.
        // =====================================================================
        // [real-only feature] Space-group reduced basis, CrysFFT and non-periodic stress are implemented for real fields; complex fields take the generic path or raise.
        if constexpr (std::is_same_v<T, double>)
        {
            // NOTE: The CrysFFT fast path computes only the diagonal sums
            // (V₁₁, V₂₂, V₃₃). This is fine because crystal systems eligible
            // for CrysFFT are orthogonal and never optimize angles. If the
            // force-off-diagonal flag is set (angle optimization active),
            // fall back to the standard path below — which can compute the
            // cross terms — instead of silently returning zeros for them.
            if (use_crysfft_ && space_group_ != nullptr && is_periodic_ && DIM == 3
                && this->cb->is_orthogonal()
                && !this->force_off_diagonal_stress_
                && (crysfft_mode_ == CudaCrysFFTMode::PmmmDct || crysfft_mode_ == CudaCrysFFTMode::Recursive3m))
            {
                const std::vector<double> lx_vec = this->cb->get_lx();

                CudaCrysFFTStressArgs args{};
                args.crysfft = crysfft_[STREAM];
                args.mode = crysfft_mode_;
                args.identity_map = crysfft_identity_map_;
                args.M_phys = crysfft_physical_size_;
                args.n_basis = n_basis_;
                args.M_full = M;
                args.d_phys_to_reduced = d_crysfft_phys_to_reduced_;
                args.d_reduced_to_phys = d_crysfft_reduced_to_phys_;
                args.d_orbit_counts = d_crysfft_orbit_counts_;
                args.d_phys_work = d_crysfft_in_[STREAM];
                args.d_phys_out = d_crysfft_out_[STREAM];
                args.d_reduce_buf = d_stress_sum[STREAM];
                args.d_cub_temp = d_temp_storage[STREAM];
                args.cub_temp_bytes = temp_storage_bytes[STREAM];
                args.space_group = space_group_;
                args.lx[0] = lx_vec[0];
                args.lx[1] = lx_vec[1];
                args.lx[2] = lx_vec[2];
                args.bond_length_sq = bond_length_sq;
                args.include_bond_factor = true;  // discrete chains
                args.global_ds = this->molecules->get_global_ds();
                args.stream = streams[STREAM][0];

                cuda_crysfft_compute_single_segment_stress(
                    args, d_q_pair, &d_q_pair[n_basis_], d_segment_stress);
                return;
            }
        }

        // Space group without CrysFFT fast path: the propagator pair arrives in
        // the reduced basis (packed at stride n_basis_). Expand both propagators
        // to the full grid into a contiguous 2*M buffer for the batched FFT.
        CuDeviceData<T>* d_q_pair_full = d_q_pair;
        if (space_group_ != nullptr)
        {
            // [real-only feature] Space-group reduced basis, CrysFFT and non-periodic stress are implemented for real fields; complex fields take the generic path or raise.
            if constexpr (std::is_same_v<T, double>)
            {
                ker_expand_reduced_basis<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                    d_q_full_in_[STREAM], d_q_pair, d_full_to_reduced_map_, M);
                ker_expand_reduced_basis<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                    &d_q_full_in_[STREAM][M], &d_q_pair[n_basis_], d_full_to_reduced_map_, M);
                gpu_error_check(cudaPeekAtLastError());
                d_q_pair_full = d_q_full_in_[STREAM];
            }
            else
            {
                throw_with_line_number("Space group reduced basis is only supported for real fields.");
            }
        }

        if (is_periodic_)
        {
            // Periodic BC: Execute a forward FFT using cuFFT (batched, two propagators)
            fft_forward(STREAM, d_q_pair_full, d_qk_in_1_two[STREAM], 2);
            gpu_error_check(cudaPeekAtLastError());

            // [type-specific algorithm] Real fields store the Hermitian half-spectrum, so the
            // conjugate product is formed directly; complex fields need the -k index map.
            // Multiply two propagators in the fourier spaces
            if constexpr (std::is_same<T, double>::value)
                ker_multi_complex_conjugate<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_q_multi[STREAM], &d_qk_in_1_two[STREAM][0], &d_qk_in_1_two[STREAM][M_COMPLEX], M_COMPLEX);
            else
            {
                ker_copy_data_with_idx<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_qk_in_1_one[STREAM], &d_qk_in_1_two[STREAM][M_COMPLEX], _d_negative_k_idx, M_COMPLEX);
                ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_q_multi[STREAM], &d_qk_in_1_two[STREAM][0], d_qk_in_1_one[STREAM], 1.0, M_COMPLEX);
            }
            gpu_error_check(cudaPeekAtLastError());
        }
        else
        {
            // Non-periodic BC: use CudaFFT (DCT/DST)
            // [real-only feature] Space-group reduced basis, CrysFFT and non-periodic stress are implemented for real fields; complex fields take the generic path or raise.
            if constexpr (!std::is_same<T, double>::value)
            {
                throw_with_line_number("Discrete chain stress computation with non-periodic BC is only supported for real fields (double).");
            }
            else
            {
                double* d_q1 = d_q_pair_full;
                double* d_q2 = &d_q_pair_full[M];
                double* rk_1 = d_rk_in_1_one[STREAM];
                double* rk_2 = d_rk_in_2_one[STREAM];

                // Transform each propagator separately using DCT/DST
                fft_[STREAM]->forward(d_q1, rk_1);
                fft_[STREAM]->forward(d_q2, rk_2);

                // Multiply the two transforms element-wise (real coefficients)
                // Per-mode Parseval weights (2 interior / 1 special modes) are folded
                // into the fourier_basis tables by Pseudo::update_weighted_fourier_basis_mixed.
                // CudaFFT's real-coefficient forward is 2x the textbook DCT-II/DST-II
                // convention per transformed dimension (CudaRealTransform matches FFTW's
                // unnormalized REDFT10/RODFT10, while the CPU FftwFFT/MklFFT scale by 0.5);
                // the coefficient product therefore carries 4^DIM, compensated here.
                // (All dimensions are r2r here: mixed periodic/non-periodic is rejected.)
                const double dct_scale = std::pow(0.25, DIM);
                ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_q_multi[STREAM], rk_1, rk_2, dct_scale, M_COMPLEX);
            }
            gpu_error_check(cudaPeekAtLastError());
        }

        // Multiply by Boltzmann bond factor and bond length squared
        ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_q_multi[STREAM], d_q_multi[STREAM], _d_boltz_bond, bond_length_sq, M_COMPLEX);
        gpu_error_check(cudaPeekAtLastError());

        // fourier_basis arrays contain v⊗v dyad components of the deformation
        // vector v = 2π g⁻¹ m (units 1/L², NOT the Cartesian wavevector):
        // _d_fourier_basis_x = v₁², _d_fourier_basis_xy = v₁v₂, etc.
        // The accumulated sums V_ij = Σ kernel·v_i·v_j are converted to
        // lattice-parameter derivatives in compute_stress().
        // No cross-term corrections needed.

        // ============================================================================
        // PERFORMANCE-CRITICAL: Orthogonal box optimization
        // ============================================================================
        // For orthogonal boxes (all angles = 90°), the cross-term sums (V₁₂, V₁₃, V₂₃)
        // are not computed: angle optimization is inactive for orthogonal crystal
        // systems, so those slots are never consumed. This optimization
        // reduces the number of GPU kernel calls from 6 to 3 for 3D boxes,
        // providing ~10% speedup when box_is_altering=True.
        //
        // DO NOT REMOVE THIS OPTIMIZATION without benchmarking! We have experienced
        // performance regressions before when this check was accidentally removed.
        // See git history for commit 6d1dc54 which caused a regression by always
        // computing all 6 components.
        //
        // Explicit opt-in: force_off_diagonal_stress_ (set when angle optimization
        // is active) disables the skip so that angle derivatives are available
        // even when the box is at exactly 90° angles. The skip remains the default.
        // ============================================================================
        const bool is_orthogonal = this->cb->is_orthogonal() && !this->force_off_diagonal_stress_;

        if ( DIM == 3 )
        {
            // V₁₁
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_stress_sum[STREAM], d_q_multi[STREAM], _d_fourier_basis_x, 1.0, M_COMPLEX);
            gpu_error_check(cudaPeekAtLastError());
            cuda_reduce_sum(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[0], M_COMPLEX, streams[STREAM][0]);
            gpu_error_check(cudaPeekAtLastError());

            // V₂₂
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_stress_sum[STREAM], d_q_multi[STREAM], _d_fourier_basis_y, 1.0, M_COMPLEX);
            gpu_error_check(cudaPeekAtLastError());
            cuda_reduce_sum(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[1], M_COMPLEX, streams[STREAM][0]);
            gpu_error_check(cudaPeekAtLastError());

            // V₃₃
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_stress_sum[STREAM], d_q_multi[STREAM], _d_fourier_basis_z, 1.0, M_COMPLEX);
            gpu_error_check(cudaPeekAtLastError());
            cuda_reduce_sum(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[2], M_COMPLEX, streams[STREAM][0]);
            gpu_error_check(cudaPeekAtLastError());

            // Cross-terms: only compute for non-orthogonal boxes (triclinic lattices)
            // For orthogonal boxes, these are mathematically zero.
            if (!is_orthogonal)
            {
                // V₁₂
                ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_stress_sum[STREAM], d_q_multi[STREAM], _d_fourier_basis_xy, 1.0, M_COMPLEX);
                gpu_error_check(cudaPeekAtLastError());
                cuda_reduce_sum(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[3], M_COMPLEX, streams[STREAM][0]);
                gpu_error_check(cudaPeekAtLastError());

                // V₁₃
                ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_stress_sum[STREAM], d_q_multi[STREAM], _d_fourier_basis_xz, 1.0, M_COMPLEX);
                gpu_error_check(cudaPeekAtLastError());
                cuda_reduce_sum(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[4], M_COMPLEX, streams[STREAM][0]);
                gpu_error_check(cudaPeekAtLastError());

                // V₂₃
                ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_stress_sum[STREAM], d_q_multi[STREAM], _d_fourier_basis_yz, 1.0, M_COMPLEX);
                gpu_error_check(cudaPeekAtLastError());
                cuda_reduce_sum(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[5], M_COMPLEX, streams[STREAM][0]);
                gpu_error_check(cudaPeekAtLastError());
            }
        }
        if ( DIM == 2 )
        {
            // For non-periodic BC, 2D grid is mapped to y-z axes internally (tnx = {1, nx[0], nx[1]})
            // so stress data is stored in fourier_basis_y and fourier_basis_z
            const double* _d_basis_xx = is_periodic_ ? _d_fourier_basis_x : _d_fourier_basis_y;
            const double* _d_basis_yy = is_periodic_ ? _d_fourier_basis_y : _d_fourier_basis_z;
            const double* _d_basis_xy = is_periodic_ ? _d_fourier_basis_xy : _d_fourier_basis_yz;

            // V₁₁
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_stress_sum[STREAM], d_q_multi[STREAM], _d_basis_xx, 1.0, M_COMPLEX);
            gpu_error_check(cudaPeekAtLastError());
            cuda_reduce_sum(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[0], M_COMPLEX, streams[STREAM][0]);
            gpu_error_check(cudaPeekAtLastError());

            // V₂₂
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_stress_sum[STREAM], d_q_multi[STREAM], _d_basis_yy, 1.0, M_COMPLEX);
            gpu_error_check(cudaPeekAtLastError());
            cuda_reduce_sum(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[1], M_COMPLEX, streams[STREAM][0]);
            gpu_error_check(cudaPeekAtLastError());

            // V₁₂: only compute for non-orthogonal boxes
            if (!is_orthogonal)
            {
                ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_stress_sum[STREAM], d_q_multi[STREAM], _d_basis_xy, 1.0, M_COMPLEX);
                gpu_error_check(cudaPeekAtLastError());
                cuda_reduce_sum(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[2], M_COMPLEX, streams[STREAM][0]);
                gpu_error_check(cudaPeekAtLastError());
            }
        }
        if ( DIM == 1 )
        {
            // For non-periodic BC, 1D grid is mapped to z-axis internally (tnx = {1, 1, nx[0]})
            // so stress data is stored in fourier_basis_z
            const double* _d_basis_xx = is_periodic_ ? _d_fourier_basis_x : _d_fourier_basis_z;

            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_stress_sum[STREAM], d_q_multi[STREAM], _d_basis_xx, 1.0, M_COMPLEX);
            gpu_error_check(cudaPeekAtLastError());
            cuda_reduce_sum(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[0], M_COMPLEX, streams[STREAM][0]);
            gpu_error_check(cudaPeekAtLastError());
        }

    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

// Explicit template instantiation

template class CudaSolverPseudoDiscrete<double>;
template class CudaSolverPseudoDiscrete<std::complex<double>>;
