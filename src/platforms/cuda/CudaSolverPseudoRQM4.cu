/**
 * @file CudaSolverPseudoRQM4.cu
 * @brief CUDA pseudo-spectral solver for continuous Gaussian chains using RQM4.
 *
 * Implements RQM4 (Ranjan-Qin-Morse 4th-order using Richardson extrapolation)
 * for advancing chain propagators using cuFFT for Fourier transforms and
 * multiple CUDA streams for concurrent computation.
 *
 * **RQM4 (Richardson Extrapolation):**
 *
 * The propagator is advanced using two step sizes combined for 4th-order:
 * - Full step ds: O(ds²) accuracy
 * - Two half steps ds/2: O(ds²) accuracy
 * - Combination: (4*q_half - q_full)/3 gives O(ds⁴) accuracy
 *
 * **cuFFT Plans:**
 *
 * - cuFFT plans and DCT/DST objects are owned by CudaSolverPseudoBase
 *   (fft_forward / fft_backward / fft_)
 * - Stream-associated for concurrent execution
 *
 * **Stress Calculation:**
 *
 * CudaSolverPseudoBase::compute_single_segment_stress() (shared with RK2 and
 * the discrete chain) computes dQ/dL by multiplying propagator products in
 * Fourier space by weighted basis functions; this class only lends its
 * workspaces (stress_scratch()) and reports no bond factor.
 *
 * **Template Instantiations:**
 *
 * - CudaSolverPseudoRQM4<double>: Real field solver
 * - CudaSolverPseudoRQM4<std::complex<double>>: Complex field solver
 *
 * @see CudaPseudo for Boltzmann factors
 * @see CudaComputationContinuous for propagator orchestration
 */

#include <iostream>
#include <cmath>
#include <numbers>
#include <array>
#include <thrust/reduce.h>

#include "CudaPseudo.h"
#include "CudaTypeDispatch.h"
#include "CudaSolverPseudoRQM4.h"
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
CudaSolverPseudoRQM4<T>::CudaSolverPseudoRQM4(
    ComputationBox<T>* cb,
    Molecules *molecules,
    int n_streams,
    cudaStream_t streams[MAX_STREAMS][2],
    [[maybe_unused]] bool reduce_memory)
    : CudaSolverPseudoBase<T>(cb, molecules, n_streams, streams, /*max_fft_batch=*/2,
                              /*q_full_in_factor=*/1, /*with_exp_dw_half=*/true,
                              /*complex_requires_periodic=*/true)
{
    try{
        const int M = cb->get_total_grid();
        const int M_COMPLEX = pseudo->get_total_complex_grid();
        for(int i=0; i<n_streams; i++)
        {
            d_rk_in_1_one[i] = nullptr;
            d_rk_in_2_one[i] = nullptr;
        }
        if (!is_periodic_)
        {
            // Real coefficient buffers for the DCT/DST path
            for(int i=0; i<n_streams; i++)
            {
                gpu_error_check(cudaMalloc((void**)&d_rk_in_1_one[i], sizeof(double)*M_COMPLEX));
                gpu_error_check(cudaMalloc((void**)&d_rk_in_2_one[i], sizeof(double)*M_COMPLEX));
            }
        }

        // Allocate memory for pseudo-spectral: advance_propagator()
        for(int i=0; i<n_streams; i++)
        {
            gpu_error_check(cudaMalloc((void**)&d_q_step_1_one[i], sizeof(T)*M));
            gpu_error_check(cudaMalloc((void**)&d_q_step_2_one[i], sizeof(T)*M));
            gpu_error_check(cudaMalloc((void**)&d_q_step_1_two[i], sizeof(T)*2*M));

            gpu_error_check(cudaMalloc((void**)&d_qk_in_2_one[i], sizeof(cuDoubleComplex)*M_COMPLEX));
            gpu_error_check(cudaMalloc((void**)&d_qk_in_1_two[i], sizeof(cuDoubleComplex)*2*M_COMPLEX));
        }
        this->update_laplacian_operator();
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
CudaSolverPseudoRQM4<T>::~CudaSolverPseudoRQM4()
{

    if (!is_periodic_)
    {
        // Free real coefficient buffers
        for(int i=0; i<n_streams; i++)
        {
            if (d_rk_in_1_one[i] != nullptr) cudaFree(d_rk_in_1_one[i]);
            if (d_rk_in_2_one[i] != nullptr) cudaFree(d_rk_in_2_one[i]);
        }
    }

    // For pseudo-spectral: advance_propagator()
    for(int i=0; i<n_streams; i++)
    {
        cudaFree(d_q_step_1_one[i]);
        cudaFree(d_q_step_2_one[i]);
        cudaFree(d_q_step_1_two[i]);
        cudaFree(d_qk_in_2_one[i]);
        cudaFree(d_qk_in_1_two[i]);
    }

}

template <typename T>
void CudaSolverPseudoRQM4<T>::update_dw(std::string device, std::map<std::string, const T*> w_input)
{
    // exp(-w ds/2) for the full step and exp(-w ds/4) for the half step
    this->update_dw_impl(device, w_input, true, 0.5, 0.25);
}
// Advance propagator using RQM4
template <typename T>
void CudaSolverPseudoRQM4<T>::advance_propagator(
        const int STREAM,
        CuDeviceData<T> *d_q_in, CuDeviceData<T> *d_q_out,
        std::string monomer_type, double *d_q_mask, int ds_index)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = cb->get_total_grid();
        const int M_COMPLEX = pseudo->get_total_complex_grid();

        CuDeviceData<T> *_d_exp_dw = this->d_exp_dw[ds_index][monomer_type];
        CuDeviceData<T> *_d_exp_dw_half = this->d_exp_dw_half[ds_index][monomer_type];
        const double* _d_boltz_bond      = pseudo->get_boltz_bond     (monomer_type, ds_index);
        const double* _d_boltz_bond_half = pseudo->get_boltz_bond_half(monomer_type, ds_index);

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
                double coeff_half = bond_length_sq * local_ds / 12.0;

                const int M_phys = crysfft_physical_size_;
                double* d_full_step = reinterpret_cast<double*>(d_q_step_1_one[STREAM]);
                double* d_half_step = reinterpret_cast<double*>(d_q_step_2_one[STREAM]);

                if (crysfft_identity_map_)
                {
                    // ===== Step 1: Full step =====
                    ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                        d_crysfft_in_[STREAM], d_q_in, _d_exp_dw, 1.0, M_phys);
                    gpu_error_check(cudaPeekAtLastError());

                    crysfft_[STREAM]->set_contour_step(coeff_full);
                    crysfft_[STREAM]->diffusion(d_crysfft_in_[STREAM], d_full_step, streams[STREAM][0]);

                    // ===== Step 2: Two half steps =====
                    ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                        d_crysfft_in_[STREAM], d_q_in, _d_exp_dw_half, 1.0, M_phys);
                    gpu_error_check(cudaPeekAtLastError());

                    crysfft_[STREAM]->set_contour_step(coeff_half);
                    crysfft_[STREAM]->diffusion(d_crysfft_in_[STREAM], d_crysfft_out_[STREAM], streams[STREAM][0]);

                    ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                        d_crysfft_out_[STREAM], d_crysfft_out_[STREAM], _d_exp_dw, 1.0, M_phys);
                    gpu_error_check(cudaPeekAtLastError());

                    crysfft_[STREAM]->diffusion(d_crysfft_out_[STREAM], d_half_step, streams[STREAM][0]);

                    // ===== Richardson extrapolation with exp_dw factors =====
                    ker_lin_comb_exp<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                        d_q_out,
                        d_half_step, _d_exp_dw_half, 4.0/3.0,
                        d_full_step, _d_exp_dw, -1.0/3.0,
                        d_q_mask, M_phys);
                    gpu_error_check(cudaPeekAtLastError());
                    return;
                }

                // ===== Step 1: Full step =====
                ker_expand_reduced_basis_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                    d_crysfft_in_[STREAM], d_q_in, _d_exp_dw, d_crysfft_phys_to_reduced_, M_phys);
                gpu_error_check(cudaPeekAtLastError());

                crysfft_[STREAM]->set_contour_step(coeff_full);
                crysfft_[STREAM]->diffusion(d_crysfft_in_[STREAM], d_crysfft_out_[STREAM], streams[STREAM][0]);

                ker_multi_map<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                    d_crysfft_out_[STREAM], _d_exp_dw, d_crysfft_phys_to_reduced_, M_phys);
                gpu_error_check(cudaPeekAtLastError());

                ker_reduce_to_basis<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                    d_full_step, d_crysfft_out_[STREAM], d_crysfft_reduced_to_phys_, n_basis_);
                gpu_error_check(cudaPeekAtLastError());

                // ===== Step 2: Two half steps =====
                ker_expand_reduced_basis_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                    d_crysfft_in_[STREAM], d_q_in, _d_exp_dw_half, d_crysfft_phys_to_reduced_, M_phys);
                gpu_error_check(cudaPeekAtLastError());

                crysfft_[STREAM]->set_contour_step(coeff_half);
                crysfft_[STREAM]->diffusion(d_crysfft_in_[STREAM], d_crysfft_out_[STREAM], streams[STREAM][0]);

                ker_multi_map<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                    d_crysfft_out_[STREAM], _d_exp_dw, d_crysfft_phys_to_reduced_, M_phys);
                gpu_error_check(cudaPeekAtLastError());

                crysfft_[STREAM]->diffusion(d_crysfft_out_[STREAM], d_crysfft_in_[STREAM], streams[STREAM][0]);

                ker_multi_map<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                    d_crysfft_in_[STREAM], _d_exp_dw_half, d_crysfft_phys_to_reduced_, M_phys);
                gpu_error_check(cudaPeekAtLastError());

                ker_reduce_to_basis<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                    d_half_step, d_crysfft_in_[STREAM], d_crysfft_reduced_to_phys_, n_basis_);
                gpu_error_check(cudaPeekAtLastError());

                // ===== Richardson extrapolation on reduced basis =====
                ker_lin_comb<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                    d_q_out, 4.0/3.0, d_half_step, -1.0/3.0, d_full_step, n_basis_);
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
            // Expand reduced basis → full grid
            ker_expand_reduced_basis<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                d_q_full_in_[STREAM], d_q_in, d_full_to_reduced_map_, M);
            gpu_error_check(cudaPeekAtLastError());
            fft_in = d_q_full_in_[STREAM];
            fft_out = d_q_full_out_[STREAM];
        }

        if (is_periodic_)
        {
            // ============================================================
            // Periodic BC: Use batched cuFFT with complex coefficients
            // ============================================================

            // step 1/2: Evaluate exp(-w*ds/2) in real space
            // step 1/4: Evaluate exp(-w*ds/4) in real space
            ker_multi_exp_dw_two<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                &d_q_step_1_two[STREAM][0], fft_in, _d_exp_dw,
                &d_q_step_1_two[STREAM][M], fft_in, _d_exp_dw_half, 1.0, M);
            gpu_error_check(cudaPeekAtLastError());

            // step 1/2: Execute a forward FFT
            // step 1/4: Execute a forward FFT
            fft_forward(STREAM, d_q_step_1_two[STREAM], d_qk_in_1_two[STREAM], 2);
            gpu_error_check(cudaPeekAtLastError());

            // step 1/2: Multiply exp(-k^2 ds/6)  in fourier space
            // step 1/4: Multiply exp(-k^2 ds/12) in fourier space
            ker_complex_real_multi_bond_two<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                &d_qk_in_1_two[STREAM][0],         _d_boltz_bond,
                &d_qk_in_1_two[STREAM][M_COMPLEX], _d_boltz_bond_half, M_COMPLEX);
            gpu_error_check(cudaPeekAtLastError());

            // step 1/2: Execute a backward FFT
            // step 1/4: Execute a backward FFT
            fft_backward(STREAM, d_qk_in_1_two[STREAM], d_q_step_1_two[STREAM], 2);
            gpu_error_check(cudaPeekAtLastError());

            // step 1/2: Evaluate exp(-w*ds/2) in real space
            // step 1/4: Evaluate exp(-w*ds/2) in real space
            ker_multi_exp_dw_two<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                d_q_step_1_one[STREAM], &d_q_step_1_two[STREAM][0], _d_exp_dw,
                d_q_step_2_one[STREAM], &d_q_step_1_two[STREAM][M], _d_exp_dw, 1.0/static_cast<double>(M), M);
            gpu_error_check(cudaPeekAtLastError());

            // step 1/4: Execute a forward FFT
            fft_forward(STREAM, d_q_step_2_one[STREAM], d_qk_in_2_one[STREAM], 1);
            gpu_error_check(cudaPeekAtLastError());

            // step 1/4: Multiply exp(-k^2 ds/12) in fourier space
            ker_multi_complex_real<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_qk_in_2_one[STREAM], _d_boltz_bond_half, 1.0, M_COMPLEX);
            gpu_error_check(cudaPeekAtLastError());

            // step 1/4: Execute a backward FFT
            fft_backward(STREAM, d_qk_in_2_one[STREAM], d_q_step_2_one[STREAM], 1);
            gpu_error_check(cudaPeekAtLastError());

            // step 1/4: Evaluate exp(-w*ds/4) in real space.
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_q_step_2_one[STREAM], d_q_step_2_one[STREAM], _d_exp_dw_half, 1.0/static_cast<double>(M), M);
            gpu_error_check(cudaPeekAtLastError());
        }
        else
        {
            // ============================================================
            // Non-periodic BC: Use CudaFFT (DCT/DST) with real coefficients
            // ============================================================
            // Each stream has its own CudaFFT object with independent work buffers;
            // transforms and kernels are issued on this propagator's stream.

            // Cast device pointers for FFT interface
            T* d_q_step_1 = reinterpret_cast<T*>(d_q_step_1_one[STREAM]);
            T* d_q_step_2 = reinterpret_cast<T*>(d_q_step_2_one[STREAM]);

            // ===== Step 1: Full step =====
            // Evaluate exp(-w*ds/2) in real space
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_q_step_1_one[STREAM], fft_in, _d_exp_dw, 1.0, M);
            gpu_error_check(cudaPeekAtLastError());

            // Forward transform (DCT/DST)
            fft_[STREAM]->forward_stream(d_q_step_1, d_rk_in_1_one[STREAM], streams[STREAM][0]);

            // Multiply by Boltzmann factor (real coefficients)
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_rk_in_1_one[STREAM], d_rk_in_1_one[STREAM], _d_boltz_bond, 1.0, M_COMPLEX);
            gpu_error_check(cudaPeekAtLastError());

            // Backward transform (inverse DCT/DST) - normalization included
            fft_[STREAM]->backward_stream(d_rk_in_1_one[STREAM], d_q_step_1, streams[STREAM][0]);

            // Evaluate exp(-w*ds/2) in real space
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_q_step_1_one[STREAM], d_q_step_1_one[STREAM], _d_exp_dw, 1.0, M);
            gpu_error_check(cudaPeekAtLastError());

            // ===== Step 2: Two half steps =====
            // First half step
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_q_step_2_one[STREAM], fft_in, _d_exp_dw_half, 1.0, M);
            gpu_error_check(cudaPeekAtLastError());

            fft_[STREAM]->forward_stream(d_q_step_2, d_rk_in_2_one[STREAM], streams[STREAM][0]);

            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_rk_in_2_one[STREAM], d_rk_in_2_one[STREAM], _d_boltz_bond_half, 1.0, M_COMPLEX);
            gpu_error_check(cudaPeekAtLastError());

            fft_[STREAM]->backward_stream(d_rk_in_2_one[STREAM], d_q_step_2, streams[STREAM][0]);

            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_q_step_2_one[STREAM], d_q_step_2_one[STREAM], _d_exp_dw, 1.0, M);
            gpu_error_check(cudaPeekAtLastError());

            // Second half step
            fft_[STREAM]->forward_stream(d_q_step_2, d_rk_in_2_one[STREAM], streams[STREAM][0]);

            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_rk_in_2_one[STREAM], d_rk_in_2_one[STREAM], _d_boltz_bond_half, 1.0, M_COMPLEX);
            gpu_error_check(cudaPeekAtLastError());

            fft_[STREAM]->backward_stream(d_rk_in_2_one[STREAM], d_q_step_2, streams[STREAM][0]);

            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_q_step_2_one[STREAM], d_q_step_2_one[STREAM], _d_exp_dw_half, 1.0, M);
            gpu_error_check(cudaPeekAtLastError());
        }

        // ===== RQM4: Richardson extrapolation =====
        // Compute linear combination with 4/3 and -1/3 ratio
        ker_lin_comb<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(fft_out, 4.0/3.0, d_q_step_2_one[STREAM], -1.0/3.0, d_q_step_1_one[STREAM], M);
        gpu_error_check(cudaPeekAtLastError());

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
            gpu_error_check(cudaPeekAtLastError());
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
typename CudaSolverPseudoBase<T>::StressScratch CudaSolverPseudoRQM4<T>::stress_scratch(int STREAM)
{
    // d_q_step_1_two (2*M) and d_qk_in_1_two (2*M_COMPLEX) are the half-step
    // workspaces, free between propagator steps
    return {d_q_step_1_two[STREAM], d_qk_in_1_two[STREAM], d_rk_in_1_one[STREAM], d_rk_in_2_one[STREAM]};
}

// Explicit template instantiation
template class CudaSolverPseudoRQM4<double>;
template class CudaSolverPseudoRQM4<std::complex<double>>;
