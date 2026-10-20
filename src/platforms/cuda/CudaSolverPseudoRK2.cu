/**
 * @file CudaSolverPseudoRK2.cu
 * @brief CUDA pseudo-spectral solver for continuous Gaussian chains using RK2.
 *
 * Implements RK2 (Rasmussen-Kalosakas 2nd-order) for advancing chain propagators
 * using cuFFT for Fourier transforms and multiple CUDA streams for concurrent
 * computation.
 *
 * **RK2 (Operator Splitting):**
 *
 * Single full step with Strang splitting:
 *     q(s+ds) = exp(-w·ds/2) · FFT⁻¹[ exp(-k²b²ds/6) · FFT[ exp(-w·ds/2) · q(s) ] ]
 *
 * **Comparison to RQM4:**
 *
 * RK2 uses only a single full step (2 FFTs) while RQM4 combines full and half
 * steps via Richardson extrapolation (6 FFTs). RK2 is faster but only
 * 2nd-order accurate (vs 4th-order for RQM4).
 *
 * **Template Instantiations:**
 *
 * - CudaSolverPseudoRK2<double>: Real field solver
 * - CudaSolverPseudoRK2<std::complex<double>>: Complex field solver
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
#include "CudaSolverPseudoRK2.h"
#include "CudaCrysFFT.h"
#include "CudaCrysFFTRecursive3m.h"
#include "CudaCrysFFTObliqueZ.h"
#include "CudaCrysFFTStress.h"
#include "CrysFFTSelector.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

template <typename T>
CudaSolverPseudoRK2<T>::CudaSolverPseudoRK2(
    ComputationBox<T>* cb,
    Molecules *molecules,
    int n_streams,
    cudaStream_t streams[MAX_STREAMS][2],
    [[maybe_unused]] bool reduce_memory)
    : CudaSolverPseudoBase<T>(cb, molecules, n_streams, streams, /*max_fft_batch=*/2,
                              /*q_full_in_factor=*/1, /*with_exp_dw_half=*/false,
                              /*complex_requires_periodic=*/true)
{
    try{
        const int M = cb->get_total_grid();
        const int M_COMPLEX = pseudo->get_total_complex_grid();
        for(int i=0; i<n_streams; i++)
        {
            d_rk_in_1[i] = nullptr;
            d_rk_in_2[i] = nullptr;
        }
        if (!is_periodic_)
        {
            // Real coefficient buffers for the DCT/DST path
            for(int i=0; i<n_streams; i++)
            {
                gpu_error_check(cudaMalloc((void**)&d_rk_in_1[i], sizeof(double)*M_COMPLEX));
                gpu_error_check(cudaMalloc((void**)&d_rk_in_2[i], sizeof(double)*M_COMPLEX));
            }
        }

        // Workspaces: advance_propagator() uses the first half of each, the
        // stress computation both halves as the contiguous propagator pair
        for(int i=0; i<n_streams; i++)
        {
            gpu_error_check(cudaMalloc((void**)&d_q_work[i], sizeof(T)*2*M));
            gpu_error_check(cudaMalloc((void**)&d_qk_work[i], sizeof(cuDoubleComplex)*2*M_COMPLEX));
        }
        this->update_laplacian_operator();
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

template <typename T>
CudaSolverPseudoRK2<T>::~CudaSolverPseudoRK2()
{

    if (!is_periodic_)
    {
        // Free real coefficient buffers
        for(int i=0; i<n_streams; i++)
        {
            if (d_rk_in_1[i] != nullptr) cudaFree(d_rk_in_1[i]);
            if (d_rk_in_2[i] != nullptr) cudaFree(d_rk_in_2[i]);
        }
    }

    // Free workspace arrays
    for(int i=0; i<n_streams; i++)
    {
        cudaFree(d_q_work[i]);
        cudaFree(d_qk_work[i]);
    }

}

template <typename T>
void CudaSolverPseudoRK2<T>::update_dw(std::string device, std::map<std::string, const T*> w_input)
{
    // exp(-w ds/2): the half-step factor applied on both sides of the diffusion step
    this->update_dw_impl(device, w_input, false, 0.5, 0.0);
}

// Advance propagator using RK2 (single full step)
template <typename T>
void CudaSolverPseudoRK2<T>::advance_propagator(
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
                    ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                        d_crysfft_in_[STREAM], d_q_in, _d_exp_dw, 1.0, M_phys);
                    gpu_error_check(cudaPeekAtLastError());

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
                ker_expand_reduced_basis_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                    d_crysfft_in_[STREAM], d_q_in, _d_exp_dw, d_crysfft_phys_to_reduced_, M_phys);
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

        CuDeviceData<T> *fft_in = d_q_in;
        CuDeviceData<T> *fft_out = d_q_out;
        if (space_group_ != nullptr)
        {
            // [real-only feature] Space-group reduced basis, CrysFFT and non-periodic stress are implemented for real fields; complex fields take the generic path or raise.
            if constexpr (std::is_same_v<T, double>)
            {
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

        if (is_periodic_)
        {
            // ============================================================
            // Periodic BC: Use cuFFT with complex coefficients
            // ============================================================

            // Step 1: Evaluate exp(-w*ds/2) in real space
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                d_q_work[STREAM], fft_in, _d_exp_dw, 1.0, M);
            gpu_error_check(cudaPeekAtLastError());

            // Step 2: Execute a forward FFT
            fft_forward(STREAM, d_q_work[STREAM], d_qk_work[STREAM], 1);
            gpu_error_check(cudaPeekAtLastError());

            // Step 3: Multiply exp(-k^2 ds/6) in fourier space
            ker_multi_complex_real<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                d_qk_work[STREAM], _d_boltz_bond, 1.0, M_COMPLEX);
            gpu_error_check(cudaPeekAtLastError());

            // Step 4: Execute a backward FFT
            fft_backward(STREAM, d_qk_work[STREAM], fft_out, 1);
            gpu_error_check(cudaPeekAtLastError());

            // Step 5: Evaluate exp(-w*ds/2) in real space and normalize
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                fft_out, fft_out, _d_exp_dw, 1.0/static_cast<double>(M), M);
            gpu_error_check(cudaPeekAtLastError());
        }
        else
        {
            // ============================================================
            // Non-periodic BC: Use CudaFFT (DCT/DST) with real coefficients
            // ============================================================

            // Cast device pointers for FFT interface
            T* d_q_step = reinterpret_cast<T*>(d_q_work[STREAM]);

            // Step 1: Evaluate exp(-w*ds/2) in real space
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_q_work[STREAM], fft_in, _d_exp_dw, 1.0, M);
            gpu_error_check(cudaPeekAtLastError());

            // Step 2: Forward transform (DCT/DST)
            fft_[STREAM]->forward_stream(d_q_step, d_rk_in_1[STREAM], streams[STREAM][0]);

            // Step 3: Multiply by Boltzmann factor (real coefficients)
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_rk_in_1[STREAM], d_rk_in_1[STREAM], _d_boltz_bond, 1.0, M_COMPLEX);
            gpu_error_check(cudaPeekAtLastError());

            // Step 4: Backward transform (inverse DCT/DST) - normalization included
            fft_[STREAM]->backward_stream(d_rk_in_1[STREAM], reinterpret_cast<T*>(fft_out), streams[STREAM][0]);

            // Step 5: Evaluate exp(-w*ds/2) in real space
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(fft_out, fft_out, _d_exp_dw, 1.0, M);
            gpu_error_check(cudaPeekAtLastError());
        }

        // Multiply mask
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
typename CudaSolverPseudoBase<T>::StressScratch CudaSolverPseudoRK2<T>::stress_scratch(int STREAM)
{
    return {d_q_work[STREAM], d_qk_work[STREAM], d_rk_in_1[STREAM], d_rk_in_2[STREAM]};
}

// Explicit template instantiation
template class CudaSolverPseudoRK2<double>;
template class CudaSolverPseudoRK2<std::complex<double>>;
