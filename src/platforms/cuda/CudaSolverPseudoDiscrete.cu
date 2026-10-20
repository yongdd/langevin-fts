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
 * - get_stress_boltz_bond(): Bond factor of the shared stress computation
 *   (CudaSolverPseudoBase::compute_single_segment_stress())
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
        }
        if (!is_periodic_)
        {
            // Real-valued buffers for the non-periodic BC (DCT/DST) propagator and stress paths.
            // For complex fields the DCT/DST coefficients of the real and
            // imaginary parts are stored interleaved (complex pairs).
            const int rk_size = std::is_same<T, std::complex<double>>::value
                                ? 2 * M_COMPLEX : M_COMPLEX;
            for(int i=0; i<n_streams; i++)
            {
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
        if (d_rk_in_1_one[i] != nullptr) cudaFree(d_rk_in_1_one[i]);
        if (d_rk_in_2_one[i] != nullptr) cudaFree(d_rk_in_2_one[i]);
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
            gpu_error_check(cudaPeekAtLastError());

            // Execute a backward FFT
            fft_backward(STREAM, d_qk_in_1_one[STREAM], fft_out, 1);

            // Evaluate exp(-w*ds) in real space (1/M normalization from unnormalized cuFFT)
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(fft_out, fft_out, _d_exp_dw, 1.0/static_cast<double>(M), M);
            gpu_error_check(cudaPeekAtLastError());
        }
        else
        {
            // Non-periodic BC (reflecting/absorbing): use CudaFFT (DCT/DST) with real
            // coefficients. forward_stream() copies the input into its own work buffer,
            // so fft_in is preserved; everything is issued on this propagator's stream.
            fft_[STREAM]->forward_stream(reinterpret_cast<T*>(fft_in), d_rk_in_1_one[STREAM], streams[STREAM][0]);

            // Multiply bond Boltzmann factor (real factor; coefficients are
            // real for real fields, interleaved complex pairs for complex)
            // [type-specific algorithm] DCT/DST coefficients are real for real fields and interleaved complex pairs for complex fields.
            if constexpr (std::is_same<T, double>::value)
                ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_rk_in_1_one[STREAM], d_rk_in_1_one[STREAM], _d_boltz_bond, 1.0, M_COMPLEX);
            else
                ker_multi_complex_real<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(reinterpret_cast<cuDoubleComplex*>(d_rk_in_1_one[STREAM]), _d_boltz_bond, 1.0, M_COMPLEX);
            gpu_error_check(cudaPeekAtLastError());

            fft_[STREAM]->backward_stream(d_rk_in_1_one[STREAM], reinterpret_cast<T*>(fft_out), streams[STREAM][0]);

            // Evaluate exp(-w*ds); DCT/DST backward is already normalized (no 1/M).
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(fft_out, fft_out, _d_exp_dw, 1.0, M);
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
            // coefficients, on this propagator's stream. backward_stream() includes
            // normalization (no 1/M).
            fft_[STREAM]->forward_stream(reinterpret_cast<T*>(fft_in), d_rk_in_1_one[STREAM], streams[STREAM][0]);
            // [type-specific algorithm] DCT/DST coefficients are real for real fields and interleaved complex pairs for complex fields.
            if constexpr (std::is_same<T, double>::value)
                ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_rk_in_1_one[STREAM], d_rk_in_1_one[STREAM], _d_boltz_bond_half, 1.0, M_COMPLEX);
            else
                ker_multi_complex_real<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(reinterpret_cast<cuDoubleComplex*>(d_rk_in_1_one[STREAM]), _d_boltz_bond_half, 1.0, M_COMPLEX);
            gpu_error_check(cudaPeekAtLastError());
            fft_[STREAM]->backward_stream(d_rk_in_1_one[STREAM], reinterpret_cast<T*>(fft_out), streams[STREAM][0]);
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

// Explicit template instantiation

template <typename T>
typename CudaSolverPseudoBase<T>::StressScratch CudaSolverPseudoDiscrete<T>::stress_scratch(int STREAM)
{
    // d_q_full_in_ holds two full-grid fields (q_full_in_factor = 2) when a
    // space group is set; d_qk_in_1_two is the 2*M_COMPLEX FFT workspace
    return {d_q_full_in_[STREAM], d_qk_in_1_two[STREAM], d_rk_in_1_one[STREAM], d_rk_in_2_one[STREAM]};
}

template <typename T>
const double* CudaSolverPseudoDiscrete<T>::get_stress_boltz_bond(
    const std::string& monomer_type, bool is_half_bond_length) const
{
    // Discrete chains include the Boltzmann bond factor exp(-k²·b²·ds/6) in the
    // stress integrand; they always use ds_index = 0 (global ds)
    if (is_half_bond_length)
        return pseudo->get_boltz_bond_half(monomer_type, 0);
    return pseudo->get_boltz_bond(monomer_type, 0);
}

template class CudaSolverPseudoDiscrete<double>;
template class CudaSolverPseudoDiscrete<std::complex<double>>;
