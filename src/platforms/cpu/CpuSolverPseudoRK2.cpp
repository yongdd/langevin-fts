/**
 * @file CpuSolverPseudoRK2.cpp
 * @brief CPU pseudo-spectral solver for continuous chain propagators using RK2.
 *
 * Implements the operator splitting pseudo-spectral method with
 * RK2 (Rasmussen-Kalosakas 2nd-order) for continuous Gaussian chains.
 * Supports all boundary conditions (periodic, reflecting, absorbing).
 *
 * **Operator Splitting Scheme:**
 *
 * The diffusion equation dq/ds = (b^2/6)nabla^2 q - w(r)q is split as:
 * 1. Potential half-step: exp(-w·ds/2)
 * 2. Diffusion step: FFT -> multiply by exp(-b^2 k^2 ds/6) -> IFFT
 * 3. Potential half-step: exp(-w·ds/2)
 *
 * **Comparison to RQM4:**
 *
 * RK2 uses only a single full step, while RQM4 combines full and half steps
 * via Richardson extrapolation. RK2 is faster (2 FFTs vs 6 FFTs) but only
 * 2nd-order accurate (vs 4th-order for RQM4).
 *
 * **Template Instantiations:**
 *
 * - CpuSolverPseudoRK2<double>: Real fields with r2c FFT
 * - CpuSolverPseudoRK2<std::complex<double>>: Complex fields
 *
 * @see Pseudo for Boltzmann factor computation
 * @see ComputationContinuous for usage context
 */

#include <iostream>
#include <cmath>
#include <complex>
#include <type_traits>

#include "CpuSolverPseudoRK2.h"

//------------------------------------------------------------------------------
// Constructor
//------------------------------------------------------------------------------
template <typename T>
CpuSolverPseudoRK2<T>::CpuSolverPseudoRK2(ComputationBox<T>* cb, Molecules *molecules, FFTBackend backend)
{
    try
    {
        // Initialize shared components (FFT, Pseudo, etc.)
        this->init_shared(cb, molecules, backend);

        const int M = cb->get_total_grid();

        // Ensure ContourLengthMapping is finalized before using it
        molecules->finalize_contour_length_mapping();

        // Get unique ds values from ContourLengthMapping
        const ContourLengthMapping& mapping = molecules->get_contour_length_mapping();
        int n_unique_ds = mapping.get_n_unique_ds();

        // Create exp_dw vectors for each ds_index and monomer type
        // Note: RK2 only needs exp_dw, not exp_dw_half (no Richardson extrapolation)
        // Also register local_ds values with Pseudo for boltz_bond computation
        for (int ds_idx = 0; ds_idx < n_unique_ds; ++ds_idx)
        {
            double local_ds = mapping.get_ds_from_index(ds_idx);
            this->pseudo->add_ds_value(ds_idx, local_ds);
            this->register_ds_value(ds_idx, local_ds);

            for (const auto& item : molecules->get_bond_lengths())
            {
                std::string monomer_type = item.first;
                this->exp_dw[ds_idx][monomer_type].resize(M);
            }
        }

        // Finalize ds values to allocate boltz_bond arrays
        // (update_laplacian_operator will compute the actual values)
        this->pseudo->finalize_ds_values();
        this->update_laplacian_operator();
    }
    catch (std::exception& exc)
    {
        // Nothing to free here: the FFT and Pseudo objects created by
        // init_shared() are owning members of CpuSolverPseudoBase
        throw_without_line_number(exc.what());
    }
}

//------------------------------------------------------------------------------
// Destructor
//------------------------------------------------------------------------------
template <typename T>
CpuSolverPseudoRK2<T>::~CpuSolverPseudoRK2()
{
    // exp_dw vectors are automatically cleaned up;
    // the shared components (FFT, Pseudo) are owning members of the base
}

//------------------------------------------------------------------------------
// Update dw (Boltzmann factors from field)
//------------------------------------------------------------------------------
template <typename T>
void CpuSolverPseudoRK2<T>::update_dw(std::map<std::string, const T*> w_input)
{
    // Fields on the grid the Boltzmann factors live on (full grid, or the
    // reduced basis on the CrysFFT path); M_use is the size of each exp_dw array
    std::map<std::string, std::vector<T>> w_full_cache;
    std::map<std::string, const T*> w_use;
    const int M_use = this->expand_w_for_update_dw(w_input, w_full_cache, w_use);

    // Get unique ds values from ContourLengthMapping
    const ContourLengthMapping& mapping = this->molecules->get_contour_length_mapping();
    int n_unique_ds = mapping.get_n_unique_ds();

    // Compute exp_dw for each ds_index and monomer type
    for (int ds_idx = 0; ds_idx < n_unique_ds; ++ds_idx)
    {
        double local_ds = mapping.get_ds_from_index(ds_idx);

        for (const auto& item : w_use)
        {
            const std::string& monomer_type = item.first;
            const T* w = item.second;

            if (!this->exp_dw[ds_idx].contains(monomer_type))
                throw_with_line_number("monomer_type \"" + monomer_type + "\" is not in exp_dw[" + std::to_string(ds_idx) + "].");

            std::vector<T>& exp_dw_vec = this->exp_dw.at(ds_idx).at(monomer_type);
            exp_dw_vec.resize(M_use);

            for (int i = 0; i < M_use; ++i)
            {
                exp_dw_vec[i] = std::exp(-w[i] * local_ds * 0.5);
            }
        }
    }
}

//------------------------------------------------------------------------------
// Advance propagator (RK2 - single full step)
//------------------------------------------------------------------------------
template <typename T>
void CpuSolverPseudoRK2<T>::advance_propagator(
    T* q_in, T* q_out, std::string monomer_type, const double* q_mask, int ds_index)
{
    try
    {
        const int M_full = this->cb->get_total_grid();
        const int M_reduced = (this->space_group_ != nullptr)
            ? this->space_group_->get_n_reduced_basis()
            : M_full;
        const int M_COMPLEX = this->pseudo->get_total_complex_grid();

        // For periodic BC, coefficient array is actually complex (interleaved real/imag)
        // so allocate extra space
        int coeff_size = this->complex_coeffs() ? M_COMPLEX * 2 : M_COMPLEX;

        // The CrysFFT path works on the physical grid and returns the reduced basis;
        // the other paths work on the full grid
        const bool crysfft_path = this->use_crysfft() && this->space_group_ != nullptr;

        // Work array, kept per thread (called from OpenMP parallel regions)
        typename CpuSolverPseudoBase<T>::WorkArrays& work = this->work_arrays();
        std::vector<double>& k_q_in = work.k_1;
        if (!crysfft_path)
            k_q_in.resize(coeff_size);

        // Get Boltzmann factors for the correct ds_index
        const T* _exp_dw = this->exp_dw.at(ds_index).at(monomer_type).data();
        const double* _boltz_bond = this->pseudo->get_boltz_bond(monomer_type, ds_index);

        // Determine input/output pointers based on space group
        T* fft_in = q_in;
        T* fft_out = q_out;

        // Thread-local buffers for expand/reduce (called from OpenMP parallel regions);
        // the CrysFFT path reads the reduced basis directly
        std::vector<T>& q_full_in_local = work.q_full_in;
        std::vector<T>& q_full_out_local = work.q_full_out;

        if (this->space_group_ != nullptr && !crysfft_path)
        {
            q_full_in_local.resize(M_full);
            q_full_out_local.resize(M_full);

            // Expand reduced basis → full grid
            if constexpr (std::is_same_v<T, double>)
                this->space_group_->from_reduced_basis(q_in, q_full_in_local.data(), 1);

            fft_in = q_full_in_local.data();
            fft_out = q_full_out_local.data();
        }

        // ===== RK2: Single full step =====
        // Step 1: exp(-w·ds/2) * q_in
        if (!crysfft_path)
        {
            for (int i = 0; i < M_full; ++i)
                fft_out[i] = _exp_dw[i] * fft_in[i];
        }

        // Step 2-4: Forward FFT -> multiply by Boltzmann factor -> Backward FFT
        if constexpr (std::is_same_v<T, double>)
        {
            if (crysfft_path)
            {
                std::vector<double>& phys_in = work.phys_in;
                std::vector<double>& phys_out = work.phys_out;
                const int M_phys = this->get_crysfft_physical_size();
                const bool identity = this->use_crysfft_identity_map();
                if (static_cast<int>(phys_in.size()) != M_phys)
                {
                    phys_in.resize(M_phys);
                    phys_out.resize(M_phys);
                }

                this->fill_crysfft_from_reduced(q_in, phys_in.data());
                for (int i = 0; i < M_phys; ++i)
                {
                    const int ridx = identity ? i : this->crysfft_reduced_indices_[i];
                    phys_in[i] *= _exp_dw[ridx];
                }

                double coeff_full = this->get_effective_diffusion_coeff(monomer_type, ds_index, false);
                this->crysfft_set_contour_step(coeff_full);
                this->crysfft_diffusion(phys_in.data(), phys_out.data());

                for (int i = 0; i < M_phys; ++i)
                {
                    const int ridx = identity ? i : this->crysfft_reduced_indices_[i];
                    phys_out[i] *= _exp_dw[ridx];
                }

                q_full_out_local.resize(M_reduced);
                this->reduce_crysfft_to_reduced(phys_out.data(), q_full_out_local.data());
                fft_out = q_full_out_local.data();
            }
            else
            {
                this->transform_forward(fft_out, k_q_in.data());
                this->multiply_fourier_coeffs(k_q_in.data(), _boltz_bond, M_COMPLEX);
                this->transform_backward(k_q_in.data(), fft_out);
            }
        }
        else
        {
            this->transform_forward(fft_out, k_q_in.data());
            this->multiply_fourier_coeffs(k_q_in.data(), _boltz_bond, M_COMPLEX);
            this->transform_backward(k_q_in.data(), fft_out);
        }

        // Step 5: exp(-w·ds/2) * result
        if constexpr (std::is_same_v<T, double>)
        {
            if (!crysfft_path)
            {
                for (int i = 0; i < M_full; ++i)
                    fft_out[i] *= _exp_dw[i];
            }
        }
        else
        {
            for (int i = 0; i < M_full; ++i)
                fft_out[i] *= _exp_dw[i];
        }

        // Reduce full grid → reduced basis
        if (this->space_group_ != nullptr)
        {
            if constexpr (std::is_same_v<T, double>)
            {
                if (this->use_crysfft())
                    std::copy(fft_out, fft_out + M_reduced, q_out);
                else
                    this->space_group_->to_reduced_basis(fft_out, q_out, 1);
            }
        }

        // Apply the mask on the output. The computation box stores the mask in the
        // basis of the propagators (reduced with a space group, full otherwise), so it
        // is applied after the reduction; the mask is symmetric, so this equals masking
        // on the full grid.
        if (q_mask != nullptr)
        {
            for (int i = 0; i < M_reduced; ++i)
                q_out[i] *= q_mask[i];
        }
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

// Explicit template instantiation
#include "TemplateInstantiations.h"
INSTANTIATE_CLASS(CpuSolverPseudoRK2);
