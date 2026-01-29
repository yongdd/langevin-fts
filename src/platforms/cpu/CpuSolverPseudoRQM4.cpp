/**
 * @file CpuSolverPseudoRQM4.cpp
 * @brief CPU pseudo-spectral solver for continuous chain propagators using RQM4.
 *
 * Implements the operator splitting pseudo-spectral method with
 * RQM4 (Ranjan-Qin-Morse 4th-order using Richardson extrapolation)
 * for continuous Gaussian chains. Supports all boundary conditions
 * (periodic, reflecting, absorbing).
 *
 * **Operator Splitting Scheme:**
 *
 * The diffusion equation dq/ds = (b^2/6)nabla^2 q - w(r)q is split as:
 * 1. Potential half-step: exp(-w·ds/2)
 * 2. Diffusion step: FFT -> multiply by exp(-b^2 k^2 ds/6) -> IFFT
 * 3. Potential half-step: exp(-w·ds/2)
 *
 * **RQM4 (Richardson Extrapolation):**
 *
 * Combines one full step (ds) and two half-steps (ds/2 each):
 *     q_out = (4·q_half - q_full) / 3
 *
 * This yields 4th-order accuracy in ds.
 *
 * **Template Instantiations:**
 *
 * - CpuSolverPseudoRQM4<double>: Real fields with r2c FFT
 * - CpuSolverPseudoRQM4<std::complex<double>>: Complex fields
 *
 * @see Pseudo for Boltzmann factor computation
 * @see CpuComputationContinuous for usage context
 */

#include <iostream>
#include <cmath>
#include <complex>
#include <type_traits>

#include "CpuSolverPseudoRQM4.h"

//------------------------------------------------------------------------------
// Constructor
//------------------------------------------------------------------------------
template <typename T>
CpuSolverPseudoRQM4<T>::CpuSolverPseudoRQM4(ComputationBox<T>* cb, Molecules *molecules, FFTBackend backend)
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
                this->exp_dw_half[ds_idx][monomer_type].resize(M);
            }
        }

        // Finalize ds values to allocate boltz_bond arrays
        // (update_laplacian_operator will compute the actual values)
        this->pseudo->finalize_ds_values();
        this->update_laplacian_operator();
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

//------------------------------------------------------------------------------
// Destructor
//------------------------------------------------------------------------------
template <typename T>
CpuSolverPseudoRQM4<T>::~CpuSolverPseudoRQM4()
{
    // exp_dw and exp_dw_half vectors are automatically cleaned up

    // Clean up shared components (FFT, Pseudo)
    this->cleanup_shared();
}

//------------------------------------------------------------------------------
// Get stress Boltzmann bond factor
//------------------------------------------------------------------------------
template <typename T>
const double* CpuSolverPseudoRQM4<T>::get_stress_boltz_bond(
    std::string /*monomer_type*/, bool /*is_half_bond_length*/) const
{
    // Continuous chains don't use boltz_bond factor in stress computation
    return nullptr;
}

//------------------------------------------------------------------------------
// Update dw (Boltzmann factors from field)
//------------------------------------------------------------------------------
template <typename T>
void CpuSolverPseudoRQM4<T>::update_dw(std::map<std::string, const T*> w_input)
{
    const int M_full = this->cb->get_total_grid();
    const int M_reduced = (this->space_group_ != nullptr)
        ? this->space_group_->get_n_reduced_basis()
        : M_full;
    const bool use_reduced_basis = (this->space_group_ != nullptr);
    const bool use_crysfft = (this->use_crysfft() && this->space_group_ != nullptr);

    std::map<std::string, std::vector<T>> w_full_cache;
    if (use_reduced_basis && !use_crysfft)
    {
        if constexpr (std::is_same_v<T, double>)
        {
            for (const auto& item : w_input)
            {
                const std::string& monomer_type = item.first;
                w_full_cache[monomer_type].resize(M_full);
                this->space_group_->from_reduced_basis(item.second, w_full_cache[monomer_type].data(), 1);
            }
        }
        else
        {
            throw_with_line_number("Space group reduced basis is only supported for real fields.");
        }
    }

    // Get unique ds values from ContourLengthMapping
    const ContourLengthMapping& mapping = this->molecules->get_contour_length_mapping();
    int n_unique_ds = mapping.get_n_unique_ds();

    // Compute exp_dw for each ds_index and monomer type
    for (int ds_idx = 0; ds_idx < n_unique_ds; ++ds_idx)
    {
        double local_ds = mapping.get_ds_from_index(ds_idx);

        for (const auto& item : w_input)
        {
            const std::string& monomer_type = item.first;
            const T* w = (use_reduced_basis && !use_crysfft) ? w_full_cache[monomer_type].data() : item.second;

            if (!this->exp_dw[ds_idx].contains(monomer_type))
                throw_with_line_number("monomer_type \"" + monomer_type + "\" is not in exp_dw[" + std::to_string(ds_idx) + "].");

            std::vector<T>& exp_dw_vec = this->exp_dw[ds_idx][monomer_type];
            std::vector<T>& exp_dw_half_vec = this->exp_dw_half[ds_idx][monomer_type];

            const int M_use = use_crysfft ? M_reduced : M_full;
            exp_dw_vec.resize(M_use);
            exp_dw_half_vec.resize(M_use);

            for (int i = 0; i < M_use; ++i)
            {
                exp_dw_vec[i] = std::exp(-w[i] * local_ds * 0.5);
                exp_dw_half_vec[i] = std::exp(-w[i] * local_ds * 0.25);
            }
        }
    }
}

//------------------------------------------------------------------------------
// Advance propagator
//------------------------------------------------------------------------------
template <typename T>
void CpuSolverPseudoRQM4<T>::advance_propagator(
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
        int coeff_size = this->is_periodic_ ? M_COMPLEX * 2 : M_COMPLEX;

        // Temporary arrays
        std::vector<T> q_out1(M_full), q_out2(M_full);
        std::vector<double> k_q_in1(coeff_size);
        std::vector<double> k_q_in2(coeff_size);

        // Get Boltzmann factors for the correct ds_index
        const T* _exp_dw = this->exp_dw[ds_index][monomer_type].data();
        const T* _exp_dw_half = this->exp_dw_half[ds_index][monomer_type].data();

        const double* _boltz_bond = this->pseudo->get_boltz_bond(monomer_type, ds_index);
        const double* _boltz_bond_half = this->pseudo->get_boltz_bond_half(monomer_type, ds_index);

        // Determine input pointer (expand reduced basis if space_group is set)
        T* fft_in = q_in;

        // Thread-local buffer for expand (called from OpenMP parallel regions)
        thread_local std::vector<T> q_full_in_local;

        if (this->space_group_ != nullptr)
        {
            q_full_in_local.resize(M_full);
            // Expand reduced basis → full grid
            if constexpr (std::is_same_v<T, double>)
                this->space_group_->from_reduced_basis(q_in, q_full_in_local.data(), 1);
            fft_in = q_full_in_local.data();
        }

        // ===== Step 1: Full step =====
        if constexpr (std::is_same_v<T, double>)
        {
            if (this->use_crysfft() && this->space_group_ != nullptr)
            {
                thread_local std::vector<double> phys_in;
                thread_local std::vector<double> phys_out;
                const int M_phys = this->get_crysfft_physical_size();
                const bool identity = this->use_crysfft_identity_map();
                if (static_cast<int>(phys_in.size()) != M_phys)
                {
                    phys_in.resize(M_phys);
                    phys_out.resize(M_phys);
                }

                // exp(-w·ds/2) on Pmmm grid
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

                q_out1.resize(M_reduced);
                this->reduce_crysfft_to_reduced(phys_out.data(), q_out1.data());
            }
            else
            {
                for (int i = 0; i < M_full; ++i)
                    q_out1[i] = _exp_dw[i] * fft_in[i];

                this->transform_forward(q_out1.data(), k_q_in1.data());
                this->multiply_fourier_coeffs(k_q_in1.data(), _boltz_bond, M_COMPLEX);
                this->transform_backward(k_q_in1.data(), q_out1.data());

                for (int i = 0; i < M_full; ++i)
                    q_out1[i] *= _exp_dw[i];
            }
        }
        else
        {
            for (int i = 0; i < M_full; ++i)
                q_out1[i] = _exp_dw[i] * fft_in[i];

            this->transform_forward(q_out1.data(), k_q_in1.data());
            this->multiply_fourier_coeffs(k_q_in1.data(), _boltz_bond, M_COMPLEX);
            this->transform_backward(k_q_in1.data(), q_out1.data());

            for (int i = 0; i < M_full; ++i)
                q_out1[i] *= _exp_dw[i];
        }

        // ===== Step 2: Two half steps =====
        if constexpr (std::is_same_v<T, double>)
        {
            if (this->use_crysfft() && this->space_group_ != nullptr)
            {
                thread_local std::vector<double> phys_in;
                thread_local std::vector<double> phys_out;
                const int M_phys = this->get_crysfft_physical_size();
                const bool identity = this->use_crysfft_identity_map();
                if (static_cast<int>(phys_in.size()) != M_phys)
                {
                    phys_in.resize(M_phys);
                    phys_out.resize(M_phys);
                }

                double coeff_half = this->get_effective_diffusion_coeff(monomer_type, ds_index, true);
                this->crysfft_set_contour_step(coeff_half);

                // First half step
                this->fill_crysfft_from_reduced(q_in, phys_in.data());
                for (int i = 0; i < M_phys; ++i)
                {
                    const int ridx = identity ? i : this->crysfft_reduced_indices_[i];
                    phys_in[i] *= _exp_dw_half[ridx];
                }
                this->crysfft_diffusion(phys_in.data(), phys_out.data());
                for (int i = 0; i < M_phys; ++i)
                {
                    const int ridx = identity ? i : this->crysfft_reduced_indices_[i];
                    phys_out[i] *= _exp_dw[ridx];
                }

                // Second half step
                this->crysfft_diffusion(phys_out.data(), phys_in.data());
                for (int i = 0; i < M_phys; ++i)
                {
                    const int ridx = identity ? i : this->crysfft_reduced_indices_[i];
                    phys_in[i] *= _exp_dw_half[ridx];
                }

                q_out2.resize(M_reduced);
                this->reduce_crysfft_to_reduced(phys_in.data(), q_out2.data());
            }
            else
            {
                // First half step
                for (int i = 0; i < M_full; ++i)
                    q_out2[i] = _exp_dw_half[i] * fft_in[i];

                this->transform_forward(q_out2.data(), k_q_in2.data());
                this->multiply_fourier_coeffs(k_q_in2.data(), _boltz_bond_half, M_COMPLEX);
                this->transform_backward(k_q_in2.data(), q_out2.data());

                for (int i = 0; i < M_full; ++i)
                    q_out2[i] *= _exp_dw[i];

                // Second half step
                this->transform_forward(q_out2.data(), k_q_in2.data());
                this->multiply_fourier_coeffs(k_q_in2.data(), _boltz_bond_half, M_COMPLEX);
                this->transform_backward(k_q_in2.data(), q_out2.data());

                for (int i = 0; i < M_full; ++i)
                    q_out2[i] *= _exp_dw_half[i];
            }
        }
        else
        {
            // First half step
            for (int i = 0; i < M_full; ++i)
                q_out2[i] = _exp_dw_half[i] * fft_in[i];

            this->transform_forward(q_out2.data(), k_q_in2.data());
            this->multiply_fourier_coeffs(k_q_in2.data(), _boltz_bond_half, M_COMPLEX);
            this->transform_backward(k_q_in2.data(), q_out2.data());

            for (int i = 0; i < M_full; ++i)
                q_out2[i] *= _exp_dw[i];

            // Second half step
            this->transform_forward(q_out2.data(), k_q_in2.data());
            this->multiply_fourier_coeffs(k_q_in2.data(), _boltz_bond_half, M_COMPLEX);
            this->transform_backward(k_q_in2.data(), q_out2.data());

            for (int i = 0; i < M_full; ++i)
                q_out2[i] *= _exp_dw_half[i];
        }

        // ===== RQM4: Richardson extrapolation =====
        // Result goes into q_out2 (reuse buffer)
        const int M_out = (this->use_crysfft() && this->space_group_ != nullptr) ? M_reduced : M_full;
        for (int i = 0; i < M_out; ++i)
            q_out2[i] = (4.0 * q_out2[i] - q_out1[i]) / 3.0;

        // Apply mask if provided
        if (q_mask != nullptr)
        {
            for (int i = 0; i < M_out; ++i)
                q_out2[i] *= q_mask[i];
        }

        // Reduce output to reduced basis if space_group is set
        if (this->space_group_ != nullptr)
        {
            if constexpr (std::is_same_v<T, double>)
            {
                if (this->use_crysfft())
                    std::copy(q_out2.begin(), q_out2.begin() + M_out, q_out);
                else
                    this->space_group_->to_reduced_basis(q_out2.data(), q_out, 1);
            }
        }
        else
        {
            // No space group - copy result to output
            std::copy(q_out2.begin(), q_out2.end(), q_out);
        }
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

//------------------------------------------------------------------------------
// Update Laplacian operator
//------------------------------------------------------------------------------
template <typename T>
void CpuSolverPseudoRQM4<T>::update_laplacian_operator()
{
    try
    {
        // Call base class implementation (updates Fourier basis and recomputes Boltzmann factors)
        // Note: local_ds values are registered once in constructor via add_ds_value()
        // pseudo->update() recomputes boltz_bond for all registered ds values
        CpuSolverPseudoBase<T>::update_laplacian_operator();
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

// Explicit template instantiation
#include "TemplateInstantiations.h"
INSTANTIATE_CLASS(CpuSolverPseudoRQM4);
