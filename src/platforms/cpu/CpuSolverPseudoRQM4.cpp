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

#include "CpuSolverPseudoRQM4.h"

//------------------------------------------------------------------------------
// Constructor
//------------------------------------------------------------------------------
template <typename T>
CpuSolverPseudoRQM4<T>::CpuSolverPseudoRQM4(ComputationBox<T>* cb, Molecules *molecules)
{
    try
    {
        // Initialize shared components (FFT, Pseudo, etc.)
        this->init_shared(cb, molecules);

        const int M = cb->get_total_grid();

        // Ensure ContourLengthMapping is finalized before using it
        molecules->finalize_contour_length_mapping();

        // Get unique ds values from ContourLengthMapping
        const ContourLengthMapping& mapping = molecules->get_contour_length_mapping();
        int n_unique_ds = mapping.get_n_unique_ds();

        // Create exp_dw vectors for each ds_index and monomer type
        for (int ds_idx = 1; ds_idx <= n_unique_ds; ++ds_idx)
        {
            for (const auto& item : molecules->get_bond_lengths())
            {
                std::string monomer_type = item.first;
                this->exp_dw[ds_idx][monomer_type].resize(M);
                this->exp_dw_half[ds_idx][monomer_type].resize(M);
            }
        }

        // update_laplacian_operator() handles registration of local_ds values
        // and calls finalize_ds_values() to compute boltz_bond with correct local_ds
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
    const int M = this->cb->get_total_grid();

    // Get unique ds values from ContourLengthMapping
    const ContourLengthMapping& mapping = this->molecules->get_contour_length_mapping();
    int n_unique_ds = mapping.get_n_unique_ds();

    // Compute exp_dw for each ds_index and monomer type
    for (int ds_idx = 1; ds_idx <= n_unique_ds; ++ds_idx)
    {
        double local_ds = mapping.get_ds_from_index(ds_idx);

        for (const auto& item : w_input)
        {
            const std::string& monomer_type = item.first;
            const T* w = item.second;

            if (!this->exp_dw[ds_idx].contains(monomer_type))
                throw_with_line_number("monomer_type \"" + monomer_type + "\" is not in exp_dw[" + std::to_string(ds_idx) + "].");

            std::vector<T>& exp_dw_vec = this->exp_dw[ds_idx][monomer_type];
            std::vector<T>& exp_dw_half_vec = this->exp_dw_half[ds_idx][monomer_type];

            for (int i = 0; i < M; ++i)
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
        const int M = this->cb->get_total_grid();
        const int M_COMPLEX = this->pseudo->get_total_complex_grid();

        // For periodic BC, coefficient array is actually complex (interleaved real/imag)
        // so allocate extra space
        int coeff_size = this->is_periodic_ ? M_COMPLEX * 2 : M_COMPLEX;

        // Temporary arrays
        std::vector<T> q_out1(M), q_out2(M);
        std::vector<double> k_q_in1(coeff_size);
        std::vector<double> k_q_in2(coeff_size);

        // Get Boltzmann factors for the correct ds_index
        const T* _exp_dw = this->exp_dw[ds_index][monomer_type].data();
        const T* _exp_dw_half = this->exp_dw_half[ds_index][monomer_type].data();

        const double* _boltz_bond = this->pseudo->get_boltz_bond(monomer_type, ds_index);
        const double* _boltz_bond_half = this->pseudo->get_boltz_bond_half(monomer_type, ds_index);

        // ===== Step 1: Full step =====
        for (int i = 0; i < M; ++i)
            q_out1[i] = _exp_dw[i] * q_in[i];

        this->transform_forward(q_out1.data(), k_q_in1.data());

        // Multiply by Boltzmann factor in Fourier space
        if (this->is_periodic_)
        {
            std::complex<double>* k_q_complex = reinterpret_cast<std::complex<double>*>(k_q_in1.data());
            for (int i = 0; i < M_COMPLEX; ++i)
                k_q_complex[i] *= _boltz_bond[i];
        }
        else
        {
            for (int i = 0; i < M_COMPLEX; ++i)
                k_q_in1[i] *= _boltz_bond[i];
        }

        this->transform_backward(k_q_in1.data(), q_out1.data());

        for (int i = 0; i < M; ++i)
            q_out1[i] *= _exp_dw[i];

        // ===== Step 2: Two half steps =====
        // First half step
        for (int i = 0; i < M; ++i)
            q_out2[i] = _exp_dw_half[i] * q_in[i];

        this->transform_forward(q_out2.data(), k_q_in2.data());

        if (this->is_periodic_)
        {
            std::complex<double>* k_q_complex = reinterpret_cast<std::complex<double>*>(k_q_in2.data());
            for (int i = 0; i < M_COMPLEX; ++i)
                k_q_complex[i] *= _boltz_bond_half[i];
        }
        else
        {
            for (int i = 0; i < M_COMPLEX; ++i)
                k_q_in2[i] *= _boltz_bond_half[i];
        }

        this->transform_backward(k_q_in2.data(), q_out2.data());

        for (int i = 0; i < M; ++i)
            q_out2[i] *= _exp_dw[i];

        // Second half step
        this->transform_forward(q_out2.data(), k_q_in2.data());

        if (this->is_periodic_)
        {
            std::complex<double>* k_q_complex = reinterpret_cast<std::complex<double>*>(k_q_in2.data());
            for (int i = 0; i < M_COMPLEX; ++i)
                k_q_complex[i] *= _boltz_bond_half[i];
        }
        else
        {
            for (int i = 0; i < M_COMPLEX; ++i)
                k_q_in2[i] *= _boltz_bond_half[i];
        }

        this->transform_backward(k_q_in2.data(), q_out2.data());

        for (int i = 0; i < M; ++i)
            q_out2[i] *= _exp_dw_half[i];

        // ===== RQM4: Richardson extrapolation =====
        for (int i = 0; i < M; ++i)
            q_out[i] = (4.0 * q_out2[i] - q_out1[i]) / 3.0;

        // Apply mask if provided
        if (q_mask != nullptr)
        {
            for (int i = 0; i < M; ++i)
                q_out[i] *= q_mask[i];
        }
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

//------------------------------------------------------------------------------
// Update Laplacian operator and re-register local ds values
//------------------------------------------------------------------------------
template <typename T>
void CpuSolverPseudoRQM4<T>::update_laplacian_operator()
{
    try
    {
        // Call base class implementation (updates Pseudo with global_ds)
        CpuSolverPseudoBase<T>::update_laplacian_operator();

        // Re-register local_ds values for each block
        // (base class's pseudo->update() resets ds_values[1] to global_ds)
        const ContourLengthMapping& mapping = this->molecules->get_contour_length_mapping();
        int n_unique_ds = mapping.get_n_unique_ds();

        for (int ds_idx = 1; ds_idx <= n_unique_ds; ++ds_idx)
        {
            double local_ds = mapping.get_ds_from_index(ds_idx);
            this->pseudo->add_ds_value(ds_idx, local_ds);
        }

        // Finalize Pseudo to compute boltz_bond with correct local_ds
        this->pseudo->finalize_ds_values();
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

// Explicit template instantiation
#include "TemplateInstantiations.h"
INSTANTIATE_CLASS(CpuSolverPseudoRQM4);
