/**
 * @file CpuSolverPseudoDiscrete.cpp
 * @brief CPU pseudo-spectral solver for discrete chain propagators.
 *
 * Implements the pseudo-spectral method for discrete chain models using the
 * Chapman-Kolmogorov integral equation. Supports all boundary conditions
 * (periodic, reflecting, absorbing).
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
 * **Template Instantiations:**
 *
 * - CpuSolverPseudoDiscrete<double>: Real fields
 * - CpuSolverPseudoDiscrete<std::complex<double>>: Complex fields
 *
 * @see CpuComputationDiscrete for usage context
 */

#include <iostream>
#include <cmath>
#include <complex>

#include "CpuSolverPseudoDiscrete.h"

//------------------------------------------------------------------------------
// Constructor
//------------------------------------------------------------------------------
template <typename T>
CpuSolverPseudoDiscrete<T>::CpuSolverPseudoDiscrete(ComputationBox<T>* cb, Molecules *molecules)
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
        // Also register local_ds values with Pseudo for boltz_bond computation
        for (int ds_idx = 1; ds_idx <= n_unique_ds; ++ds_idx)
        {
            double local_ds = mapping.get_ds_from_index(ds_idx);
            this->pseudo->add_ds_value(ds_idx, local_ds);

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
        throw_without_line_number(exc.what());
    }
}

//------------------------------------------------------------------------------
// Destructor
//------------------------------------------------------------------------------
template <typename T>
CpuSolverPseudoDiscrete<T>::~CpuSolverPseudoDiscrete()
{
    // exp_dw vectors are automatically cleaned up

    // Clean up shared components (FFT, Pseudo)
    this->cleanup_shared();
}

//------------------------------------------------------------------------------
// Get stress Boltzmann bond factor
//------------------------------------------------------------------------------
template <typename T>
const double* CpuSolverPseudoDiscrete<T>::get_stress_boltz_bond(
    std::string monomer_type, bool is_half_bond_length) const
{
    // Discrete chains include boltz_bond factor in stress computation
    // Discrete chains always use ds_index=1 (global ds)
    if (is_half_bond_length)
        return this->pseudo->get_boltz_bond_half(monomer_type, 1);
    else
        return this->pseudo->get_boltz_bond(monomer_type, 1);
}

//------------------------------------------------------------------------------
// Update dw (Boltzmann factors from field)
//------------------------------------------------------------------------------
template <typename T>
void CpuSolverPseudoDiscrete<T>::update_dw(std::map<std::string, const T*> w_input)
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

            for (int i = 0; i < M; ++i)
                exp_dw_vec[i] = std::exp(-w[i] * local_ds);
        }
    }
}

//------------------------------------------------------------------------------
// Advance propagator
//------------------------------------------------------------------------------
template <typename T>
void CpuSolverPseudoDiscrete<T>::advance_propagator(
    T* q_in, T* q_out, std::string monomer_type, const double* q_mask, int ds_index)
{
    try
    {
        const int M = this->cb->get_total_grid();
        const int M_COMPLEX = this->pseudo->get_total_complex_grid();

        // Use local buffer for thread safety (called from OpenMP parallel regions)
        int coeff_size = this->is_periodic_ ? M_COMPLEX * 2 : M_COMPLEX;
        std::vector<double> k_q_in(coeff_size);

        // Get Boltzmann factors for the correct ds_index
        const T* _exp_dw = this->exp_dw[ds_index][monomer_type].data();
        const double* _boltz_bond = this->pseudo->get_boltz_bond(monomer_type, ds_index);

        // Forward transform -> multiply by bond function ĝ(k) -> Backward transform
        this->transform_forward(q_in, k_q_in.data());
        this->multiply_fourier_coeffs(k_q_in.data(), _boltz_bond, M_COMPLEX);
        this->transform_backward(k_q_in.data(), q_out);

        // Evaluate exp(-w*ds) in real space and apply mask if provided
        // Combined into single loop to reduce memory traffic
        if (q_mask != nullptr)
        {
            for (int i = 0; i < M; ++i)
                q_out[i] *= _exp_dw[i] * q_mask[i];
        }
        else
        {
            for (int i = 0; i < M; ++i)
                q_out[i] *= _exp_dw[i];
        }
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

//------------------------------------------------------------------------------
// Advance propagator half bond step
//------------------------------------------------------------------------------
template <typename T>
void CpuSolverPseudoDiscrete<T>::advance_propagator_half_bond_step(
    T* q_in, T* q_out, std::string monomer_type)
{
    try
    {
        const int M_COMPLEX = this->pseudo->get_total_complex_grid();

        // Use local buffer for thread safety (called from OpenMP parallel regions)
        int coeff_size = this->is_periodic_ ? M_COMPLEX * 2 : M_COMPLEX;
        std::vector<double> k_q_in(coeff_size);

        // Discrete chains always use ds_index=1 (global ds)
        const double* _boltz_bond_half = this->pseudo->get_boltz_bond_half(monomer_type, 1);

        // Forward transform -> multiply by half-bond function ĝ^(1/2)(k) -> Backward transform
        this->transform_forward(q_in, k_q_in.data());
        this->multiply_fourier_coeffs(k_q_in.data(), _boltz_bond_half, M_COMPLEX);
        this->transform_backward(k_q_in.data(), q_out);
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
void CpuSolverPseudoDiscrete<T>::update_laplacian_operator()
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
INSTANTIATE_CLASS(CpuSolverPseudoDiscrete);
