/**
 * @file CpuSolverPseudoDiscrete.cpp
 * @brief CPU pseudo-spectral solver for discrete chain propagators.
 *
 * Implements the pseudo-spectral method for discrete (freely-jointed)
 * chain models. Supports all boundary conditions (periodic, reflecting,
 * absorbing). Unlike continuous chains, discrete chains use single
 * diffusion/potential steps per segment with half-bond step support.
 *
 * **Propagator Equation:**
 *
 * For discrete chains, the propagator satisfies:
 *     q(r,n+1) = exp(-w(r)*ds) * FFT^-1[exp(-b^2 k^2 ds/6) * FFT[q(r,n)]]
 *
 * **Half-Bond Steps:**
 *
 * At junction points, half-bond steps are used:
 *     exp(-b^2 k^2 ds/12)
 *
 * This ensures proper handling of chain junctions where multiple
 * propagators meet.
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

        // Create exp_dw vectors for discrete chains (no exp_dw_half needed)
        const int M = cb->get_total_grid();
        for (const auto& item : molecules->get_bond_lengths())
        {
            std::string monomer_type = item.first;
            this->exp_dw[monomer_type].resize(M);
        }

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
    if (is_half_bond_length)
        return this->pseudo->get_boltz_bond_half(monomer_type);
    else
        return this->pseudo->get_boltz_bond(monomer_type);
}

//------------------------------------------------------------------------------
// Update dw (Boltzmann factors from field)
//------------------------------------------------------------------------------
template <typename T>
void CpuSolverPseudoDiscrete<T>::update_dw(std::map<std::string, const T*> w_input)
{
    const int M = this->cb->get_total_grid();
    const double ds = this->molecules->get_ds();

    for (const auto& item : w_input)
    {
        if (!this->exp_dw.contains(item.first))
            throw_with_line_number("monomer_type \"" + item.first + "\" is not in exp_dw.");
    }

    for (const auto& item : w_input)
    {
        const std::string& monomer_type = item.first;
        const T* w = item.second;
        std::vector<T>& exp_dw_vec = this->exp_dw[monomer_type];

        for (int i = 0; i < M; ++i)
            exp_dw_vec[i] = std::exp(-w[i] * ds);
    }
}

//------------------------------------------------------------------------------
// Advance propagator
//------------------------------------------------------------------------------
template <typename T>
void CpuSolverPseudoDiscrete<T>::advance_propagator(
    T* q_in, T* q_out, std::string monomer_type, const double* q_mask)
{
    try
    {
        const int M = this->cb->get_total_grid();
        const int M_COMPLEX = this->pseudo->get_total_complex_grid();

        // Use local buffer for thread safety (called from OpenMP parallel regions)
        int coeff_size = this->is_periodic_ ? M_COMPLEX * 2 : M_COMPLEX;
        std::vector<double> k_q_in(coeff_size);

        const T* _exp_dw = this->exp_dw[monomer_type].data();
        const double* _boltz_bond = this->pseudo->get_boltz_bond(monomer_type);

        // Forward transform
        this->transform_forward(q_in, k_q_in.data());

        // Multiply exp(-k^2 ds/6) in Fourier space
        if (this->is_periodic_)
        {
            std::complex<double>* k_q_complex = reinterpret_cast<std::complex<double>*>(k_q_in.data());
            for (int i = 0; i < M_COMPLEX; ++i)
                k_q_complex[i] *= _boltz_bond[i];
        }
        else
        {
            for (int i = 0; i < M_COMPLEX; ++i)
                k_q_in[i] *= _boltz_bond[i];
        }

        // Backward transform
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

        const double* _boltz_bond_half = this->pseudo->get_boltz_bond_half(monomer_type);

        // Forward transform
        this->transform_forward(q_in, k_q_in.data());

        // Multiply exp(-k^2 ds/12) in Fourier space
        if (this->is_periodic_)
        {
            std::complex<double>* k_q_complex = reinterpret_cast<std::complex<double>*>(k_q_in.data());
            for (int i = 0; i < M_COMPLEX; ++i)
                k_q_complex[i] *= _boltz_bond_half[i];
        }
        else
        {
            for (int i = 0; i < M_COMPLEX; ++i)
                k_q_in[i] *= _boltz_bond_half[i];
        }

        // Backward transform
        this->transform_backward(k_q_in.data(), q_out);
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

// Explicit template instantiation
#include "TemplateInstantiations.h"
INSTANTIATE_CLASS(CpuSolverPseudoDiscrete);
