/**
 * @file CpuSolverPseudoContinuous.cpp
 * @brief CPU pseudo-spectral solver for continuous chain propagators.
 *
 * Implements the operator splitting pseudo-spectral method with
 * 4th-order Richardson extrapolation for continuous Gaussian chains.
 * Supports all boundary conditions (periodic, reflecting, absorbing).
 *
 * **Operator Splitting Scheme:**
 *
 * The diffusion equation dq/ds = (b^2/6)nabla^2 q - w(r)q is split as:
 * 1. Potential half-step: exp(-w·ds/2)
 * 2. Diffusion step: FFT -> multiply by exp(-b^2 k^2 ds/6) -> IFFT
 * 3. Potential half-step: exp(-w·ds/2)
 *
 * **Richardson Extrapolation:**
 *
 * Combines one full step (ds) and two half-steps (ds/2 each):
 *     q_out = (4·q_half - q_full) / 3
 *
 * This yields 4th-order accuracy in ds.
 *
 * **Template Instantiations:**
 *
 * - CpuSolverPseudoContinuous<double>: Real fields with r2c FFT
 * - CpuSolverPseudoContinuous<std::complex<double>>: Complex fields
 *
 * @see Pseudo for Boltzmann factor computation
 * @see CpuComputationContinuous for usage context
 */

#include <iostream>
#include <cmath>
#include <complex>

#include "CpuSolverPseudoContinuous.h"

//------------------------------------------------------------------------------
// Constructor
//------------------------------------------------------------------------------
template <typename T>
CpuSolverPseudoContinuous<T>::CpuSolverPseudoContinuous(ComputationBox<T>* cb, Molecules *molecules)
{
    try
    {
        // Initialize shared components (FFT, Pseudo, etc.)
        this->init_shared(cb, molecules);

        // Create exp_dw arrays for continuous chains
        const int M = cb->get_total_grid();
        for (const auto& item : molecules->get_bond_lengths())
        {
            std::string monomer_type = item.first;
            this->exp_dw[monomer_type] = new T[M];
            this->exp_dw_half[monomer_type] = new T[M];
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
CpuSolverPseudoContinuous<T>::~CpuSolverPseudoContinuous()
{
    // Clean up exp_dw arrays
    for (const auto& item : this->exp_dw)
        delete[] item.second;
    for (const auto& item : this->exp_dw_half)
        delete[] item.second;

    // Clean up shared components (FFT, Pseudo)
    this->cleanup_shared();
}

//------------------------------------------------------------------------------
// Get stress Boltzmann bond factor
//------------------------------------------------------------------------------
template <typename T>
const double* CpuSolverPseudoContinuous<T>::get_stress_boltz_bond(
    std::string /*monomer_type*/, bool /*is_half_bond_length*/) const
{
    // Continuous chains don't use boltz_bond factor in stress computation
    return nullptr;
}

//------------------------------------------------------------------------------
// Update dw (Boltzmann factors from field)
//------------------------------------------------------------------------------
template <typename T>
void CpuSolverPseudoContinuous<T>::update_dw(std::map<std::string, const T*> w_input)
{
    const int M = this->cb->get_total_grid();
    const double ds = this->molecules->get_ds();

    for (const auto& item : w_input)
    {
        if (this->exp_dw.find(item.first) == this->exp_dw.end())
            throw_with_line_number("monomer_type \"" + item.first + "\" is not in exp_dw.");
    }

    for (const auto& item : w_input)
    {
        std::string monomer_type = item.first;
        const T* w = item.second;

        for (int i = 0; i < M; ++i)
        {
            this->exp_dw[monomer_type][i] = std::exp(-w[i] * ds * 0.5);
            this->exp_dw_half[monomer_type][i] = std::exp(-w[i] * ds * 0.25);
        }
    }
}

//------------------------------------------------------------------------------
// Advance propagator
//------------------------------------------------------------------------------
template <typename T>
void CpuSolverPseudoContinuous<T>::advance_propagator(
    T* q_in, T* q_out, std::string monomer_type, const double* q_mask)
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

        T* _exp_dw = this->exp_dw[monomer_type];
        T* _exp_dw_half = this->exp_dw_half[monomer_type];

        const double* _boltz_bond = this->pseudo->get_boltz_bond(monomer_type);
        const double* _boltz_bond_half = this->pseudo->get_boltz_bond_half(monomer_type);

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

        // ===== Richardson extrapolation =====
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

// Explicit template instantiation
template class CpuSolverPseudoContinuous<double>;
template class CpuSolverPseudoContinuous<std::complex<double>>;
