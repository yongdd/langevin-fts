/**
 * @file CpuSolverPseudoETDRK4.cpp
 * @brief CPU ETDRK4 pseudo-spectral solver for continuous chain propagators.
 *
 * Implements the ETDRK4 (Exponential Time Differencing Runge-Kutta 4th order)
 * method for solving the modified diffusion equation. This provides an
 * alternative to RQM4 with L-stability properties.
 *
 * **ETDRK4 Stages:**
 *
 * Given dq/ds = L*q + N(q) where L = (b^2/6)*nabla^2 and N(q) = -w*q:
 *
 * 1. a = E2*q_hat + alpha*N_hat_n     (N_n = -w*q_in)
 * 2. b = E2*q_hat + alpha*N_hat_a     (N_a = -w*a)
 * 3. c = E2*a_hat + alpha*(2*N_hat_b - N_hat_n)  (N_b = -w*b)
 * 4. q_{n+1} = E*q_hat + f1*N_hat_n + f2*(N_hat_a + N_hat_b) + f3*N_hat_c
 *
 * **Template Instantiations:**
 *
 * - CpuSolverPseudoETDRK4<double>: Real fields with r2c FFT
 * - CpuSolverPseudoETDRK4<std::complex<double>>: Complex fields
 *
 * @see ETDRK4Coefficients for coefficient computation
 * @see CpuSolverPseudoRQM4 for RQM4 alternative
 */

#include <iostream>
#include <cmath>
#include <complex>

#include "CpuSolverPseudoETDRK4.h"

//------------------------------------------------------------------------------
// Constructor
//------------------------------------------------------------------------------
template <typename T>
CpuSolverPseudoETDRK4<T>::CpuSolverPseudoETDRK4(ComputationBox<T>* cb, Molecules *molecules)
{
    try
    {
        // Initialize shared components (FFT, Pseudo, etc.)
        this->init_shared(cb, molecules);

        const int M = cb->get_total_grid();

        // Create field vectors for each monomer type
        // ETDRK4 uses only global ds (ds_index=1)
        const int ds_index = 1;
        for (const auto& item : molecules->get_bond_lengths())
        {
            std::string monomer_type = item.first;
            this->exp_dw[ds_index][monomer_type].resize(M);
            this->exp_dw_half[ds_index][monomer_type].resize(M);
            this->w_field[monomer_type].resize(M);
        }

        // Initialize ETDRK4 coefficients
        this->etdrk4_coefficients_ = std::make_unique<ETDRK4Coefficients<T>>(
            molecules->get_bond_lengths(),
            cb->get_boundary_conditions(),
            cb->get_nx(),
            cb->get_dx(),
            molecules->get_ds(),
            cb->get_recip_metric()
        );

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
CpuSolverPseudoETDRK4<T>::~CpuSolverPseudoETDRK4()
{
    // Vectors are automatically cleaned up

    // Clean up shared components (FFT, Pseudo)
    this->cleanup_shared();
}

//------------------------------------------------------------------------------
// Get stress Boltzmann bond factor
//------------------------------------------------------------------------------
template <typename T>
const double* CpuSolverPseudoETDRK4<T>::get_stress_boltz_bond(
    std::string /*monomer_type*/, bool /*is_half_bond_length*/) const
{
    // ETDRK4/continuous chains don't use boltz_bond factor in stress computation
    return nullptr;
}

//------------------------------------------------------------------------------
// Update dw (store raw field and compute Boltzmann factors)
//------------------------------------------------------------------------------
template <typename T>
void CpuSolverPseudoETDRK4<T>::update_dw(std::map<std::string, const T*> w_input)
{
    const int M = this->cb->get_total_grid();
    const double ds = this->molecules->get_ds();
    const int ds_index = 1;  // ETDRK4 uses only global ds

    for (const auto& item : w_input)
    {
        if (!this->exp_dw[ds_index].contains(item.first))
            throw_with_line_number("monomer_type \"" + item.first + "\" is not in exp_dw.");
    }

    for (const auto& item : w_input)
    {
        const std::string& monomer_type = item.first;
        const T* w = item.second;
        std::vector<T>& exp_dw_vec = this->exp_dw[ds_index][monomer_type];
        std::vector<T>& exp_dw_half_vec = this->exp_dw_half[ds_index][monomer_type];
        std::vector<T>& w_field_vec = this->w_field[monomer_type];

        for (int i = 0; i < M; ++i)
        {
            w_field_vec[i] = w[i];  // Store raw w for ETDRK4
            exp_dw_vec[i] = std::exp(-w[i] * ds * 0.5);
            exp_dw_half_vec[i] = std::exp(-w[i] * ds * 0.25);
        }
    }
}

//------------------------------------------------------------------------------
// Advance propagator using ETDRK4
//------------------------------------------------------------------------------
template <typename T>
void CpuSolverPseudoETDRK4<T>::advance_propagator(
    T* q_in, T* q_out, std::string monomer_type, const double* q_mask, int /*ds_index*/)
{
    // Note: ETDRK4 currently uses global ds for all blocks.
    // Per-block ds support would require recomputing coefficients.
    try
    {
        const int M = this->cb->get_total_grid();
        const int M_COMPLEX = this->pseudo->get_total_complex_grid();

        // For periodic BC, coefficient array is actually complex (interleaved real/imag)
        int coeff_size = this->is_periodic_ ? M_COMPLEX * 2 : M_COMPLEX;

        // Get ETDRK4 coefficients
        const double* _E = this->etdrk4_coefficients_->get_E(monomer_type);
        const double* _E2 = this->etdrk4_coefficients_->get_E2(monomer_type);
        const double* _alpha = this->etdrk4_coefficients_->get_alpha(monomer_type);
        const double* _f1 = this->etdrk4_coefficients_->get_f1(monomer_type);
        const double* _f2 = this->etdrk4_coefficients_->get_f2(monomer_type);
        const double* _f3 = this->etdrk4_coefficients_->get_f3(monomer_type);

        // Get raw w field for nonlinear term
        const T* _w = this->w_field[monomer_type].data();

        // Temporary arrays
        std::vector<T> a(M), b(M), c(M);
        std::vector<T> N_n(M), N_a(M), N_b(M), N_c(M);
        std::vector<double> k_q(coeff_size);
        std::vector<double> k_N_n(coeff_size), k_N_a(coeff_size), k_N_b(coeff_size), k_N_c(coeff_size);
        std::vector<double> k_a(coeff_size), k_work(coeff_size);

        // Step 1: Compute N_n = -w*q_in (nonlinear term at initial point)
        for (int i = 0; i < M; ++i)
            N_n[i] = -_w[i] * q_in[i];

        // Step 2: FFT of q_in and N_n
        this->transform_forward(q_in, k_q.data());
        this->transform_forward(N_n.data(), k_N_n.data());

        // ETDRK4 stages
        if (this->is_periodic_)
        {
            // Complex coefficients for periodic BC
            std::complex<double>* k_q_c = reinterpret_cast<std::complex<double>*>(k_q.data());
            std::complex<double>* k_N_n_c = reinterpret_cast<std::complex<double>*>(k_N_n.data());
            std::complex<double>* k_N_a_c = reinterpret_cast<std::complex<double>*>(k_N_a.data());
            std::complex<double>* k_N_b_c = reinterpret_cast<std::complex<double>*>(k_N_b.data());
            std::complex<double>* k_N_c_c = reinterpret_cast<std::complex<double>*>(k_N_c.data());
            std::complex<double>* k_a_c = reinterpret_cast<std::complex<double>*>(k_a.data());
            std::complex<double>* k_work_c = reinterpret_cast<std::complex<double>*>(k_work.data());

            // Stage a: a_hat = E2*q_hat + alpha*N_hat_n
            for (int i = 0; i < M_COMPLEX; ++i)
                k_a_c[i] = _E2[i] * k_q_c[i] + _alpha[i] * k_N_n_c[i];

            // IFFT to get a, compute N_a = -w*a
            this->transform_backward(k_a.data(), a.data());
            for (int i = 0; i < M; ++i)
                N_a[i] = -_w[i] * a[i];
            this->transform_forward(N_a.data(), k_N_a.data());

            // Stage b: b_hat = E2*q_hat + alpha*N_hat_a
            for (int i = 0; i < M_COMPLEX; ++i)
                k_work_c[i] = _E2[i] * k_q_c[i] + _alpha[i] * k_N_a_c[i];

            // IFFT to get b, compute N_b = -w*b
            this->transform_backward(k_work.data(), b.data());
            for (int i = 0; i < M; ++i)
                N_b[i] = -_w[i] * b[i];
            this->transform_forward(N_b.data(), k_N_b.data());

            // Stage c: c_hat = E2*a_hat + alpha*(2*N_hat_b - N_hat_n)
            for (int i = 0; i < M_COMPLEX; ++i)
                k_work_c[i] = _E2[i] * k_a_c[i] + _alpha[i] * (2.0 * k_N_b_c[i] - k_N_n_c[i]);

            // IFFT to get c, compute N_c = -w*c
            this->transform_backward(k_work.data(), c.data());
            for (int i = 0; i < M; ++i)
                N_c[i] = -_w[i] * c[i];
            this->transform_forward(N_c.data(), k_N_c.data());

            // Final step: q_hat_{n+1} = E*q_hat + f1*N_hat_n + f2*(N_hat_a + N_hat_b) + f3*N_hat_c
            for (int i = 0; i < M_COMPLEX; ++i)
            {
                k_work_c[i] = _E[i] * k_q_c[i]
                            + _f1[i] * k_N_n_c[i]
                            + _f2[i] * (k_N_a_c[i] + k_N_b_c[i])
                            + _f3[i] * k_N_c_c[i];
            }

            // IFFT to get final result
            this->transform_backward(k_work.data(), q_out);
        }
        else
        {
            // Real coefficients for non-periodic BC (DCT/DST)

            // Stage a: a_hat = E2*q_hat + alpha*N_hat_n
            for (int i = 0; i < M_COMPLEX; ++i)
                k_a[i] = _E2[i] * k_q[i] + _alpha[i] * k_N_n[i];

            // IFFT to get a, compute N_a = -w*a
            this->transform_backward(k_a.data(), a.data());
            for (int i = 0; i < M; ++i)
                N_a[i] = -_w[i] * a[i];
            this->transform_forward(N_a.data(), k_N_a.data());

            // Stage b: b_hat = E2*q_hat + alpha*N_hat_a
            for (int i = 0; i < M_COMPLEX; ++i)
                k_work[i] = _E2[i] * k_q[i] + _alpha[i] * k_N_a[i];

            // IFFT to get b, compute N_b = -w*b
            this->transform_backward(k_work.data(), b.data());
            for (int i = 0; i < M; ++i)
                N_b[i] = -_w[i] * b[i];
            this->transform_forward(N_b.data(), k_N_b.data());

            // Stage c: c_hat = E2*a_hat + alpha*(2*N_hat_b - N_hat_n)
            for (int i = 0; i < M_COMPLEX; ++i)
                k_work[i] = _E2[i] * k_a[i] + _alpha[i] * (2.0 * k_N_b[i] - k_N_n[i]);

            // IFFT to get c, compute N_c = -w*c
            this->transform_backward(k_work.data(), c.data());
            for (int i = 0; i < M; ++i)
                N_c[i] = -_w[i] * c[i];
            this->transform_forward(N_c.data(), k_N_c.data());

            // Final step: q_hat_{n+1} = E*q_hat + f1*N_hat_n + f2*(N_hat_a + N_hat_b) + f3*N_hat_c
            for (int i = 0; i < M_COMPLEX; ++i)
            {
                k_work[i] = _E[i] * k_q[i]
                          + _f1[i] * k_N_n[i]
                          + _f2[i] * (k_N_a[i] + k_N_b[i])
                          + _f3[i] * k_N_c[i];
            }

            // IFFT to get final result
            this->transform_backward(k_work.data(), q_out);
        }

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
#include "TemplateInstantiations.h"
INSTANTIATE_CLASS(CpuSolverPseudoETDRK4);
