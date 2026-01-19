/**
 * @file CpuSolverPseudoETDRK4.cpp
 * @brief CPU ETDRK4 pseudo-spectral solver for continuous chain propagators.
 *
 * Implements the ETDRK4 (Exponential Time Differencing Runge-Kutta 4th order)
 * method for solving the modified diffusion equation using the Krogstad scheme
 * (Song et al. 2018).
 *
 * **Krogstad ETDRK4 Stages (Song et al. 2018, Eq. 7a-7d):**
 *
 * Given dq/ds = L*q + N(q) where L = (b^2/6)*nabla^2 and N(q) = -w*q:
 *
 * 1. a_hat = E2*q_hat + alpha*N_hat_n                       (7a)
 * 2. b_hat = a_hat + phi2_half*(N_hat_a - N_hat_n)          (7b)
 * 3. c_hat = E*q_hat + phi1*N_hat_n + 2*phi2*(N_hat_b - N_hat_n)  (7c)
 * 4. q_{n+1}_hat = c_hat + (4*phi3 - phi2)*(N_hat_n + N_hat_c)
 *                  + 2*phi2*N_hat_a - 4*phi3*(N_hat_a + N_hat_b)  (7d)
 *
 * @see ETDRK4Coefficients for coefficient computation
 * @see CpuSolverPseudoRQM4 for RQM4 alternative
 */

#include <iostream>
#include <cmath>
#include <complex>
#include <algorithm>

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
            molecules->get_global_ds(),
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
    const double ds = this->molecules->get_global_ds();
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

        // Get ETDRK4 coefficients (Krogstad scheme)
        const double* _E = this->etdrk4_coefficients_->get_E(monomer_type);
        const double* _E2 = this->etdrk4_coefficients_->get_E2(monomer_type);
        const double* _alpha = this->etdrk4_coefficients_->get_alpha(monomer_type);
        const double* _phi2_half = this->etdrk4_coefficients_->get_phi2_half(monomer_type);
        const double* _phi1 = this->etdrk4_coefficients_->get_phi1(monomer_type);
        const double* _phi2 = this->etdrk4_coefficients_->get_phi2(monomer_type);
        const double* _phi3 = this->etdrk4_coefficients_->get_phi3(monomer_type);

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

        // Krogstad ETDRK4 stages (Song et al. 2018)
        // Uses base class helpers that handle periodic vs non-periodic automatically

        // Stage a (Eq. 7a): a_hat = E2*q_hat + alpha*N_hat_n
        this->fourier_linear_combination_2(k_a.data(), _E2, k_q.data(), _alpha, k_N_n.data(), M_COMPLEX);

        // IFFT to get a, compute N_a = -w*a
        this->transform_backward(k_a.data(), a.data());
        for (int i = 0; i < M; ++i)
            N_a[i] = -_w[i] * a[i];
        this->transform_forward(N_a.data(), k_N_a.data());

        // Stage b (Eq. 7b): b_hat = a_hat + phi2_half*(N_hat_a - N_hat_n)
        std::copy(k_a.begin(), k_a.end(), k_work.begin());
        this->fourier_add_scaled_diff(k_work.data(), _phi2_half, k_N_a.data(), k_N_n.data(), M_COMPLEX);

        // IFFT to get b, compute N_b = -w*b
        this->transform_backward(k_work.data(), b.data());
        for (int i = 0; i < M; ++i)
            N_b[i] = -_w[i] * b[i];
        this->transform_forward(N_b.data(), k_N_b.data());

        // Stage c (Eq. 7c): c_hat = E*q_hat + phi1*N_hat_n + 2*phi2*(N_hat_b - N_hat_n)
        this->fourier_linear_combination_2(k_work.data(), _E, k_q.data(), _phi1, k_N_n.data(), M_COMPLEX);
        // Add 2*phi2*(N_hat_b - N_hat_n) using scaled diff with factor 2
        if (this->is_periodic_)
        {
            std::complex<double>* k_work_c = reinterpret_cast<std::complex<double>*>(k_work.data());
            const std::complex<double>* k_N_b_c = reinterpret_cast<const std::complex<double>*>(k_N_b.data());
            const std::complex<double>* k_N_n_c = reinterpret_cast<const std::complex<double>*>(k_N_n.data());
            for (int i = 0; i < M_COMPLEX; ++i)
                k_work_c[i] += 2.0 * _phi2[i] * (k_N_b_c[i] - k_N_n_c[i]);
        }
        else
        {
            for (int i = 0; i < M_COMPLEX; ++i)
                k_work[i] += 2.0 * _phi2[i] * (k_N_b[i] - k_N_n[i]);
        }

        // IFFT to get c, compute N_c = -w*c
        this->transform_backward(k_work.data(), c.data());
        for (int i = 0; i < M; ++i)
            N_c[i] = -_w[i] * c[i];
        this->transform_forward(N_c.data(), k_N_c.data());

        // Final step (Eq. 7d): q_{n+1}_hat = c_hat + (4*phi3 - phi2)*(N_hat_n + N_hat_c)
        //                                   + 2*phi2*N_hat_a - 4*phi3*(N_hat_a + N_hat_b)
        if (this->is_periodic_)
        {
            std::complex<double>* k_work_c = reinterpret_cast<std::complex<double>*>(k_work.data());
            const std::complex<double>* k_N_n_c = reinterpret_cast<const std::complex<double>*>(k_N_n.data());
            const std::complex<double>* k_N_a_c = reinterpret_cast<const std::complex<double>*>(k_N_a.data());
            const std::complex<double>* k_N_b_c = reinterpret_cast<const std::complex<double>*>(k_N_b.data());
            const std::complex<double>* k_N_c_c = reinterpret_cast<const std::complex<double>*>(k_N_c.data());
            for (int i = 0; i < M_COMPLEX; ++i)
            {
                double coeff_nc = 4.0 * _phi3[i] - _phi2[i];
                k_work_c[i] += coeff_nc * (k_N_n_c[i] + k_N_c_c[i])
                             + 2.0 * _phi2[i] * k_N_a_c[i]
                             - 4.0 * _phi3[i] * (k_N_a_c[i] + k_N_b_c[i]);
            }
        }
        else
        {
            for (int i = 0; i < M_COMPLEX; ++i)
            {
                double coeff_nc = 4.0 * _phi3[i] - _phi2[i];
                k_work[i] += coeff_nc * (k_N_n[i] + k_N_c[i])
                           + 2.0 * _phi2[i] * k_N_a[i]
                           - 4.0 * _phi3[i] * (k_N_a[i] + k_N_b[i]);
            }
        }

        // IFFT to get final result
        this->transform_backward(k_work.data(), q_out);

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
