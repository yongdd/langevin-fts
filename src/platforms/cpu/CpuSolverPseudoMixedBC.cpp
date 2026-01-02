/**
 * @file CpuSolverPseudoMixedBC.cpp
 * @brief Implementation of pseudo-spectral solver with mixed BCs.
 *
 * Supports reflecting (DCT) and absorbing (DST) boundary conditions
 * in addition to periodic (FFT).
 *
 * @see CpuSolverPseudoMixedBC.h for class documentation
 */

#include <iostream>
#include <cmath>
#include <numbers>
#include <complex>

#include "CpuSolverPseudoMixedBC.h"
#include "MklFFT.h"

//------------------------------------------------------------------------------
// Constructor
//------------------------------------------------------------------------------
template <typename T>
CpuSolverPseudoMixedBC<T>::CpuSolverPseudoMixedBC(ComputationBox<T>* cb, Molecules* molecules)
    : fft_periodic(nullptr), fft_mixed_1d(nullptr), fft_mixed_2d(nullptr), fft_mixed_3d(nullptr)
{
    try
    {
        this->cb = cb;
        this->molecules = molecules;
        this->chain_model = molecules->get_model_name();

        // Check if all BCs are periodic
        auto bc_vec = cb->get_boundary_conditions();
        is_periodic_ = true;
        for (const auto& b : bc_vec)
        {
            if (b != BoundaryCondition::PERIODIC)
            {
                is_periodic_ = false;
                break;
            }
        }

        // Create appropriate FFT object
        int dim = cb->get_dim();
        if (is_periodic_)
        {
            // Use standard periodic FFT
            if (dim == 3)
                fft_periodic = new MklFFT<T, 3>({cb->get_nx(0), cb->get_nx(1), cb->get_nx(2)});
            else if (dim == 2)
                fft_periodic = new MklFFT<T, 2>({cb->get_nx(0), cb->get_nx(1)});
            else if (dim == 1)
                fft_periodic = new MklFFT<T, 1>({cb->get_nx(0)});
        }
        else
        {
            // Use mixed BC FFT (DCT/DST)
            // bc_vec has 2 entries per dimension: [x_low, x_high, y_low, y_high, z_low, z_high]
            // We take one BC per dimension (both boundaries must match per ComputationBox validation)
            if (dim == 3)
            {
                std::array<int, 3> nx_arr = {cb->get_nx(0), cb->get_nx(1), cb->get_nx(2)};
                std::array<BoundaryCondition, 3> bc_arr = {bc_vec[0], bc_vec[2], bc_vec[4]};
                fft_mixed_3d = new MklFFTMixedBC<T, 3>(nx_arr, bc_arr);
            }
            else if (dim == 2)
            {
                std::array<int, 2> nx_arr = {cb->get_nx(0), cb->get_nx(1)};
                std::array<BoundaryCondition, 2> bc_arr = {bc_vec[0], bc_vec[2]};
                fft_mixed_2d = new MklFFTMixedBC<T, 2>(nx_arr, bc_arr);
            }
            else if (dim == 1)
            {
                std::array<int, 1> nx_arr = {cb->get_nx(0)};
                std::array<BoundaryCondition, 1> bc_arr = {bc_vec[0]};
                fft_mixed_1d = new MklFFTMixedBC<T, 1>(nx_arr, bc_arr);
            }
        }

        // Create Pseudo object
        // Extract one BC per dimension from the 2-per-dimension vector
        std::vector<BoundaryCondition> bc_per_dim;
        for (int d = 0; d < dim; ++d)
            bc_per_dim.push_back(bc_vec[2 * d]);  // Take low boundary (high must match)

        pseudo = new PseudoMixedBC<T>(
            molecules->get_bond_lengths(),
            bc_per_dim,
            cb->get_nx(), cb->get_dx(), molecules->get_ds());

        // Create exp_dw arrays
        const int M = cb->get_total_grid();
        for (const auto& item : molecules->get_bond_lengths())
        {
            std::string monomer_type = item.first;
            this->exp_dw[monomer_type] = new T[M];
            this->exp_dw_half[monomer_type] = new T[M];
        }

        update_laplacian_operator();
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
CpuSolverPseudoMixedBC<T>::~CpuSolverPseudoMixedBC()
{
    delete fft_periodic;
    delete fft_mixed_1d;
    delete fft_mixed_2d;
    delete fft_mixed_3d;
    delete pseudo;

    for (const auto& item : this->exp_dw)
        delete[] item.second;
    for (const auto& item : this->exp_dw_half)
        delete[] item.second;
}

//------------------------------------------------------------------------------
// Transform forward (dispatch to appropriate FFT)
//------------------------------------------------------------------------------
template <typename T>
void CpuSolverPseudoMixedBC<T>::transform_forward(T* rdata, double* cdata)
{
    if (is_periodic_)
    {
        // For periodic, we need to handle complex output
        const int M_COMPLEX = pseudo->get_total_complex_grid();
        std::complex<double>* cdata_complex = reinterpret_cast<std::complex<double>*>(cdata);
        fft_periodic->forward(rdata, cdata_complex);
    }
    else
    {
        int dim = cb->get_dim();
        if (dim == 3)
            fft_mixed_3d->forward(rdata, cdata);
        else if (dim == 2)
            fft_mixed_2d->forward(rdata, cdata);
        else if (dim == 1)
            fft_mixed_1d->forward(rdata, cdata);
    }
}

//------------------------------------------------------------------------------
// Transform backward (dispatch to appropriate FFT)
//------------------------------------------------------------------------------
template <typename T>
void CpuSolverPseudoMixedBC<T>::transform_backward(double* cdata, T* rdata)
{
    if (is_periodic_)
    {
        std::complex<double>* cdata_complex = reinterpret_cast<std::complex<double>*>(cdata);
        fft_periodic->backward(cdata_complex, rdata);
    }
    else
    {
        int dim = cb->get_dim();
        if (dim == 3)
            fft_mixed_3d->backward(cdata, rdata);
        else if (dim == 2)
            fft_mixed_2d->backward(cdata, rdata);
        else if (dim == 1)
            fft_mixed_1d->backward(cdata, rdata);
    }
}

//------------------------------------------------------------------------------
// Update Laplacian operator
//------------------------------------------------------------------------------
template <typename T>
void CpuSolverPseudoMixedBC<T>::update_laplacian_operator()
{
    try
    {
        // Extract one BC per dimension from the 2-per-dimension vector
        auto bc_vec = this->cb->get_boundary_conditions();
        int dim = this->cb->get_dim();
        std::vector<BoundaryCondition> bc_per_dim;
        for (int d = 0; d < dim; ++d)
            bc_per_dim.push_back(bc_vec[2 * d]);

        pseudo->update(
            bc_per_dim,
            this->molecules->get_bond_lengths(),
            this->cb->get_dx(), this->molecules->get_ds());
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

//------------------------------------------------------------------------------
// Update dw (Boltzmann factors from field)
//------------------------------------------------------------------------------
template <typename T>
void CpuSolverPseudoMixedBC<T>::update_dw(std::map<std::string, const T*> w_input)
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
void CpuSolverPseudoMixedBC<T>::advance_propagator(
    T* q_in, T* q_out, std::string monomer_type, const double* q_mask)
{
    try
    {
        const int M = this->cb->get_total_grid();
        const int M_COMPLEX = pseudo->get_total_complex_grid();

        // Temporary arrays
        std::vector<T> q_out1(M), q_out2(M);
        std::vector<double> k_q_in1(is_periodic_ ? M_COMPLEX * 2 : M_COMPLEX);
        std::vector<double> k_q_in2(is_periodic_ ? M_COMPLEX * 2 : M_COMPLEX);

        T* _exp_dw = this->exp_dw[monomer_type];
        T* _exp_dw_half = this->exp_dw_half[monomer_type];

        const double* _boltz_bond = pseudo->get_boltz_bond(monomer_type);
        const double* _boltz_bond_half = pseudo->get_boltz_bond_half(monomer_type);

        // ===== Step 1: Full step =====
        // Apply exp(-w*ds/2)
        for (int i = 0; i < M; ++i)
            q_out1[i] = _exp_dw[i] * q_in[i];

        // Forward transform
        transform_forward(q_out1.data(), k_q_in1.data());

        // Multiply by exp(-k^2 ds/6) in Fourier space
        if (is_periodic_)
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

        // Backward transform
        transform_backward(k_q_in1.data(), q_out1.data());

        // Apply exp(-w*ds/2)
        for (int i = 0; i < M; ++i)
            q_out1[i] *= _exp_dw[i];

        // ===== Step 2: Two half steps =====
        // First half step
        for (int i = 0; i < M; ++i)
            q_out2[i] = _exp_dw_half[i] * q_in[i];

        transform_forward(q_out2.data(), k_q_in2.data());

        if (is_periodic_)
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

        transform_backward(k_q_in2.data(), q_out2.data());

        for (int i = 0; i < M; ++i)
            q_out2[i] *= _exp_dw[i];

        // Second half step
        transform_forward(q_out2.data(), k_q_in2.data());

        if (is_periodic_)
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

        transform_backward(k_q_in2.data(), q_out2.data());

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

//------------------------------------------------------------------------------
// Compute single segment stress
//------------------------------------------------------------------------------
template <typename T>
std::vector<T> CpuSolverPseudoMixedBC<T>::compute_single_segment_stress(
    T* q_1, T* q_2, std::string monomer_type, bool is_half_bond_length)
{
    try
    {
        const int DIM = this->cb->get_dim();
        const int M_COMPLEX = pseudo->get_total_complex_grid();
        auto bond_lengths = this->molecules->get_bond_lengths();
        double bond_length_sq = bond_lengths[monomer_type] * bond_lengths[monomer_type];

        std::vector<T> stress(DIM, 0.0);

        // For non-periodic BCs, compute stress in real space
        // (simplified implementation - may need refinement for accuracy)
        if (!is_periodic_)
        {
            // Allocate Fourier coefficient arrays
            std::vector<double> qk_1(M_COMPLEX);
            std::vector<double> qk_2(M_COMPLEX);

            // Transform to Fourier space
            transform_forward(q_1, qk_1.data());
            transform_forward(q_2, qk_2.data());

            const double* _fourier_basis_x = pseudo->get_fourier_basis_x();
            const double* _fourier_basis_y = pseudo->get_fourier_basis_y();
            const double* _fourier_basis_z = pseudo->get_fourier_basis_z();

            // Compute stress from Fourier coefficients
            for (int i = 0; i < M_COMPLEX; ++i)
            {
                double coeff = bond_length_sq * qk_1[i] * qk_2[i];

                if (DIM >= 1)
                    stress[DIM - 1] += coeff * _fourier_basis_z[i];
                if (DIM >= 2)
                    stress[DIM - 2] += coeff * _fourier_basis_y[i];
                if (DIM >= 3)
                    stress[DIM - 3] += coeff * _fourier_basis_x[i];
            }

            return stress;
        }

        // For periodic BCs, use complex FFT
        std::vector<std::complex<double>> qk_1(M_COMPLEX);
        std::vector<std::complex<double>> qk_2(M_COMPLEX);

        fft_periodic->forward(q_1, qk_1.data());
        fft_periodic->forward(q_2, qk_2.data());

        const double* _fourier_basis_x = pseudo->get_fourier_basis_x();
        const double* _fourier_basis_y = pseudo->get_fourier_basis_y();
        const double* _fourier_basis_z = pseudo->get_fourier_basis_z();

        for (int i = 0; i < M_COMPLEX; ++i)
        {
            T coeff;
            if constexpr (std::is_same<T, double>::value)
                coeff = bond_length_sq * (qk_1[i] * std::conj(qk_2[i])).real();
            else
                coeff = bond_length_sq * qk_1[i] * std::conj(qk_2[i]);

            if (DIM == 3)
            {
                stress[0] += coeff * _fourier_basis_x[i];
                stress[1] += coeff * _fourier_basis_y[i];
                stress[2] += coeff * _fourier_basis_z[i];
            }
            else if (DIM == 2)
            {
                stress[0] += coeff * _fourier_basis_y[i];
                stress[1] += coeff * _fourier_basis_z[i];
            }
            else if (DIM == 1)
            {
                stress[0] += coeff * _fourier_basis_z[i];
            }
        }

        return stress;
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

// Explicit template instantiation
template class CpuSolverPseudoMixedBC<double>;
template class CpuSolverPseudoMixedBC<std::complex<double>>;
