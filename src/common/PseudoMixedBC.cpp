/**
 * @file PseudoMixedBC.cpp
 * @brief Implementation of pseudo-spectral utilities for mixed BCs.
 *
 * Computes Boltzmann factors for non-periodic boundary conditions:
 * - REFLECTING (DCT): k = π*n/L, n = 0, 1, ..., N-1
 * - ABSORBING (DST): k = π*(n+1)/L, n = 0, 1, ..., N-1
 *
 * @see PseudoMixedBC.h for class documentation
 */

#include <iostream>
#include <cmath>
#include <numbers>
#include "PseudoMixedBC.h"

//------------------------------------------------------------------------------
// Constructor
//------------------------------------------------------------------------------
template <typename T>
PseudoMixedBC<T>::PseudoMixedBC(
    std::map<std::string, double> bond_lengths,
    std::vector<BoundaryCondition> bc,
    std::vector<int> nx, std::vector<double> dx, double ds)
{
    try
    {
        this->bond_lengths = bond_lengths;
        this->bc = bc;
        this->nx = nx;
        this->dx = dx;
        this->ds = ds;

        // Compute total grid
        total_grid = 1;
        for (size_t d = 0; d < nx.size(); ++d)
            total_grid *= nx[d];

        update_total_complex_grid();
        const int M_COMPLEX = get_total_complex_grid();

        // Create boltz_bond arrays
        for (const auto& item : bond_lengths)
        {
            std::string monomer_type = item.first;
            boltz_bond[monomer_type] = new double[M_COMPLEX];
            boltz_bond_half[monomer_type] = new double[M_COMPLEX];
        }

        // Allocate Fourier basis arrays for stress calculation
        fourier_basis_x = new double[M_COMPLEX];
        fourier_basis_y = new double[M_COMPLEX];
        fourier_basis_z = new double[M_COMPLEX];

        update_boltz_bond();
        update_weighted_fourier_basis();
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
PseudoMixedBC<T>::~PseudoMixedBC()
{
    delete[] fourier_basis_x;
    delete[] fourier_basis_y;
    delete[] fourier_basis_z;

    for (const auto& item : boltz_bond)
        delete[] item.second;
    for (const auto& item : boltz_bond_half)
        delete[] item.second;
}

//------------------------------------------------------------------------------
// Check if all periodic
//------------------------------------------------------------------------------
template <typename T>
bool PseudoMixedBC<T>::is_all_periodic() const
{
    for (const auto& b : bc)
    {
        if (b != BoundaryCondition::PERIODIC)
            return false;
    }
    return true;
}

//------------------------------------------------------------------------------
// Update total complex grid
//------------------------------------------------------------------------------
template <typename T>
void PseudoMixedBC<T>::update_total_complex_grid()
{
    if (is_all_periodic())
    {
        // Standard r2c FFT: last dimension halved
        if (nx.size() == 3)
            total_complex_grid = nx[0] * nx[1] * (nx[2] / 2 + 1);
        else if (nx.size() == 2)
            total_complex_grid = nx[0] * (nx[1] / 2 + 1);
        else if (nx.size() == 1)
            total_complex_grid = nx[0] / 2 + 1;
    }
    else
    {
        // DCT/DST: real-to-real, same size
        total_complex_grid = total_grid;
    }
}

//------------------------------------------------------------------------------
// Get total complex grid
//------------------------------------------------------------------------------
template <typename T>
int PseudoMixedBC<T>::get_total_complex_grid()
{
    return total_complex_grid;
}

//------------------------------------------------------------------------------
// Get Boltzmann factors
//------------------------------------------------------------------------------
template <typename T>
double* PseudoMixedBC<T>::get_boltz_bond(std::string monomer_type)
{
    return boltz_bond[monomer_type];
}

template <typename T>
double* PseudoMixedBC<T>::get_boltz_bond_half(std::string monomer_type)
{
    return boltz_bond_half[monomer_type];
}

//------------------------------------------------------------------------------
// Get Fourier basis
//------------------------------------------------------------------------------
template <typename T>
const double* PseudoMixedBC<T>::get_fourier_basis_x()
{
    return fourier_basis_x;
}

template <typename T>
const double* PseudoMixedBC<T>::get_fourier_basis_y()
{
    return fourier_basis_y;
}

template <typename T>
const double* PseudoMixedBC<T>::get_fourier_basis_z()
{
    return fourier_basis_z;
}

//------------------------------------------------------------------------------
// Update Boltzmann factors
//------------------------------------------------------------------------------
template <typename T>
void PseudoMixedBC<T>::update_boltz_bond()
{
    try
    {
        const double PI = std::numbers::pi;
        const int DIM = nx.size();

        // Expand to 3D for uniform processing
        std::vector<int> tnx(3, 1);
        std::vector<double> tdx(3, 1.0);
        std::vector<BoundaryCondition> tbc(3, BoundaryCondition::PERIODIC);

        for (int d = 0; d < DIM; ++d)
        {
            tnx[3 - DIM + d] = nx[d];
            tdx[3 - DIM + d] = dx[d];
            tbc[3 - DIM + d] = bc[d];
        }

        for (const auto& item : bond_lengths)
        {
            std::string monomer_type = item.first;
            double bond_length_sq = item.second * item.second;
            double* _boltz_bond = boltz_bond[monomer_type];
            double* _boltz_bond_half = boltz_bond_half[monomer_type];

            // Compute wavenumber factors based on boundary conditions
            // PERIODIC: k = 2π*n/L -> factor = -(2π/L)² * b² * ds/6
            // REFLECTING: k = π*n/L -> factor = -(π/L)² * b² * ds/6
            // ABSORBING: k = π*(n+1)/L -> factor = -(π/L)² * b² * ds/6 (with n+1 offset)
            double xfactor[3];
            for (int d = 0; d < 3; ++d)
            {
                double L = tnx[d] * tdx[d];
                if (tbc[d] == BoundaryCondition::PERIODIC)
                    xfactor[d] = -bond_length_sq * std::pow(2 * PI / L, 2) * ds / 6.0;
                else  // REFLECTING or ABSORBING
                    xfactor[d] = -bond_length_sq * std::pow(PI / L, 2) * ds / 6.0;
            }

            // Loop over all grid points
            for (int i = 0; i < tnx[0]; ++i)
            {
                // Wavenumber index for dimension 0
                int ki;
                if (tbc[0] == BoundaryCondition::PERIODIC)
                {
                    if (i > tnx[0] / 2)
                        ki = tnx[0] - i;
                    else
                        ki = i;
                }
                else if (tbc[0] == BoundaryCondition::REFLECTING)
                {
                    ki = i;  // DCT: k = π*i/L
                }
                else  // ABSORBING
                {
                    ki = i + 1;  // DST: k = π*(i+1)/L
                }

                for (int j = 0; j < tnx[1]; ++j)
                {
                    // Wavenumber index for dimension 1
                    int kj;
                    if (tbc[1] == BoundaryCondition::PERIODIC)
                    {
                        if (j > tnx[1] / 2)
                            kj = tnx[1] - j;
                        else
                            kj = j;
                    }
                    else if (tbc[1] == BoundaryCondition::REFLECTING)
                    {
                        kj = j;
                    }
                    else  // ABSORBING
                    {
                        kj = j + 1;
                    }

                    for (int k = 0; k < tnx[2]; ++k)
                    {
                        // Wavenumber index for dimension 2
                        int kk;
                        if (tbc[2] == BoundaryCondition::PERIODIC)
                        {
                            if (k > tnx[2] / 2)
                                kk = tnx[2] - k;
                            else
                                kk = k;
                        }
                        else if (tbc[2] == BoundaryCondition::REFLECTING)
                        {
                            kk = k;
                        }
                        else  // ABSORBING
                        {
                            kk = k + 1;
                        }

                        int idx = i * tnx[1] * tnx[2] + j * tnx[2] + k;

                        double mag_q2 = ki * ki * xfactor[0] +
                                       kj * kj * xfactor[1] +
                                       kk * kk * xfactor[2];

                        _boltz_bond[idx] = std::exp(mag_q2);
                        _boltz_bond_half[idx] = std::exp(mag_q2 / 2.0);
                    }
                }
            }
        }
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

//------------------------------------------------------------------------------
// Update weighted Fourier basis for stress calculation
//------------------------------------------------------------------------------
template <typename T>
void PseudoMixedBC<T>::update_weighted_fourier_basis()
{
    const double PI = std::numbers::pi;
    const int DIM = nx.size();

    // Expand to 3D
    std::vector<int> tnx(3, 1);
    std::vector<double> tdx(3, 1.0);
    std::vector<BoundaryCondition> tbc(3, BoundaryCondition::PERIODIC);

    for (int d = 0; d < DIM; ++d)
    {
        tnx[3 - DIM + d] = nx[d];
        tdx[3 - DIM + d] = dx[d];
        tbc[3 - DIM + d] = bc[d];
    }

    // Compute wavenumber factors
    double xfactor[3];
    for (int d = 0; d < 3; ++d)
    {
        double L = tnx[d] * tdx[d];
        if (tbc[d] == BoundaryCondition::PERIODIC)
            xfactor[d] = std::pow(2 * PI / L, 2);
        else
            xfactor[d] = std::pow(PI / L, 2);
    }

    for (int i = 0; i < tnx[0]; ++i)
    {
        int ki;
        if (tbc[0] == BoundaryCondition::PERIODIC)
        {
            if (i > tnx[0] / 2)
                ki = tnx[0] - i;
            else
                ki = i;
        }
        else if (tbc[0] == BoundaryCondition::REFLECTING)
            ki = i;
        else
            ki = i + 1;

        for (int j = 0; j < tnx[1]; ++j)
        {
            int kj;
            if (tbc[1] == BoundaryCondition::PERIODIC)
            {
                if (j > tnx[1] / 2)
                    kj = tnx[1] - j;
                else
                    kj = j;
            }
            else if (tbc[1] == BoundaryCondition::REFLECTING)
                kj = j;
            else
                kj = j + 1;

            for (int k = 0; k < tnx[2]; ++k)
            {
                int kk;
                if (tbc[2] == BoundaryCondition::PERIODIC)
                {
                    if (k > tnx[2] / 2)
                        kk = tnx[2] - k;
                    else
                        kk = k;
                }
                else if (tbc[2] == BoundaryCondition::REFLECTING)
                    kk = k;
                else
                    kk = k + 1;

                int idx = i * tnx[1] * tnx[2] + j * tnx[2] + k;

                fourier_basis_x[idx] = ki * ki * xfactor[0];
                fourier_basis_y[idx] = kj * kj * xfactor[1];
                fourier_basis_z[idx] = kk * kk * xfactor[2];
            }
        }
    }
}

//------------------------------------------------------------------------------
// Update all arrays
//------------------------------------------------------------------------------
template <typename T>
void PseudoMixedBC<T>::update(
    std::vector<BoundaryCondition> bc,
    std::map<std::string, double> bond_lengths,
    std::vector<double> dx, double ds)
{
    this->bond_lengths = bond_lengths;
    this->bc = bc;
    this->dx = dx;
    this->ds = ds;

    update_total_complex_grid();
    update_boltz_bond();
    update_weighted_fourier_basis();
}

// Explicit template instantiation
template class PseudoMixedBC<double>;
template class PseudoMixedBC<std::complex<double>>;
