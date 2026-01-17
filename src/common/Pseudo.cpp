/**
 * @file Pseudo.cpp
 * @brief Unified implementation of pseudo-spectral method utilities.
 *
 * Supports all boundary conditions and non-orthogonal crystal systems:
 * - PERIODIC: FFT with recip_metric for non-orthogonal lattices
 * - REFLECTING (DCT): k = π*n/L, n = 0, 1, ..., N-1
 * - ABSORBING (DST): k = π*(n+1)/L, n = 0, 1, ..., N-1
 *
 * @see Pseudo.h for class documentation
 */

#include <iostream>
#include <cmath>
#include <numbers>
#include <array>
#include <complex>
#include <utility>
#include "Pseudo.h"

//------------------------------------------------------------------------------
// Sinc function: sin(x)/x with sinc(0) = 1
//------------------------------------------------------------------------------
static inline double sinc(double x)
{
    if (std::abs(x) < 1e-10)
        return 1.0;
    return std::sin(x) / x;
}

//------------------------------------------------------------------------------
// Constructor
//------------------------------------------------------------------------------
template <typename T>
Pseudo<T>::Pseudo(
    std::map<std::string, double> bond_lengths,
    std::vector<BoundaryCondition> bc,
    std::vector<int> nx, std::vector<double> dx, double ds,
    std::array<double, 6> recip_metric)
{
    try
    {
        this->bond_lengths = bond_lengths;
        this->bc = bc;
        this->nx = nx;
        this->dx = dx;
        this->ds = ds;
        this->recip_metric_ = recip_metric;
        this->use_cell_averaged_bond_ = false;
        this->n_cell_average_momentum_ = 1;

        // Compute total grid
        total_grid = 1;
        for (size_t d = 0; d < nx.size(); ++d)
            total_grid *= nx[d];

        update_total_complex_grid();
        const int M_COMPLEX = get_total_complex_grid();

        // Store global ds as ds_index=1
        ds_values[1] = ds;

        // Allocate Boltzmann factors for ds_index=1 (global ds)
        for (const auto& item : bond_lengths)
        {
            std::string monomer_type = item.first;
            boltz_bond[1][monomer_type] = new double[M_COMPLEX];
            boltz_bond_half[1][monomer_type] = new double[M_COMPLEX];
        }

        // Allocate Fourier basis arrays (diagonal terms)
        fourier_basis_x = new double[M_COMPLEX];
        fourier_basis_y = new double[M_COMPLEX];
        fourier_basis_z = new double[M_COMPLEX];
        // Cross-terms for non-orthogonal systems (only used for periodic BC)
        fourier_basis_xy = new double[M_COMPLEX];
        fourier_basis_xz = new double[M_COMPLEX];
        fourier_basis_yz = new double[M_COMPLEX];

        // Negative frequency mapping only for complex fields with periodic BC
        if constexpr (std::is_same<T, std::complex<double>>::value)
        {
            if (is_all_periodic())
                negative_k_idx = new int[M_COMPLEX];
            else
                negative_k_idx = nullptr;
        }

        update_boltz_bond();
        update_weighted_fourier_basis();
        update_negative_frequency_mapping();
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
Pseudo<T>::~Pseudo()
{
    // Delete diagonal basis arrays
    delete[] fourier_basis_x;
    delete[] fourier_basis_y;
    delete[] fourier_basis_z;
    // Delete cross-term basis arrays
    delete[] fourier_basis_xy;
    delete[] fourier_basis_xz;
    delete[] fourier_basis_yz;

    if constexpr (std::is_same<T, std::complex<double>>::value)
    {
        if (negative_k_idx != nullptr)
            delete[] negative_k_idx;
    }

    // Delete Boltzmann factors for all ds_index values
    for (const auto& ds_pair : boltz_bond)
    {
        for (const auto& item : ds_pair.second)
            delete[] item.second;
    }
    for (const auto& ds_pair : boltz_bond_half)
    {
        for (const auto& item : ds_pair.second)
            delete[] item.second;
    }
}

//------------------------------------------------------------------------------
// Check if all BCs are periodic
//------------------------------------------------------------------------------
template <typename T>
bool Pseudo<T>::is_all_periodic() const
{
    for (const auto& b : bc)
    {
        if (b != BoundaryCondition::PERIODIC)
            return false;
    }
    return true;
}

//------------------------------------------------------------------------------
// Getters
//------------------------------------------------------------------------------
template <typename T>
int Pseudo<T>::get_total_complex_grid()
{
    return total_complex_grid;
}

template <typename T>
double* Pseudo<T>::get_boltz_bond(std::string monomer_type, int ds_index)
{
    if (!boltz_bond.contains(ds_index))
        throw_with_line_number("ds_index " + std::to_string(ds_index) + " not found in boltz_bond. Call add_ds_value() and finalize_ds_values() first.");
    if (!boltz_bond[ds_index].contains(monomer_type))
        throw_with_line_number("monomer_type \"" + monomer_type + "\" not found in boltz_bond[" + std::to_string(ds_index) + "].");
    return boltz_bond[ds_index][monomer_type];
}

template <typename T>
double* Pseudo<T>::get_boltz_bond_half(std::string monomer_type, int ds_index)
{
    if (!boltz_bond_half.contains(ds_index))
        throw_with_line_number("ds_index " + std::to_string(ds_index) + " not found in boltz_bond_half. Call add_ds_value() and finalize_ds_values() first.");
    if (!boltz_bond_half[ds_index].contains(monomer_type))
        throw_with_line_number("monomer_type \"" + monomer_type + "\" not found in boltz_bond_half[" + std::to_string(ds_index) + "].");
    return boltz_bond_half[ds_index][monomer_type];
}

template <typename T>
const double* Pseudo<T>::get_fourier_basis_x()
{
    return fourier_basis_x;
}

template <typename T>
const double* Pseudo<T>::get_fourier_basis_y()
{
    return fourier_basis_y;
}

template <typename T>
const double* Pseudo<T>::get_fourier_basis_z()
{
    return fourier_basis_z;
}

template <typename T>
const double* Pseudo<T>::get_fourier_basis_xy()
{
    return fourier_basis_xy;
}

template <typename T>
const double* Pseudo<T>::get_fourier_basis_xz()
{
    return fourier_basis_xz;
}

template <typename T>
const double* Pseudo<T>::get_fourier_basis_yz()
{
    return fourier_basis_yz;
}

template <typename T>
const int* Pseudo<T>::get_negative_frequency_mapping()
{
    return negative_k_idx;
}

//------------------------------------------------------------------------------
// Update total complex grid size
//------------------------------------------------------------------------------
template <typename T>
void Pseudo<T>::update_total_complex_grid()
{
    if (is_all_periodic())
    {
        // Periodic BC: use r2c FFT for double, c2c for complex
        if constexpr (std::is_same<T, double>::value)
        {
            if (nx.size() == 3)
                total_complex_grid = nx[0] * nx[1] * (nx[2] / 2 + 1);
            else if (nx.size() == 2)
                total_complex_grid = nx[0] * (nx[1] / 2 + 1);
            else if (nx.size() == 1)
                total_complex_grid = nx[0] / 2 + 1;
        }
        else
        {
            total_complex_grid = total_grid;
        }
    }
    else
    {
        // Non-periodic BC: DCT/DST uses full grid
        total_complex_grid = total_grid;
    }
}

//------------------------------------------------------------------------------
// Update Boltzmann factors
//------------------------------------------------------------------------------
template <typename T>
void Pseudo<T>::update_boltz_bond()
{
    try
    {
        if (is_all_periodic())
        {
            // Periodic BC: use recip_metric for non-orthogonal systems
            update_boltz_bond_periodic();
        }
        else
        {
            // Non-periodic BC: use mixed BC formula
            update_boltz_bond_mixed();
        }
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

//------------------------------------------------------------------------------
// Update Boltzmann factors for periodic BC (with recip_metric support)
//------------------------------------------------------------------------------
template <typename T>
void update_boltz_bond_periodic_impl(
    std::map<std::string, double>& bond_lengths,
    std::map<std::string, double*>& boltz_bond,
    std::map<std::string, double*>& boltz_bond_half,
    const std::vector<int>& nx,
    double ds,
    const std::array<double, 6>& recip_metric_)
{
    const double PI = std::numbers::pi;
    const double FOUR_PI_SQ = 4.0 * PI * PI;
    const int DIM = nx.size();

    // Pad to 3D for unified loop
    std::vector<int> tnx(3, 1);
    if (DIM == 3)
        tnx = {nx[0], nx[1], nx[2]};
    else if (DIM == 2)
        tnx = {1, nx[0], nx[1]};
    else if (DIM == 1)
        tnx = {1, 1, nx[0]};

    // Extract reciprocal metric components
    double Gii, Gjj, Gkk, Gij, Gik, Gjk;
    if (DIM == 3) {
        Gii = recip_metric_[0]; Gjj = recip_metric_[3]; Gkk = recip_metric_[5];
        Gij = recip_metric_[1]; Gik = recip_metric_[2]; Gjk = recip_metric_[4];
    } else if (DIM == 2) {
        Gii = 0.0; Gjj = recip_metric_[0]; Gkk = recip_metric_[3];
        Gij = 0.0; Gik = 0.0; Gjk = recip_metric_[1];
    } else {
        Gii = 0.0; Gjj = 0.0; Gkk = recip_metric_[0];
        Gij = 0.0; Gik = 0.0; Gjk = 0.0;
    }

    for (const auto& item : bond_lengths)
    {
        std::string monomer_type = item.first;
        double bond_length_sq = item.second * item.second;
        double* _boltz_bond = boltz_bond[monomer_type];
        double* _boltz_bond_half = boltz_bond_half[monomer_type];

        double prefactor = -bond_length_sq * FOUR_PI_SQ * ds / 6.0;

        for (int i = 0; i < tnx[0]; i++)
        {
            int itemp = (i > tnx[0]/2) ? tnx[0] - i : i;
            int i_signed = (i > tnx[0]/2) ? i - tnx[0] : i;

            for (int j = 0; j < tnx[1]; j++)
            {
                int jtemp = (j > tnx[1]/2) ? tnx[1] - j : j;
                int j_signed = (j > tnx[1]/2) ? j - tnx[1] : j;

                if constexpr (std::is_same<T, double>::value)
                {
                    for (int k = 0; k < tnx[2]/2+1; k++)
                    {
                        int ktemp = k;
                        int k_signed = k;
                        int idx = i * tnx[1]*(tnx[2]/2+1) + j*(tnx[2]/2+1) + k;

                        double mag_q2 = prefactor * (
                            Gii * itemp * itemp + Gjj * jtemp * jtemp + Gkk * ktemp * ktemp +
                            2.0 * Gij * i_signed * j_signed +
                            2.0 * Gik * i_signed * k_signed +
                            2.0 * Gjk * j_signed * k_signed
                        );
                        _boltz_bond[idx] = std::exp(mag_q2);
                        _boltz_bond_half[idx] = std::exp(mag_q2 / 2.0);
                    }
                }
                else
                {
                    for (int k = 0; k < tnx[2]; k++)
                    {
                        int ktemp = (k > tnx[2]/2) ? tnx[2] - k : k;
                        int k_signed = (k > tnx[2]/2) ? k - tnx[2] : k;
                        int idx = i * tnx[1]*tnx[2] + j*tnx[2] + k;

                        double mag_q2 = prefactor * (
                            Gii * itemp * itemp + Gjj * jtemp * jtemp + Gkk * ktemp * ktemp +
                            2.0 * Gij * i_signed * j_signed +
                            2.0 * Gik * i_signed * k_signed +
                            2.0 * Gjk * j_signed * k_signed
                        );
                        _boltz_bond[idx] = std::exp(mag_q2);
                        _boltz_bond_half[idx] = std::exp(mag_q2 / 2.0);
                    }
                }
            }
        }
    }
}

template <typename T>
void Pseudo<T>::update_boltz_bond_periodic()
{
    // Update for ds_index=1 (global ds)
    update_boltz_bond_periodic_for_ds_index(1);
}

//------------------------------------------------------------------------------
// Update Boltzmann factors for mixed BC
//------------------------------------------------------------------------------
template <typename T>
void Pseudo<T>::update_boltz_bond_mixed()
{
    // Update for ds_index=1 (global ds)
    update_boltz_bond_mixed_for_ds_index(1);
}

//------------------------------------------------------------------------------
// Update weighted Fourier basis for stress calculation
//------------------------------------------------------------------------------
template <typename T>
void Pseudo<T>::update_weighted_fourier_basis()
{
    if (is_all_periodic())
        update_weighted_fourier_basis_periodic();
    else
        update_weighted_fourier_basis_mixed();
}

//------------------------------------------------------------------------------
// Update Fourier basis for periodic BC (with cross-terms)
//------------------------------------------------------------------------------
template <typename T>
void update_weighted_fourier_basis_periodic_impl(
    double* fourier_basis_x, double* fourier_basis_y, double* fourier_basis_z,
    double* fourier_basis_xy, double* fourier_basis_xz, double* fourier_basis_yz,
    const std::vector<int>& nx, const std::vector<double>& dx,
    const std::array<double, 6>& recip_metric_)
{
    const double PI = std::numbers::pi;
    const int DIM = nx.size();

    // Pad to 3D
    std::vector<int> tnx(3, 1);
    std::vector<double> tdx(3, 1.0);
    if (DIM == 3) {
        tnx = {nx[0], nx[1], nx[2]};
        tdx = {dx[0], dx[1], dx[2]};
    } else if (DIM == 2) {
        tnx = {1, nx[0], nx[1]};
        tdx = {1.0, dx[0], dx[1]};
    } else {
        tnx = {1, 1, nx[0]};
        tdx = {1.0, 1.0, dx[0]};
    }

    // Calculate factors
    double xfactor[3] = {0.0, 0.0, 0.0};
    double k_factor[3] = {0.0, 0.0, 0.0};
    for (int d = 0; d < DIM; d++) {
        double L = nx[d] * dx[d];
        xfactor[d] = std::pow(2*PI/L, 2);
        k_factor[d] = 2*PI/L;
    }

    // Cross factors
    double Gstar_01 = (DIM >= 2) ? recip_metric_[1] : 0.0;
    double Gstar_02 = (DIM >= 3) ? recip_metric_[2] : 0.0;
    double Gstar_12 = (DIM >= 3) ? recip_metric_[4] : 0.0;

    double cross_factor_01 = (DIM >= 2) ? 2.0 * k_factor[0] * k_factor[1] * Gstar_01 : 0.0;
    double cross_factor_02 = (DIM >= 3) ? 2.0 * k_factor[0] * k_factor[2] * Gstar_02 : 0.0;
    double cross_factor_12 = (DIM >= 3) ? 2.0 * k_factor[1] * k_factor[2] * Gstar_12 : 0.0;

    for (int i = 0; i < tnx[0]; i++)
    {
        int itemp = (i > tnx[0]/2) ? tnx[0] - i : i;
        int i_signed = (i > tnx[0]/2) ? i - tnx[0] : i;

        for (int j = 0; j < tnx[1]; j++)
        {
            int jtemp = (j > tnx[1]/2) ? tnx[1] - j : j;
            int j_signed = (j > tnx[1]/2) ? j - tnx[1] : j;

            if constexpr (std::is_same<T, double>::value)
            {
                for (int k = 0; k < tnx[2]/2+1; k++)
                {
                    int ktemp = k;
                    int k_signed = k;
                    int idx = i * tnx[1]*(tnx[2]/2+1) + j*(tnx[2]/2+1) + k;

                    int n0, n1, n2, n0_s, n1_s, n2_s;
                    if (DIM == 3) {
                        n0 = itemp; n1 = jtemp; n2 = ktemp;
                        n0_s = i_signed; n1_s = j_signed; n2_s = k_signed;
                    } else if (DIM == 2) {
                        n0 = jtemp; n1 = ktemp; n2 = 0;
                        n0_s = j_signed; n1_s = k_signed; n2_s = 0;
                    } else {
                        n0 = ktemp; n1 = 0; n2 = 0;
                        n0_s = k_signed; n1_s = 0; n2_s = 0;
                    }

                    fourier_basis_x[idx] = n0*n0*xfactor[0];
                    fourier_basis_y[idx] = (DIM >= 2) ? n1*n1*xfactor[1] : 0.0;
                    fourier_basis_z[idx] = (DIM >= 3) ? n2*n2*xfactor[2] : 0.0;
                    fourier_basis_xy[idx] = (DIM >= 2) ? n0_s*n1_s*cross_factor_01 : 0.0;
                    fourier_basis_xz[idx] = (DIM >= 3) ? n0_s*n2_s*cross_factor_02 : 0.0;
                    fourier_basis_yz[idx] = (DIM >= 3) ? n1_s*n2_s*cross_factor_12 : 0.0;

                    // Weight factor of 2 for interior k modes
                    if (k != 0 && 2*k != tnx[2]) {
                        fourier_basis_x[idx] *= 2;
                        fourier_basis_y[idx] *= 2;
                        fourier_basis_z[idx] *= 2;
                        fourier_basis_xy[idx] *= 2;
                        fourier_basis_xz[idx] *= 2;
                        fourier_basis_yz[idx] *= 2;
                    }
                }
            }
            else
            {
                for (int k = 0; k < tnx[2]; k++)
                {
                    int ktemp = (k > tnx[2]/2) ? tnx[2] - k : k;
                    int k_signed = (k > tnx[2]/2) ? k - tnx[2] : k;
                    int idx = i * tnx[1]*tnx[2] + j*tnx[2] + k;

                    int n0, n1, n2, n0_s, n1_s, n2_s;
                    if (DIM == 3) {
                        n0 = itemp; n1 = jtemp; n2 = ktemp;
                        n0_s = i_signed; n1_s = j_signed; n2_s = k_signed;
                    } else if (DIM == 2) {
                        n0 = jtemp; n1 = ktemp; n2 = 0;
                        n0_s = j_signed; n1_s = k_signed; n2_s = 0;
                    } else {
                        n0 = ktemp; n1 = 0; n2 = 0;
                        n0_s = k_signed; n1_s = 0; n2_s = 0;
                    }

                    fourier_basis_x[idx] = n0*n0*xfactor[0];
                    fourier_basis_y[idx] = (DIM >= 2) ? n1*n1*xfactor[1] : 0.0;
                    fourier_basis_z[idx] = (DIM >= 3) ? n2*n2*xfactor[2] : 0.0;
                    fourier_basis_xy[idx] = (DIM >= 2) ? n0_s*n1_s*cross_factor_01 : 0.0;
                    fourier_basis_xz[idx] = (DIM >= 3) ? n0_s*n2_s*cross_factor_02 : 0.0;
                    fourier_basis_yz[idx] = (DIM >= 3) ? n1_s*n2_s*cross_factor_12 : 0.0;
                }
            }
        }
    }
}

template <typename T>
void Pseudo<T>::update_weighted_fourier_basis_periodic()
{
    update_weighted_fourier_basis_periodic_impl<T>(
        fourier_basis_x, fourier_basis_y, fourier_basis_z,
        fourier_basis_xy, fourier_basis_xz, fourier_basis_yz,
        nx, dx, recip_metric_);
}

//------------------------------------------------------------------------------
// Update Fourier basis for mixed BC (no cross-terms)
//------------------------------------------------------------------------------
template <typename T>
void Pseudo<T>::update_weighted_fourier_basis_mixed()
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
            ki = (i > tnx[0]/2) ? tnx[0] - i : i;
        else if (tbc[0] == BoundaryCondition::REFLECTING)
            ki = i;
        else
            ki = i + 1;

        for (int j = 0; j < tnx[1]; ++j)
        {
            int kj;
            if (tbc[1] == BoundaryCondition::PERIODIC)
                kj = (j > tnx[1]/2) ? tnx[1] - j : j;
            else if (tbc[1] == BoundaryCondition::REFLECTING)
                kj = j;
            else
                kj = j + 1;

            for (int k = 0; k < tnx[2]; ++k)
            {
                int kk;
                if (tbc[2] == BoundaryCondition::PERIODIC)
                    kk = (k > tnx[2]/2) ? tnx[2] - k : k;
                else if (tbc[2] == BoundaryCondition::REFLECTING)
                    kk = k;
                else
                    kk = k + 1;

                int idx = i * tnx[1] * tnx[2] + j * tnx[2] + k;

                fourier_basis_x[idx] = ki * ki * xfactor[0];
                fourier_basis_y[idx] = kj * kj * xfactor[1];
                fourier_basis_z[idx] = kk * kk * xfactor[2];
                // Cross-terms are zero for non-periodic BC
                fourier_basis_xy[idx] = 0.0;
                fourier_basis_xz[idx] = 0.0;
                fourier_basis_yz[idx] = 0.0;
            }
        }
    }
}

//------------------------------------------------------------------------------
// Update negative frequency mapping (only for periodic BC with complex fields)
//------------------------------------------------------------------------------
template <typename T>
void Pseudo<T>::update_negative_frequency_mapping() {}

template <>
void Pseudo<std::complex<double>>::update_negative_frequency_mapping()
{
    if (!is_all_periodic() || negative_k_idx == nullptr)
        return;

    const int DIM = nx.size();
    std::vector<int> tnx(3, 1);

    if (DIM == 3)
        tnx = {nx[0], nx[1], nx[2]};
    else if (DIM == 2)
        tnx = {1, nx[0], nx[1]};
    else if (DIM == 1)
        tnx = {1, 1, nx[0]};

    for (int i = 0; i < tnx[0]; i++)
    {
        int itemp = (i == 0) ? 0 : tnx[0] - i;
        for (int j = 0; j < tnx[1]; j++)
        {
            int jtemp = (j == 0) ? 0 : tnx[1] - j;
            for (int k = 0; k < tnx[2]; k++)
            {
                int ktemp = (k == 0) ? 0 : tnx[2] - k;

                int idx = i * tnx[1] * tnx[2] + j * tnx[2] + k;
                int idx_minus = itemp * tnx[1] * tnx[2] + jtemp * tnx[2] + ktemp;

                negative_k_idx[idx] = idx_minus;
            }
        }
    }
}

//------------------------------------------------------------------------------
// Update all arrays
//------------------------------------------------------------------------------
template <typename T>
void Pseudo<T>::update(
    std::vector<BoundaryCondition> bc,
    std::map<std::string, double> bond_lengths,
    std::vector<double> dx, double ds,
    std::array<double, 6> recip_metric)
{
    this->bond_lengths = bond_lengths;
    this->bc = bc;
    this->dx = dx;
    this->ds = ds;
    this->ds_values[1] = ds;  // Update global ds
    this->recip_metric_ = recip_metric;

    update_total_complex_grid();
    update_boltz_bond();
    update_weighted_fourier_basis();
}

//------------------------------------------------------------------------------
// Set cell-averaged bond function mode
//------------------------------------------------------------------------------
template <typename T>
void Pseudo<T>::set_cell_averaged_bond(bool enabled)
{
    if (use_cell_averaged_bond_ != enabled)
    {
        use_cell_averaged_bond_ = enabled;
        // Recompute all Boltzmann factors with the new setting
        for (const auto& ds_pair : ds_values)
        {
            int ds_idx = ds_pair.first;
            double local_ds = ds_pair.second;

            double saved_ds = this->ds;
            this->ds = local_ds;

            if (is_all_periodic())
                update_boltz_bond_periodic_for_ds_index(ds_idx);
            else
                update_boltz_bond_mixed_for_ds_index(ds_idx);

            this->ds = saved_ds;
        }
    }
}

//------------------------------------------------------------------------------
// Set cell-average momentum parameter
//------------------------------------------------------------------------------
template <typename T>
void Pseudo<T>::set_cell_average_momentum(int n)
{
    if (n < 0)
        throw_with_line_number("n_cell_average_momentum must be >= 0, got " + std::to_string(n));

    if (n_cell_average_momentum_ != n)
    {
        n_cell_average_momentum_ = n;
        // Recompute all Boltzmann factors with the new setting
        for (const auto& ds_pair : ds_values)
        {
            int ds_idx = ds_pair.first;
            double local_ds = ds_pair.second;

            double saved_ds = this->ds;
            this->ds = local_ds;

            if (is_all_periodic())
                update_boltz_bond_periodic_for_ds_index(ds_idx);
            else
                update_boltz_bond_mixed_for_ds_index(ds_idx);

            this->ds = saved_ds;
        }
    }
}

//------------------------------------------------------------------------------
// Add ds value for pre-computation
//------------------------------------------------------------------------------
template <typename T>
void Pseudo<T>::add_ds_value(int ds_index, double ds_value)
{
    if (ds_index < 1)
        throw_with_line_number("ds_index must be >= 1, got " + std::to_string(ds_index));

    // Store the ds value
    ds_values[ds_index] = ds_value;

    // Allocate arrays for this ds_index if not already present
    if (!boltz_bond.contains(ds_index))
    {
        const int M_COMPLEX = get_total_complex_grid();
        for (const auto& item : bond_lengths)
        {
            std::string monomer_type = item.first;
            boltz_bond[ds_index][monomer_type] = new double[M_COMPLEX];
            boltz_bond_half[ds_index][monomer_type] = new double[M_COMPLEX];
        }
    }
}

//------------------------------------------------------------------------------
// Finalize ds values and compute Boltzmann factors
//------------------------------------------------------------------------------
template <typename T>
void Pseudo<T>::finalize_ds_values()
{
    // Compute Boltzmann factors for all ds_index values
    for (const auto& ds_pair : ds_values)
    {
        int ds_idx = ds_pair.first;
        double local_ds = ds_pair.second;

        // Temporarily set ds to local_ds, compute boltz_bond, then restore
        double saved_ds = this->ds;
        this->ds = local_ds;

        if (is_all_periodic())
        {
            update_boltz_bond_periodic_for_ds_index(ds_idx);
        }
        else
        {
            update_boltz_bond_mixed_for_ds_index(ds_idx);
        }

        this->ds = saved_ds;
    }
}

//------------------------------------------------------------------------------
// Compute cell-averaged bond function at n=1 for a single dimension (periodic BC)
// Used for root-finding sinc correction
// Returns both full-bond and half-bond values
//------------------------------------------------------------------------------
static std::pair<double, double> compute_cell_avg_bond_at_n1_periodic(
    int N,           // grid points in this dimension
    int n_mom,       // momentum truncation
    double prefactor // -b²(2π)²ds/6 * G_dd
)
{
    const double PI = std::numbers::pi;
    double sum_full = 0.0;
    double sum_half = 0.0;

    // n=1 case: compute Σ_m exp(prefactor * (1-2mN)²) * sinc(π(1-2mN)/N)
    for (int m = -n_mom; m <= n_mom; m++)
    {
        int n_shifted = 1 - 2 * m * N;
        double mag_q2 = prefactor * n_shifted * n_shifted;
        double sinc_arg = PI * n_shifted / static_cast<double>(N);
        double sinc_val = sinc(sinc_arg);
        sum_full += std::exp(mag_q2) * sinc_val;
        sum_half += std::exp(mag_q2 / 2.0) * sinc_val;
    }
    return {sum_full, sum_half};
}

//------------------------------------------------------------------------------
// Update Boltzmann factors for a specific ds_index (periodic BC)
//------------------------------------------------------------------------------
template <typename T>
void Pseudo<T>::update_boltz_bond_periodic_for_ds_index(int ds_idx)
{
    const double PI = std::numbers::pi;
    const double FOUR_PI_SQ = 4.0 * PI * PI;
    const int DIM = nx.size();
    double local_ds = ds_values[ds_idx];

    // Pad to 3D for unified loop
    std::vector<int> tnx(3, 1);
    if (DIM == 3)
        tnx = {nx[0], nx[1], nx[2]};
    else if (DIM == 2)
        tnx = {1, nx[0], nx[1]};
    else if (DIM == 1)
        tnx = {1, 1, nx[0]};

    // Extract reciprocal metric components
    double Gii, Gjj, Gkk, Gij, Gik, Gjk;
    if (DIM == 3) {
        Gii = recip_metric_[0]; Gjj = recip_metric_[3]; Gkk = recip_metric_[5];
        Gij = recip_metric_[1]; Gik = recip_metric_[2]; Gjk = recip_metric_[4];
    } else if (DIM == 2) {
        Gii = 0.0; Gjj = recip_metric_[0]; Gkk = recip_metric_[3];
        Gij = 0.0; Gik = 0.0; Gjk = recip_metric_[1];
    } else {
        Gii = 0.0; Gjj = 0.0; Gkk = recip_metric_[0];
        Gij = 0.0; Gik = 0.0; Gjk = 0.0;
    }

    // Determine whether to use multi-momentum summation
    // n_cell_average_momentum_ > 0: use multi-momentum summation (Park et al. 2019, Eq. 30)
    // n_cell_average_momentum_: number of aliased copies to sum (Park et al. 2019, Eq. 30)
    // n=0: single sinc, n>=1: multi-momentum summation
    const int n_mom = n_cell_average_momentum_;

    for (const auto& item : bond_lengths)
    {
        std::string monomer_type = item.first;
        double bond_length_sq = item.second * item.second;
        double* _boltz_bond = boltz_bond[ds_idx][monomer_type];
        double* _boltz_bond_half = boltz_bond_half[ds_idx][monomer_type];

        double prefactor = -bond_length_sq * FOUR_PI_SQ * local_ds / 6.0;

        // Compute sinc correction factors using root-finding approach
        // For each dimension d: sinc_corr_d = target_d - ln(g_ca_d(1))
        // where target_d = ln(ĝ_gaussian(1)) = prefactor * G_dd
        double sinc_corr_i = 0.0, sinc_corr_j = 0.0, sinc_corr_k = 0.0;
        double sinc_corr_half_i = 0.0, sinc_corr_half_j = 0.0, sinc_corr_half_k = 0.0;
        if (use_cell_averaged_bond_)
        {
            // Compute cell-averaged bond at n=1 for each dimension
            // Use prefactor * G_dd as the 1D prefactor for dimension d
            if (tnx[0] > 1) {
                double prefactor_i = prefactor * Gii;
                auto [g_ca_i, g_ca_half_i] = compute_cell_avg_bond_at_n1_periodic(tnx[0], n_mom, prefactor_i);
                double target_i = prefactor_i;  // ln(ĝ_gaussian(1)) = prefactor * Gii * 1²
                double target_half_i = prefactor_i / 2.0;  // half-bond has half the exponent
                sinc_corr_i = target_i - std::log(g_ca_i);
                sinc_corr_half_i = target_half_i - std::log(g_ca_half_i);
            }
            if (tnx[1] > 1) {
                double prefactor_j = prefactor * Gjj;
                auto [g_ca_j, g_ca_half_j] = compute_cell_avg_bond_at_n1_periodic(tnx[1], n_mom, prefactor_j);
                double target_j = prefactor_j;  // ln(ĝ_gaussian(1)) = prefactor * Gjj * 1²
                double target_half_j = prefactor_j / 2.0;
                sinc_corr_j = target_j - std::log(g_ca_j);
                sinc_corr_half_j = target_half_j - std::log(g_ca_half_j);
            }
            if (tnx[2] > 1) {
                double prefactor_k = prefactor * Gkk;
                auto [g_ca_k, g_ca_half_k] = compute_cell_avg_bond_at_n1_periodic(tnx[2], n_mom, prefactor_k);
                double target_k = prefactor_k;  // ln(ĝ_gaussian(1)) = prefactor * Gkk * 1²
                double target_half_k = prefactor_k / 2.0;
                sinc_corr_k = target_k - std::log(g_ca_k);
                sinc_corr_half_k = target_half_k - std::log(g_ca_half_k);
            }
        }

        for (int i = 0; i < tnx[0]; i++)
        {
            // Wavenumber index for i-direction
            int i_signed = (i > tnx[0]/2) ? i - tnx[0] : i;
            int itemp = std::abs(i_signed);

            for (int j = 0; j < tnx[1]; j++)
            {
                // Wavenumber index for j-direction
                int j_signed = (j > tnx[1]/2) ? j - tnx[1] : j;
                int jtemp = std::abs(j_signed);

                if constexpr (std::is_same<T, double>::value)
                {
                    for (int k = 0; k < tnx[2]/2+1; k++)
                    {
                        int ktemp = k;
                        int k_signed = k;
                        int idx = i * tnx[1]*(tnx[2]/2+1) + j*(tnx[2]/2+1) + k;

                        if (use_cell_averaged_bond_)
                        {
                            // Cell-averaged bond function (Park et al. 2019, Eq. 30)
                            // Unified implementation for n_mom = 0 (single sinc) and n_mom >= 1 (multi-momentum)
                            double sum_full = 0.0;
                            double sum_half = 0.0;

                            for (int mi = -n_mom; mi <= n_mom; mi++)
                            {
                                int i_shifted = i_signed - 2 * mi * tnx[0];
                                double sinc_arg_i = PI * i_shifted / static_cast<double>(tnx[0]);

                                for (int mj = -n_mom; mj <= n_mom; mj++)
                                {
                                    int j_shifted = j_signed - 2 * mj * tnx[1];
                                    double sinc_arg_j = PI * j_shifted / static_cast<double>(tnx[1]);

                                    for (int mk = -n_mom; mk <= n_mom; mk++)
                                    {
                                        int k_shifted = k_signed - 2 * mk * tnx[2];
                                        double sinc_arg_k = PI * k_shifted / static_cast<double>(tnx[2]);

                                        double mag_q2 = prefactor * (
                                            Gii * i_shifted * i_shifted +
                                            Gjj * j_shifted * j_shifted +
                                            Gkk * k_shifted * k_shifted +
                                            2.0 * Gij * i_shifted * j_shifted +
                                            2.0 * Gik * i_shifted * k_shifted +
                                            2.0 * Gjk * j_shifted * k_shifted
                                        );

                                        // Sinc filter is a spatial filter (independent of ds)
                                        double sinc_factor = sinc(sinc_arg_i) * sinc(sinc_arg_j) * sinc(sinc_arg_k);

                                        sum_full += std::exp(mag_q2) * sinc_factor;
                                        // Half-bond uses same sinc filter but half the Gaussian exponent
                                        sum_half += std::exp(mag_q2 / 2.0) * sinc_factor;
                                    }
                                }
                            }

                            // Apply end-to-end distance correction as multiplicative factor
                            double sinc_corr = sinc_corr_i * itemp * itemp +
                                               sinc_corr_j * jtemp * jtemp +
                                               sinc_corr_k * ktemp * ktemp;
                            double sinc_corr_half = sinc_corr_half_i * itemp * itemp +
                                                    sinc_corr_half_j * jtemp * jtemp +
                                                    sinc_corr_half_k * ktemp * ktemp;

                            _boltz_bond[idx] = sum_full * std::exp(sinc_corr);
                            _boltz_bond_half[idx] = sum_half * std::exp(sinc_corr_half);
                        }
                        else
                        {
                            // No cell-averaging: standard Gaussian bond
                            double mag_q2 = prefactor * (
                                Gii * itemp * itemp + Gjj * jtemp * jtemp + Gkk * ktemp * ktemp +
                                2.0 * Gij * i_signed * j_signed +
                                2.0 * Gik * i_signed * k_signed +
                                2.0 * Gjk * j_signed * k_signed
                            );

                            _boltz_bond[idx] = std::exp(mag_q2);
                            _boltz_bond_half[idx] = std::exp(mag_q2 / 2.0);
                        }
                    }
                }
                else  // Complex field type
                {
                    for (int k = 0; k < tnx[2]; k++)
                    {
                        int ktemp = (k > tnx[2]/2) ? tnx[2] - k : k;
                        int k_signed = (k > tnx[2]/2) ? k - tnx[2] : k;
                        int idx = i * tnx[1]*tnx[2] + j*tnx[2] + k;

                        if (use_cell_averaged_bond_)
                        {
                            // Cell-averaged bond function (Park et al. 2019, Eq. 30)
                            // Unified implementation for n_mom = 0 (single sinc) and n_mom >= 1 (multi-momentum)
                            double sum_full = 0.0;
                            double sum_half = 0.0;

                            for (int mi = -n_mom; mi <= n_mom; mi++)
                            {
                                int i_shifted = i_signed - 2 * mi * tnx[0];
                                double sinc_arg_i = PI * i_shifted / static_cast<double>(tnx[0]);

                                for (int mj = -n_mom; mj <= n_mom; mj++)
                                {
                                    int j_shifted = j_signed - 2 * mj * tnx[1];
                                    double sinc_arg_j = PI * j_shifted / static_cast<double>(tnx[1]);

                                    for (int mk = -n_mom; mk <= n_mom; mk++)
                                    {
                                        int k_shifted = k_signed - 2 * mk * tnx[2];
                                        double sinc_arg_k = PI * k_shifted / static_cast<double>(tnx[2]);

                                        double mag_q2 = prefactor * (
                                            Gii * i_shifted * i_shifted +
                                            Gjj * j_shifted * j_shifted +
                                            Gkk * k_shifted * k_shifted +
                                            2.0 * Gij * i_shifted * j_shifted +
                                            2.0 * Gik * i_shifted * k_shifted +
                                            2.0 * Gjk * j_shifted * k_shifted
                                        );

                                        // Sinc filter is a spatial filter (independent of ds)
                                        double sinc_factor = sinc(sinc_arg_i) * sinc(sinc_arg_j) * sinc(sinc_arg_k);

                                        sum_full += std::exp(mag_q2) * sinc_factor;
                                        // Half-bond uses same sinc filter but half the Gaussian exponent
                                        sum_half += std::exp(mag_q2 / 2.0) * sinc_factor;
                                    }
                                }
                            }

                            // Apply end-to-end distance correction as multiplicative factor
                            double sinc_corr = sinc_corr_i * itemp * itemp +
                                               sinc_corr_j * jtemp * jtemp +
                                               sinc_corr_k * ktemp * ktemp;
                            double sinc_corr_half = sinc_corr_half_i * itemp * itemp +
                                                    sinc_corr_half_j * jtemp * jtemp +
                                                    sinc_corr_half_k * ktemp * ktemp;

                            _boltz_bond[idx] = sum_full * std::exp(sinc_corr);
                            _boltz_bond_half[idx] = sum_half * std::exp(sinc_corr_half);
                        }
                        else
                        {
                            // No cell-averaging: standard Gaussian bond
                            double mag_q2 = prefactor * (
                                Gii * itemp * itemp + Gjj * jtemp * jtemp + Gkk * ktemp * ktemp +
                                2.0 * Gij * i_signed * j_signed +
                                2.0 * Gik * i_signed * k_signed +
                                2.0 * Gjk * j_signed * k_signed
                            );

                            _boltz_bond[idx] = std::exp(mag_q2);
                            _boltz_bond_half[idx] = std::exp(mag_q2 / 2.0);
                        }
                    }
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
// Compute cell-averaged bond function at n=1 for a single dimension (non-periodic BC)
// Used for root-finding sinc correction
// Returns both full-bond and half-bond values
// For REFLECTING: k_n = πn/L, n = 0, 1, ..., N-1 → first non-zero is n=1
// For ABSORBING: k_n = π(n+1)/L, n = 0, 1, ..., N-1 → first is n=0 (ki=1)
//------------------------------------------------------------------------------
static std::pair<double, double> compute_cell_avg_bond_at_n1_nonperiodic(
    int N,           // grid points in this dimension
    int n_mom,       // momentum truncation
    double prefactor // -b²(π/L)²ds/6
)
{
    const double PI = std::numbers::pi;
    double sum_full = 0.0;
    double sum_half = 0.0;

    // For non-periodic BC, compute at ki=1 (first non-trivial wavenumber)
    // k_shifted = 1 - 2*m*N, sinc_arg = π*k_shifted/(2N)
    for (int m = -n_mom; m <= n_mom; m++)
    {
        int k_shifted = 1 - 2 * m * N;
        double mag_q2 = prefactor * k_shifted * k_shifted;
        double sinc_arg = PI * k_shifted / (2.0 * N);
        double sinc_val = sinc(sinc_arg);
        sum_full += std::exp(mag_q2) * sinc_val;
        sum_half += std::exp(mag_q2 / 2.0) * sinc_val;
    }
    return {sum_full, sum_half};
}

//------------------------------------------------------------------------------
// Update Boltzmann factors for a specific ds_index (mixed BC)
//------------------------------------------------------------------------------
template <typename T>
void Pseudo<T>::update_boltz_bond_mixed_for_ds_index(int ds_idx)
{
    const double PI = std::numbers::pi;
    const int DIM = nx.size();
    double local_ds = ds_values[ds_idx];

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

    // Determine whether to use multi-momentum summation
    // n_cell_average_momentum_: number of aliased copies to sum (Park et al. 2019, Eq. 30)
    // n=0: single sinc, n>=1: multi-momentum summation
    const int n_mom = n_cell_average_momentum_;

    for (const auto& item : bond_lengths)
    {
        std::string monomer_type = item.first;
        double bond_length_sq = item.second * item.second;
        double* _boltz_bond = boltz_bond[ds_idx][monomer_type];
        double* _boltz_bond_half = boltz_bond_half[ds_idx][monomer_type];

        // Compute wavenumber factors based on BC
        double xfactor[3];
        for (int d = 0; d < 3; ++d)
        {
            double L = tnx[d] * tdx[d];
            if (tbc[d] == BoundaryCondition::PERIODIC)
                xfactor[d] = -bond_length_sq * std::pow(2 * PI / L, 2) * local_ds / 6.0;
            else
                xfactor[d] = -bond_length_sq * std::pow(PI / L, 2) * local_ds / 6.0;
        }

        // Compute sinc correction factors using root-finding approach
        // For each dimension d: sinc_corr_d = target_d - ln(g_ca_d(1))
        // where target_d = ln(ĝ_gaussian(1)) = xfactor[d] * 1²
        double sinc_corr[3] = {0.0, 0.0, 0.0};
        double sinc_corr_half[3] = {0.0, 0.0, 0.0};
        if (use_cell_averaged_bond_)
        {
            for (int d = 0; d < 3; ++d)
            {
                if (tnx[d] <= 1) continue;

                double g_ca, g_ca_half;
                if (tbc[d] == BoundaryCondition::PERIODIC) {
                    auto [full, half] = compute_cell_avg_bond_at_n1_periodic(tnx[d], n_mom, xfactor[d]);
                    g_ca = full;
                    g_ca_half = half;
                } else {
                    auto [full, half] = compute_cell_avg_bond_at_n1_nonperiodic(tnx[d], n_mom, xfactor[d]);
                    g_ca = full;
                    g_ca_half = half;
                }

                double target = xfactor[d];  // ln(ĝ_gaussian(1)) = xfactor[d] * 1²
                double target_half = xfactor[d] / 2.0;  // half-bond has half the exponent
                sinc_corr[d] = target - std::log(g_ca);
                sinc_corr_half[d] = target_half - std::log(g_ca_half);
            }
        }

        for (int i = 0; i < tnx[0]; ++i)
        {
            int ki;
            if (tbc[0] == BoundaryCondition::PERIODIC)
                ki = (i > tnx[0]/2) ? tnx[0] - i : i;
            else if (tbc[0] == BoundaryCondition::REFLECTING)
                ki = i;
            else
                ki = i + 1;

            for (int j = 0; j < tnx[1]; ++j)
            {
                int kj;
                if (tbc[1] == BoundaryCondition::PERIODIC)
                    kj = (j > tnx[1]/2) ? tnx[1] - j : j;
                else if (tbc[1] == BoundaryCondition::REFLECTING)
                    kj = j;
                else
                    kj = j + 1;

                for (int k = 0; k < tnx[2]; ++k)
                {
                    int kk;
                    if (tbc[2] == BoundaryCondition::PERIODIC)
                        kk = (k > tnx[2]/2) ? tnx[2] - k : k;
                    else if (tbc[2] == BoundaryCondition::REFLECTING)
                        kk = k;
                    else
                        kk = k + 1;

                    int idx = i * tnx[1] * tnx[2] + j * tnx[2] + k;

                    if (use_cell_averaged_bond_)
                    {
                        // Cell-averaged bond function (Park et al. 2019, Eq. 30)
                        // Unified implementation for n_mom = 0 (single sinc) and n_mom >= 1 (multi-momentum)
                        double sum_full = 0.0;
                        double sum_half = 0.0;

                        for (int mi = -n_mom; mi <= n_mom; mi++)
                        {
                            int ki_shifted;
                            double sinc_arg_i;
                            if (tbc[0] == BoundaryCondition::PERIODIC) {
                                ki_shifted = ki - 2 * mi * tnx[0];
                                sinc_arg_i = PI * ki_shifted / static_cast<double>(tnx[0]);
                            } else {
                                ki_shifted = ki - 2 * mi * tnx[0];
                                sinc_arg_i = PI * ki_shifted / (2.0 * tnx[0]);
                            }

                            for (int mj = -n_mom; mj <= n_mom; mj++)
                            {
                                int kj_shifted;
                                double sinc_arg_j;
                                if (tbc[1] == BoundaryCondition::PERIODIC) {
                                    kj_shifted = kj - 2 * mj * tnx[1];
                                    sinc_arg_j = PI * kj_shifted / static_cast<double>(tnx[1]);
                                } else {
                                    kj_shifted = kj - 2 * mj * tnx[1];
                                    sinc_arg_j = PI * kj_shifted / (2.0 * tnx[1]);
                                }

                                for (int mk = -n_mom; mk <= n_mom; mk++)
                                {
                                    int kk_shifted;
                                    double sinc_arg_k;
                                    if (tbc[2] == BoundaryCondition::PERIODIC) {
                                        kk_shifted = kk - 2 * mk * tnx[2];
                                        sinc_arg_k = PI * kk_shifted / static_cast<double>(tnx[2]);
                                    } else {
                                        kk_shifted = kk - 2 * mk * tnx[2];
                                        sinc_arg_k = PI * kk_shifted / (2.0 * tnx[2]);
                                    }

                                    double mag_q2 = ki_shifted * ki_shifted * xfactor[0] +
                                                    kj_shifted * kj_shifted * xfactor[1] +
                                                    kk_shifted * kk_shifted * xfactor[2];

                                    // Sinc filter is a spatial filter (independent of ds)
                                    double sinc_factor = sinc(sinc_arg_i) * sinc(sinc_arg_j) * sinc(sinc_arg_k);

                                    sum_full += std::exp(mag_q2) * sinc_factor;
                                    // Half-bond uses same sinc filter but half the Gaussian exponent
                                    sum_half += std::exp(mag_q2 / 2.0) * sinc_factor;
                                }
                            }
                        }

                        // Apply end-to-end distance correction as multiplicative factor
                        double sinc_corr_total = sinc_corr[0] * ki * ki +
                                                 sinc_corr[1] * kj * kj +
                                                 sinc_corr[2] * kk * kk;
                        double sinc_corr_half_total = sinc_corr_half[0] * ki * ki +
                                                      sinc_corr_half[1] * kj * kj +
                                                      sinc_corr_half[2] * kk * kk;

                        _boltz_bond[idx] = sum_full * std::exp(sinc_corr_total);
                        _boltz_bond_half[idx] = sum_half * std::exp(sinc_corr_half_total);
                    }
                    else
                    {
                        // No cell-averaging: standard Gaussian bond
                        double mag_q2 = ki*ki*xfactor[0] + kj*kj*xfactor[1] + kk*kk*xfactor[2];

                        _boltz_bond[idx] = std::exp(mag_q2);
                        _boltz_bond_half[idx] = std::exp(mag_q2 / 2.0);
                    }
                }
            }
        }
    }
}

// Explicit template instantiation
template class Pseudo<double>;
template class Pseudo<std::complex<double>>;
