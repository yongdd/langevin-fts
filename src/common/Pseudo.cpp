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
// Constructor
//------------------------------------------------------------------------------
template <typename T>
Pseudo<T>::Pseudo(
    std::map<std::string, double> bond_lengths,
    std::vector<BoundaryCondition> bc,
    std::vector<int> nx, std::vector<double> dx,
    std::array<double, 6> recip_metric,
    std::array<double, 9> recip_vec)
{
    try
    {
        this->bond_lengths = bond_lengths;
        this->bc = bc;
        this->nx = nx;
        this->dx = dx;
        this->ds = 0.0;  // Not used; ds values come from add_ds_value()
        this->recip_metric_ = recip_metric;
        this->recip_vec_ = recip_vec;

        // Compute total grid
        total_grid = 1;
        for (size_t d = 0; d < nx.size(); ++d)
            total_grid *= nx[d];

        update_total_complex_grid();
        const int M_COMPLEX = get_total_complex_grid();

        // Don't allocate Boltzmann factors here; use add_ds_value() instead

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
// Update Fourier basis for periodic BC using v⊗v dyad product
//------------------------------------------------------------------------------
/**
 * @brief Compute v⊗v dyad product components for stress calculation.
 *
 * Uses the deformation vector: v = 2π g⁻¹ m
 * where g⁻¹ is the inverse metric tensor (recip_metric_) and m is Miller index.
 *
 * The deformation vector components are:
 *   v₁ = 2π (G₁₁m₁ + G₁₂m₂ + G₁₃m₃)
 *   v₂ = 2π (G₁₂m₁ + G₂₂m₂ + G₂₃m₃)
 *   v₃ = 2π (G₁₃m₁ + G₂₃m₂ + G₃₃m₃)
 *
 * Stores:
 * - fourier_basis_x = v₁²  (V₁₁)
 * - fourier_basis_y = v₂²  (V₂₂)
 * - fourier_basis_z = v₃²  (V₃₃)
 * - fourier_basis_xy = v₁ × v₂  (V₁₂)
 * - fourier_basis_xz = v₁ × v₃  (V₁₃)
 * - fourier_basis_yz = v₂ × v₃  (V₂₃)
 *
 * @see docs/StressTensorCalculation.md for derivation
 */
template <typename T>
void update_weighted_fourier_basis_periodic_impl(
    double* fourier_basis_x, double* fourier_basis_y, double* fourier_basis_z,
    double* fourier_basis_xy, double* fourier_basis_xz, double* fourier_basis_yz,
    const std::vector<int>& nx,
    const std::array<double, 6>& recip_metric_)
{
    const double PI = std::numbers::pi;
    const double TWO_PI = 2.0 * PI;
    const int DIM = nx.size();

    // Pad to 3D
    std::vector<int> tnx(3, 1);
    if (DIM == 3) {
        tnx = {nx[0], nx[1], nx[2]};
    } else if (DIM == 2) {
        tnx = {1, nx[0], nx[1]};
    } else {
        tnx = {1, 1, nx[0]};
    }

    // Extract inverse metric tensor components g⁻¹
    // recip_metric_ layout: [G₁₁, G₁₂, G₁₃, G₂₂, G₂₃, G₃₃]
    double G11, G22, G33, G12, G13, G23;
    if (DIM == 3) {
        G11 = recip_metric_[0]; G12 = recip_metric_[1]; G13 = recip_metric_[2];
        G22 = recip_metric_[3]; G23 = recip_metric_[4]; G33 = recip_metric_[5];
    } else if (DIM == 2) {
        // 2D: indices map to (1,2) plane
        G11 = recip_metric_[0]; G12 = recip_metric_[1]; G13 = 0.0;
        G22 = recip_metric_[3]; G23 = 0.0; G33 = 0.0;
    } else {
        // 1D: only G₁₁
        G11 = recip_metric_[0]; G12 = 0.0; G13 = 0.0;
        G22 = 0.0; G23 = 0.0; G33 = 0.0;
    }

    for (int i = 0; i < tnx[0]; i++)
    {
        int i_signed = (i > tnx[0]/2) ? i - tnx[0] : i;

        for (int j = 0; j < tnx[1]; j++)
        {
            int j_signed = (j > tnx[1]/2) ? j - tnx[1] : j;

            if constexpr (std::is_same<T, double>::value)
            {
                for (int k = 0; k < tnx[2]/2+1; k++)
                {
                    int k_signed = k;
                    int idx = i * tnx[1]*(tnx[2]/2+1) + j*(tnx[2]/2+1) + k;

                    // Get Miller indices based on dimension
                    int m1, m2, m3;
                    if (DIM == 3) {
                        m1 = i_signed; m2 = j_signed; m3 = k_signed;
                    } else if (DIM == 2) {
                        m1 = j_signed; m2 = k_signed; m3 = 0;
                    } else {
                        m1 = k_signed; m2 = 0; m3 = 0;
                    }

                    // Compute deformation vector v = 2π g⁻¹ m
                    double v1 = TWO_PI * (G11 * m1 + G12 * m2 + G13 * m3);
                    double v2 = TWO_PI * (G12 * m1 + G22 * m2 + G23 * m3);
                    double v3 = TWO_PI * (G13 * m1 + G23 * m2 + G33 * m3);

                    // Store v⊗v components
                    fourier_basis_x[idx] = v1 * v1;
                    fourier_basis_y[idx] = v2 * v2;
                    fourier_basis_z[idx] = v3 * v3;
                    fourier_basis_xy[idx] = v1 * v2;
                    fourier_basis_xz[idx] = v1 * v3;
                    fourier_basis_yz[idx] = v2 * v3;

                    // Weight factor of 2 for interior k modes (r2c symmetry)
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
                    int k_signed = (k > tnx[2]/2) ? k - tnx[2] : k;
                    int idx = i * tnx[1]*tnx[2] + j*tnx[2] + k;

                    // Get Miller indices based on dimension
                    int m1, m2, m3;
                    if (DIM == 3) {
                        m1 = i_signed; m2 = j_signed; m3 = k_signed;
                    } else if (DIM == 2) {
                        m1 = j_signed; m2 = k_signed; m3 = 0;
                    } else {
                        m1 = k_signed; m2 = 0; m3 = 0;
                    }

                    // Compute deformation vector v = 2π g⁻¹ m
                    double v1 = TWO_PI * (G11 * m1 + G12 * m2 + G13 * m3);
                    double v2 = TWO_PI * (G12 * m1 + G22 * m2 + G23 * m3);
                    double v3 = TWO_PI * (G13 * m1 + G23 * m2 + G33 * m3);

                    // Store v⊗v components
                    fourier_basis_x[idx] = v1 * v1;
                    fourier_basis_y[idx] = v2 * v2;
                    fourier_basis_z[idx] = v3 * v3;
                    fourier_basis_xy[idx] = v1 * v2;
                    fourier_basis_xz[idx] = v1 * v3;
                    fourier_basis_yz[idx] = v2 * v3;
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
        nx, recip_metric_);
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

    // Compute deformation vector factors v² = (π g⁻¹ m)²
    // For orthogonal boxes: g⁻¹ = 1/L², so v = πm/L² and v² = (πm)²/L⁴
    // This is consistent with the periodic BC formula where v = 2π g⁻¹ m
    double xfactor[3];
    for (int d = 0; d < 3; ++d)
    {
        double L = tnx[d] * tdx[d];
        if (tbc[d] == BoundaryCondition::PERIODIC)
            xfactor[d] = std::pow(2 * PI / L, 2);
        else
            // Deformation vector: v² = k²/L² = (πm/L)²/L² = (πm)²/L⁴
            xfactor[d] = PI * PI / (L * L * L * L);
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
    std::vector<double> dx,
    std::array<double, 6> recip_metric,
    std::array<double, 9> recip_vec)
{
    this->bond_lengths = bond_lengths;
    this->bc = bc;
    this->dx = dx;
    this->recip_metric_ = recip_metric;
    this->recip_vec_ = recip_vec;

    update_total_complex_grid();
    update_weighted_fourier_basis();

    // Recompute Boltzmann factors for all registered ds values
    for (const auto& ds_pair : ds_values)
    {
        int ds_idx = ds_pair.first;
        if (is_all_periodic())
            update_boltz_bond_periodic_for_ds_index(ds_idx);
        else
            update_boltz_bond_mixed_for_ds_index(ds_idx);
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
// Update Boltzmann factors for a specific ds_index (periodic BC)
//------------------------------------------------------------------------------
template <typename T>
void Pseudo<T>::update_boltz_bond_periodic_for_ds_index(int ds_idx)
{
    const double PI = std::numbers::pi;
    const int DIM = nx.size();
    double local_ds = ds_values[ds_idx];

    // Pad to 3D for unified loop
    std::array<int, 3> tnx = {1, 1, 1};
    if (DIM == 3)      tnx = {nx[0], nx[1], nx[2]};
    else if (DIM == 2) tnx = {1, nx[0], nx[1]};
    else               tnx = {1, 1, nx[0]};

    // Extract reciprocal metric: G = [Gii, Gij, Gik; Gij, Gjj, Gjk; Gik, Gjk, Gkk]
    std::array<double, 3> Gd;  // diagonal: Gii, Gjj, Gkk
    double Gij, Gik, Gjk;      // off-diagonal
    if (DIM == 3) {
        Gd = {recip_metric_[0], recip_metric_[3], recip_metric_[5]};
        Gij = recip_metric_[1]; Gik = recip_metric_[2]; Gjk = recip_metric_[4];
    } else if (DIM == 2) {
        Gd = {0.0, recip_metric_[0], recip_metric_[3]};
        Gij = 0.0; Gik = 0.0; Gjk = recip_metric_[1];
    } else {
        Gd = {0.0, 0.0, recip_metric_[0]};
        Gij = 0.0; Gik = 0.0; Gjk = 0.0;
    }

    for (const auto& [monomer_type, bond_length] : bond_lengths)
    {
        double bond_length_sq = bond_length * bond_length;
        double* _boltz_bond = boltz_bond[ds_idx][monomer_type];
        double* _boltz_bond_half = boltz_bond_half[ds_idx][monomer_type];
        double prefactor = -bond_length_sq * 4.0 * PI * PI * local_ds / 6.0;

        for (int i = 0; i < tnx[0]; i++)
        {
            int i_signed = (i > tnx[0]/2) ? i - tnx[0] : i;
            int ni = std::abs(i_signed);

            for (int j = 0; j < tnx[1]; j++)
            {
                int j_signed = (j > tnx[1]/2) ? j - tnx[1] : j;
                int nj = std::abs(j_signed);

                if constexpr (std::is_same<T, double>::value)
                {
                    for (int k = 0; k < tnx[2]/2+1; k++)
                    {
                        int idx = i * tnx[1]*(tnx[2]/2+1) + j*(tnx[2]/2+1) + k;
                        double mag_q2 = prefactor * (Gd[0]*ni*ni + Gd[1]*nj*nj + Gd[2]*k*k +
                            2.0*(Gij*i_signed*j_signed + Gik*i_signed*k + Gjk*j_signed*k));
                        _boltz_bond[idx] = std::exp(mag_q2);
                        _boltz_bond_half[idx] = std::exp(mag_q2 / 2.0);
                    }
                }
                else  // Complex field type
                {
                    for (int k = 0; k < tnx[2]; k++)
                    {
                        int k_signed = (k > tnx[2]/2) ? k - tnx[2] : k;
                        int nk = std::abs(k_signed);
                        int idx = i * tnx[1]*tnx[2] + j*tnx[2] + k;
                        double mag_q2 = prefactor * (Gd[0]*ni*ni + Gd[1]*nj*nj + Gd[2]*nk*nk +
                            2.0*(Gij*i_signed*j_signed + Gik*i_signed*k_signed + Gjk*j_signed*k_signed));
                        _boltz_bond[idx] = std::exp(mag_q2);
                        _boltz_bond_half[idx] = std::exp(mag_q2 / 2.0);
                    }
                }
            }
        }
    }
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
    std::array<int, 3> tnx = {1, 1, 1};
    std::array<double, 3> tdx = {1.0, 1.0, 1.0};
    std::array<BoundaryCondition, 3> tbc = {BoundaryCondition::PERIODIC,
                                             BoundaryCondition::PERIODIC,
                                             BoundaryCondition::PERIODIC};
    for (int d = 0; d < DIM; ++d) {
        tnx[3 - DIM + d] = nx[d];
        tdx[3 - DIM + d] = dx[d];
        tbc[3 - DIM + d] = bc[d];
    }

    // Wavenumber index based on BC
    auto get_ki = [](int idx, int N, BoundaryCondition bc_type) {
        if (bc_type == BoundaryCondition::PERIODIC)
            return (idx > N/2) ? N - idx : idx;
        else if (bc_type == BoundaryCondition::REFLECTING)
            return idx;
        else  // ABSORBING
            return idx + 1;
    };

    for (const auto& [monomer_type, bond_length] : bond_lengths)
    {
        double bond_length_sq = bond_length * bond_length;
        double* _boltz_bond = boltz_bond[ds_idx][monomer_type];
        double* _boltz_bond_half = boltz_bond_half[ds_idx][monomer_type];

        // Compute prefactors based on BC
        std::array<double, 3> xfactor;
        for (int d = 0; d < 3; ++d) {
            double L = tnx[d] * tdx[d];
            double k_scale = (tbc[d] == BoundaryCondition::PERIODIC) ? 2*PI/L : PI/L;
            xfactor[d] = -bond_length_sq * k_scale * k_scale * local_ds / 6.0;
        }

        for (int i = 0; i < tnx[0]; ++i) {
            int ki = get_ki(i, tnx[0], tbc[0]);
            for (int j = 0; j < tnx[1]; ++j) {
                int kj = get_ki(j, tnx[1], tbc[1]);
                for (int k = 0; k < tnx[2]; ++k) {
                    int kk = get_ki(k, tnx[2], tbc[2]);
                    int idx = i*tnx[1]*tnx[2] + j*tnx[2] + k;

                    double mag_q2 = ki*ki*xfactor[0] + kj*kj*xfactor[1] + kk*kk*xfactor[2];
                    _boltz_bond[idx] = std::exp(mag_q2);
                    _boltz_bond_half[idx] = std::exp(mag_q2/2.0);
                }
            }
        }
    }
}

// Explicit template instantiation
template class Pseudo<double>;
template class Pseudo<std::complex<double>>;
