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
namespace {
/**
 * @brief Replace Miller indices by the alias representative minimizing the
 *        reciprocal-metric form (first-Brillouin-zone representative).
 *
 * On the discrete torus each mode k is only defined modulo N, and the
 * multiplier must pick one alias representative. The componentwise
 * principal-zone choice (|m_d| <= N_d/2) is NOT invariant under point-group
 * operations that mix axes with a nonzero metric cross term: e.g. the
 * hexagonal rotation (kx,ky) -> (kx+ky,-kx) can push kx+ky out of the zone,
 * and re-aliasing changes kx^2 + kx*ky + ky^2. The minimal-form
 * representative is group-invariant: symmetry operations permute alias
 * classes and preserve the form, so the minimum over the class is preserved.
 * This makes the pseudo-spectral evolution operator exactly symmetric under
 * the crystal point group (required for space-group reduced-basis
 * computations to match full-grid computations).
 *
 * For orthogonal cells the form is separable and the componentwise
 * representative is already minimal, so callers only invoke this for cells
 * with a nonzero off-diagonal metric (keeping orthogonal results bit-exact).
 * A search window of +/-1 period per axis is sufficient for valid cell
 * geometries (the Voronoi cell of a reduced lattice basis is contained in
 * the union of neighboring translates).
 */
inline void min_form_alias_rep(
    int& m1, int& m2, int& m3,
    const std::array<int, 3>& tnx,
    double G11, double G22, double G33,
    double G12, double G13, double G23)
{
    auto form = [&](double a, double b, double c) {
        return G11 * a * a + G22 * b * b + G33 * c * c +
               2.0 * (G12 * a * b + G13 * a * c + G23 * b * c);
    };

    const int s1 = (tnx[0] > 1) ? 1 : 0;
    const int s2 = (tnx[1] > 1) ? 1 : 0;
    const int s3 = (tnx[2] > 1) ? 1 : 0;

    int b1 = m1, b2 = m2, b3 = m3;
    double best = form(m1, m2, m3);
    for (int a = -s1; a <= s1; ++a)
    {
        for (int b = -s2; b <= s2; ++b)
        {
            for (int c = -s3; c <= s3; ++c)
            {
                if (a == 0 && b == 0 && c == 0)
                    continue;
                const int t1 = m1 + a * tnx[0];
                const int t2 = m2 + b * tnx[1];
                const int t3 = m3 + c * tnx[2];
                const double f = form(t1, t2, t3);
                if (f < best)
                {
                    best = f;
                    b1 = t1; b2 = t2; b3 = t3;
                }
            }
        }
    }
    m1 = b1; m2 = b2; m3 = b3;
}
}  // namespace

// (A duplicated free-function implementation of the periodic Boltzmann
// table builder used to live here; it had no callers and risked drifting
// from the live member implementation, so it was removed.)

template <typename T>
void Pseudo<T>::update_boltz_bond_periodic()
{
    // Recompute for every registered ds value (writing a hard-coded index
    // would default-insert a nullptr entry when that index was never
    // registered via add_ds_value()).
    for (const auto& ds_pair : ds_values)
        update_boltz_bond_periodic_for_ds_index(ds_pair.first);
}

//------------------------------------------------------------------------------
// Update Boltzmann factors for mixed BC
//------------------------------------------------------------------------------
template <typename T>
void Pseudo<T>::update_boltz_bond_mixed()
{
    // Recompute for every registered ds value (see update_boltz_bond_periodic).
    for (const auto& ds_pair : ds_values)
        update_boltz_bond_mixed_for_ds_index(ds_pair.first);
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
 * @see docs/theory/StressTensor.md for derivation
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

    // Oblique cells need the group-invariant (minimal-form) alias
    // representative, consistent with the diffusion Boltzmann factors;
    // see min_form_alias_rep(). Note the search runs in loop-index space
    // (i,j,k with padded leading dims), before mapping to Miller indices.
    // Relative-tolerance gate: exact 90-degree angles produce cos(pi/2)
    // ~ 6e-17, which must not flip orthogonal cells onto this path.
    const double G_diag_max2 = std::max({std::abs(G11), std::abs(G22), std::abs(G33)});
    const bool oblique = (std::abs(G12) > 1e-12 * G_diag_max2 ||
                          std::abs(G13) > 1e-12 * G_diag_max2 ||
                          std::abs(G23) > 1e-12 * G_diag_max2);
    const std::array<int, 3> tnx3 = {tnx[0], tnx[1], tnx[2]};
    auto to_min_form_rep = [&](int& i_s, int& j_s, int& k_s) {
        // recip metric in loop-index space: for DIM==3 (i,j,k)=(m1,m2,m3);
        // DIM==2 (j,k)=(m1,m2); DIM==1 k=m1.
        if (DIM == 3)
            min_form_alias_rep(i_s, j_s, k_s, tnx3, G11, G22, G33, G12, G13, G23);
        else if (DIM == 2)
        {
            int dummy = 0;
            min_form_alias_rep(dummy, j_s, k_s, tnx3, 0.0, G11, G22, 0.0, 0.0, G12);
        }
        // DIM==1: separable, componentwise representative already minimal
    };

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
                    int i_rep = i_signed, j_rep = j_signed, k_rep = k_signed;
                    if (oblique)
                        to_min_form_rep(i_rep, j_rep, k_rep);
                    if (DIM == 3) {
                        m1 = i_rep; m2 = j_rep; m3 = k_rep;
                    } else if (DIM == 2) {
                        m1 = j_rep; m2 = k_rep; m3 = 0;
                    } else {
                        m1 = k_rep; m2 = 0; m3 = 0;
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
                    int i_rep = i_signed, j_rep = j_signed, k_rep = k_signed;
                    if (oblique)
                        to_min_form_rep(i_rep, j_rep, k_rep);
                    if (DIM == 3) {
                        m1 = i_rep; m2 = j_rep; m3 = k_rep;
                    } else if (DIM == 2) {
                        m1 = j_rep; m2 = k_rep; m3 = 0;
                    } else {
                        m1 = k_rep; m2 = 0; m3 = 0;
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

    // Compute deformation vector factors v² (units 1/L⁴), consistent with the
    // periodic-BC path where v = 2π g⁻¹ m and g⁻¹ = 1/L² for orthogonal boxes:
    //   Periodic:      k = 2πm/L → v² = k²/L² = (2πm)²/L⁴
    //   Reflecting:    k = πm/L  → v² = k²/L² = (πm)²/L⁴
    //   Absorbing:     k = π(m+1)/L, same scaling
    // NOTE: stress with non-periodic BCs is currently rejected by
    // compute_stress() in the computation classes; this path is kept
    // unit-consistent for when that support is added.
    double xfactor[3];
    for (int d = 0; d < 3; ++d)
    {
        double L = tnx[d] * tdx[d];
        if (tbc[d] == BoundaryCondition::PERIODIC)
            xfactor[d] = std::pow(2 * PI, 2) / (L * L * L * L);
        else
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
    if (ds_index < 0)
        throw_with_line_number("ds_index must be >= 0, got " + std::to_string(ds_index));

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

    // Oblique cells need the group-invariant (minimal-form) alias
    // representative; see min_form_alias_rep().
    // Treat the cell as oblique only when the off-diagonal metric is
    // significant relative to the diagonal: angles of exactly 90 degrees
    // produce cos(pi/2) ~ 6e-17, which must not flip orthogonal cells
    // onto the minimal-form alias path (bit-identical behavior there).
    const double G_diag_max = std::max({std::abs(Gd[0]), std::abs(Gd[1]), std::abs(Gd[2])});
    const bool oblique = (std::abs(Gij) > 1e-12 * G_diag_max ||
                          std::abs(Gik) > 1e-12 * G_diag_max ||
                          std::abs(Gjk) > 1e-12 * G_diag_max);

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
                        double mag_q2;
                        if (oblique)
                        {
                            int m1 = i_signed, m2 = j_signed, m3 = k;
                            min_form_alias_rep(m1, m2, m3, tnx, Gd[0], Gd[1], Gd[2], Gij, Gik, Gjk);
                            mag_q2 = prefactor * (Gd[0]*m1*m1 + Gd[1]*m2*m2 + Gd[2]*m3*m3 +
                                2.0*(Gij*m1*m2 + Gik*m1*m3 + Gjk*m2*m3));
                        }
                        else
                        {
                            mag_q2 = prefactor * (Gd[0]*ni*ni + Gd[1]*nj*nj + Gd[2]*k*k +
                                2.0*(Gij*i_signed*j_signed + Gik*i_signed*k + Gjk*j_signed*k));
                        }
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
                        double mag_q2;
                        if (oblique)
                        {
                            int m1 = i_signed, m2 = j_signed, m3 = k_signed;
                            min_form_alias_rep(m1, m2, m3, tnx, Gd[0], Gd[1], Gd[2], Gij, Gik, Gjk);
                            mag_q2 = prefactor * (Gd[0]*m1*m1 + Gd[1]*m2*m2 + Gd[2]*m3*m3 +
                                2.0*(Gij*m1*m2 + Gik*m1*m3 + Gjk*m2*m3));
                        }
                        else
                        {
                            mag_q2 = prefactor * (Gd[0]*ni*ni + Gd[1]*nj*nj + Gd[2]*nk*nk +
                                2.0*(Gij*i_signed*j_signed + Gik*i_signed*k_signed + Gjk*j_signed*k_signed));
                        }
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
