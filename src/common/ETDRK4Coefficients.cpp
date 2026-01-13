/**
 * @file ETDRK4Coefficients.cpp
 * @brief Implementation of ETDRK4 coefficient computation (Krogstad scheme).
 *
 * Implements the Kassam-Trefethen (2005) contour integral method for
 * stable computation of ETDRK4 coefficients, especially for small eigenvalues
 * where direct formulas suffer from catastrophic cancellation.
 *
 * Uses the Krogstad scheme as described in Song et al. (2018).
 *
 * @see ETDRK4Coefficients.h for class documentation
 */

#include <iostream>
#include <cmath>
#include <numbers>
#include <complex>
#include "ETDRK4Coefficients.h"

//------------------------------------------------------------------------------
// Phi Function Utilities for Krogstad ETDRK4 (Kassam-Trefethen 2005)
//------------------------------------------------------------------------------

namespace {

/**
 * @brief Compute phi_1(z) = (exp(z) - 1) / z with Taylor series for small z.
 *
 * Limit as z->0: phi_1 -> 1
 */
std::complex<double> phi1_stable(std::complex<double> z) {
    if (std::abs(z) < 1e-4) {
        // Taylor series: 1 + z/2 + z^2/6 + z^3/24 + z^4/120
        return 1.0 + z/2.0 + z*z/6.0 + z*z*z/24.0 + z*z*z*z/120.0;
    }
    return (std::exp(z) - 1.0) / z;
}

/**
 * @brief Compute phi_2(z) = (exp(z) - 1 - z) / z^2 with Taylor series for small z.
 *
 * Limit as z->0: phi_2 -> 1/2
 */
std::complex<double> phi2_stable(std::complex<double> z) {
    if (std::abs(z) < 1e-4) {
        // Taylor series: 1/2 + z/6 + z^2/24 + z^3/120 + z^4/720
        return 1.0/2.0 + z/6.0 + z*z/24.0 + z*z*z/120.0 + z*z*z*z/720.0;
    }
    return (std::exp(z) - 1.0 - z) / (z*z);
}

/**
 * @brief Compute phi_3(z) = (exp(z) - 1 - z - z^2/2) / z^3 with Taylor series for small z.
 *
 * Limit as z->0: phi_3 -> 1/6
 */
std::complex<double> phi3_stable(std::complex<double> z) {
    if (std::abs(z) < 1e-4) {
        // Taylor series: 1/6 + z/24 + z^2/120 + z^3/720 + z^4/5040
        return 1.0/6.0 + z/24.0 + z*z/120.0 + z*z*z/720.0 + z*z*z*z/5040.0;
    }
    return (std::exp(z) - 1.0 - z - z*z/2.0) / (z*z*z);
}

/**
 * @brief Compute phi_n using Kassam-Trefethen contour integral.
 *
 * @param n Phi function index (1, 2, or 3)
 * @param z Real argument (c*h where c is eigenvalue, h is step size)
 * @param M Number of contour points (default 32)
 * @param r Contour radius (default 1.0)
 * @return phi_n(z) computed via contour integral
 */
double phi_contour(int n, double z, int M = 32, double r = 1.0) {
    const double PI = std::numbers::pi;
    std::complex<double> sum = 0.0;

    for (int m = 0; m < M; m++) {
        double theta = 2.0 * PI * m / M;
        std::complex<double> zeta = z + r * std::exp(std::complex<double>(0.0, theta));

        if (n == 1)
            sum += phi1_stable(zeta);
        else if (n == 2)
            sum += phi2_stable(zeta);
        else if (n == 3)
            sum += phi3_stable(zeta);
    }

    return std::real(sum) / M;
}

} // anonymous namespace

//------------------------------------------------------------------------------
// Constructor
//------------------------------------------------------------------------------
template <typename T>
ETDRK4Coefficients<T>::ETDRK4Coefficients(
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

        compute_total_complex_grid();

        // Allocate coefficient arrays (Krogstad scheme)
        for (const auto& item : bond_lengths)
        {
            std::string monomer_type = item.first;
            E_[monomer_type] = new double[total_complex_grid];
            E2_[monomer_type] = new double[total_complex_grid];
            alpha_[monomer_type] = new double[total_complex_grid];
            phi2_half_[monomer_type] = new double[total_complex_grid];
            phi1_[monomer_type] = new double[total_complex_grid];
            phi2_[monomer_type] = new double[total_complex_grid];
            phi3_[monomer_type] = new double[total_complex_grid];
        }

        // Compute coefficients
        if (is_all_periodic())
            compute_coefficients_periodic();
        else
            compute_coefficients_mixed();
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
ETDRK4Coefficients<T>::~ETDRK4Coefficients()
{
    for (const auto& item : E_)
        delete[] item.second;
    for (const auto& item : E2_)
        delete[] item.second;
    for (const auto& item : alpha_)
        delete[] item.second;
    for (const auto& item : phi2_half_)
        delete[] item.second;
    for (const auto& item : phi1_)
        delete[] item.second;
    for (const auto& item : phi2_)
        delete[] item.second;
    for (const auto& item : phi3_)
        delete[] item.second;
}

//------------------------------------------------------------------------------
// Check if all BCs are periodic
//------------------------------------------------------------------------------
template <typename T>
bool ETDRK4Coefficients<T>::is_all_periodic() const
{
    for (const auto& b : bc)
    {
        if (b != BoundaryCondition::PERIODIC)
            return false;
    }
    return true;
}

//------------------------------------------------------------------------------
// Compute total complex grid size
//------------------------------------------------------------------------------
template <typename T>
void ETDRK4Coefficients<T>::compute_total_complex_grid()
{
    int total_grid = 1;
    for (size_t d = 0; d < nx.size(); ++d)
        total_grid *= nx[d];

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
// Getters
//------------------------------------------------------------------------------
template <typename T>
const double* ETDRK4Coefficients<T>::get_E(const std::string& monomer_type) const
{
    auto it = E_.find(monomer_type);
    if (it == E_.end())
        throw_with_line_number("Monomer type \"" + monomer_type + "\" not found.");
    return it->second;
}

template <typename T>
const double* ETDRK4Coefficients<T>::get_E2(const std::string& monomer_type) const
{
    auto it = E2_.find(monomer_type);
    if (it == E2_.end())
        throw_with_line_number("Monomer type \"" + monomer_type + "\" not found.");
    return it->second;
}

template <typename T>
const double* ETDRK4Coefficients<T>::get_alpha(const std::string& monomer_type) const
{
    auto it = alpha_.find(monomer_type);
    if (it == alpha_.end())
        throw_with_line_number("Monomer type \"" + monomer_type + "\" not found.");
    return it->second;
}

template <typename T>
const double* ETDRK4Coefficients<T>::get_phi2_half(const std::string& monomer_type) const
{
    auto it = phi2_half_.find(monomer_type);
    if (it == phi2_half_.end())
        throw_with_line_number("Monomer type \"" + monomer_type + "\" not found.");
    return it->second;
}

template <typename T>
const double* ETDRK4Coefficients<T>::get_phi1(const std::string& monomer_type) const
{
    auto it = phi1_.find(monomer_type);
    if (it == phi1_.end())
        throw_with_line_number("Monomer type \"" + monomer_type + "\" not found.");
    return it->second;
}

template <typename T>
const double* ETDRK4Coefficients<T>::get_phi2(const std::string& monomer_type) const
{
    auto it = phi2_.find(monomer_type);
    if (it == phi2_.end())
        throw_with_line_number("Monomer type \"" + monomer_type + "\" not found.");
    return it->second;
}

template <typename T>
const double* ETDRK4Coefficients<T>::get_phi3(const std::string& monomer_type) const
{
    auto it = phi3_.find(monomer_type);
    if (it == phi3_.end())
        throw_with_line_number("Monomer type \"" + monomer_type + "\" not found.");
    return it->second;
}

//------------------------------------------------------------------------------
// Compute coefficients for periodic BC
//------------------------------------------------------------------------------
template <typename T>
void ETDRK4Coefficients<T>::compute_coefficients_periodic()
{
    const double PI = std::numbers::pi;
    const double FOUR_PI_SQ = 4.0 * PI * PI;
    const int DIM = nx.size();

    // Pad to 3D
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

        double* _E = E_[monomer_type];
        double* _E2 = E2_[monomer_type];
        double* _alpha = alpha_[monomer_type];
        double* _phi2_half = phi2_half_[monomer_type];
        double* _phi1 = phi1_[monomer_type];
        double* _phi2 = phi2_[monomer_type];
        double* _phi3 = phi3_[monomer_type];

        // c_prefactor: multiply by k^2 to get eigenvalue c = -b^2*k^2/6
        double c_prefactor = -bond_length_sq * FOUR_PI_SQ / 6.0;

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

                        // Compute |k|^2 with metric
                        double k2 = Gii * itemp * itemp + Gjj * jtemp * jtemp + Gkk * ktemp * ktemp +
                                    2.0 * Gij * i_signed * j_signed +
                                    2.0 * Gik * i_signed * k_signed +
                                    2.0 * Gjk * j_signed * k_signed;

                        // Eigenvalue c and arguments
                        double c = c_prefactor * k2;
                        double ch = c * ds;      // c*h for full step
                        double ch2 = ch / 2.0;   // c*h/2 for half step

                        // Krogstad ETDRK4 coefficients (Song et al. 2018)
                        // E = exp(c*h), E2 = exp(c*h/2)
                        _E[idx] = std::exp(ch);
                        _E2[idx] = std::exp(ch2);

                        // alpha = (h/2) * phi_1(c*h/2) for stage a
                        _alpha[idx] = (ds / 2.0) * phi_contour(1, ch2);

                        // phi2_half = h * phi_2(c*h/2) for stage b
                        _phi2_half[idx] = ds * phi_contour(2, ch2);

                        // phi1 = h * phi_1(c*h) for stage c
                        _phi1[idx] = ds * phi_contour(1, ch);

                        // phi2 = h * phi_2(c*h) for stages c and final
                        _phi2[idx] = ds * phi_contour(2, ch);

                        // phi3 = h * phi_3(c*h) for final step
                        _phi3[idx] = ds * phi_contour(3, ch);
                    }
                }
                else
                {
                    for (int k = 0; k < tnx[2]; k++)
                    {
                        int ktemp = (k > tnx[2]/2) ? tnx[2] - k : k;
                        int k_signed = (k > tnx[2]/2) ? k - tnx[2] : k;
                        int idx = i * tnx[1]*tnx[2] + j*tnx[2] + k;

                        double k2 = Gii * itemp * itemp + Gjj * jtemp * jtemp + Gkk * ktemp * ktemp +
                                    2.0 * Gij * i_signed * j_signed +
                                    2.0 * Gik * i_signed * k_signed +
                                    2.0 * Gjk * j_signed * k_signed;

                        double c = c_prefactor * k2;
                        double ch = c * ds;
                        double ch2 = ch / 2.0;

                        // Krogstad ETDRK4 coefficients
                        _E[idx] = std::exp(ch);
                        _E2[idx] = std::exp(ch2);
                        _alpha[idx] = (ds / 2.0) * phi_contour(1, ch2);
                        _phi2_half[idx] = ds * phi_contour(2, ch2);
                        _phi1[idx] = ds * phi_contour(1, ch);
                        _phi2[idx] = ds * phi_contour(2, ch);
                        _phi3[idx] = ds * phi_contour(3, ch);
                    }
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
// Compute coefficients for mixed BC
//------------------------------------------------------------------------------
template <typename T>
void ETDRK4Coefficients<T>::compute_coefficients_mixed()
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

    for (const auto& item : bond_lengths)
    {
        std::string monomer_type = item.first;
        double bond_length_sq = item.second * item.second;

        double* _E = E_[monomer_type];
        double* _E2 = E2_[monomer_type];
        double* _alpha = alpha_[monomer_type];
        double* _phi2_half = phi2_half_[monomer_type];
        double* _phi1 = phi1_[monomer_type];
        double* _phi2 = phi2_[monomer_type];
        double* _phi3 = phi3_[monomer_type];

        // Compute wavenumber factors based on BC
        // c_factor[d] = -(b^2/6) * (k_factor[d])^2
        double c_factor[3];
        for (int d = 0; d < 3; ++d)
        {
            double L = tnx[d] * tdx[d];
            if (tbc[d] == BoundaryCondition::PERIODIC)
                c_factor[d] = -bond_length_sq * std::pow(2 * PI / L, 2) / 6.0;
            else
                c_factor[d] = -bond_length_sq * std::pow(PI / L, 2) / 6.0;
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

                    // Eigenvalue c = sum of c_factor[d] * k[d]^2
                    double c = ki*ki*c_factor[0] + kj*kj*c_factor[1] + kk*kk*c_factor[2];
                    double ch = c * ds;
                    double ch2 = ch / 2.0;

                    // Krogstad ETDRK4 coefficients (Song et al. 2018)
                    _E[idx] = std::exp(ch);
                    _E2[idx] = std::exp(ch2);
                    _alpha[idx] = (ds / 2.0) * phi_contour(1, ch2);
                    _phi2_half[idx] = ds * phi_contour(2, ch2);
                    _phi1[idx] = ds * phi_contour(1, ch);
                    _phi2[idx] = ds * phi_contour(2, ch);
                    _phi3[idx] = ds * phi_contour(3, ch);
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
// Update coefficients
//------------------------------------------------------------------------------
template <typename T>
void ETDRK4Coefficients<T>::update(
    std::vector<BoundaryCondition> bc,
    std::map<std::string, double> bond_lengths,
    std::vector<double> dx, double ds,
    std::array<double, 6> recip_metric)
{
    this->bc = bc;
    this->bond_lengths = bond_lengths;
    this->dx = dx;
    this->ds = ds;
    this->recip_metric_ = recip_metric;

    compute_total_complex_grid();

    if (is_all_periodic())
        compute_coefficients_periodic();
    else
        compute_coefficients_mixed();
}

// Explicit template instantiation
template class ETDRK4Coefficients<double>;
template class ETDRK4Coefficients<std::complex<double>>;
