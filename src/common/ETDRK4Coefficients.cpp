/**
 * @file ETDRK4Coefficients.cpp
 * @brief Implementation of ETDRK4 coefficient computation.
 *
 * Implements the Kassam-Trefethen (2005) contour integral method for
 * stable computation of ETDRK4 coefficients, especially for small eigenvalues
 * where direct formulas suffer from catastrophic cancellation.
 *
 * @see ETDRK4Coefficients.h for class documentation
 */

#include <iostream>
#include <cmath>
#include <numbers>
#include <complex>
#include "ETDRK4Coefficients.h"

//------------------------------------------------------------------------------
// Phi and Beta Function Utilities (Kassam-Trefethen 2005)
//------------------------------------------------------------------------------

namespace {

/**
 * @brief Compute phi_1(z) = (exp(z) - 1) / z with Taylor series for small z.
 */
std::complex<double> phi1_stable(std::complex<double> z) {
    if (std::abs(z) < 1e-4) {
        // Taylor series: 1 + z/2 + z^2/6 + z^3/24 + z^4/120
        return 1.0 + z/2.0 + z*z/6.0 + z*z*z/24.0 + z*z*z*z/120.0;
    }
    return (std::exp(z) - 1.0) / z;
}

/**
 * @brief ETDRK4 coefficient beta_1 = (-4 - z + e^z(4 - 3z + z^2))/z^3.
 *
 * Coefficient for N_n in the final ETDRK4 combination step.
 * Limit as z->0: beta_1 -> 1/6
 */
std::complex<double> beta1_stable(std::complex<double> z) {
    if (std::abs(z) < 1e-4) {
        // Taylor series: 1/6 + z/6 + 3z^2/40 + z^3/45 + 5z^4/1008 + ...
        return 1.0/6.0 + z/6.0 + 3.0*z*z/40.0 + z*z*z/45.0 + 5.0*z*z*z*z/1008.0;
    }
    std::complex<double> ez = std::exp(z);
    return (-4.0 - z + ez*(4.0 - 3.0*z + z*z)) / (z*z*z);
}

/**
 * @brief ETDRK4 coefficient beta_2 = 2(2 + z + e^z(-2 + z))/z^3.
 *
 * Coefficient for (N_a + N_b) in the final ETDRK4 combination step.
 * Limit as z->0: beta_2 -> 2/3
 */
std::complex<double> beta2_stable(std::complex<double> z) {
    if (std::abs(z) < 1e-4) {
        // Taylor series: 2/3 + z/3 + z^2/10 + z^3/36 + z^4/168 + ...
        return 2.0/3.0 + z/3.0 + z*z/10.0 + z*z*z/36.0 + z*z*z*z/168.0;
    }
    std::complex<double> ez = std::exp(z);
    return 2.0*(2.0 + z + ez*(-2.0 + z)) / (z*z*z);
}

/**
 * @brief ETDRK4 coefficient beta_3 = (-4 - 3z - z^2 + e^z(4 - z))/z^3.
 *
 * Coefficient for N_c in the final ETDRK4 combination step.
 * Limit as z->0: beta_3 -> 1/6
 */
std::complex<double> beta3_stable(std::complex<double> z) {
    if (std::abs(z) < 1e-4) {
        // Taylor series: 1/6 + z/6 + z^2/20 + z^3/90 + z^4/504 + ...
        return 1.0/6.0 + z/6.0 + z*z/20.0 + z*z*z/90.0 + z*z*z*z/504.0;
    }
    std::complex<double> ez = std::exp(z);
    return (-4.0 - 3.0*z - z*z + ez*(4.0 - z)) / (z*z*z);
}

/**
 * @brief Compute phi_1 using Kassam-Trefethen contour integral.
 *
 * @param z Real argument
 * @param M Number of contour points (default 32)
 * @param r Contour radius (default 1.0)
 * @return phi_1(z) computed via contour integral
 */
double phi1_contour(double z, int M = 32, double r = 1.0) {
    const double PI = std::numbers::pi;
    std::complex<double> sum = 0.0;

    for (int m = 0; m < M; m++) {
        double theta = 2.0 * PI * m / M;
        std::complex<double> zeta = z + r * std::exp(std::complex<double>(0.0, theta));
        sum += phi1_stable(zeta);
    }

    return std::real(sum) / M;
}

/**
 * @brief Compute beta_n using Kassam-Trefethen contour integral.
 *
 * @param n Beta function index (1, 2, or 3)
 * @param z Real argument (c*h where c is eigenvalue, h is step size)
 * @param M Number of contour points (default 32)
 * @param r Contour radius (default 1.0)
 * @return beta_n(z) computed via contour integral
 */
double beta_contour(int n, double z, int M = 32, double r = 1.0) {
    const double PI = std::numbers::pi;
    std::complex<double> sum = 0.0;

    for (int m = 0; m < M; m++) {
        double theta = 2.0 * PI * m / M;
        std::complex<double> zeta = z + r * std::exp(std::complex<double>(0.0, theta));

        if (n == 1)
            sum += beta1_stable(zeta);
        else if (n == 2)
            sum += beta2_stable(zeta);
        else if (n == 3)
            sum += beta3_stable(zeta);
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

        // Allocate coefficient arrays
        for (const auto& item : bond_lengths)
        {
            std::string monomer_type = item.first;
            E_[monomer_type] = new double[total_complex_grid];
            E2_[monomer_type] = new double[total_complex_grid];
            alpha_[monomer_type] = new double[total_complex_grid];
            f1_[monomer_type] = new double[total_complex_grid];
            f2_[monomer_type] = new double[total_complex_grid];
            f3_[monomer_type] = new double[total_complex_grid];
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
    for (const auto& item : f1_)
        delete[] item.second;
    for (const auto& item : f2_)
        delete[] item.second;
    for (const auto& item : f3_)
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
const double* ETDRK4Coefficients<T>::get_f1(const std::string& monomer_type) const
{
    auto it = f1_.find(monomer_type);
    if (it == f1_.end())
        throw_with_line_number("Monomer type \"" + monomer_type + "\" not found.");
    return it->second;
}

template <typename T>
const double* ETDRK4Coefficients<T>::get_f2(const std::string& monomer_type) const
{
    auto it = f2_.find(monomer_type);
    if (it == f2_.end())
        throw_with_line_number("Monomer type \"" + monomer_type + "\" not found.");
    return it->second;
}

template <typename T>
const double* ETDRK4Coefficients<T>::get_f3(const std::string& monomer_type) const
{
    auto it = f3_.find(monomer_type);
    if (it == f3_.end())
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
        double* _f1 = f1_[monomer_type];
        double* _f2 = f2_[monomer_type];
        double* _f3 = f3_[monomer_type];

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

                        // Eigenvalue c and c*h
                        double c = c_prefactor * k2;
                        double ch = c * ds;
                        double ch2 = ch / 2.0;

                        // ETDRK4 coefficients using contour integral
                        _E[idx] = std::exp(ch);
                        _E2[idx] = std::exp(ch2);
                        _alpha[idx] = ds * phi1_contour(ch2);
                        _f1[idx] = ds * beta_contour(1, ch);
                        _f2[idx] = ds * beta_contour(2, ch);
                        _f3[idx] = ds * beta_contour(3, ch);
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

                        _E[idx] = std::exp(ch);
                        _E2[idx] = std::exp(ch2);
                        _alpha[idx] = ds * phi1_contour(ch2);
                        _f1[idx] = ds * beta_contour(1, ch);
                        _f2[idx] = ds * beta_contour(2, ch);
                        _f3[idx] = ds * beta_contour(3, ch);
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
        double* _f1 = f1_[monomer_type];
        double* _f2 = f2_[monomer_type];
        double* _f3 = f3_[monomer_type];

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

                    // ETDRK4 coefficients using contour integral
                    _E[idx] = std::exp(ch);
                    _E2[idx] = std::exp(ch2);
                    _alpha[idx] = ds * phi1_contour(ch2);
                    _f1[idx] = ds * beta_contour(1, ch);
                    _f2[idx] = ds * beta_contour(2, ch);
                    _f3[idx] = ds * beta_contour(3, ch);
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
