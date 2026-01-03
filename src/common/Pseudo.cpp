/**
 * @file Pseudo.cpp
 * @brief Implementation of pseudo-spectral method utilities.
 *
 * Provides the base implementation for computing Fourier-space operators
 * used in the pseudo-spectral propagator solver. Platform-specific versions
 * (CpuPseudo, CudaPseudo) copy these arrays to appropriate memory.
 *
 * **Boltzmann Factors:**
 *
 * In Fourier space, the diffusion operator becomes multiplication:
 *     exp(-b²k²ds/6) for full step
 *     exp(-b²k²ds/12) for half step (discrete chains)
 *
 * **Fourier Basis Vectors:**
 *
 * Weighted wavenumber squares for stress calculation:
 *     fourier_basis_x[k] = kx² × weight
 *
 * Weight factor of 2 for interior k modes in r2c transform (Hermitian symmetry).
 *
 * **Negative Frequency Mapping:**
 *
 * For complex fields, maps each frequency to its conjugate partner:
 *     negative_k_idx[k] = index of -k
 *
 * Required for computing real-valued stress from complex propagators.
 *
 * **Template Instantiations:**
 *
 * - Pseudo<double>: Real fields with r2c FFT (reduced storage)
 * - Pseudo<std::complex<double>>: Complex fields with c2c FFT
 */

#include <iostream>
#include <cmath>
#include <numbers>
#include <array>
#include "Pseudo.h"

/**
 * @brief Construct pseudo-spectral utilities.
 *
 * Allocates Boltzmann factors and Fourier basis arrays, then initializes
 * all operators for the given grid and contour step.
 *
 * @param bond_lengths  Map of monomer type to segment length (b)
 * @param bc            Boundary conditions (must be periodic)
 * @param nx            Grid points per dimension
 * @param dx            Grid spacing per dimension
 * @param ds            Contour step size
 * @param recip_metric  Reciprocal metric tensor [G*_00, G*_01, G*_02, G*_11, G*_12, G*_22]
 */
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

        update_total_complex_grid();
        const int M_COMPLEX = get_total_complex_grid();

        // Create boltz_bond, boltz_bond_half, exp_dw, and exp_dw_half
        for(const auto& item: bond_lengths)
        {
            std::string monomer_type = item.first;
            boltz_bond     [monomer_type] = new double[M_COMPLEX];
            boltz_bond_half[monomer_type] = new double[M_COMPLEX];
        }

        // Allocate memory for stress calculation: compute_stress()
        // Diagonal terms
        fourier_basis_x = new double[M_COMPLEX];
        fourier_basis_y = new double[M_COMPLEX];
        fourier_basis_z = new double[M_COMPLEX];
        // Cross-terms for non-orthogonal systems
        fourier_basis_xy = new double[M_COMPLEX];
        fourier_basis_xz = new double[M_COMPLEX];
        fourier_basis_yz = new double[M_COMPLEX];

        if constexpr (std::is_same<T, std::complex<double>>::value)
            negative_k_idx = new int[M_COMPLEX];

        update_boltz_bond();
        update_weighted_fourier_basis();
        update_negative_frequency_mapping();
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
//----------------- Destructor -----------------------------
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
        delete[] negative_k_idx;

    for(const auto& item: boltz_bond)
        delete[] item.second;
    for(const auto& item: boltz_bond_half)
        delete[] item.second;
}
template <typename T>
int Pseudo<T>::get_total_complex_grid()
{
    return total_complex_grid;
}
template <typename T>
double* Pseudo<T>::get_boltz_bond(std::string monomer_type)
{
    return boltz_bond[monomer_type];
}
template <typename T>
double* Pseudo<T>::get_boltz_bond_half(std::string monomer_type)
{
    return boltz_bond_half[monomer_type];
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
const int* Pseudo<T>::get_negative_frequency_mapping()
{
    return negative_k_idx;
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

//----------------- update_boltz_bond -------------------
template <typename T>
void Pseudo<T>::update_total_complex_grid()
{
    std::vector<int> nx = this->nx;
    int total_complex_grid;
    if constexpr (std::is_same<T, double>::value)
    {
        if (nx.size() == 3)
            total_complex_grid = nx[0]*nx[1]*(nx[2]/2+1);
        else if (nx.size() == 2)
            total_complex_grid = nx[0]*(nx[1]/2+1);
        else if (nx.size() == 1)
            total_complex_grid = nx[0]/2+1;
    }
    else
    {
        if (nx.size() == 3)
            total_complex_grid = nx[0]*nx[1]*nx[2];
        else if (nx.size() == 2)
            total_complex_grid = nx[0]*nx[1];
        else if (nx.size() == 1)
            total_complex_grid = nx[0];
    }
    this->total_complex_grid = total_complex_grid;
}

//----------------- update_boltz_bond -------------------
/**
 * @brief Update Boltzmann factors using reciprocal metric tensor.
 *
 * Computes exp(-b²|k|²ds/6) where |k|² = (2π)² * G*_ij h_i h_j
 * and G* is the reciprocal metric tensor, h_i are Miller indices.
 *
 * For orthogonal systems (G*_ij diagonal): reduces to standard formula.
 * For non-orthogonal systems: includes cross-term contributions.
 */
template <typename T>
void Pseudo<T>::update_boltz_bond()
{
    try
    {
        int itemp, jtemp, ktemp, idx;
        const double PI = std::numbers::pi;
        const double FOUR_PI_SQ = 4.0 * PI * PI;

        const int DIM = nx.size();
        std::vector<int> tnx;

        // Set up grid dimensions (pad to 3D for unified loop)
        if (DIM == 3)
            tnx = {nx[0], nx[1], nx[2]};
        else if (DIM == 2)
            tnx = {1, nx[0], nx[1]};
        else if (DIM == 1)
            tnx = {1, 1, nx[0]};

        for(size_t i=0; i<bc.size(); i++)
        {
            if (bc[i] != BoundaryCondition::PERIODIC)
                throw_with_line_number("Currently, pseudo-spectral method only supports periodic boundary conditions");
        }

        // Extract reciprocal metric components and remap based on dimension
        // The metric tensor is computed from lx[0], lx[1], lx[2] (or padded with 1.0)
        // But the loop indices are padded differently:
        //   3D: (i,j,k) -> (lx[0], lx[1], lx[2]) -> use G00, G11, G22 directly
        //   2D: (i,j,k) = (pad, nx[0], nx[1]) -> need G00 for j, G11 for k
        //   1D: (i,j,k) = (pad, pad, nx[0]) -> need G00 for k
        // Layout: [G*_00, G*_01, G*_02, G*_11, G*_12, G*_22]
        double Gii, Gjj, Gkk, Gij, Gik, Gjk;
        if (DIM == 3) {
            Gii = recip_metric_[0];  // G*_00
            Gjj = recip_metric_[3];  // G*_11
            Gkk = recip_metric_[5];  // G*_22
            Gij = recip_metric_[1];  // G*_01
            Gik = recip_metric_[2];  // G*_02
            Gjk = recip_metric_[4];  // G*_12
        } else if (DIM == 2) {
            // For 2D: j corresponds to lx[0], k corresponds to lx[1]
            Gii = 0.0;               // padded dimension, no contribution
            Gjj = recip_metric_[0];  // G*_00 for lx[0]
            Gkk = recip_metric_[3];  // G*_11 for lx[1]
            Gij = 0.0;               // cross with padded dimension
            Gik = 0.0;               // cross with padded dimension
            Gjk = recip_metric_[1];  // G*_01 for lx[0]-lx[1] cross term
        } else { // DIM == 1
            // For 1D: k corresponds to lx[0]
            Gii = 0.0;               // padded dimension
            Gjj = 0.0;               // padded dimension
            Gkk = recip_metric_[0];  // G*_00 for lx[0]
            Gij = 0.0;               // cross with padded dimension
            Gik = 0.0;               // cross with padded dimension
            Gjk = 0.0;               // cross with padded dimension
        }

        for(const auto& item: this->bond_lengths)
        {
            std::string monomer_type = item.first;
            double bond_length_sq    = item.second*item.second;
            double* _boltz_bond      = boltz_bond[monomer_type];
            double* _boltz_bond_half = boltz_bond_half[monomer_type];
            double mag_q2;

            // Pre-compute constant factor: -b² * (2π)² * ds / 6
            double prefactor = -bond_length_sq * FOUR_PI_SQ * ds / 6.0;

            for(int i=0; i<tnx[0]; i++)
            {
                if( i > tnx[0]/2)
                    itemp = tnx[0]-i;
                else
                    itemp = i;
                for(int j=0; j<tnx[1]; j++)
                {
                    if( j > tnx[1]/2)
                        jtemp = tnx[1]-j;
                    else
                        jtemp = j;

                    if constexpr (std::is_same<T, double>::value)
                    {
                        for(int k=0; k<tnx[2]/2+1; k++)
                        {
                            ktemp = k;
                            idx = i * tnx[1]*(tnx[2]/2+1) + j*(tnx[2]/2+1) + k;

                            // |k|² = (2π)² * (G*_ij h_i h_j)
                            // mag_q2 = -b²*ds/6 * |k|²
                            mag_q2 = prefactor * (
                                Gii * itemp * itemp +
                                Gjj * jtemp * jtemp +
                                Gkk * ktemp * ktemp +
                                2.0 * Gij * itemp * jtemp +
                                2.0 * Gik * itemp * ktemp +
                                2.0 * Gjk * jtemp * ktemp
                            );
                            _boltz_bond     [idx] = exp(mag_q2);
                            _boltz_bond_half[idx] = exp(mag_q2/2.0);
                        }
                    }
                    else
                    {
                        for(int k=0; k<tnx[2]; k++)
                        {
                            if( k > tnx[2]/2)
                                ktemp = tnx[2]-k;
                            else
                                ktemp = k;
                            idx = i * tnx[1]*tnx[2] + j*tnx[2] + k;

                            // |k|² = (2π)² * (G*_ij h_i h_j)
                            mag_q2 = prefactor * (
                                Gii * itemp * itemp +
                                Gjj * jtemp * jtemp +
                                Gkk * ktemp * ktemp +
                                2.0 * Gij * itemp * jtemp +
                                2.0 * Gik * itemp * ktemp +
                                2.0 * Gjk * jtemp * ktemp
                            );
                            _boltz_bond     [idx] = exp(mag_q2);
                            _boltz_bond_half[idx] = exp(mag_q2/2.0);
                        }
                    }
                }
            }
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
void Pseudo<T>::update_weighted_fourier_basis()
{
    int itemp, jtemp, ktemp, idx;
    const double PI = std::numbers::pi;

    const int DIM = nx.size();
    std::vector<int> tnx;
    std::vector<double> tdx;
    if (DIM == 3)
    {
        tnx = {nx[0], nx[1], nx[2]};
        tdx = {dx[0], dx[1], dx[2]};
    }
    else if (DIM == 2)
    {
        tnx = {1,   nx[0], nx[1]};
        tdx = {1.0, dx[0], dx[1]};
    }
    else if (DIM == 1)
    {
        tnx = {1,   1,   nx[0]};
        tdx = {1.0, 1.0, dx[0]};
    }

    for(size_t i=0; i<bc.size(); i++)
    {
        if (bc[i] != BoundaryCondition::PERIODIC)
            throw_with_line_number("Currently, pseudo-spectral method only supports periodic boundary conditions");
    }

    // Calculate diagonal factors for each real dimension: (2π/L_i)²
    // xfactor[d] corresponds to lx[d], not to the padded loop index
    double xfactor[3] = {0.0, 0.0, 0.0};
    double k_factor[3] = {0.0, 0.0, 0.0};  // 2π/L_i for each real dimension
    for(int d=0; d<DIM; d++) {
        double L = nx[d] * dx[d];
        xfactor[d] = std::pow(2*PI/L, 2);
        k_factor[d] = 2*PI/L;
    }

    // Cross factors for non-orthogonal systems
    double cross_factor_01 = (DIM >= 2) ? 2.0 * k_factor[0] * k_factor[1] : 0.0;
    double cross_factor_02 = (DIM >= 3) ? 2.0 * k_factor[0] * k_factor[2] : 0.0;
    double cross_factor_12 = (DIM >= 3) ? 2.0 * k_factor[1] * k_factor[2] : 0.0;

    for(int i=0; i<tnx[0]; i++)
    {
        if( i > tnx[0]/2)
            itemp = tnx[0]-i;
        else
            itemp = i;
        for(int j=0; j<tnx[1]; j++)
        {
            if( j > tnx[1]/2)
                jtemp = tnx[1]-j;
            else
                jtemp = j;

            if constexpr (std::is_same<T, double>::value)
            {
                for(int k=0; k<tnx[2]/2+1; k++)
                {
                    ktemp = k;
                    idx = i* tnx[1]*(tnx[2]/2+1) + j*(tnx[2]/2+1) + k;

                    // Map loop indices to real dimension indices based on DIM
                    // For 3D: n0=itemp, n1=jtemp, n2=ktemp
                    // For 2D: n0=jtemp, n1=ktemp (itemp=0)
                    // For 1D: n0=ktemp (itemp=jtemp=0)
                    int n0, n1, n2;
                    if (DIM == 3) {
                        n0 = itemp; n1 = jtemp; n2 = ktemp;
                    } else if (DIM == 2) {
                        n0 = jtemp; n1 = ktemp; n2 = 0;
                    } else { // DIM == 1
                        n0 = ktemp; n1 = 0; n2 = 0;
                    }

                    // Diagonal terms: fourier_basis_x for lx[0], etc.
                    fourier_basis_x[idx] = n0*n0*xfactor[0];
                    fourier_basis_y[idx] = (DIM >= 2) ? n1*n1*xfactor[1] : 0.0;
                    fourier_basis_z[idx] = (DIM >= 3) ? n2*n2*xfactor[2] : 0.0;
                    // Cross-terms
                    fourier_basis_xy[idx] = (DIM >= 2) ? n0*n1*cross_factor_01 : 0.0;
                    fourier_basis_xz[idx] = (DIM >= 3) ? n0*n2*cross_factor_02 : 0.0;
                    fourier_basis_yz[idx] = (DIM >= 3) ? n1*n2*cross_factor_12 : 0.0;

                    // Weight factor of 2 for interior k modes (Hermitian symmetry in r2c)
                    if (k != 0 && 2*k != tnx[2])
                    {
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
                for(int k=0; k<tnx[2]; k++)
                {
                    if( k > tnx[2]/2)
                        ktemp = tnx[2]-k;
                    else
                        ktemp = k;
                    idx = i* tnx[1]*tnx[2] + j*tnx[2] + k;

                    // Map loop indices to real dimension indices based on DIM
                    int n0, n1, n2;
                    if (DIM == 3) {
                        n0 = itemp; n1 = jtemp; n2 = ktemp;
                    } else if (DIM == 2) {
                        n0 = jtemp; n1 = ktemp; n2 = 0;
                    } else { // DIM == 1
                        n0 = ktemp; n1 = 0; n2 = 0;
                    }

                    // Diagonal terms
                    fourier_basis_x[idx] = n0*n0*xfactor[0];
                    fourier_basis_y[idx] = (DIM >= 2) ? n1*n1*xfactor[1] : 0.0;
                    fourier_basis_z[idx] = (DIM >= 3) ? n2*n2*xfactor[2] : 0.0;
                    // Cross-terms
                    fourier_basis_xy[idx] = (DIM >= 2) ? n0*n1*cross_factor_01 : 0.0;
                    fourier_basis_xz[idx] = (DIM >= 3) ? n0*n2*cross_factor_02 : 0.0;
                    fourier_basis_yz[idx] = (DIM >= 3) ? n1*n2*cross_factor_12 : 0.0;
                }
            }
        }
    }
}

template <typename T>
void Pseudo<T>::update_negative_frequency_mapping(){}

template <>
void Pseudo<std::complex<double>>::update_negative_frequency_mapping()
{
    const int DIM = nx.size();
    int itemp, jtemp, ktemp, idx, idx_minus;
    std::vector<int> tnx;

    if (DIM == 3)
    {
        tnx = {nx[0], nx[1], nx[2]};
    }
    else if (DIM == 2)
    {
        tnx = {1,   nx[0], nx[1]};
    }
    else if (DIM == 1)
    {
        tnx = {1,   1,   nx[0]};
    }

    for(int i=0; i<tnx[0]; i++)
    {
        if(i == 0)
            itemp = 0;
        else
            itemp = tnx[0]-i;
        for(int j=0; j<tnx[1]; j++)
        {
            if(j == 0)
                jtemp = 0;
            else
                jtemp = tnx[1]-j;
            for(int k=0; k<tnx[2]; k++)
            {
                if(k == 0)
                    ktemp = 0;
                else
                    ktemp = tnx[2]-k;

                idx       = i    *tnx[1]*tnx[2] + j    *tnx[2] + k;
                idx_minus = itemp*tnx[1]*tnx[2] + jtemp*tnx[2] + ktemp;

                negative_k_idx[idx] = idx_minus;

            //     0   0   0
            //     1   1   6
            //     2   2   5
            //     3   3   4
            //    -3   4   3
            //    -2   5   2
            //    -1   6   1

            //     0   0   0
            //     1   1   7
            //     2   2   6
            //     3   3   5
            //    -4   4   4
            //    -3   5   3
            //    -2   6   2
            //    -1   7   1

            }
        }
    }
    // const int M_COMPLEX = get_total_complex_grid();
    // for(int i=0;i<M_COMPLEX;i++)
    //     std::cout << i << ", " << negative_k_idx[i] << ", " << std::endl;
}

// Explicit template instantiation
template class Pseudo<double>;
template class Pseudo<std::complex<double>>;
