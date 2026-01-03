/**
 * @file CudaPseudo.cu
 * @brief CUDA implementation of pseudo-spectral method base class.
 *
 * Extends Pseudo base class with GPU device memory for Boltzmann bond
 * factors and Fourier basis arrays used in stress calculations.
 *
 * **Device Arrays:**
 *
 * - d_boltz_bond: exp(-k² b² ds/6) for full step propagation
 * - d_boltz_bond_half: exp(-k² b² ds/12) for half step
 * - d_fourier_basis_x/y/z: Fourier basis for stress computation
 * - d_negative_k_idx: Mapping for conjugate symmetry (complex fields)
 *
 * **Update Mechanism:**
 *
 * When box dimensions change, update() recalculates Fourier space
 * operators and copies them to device memory.
 *
 * **Template Instantiations:**
 *
 * - CudaPseudo<double>: Real field computations
 * - CudaPseudo<std::complex<double>>: Complex field computations
 *
 * @see Pseudo for base class interface
 * @see CudaSolverPseudoContinuous for continuous chain solver
 * @see CudaSolverPseudoDiscrete for discrete chain solver
 */

#include <iostream>
#include <cmath>
#include <numbers>

#include "CudaCommon.h"
#include "CudaPseudo.h"

//----------------- Constructor -----------------------------
template <typename T>
CudaPseudo<T>::CudaPseudo(
    std::map<std::string, double> bond_lengths,
    std::vector<BoundaryCondition> bc,
    std::vector<int> nx, std::vector<double> dx, double ds,
    std::array<double, 6> recip_metric)
        : Pseudo<T>(bond_lengths, bc, nx, dx, ds, recip_metric)
{
    try
    {
        const int M_COMPLEX = Pseudo<T>::get_total_complex_grid();

        // Create boltz_bond, boltz_bond_half, exp_dw, and exp_dw_half
        for(const auto& item: this->bond_lengths)
        {
            std::string monomer_type = item.first;
            d_boltz_bond       [monomer_type] = nullptr;
            d_boltz_bond_half  [monomer_type] = nullptr;

            gpu_error_check(cudaMalloc((void**)&d_boltz_bond       [monomer_type], sizeof(double)*M_COMPLEX));
            gpu_error_check(cudaMalloc((void**)&d_boltz_bond_half  [monomer_type], sizeof(double)*M_COMPLEX));
        }

        // Allocate memory for stress calculation: compute_stress()
        // Diagonal terms
        gpu_error_check(cudaMalloc((void**)&d_fourier_basis_x, sizeof(double)*M_COMPLEX));
        gpu_error_check(cudaMalloc((void**)&d_fourier_basis_y, sizeof(double)*M_COMPLEX));
        gpu_error_check(cudaMalloc((void**)&d_fourier_basis_z, sizeof(double)*M_COMPLEX));
        // Cross-terms for non-orthogonal systems
        gpu_error_check(cudaMalloc((void**)&d_fourier_basis_xy, sizeof(double)*M_COMPLEX));
        gpu_error_check(cudaMalloc((void**)&d_fourier_basis_xz, sizeof(double)*M_COMPLEX));
        gpu_error_check(cudaMalloc((void**)&d_fourier_basis_yz, sizeof(double)*M_COMPLEX));

        if constexpr (std::is_same<T, std::complex<double>>::value)
        {
            const int* negative_k_idx = Pseudo<T>::get_negative_frequency_mapping();
            gpu_error_check(cudaMalloc((void**)&d_negative_k_idx, sizeof(int)*M_COMPLEX));
            gpu_error_check(cudaMemcpy(d_negative_k_idx, negative_k_idx, sizeof(int)*M_COMPLEX, cudaMemcpyHostToDevice));
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
//----------------- Destructor -----------------------------
template <typename T>
CudaPseudo<T>::~CudaPseudo()
{
    for(const auto& item: d_boltz_bond)
        cudaFree(item.second);
    for(const auto& item: d_boltz_bond_half)
        cudaFree(item.second);

    // For stress calculation: compute_stress()
    // Diagonal terms
    cudaFree(d_fourier_basis_x);
    cudaFree(d_fourier_basis_y);
    cudaFree(d_fourier_basis_z);
    // Cross-terms
    cudaFree(d_fourier_basis_xy);
    cudaFree(d_fourier_basis_xz);
    cudaFree(d_fourier_basis_yz);

    if constexpr (std::is_same<T, std::complex<double>>::value)
        cudaFree(d_negative_k_idx);
}
template <typename T>
void CudaPseudo<T>::update(
    std::vector<BoundaryCondition> bc,
    std::map<std::string, double> bond_lengths,
    std::vector<double> dx, double ds,
    std::array<double, 6> recip_metric)
{
    Pseudo<T>::update(bc, bond_lengths, dx, ds, recip_metric);

    const int M_COMPLEX = Pseudo<T>::get_total_complex_grid();;
    for(const auto& item: this->bond_lengths)
    {
        std::string monomer_type = item.first;
        gpu_error_check(cudaMemcpy(d_boltz_bond     [monomer_type], this->boltz_bond     [monomer_type], sizeof(double)*M_COMPLEX, cudaMemcpyHostToDevice));
        gpu_error_check(cudaMemcpy(d_boltz_bond_half[monomer_type], this->boltz_bond_half[monomer_type], sizeof(double)*M_COMPLEX, cudaMemcpyHostToDevice));
    }
    // Diagonal terms
    gpu_error_check(cudaMemcpy(d_fourier_basis_x, this->fourier_basis_x, sizeof(double)*M_COMPLEX, cudaMemcpyHostToDevice));
    gpu_error_check(cudaMemcpy(d_fourier_basis_y, this->fourier_basis_y, sizeof(double)*M_COMPLEX, cudaMemcpyHostToDevice));
    gpu_error_check(cudaMemcpy(d_fourier_basis_z, this->fourier_basis_z, sizeof(double)*M_COMPLEX, cudaMemcpyHostToDevice));
    // Cross-terms
    gpu_error_check(cudaMemcpy(d_fourier_basis_xy, this->fourier_basis_xy, sizeof(double)*M_COMPLEX, cudaMemcpyHostToDevice));
    gpu_error_check(cudaMemcpy(d_fourier_basis_xz, this->fourier_basis_xz, sizeof(double)*M_COMPLEX, cudaMemcpyHostToDevice));
    gpu_error_check(cudaMemcpy(d_fourier_basis_yz, this->fourier_basis_yz, sizeof(double)*M_COMPLEX, cudaMemcpyHostToDevice));

    // update_negative_frequency_mapping();
} 

// Explicit template instantiation
template class CudaPseudo<double>;
template class CudaPseudo<std::complex<double>>;
