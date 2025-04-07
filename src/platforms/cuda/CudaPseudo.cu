#include <iostream>
#include <cmath>

#include "CudaCommon.h"
#include "CudaPseudo.h"

//----------------- Constructor -----------------------------
template <typename T>
CudaPseudo<T>::CudaPseudo(
    std::map<std::string, double> bond_lengths,
    std::vector<BoundaryCondition> bc,
    std::vector<int> nx, std::vector<double> dx, double ds)
        : Pseudo<T>(bond_lengths, bc, nx, dx, ds)
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
        gpu_error_check(cudaMalloc((void**)&d_fourier_basis_x, sizeof(double)*M_COMPLEX));
        gpu_error_check(cudaMalloc((void**)&d_fourier_basis_y, sizeof(double)*M_COMPLEX));
        gpu_error_check(cudaMalloc((void**)&d_fourier_basis_z, sizeof(double)*M_COMPLEX));

        if constexpr (std::is_same<T, std::complex<double>>::value)
        {
            const int* k_idx = Pseudo<T>::get_negative_frequency_mapping();
            gpu_error_check(cudaMalloc((void**)&d_k_idx, sizeof(int)*M_COMPLEX));
            gpu_error_check(cudaMemcpy(d_k_idx, k_idx, sizeof(int)*M_COMPLEX,cudaMemcpyHostToDevice));
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
    cudaFree(d_fourier_basis_x);
    cudaFree(d_fourier_basis_y);
    cudaFree(d_fourier_basis_z);
    if constexpr (std::is_same<T, std::complex<double>>::value)
        cudaFree(d_k_idx);
}
template <typename T>
void CudaPseudo<T>::update(
    std::vector<BoundaryCondition> bc, std::map<std::string, double> bond_lengths,
    std::vector<int> nx, std::vector<double> dx, double ds)
{
    Pseudo<T>::update(bc, bond_lengths, nx, dx, ds);

    const int M_COMPLEX = Pseudo<T>::get_total_complex_grid();;
    for(const auto& item: this->bond_lengths)
    {
        std::string monomer_type = item.first;
        gpu_error_check(cudaMemcpy(d_boltz_bond     [monomer_type], this->boltz_bond     [monomer_type], sizeof(double)*M_COMPLEX, cudaMemcpyHostToDevice));
        gpu_error_check(cudaMemcpy(d_boltz_bond_half[monomer_type], this->boltz_bond_half[monomer_type], sizeof(double)*M_COMPLEX, cudaMemcpyHostToDevice));
    }
    gpu_error_check(cudaMemcpy(d_fourier_basis_x, this->fourier_basis_x, sizeof(double)*M_COMPLEX, cudaMemcpyHostToDevice));
    gpu_error_check(cudaMemcpy(d_fourier_basis_y, this->fourier_basis_y, sizeof(double)*M_COMPLEX, cudaMemcpyHostToDevice));
    gpu_error_check(cudaMemcpy(d_fourier_basis_z, this->fourier_basis_z, sizeof(double)*M_COMPLEX, cudaMemcpyHostToDevice));

    // update_negative_frequency_mapping();
} 

// Explicit template instantiation
template class CudaPseudo<double>;
template class CudaPseudo<std::complex<double>>;
