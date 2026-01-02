/**
 * @file CudaPseudoMixedBC.cu
 * @brief GPU implementation of pseudo-spectral utilities for mixed BCs.
 *
 * Computes Boltzmann factors on CPU using PseudoMixedBC, then uploads
 * to GPU device memory for efficient access during propagator computation.
 *
 * @see CudaPseudoMixedBC.h for class documentation
 */

#include <iostream>
#include <cmath>
#include <numbers>

#include "CudaPseudoMixedBC.h"
#include "CudaCommon.h"

//------------------------------------------------------------------------------
// Constructor
//------------------------------------------------------------------------------
template <typename T>
CudaPseudoMixedBC<T>::CudaPseudoMixedBC(
    std::map<std::string, double> bond_lengths,
    std::vector<BoundaryCondition> bc,
    std::vector<int> nx, std::vector<double> dx, double ds)
    : PseudoMixedBC<T>(bond_lengths, bc, nx, dx, ds),
      d_fourier_basis_x(nullptr), d_fourier_basis_y(nullptr), d_fourier_basis_z(nullptr)
{
    try
    {
        const int M = this->total_complex_grid;

        // Allocate GPU memory for Fourier basis
        gpu_error_check(cudaMalloc((void**)&d_fourier_basis_x, sizeof(double) * M));
        gpu_error_check(cudaMalloc((void**)&d_fourier_basis_y, sizeof(double) * M));
        gpu_error_check(cudaMalloc((void**)&d_fourier_basis_z, sizeof(double) * M));

        // Allocate GPU memory for Boltzmann factors
        for (const auto& item : bond_lengths)
        {
            std::string monomer_type = item.first;
            d_boltz_bond[monomer_type] = nullptr;
            d_boltz_bond_half[monomer_type] = nullptr;
            gpu_error_check(cudaMalloc((void**)&d_boltz_bond[monomer_type], sizeof(double) * M));
            gpu_error_check(cudaMalloc((void**)&d_boltz_bond_half[monomer_type], sizeof(double) * M));
        }

        // Upload data to GPU
        upload_boltz_bond();
        upload_fourier_basis();
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
CudaPseudoMixedBC<T>::~CudaPseudoMixedBC()
{
    cudaFree(d_fourier_basis_x);
    cudaFree(d_fourier_basis_y);
    cudaFree(d_fourier_basis_z);

    for (const auto& item : d_boltz_bond)
        cudaFree(item.second);
    for (const auto& item : d_boltz_bond_half)
        cudaFree(item.second);
}

//------------------------------------------------------------------------------
// Upload Boltzmann factors to GPU
//------------------------------------------------------------------------------
template <typename T>
void CudaPseudoMixedBC<T>::upload_boltz_bond()
{
    const int M = this->total_complex_grid;

    for (const auto& item : this->boltz_bond)
    {
        std::string monomer_type = item.first;
        gpu_error_check(cudaMemcpy(d_boltz_bond[monomer_type], item.second,
                                   sizeof(double) * M, cudaMemcpyHostToDevice));
    }

    for (const auto& item : this->boltz_bond_half)
    {
        std::string monomer_type = item.first;
        gpu_error_check(cudaMemcpy(d_boltz_bond_half[monomer_type], item.second,
                                   sizeof(double) * M, cudaMemcpyHostToDevice));
    }
}

//------------------------------------------------------------------------------
// Upload Fourier basis to GPU
//------------------------------------------------------------------------------
template <typename T>
void CudaPseudoMixedBC<T>::upload_fourier_basis()
{
    const int M = this->total_complex_grid;

    gpu_error_check(cudaMemcpy(d_fourier_basis_x, this->fourier_basis_x,
                               sizeof(double) * M, cudaMemcpyHostToDevice));
    gpu_error_check(cudaMemcpy(d_fourier_basis_y, this->fourier_basis_y,
                               sizeof(double) * M, cudaMemcpyHostToDevice));
    gpu_error_check(cudaMemcpy(d_fourier_basis_z, this->fourier_basis_z,
                               sizeof(double) * M, cudaMemcpyHostToDevice));
}

//------------------------------------------------------------------------------
// Update operators for new box dimensions
//------------------------------------------------------------------------------
template <typename T>
void CudaPseudoMixedBC<T>::update(
    std::vector<BoundaryCondition> bc,
    std::map<std::string, double> bond_lengths,
    std::vector<double> dx, double ds)
{
    try
    {
        // Update on CPU (base class)
        PseudoMixedBC<T>::update(bc, bond_lengths, dx, ds);

        // Upload updated data to GPU
        upload_boltz_bond();
        upload_fourier_basis();
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

// Explicit template instantiations
template class CudaPseudoMixedBC<double>;
template class CudaPseudoMixedBC<std::complex<double>>;
