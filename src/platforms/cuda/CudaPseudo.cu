/**
 * @file CudaPseudo.cu
 * @brief CUDA implementation of pseudo-spectral method base class.
 *
 * Extends Pseudo base class with GPU device memory for Boltzmann bond
 * factors and Fourier basis arrays used in stress calculations.
 *
 * **Device Arrays:**
 *
 * - d_boltz_bond: exp(-b²|k|²ds/6) for full step propagation
 *     - Continuous chains: diffusion propagator
 *     - Discrete chains: bond function ĝ(k)
 * - d_boltz_bond_half: exp(-b²|k|²ds/12) for half step
 *     - Continuous chains: half-step diffusion
 *     - Discrete chains: half-bond function ĝ^(1/2)(k)
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
 * @see CudaSolverPseudoRQM4 for continuous chain solver
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

        // Create boltz_bond, boltz_bond_half for each ds_index and monomer_type
        // The base class Pseudo computes boltz_bond[ds_index][monomer_type]
        for (const auto& ds_entry : this->boltz_bond)
        {
            int ds_index = ds_entry.first;
            for (const auto& item : ds_entry.second)
            {
                std::string monomer_type = item.first;
                d_boltz_bond[ds_index][monomer_type] = nullptr;
                gpu_error_check(cudaMalloc((void**)&d_boltz_bond[ds_index][monomer_type], sizeof(double)*M_COMPLEX));
            }
        }
        for (const auto& ds_entry : this->boltz_bond_half)
        {
            int ds_index = ds_entry.first;
            for (const auto& item : ds_entry.second)
            {
                std::string monomer_type = item.first;
                d_boltz_bond_half[ds_index][monomer_type] = nullptr;
                gpu_error_check(cudaMalloc((void**)&d_boltz_bond_half[ds_index][monomer_type], sizeof(double)*M_COMPLEX));
            }
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
            gpu_error_check(cudaMalloc((void**)&d_negative_k_idx, sizeof(int)*M_COMPLEX));
        }

        // Upload computed data to GPU
        upload_boltz_bond();
        upload_fourier_basis();
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
    // Free GPU memory for nested maps: d_boltz_bond[ds_index][monomer_type]
    for (const auto& ds_entry : d_boltz_bond)
        for (const auto& item : ds_entry.second)
            cudaFree(item.second);
    for (const auto& ds_entry : d_boltz_bond_half)
        for (const auto& item : ds_entry.second)
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

//----------------- Upload Boltzmann factors to GPU -----------------------------
template <typename T>
void CudaPseudo<T>::upload_boltz_bond()
{
    const int M_COMPLEX = Pseudo<T>::get_total_complex_grid();

    // Upload all ds_index values: boltz_bond[ds_index][monomer_type]
    for (const auto& ds_entry : this->boltz_bond)
    {
        int ds_index = ds_entry.first;
        for (const auto& item : ds_entry.second)
        {
            std::string monomer_type = item.first;
            gpu_error_check(cudaMemcpy(d_boltz_bond[ds_index][monomer_type], item.second,
                                       sizeof(double) * M_COMPLEX, cudaMemcpyHostToDevice));
        }
    }

    for (const auto& ds_entry : this->boltz_bond_half)
    {
        int ds_index = ds_entry.first;
        for (const auto& item : ds_entry.second)
        {
            std::string monomer_type = item.first;
            gpu_error_check(cudaMemcpy(d_boltz_bond_half[ds_index][monomer_type], item.second,
                                       sizeof(double) * M_COMPLEX, cudaMemcpyHostToDevice));
        }
    }
}

//----------------- Upload Fourier basis to GPU -----------------------------
template <typename T>
void CudaPseudo<T>::upload_fourier_basis()
{
    const int M_COMPLEX = Pseudo<T>::get_total_complex_grid();

    // Diagonal terms
    gpu_error_check(cudaMemcpy(d_fourier_basis_x, this->fourier_basis_x,
                               sizeof(double) * M_COMPLEX, cudaMemcpyHostToDevice));
    gpu_error_check(cudaMemcpy(d_fourier_basis_y, this->fourier_basis_y,
                               sizeof(double) * M_COMPLEX, cudaMemcpyHostToDevice));
    gpu_error_check(cudaMemcpy(d_fourier_basis_z, this->fourier_basis_z,
                               sizeof(double) * M_COMPLEX, cudaMemcpyHostToDevice));
    // Cross-terms for non-orthogonal systems
    gpu_error_check(cudaMemcpy(d_fourier_basis_xy, this->fourier_basis_xy,
                               sizeof(double) * M_COMPLEX, cudaMemcpyHostToDevice));
    gpu_error_check(cudaMemcpy(d_fourier_basis_xz, this->fourier_basis_xz,
                               sizeof(double) * M_COMPLEX, cudaMemcpyHostToDevice));
    gpu_error_check(cudaMemcpy(d_fourier_basis_yz, this->fourier_basis_yz,
                               sizeof(double) * M_COMPLEX, cudaMemcpyHostToDevice));

    // negative_k_idx for complex fields with periodic BC
    if constexpr (std::is_same<T, std::complex<double>>::value)
    {
        const int* negative_k_idx = Pseudo<T>::get_negative_frequency_mapping();
        gpu_error_check(cudaMemcpy(d_negative_k_idx, negative_k_idx,
                                   sizeof(int) * M_COMPLEX, cudaMemcpyHostToDevice));
    }
}

//----------------- Update -----------------------------
template <typename T>
void CudaPseudo<T>::update(
    std::vector<BoundaryCondition> bc,
    std::map<std::string, double> bond_lengths,
    std::vector<double> dx, double ds,
    std::array<double, 6> recip_metric)
{
    Pseudo<T>::update(bc, bond_lengths, dx, ds, recip_metric);

    // Upload updated data to GPU
    upload_boltz_bond();
    upload_fourier_basis();
}

//----------------- finalize_ds_values -----------------------------
template <typename T>
void CudaPseudo<T>::finalize_ds_values()
{
    // Call base class to compute host-side Boltzmann factors for all ds values
    Pseudo<T>::finalize_ds_values();

    const int M_COMPLEX = Pseudo<T>::get_total_complex_grid();

    // Allocate GPU memory for any new ds_index values not already allocated
    for (const auto& ds_entry : this->boltz_bond)
    {
        int ds_index = ds_entry.first;
        for (const auto& item : ds_entry.second)
        {
            std::string monomer_type = item.first;
            // Check if GPU memory already exists for this ds_index/monomer_type
            if (d_boltz_bond.find(ds_index) == d_boltz_bond.end() ||
                d_boltz_bond[ds_index].find(monomer_type) == d_boltz_bond[ds_index].end())
            {
                d_boltz_bond[ds_index][monomer_type] = nullptr;
                gpu_error_check(cudaMalloc((void**)&d_boltz_bond[ds_index][monomer_type], sizeof(double)*M_COMPLEX));
            }
        }
    }
    for (const auto& ds_entry : this->boltz_bond_half)
    {
        int ds_index = ds_entry.first;
        for (const auto& item : ds_entry.second)
        {
            std::string monomer_type = item.first;
            // Check if GPU memory already exists for this ds_index/monomer_type
            if (d_boltz_bond_half.find(ds_index) == d_boltz_bond_half.end() ||
                d_boltz_bond_half[ds_index].find(monomer_type) == d_boltz_bond_half[ds_index].end())
            {
                d_boltz_bond_half[ds_index][monomer_type] = nullptr;
                gpu_error_check(cudaMalloc((void**)&d_boltz_bond_half[ds_index][monomer_type], sizeof(double)*M_COMPLEX));
            }
        }
    }

    // Upload all Boltzmann factors to GPU
    upload_boltz_bond();
}

//----------------- set_cell_averaged_bond -----------------------------
template <typename T>
void CudaPseudo<T>::set_cell_averaged_bond(bool enabled)
{
    // Call base class to update host-side Boltzmann factors
    Pseudo<T>::set_cell_averaged_bond(enabled);

    // Re-upload to GPU
    upload_boltz_bond();
}

//----------------- set_cell_average_momentum -----------------------------
template <typename T>
void CudaPseudo<T>::set_cell_average_momentum(int n)
{
    // Call base class to update host-side Boltzmann factors
    Pseudo<T>::set_cell_average_momentum(n);

    // Re-upload to GPU
    upload_boltz_bond();
}

// Explicit template instantiation
template class CudaPseudo<double>;
template class CudaPseudo<std::complex<double>>;
