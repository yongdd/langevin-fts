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
    std::vector<int> nx, std::vector<double> dx,
    std::array<double, 6> recip_metric,
    std::array<double, 9> recip_vec)
        : Pseudo<T>(bond_lengths, bc, nx, dx, recip_metric, recip_vec)
{
    // The device arrays are owning buffers: a failed allocation below releases
    // the ones made so far, and the Pseudo<T> base frees the host arrays.
    const int M_COMPLEX = Pseudo<T>::get_total_complex_grid();

    // Boltzmann factors are allocated in finalize_ds_values()
    // after add_ds_value() is called for each unique ds value

    // Stress calculation: compute_stress(). Diagonal terms, then the
    // cross-terms for non-orthogonal systems
    d_fourier_basis_x.allocate(M_COMPLEX);
    d_fourier_basis_y.allocate(M_COMPLEX);
    d_fourier_basis_z.allocate(M_COMPLEX);
    d_fourier_basis_xy.allocate(M_COMPLEX);
    d_fourier_basis_xz.allocate(M_COMPLEX);
    d_fourier_basis_yz.allocate(M_COMPLEX);

    if constexpr (std::is_same<T, std::complex<double>>::value)
    {
        // Mirror the host convention: the mapping exists only for
        // periodic BCs; leave the device pointer nullptr otherwise so
        // get_negative_frequency_mapping() returns nullptr on both
        // sides for non-periodic boxes.
        if (Pseudo<T>::get_negative_frequency_mapping() != nullptr)
            d_negative_k_idx.allocate(M_COMPLEX);
    }

    // Upload Fourier basis to GPU
    upload_fourier_basis();
}
//----------------- Destructor -----------------------------
template <typename T>
CudaPseudo<T>::~CudaPseudo() = default;

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
            gpu_error_check(cudaMemcpy(d_boltz_bond[ds_index][monomer_type], item.second.data(),
                                       sizeof(double) * M_COMPLEX, cudaMemcpyHostToDevice));
        }
    }

    for (const auto& ds_entry : this->boltz_bond_half)
    {
        int ds_index = ds_entry.first;
        for (const auto& item : ds_entry.second)
        {
            std::string monomer_type = item.first;
            gpu_error_check(cudaMemcpy(d_boltz_bond_half[ds_index][monomer_type], item.second.data(),
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
    gpu_error_check(cudaMemcpy(d_fourier_basis_x, this->fourier_basis_x.data(),
                               sizeof(double) * M_COMPLEX, cudaMemcpyHostToDevice));
    gpu_error_check(cudaMemcpy(d_fourier_basis_y, this->fourier_basis_y.data(),
                               sizeof(double) * M_COMPLEX, cudaMemcpyHostToDevice));
    gpu_error_check(cudaMemcpy(d_fourier_basis_z, this->fourier_basis_z.data(),
                               sizeof(double) * M_COMPLEX, cudaMemcpyHostToDevice));
    // Cross-terms for non-orthogonal systems
    gpu_error_check(cudaMemcpy(d_fourier_basis_xy, this->fourier_basis_xy.data(),
                               sizeof(double) * M_COMPLEX, cudaMemcpyHostToDevice));
    gpu_error_check(cudaMemcpy(d_fourier_basis_xz, this->fourier_basis_xz.data(),
                               sizeof(double) * M_COMPLEX, cudaMemcpyHostToDevice));
    gpu_error_check(cudaMemcpy(d_fourier_basis_yz, this->fourier_basis_yz.data(),
                               sizeof(double) * M_COMPLEX, cudaMemcpyHostToDevice));

    // negative_k_idx for complex fields with periodic BC
    if constexpr (std::is_same<T, std::complex<double>>::value)
    {
        // The negative-frequency mapping exists only for periodic BCs
        // (conjugate symmetry of the full complex FFT); for non-periodic
        // boxes the host pointer is nullptr and there is nothing to upload.
        const int* negative_k_idx = Pseudo<T>::get_negative_frequency_mapping();
        if (negative_k_idx != nullptr)
            gpu_error_check(cudaMemcpy(d_negative_k_idx, negative_k_idx,
                                       sizeof(int) * M_COMPLEX, cudaMemcpyHostToDevice));
    }
}

//----------------- Update -----------------------------
template <typename T>
void CudaPseudo<T>::update(
    std::vector<BoundaryCondition> bc,
    std::map<std::string, double> bond_lengths,
    std::vector<double> dx,
    std::array<double, 6> recip_metric,
    std::array<double, 9> recip_vec)
{
    Pseudo<T>::update(bc, bond_lengths, dx, recip_metric, recip_vec);

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
            // Allocate only if no GPU memory exists for this ds_index/monomer_type
            CudaDeviceBuffer<double>& slot = d_boltz_bond[ds_index][monomer_type];
            if (slot.empty())
                slot.allocate(M_COMPLEX);
        }
    }
    for (const auto& ds_entry : this->boltz_bond_half)
    {
        int ds_index = ds_entry.first;
        for (const auto& item : ds_entry.second)
        {
            std::string monomer_type = item.first;
            // Allocate only if no GPU memory exists for this ds_index/monomer_type
            CudaDeviceBuffer<double>& slot = d_boltz_bond_half[ds_index][monomer_type];
            if (slot.empty())
                slot.allocate(M_COMPLEX);
        }
    }

    // Upload all Boltzmann factors to GPU
    upload_boltz_bond();
}

// Explicit template instantiation
template class CudaPseudo<double>;
template class CudaPseudo<std::complex<double>>;
