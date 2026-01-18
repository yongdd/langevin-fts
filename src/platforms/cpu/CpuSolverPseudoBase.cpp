/**
 * @file CpuSolverPseudoBase.cpp
 * @brief Implementation of common pseudo-spectral solver functionality.
 *
 * Implements shared code between CpuSolverPseudoRQM4 and
 * CpuSolverPseudoDiscrete, including:
 *
 * - FFT object initialization and cleanup
 * - Laplacian operator updates
 * - Stress computation with chain-model-specific factors
 *
 * Transform dispatch is handled via virtual functions in FFTMixedBC.
 *
 * @see CpuSolverPseudoBase.h for interface documentation
 */

#include <iostream>
#include <cmath>
#include <complex>

#include "CpuSolverPseudoBase.h"

//------------------------------------------------------------------------------
// Initialize shared components
//------------------------------------------------------------------------------
template <typename T>
void CpuSolverPseudoBase<T>::init_shared(ComputationBox<T>* cb, Molecules* molecules)
{
    this->cb = cb;
    this->molecules = molecules;
    this->chain_model = molecules->get_model_name();

    // Check if all BCs are periodic
    auto bc_vec = cb->get_boundary_conditions();

    // Validate that both sides of each direction have matching BC
    // (required for pseudo-spectral: FFT/DCT/DST apply to entire direction)
    int dim = cb->get_dim();
    for (int d = 0; d < dim; ++d)
    {
        if (bc_vec[2*d] != bc_vec[2*d + 1])
        {
            throw_with_line_number("Pseudo-spectral method requires matching boundary conditions on both sides of each direction. "
                "Direction " + std::to_string(d) + " has mismatched BCs. Use real-space method for mixed BCs.");
        }
    }
    is_periodic_ = true;
    for (const auto& b : bc_vec)
    {
        if (b != BoundaryCondition::PERIODIC)
        {
            is_periodic_ = false;
            break;
        }
    }

    // Create unified FFT object (handles all BCs)
    // All simulations are conducted in fixed dimensions
    dim_ = cb->get_dim();

    // Extract one BC per dimension and create appropriate FFT
    if (dim_ == 3)
    {
        std::array<int, 3> nx_arr = {cb->get_nx(0), cb->get_nx(1), cb->get_nx(2)};
        std::array<BoundaryCondition, 3> bc_arr = {bc_vec[0], bc_vec[2], bc_vec[4]};
        fft_ = new MklFFT<T, 3>(nx_arr, bc_arr);
    }
    else if (dim_ == 2)
    {
        std::array<int, 2> nx_arr = {cb->get_nx(0), cb->get_nx(1)};
        std::array<BoundaryCondition, 2> bc_arr = {bc_vec[0], bc_vec[2]};
        fft_ = new MklFFT<T, 2>(nx_arr, bc_arr);
    }
    else if (dim_ == 1)
    {
        std::array<int, 1> nx_arr = {cb->get_nx(0)};
        std::array<BoundaryCondition, 1> bc_arr = {bc_vec[0]};
        fft_ = new MklFFT<T, 1>(nx_arr, bc_arr);
    }

    // Create Pseudo object
    std::vector<BoundaryCondition> bc_per_dim;
    for (int d = 0; d < dim_; ++d)
        bc_per_dim.push_back(bc_vec[2 * d]);

    pseudo = new Pseudo<T>(
        molecules->get_bond_lengths(),
        bc_per_dim,
        cb->get_nx(), cb->get_dx(), molecules->get_ds(),
        cb->get_recip_metric());
}

//------------------------------------------------------------------------------
// Clean up shared components
//------------------------------------------------------------------------------
template <typename T>
void CpuSolverPseudoBase<T>::cleanup_shared()
{
    // FFTMixedBC has virtual destructor, so delete works polymorphically
    delete fft_;
    fft_ = nullptr;

    delete pseudo;
    pseudo = nullptr;
}

//------------------------------------------------------------------------------
// Update Laplacian operator
//------------------------------------------------------------------------------
template <typename T>
void CpuSolverPseudoBase<T>::update_laplacian_operator()
{
    try
    {
        // Extract one BC per dimension
        auto bc_vec = this->cb->get_boundary_conditions();
        std::vector<BoundaryCondition> bc_per_dim;
        for (int d = 0; d < dim_; ++d)
            bc_per_dim.push_back(bc_vec[2 * d]);

        pseudo->update(
            bc_per_dim,
            this->molecules->get_bond_lengths(),
            this->cb->get_dx(), this->molecules->get_ds(),
            this->cb->get_recip_metric());
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

//------------------------------------------------------------------------------
// Compute single segment stress
//------------------------------------------------------------------------------
template <typename T>
std::vector<T> CpuSolverPseudoBase<T>::compute_single_segment_stress(
    T* q_1, T* q_2, std::string monomer_type, bool is_half_bond_length)
{
    try
    {
        const int DIM = dim_;
        const int N_STRESS = 6;  // Full stress tensor: xx, yy, zz, xy, xz, yz
        const int M_COMPLEX = pseudo->get_total_complex_grid();
        auto bond_lengths = this->molecules->get_bond_lengths();

        double bond_length_sq;
        if (is_half_bond_length)
            bond_length_sq = 0.5 * bond_lengths[monomer_type] * bond_lengths[monomer_type];
        else
            bond_length_sq = bond_lengths[monomer_type] * bond_lengths[monomer_type];

        std::vector<T> stress(N_STRESS, 0.0);

        // Get chain-model-specific Boltzmann bond factor
        const double* _boltz_bond = get_stress_boltz_bond(monomer_type, is_half_bond_length);

        // Allocate Fourier coefficient arrays
        int coeff_size = is_periodic_ ? M_COMPLEX * 2 : M_COMPLEX;
        std::vector<double> qk_1(coeff_size);
        std::vector<double> qk_2(coeff_size);

        // Diagonal Fourier basis
        const double* _fourier_basis_x = pseudo->get_fourier_basis_x();
        const double* _fourier_basis_y = pseudo->get_fourier_basis_y();
        const double* _fourier_basis_z = pseudo->get_fourier_basis_z();
        // Cross-term Fourier basis
        const double* _fourier_basis_xy = pseudo->get_fourier_basis_xy();
        const double* _fourier_basis_xz = pseudo->get_fourier_basis_xz();
        const double* _fourier_basis_yz = pseudo->get_fourier_basis_yz();

        // Transform to Fourier space
        transform_forward(q_1, qk_1.data());
        transform_forward(q_2, qk_2.data());

        if (is_periodic_)
        {
            std::complex<double>* qk_1_complex = reinterpret_cast<std::complex<double>*>(qk_1.data());
            std::complex<double>* qk_2_complex = reinterpret_cast<std::complex<double>*>(qk_2.data());
            const int* _negative_k_idx = pseudo->get_negative_frequency_mapping();

            // Get reciprocal metric tensor for non-orthogonal correction
            // G*_ij layout: [G*_00, G*_01, G*_02, G*_11, G*_12, G*_22]
            const auto& recip_metric = this->cb->get_recip_metric();

            if (DIM == 3)
            {
                // For non-orthogonal 3D boxes:
                // stress[i] = ∂H/∂L_i needs contributions from cross-terms
                // ∂|k|²/∂a ∝ G*_00×n₀² + G*_01×n₀n₁ + G*_02×n₀n₂
                //          = fourier_basis_x + 0.5×fourier_basis_xy + 0.5×fourier_basis_xz
                // Correction factor is 0.5 for all cross-terms (because fourier_basis_xy already has factor of 2)
                double cross_xy_to_x = 0.5;
                double cross_xz_to_x = 0.5;
                double cross_xy_to_y = 0.5;
                double cross_yz_to_y = 0.5;
                double cross_xz_to_z = 0.5;
                double cross_yz_to_z = 0.5;

                for (int i = 0; i < M_COMPLEX; ++i)
                {
                    T coeff;
                    double boltz_factor = (_boltz_bond != nullptr) ? _boltz_bond[i] : 1.0;
                    if constexpr (std::is_same<T, double>::value)
                        coeff = bond_length_sq * boltz_factor * (qk_1_complex[i] * std::conj(qk_2_complex[i])).real();
                    else
                        coeff = bond_length_sq * boltz_factor * qk_1_complex[i] * qk_2_complex[_negative_k_idx[i]];

                    // Diagonal components with cross-term corrections for non-orthogonal boxes
                    stress[0] += coeff * (_fourier_basis_x[i] + cross_xy_to_x * _fourier_basis_xy[i] + cross_xz_to_x * _fourier_basis_xz[i]);
                    stress[1] += coeff * (_fourier_basis_y[i] + cross_xy_to_y * _fourier_basis_xy[i] + cross_yz_to_y * _fourier_basis_yz[i]);
                    stress[2] += coeff * (_fourier_basis_z[i] + cross_xz_to_z * _fourier_basis_xz[i] + cross_yz_to_z * _fourier_basis_yz[i]);
                    // Cross-term components (shear stress)
                    stress[3] += coeff * _fourier_basis_xy[i];  // xy
                    stress[4] += coeff * _fourier_basis_xz[i];  // xz
                    stress[5] += coeff * _fourier_basis_yz[i];  // yz
                }
            }
            else if (DIM == 2)
            {
                // For non-orthogonal 2D boxes:
                // stress[0] = ∂H/∂L_0 needs contribution from G*_01 term
                // stress[1] = ∂H/∂L_1 needs contribution from G*_01 term
                // ∂|k|²/∂a = (2π)² × (∂G*_00/∂a × n₀² + 2∂G*_01/∂a × n₀n₁)
                //          = (2π)² × (-2G*_00/a × n₀² - 2G*_01/a × n₀n₁)
                //          ∝ G*_00×n₀² + G*_01×n₀n₁
                //          = fourier_basis_x + 0.5×fourier_basis_xy
                // Similarly for ∂|k|²/∂b: ∝ G*_11×n₁² + G*_01×n₀n₁
                double cross_xy_to_x = 0.5;  // Factor to convert fourier_basis_xy contribution
                double cross_xy_to_y = 0.5;

                for (int i = 0; i < M_COMPLEX; ++i)
                {
                    T coeff;
                    double boltz_factor = (_boltz_bond != nullptr) ? _boltz_bond[i] : 1.0;
                    if constexpr (std::is_same<T, double>::value)
                        coeff = bond_length_sq * boltz_factor * (qk_1_complex[i] * std::conj(qk_2_complex[i])).real();
                    else
                        coeff = bond_length_sq * boltz_factor * qk_1_complex[i] * qk_2_complex[_negative_k_idx[i]];

                    // Diagonal components with cross-term correction for non-orthogonal boxes
                    stress[0] += coeff * (_fourier_basis_x[i] + cross_xy_to_x * _fourier_basis_xy[i]);
                    stress[1] += coeff * (_fourier_basis_y[i] + cross_xy_to_y * _fourier_basis_xy[i]);
                    stress[2] += coeff * _fourier_basis_xy[i];  // cross term (shear stress)
                }
            }
            else if (DIM == 1)
            {
                for (int i = 0; i < M_COMPLEX; ++i)
                {
                    T coeff;
                    double boltz_factor = (_boltz_bond != nullptr) ? _boltz_bond[i] : 1.0;
                    if constexpr (std::is_same<T, double>::value)
                        coeff = bond_length_sq * boltz_factor * (qk_1_complex[i] * std::conj(qk_2_complex[i])).real();
                    else
                        coeff = bond_length_sq * boltz_factor * qk_1_complex[i] * qk_2_complex[_negative_k_idx[i]];

                    stress[0] += coeff * _fourier_basis_x[i];   // lx[0] dimension
                }
            }
        }
        else
        {
            // For non-periodic BCs (real coefficients)
            // Use the same structure as periodic BC for consistency
            // Cross-terms are currently zero for non-periodic BC (orthogonal grids only)
            if (DIM == 3)
            {
                double cross_xy_to_x = 0.5;
                double cross_xz_to_x = 0.5;
                double cross_xy_to_y = 0.5;
                double cross_yz_to_y = 0.5;
                double cross_xz_to_z = 0.5;
                double cross_yz_to_z = 0.5;

                for (int i = 0; i < M_COMPLEX; ++i)
                {
                    double boltz_factor = (_boltz_bond != nullptr) ? _boltz_bond[i] : 1.0;
                    double coeff = bond_length_sq * boltz_factor * qk_1[i] * qk_2[i];

                    // Diagonal components with cross-term corrections for non-orthogonal boxes
                    stress[0] += coeff * (_fourier_basis_x[i] + cross_xy_to_x * _fourier_basis_xy[i] + cross_xz_to_x * _fourier_basis_xz[i]);
                    stress[1] += coeff * (_fourier_basis_y[i] + cross_xy_to_y * _fourier_basis_xy[i] + cross_yz_to_y * _fourier_basis_yz[i]);
                    stress[2] += coeff * (_fourier_basis_z[i] + cross_xz_to_z * _fourier_basis_xz[i] + cross_yz_to_z * _fourier_basis_yz[i]);
                    // Cross-term components (shear stress)
                    stress[3] += coeff * _fourier_basis_xy[i];
                    stress[4] += coeff * _fourier_basis_xz[i];
                    stress[5] += coeff * _fourier_basis_yz[i];
                }
            }
            else if (DIM == 2)
            {
                double cross_xy_to_x = 0.5;
                double cross_xy_to_y = 0.5;

                for (int i = 0; i < M_COMPLEX; ++i)
                {
                    double boltz_factor = (_boltz_bond != nullptr) ? _boltz_bond[i] : 1.0;
                    double coeff = bond_length_sq * boltz_factor * qk_1[i] * qk_2[i];

                    // Diagonal components with cross-term correction for non-orthogonal boxes
                    stress[0] += coeff * (_fourier_basis_x[i] + cross_xy_to_x * _fourier_basis_xy[i]);
                    stress[1] += coeff * (_fourier_basis_y[i] + cross_xy_to_y * _fourier_basis_xy[i]);
                    stress[2] += coeff * _fourier_basis_xy[i];  // cross term (shear stress)
                }
            }
            else if (DIM == 1)
            {
                for (int i = 0; i < M_COMPLEX; ++i)
                {
                    double boltz_factor = (_boltz_bond != nullptr) ? _boltz_bond[i] : 1.0;
                    double coeff = bond_length_sq * boltz_factor * qk_1[i] * qk_2[i];

                    stress[0] += coeff * _fourier_basis_x[i];
                }
            }
        }

        return stress;
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

// Explicit template instantiation
#include "TemplateInstantiations.h"
INSTANTIATE_CLASS(CpuSolverPseudoBase);
