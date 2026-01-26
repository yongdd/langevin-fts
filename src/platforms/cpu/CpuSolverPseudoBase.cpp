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
void CpuSolverPseudoBase<T>::init_shared(ComputationBox<T>* cb, Molecules* molecules, FFTBackend backend)
{
    this->cb = cb;
    this->molecules = molecules;
    this->chain_model = molecules->get_model_name();
    this->fft_backend_ = backend;

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

    // Create FFT object using factory (handles all BCs and backends)
    // All simulations are conducted in fixed dimensions
    dim_ = cb->get_dim();

    // Extract one BC per dimension and create appropriate FFT
    if (dim_ == 3)
    {
        std::array<int, 3> nx_arr = {cb->get_nx(0), cb->get_nx(1), cb->get_nx(2)};
        std::array<BoundaryCondition, 3> bc_arr = {bc_vec[0], bc_vec[2], bc_vec[4]};
        fft_ = createFFT<T, 3>(nx_arr, bc_arr, backend);
    }
    else if (dim_ == 2)
    {
        std::array<int, 2> nx_arr = {cb->get_nx(0), cb->get_nx(1)};
        std::array<BoundaryCondition, 2> bc_arr = {bc_vec[0], bc_vec[2]};
        fft_ = createFFT<T, 2>(nx_arr, bc_arr, backend);
    }
    else if (dim_ == 1)
    {
        std::array<int, 1> nx_arr = {cb->get_nx(0)};
        std::array<BoundaryCondition, 1> bc_arr = {bc_vec[0]};
        fft_ = createFFT<T, 1>(nx_arr, bc_arr, backend);
    }

    // Create Pseudo object
    std::vector<BoundaryCondition> bc_per_dim;
    for (int d = 0; d < dim_; ++d)
        bc_per_dim.push_back(bc_vec[2 * d]);

    pseudo = new Pseudo<T>(
        molecules->get_bond_lengths(),
        bc_per_dim,
        cb->get_nx(), cb->get_dx(),
        cb->get_recip_metric(),
        cb->get_recip_vec());
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
// Set space group for reduced basis
//------------------------------------------------------------------------------
template <typename T>
void CpuSolverPseudoBase<T>::set_space_group(SpaceGroup* sg)
{
    space_group_ = sg;

    if (sg != nullptr)
    {
        // Allocate full grid buffers for expand/reduce around FFT
        const int M = cb->get_total_grid();
        q_full_in_.resize(M);
        q_full_out_.resize(M);
    }
    else
    {
        // Free buffers when space group is disabled
        q_full_in_.clear();
        q_full_in_.shrink_to_fit();
        q_full_out_.clear();
        q_full_out_.shrink_to_fit();
    }
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
            this->cb->get_dx(),
            this->cb->get_recip_metric(),
            this->cb->get_recip_vec());
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

//------------------------------------------------------------------------------
// Compute single segment stress
//------------------------------------------------------------------------------
/**
 * @brief Compute stress tensor contribution from a single bond/segment.
 *
 * Uses k⊗k dyad product components stored in fourier_basis arrays:
 * - fourier_basis_x = k_x² (Cartesian)
 * - fourier_basis_y = k_y² (Cartesian)
 * - fourier_basis_z = k_z² (Cartesian)
 * - fourier_basis_xy = k_x × k_y (Cartesian)
 * - fourier_basis_xz = k_x × k_z (Cartesian)
 * - fourier_basis_yz = k_y × k_z (Cartesian)
 *
 * Returns Cartesian stress tensor components [σ_xx, σ_yy, σ_zz, σ_xy, σ_xz, σ_yz].
 */
template <typename T>
std::vector<T> CpuSolverPseudoBase<T>::compute_single_segment_stress(
    T* q_1, T* q_2, std::string monomer_type, bool is_half_bond_length)
{
    try
    {
        const int DIM = dim_;
        const int N_STRESS = 6;  // Cartesian stress tensor: xx, yy, zz, xy, xz, yz
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

        // k⊗k dyad components (Cartesian)
        const double* kk_xx = pseudo->get_fourier_basis_x();
        const double* kk_yy = pseudo->get_fourier_basis_y();
        const double* kk_zz = pseudo->get_fourier_basis_z();
        const double* kk_xy = pseudo->get_fourier_basis_xy();
        const double* kk_xz = pseudo->get_fourier_basis_xz();
        const double* kk_yz = pseudo->get_fourier_basis_yz();

        // ============================================================================
        // PERFORMANCE-CRITICAL: Orthogonal box optimization
        // ============================================================================
        // For orthogonal boxes (all angles = 90°), the cross-terms (σ_xy, σ_xz, σ_yz)
        // are mathematically zero and do not need to be computed. This optimization
        // skips unnecessary calculations for the common case of orthogonal boxes.
        //
        // DO NOT REMOVE THIS OPTIMIZATION without benchmarking! We have experienced
        // performance regressions before when this check was accidentally removed.
        // See git history for commit 6d1dc54 which caused a regression by always
        // computing all 6 components.
        // ============================================================================
        const bool is_orthogonal = this->cb->is_orthogonal();

        // Determine pointers based on space group
        T* q_1_full = q_1;
        T* q_2_full = q_2;

        if (space_group_ != nullptr)
        {
            // Expand reduced basis → full grid
            if constexpr (std::is_same<T, double>::value)
            {
                space_group_->from_reduced_basis(q_1, q_full_in_.data(), 1);
                space_group_->from_reduced_basis(q_2, q_full_out_.data(), 1);
            }
            else // complex<double>: expand as 2 interleaved real fields
            {
                space_group_->from_reduced_basis(
                    reinterpret_cast<const double*>(q_1),
                    reinterpret_cast<double*>(q_full_in_.data()), 2);
                space_group_->from_reduced_basis(
                    reinterpret_cast<const double*>(q_2),
                    reinterpret_cast<double*>(q_full_out_.data()), 2);
            }
            q_1_full = q_full_in_.data();
            q_2_full = q_full_out_.data();
        }

        // Transform to Fourier space
        transform_forward(q_1_full, qk_1.data());
        transform_forward(q_2_full, qk_2.data());

        if (is_periodic_)
        {
            std::complex<double>* qk_1_complex = reinterpret_cast<std::complex<double>*>(qk_1.data());
            std::complex<double>* qk_2_complex = reinterpret_cast<std::complex<double>*>(qk_2.data());
            const int* _negative_k_idx = pseudo->get_negative_frequency_mapping();

            if (DIM == 3)
            {
                // Diagonal components (always computed)
                for (int i = 0; i < M_COMPLEX; ++i)
                {
                    T coeff;
                    double boltz_factor = (_boltz_bond != nullptr) ? _boltz_bond[i] : 1.0;
                    if constexpr (std::is_same<T, double>::value)
                        coeff = bond_length_sq * boltz_factor * (qk_1_complex[i] * std::conj(qk_2_complex[i])).real();
                    else
                        coeff = bond_length_sq * boltz_factor * qk_1_complex[i] * qk_2_complex[_negative_k_idx[i]];

                    // Cartesian stress tensor diagonal components from k⊗k dyad
                    stress[0] += coeff * kk_xx[i];  // σ_xx
                    stress[1] += coeff * kk_yy[i];  // σ_yy
                    stress[2] += coeff * kk_zz[i];  // σ_zz
                }

                // Cross-terms: only compute for non-orthogonal boxes (triclinic lattices)
                // For orthogonal boxes, these are mathematically zero.
                if (!is_orthogonal)
                {
                    for (int i = 0; i < M_COMPLEX; ++i)
                    {
                        T coeff;
                        double boltz_factor = (_boltz_bond != nullptr) ? _boltz_bond[i] : 1.0;
                        if constexpr (std::is_same<T, double>::value)
                            coeff = bond_length_sq * boltz_factor * (qk_1_complex[i] * std::conj(qk_2_complex[i])).real();
                        else
                            coeff = bond_length_sq * boltz_factor * qk_1_complex[i] * qk_2_complex[_negative_k_idx[i]];

                        stress[3] += coeff * kk_xy[i];  // σ_xy
                        stress[4] += coeff * kk_xz[i];  // σ_xz
                        stress[5] += coeff * kk_yz[i];  // σ_yz
                    }
                }
            }
            else if (DIM == 2)
            {
                for (int i = 0; i < M_COMPLEX; ++i)
                {
                    T coeff;
                    double boltz_factor = (_boltz_bond != nullptr) ? _boltz_bond[i] : 1.0;
                    if constexpr (std::is_same<T, double>::value)
                        coeff = bond_length_sq * boltz_factor * (qk_1_complex[i] * std::conj(qk_2_complex[i])).real();
                    else
                        coeff = bond_length_sq * boltz_factor * qk_1_complex[i] * qk_2_complex[_negative_k_idx[i]];

                    // Cartesian stress tensor components for 2D
                    stress[0] += coeff * kk_xx[i];  // σ_xx
                    stress[1] += coeff * kk_yy[i];  // σ_yy
                    stress[2] += coeff * kk_xy[i];  // σ_xy (stored in index 2 for 2D)
                }
            }
            else if (DIM == 1)
            {
                // For periodic 1D: Miller index is mapped to m1, so stress is in fourier_basis_x
                for (int i = 0; i < M_COMPLEX; ++i)
                {
                    T coeff;
                    double boltz_factor = (_boltz_bond != nullptr) ? _boltz_bond[i] : 1.0;
                    if constexpr (std::is_same<T, double>::value)
                        coeff = bond_length_sq * boltz_factor * (qk_1_complex[i] * std::conj(qk_2_complex[i])).real();
                    else
                        coeff = bond_length_sq * boltz_factor * qk_1_complex[i] * qk_2_complex[_negative_k_idx[i]];

                    stress[0] += coeff * kk_xx[i];  // σ_xx (Miller index maps to m1 for periodic)
                }
            }
        }
        else
        {
            // For non-periodic BCs (real coefficients from DCT/DST)
            // Unlike FFT which has conjugate pairs (±m modes contributing 2× each),
            // DCT/DST modes are purely real and each mode contributes once.
            // The factor of 2 accounts for the Parseval relation for real transforms.
            // Cross-terms are zero for non-periodic BC (orthogonal grids only)
            const double FACTOR = 2.0;

            if (DIM == 3)
            {
                // Diagonal components (always computed)
                for (int i = 0; i < M_COMPLEX; ++i)
                {
                    double boltz_factor = (_boltz_bond != nullptr) ? _boltz_bond[i] : 1.0;
                    double coeff = FACTOR * bond_length_sq * boltz_factor * qk_1[i] * qk_2[i];

                    // Cartesian stress tensor diagonal components from k⊗k dyad
                    stress[0] += coeff * kk_xx[i];  // σ_xx
                    stress[1] += coeff * kk_yy[i];  // σ_yy
                    stress[2] += coeff * kk_zz[i];  // σ_zz
                }

                // Cross-terms: only compute for non-orthogonal boxes (triclinic lattices)
                // For orthogonal boxes, these are mathematically zero.
                if (!is_orthogonal)
                {
                    for (int i = 0; i < M_COMPLEX; ++i)
                    {
                        double boltz_factor = (_boltz_bond != nullptr) ? _boltz_bond[i] : 1.0;
                        double coeff = FACTOR * bond_length_sq * boltz_factor * qk_1[i] * qk_2[i];

                        stress[3] += coeff * kk_xy[i];  // σ_xy
                        stress[4] += coeff * kk_xz[i];  // σ_xz
                        stress[5] += coeff * kk_yz[i];  // σ_yz
                    }
                }
            }
            else if (DIM == 2)
            {
                // Note: 2D grid is mapped to y-z axes internally (tnx = {1, nx[0], nx[1]})
                // so stress data is stored in fourier_basis_y and fourier_basis_z
                for (int i = 0; i < M_COMPLEX; ++i)
                {
                    double boltz_factor = (_boltz_bond != nullptr) ? _boltz_bond[i] : 1.0;
                    double coeff = FACTOR * bond_length_sq * boltz_factor * qk_1[i] * qk_2[i];

                    // Cartesian stress tensor components for 2D (mapped to y-z internally)
                    stress[0] += coeff * kk_yy[i];  // σ_xx (stored in y due to internal mapping)
                    stress[1] += coeff * kk_zz[i];  // σ_yy (stored in z due to internal mapping)
                    stress[2] += coeff * kk_yz[i];  // σ_xy (stored in yz due to internal mapping)
                }
            }
            else if (DIM == 1)
            {
                // Note: 1D grid is mapped to z-axis internally (tnx = {1, 1, nx[0]})
                // so the stress data is stored in fourier_basis_z (kk_zz)
                for (int i = 0; i < M_COMPLEX; ++i)
                {
                    double boltz_factor = (_boltz_bond != nullptr) ? _boltz_bond[i] : 1.0;
                    double coeff = FACTOR * bond_length_sq * boltz_factor * qk_1[i] * qk_2[i];

                    stress[0] += coeff * kk_zz[i];  // σ_xx (stored in z due to internal mapping)
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
