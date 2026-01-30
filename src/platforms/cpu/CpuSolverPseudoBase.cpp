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
#include <cstring>
#include <type_traits>
#include <algorithm>

#include "CpuSolverPseudoBase.h"
#include "CrysFFTSelector.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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

    auto reset_crysfft = [&]() {
        crysfft_mode_ = CrysFFTMode::None;
        crysfft_pmmm_.reset();
        crysfft_recursive_.reset();
        crysfft_oblique_.reset();
        crysfft_full_indices_.clear();
        crysfft_reduced_indices_.clear();
        crysfft_kx2_.clear();
        crysfft_ky2_.clear();
        crysfft_kz2_.clear();
        crysfft_k_cache_lx_ = {{-1.0, -1.0, -1.0}};
        crysfft_identity_map_ = false;
    };

    if (sg != nullptr)
    {
        // Allocate full grid buffers for expand/reduce around FFT
        const int M = cb->get_total_grid();
        q_full_in_.resize(M);
        q_full_out_.resize(M);

        // Initialize crystallographic FFT when applicable
        if constexpr (std::is_same_v<T, double>)
        {
            reset_crysfft();

            if (dim_ == 3)
            {
                const auto nx = cb->get_nx();
                std::array<int, 3> nx_logical = {nx[0], nx[1], nx[2]};
                const auto angles = cb->get_angles();
                const bool z_axis_orthogonal =
                    (angles.size() >= 3 &&
                     std::abs(angles[0] - M_PI / 2.0) < 1e-10 &&
                     std::abs(angles[1] - M_PI / 2.0) < 1e-10);
                const auto selection = select_crysfft_mode(
                    space_group_, nx_logical, dim_, is_periodic_, cb->is_orthogonal(), z_axis_orthogonal);
                if (selection.mode != CrysFFTChoice::None || selection.can_pmmm || selection.can_oblique_z)
                {
                    auto lx = cb->get_lx();
                    auto angles = cb->get_angles();
                    std::array<double, 6> cell_para = {lx[0], lx[1], lx[2], angles[0], angles[1], angles[2]};

                    const int M_reduced = space_group_->get_n_reduced_basis();
                    const auto& full_to_reduced = space_group_->get_full_to_reduced_map();
                    const bool use_m3_basis = space_group_->using_m3_physical_basis();
                    const bool use_pmmm_basis = space_group_->using_pmmm_physical_basis();
                    const bool use_hex_basis = space_group_->using_z_mirror_physical_basis();

                    auto check_identity = [&](bool even_indices) -> bool {
                        const int Nx2 = nx[0] / 2;
                        const int Ny2 = nx[1] / 2;
                        const int Nz2 = nx[2] / 2;
                        const int M_phys = Nx2 * Ny2 * Nz2;
                        if (M_reduced != M_phys)
                            return false;

                        int idx = 0;
                        for (int ix = 0; ix < Nx2; ++ix)
                        {
                            const int fx = even_indices ? (2 * ix) : ix;
                            for (int iy = 0; iy < Ny2; ++iy)
                            {
                                const int fy = even_indices ? (2 * iy) : iy;
                                for (int iz = 0; iz < Nz2; ++iz)
                                {
                                    const int fz = even_indices ? (2 * iz) : iz;
                                    const int full_idx = (fx * nx[1] + fy) * nx[2] + fz;
                                    const int reduced_idx = full_to_reduced[full_idx];
                                    if (reduced_idx != idx)
                                        return false;
                                    ++idx;
                                }
                            }
                        }
                        return true;
                    };

                    auto build_mapping = [&](bool even_indices) -> bool {
                        const int Nx2 = nx[0] / 2;
                        const int Ny2 = nx[1] / 2;
                        const int Nz2 = nx[2] / 2;
                        const int M_phys = Nx2 * Ny2 * Nz2;

                        crysfft_full_indices_.resize(M_phys);
                        crysfft_reduced_indices_.resize(M_phys);
                        crysfft_identity_map_ = false;

                        std::vector<int> coverage(M_reduced, 0);
                        int idx = 0;
                        for (int ix = 0; ix < Nx2; ++ix)
                        {
                            const int fx = even_indices ? (2 * ix) : ix;
                            for (int iy = 0; iy < Ny2; ++iy)
                            {
                                const int fy = even_indices ? (2 * iy) : iy;
                                for (int iz = 0; iz < Nz2; ++iz)
                                {
                                    const int fz = even_indices ? (2 * iz) : iz;
                                    const int full_idx = (fx * nx[1] + fy) * nx[2] + fz;
                                    crysfft_full_indices_[idx] = full_idx;
                                    const int reduced_idx = full_to_reduced[full_idx];
                                    crysfft_reduced_indices_[idx] = reduced_idx;
                                    coverage[reduced_idx] += 1;
                                    ++idx;
                                }
                            }
                        }

                        for (int i = 0; i < M_reduced; ++i)
                        {
                            if (coverage[i] == 0)
                                return false;
                        }

                        if (M_reduced == M_phys)
                        {
                            bool identity = true;
                            for (int i = 0; i < M_phys; ++i)
                            {
                                if (crysfft_reduced_indices_[i] != i)
                                {
                                    identity = false;
                                    break;
                                }
                            }
                            crysfft_identity_map_ = identity;
                        }
                        return true;
                    };

                    auto check_identity_hex = [&](int z_shift) -> bool {
                        const int Nz2 = nx[2] / 2;
                        const int M_phys = nx[0] * nx[1] * Nz2;
                        if (M_reduced != M_phys)
                            return false;

                        int idx = 0;
                        for (int ix = 0; ix < nx[0]; ++ix)
                        {
                            for (int iy = 0; iy < nx[1]; ++iy)
                            {
                                for (int iz = 0; iz < Nz2; ++iz)
                                {
                                    int iz_full = iz + z_shift;
                                    if (iz_full >= nx[2])
                                        iz_full -= nx[2];
                                    const int full_idx = (ix * nx[1] + iy) * nx[2] + iz_full;
                                    const int reduced_idx = full_to_reduced[full_idx];
                                    if (reduced_idx != idx)
                                        return false;
                                    ++idx;
                                }
                            }
                        }
                        return true;
                    };

                    auto build_mapping_hex = [&](int z_shift) -> bool {
                        const int Nz2 = nx[2] / 2;
                        const int M_phys = nx[0] * nx[1] * Nz2;

                        crysfft_full_indices_.resize(M_phys);
                        crysfft_reduced_indices_.resize(M_phys);
                        crysfft_identity_map_ = false;

                        std::vector<int> coverage(M_reduced, 0);
                        int idx = 0;
                        for (int ix = 0; ix < nx[0]; ++ix)
                        {
                            for (int iy = 0; iy < nx[1]; ++iy)
                            {
                                for (int iz = 0; iz < Nz2; ++iz)
                                {
                                    int iz_full = iz + z_shift;
                                    if (iz_full >= nx[2])
                                        iz_full -= nx[2];
                                    const int full_idx = (ix * nx[1] + iy) * nx[2] + iz_full;
                                    crysfft_full_indices_[idx] = full_idx;
                                    const int reduced_idx = full_to_reduced[full_idx];
                                    crysfft_reduced_indices_[idx] = reduced_idx;
                                    coverage[reduced_idx] += 1;
                                    ++idx;
                                }
                            }
                        }

                        for (int i = 0; i < M_reduced; ++i)
                        {
                            if (coverage[i] == 0)
                                return false;
                        }

                        if (M_reduced == M_phys)
                        {
                            bool identity = true;
                            for (int i = 0; i < M_phys; ++i)
                            {
                                if (crysfft_reduced_indices_[i] != i)
                                {
                                    identity = false;
                                    break;
                                }
                            }
                            crysfft_identity_map_ = identity;
                        }
                        return true;
                    };

                    if (selection.mode == CrysFFTChoice::Recursive3m)
                    {
                        if (use_pmmm_basis)
                            throw_with_line_number("Pmmm physical basis is enabled but recursive 3m CrysFFT is selected.");
                        crysfft_recursive_ = std::make_unique<FftwCrysFFTRecursive3m>(nx_logical, cell_para, selection.m3_translations);
                        if (use_m3_basis)
                        {
                            if (!check_identity(true))
                                throw_with_line_number("M3 physical basis does not match recursive 3m CrysFFT grid ordering.");
                            crysfft_identity_map_ = true;
                            crysfft_mode_ = CrysFFTMode::Recursive3m;
                        }
                        else if (build_mapping(true))
                        {
                            crysfft_mode_ = CrysFFTMode::Recursive3m;
                        }
                        else
                        {
                            crysfft_recursive_.reset();
                            crysfft_full_indices_.clear();
                            crysfft_reduced_indices_.clear();
                            crysfft_identity_map_ = false;
                        }
                    }

                    if (crysfft_mode_ == CrysFFTMode::None && selection.can_pmmm)
                    {
                        if (use_m3_basis)
                            throw_with_line_number("M3 physical basis is enabled but recursive 3m CrysFFT is unavailable.");
                        crysfft_pmmm_ = std::make_unique<FftwCrysFFTPmmm>(nx_logical, cell_para);
                        if (use_pmmm_basis)
                        {
                            if (!check_identity(false))
                                throw_with_line_number("Pmmm physical basis does not match Pmmm CrysFFT grid ordering.");
                            crysfft_identity_map_ = true;
                            crysfft_mode_ = CrysFFTMode::PmmmDct;
                        }
                        else if (build_mapping(false))
                        {
                            crysfft_mode_ = CrysFFTMode::PmmmDct;
                        }
                        else
                        {
                            crysfft_pmmm_.reset();
                            crysfft_full_indices_.clear();
                            crysfft_reduced_indices_.clear();
                            crysfft_identity_map_ = false;
                        }
                    }

                    const bool want_hex = (selection.mode == CrysFFTChoice::ObliqueZ);
                    if (crysfft_mode_ == CrysFFTMode::None && want_hex)
                    {
                        if (use_m3_basis || use_pmmm_basis)
                            throw_with_line_number("ObliqueZ CrysFFT selected but Pmmm/M3 physical basis is enabled.");

                        const int z_shift = use_hex_basis ? space_group_->get_z_mirror_shift()
                                                          : selection.oblique_z_shift;
                        if (use_hex_basis && z_shift != selection.oblique_z_shift)
                            throw_with_line_number("Z-mirror physical basis shift does not match CrysFFT selection.");

                        crysfft_oblique_ = std::make_unique<FftwCrysFFTObliqueZ>(nx_logical, cell_para);
                        if (use_hex_basis)
                        {
                            if (!check_identity_hex(z_shift))
                                throw_with_line_number("Z-mirror physical basis does not match ObliqueZ CrysFFT grid ordering.");
                            crysfft_identity_map_ = true;
                            crysfft_mode_ = CrysFFTMode::ObliqueZ;
                        }
                        else if (build_mapping_hex(z_shift))
                        {
                            crysfft_mode_ = CrysFFTMode::ObliqueZ;
                        }
                        else
                        {
                            crysfft_oblique_.reset();
                            crysfft_full_indices_.clear();
                            crysfft_reduced_indices_.clear();
                            crysfft_identity_map_ = false;
                        }
                    }
                }
            }
        }
    }
    else
    {
        // Free buffers when space group is disabled
        q_full_in_.clear();
        q_full_in_.shrink_to_fit();
        q_full_out_.clear();
        q_full_out_.shrink_to_fit();

        if constexpr (std::is_same_v<T, double>)
        {
            reset_crysfft();
        }
    }
}

//------------------------------------------------------------------------------
// ds value helpers
//------------------------------------------------------------------------------
template <typename T>
double CpuSolverPseudoBase<T>::get_ds_value(int ds_index) const
{
    auto it = ds_values_.find(ds_index);
    if (it == ds_values_.end())
        throw_with_line_number("ds_index " + std::to_string(ds_index) + " not registered in solver.");
    return it->second;
}

template <typename T>
double CpuSolverPseudoBase<T>::get_effective_diffusion_coeff(
    const std::string& monomer_type, int ds_index, bool half_step) const
{
    double ds = get_ds_value(ds_index);
    if (half_step)
        ds *= 0.5;

    auto& bond_lengths = molecules->get_bond_lengths();
    auto it = bond_lengths.find(monomer_type);
    if (it == bond_lengths.end())
        throw_with_line_number("monomer_type \"" + monomer_type + "\" not found in bond_lengths.");
    double bond_length_sq = it->second * it->second;
    return bond_length_sq * ds / 6.0;
}

//------------------------------------------------------------------------------
// CrysFFT grid helpers (reduced basis <-> physical grid)
//------------------------------------------------------------------------------
template <typename T>
void CpuSolverPseudoBase<T>::fill_crysfft_from_reduced(const double* reduced_in, double* phys_out) const
{
    if (!use_crysfft())
        throw_with_line_number("CrysFFT requested but CrysFFT is not initialized.");

    const int M_phys = get_crysfft_physical_size();
    if (crysfft_identity_map_)
    {
        std::memcpy(phys_out, reduced_in, sizeof(double) * M_phys);
        return;
    }
    for (int i = 0; i < M_phys; ++i)
    {
        phys_out[i] = reduced_in[crysfft_reduced_indices_[i]];
    }
}

template <typename T>
void CpuSolverPseudoBase<T>::reduce_crysfft_to_reduced(const double* phys_in, double* reduced_out) const
{
    if (!use_crysfft())
        throw_with_line_number("CrysFFT requested but CrysFFT is not initialized.");

    const int M_phys = get_crysfft_physical_size();
    if (crysfft_identity_map_)
    {
        std::memcpy(reduced_out, phys_in, sizeof(double) * M_phys);
        return;
    }

    const int M_reduced = space_group_->get_n_reduced_basis();
    std::fill(reduced_out, reduced_out + M_reduced, 0.0);

    for (int i = 0; i < M_phys; ++i)
    {
        reduced_out[crysfft_reduced_indices_[i]] = phys_in[i];
    }
}

//------------------------------------------------------------------------------
// CrysFFT helpers
//------------------------------------------------------------------------------
template <typename T>
int CpuSolverPseudoBase<T>::get_crysfft_physical_size() const
{
    if (use_crysfft_recursive() && crysfft_recursive_)
        return crysfft_recursive_->get_M_physical();
    if (use_crysfft_pmmm() && crysfft_pmmm_)
        return crysfft_pmmm_->get_M_physical();
    if (use_crysfft_oblique() && crysfft_oblique_)
        return crysfft_oblique_->get_M_physical();
    throw_with_line_number("CrysFFT requested but CrysFFT is not initialized.");
}

template <typename T>
void CpuSolverPseudoBase<T>::crysfft_set_cell_para(const std::array<double, 6>& cell_para)
{
    if (use_crysfft_recursive() && crysfft_recursive_)
        crysfft_recursive_->set_cell_para(cell_para);
    if (use_crysfft_pmmm() && crysfft_pmmm_)
        crysfft_pmmm_->set_cell_para(cell_para);
    if (use_crysfft_oblique() && crysfft_oblique_)
        crysfft_oblique_->set_cell_para(cell_para);
}

template <typename T>
void CpuSolverPseudoBase<T>::crysfft_set_contour_step(double coeff)
{
    if (use_crysfft_recursive() && crysfft_recursive_)
    {
        crysfft_recursive_->set_contour_step(coeff);
        return;
    }
    if (use_crysfft_pmmm() && crysfft_pmmm_)
    {
        crysfft_pmmm_->set_contour_step(coeff);
        return;
    }
    if (use_crysfft_oblique() && crysfft_oblique_)
    {
        crysfft_oblique_->set_contour_step(coeff);
        return;
    }
    throw_with_line_number("CrysFFT requested but CrysFFT is not initialized.");
}

template <typename T>
void CpuSolverPseudoBase<T>::crysfft_diffusion(double* q_in, double* q_out) const
{
    if (use_crysfft_recursive() && crysfft_recursive_)
    {
        crysfft_recursive_->diffusion(q_in, q_out);
        return;
    }
    if (use_crysfft_pmmm() && crysfft_pmmm_)
    {
        crysfft_pmmm_->diffusion(q_in, q_out);
        return;
    }
    if (use_crysfft_oblique() && crysfft_oblique_)
    {
        crysfft_oblique_->diffusion(q_in, q_out);
        return;
    }
    throw_with_line_number("CrysFFT requested but CrysFFT is not initialized.");
}

template <typename T>
void CpuSolverPseudoBase<T>::update_crysfft_k_cache()
{
    if (!use_crysfft_pmmm() || !crysfft_pmmm_)
        return;

    if (dim_ != 3 || !is_periodic_ || !cb->is_orthogonal())
        return;

    auto lx = cb->get_lx();
    if (crysfft_k_cache_lx_[0] == lx[0] &&
        crysfft_k_cache_lx_[1] == lx[1] &&
        crysfft_k_cache_lx_[2] == lx[2] &&
        !crysfft_kx2_.empty())
    {
        return;
    }

    crysfft_k_cache_lx_[0] = lx[0];
    crysfft_k_cache_lx_[1] = lx[1];
    crysfft_k_cache_lx_[2] = lx[2];

    const std::vector<int>& nx = cb->get_nx();
    const int Nx2 = nx[0] / 2;
    const int Ny2 = nx[1] / 2;
    const int Nz2 = nx[2] / 2;
    const int M_phys = get_crysfft_physical_size();

    crysfft_kx2_.assign(M_phys, 0.0);
    crysfft_ky2_.assign(M_phys, 0.0);
    crysfft_kz2_.assign(M_phys, 0.0);

    const double factor_x = 2.0 * M_PI / lx[0];
    const double factor_y = 2.0 * M_PI / lx[1];
    const double factor_z = 2.0 * M_PI / lx[2];

    int idx = 0;
    for (int ix = 0; ix < Nx2; ++ix)
    {
        double kx = ix * factor_x;
        double kx2 = kx * kx;
        for (int iy = 0; iy < Ny2; ++iy)
        {
            double ky = iy * factor_y;
            double ky2 = ky * ky;
            for (int iz = 0; iz < Nz2; ++iz)
            {
                double kz = iz * factor_z;
                double kz2 = kz * kz;
                crysfft_kx2_[idx] = kx2;
                crysfft_ky2_[idx] = ky2;
                crysfft_kz2_[idx] = kz2;
                ++idx;
            }
        }
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

        if constexpr (std::is_same_v<T, double>)
        {
            if (use_crysfft())
            {
                auto lx = cb->get_lx();
                auto angles = cb->get_angles();
                crysfft_set_cell_para({lx[0], lx[1], lx[2], angles[0], angles[1], angles[2]});
            }
            if (use_crysfft_pmmm())
            {
                update_crysfft_k_cache();
            }
        }
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

        // =====================================================================
        // CrysFFT stress path (reduced grid, no full-grid FFT)
        // =====================================================================
        if constexpr (std::is_same_v<T, double>)
        {
            if (use_crysfft() && space_group_ != nullptr && is_periodic_ && dim_ == 3 && cb->is_orthogonal())
            {
                const int M_full = cb->get_total_grid();
                const int M_reduced = space_group_->get_n_reduced_basis();
                const int M_phys = get_crysfft_physical_size();
                const auto& orbit_counts = space_group_->get_orbit_counts();

                thread_local std::vector<double> phys_q1;
                thread_local std::vector<double> phys_q2;
                thread_local std::vector<double> phys_tmp;
                thread_local std::vector<double> reduced_tmp;

                if (static_cast<int>(phys_q1.size()) != M_phys)
                {
                    phys_q1.resize(M_phys);
                    phys_q2.resize(M_phys);
                    phys_tmp.resize(M_phys);
                }
                if (static_cast<int>(reduced_tmp.size()) != M_reduced)
                {
                    reduced_tmp.resize(M_reduced);
                }

                // Reduced -> physical grid
                fill_crysfft_from_reduced(q_1, phys_q1.data());
                fill_crysfft_from_reduced(q_2, phys_q2.data());

                if (use_crysfft_pmmm())
                {
                    // Ensure k-space cache for DCT physical grid
                    update_crysfft_k_cache();

                    auto accumulate_component = [&](const std::vector<double>& multiplier) -> double
                    {
                        crysfft_pmmm_->apply_multiplier(phys_q2.data(), phys_tmp.data(), multiplier.data());
                        reduce_crysfft_to_reduced(phys_tmp.data(), reduced_tmp.data());
                        double sum = 0.0;
                        for (int i = 0; i < M_reduced; ++i)
                            sum += static_cast<double>(orbit_counts[i]) * q_1[i] * reduced_tmp[i];
                        return sum;
                    };

                    if (_boltz_bond == nullptr)
                    {
                        // Continuous chains: no Boltzmann bond factor
                        double sum_xx = accumulate_component(crysfft_kx2_);
                        double sum_yy = accumulate_component(crysfft_ky2_);
                        double sum_zz = accumulate_component(crysfft_kz2_);

                        stress[0] = bond_length_sq * M_full * sum_xx;
                        stress[1] = bond_length_sq * M_full * sum_yy;
                        stress[2] = bond_length_sq * M_full * sum_zz;
                    }
                    else
                    {
                        // Discrete chains: include Boltzmann bond factor in multiplier
                        const double ds = molecules->get_global_ds();
                        const double coeff = bond_length_sq * ds / 6.0;

                        thread_local std::vector<double> boltz;
                        thread_local std::vector<double> multiplier;

                        if (static_cast<int>(boltz.size()) != M_phys)
                        {
                            boltz.resize(M_phys);
                            multiplier.resize(M_phys);
                        }

                        for (int i = 0; i < M_phys; ++i)
                        {
                            double k2 = crysfft_kx2_[i] + crysfft_ky2_[i] + crysfft_kz2_[i];
                            boltz[i] = std::exp(-k2 * coeff);
                        }

                        for (int i = 0; i < M_phys; ++i)
                            multiplier[i] = boltz[i] * crysfft_kx2_[i];
                        double sum_xx = accumulate_component(multiplier);

                        for (int i = 0; i < M_phys; ++i)
                            multiplier[i] = boltz[i] * crysfft_ky2_[i];
                        double sum_yy = accumulate_component(multiplier);

                        for (int i = 0; i < M_phys; ++i)
                            multiplier[i] = boltz[i] * crysfft_kz2_[i];
                        double sum_zz = accumulate_component(multiplier);

                        stress[0] = bond_length_sq * M_full * sum_xx;
                        stress[1] = bond_length_sq * M_full * sum_yy;
                        stress[2] = bond_length_sq * M_full * sum_zz;
                    }

                    return stress;
                }
                else if (use_crysfft_recursive())
                {
                    auto accumulate_component = [&](FftwCrysFFTRecursive3m::MultiplierType type, double coeff) -> double
                    {
                        crysfft_recursive_->apply_multiplier(phys_q2.data(), phys_tmp.data(), type, coeff);
                        reduce_crysfft_to_reduced(phys_tmp.data(), reduced_tmp.data());
                        double sum = 0.0;
                        for (int i = 0; i < M_reduced; ++i)
                            sum += static_cast<double>(orbit_counts[i]) * q_1[i] * reduced_tmp[i];
                        return sum;
                    };

                    if (_boltz_bond == nullptr)
                    {
                        double sum_xx = accumulate_component(FftwCrysFFTRecursive3m::MultiplierType::Kx2, 0.0);
                        double sum_yy = accumulate_component(FftwCrysFFTRecursive3m::MultiplierType::Ky2, 0.0);
                        double sum_zz = accumulate_component(FftwCrysFFTRecursive3m::MultiplierType::Kz2, 0.0);

                        stress[0] = bond_length_sq * M_full * sum_xx;
                        stress[1] = bond_length_sq * M_full * sum_yy;
                        stress[2] = bond_length_sq * M_full * sum_zz;
                    }
                    else
                    {
                        const double ds = molecules->get_global_ds();
                        const double coeff = bond_length_sq * ds / 6.0;

                        double sum_xx = accumulate_component(FftwCrysFFTRecursive3m::MultiplierType::ExpKx2, coeff);
                        double sum_yy = accumulate_component(FftwCrysFFTRecursive3m::MultiplierType::ExpKy2, coeff);
                        double sum_zz = accumulate_component(FftwCrysFFTRecursive3m::MultiplierType::ExpKz2, coeff);

                        stress[0] = bond_length_sq * M_full * sum_xx;
                        stress[1] = bond_length_sq * M_full * sum_yy;
                        stress[2] = bond_length_sq * M_full * sum_zz;
                    }

                    return stress;
                }
            }
        }

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
