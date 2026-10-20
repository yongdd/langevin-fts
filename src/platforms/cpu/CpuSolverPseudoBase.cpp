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
 * Transform dispatch goes through the FFT<T> interface (FftwFFT or MklFFT).
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
#include "CrysFFTBasisMapping.h"

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
    else
    {
        throw_with_line_number("Pseudo-spectral solver supports 1D, 2D, or 3D only, got dim = "
                               + std::to_string(dim_) + ".");
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
    // FFT<T> has a virtual destructor, so delete works polymorphically
    delete fft_;
    fft_ = nullptr;

    delete pseudo;
    pseudo = nullptr;
}

//------------------------------------------------------------------------------
// Space group / CrysFFT setup
//------------------------------------------------------------------------------
template <typename T>
void CpuSolverPseudoBase<T>::reset_crysfft()
{
    crysfft_mode_ = CrysFFTMode::None;
#ifdef USE_CPU_FFTW
    crysfft_pmmm_fftw_.reset();
    crysfft_recursive_fftw_.reset();
    crysfft_oblique_fftw_.reset();
#endif
#ifdef USE_CPU_MKL
    crysfft_pmmm_mkl_.reset();
    crysfft_recursive_mkl_.reset();
    crysfft_oblique_mkl_.reset();
#endif
    crysfft_full_indices_.clear();
    crysfft_reduced_indices_.clear();
    crysfft_kx2_.clear();
    crysfft_ky2_.clear();
    crysfft_kz2_.clear();
    crysfft_k_cache_lx_ = {{-1.0, -1.0, -1.0}};
    crysfft_identity_map_ = false;
}

template <typename T>
bool CpuSolverPseudoBase<T>::adopt_crysfft_mapping(
    const CrysFFTBasisMapping& map, bool physical_basis_enabled, const char* mismatch_message)
{
    if (!crysfft_mapping_usable(map, physical_basis_enabled, mismatch_message))
        return false;
    if (physical_basis_enabled)
    {
        // The fields already live on the physical grid: no index maps needed.
        crysfft_full_indices_.clear();
        crysfft_reduced_indices_.clear();
        crysfft_identity_map_ = true;
    }
    else
    {
        crysfft_full_indices_ = map.phys_to_full;
        crysfft_reduced_indices_ = map.phys_to_reduced;
        crysfft_identity_map_ = map.identity;
    }
    return true;
}

template <typename T>
void CpuSolverPseudoBase<T>::set_space_group(SpaceGroup* sg)
{
    space_group_ = sg;

    if (sg == nullptr)
    {
        if constexpr (std::is_same_v<T, double>)
            reset_crysfft();
        return;
    }

    // [real-only feature] The CrysFFT fast paths fold real fields on the
    // physical sub-grid; complex fields always use the full-grid reduced basis.
    if constexpr (std::is_same_v<T, double>)
    {
        reset_crysfft();
        if (dim_ == 3)
            init_crysfft();
    }

    // The CrysFFT maps above are snapshots: freeze the physical basis
    sg->mark_in_use();
}

template <typename T>
void CpuSolverPseudoBase<T>::init_crysfft()
{
    const auto nx = cb->get_nx();
    const std::array<int, 3> nx_logical = {nx[0], nx[1], nx[2]};
    const auto angles = cb->get_angles();
    const bool z_axis_orthogonal =
        (angles.size() >= 3 &&
         std::abs(angles[0] - M_PI / 2.0) < 1e-10 &&
         std::abs(angles[1] - M_PI / 2.0) < 1e-10);
    const auto selection = select_crysfft_mode(
        space_group_, nx_logical, dim_, is_periodic_, cb->is_orthogonal(), z_axis_orthogonal);
    if (selection.mode == CrysFFTChoice::None && !selection.can_pmmm && !selection.can_oblique_z)
        return;

    const auto lx = cb->get_lx();
    const std::array<double, 6> cell_para = {lx[0], lx[1], lx[2], angles[0], angles[1], angles[2]};
    const int M_reduced = space_group_->get_n_reduced_basis();
    const auto& full_to_reduced = space_group_->get_full_to_reduced_map();
    const bool use_m3_basis = space_group_->using_m3_physical_basis();
    const bool use_pmmm_basis = space_group_->using_pmmm_physical_basis();
    const bool use_hex_basis = space_group_->using_z_mirror_physical_basis();

    // Recursive 3m: octant sampled at even full-grid indices
    if (selection.mode == CrysFFTChoice::Recursive3m)
    {
        if (use_pmmm_basis)
            throw_with_line_number("Pmmm physical basis is enabled but recursive 3m CrysFFT is selected.");
        const auto map = build_crysfft_mapping_octant(nx_logical, full_to_reduced, M_reduced, /*even_indices=*/true);
        if (adopt_crysfft_mapping(map, use_m3_basis, "M3 physical basis does not match recursive 3m CrysFFT grid ordering."))
        {
#ifdef USE_CPU_MKL
            if (fft_backend_ == FFTBackend::MKL)
                crysfft_recursive_mkl_ = std::make_unique<MklCrysFFTRecursive3m>(nx_logical, cell_para, selection.m3_translations);
#endif
#ifdef USE_CPU_FFTW
            if (fft_backend_ == FFTBackend::FFTW)
                crysfft_recursive_fftw_ = std::make_unique<FftwCrysFFTRecursive3m>(nx_logical, cell_para, selection.m3_translations);
#endif
            crysfft_mode_ = CrysFFTMode::Recursive3m;
        }
    }

    // Pmmm DCT: octant at consecutive indices
    if (crysfft_mode_ == CrysFFTMode::None && selection.can_pmmm)
    {
        if (use_m3_basis)
            throw_with_line_number("M3 physical basis is enabled but recursive 3m CrysFFT is unavailable.");
        const auto map = build_crysfft_mapping_octant(nx_logical, full_to_reduced, M_reduced, /*even_indices=*/false);
        if (adopt_crysfft_mapping(map, use_pmmm_basis, "Pmmm physical basis does not match Pmmm CrysFFT grid ordering."))
        {
#ifdef USE_CPU_MKL
            if (fft_backend_ == FFTBackend::MKL)
                crysfft_pmmm_mkl_ = std::make_unique<MklCrysFFTPmmm>(nx_logical, cell_para);
#endif
#ifdef USE_CPU_FFTW
            if (fft_backend_ == FFTBackend::FFTW)
                crysfft_pmmm_fftw_ = std::make_unique<FftwCrysFFTPmmm>(nx_logical, cell_para);
#endif
            crysfft_mode_ = CrysFFTMode::PmmmDct;
        }
    }

    // ObliqueZ: lower z half (z-mirror)
    if (crysfft_mode_ == CrysFFTMode::None && selection.mode == CrysFFTChoice::ObliqueZ)
    {
        if (use_m3_basis || use_pmmm_basis)
            throw_with_line_number("ObliqueZ CrysFFT selected but Pmmm/M3 physical basis is enabled.");
        const int z_shift = use_hex_basis ? space_group_->get_z_mirror_shift()
                                          : selection.oblique_z_shift;
        if (use_hex_basis && z_shift != selection.oblique_z_shift)
            throw_with_line_number("Z-mirror physical basis shift does not match CrysFFT selection.");
        const auto map = build_crysfft_mapping_z_mirror(nx_logical, full_to_reduced, M_reduced, z_shift);
        if (adopt_crysfft_mapping(map, use_hex_basis, "Z-mirror physical basis does not match ObliqueZ CrysFFT grid ordering."))
        {
#ifdef USE_CPU_MKL
            if (fft_backend_ == FFTBackend::MKL)
                crysfft_oblique_mkl_ = std::make_unique<MklCrysFFTObliqueZ>(nx_logical, cell_para);
#endif
#ifdef USE_CPU_FFTW
            if (fft_backend_ == FFTBackend::FFTW)
                crysfft_oblique_fftw_ = std::make_unique<FftwCrysFFTObliqueZ>(nx_logical, cell_para);
#endif
            crysfft_mode_ = CrysFFTMode::ObliqueZ;
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
// update_dw() helper: fields on the grid the Boltzmann factors live on
//------------------------------------------------------------------------------
template <typename T>
int CpuSolverPseudoBase<T>::expand_w_for_update_dw(
    const std::map<std::string, const T*>& w_input,
    std::map<std::string, std::vector<T>>& w_full_cache,
    std::map<std::string, const T*>& w_use) const
{
    const int M_full = cb->get_total_grid();
    const int M_reduced = (space_group_ != nullptr)
        ? space_group_->get_n_reduced_basis()
        : M_full;
    const bool use_reduced_basis = (space_group_ != nullptr);
    const bool use_crysfft_path = (use_crysfft() && space_group_ != nullptr);

    w_full_cache.clear();
    w_use.clear();
    if (use_reduced_basis && !use_crysfft_path)
    {
        // Standard FFT path: exp_dw lives on the full grid, so expand the
        // reduced-basis fields. [real-only feature]
        if constexpr (std::is_same_v<T, double>)
        {
            for (const auto& item : w_input)
            {
                const std::string& monomer_type = item.first;
                std::vector<T>& w_full = w_full_cache[monomer_type];
                w_full.resize(M_full);
                space_group_->from_reduced_basis(item.second, w_full.data(), 1);
                w_use[monomer_type] = w_full.data();
            }
        }
        else
        {
            throw_with_line_number("Space group reduced basis is only supported for real fields.");
        }
    }
    else
    {
        // No space group (full grid), or CrysFFT path (exp_dw stays in the reduced basis)
        for (const auto& item : w_input)
            w_use[item.first] = item.second;
    }

    return use_crysfft_path ? M_reduced : M_full;
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
#ifdef USE_CPU_MKL
    if (use_crysfft_recursive() && crysfft_recursive_mkl_)
        return crysfft_recursive_mkl_->get_M_physical();
    if (use_crysfft_pmmm() && crysfft_pmmm_mkl_)
        return crysfft_pmmm_mkl_->get_M_physical();
    if (use_crysfft_oblique() && crysfft_oblique_mkl_)
        return crysfft_oblique_mkl_->get_M_physical();
#endif
#ifdef USE_CPU_FFTW
    if (use_crysfft_recursive() && crysfft_recursive_fftw_)
        return crysfft_recursive_fftw_->get_M_physical();
    if (use_crysfft_pmmm() && crysfft_pmmm_fftw_)
        return crysfft_pmmm_fftw_->get_M_physical();
    if (use_crysfft_oblique() && crysfft_oblique_fftw_)
        return crysfft_oblique_fftw_->get_M_physical();
#endif
    throw_with_line_number("CrysFFT requested but CrysFFT is not initialized.");
}

template <typename T>
void CpuSolverPseudoBase<T>::crysfft_set_cell_para(const std::array<double, 6>& cell_para)
{
#ifdef USE_CPU_MKL
    if (use_crysfft_recursive() && crysfft_recursive_mkl_)
        crysfft_recursive_mkl_->set_cell_para(cell_para);
    if (use_crysfft_pmmm() && crysfft_pmmm_mkl_)
        crysfft_pmmm_mkl_->set_cell_para(cell_para);
    if (use_crysfft_oblique() && crysfft_oblique_mkl_)
        crysfft_oblique_mkl_->set_cell_para(cell_para);
#endif
#ifdef USE_CPU_FFTW
    if (use_crysfft_recursive() && crysfft_recursive_fftw_)
        crysfft_recursive_fftw_->set_cell_para(cell_para);
    if (use_crysfft_pmmm() && crysfft_pmmm_fftw_)
        crysfft_pmmm_fftw_->set_cell_para(cell_para);
    if (use_crysfft_oblique() && crysfft_oblique_fftw_)
        crysfft_oblique_fftw_->set_cell_para(cell_para);
#endif
}

template <typename T>
void CpuSolverPseudoBase<T>::crysfft_set_contour_step(double coeff)
{
#ifdef USE_CPU_MKL
    if (use_crysfft_recursive() && crysfft_recursive_mkl_)
    {
        crysfft_recursive_mkl_->set_contour_step(coeff);
        return;
    }
    if (use_crysfft_pmmm() && crysfft_pmmm_mkl_)
    {
        crysfft_pmmm_mkl_->set_contour_step(coeff);
        return;
    }
    if (use_crysfft_oblique() && crysfft_oblique_mkl_)
    {
        crysfft_oblique_mkl_->set_contour_step(coeff);
        return;
    }
#endif
#ifdef USE_CPU_FFTW
    if (use_crysfft_recursive() && crysfft_recursive_fftw_)
    {
        crysfft_recursive_fftw_->set_contour_step(coeff);
        return;
    }
    if (use_crysfft_pmmm() && crysfft_pmmm_fftw_)
    {
        crysfft_pmmm_fftw_->set_contour_step(coeff);
        return;
    }
    if (use_crysfft_oblique() && crysfft_oblique_fftw_)
    {
        crysfft_oblique_fftw_->set_contour_step(coeff);
        return;
    }
#endif
    throw_with_line_number("CrysFFT requested but CrysFFT is not initialized.");
}

template <typename T>
void CpuSolverPseudoBase<T>::crysfft_diffusion(double* q_in, double* q_out) const
{
#ifdef USE_CPU_MKL
    if (use_crysfft_recursive() && crysfft_recursive_mkl_)
    {
        crysfft_recursive_mkl_->diffusion(q_in, q_out);
        return;
    }
    if (use_crysfft_pmmm() && crysfft_pmmm_mkl_)
    {
        crysfft_pmmm_mkl_->diffusion(q_in, q_out);
        return;
    }
    if (use_crysfft_oblique() && crysfft_oblique_mkl_)
    {
        crysfft_oblique_mkl_->diffusion(q_in, q_out);
        return;
    }
#endif
#ifdef USE_CPU_FFTW
    if (use_crysfft_recursive() && crysfft_recursive_fftw_)
    {
        crysfft_recursive_fftw_->diffusion(q_in, q_out);
        return;
    }
    if (use_crysfft_pmmm() && crysfft_pmmm_fftw_)
    {
        crysfft_pmmm_fftw_->diffusion(q_in, q_out);
        return;
    }
    if (use_crysfft_oblique() && crysfft_oblique_fftw_)
    {
        crysfft_oblique_fftw_->diffusion(q_in, q_out);
        return;
    }
#endif
    throw_with_line_number("CrysFFT requested but CrysFFT is not initialized.");
}

template <typename T>
void CpuSolverPseudoBase<T>::update_crysfft_k_cache()
{
    bool has_pmmm = false;
#ifdef USE_CPU_MKL
    has_pmmm = has_pmmm || crysfft_pmmm_mkl_;
#endif
#ifdef USE_CPU_FFTW
    has_pmmm = has_pmmm || crysfft_pmmm_fftw_;
#endif
    if (!use_crysfft_pmmm() || !has_pmmm)
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
 * @brief Compute stress contribution from a single bond/segment.
 *
 * Uses v⊗v dyad product components stored in fourier_basis arrays, where
 * v = 2π g⁻¹ m is the deformation vector (units 1/L², NOT the Cartesian
 * wavevector — see Pseudo::update_weighted_fourier_basis_periodic):
 * - fourier_basis_x = v₁², fourier_basis_y = v₂², fourier_basis_z = v₃²
 * - fourier_basis_xy = v₁v₂, fourier_basis_xz = v₁v₃, fourier_basis_yz = v₂v₃
 *
 * Returns the accumulated sums [V₁₁, V₂₂, V₃₃, V₁₂, V₁₃, V₂₃] where
 * V_ij = Σ_k kernel(k)·v_i·v_j. The caller (compute_stress in the
 * computation classes) converts these to lattice-parameter derivatives
 * [dH/dL₁, dH/dL₂, dH/dL₃, dH/dγ, dH/dβ, dH/dα] via the metric tensor.
 *
 * @see docs/theory/StressTensor.md for the derivation
 */
template <typename T>
void CpuSolverPseudoBase<T>::crysfft_pmmm_axis_multiplier(
    double* phys_in, double* phys_out, int axes, const double* boltz)
{
    // Σ k_d² over the axes in the bit mask (1: x, 2: y, 4: z); discrete chains
    // fold the bond factor ĝ(k) into the multiplier
    thread_local std::vector<double> weighted;
    const int M_phys = static_cast<int>(crysfft_kx2_.size());
    weighted.resize(M_phys);
    for (int i = 0; i < M_phys; ++i)
    {
        const double k_axes = ((axes & 1) ? crysfft_kx2_[i] : 0.0) + ((axes & 2) ? crysfft_ky2_[i] : 0.0)
                            + ((axes & 4) ? crysfft_kz2_[i] : 0.0);
        weighted[i] = (boltz != nullptr) ? boltz[i] * k_axes : k_axes;
    }
    const double* multiplier = weighted.data();
#ifdef USE_CPU_MKL
    if (crysfft_pmmm_mkl_)
    {
        crysfft_pmmm_mkl_->apply_multiplier(phys_in, phys_out, multiplier);
        return;
    }
#endif
#ifdef USE_CPU_FFTW
    if (crysfft_pmmm_fftw_)
    {
        crysfft_pmmm_fftw_->apply_multiplier(phys_in, phys_out, multiplier);
        return;
    }
#endif
    throw_with_line_number("Pmmm CrysFFT requested but it is not initialized.");
}

// Recursive 3m multiplier type for an axis mask (1: x, 2: y, 4: z, 3: x+y, 7: all)
template <typename MT>
static MT recursive_multiplier_type(int axes, bool with_bond)
{
    switch (axes)
    {
        case 1: return with_bond ? MT::ExpKx2  : MT::Kx2;
        case 2: return with_bond ? MT::ExpKy2  : MT::Ky2;
        case 4: return with_bond ? MT::ExpKz2  : MT::Kz2;
        case 3: return with_bond ? MT::ExpKxy2 : MT::Kxy2;
        case 7: return with_bond ? MT::ExpK2   : MT::K2;
    }
    throw_with_line_number("Unsupported CrysFFT stress axis group " + std::to_string(axes) + ".");
}

template <typename T>
bool CpuSolverPseudoBase<T>::crysfft_recursive_axis_multiplier(
    double* phys_in, double* phys_out, int axes, bool with_bond, double coeff)
{
#ifdef USE_CPU_MKL
    if (crysfft_recursive_mkl_)
    {
        crysfft_recursive_mkl_->apply_multiplier(phys_in, phys_out,
            recursive_multiplier_type<MklCrysFFTRecursive3m::MultiplierType>(axes, with_bond), with_bond ? coeff : 0.0);
        return true;
    }
#endif
#ifdef USE_CPU_FFTW
    if (crysfft_recursive_fftw_)
    {
        crysfft_recursive_fftw_->apply_multiplier(phys_in, phys_out,
            recursive_multiplier_type<FftwCrysFFTRecursive3m::MultiplierType>(axes, with_bond), with_bond ? coeff : 0.0);
        return true;
    }
#endif
    return false;
}

template <typename T>
bool CpuSolverPseudoBase<T>::compute_stress_crysfft(
    const T* q_1, const T* q_2, double bond_length_sq, const double* boltz_bond, std::vector<T>& stress)
{
    // [real-only feature] CrysFFT folds real fields on the physical sub-grid.
    if constexpr (!std::is_same_v<T, double>)
    {
        (void)q_1; (void)q_2; (void)bond_length_sq; (void)boltz_bond; (void)stress;
        return false;
    }
    else
    {
        // Only the Pmmm and recursive-3m modes have a stress fast path (ObliqueZ
        // has none). The fast path computes the diagonal sums V₁₁, V₂₂, V₃₃
        // only, which is fine because the eligible crystal systems are
        // orthogonal and never optimize angles. If angle derivatives are forced
        // (force_off_diagonal_stress_), fall back to the standard path, which
        // can compute the cross terms, instead of returning zeros for them.
        if (!(use_crysfft_pmmm() || use_crysfft_recursive())
            || space_group_ == nullptr || !is_periodic_ || dim_ != 3 || !cb->is_orthogonal()
            || this->force_off_diagonal_stress_)
            return false;

        const int M_full = cb->get_total_grid();
        const int M_reduced = space_group_->get_n_reduced_basis();
        const int M_phys = get_crysfft_physical_size();
        const auto& orbit_counts = space_group_->get_orbit_counts();

        thread_local std::vector<double> phys_q1;
        thread_local std::vector<double> phys_q2;
        thread_local std::vector<double> phys_tmp;
        thread_local std::vector<double> reduced_tmp;
        thread_local std::vector<double> boltz;
        if (static_cast<int>(phys_q1.size()) != M_phys)
        {
            phys_q1.resize(M_phys);
            phys_q2.resize(M_phys);
            phys_tmp.resize(M_phys);
        }
        if (static_cast<int>(reduced_tmp.size()) != M_reduced)
            reduced_tmp.resize(M_reduced);

        // Reduced -> physical grid
        fill_crysfft_from_reduced(q_1, phys_q1.data());
        fill_crysfft_from_reduced(q_2, phys_q2.data());

        // Discrete chains carry the bond factor ĝ(k) = exp(-b² k² Δs / 6)
        const bool with_bond = (boltz_bond != nullptr);
        const double coeff = with_bond ? bond_length_sq * molecules->get_global_ds() / 6.0 : 0.0;
        if (use_crysfft_pmmm())
        {
            // k-space tables of the DCT physical grid. This runs inside the
            // OpenMP stress loop and a refill writes shared members, so it is
            // serialized: the first thread fills the cache, the others return
            // early once the tables are valid.
            #pragma omp critical(crysfft_k_cache)
            {
                update_crysfft_k_cache();
            }
            if (with_bond)
            {
                boltz.resize(M_phys);
                for (int i = 0; i < M_phys; ++i)
                {
                    const double k2 = crysfft_kx2_[i] + crysfft_ky2_[i] + crysfft_kz2_[i];
                    boltz[i] = std::exp(-k2 * coeff);
                }
            }
        }

        // sums[d] = Σ_r n_orbit(r) q1(r) [F⁻¹ k_d² ĝ F q2](r) over the reduced basis, one
        // multiplier Σ_{d in group} k_d² per group of equivalent axes, shared equally by
        // the axes of the group (see crysfft_stress_axis_groups)
        double sums[3] = {0.0, 0.0, 0.0};
        for (const int axes : crysfft_stress_axis_groups(space_group_->get_crystal_system()))
        {
            if (use_crysfft_pmmm())
                crysfft_pmmm_axis_multiplier(phys_q2.data(), phys_tmp.data(), axes, with_bond ? boltz.data() : nullptr);
            else if (!crysfft_recursive_axis_multiplier(phys_q2.data(), phys_tmp.data(), axes, with_bond, coeff))
                return false;
            reduce_crysfft_to_reduced(phys_tmp.data(), reduced_tmp.data());
            double sum = 0.0;
            for (int i = 0; i < M_reduced; ++i)
                sum += static_cast<double>(orbit_counts[i]) * q_1[i] * reduced_tmp[i];
            const int n_axes = ((axes & 1) != 0) + ((axes & 2) != 0) + ((axes & 4) != 0);
            for (int d = 0; d < 3; ++d)
                if (axes & (1 << d))
                    sums[d] = sum / n_axes;
        }

        // The CrysFFT multipliers are Cartesian k_d² = (2πm_d/L_d)², while the
        // standard path accumulates deformation-vector components v_d² = k_d²/L_d²
        // (see Pseudo::update_weighted_fourier_basis_periodic). Divide by L_d²
        // so both paths return the same V_dd.
        const std::vector<double> lx = cb->get_lx();
        for (int d = 0; d < 3; ++d)
        {
            const double inv_L2 = 1.0 / (lx[d] * lx[d]);
            stress[d] = bond_length_sq * M_full * sums[d] * inv_L2;
        }
        return true;
    }
}

template <typename T>
void CpuSolverPseudoBase<T>::accumulate_stress_periodic(
    const double* qk_1, const double* qk_2, double bond_length_sq,
    const double* boltz_bond, bool is_orthogonal, std::vector<T>& stress) const
{
    const int DIM = dim_;
    const int M_COMPLEX = pseudo->get_total_complex_grid();
    const std::complex<double>* qk_1_complex = reinterpret_cast<const std::complex<double>*>(qk_1);
    const std::complex<double>* qk_2_complex = reinterpret_cast<const std::complex<double>*>(qk_2);
    const int* negative_k_idx = pseudo->get_negative_frequency_mapping();

    // v⊗v dyad components (deformation vector v = 2π g⁻¹ m, units 1/L²)
    const double* kk_xx = pseudo->get_fourier_basis_x();
    const double* kk_yy = pseudo->get_fourier_basis_y();
    const double* kk_zz = pseudo->get_fourier_basis_z();
    const double* kk_xy = pseudo->get_fourier_basis_xy();
    const double* kk_xz = pseudo->get_fourier_basis_xz();
    const double* kk_yz = pseudo->get_fourier_basis_yz();

    // [type-specific algorithm] Real fields store the Hermitian half-spectrum,
    // so the conjugate product is formed directly; complex fields need the -k map.
    auto kernel = [&](int i) -> T
    {
        const double boltz_factor = (boltz_bond != nullptr) ? boltz_bond[i] : 1.0;
        if constexpr (std::is_same<T, double>::value)
            return bond_length_sq * boltz_factor * (qk_1_complex[i] * std::conj(qk_2_complex[i])).real();
        else
            return bond_length_sq * boltz_factor * qk_1_complex[i] * qk_2_complex[negative_k_idx[i]];
    };

    if (DIM == 3)
    {
        for (int i = 0; i < M_COMPLEX; ++i)
        {
            const T coeff = kernel(i);
            stress[0] += coeff * kk_xx[i];  // V₁₁
            stress[1] += coeff * kk_yy[i];  // V₂₂
            stress[2] += coeff * kk_zz[i];  // V₃₃
        }
        // Cross terms only for non-orthogonal boxes (see the note in compute_single_segment_stress)
        if (!is_orthogonal)
        {
            for (int i = 0; i < M_COMPLEX; ++i)
            {
                const T coeff = kernel(i);
                stress[3] += coeff * kk_xy[i];  // V₁₂ (→ dH/dγ)
                stress[4] += coeff * kk_xz[i];  // V₁₃ (→ dH/dβ)
                stress[5] += coeff * kk_yz[i];  // V₂₃ (→ dH/dα)
            }
        }
    }
    else if (DIM == 2)
    {
        for (int i = 0; i < M_COMPLEX; ++i)
        {
            const T coeff = kernel(i);
            stress[0] += coeff * kk_xx[i];  // V₁₁
            stress[1] += coeff * kk_yy[i];  // V₂₂
        }
        // Cross term only for non-orthogonal boxes (as in 3D and on the CUDA side)
        if (!is_orthogonal)
        {
            for (int i = 0; i < M_COMPLEX; ++i)
                stress[2] += kernel(i) * kk_xy[i];  // V₁₂ (stored in index 2 for 2D)
        }
    }
    else if (DIM == 1)
    {
        // Periodic 1D: the Miller index is mapped to m1, so the stress is in fourier_basis_x
        for (int i = 0; i < M_COMPLEX; ++i)
            stress[0] += kernel(i) * kk_xx[i];
    }
}

template <typename T>
void CpuSolverPseudoBase<T>::accumulate_stress_nonperiodic(
    const double* qk_1, const double* qk_2, double bond_length_sq,
    const double* boltz_bond, bool is_orthogonal, std::vector<T>& stress) const
{
    const int DIM = dim_;
    const int M_COMPLEX = pseudo->get_total_complex_grid();
    const double* kk_xx = pseudo->get_fourier_basis_x();
    const double* kk_yy = pseudo->get_fourier_basis_y();
    const double* kk_zz = pseudo->get_fourier_basis_z();
    const double* kk_xy = pseudo->get_fourier_basis_xy();
    const double* kk_xz = pseudo->get_fourier_basis_xz();
    const double* kk_yz = pseudo->get_fourier_basis_yz();

    // Real coefficients from DCT/DST. The per-mode Parseval weights (2 for
    // interior modes, 1 for the DCT m=0 / DST k=n special modes, composed
    // across dimensions) are folded into the fourier_basis tables by
    // Pseudo::update_weighted_fourier_basis_mixed, mirroring the periodic
    // path, so no flat factor is applied here.
    auto kernel = [&](int i) -> double
    {
        const double boltz_factor = (boltz_bond != nullptr) ? boltz_bond[i] : 1.0;
        return bond_length_sq * boltz_factor * qk_1[i] * qk_2[i];
    };

    if (DIM == 3)
    {
        for (int i = 0; i < M_COMPLEX; ++i)
        {
            const double coeff = kernel(i);
            stress[0] += coeff * kk_xx[i];  // V₁₁
            stress[1] += coeff * kk_yy[i];  // V₂₂
            stress[2] += coeff * kk_zz[i];  // V₃₃
        }
        // Cross terms are zero for non-periodic BC (orthogonal grids only)
        if (!is_orthogonal)
        {
            for (int i = 0; i < M_COMPLEX; ++i)
            {
                const double coeff = kernel(i);
                stress[3] += coeff * kk_xy[i];  // V₁₂ (→ dH/dγ)
                stress[4] += coeff * kk_xz[i];  // V₁₃ (→ dH/dβ)
                stress[5] += coeff * kk_yz[i];  // V₂₃ (→ dH/dα)
            }
        }
    }
    else if (DIM == 2)
    {
        // The 2D grid is mapped to the y-z axes internally (tnx = {1, nx[0], nx[1]}),
        // so the stress data live in fourier_basis_y and fourier_basis_z
        for (int i = 0; i < M_COMPLEX; ++i)
        {
            const double coeff = kernel(i);
            stress[0] += coeff * kk_yy[i];  // V₁₁
            stress[1] += coeff * kk_zz[i];  // V₂₂
        }
        // V₁₂ only for non-orthogonal boxes (the mixed-BC tables are zero anyway)
        if (!is_orthogonal)
        {
            for (int i = 0; i < M_COMPLEX; ++i)
                stress[2] += kernel(i) * kk_yz[i];
        }
    }
    else if (DIM == 1)
    {
        // The 1D grid is mapped to the z axis internally (tnx = {1, 1, nx[0]})
        for (int i = 0; i < M_COMPLEX; ++i)
            stress[0] += kernel(i) * kk_zz[i];
    }
}

template <typename T>
const double* CpuSolverPseudoBase<T>::get_stress_boltz_bond(
    std::string /*monomer_type*/, bool /*is_half_bond_length*/) const
{
    return nullptr;   // continuous chains: the bond factor is already in the propagator
}

template <typename T>
std::vector<T> CpuSolverPseudoBase<T>::compute_single_segment_stress(
    T* q_1, T* q_2, std::string monomer_type, bool is_half_bond_length)
{
    try
    {
        const int N_STRESS = 6;  // v⊗v dyad sums: V₁₁, V₂₂, V₃₃, V₁₂, V₁₃, V₂₃
        const int M_COMPLEX = pseudo->get_total_complex_grid();
        const auto& bond_lengths = this->molecules->get_bond_lengths();
        auto bond_it = bond_lengths.find(monomer_type);
        if (bond_it == bond_lengths.end())
            throw_with_line_number("monomer_type \"" + monomer_type + "\" not found in bond_lengths.");

        double bond_length_sq = bond_it->second * bond_it->second;
        if (is_half_bond_length)
            bond_length_sq *= 0.5;

        std::vector<T> stress(N_STRESS, 0.0);

        // Chain-model-specific Boltzmann bond factor (nullptr for continuous chains)
        const double* _boltz_bond = get_stress_boltz_bond(monomer_type, is_half_bond_length);

        // CrysFFT fast path (reduced grid, no full-grid FFT)
        if (compute_stress_crysfft(q_1, q_2, bond_length_sq, _boltz_bond, stress))
            return stress;

        if constexpr (std::is_same<T, std::complex<double>>::value)
        {
            if (!is_periodic_)
                throw_with_line_number(
                    "Stress computation for complex fields with non-periodic "
                    "boundary conditions is not implemented.");
        }
        const int coeff_size = complex_coeffs() ? M_COMPLEX * 2 : M_COMPLEX;
        std::vector<double> qk_1(coeff_size);
        std::vector<double> qk_2(coeff_size);

        // ============================================================================
        // PERFORMANCE-CRITICAL: Orthogonal box optimization
        // ============================================================================
        // For orthogonal boxes (all angles = 90°), the cross-term sums (V₁₂, V₁₃, V₂₃)
        // are not computed: angle optimization is inactive for orthogonal crystal
        // systems, so those slots are simply left at zero. (For generic asymmetric
        // fields the true values need not vanish, but they are never consumed.)
        //
        // DO NOT REMOVE THIS OPTIMIZATION without benchmarking! We have experienced
        // performance regressions before when this check was accidentally removed.
        // See git history for commit 6d1dc54 which caused a regression by always
        // computing all 6 components.
        //
        // Explicit opt-in: force_off_diagonal_stress_ (set when angle optimization
        // is active) disables the skip so that angle derivatives are available
        // even when the box is at exactly 90° angles. The skip remains the default.
        // ============================================================================
        const bool is_orthogonal = this->cb->is_orthogonal() && !this->force_off_diagonal_stress_;

        // Expand reduced basis → full grid. Thread-local buffers: this function
        // is called from OpenMP parallel regions (block loops in compute_stress),
        // so no shared class member may be used as scratch space here.
        T* q_1_full = q_1;
        T* q_2_full = q_2;
        thread_local std::vector<T> q_full_1_local;
        thread_local std::vector<T> q_full_2_local;
        if (space_group_ != nullptr)
        {
            const int M_full = cb->get_total_grid();
            if (static_cast<int>(q_full_1_local.size()) != M_full)
            {
                q_full_1_local.resize(M_full);
                q_full_2_local.resize(M_full);
            }
            if constexpr (std::is_same<T, double>::value)
            {
                space_group_->from_reduced_basis(q_1, q_full_1_local.data(), 1);
                space_group_->from_reduced_basis(q_2, q_full_2_local.data(), 1);
            }
            else // complex<double>: expand as 2 interleaved real fields
            {
                space_group_->from_reduced_basis(
                    reinterpret_cast<const double*>(q_1),
                    reinterpret_cast<double*>(q_full_1_local.data()), 2);
                space_group_->from_reduced_basis(
                    reinterpret_cast<const double*>(q_2),
                    reinterpret_cast<double*>(q_full_2_local.data()), 2);
            }
            q_1_full = q_full_1_local.data();
            q_2_full = q_full_2_local.data();
        }

        // Transform to Fourier space and accumulate Σ_k kernel(k) v_i v_j
        transform_forward(q_1_full, qk_1.data());
        transform_forward(q_2_full, qk_2.data());
        if (is_periodic_)
            accumulate_stress_periodic(qk_1.data(), qk_2.data(), bond_length_sq, _boltz_bond, is_orthogonal, stress);
        else
            accumulate_stress_nonperiodic(qk_1.data(), qk_2.data(), bond_length_sq, _boltz_bond, is_orthogonal, stress);

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
