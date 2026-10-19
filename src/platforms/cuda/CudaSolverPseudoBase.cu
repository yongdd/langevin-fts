/**
 * @file CudaSolverPseudoBase.cu
 * @brief Shared implementation of the CUDA pseudo-spectral solvers.
 *
 * See CudaSolverPseudoBase.h. The space-group / CrysFFT setup below used to be
 * triplicated in the RQM4, RK2 and discrete solvers.
 */

#include <iostream>
#include <cmath>
#include <array>
#include <vector>

#include "CudaSolverPseudoBase.h"
#include "CudaPseudo.h"
#include "CudaTypeDispatch.h"
#include "CudaCrysFFT.h"
#include "CudaCrysFFTRecursive3m.h"
#include "CudaCrysFFTObliqueZ.h"
#include "CrysFFTSelector.h"
#include "SpaceGroup.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

template <typename T>
CudaSolverPseudoBase<T>::CudaSolverPseudoBase(
    ComputationBox<T>* cb,
    Molecules *molecules,
    int n_streams,
    cudaStream_t streams[MAX_STREAMS][2],
    int q_full_in_factor,
    bool with_exp_dw_half,
    bool complex_requires_periodic)
{
    try{
        this->cb = cb;
        this->molecules = molecules;
        this->pseudo = nullptr;
        this->chain_model = molecules->get_model_name();
        this->n_streams = n_streams;
        this->dim_ = cb->get_dim();
        this->q_full_in_factor_ = q_full_in_factor;

        for(int i=0; i<n_streams; i++)
        {
            this->streams[i][0] = streams[i][0];
            this->streams[i][1] = streams[i][1];
        }
        for(int i=0; i<MAX_STREAMS; i++)
        {
            d_temp_storage[i] = nullptr;
            temp_storage_bytes[i] = 0;
            d_stress_sum[i] = nullptr;
            d_stress_sum_out[i] = nullptr;
            d_q_multi[i] = nullptr;
            d_q_full_in_[i] = nullptr;
            d_q_full_out_[i] = nullptr;
            crysfft_[i] = nullptr;
            d_crysfft_in_[i] = nullptr;
            d_crysfft_out_[i] = nullptr;
        }
        space_group_ = nullptr;
        d_reduced_basis_indices_ = nullptr;
        d_full_to_reduced_map_ = nullptr;
        n_basis_ = 0;
        use_crysfft_ = false;
        crysfft_mode_ = CudaCrysFFTMode::None;
        crysfft_identity_map_ = false;
        crysfft_physical_size_ = 0;
        crysfft_reduced_size_ = 0;
        d_crysfft_phys_to_reduced_ = nullptr;
        d_crysfft_reduced_to_phys_ = nullptr;
        d_crysfft_orbit_counts_ = nullptr;

        // Boundary conditions: the pseudo-spectral method needs the same BC on
        // both faces of each direction.
        auto bc_vec = cb->get_boundary_conditions();
        for (int d = 0; d < dim_; ++d)
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
        // [type-specific algorithm] The continuous-chain solvers have no DCT/DST
        // path for complex fields; the discrete solver has.
        if constexpr (std::is_same<T, std::complex<double>>::value)
        {
            if (complex_requires_periodic && !is_periodic_)
                throw_with_line_number(
                    "Complex fields with non-periodic boundary conditions are "
                    "not yet ported to this CUDA solver; use the discrete "
                    "chain model or a CPU platform.");
        }

        // Pseudo-spectral helper (Fourier basis, Boltzmann bond factors).
        // Squeeze face BCs to one per dimension.
        {
            std::vector<BoundaryCondition> bc_dim_ctor;
            for (int d = 0; d < dim_; ++d)
                bc_dim_ctor.push_back(bc_vec[2 * d]);
            pseudo = new CudaPseudo<T>(
                molecules->get_bond_lengths(),
                bc_dim_ctor,
                cb->get_nx(), cb->get_dx(),
                cb->get_recip_metric(),
                cb->get_recip_vec());
        }
        const int M = cb->get_total_grid();
        const int M_COMPLEX = pseudo->get_total_complex_grid();

        // Register every distinct contour step and allocate the Boltzmann
        // factor buffers exp(-w ds) per (ds, monomer type).
        molecules->finalize_contour_length_mapping();
        const ContourLengthMapping& mapping = molecules->get_contour_length_mapping();
        const int n_unique_ds = mapping.get_n_unique_ds();
        for (int ds_idx = 0; ds_idx < n_unique_ds; ++ds_idx)
        {
            const double local_ds = mapping.get_ds_from_index(ds_idx);
            pseudo->add_ds_value(ds_idx, local_ds);
            for(const auto& item: molecules->get_bond_lengths())
            {
                const std::string& monomer_type = item.first;
                this->d_exp_dw[ds_idx][monomer_type] = nullptr;
                gpu_error_check(cudaMalloc((void**)&this->d_exp_dw[ds_idx][monomer_type], sizeof(T)*M));
                if (with_exp_dw_half)
                {
                    this->d_exp_dw_half[ds_idx][monomer_type] = nullptr;
                    gpu_error_check(cudaMalloc((void**)&this->d_exp_dw_half[ds_idx][monomer_type], sizeof(T)*M));
                }
            }
        }
        pseudo->finalize_ds_values();

        // Stress reduction workspace
        for(int i=0; i<n_streams; i++)
        {
            gpu_error_check(cudaMalloc((void**)&d_stress_sum[i],      sizeof(T)*M_COMPLEX));
            gpu_error_check(cudaMalloc((void**)&d_stress_sum_out[i],  sizeof(T)*1));
            gpu_error_check(cudaMalloc((void**)&d_q_multi[i],         sizeof(T)*M_COMPLEX));
        }
        for(int i=0; i<n_streams; i++)
        {
            d_temp_storage[i] = nullptr;
            temp_storage_bytes[i] = 0;
            cuda_reduce_sum(d_temp_storage[i], temp_storage_bytes[i], d_stress_sum[i], d_stress_sum_out[i], M_COMPLEX, streams[i][0]);
            gpu_error_check(cudaMalloc(&d_temp_storage[i], temp_storage_bytes[i]));
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

template <typename T>
CudaSolverPseudoBase<T>::~CudaSolverPseudoBase()
{
    cleanup_crysfft();
    delete pseudo;

    for(const auto& ds_entry: this->d_exp_dw)
        for(const auto& item: ds_entry.second)
            cudaFree(item.second);
    for(const auto& ds_entry: this->d_exp_dw_half)
        for(const auto& item: ds_entry.second)
            cudaFree(item.second);

    for(int i=0; i<n_streams; i++)
    {
        cudaFree(d_stress_sum[i]);
        cudaFree(d_stress_sum_out[i]);
        cudaFree(d_q_multi[i]);
        cudaFree(d_temp_storage[i]);
    }
    for(int i=0; i<n_streams; i++)
    {
        if (d_q_full_in_[i] != nullptr) cudaFree(d_q_full_in_[i]);
        if (d_q_full_out_[i] != nullptr) cudaFree(d_q_full_out_[i]);
    }
}

template <typename T>
void CudaSolverPseudoBase<T>::cleanup_crysfft()
{
    use_crysfft_ = false;
    crysfft_mode_ = CudaCrysFFTMode::None;
    crysfft_identity_map_ = false;
    crysfft_physical_size_ = 0;
    crysfft_reduced_size_ = 0;

    if (d_crysfft_phys_to_reduced_ != nullptr)
    {
        cudaFree(d_crysfft_phys_to_reduced_);
        d_crysfft_phys_to_reduced_ = nullptr;
    }
    if (d_crysfft_reduced_to_phys_ != nullptr)
    {
        cudaFree(d_crysfft_reduced_to_phys_);
        d_crysfft_reduced_to_phys_ = nullptr;
    }
    if (d_crysfft_orbit_counts_ != nullptr)
    {
        cudaFree(d_crysfft_orbit_counts_);
        d_crysfft_orbit_counts_ = nullptr;
    }

    for (int i = 0; i < MAX_STREAMS; i++)
    {
        if (crysfft_[i] != nullptr)
        {
            delete crysfft_[i];
            crysfft_[i] = nullptr;
        }
        if (d_crysfft_in_[i] != nullptr)
        {
            cudaFree(d_crysfft_in_[i]);
            d_crysfft_in_[i] = nullptr;
        }
        if (d_crysfft_out_[i] != nullptr)
        {
            cudaFree(d_crysfft_out_[i]);
            d_crysfft_out_[i] = nullptr;
        }
    }
}

template <typename T>
void CudaSolverPseudoBase<T>::set_space_group(
    SpaceGroup* sg,
    int* d_reduced_basis_indices,
    int* d_full_to_reduced_map,
    int n_basis)
{
    space_group_ = sg;
    d_reduced_basis_indices_ = d_reduced_basis_indices;
    d_full_to_reduced_map_ = d_full_to_reduced_map;
    n_basis_ = n_basis;

    cleanup_crysfft();

    if (sg != nullptr)
    {
        const int M = cb->get_total_grid();
        for(int i=0; i<n_streams; i++)
        {
            if (d_q_full_in_[i] == nullptr)
                gpu_error_check(cudaMalloc((void**)&d_q_full_in_[i], sizeof(T)*M*q_full_in_factor_));
            if (d_q_full_out_[i] == nullptr)
                gpu_error_check(cudaMalloc((void**)&d_q_full_out_[i], sizeof(T)*M));
        }
    }
    else
    {
        // Free work buffers if space group is disabled
        for(int i=0; i<n_streams; i++)
        {
            if (d_q_full_in_[i] != nullptr)
            {
                cudaFree(d_q_full_in_[i]);
                d_q_full_in_[i] = nullptr;
            }
            if (d_q_full_out_[i] != nullptr)
            {
                cudaFree(d_q_full_out_[i]);
                d_q_full_out_[i] = nullptr;
            }
        }
    }

    // [type-specific algorithm] The CrysFFT fast path folds real fields on the
    // physical sub-grid; complex fields always use the full-grid reduced basis.
    if constexpr (std::is_same_v<T, double>)
    {
        if (space_group_ != nullptr && dim_ == 3 && is_periodic_)
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
                const int Nx2 = nx[0] / 2;
                const int Ny2 = nx[1] / 2;
                const int Nz2 = nx[2] / 2;
                const int M_phys_pmmm = Nx2 * Ny2 * Nz2;
                const int M_phys_hex = nx[0] * nx[1] * Nz2;
                const int M_reduced = space_group_->get_n_reduced_basis();
                const auto& full_to_reduced = space_group_->get_full_to_reduced_map();
                const bool use_m3_basis = space_group_->using_m3_physical_basis();
                const bool use_pmmm_basis = space_group_->using_pmmm_physical_basis();
                const bool use_hex_basis = space_group_->using_z_mirror_physical_basis();

                std::vector<int> phys_to_reduced;
                std::vector<int> reduced_to_phys;

                auto check_identity = [&](bool even_indices) -> bool {
                    if (M_reduced != M_phys_pmmm)
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
                    phys_to_reduced.resize(M_phys_pmmm);
                    reduced_to_phys.assign(M_reduced, -1);
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
                                const int reduced_idx = full_to_reduced[full_idx];
                                phys_to_reduced[idx] = reduced_idx;
                                if (reduced_to_phys[reduced_idx] < 0)
                                    reduced_to_phys[reduced_idx] = idx;
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
                    return true;
                };

                auto check_identity_hex = [&](int z_shift) -> bool {
                    if (M_reduced != M_phys_hex)
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
                    phys_to_reduced.resize(M_phys_hex);
                    reduced_to_phys.assign(M_reduced, -1);
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
                                const int reduced_idx = full_to_reduced[full_idx];
                                phys_to_reduced[idx] = reduced_idx;
                                if (reduced_to_phys[reduced_idx] < 0)
                                    reduced_to_phys[reduced_idx] = idx;
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
                    return true;
                };

                auto finalize_crysfft = [&](CudaCrysFFTMode mode, int M_phys, bool identity_forced) {
                    crysfft_physical_size_ = M_phys;
                    crysfft_reduced_size_ = M_reduced;

                    bool identity = identity_forced;
                    if (!identity_forced)
                    {
                        identity = (M_reduced == M_phys);
                        if (identity)
                        {
                            for (int i = 0; i < M_phys; ++i)
                            {
                                if (phys_to_reduced[i] != i || reduced_to_phys[i] != i)
                                {
                                    identity = false;
                                    break;
                                }
                            }
                        }
                    }
                    crysfft_identity_map_ = identity;

                    if (!crysfft_identity_map_)
                    {
                        gpu_error_check(cudaMalloc((void**)&d_crysfft_phys_to_reduced_, sizeof(int) * M_phys));
                        gpu_error_check(cudaMalloc((void**)&d_crysfft_reduced_to_phys_, sizeof(int) * M_reduced));
                        gpu_error_check(cudaMemcpy(d_crysfft_phys_to_reduced_, phys_to_reduced.data(),
                                                   sizeof(int) * M_phys, cudaMemcpyHostToDevice));
                        gpu_error_check(cudaMemcpy(d_crysfft_reduced_to_phys_, reduced_to_phys.data(),
                                                   sizeof(int) * M_reduced, cudaMemcpyHostToDevice));
                    }

                    // Orbit counts for the CrysFFT stress fast path
                    const auto& orbit_counts = space_group_->get_orbit_counts();
                    gpu_error_check(cudaMalloc((void**)&d_crysfft_orbit_counts_, sizeof(int) * M_reduced));
                    gpu_error_check(cudaMemcpy(d_crysfft_orbit_counts_, orbit_counts.data(),
                                               sizeof(int) * M_reduced, cudaMemcpyHostToDevice));

                    auto lx = cb->get_lx();
                    auto angles = cb->get_angles();
                    std::array<double, 6> cell_para = {lx[0], lx[1], lx[2], angles[0], angles[1], angles[2]};

                    for (int i = 0; i < n_streams; i++)
                    {
                        if (mode == CudaCrysFFTMode::Recursive3m)
                            crysfft_[i] = new CudaCrysFFTRecursive3m(nx_logical, cell_para, selection.m3_translations);
                        else if (mode == CudaCrysFFTMode::ObliqueZ)
                            crysfft_[i] = new CudaCrysFFTObliqueZ(nx_logical, cell_para);
                        else
                            crysfft_[i] = new CudaCrysFFT(nx_logical, cell_para);

                        gpu_error_check(cudaMalloc((void**)&d_crysfft_in_[i], sizeof(double) * M_phys));
                        gpu_error_check(cudaMalloc((void**)&d_crysfft_out_[i], sizeof(double) * M_phys));
                    }
                    crysfft_mode_ = mode;
                    use_crysfft_ = true;
                };

                if (selection.mode == CrysFFTChoice::Recursive3m)
                {
                    if (use_pmmm_basis)
                        throw_with_line_number("Pmmm physical basis is enabled but recursive 3m CrysFFT is selected.");
                    if (use_m3_basis)
                    {
                        if (!check_identity(true))
                            throw_with_line_number("M3 physical basis does not match recursive 3m CrysFFT grid ordering.");
                        finalize_crysfft(CudaCrysFFTMode::Recursive3m, M_phys_pmmm, true);
                    }
                    else if (build_mapping(true))
                    {
                        finalize_crysfft(CudaCrysFFTMode::Recursive3m, M_phys_pmmm, false);
                    }
                }

                if (!use_crysfft_ && selection.mode == CrysFFTChoice::ObliqueZ)
                {
                    if (use_m3_basis || use_pmmm_basis)
                        throw_with_line_number("ObliqueZ CrysFFT selected but Pmmm/M3 physical basis is enabled.");

                    if (!selection.can_oblique_z)
                        throw_with_line_number("ObliqueZ CrysFFT selected but z-mirror symmetry is unavailable.");

                    const int z_shift = use_hex_basis ? space_group_->get_z_mirror_shift()
                                                      : selection.oblique_z_shift;
                    if (use_hex_basis && z_shift != selection.oblique_z_shift)
                        throw_with_line_number("Z-mirror physical basis shift does not match CrysFFT selection.");

                    if (use_hex_basis)
                    {
                        if (!check_identity_hex(z_shift))
                            throw_with_line_number("Z-mirror physical basis does not match ObliqueZ CrysFFT grid ordering.");
                        finalize_crysfft(CudaCrysFFTMode::ObliqueZ, M_phys_hex, true);
                    }
                    else if (build_mapping_hex(z_shift))
                    {
                        finalize_crysfft(CudaCrysFFTMode::ObliqueZ, M_phys_hex, false);
                    }
                }

                if (!use_crysfft_ && selection.can_pmmm)
                {
                    if (use_m3_basis)
                        throw_with_line_number("M3 physical basis is enabled but recursive 3m CrysFFT is unavailable.");
                    if (use_pmmm_basis)
                    {
                        if (!check_identity(false))
                            throw_with_line_number("Pmmm physical basis does not match Pmmm CrysFFT grid ordering.");
                        finalize_crysfft(CudaCrysFFTMode::PmmmDct, M_phys_pmmm, true);
                    }
                    else if (build_mapping(false))
                    {
                        finalize_crysfft(CudaCrysFFTMode::PmmmDct, M_phys_pmmm, false);
                    }
                }

            }
        }
    }
}

template <typename T>
void CudaSolverPseudoBase<T>::update_laplacian_operator()
{
    try{
        // Update Pseudo Fourier basis arrays and boltz_bond for new box dimensions
        // Note: local_ds values are registered once in constructor via add_ds_value()
        // pseudo->update() recomputes boltz_bond for all registered ds values
        // Squeeze face BCs to one per dimension (see constructor note).
        auto bc_vec_upd = this->cb->get_boundary_conditions();
        std::vector<BoundaryCondition> bc_dim_upd;
        for (int d = 0; d < this->cb->get_dim(); ++d)
            bc_dim_upd.push_back(bc_vec_upd[2 * d]);
        pseudo->update(
            bc_dim_upd,
            this->molecules->get_bond_lengths(),
            this->cb->get_dx(),
            this->cb->get_recip_metric(),
            this->cb->get_recip_vec());

        if (use_crysfft_)
        {
            auto lx = cb->get_lx();
            auto angles = cb->get_angles();
            std::array<double, 6> cell_para = {lx[0], lx[1], lx[2], angles[0], angles[1], angles[2]};
            for (int i = 0; i < n_streams; ++i)
            {
                if (crysfft_[i] != nullptr)
                    crysfft_[i]->set_cell_para(cell_para);
            }
        }
    }
    catch(std::exception& exc)
    {
        throw_with_line_number(exc.what());
    }
}

template <typename T>
void CudaSolverPseudoBase<T>::update_dw_impl(
    const std::string& device, const std::map<std::string, const T*>& w_input,
    bool with_half, double factor, double factor_half)
{
    try{
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();
        const int M = cb->get_total_grid();

        const bool use_reduced_basis = (space_group_ != nullptr);
        const bool use_crysfft = (use_crysfft_ && space_group_ != nullptr);
        const int M_use = use_crysfft ? n_basis_ : M;

        // Get unique ds values from ContourLengthMapping
        const ContourLengthMapping& mapping = this->molecules->get_contour_length_mapping();
        const int n_unique_ds = mapping.get_n_unique_ds();

        cudaMemcpyKind cudaMemcpyInputToDevice;
        if (device == "gpu")
            cudaMemcpyInputToDevice = cudaMemcpyDeviceToDevice;
        else if(device == "cpu")
            cudaMemcpyInputToDevice = cudaMemcpyHostToDevice;
        else
        {
            throw_with_line_number("Invalid device \"" + device + "\".");
        }

        // Compute exp_dw for each ds_index and monomer type
        for (int ds_idx = 0; ds_idx < n_unique_ds; ++ds_idx)
        {
            const double local_ds = mapping.get_ds_from_index(ds_idx);
            for(const auto& item: w_input)
            {
                const std::string& monomer_type = item.first;
                const T *w = item.second;

                if (this->d_exp_dw[ds_idx].find(monomer_type) == this->d_exp_dw[ds_idx].end())
                    throw_with_line_number("monomer_type \"" + monomer_type + "\" is not in d_exp_dw[" + std::to_string(ds_idx) + "].");
                CuDeviceData<T>* d_full = this->d_exp_dw[ds_idx][monomer_type];
                CuDeviceData<T>* d_half = with_half ? this->d_exp_dw_half[ds_idx][monomer_type] : nullptr;

                if (use_reduced_basis && device == "gpu" && !use_crysfft)
                {
                    if constexpr (std::is_same_v<T, double>)
                    {
                        // Expand reduced basis -> full grid on device
                        ker_expand_reduced_basis<<<N_BLOCKS, N_THREADS>>>(d_full, w, d_full_to_reduced_map_, M);
                        if (with_half)
                            ker_expand_reduced_basis<<<N_BLOCKS, N_THREADS>>>(d_half, w, d_full_to_reduced_map_, M);
                    }
                    else
                    {
                        throw_with_line_number("Space group reduced basis is only supported for real fields.");
                    }
                }
                else
                {
                    const T* w_full = w;
                    std::vector<T> w_full_host;
                    if (use_reduced_basis && !use_crysfft)
                    {
                        if constexpr (std::is_same_v<T, double>)
                        {
                            w_full_host.resize(M);
                            space_group_->from_reduced_basis(w, w_full_host.data(), 1);
                            w_full = w_full_host.data();
                        }
                        else
                        {
                            throw_with_line_number("Space group reduced basis is only supported for real fields.");
                        }
                    }
                    // Copy field configurations to the device
                    gpu_error_check(cudaMemcpyAsync(d_full, w_full, sizeof(T)*M_use, cudaMemcpyInputToDevice));
                    if (with_half)
                        gpu_error_check(cudaMemcpyAsync(d_half, w_full, sizeof(T)*M_use, cudaMemcpyInputToDevice));
                }

                // d_exp_dw = exp(-w * local_ds * factor); d_exp_dw_half = exp(-w * local_ds * factor_half)
                ker_exp<<<N_BLOCKS, N_THREADS>>>(d_full, d_full, 1.0, -factor*local_ds, M_use);
                if (with_half)
                    ker_exp<<<N_BLOCKS, N_THREADS>>>(d_half, d_half, 1.0, -factor_half*local_ds, M_use);
            }
        }
        gpu_error_check(cudaPeekAtLastError());
        gpu_error_check(cudaDeviceSynchronize());
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

// Explicit template instantiation
template class CudaSolverPseudoBase<double>;
template class CudaSolverPseudoBase<std::complex<double>>;
