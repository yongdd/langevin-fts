/**
 * @file CudaSolverPseudoRK2.cu
 * @brief CUDA pseudo-spectral solver for continuous Gaussian chains using RK2.
 *
 * Implements RK2 (Rasmussen-Kalosakas 2nd-order) for advancing chain propagators
 * using cuFFT for Fourier transforms and multiple CUDA streams for concurrent
 * computation.
 *
 * **RK2 (Operator Splitting):**
 *
 * Single full step with Strang splitting:
 *     q(s+ds) = exp(-w·ds/2) · FFT⁻¹[ exp(-k²b²ds/6) · FFT[ exp(-w·ds/2) · q(s) ] ]
 *
 * **Comparison to RQM4:**
 *
 * RK2 uses only a single full step (2 FFTs) while RQM4 combines full and half
 * steps via Richardson extrapolation (6 FFTs). RK2 is faster but only
 * 2nd-order accurate (vs 4th-order for RQM4).
 *
 * **Template Instantiations:**
 *
 * - CudaSolverPseudoRK2<double>: Real field solver
 * - CudaSolverPseudoRK2<std::complex<double>>: Complex field solver
 *
 * @see CudaPseudo for Boltzmann factors
 * @see CudaComputationContinuous for propagator orchestration
 */

#include <iostream>
#include <cmath>
#include <numbers>
#include <array>
#include <thrust/reduce.h>

#include "CudaPseudo.h"
#include "CudaSolverPseudoRK2.h"
#include "CudaCrysFFT.h"
#include "CudaCrysFFTRecursive3m.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

template <typename T>
CudaSolverPseudoRK2<T>::CudaSolverPseudoRK2(
    ComputationBox<T>* cb,
    Molecules *molecules,
    int n_streams,
    cudaStream_t streams[MAX_STREAMS][2],
    [[maybe_unused]] bool reduce_memory)
{
    try{
        this->cb = cb;
        this->molecules = molecules;
        this->chain_model = molecules->get_model_name();
        this->n_streams = n_streams;
        this->dim_ = cb->get_dim();
        this->space_group_ = nullptr;
        this->d_reduced_basis_indices_ = nullptr;
        this->d_full_to_reduced_map_ = nullptr;
        this->n_basis_ = 0;
        this->use_crysfft_ = false;
        this->crysfft_mode_ = CudaCrysFFTMode::None;
        this->crysfft_identity_map_ = false;
        this->crysfft_physical_size_ = 0;
        this->crysfft_reduced_size_ = 0;
        this->d_crysfft_phys_to_reduced_ = nullptr;
        this->d_crysfft_reduced_to_phys_ = nullptr;
        for (int i = 0; i < MAX_STREAMS; i++)
        {
            this->d_q_full_in_[i] = nullptr;
            this->d_q_full_out_[i] = nullptr;
            this->crysfft_[i] = nullptr;
            this->d_crysfft_in_[i] = nullptr;
            this->d_crysfft_out_[i] = nullptr;
        }

        // Initialize fft_ pointers to nullptr
        for(int i=0; i<MAX_STREAMS; i++)
            this->fft_[i] = nullptr;

        // Check if all BCs are periodic
        auto bc_vec = cb->get_boundary_conditions();

        // Validate that both sides of each direction have matching BC
        for (int d = 0; d < this->dim_; ++d)
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

        pseudo = new CudaPseudo<T>(
            molecules->get_bond_lengths(),
            cb->get_boundary_conditions(),
            cb->get_nx(), cb->get_dx(),
            cb->get_recip_metric(),
            cb->get_recip_vec());

        const int M = cb->get_total_grid();
        const int M_COMPLEX = pseudo->get_total_complex_grid();

        // Copy streams
        for(int i=0; i<n_streams; i++)
        {
            this->streams[i][0] = streams[i][0];
            this->streams[i][1] = streams[i][1];
        }

        // Ensure ContourLengthMapping is finalized before using it
        molecules->finalize_contour_length_mapping();

        // Get unique ds values from ContourLengthMapping
        const ContourLengthMapping& mapping = molecules->get_contour_length_mapping();
        int n_unique_ds = mapping.get_n_unique_ds();

        // Create exp_dw for each ds_index and monomer_type
        // Note: RK2 doesn't need exp_dw_half (no Richardson extrapolation)
        // Also register local_ds values with Pseudo for boltz_bond computation
        for (int ds_idx = 0; ds_idx < n_unique_ds; ++ds_idx)
        {
            double local_ds = mapping.get_ds_from_index(ds_idx);
            pseudo->add_ds_value(ds_idx, local_ds);

            for(const auto& item: molecules->get_bond_lengths())
            {
                std::string monomer_type = item.first;
                this->d_exp_dw     [ds_idx][monomer_type] = nullptr;
                this->d_exp_dw_half[ds_idx][monomer_type] = nullptr;  // Allocate for compatibility
                gpu_error_check(cudaMalloc((void**)&this->d_exp_dw     [ds_idx][monomer_type], sizeof(T)*M));
                gpu_error_check(cudaMalloc((void**)&this->d_exp_dw_half[ds_idx][monomer_type], sizeof(T)*M));
            }
        }

        // Finalize ds values to allocate boltz_bond arrays
        // (update_laplacian_operator will compute the actual values)
        pseudo->finalize_ds_values();

        // Initialize cuFFT plans to 0
        for(int i=0; i<n_streams; i++)
        {
            plan_for_one[i] = 0;
            plan_bak_one[i] = 0;
            d_rk_in_1[i] = nullptr;
            d_rk_in_2[i] = nullptr;
        }

        if (is_periodic_)
        {
            // Create cuFFT plans for periodic BC
            const int NRANK{cb->get_dim()};
            int total_grid[NRANK];

            if(cb->get_dim() == 3)
            {
                total_grid[0] = cb->get_nx(0);
                total_grid[1] = cb->get_nx(1);
                total_grid[2] = cb->get_nx(2);
            }
            else if(cb->get_dim() == 2)
            {
                total_grid[0] = cb->get_nx(0);
                total_grid[1] = cb->get_nx(1);
            }
            else if(cb->get_dim() == 1)
            {
                total_grid[0] = cb->get_nx(0);
            }

            cufftType cufft_forward;
            cufftType cufft_backward;
            if constexpr (std::is_same<T, double>::value)
            {
                cufft_forward = CUFFT_D2Z;
                cufft_backward = CUFFT_Z2D;
            }
            else
            {
                cufft_forward = CUFFT_Z2Z;
                cufft_backward = CUFFT_Z2Z;
            }

            for(int i=0; i<n_streams; i++)
            {
                cufftPlanMany(&plan_for_one[i], NRANK, total_grid, nullptr, 1, 0, nullptr, 1, 0, cufft_forward, 1);
                cufftPlanMany(&plan_bak_one[i], NRANK, total_grid, nullptr, 1, 0, nullptr, 1, 0, cufft_backward, 1);
                cufftSetStream(plan_for_one[i], streams[i][0]);
                cufftSetStream(plan_bak_one[i], streams[i][0]);
            }
        }
        else
        {
            // Create per-stream CudaFFT objects for non-periodic BC (DCT/DST)
            for(int i=0; i<n_streams; i++)
            {
                if (dim_ == 3)
                {
                    std::array<int, 3> nx_arr = {cb->get_nx(0), cb->get_nx(1), cb->get_nx(2)};
                    std::array<BoundaryCondition, 3> bc_arr = {bc_vec[0], bc_vec[2], bc_vec[4]};
                    fft_[i] = new CudaFFT<T, 3>(nx_arr, bc_arr);
                }
                else if (dim_ == 2)
                {
                    std::array<int, 2> nx_arr = {cb->get_nx(0), cb->get_nx(1)};
                    std::array<BoundaryCondition, 2> bc_arr = {bc_vec[0], bc_vec[2]};
                    fft_[i] = new CudaFFT<T, 2>(nx_arr, bc_arr);
                }
                else if (dim_ == 1)
                {
                    std::array<int, 1> nx_arr = {cb->get_nx(0)};
                    std::array<BoundaryCondition, 1> bc_arr = {bc_vec[0]};
                    fft_[i] = new CudaFFT<T, 1>(nx_arr, bc_arr);
                }
            }

            // Allocate real coefficient buffers for non-periodic BC
            for(int i=0; i<n_streams; i++)
            {
                gpu_error_check(cudaMalloc((void**)&d_rk_in_1[i], sizeof(double)*M_COMPLEX));
                gpu_error_check(cudaMalloc((void**)&d_rk_in_2[i], sizeof(double)*M_COMPLEX));
            }
        }

        // Allocate memory for pseudo-spectral: advance_propagator()
        for(int i=0; i<n_streams; i++)
        {
            gpu_error_check(cudaMalloc((void**)&d_q_step_1[i], sizeof(T)*M));
            gpu_error_check(cudaMalloc((void**)&d_q_step_2[i], sizeof(T)*M));
            gpu_error_check(cudaMalloc((void**)&d_qk_in_1[i], sizeof(cuDoubleComplex)*M_COMPLEX));
            gpu_error_check(cudaMalloc((void**)&d_qk_in_2[i], sizeof(cuDoubleComplex)*M_COMPLEX));
        }
        for(int i=0; i<n_streams; i++)
        {
            gpu_error_check(cudaMalloc((void**)&d_stress_sum[i],      sizeof(T)*M_COMPLEX));
            gpu_error_check(cudaMalloc((void**)&d_stress_sum_out[i],  sizeof(T)*1));
            gpu_error_check(cudaMalloc((void**)&d_q_multi[i],         sizeof(T)*M_COMPLEX));
        }

        // Allocate memory for cub reduction sum
        for(int i=0; i<n_streams; i++)
        {
            d_temp_storage[i] = nullptr;
            temp_storage_bytes[i] = 0;
            if constexpr (std::is_same<T, double>::value)
                cub::DeviceReduce::Sum(d_temp_storage[i], temp_storage_bytes[i], d_stress_sum[i], d_stress_sum_out[i], M_COMPLEX, streams[i][0]);
            else
                cub::DeviceReduce::Reduce(d_temp_storage[i], temp_storage_bytes[i], d_stress_sum[i], d_stress_sum_out[i], M_COMPLEX, ComplexSumOp(), CuDeviceData<T>{0.0,0.0}, streams[i][0]);
            gpu_error_check(cudaMalloc(&d_temp_storage[i], temp_storage_bytes[i]));
        }

        // update_laplacian_operator() handles registration of local_ds values
        update_laplacian_operator();
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

template <typename T>
CudaSolverPseudoRK2<T>::~CudaSolverPseudoRK2()
{
    cleanup_crysfft();

    delete pseudo;

    // Clean up FFT resources
    if (is_periodic_)
    {
        for(int i=0; i<n_streams; i++)
        {
            if (plan_for_one[i] != 0) cufftDestroy(plan_for_one[i]);
            if (plan_bak_one[i] != 0) cufftDestroy(plan_bak_one[i]);
        }
    }
    else
    {
        // Delete per-stream CudaFFT objects
        for(int i=0; i<n_streams; i++)
        {
            if (fft_[i] != nullptr)
            {
                delete fft_[i];
                fft_[i] = nullptr;
            }
        }

        // Free real coefficient buffers
        for(int i=0; i<n_streams; i++)
        {
            if (d_rk_in_1[i] != nullptr) cudaFree(d_rk_in_1[i]);
            if (d_rk_in_2[i] != nullptr) cudaFree(d_rk_in_2[i]);
        }
    }

    // Free d_exp_dw nested maps
    for(const auto& ds_entry: this->d_exp_dw)
        for(const auto& item: ds_entry.second)
            cudaFree(item.second);
    for(const auto& ds_entry: this->d_exp_dw_half)
        for(const auto& item: ds_entry.second)
            cudaFree(item.second);

    // Free workspace arrays
    for(int i=0; i<n_streams; i++)
    {
        cudaFree(d_q_step_1[i]);
        cudaFree(d_q_step_2[i]);
        cudaFree(d_qk_in_1[i]);
        cudaFree(d_qk_in_2[i]);
    }

    for(int i=0; i<n_streams; i++)
    {
        cudaFree(d_stress_sum[i]);
        cudaFree(d_stress_sum_out[i]);
        cudaFree(d_q_multi[i]);
        cudaFree(d_temp_storage[i]);
        if (d_q_full_in_[i] != nullptr)
            cudaFree(d_q_full_in_[i]);
        if (d_q_full_out_[i] != nullptr)
            cudaFree(d_q_full_out_[i]);
    }
}

template <typename T>
void CudaSolverPseudoRK2<T>::cleanup_crysfft()
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
void CudaSolverPseudoRK2<T>::set_space_group(
    SpaceGroup* sg,
    int* d_reduced_basis_indices,
    int* d_full_to_reduced_map,
    int n_basis)
{
    space_group_ = sg;
    d_reduced_basis_indices_ = d_reduced_basis_indices;
    d_full_to_reduced_map_ = d_full_to_reduced_map;
    n_basis_ = n_basis;

    const int M = cb->get_total_grid();
    if (space_group_ != nullptr)
    {
        for (int i = 0; i < n_streams; i++)
        {
            if (d_q_full_in_[i] == nullptr)
                gpu_error_check(cudaMalloc((void**)&d_q_full_in_[i], sizeof(T)*M));
            if (d_q_full_out_[i] == nullptr)
                gpu_error_check(cudaMalloc((void**)&d_q_full_out_[i], sizeof(T)*M));
        }
    }

    cleanup_crysfft();

    if constexpr (std::is_same_v<T, double>)
    {
        if (space_group_ != nullptr && dim_ == 3 && is_periodic_ && cb->is_orthogonal())
        {
            const auto nx = cb->get_nx();
            const bool even_grid = (nx[0] % 2 == 0 && nx[1] % 2 == 0 && nx[2] % 2 == 0);
            if (even_grid)
            {
                std::array<double, 9> trans_part = {0,0,0, 0,0,0, 0,0,0};
                const bool has_3m = space_group_->get_m3_translations(trans_part);
                const bool has_pmmm = space_group_->has_mirror_planes_xyz();
                const bool recursive_ok = ((nx[2] / 2) % 8) == 0;

                const int Nx2 = nx[0] / 2;
                const int Ny2 = nx[1] / 2;
                const int Nz2 = nx[2] / 2;
                const int M_phys = Nx2 * Ny2 * Nz2;
                const int M_reduced = space_group_->get_n_reduced_basis();
                const auto& full_to_reduced = space_group_->get_full_to_reduced_map();

                std::vector<int> phys_to_reduced;
                std::vector<int> reduced_to_phys;

                auto build_mapping = [&](bool even_indices) -> bool {
                    phys_to_reduced.resize(M_phys);
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

                auto finalize_crysfft = [&](CudaCrysFFTMode mode) {
                    crysfft_physical_size_ = M_phys;
                    crysfft_reduced_size_ = M_reduced;

                    bool identity = (M_reduced == M_phys);
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

                    std::array<int, 3> nx_logical = {nx[0], nx[1], nx[2]};
                    auto lx = cb->get_lx();
                    std::array<double, 6> cell_para = {lx[0], lx[1], lx[2], M_PI/2, M_PI/2, M_PI/2};

                    for (int i = 0; i < n_streams; i++)
                    {
                        if (mode == CudaCrysFFTMode::Recursive3m)
                            crysfft_[i] = new CudaCrysFFTRecursive3m(nx_logical, cell_para, trans_part);
                        else
                            crysfft_[i] = new CudaCrysFFT(nx_logical, cell_para);
                        gpu_error_check(cudaMalloc((void**)&d_crysfft_in_[i], sizeof(double) * M_phys));
                        gpu_error_check(cudaMalloc((void**)&d_crysfft_out_[i], sizeof(double) * M_phys));
                    }
                    crysfft_mode_ = mode;
                    use_crysfft_ = true;
                };

                if (has_3m && recursive_ok)
                {
                    if (build_mapping(true))
                        finalize_crysfft(CudaCrysFFTMode::Recursive3m);
                }

                if (!use_crysfft_ && has_pmmm)
                {
                    if (build_mapping(false))
                        finalize_crysfft(CudaCrysFFTMode::PmmmDct);
                }
            }
        }
    }
}

template <typename T>
void CudaSolverPseudoRK2<T>::update_laplacian_operator()
{
    try{
        // Update Pseudo Fourier basis arrays and boltz_bond for new box dimensions
        // Note: local_ds values are registered once in constructor via add_ds_value()
        // pseudo->update() recomputes boltz_bond for all registered ds values
        pseudo->update(
            this->cb->get_boundary_conditions(),
            this->molecules->get_bond_lengths(),
            this->cb->get_dx(),
            this->cb->get_recip_metric(),
            this->cb->get_recip_vec());

        if (use_crysfft_)
        {
            auto lx = cb->get_lx();
            std::array<double, 6> cell_para = {lx[0], lx[1], lx[2], M_PI/2, M_PI/2, M_PI/2};
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
void CudaSolverPseudoRK2<T>::update_dw(std::string device, std::map<std::string, const T*> w_input)
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
        int n_unique_ds = mapping.get_n_unique_ds();

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
            double local_ds = mapping.get_ds_from_index(ds_idx);

            for(const auto& item: w_input)
            {
                std::string monomer_type = item.first;
                const T *w = item.second;

                if (this->d_exp_dw[ds_idx].find(monomer_type) == this->d_exp_dw[ds_idx].end())
                    throw_with_line_number("monomer_type \"" + monomer_type + "\" is not in d_exp_dw[" + std::to_string(ds_idx) + "].");

                if (use_reduced_basis && device == "gpu" && !use_crysfft)
                {
                    if constexpr (std::is_same_v<T, double>)
                    {
                        ker_expand_reduced_basis<<<N_BLOCKS, N_THREADS>>>(
                            this->d_exp_dw[ds_idx][monomer_type], w, d_full_to_reduced_map_, M);
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

                    gpu_error_check(cudaMemcpyAsync(
                        this->d_exp_dw[ds_idx][monomer_type], w_full,
                        sizeof(T)*M_use, cudaMemcpyInputToDevice));
                }

                // Compute d_exp_dw = exp(-w * local_ds * 0.5)
                ker_exp<<<N_BLOCKS, N_THREADS>>>
                    (this->d_exp_dw[ds_idx][monomer_type], this->d_exp_dw[ds_idx][monomer_type], 1.0, -0.50*local_ds, M_use);
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

// Advance propagator using RK2 (single full step)
template <typename T>
void CudaSolverPseudoRK2<T>::advance_propagator(
        const int STREAM,
        CuDeviceData<T> *d_q_in, CuDeviceData<T> *d_q_out,
        std::string monomer_type, double *d_q_mask, int ds_index)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = cb->get_total_grid();
        const int M_COMPLEX = pseudo->get_total_complex_grid();

        CuDeviceData<T> *_d_exp_dw = this->d_exp_dw[ds_index][monomer_type];
        const double* _d_boltz_bond = pseudo->get_boltz_bond(monomer_type, ds_index);

        const bool use_crysfft = (use_crysfft_ && space_group_ != nullptr);
        if (use_crysfft)
        {
            if constexpr (!std::is_same_v<T, double>)
            {
                throw_with_line_number("CrysFFT path is only supported for real fields.");
            }
            else
            {
                const ContourLengthMapping& mapping = this->molecules->get_contour_length_mapping();
                double local_ds = mapping.get_ds_from_index(ds_index);
                auto bond_lengths = this->molecules->get_bond_lengths();
                double bond_length_sq = bond_lengths[monomer_type] * bond_lengths[monomer_type];
                double coeff_full = bond_length_sq * local_ds / 6.0;

                const int M_phys = crysfft_physical_size_;
                if (crysfft_identity_map_)
                {
                    ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                        d_crysfft_in_[STREAM], d_q_in, _d_exp_dw, 1.0, M_phys);
                    gpu_error_check(cudaPeekAtLastError());

                    crysfft_[STREAM]->set_contour_step(coeff_full);
                    crysfft_[STREAM]->diffusion(d_crysfft_in_[STREAM], d_crysfft_out_[STREAM], streams[STREAM][0]);

                    ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                        d_q_out, d_crysfft_out_[STREAM], _d_exp_dw, 1.0, M_phys);
                    gpu_error_check(cudaPeekAtLastError());

                    if (d_q_mask != nullptr)
                    {
                        ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                            d_q_out, d_q_out, d_q_mask, 1.0, M_phys);
                        gpu_error_check(cudaPeekAtLastError());
                    }
                    return;
                }
                ker_expand_reduced_basis<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                    d_crysfft_in_[STREAM], d_q_in, d_crysfft_phys_to_reduced_, M_phys);
                gpu_error_check(cudaPeekAtLastError());

                ker_multi_map<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                    d_crysfft_in_[STREAM], _d_exp_dw, d_crysfft_phys_to_reduced_, M_phys);
                gpu_error_check(cudaPeekAtLastError());

                crysfft_[STREAM]->set_contour_step(coeff_full);
                crysfft_[STREAM]->diffusion(d_crysfft_in_[STREAM], d_crysfft_out_[STREAM], streams[STREAM][0]);

                ker_multi_map<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                    d_crysfft_out_[STREAM], _d_exp_dw, d_crysfft_phys_to_reduced_, M_phys);
                gpu_error_check(cudaPeekAtLastError());

                ker_reduce_to_basis<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                    d_q_out, d_crysfft_out_[STREAM], d_crysfft_reduced_to_phys_, n_basis_);
                gpu_error_check(cudaPeekAtLastError());

                if (d_q_mask != nullptr)
                {
                    ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                        d_q_out, d_q_out, d_q_mask, 1.0, n_basis_);
                    gpu_error_check(cudaPeekAtLastError());
                }
                return;
            }
        }

        CuDeviceData<T> *fft_in = d_q_in;
        CuDeviceData<T> *fft_out = d_q_out;
        if (space_group_ != nullptr)
        {
            if constexpr (std::is_same_v<T, double>)
            {
                ker_expand_reduced_basis<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                    d_q_full_in_[STREAM], d_q_in, d_full_to_reduced_map_, M);
                gpu_error_check(cudaPeekAtLastError());
                fft_in = d_q_full_in_[STREAM];
                fft_out = d_q_full_out_[STREAM];
            }
            else
            {
                throw_with_line_number("Space group reduced basis is only supported for real fields.");
            }
        }

        if (is_periodic_)
        {
            // ============================================================
            // Periodic BC: Use cuFFT with complex coefficients
            // ============================================================

            // Step 1: Evaluate exp(-w*ds/2) in real space
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                d_q_step_1[STREAM], fft_in, _d_exp_dw, 1.0, M);
            gpu_error_check(cudaPeekAtLastError());

            // Step 2: Execute a forward FFT
            if constexpr (std::is_same<T, double>::value)
                cufftExecD2Z(plan_for_one[STREAM], d_q_step_1[STREAM], d_qk_in_1[STREAM]);
            else
                cufftExecZ2Z(plan_for_one[STREAM], d_q_step_1[STREAM], d_qk_in_1[STREAM], CUFFT_FORWARD);
            gpu_error_check(cudaPeekAtLastError());

            // Step 3: Multiply exp(-k^2 ds/6) in fourier space
            ker_multi_complex_real<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                d_qk_in_1[STREAM], _d_boltz_bond, 1.0, M_COMPLEX);
            gpu_error_check(cudaPeekAtLastError());

            // Step 4: Execute a backward FFT
            if constexpr (std::is_same<T, double>::value)
                cufftExecZ2D(plan_bak_one[STREAM], d_qk_in_1[STREAM], fft_out);
            else
                cufftExecZ2Z(plan_bak_one[STREAM], d_qk_in_1[STREAM], fft_out, CUFFT_INVERSE);
            gpu_error_check(cudaPeekAtLastError());

            // Step 5: Evaluate exp(-w*ds/2) in real space and normalize
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                fft_out, fft_out, _d_exp_dw, 1.0/static_cast<double>(M), M);
            gpu_error_check(cudaPeekAtLastError());
        }
        else
        {
            // ============================================================
            // Non-periodic BC: Use CudaFFT (DCT/DST) with real coefficients
            // ============================================================

            // Cast device pointers for FFT interface
            T* d_q_step = reinterpret_cast<T*>(d_q_step_1[STREAM]);

            // Step 1: Evaluate exp(-w*ds/2) in real space
            ker_multi<<<N_BLOCKS, N_THREADS>>>(d_q_step_1[STREAM], fft_in, _d_exp_dw, 1.0, M);
            gpu_error_check(cudaPeekAtLastError());

            // Step 2: Forward transform (DCT/DST)
            fft_[STREAM]->forward(d_q_step, d_rk_in_1[STREAM]);

            // Step 3: Multiply by Boltzmann factor (real coefficients)
            ker_multi<<<N_BLOCKS, N_THREADS>>>(d_rk_in_1[STREAM], d_rk_in_1[STREAM], _d_boltz_bond, 1.0, M_COMPLEX);
            gpu_error_check(cudaPeekAtLastError());

            // Step 4: Backward transform (inverse DCT/DST) - normalization included
            fft_[STREAM]->backward(d_rk_in_1[STREAM], reinterpret_cast<T*>(fft_out));

            // Step 5: Evaluate exp(-w*ds/2) in real space
            ker_multi<<<N_BLOCKS, N_THREADS>>>(fft_out, fft_out, _d_exp_dw, 1.0, M);
            gpu_error_check(cudaPeekAtLastError());
        }

        // Multiply mask
        if (d_q_mask != nullptr)
        {
            double* d_q_mask_full = d_q_mask;
            if (space_group_ != nullptr)
            {
                if constexpr (std::is_same_v<T, double>)
                {
                    ker_expand_reduced_basis<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                        d_q_full_in_[STREAM], d_q_mask, d_full_to_reduced_map_, M);
                    gpu_error_check(cudaPeekAtLastError());
                    d_q_mask_full = d_q_full_in_[STREAM];
                }
                else
                {
                    throw_with_line_number("Space group reduced basis is only supported for real fields.");
                }
            }
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(fft_out, fft_out, d_q_mask_full, 1.0, M);
            gpu_error_check(cudaPeekAtLastError());
        }

        if (space_group_ != nullptr)
        {
            ker_reduce_to_basis<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                d_q_out, fft_out, d_reduced_basis_indices_, n_basis_);
            gpu_error_check(cudaPeekAtLastError());
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

template <typename T>
void CudaSolverPseudoRK2<T>::compute_single_segment_stress(
    const int STREAM,
    CuDeviceData<T> *d_q_pair, CuDeviceData<T>  *d_segment_stress,
    std::string monomer_type, [[maybe_unused]] bool is_half_bond_length)
{
    try{
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int DIM = this->cb->get_dim();
        const int M = cb->get_total_grid();
        const int M_COMPLEX = pseudo->get_total_complex_grid();

        auto bond_lengths = this->molecules->get_bond_lengths();
        double bond_length_sq = bond_lengths[monomer_type]*bond_lengths[monomer_type];

        const double* _d_fourier_basis_x = pseudo->get_fourier_basis_x();
        const double* _d_fourier_basis_y = pseudo->get_fourier_basis_y();
        const double* _d_fourier_basis_z = pseudo->get_fourier_basis_z();
        const double* _d_fourier_basis_xy = pseudo->get_fourier_basis_xy();
        const double* _d_fourier_basis_xz = pseudo->get_fourier_basis_xz();
        const double* _d_fourier_basis_yz = pseudo->get_fourier_basis_yz();
        const int* _d_negative_k_idx = pseudo->get_negative_frequency_mapping();

        if (is_periodic_)
        {
            // Periodic BC: Execute a forward FFT (batched for 2 fields)
            // Copy to workspace for batched FFT
            if (space_group_ != nullptr)
            {
                if constexpr (std::is_same_v<T, double>)
                {
                    ker_expand_reduced_basis<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                        d_q_step_1[STREAM], d_q_pair, d_full_to_reduced_map_, M);
                    ker_expand_reduced_basis<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                        d_q_step_2[STREAM], &d_q_pair[n_basis_], d_full_to_reduced_map_, M);
                    gpu_error_check(cudaPeekAtLastError());
                }
                else
                {
                    throw_with_line_number("Space group reduced basis is only supported for real fields.");
                }
            }
            else
            {
                gpu_error_check(cudaMemcpyAsync(d_q_step_1[STREAM], d_q_pair, sizeof(T)*M, cudaMemcpyDeviceToDevice, streams[STREAM][0]));
                gpu_error_check(cudaMemcpyAsync(d_q_step_2[STREAM], &d_q_pair[M], sizeof(T)*M, cudaMemcpyDeviceToDevice, streams[STREAM][0]));
            }

            // Forward FFT for first field
            if constexpr (std::is_same<T, double>::value)
                cufftExecD2Z(plan_for_one[STREAM], d_q_step_1[STREAM], d_qk_in_1[STREAM]);
            else
                cufftExecZ2Z(plan_for_one[STREAM], d_q_step_1[STREAM], d_qk_in_1[STREAM], CUFFT_FORWARD);
            gpu_error_check(cudaPeekAtLastError());

            // Forward FFT for second field
            if constexpr (std::is_same<T, double>::value)
                cufftExecD2Z(plan_for_one[STREAM], d_q_step_2[STREAM], d_qk_in_2[STREAM]);
            else
                cufftExecZ2Z(plan_for_one[STREAM], d_q_step_2[STREAM], d_qk_in_2[STREAM], CUFFT_FORWARD);
            gpu_error_check(cudaPeekAtLastError());

            // Multiply two propagators in the fourier spaces
            if constexpr (std::is_same<T, double>::value)
                ker_multi_complex_conjugate<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_q_multi[STREAM], d_qk_in_1[STREAM], d_qk_in_2[STREAM], M_COMPLEX);
            else
            {
                // For complex fields, need to handle conjugation properly
                cuDoubleComplex* d_qk_conj = reinterpret_cast<cuDoubleComplex*>(d_qk_in_2[STREAM]);
                ker_copy_data_with_idx<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_qk_conj, d_qk_in_2[STREAM], _d_negative_k_idx, M_COMPLEX);
                ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_q_multi[STREAM], d_qk_in_1[STREAM], d_qk_conj, 1.0, M_COMPLEX);
            }
            gpu_error_check(cudaPeekAtLastError());
        }
        else
        {
            // Non-periodic BC: use CudaFFT (DCT/DST)
            if constexpr (!std::is_same<T, double>::value)
            {
                throw_with_line_number("RK2 stress computation with non-periodic BC is only supported for real fields (double).");
            }
            else
            {
                if (space_group_ != nullptr)
                {
                    ker_expand_reduced_basis<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                        d_q_step_1[STREAM], d_q_pair, d_full_to_reduced_map_, M);
                    ker_expand_reduced_basis<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                        d_q_step_2[STREAM], &d_q_pair[n_basis_], d_full_to_reduced_map_, M);
                    gpu_error_check(cudaPeekAtLastError());
                }
                else
                {
                    gpu_error_check(cudaMemcpyAsync(d_q_step_1[STREAM], d_q_pair, sizeof(double)*M, cudaMemcpyDeviceToDevice, streams[STREAM][0]));
                    gpu_error_check(cudaMemcpyAsync(d_q_step_2[STREAM], &d_q_pair[M], sizeof(double)*M, cudaMemcpyDeviceToDevice, streams[STREAM][0]));
                }
                double* d_q1 = d_q_step_1[STREAM];
                double* d_q2 = d_q_step_2[STREAM];
                double* rk_1 = reinterpret_cast<double*>(d_qk_in_1[STREAM]);
                double* rk_2 = reinterpret_cast<double*>(d_qk_in_2[STREAM]);

                fft_[STREAM]->forward(d_q1, rk_1);
                fft_[STREAM]->forward(d_q2, rk_2);

                // Multiply (real coefficients for non-periodic BC)
                ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_q_multi[STREAM], rk_1, rk_2, 1.0, M_COMPLEX);
                gpu_error_check(cudaPeekAtLastError());
            }
        }

        // With k⊗k dyad product, fourier_basis arrays contain Cartesian components directly:
        // _d_fourier_basis_x = k_x², _d_fourier_basis_y = k_y², etc.
        // No cross-term corrections needed.

        // ============================================================================
        // PERFORMANCE-CRITICAL: Orthogonal box optimization
        // ============================================================================
        // For orthogonal boxes (all angles = 90°), the cross-terms (σ_xy, σ_xz, σ_yz)
        // are mathematically zero and do not need to be computed. This optimization
        // reduces the number of GPU kernel calls from 6 to 3 for 3D boxes,
        // providing ~10% speedup when box_is_altering=True.
        //
        // DO NOT REMOVE THIS OPTIMIZATION without benchmarking! We have experienced
        // performance regressions before when this check was accidentally removed.
        // See git history for commit 6d1dc54 which caused a regression by always
        // computing all 6 components.
        // ============================================================================
        const bool is_orthogonal = this->cb->is_orthogonal();

        if ( DIM == 3 )
        {
            // σ_xx
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_stress_sum[STREAM], d_q_multi[STREAM], _d_fourier_basis_x, bond_length_sq, M_COMPLEX);
            gpu_error_check(cudaPeekAtLastError());
            if constexpr (std::is_same<T, double>::value)
                cub::DeviceReduce::Sum(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[0], M_COMPLEX, streams[STREAM][0]);
            else
                cub::DeviceReduce::Reduce(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[0], M_COMPLEX, ComplexSumOp(), CuDeviceData<T>{0.0,0.0}, streams[STREAM][0]);
            gpu_error_check(cudaPeekAtLastError());

            // σ_yy
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_stress_sum[STREAM], d_q_multi[STREAM], _d_fourier_basis_y, bond_length_sq, M_COMPLEX);
            gpu_error_check(cudaPeekAtLastError());
            if constexpr (std::is_same<T, double>::value)
                cub::DeviceReduce::Sum(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[1], M_COMPLEX, streams[STREAM][0]);
            else
                cub::DeviceReduce::Reduce(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[1], M_COMPLEX, ComplexSumOp(), CuDeviceData<T>{0.0,0.0}, streams[STREAM][0]);
            gpu_error_check(cudaPeekAtLastError());

            // σ_zz
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_stress_sum[STREAM], d_q_multi[STREAM], _d_fourier_basis_z, bond_length_sq, M_COMPLEX);
            gpu_error_check(cudaPeekAtLastError());
            if constexpr (std::is_same<T, double>::value)
                cub::DeviceReduce::Sum(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[2], M_COMPLEX, streams[STREAM][0]);
            else
                cub::DeviceReduce::Reduce(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[2], M_COMPLEX, ComplexSumOp(), CuDeviceData<T>{0.0,0.0}, streams[STREAM][0]);
            gpu_error_check(cudaPeekAtLastError());

            // Cross-terms: only compute for non-orthogonal boxes (triclinic lattices)
            // For orthogonal boxes, these are mathematically zero.
            if (!is_orthogonal)
            {
                // σ_xy
                ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_stress_sum[STREAM], d_q_multi[STREAM], _d_fourier_basis_xy, bond_length_sq, M_COMPLEX);
                gpu_error_check(cudaPeekAtLastError());
                if constexpr (std::is_same<T, double>::value)
                    cub::DeviceReduce::Sum(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[3], M_COMPLEX, streams[STREAM][0]);
                else
                    cub::DeviceReduce::Reduce(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[3], M_COMPLEX, ComplexSumOp(), CuDeviceData<T>{0.0,0.0}, streams[STREAM][0]);
                gpu_error_check(cudaPeekAtLastError());

                // σ_xz
                ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_stress_sum[STREAM], d_q_multi[STREAM], _d_fourier_basis_xz, bond_length_sq, M_COMPLEX);
                gpu_error_check(cudaPeekAtLastError());
                if constexpr (std::is_same<T, double>::value)
                    cub::DeviceReduce::Sum(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[4], M_COMPLEX, streams[STREAM][0]);
                else
                    cub::DeviceReduce::Reduce(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[4], M_COMPLEX, ComplexSumOp(), CuDeviceData<T>{0.0,0.0}, streams[STREAM][0]);
                gpu_error_check(cudaPeekAtLastError());

                // σ_yz
                ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_stress_sum[STREAM], d_q_multi[STREAM], _d_fourier_basis_yz, bond_length_sq, M_COMPLEX);
                gpu_error_check(cudaPeekAtLastError());
                if constexpr (std::is_same<T, double>::value)
                    cub::DeviceReduce::Sum(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[5], M_COMPLEX, streams[STREAM][0]);
                else
                    cub::DeviceReduce::Reduce(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[5], M_COMPLEX, ComplexSumOp(), CuDeviceData<T>{0.0,0.0}, streams[STREAM][0]);
                gpu_error_check(cudaPeekAtLastError());
            }
        }
        if ( DIM == 2 )
        {
            // σ_xx
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_stress_sum[STREAM], d_q_multi[STREAM], _d_fourier_basis_x, bond_length_sq, M_COMPLEX);
            gpu_error_check(cudaPeekAtLastError());
            if constexpr (std::is_same<T, double>::value)
                cub::DeviceReduce::Sum(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[0], M_COMPLEX, streams[STREAM][0]);
            else
                cub::DeviceReduce::Reduce(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[0], M_COMPLEX, ComplexSumOp(), CuDeviceData<T>{0.0,0.0}, streams[STREAM][0]);
            gpu_error_check(cudaPeekAtLastError());

            // σ_yy
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_stress_sum[STREAM], d_q_multi[STREAM], _d_fourier_basis_y, bond_length_sq, M_COMPLEX);
            gpu_error_check(cudaPeekAtLastError());
            if constexpr (std::is_same<T, double>::value)
                cub::DeviceReduce::Sum(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[1], M_COMPLEX, streams[STREAM][0]);
            else
                cub::DeviceReduce::Reduce(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[1], M_COMPLEX, ComplexSumOp(), CuDeviceData<T>{0.0,0.0}, streams[STREAM][0]);
            gpu_error_check(cudaPeekAtLastError());

            // σ_xy: only compute for non-orthogonal boxes
            if (!is_orthogonal)
            {
                ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_stress_sum[STREAM], d_q_multi[STREAM], _d_fourier_basis_xy, bond_length_sq, M_COMPLEX);
                gpu_error_check(cudaPeekAtLastError());
                if constexpr (std::is_same<T, double>::value)
                    cub::DeviceReduce::Sum(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[2], M_COMPLEX, streams[STREAM][0]);
                else
                    cub::DeviceReduce::Reduce(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[2], M_COMPLEX, ComplexSumOp(), CuDeviceData<T>{0.0,0.0}, streams[STREAM][0]);
                gpu_error_check(cudaPeekAtLastError());
            }
        }
        if ( DIM == 1 )
        {
            // σ_xx (only x-direction in 1D)
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_stress_sum[STREAM], d_q_multi[STREAM], _d_fourier_basis_x, bond_length_sq, M_COMPLEX);
            gpu_error_check(cudaPeekAtLastError());
            if constexpr (std::is_same<T, double>::value)
                cub::DeviceReduce::Sum(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[0], M_COMPLEX, streams[STREAM][0]);
            else
                cub::DeviceReduce::Reduce(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[0], M_COMPLEX, ComplexSumOp(), CuDeviceData<T>{0.0,0.0}, streams[STREAM][0]);
            gpu_error_check(cudaPeekAtLastError());
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

// Explicit template instantiation
template class CudaSolverPseudoRK2<double>;
template class CudaSolverPseudoRK2<std::complex<double>>;
