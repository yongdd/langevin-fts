/**
 * @file CudaSolverPseudoDiscrete.cu
 * @brief CUDA pseudo-spectral solver for discrete chain model.
 *
 * Implements propagator advancement for discrete chains using the
 * Chapman-Kolmogorov integral equation via cuFFT.
 *
 * **Chapman-Kolmogorov Equation (N-1 Bond Model):**
 *
 * For discrete chains, the propagator satisfies:
 *     q(r, i+1) = exp(-w(r)*ds) * integral g(r-r') q(r', i) dr'
 *
 * In Fourier space:
 *     q(i+1) = exp(-w*ds) * FFT^-1[ ĝ(k) * FFT[q(i)] ]
 *
 * where ĝ(k) = exp(-b²|k|²ds/6) is the full bond function.
 * See Park et al. J. Chem. Phys. 150, 234901 (2019).
 *
 * **Methods:**
 *
 * - advance_propagator(): Full segment step (bond convolution + full-segment weight)
 * - advance_propagator_half_bond_step(): Half bond convolution at chain ends/junctions
 * - compute_single_segment_stress(): Stress contribution per segment
 *
 * **cuFFT Plans:**
 *
 * - plan_for_one/two: Forward FFTs (D2Z for real, Z2Z for complex)
 * - plan_bak_one/two: Backward FFTs
 * - Stream-associated for concurrent execution
 *
 * **Template Instantiations:**
 *
 * - CudaSolverPseudoDiscrete<double>: Real field solver
 * - CudaSolverPseudoDiscrete<std::complex<double>>: Complex field solver
 *
 * @see CudaComputationDiscrete for discrete chain orchestration
 */

#include <iostream>
#include <cmath>
#include <numbers>
#include <array>
#include <thrust/reduce.h>

#include "CudaPseudo.h"
#include "CudaSolverPseudoDiscrete.h"
#include "CudaCrysFFT.h"
#include "CudaCrysFFTRecursive3m.h"
#include "CrysFFTSelector.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include "SpaceGroup.h"

template <typename T>
CudaSolverPseudoDiscrete<T>::CudaSolverPseudoDiscrete(
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

        // Validate that both sides of each direction have matching BC
        // (required for pseudo-spectral: FFT/DCT/DST apply to entire direction)
        auto bc_vec = cb->get_boundary_conditions();
        int dim = cb->get_dim();
        for (int d = 0; d < dim; ++d)
        {
            if (bc_vec[2*d] != bc_vec[2*d + 1])
            {
                throw_with_line_number("Pseudo-spectral method requires matching boundary conditions on both sides of each direction. "
                    "Direction " + std::to_string(d) + " has mismatched BCs. Use real-space method for mixed BCs.");
            }
        }

        // Check if all BCs are periodic
        is_periodic_ = true;
        for (const auto& bc : bc_vec)
        {
            if (bc != BoundaryCondition::PERIODIC)
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

        // Create d_exp_dw for each ds_index and monomer type
        // Also register local_ds values with Pseudo for boltz_bond computation
        for (int ds_idx = 0; ds_idx < n_unique_ds; ++ds_idx)
        {
            double local_ds = mapping.get_ds_from_index(ds_idx);
            pseudo->add_ds_value(ds_idx, local_ds);

            for(const auto& item: molecules->get_bond_lengths())
            {
                std::string monomer_type = item.first;
                this->d_exp_dw[ds_idx][monomer_type] = nullptr;
                gpu_error_check(cudaMalloc((void**)&this->d_exp_dw[ds_idx][monomer_type], sizeof(T)*M));
            }
        }

        // Finalize ds values to allocate boltz_bond arrays
        // (update_laplacian_operator will compute the actual values)
        pseudo->finalize_ds_values();

        // Create FFT plan
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
            cufftPlanMany(&plan_for_two[i], NRANK, total_grid, nullptr, 1, 0, nullptr, 1, 0, cufft_forward, 2);
            cufftPlanMany(&plan_bak_one[i], NRANK, total_grid, nullptr, 1, 0, nullptr, 1, 0, cufft_backward, 1);
            cufftPlanMany(&plan_bak_two[i], NRANK, total_grid, nullptr, 1, 0, nullptr, 1, 0, cufft_backward, 2);
            cufftSetStream(plan_for_one[i], streams[i][0]);
            cufftSetStream(plan_for_two[i], streams[i][0]);
            cufftSetStream(plan_bak_one[i], streams[i][0]);
            cufftSetStream(plan_bak_two[i], streams[i][0]);
        }

        // Create FFT objects for non-periodic BC (DCT/DST)
        // Extract one BC per dimension
        std::vector<BoundaryCondition> bc_per_dim;
        for (int d = 0; d < dim; ++d)
            bc_per_dim.push_back(bc_vec[2 * d]);

        for(int i=0; i<n_streams; i++)
        {
            if (dim == 3)
            {
                std::array<int, 3> nx_arr = {cb->get_nx(0), cb->get_nx(1), cb->get_nx(2)};
                std::array<BoundaryCondition, 3> bc_arr = {bc_per_dim[0], bc_per_dim[1], bc_per_dim[2]};
                fft_[i] = new CudaFFT<T, 3>(nx_arr, bc_arr);
            }
            else if (dim == 2)
            {
                std::array<int, 2> nx_arr = {cb->get_nx(0), cb->get_nx(1)};
                std::array<BoundaryCondition, 2> bc_arr = {bc_per_dim[0], bc_per_dim[1]};
                fft_[i] = new CudaFFT<T, 2>(nx_arr, bc_arr);
            }
            else if (dim == 1)
            {
                std::array<int, 1> nx_arr = {cb->get_nx(0)};
                std::array<BoundaryCondition, 1> bc_arr = {bc_per_dim[0]};
                fft_[i] = new CudaFFT<T, 1>(nx_arr, bc_arr);
            }
        }

        // Allocate memory for pseudo-spectral: advance_propagator()
        for(int i=0; i<n_streams; i++)
        {
            gpu_error_check(cudaMalloc((void**)&d_qk_in_1_one[i], sizeof(cuDoubleComplex)*M_COMPLEX));
            gpu_error_check(cudaMalloc((void**)&d_qk_in_1_two[i], sizeof(cuDoubleComplex)*2*M_COMPLEX));
            // Real-valued buffers for non-periodic BC stress calculation (DCT/DST)
            gpu_error_check(cudaMalloc((void**)&d_rk_in_1_one[i], sizeof(double)*M_COMPLEX));
            gpu_error_check(cudaMalloc((void**)&d_rk_in_2_one[i], sizeof(double)*M_COMPLEX));
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

        // Initialize space group support
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
        for(int i=0; i<MAX_STREAMS; i++)
        {
            d_q_full_in_[i] = nullptr;
            d_q_full_out_[i] = nullptr;
            crysfft_[i] = nullptr;
            d_crysfft_in_[i] = nullptr;
            d_crysfft_out_[i] = nullptr;
        }

        update_laplacian_operator();
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
CudaSolverPseudoDiscrete<T>::~CudaSolverPseudoDiscrete()
{
    cleanup_crysfft();

    delete pseudo;

    for(int i=0; i<n_streams; i++)
    {
        cufftDestroy(plan_for_one[i]);
        cufftDestroy(plan_for_two[i]);
        cufftDestroy(plan_bak_one[i]);
        cufftDestroy(plan_bak_two[i]);
    }

    // Free nested maps: d_exp_dw[ds_index][monomer_type]
    for(const auto& ds_entry: this->d_exp_dw)
        for(const auto& item: ds_entry.second)
            cudaFree(item.second);

    // For pseudo-spectral: advance_propagator()
    for(int i=0; i<n_streams; i++)
    {
        cudaFree(d_qk_in_1_one[i]);
        cudaFree(d_qk_in_1_two[i]);
        cudaFree(d_rk_in_1_one[i]);
        cudaFree(d_rk_in_2_one[i]);
        delete fft_[i];
    }

    for(int i=0; i<n_streams; i++)
    {
        cudaFree(d_stress_sum[i]);
        cudaFree(d_stress_sum_out[i]);
        cudaFree(d_q_multi[i]);
        cudaFree(d_temp_storage[i]);
    }

    // Free space group work buffers
    for(int i=0; i<n_streams; i++)
    {
        if (d_q_full_in_[i] != nullptr) cudaFree(d_q_full_in_[i]);
        if (d_q_full_out_[i] != nullptr) cudaFree(d_q_full_out_[i]);
    }
}

template <typename T>
void CudaSolverPseudoDiscrete<T>::cleanup_crysfft()
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
void CudaSolverPseudoDiscrete<T>::set_space_group(
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
                gpu_error_check(cudaMalloc((void**)&d_q_full_in_[i], sizeof(T)*M));
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

    if constexpr (std::is_same_v<T, double>)
    {
        if (space_group_ != nullptr && cb->get_dim() == 3 && is_periodic_ && cb->is_orthogonal())
        {
            const auto nx = cb->get_nx();
            std::array<int, 3> nx_logical = {nx[0], nx[1], nx[2]};
            const auto selection = select_crysfft_mode(space_group_, nx_logical, cb->get_dim(), is_periodic_, cb->is_orthogonal());
            if (selection.mode != CrysFFTChoice::None || selection.can_pmmm)
            {
                const int Nx2 = nx[0] / 2;
                const int Ny2 = nx[1] / 2;
                const int Nz2 = nx[2] / 2;
                const int M_phys = Nx2 * Ny2 * Nz2;
                const int M_reduced = space_group_->get_n_reduced_basis();
                const auto& full_to_reduced = space_group_->get_full_to_reduced_map();
                const bool use_m3_basis = space_group_->using_m3_physical_basis();
                const bool use_pmmm_basis = space_group_->using_pmmm_physical_basis();

                std::vector<int> phys_to_reduced;
                std::vector<int> reduced_to_phys;

                auto check_identity = [&](bool even_indices) -> bool {
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

                auto finalize_crysfft = [&](CudaCrysFFTMode mode, bool identity_forced) {
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

                    auto lx = cb->get_lx();
                    std::array<double, 6> cell_para = {lx[0], lx[1], lx[2], M_PI/2, M_PI/2, M_PI/2};

                    for (int i = 0; i < n_streams; i++)
                    {
                        if (mode == CudaCrysFFTMode::Recursive3m)
                            crysfft_[i] = new CudaCrysFFTRecursive3m(nx_logical, cell_para, selection.m3_translations);
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
                        finalize_crysfft(CudaCrysFFTMode::Recursive3m, true);
                    }
                    else if (build_mapping(true))
                    {
                        finalize_crysfft(CudaCrysFFTMode::Recursive3m, false);
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
                        finalize_crysfft(CudaCrysFFTMode::PmmmDct, true);
                    }
                    else if (build_mapping(false))
                    {
                        finalize_crysfft(CudaCrysFFTMode::PmmmDct, false);
                    }
                }
            }
        }
    }
}
template <typename T>
void CudaSolverPseudoDiscrete<T>::update_laplacian_operator()
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
void CudaSolverPseudoDiscrete<T>::update_dw(std::string device, std::map<std::string, const T*> w_input)
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
                        // Expand reduced basis → full grid on device
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

                    // Copy field configurations from host to device
                    gpu_error_check(cudaMemcpy(
                        this->d_exp_dw[ds_idx][monomer_type], w_full,
                        sizeof(T)*M_use, cudaMemcpyInputToDevice));
                }

                // Compute exp_dw: exp(-w * local_ds)
                ker_exp<<<N_BLOCKS, N_THREADS>>>
                    (this->d_exp_dw[ds_idx][monomer_type],
                     this->d_exp_dw[ds_idx][monomer_type], 1.0, -1.0*local_ds, M_use);
            }
        }
        gpu_error_check(cudaDeviceSynchronize());
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
void CudaSolverPseudoDiscrete<T>::advance_propagator(
    const int STREAM,
    CuDeviceData<T> *d_q_in, CuDeviceData<T> *d_q_out,
    std::string monomer_type,
    double *d_q_mask, int ds_index)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = cb->get_total_grid();
        const int M_COMPLEX = pseudo->get_total_complex_grid();

        // Get Boltzmann factors for the correct ds_index
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
                    gpu_error_check(cudaMemcpyAsync(
                        d_crysfft_in_[STREAM], d_q_in,
                        sizeof(double) * M_phys, cudaMemcpyDeviceToDevice, streams[STREAM][0]));

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

        // Determine input/output pointers based on space group
        CuDeviceData<T> *fft_in = d_q_in;
        CuDeviceData<T> *fft_out = d_q_out;

        if (space_group_ != nullptr)
        {
            if constexpr (std::is_same_v<T, double>)
            {
                // Expand reduced basis → full grid
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

        // Execute a forward FFT
        if constexpr (std::is_same<T, double>::value)
            cufftExecD2Z(plan_for_one[STREAM], fft_in, d_qk_in_1_one[STREAM]);
        else
            cufftExecZ2Z(plan_for_one[STREAM], fft_in, d_qk_in_1_one[STREAM], CUFFT_FORWARD);

        // Multiply exp(-k^2 ds/6) in fourier space
        ker_multi_complex_real<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_qk_in_1_one[STREAM], _d_boltz_bond, 1.0, M_COMPLEX);

        // Execute a backward FFT
        if constexpr (std::is_same<T, double>::value)
            cufftExecZ2D(plan_bak_one[STREAM], d_qk_in_1_one[STREAM], fft_out);
        else
            cufftExecZ2Z(plan_bak_one[STREAM], d_qk_in_1_one[STREAM], fft_out, CUFFT_INVERSE);

        // Evaluate exp(-w*ds) in real space
        ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(fft_out, fft_out, _d_exp_dw, 1.0/static_cast<double>(M), M);

        // Multiply mask (on full grid)
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
        }

        // Reduce full grid → reduced basis
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
void CudaSolverPseudoDiscrete<T>::advance_propagator_half_bond_step(
    const int STREAM,
    CuDeviceData<T> *d_q_in, CuDeviceData<T> *d_q_out, std::string monomer_type)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = cb->get_total_grid();
        const int M_COMPLEX = pseudo->get_total_complex_grid();
        // Discrete chains always use ds_index=0 (global ds)
        const double* _d_boltz_bond_half = pseudo->get_boltz_bond_half(monomer_type, 0);

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
                double local_ds = mapping.get_ds_from_index(0);
                auto bond_lengths = this->molecules->get_bond_lengths();
                double bond_length_sq = bond_lengths[monomer_type] * bond_lengths[monomer_type];
                double coeff_half = bond_length_sq * local_ds / 12.0;

                const int M_phys = crysfft_physical_size_;
                if (crysfft_identity_map_)
                {
                    gpu_error_check(cudaMemcpyAsync(
                        d_crysfft_in_[STREAM], d_q_in,
                        sizeof(double) * M_phys, cudaMemcpyDeviceToDevice, streams[STREAM][0]));

                    crysfft_[STREAM]->set_contour_step(coeff_half);
                    crysfft_[STREAM]->diffusion(d_crysfft_in_[STREAM], d_crysfft_out_[STREAM], streams[STREAM][0]);

                    gpu_error_check(cudaMemcpyAsync(
                        d_q_out, d_crysfft_out_[STREAM],
                        sizeof(double) * M_phys, cudaMemcpyDeviceToDevice, streams[STREAM][0]));
                    return;
                }
                ker_expand_reduced_basis<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                    d_crysfft_in_[STREAM], d_q_in, d_crysfft_phys_to_reduced_, M_phys);
                gpu_error_check(cudaPeekAtLastError());

                crysfft_[STREAM]->set_contour_step(coeff_half);
                crysfft_[STREAM]->diffusion(d_crysfft_in_[STREAM], d_crysfft_out_[STREAM], streams[STREAM][0]);

                ker_reduce_to_basis<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                    d_q_out, d_crysfft_out_[STREAM], d_crysfft_reduced_to_phys_, n_basis_);
                gpu_error_check(cudaPeekAtLastError());
                return;
            }
        }

        // Determine input/output pointers based on space group
        CuDeviceData<T> *fft_in = d_q_in;
        CuDeviceData<T> *fft_out = d_q_out;

        if (space_group_ != nullptr)
        {
            if constexpr (std::is_same_v<T, double>)
            {
                // Expand reduced basis → full grid
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

        // 3D fourier discrete transform, forward
        if constexpr (std::is_same<T, double>::value)
            cufftExecD2Z(plan_for_one[STREAM], fft_in, d_qk_in_1_one[STREAM]);
        else
            cufftExecZ2Z(plan_for_one[STREAM], fft_in, d_qk_in_1_one[STREAM], CUFFT_FORWARD);
        gpu_error_check(cudaPeekAtLastError());

        // Multiply exp(-k^2 ds/12) in fourier space, in all 3 directions
        ker_multi_complex_real<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_qk_in_1_one[STREAM], _d_boltz_bond_half, 1.0/static_cast<double>(M), M_COMPLEX);
        gpu_error_check(cudaPeekAtLastError());

        // 3D fourier discrete transform, backward
        if constexpr (std::is_same<T, double>::value)
            cufftExecZ2D(plan_bak_one[STREAM], d_qk_in_1_one[STREAM], fft_out);
        else
            cufftExecZ2Z(plan_bak_one[STREAM], d_qk_in_1_one[STREAM], fft_out, CUFFT_INVERSE);
        gpu_error_check(cudaPeekAtLastError());

        // Reduce full grid → reduced basis
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
void CudaSolverPseudoDiscrete<T>::compute_single_segment_stress(
    const int STREAM,
    CuDeviceData<T> *d_q_pair, CuDeviceData<T> *d_segment_stress,
    std::string monomer_type, bool is_half_bond_length)
{
    try{
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int DIM = this->cb->get_dim();
        // const int M   = total_grid;
        const int M_COMPLEX = pseudo->get_total_complex_grid();

        auto bond_lengths = this->molecules->get_bond_lengths();
        double bond_length_sq;
        double *_d_boltz_bond;

        const double* _d_fourier_basis_x = pseudo->get_fourier_basis_x();
        const double* _d_fourier_basis_y = pseudo->get_fourier_basis_y();
        const double* _d_fourier_basis_z = pseudo->get_fourier_basis_z();
        const double* _d_fourier_basis_xy = pseudo->get_fourier_basis_xy();
        const double* _d_fourier_basis_xz = pseudo->get_fourier_basis_xz();
        const double* _d_fourier_basis_yz = pseudo->get_fourier_basis_yz();
        const int* _d_negative_k_idx = pseudo->get_negative_frequency_mapping();

        // Discrete chains always use ds_index=0 (global ds)
        if (is_half_bond_length)
        {
            bond_length_sq = 0.5*bond_lengths[monomer_type]*bond_lengths[monomer_type];
            _d_boltz_bond = pseudo->get_boltz_bond_half(monomer_type, 0);
        }
        else
        {
            bond_length_sq = bond_lengths[monomer_type]*bond_lengths[monomer_type];
            _d_boltz_bond = pseudo->get_boltz_bond(monomer_type, 0);
        }

        const int M = this->cb->get_total_grid();

        if (is_periodic_)
        {
            // Periodic BC: Execute a forward FFT using cuFFT (batched, two propagators)
            if constexpr (std::is_same<T, double>::value)
                cufftExecD2Z(plan_for_two[STREAM], d_q_pair, d_qk_in_1_two[STREAM]);
            else
                cufftExecZ2Z(plan_for_two[STREAM], d_q_pair, d_qk_in_1_two[STREAM], CUFFT_FORWARD);
            gpu_error_check(cudaPeekAtLastError());

            // Multiply two propagators in the fourier spaces
            if constexpr (std::is_same<T, double>::value)
                ker_multi_complex_conjugate<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_q_multi[STREAM], &d_qk_in_1_two[STREAM][0], &d_qk_in_1_two[STREAM][M_COMPLEX], M_COMPLEX);
            else
            {
                ker_copy_data_with_idx<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_qk_in_1_one[STREAM], &d_qk_in_1_two[STREAM][M_COMPLEX], _d_negative_k_idx, M_COMPLEX);
                ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_q_multi[STREAM], &d_qk_in_1_two[STREAM][0], d_qk_in_1_one[STREAM], 1.0, M_COMPLEX);
            }
            gpu_error_check(cudaPeekAtLastError());
        }
        else
        {
            // Non-periodic BC: use CudaFFT (DCT/DST)
            if constexpr (!std::is_same<T, double>::value)
            {
                throw_with_line_number("Discrete chain stress computation with non-periodic BC is only supported for real fields (double).");
            }
            else
            {
                double* d_q1 = d_q_pair;
                double* d_q2 = &d_q_pair[M];
                double* rk_1 = d_rk_in_1_one[STREAM];
                double* rk_2 = d_rk_in_2_one[STREAM];

                // Transform each propagator separately using DCT/DST
                fft_[STREAM]->forward(d_q1, rk_1);
                fft_[STREAM]->forward(d_q2, rk_2);

                // Multiply the two transforms element-wise (real coefficients)
                // Factor of 2 for DCT/DST Parseval relation (no conjugate pairs like FFT)
                ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_q_multi[STREAM], rk_1, rk_2, 2.0, M_COMPLEX);
            }
            gpu_error_check(cudaPeekAtLastError());
        }

        // Multiply by Boltzmann bond factor and bond length squared
        ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_q_multi[STREAM], d_q_multi[STREAM], _d_boltz_bond, bond_length_sq, M_COMPLEX);
        gpu_error_check(cudaPeekAtLastError());

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
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_stress_sum[STREAM], d_q_multi[STREAM], _d_fourier_basis_x, 1.0, M_COMPLEX);
            gpu_error_check(cudaPeekAtLastError());
            if constexpr (std::is_same<T, double>::value)
                cub::DeviceReduce::Sum(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[0], M_COMPLEX, streams[STREAM][0]);
            else
                cub::DeviceReduce::Reduce(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[0], M_COMPLEX, ComplexSumOp(), CuDeviceData<T>{0.0,0.0}, streams[STREAM][0]);
            gpu_error_check(cudaPeekAtLastError());

            // σ_yy
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_stress_sum[STREAM], d_q_multi[STREAM], _d_fourier_basis_y, 1.0, M_COMPLEX);
            gpu_error_check(cudaPeekAtLastError());
            if constexpr (std::is_same<T, double>::value)
                cub::DeviceReduce::Sum(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[1], M_COMPLEX, streams[STREAM][0]);
            else
                cub::DeviceReduce::Reduce(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[1], M_COMPLEX, ComplexSumOp(), CuDeviceData<T>{0.0,0.0}, streams[STREAM][0]);
            gpu_error_check(cudaPeekAtLastError());

            // σ_zz
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_stress_sum[STREAM], d_q_multi[STREAM], _d_fourier_basis_z, 1.0, M_COMPLEX);
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
                ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_stress_sum[STREAM], d_q_multi[STREAM], _d_fourier_basis_xy, 1.0, M_COMPLEX);
                gpu_error_check(cudaPeekAtLastError());
                if constexpr (std::is_same<T, double>::value)
                    cub::DeviceReduce::Sum(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[3], M_COMPLEX, streams[STREAM][0]);
                else
                    cub::DeviceReduce::Reduce(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[3], M_COMPLEX, ComplexSumOp(), CuDeviceData<T>{0.0,0.0}, streams[STREAM][0]);
                gpu_error_check(cudaPeekAtLastError());

                // σ_xz
                ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_stress_sum[STREAM], d_q_multi[STREAM], _d_fourier_basis_xz, 1.0, M_COMPLEX);
                gpu_error_check(cudaPeekAtLastError());
                if constexpr (std::is_same<T, double>::value)
                    cub::DeviceReduce::Sum(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[4], M_COMPLEX, streams[STREAM][0]);
                else
                    cub::DeviceReduce::Reduce(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[4], M_COMPLEX, ComplexSumOp(), CuDeviceData<T>{0.0,0.0}, streams[STREAM][0]);
                gpu_error_check(cudaPeekAtLastError());

                // σ_yz
                ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_stress_sum[STREAM], d_q_multi[STREAM], _d_fourier_basis_yz, 1.0, M_COMPLEX);
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
            // For non-periodic BC, 2D grid is mapped to y-z axes internally (tnx = {1, nx[0], nx[1]})
            // so stress data is stored in fourier_basis_y and fourier_basis_z
            const double* _d_basis_xx = is_periodic_ ? _d_fourier_basis_x : _d_fourier_basis_y;
            const double* _d_basis_yy = is_periodic_ ? _d_fourier_basis_y : _d_fourier_basis_z;
            const double* _d_basis_xy = is_periodic_ ? _d_fourier_basis_xy : _d_fourier_basis_yz;

            // σ_xx
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_stress_sum[STREAM], d_q_multi[STREAM], _d_basis_xx, 1.0, M_COMPLEX);
            gpu_error_check(cudaPeekAtLastError());
            if constexpr (std::is_same<T, double>::value)
                cub::DeviceReduce::Sum(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[0], M_COMPLEX, streams[STREAM][0]);
            else
                cub::DeviceReduce::Reduce(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[0], M_COMPLEX, ComplexSumOp(), CuDeviceData<T>{0.0,0.0}, streams[STREAM][0]);
            gpu_error_check(cudaPeekAtLastError());

            // σ_yy
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_stress_sum[STREAM], d_q_multi[STREAM], _d_basis_yy, 1.0, M_COMPLEX);
            gpu_error_check(cudaPeekAtLastError());
            if constexpr (std::is_same<T, double>::value)
                cub::DeviceReduce::Sum(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[1], M_COMPLEX, streams[STREAM][0]);
            else
                cub::DeviceReduce::Reduce(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[1], M_COMPLEX, ComplexSumOp(), CuDeviceData<T>{0.0,0.0}, streams[STREAM][0]);
            gpu_error_check(cudaPeekAtLastError());

            // σ_xy: only compute for non-orthogonal boxes
            if (!is_orthogonal)
            {
                ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_stress_sum[STREAM], d_q_multi[STREAM], _d_basis_xy, 1.0, M_COMPLEX);
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
            // For non-periodic BC, 1D grid is mapped to z-axis internally (tnx = {1, 1, nx[0]})
            // so stress data is stored in fourier_basis_z
            const double* _d_basis_xx = is_periodic_ ? _d_fourier_basis_x : _d_fourier_basis_z;

            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_stress_sum[STREAM], d_q_multi[STREAM], _d_basis_xx, 1.0, M_COMPLEX);
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

template class CudaSolverPseudoDiscrete<double>;
template class CudaSolverPseudoDiscrete<std::complex<double>>;
