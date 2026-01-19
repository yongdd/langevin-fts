/**
 * @file CudaSolverPseudoETDRK4.cu
 * @brief CUDA ETDRK4 pseudo-spectral solver for continuous Gaussian chains.
 *
 * Implements the ETDRK4 (Exponential Time Differencing Runge-Kutta 4th order)
 * method for advancing chain propagators using cuFFT and multiple CUDA streams.
 * Uses the Krogstad scheme from Song et al. 2018.
 *
 * **Krogstad ETDRK4 Algorithm (Song et al. 2018, Eq. 7a-7d):**
 *
 * For dq/ds = L·q + N(q) where L = (b²/6)∇² and N(q) = -w·q:
 *
 * Stage a (7a): â = E2·q̂ + α·N̂_n                        where N_n = -w·q
 * Stage b (7b): b̂ = â + φ₂_half·(N̂_a - N̂_n)            where N_a = -w·a
 * Stage c (7c): ĉ = E·q̂ + φ₁·N̂_n + 2φ₂·(N̂_b - N̂_n)   where N_b = -w·b
 * Final (7d):   q̂_{n+1} = ĉ + (4φ₃ - φ₂)·(N̂_n + N̂_c)
 *               + 2φ₂·N̂_a - 4φ₃·(N̂_a + N̂_b)           where N_c = -w·c
 *
 * Coefficients are computed using the Kassam-Trefethen contour integral method.
 *
 * @see ETDRK4Coefficients for coefficient computation
 * @see CudaSolverPseudoRQM4 for RQM4 alternative
 */

#include <iostream>
#include <cmath>
#include <numbers>
#include <thrust/reduce.h>

#include "CudaPseudo.h"
#include "CudaSolverPseudoETDRK4.h"

template <typename T>
CudaSolverPseudoETDRK4<T>::CudaSolverPseudoETDRK4(
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

        // Create exp_dw, exp_dw_half for each ds_index and monomer type
        for (int ds_idx = 1; ds_idx <= n_unique_ds; ++ds_idx)
        {
            for(const auto& item: molecules->get_bond_lengths())
            {
                std::string monomer_type = item.first;
                this->d_exp_dw     [ds_idx][monomer_type] = nullptr;
                this->d_exp_dw_half[ds_idx][monomer_type] = nullptr;
                gpu_error_check(cudaMalloc((void**)&this->d_exp_dw     [ds_idx][monomer_type], sizeof(T)*M));
                gpu_error_check(cudaMalloc((void**)&this->d_exp_dw_half[ds_idx][monomer_type], sizeof(T)*M));
            }
        }

        // Create w_field storage (independent of ds)
        for(const auto& item: molecules->get_bond_lengths())
        {
            std::string monomer_type = item.first;
            this->d_w_field[monomer_type] = nullptr;
            gpu_error_check(cudaMalloc((void**)&this->d_w_field[monomer_type], sizeof(T)*M));
        }

        // Initialize cuFFT plans to 0
        for(int i=0; i<n_streams; i++)
        {
            plan_for_one[i] = 0;
            plan_bak_one[i] = 0;
        }

        if (is_periodic_)
        {
            // Create cuFFT plans for periodic BC
            // Use fixed-size array (max 3 dims) instead of VLA for portability
            int total_grid[3] = {1, 1, 1};

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
                cufftPlanMany(&plan_for_one[i], dim_, total_grid, nullptr, 1, 0, nullptr, 1, 0, cufft_forward, 1);
                cufftPlanMany(&plan_bak_one[i], dim_, total_grid, nullptr, 1, 0, nullptr, 1, 0, cufft_backward, 1);
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
        }

        // Initialize ETDRK4 coefficients for each ds_index
        for (int ds_idx = 1; ds_idx <= n_unique_ds; ++ds_idx)
        {
            double local_ds = mapping.get_ds_from_index(ds_idx);
            etdrk4_coefficients_[ds_idx] = std::make_unique<ETDRK4Coefficients<T>>(
                molecules->get_bond_lengths(),
                cb->get_boundary_conditions(),
                cb->get_nx(),
                cb->get_dx(),
                local_ds,
                cb->get_recip_metric()
            );

            int coeff_size = etdrk4_coefficients_[ds_idx]->get_total_complex_grid();

            // Allocate and copy Krogstad coefficients to device for each ds_index
            for (const auto& item : molecules->get_bond_lengths())
            {
                std::string type = item.first;

                gpu_error_check(cudaMalloc((void**)&d_etdrk4_E[ds_idx][type], sizeof(double)*coeff_size));
                gpu_error_check(cudaMalloc((void**)&d_etdrk4_E2[ds_idx][type], sizeof(double)*coeff_size));
                gpu_error_check(cudaMalloc((void**)&d_etdrk4_alpha[ds_idx][type], sizeof(double)*coeff_size));
                gpu_error_check(cudaMalloc((void**)&d_etdrk4_phi2_half[ds_idx][type], sizeof(double)*coeff_size));
                gpu_error_check(cudaMalloc((void**)&d_etdrk4_phi1[ds_idx][type], sizeof(double)*coeff_size));
                gpu_error_check(cudaMalloc((void**)&d_etdrk4_phi2[ds_idx][type], sizeof(double)*coeff_size));
                gpu_error_check(cudaMalloc((void**)&d_etdrk4_phi3[ds_idx][type], sizeof(double)*coeff_size));

                gpu_error_check(cudaMemcpy(d_etdrk4_E[ds_idx][type], etdrk4_coefficients_[ds_idx]->get_E(type),
                    sizeof(double)*coeff_size, cudaMemcpyHostToDevice));
                gpu_error_check(cudaMemcpy(d_etdrk4_E2[ds_idx][type], etdrk4_coefficients_[ds_idx]->get_E2(type),
                    sizeof(double)*coeff_size, cudaMemcpyHostToDevice));
                gpu_error_check(cudaMemcpy(d_etdrk4_alpha[ds_idx][type], etdrk4_coefficients_[ds_idx]->get_alpha(type),
                    sizeof(double)*coeff_size, cudaMemcpyHostToDevice));
                gpu_error_check(cudaMemcpy(d_etdrk4_phi2_half[ds_idx][type], etdrk4_coefficients_[ds_idx]->get_phi2_half(type),
                    sizeof(double)*coeff_size, cudaMemcpyHostToDevice));
                gpu_error_check(cudaMemcpy(d_etdrk4_phi1[ds_idx][type], etdrk4_coefficients_[ds_idx]->get_phi1(type),
                    sizeof(double)*coeff_size, cudaMemcpyHostToDevice));
                gpu_error_check(cudaMemcpy(d_etdrk4_phi2[ds_idx][type], etdrk4_coefficients_[ds_idx]->get_phi2(type),
                    sizeof(double)*coeff_size, cudaMemcpyHostToDevice));
                gpu_error_check(cudaMemcpy(d_etdrk4_phi3[ds_idx][type], etdrk4_coefficients_[ds_idx]->get_phi3(type),
                    sizeof(double)*coeff_size, cudaMemcpyHostToDevice));
            }
        }

        // Allocate ETDRK4 workspace arrays for each stream
        for (int i = 0; i < n_streams; i++)
        {
            // Real-space arrays
            gpu_error_check(cudaMalloc((void**)&d_etdrk4_a[i], sizeof(T)*M));
            gpu_error_check(cudaMalloc((void**)&d_etdrk4_b[i], sizeof(T)*M));
            gpu_error_check(cudaMalloc((void**)&d_etdrk4_c[i], sizeof(T)*M));
            gpu_error_check(cudaMalloc((void**)&d_etdrk4_N_n[i], sizeof(T)*M));
            gpu_error_check(cudaMalloc((void**)&d_etdrk4_N_a[i], sizeof(T)*M));
            gpu_error_check(cudaMalloc((void**)&d_etdrk4_N_b[i], sizeof(T)*M));
            gpu_error_check(cudaMalloc((void**)&d_etdrk4_N_c[i], sizeof(T)*M));

            // Fourier-space arrays
            gpu_error_check(cudaMalloc((void**)&d_k_q[i], sizeof(cuDoubleComplex)*M_COMPLEX));
            gpu_error_check(cudaMalloc((void**)&d_k_a[i], sizeof(cuDoubleComplex)*M_COMPLEX));
            gpu_error_check(cudaMalloc((void**)&d_k_N_n[i], sizeof(cuDoubleComplex)*M_COMPLEX));
            gpu_error_check(cudaMalloc((void**)&d_k_N_a[i], sizeof(cuDoubleComplex)*M_COMPLEX));
            gpu_error_check(cudaMalloc((void**)&d_k_N_b[i], sizeof(cuDoubleComplex)*M_COMPLEX));
            gpu_error_check(cudaMalloc((void**)&d_k_N_c[i], sizeof(cuDoubleComplex)*M_COMPLEX));
            gpu_error_check(cudaMalloc((void**)&d_k_work[i], sizeof(cuDoubleComplex)*M_COMPLEX));

            if (!is_periodic_)
            {
                gpu_error_check(cudaMalloc((void**)&d_rk_work[i], sizeof(double)*M_COMPLEX));
            }
            else
            {
                d_rk_work[i] = nullptr;
            }
        }

        // Allocate stress computation arrays
        for(int i=0; i<n_streams; i++)
        {
            gpu_error_check(cudaMalloc((void**)&d_stress_sum[i],      sizeof(T)*M_COMPLEX));
            gpu_error_check(cudaMalloc((void**)&d_stress_sum_out[i],  sizeof(T)*1));
            gpu_error_check(cudaMalloc((void**)&d_q_multi[i],         sizeof(T)*M_COMPLEX));
            gpu_error_check(cudaMalloc((void**)&d_qk_stress[i],       sizeof(cuDoubleComplex)*2*M_COMPLEX));
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

        update_laplacian_operator();
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

template <typename T>
CudaSolverPseudoETDRK4<T>::~CudaSolverPseudoETDRK4()
{
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
        for(int i=0; i<n_streams; i++)
        {
            if (fft_[i] != nullptr)
            {
                delete fft_[i];
                fft_[i] = nullptr;
            }
        }
    }

    // Free Boltzmann factors (nested maps: d_exp_dw[ds_index][monomer_type])
    for(const auto& ds_entry: this->d_exp_dw)
        for(const auto& item: ds_entry.second)
            cudaFree(item.second);
    for(const auto& ds_entry: this->d_exp_dw_half)
        for(const auto& item: ds_entry.second)
            cudaFree(item.second);
    for(const auto& item: this->d_w_field)
        cudaFree(item.second);

    // Free ETDRK4 coefficient arrays (Krogstad scheme) - nested maps: [ds_index][monomer_type]
    for(const auto& ds_entry: this->d_etdrk4_E)
        for(const auto& item: ds_entry.second)
            cudaFree(item.second);
    for(const auto& ds_entry: this->d_etdrk4_E2)
        for(const auto& item: ds_entry.second)
            cudaFree(item.second);
    for(const auto& ds_entry: this->d_etdrk4_alpha)
        for(const auto& item: ds_entry.second)
            cudaFree(item.second);
    for(const auto& ds_entry: this->d_etdrk4_phi2_half)
        for(const auto& item: ds_entry.second)
            cudaFree(item.second);
    for(const auto& ds_entry: this->d_etdrk4_phi1)
        for(const auto& item: ds_entry.second)
            cudaFree(item.second);
    for(const auto& ds_entry: this->d_etdrk4_phi2)
        for(const auto& item: ds_entry.second)
            cudaFree(item.second);
    for(const auto& ds_entry: this->d_etdrk4_phi3)
        for(const auto& item: ds_entry.second)
            cudaFree(item.second);

    // Free ETDRK4 workspace arrays
    for(int i=0; i<n_streams; i++)
    {
        cudaFree(d_etdrk4_a[i]);
        cudaFree(d_etdrk4_b[i]);
        cudaFree(d_etdrk4_c[i]);
        cudaFree(d_etdrk4_N_n[i]);
        cudaFree(d_etdrk4_N_a[i]);
        cudaFree(d_etdrk4_N_b[i]);
        cudaFree(d_etdrk4_N_c[i]);
        cudaFree(d_k_q[i]);
        cudaFree(d_k_a[i]);
        cudaFree(d_k_N_n[i]);
        cudaFree(d_k_N_a[i]);
        cudaFree(d_k_N_b[i]);
        cudaFree(d_k_N_c[i]);
        cudaFree(d_k_work[i]);
        if (d_rk_work[i] != nullptr) cudaFree(d_rk_work[i]);
    }

    // Free stress computation arrays
    for(int i=0; i<n_streams; i++)
    {
        cudaFree(d_stress_sum[i]);
        cudaFree(d_stress_sum_out[i]);
        cudaFree(d_q_multi[i]);
        cudaFree(d_qk_stress[i]);
        cudaFree(d_temp_storage[i]);
    }
}

template <typename T>
void CudaSolverPseudoETDRK4<T>::update_laplacian_operator()
{
    try{
        // Update Pseudo (without global_ds - all ds values come from ContourLengthMapping)
        pseudo->update(
            this->cb->get_boundary_conditions(),
            this->molecules->get_bond_lengths(),
            this->cb->get_dx(),
            this->cb->get_recip_metric(),
            this->cb->get_recip_vec());

        // Update ETDRK4 coefficients for each ds_index
        const ContourLengthMapping& mapping = this->molecules->get_contour_length_mapping();
        int n_unique_ds = mapping.get_n_unique_ds();

        for (int ds_idx = 1; ds_idx <= n_unique_ds; ++ds_idx)
        {
            double local_ds = mapping.get_ds_from_index(ds_idx);
            etdrk4_coefficients_[ds_idx]->update(
                this->cb->get_boundary_conditions(),
                this->molecules->get_bond_lengths(),
                this->cb->get_dx(),
                local_ds,
                this->cb->get_recip_metric()
            );

            // Copy updated Krogstad coefficients to device
            int coeff_size = etdrk4_coefficients_[ds_idx]->get_total_complex_grid();
            for (const auto& item : molecules->get_bond_lengths())
            {
                std::string type = item.first;
                gpu_error_check(cudaMemcpy(d_etdrk4_E[ds_idx][type], etdrk4_coefficients_[ds_idx]->get_E(type),
                    sizeof(double)*coeff_size, cudaMemcpyHostToDevice));
                gpu_error_check(cudaMemcpy(d_etdrk4_E2[ds_idx][type], etdrk4_coefficients_[ds_idx]->get_E2(type),
                    sizeof(double)*coeff_size, cudaMemcpyHostToDevice));
                gpu_error_check(cudaMemcpy(d_etdrk4_alpha[ds_idx][type], etdrk4_coefficients_[ds_idx]->get_alpha(type),
                    sizeof(double)*coeff_size, cudaMemcpyHostToDevice));
                gpu_error_check(cudaMemcpy(d_etdrk4_phi2_half[ds_idx][type], etdrk4_coefficients_[ds_idx]->get_phi2_half(type),
                    sizeof(double)*coeff_size, cudaMemcpyHostToDevice));
                gpu_error_check(cudaMemcpy(d_etdrk4_phi1[ds_idx][type], etdrk4_coefficients_[ds_idx]->get_phi1(type),
                    sizeof(double)*coeff_size, cudaMemcpyHostToDevice));
                gpu_error_check(cudaMemcpy(d_etdrk4_phi2[ds_idx][type], etdrk4_coefficients_[ds_idx]->get_phi2(type),
                    sizeof(double)*coeff_size, cudaMemcpyHostToDevice));
                gpu_error_check(cudaMemcpy(d_etdrk4_phi3[ds_idx][type], etdrk4_coefficients_[ds_idx]->get_phi3(type),
                    sizeof(double)*coeff_size, cudaMemcpyHostToDevice));
            }
        }
    }
    catch(std::exception& exc)
    {
        throw_with_line_number(exc.what());
    }
}

template <typename T>
void CudaSolverPseudoETDRK4<T>::update_dw(std::string device, std::map<std::string, const T*> w_input)
{
    try{
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = cb->get_total_grid();

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

        // Copy raw w field for ETDRK4 (independent of ds)
        for(const auto& item: w_input)
        {
            std::string monomer_type = item.first;
            const T *w = item.second;

            gpu_error_check(cudaMemcpyAsync(
                this->d_w_field[monomer_type], w,
                sizeof(T)*M, cudaMemcpyInputToDevice));
        }

        // Compute exp_dw and exp_dw_half for each ds_index
        for (int ds_idx = 1; ds_idx <= n_unique_ds; ++ds_idx)
        {
            double local_ds = mapping.get_ds_from_index(ds_idx);

            for(const auto& item: w_input)
            {
                std::string monomer_type = item.first;
                const T *w = item.second;

                if (this->d_exp_dw[ds_idx].find(monomer_type) == this->d_exp_dw[ds_idx].end())
                    throw_with_line_number("monomer_type \"" + monomer_type + "\" is not in d_exp_dw[" + std::to_string(ds_idx) + "].");

                // Copy for exp computation
                gpu_error_check(cudaMemcpyAsync(
                    this->d_exp_dw     [ds_idx][monomer_type], w,
                    sizeof(T)*M, cudaMemcpyInputToDevice));
                gpu_error_check(cudaMemcpyAsync(
                    this->d_exp_dw_half[ds_idx][monomer_type], w,
                    sizeof(T)*M, cudaMemcpyInputToDevice));

                // Compute d_exp_dw and d_exp_dw_half with local_ds
                ker_exp<<<N_BLOCKS, N_THREADS>>>
                    (this->d_exp_dw[ds_idx][monomer_type],      this->d_exp_dw[ds_idx][monomer_type],      1.0, -0.50*local_ds, M);
                ker_exp<<<N_BLOCKS, N_THREADS>>>
                    (this->d_exp_dw_half[ds_idx][monomer_type], this->d_exp_dw_half[ds_idx][monomer_type], 1.0, -0.25*local_ds, M);
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

// Advance propagator using ETDRK4
template <typename T>
void CudaSolverPseudoETDRK4<T>::advance_propagator(
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

        // Get Krogstad ETDRK4 coefficients for this ds_index and monomer type
        const double* _d_E = d_etdrk4_E[ds_index][monomer_type];
        const double* _d_E2 = d_etdrk4_E2[ds_index][monomer_type];
        const double* _d_alpha = d_etdrk4_alpha[ds_index][monomer_type];
        const double* _d_phi2_half = d_etdrk4_phi2_half[ds_index][monomer_type];
        const double* _d_phi1 = d_etdrk4_phi1[ds_index][monomer_type];
        const double* _d_phi2 = d_etdrk4_phi2[ds_index][monomer_type];
        const double* _d_phi3 = d_etdrk4_phi3[ds_index][monomer_type];

        // Get raw w field for nonlinear term
        CuDeviceData<T>* _d_w = d_w_field[monomer_type];

        if (is_periodic_)
        {
            // ============================================================
            // Periodic BC: Use cuFFT with complex coefficients
            // ============================================================

            // Step 1: Compute N_n = -w * q_in
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                d_etdrk4_N_n[STREAM], _d_w, d_q_in, -1.0, M);
            gpu_error_check(cudaPeekAtLastError());

            // Step 2: FFT of q_in and N_n
            if constexpr (std::is_same<T, double>::value)
            {
                cufftExecD2Z(plan_for_one[STREAM], d_q_in, d_k_q[STREAM]);
                cufftExecD2Z(plan_for_one[STREAM], d_etdrk4_N_n[STREAM], d_k_N_n[STREAM]);
            }
            else
            {
                cufftExecZ2Z(plan_for_one[STREAM], d_q_in, d_k_q[STREAM], CUFFT_FORWARD);
                cufftExecZ2Z(plan_for_one[STREAM], d_etdrk4_N_n[STREAM], d_k_N_n[STREAM], CUFFT_FORWARD);
            }
            gpu_error_check(cudaPeekAtLastError());

            // Stage a (Eq. 7a): â = E2·q̂ + α·N̂_n
            ker_etdrk4_stage_a<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                d_k_a[STREAM], d_k_q[STREAM], d_k_N_n[STREAM], _d_E2, _d_alpha, M_COMPLEX);
            gpu_error_check(cudaPeekAtLastError());

            // IFFT to get a
            // IMPORTANT: Copy k_a to work buffer first because cuFFT Z2D may corrupt
            // the input array, and we need k_a again for stage b.
            gpu_error_check(cudaMemcpyAsync(d_k_work[STREAM], d_k_a[STREAM],
                sizeof(cuDoubleComplex)*M_COMPLEX, cudaMemcpyDeviceToDevice, streams[STREAM][0]));
            if constexpr (std::is_same<T, double>::value)
                cufftExecZ2D(plan_bak_one[STREAM], d_k_work[STREAM], d_etdrk4_a[STREAM]);
            else
                cufftExecZ2Z(plan_bak_one[STREAM], d_k_work[STREAM], d_etdrk4_a[STREAM], CUFFT_INVERSE);
            gpu_error_check(cudaPeekAtLastError());

            // Normalize and compute N_a = -w * a
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                d_etdrk4_N_a[STREAM], _d_w, d_etdrk4_a[STREAM], -1.0/static_cast<double>(M), M);
            gpu_error_check(cudaPeekAtLastError());

            // FFT of N_a
            if constexpr (std::is_same<T, double>::value)
                cufftExecD2Z(plan_for_one[STREAM], d_etdrk4_N_a[STREAM], d_k_N_a[STREAM]);
            else
                cufftExecZ2Z(plan_for_one[STREAM], d_etdrk4_N_a[STREAM], d_k_N_a[STREAM], CUFFT_FORWARD);
            gpu_error_check(cudaPeekAtLastError());

            // Stage b (Eq. 7b): b̂ = â + φ₂_half·(N̂_a - N̂_n)
            ker_etdrk4_krogstad_stage_b<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                d_k_work[STREAM], d_k_a[STREAM], d_k_N_a[STREAM], d_k_N_n[STREAM], _d_phi2_half, M_COMPLEX);
            gpu_error_check(cudaPeekAtLastError());

            // IFFT to get b
            if constexpr (std::is_same<T, double>::value)
                cufftExecZ2D(plan_bak_one[STREAM], d_k_work[STREAM], d_etdrk4_b[STREAM]);
            else
                cufftExecZ2Z(plan_bak_one[STREAM], d_k_work[STREAM], d_etdrk4_b[STREAM], CUFFT_INVERSE);
            gpu_error_check(cudaPeekAtLastError());

            // Normalize and compute N_b = -w * b
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                d_etdrk4_N_b[STREAM], _d_w, d_etdrk4_b[STREAM], -1.0/static_cast<double>(M), M);
            gpu_error_check(cudaPeekAtLastError());

            // FFT of N_b
            if constexpr (std::is_same<T, double>::value)
                cufftExecD2Z(plan_for_one[STREAM], d_etdrk4_N_b[STREAM], d_k_N_b[STREAM]);
            else
                cufftExecZ2Z(plan_for_one[STREAM], d_etdrk4_N_b[STREAM], d_k_N_b[STREAM], CUFFT_FORWARD);
            gpu_error_check(cudaPeekAtLastError());

            // Stage c (Eq. 7c): ĉ = E·q̂ + φ₁·N̂_n + 2φ₂·(N̂_b - N̂_n)
            ker_etdrk4_krogstad_stage_c<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                d_k_work[STREAM], d_k_q[STREAM], d_k_N_n[STREAM], d_k_N_b[STREAM],
                _d_E, _d_phi1, _d_phi2, M_COMPLEX);
            gpu_error_check(cudaPeekAtLastError());

            // IMPORTANT: Save c_hat to d_k_a before IFFT because cuFFT Z2D corrupts input.
            // We reuse d_k_a since it's no longer needed after stage b.
            gpu_error_check(cudaMemcpyAsync(d_k_a[STREAM], d_k_work[STREAM],
                sizeof(cuDoubleComplex)*M_COMPLEX, cudaMemcpyDeviceToDevice, streams[STREAM][0]));

            // IFFT to get c (this corrupts d_k_work)
            if constexpr (std::is_same<T, double>::value)
                cufftExecZ2D(plan_bak_one[STREAM], d_k_work[STREAM], d_etdrk4_c[STREAM]);
            else
                cufftExecZ2Z(plan_bak_one[STREAM], d_k_work[STREAM], d_etdrk4_c[STREAM], CUFFT_INVERSE);
            gpu_error_check(cudaPeekAtLastError());

            // Normalize and compute N_c = -w * c
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                d_etdrk4_N_c[STREAM], _d_w, d_etdrk4_c[STREAM], -1.0/static_cast<double>(M), M);
            gpu_error_check(cudaPeekAtLastError());

            // FFT of N_c
            if constexpr (std::is_same<T, double>::value)
                cufftExecD2Z(plan_for_one[STREAM], d_etdrk4_N_c[STREAM], d_k_N_c[STREAM]);
            else
                cufftExecZ2Z(plan_for_one[STREAM], d_etdrk4_N_c[STREAM], d_k_N_c[STREAM], CUFFT_FORWARD);
            gpu_error_check(cudaPeekAtLastError());

            // Final step (Eq. 7d): q̂_{n+1} = ĉ + (4φ₃ - φ₂)·(N̂_n + N̂_c) + 2φ₂·N̂_a - 4φ₃·(N̂_a + N̂_b)
            // Use d_k_a which has the saved c_hat (d_k_work was corrupted by IFFT)
            ker_etdrk4_krogstad_final<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                d_k_work[STREAM], d_k_a[STREAM], d_k_N_n[STREAM], d_k_N_a[STREAM],
                d_k_N_b[STREAM], d_k_N_c[STREAM], _d_phi2, _d_phi3, M_COMPLEX);
            gpu_error_check(cudaPeekAtLastError());

            // IFFT to get final result
            if constexpr (std::is_same<T, double>::value)
                cufftExecZ2D(plan_bak_one[STREAM], d_k_work[STREAM], d_q_out);
            else
                cufftExecZ2Z(plan_bak_one[STREAM], d_k_work[STREAM], d_q_out, CUFFT_INVERSE);
            gpu_error_check(cudaPeekAtLastError());

            // Normalize final result (divide by M for IFFT normalization)
            if constexpr (std::is_same<T, double>::value)
            {
                ker_linear_scaling<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                    d_q_out, d_q_out, 1.0/static_cast<double>(M), 0.0, M);
            }
            else
            {
                cuDoubleComplex zero = make_cuDoubleComplex(0.0, 0.0);
                ker_linear_scaling<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                    d_q_out, d_q_out, 1.0/static_cast<double>(M), zero, M);
            }
            gpu_error_check(cudaPeekAtLastError());
        }
        else
        {
            // ============================================================
            // Non-periodic BC: Use CudaFFT (DCT/DST) with real coefficients
            // Only supports real fields (T = double). Non-periodic BC with
            // complex fields is not commonly needed in polymer field theory.
            // ============================================================

            if constexpr (!std::is_same<T, double>::value)
            {
                throw_with_line_number("ETDRK4 with non-periodic BC is only supported for real fields (double).");
            }
            else
            {
                // Workspace arrays (use direct pointers since T = double)
                T* d_a = d_etdrk4_a[STREAM];
                T* d_b = d_etdrk4_b[STREAM];
                T* d_c = d_etdrk4_c[STREAM];
                T* d_N_n = d_etdrk4_N_n[STREAM];
                T* d_N_a = d_etdrk4_N_a[STREAM];
                T* d_N_b = d_etdrk4_N_b[STREAM];
                T* d_N_c = d_etdrk4_N_c[STREAM];
                double* rk_q = d_rk_work[STREAM];  // Use rk_work as temp buffer
                double* rk_a = reinterpret_cast<double*>(d_k_a[STREAM]);
                double* rk_N_n = reinterpret_cast<double*>(d_k_N_n[STREAM]);
                double* rk_N_a = reinterpret_cast<double*>(d_k_N_a[STREAM]);
                double* rk_N_b = reinterpret_cast<double*>(d_k_N_b[STREAM]);
                double* rk_N_c = reinterpret_cast<double*>(d_k_N_c[STREAM]);
                double* rk_work = reinterpret_cast<double*>(d_k_work[STREAM]);

                // Step 1: Compute N_n = -w * q_in
                ker_multi<<<N_BLOCKS, N_THREADS>>>(d_N_n, _d_w, d_q_in, -1.0, M);
                gpu_error_check(cudaPeekAtLastError());

                // Step 2: Forward transforms
                fft_[STREAM]->forward(d_q_in, rk_q);
                fft_[STREAM]->forward(d_N_n, rk_N_n);

                // Stage a (Eq. 7a): â = E2·q̂ + α·N̂_n
                ker_etdrk4_stage_a_real<<<N_BLOCKS, N_THREADS>>>(
                    rk_a, rk_q, rk_N_n, _d_E2, _d_alpha, M_COMPLEX);
                gpu_error_check(cudaPeekAtLastError());

                // IFFT to get a
                fft_[STREAM]->backward(rk_a, d_a);

                // Compute N_a = -w * a
                ker_multi<<<N_BLOCKS, N_THREADS>>>(d_N_a, _d_w, d_a, -1.0, M);
                gpu_error_check(cudaPeekAtLastError());

                // FFT of N_a
                fft_[STREAM]->forward(d_N_a, rk_N_a);

                // Stage b (Eq. 7b): b̂ = â + φ₂_half·(N̂_a - N̂_n)
                ker_etdrk4_krogstad_stage_b_real<<<N_BLOCKS, N_THREADS>>>(
                    rk_work, rk_a, rk_N_a, rk_N_n, _d_phi2_half, M_COMPLEX);
                gpu_error_check(cudaPeekAtLastError());

                // IFFT to get b
                fft_[STREAM]->backward(rk_work, d_b);

                // Compute N_b = -w * b
                ker_multi<<<N_BLOCKS, N_THREADS>>>(d_N_b, _d_w, d_b, -1.0, M);
                gpu_error_check(cudaPeekAtLastError());

                // FFT of N_b
                fft_[STREAM]->forward(d_N_b, rk_N_b);

                // Stage c (Eq. 7c): ĉ = E·q̂ + φ₁·N̂_n + 2φ₂·(N̂_b - N̂_n)
                ker_etdrk4_krogstad_stage_c_real<<<N_BLOCKS, N_THREADS>>>(
                    rk_work, rk_q, rk_N_n, rk_N_b, _d_E, _d_phi1, _d_phi2, M_COMPLEX);
                gpu_error_check(cudaPeekAtLastError());

                // Save c_hat to rk_a before IFFT (rk_a is no longer needed after stage b)
                gpu_error_check(cudaMemcpy(rk_a, rk_work, sizeof(double)*M_COMPLEX, cudaMemcpyDeviceToDevice));

                // IFFT to get c
                fft_[STREAM]->backward(rk_work, d_c);

                // Compute N_c = -w * c
                ker_multi<<<N_BLOCKS, N_THREADS>>>(d_N_c, _d_w, d_c, -1.0, M);
                gpu_error_check(cudaPeekAtLastError());

                // FFT of N_c
                fft_[STREAM]->forward(d_N_c, rk_N_c);

                // Final step (Eq. 7d): q̂_{n+1} = ĉ + (4φ₃ - φ₂)·(N̂_n + N̂_c) + 2φ₂·N̂_a - 4φ₃·(N̂_a + N̂_b)
                // Use rk_a which has the saved c_hat
                ker_etdrk4_krogstad_final_real<<<N_BLOCKS, N_THREADS>>>(
                    rk_work, rk_a, rk_N_n, rk_N_a, rk_N_b, rk_N_c,
                    _d_phi2, _d_phi3, M_COMPLEX);
                gpu_error_check(cudaPeekAtLastError());

                // IFFT to get final result
                fft_[STREAM]->backward(rk_work, d_q_out);
            }
        }

        // Apply mask if provided
        if (d_q_mask != nullptr)
        {
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_q_out, d_q_out, d_q_mask, 1.0, M);
            gpu_error_check(cudaPeekAtLastError());
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

template <typename T>
void CudaSolverPseudoETDRK4<T>::compute_single_segment_stress(
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
            // Execute forward FFT for both propagators
            // Use out-of-place transforms to avoid cuFFT in-place D2Z issues
            if constexpr (std::is_same<T, double>::value)
            {
                // Out-of-place D2Z: read from d_q_pair, write to d_qk_stress
                cufftExecD2Z(plan_for_one[STREAM], d_q_pair, &d_qk_stress[STREAM][0]);
                cufftExecD2Z(plan_for_one[STREAM], &d_q_pair[M], &d_qk_stress[STREAM][M_COMPLEX]);
            }
            else
            {
                cufftExecZ2Z(plan_for_one[STREAM], d_q_pair, &d_qk_stress[STREAM][0], CUFFT_FORWARD);
                cufftExecZ2Z(plan_for_one[STREAM], &d_q_pair[M], &d_qk_stress[STREAM][M_COMPLEX], CUFFT_FORWARD);
            }
            gpu_error_check(cudaPeekAtLastError());

            // Multiply two propagators in the fourier spaces
            if constexpr (std::is_same<T, double>::value)
                ker_multi_complex_conjugate<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                    d_q_multi[STREAM], &d_qk_stress[STREAM][0], &d_qk_stress[STREAM][M_COMPLEX], M_COMPLEX);
            else
            {
                ker_copy_data_with_idx<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                    d_k_work[STREAM], &d_qk_stress[STREAM][M_COMPLEX], _d_negative_k_idx, M_COMPLEX);
                ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                    d_q_multi[STREAM], &d_qk_stress[STREAM][0], d_k_work[STREAM], 1.0, M_COMPLEX);
            }
            gpu_error_check(cudaPeekAtLastError());
        }
        else
        {
            // Non-periodic BC only supports real fields (double)
            if constexpr (!std::is_same<T, double>::value)
            {
                throw_with_line_number("ETDRK4 stress computation with non-periodic BC is only supported for real fields (double).");
            }
            else
            {
                // Non-periodic: use CudaFFT
                double* d_q1 = d_q_pair;
                double* d_q2 = &d_q_pair[M];
                double* rk_1 = reinterpret_cast<double*>(d_k_q[STREAM]);
                double* rk_2 = reinterpret_cast<double*>(d_k_work[STREAM]);

                fft_[STREAM]->forward(d_q1, rk_1);
                fft_[STREAM]->forward(d_q2, rk_2);

                // Multiply
                ker_multi<<<N_BLOCKS, N_THREADS>>>(d_q_multi[STREAM], rk_1, rk_2, 1.0, M_COMPLEX);
                gpu_error_check(cudaPeekAtLastError());
            }
        }

        // With k⊗k dyad product, fourier_basis arrays contain Cartesian components directly:
        // _d_fourier_basis_x = k_x², _d_fourier_basis_y = k_y², etc.
        // No cross-term corrections needed.

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
        else if ( DIM == 2 )
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

            // σ_xy (stored in index 2 for 2D)
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_stress_sum[STREAM], d_q_multi[STREAM], _d_fourier_basis_xy, bond_length_sq, M_COMPLEX);
            gpu_error_check(cudaPeekAtLastError());
            if constexpr (std::is_same<T, double>::value)
                cub::DeviceReduce::Sum(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[2], M_COMPLEX, streams[STREAM][0]);
            else
                cub::DeviceReduce::Reduce(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[2], M_COMPLEX, ComplexSumOp(), CuDeviceData<T>{0.0,0.0}, streams[STREAM][0]);
            gpu_error_check(cudaPeekAtLastError());
        }
        else if ( DIM == 1 )
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
template class CudaSolverPseudoETDRK4<double>;
template class CudaSolverPseudoETDRK4<std::complex<double>>;
