/**
 * @file CudaSolverPseudoETDRK4.cu
 * @brief CUDA ETDRK4 pseudo-spectral solver for continuous Gaussian chains.
 *
 * Implements the ETDRK4 (Exponential Time Differencing Runge-Kutta 4th order)
 * method for advancing chain propagators using cuFFT and multiple CUDA streams.
 *
 * **ETDRK4 Algorithm (Cox & Matthews 2002):**
 *
 * For dq/ds = L·q + N(q) where L = (b²/6)∇² and N(q) = -w·q:
 *
 * Stage a: â = E2·q̂ + α·N̂_n           where N_n = -w·q
 * Stage b: b̂ = E2·q̂ + α·N̂_a           where N_a = -w·a
 * Stage c: ĉ = E2·â + α·(2N̂_b - N̂_n)  where N_b = -w·b
 * Final:   q̂_{n+1} = E·q̂ + f1·N̂_n + f2·(N̂_a + N̂_b) + f3·N̂_c  where N_c = -w·c
 *
 * Coefficients E, E2, α, f1, f2, f3 are computed using the Kassam-Trefethen
 * contour integral method for numerical stability.
 *
 * **Template Instantiations:**
 *
 * - CudaSolverPseudoETDRK4<double>: Real field solver
 * - CudaSolverPseudoETDRK4<std::complex<double>>: Complex field solver
 *
 * @see ETDRK4Coefficients for coefficient computation
 * @see CudaSolverPseudoContinuous for RQM4 alternative
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
    [[maybe_unused]] bool reduce_memory_usage)
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
            cb->get_nx(), cb->get_dx(), molecules->get_ds(),
            cb->get_recip_metric());

        const int M = cb->get_total_grid();
        const int M_COMPLEX = pseudo->get_total_complex_grid();

        // Copy streams
        for(int i=0; i<n_streams; i++)
        {
            this->streams[i][0] = streams[i][0];
            this->streams[i][1] = streams[i][1];
        }

        // Create exp_dw, exp_dw_half, and w_field for each monomer type
        for(const auto& item: molecules->get_bond_lengths())
        {
            std::string monomer_type = item.first;
            this->d_exp_dw     [monomer_type] = nullptr;
            this->d_exp_dw_half[monomer_type] = nullptr;
            this->d_w_field    [monomer_type] = nullptr;
            gpu_error_check(cudaMalloc((void**)&this->d_exp_dw     [monomer_type], sizeof(T)*M));
            gpu_error_check(cudaMalloc((void**)&this->d_exp_dw_half[monomer_type], sizeof(T)*M));
            gpu_error_check(cudaMalloc((void**)&this->d_w_field    [monomer_type], sizeof(T)*M));
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
        }

        // Initialize ETDRK4 coefficients
        etdrk4_coefficients_ = std::make_unique<ETDRK4Coefficients<T>>(
            molecules->get_bond_lengths(),
            cb->get_boundary_conditions(),
            cb->get_nx(),
            cb->get_dx(),
            molecules->get_ds(),
            cb->get_recip_metric()
        );

        int coeff_size = etdrk4_coefficients_->get_total_complex_grid();

        // Allocate and copy coefficients to device
        for (const auto& item : molecules->get_bond_lengths())
        {
            std::string type = item.first;

            gpu_error_check(cudaMalloc((void**)&d_etdrk4_E[type], sizeof(double)*coeff_size));
            gpu_error_check(cudaMalloc((void**)&d_etdrk4_E2[type], sizeof(double)*coeff_size));
            gpu_error_check(cudaMalloc((void**)&d_etdrk4_alpha[type], sizeof(double)*coeff_size));
            gpu_error_check(cudaMalloc((void**)&d_etdrk4_f1[type], sizeof(double)*coeff_size));
            gpu_error_check(cudaMalloc((void**)&d_etdrk4_f2[type], sizeof(double)*coeff_size));
            gpu_error_check(cudaMalloc((void**)&d_etdrk4_f3[type], sizeof(double)*coeff_size));

            gpu_error_check(cudaMemcpy(d_etdrk4_E[type], etdrk4_coefficients_->get_E(type),
                sizeof(double)*coeff_size, cudaMemcpyHostToDevice));
            gpu_error_check(cudaMemcpy(d_etdrk4_E2[type], etdrk4_coefficients_->get_E2(type),
                sizeof(double)*coeff_size, cudaMemcpyHostToDevice));
            gpu_error_check(cudaMemcpy(d_etdrk4_alpha[type], etdrk4_coefficients_->get_alpha(type),
                sizeof(double)*coeff_size, cudaMemcpyHostToDevice));
            gpu_error_check(cudaMemcpy(d_etdrk4_f1[type], etdrk4_coefficients_->get_f1(type),
                sizeof(double)*coeff_size, cudaMemcpyHostToDevice));
            gpu_error_check(cudaMemcpy(d_etdrk4_f2[type], etdrk4_coefficients_->get_f2(type),
                sizeof(double)*coeff_size, cudaMemcpyHostToDevice));
            gpu_error_check(cudaMemcpy(d_etdrk4_f3[type], etdrk4_coefficients_->get_f3(type),
                sizeof(double)*coeff_size, cudaMemcpyHostToDevice));
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

    // Free Boltzmann factors
    for(const auto& item: this->d_exp_dw)
        cudaFree(item.second);
    for(const auto& item: this->d_exp_dw_half)
        cudaFree(item.second);
    for(const auto& item: this->d_w_field)
        cudaFree(item.second);

    // Free ETDRK4 coefficient arrays
    for(const auto& item: this->d_etdrk4_E)
        cudaFree(item.second);
    for(const auto& item: this->d_etdrk4_E2)
        cudaFree(item.second);
    for(const auto& item: this->d_etdrk4_alpha)
        cudaFree(item.second);
    for(const auto& item: this->d_etdrk4_f1)
        cudaFree(item.second);
    for(const auto& item: this->d_etdrk4_f2)
        cudaFree(item.second);
    for(const auto& item: this->d_etdrk4_f3)
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
        pseudo->update(
            this->cb->get_boundary_conditions(),
            this->molecules->get_bond_lengths(),
            this->cb->get_dx(), this->molecules->get_ds(),
            this->cb->get_recip_metric());

        // Update ETDRK4 coefficients for new box dimensions
        etdrk4_coefficients_->update(
            this->cb->get_boundary_conditions(),
            this->molecules->get_bond_lengths(),
            this->cb->get_dx(),
            this->molecules->get_ds(),
            this->cb->get_recip_metric()
        );

        // Copy updated coefficients to device
        int coeff_size = etdrk4_coefficients_->get_total_complex_grid();
        for (const auto& item : molecules->get_bond_lengths())
        {
            std::string type = item.first;
            gpu_error_check(cudaMemcpy(d_etdrk4_E[type], etdrk4_coefficients_->get_E(type),
                sizeof(double)*coeff_size, cudaMemcpyHostToDevice));
            gpu_error_check(cudaMemcpy(d_etdrk4_E2[type], etdrk4_coefficients_->get_E2(type),
                sizeof(double)*coeff_size, cudaMemcpyHostToDevice));
            gpu_error_check(cudaMemcpy(d_etdrk4_alpha[type], etdrk4_coefficients_->get_alpha(type),
                sizeof(double)*coeff_size, cudaMemcpyHostToDevice));
            gpu_error_check(cudaMemcpy(d_etdrk4_f1[type], etdrk4_coefficients_->get_f1(type),
                sizeof(double)*coeff_size, cudaMemcpyHostToDevice));
            gpu_error_check(cudaMemcpy(d_etdrk4_f2[type], etdrk4_coefficients_->get_f2(type),
                sizeof(double)*coeff_size, cudaMemcpyHostToDevice));
            gpu_error_check(cudaMemcpy(d_etdrk4_f3[type], etdrk4_coefficients_->get_f3(type),
                sizeof(double)*coeff_size, cudaMemcpyHostToDevice));
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
        const double ds = this->molecules->get_ds();

        for(const auto& item: w_input)
        {
            if( this->d_exp_dw.find(item.first) == this->d_exp_dw.end())
                throw_with_line_number("monomer_type \"" + item.first + "\" is not in d_exp_dw.");
        }

        cudaMemcpyKind cudaMemcpyInputToDevice;
        if (device == "gpu")
            cudaMemcpyInputToDevice = cudaMemcpyDeviceToDevice;
        else if(device == "cpu")
            cudaMemcpyInputToDevice = cudaMemcpyHostToDevice;
        else
        {
            throw_with_line_number("Invalid device \"" + device + "\".");
        }

        // Compute exp_dw and exp_dw_half, and copy raw w field
        for(const auto& item: w_input)
        {
            std::string monomer_type = item.first;
            const T *w = item.second;

            // Copy raw w field for ETDRK4
            gpu_error_check(cudaMemcpyAsync(
                this->d_w_field[monomer_type], w,
                sizeof(T)*M, cudaMemcpyInputToDevice));

            // Copy for exp computation
            gpu_error_check(cudaMemcpyAsync(
                this->d_exp_dw     [monomer_type], w,
                sizeof(T)*M, cudaMemcpyInputToDevice));
            gpu_error_check(cudaMemcpyAsync(
                this->d_exp_dw_half[monomer_type], w,
                sizeof(T)*M, cudaMemcpyInputToDevice));

            // Compute d_exp_dw and d_exp_dw_half
            ker_exp<<<N_BLOCKS, N_THREADS>>>
                (this->d_exp_dw[monomer_type],      this->d_exp_dw[monomer_type],      1.0, -0.50*ds, M);
            ker_exp<<<N_BLOCKS, N_THREADS>>>
                (this->d_exp_dw_half[monomer_type], this->d_exp_dw_half[monomer_type], 1.0, -0.25*ds, M);
            gpu_error_check(cudaPeekAtLastError());
            gpu_error_check(cudaDeviceSynchronize());
        }
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
        std::string monomer_type, double *d_q_mask)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = cb->get_total_grid();
        const int M_COMPLEX = pseudo->get_total_complex_grid();

        // Get ETDRK4 coefficients for this monomer type
        const double* _d_E = d_etdrk4_E[monomer_type];
        const double* _d_E2 = d_etdrk4_E2[monomer_type];
        const double* _d_alpha = d_etdrk4_alpha[monomer_type];
        const double* _d_f1 = d_etdrk4_f1[monomer_type];
        const double* _d_f2 = d_etdrk4_f2[monomer_type];
        const double* _d_f3 = d_etdrk4_f3[monomer_type];

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

            // Stage a: â = E2·q̂ + α·N̂_n
            ker_etdrk4_stage_a<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                d_k_a[STREAM], d_k_q[STREAM], d_k_N_n[STREAM], _d_E2, _d_alpha, M_COMPLEX);
            gpu_error_check(cudaPeekAtLastError());

            // IFFT to get a
            if constexpr (std::is_same<T, double>::value)
                cufftExecZ2D(plan_bak_one[STREAM], d_k_a[STREAM], d_etdrk4_a[STREAM]);
            else
                cufftExecZ2Z(plan_bak_one[STREAM], d_k_a[STREAM], d_etdrk4_a[STREAM], CUFFT_INVERSE);
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

            // Stage b: b̂ = E2·q̂ + α·N̂_a
            ker_etdrk4_stage_a<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                d_k_work[STREAM], d_k_q[STREAM], d_k_N_a[STREAM], _d_E2, _d_alpha, M_COMPLEX);
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

            // Stage c: ĉ = E2·â + α·(2N̂_b - N̂_n)
            ker_etdrk4_stage_c<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                d_k_work[STREAM], d_k_a[STREAM], d_k_N_b[STREAM], d_k_N_n[STREAM], _d_E2, _d_alpha, M_COMPLEX);
            gpu_error_check(cudaPeekAtLastError());

            // IFFT to get c
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

            // Final step: q̂_{n+1} = E·q̂ + f1·N̂_n + f2·(N̂_a + N̂_b) + f3·N̂_c
            ker_etdrk4_final<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                d_k_work[STREAM], d_k_q[STREAM], d_k_N_n[STREAM], d_k_N_a[STREAM],
                d_k_N_b[STREAM], d_k_N_c[STREAM], _d_E, _d_f1, _d_f2, _d_f3, M_COMPLEX);
            gpu_error_check(cudaPeekAtLastError());

            // IFFT to get final result
            if constexpr (std::is_same<T, double>::value)
                cufftExecZ2D(plan_bak_one[STREAM], d_k_work[STREAM], d_q_out);
            else
                cufftExecZ2Z(plan_bak_one[STREAM], d_k_work[STREAM], d_q_out, CUFFT_INVERSE);
            gpu_error_check(cudaPeekAtLastError());

            // Normalize final result
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                d_q_out, d_q_out, d_q_out, 1.0/static_cast<double>(M)/static_cast<double>(M), M);
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

                // Stage a: â = E2·q̂ + α·N̂_n
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

                // Stage b: b̂ = E2·q̂ + α·N̂_a
                ker_etdrk4_stage_a_real<<<N_BLOCKS, N_THREADS>>>(
                    rk_work, rk_q, rk_N_a, _d_E2, _d_alpha, M_COMPLEX);
                gpu_error_check(cudaPeekAtLastError());

                // IFFT to get b
                fft_[STREAM]->backward(rk_work, d_b);

                // Compute N_b = -w * b
                ker_multi<<<N_BLOCKS, N_THREADS>>>(d_N_b, _d_w, d_b, -1.0, M);
                gpu_error_check(cudaPeekAtLastError());

                // FFT of N_b
                fft_[STREAM]->forward(d_N_b, rk_N_b);

                // Stage c: ĉ = E2·â + α·(2N̂_b - N̂_n)
                ker_etdrk4_stage_c_real<<<N_BLOCKS, N_THREADS>>>(
                    rk_work, rk_a, rk_N_b, rk_N_n, _d_E2, _d_alpha, M_COMPLEX);
                gpu_error_check(cudaPeekAtLastError());

                // IFFT to get c
                fft_[STREAM]->backward(rk_work, d_c);

                // Compute N_c = -w * c
                ker_multi<<<N_BLOCKS, N_THREADS>>>(d_N_c, _d_w, d_c, -1.0, M);
                gpu_error_check(cudaPeekAtLastError());

                // FFT of N_c
                fft_[STREAM]->forward(d_N_c, rk_N_c);

                // Final step: q̂_{n+1} = E·q̂ + f1·N̂_n + f2·(N̂_a + N̂_b) + f3·N̂_c
                ker_etdrk4_final_real<<<N_BLOCKS, N_THREADS>>>(
                    rk_work, rk_q, rk_N_n, rk_N_a, rk_N_b, rk_N_c,
                    _d_E, _d_f1, _d_f2, _d_f3, M_COMPLEX);
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
            // Execute forward FFT for both propagators (batched)
            // We need two FFTs for forward and backward propagator
            if constexpr (std::is_same<T, double>::value)
            {
                // Copy both propagators to stress buffer
                gpu_error_check(cudaMemcpyAsync(&d_qk_stress[STREAM][0], d_q_pair,
                    sizeof(T)*M, cudaMemcpyDeviceToDevice, streams[STREAM][0]));
                gpu_error_check(cudaMemcpyAsync(&d_qk_stress[STREAM][M_COMPLEX], &d_q_pair[M],
                    sizeof(T)*M, cudaMemcpyDeviceToDevice, streams[STREAM][0]));

                cufftExecD2Z(plan_for_one[STREAM], reinterpret_cast<double*>(&d_qk_stress[STREAM][0]), &d_qk_stress[STREAM][0]);
                cufftExecD2Z(plan_for_one[STREAM], reinterpret_cast<double*>(&d_qk_stress[STREAM][M_COMPLEX]), &d_qk_stress[STREAM][M_COMPLEX]);
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

        if ( DIM == 3 )
        {
            // xx direction
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_stress_sum[STREAM], d_q_multi[STREAM], _d_fourier_basis_x, bond_length_sq, M_COMPLEX);
            gpu_error_check(cudaPeekAtLastError());
            if constexpr (std::is_same<T, double>::value)
                cub::DeviceReduce::Sum(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[0], M_COMPLEX, streams[STREAM][0]);
            else
                cub::DeviceReduce::Reduce(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[0], M_COMPLEX, ComplexSumOp(), CuDeviceData<T>{0.0,0.0}, streams[STREAM][0]);
            gpu_error_check(cudaPeekAtLastError());

            // yy direction
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_stress_sum[STREAM], d_q_multi[STREAM], _d_fourier_basis_y, bond_length_sq, M_COMPLEX);
            gpu_error_check(cudaPeekAtLastError());
            if constexpr (std::is_same<T, double>::value)
                cub::DeviceReduce::Sum(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[1], M_COMPLEX, streams[STREAM][0]);
            else
                cub::DeviceReduce::Reduce(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[1], M_COMPLEX, ComplexSumOp(), CuDeviceData<T>{0.0,0.0}, streams[STREAM][0]);
            gpu_error_check(cudaPeekAtLastError());

            // zz direction
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_stress_sum[STREAM], d_q_multi[STREAM], _d_fourier_basis_z, bond_length_sq, M_COMPLEX);
            gpu_error_check(cudaPeekAtLastError());
            if constexpr (std::is_same<T, double>::value)
                cub::DeviceReduce::Sum(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[2], M_COMPLEX, streams[STREAM][0]);
            else
                cub::DeviceReduce::Reduce(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[2], M_COMPLEX, ComplexSumOp(), CuDeviceData<T>{0.0,0.0}, streams[STREAM][0]);
            gpu_error_check(cudaPeekAtLastError());

            // Cross-terms for non-orthogonal systems
            if (!this->cb->is_orthogonal())
            {
                // xy cross-term
                ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_stress_sum[STREAM], d_q_multi[STREAM], _d_fourier_basis_xy, bond_length_sq, M_COMPLEX);
                gpu_error_check(cudaPeekAtLastError());
                if constexpr (std::is_same<T, double>::value)
                    cub::DeviceReduce::Sum(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[3], M_COMPLEX, streams[STREAM][0]);
                else
                    cub::DeviceReduce::Reduce(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[3], M_COMPLEX, ComplexSumOp(), CuDeviceData<T>{0.0,0.0}, streams[STREAM][0]);
                gpu_error_check(cudaPeekAtLastError());

                // xz cross-term
                ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_stress_sum[STREAM], d_q_multi[STREAM], _d_fourier_basis_xz, bond_length_sq, M_COMPLEX);
                gpu_error_check(cudaPeekAtLastError());
                if constexpr (std::is_same<T, double>::value)
                    cub::DeviceReduce::Sum(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[4], M_COMPLEX, streams[STREAM][0]);
                else
                    cub::DeviceReduce::Reduce(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[4], M_COMPLEX, ComplexSumOp(), CuDeviceData<T>{0.0,0.0}, streams[STREAM][0]);
                gpu_error_check(cudaPeekAtLastError());

                // yz cross-term
                ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_stress_sum[STREAM], d_q_multi[STREAM], _d_fourier_basis_yz, bond_length_sq, M_COMPLEX);
                gpu_error_check(cudaPeekAtLastError());
                if constexpr (std::is_same<T, double>::value)
                    cub::DeviceReduce::Sum(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[5], M_COMPLEX, streams[STREAM][0]);
                else
                    cub::DeviceReduce::Reduce(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[5], M_COMPLEX, ComplexSumOp(), CuDeviceData<T>{0.0,0.0}, streams[STREAM][0]);
                gpu_error_check(cudaPeekAtLastError());
            }
        }
        else if ( DIM == 2 )
        {
            // lx[0] direction
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_stress_sum[STREAM], d_q_multi[STREAM], _d_fourier_basis_x, bond_length_sq, M_COMPLEX);
            gpu_error_check(cudaPeekAtLastError());
            if constexpr (std::is_same<T, double>::value)
                cub::DeviceReduce::Sum(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[0], M_COMPLEX, streams[STREAM][0]);
            else
                cub::DeviceReduce::Reduce(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[0], M_COMPLEX, ComplexSumOp(), CuDeviceData<T>{0.0,0.0}, streams[STREAM][0]);
            gpu_error_check(cudaPeekAtLastError());

            // lx[1] direction
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_stress_sum[STREAM], d_q_multi[STREAM], _d_fourier_basis_y, bond_length_sq, M_COMPLEX);
            gpu_error_check(cudaPeekAtLastError());
            if constexpr (std::is_same<T, double>::value)
                cub::DeviceReduce::Sum(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[1], M_COMPLEX, streams[STREAM][0]);
            else
                cub::DeviceReduce::Reduce(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[1], M_COMPLEX, ComplexSumOp(), CuDeviceData<T>{0.0,0.0}, streams[STREAM][0]);
            gpu_error_check(cudaPeekAtLastError());

            // xy cross-term
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
            // lx[0] direction
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
