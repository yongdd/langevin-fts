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
#include <thrust/reduce.h>

#include "CudaPseudo.h"
#include "CudaSolverPseudoRK2.h"

template <typename T>
CudaSolverPseudoRK2<T>::CudaSolverPseudoRK2(
    ComputationBox<T>* cb,
    Molecules *molecules,
    int n_streams,
    cudaStream_t streams[MAX_STREAMS][2],
    [[maybe_unused]] bool use_checkpointing)
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

        // Ensure ContourLengthMapping is finalized before using it
        molecules->finalize_contour_length_mapping();

        // Get unique ds values from ContourLengthMapping
        const ContourLengthMapping& mapping = molecules->get_contour_length_mapping();
        int n_unique_ds = mapping.get_n_unique_ds();

        // Create exp_dw for each ds_index and monomer_type
        // Note: RK2 doesn't need exp_dw_half (no Richardson extrapolation)
        for (int ds_idx = 1; ds_idx <= n_unique_ds; ++ds_idx)
        {
            for(const auto& item: molecules->get_bond_lengths())
            {
                std::string monomer_type = item.first;
                this->d_exp_dw     [ds_idx][monomer_type] = nullptr;
                this->d_exp_dw_half[ds_idx][monomer_type] = nullptr;  // Allocate for compatibility
                gpu_error_check(cudaMalloc((void**)&this->d_exp_dw     [ds_idx][monomer_type], sizeof(T)*M));
                gpu_error_check(cudaMalloc((void**)&this->d_exp_dw_half[ds_idx][monomer_type], sizeof(T)*M));
            }
        }

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
    }
}

template <typename T>
void CudaSolverPseudoRK2<T>::update_laplacian_operator()
{
    try{
        // Update Pseudo with global_ds
        pseudo->update(
            this->cb->get_boundary_conditions(),
            this->molecules->get_bond_lengths(),
            this->cb->get_dx(), this->molecules->get_ds(),
            this->cb->get_recip_metric());

        // Re-register local_ds values for each block
        const ContourLengthMapping& mapping = this->molecules->get_contour_length_mapping();
        int n_unique_ds = mapping.get_n_unique_ds();

        for (int ds_idx = 1; ds_idx <= n_unique_ds; ++ds_idx)
        {
            double local_ds = mapping.get_ds_from_index(ds_idx);
            pseudo->add_ds_value(ds_idx, local_ds);
        }

        // Finalize Pseudo to compute boltz_bond with correct local_ds
        pseudo->finalize_ds_values();
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
        for (int ds_idx = 1; ds_idx <= n_unique_ds; ++ds_idx)
        {
            double local_ds = mapping.get_ds_from_index(ds_idx);

            for(const auto& item: w_input)
            {
                std::string monomer_type = item.first;
                const T *w = item.second;

                if (this->d_exp_dw[ds_idx].find(monomer_type) == this->d_exp_dw[ds_idx].end())
                    throw_with_line_number("monomer_type \"" + monomer_type + "\" is not in d_exp_dw[" + std::to_string(ds_idx) + "].");

                // Copy field configurations from host to device
                gpu_error_check(cudaMemcpyAsync(
                    this->d_exp_dw[ds_idx][monomer_type], w,
                    sizeof(T)*M, cudaMemcpyInputToDevice));

                // Compute d_exp_dw = exp(-w * local_ds * 0.5)
                ker_exp<<<N_BLOCKS, N_THREADS>>>
                    (this->d_exp_dw[ds_idx][monomer_type], this->d_exp_dw[ds_idx][monomer_type], 1.0, -0.50*local_ds, M);
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

        if (is_periodic_)
        {
            // ============================================================
            // Periodic BC: Use cuFFT with complex coefficients
            // ============================================================

            // Step 1: Evaluate exp(-w*ds/2) in real space
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                d_q_step_1[STREAM], d_q_in, _d_exp_dw, 1.0, M);
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
                cufftExecZ2D(plan_bak_one[STREAM], d_qk_in_1[STREAM], d_q_out);
            else
                cufftExecZ2Z(plan_bak_one[STREAM], d_qk_in_1[STREAM], d_q_out, CUFFT_INVERSE);
            gpu_error_check(cudaPeekAtLastError());

            // Step 5: Evaluate exp(-w*ds/2) in real space and normalize
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                d_q_out, d_q_out, _d_exp_dw, 1.0/static_cast<double>(M), M);
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
            ker_multi<<<N_BLOCKS, N_THREADS>>>(d_q_step_1[STREAM], d_q_in, _d_exp_dw, 1.0, M);
            gpu_error_check(cudaPeekAtLastError());

            // Step 2: Forward transform (DCT/DST)
            fft_[STREAM]->forward(d_q_step, d_rk_in_1[STREAM]);

            // Step 3: Multiply by Boltzmann factor (real coefficients)
            ker_multi<<<N_BLOCKS, N_THREADS>>>(d_rk_in_1[STREAM], d_rk_in_1[STREAM], _d_boltz_bond, 1.0, M_COMPLEX);
            gpu_error_check(cudaPeekAtLastError());

            // Step 4: Backward transform (inverse DCT/DST) - normalization included
            fft_[STREAM]->backward(d_rk_in_1[STREAM], reinterpret_cast<T*>(d_q_out));

            // Step 5: Evaluate exp(-w*ds/2) in real space
            ker_multi<<<N_BLOCKS, N_THREADS>>>(d_q_out, d_q_out, _d_exp_dw, 1.0, M);
            gpu_error_check(cudaPeekAtLastError());
        }

        // Multiply mask
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

        // Execute a forward FFT (batched for 2 fields)
        // Note: We need 2 input fields for stress computation, reuse d_qk_in_1/2
        // Copy to workspace for batched FFT
        gpu_error_check(cudaMemcpyAsync(d_q_step_1[STREAM], d_q_pair, sizeof(T)*M, cudaMemcpyDeviceToDevice, streams[STREAM][0]));
        gpu_error_check(cudaMemcpyAsync(d_q_step_2[STREAM], &d_q_pair[M], sizeof(T)*M, cudaMemcpyDeviceToDevice, streams[STREAM][0]));

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
        if ( DIM == 2 )
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

            // xy cross-term
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_stress_sum[STREAM], d_q_multi[STREAM], _d_fourier_basis_xy, bond_length_sq, M_COMPLEX);
            gpu_error_check(cudaPeekAtLastError());
            if constexpr (std::is_same<T, double>::value)
                cub::DeviceReduce::Sum(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[2], M_COMPLEX, streams[STREAM][0]);
            else
                cub::DeviceReduce::Reduce(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[2], M_COMPLEX, ComplexSumOp(), CuDeviceData<T>{0.0,0.0}, streams[STREAM][0]);
            gpu_error_check(cudaPeekAtLastError());
        }
        if ( DIM == 1 )
        {
            // xx direction
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
