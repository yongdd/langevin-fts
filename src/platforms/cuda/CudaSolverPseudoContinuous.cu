#include <iostream>
#include <cmath>
#include <thrust/reduce.h>
#include "CudaSolverPseudoContinuous.h"

template <typename T>
CudaSolverPseudoContinuous<T>::CudaSolverPseudoContinuous(
    ComputationBox<T>* cb,
    Molecules *molecules,
    int n_streams,
    cudaStream_t streams[MAX_STREAMS][2],
    bool reduce_gpu_memory_usage)
{
    try{
        this->cb = cb;
        this->molecules = molecules;
        this->chain_model = molecules->get_model_name();
        this->n_streams = n_streams;
        
        const int M = cb->get_total_grid(); 
        const int M_COMPLEX = Pseudo<T>::get_total_complex_grid(cb->get_nx());;

        // Copy streams
        for(int i=0; i<n_streams; i++)
        {
            this->streams[i][0] = streams[i][0];
            this->streams[i][1] = streams[i][1];
        }

        // Create boltz_bond, boltz_bond_half, exp_dw, and exp_dw_half
        for(const auto& item: molecules->get_bond_lengths())
        {
            std::string monomer_type = item.first;
            d_boltz_bond       [monomer_type] = nullptr;
            d_boltz_bond_half  [monomer_type] = nullptr;
            this->d_exp_dw     [monomer_type] = nullptr;
            this->d_exp_dw_half[monomer_type] = nullptr;

            gpu_error_check(cudaMalloc((void**)&d_boltz_bond       [monomer_type], sizeof(double)*M_COMPLEX));
            gpu_error_check(cudaMalloc((void**)&d_boltz_bond_half  [monomer_type], sizeof(double)*M_COMPLEX));
            gpu_error_check(cudaMalloc((void**)&this->d_exp_dw     [monomer_type], sizeof(T)*M));
            gpu_error_check(cudaMalloc((void**)&this->d_exp_dw_half[monomer_type], sizeof(T)*M));
        }

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
            cufftPlanMany(&plan_for_one[i], NRANK, total_grid, NULL, 1, 0, NULL, 1, 0, cufft_forward, 1);
            cufftPlanMany(&plan_for_two[i], NRANK, total_grid, NULL, 1, 0, NULL, 1, 0, cufft_forward, 2);
            cufftPlanMany(&plan_bak_one[i], NRANK, total_grid, NULL, 1, 0, NULL, 1, 0, cufft_backward, 1);
            cufftPlanMany(&plan_bak_two[i], NRANK, total_grid, NULL, 1, 0, NULL, 1, 0, cufft_backward, 2);
            cufftSetStream(plan_for_one[i], streams[i][0]);
            cufftSetStream(plan_for_two[i], streams[i][0]);
            cufftSetStream(plan_bak_one[i], streams[i][0]);
            cufftSetStream(plan_bak_two[i], streams[i][0]);
        }

        // Allocate memory for pseudo-spectral: advance_propagator()
        for(int i=0; i<n_streams; i++)
        {  
            gpu_error_check(cudaMalloc((void**)&d_q_step_1_one[i], sizeof(T)*M));
            gpu_error_check(cudaMalloc((void**)&d_q_step_2_one[i], sizeof(T)*M));
            gpu_error_check(cudaMalloc((void**)&d_q_step_1_two[i], sizeof(T)*2*M));
            
            gpu_error_check(cudaMalloc((void**)&d_qk_in_2_one[i], sizeof(cuDoubleComplex)*M_COMPLEX));
            gpu_error_check(cudaMalloc((void**)&d_qk_in_1_two[i], sizeof(cuDoubleComplex)*2*M_COMPLEX));
        }
    
        // Allocate memory for stress calculation: compute_stress()
        gpu_error_check(cudaMalloc((void**)&d_fourier_basis_x, sizeof(double)*M_COMPLEX));
        gpu_error_check(cudaMalloc((void**)&d_fourier_basis_y, sizeof(double)*M_COMPLEX));
        gpu_error_check(cudaMalloc((void**)&d_fourier_basis_z, sizeof(double)*M_COMPLEX));

        if constexpr (std::is_same<T, std::complex<double>>::value)
        {
            int k_idx[M_COMPLEX];
            Pseudo<T>::get_negative_frequency_mapping(cb->get_nx(), k_idx);
            gpu_error_check(cudaMalloc((void**)&d_k_idx, sizeof(int)*M_COMPLEX));
            gpu_error_check(cudaMemcpy(d_k_idx, k_idx, sizeof(int)*M_COMPLEX,cudaMemcpyHostToDevice));
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
        update_laplacian_operator();
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
CudaSolverPseudoContinuous<T>::~CudaSolverPseudoContinuous()
{
    for(int i=0; i<n_streams; i++)
    {
        cufftDestroy(plan_for_one[i]);
        cufftDestroy(plan_for_two[i]);
        cufftDestroy(plan_bak_one[i]);
        cufftDestroy(plan_bak_two[i]);
    }

    for(const auto& item: d_boltz_bond)
        cudaFree(item.second);
    for(const auto& item: d_boltz_bond_half)
        cudaFree(item.second);
    for(const auto& item: this->d_exp_dw)
        cudaFree(item.second);
    for(const auto& item: this->d_exp_dw_half)
        cudaFree(item.second);

    // For pseudo-spectral: advance_propagator()
    for(int i=0; i<n_streams; i++)
    {
        cudaFree(d_q_step_1_one[i]);
        cudaFree(d_q_step_2_one[i]);
        cudaFree(d_q_step_1_two[i]);
        cudaFree(d_qk_in_2_one[i]);
        cudaFree(d_qk_in_1_two[i]);
    }

    // For stress calculation: compute_stress()
    cudaFree(d_fourier_basis_x);
    cudaFree(d_fourier_basis_y);
    cudaFree(d_fourier_basis_z);
    if constexpr (std::is_same<T, std::complex<double>>::value)
        cudaFree(d_k_idx);

    for(int i=0; i<n_streams; i++)
    {
        cudaFree(d_stress_sum[i]);
        cudaFree(d_stress_sum_out[i]);
        cudaFree(d_q_multi[i]);
        cudaFree(d_temp_storage[i]);
    }
}
template <typename T>
void CudaSolverPseudoContinuous<T>::update_laplacian_operator()
{
    try{
        // For pseudo-spectral: advance_propagator()
        const int M_COMPLEX = Pseudo<T>::get_total_complex_grid(cb->get_nx());;
        double boltz_bond[M_COMPLEX], boltz_bond_half[M_COMPLEX];

        for(const auto& item: this->molecules->get_bond_lengths())
        {
            std::string monomer_type = item.first;
            double bond_length_sq = item.second*item.second;
            
            Pseudo<T>::get_boltz_bond(this->cb->get_boundary_conditions(), boltz_bond     , bond_length_sq,   this->cb->get_nx(), this->cb->get_dx(), this->molecules->get_ds());
            Pseudo<T>::get_boltz_bond(this->cb->get_boundary_conditions(), boltz_bond_half, bond_length_sq/2, this->cb->get_nx(), this->cb->get_dx(), this->molecules->get_ds());    

            gpu_error_check(cudaMemcpy(d_boltz_bond     [monomer_type], boltz_bond,      sizeof(double)*M_COMPLEX,cudaMemcpyHostToDevice));
            gpu_error_check(cudaMemcpy(d_boltz_bond_half[monomer_type], boltz_bond_half, sizeof(double)*M_COMPLEX,cudaMemcpyHostToDevice));
        }

        // For stress calculation: compute_stress()
        double fourier_basis_x[M_COMPLEX];
        double fourier_basis_y[M_COMPLEX];
        double fourier_basis_z[M_COMPLEX];

        Pseudo<T>::get_weighted_fourier_basis(this->cb->get_boundary_conditions(), fourier_basis_x, fourier_basis_y, fourier_basis_z, this->cb->get_nx(), this->cb->get_dx());

        gpu_error_check(cudaMemcpy(d_fourier_basis_x, fourier_basis_x, sizeof(double)*M_COMPLEX,cudaMemcpyHostToDevice));
        gpu_error_check(cudaMemcpy(d_fourier_basis_y, fourier_basis_y, sizeof(double)*M_COMPLEX,cudaMemcpyHostToDevice));
        gpu_error_check(cudaMemcpy(d_fourier_basis_z, fourier_basis_z, sizeof(double)*M_COMPLEX,cudaMemcpyHostToDevice));
    }
    catch(std::exception& exc)
    {
        throw_with_line_number(exc.what());
    }
}
template <typename T>
void CudaSolverPseudoContinuous<T>::update_dw(std::string device, std::map<std::string, const T*> w_input)
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

        // Compute exp_dw and exp_dw_half
        for(const auto& item: w_input)
        {
            std::string monomer_type = item.first;
            const T *w = item.second;

            // Copy field configurations from host to device
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

            gpu_error_check(cudaDeviceSynchronize());
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
// Advance propagator using Richardson extrapolation
template <typename T>
void CudaSolverPseudoContinuous<T>::advance_propagator(
        const int STREAM,
        CuDeviceData<T> *d_q_in, CuDeviceData<T> *d_q_out,
        std::string monomer_type, double *d_q_mask)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = cb->get_total_grid();
        const int M_COMPLEX = Pseudo<T>::get_total_complex_grid(cb->get_nx());;

        CuDeviceData<T> *_d_exp_dw = this->d_exp_dw[monomer_type];
        CuDeviceData<T> *_d_exp_dw_half = this->d_exp_dw_half[monomer_type];
        double *_d_boltz_bond = d_boltz_bond[monomer_type];
        double *_d_boltz_bond_half = d_boltz_bond_half[monomer_type];

        // step 1/2: Evaluate exp(-w*ds/2) in real space
        // step 1/4: Evaluate exp(-w*ds/4) in real space
        ker_multi_exp_dw_two<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
            &d_q_step_1_two[STREAM][0], d_q_in, _d_exp_dw,
            &d_q_step_1_two[STREAM][M], d_q_in, _d_exp_dw_half, 1.0, M);

        // step 1/2: Execute a forward FFT
        // step 1/4: Execute a forward FFT
        if constexpr (std::is_same<T, double>::value)
            cufftExecD2Z(plan_for_two[STREAM], d_q_step_1_two[STREAM], d_qk_in_1_two[STREAM]);
        else
            cufftExecZ2Z(plan_for_two[STREAM], d_q_step_1_two[STREAM], d_qk_in_1_two[STREAM], CUFFT_FORWARD);

        // step 1/2: Multiply exp(-k^2 ds/6)  in fourier space
        // step 1/4: Multiply exp(-k^2 ds/12) in fourier space
        ker_complex_real_multi_bond_two<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
            &d_qk_in_1_two[STREAM][0],         _d_boltz_bond,
            &d_qk_in_1_two[STREAM][M_COMPLEX], _d_boltz_bond_half, M_COMPLEX);

        // step 1/2: Execute a backward FFT
        // step 1/4: Execute a backward FFT
        if constexpr (std::is_same<T, double>::value)
            cufftExecZ2D(plan_bak_two[STREAM], d_qk_in_1_two[STREAM], d_q_step_1_two[STREAM]);
        else
            cufftExecZ2Z(plan_bak_two[STREAM], d_qk_in_1_two[STREAM], d_q_step_1_two[STREAM], CUFFT_INVERSE);

        // step 1/2: Evaluate exp(-w*ds/2) in real space
        // step 1/4: Evaluate exp(-w*ds/2) in real space
        ker_multi_exp_dw_two<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
            d_q_step_1_one[STREAM], &d_q_step_1_two[STREAM][0], _d_exp_dw,
            d_q_step_2_one[STREAM], &d_q_step_1_two[STREAM][M], _d_exp_dw, 1.0/static_cast<double>(M), M);

        // step 1/4: Execute a forward FFT
        if constexpr (std::is_same<T, double>::value)
            cufftExecD2Z(plan_for_one[STREAM], d_q_step_2_one[STREAM], d_qk_in_2_one[STREAM]);
        else
            cufftExecZ2Z(plan_for_one[STREAM], d_q_step_2_one[STREAM], d_qk_in_2_one[STREAM], CUFFT_FORWARD);

        // step 1/4: Multiply exp(-k^2 ds/12) in fourier space
        ker_multi_complex_real<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_qk_in_2_one[STREAM], _d_boltz_bond_half, 1.0, M_COMPLEX);

        // step 1/4: Execute a backward FFT
        if constexpr (std::is_same<T, double>::value)
            cufftExecZ2D(plan_bak_one[STREAM], d_qk_in_2_one[STREAM], d_q_step_2_one[STREAM]);
        else
            cufftExecZ2Z(plan_bak_one[STREAM], d_qk_in_2_one[STREAM], d_q_step_2_one[STREAM], CUFFT_INVERSE);

        // step 1/4: Evaluate exp(-w*ds/4) in real space.
        ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_q_step_2_one[STREAM], d_q_step_2_one[STREAM], _d_exp_dw_half, 1.0/static_cast<double>(M), M);

        // Compute linear combination with 4/3 and -1/3 ratio
        ker_lin_comb<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_q_out, 4.0/3.0, d_q_step_2_one[STREAM], -1.0/3.0, d_q_step_1_one[STREAM], M);

        // Multiply mask
        if (d_q_mask != nullptr)
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_q_out, d_q_out, d_q_mask, 1.0, M);
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
void CudaSolverPseudoContinuous<T>::compute_single_segment_stress(
    const int STREAM,
    CuDeviceData<T> *d_q_pair, CuDeviceData<T>  *d_segment_stress,
    std::string monomer_type, bool is_half_bond_length)
{
    try{
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int DIM = this->cb->get_dim();
        // const int M   = total_grid;
        const int M_COMPLEX = Pseudo<T>::get_total_complex_grid(cb->get_nx());;

        auto bond_lengths = this->molecules->get_bond_lengths();
        double bond_length_sq = bond_lengths[monomer_type]*bond_lengths[monomer_type];

        // Execute a forward FFT
        if constexpr (std::is_same<T, double>::value)
            cufftExecD2Z(plan_for_two[STREAM], d_q_pair, d_qk_in_1_two[STREAM]);
        else
            cufftExecZ2Z(plan_for_two[STREAM], d_q_pair, d_qk_in_1_two[STREAM], CUFFT_FORWARD);

        // Multiply two propagators in the fourier spaces
        if constexpr (std::is_same<T, double>::value)
            ker_multi_complex_conjugate<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_q_multi[STREAM], &d_qk_in_1_two[STREAM][0], &d_qk_in_1_two[STREAM][M_COMPLEX], M_COMPLEX);
        else
        {
            // TODO
            // ker_copy_data_with_idx<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_qk_in_2_one[STREAM], &d_qk_in_1_two[STREAM][M_COMPLEX], d_k_idx, M_COMPLEX);
            // ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_q_multi[STREAM], &d_qk_in_1_two[STREAM][0], d_qk_in_2_one[STREAM], 1.0, M_COMPLEX);
        }
        if ( DIM == 3 )
        {
            // x direction
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_stress_sum[STREAM], d_q_multi[STREAM], d_fourier_basis_x, bond_length_sq, M_COMPLEX);
            if constexpr (std::is_same<T, double>::value)
                cub::DeviceReduce::Sum(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[0], M_COMPLEX, streams[STREAM][0]);
            else
                cub::DeviceReduce::Reduce(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[0], M_COMPLEX, ComplexSumOp(), CuDeviceData<T>{0.0,0.0}, streams[STREAM][0]);

            // y direction
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_stress_sum[STREAM], d_q_multi[STREAM], d_fourier_basis_y, bond_length_sq, M_COMPLEX);
            if constexpr (std::is_same<T, double>::value)
                cub::DeviceReduce::Sum(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[1], M_COMPLEX, streams[STREAM][0]);
            else
                cub::DeviceReduce::Reduce(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[1], M_COMPLEX, ComplexSumOp(), CuDeviceData<T>{0.0,0.0}, streams[STREAM][0]);

            // z direction
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_stress_sum[STREAM], d_q_multi[STREAM], d_fourier_basis_z, bond_length_sq, M_COMPLEX);
            if constexpr (std::is_same<T, double>::value)
                cub::DeviceReduce::Sum(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[2], M_COMPLEX, streams[STREAM][0]);
            else
                cub::DeviceReduce::Reduce(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[2], M_COMPLEX, ComplexSumOp(), CuDeviceData<T>{0.0,0.0}, streams[STREAM][0]);
        }
        if ( DIM == 2 )
        {
            // y direction
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_stress_sum[STREAM], d_q_multi[STREAM], d_fourier_basis_y, bond_length_sq, M_COMPLEX);
            if constexpr (std::is_same<T, double>::value)
                cub::DeviceReduce::Sum(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[0], M_COMPLEX, streams[STREAM][0]);
            else
                cub::DeviceReduce::Reduce(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[0], M_COMPLEX, ComplexSumOp(), CuDeviceData<T>{0.0,0.0}, streams[STREAM][0]);

            // z direction
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_stress_sum[STREAM], d_q_multi[STREAM], d_fourier_basis_z, bond_length_sq, M_COMPLEX);
            if constexpr (std::is_same<T, double>::value)
                cub::DeviceReduce::Sum(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[1], M_COMPLEX, streams[STREAM][0]);
            else
                cub::DeviceReduce::Reduce(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[1], M_COMPLEX, ComplexSumOp(), CuDeviceData<T>{0.0,0.0}, streams[STREAM][0]);
        }
        if ( DIM == 1 )
        {
            // z direction
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_stress_sum[STREAM], d_q_multi[STREAM], d_fourier_basis_z, bond_length_sq, M_COMPLEX);
            if constexpr (std::is_same<T, double>::value)
                cub::DeviceReduce::Sum(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[0], M_COMPLEX, streams[STREAM][0]);
            else
                cub::DeviceReduce::Reduce(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[0], M_COMPLEX, ComplexSumOp(), CuDeviceData<T>{0.0,0.0}, streams[STREAM][0]);
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

// Explicit template instantiation
template class CudaSolverPseudoContinuous<double>;
template class CudaSolverPseudoContinuous<std::complex<double>>;