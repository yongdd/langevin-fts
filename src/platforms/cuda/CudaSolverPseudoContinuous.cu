#include <iostream>
#include <cmath>
#include <thrust/reduce.h>
#include "CudaSolverPseudoContinuous.h"

CudaSolverPseudoContinuous::CudaSolverPseudoContinuous(
    ComputationBox<double>* cb,
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
        const int M_COMPLEX = Pseudo::get_total_complex_grid<double>(cb->get_nx());
        const int N_GPUS = CudaCommon::get_instance().get_n_gpus();

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
            for(int gpu=0; gpu<N_GPUS; gpu++)
            {
                gpu_error_check(cudaSetDevice(gpu));
                d_boltz_bond       [gpu][monomer_type] = nullptr;
                d_boltz_bond_half  [gpu][monomer_type] = nullptr;
                this->d_exp_dw     [gpu][monomer_type] = nullptr;
                this->d_exp_dw_half[gpu][monomer_type] = nullptr;

                gpu_error_check(cudaMalloc((void**)&d_boltz_bond       [gpu][monomer_type], sizeof(double)*M_COMPLEX));
                gpu_error_check(cudaMalloc((void**)&d_boltz_bond_half  [gpu][monomer_type], sizeof(double)*M_COMPLEX));
                gpu_error_check(cudaMalloc((void**)&this->d_exp_dw     [gpu][monomer_type], sizeof(double)*M));
                gpu_error_check(cudaMalloc((void**)&this->d_exp_dw_half[gpu][monomer_type], sizeof(double)*M));
            }
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

        for(int i=0; i<n_streams; i++)
        {  
            gpu_error_check(cudaSetDevice(i % N_GPUS));
                
            cufftPlanMany(&plan_for_one[i], NRANK, total_grid, NULL, 1, 0, NULL, 1, 0, CUFFT_D2Z,1);
            cufftPlanMany(&plan_for_two[i], NRANK, total_grid, NULL, 1, 0, NULL, 1, 0, CUFFT_D2Z,2);
            cufftPlanMany(&plan_bak_one[i], NRANK, total_grid, NULL, 1, 0, NULL, 1, 0, CUFFT_Z2D,1);
            cufftPlanMany(&plan_bak_two[i], NRANK, total_grid, NULL, 1, 0, NULL, 1, 0, CUFFT_Z2D,2);
            cufftSetStream(plan_for_one[i], streams[i][0]);
            cufftSetStream(plan_for_two[i], streams[i][0]);
            cufftSetStream(plan_bak_one[i], streams[i][0]);
            cufftSetStream(plan_bak_two[i], streams[i][0]);
        }

        gpu_error_check(cudaSetDevice(0));
        // Allocate memory for pseudo-spectral: advance_propagator()
        for(int i=0; i<n_streams; i++)
        {  
            gpu_error_check(cudaSetDevice(i % N_GPUS));
                
            gpu_error_check(cudaMalloc((void**)&d_q_step_1_one[i], sizeof(double)*M));
            gpu_error_check(cudaMalloc((void**)&d_q_step_2_one[i], sizeof(double)*M));
            gpu_error_check(cudaMalloc((void**)&d_q_step_1_two[i], sizeof(double)*2*M));
            
            gpu_error_check(cudaMalloc((void**)&d_qk_in_2_one[i], sizeof(ftsComplex)*M_COMPLEX));
            gpu_error_check(cudaMalloc((void**)&d_qk_in_1_two[i], sizeof(ftsComplex)*2*M_COMPLEX));
        }
    
        // Allocate memory for stress calculation: compute_stress()
        for(int gpu=0; gpu<N_GPUS; gpu++)
        {
            gpu_error_check(cudaSetDevice(gpu));
            gpu_error_check(cudaMalloc((void**)&d_fourier_basis_x[gpu], sizeof(double)*M_COMPLEX));
            gpu_error_check(cudaMalloc((void**)&d_fourier_basis_y[gpu], sizeof(double)*M_COMPLEX));
            gpu_error_check(cudaMalloc((void**)&d_fourier_basis_z[gpu], sizeof(double)*M_COMPLEX));
        }

        for(int i=0; i<n_streams; i++)
        {  
            gpu_error_check(cudaSetDevice(i % N_GPUS));

            gpu_error_check(cudaMalloc((void**)&d_stress_sum[i],      sizeof(double)*M_COMPLEX));
            gpu_error_check(cudaMalloc((void**)&d_stress_sum_out[i],  sizeof(double)*1));
            gpu_error_check(cudaMalloc((void**)&d_q_multi[i],         sizeof(double)*M_COMPLEX));
        }

        // Allocate memory for cub reduction sum
        for(int i=0; i<n_streams; i++)
        {  
            gpu_error_check(cudaSetDevice(i % N_GPUS));
            d_temp_storage[i] = nullptr;
            temp_storage_bytes[i] = 0;
            cub::DeviceReduce::Sum(d_temp_storage[i], temp_storage_bytes[i], d_stress_sum[i], d_stress_sum_out[i], M_COMPLEX, streams[i][0]);
            gpu_error_check(cudaMalloc(&d_temp_storage[i], temp_storage_bytes[i]));
        }
        update_laplacian_operator();
        gpu_error_check(cudaSetDevice(0));
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
CudaSolverPseudoContinuous::~CudaSolverPseudoContinuous()
{
    const int N_GPUS = CudaCommon::get_instance().get_n_gpus();

    for(int i=0; i<n_streams; i++)
    {
        cufftDestroy(plan_for_one[i]);
        cufftDestroy(plan_for_two[i]);
        cufftDestroy(plan_bak_one[i]);
        cufftDestroy(plan_bak_two[i]);
    }

    for(int gpu=0; gpu<N_GPUS; gpu++)
    {
        for(const auto& item: d_boltz_bond[gpu])
            cudaFree(item.second);
        for(const auto& item: d_boltz_bond_half[gpu])
            cudaFree(item.second);
        for(const auto& item: this->d_exp_dw[gpu])
            cudaFree(item.second);
        for(const auto& item: this->d_exp_dw_half[gpu])
            cudaFree(item.second);
    }

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
    for(int gpu=0; gpu<N_GPUS; gpu++)
    {
        cudaFree(d_fourier_basis_x[gpu]);
        cudaFree(d_fourier_basis_y[gpu]);
        cudaFree(d_fourier_basis_z[gpu]);
    }
    for(int i=0; i<n_streams; i++)
    {
        cudaFree(d_stress_sum[i]);
        cudaFree(d_stress_sum_out[i]);
        cudaFree(d_q_multi[i]);
        cudaFree(d_temp_storage[i]);
    }
}
void CudaSolverPseudoContinuous::update_laplacian_operator()
{
    try{
        // For pseudo-spectral: advance_propagator()
        const int M_COMPLEX = Pseudo::get_total_complex_grid<double>(this->cb->get_nx());
        const int N_GPUS = CudaCommon::get_instance().get_n_gpus();
        double boltz_bond[M_COMPLEX], boltz_bond_half[M_COMPLEX];

        for(const auto& item: this->molecules->get_bond_lengths())
        {
            std::string monomer_type = item.first;
            double bond_length_sq = item.second*item.second;
            
            Pseudo::get_boltz_bond<double>(this->cb->get_boundary_conditions(), boltz_bond     , bond_length_sq,   this->cb->get_nx(), this->cb->get_dx(), this->molecules->get_ds());
            Pseudo::get_boltz_bond<double>(this->cb->get_boundary_conditions(), boltz_bond_half, bond_length_sq/2, this->cb->get_nx(), this->cb->get_dx(), this->molecules->get_ds());
        
            for(int gpu=0; gpu<N_GPUS; gpu++)
            {
                gpu_error_check(cudaSetDevice(gpu));
                gpu_error_check(cudaMemcpy(d_boltz_bond     [gpu][monomer_type], boltz_bond,      sizeof(double)*M_COMPLEX,cudaMemcpyHostToDevice));
                gpu_error_check(cudaMemcpy(d_boltz_bond_half[gpu][monomer_type], boltz_bond_half, sizeof(double)*M_COMPLEX,cudaMemcpyHostToDevice));
            }
        }

        // For stress calculation: compute_stress()
        double fourier_basis_x[M_COMPLEX];
        double fourier_basis_y[M_COMPLEX];
        double fourier_basis_z[M_COMPLEX];
        Pseudo::get_weighted_fourier_basis<double>(this->cb->get_boundary_conditions(), fourier_basis_x, fourier_basis_y, fourier_basis_z, this->cb->get_nx(), this->cb->get_dx());
        for(int gpu=0; gpu<N_GPUS; gpu++)
        {
            gpu_error_check(cudaSetDevice(gpu));
            gpu_error_check(cudaMemcpy(d_fourier_basis_x[gpu], fourier_basis_x, sizeof(double)*M_COMPLEX,cudaMemcpyHostToDevice));
            gpu_error_check(cudaMemcpy(d_fourier_basis_y[gpu], fourier_basis_y, sizeof(double)*M_COMPLEX,cudaMemcpyHostToDevice));
            gpu_error_check(cudaMemcpy(d_fourier_basis_z[gpu], fourier_basis_z, sizeof(double)*M_COMPLEX,cudaMemcpyHostToDevice));
        }
        gpu_error_check(cudaSetDevice(0));
    }
    catch(std::exception& exc)
    {
        throw_with_line_number(exc.what());
    }
}
void CudaSolverPseudoContinuous::update_dw(std::string device, std::map<std::string, const double*> w_input)
{
    try{
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();
        const int N_GPUS = CudaCommon::get_instance().get_n_gpus();

        const int M = this->cb->get_total_grid();
        const double ds = this->molecules->get_ds();

        for(const auto& item: w_input)
        {
            if( this->d_exp_dw[0].find(item.first) == this->d_exp_dw[0].end())
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
            const double *w = item.second;

            // Copy field configurations from host to device
            for(int gpu=0; gpu<N_GPUS; gpu++)
            {
                gpu_error_check(cudaSetDevice(gpu));
                gpu_error_check(cudaMemcpyAsync(
                    this->d_exp_dw     [gpu][monomer_type], w,      
                    sizeof(double)*M, cudaMemcpyInputToDevice, streams[gpu][1]));
                gpu_error_check(cudaMemcpyAsync(
                    this->d_exp_dw_half[gpu][monomer_type], w,
                    sizeof(double)*M, cudaMemcpyInputToDevice, streams[gpu][1]));
            }

            // Compute d_exp_dw and d_exp_dw_half
            for(int gpu=0; gpu<N_GPUS; gpu++)
            {
                gpu_error_check(cudaSetDevice(gpu));
                ker_exp<double><<<N_BLOCKS, N_THREADS, 0, streams[gpu][1]>>>
                    (this->d_exp_dw[gpu][monomer_type],      this->d_exp_dw[gpu][monomer_type],      1.0, -0.50*ds, M);
                ker_exp<double><<<N_BLOCKS, N_THREADS, 0, streams[gpu][1]>>>
                    (this->d_exp_dw_half[gpu][monomer_type], this->d_exp_dw_half[gpu][monomer_type], 1.0, -0.25*ds, M);
            }

            // Synchronize all GPUs
            for(int gpu=0; gpu<N_GPUS; gpu++)
            {
                gpu_error_check(cudaSetDevice(gpu));
                gpu_error_check(cudaDeviceSynchronize());
            }
        }
        gpu_error_check(cudaSetDevice(0));
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
// Advance propagator using Richardson extrapolation
void CudaSolverPseudoContinuous::advance_propagator(
        const int GPU, const int STREAM,
        double *d_q_in, double *d_q_out,
        std::string monomer_type, double *d_q_mask)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = this->cb->get_total_grid();
        const int M_COMPLEX = Pseudo::get_total_complex_grid<double>(this->cb->get_nx());

        double *_d_exp_dw = this->d_exp_dw[GPU][monomer_type];
        double *_d_exp_dw_half = this->d_exp_dw_half[GPU][monomer_type];
        double *_d_boltz_bond = d_boltz_bond[GPU][monomer_type];
        double *_d_boltz_bond_half = d_boltz_bond_half[GPU][monomer_type];

        // step 1/2: Evaluate exp(-w*ds/2) in real space
        // step 1/4: Evaluate exp(-w*ds/4) in real space
        ker_multi_exp_dw_two<double><<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
            &d_q_step_1_two[STREAM][0], d_q_in, _d_exp_dw,
            &d_q_step_1_two[STREAM][M], d_q_in, _d_exp_dw_half, 1.0, M);

        // step 1/2: Execute a forward FFT
        // step 1/4: Execute a forward FFT
        cufftExecD2Z(plan_for_two[STREAM], d_q_step_1_two[STREAM], d_qk_in_1_two[STREAM]);

        // step 1/2: Multiply exp(-k^2 ds/6)  in fourier space
        // step 1/4: Multiply exp(-k^2 ds/12) in fourier space
        ker_complex_real_multi_bond_two<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
            &d_qk_in_1_two[STREAM][0],         _d_boltz_bond,
            &d_qk_in_1_two[STREAM][M_COMPLEX], _d_boltz_bond_half, M_COMPLEX);

        // step 1/2: Execute a backward FFT
        // step 1/4: Execute a backward FFT
        cufftExecZ2D(plan_bak_two[STREAM], d_qk_in_1_two[STREAM], d_q_step_1_two[STREAM]);

        // step 1/2: Evaluate exp(-w*ds/2) in real space
        // step 1/4: Evaluate exp(-w*ds/2) in real space
        ker_multi_exp_dw_two<double><<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
            d_q_step_1_one[STREAM], &d_q_step_1_two[STREAM][0], _d_exp_dw,
            d_q_step_2_one[STREAM], &d_q_step_1_two[STREAM][M], _d_exp_dw, 1.0/((double)M), M);

        // step 1/4: Execute a forward FFT
        cufftExecD2Z(plan_for_one[STREAM], d_q_step_2_one[STREAM], d_qk_in_2_one[STREAM]);

        // step 1/4: Multiply exp(-k^2 ds/12) in fourier space
        ker_multi_complex_real<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_qk_in_2_one[STREAM], _d_boltz_bond_half, 1.0, M_COMPLEX);

        // step 1/4: Execute a backward FFT
        cufftExecZ2D(plan_bak_one[STREAM], d_qk_in_2_one[STREAM], d_q_step_2_one[STREAM]);

        // step 1/4: Evaluate exp(-w*ds/4) in real space.
        ker_multi<double><<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_q_step_2_one[STREAM], d_q_step_2_one[STREAM], _d_exp_dw_half, 1.0/((double)M), M);

        // Compute linear combination with 4/3 and -1/3 ratio
        ker_lin_comb<double><<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_q_out, 4.0/3.0, d_q_step_2_one[STREAM], -1.0/3.0, d_q_step_1_one[STREAM], M);

        // Multiply mask
        if (d_q_mask != nullptr)
            ker_multi<double><<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_q_out, d_q_out, d_q_mask, 1.0, M);
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaSolverPseudoContinuous::compute_single_segment_stress(
    const int GPU, const int STREAM,
    double *d_q_pair, double *d_segment_stress,
    std::string monomer_type, bool is_half_bond_length)
{
    try{
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();
        const int N_GPUS    = CudaCommon::get_instance().get_n_gpus();

        const int DIM = this->cb->get_dim();
        const int M   = this->cb->get_total_grid();
        const int M_COMPLEX = Pseudo::get_total_complex_grid<double>(this->cb->get_nx());

        auto bond_lengths = this->molecules->get_bond_lengths();
        std::vector<double> stress(DIM);
        double bond_length_sq = bond_lengths[monomer_type]*bond_lengths[monomer_type];

        // Execute a forward FFT
        cufftExecD2Z(plan_for_two[STREAM], d_q_pair, d_qk_in_1_two[STREAM]);

        // Multiply two propagators in the fourier spaces
        ker_multi_complex_conjugate<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_q_multi[STREAM], &d_qk_in_1_two[STREAM][0], &d_qk_in_1_two[STREAM][M_COMPLEX], M_COMPLEX);
        if ( DIM == 3 )
        {
            // x direction
            ker_multi<double><<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_stress_sum[STREAM], d_q_multi[STREAM], d_fourier_basis_x[GPU], bond_length_sq, M_COMPLEX);
            cub::DeviceReduce::Sum(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[0], M_COMPLEX, streams[STREAM][0]);

            // y direction
            ker_multi<double><<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_stress_sum[STREAM], d_q_multi[STREAM], d_fourier_basis_y[GPU], bond_length_sq, M_COMPLEX);
            cub::DeviceReduce::Sum(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[1], M_COMPLEX, streams[STREAM][0]);

            // z direction
            ker_multi<double><<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_stress_sum[STREAM], d_q_multi[STREAM], d_fourier_basis_z[GPU], bond_length_sq, M_COMPLEX);
            cub::DeviceReduce::Sum(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[2], M_COMPLEX, streams[STREAM][0]);
        }
        if ( DIM == 2 )
        {
            // y direction
            ker_multi<double><<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_stress_sum[STREAM], d_q_multi[STREAM], d_fourier_basis_y[GPU], bond_length_sq, M_COMPLEX);
            cub::DeviceReduce::Sum(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[0], M_COMPLEX, streams[STREAM][0]);

            // z direction
            ker_multi<double><<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_stress_sum[STREAM], d_q_multi[STREAM], d_fourier_basis_z[GPU], bond_length_sq, M_COMPLEX);
            cub::DeviceReduce::Sum(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[1], M_COMPLEX, streams[STREAM][0]);
        }
        if ( DIM == 1 )
        {
            // z direction
            ker_multi<double><<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_stress_sum[STREAM], d_q_multi[STREAM], d_fourier_basis_z[GPU], bond_length_sq, M_COMPLEX);
            cub::DeviceReduce::Sum(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[0], M_COMPLEX, streams[STREAM][0]);
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}