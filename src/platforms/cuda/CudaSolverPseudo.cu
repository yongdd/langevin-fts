#include <iostream>
#include <cmath>
#include <thrust/reduce.h>
#include "CudaSolverPseudo.h"

CudaSolverPseudo::CudaSolverPseudo(ComputationBox *cb, Molecules *molecules, cudaStream_t streams[MAX_GPUS][2], bool reduce_gpu_memory_usage)
{
    try{
        this->cb = cb;
        this->molecules = molecules;
        this->chain_model = molecules->get_model_name();
        this->reduce_gpu_memory_usage = reduce_gpu_memory_usage;

        const int M = cb->get_n_grid();
        const int M_COMPLEX = Pseudo::get_n_complex_grid(cb->get_nx());
        const int N_GPUS = CudaCommon::get_instance().get_n_gpus();

        // Copy streams
        for(int gpu=0; gpu<N_GPUS; gpu++)
        {
            this->streams[gpu][0] = streams[gpu][0];
            this->streams[gpu][1] = streams[gpu][1];
        }

        // Create boltz_bond, boltz_bond_half, exp_dw, and exp_dw_half
        for(const auto& item: molecules->get_bond_lengths())
        {
            std::string monomer_type = item.first;
            for(int gpu=0; gpu<N_GPUS; gpu++)
            {
                gpu_error_check(cudaSetDevice(gpu));
                d_exp_dw         [gpu][monomer_type] = nullptr;
                d_boltz_bond     [gpu][monomer_type] = nullptr;
                d_boltz_bond_half[gpu][monomer_type] = nullptr;

                gpu_error_check(cudaMalloc((void**)&d_exp_dw         [gpu][monomer_type], sizeof(double)*M));
                gpu_error_check(cudaMalloc((void**)&d_boltz_bond     [gpu][monomer_type], sizeof(double)*M_COMPLEX));
                gpu_error_check(cudaMalloc((void**)&d_boltz_bond_half[gpu][monomer_type], sizeof(double)*M_COMPLEX));

                if(chain_model == "continuous")
                {
                    d_exp_dw_half    [gpu][monomer_type] = nullptr;
                    gpu_error_check(cudaMalloc((void**)&d_exp_dw_half    [gpu][monomer_type], sizeof(double)*M));
                }
            }
        }

        // Create FFT plan
        const int NRANK{cb->get_dim()};
        int n_grid[NRANK];

        if(cb->get_dim() == 3)
        {
            n_grid[0] = cb->get_nx(0);
            n_grid[1] = cb->get_nx(1);
            n_grid[2] = cb->get_nx(2);
        }
        else if(cb->get_dim() == 2)
        {
            n_grid[0] = cb->get_nx(0);
            n_grid[1] = cb->get_nx(1);
        }
        else if(cb->get_dim() == 1)
        {
            n_grid[0] = cb->get_nx(0);
        }

        for(int gpu=0; gpu<N_GPUS; gpu++)
        {
            gpu_error_check(cudaSetDevice(gpu));
            cufftPlanMany(&plan_for_one[gpu], NRANK, n_grid, NULL, 1, 0, NULL, 1, 0, CUFFT_D2Z,1);
            cufftPlanMany(&plan_for_two[gpu], NRANK, n_grid, NULL, 1, 0, NULL, 1, 0, CUFFT_D2Z,2);
            cufftPlanMany(&plan_bak_one[gpu], NRANK, n_grid, NULL, 1, 0, NULL, 1, 0, CUFFT_Z2D,1);
            cufftPlanMany(&plan_bak_two[gpu], NRANK, n_grid, NULL, 1, 0, NULL, 1, 0, CUFFT_Z2D,2);
            cufftSetStream(plan_for_one[gpu], streams[gpu][0]);
            cufftSetStream(plan_for_two[gpu], streams[gpu][0]);
            cufftSetStream(plan_bak_one[gpu], streams[gpu][0]);
            cufftSetStream(plan_bak_two[gpu], streams[gpu][0]);
        }
        gpu_error_check(cudaSetDevice(0));
        cufftPlanMany(&plan_for_four, NRANK, n_grid, NULL, 1, 0, NULL, 1, 0, CUFFT_D2Z,4);
        cufftPlanMany(&plan_bak_four, NRANK, n_grid, NULL, 1, 0, NULL, 1, 0, CUFFT_Z2D,4);
        cufftSetStream(plan_for_four, streams[0][0]);
        cufftSetStream(plan_bak_four, streams[0][0]);

        // Allocate memory for pseudo-spectral: advance_propagator()
        if(chain_model == "continuous" && !reduce_gpu_memory_usage)
        {
            for(int gpu=0; gpu<N_GPUS; gpu++)
            {
                gpu_error_check(cudaSetDevice(gpu));
                gpu_error_check(cudaMalloc((void**)&d_q_step_1_one[gpu], sizeof(double)*M));
                gpu_error_check(cudaMalloc((void**)&d_q_step_2_one[gpu], sizeof(double)*M));
                gpu_error_check(cudaMalloc((void**)&d_q_step_1_two[gpu], sizeof(double)*2*M));
                gpu_error_check(cudaMalloc((void**)&d_q_step_2_two[gpu], sizeof(double)*2*M));

                gpu_error_check(cudaMalloc((void**)&d_qk_in_2_one[gpu], sizeof(ftsComplex)*M_COMPLEX));
                gpu_error_check(cudaMalloc((void**)&d_qk_in_1_two[gpu], sizeof(ftsComplex)*2*M_COMPLEX));
                gpu_error_check(cudaMalloc((void**)&d_qk_in_2_two[gpu], sizeof(ftsComplex)*2*M_COMPLEX));
            }

            gpu_error_check(cudaSetDevice(0));
            gpu_error_check(cudaMalloc((void**)&d_q_step_1_four, sizeof(double)*4*M));
            gpu_error_check(cudaMalloc((void**)&d_qk_in_1_four,  sizeof(ftsComplex)*4*M_COMPLEX));
        }
        else if(chain_model == "continuous" && reduce_gpu_memory_usage)
        {
            for(int gpu=0; gpu<N_GPUS; gpu++)
            {
                gpu_error_check(cudaSetDevice(gpu));
                gpu_error_check(cudaMalloc((void**)&d_q_step_1_one[gpu], sizeof(double)*M));
                gpu_error_check(cudaMalloc((void**)&d_q_step_2_one[gpu], sizeof(double)*M));
                gpu_error_check(cudaMalloc((void**)&d_q_step_1_two[gpu], sizeof(double)*2*M));
                gpu_error_check(cudaMalloc((void**)&d_qk_in_2_one[gpu], sizeof(ftsComplex)*M_COMPLEX));
                gpu_error_check(cudaMalloc((void**)&d_qk_in_1_two[gpu], sizeof(ftsComplex)*2*M_COMPLEX));
            }
        }
        else if(chain_model == "discrete")
        {
            for(int gpu=0; gpu<N_GPUS; gpu++)
            {
                gpu_error_check(cudaMalloc((void**)&d_q_step_1_two[gpu], sizeof(double)*2*M));
                gpu_error_check(cudaMalloc((void**)&d_qk_in_1_one[gpu], sizeof(ftsComplex)*M_COMPLEX));
                gpu_error_check(cudaMalloc((void**)&d_qk_in_1_two[gpu], sizeof(ftsComplex)*2*M_COMPLEX));
            }
        }

        // Allocate memory for stress calculation: compute_stress()
        for(int gpu=0; gpu<N_GPUS; gpu++)
        {
            gpu_error_check(cudaSetDevice(gpu));
            gpu_error_check(cudaMalloc((void**)&d_fourier_basis_x[gpu], sizeof(double)*M_COMPLEX));
            gpu_error_check(cudaMalloc((void**)&d_fourier_basis_y[gpu], sizeof(double)*M_COMPLEX));
            gpu_error_check(cudaMalloc((void**)&d_fourier_basis_z[gpu], sizeof(double)*M_COMPLEX));
            gpu_error_check(cudaMalloc((void**)&d_stress_sum[gpu],      sizeof(double)*M_COMPLEX));
            gpu_error_check(cudaMalloc((void**)&d_stress_sum_out[gpu],  sizeof(double)*1));
            gpu_error_check(cudaMalloc((void**)&d_q_multi[gpu],         sizeof(double)*M_COMPLEX));
        }

        // Allocate memory for cub reduction sum
        for(int gpu=0; gpu<N_GPUS; gpu++)
        {
            gpu_error_check(cudaSetDevice(gpu));
            d_temp_storage[gpu] = nullptr;
            temp_storage_bytes[gpu] = 0;
            cub::DeviceReduce::Sum(d_temp_storage[gpu], temp_storage_bytes[gpu], d_stress_sum[gpu], d_stress_sum_out[gpu], M_COMPLEX, streams[gpu][0]);
            gpu_error_check(cudaMalloc(&d_temp_storage[gpu], temp_storage_bytes[gpu]));
        }
        update_laplacian_operator();
        gpu_error_check(cudaSetDevice(0));
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
CudaSolverPseudo::~CudaSolverPseudo()
{
    const int N_GPUS = CudaCommon::get_instance().get_n_gpus();

    cufftDestroy(plan_for_four);
    cufftDestroy(plan_bak_four);

    for(int gpu=0; gpu<N_GPUS; gpu++)
    {
        for(const auto& item: d_boltz_bond[gpu])
            cudaFree(item.second);
        for(const auto& item: d_boltz_bond_half[gpu])
            cudaFree(item.second);
        for(const auto& item: d_exp_dw[gpu])
            cudaFree(item.second);
        if(chain_model == "continuous")
        {
            for(const auto& item: d_exp_dw_half[gpu])
                cudaFree(item.second);
        }
    }

    // For pseudo-spectral: advance_propagator()
    if(chain_model == "continuous" && !reduce_gpu_memory_usage)
    {
        for(int gpu=0; gpu<N_GPUS; gpu++)
        {
            cudaFree(d_q_step_1_one[gpu]);
            cudaFree(d_q_step_2_one[gpu]);
            cudaFree(d_q_step_1_two[gpu]);
            cudaFree(d_q_step_2_two[gpu]);
            cudaFree(d_qk_in_2_one[gpu]);
            cudaFree(d_qk_in_1_two[gpu]);
            cudaFree(d_qk_in_2_two[gpu]);
        }
        cudaFree(d_qk_in_1_four);
    }
    if(chain_model == "continuous" && reduce_gpu_memory_usage)
    {
        for(int gpu=0; gpu<N_GPUS; gpu++)
        {
            cudaFree(d_q_step_1_one[gpu]);
            cudaFree(d_q_step_2_one[gpu]);
            cudaFree(d_q_step_1_two[gpu]);
            cudaFree(d_qk_in_2_one[gpu]);
            cudaFree(d_qk_in_1_two[gpu]);
        }
    }
    else if(chain_model == "discrete")
    {
        for(int gpu=0; gpu<N_GPUS; gpu++)
        {
            cudaFree(d_q_step_1_two[gpu]);
            cudaFree(d_qk_in_1_one[gpu]);
            cudaFree(d_qk_in_1_two[gpu]);
        }
    }

    // For stress calculation: compute_stress()
    for(int gpu=0; gpu<N_GPUS; gpu++)
    {
        cudaFree(d_fourier_basis_x[gpu]);
        cudaFree(d_fourier_basis_y[gpu]);
        cudaFree(d_fourier_basis_z[gpu]);
        cudaFree(d_stress_sum[gpu]);
        cudaFree(d_stress_sum_out[gpu]);
        cudaFree(d_q_multi[gpu]);
        cudaFree(d_temp_storage[gpu]);
    }

    // Destroy streams
    for(int gpu=0; gpu<N_GPUS; gpu++)
    {
        cudaStreamDestroy(streams[gpu][0]);
        cudaStreamDestroy(streams[gpu][1]);
    }
}
void CudaSolverPseudo::update_laplacian_operator()
{
    try{
        // For pseudo-spectral: advance_propagator()
        const int M_COMPLEX = Pseudo::get_n_complex_grid(cb->get_nx());
        const int N_GPUS = CudaCommon::get_instance().get_n_gpus();
        double boltz_bond[M_COMPLEX], boltz_bond_half[M_COMPLEX];

        for(const auto& item: molecules->get_bond_lengths())
        {
            std::string monomer_type = item.first;
            double bond_length_sq = item.second*item.second;
            
            Pseudo::get_boltz_bond(boltz_bond     , bond_length_sq,   cb->get_nx(), cb->get_dx(), molecules->get_ds());
            Pseudo::get_boltz_bond(boltz_bond_half, bond_length_sq/2, cb->get_nx(), cb->get_dx(), molecules->get_ds());
        
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
        Pseudo::get_weighted_fourier_basis(fourier_basis_x, fourier_basis_y, fourier_basis_z, cb->get_nx(), cb->get_dx());
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
void CudaSolverPseudo::update_dw(std::string device, std::map<std::string, const double*> w_input)
{
    try{
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();
        const int N_GPUS = CudaCommon::get_instance().get_n_gpus();

        const int M = cb->get_n_grid();
        const double ds = molecules->get_ds();

        for(const auto& item: w_input)
        {
            if( d_exp_dw[0].find(item.first) == d_exp_dw[0].end())
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

            if(chain_model == "continuous")
            {
                // Copy field configurations from host to device
                for(int gpu=0; gpu<N_GPUS; gpu++)
                {
                    gpu_error_check(cudaSetDevice(gpu));
                    gpu_error_check(cudaMemcpyAsync(
                        d_exp_dw     [gpu][monomer_type], w,      
                        sizeof(double)*M, cudaMemcpyInputToDevice, streams[gpu][1]));
                    gpu_error_check(cudaMemcpyAsync(
                        d_exp_dw_half[gpu][monomer_type], w,
                        sizeof(double)*M, cudaMemcpyInputToDevice, streams[gpu][1]));
                }

                // Compute exp_dw and exp_dw_half
                for(int gpu=0; gpu<N_GPUS; gpu++)
                {
                    gpu_error_check(cudaSetDevice(gpu));
                    exp_real<<<N_BLOCKS, N_THREADS, 0, streams[gpu][1]>>>
                        (d_exp_dw[gpu][monomer_type],      d_exp_dw[gpu][monomer_type],      1.0, -0.50*ds, M);
                    exp_real<<<N_BLOCKS, N_THREADS, 0, streams[gpu][1]>>>
                        (d_exp_dw_half[gpu][monomer_type], d_exp_dw_half[gpu][monomer_type], 1.0, -0.25*ds, M);
                }
            }
            else if(chain_model == "discrete")
            {
                // Copy field configurations from host to device
                for(int gpu=0; gpu<N_GPUS; gpu++)
                {
                    gpu_error_check(cudaSetDevice(gpu));
                    gpu_error_check(cudaMemcpyAsync(
                        d_exp_dw[gpu][monomer_type], w,      
                        sizeof(double)*M, cudaMemcpyInputToDevice, streams[gpu][1]));
                }

                // Compute exp_dw and exp_dw_half
                for(int gpu=0; gpu<N_GPUS; gpu++)
                {
                    gpu_error_check(cudaSetDevice(gpu));
                    exp_real<<<N_BLOCKS, N_THREADS, 0, streams[gpu][1]>>>
                        (d_exp_dw[gpu][monomer_type], d_exp_dw[gpu][monomer_type], 1.0, -1.0*ds, M);
                }
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
void CudaSolverPseudo::advance_one_propagator_continuous(
        const int GPU, double *d_q_in, double *d_q_out,
        std::string monomer_type, double *d_q_mask)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = cb->get_n_grid();
        const int M_COMPLEX = Pseudo::get_n_complex_grid(cb->get_nx());

        double *_d_exp_dw = d_exp_dw[GPU][monomer_type];
        double *_d_exp_dw_half = d_exp_dw_half[GPU][monomer_type];
        double *_d_boltz_bond = d_boltz_bond[GPU][monomer_type];
        double *_d_boltz_bond_half = d_boltz_bond_half[GPU][monomer_type];

        // step 1/2: Evaluate exp(-w*ds/2) in real space
        // step 1/4: Evaluate exp(-w*ds/4) in real space
        real_multi_exp_dw_two<<<N_BLOCKS, N_THREADS, 0, streams[GPU][0]>>>(
            &d_q_step_1_two[GPU][0], d_q_in, _d_exp_dw,
            &d_q_step_1_two[GPU][M], d_q_in, _d_exp_dw_half, 1.0, M);

        // step 1/2: Execute a forward FFT
        // step 1/4: Execute a forward FFT
        cufftExecD2Z(plan_for_two[GPU], d_q_step_1_two[GPU], d_qk_in_1_two[GPU]);

        // step 1/2: Multiply exp(-k^2 ds/6)  in fourier space
        // step 1/4: Multiply exp(-k^2 ds/12) in fourier space
        complex_real_multi_bond_two<<<N_BLOCKS, N_THREADS, 0, streams[GPU][0]>>>(
            &d_qk_in_1_two[GPU][0],         _d_boltz_bond,
            &d_qk_in_1_two[GPU][M_COMPLEX], _d_boltz_bond_half, M_COMPLEX);

        // step 1/2: Execute a backward FFT
        // step 1/4: Execute a backward FFT
        cufftExecZ2D(plan_bak_two[GPU], d_qk_in_1_two[GPU], d_q_step_1_two[GPU]);

        // step 1/2: Evaluate exp(-w*ds/2) in real space
        // step 1/4: Evaluate exp(-w*ds/2) in real space
        real_multi_exp_dw_two<<<N_BLOCKS, N_THREADS, 0, streams[GPU][0]>>>(
            d_q_step_1_one[GPU], &d_q_step_1_two[GPU][0], _d_exp_dw,
            d_q_step_2_one[GPU], &d_q_step_1_two[GPU][M], _d_exp_dw, 1.0/((double)M), M);

        // step 1/4: Execute a forward FFT
        cufftExecD2Z(plan_for_one[GPU], d_q_step_2_one[GPU], d_qk_in_2_one[GPU]);

        // step 1/4: Multiply exp(-k^2 ds/12) in fourier space
        multi_complex_real<<<N_BLOCKS, N_THREADS, 0, streams[GPU][0]>>>(d_qk_in_2_one[GPU], _d_boltz_bond_half, M_COMPLEX);

        // step 1/4: Execute a backward FFT
        cufftExecZ2D(plan_bak_one[GPU], d_qk_in_2_one[GPU], d_q_step_2_one[GPU]);

        // step 1/4: Evaluate exp(-w*ds/4) in real space.
        multi_real<<<N_BLOCKS, N_THREADS, 0, streams[GPU][0]>>>(d_q_step_2_one[GPU], d_q_step_2_one[GPU], _d_exp_dw_half, 1.0/((double)M), M);

        // Compute linear combination with 4/3 and -1/3 ratio
        lin_comb<<<N_BLOCKS, N_THREADS, 0, streams[GPU][0]>>>(d_q_out, 4.0/3.0, d_q_step_2_one[GPU], -1.0/3.0, d_q_step_1_one[GPU], M);

        // Multiply mask
        if (d_q_mask != nullptr)
            multi_real<<<N_BLOCKS, N_THREADS, 0, streams[GPU][0]>>>(d_q_out, d_q_out, d_q_mask, 1.0, M);
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaSolverPseudo::advance_two_propagators_continuous(
            double *d_q_in_1,  double *d_q_in_2,
            double *d_q_out_1, double *d_q_out_2,
            std::string monomer_type_1, std::string monomer_type_2,
            double *d_q_mask)
{
    // Overlapping computations for 1/2 step and 1/4 step using 4-batch cuFFT
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = cb->get_n_grid();
        const int M_COMPLEX = Pseudo::get_n_complex_grid(cb->get_nx());

        double *_d_exp_dw_1 = d_exp_dw[0][monomer_type_1];
        double *_d_exp_dw_half_1 = d_exp_dw_half[0][monomer_type_1];
        double *_d_boltz_bond_1 = d_boltz_bond[0][monomer_type_1];
        double *_d_boltz_bond_half_1 = d_boltz_bond_half[0][monomer_type_1];

        double *_d_exp_dw_2 = d_exp_dw[0][monomer_type_2];
        double *_d_exp_dw_half_2 = d_exp_dw_half[0][monomer_type_2];
        double *_d_boltz_bond_2 = d_boltz_bond[0][monomer_type_2];
        double *_d_boltz_bond_half_2 = d_boltz_bond_half[0][monomer_type_2];

        // step 1/2: Evaluate exp(-w*ds/2) in real space
        // step 1/4: Evaluate exp(-w*ds/4) in real space
        real_multi_exp_dw_four<<<N_BLOCKS, N_THREADS, 0, streams[0][0]>>>(
            &d_q_step_1_four[0],   d_q_in_1, _d_exp_dw_1,
            &d_q_step_1_four[M],   d_q_in_2, _d_exp_dw_2,
            &d_q_step_1_four[2*M], d_q_in_1, _d_exp_dw_half_1,
            &d_q_step_1_four[3*M], d_q_in_2, _d_exp_dw_half_2, 1.0, M);

        // step 1/2: Execute a forward FFT
        // step 1/4: Execute a forward FFT
        cufftExecD2Z(plan_for_four, d_q_step_1_four, d_qk_in_1_four);

        // step 1/2: Multiply exp(-k^2 ds/6)  in fourier space
        // step 1/4: Multiply exp(-k^2 ds/12) in fourier space
        complex_real_multi_bond_four<<<N_BLOCKS, N_THREADS, 0, streams[0][0]>>>(
            &d_qk_in_1_four[0],           _d_boltz_bond_1,
            &d_qk_in_1_four[M_COMPLEX],   _d_boltz_bond_2,
            &d_qk_in_1_four[2*M_COMPLEX], _d_boltz_bond_half_1,
            &d_qk_in_1_four[3*M_COMPLEX], _d_boltz_bond_half_2, M_COMPLEX);

        // step 1/2: Execute a backward FFT
        // step 1/4: Execute a backward FFT
        cufftExecZ2D(plan_bak_four, d_qk_in_1_four, d_q_step_1_four);

        // step 1/2: Evaluate exp(-w*ds/2) in real space
        // step 1/4: Evaluate exp(-w*ds/2) in real space
        real_multi_exp_dw_four<<<N_BLOCKS, N_THREADS, 0, streams[0][0]>>>(
            &d_q_step_1_two[0][0], &d_q_step_1_four[0],   _d_exp_dw_1,
            &d_q_step_1_two[0][M], &d_q_step_1_four[M],   _d_exp_dw_2,
            &d_q_step_2_two[0][0], &d_q_step_1_four[2*M], _d_exp_dw_1,
            &d_q_step_2_two[0][M], &d_q_step_1_four[3*M], _d_exp_dw_2, 1.0/((double)M), M);

        // step 1/4: Execute a forward FFT
        cufftExecD2Z(plan_for_two[0], d_q_step_2_two[0], d_qk_in_2_two[0]);

        // step 1/4: Multiply exp(-k^2 ds/12) in fourier space
        complex_real_multi_bond_two<<<N_BLOCKS, N_THREADS, 0, streams[0][0]>>>(
            &d_qk_in_2_two[0][0],         _d_boltz_bond_half_1, 
            &d_qk_in_2_two[0][M_COMPLEX], _d_boltz_bond_half_2, M_COMPLEX);

        // step 1/4: Execute a backward FFT
        cufftExecZ2D(plan_bak_two[0], d_qk_in_2_two[0], d_q_step_2_two[0]);

        // step 1/4: Evaluate exp(-w*ds/4) in real space
        real_multi_exp_dw_two<<<N_BLOCKS, N_THREADS, 0, streams[0][0]>>>(
            &d_q_step_2_two[0][0], &d_q_step_2_two[0][0], _d_exp_dw_half_1,
            &d_q_step_2_two[0][M], &d_q_step_2_two[0][M], _d_exp_dw_half_2, 1.0/((double)M), M);

        // Compute linear combination with 4/3 and -1/3 ratio
        lin_comb<<<N_BLOCKS, N_THREADS, 0, streams[0][0]>>>(d_q_out_1, 4.0/3.0, &d_q_step_2_two[0][0], -1.0/3.0, &d_q_step_1_two[0][0], M);
        lin_comb<<<N_BLOCKS, N_THREADS, 0, streams[0][0]>>>(d_q_out_2, 4.0/3.0, &d_q_step_2_two[0][M], -1.0/3.0, &d_q_step_1_two[0][M], M);

        // Multiply mask
        if (d_q_mask != nullptr)
        {
            multi_real<<<N_BLOCKS, N_THREADS, 0, streams[0][0]>>>(d_q_out_1, d_q_out_1, d_q_mask, 1.0, M);
            multi_real<<<N_BLOCKS, N_THREADS, 0, streams[0][0]>>>(d_q_out_2, d_q_out_2, d_q_mask, 1.0, M);
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaSolverPseudo::advance_two_propagators_continuous_two_gpus(
            double *d_q_in_1,  double *d_q_in_2,
            double *d_q_out_1, double *d_q_out_2,
            std::string monomer_type_1, std::string monomer_type_2,
            double **d_q_mask)
{
    // Overlapping computations for 1/2 step and 1/4 step
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();
        const int N_GPUS = CudaCommon::get_instance().get_n_gpus();

        const int M = cb->get_n_grid();
        const int M_COMPLEX = Pseudo::get_n_complex_grid(cb->get_nx());

        double *_d_exp_dw_1 = d_exp_dw[0][monomer_type_1];
        double *_d_exp_dw_half_1 = d_exp_dw_half[0][monomer_type_1];
        double *_d_boltz_bond_1 = d_boltz_bond[0][monomer_type_1];
        double *_d_boltz_bond_half_1 = d_boltz_bond_half[0][monomer_type_1];

        double *_d_exp_dw_2 = d_exp_dw[1][monomer_type_2];
        double *_d_exp_dw_half_2 = d_exp_dw_half[1][monomer_type_2];
        double *_d_boltz_bond_2 = d_boltz_bond[1][monomer_type_2];
        double *_d_boltz_bond_half_2 = d_boltz_bond_half[1][monomer_type_2];

        // step 1/2: Evaluate exp(-w*ds/2) in real space
        // step 1/4: Evaluate exp(-w*ds/4) in real space
        gpu_error_check(cudaSetDevice(0));
        real_multi_exp_dw_two<<<N_BLOCKS, N_THREADS, 0, streams[0][0]>>>(
            &d_q_step_1_two[0][0], d_q_in_1, _d_exp_dw_1,
            &d_q_step_1_two[0][M], d_q_in_1, _d_exp_dw_half_1, 1.0, M);
        gpu_error_check(cudaSetDevice(1));
        real_multi_exp_dw_two<<<N_BLOCKS, N_THREADS, 0, streams[1][0]>>>(
            &d_q_step_1_two[1][0], d_q_in_2, _d_exp_dw_2,
            &d_q_step_1_two[1][M], d_q_in_2, _d_exp_dw_half_2, 1.0, M);

        // step 1/2: Execute a forward FFT
        // step 1/4: Execute a forward FFT
        gpu_error_check(cudaSetDevice(0));
        cufftExecD2Z(plan_for_two[0], d_q_step_1_two[0], d_qk_in_1_two[0]);
        gpu_error_check(cudaSetDevice(1));
        cufftExecD2Z(plan_for_two[1], d_q_step_1_two[1], d_qk_in_1_two[1]);

        // step 1/2: Multiply exp(-k^2 ds/6)  in fourier space
        // step 1/4: Multiply exp(-k^2 ds/12) in fourier space
        gpu_error_check(cudaSetDevice(0));
        complex_real_multi_bond_two<<<N_BLOCKS, N_THREADS, 0, streams[0][0]>>>(
            &d_qk_in_1_two[0][0],         _d_boltz_bond_1,
            &d_qk_in_1_two[0][M_COMPLEX], _d_boltz_bond_half_1, M_COMPLEX);
        gpu_error_check(cudaSetDevice(1));
        complex_real_multi_bond_two<<<N_BLOCKS, N_THREADS, 0, streams[1][0]>>>(
            &d_qk_in_1_two[1][0],         _d_boltz_bond_2,
            &d_qk_in_1_two[1][M_COMPLEX], _d_boltz_bond_half_2, M_COMPLEX);

        // step 1/2: Execute a backward FFT
        // step 1/4: Execute a backward FFT
        gpu_error_check(cudaSetDevice(0));
        cufftExecZ2D(plan_bak_two[0], d_qk_in_1_two[0], d_q_step_1_two[0]);
        gpu_error_check(cudaSetDevice(1));
        cufftExecZ2D(plan_bak_two[1], d_qk_in_1_two[1], d_q_step_1_two[1]);

        // step 1/2: Evaluate exp(-w*ds/2) in real space
        // step 1/4: Evaluate exp(-w*ds/2) in real space
        gpu_error_check(cudaSetDevice(0));
        real_multi_exp_dw_two<<<N_BLOCKS, N_THREADS, 0, streams[0][0]>>>(
            d_q_step_1_one[0], &d_q_step_1_two[0][0], _d_exp_dw_1,
            d_q_step_2_one[0], &d_q_step_1_two[0][M], _d_exp_dw_1, 1.0/((double)M), M);
        gpu_error_check(cudaSetDevice(1));
        real_multi_exp_dw_two<<<N_BLOCKS, N_THREADS, 0, streams[1][0]>>>(
            d_q_step_1_one[1], &d_q_step_1_two[1][0], _d_exp_dw_2,
            d_q_step_2_one[1], &d_q_step_1_two[1][M], _d_exp_dw_2, 1.0/((double)M), M);

        // step 1/4: Execute a forward FFT
        gpu_error_check(cudaSetDevice(0));
        cufftExecD2Z(plan_for_one[0], d_q_step_2_one[0], d_qk_in_2_one[0]);
        gpu_error_check(cudaSetDevice(1));
        cufftExecD2Z(plan_for_one[1], d_q_step_2_one[1], d_qk_in_2_one[1]);

        // step 1/4: Multiply exp(-k^2 ds/12) in fourier space
        gpu_error_check(cudaSetDevice(0));
        multi_complex_real<<<N_BLOCKS, N_THREADS, 0, streams[0][0]>>>(d_qk_in_2_one[0], _d_boltz_bond_half_1, M_COMPLEX);
        gpu_error_check(cudaSetDevice(1));
        multi_complex_real<<<N_BLOCKS, N_THREADS, 0, streams[1][0]>>>(d_qk_in_2_one[1], _d_boltz_bond_half_2, M_COMPLEX);

        // step 1/4: Execute a backward FFT
        gpu_error_check(cudaSetDevice(0));
        cufftExecZ2D(plan_bak_one[0], d_qk_in_2_one[0], d_q_step_2_one[0]);
        gpu_error_check(cudaSetDevice(1));
        cufftExecZ2D(plan_bak_one[1], d_qk_in_2_one[1], d_q_step_2_one[1]);

        // step 1/4: Evaluate exp(-w*ds/4) in real space.
        gpu_error_check(cudaSetDevice(0));
        multi_real<<<N_BLOCKS, N_THREADS, 0, streams[0][0]>>>(d_q_step_2_one[0], d_q_step_2_one[0], _d_exp_dw_half_1, 1.0/((double)M), M);
        gpu_error_check(cudaSetDevice(1));
        multi_real<<<N_BLOCKS, N_THREADS, 0, streams[1][0]>>>(d_q_step_2_one[1], d_q_step_2_one[1], _d_exp_dw_half_2, 1.0/((double)M), M);

        // Compute linear combination with 4/3 and -1/3 ratio
        gpu_error_check(cudaSetDevice(0));
        lin_comb<<<N_BLOCKS, N_THREADS, 0, streams[0][0]>>>(d_q_out_1, 4.0/3.0, d_q_step_2_one[0], -1.0/3.0, d_q_step_1_one[0], M);
        gpu_error_check(cudaSetDevice(1));
        lin_comb<<<N_BLOCKS, N_THREADS, 0, streams[1][0]>>>(d_q_out_2, 4.0/3.0, d_q_step_2_one[1], -1.0/3.0, d_q_step_1_one[1], M);

        // Multiply mask
        if (d_q_mask[0] != nullptr && d_q_mask[1] != nullptr)
        {
            gpu_error_check(cudaSetDevice(0));
            multi_real<<<N_BLOCKS, N_THREADS, 0, streams[0][0]>>>(d_q_out_1, d_q_out_1, d_q_mask[0], 1.0, M);
            gpu_error_check(cudaSetDevice(1));
            multi_real<<<N_BLOCKS, N_THREADS, 0, streams[1][0]>>>(d_q_out_2, d_q_out_2, d_q_mask[1], 1.0, M);
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaSolverPseudo::advance_one_propagator_discrete(
    const int GPU,
    double *d_q_in, double *d_q_out,
    std::string monomer_type,
    double *d_q_mask)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = cb->get_n_grid();
        const int M_COMPLEX = Pseudo::get_n_complex_grid(cb->get_nx());

        double *_d_exp_dw = d_exp_dw[GPU][monomer_type];
        double *_d_boltz_bond = d_boltz_bond[GPU][monomer_type];

        // Execute a forward FFT
        cufftExecD2Z(plan_for_one[GPU], d_q_in, d_qk_in_1_one[GPU]);

        // Multiply exp(-k^2 ds/6) in fourier space
        multi_complex_real<<<N_BLOCKS, N_THREADS, 0, streams[GPU][0]>>>(d_qk_in_1_one[GPU], _d_boltz_bond, M_COMPLEX);

        // Execute a backward FFT
        cufftExecZ2D(plan_bak_one[GPU], d_qk_in_1_one[GPU], d_q_out);

        // Evaluate exp(-w*ds) in real space
        multi_real<<<N_BLOCKS, N_THREADS, 0, streams[GPU][0]>>>(d_q_out, d_q_out, _d_exp_dw, 1.0/((double)M), M);

        // Multiply mask
        if (d_q_mask != nullptr)
            multi_real<<<N_BLOCKS, N_THREADS, 0, streams[GPU][0]>>>(d_q_out, d_q_out, d_q_mask, 1.0, M);
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaSolverPseudo::advance_two_propagators_discrete(
    double *d_q_in_1, double *d_q_in_2,
    double *d_q_out_1, double *d_q_out_2,
    std::string monomer_type_1, std::string monomer_type_2,
    double *d_q_mask)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = cb->get_n_grid();
        const int M_COMPLEX = Pseudo::get_n_complex_grid(cb->get_nx());

        double *_d_exp_dw_1 = d_exp_dw[0][monomer_type_1];
        double *_d_boltz_bond_1 = d_boltz_bond[0][monomer_type_1];

        double *_d_exp_dw_2 = d_exp_dw[0][monomer_type_2];
        double *_d_boltz_bond_2 = d_boltz_bond[0][monomer_type_2];

        gpu_error_check(cudaMemcpyAsync(&d_q_step_1_two[0][0], d_q_in_1, sizeof(double)*M, cudaMemcpyDeviceToDevice, streams[0][0]));
        gpu_error_check(cudaMemcpyAsync(&d_q_step_1_two[0][M], d_q_in_2, sizeof(double)*M, cudaMemcpyDeviceToDevice, streams[0][0]));

        // Execute a forward FFT
        cufftExecD2Z(plan_for_two[0], d_q_step_1_two[0], d_qk_in_1_two[0]);

        // Multiply exp(-k^2 ds/6) in fourier space
        complex_real_multi_bond_two<<<N_BLOCKS, N_THREADS, 0, streams[0][0]>>>(
            &d_qk_in_1_two[0][0],         _d_boltz_bond_1, 
            &d_qk_in_1_two[0][M_COMPLEX], _d_boltz_bond_2, M_COMPLEX);

        // Execute a backward FFT
        cufftExecZ2D(plan_bak_two[0], d_qk_in_1_two[0], d_q_step_1_two[0]);

        // Evaluate exp(-w*ds) in real space
        real_multi_exp_dw_two<<<N_BLOCKS, N_THREADS, 0, streams[0][0]>>>(
            d_q_out_1, &d_q_step_1_two[0][0], _d_exp_dw_1,
            d_q_out_2, &d_q_step_1_two[0][M], _d_exp_dw_2, 1.0/((double)M), M);

        // Multiply mask
        if (d_q_mask != nullptr)
        {
            multi_real<<<N_BLOCKS, N_THREADS, 0, streams[0][0]>>>(d_q_out_1, d_q_out_1, d_q_mask, 1.0, M);
            multi_real<<<N_BLOCKS, N_THREADS, 0, streams[0][0]>>>(d_q_out_2, d_q_out_2, d_q_mask, 1.0, M);
        }

    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaSolverPseudo::advance_two_propagators_discrete_without_copy(
    double *d_q_in_two, double *d_q_out_two,
    std::string monomer_type_1, std::string monomer_type_2,
    double *d_q_mask)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = cb->get_n_grid();
        const int M_COMPLEX = Pseudo::get_n_complex_grid(cb->get_nx());

        double *_d_exp_dw_1 = d_exp_dw[0][monomer_type_1];
        double *_d_boltz_bond_1 = d_boltz_bond[0][monomer_type_1];
        double *_d_exp_dw_2 = d_exp_dw[0][monomer_type_2];
        double *_d_boltz_bond_2 = d_boltz_bond[0][monomer_type_2];

        // Execute a forward FFT
        cufftExecD2Z(plan_for_two[0], d_q_in_two, d_qk_in_1_two[0]);

        // Multiply exp(-k^2 ds/6) in fourier space
        complex_real_multi_bond_two<<<N_BLOCKS, N_THREADS, 0, streams[0][0]>>>(
            &d_qk_in_1_two[0][0],         _d_boltz_bond_1, 
            &d_qk_in_1_two[0][M_COMPLEX], _d_boltz_bond_2, M_COMPLEX);

        // Execute a backward FFT
        cufftExecZ2D(plan_bak_two[0], d_qk_in_1_two[0], d_q_step_1_two[0]);

        // Evaluate exp(-w*ds) in real space
        real_multi_exp_dw_two<<<N_BLOCKS, N_THREADS, 0, streams[0][0]>>>(
            &d_q_out_two[0], &d_q_step_1_two[0][0], _d_exp_dw_1,
            &d_q_out_two[M], &d_q_step_1_two[0][M], _d_exp_dw_2, 1.0/((double)M), M);

        // Multiply mask
        if (d_q_mask != nullptr)
        {
            multi_real<<<N_BLOCKS, N_THREADS, 0, streams[0][0]>>>(&d_q_out_two[0], &d_q_out_two[0], d_q_mask, 1.0, M);
            multi_real<<<N_BLOCKS, N_THREADS, 0, streams[0][0]>>>(&d_q_out_two[M], &d_q_out_two[M], d_q_mask, 1.0, M);
        }

    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaSolverPseudo::advance_two_propagators_discrete_two_gpus(
    double *d_q_in_1, double *d_q_in_2,
    double *d_q_out_1, double *d_q_out_2,
    std::string monomer_type_1, std::string monomer_type_2,
    double **d_q_mask)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = cb->get_n_grid();
        const int M_COMPLEX = Pseudo::get_n_complex_grid(cb->get_nx());

        double *_d_exp_dw_1 = d_exp_dw[0][monomer_type_1];
        double *_d_boltz_bond_1 = d_boltz_bond[0][monomer_type_1];

        double *_d_exp_dw_2 = d_exp_dw[1][monomer_type_2];
        double *_d_boltz_bond_2 = d_boltz_bond[1][monomer_type_2];

        // Execute a forward FFT
        gpu_error_check(cudaSetDevice(0));
        cufftExecD2Z(plan_for_one[0], d_q_in_1, d_qk_in_1_one[0]);
        gpu_error_check(cudaSetDevice(1));
        cufftExecD2Z(plan_for_one[1], d_q_in_2, d_qk_in_1_one[1]);

        // Multiply exp(-k^2 ds/6) in fourier space
        gpu_error_check(cudaSetDevice(0));
        multi_complex_real<<<N_BLOCKS, N_THREADS, 0, streams[0][0]>>>(d_qk_in_1_one[0], _d_boltz_bond_1, M_COMPLEX);
        gpu_error_check(cudaSetDevice(1));
        multi_complex_real<<<N_BLOCKS, N_THREADS, 0, streams[1][0]>>>(d_qk_in_1_one[1], _d_boltz_bond_2, M_COMPLEX);

        // Execute a backward FFT
        gpu_error_check(cudaSetDevice(0));
        cufftExecZ2D(plan_bak_one[0], d_qk_in_1_one[0], d_q_out_1);
        gpu_error_check(cudaSetDevice(1));
        cufftExecZ2D(plan_bak_one[1], d_qk_in_1_one[1], d_q_out_2);

        // Evaluate exp(-w*ds) in real space
        gpu_error_check(cudaSetDevice(0));
        multi_real<<<N_BLOCKS, N_THREADS, 0, streams[0][0]>>>(d_q_out_1, d_q_out_1, _d_exp_dw_1, 1.0/((double)M), M);
        gpu_error_check(cudaSetDevice(1));
        multi_real<<<N_BLOCKS, N_THREADS, 0, streams[1][0]>>>(d_q_out_2, d_q_out_2, _d_exp_dw_2, 1.0/((double)M), M);

        // Multiply mask
        if (d_q_mask[0] != nullptr && d_q_mask[1] != nullptr)
        {
            gpu_error_check(cudaSetDevice(0));
            multi_real<<<N_BLOCKS, N_THREADS, 0, streams[0][0]>>>(d_q_out_1, d_q_out_1, d_q_mask[0], 1.0, M);
            gpu_error_check(cudaSetDevice(1));
            multi_real<<<N_BLOCKS, N_THREADS, 0, streams[1][0]>>>(d_q_out_2, d_q_out_2, d_q_mask[1], 1.0, M);
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaSolverPseudo::advance_propagator_discrete_half_bond_step(const int GPU, double *d_q_in, double *d_q_out, std::string monomer_type)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = cb->get_n_grid();
        const int M_COMPLEX = Pseudo::get_n_complex_grid(cb->get_nx());

        double *_d_boltz_bond_half = d_boltz_bond_half[GPU][monomer_type];

        // 3D fourier discrete transform, forward and inplace
        cufftExecD2Z(plan_for_one[GPU], d_q_in, d_qk_in_1_one[GPU]);
        // Multiply exp(-k^2 ds/12) in fourier space, in all 3 directions
        multi_complex_real<<<N_BLOCKS, N_THREADS, 0, streams[GPU][0]>>>(d_qk_in_1_one[GPU], _d_boltz_bond_half, 1.0/((double)M), M_COMPLEX);
        // 3D fourier discrete transform, backward and inplace
        cufftExecZ2D(plan_bak_one[GPU], d_qk_in_1_one[GPU], d_q_out);
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaSolverPseudo::compute_single_segment_stress_fourier(const int GPU, double *d_q)
{
    // Execute a forward FFT
    cufftExecD2Z(plan_for_two[GPU], d_q, d_qk_in_1_two[GPU]);
}
std::vector<double> CudaSolverPseudo::compute_single_segment_stress_continuous(
    const int GPU, std::string monomer_type)
{
    try{
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();
        const int N_GPUS    = CudaCommon::get_instance().get_n_gpus();

        const int DIM = cb->get_dim();
        const int M   = cb->get_n_grid();
        const int M_COMPLEX = Pseudo::get_n_complex_grid(cb->get_nx());

        auto bond_lengths = molecules->get_bond_lengths();
        std::vector<double> stress(DIM);
        double bond_length_sq = bond_lengths[monomer_type]*bond_lengths[monomer_type];
        double stress_sum_out[3];
        
        // Multiply two propagators in the fourier spaces
        multi_complex_conjugate<<<N_BLOCKS, N_THREADS, 0, streams[GPU][0]>>>(d_q_multi[GPU], &d_qk_in_1_two[GPU][0], &d_qk_in_1_two[GPU][M_COMPLEX], M_COMPLEX);
        if ( DIM == 3 )
        {
            // x direction
            multi_real<<<N_BLOCKS, N_THREADS, 0, streams[GPU][0]>>>(d_stress_sum[GPU], d_q_multi[GPU], d_fourier_basis_x[GPU], bond_length_sq, M_COMPLEX);
            cub::DeviceReduce::Sum(d_temp_storage[GPU], temp_storage_bytes[GPU], d_stress_sum[GPU], d_stress_sum_out[GPU], M_COMPLEX, streams[GPU][0]);
            gpu_error_check(cudaMemcpyAsync(&stress_sum_out[0], d_stress_sum_out[GPU], sizeof(double), cudaMemcpyDeviceToHost, streams[GPU][0]));

            // y direction
            multi_real<<<N_BLOCKS, N_THREADS, 0, streams[GPU][0]>>>(d_stress_sum[GPU], d_q_multi[GPU], d_fourier_basis_y[GPU], bond_length_sq, M_COMPLEX);
            cub::DeviceReduce::Sum(d_temp_storage[GPU], temp_storage_bytes[GPU], d_stress_sum[GPU], d_stress_sum_out[GPU], M_COMPLEX, streams[GPU][0]);
            gpu_error_check(cudaMemcpyAsync(&stress_sum_out[1], d_stress_sum_out[GPU], sizeof(double), cudaMemcpyDeviceToHost, streams[GPU][0]));

            // z direction
            multi_real<<<N_BLOCKS, N_THREADS, 0, streams[GPU][0]>>>(d_stress_sum[GPU], d_q_multi[GPU], d_fourier_basis_z[GPU], bond_length_sq, M_COMPLEX);
            cub::DeviceReduce::Sum(d_temp_storage[GPU], temp_storage_bytes[GPU], d_stress_sum[GPU], d_stress_sum_out[GPU], M_COMPLEX, streams[GPU][0]);
            gpu_error_check(cudaMemcpyAsync(&stress_sum_out[2], d_stress_sum_out[GPU], sizeof(double), cudaMemcpyDeviceToHost, streams[GPU][0]));
        }
        if ( DIM == 2 )
        {
            // y direction
            multi_real<<<N_BLOCKS, N_THREADS, 0, streams[GPU][0]>>>(d_stress_sum[GPU], d_q_multi[GPU], d_fourier_basis_y[GPU], bond_length_sq, M_COMPLEX);
            cub::DeviceReduce::Sum(d_temp_storage[GPU], temp_storage_bytes[GPU], d_stress_sum[GPU], d_stress_sum_out[GPU], M_COMPLEX, streams[GPU][0]);
            gpu_error_check(cudaMemcpyAsync(&stress_sum_out[0], d_stress_sum_out[GPU], sizeof(double), cudaMemcpyDeviceToHost, streams[GPU][0]));

            // z direction
            multi_real<<<N_BLOCKS, N_THREADS, 0, streams[GPU][0]>>>(d_stress_sum[GPU], d_q_multi[GPU], d_fourier_basis_z[GPU], bond_length_sq, M_COMPLEX);
            cub::DeviceReduce::Sum(d_temp_storage[GPU], temp_storage_bytes[GPU], d_stress_sum[GPU], d_stress_sum_out[GPU], M_COMPLEX, streams[GPU][0]);
            gpu_error_check(cudaMemcpyAsync(&stress_sum_out[1], d_stress_sum_out[GPU], sizeof(double), cudaMemcpyDeviceToHost, streams[GPU][0]));
        }
        if ( DIM == 1 )
        {
            // z direction
            multi_real<<<N_BLOCKS, N_THREADS, 0, streams[GPU][0]>>>(d_stress_sum[GPU], d_q_multi[GPU], d_fourier_basis_z[GPU], bond_length_sq, M_COMPLEX);
            cub::DeviceReduce::Sum(d_temp_storage[GPU], temp_storage_bytes[GPU], d_stress_sum[GPU], d_stress_sum_out[GPU], M_COMPLEX, streams[GPU][0]);
            gpu_error_check(cudaMemcpyAsync(&stress_sum_out[0], d_stress_sum_out[GPU], sizeof(double), cudaMemcpyDeviceToHost, streams[GPU][0]));
        }
        // Synchronize streams and add results
        gpu_error_check(cudaStreamSynchronize(streams[GPU][0]));

        for(int d=0; d<DIM; d++)
            stress[d] += stress_sum_out[d];

        return stress;
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

std::vector<double> CudaSolverPseudo::compute_single_segment_stress_discrete(
    const int GPU, std::string monomer_type, bool is_half_bond_length)
{
    try{
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();
        const int N_GPUS    = CudaCommon::get_instance().get_n_gpus();

        const int DIM = cb->get_dim();
        const int M   = cb->get_n_grid();
        const int M_COMPLEX = Pseudo::get_n_complex_grid(cb->get_nx());

        auto bond_lengths = molecules->get_bond_lengths();
        std::vector<double> stress(DIM);
        double stress_sum_out[3];  
        double bond_length_sq;
        double *_d_boltz_bond;

        if (is_half_bond_length)
        {
            bond_length_sq = 0.5*bond_lengths[monomer_type]*bond_lengths[monomer_type];
            _d_boltz_bond = d_boltz_bond_half[GPU][monomer_type];
        }
        else
        {
            bond_length_sq = bond_lengths[monomer_type]*bond_lengths[monomer_type];
            _d_boltz_bond = d_boltz_bond[GPU][monomer_type];
        }

        // Multiply two propagators in the fourier spaces
        multi_complex_conjugate<<<N_BLOCKS, N_THREADS, 0, streams[GPU][0]>>>(d_q_multi[GPU], &d_qk_in_1_two[GPU][0], &d_qk_in_1_two[GPU][M_COMPLEX], M_COMPLEX);
        multi_real<<<N_BLOCKS, N_THREADS, 0, streams[GPU][0]>>>(d_q_multi[GPU], d_q_multi[GPU], _d_boltz_bond, bond_length_sq, M_COMPLEX);

        if ( DIM == 3 )
        {
            // x direction
            multi_real<<<N_BLOCKS, N_THREADS, 0, streams[GPU][0]>>>(d_stress_sum[GPU], d_q_multi[GPU], d_fourier_basis_x[GPU], 1.0, M_COMPLEX);
            cub::DeviceReduce::Sum(d_temp_storage[GPU], temp_storage_bytes[GPU], d_stress_sum[GPU], d_stress_sum_out[GPU], M_COMPLEX, streams[GPU][0]);
            gpu_error_check(cudaMemcpyAsync(&stress_sum_out[0], d_stress_sum_out[GPU], sizeof(double), cudaMemcpyDeviceToHost, streams[GPU][0]));

            // y direction
            multi_real<<<N_BLOCKS, N_THREADS, 0, streams[GPU][0]>>>(d_stress_sum[GPU], d_q_multi[GPU], d_fourier_basis_y[GPU], 1.0, M_COMPLEX);
            cub::DeviceReduce::Sum(d_temp_storage[GPU], temp_storage_bytes[GPU], d_stress_sum[GPU], d_stress_sum_out[GPU], M_COMPLEX, streams[GPU][0]);
            gpu_error_check(cudaMemcpyAsync(&stress_sum_out[1], d_stress_sum_out[GPU], sizeof(double), cudaMemcpyDeviceToHost, streams[GPU][0]));

            // z direction
            multi_real<<<N_BLOCKS, N_THREADS, 0, streams[GPU][0]>>>(d_stress_sum[GPU], d_q_multi[GPU], d_fourier_basis_z[GPU], 1.0, M_COMPLEX);
            cub::DeviceReduce::Sum(d_temp_storage[GPU], temp_storage_bytes[GPU], d_stress_sum[GPU], d_stress_sum_out[GPU], M_COMPLEX, streams[GPU][0]);
            gpu_error_check(cudaMemcpyAsync(&stress_sum_out[2], d_stress_sum_out[GPU], sizeof(double), cudaMemcpyDeviceToHost, streams[GPU][0]));
        }
        if ( DIM == 2 )
        {
            // y direction
            multi_real<<<N_BLOCKS, N_THREADS, 0, streams[GPU][0]>>>(d_stress_sum[GPU], d_q_multi[GPU], d_fourier_basis_y[GPU], 1.0, M_COMPLEX);
            cub::DeviceReduce::Sum(d_temp_storage[GPU], temp_storage_bytes[GPU], d_stress_sum[GPU], d_stress_sum_out[GPU], M_COMPLEX, streams[GPU][0]);
            gpu_error_check(cudaMemcpyAsync(&stress_sum_out[0], d_stress_sum_out[GPU], sizeof(double), cudaMemcpyDeviceToHost, streams[GPU][0]));

            // z direction
            multi_real<<<N_BLOCKS, N_THREADS, 0, streams[GPU][0]>>>(d_stress_sum[GPU], d_q_multi[GPU], d_fourier_basis_z[GPU], 1.0, M_COMPLEX);
            cub::DeviceReduce::Sum(d_temp_storage[GPU], temp_storage_bytes[GPU], d_stress_sum[GPU], d_stress_sum_out[GPU], M_COMPLEX, streams[GPU][0]);
            gpu_error_check(cudaMemcpyAsync(&stress_sum_out[1], d_stress_sum_out[GPU], sizeof(double), cudaMemcpyDeviceToHost, streams[GPU][0]));
        }
        if ( DIM == 1 )
        {
            // z direction
            multi_real<<<N_BLOCKS, N_THREADS, 0, streams[GPU][0]>>>(d_stress_sum[GPU], d_q_multi[GPU], d_fourier_basis_z[GPU], 1.0, M_COMPLEX);
            cub::DeviceReduce::Sum(d_temp_storage[GPU], temp_storage_bytes[GPU], d_stress_sum[GPU], d_stress_sum_out[GPU], M_COMPLEX, streams[GPU][0]);
            gpu_error_check(cudaMemcpyAsync(&stress_sum_out[0], d_stress_sum_out[GPU], sizeof(double), cudaMemcpyDeviceToHost, streams[GPU][0]));
        }
        // Synchronize streams and add results
        gpu_error_check(cudaStreamSynchronize(streams[GPU][0]));

        for(int d=0; d<DIM; d++)
            stress[d] += stress_sum_out[d];

        return stress;
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}