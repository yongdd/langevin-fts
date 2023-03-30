#define THRUST_IGNORE_DEPRECATED_CPP_DIALECToptimal
#define CUB_IGNORE_DEPRECATED_CPP_DIALECT

#include <complex>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <iostream>
#include "CudaPseudoDiscrete.h"
#include "CudaComputationBox.h"

CudaPseudoDiscrete::CudaPseudoDiscrete(
    ComputationBox *cb,
    Mixture *mx)
    : Pseudo(cb, mx)
{
    try
    {
        const int M = cb->get_n_grid();
        const int M_COMPLEX = this->n_complex_grid;
        const int N_GPUS = CudaCommon::get_instance().get_n_gpus();

        // allocate memory for propagators
        gpu_error_check(cudaSetDevice(0));
        if( mx->get_essential_propagator_codes().size() == 0)
            throw_with_line_number("There is no propagator code. Add polymers first.");
        for(const auto& item: mx->get_essential_propagator_codes())
        {
            std::string dep = item.first;
            int max_n_segment = item.second.max_n_segment;
             // There are N segments

             // Example (N==5)
             // O--O--O--O--O
             // 0  1  2  3  4 essential_blocks

             // Legend)
             // -- : full bond
             // O  : full segment
            d_propagator[dep] = new double*[max_n_segment];
            propagator_size[dep] = max_n_segment;
            for(int i=0; i<propagator_size[dep]; i++)
                gpu_error_check(cudaMalloc((void**)&d_propagator[dep][i], sizeof(double)*M));

            #ifndef NDEBUG
            propagator_finished[dep] = new bool[max_n_segment];
            for(int i=0; i<max_n_segment;i++)
                propagator_finished[dep][i] = false;
            #endif
        }

        // allocate memory for q_junction_cache, which contain partition function at junction of discrete chain
        for(const auto& item: mx->get_essential_propagator_codes())
        {
            std::string dep = item.first;
            d_q_junction_cache[dep] = nullptr;
            gpu_error_check(cudaMalloc((void**)&d_q_junction_cache[dep], sizeof(double)*M));
        }

        // allocate memory for concentrations
        if( mx->get_essential_blocks().size() == 0)
            throw_with_line_number("There is no block. Add polymers first.");
        for(const auto& item: mx->get_essential_blocks())
        {
            d_block_phi[item.first] = nullptr;
            gpu_error_check(cudaMalloc((void**)&d_block_phi[item.first], sizeof(double)*M));
        }

        // create boltz_bond, boltz_bond_half, and exp_dw
        for(const auto& item: mx->get_bond_lengths())
        {
            std::string monomer_type = item.first;
            for(int gpu=0; gpu<N_GPUS; gpu++)
            {
                gpu_error_check(cudaSetDevice(gpu));
                d_boltz_bond     [gpu][monomer_type] = nullptr;
                d_boltz_bond_half[gpu][monomer_type] = nullptr;
                d_exp_dw         [gpu][monomer_type] = nullptr;

                gpu_error_check(cudaMalloc((void**)&d_exp_dw         [gpu][monomer_type], sizeof(double)*M));
                gpu_error_check(cudaMalloc((void**)&d_boltz_bond     [gpu][monomer_type], sizeof(double)*M_COMPLEX));
                gpu_error_check(cudaMalloc((void**)&d_boltz_bond_half[gpu][monomer_type], sizeof(double)*M_COMPLEX));
            }
        }

        // total partition functions for each polymer
        single_partitions = new double[mx->get_n_polymers()];

        // create scheduler for computation of propagator
        sc = new Scheduler(mx->get_essential_propagator_codes(), N_PARALLEL_STREAMS); 

        // create streams
        for(int gpu=0; gpu<N_GPUS; gpu++)
        {
            gpu_error_check(cudaSetDevice(gpu));
            cudaStreamCreate(&streams[gpu]); // for cuFFT
        }
        gpu_error_check(cudaSetDevice(N_GPUS-1));
        cudaStreamCreate(&streams[N_GPUS]); // for PtoP Memcpy

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
            cufftPlanMany(&plan_for_1[gpu], NRANK, n_grid, NULL, 1, 0, NULL, 1, 0, CUFFT_D2Z,1);
            cufftPlanMany(&plan_bak_1[gpu], NRANK, n_grid, NULL, 1, 0, NULL, 1, 0, CUFFT_Z2D,1);
            cufftSetStream(plan_for_1[gpu], streams[gpu]);
            cufftSetStream(plan_bak_1[gpu], streams[gpu]);
        }
        gpu_error_check(cudaSetDevice(0));
        cufftPlanMany(&plan_for_two, NRANK, n_grid, NULL, 1, 0, NULL, 1, 0, CUFFT_D2Z,2);
        cufftPlanMany(&plan_bak_two, NRANK, n_grid, NULL, 1, 0, NULL, 1, 0, CUFFT_Z2D,2);

        // allocate memory for get_concentration
        gpu_error_check(cudaMalloc((void**)&d_phi, sizeof(double)*M));

        // allocate memory for pseudo-spectral: one_step()
        for(int gpu=0; gpu<N_GPUS; gpu++)
        {
            gpu_error_check(cudaSetDevice(gpu));
            gpu_error_check(cudaMalloc((void**)&d_qk_in_1[gpu], sizeof(ftsComplex)*M_COMPLEX));
        }

        if (N_GPUS > 1)
        {
            gpu_error_check(cudaSetDevice(1));
            gpu_error_check(cudaMalloc((void**)&d_propagator_device_1[0], sizeof(double)*M));  // prev
            gpu_error_check(cudaMalloc((void**)&d_propagator_device_1[1], sizeof(double)*M));  // next
        }

        gpu_error_check(cudaSetDevice(0));
        gpu_error_check(cudaMalloc((void**)&d_qk_in_two, sizeof(ftsComplex)*2*M_COMPLEX));
        gpu_error_check(cudaMalloc((void**)&d_q_in_temp_2, sizeof(double)*2*M));
        gpu_error_check(cudaMalloc((void**)&d_q_out_temp_2, sizeof(double)*2*M));

        gpu_error_check(cudaMalloc((void**)&d_q_half_step, sizeof(double)*M));
        gpu_error_check(cudaMalloc((void**)&d_q_junction,  sizeof(double)*M));

        double q_unity[M];
        for(int i=0; i<M; i++)
            q_unity[i] = 1.0;
        gpu_error_check(cudaMalloc((void**)&d_q_unity, sizeof(double)*M));
        gpu_error_check(cudaMemcpy(d_q_unity, q_unity, sizeof(double)*M, cudaMemcpyHostToDevice));

        // allocate memory for stress calculation: compute_stress()
        gpu_error_check(cudaMalloc((void**)&d_fourier_basis_x, sizeof(double)*M_COMPLEX));
        gpu_error_check(cudaMalloc((void**)&d_fourier_basis_y, sizeof(double)*M_COMPLEX));
        gpu_error_check(cudaMalloc((void**)&d_fourier_basis_z, sizeof(double)*M_COMPLEX));
        gpu_error_check(cudaMalloc((void**)&d_q_multi,         sizeof(double)*M_COMPLEX));
        gpu_error_check(cudaMalloc((void**)&d_stress_sum,      sizeof(double)*M_COMPLEX));

        update_bond_function();
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
CudaPseudoDiscrete::~CudaPseudoDiscrete()
{
    const int N_GPUS = CudaCommon::get_instance().get_n_gpus();
    
    for(int gpu=0; gpu<N_GPUS; gpu++)
    {
        cufftDestroy(plan_for_1[gpu]);
        cufftDestroy(plan_bak_1[gpu]);
    }
    cufftDestroy(plan_for_two);
    cufftDestroy(plan_bak_two);

    delete sc;

    delete[] single_partitions;

    for(int gpu=0; gpu<N_GPUS; gpu++)
    {
        for(const auto& item: d_boltz_bond[gpu])
            cudaFree(item.second);
        for(const auto& item: d_boltz_bond_half[gpu])
            cudaFree(item.second);
        for(const auto& item: d_exp_dw[gpu])
            cudaFree(item.second);
    }

    for(const auto& item: d_propagator)
    {
        for(int i=0; i<propagator_size[item.first]; i++)
            cudaFree(item.second[i]);
        delete[] item.second;
    }
    for(const auto& item: d_block_phi)
        cudaFree(item.second);
    for(const auto& item: d_q_junction_cache)
        cudaFree(item.second);

    #ifndef NDEBUG
    for(const auto& item: propagator_finished)
        delete[] item.second;
    #endif

    // for get_concentration
    cudaFree(d_phi);

    // for pseudo-spectral: one_step()
    for(int gpu=0; gpu<N_GPUS; gpu++)
    {
        cudaFree(d_qk_in_1[gpu]);
    }
    if (N_GPUS > 1)
    {
        cudaFree(d_propagator_device_1[0]);
        cudaFree(d_propagator_device_1[1]);
    }
    cudaFree(d_q_unity);
    cudaFree(d_qk_in_two);
    cudaFree(d_q_in_temp_2);
    cudaFree(d_q_out_temp_2);
    cudaFree(d_q_half_step);
    cudaFree(d_q_junction);

    // for stress calculation: compute_stress()
    cudaFree(d_fourier_basis_x);
    cudaFree(d_fourier_basis_y);
    cudaFree(d_fourier_basis_z);
    cudaFree(d_q_multi);
    cudaFree(d_stress_sum);

    // destory streams
    for (int i = 0; i < N_GPUS+1; i++)
        cudaStreamDestroy(streams[i]);
}

void CudaPseudoDiscrete::update_bond_function()
{
    try
    {
        // for pseudo-spectral: one_step()
        const int M_COMPLEX = this->n_complex_grid;
        const int N_GPUS = CudaCommon::get_instance().get_n_gpus();
        double boltz_bond[M_COMPLEX], boltz_bond_half[M_COMPLEX];
        
        for(const auto& item: mx->get_bond_lengths())
        {
            std::string monomer_type = item.first;
            double bond_length_sq = item.second*item.second;

            get_boltz_bond(boltz_bond     , bond_length_sq,   cb->get_nx(), cb->get_dx(), mx->get_ds());
            get_boltz_bond(boltz_bond_half, bond_length_sq/2, cb->get_nx(), cb->get_dx(), mx->get_ds());
            for(int gpu=0; gpu<N_GPUS; gpu++)
            {
                gpu_error_check(cudaSetDevice(gpu));
                gpu_error_check(cudaMemcpy(d_boltz_bond     [gpu][monomer_type], boltz_bond,      sizeof(double)*M_COMPLEX,cudaMemcpyHostToDevice));
                gpu_error_check(cudaMemcpy(d_boltz_bond_half[gpu][monomer_type], boltz_bond_half, sizeof(double)*M_COMPLEX,cudaMemcpyHostToDevice));
            }
        }
        gpu_error_check(cudaSetDevice(0));

        // for stress calculation: compute_stress()
        double fourier_basis_x[M_COMPLEX];
        double fourier_basis_y[M_COMPLEX];
        double fourier_basis_z[M_COMPLEX];
        get_weighted_fourier_basis(fourier_basis_x, fourier_basis_y, fourier_basis_z, cb->get_nx(), cb->get_dx());
        gpu_error_check(cudaMemcpy(d_fourier_basis_x, fourier_basis_x, sizeof(double)*M_COMPLEX,cudaMemcpyHostToDevice));
        gpu_error_check(cudaMemcpy(d_fourier_basis_y, fourier_basis_y, sizeof(double)*M_COMPLEX,cudaMemcpyHostToDevice));
        gpu_error_check(cudaMemcpy(d_fourier_basis_z, fourier_basis_z, sizeof(double)*M_COMPLEX,cudaMemcpyHostToDevice));
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaPseudoDiscrete::compute_statistics(
    std::map<std::string, double*> w_input,
    std::map<std::string, double*> q_init)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();
        const int N_GPUS = CudaCommon::get_instance().get_n_gpus();

        const int M = cb->get_n_grid();
        const double ds = mx->get_ds();

        for(const auto& item: mx->get_essential_propagator_codes())
        {
            if( w_input.find(item.second.monomer_type) == w_input.end())
                throw_with_line_number("monomer_type \"" + item.second.monomer_type + "\" is not in w_input.");
        }

        for(const auto& item: w_input)
        {
            if( d_exp_dw[0].find(item.first) == d_exp_dw[0].end())
                throw_with_line_number("monomer_type \"" + item.first + "\" is not in d_exp_dw.");     
        }

        // exp_dw
        double exp_dw[M];
        for(const auto& item: w_input)
        {
            std::string monomer_type = item.first;
            double *w = item.second;
            for(int i=0; i<M; i++)
                exp_dw[i] = exp(-w[i]*ds);

            for(int gpu=0; gpu<N_GPUS; gpu++)
            {
                gpu_error_check(cudaSetDevice(gpu));
                gpu_error_check(cudaMemcpy(d_exp_dw[gpu][monomer_type], exp_dw, sizeof(double)*M,cudaMemcpyHostToDevice));
            }
        }

        // for each time span
        auto& branch_schedule = sc->get_schedule();
        for (auto parallel_job = branch_schedule.begin(); parallel_job != branch_schedule.end(); parallel_job++)
        {
            gpu_error_check(cudaSetDevice(0));
            // for each propagator
            for(size_t job=0; job<parallel_job->size(); job++)
            {
                auto& key = std::get<0>((*parallel_job)[job]);
                int n_segment_from = std::get<1>((*parallel_job)[job]);
                int n_segment_to = std::get<2>((*parallel_job)[job]);
                auto& deps = mx->get_essential_propagator_code(key).deps;
                auto monomer_type = mx->get_essential_propagator_code(key).monomer_type;

                // check key
                #ifndef NDEBUG
                if (d_propagator.find(key) == d_propagator.end())
                    throw_with_line_number("Could not find key '" + key + "'. ");
                #endif
                double **_d_propagator = d_propagator[key];

                // calculate one block end
                if(n_segment_from == 1 && deps.size() == 0) // if it is leaf node
                {
                     // q_init
                    if (key[0] == '{')
                    {
                        std::string g = Mixture::get_q_input_idx_from_key(key);
                        if (q_init.find(g) == q_init.end())
                            throw_with_line_number("Could not find q_init[\"" + g + "\"].");
                        gpu_error_check(cudaMemcpy(_d_propagator[0], q_init[g], sizeof(double)*M, cudaMemcpyHostToDevice));
                        multi_real<<<N_BLOCKS, N_THREADS>>>(_d_propagator[0], _d_propagator[0], d_exp_dw[0][monomer_type], 1.0, M);
                    }
                    else
                    {
                        gpu_error_check(cudaMemcpy(_d_propagator[0], d_exp_dw[0][monomer_type], sizeof(double)*M, cudaMemcpyDeviceToDevice));
                    }
                    
                    #ifndef NDEBUG
                    propagator_finished[key][0] = true;
                    #endif
                }
                else if (n_segment_from == 1 && deps.size() > 0) // if it is not leaf node
                {
                    // if it is superposed
                    if (key[0] == '[')
                    {
                        // initialize to zero
                        gpu_error_check(cudaMemset(_d_propagator[0], 0, sizeof(double)*M));

                        for(size_t d=0; d<deps.size(); d++)
                        {
                            std::string sub_dep = std::get<0>(deps[d]);
                            int sub_n_segment   = std::get<1>(deps[d]);
                            int sub_n_repeated  = std::get<2>(deps[d]);

                            // check sub key
                            #ifndef NDEBUG
                            if (d_propagator.find(sub_dep) == d_propagator.end())
                                throw_with_line_number("Could not find sub key '" + sub_dep + "'. ");
                            if (!propagator_finished[sub_dep][sub_n_segment-1])
                                throw_with_line_number("Could not compute '" + key +  "', since '"+ sub_dep + std::to_string(sub_n_segment) + "' is not prepared.");
                            #endif

                            lin_comb<<<N_BLOCKS, N_THREADS>>>(
                                _d_propagator[0], 1.0, _d_propagator[0],
                                sub_n_repeated, d_propagator[sub_dep][sub_n_segment-1], M);
                        }
                        one_step_1(0,
                            _d_propagator[0],
                            _d_propagator[0],
                            d_boltz_bond[0][monomer_type],
                            d_exp_dw[0][monomer_type]);

                        #ifndef NDEBUG
                        propagator_finished[key][0] = true;
                        #endif
                    }
                    else
                    {
                        // Example (four branches)
                        //     A
                        //     |
                        // O - . - B
                        //     |
                        //     C

                        // Legend)
                        // .       : junction
                        // O       : full segment
                        // -, |    : half bonds
                        // A, B, C : other full segments

                        // combine branches
                        gpu_error_check(cudaMemcpy(d_q_junction, d_q_unity, sizeof(double)*M, cudaMemcpyDeviceToDevice));

                        for(size_t d=0; d<deps.size(); d++)
                        {
                            std::string sub_dep = std::get<0>(deps[d]);
                            int sub_n_segment   = std::get<1>(deps[d]);

                            // check sub key
                            #ifndef NDEBUG
                            if (d_propagator.find(sub_dep) == d_propagator.end())
                                throw_with_line_number("Could not find sub key '" + sub_dep + "'. ");
                            if (!propagator_finished[sub_dep][sub_n_segment-1])
                                throw_with_line_number("Could not compute '" + key +  "', since '"+ sub_dep + std::to_string(sub_n_segment) + "' is not prepared.");
                            #endif

                            half_bond_step(0, 
                                d_propagator[sub_dep][sub_n_segment-1],
                                d_q_half_step, d_boltz_bond_half[0][mx->get_essential_propagator_code(sub_dep).monomer_type]);

                            multi_real<<<N_BLOCKS, N_THREADS>>>(d_q_junction, d_q_junction, d_q_half_step, 1.0, M);
                        }
                        gpu_error_check(cudaMemcpy(d_q_junction_cache[key], d_q_junction, sizeof(double)*M, cudaMemcpyDeviceToDevice));

                        // add half bond
                        half_bond_step(0, d_q_junction, _d_propagator[0], d_boltz_bond_half[0][monomer_type]);

                        // add full segment
                        multi_real<<<N_BLOCKS, N_THREADS>>>(_d_propagator[0], _d_propagator[0], d_exp_dw[0][monomer_type], 1.0, M);

                        #ifndef NDEBUG
                        propagator_finished[key][0] = true;
                        #endif
                    }
                }
                else
                {
                    int n = n_segment_from-1;

                    #ifndef NDEBUG
                    if (!propagator_finished[key][n-1])
                        throw_with_line_number("unfinished, key: " + key + ", " + std::to_string(n-1));
                    #endif

                    one_step_1(0, 
                        _d_propagator[n-1],
                        _d_propagator[n],
                        d_boltz_bond[0][monomer_type],
                        d_exp_dw[0][monomer_type]);

                    #ifndef NDEBUG
                    propagator_finished[key][n] = true;
                    #endif
                }
            }
                
            // advance propagator successively
            if(parallel_job->size()==1)
            {
                auto& key = std::get<0>((*parallel_job)[0]);
                int n_segment_from = std::get<1>((*parallel_job)[0]);
                int n_segment_to = std::get<2>((*parallel_job)[0]);
                auto monomer_type = mx->get_essential_propagator_code(key).monomer_type;
                double **_d_propagator_key = d_propagator[key];

                for(int n=n_segment_from; n<n_segment_to; n++)
                {
                    #ifndef NDEBUG
                    if (!propagator_finished[key][n-1])
                        throw_with_line_number("unfinished, key: " + key + ", " + std::to_string(n-1));
                    #endif

                    one_step_1(0, 
                        _d_propagator_key[n-1],
                        _d_propagator_key[n],
                        d_boltz_bond[0][monomer_type],
                        d_exp_dw[0][monomer_type]);

                    #ifndef NDEBUG
                    propagator_finished[key][n] = true;
                    #endif
                }
            }
            else if(parallel_job->size()==2)
            {
                auto& key_0 = std::get<0>((*parallel_job)[0]);
                int n_segment_from_0 = std::get<1>((*parallel_job)[0]);
                int n_segment_to_0 = std::get<2>((*parallel_job)[0]);
                auto monomer_type_0 = mx->get_essential_propagator_code(key_0).monomer_type;

                auto& key_1 = std::get<0>((*parallel_job)[1]);
                int n_segment_from_1 = std::get<1>((*parallel_job)[1]);
                int n_segment_to_1 = std::get<2>((*parallel_job)[1]);
                auto monomer_type_1 = mx->get_essential_propagator_code(key_1).monomer_type;

                double **_d_propagator_key_0 = d_propagator[key_0];
                double **_d_propagator_key_1 = d_propagator[key_1];

                if (CudaCommon::get_instance().get_n_gpus() > 1)
                {
                    int prev, next;
                    prev = 0;
                    next = 1;

                    // copy propagator of key1 from device0 to device1
                    gpu_error_check(cudaMemcpy(
                        d_propagator_device_1[prev],
                        _d_propagator_key_1[n_segment_from_1-1],
                        sizeof(double)*M, cudaMemcpyDeviceToDevice));

                    for(int n=0; n<n_segment_to_0-n_segment_from_0; n++)
                    {
                        #ifndef NDEBUG
                        if (!propagator_finished[key_0][n-1+n_segment_from_0])
                            throw_with_line_number("unfinished, key: " + key_0 + ", " + std::to_string(n-1+n_segment_from_0));
                        if (!propagator_finished[key_1][n-1+n_segment_from_1])
                            throw_with_line_number("unfinished, key: " + key_1 + ", " + std::to_string(n-1+n_segment_from_1));
                        #endif

                        // DEVICE 0: calculate propagator of key0
                        gpu_error_check(cudaSetDevice(0));
                        one_step_1(0,
                            _d_propagator_key_0[n-1+n_segment_from_0],
                            _d_propagator_key_0[n+n_segment_from_0],
                            d_boltz_bond[0][monomer_type_0],
                            d_exp_dw[0][monomer_type_0]);

                        // DEVICE 1: calculate propagator of key1
                        gpu_error_check(cudaSetDevice(1));
                        one_step_1(1,
                            d_propagator_device_1[prev],
                            d_propagator_device_1[next],
                            d_boltz_bond[1][monomer_type_1],
                            d_exp_dw[1][monomer_type_1]);

                        // DEVICE 1: copy memory from device 1 to device 0
                        if (n > 0)
                        {
                            gpu_error_check(cudaMemcpyAsync(
                                _d_propagator_key_1[n-1+n_segment_from_1],
                                d_propagator_device_1[prev],
                                sizeof(double)*M, cudaMemcpyDeviceToDevice, streams[2]));
                        }
                        gpu_error_check(cudaStreamSynchronize(streams[1]));
                        gpu_error_check(cudaStreamSynchronize(streams[2]));

                        std::swap(prev, next);

                        #ifndef NDEBUG
                        propagator_finished[key_0][n+n_segment_from_0] = true;
                        propagator_finished[key_1][n+n_segment_from_1] = true;
                        #endif
                    }
                    gpu_error_check(cudaMemcpy(
                        _d_propagator_key_1[n_segment_to_1-1],
                        d_propagator_device_1[prev],
                        sizeof(double)*M, cudaMemcpyDeviceToDevice));
                }
                else
                {
                    for(int n=0; n<n_segment_to_0-n_segment_from_0; n++)
                    {
                        gpu_error_check(cudaSetDevice(0));
                        #ifndef NDEBUG
                        if (!propagator_finished[key_0][n-1+n_segment_from_0])
                            throw_with_line_number("unfinished, key: " + key_0 + ", " + std::to_string(n-n_segment_from_0));
                        if (!propagator_finished[key_1][n-1+n_segment_from_1])
                            throw_with_line_number("unfinished, key: " + key_1 + ", " + std::to_string(n-n_segment_from_1));
                        #endif

                        one_step_2(
                            _d_propagator_key_0[n-1+n_segment_from_0],
                            _d_propagator_key_1[n-1+n_segment_from_1],
                            _d_propagator_key_0[n+n_segment_from_0],
                            _d_propagator_key_1[n+n_segment_from_1],
                            d_boltz_bond[0][monomer_type_0],
                            d_boltz_bond[0][monomer_type_1],
                            d_exp_dw[0][monomer_type_0],
                            d_exp_dw[0][monomer_type_1]);

                        #ifndef NDEBUG
                        propagator_finished[key_0][n+n_segment_from_0] = true;
                        propagator_finished[key_1][n+n_segment_from_1] = true;
                        #endif
                    }
                }
            }
            for(int gpu=0; gpu<N_GPUS; gpu++)
            {
                gpu_error_check(cudaSetDevice(gpu));
                gpu_error_check(cudaDeviceSynchronize());
            }
        }
        gpu_error_check(cudaSetDevice(0));

        // compute total partition function of each distinct polymers
        int current_p = 0;
        for(const auto& block: d_block_phi)
        {
            const auto& key = block.first;
            int p                = std::get<0>(key);
            std::string dep_v    = std::get<1>(key);
            std::string dep_u    = std::get<2>(key);

            // already computed
            if (p != current_p)
                continue;

            int n_superposed;
            // int n_segment_allocated = mx->get_essential_block(key).n_segment_allocated;
            int n_segment_offset    = mx->get_essential_block(key).n_segment_offset;
            int n_segment_original  = mx->get_essential_block(key).n_segment_original;
            std::string monomer_type = mx->get_essential_block(key).monomer_type;

            // contains no '['
            if (dep_u.find('[') == std::string::npos)
                n_superposed = 1;
            else
                n_superposed = mx->get_essential_block(key).v_u.size();

            // check keys
            #ifndef NDEBUG
            if (d_propagator.find(dep_v) == d_propagator.end())
                throw_with_line_number("Could not find dep_v key'" + dep_v + "'. ");
            if (d_propagator.find(dep_u) == d_propagator.end())
                throw_with_line_number("Could not find dep_u key'" + dep_u + "'. ");
            #endif

            single_partitions[p] = ((CudaComputationBox *)cb)->inner_product_inverse_weight_gpu(
                d_propagator[dep_v][n_segment_original-n_segment_offset-1],  // q
                d_propagator[dep_u][0],                                      // q^dagger
                d_exp_dw[0][monomer_type])/n_superposed/cb->get_volume();        

            // std::cout << p << ", " << single_partitions[p] << std::endl;
            // std::cout << p <<", "<< n_segment <<", "<< n_segment_offset <<", "<< single_partitions[p] << std::endl;
            current_p++;
        }

        // calculate segment concentrations
        for(const auto& block: d_block_phi)
        {
            const auto& key = block.first;
            int p                = std::get<0>(key);
            std::string dep_v    = std::get<1>(key);
            std::string dep_u    = std::get<2>(key);

            int n_repeated;
            int n_segment_allocated = mx->get_essential_block(key).n_segment_allocated;
            int n_segment_offset    = mx->get_essential_block(key).n_segment_offset;
            int n_segment_original  = mx->get_essential_block(key).n_segment_original;
            std::string monomer_type = mx->get_essential_block(key).monomer_type;

            // contains no '['
            if (dep_u.find('[') == std::string::npos)
                n_repeated = mx->get_essential_block(key).v_u.size();
            else
                n_repeated = 1;

            // check keys
            #ifndef NDEBUG
            if (d_propagator.find(dep_v) == d_propagator.end())
                throw_with_line_number("Could not find dep_v key'" + dep_v + "'. ");
            if (d_propagator.find(dep_u) == d_propagator.end())
                throw_with_line_number("Could not find dep_u key'" + dep_u + "'. ");
            #endif

            // calculate phi of one block (possibly multiple blocks when using superposition)
            calculate_phi_one_block(
                block.second,             // phi
                d_propagator[dep_v],  // dependency v
                d_propagator[dep_u],  // dependency u
                d_exp_dw[0][monomer_type],     // exp_dw
                n_segment_allocated,
                n_segment_offset,
                n_segment_original);
            
            // normalize concentration
            PolymerChain& pc = mx->get_polymer(p);
            double norm = mx->get_ds()*pc.get_volume_fraction()/pc.get_alpha()/single_partitions[p]*n_repeated;
            lin_comb<<<N_BLOCKS, N_THREADS>>>(block.second, norm, block.second, 0.0, block.second, M);
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaPseudoDiscrete::one_step_1(
    const int GPU,
    double *d_q_in, double *d_q_out,
    double *d_boltz_bond, double *d_exp_dw)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = cb->get_n_grid();
        const int M_COMPLEX = this->n_complex_grid;

        //-------------- step 1 ----------
        // Execute a Forward FFT
        cufftExecD2Z(plan_for_1[GPU], d_q_in, d_qk_in_1[GPU]);

        // Multiply e^(-k^2 ds/6) in fourier space
        multi_complex_real<<<N_BLOCKS, N_THREADS, 0, streams[GPU]>>>(d_qk_in_1[GPU], d_boltz_bond, M_COMPLEX);

        // Execute a backward FFT
        cufftExecZ2D(plan_bak_1[GPU], d_qk_in_1[GPU], d_q_out);

        // Evaluate e^(-w*ds) in real space
        multi_real<<<N_BLOCKS, N_THREADS, 0, streams[GPU]>>>(d_q_out, d_q_out, d_exp_dw, 1.0/((double)M), M);
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaPseudoDiscrete::one_step_2(
    double *d_q_in_1, double *d_q_in_2,
    double *d_q_out_1, double *d_q_out_2,
    double *d_boltz_bond_1, double *d_boltz_bond_2,  
    double *d_exp_dw_1, double *d_exp_dw_2)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = cb->get_n_grid();
        const int M_COMPLEX = this->n_complex_grid;

        gpu_error_check(cudaMemcpy(&d_q_in_temp_2[0], d_q_in_1, sizeof(double)*M, cudaMemcpyDeviceToDevice));
        gpu_error_check(cudaMemcpy(&d_q_in_temp_2[M], d_q_in_2, sizeof(double)*M, cudaMemcpyDeviceToDevice));

        //-------------- step 1 ----------
        // Execute a Forward FFT
        cufftExecD2Z(plan_for_two, d_q_in_temp_2, d_qk_in_two);

        // Multiply e^(-k^2 ds/6) in fourier space
        multi_complex_real<<<N_BLOCKS, N_THREADS>>>(&d_qk_in_two[0],         d_boltz_bond_1, M_COMPLEX);
        multi_complex_real<<<N_BLOCKS, N_THREADS>>>(&d_qk_in_two[M_COMPLEX], d_boltz_bond_2, M_COMPLEX);

        // Execute a backward FFT
        cufftExecZ2D(plan_bak_two, d_qk_in_two, d_q_out_temp_2);

        // Evaluate e^(-w*ds) in real space
        multi_real<<<N_BLOCKS, N_THREADS>>>(d_q_out_1, &d_q_out_temp_2[0], d_exp_dw_1, 1.0/((double)M), M);
        multi_real<<<N_BLOCKS, N_THREADS>>>(d_q_out_2, &d_q_out_temp_2[M], d_exp_dw_2, 1.0/((double)M), M);
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaPseudoDiscrete::half_bond_step(const int GPU, double *d_q_in, double *d_q_out, double *d_boltz_bond_half)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = cb->get_n_grid();
        const int M_COMPLEX = this->n_complex_grid;

        // 3D fourier discrete transform, forward and inplace
        cufftExecD2Z(plan_for_1[GPU], d_q_in, d_qk_in_1[GPU]);
        // multiply e^(-k^2 ds/12) in fourier space, in all 3 directions
        multi_complex_real<<<N_BLOCKS, N_THREADS, 0, streams[GPU]>>>(d_qk_in_1[GPU], d_boltz_bond_half, 1.0/((double)M), M_COMPLEX);
        // 3D fourier discrete transform, backward and inplace
        cufftExecZ2D(plan_bak_1[GPU], d_qk_in_1[GPU], d_q_out);
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaPseudoDiscrete::calculate_phi_one_block(
    double *d_phi, double **d_q_1, double **d_q_2, double *d_exp_dw, const int N, const int N_OFFSET, const int N_ORIGINAL)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = cb->get_n_grid();
        // Compute segment concentration
        multi_real<<<N_BLOCKS, N_THREADS>>>(d_phi,d_q_1[N_ORIGINAL-N_OFFSET-1], d_q_2[0], 1.0, M);
        for(int n=1; n<N; n++)
        {
            add_multi_real<<<N_BLOCKS, N_THREADS>>>(d_phi, d_q_1[N_ORIGINAL-N_OFFSET-n-1], d_q_2[n], 1.0, M);
        }
        divide_real<<<N_BLOCKS, N_THREADS>>>(d_phi, d_phi, d_exp_dw, 1.0, M);
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
double CudaPseudoDiscrete::get_total_partition(int polymer)
{
    try
    {
        return single_partitions[polymer];
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaPseudoDiscrete::get_monomer_concentration(std::string monomer_type, double *phi)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = cb->get_n_grid();
        // initialize to zero
        gpu_error_check(cudaMemset(d_phi, 0, sizeof(double)*M));

        // for each block
        for(const auto& block: d_block_phi)
        {
            const auto& key = block.first;
            std::string dep_v = std::get<1>(key);
            int n_segment_allocated = mx->get_essential_block(key).n_segment_allocated;
            if (Mixture::get_monomer_type_from_key(dep_v) == monomer_type && n_segment_allocated != 0)
                lin_comb<<<N_BLOCKS, N_THREADS>>>(d_phi, 1.0, d_phi, 1.0, block.second, M);
        }
        gpu_error_check(cudaMemcpy(phi, d_phi, sizeof(double)*M, cudaMemcpyDeviceToHost));
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaPseudoDiscrete::get_polymer_concentration(int p, double *phi)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = cb->get_n_grid();
        const int P = mx->get_n_polymers();

        if (p < 0 || p > P-1)
            throw_with_line_number("Index (" + std::to_string(p) + ") must be in range [0, " + std::to_string(P-1) + "]");

        if (mx->is_using_superposition())
            throw_with_line_number("Disable 'superposition' option to obtain concentration of each block.");

        PolymerChain& pc = mx->get_polymer(p);
        std::vector<PolymerChainBlock>& blocks = pc.get_blocks();

        for(size_t b=0; b<blocks.size(); b++)
        {
            std::string dep_v = pc.get_propagator_key(blocks[b].v, blocks[b].u);
            std::string dep_u = pc.get_propagator_key(blocks[b].u, blocks[b].v);
            if (dep_v < dep_u)
                dep_v.swap(dep_u);

            // copy normalized concentration
            double norm = cb->get_volume()*mx->get_ds()*pc.get_volume_fraction()/pc.get_alpha()/single_partitions[p];
            lin_comb<<<N_BLOCKS, N_THREADS>>>(d_phi, 0.0, d_phi, norm, d_block_phi[std::make_tuple(p, dep_v, dep_u)], M);
            gpu_error_check(cudaMemcpy(&phi[b*M], d_phi, sizeof(double)*M, cudaMemcpyDeviceToHost));
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
std::vector<double> CudaPseudoDiscrete::compute_stress()
{
    // This method should be invoked after invoking compute_statistics().

    // To calculate stress, we multiply weighted fourier basis to q(k)*q^dagger(-k).
    // We only need the real part of stress calculation.
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int DIM  = cb->get_dim();
        const int M    = cb->get_n_grid();
        const int M_COMPLEX = this->n_complex_grid;

        auto bond_lengths = mx->get_bond_lengths();
        std::vector<double> stress(cb->get_dim());
        std::map<std::tuple<int, std::string, std::string>, std::array<double,3>> block_dq_dl;
        thrust::device_ptr<double> temp_gpu_ptr(d_stress_sum);

        // reset stress map
        for(const auto& item: d_block_phi)
        {
            for(int d=0; d<3; d++)
                block_dq_dl[item.first][d] = 0.0;
        }

        // compute stress for each block
        for(const auto& block: d_block_phi)
        {
            const auto& key = block.first;
            int p                = std::get<0>(key);
            std::string dep_v    = std::get<1>(key);
            std::string dep_u    = std::get<2>(key);

            const int N           = mx->get_essential_block(key).n_segment_allocated;
            const int N_OFFSET    = mx->get_essential_block(key).n_segment_offset;
            const int N_ORIGINAL  = mx->get_essential_block(key).n_segment_original;
            std::string monomer_type = mx->get_essential_block(key).monomer_type;

            // contains no '['
            int n_repeated;
            if (dep_u.find('[') == std::string::npos)
                n_repeated = mx->get_essential_block(key).v_u.size();
            else
                n_repeated = 1;

            double **d_q_1 = d_propagator[dep_v];    // dependency v
            double **d_q_2 = d_propagator[dep_u];    // dependency u

            double bond_length_sq;
            double *d_boltz_bond_now;

            std::array<double,3> _block_dq_dl = block_dq_dl[key];

            // std::cout << "dep_v: " << dep_v << std::endl;
            // std::cout << "dep_u: " << dep_u << std::endl;

            // compute stress
            for(int n=0; n<=N; n++)
            {
                // at v
                if (n + N_OFFSET == N_ORIGINAL)
                {
                    // std::cout << "case 1: " << q_junction_cache[dep_v][0] << ", " << q_2[(N-1)*M] << std::endl;
                    if (mx->get_essential_propagator_code(dep_v).deps.size() == 0) // if v is leaf node, skip
                        continue;
                    
                    gpu_error_check(cudaMemcpy(&d_q_in_temp_2[0], d_q_junction_cache[dep_v], sizeof(double)*M, cudaMemcpyDeviceToDevice));
                    gpu_error_check(cudaMemcpy(&d_q_in_temp_2[M], d_q_2[N-1],                sizeof(double)*M, cudaMemcpyDeviceToDevice));

                    bond_length_sq = 0.5*bond_lengths[monomer_type]*bond_lengths[monomer_type];
                    d_boltz_bond_now = d_boltz_bond_half[0][monomer_type];
                }
                // at u
                else if (n + N_OFFSET == 0){
                    // std::cout << "case 2: " << q_1[(N_ORIGINAL-N_OFFSET-1)*M] << ", " << q_junction_cache[dep_u][0] << std::endl;
                    if (mx->get_essential_propagator_code(dep_u).deps.size() == 0) // if u is leaf node, skip
                        continue;

                    gpu_error_check(cudaMemcpy(&d_q_in_temp_2[0], d_q_1[N_ORIGINAL-1],       sizeof(double)*M, cudaMemcpyDeviceToDevice));
                    gpu_error_check(cudaMemcpy(&d_q_in_temp_2[M], d_q_junction_cache[dep_u], sizeof(double)*M, cudaMemcpyDeviceToDevice));
                    bond_length_sq = 0.5*bond_lengths[monomer_type]*bond_lengths[monomer_type];
                    d_boltz_bond_now = d_boltz_bond_half[0][monomer_type];
                }
                // at superposition junction
                else if (n == 0)
                {
                    // std::cout << "case 4" << std::endl;
                    continue;
                }
                // within the blocks
                else
                {
                    gpu_error_check(cudaMemcpy(&d_q_in_temp_2[0], d_q_1[N_ORIGINAL-N_OFFSET-n-1], sizeof(double)*M, cudaMemcpyDeviceToDevice));
                    gpu_error_check(cudaMemcpy(&d_q_in_temp_2[M], d_q_2[n-1],                     sizeof(double)*M, cudaMemcpyDeviceToDevice));
                    bond_length_sq = bond_lengths[monomer_type]*bond_lengths[monomer_type];
                    d_boltz_bond_now = d_boltz_bond[0][monomer_type];
                }

                // execute a Forward FFT
                cufftExecD2Z(plan_for_two, d_q_in_temp_2, d_qk_in_two);

                // multiply two propagators in the fourier spaces
                multi_complex_conjugate<<<N_BLOCKS, N_THREADS>>>(d_q_multi, &d_qk_in_two[0], &d_qk_in_two[M_COMPLEX], M_COMPLEX);
                multi_real<<<N_BLOCKS, N_THREADS>>>(d_q_multi, d_q_multi, d_boltz_bond_now, bond_length_sq, M_COMPLEX);
                
                if ( DIM == 3 )
                {
                    multi_real<<<N_BLOCKS, N_THREADS>>>(d_stress_sum, d_q_multi, d_fourier_basis_x, 1.0, M_COMPLEX);
                    _block_dq_dl[0] += thrust::reduce(temp_gpu_ptr, temp_gpu_ptr + M_COMPLEX)*n_repeated;

                    multi_real<<<N_BLOCKS, N_THREADS>>>(d_stress_sum, d_q_multi, d_fourier_basis_y, 1.0, M_COMPLEX);
                    _block_dq_dl[1] += thrust::reduce(temp_gpu_ptr, temp_gpu_ptr + M_COMPLEX)*n_repeated;

                    multi_real<<<N_BLOCKS, N_THREADS>>>(d_stress_sum, d_q_multi, d_fourier_basis_z, 1.0, M_COMPLEX);
                    _block_dq_dl[2] += thrust::reduce(temp_gpu_ptr, temp_gpu_ptr + M_COMPLEX)*n_repeated;
                }
                if ( DIM == 2 )
                {
                    multi_real<<<N_BLOCKS, N_THREADS>>>(d_stress_sum, d_q_multi, d_fourier_basis_y, 1.0, M_COMPLEX);
                    _block_dq_dl[0] += thrust::reduce(temp_gpu_ptr, temp_gpu_ptr + M_COMPLEX)*n_repeated;
                    
                    multi_real<<<N_BLOCKS, N_THREADS>>>(d_stress_sum, d_q_multi, d_fourier_basis_z, 1.0, M_COMPLEX);
                    _block_dq_dl[1] += thrust::reduce(temp_gpu_ptr, temp_gpu_ptr + M_COMPLEX)*n_repeated;
                }
                if ( DIM == 1 )
                {
                    multi_real<<<N_BLOCKS, N_THREADS>>>(d_stress_sum, d_q_multi, d_fourier_basis_z, 1.0, M_COMPLEX);
                    _block_dq_dl[0] += thrust::reduce(temp_gpu_ptr, temp_gpu_ptr + M_COMPLEX)*n_repeated;
                }
            }
            block_dq_dl[key] = _block_dq_dl;
        }

        // compute total stress
        for(int d=0; d<cb->get_dim(); d++)
            stress[d] = 0.0;
        for(const auto& block: d_block_phi)
        {
            const auto& key = block.first;
            int p                = std::get<0>(key);
            std::string dep_v    = std::get<1>(key);
            std::string dep_u    = std::get<2>(key);
            PolymerChain& pc = mx->get_polymer(p);

            for(int d=0; d<cb->get_dim(); d++)
                stress[d] += block_dq_dl[key][d]*pc.get_volume_fraction()/pc.get_alpha()/single_partitions[p];
        }
        for(int d=0; d<cb->get_dim(); d++)
            stress[d] /= -3.0*cb->get_lx(d)*M*M/mx->get_ds();
            
        return stress;
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaPseudoDiscrete::get_chain_propagator(double *q_out, int polymer, int v, int u, int n)
{ 
    // This method should be invoked after invoking compute_statistics()

    // Get chain propagator for a selected polymer, block and direction.
    // This is made for debugging and testing.
    try
    {
        const int M = cb->get_n_grid();
        PolymerChain& pc = mx->get_polymer(polymer);
        std::string dep = pc.get_propagator_key(v,u);

        if (mx->get_essential_propagator_codes().find(dep) == mx->get_essential_propagator_codes().end())
            throw_with_line_number("Could not find the propagator code '" + dep + "'. Disable 'superposition' option to obtain propagators.");

        const int N = mx->get_essential_propagator_codes()[dep].max_n_segment;
        if (n < 1 || n > N)
            throw_with_line_number("n (" + std::to_string(n) + ") must be in range [1, " + std::to_string(N) + "]");

        gpu_error_check(cudaMemcpy(q_out, d_propagator[dep][n-1], sizeof(double)*M,cudaMemcpyDeviceToHost));
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
