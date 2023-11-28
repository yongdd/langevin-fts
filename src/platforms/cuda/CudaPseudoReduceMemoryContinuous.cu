#include <complex>
#include <thrust/reduce.h>
#include "CudaPseudoReduceMemoryContinuous.h"
#include "CudaComputationBox.h"
#include "SimpsonRule.h"

CudaPseudoReduceMemoryContinuous::CudaPseudoReduceMemoryContinuous(
    ComputationBox *cb,
    Molecules *molecules)
    : Pseudo(cb, molecules)
{
    try{
        const int M = cb->get_n_grid();
        const int M_COMPLEX = this->n_complex_grid;
        const int N_GPUS = CudaCommon::get_instance().get_n_gpus();

        // Allocate memory for propagators
        gpu_error_check(cudaSetDevice(0));
        if( molecules->get_essential_propagator_codes().size() == 0)
            throw_with_line_number("There is no propagator code. Add polymers first.");
        for(const auto& item: molecules->get_essential_propagator_codes())
        {
            std::string key = item.first;
            int max_n_segment = item.second.max_n_segment;

            propagator_size[key] = max_n_segment+1;
            propagator[key] = new double*[max_n_segment+1];
            // Allocate pinned memory for device overlapping
            for(int i=0; i<propagator_size[key]; i++)
                gpu_error_check(cudaMallocHost((void**)&propagator[key][i], sizeof(double)*M));

            #ifndef NDEBUG
            propagator_finished[key] = new bool[max_n_segment+1];
            for(int i=0; i<=max_n_segment;i++)
                propagator_finished[key][i] = false;
            #endif
        }

        // Allocate memory for concentrations
        if( molecules->get_essential_blocks().size() == 0)
            throw_with_line_number("There is no block. Add polymers first.");
        for(const auto& item: molecules->get_essential_blocks())
        {
            block_phi[item.first] = nullptr;
            // Allocate pinned memory
            gpu_error_check(cudaMallocHost((void**)&block_phi[item.first], sizeof(double)*M));
        }

        // Create boltz_bond, boltz_bond_half, exp_dw, and exp_dw_half
        for(const auto& item: molecules->get_bond_lengths())
        {
            std::string monomer_type = item.first;
            for(int gpu=0; gpu<N_GPUS; gpu++)
            {
                gpu_error_check(cudaSetDevice(gpu));
                d_boltz_bond     [gpu][monomer_type] = nullptr;
                d_boltz_bond_half[gpu][monomer_type] = nullptr;
                d_exp_dw         [gpu][monomer_type] = nullptr;
                d_exp_dw_half    [gpu][monomer_type] = nullptr;

                gpu_error_check(cudaMalloc((void**)&d_exp_dw         [gpu][monomer_type], sizeof(double)*M));
                gpu_error_check(cudaMalloc((void**)&d_exp_dw_half    [gpu][monomer_type], sizeof(double)*M));
                gpu_error_check(cudaMalloc((void**)&d_boltz_bond     [gpu][monomer_type], sizeof(double)*M_COMPLEX));
                gpu_error_check(cudaMalloc((void**)&d_boltz_bond_half[gpu][monomer_type], sizeof(double)*M_COMPLEX));
            }
        }

        // Total partition functions for each polymer
        single_partitions = new double[molecules->get_n_polymer_types()];

        // Remember one segment for each polymer chain to compute total partition function
        int current_p = 0;
        for(const auto& block: block_phi)
        {
            const auto& key = block.first;
            int p                = std::get<0>(key);
            std::string dep_v    = std::get<1>(key);
            std::string dep_u    = std::get<2>(key);

            // Skip if already found one segment
            if (p != current_p)
                continue;

            int n_superposed;
            int n_segment_offset    = molecules->get_essential_block(key).n_segment_offset;
            int n_segment_original  = molecules->get_essential_block(key).n_segment_original;

            // Contains no '['
            if (dep_u.find('[') == std::string::npos)
                n_superposed = 1;
            else
                n_superposed = molecules->get_essential_block(key).v_u.size();

            single_partition_segment.push_back(std::make_tuple(
                p,
                propagator[dep_v][n_segment_original-n_segment_offset],   // q
                propagator[dep_u][0],                                   // Q_dagger
                n_superposed                    // How many propagators are aggregated
                ));
            current_p++;
        }

        // Create scheduler for computation of propagator
        sc = new Scheduler(molecules->get_essential_propagator_codes(), N_SCHEDULER_STREAMS); 

        // Create streams
        for(int gpu=0; gpu<N_GPUS; gpu++)
        {
            gpu_error_check(cudaSetDevice(gpu));
            gpu_error_check(cudaStreamCreate(&streams[gpu][0])); // for kernel execution
            gpu_error_check(cudaStreamCreate(&streams[gpu][1])); // for memcpy
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
        // Allocate memory for pseudo-spectral: advance_one_propagator()
        gpu_error_check(cudaMalloc((void**)&d_propagator_sub_dep[0], sizeof(double)*M)); // for prev
        gpu_error_check(cudaMalloc((void**)&d_propagator_sub_dep[1], sizeof(double)*M)); // for next

        for(int gpu=0; gpu<N_GPUS; gpu++)
        {
            gpu_error_check(cudaSetDevice(gpu));
            // Allocate memory for propagator computation
            gpu_error_check(cudaMalloc((void**)&d_q_one[gpu][0], sizeof(double)*M)); // for prev
            gpu_error_check(cudaMalloc((void**)&d_q_one[gpu][1], sizeof(double)*M)); // for next
            // Allocate memory for pseudo-spectral: advance_one_propagator()
            gpu_error_check(cudaMalloc((void**)&d_q_step_1_one[gpu], sizeof(double)*M));
            gpu_error_check(cudaMalloc((void**)&d_q_step_2_one[gpu], sizeof(double)*M));
            gpu_error_check(cudaMalloc((void**)&d_q_step_1_two[gpu], sizeof(double)*2*M));
            gpu_error_check(cudaMalloc((void**)&d_qk_in_2_one[gpu], sizeof(ftsComplex)*M_COMPLEX));
            gpu_error_check(cudaMalloc((void**)&d_qk_in_1_two[gpu], sizeof(ftsComplex)*2*M_COMPLEX));
        }

        gpu_error_check(cudaSetDevice(0));
        double q_unity[M];
        for(int i=0; i<M; i++)
            q_unity[i] = 1.0;
        gpu_error_check(cudaMalloc((void**)&d_q_unity, sizeof(double)*M));
        gpu_error_check(cudaMemcpy(d_q_unity, q_unity, sizeof(double)*M, cudaMemcpyHostToDevice));

        // For concentration computation
        gpu_error_check(cudaMalloc((void**)&d_q_block_v[0], sizeof(double)*M)); // for prev
        gpu_error_check(cudaMalloc((void**)&d_q_block_v[1], sizeof(double)*M)); // for next
        gpu_error_check(cudaMalloc((void**)&d_q_block_u[0], sizeof(double)*M)); // for prev
        gpu_error_check(cudaMalloc((void**)&d_q_block_u[1], sizeof(double)*M)); // for next
        gpu_error_check(cudaMalloc((void**)&d_phi,          sizeof(double)*M));

        // Allocate memory for stress calculation: compute_stress()
        for(int gpu=0; gpu<N_GPUS; gpu++)
        {
            gpu_error_check(cudaSetDevice(gpu));
            gpu_error_check(cudaMalloc((void**)&d_fourier_basis_x[gpu], sizeof(double)*M_COMPLEX));
            gpu_error_check(cudaMalloc((void**)&d_fourier_basis_y[gpu], sizeof(double)*M_COMPLEX));
            gpu_error_check(cudaMalloc((void**)&d_fourier_basis_z[gpu], sizeof(double)*M_COMPLEX));
            gpu_error_check(cudaMalloc((void**)&d_stress_sum[gpu],      sizeof(double)*M_COMPLEX));
            gpu_error_check(cudaMalloc((void**)&d_stress_sum_out[gpu],  sizeof(double)*1));
            gpu_error_check(cudaMalloc((void**)&d_stress_q[gpu][0],     sizeof(double)*2*M)); // prev
            gpu_error_check(cudaMalloc((void**)&d_stress_q[gpu][1],     sizeof(double)*2*M)); // next
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
        update_bond_function();

        gpu_error_check(cudaSetDevice(0));
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
CudaPseudoReduceMemoryContinuous::~CudaPseudoReduceMemoryContinuous()
{
    const int N_GPUS = CudaCommon::get_instance().get_n_gpus();
    
    for(int gpu=0; gpu<N_GPUS; gpu++)
    {
        cufftDestroy(plan_for_one[gpu]);
        cufftDestroy(plan_for_two[gpu]);
        cufftDestroy(plan_bak_one[gpu]);
        cufftDestroy(plan_bak_two[gpu]);
    }

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
        for(const auto& item: d_exp_dw_half[gpu])
            cudaFree(item.second);
    }

    for(const auto& item: propagator)
    {
        for(int i=0; i<propagator_size[item.first]; i++)
            cudaFreeHost(item.second[i]);
        delete[] item.second;
    }
    for(const auto& item: block_phi)
        cudaFreeHost(item.second);

    #ifndef NDEBUG
    for(const auto& item: propagator_finished)
        delete[] item.second;
    #endif

    for(int gpu=0; gpu<N_GPUS; gpu++)
    {
        cudaFree(d_q_one[gpu][0]);
        cudaFree(d_q_one[gpu][1]);
        cudaFree(d_q_step_1_one[gpu]);
        cudaFree(d_q_step_2_one[gpu]);
        cudaFree(d_q_step_1_two[gpu]);
        cudaFree(d_qk_in_2_one[gpu]);
        cudaFree(d_qk_in_1_two[gpu]);
    }
    cudaFree(d_q_unity);

    // For pseudo-spectral: advance_one_propagator()
    cudaFree(d_propagator_sub_dep[0]);
    cudaFree(d_propagator_sub_dep[1]);

    // For stress calculation: compute_stress()
    for(int gpu=0; gpu<N_GPUS; gpu++)
    {
        cudaFree(d_fourier_basis_x[gpu]);
        cudaFree(d_fourier_basis_y[gpu]);
        cudaFree(d_fourier_basis_z[gpu]);
        cudaFree(d_stress_q[gpu][0]);
        cudaFree(d_stress_q[gpu][1]);
        cudaFree(d_stress_sum[gpu]);
        cudaFree(d_stress_sum_out[gpu]);
        cudaFree(d_q_multi[gpu]);
        cudaFree(d_temp_storage[gpu]);
    }

    // For concentration computation
    cudaFree(d_q_block_v[0]);
    cudaFree(d_q_block_v[1]);
    cudaFree(d_q_block_u[0]);
    cudaFree(d_q_block_u[1]);
    cudaFree(d_phi);

    // Destroy streams
    for(int gpu=0; gpu<N_GPUS; gpu++)
    {
        cudaStreamDestroy(streams[gpu][0]);
        cudaStreamDestroy(streams[gpu][1]);
    }
}

void CudaPseudoReduceMemoryContinuous::update_bond_function()
{
    try{
        // For pseudo-spectral: advance_one_propagator()
        const int M_COMPLEX = this->n_complex_grid;
        const int N_GPUS = CudaCommon::get_instance().get_n_gpus();
        double boltz_bond[M_COMPLEX], boltz_bond_half[M_COMPLEX];

        for(const auto& item: molecules->get_bond_lengths())
        {
            std::string monomer_type = item.first;
            double bond_length_sq = item.second*item.second;
            
            get_boltz_bond(boltz_bond     , bond_length_sq,   cb->get_nx(), cb->get_dx(), molecules->get_ds());
            get_boltz_bond(boltz_bond_half, bond_length_sq/2, cb->get_nx(), cb->get_dx(), molecules->get_ds());
        
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
        get_weighted_fourier_basis(fourier_basis_x, fourier_basis_y, fourier_basis_z, cb->get_nx(), cb->get_dx());
        for(int gpu=0; gpu<N_GPUS; gpu++)
        {
            gpu_error_check(cudaSetDevice(gpu));
            gpu_error_check(cudaMemcpy(d_fourier_basis_x[gpu], fourier_basis_x, sizeof(double)*M_COMPLEX, cudaMemcpyHostToDevice));
            gpu_error_check(cudaMemcpy(d_fourier_basis_y[gpu], fourier_basis_y, sizeof(double)*M_COMPLEX, cudaMemcpyHostToDevice));
            gpu_error_check(cudaMemcpy(d_fourier_basis_z[gpu], fourier_basis_z, sizeof(double)*M_COMPLEX, cudaMemcpyHostToDevice));
        }
        gpu_error_check(cudaSetDevice(0));
    }
    catch(std::exception& exc)
    {
        throw_with_line_number(exc.what());
    }
}
void CudaPseudoReduceMemoryContinuous::compute_statistics(
    std::map<std::string, const double*> w_input,
    std::map<std::string, const double*> q_init, std::string device)
{
    try{
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();
        const int N_GPUS = CudaCommon::get_instance().get_n_gpus();

        const int M = cb->get_n_grid();
        const double ds = molecules->get_ds();

        cudaMemcpyKind cudaMemcpyInputToDevice;
        if (device == "gpu")
            cudaMemcpyInputToDevice = cudaMemcpyDeviceToDevice;
        else if(device == "cpu")
            cudaMemcpyInputToDevice = cudaMemcpyHostToDevice;
        else
        {
            throw_with_line_number("Invalid device \"" + device + "\".");
        }

        for(const auto& item: molecules->get_essential_propagator_codes())
        {
            if( w_input.find(item.second.monomer_type) == w_input.end())
                throw_with_line_number("monomer_type \"" + item.second.monomer_type + "\" is not in w_input.");
        }

        for(const auto& item: w_input)
        {
            if( d_exp_dw[0].find(item.first) == d_exp_dw[0].end())
                throw_with_line_number("monomer_type \"" + item.first + "\" is not in d_exp_dw.");     
        }

        // Exp_dw and exp_dw_half
        for(const auto& item: w_input)
        {
            std::string monomer_type = item.first;
            const double *w = item.second;

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
            // Synchronize all GPUs
            for(int gpu=0; gpu<N_GPUS; gpu++)
            {
                gpu_error_check(cudaSetDevice(gpu));
                gpu_error_check(cudaDeviceSynchronize());
            }
        }

        auto& branch_schedule = sc->get_schedule();
        // For each time span
        for (auto parallel_job = branch_schedule.begin(); parallel_job != branch_schedule.end(); parallel_job++)
        {
            gpu_error_check(cudaSetDevice(0));
            // For each propagator
            for(size_t job=0; job<parallel_job->size(); job++)
            {
                auto& key = std::get<0>((*parallel_job)[job]);
                int n_segment_from = std::get<1>((*parallel_job)[job]);
                int n_segment_to = std::get<2>((*parallel_job)[job]);
                auto& deps = molecules->get_essential_propagator_code(key).deps;
                auto monomer_type = molecules->get_essential_propagator_code(key).monomer_type;

                // Check key
                #ifndef NDEBUG
                if (propagator.find(key) == propagator.end())
                    throw_with_line_number("Could not find key '" + key + "'. ");
                #endif
                double *_propagator_0 = propagator[key][0];

                // If it is leaf node
                if(deps.size() == 0)
                {
                    // Q_init
                    if (key[0] == '{')
                    {
                        std::string g = Molecules::get_q_input_idx_from_key(key);
                        if (q_init.find(g) == q_init.end())
                            throw_with_line_number( "Could not find q_init[\"" + g + "\"].");
                        gpu_error_check(cudaMemcpy(d_q_one[0][0], q_init[g], sizeof(double)*M, cudaMemcpyInputToDevice));
                    }
                    else
                    {
                        gpu_error_check(cudaMemcpy(d_q_one[0][0], d_q_unity, sizeof(double)*M, cudaMemcpyDeviceToDevice));
                    }
                    gpu_error_check(cudaMemcpy(_propagator_0, d_q_one[0][0], sizeof(double)*M, cudaMemcpyDeviceToHost));

                    #ifndef NDEBUG
                    propagator_finished[key][0] = true;
                    #endif
                }
                // If it is not leaf node
                else if (n_segment_from == 1 && deps.size() > 0)
                {
                    // If it is superposed
                    if (key[0] == '[')
                    {
                        // Initialize to zero
                        gpu_error_check(cudaMemset(d_q_one[0][0], 0, sizeof(double)*M));

                        int prev, next;
                        prev = 0;
                        next = 1;

                        // Copy memory from host to device
                        std::string sub_dep = std::get<0>(deps[0]);
                        int sub_n_segment   = std::get<1>(deps[0]);
                        int sub_n_repeated;
                        gpu_error_check(cudaMemcpy(d_propagator_sub_dep[prev], propagator[sub_dep][sub_n_segment], sizeof(double)*M, cudaMemcpyHostToDevice));

                        for(size_t d=0; d<deps.size(); d++)
                        {
                            sub_dep         = std::get<0>(deps[d]);
                            sub_n_segment   = std::get<1>(deps[d]);
                            sub_n_repeated  = std::get<2>(deps[d]);

                            // Check sub key
                            #ifndef NDEBUG
                            if (propagator.find(sub_dep) == propagator.end())
                                throw_with_line_number("Could not find sub key '" + sub_dep + "'. ");
                            if (!propagator_finished[sub_dep][sub_n_segment])
                                throw_with_line_number("Could not compute '" + key +  "', since '"+ sub_dep + std::to_string(sub_n_segment) + "' is not prepared.");
                            #endif

                            // STREAM 1: copy memory from host to device
                            if (d < deps.size()-1)
                            {
                                std::string sub_dep_next = std::get<0>(deps[d+1]);
                                int sub_n_segment_next   = std::get<1>(deps[d+1]);

                                gpu_error_check(cudaMemcpyAsync(d_propagator_sub_dep[next],
                                                propagator[sub_dep_next][sub_n_segment_next], sizeof(double)*M,
                                                cudaMemcpyHostToDevice, streams[0][1]));
                            }

                            // STREAM 0: compute linear combination
                            lin_comb<<<N_BLOCKS, N_THREADS, 0, streams[0][0]>>>(
                                    d_q_one[0][0], 1.0, d_q_one[0][0],
                                    sub_n_repeated, d_propagator_sub_dep[prev], M);

                            std::swap(prev, next);
                            cudaDeviceSynchronize();
                        }
                        gpu_error_check(cudaMemcpy(_propagator_0, d_q_one[0][0], sizeof(double)*M, cudaMemcpyDeviceToHost));
                        
                        #ifndef NDEBUG
                        propagator_finished[key][0] = true;
                        #endif
                    }
                    else
                    {
                        // Initialize to one
                        gpu_error_check(cudaMemcpy(d_q_one[0][0], d_q_unity, sizeof(double)*M, cudaMemcpyDeviceToDevice));

                        int prev, next;
                        prev = 0;
                        next = 1;

                        // Copy memory from host to device
                        std::string sub_dep = std::get<0>(deps[0]);
                        int sub_n_segment   = std::get<1>(deps[0]);
                        gpu_error_check(cudaMemcpy(d_propagator_sub_dep[prev], propagator[sub_dep][sub_n_segment], sizeof(double)*M, cudaMemcpyHostToDevice));

                        for(size_t d=0; d<deps.size(); d++)
                        {
                            std::string sub_dep = std::get<0>(deps[d]);
                            int sub_n_segment   = std::get<1>(deps[d]);

                            // Check sub key
                            #ifndef NDEBUG
                            if (propagator.find(sub_dep) == propagator.end())
                                throw_with_line_number("Could not find sub key '" + sub_dep + "'. ");
                            if (!propagator_finished[sub_dep][sub_n_segment])
                                throw_with_line_number("Could not compute '" + key +  "', since '"+ sub_dep + std::to_string(sub_n_segment) + "' is not prepared.");
                            #endif

                            // STREAM 1: copy memory from host to device
                            if (d < deps.size()-1)
                            {
                                std::string sub_dep_next = std::get<0>(deps[d+1]);
                                int sub_n_segment_next   = std::get<1>(deps[d+1]);

                                gpu_error_check(cudaMemcpyAsync(d_propagator_sub_dep[next],
                                                propagator[sub_dep_next][sub_n_segment_next], sizeof(double)*M,
                                                cudaMemcpyHostToDevice, streams[0][1]));
                            }

                            // STREAM 0: multiply 
                            multi_real<<<N_BLOCKS, N_THREADS, 0, streams[0][0]>>>(
                                d_q_one[0][0], d_q_one[0][0], d_propagator_sub_dep[prev], 1.0, M);

                            std::swap(prev, next);
                            cudaDeviceSynchronize();
                        }
                        gpu_error_check(cudaMemcpy(_propagator_0, d_q_one[0][0], sizeof(double)*M, cudaMemcpyDeviceToHost));
                        #ifndef NDEBUG
                        propagator_finished[key][0] = true;
                        #endif
                    }
                }
                cudaDeviceSynchronize();
            }
            // Synchronize all GPUs
            for(int gpu=0; gpu<N_GPUS; gpu++)
            {
                gpu_error_check(cudaSetDevice(gpu));
                gpu_error_check(cudaDeviceSynchronize());
            }

            // Copy jobs that have non-zero segments
            std::vector<std::tuple<std::string, int, int>> non_zero_segment_jobs;
            for (auto it = parallel_job->begin(); it != parallel_job->end(); it++)
            {
                int n_segment_from = std::get<1>(*it);
                int n_segment_to = std::get<2>(*it);
                if(n_segment_to-n_segment_from >= 0)
                    non_zero_segment_jobs.push_back(*it);
            }

            // Advance propagator successively
            if(N_GPUS > 1 && non_zero_segment_jobs.size() == 2)
            {
                const int N_JOBS = non_zero_segment_jobs.size();
                std::string keys[N_JOBS];
                int n_segment_froms[N_JOBS];
                int n_segment_tos[N_JOBS];
                std::string monomer_types[N_JOBS];
                double **_propagator_keys[N_JOBS];
                
                for(int j=0; j<N_JOBS; j++)
                {
                    keys[j] = std::get<0>(non_zero_segment_jobs[j]);
                    n_segment_froms[j] = std::get<1>(non_zero_segment_jobs[j]);
                    n_segment_tos[j] = std::get<2>(non_zero_segment_jobs[j]);
                    monomer_types[j] = molecules->get_essential_propagator_code(keys[j]).monomer_type;
                    _propagator_keys[j] = propagator[keys[j]];
                }

                int prev, next;
                prev = 0;
                next = 1;

                // Copy propagators from host to device
                for(int gpu=0; gpu<N_GPUS; gpu++)
                    gpu_error_check(cudaMemcpy(d_q_one[gpu][prev], _propagator_keys[gpu][n_segment_froms[gpu]-1], sizeof(double)*M,
                        cudaMemcpyHostToDevice));

                for(int n=0; n<=n_segment_tos[0]-n_segment_froms[0]; n++)
                {
                    #ifndef NDEBUG
                    for(int gpu=0; gpu<N_GPUS; gpu++)
                    {
                        if (!propagator_finished[keys[gpu]][n-1+n_segment_froms[gpu]])
                            throw_with_line_number("unfinished, key: " + keys[gpu] + ", " + std::to_string(n-1+n_segment_froms[gpu]));
                    }
                    #endif

                    // DEVICE 0,1, STREAM 0: calculate propagators 
                    advance_two_propagators_two_gpus(
                        d_q_one[0][prev],
                        d_q_one[1][prev],
                        d_q_one[0][next],
                        d_q_one[1][next],
                        d_boltz_bond[0][monomer_types[0]],
                        d_boltz_bond[1][monomer_types[1]],
                        d_boltz_bond_half[0][monomer_types[0]],
                        d_boltz_bond_half[1][monomer_types[1]],
                        d_exp_dw[0][monomer_types[0]],
                        d_exp_dw[1][monomer_types[1]],
                        d_exp_dw_half[0][monomer_types[0]],
                        d_exp_dw_half[1][monomer_types[1]]);

                    // STREAM 1: copy propagators from device to host
                    for(int gpu=0; gpu<N_GPUS; gpu++)
                    {
                        gpu_error_check(cudaSetDevice(gpu));
                        if (n > 0)
                        {
                            gpu_error_check(cudaMemcpyAsync(_propagator_keys[gpu][n-1+n_segment_froms[gpu]], d_q_one[gpu][prev], sizeof(double)*M,
                                cudaMemcpyDeviceToHost, streams[gpu][1]));
                        }
                    }

                    // Synchronize all GPUs
                    for(int gpu=0; gpu<N_GPUS; gpu++)
                    {
                        gpu_error_check(cudaSetDevice(gpu));
                        gpu_error_check(cudaDeviceSynchronize());
                    }
                    std::swap(prev, next);

                    #ifndef NDEBUG
                    for(int gpu=0; gpu<N_GPUS; gpu++)
                        propagator_finished[keys[gpu]][n+n_segment_froms[gpu]] = true;
                    #endif
                }
                for(int gpu=0; gpu<N_GPUS; gpu++)
                {
                    // Copy propagators from device to host
                    gpu_error_check(cudaMemcpy(_propagator_keys[gpu][n_segment_tos[gpu]], d_q_one[gpu][prev], sizeof(double)*M,
                        cudaMemcpyDeviceToHost));
                }
            }
            else if(non_zero_segment_jobs.size() > 0)
            {
                const int N_JOBS = non_zero_segment_jobs.size();
                std::string keys[N_JOBS];
                int n_segment_froms[N_JOBS];
                int n_segment_tos[N_JOBS];
                std::string monomer_types[N_JOBS];
                double **_propagator_keys[N_JOBS];
                
                for(int j=0; j<N_JOBS; j++)
                {
                    keys[j] = std::get<0>(non_zero_segment_jobs[j]);
                    n_segment_froms[j] = std::get<1>(non_zero_segment_jobs[j]);
                    n_segment_tos[j] = std::get<2>(non_zero_segment_jobs[j]);
                    monomer_types[j] = molecules->get_essential_propagator_code(keys[j]).monomer_type;
                    _propagator_keys[j] = propagator[keys[j]];
                }
                for(int j=0; j<N_JOBS; j++)
                {
                    int prev, next;
                    prev = 0;
                    next = 1;

                    gpu_error_check(cudaSetDevice(0));
                    // Copy propagators from host to device
                    gpu_error_check(cudaMemcpy(d_q_one[0][prev], _propagator_keys[j][n_segment_froms[j]-1], sizeof(double)*M,
                        cudaMemcpyHostToDevice));

                    for(int n=n_segment_froms[j]; n<=n_segment_tos[j]; n++)
                    {
                        #ifndef NDEBUG
                        if (!propagator_finished[keys[j]][n-1])
                            throw_with_line_number("unfinished, key: " + keys[j] + ", " + std::to_string(n-1));
                        #endif

                        // STREAM 0: compute propagator
                        advance_one_propagator(0, 
                            d_q_one[0][prev],
                            d_q_one[0][next],
                            d_boltz_bond[0][monomer_types[j]],
                            d_boltz_bond_half[0][monomer_types[j]],
                            d_exp_dw[0][monomer_types[j]],
                            d_exp_dw_half[0][monomer_types[j]]);

                        // STREAM 1: copy propagators from device to host
                        if (n > n_segment_froms[j])
                        {
                            gpu_error_check(cudaMemcpyAsync(_propagator_keys[j][n-1], d_q_one[0][prev], sizeof(double)*M,
                                cudaMemcpyDeviceToHost, streams[0][1]));
                        }

                        std::swap(prev, next);
                        cudaDeviceSynchronize();

                        #ifndef NDEBUG
                        propagator_finished[keys[j]][n] = true;
                        #endif
                    }
                    // Copy propagators from device to host
                    gpu_error_check(cudaMemcpy(_propagator_keys[j][n_segment_tos[j]], d_q_one[0][prev], sizeof(double)*M,
                        cudaMemcpyDeviceToHost));
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

        // Compute total partition function of each distinct polymers
        for(const auto& segment_info: single_partition_segment)
        {
            int p                = std::get<0>(segment_info);
            double *propagator_v = std::get<1>(segment_info);
            double *propagator_u = std::get<2>(segment_info);
            int n_superposed     = std::get<3>(segment_info);

            single_partitions[p]= cb->inner_product(
                propagator_v, propagator_u)/n_superposed/cb->get_volume();
        }

        // Calculate segment concentrations
        for(const auto& block: block_phi)
        {
            const auto& key = block.first;
            int p                = std::get<0>(key);
            std::string dep_v    = std::get<1>(key);
            std::string dep_u    = std::get<2>(key);

            int n_repeated;
            int n_segment_allocated = molecules->get_essential_block(key).n_segment_allocated;
            int n_segment_offset    = molecules->get_essential_block(key).n_segment_offset;
            int n_segment_original  = molecules->get_essential_block(key).n_segment_original;

            // If there is no segment
            if(n_segment_allocated == 0)
            {
                gpu_error_check(cudaMemset(block.second, 0, sizeof(double)*M));
                continue;
            }

            // Check keys
            #ifndef NDEBUG
            if (propagator.find(dep_v) == propagator.end())
                std::cout << "Could not find dep_v key'" + dep_v + "'. " << std::endl;
            if (propagator.find(dep_u) == propagator.end())
                std::cout << "Could not find dep_u key'" + dep_u + "'. " << std::endl;
            #endif

            // Contains no '['
            if (dep_u.find('[') == std::string::npos)
                n_repeated = molecules->get_essential_block(key).v_u.size();
            else
                n_repeated = 1;

            // Normalization constant
            Polymer& pc = molecules->get_polymer(p);
            double norm = molecules->get_ds()*pc.get_volume_fraction()/pc.get_alpha()/single_partitions[p]*n_repeated;

            // Calculate phi of one block (possibly multiple blocks when using superposition)
            calculate_phi_one_block(
                block.second,       // Phi
                propagator[dep_v],  // dependency v
                propagator[dep_u],  // dependency u
                n_segment_allocated,
                n_segment_offset,
                n_segment_original,
                norm);
        }
        gpu_error_check(cudaSetDevice(0));
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

// Advance propagator using Richardson extrapolation
void CudaPseudoReduceMemoryContinuous::advance_one_propagator(const int GPU,
    double *d_q_in, double *d_q_out,
    double *d_boltz_bond, double *d_boltz_bond_half,
    double *d_exp_dw, double *d_exp_dw_half)
{
    // Overlapping computations for 1/2 step and 1/4 step
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = cb->get_n_grid();
        const int M_COMPLEX = this->n_complex_grid;

        // step 1/2: Evaluate exp(-w*ds/2) in real space
        // step 1/4: Evaluate exp(-w*ds/4) in real space
        real_multi_exp_dw_two<<<N_BLOCKS, N_THREADS, 0, streams[GPU][0]>>>(
            &d_q_step_1_two[GPU][0], d_q_in, d_exp_dw,
            &d_q_step_1_two[GPU][M], d_q_in, d_exp_dw_half, 1.0, M);

        // step 1/2: Execute a Forward FFT
        // step 1/4: Execute a Forward FFT
        cufftExecD2Z(plan_for_two[GPU], d_q_step_1_two[GPU], d_qk_in_1_two[GPU]);

        // step 1/2: Multiply exp(-k^2 ds/6)  in fourier space
        // step 1/4: Multiply exp(-k^2 ds/12) in fourier space
        complex_real_multi_bond_two<<<N_BLOCKS, N_THREADS, 0, streams[GPU][0]>>>(
            &d_qk_in_1_two[GPU][0],         d_boltz_bond,
            &d_qk_in_1_two[GPU][M_COMPLEX], d_boltz_bond_half, M_COMPLEX);

        // step 1/2: Execute a backward FFT
        // step 1/4: Execute a backward FFT
        cufftExecZ2D(plan_bak_two[GPU], d_qk_in_1_two[GPU], d_q_step_1_two[GPU]);

        // step 1/2: Evaluate exp(-w*ds/2) in real space
        // step 1/4: Evaluate exp(-w*ds/2) in real space
        real_multi_exp_dw_two<<<N_BLOCKS, N_THREADS, 0, streams[GPU][0]>>>(
            d_q_step_1_one[GPU], &d_q_step_1_two[GPU][0], d_exp_dw,
            d_q_step_2_one[GPU], &d_q_step_1_two[GPU][M], d_exp_dw, 1.0/((double)M), M);

        // step 1/4: Execute a Forward FFT
        cufftExecD2Z(plan_for_one[GPU], d_q_step_2_one[GPU], d_qk_in_2_one[GPU]);

        // step 1/4: Multiply exp(-k^2 ds/12) in fourier space
        multi_complex_real<<<N_BLOCKS, N_THREADS, 0, streams[GPU][0]>>>(d_qk_in_2_one[GPU], d_boltz_bond_half, M_COMPLEX);

        // step 1/4: Execute a backward FFT
        cufftExecZ2D(plan_bak_one[GPU], d_qk_in_2_one[GPU], d_q_step_2_one[GPU]);

        // step 1/4: Evaluate exp(-w*ds/4) in real space.
        multi_real<<<N_BLOCKS, N_THREADS, 0, streams[GPU][0]>>>(d_q_step_2_one[GPU], d_q_step_2_one[GPU], d_exp_dw_half, 1.0/((double)M), M);

        // Compute linear combination with 4/3 and -1/3 ratio
        lin_comb<<<N_BLOCKS, N_THREADS, 0, streams[GPU][0]>>>(d_q_out, 4.0/3.0, d_q_step_2_one[GPU], -1.0/3.0, d_q_step_1_one[GPU], M);
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaPseudoReduceMemoryContinuous::advance_two_propagators_two_gpus(
    double *d_q_in_1, double *d_q_in_2,
    double *d_q_out_1, double *d_q_out_2,
    double *d_boltz_bond_1, double *d_boltz_bond_2, 
    double *d_boltz_bond_half_1, double *d_boltz_bond_half_2,         
    double *d_exp_dw_1, double *d_exp_dw_2,
    double *d_exp_dw_half_1, double *d_exp_dw_half_2)
{
    // Overlapping computations for 1/2 step and 1/4 step
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();
        const int N_GPUS = CudaCommon::get_instance().get_n_gpus();

        const int M = cb->get_n_grid();
        const int M_COMPLEX = this->n_complex_grid;

        // step 1/2: Evaluate exp(-w*ds/2) in real space
        // step 1/4: Evaluate exp(-w*ds/4) in real space
        gpu_error_check(cudaSetDevice(0));
        real_multi_exp_dw_two<<<N_BLOCKS, N_THREADS, 0, streams[0][0]>>>(
            &d_q_step_1_two[0][0], d_q_in_1, d_exp_dw_1,
            &d_q_step_1_two[0][M], d_q_in_1, d_exp_dw_half_1, 1.0, M);
        gpu_error_check(cudaSetDevice(1));
        real_multi_exp_dw_two<<<N_BLOCKS, N_THREADS, 0, streams[1][0]>>>(
            &d_q_step_1_two[1][0], d_q_in_2, d_exp_dw_2,
            &d_q_step_1_two[1][M], d_q_in_2, d_exp_dw_half_2, 1.0, M);

        // step 1/2: Execute a Forward FFT
        // step 1/4: Execute a Forward FFT
        gpu_error_check(cudaSetDevice(0));
        cufftExecD2Z(plan_for_two[0], d_q_step_1_two[0], d_qk_in_1_two[0]);
        gpu_error_check(cudaSetDevice(1));
        cufftExecD2Z(plan_for_two[1], d_q_step_1_two[1], d_qk_in_1_two[1]);

        // step 1/2: Multiply exp(-k^2 ds/6)  in fourier space
        // step 1/4: Multiply exp(-k^2 ds/12) in fourier space
        gpu_error_check(cudaSetDevice(0));
        complex_real_multi_bond_two<<<N_BLOCKS, N_THREADS, 0, streams[0][0]>>>(
            &d_qk_in_1_two[0][0],         d_boltz_bond_1,
            &d_qk_in_1_two[0][M_COMPLEX], d_boltz_bond_half_1, M_COMPLEX);
        gpu_error_check(cudaSetDevice(1));
        complex_real_multi_bond_two<<<N_BLOCKS, N_THREADS, 0, streams[1][0]>>>(
            &d_qk_in_1_two[1][0],         d_boltz_bond_2,
            &d_qk_in_1_two[1][M_COMPLEX], d_boltz_bond_half_2, M_COMPLEX);

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
            d_q_step_1_one[0], &d_q_step_1_two[0][0], d_exp_dw_1,
            d_q_step_2_one[0], &d_q_step_1_two[0][M], d_exp_dw_1, 1.0/((double)M), M);
        gpu_error_check(cudaSetDevice(1));
        real_multi_exp_dw_two<<<N_BLOCKS, N_THREADS, 0, streams[1][0]>>>(
            d_q_step_1_one[1], &d_q_step_1_two[1][0], d_exp_dw_2,
            d_q_step_2_one[1], &d_q_step_1_two[1][M], d_exp_dw_2, 1.0/((double)M), M);

        // step 1/4: Execute a Forward FFT
        gpu_error_check(cudaSetDevice(0));
        cufftExecD2Z(plan_for_one[0], d_q_step_2_one[0], d_qk_in_2_one[0]);
        gpu_error_check(cudaSetDevice(1));
        cufftExecD2Z(plan_for_one[1], d_q_step_2_one[1], d_qk_in_2_one[1]);

        // step 1/4: Multiply exp(-k^2 ds/12) in fourier space
        gpu_error_check(cudaSetDevice(0));
        multi_complex_real<<<N_BLOCKS, N_THREADS, 0, streams[0][0]>>>(d_qk_in_2_one[0], d_boltz_bond_half_1, M_COMPLEX);
        gpu_error_check(cudaSetDevice(1));
        multi_complex_real<<<N_BLOCKS, N_THREADS, 0, streams[1][0]>>>(d_qk_in_2_one[1], d_boltz_bond_half_2, M_COMPLEX);

        // step 1/4: Execute a backward FFT
        gpu_error_check(cudaSetDevice(0));
        cufftExecZ2D(plan_bak_one[0], d_qk_in_2_one[0], d_q_step_2_one[0]);
        gpu_error_check(cudaSetDevice(1));
        cufftExecZ2D(plan_bak_one[1], d_qk_in_2_one[1], d_q_step_2_one[1]);

        // step 1/4: Evaluate exp(-w*ds/4) in real space.
        gpu_error_check(cudaSetDevice(0));
        multi_real<<<N_BLOCKS, N_THREADS, 0, streams[0][0]>>>(d_q_step_2_one[0], d_q_step_2_one[0], d_exp_dw_half_1, 1.0/((double)M), M);
        gpu_error_check(cudaSetDevice(1));
        multi_real<<<N_BLOCKS, N_THREADS, 0, streams[1][0]>>>(d_q_step_2_one[1], d_q_step_2_one[1], d_exp_dw_half_2, 1.0/((double)M), M);

        // Compute linear combination with 4/3 and -1/3 ratio
        gpu_error_check(cudaSetDevice(0));
        lin_comb<<<N_BLOCKS, N_THREADS, 0, streams[0][0]>>>(d_q_out_1, 4.0/3.0, d_q_step_2_one[0], -1.0/3.0, d_q_step_1_one[0], M);
        gpu_error_check(cudaSetDevice(1));
        lin_comb<<<N_BLOCKS, N_THREADS, 0, streams[1][0]>>>(d_q_out_2, 4.0/3.0, d_q_step_2_one[1], -1.0/3.0, d_q_step_1_one[1], M);
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaPseudoReduceMemoryContinuous::calculate_phi_one_block(
    double *phi, double **q_1, double **q_2, const int N, const int N_OFFSET, const int N_ORIGINAL, const double NORM)
{
    try
    {
        gpu_error_check(cudaSetDevice(0));

        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();
        const int M = cb->get_n_grid();
        std::vector<double> simpson_rule_coeff = SimpsonRule::get_coeff(N);

        int prev, next;
        prev = 0;
        next = 1;

        // Copy propagators from host to device
        gpu_error_check(cudaMemcpy(d_q_block_v[prev], q_1[N_ORIGINAL-N_OFFSET], sizeof(double)*M, cudaMemcpyHostToDevice));
        gpu_error_check(cudaMemcpy(d_q_block_u[prev], q_2[0],                   sizeof(double)*M, cudaMemcpyHostToDevice));

        // Initialize to zero
        gpu_error_check(cudaMemset(d_phi, 0, sizeof(double)*M));
 
        for(int n=0; n<=N; n++)
        {
            // STREAM 1: copy propagators from host to device
            if (n+1 <= N)
            {
                gpu_error_check(cudaMemcpyAsync(d_q_block_v[next], q_1[N_ORIGINAL-N_OFFSET-(n+1)],
                    sizeof(double)*M, cudaMemcpyHostToDevice, streams[0][1]));
                gpu_error_check(cudaMemcpyAsync(d_q_block_u[next], q_2[n+1],
                    sizeof(double)*M, cudaMemcpyHostToDevice, streams[0][1]));
            }

            // STREAM 0: multiply two propagators
            add_multi_real<<<N_BLOCKS, N_THREADS, 0, streams[0][0]>>>(d_phi, d_q_block_v[prev], d_q_block_u[prev], NORM*simpson_rule_coeff[n], M);
            std::swap(prev, next);
            cudaDeviceSynchronize();
        }
        // Copy propagators from device to host
        gpu_error_check(cudaMemcpy(phi, d_phi, sizeof(double)*M, cudaMemcpyDeviceToHost));
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
double CudaPseudoReduceMemoryContinuous::get_total_partition(int polymer)
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
void CudaPseudoReduceMemoryContinuous::get_total_concentration(std::string monomer_type, double *phi)
{
    try
    {
        const int M = cb->get_n_grid();
        // Initialize array
        for(int i=0; i<M; i++)
            phi[i] = 0.0;

        // For each block
        for(const auto& block: block_phi)
        {
            std::string dep_v = std::get<1>(block.first);
            int n_segment_allocated = molecules->get_essential_block(block.first).n_segment_allocated;
            if (Molecules::get_monomer_type_from_key(dep_v) == monomer_type && n_segment_allocated != 0)
            {
                for(int i=0; i<M; i++)
                    phi[i] += block.second[i]; 
            }
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaPseudoReduceMemoryContinuous::get_total_concentration(int p, std::string monomer_type, double *phi)
{
    try
    {
        const int M = cb->get_n_grid();
        const int P = molecules->get_n_polymer_types();

        if (p < 0 || p > P-1)
            throw_with_line_number("Index (" + std::to_string(p) + ") must be in range [0, " + std::to_string(P-1) + "]");

        // Initialize array
        for(int i=0; i<M; i++)
            phi[i] = 0.0;

        // For each block
        for(const auto& block: block_phi)
        {
            int polymer_idx = std::get<0>(block.first);
            std::string dep_v = std::get<1>(block.first);
            int n_segment_allocated = molecules->get_essential_block(block.first).n_segment_allocated;
            if (polymer_idx == p && Molecules::get_monomer_type_from_key(dep_v) == monomer_type && n_segment_allocated != 0)
            {
                for(int i=0; i<M; i++)
                    phi[i] += block.second[i]; 
            }
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaPseudoReduceMemoryContinuous::get_block_concentration(int p, double *phi)
{
    try
    {
        const int M = cb->get_n_grid();
        const int P = molecules->get_n_polymer_types();

        if (p < 0 || p > P-1)
            throw_with_line_number("Index (" + std::to_string(p) + ") must be in range [0, " + std::to_string(P-1) + "]");

        if (molecules->is_using_superposition())
            throw_with_line_number("Disable 'superposition' option to invoke 'get_block_concentration'.");

        Polymer& pc = molecules->get_polymer(p);
        std::vector<Block>& blocks = pc.get_blocks();

        for(size_t b=0; b<blocks.size(); b++)
        {
            std::string dep_v = pc.get_propagator_key(blocks[b].v, blocks[b].u);
            std::string dep_u = pc.get_propagator_key(blocks[b].u, blocks[b].v);
            if (dep_v < dep_u)
                dep_v.swap(dep_u);

            double* _essential_block_phi = block_phi[std::make_tuple(p, dep_v, dep_u)];
            for(int i=0; i<M; i++)
                phi[i+b*M] = _essential_block_phi[i]; 
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
std::vector<double> CudaPseudoReduceMemoryContinuous::compute_stress()
{
    // This method should be invoked after invoking compute_statistics().

    // To calculate stress, we multiply weighted fourier basis to q(k)*q^dagger(-k).
    // We only need the real part of stress calculation.

    try{
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();
        const int N_GPUS = CudaCommon::get_instance().get_n_gpus();

        const int DIM  = cb->get_dim();
        const int M    = cb->get_n_grid();
        const int M_COMPLEX = this->n_complex_grid;

        auto bond_lengths = molecules->get_bond_lengths();
        std::vector<double> stress(DIM);
        std::map<std::tuple<int, std::string, std::string>, std::array<double,3>> block_dq_dl[MAX_GPUS];
        double stress_sum_out[MAX_GPUS][3];

        // Compute stress for each block
        for(const auto& block: block_phi)
        {
            const auto& key = block.first;
            int p                = std::get<0>(key);
            std::string dep_v    = std::get<1>(key);
            std::string dep_u    = std::get<2>(key);

            const int N           = molecules->get_essential_block(key).n_segment_allocated;
            const int N_OFFSET    = molecules->get_essential_block(key).n_segment_offset;
            const int N_ORIGINAL  = molecules->get_essential_block(key).n_segment_original;
            std::string monomer_type = molecules->get_essential_block(key).monomer_type;

            // If there is no segment
            if(N == 0)
                continue;

            // Contains no '['
            int n_repeated;
            if (dep_u.find('[') == std::string::npos)
                n_repeated = molecules->get_essential_block(key).v_u.size();
            else
                n_repeated = 1;

            std::vector<double> s_coeff = SimpsonRule::get_coeff(N);
            double bond_length_sq = bond_lengths[monomer_type]*bond_lengths[monomer_type];
            double** q_1 = propagator[dep_v];    // dependency v
            double** q_2 = propagator[dep_u];    // dependency u

            std::array<double,3> _block_dq_dl[MAX_GPUS];
            for(int gpu=0; gpu<N_GPUS; gpu++)
                for(int d=0; d<3; d++)
                    _block_dq_dl[gpu][d] = 0.0;

            int prev, next;
            prev = 0;
            next = 1;

            // Copy memory from host to device
            for(int gpu=0; gpu<N_GPUS; gpu++)
            {
                // Index
                int idx = gpu;
                if (idx <= N)
                {
                    gpu_error_check(cudaSetDevice(gpu));
                    gpu_error_check(cudaMemcpy(&d_stress_q[gpu][prev][0], q_1[N_ORIGINAL-N_OFFSET-idx],
                            sizeof(double)*M,cudaMemcpyHostToDevice));
                    gpu_error_check(cudaMemcpy(&d_stress_q[gpu][prev][M], q_2[idx],
                            sizeof(double)*M,cudaMemcpyHostToDevice));
                }
            }

            // Compute
            for(int n=0; n<=N; n+=N_GPUS)
            {
                // STREAM 1: copy memory from host to device
                for(int gpu=0; gpu<N_GPUS; gpu++)
                {
                    // Index
                    const int idx = n + gpu;
                    const int idx_next = idx + N_GPUS;

                    gpu_error_check(cudaSetDevice(gpu));
                    if (idx_next <= N)
                    {
                        gpu_error_check(cudaMemcpyAsync(&d_stress_q[gpu][next][0], q_1[N_ORIGINAL-N_OFFSET-idx_next],
                                sizeof(double)*M,cudaMemcpyHostToDevice, streams[gpu][1]));
                        gpu_error_check(cudaMemcpyAsync(&d_stress_q[gpu][next][M], q_2[idx_next],
                                sizeof(double)*M,cudaMemcpyHostToDevice, streams[gpu][1]));
                    }
                }
                // STREAM 0: execute kernels
                // Execute a Forward FFT
                for(int gpu=0; gpu<N_GPUS; gpu++)
                {
                    const int idx = n + gpu;
                    gpu_error_check(cudaSetDevice(gpu));
                    if (idx <= N)
                        cufftExecD2Z(plan_for_two[gpu], d_stress_q[gpu][prev], d_qk_in_1_two[gpu]);
                }
                // Multiply two propagators in the fourier spaces
                for(int gpu=0; gpu<N_GPUS; gpu++)
                {
                    const int idx = n + gpu;
                    gpu_error_check(cudaSetDevice(gpu));
                    if (idx <= N)
                    {
                        multi_complex_conjugate<<<N_BLOCKS, N_THREADS, 0, streams[gpu][0]>>>(d_q_multi[gpu], &d_qk_in_1_two[gpu][0], &d_qk_in_1_two[gpu][M_COMPLEX], M_COMPLEX);
                        if ( DIM == 3 )
                        {
                            // x direction
                            multi_real<<<N_BLOCKS, N_THREADS, 0, streams[gpu][0]>>>(d_stress_sum[gpu], d_q_multi[gpu], d_fourier_basis_x[gpu], bond_length_sq, M_COMPLEX);
                            cub::DeviceReduce::Sum(d_temp_storage[gpu], temp_storage_bytes[gpu], d_stress_sum[gpu], d_stress_sum_out[gpu], M_COMPLEX, streams[gpu][0]);
                            gpu_error_check(cudaMemcpyAsync(&stress_sum_out[gpu][0],d_stress_sum_out[gpu],sizeof(double),cudaMemcpyDeviceToHost, streams[gpu][0]));

                            // y direction
                            multi_real<<<N_BLOCKS, N_THREADS, 0, streams[gpu][0]>>>(d_stress_sum[gpu], d_q_multi[gpu], d_fourier_basis_y[gpu], bond_length_sq, M_COMPLEX);
                            cub::DeviceReduce::Sum(d_temp_storage[gpu], temp_storage_bytes[gpu], d_stress_sum[gpu], d_stress_sum_out[gpu], M_COMPLEX, streams[gpu][0]);
                            gpu_error_check(cudaMemcpyAsync(&stress_sum_out[gpu][1],d_stress_sum_out[gpu],sizeof(double),cudaMemcpyDeviceToHost, streams[gpu][0]));

                            // z direction
                            multi_real<<<N_BLOCKS, N_THREADS, 0, streams[gpu][0]>>>(d_stress_sum[gpu], d_q_multi[gpu], d_fourier_basis_z[gpu], bond_length_sq, M_COMPLEX);
                            cub::DeviceReduce::Sum(d_temp_storage[gpu], temp_storage_bytes[gpu], d_stress_sum[gpu], d_stress_sum_out[gpu], M_COMPLEX, streams[gpu][0]);
                            gpu_error_check(cudaMemcpyAsync(&stress_sum_out[gpu][2],d_stress_sum_out[gpu],sizeof(double),cudaMemcpyDeviceToHost, streams[gpu][0]));
                        }
                        if ( DIM == 2 )
                        {
                            // y direction
                            multi_real<<<N_BLOCKS, N_THREADS, 0, streams[gpu][0]>>>(d_stress_sum[gpu], d_q_multi[gpu], d_fourier_basis_y[gpu], bond_length_sq, M_COMPLEX);
                            cub::DeviceReduce::Sum(d_temp_storage[gpu], temp_storage_bytes[gpu], d_stress_sum[gpu], d_stress_sum_out[gpu], M_COMPLEX, streams[gpu][0]);
                            gpu_error_check(cudaMemcpyAsync(&stress_sum_out[gpu][0],d_stress_sum_out[gpu],sizeof(double),cudaMemcpyDeviceToHost, streams[gpu][0]));

                            // z direction
                            multi_real<<<N_BLOCKS, N_THREADS, 0, streams[gpu][0]>>>(d_stress_sum[gpu], d_q_multi[gpu], d_fourier_basis_z[gpu], bond_length_sq, M_COMPLEX);
                            cub::DeviceReduce::Sum(d_temp_storage[gpu], temp_storage_bytes[gpu], d_stress_sum[gpu], d_stress_sum_out[gpu], M_COMPLEX, streams[gpu][0]);
                            gpu_error_check(cudaMemcpyAsync(&stress_sum_out[gpu][1],d_stress_sum_out[gpu],sizeof(double),cudaMemcpyDeviceToHost, streams[gpu][0]));
                        }
                        if ( DIM == 1 )
                        {
                            // z direction
                            multi_real<<<N_BLOCKS, N_THREADS, 0, streams[gpu][0]>>>(d_stress_sum[gpu], d_q_multi[gpu], d_fourier_basis_z[gpu], bond_length_sq, M_COMPLEX);
                            cub::DeviceReduce::Sum(d_temp_storage[gpu], temp_storage_bytes[gpu], d_stress_sum[gpu], d_stress_sum_out[gpu], M_COMPLEX, streams[gpu][0]);
                            gpu_error_check(cudaMemcpyAsync(&stress_sum_out[gpu][0],d_stress_sum_out[gpu],sizeof(double),cudaMemcpyDeviceToHost, streams[gpu][0]));
                        }
                        // Synchronize streams and add results
                        gpu_error_check(cudaStreamSynchronize(streams[gpu][0]));
                        for(int d=0; d<DIM; d++)
                            _block_dq_dl[gpu][d] += s_coeff[idx]*stress_sum_out[gpu][d]*n_repeated;
                    }
                }
                // Synchronize all GPUs
                for(int gpu=0; gpu<N_GPUS; gpu++)
                {
                    gpu_error_check(cudaSetDevice(gpu));
                    gpu_error_check(cudaDeviceSynchronize());
                }
                std::swap(prev, next);
            }
            // Copy stress data
            for(int gpu=0; gpu<N_GPUS; gpu++)
                block_dq_dl[gpu][key] = _block_dq_dl[gpu];
        }
        gpu_error_check(cudaSetDevice(0));

        // Compute total stress
        for(int d=0; d<DIM; d++)
            stress[d] = 0.0;
        for(const auto& block: block_phi)
        {
            const auto& key = block.first;
            int p             = std::get<0>(key);
            std::string dep_v = std::get<1>(key);
            std::string dep_u = std::get<2>(key);
            Polymer& pc  = molecules->get_polymer(p);

            for(int gpu=0; gpu<N_GPUS; gpu++)
                for(int d=0; d<DIM; d++)
                    stress[d] += block_dq_dl[gpu][key][d]*pc.get_volume_fraction()/pc.get_alpha()/single_partitions[p];
        }
        for(int d=0; d<DIM; d++)
            stress[d] /= -3.0*cb->get_lx(d)*M*M/molecules->get_ds();
            
        return stress;
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaPseudoReduceMemoryContinuous::get_chain_propagator(double *q_out, int polymer, int v, int u, int n)
{
    // This method should be invoked after invoking compute_statistics()

    // Get chain propagator for a selected polymer, block and direction.
    // This is made for debugging and testing.
    try
    {
        const int M = cb->get_n_grid();
        Polymer& pc = molecules->get_polymer(polymer);
        std::string dep = pc.get_propagator_key(v,u);

        if (molecules->get_essential_propagator_codes().find(dep) == molecules->get_essential_propagator_codes().end())
            throw_with_line_number("Could not find the propagator code '" + dep + "'. Disable 'superposition' option to obtain propagators.");

        const int N = molecules->get_essential_propagator_codes()[dep].max_n_segment;
        if (n < 0 || n > N)
            throw_with_line_number("n (" + std::to_string(n) + ") must be in range [0, " + std::to_string(N) + "]");

        double* _propagator = propagator[dep][n];
        for(int i=0; i<M; i++)
            q_out[i] = _propagator[i];
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}