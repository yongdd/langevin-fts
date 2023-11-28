#include <complex>
#include <thrust/reduce.h>
#include <iostream>
#include "CudaPseudoDiscrete.h"
#include "CudaComputationBox.h"

CudaPseudoDiscrete::CudaPseudoDiscrete(
    ComputationBox *cb,
    Molecules *molecules)
    : Pseudo(cb, molecules)
{
    try
    {
        const int M = cb->get_n_grid();
        const int M_COMPLEX = this->n_complex_grid;
        const int N_GPUS = CudaCommon::get_instance().get_n_gpus();

        // Allocate memory for propagators
        gpu_error_check(cudaSetDevice(0));
        if( molecules->get_essential_propagator_codes().size() == 0)
            throw_with_line_number("There is no propagator code. Add polymers first.");
        for(const auto& item: molecules->get_essential_propagator_codes())
        {
             // There are N segments

             // Example (N==5)
             // O--O--O--O--O
             // 0  1  2  3  4

             // Legend)
             // -- : full bond
             // O  : full segment

            std::string key = item.first;
            int max_n_segment = item.second.max_n_segment;

            propagator_size[key] = max_n_segment;
            d_propagator[key] = new double*[max_n_segment];
            for(int i=0; i<propagator_size[key]; i++)
                gpu_error_check(cudaMalloc((void**)&d_propagator[key][i], sizeof(double)*M));

            #ifndef NDEBUG
            propagator_finished[key] = new bool[max_n_segment];
            for(int i=0; i<max_n_segment;i++)
                propagator_finished[key][i] = false;
            #endif
        }

        // Allocate memory for propagator_junction, which contain propagator at junction of discrete chain
        for(const auto& item: molecules->get_essential_propagator_codes())
        {
            std::string key = item.first;
            d_propagator_junction[key] = nullptr;
            gpu_error_check(cudaMalloc((void**)&d_propagator_junction[key], sizeof(double)*M));
        }

        // Allocate memory for concentrations
        if( molecules->get_essential_blocks().size() == 0)
            throw_with_line_number("There is no block. Add polymers first.");
        for(const auto& item: molecules->get_essential_blocks())
        {
            d_block_phi[item.first] = nullptr;
            gpu_error_check(cudaMalloc((void**)&d_block_phi[item.first], sizeof(double)*M));
        }

        // Create boltz_bond, boltz_bond_half, and exp_dw
        for(const auto& item: molecules->get_bond_lengths())
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

        // Total partition functions for each polymer
        single_partitions = new double[molecules->get_n_polymer_types()];

        // Remember one segment for each polymer chain to compute total partition function
        int current_p = 0;
        for(const auto& d_block: d_block_phi)
        {
            const auto& key = d_block.first;
            int p                = std::get<0>(key);
            std::string dep_v    = std::get<1>(key);
            std::string dep_u    = std::get<2>(key);

            // Skip if already found one segment
            if (p != current_p)
                continue;

            int n_superposed;
            int n_segment_offset    = molecules->get_essential_block(key).n_segment_offset;
            int n_segment_original  = molecules->get_essential_block(key).n_segment_original;
            std::string monomer_type = molecules->get_essential_block(key).monomer_type;

            // Contains no '['
            if (dep_u.find('[') == std::string::npos)
                n_superposed = 1;
            else
                n_superposed = molecules->get_essential_block(key).v_u.size();

            single_partition_segment.push_back(std::make_tuple(
                p,
                d_propagator[dep_v][n_segment_original-n_segment_offset-1],  // q
                d_propagator[dep_u][0],                                      // Q_dagger
                monomer_type,       
                n_superposed                   // How many propagators are aggregated
                ));
            current_p++;
        }

       // Find propagators and bond length for each segment to prepare stress computation
        for(const auto& block: d_block_phi)
        {
            const auto& key = block.first;
            int p                = std::get<0>(key);
            std::string dep_v    = std::get<1>(key);
            std::string dep_u    = std::get<2>(key);

            const int N           = molecules->get_essential_block(key).n_segment_allocated;
            const int N_OFFSET    = molecules->get_essential_block(key).n_segment_offset;
            const int N_ORIGINAL  = molecules->get_essential_block(key).n_segment_original;

            double **d_q_1 = d_propagator[dep_v];    // dependency v
            double **d_q_2 = d_propagator[dep_u];    // dependency u

            auto& _block_stress_info_key = block_stress_info[key];

            // Find propagators and bond length
            for(int n=0; n<=N; n++)
            {
                double *d_propagator_v = nullptr;
                double *d_propagator_u = nullptr;
                bool is_half_bond_length = false;

                // At v
                if (n + N_OFFSET == N_ORIGINAL)
                {
                    if (molecules->get_essential_propagator_code(dep_v).deps.size() == 0) // if v is leaf node, skip
                    {
                        _block_stress_info_key.push_back(std::make_tuple(d_propagator_v, d_propagator_u, is_half_bond_length));
                        continue;
                    }
                    
                    d_propagator_v = d_propagator_junction[dep_v];
                    d_propagator_u = d_q_2[N-1];
                    is_half_bond_length = true;
                }
                // At u
                else if (n + N_OFFSET == 0){
                    if (molecules->get_essential_propagator_code(dep_u).deps.size() == 0) // if u is leaf node, skip
                    {
                        _block_stress_info_key.push_back(std::make_tuple(d_propagator_v, d_propagator_u, is_half_bond_length));
                        continue;
                    }

                    d_propagator_v = d_q_1[N_ORIGINAL-1];
                    d_propagator_u = d_propagator_junction[dep_u];
                    is_half_bond_length = true;
                }
                // At superposition junction
                else if (n == 0)
                {
                    _block_stress_info_key.push_back(std::make_tuple(d_propagator_v, d_propagator_u, is_half_bond_length));
                    continue;
                }
                // Within the blocks
                else
                {
                    d_propagator_v = d_q_1[N_ORIGINAL-N_OFFSET-n-1];
                    d_propagator_u = d_q_2[n-1];
                    is_half_bond_length = false;
                }
                _block_stress_info_key.push_back(std::make_tuple(d_propagator_v, d_propagator_u, is_half_bond_length));
            }
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

        // Allocate memory for pseudo-spectral: advance_propagator()
        for(int gpu=0; gpu<N_GPUS; gpu++)
        {
            gpu_error_check(cudaSetDevice(gpu));
            gpu_error_check(cudaMalloc((void**)&d_q_step_1_two[gpu], sizeof(double)*2*M));
            gpu_error_check(cudaMalloc((void**)&d_qk_in_1_one[gpu], sizeof(ftsComplex)*M_COMPLEX));
            gpu_error_check(cudaMalloc((void**)&d_qk_in_1_two[gpu], sizeof(ftsComplex)*2*M_COMPLEX));
        }
        if (N_GPUS > 1)
        {
            gpu_error_check(cudaSetDevice(1));
            gpu_error_check(cudaMalloc((void**)&d_propagator_device_1[0], sizeof(double)*M));  // prev
            gpu_error_check(cudaMalloc((void**)&d_propagator_device_1[1], sizeof(double)*M));  // next
        }

        gpu_error_check(cudaSetDevice(0));
        gpu_error_check(cudaMalloc((void**)&d_q_half_step, sizeof(double)*M));
        gpu_error_check(cudaMalloc((void**)&d_q_junction,  sizeof(double)*M));
        gpu_error_check(cudaMalloc((void**)&d_phi, sizeof(double)*M));

        double q_unity[M];
        for(int i=0; i<M; i++)
            q_unity[i] = 1.0;
        gpu_error_check(cudaMalloc((void**)&d_q_unity, sizeof(double)*M));
        gpu_error_check(cudaMemcpy(d_q_unity, q_unity, sizeof(double)*M, cudaMemcpyHostToDevice));

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
CudaPseudoDiscrete::~CudaPseudoDiscrete()
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
    }

    for(const auto& item: d_propagator)
    {
        for(int i=0; i<propagator_size[item.first]; i++)
            cudaFree(item.second[i]);
        delete[] item.second;
    }
    for(const auto& item: d_block_phi)
        cudaFree(item.second);
    for(const auto& item: d_propagator_junction)
        cudaFree(item.second);

    #ifndef NDEBUG
    for(const auto& item: propagator_finished)
        delete[] item.second;
    #endif

    // For pseudo-spectral: advance_propagator()
    for(int gpu=0; gpu<N_GPUS; gpu++)
    {
        cudaFree(d_q_step_1_two[gpu]);
        cudaFree(d_qk_in_1_one[gpu]);
        cudaFree(d_qk_in_1_two[gpu]);
    }

    if (N_GPUS > 1)
    {
        cudaFree(d_propagator_device_1[0]);
        cudaFree(d_propagator_device_1[1]);
    }

    cudaFree(d_phi);
    cudaFree(d_q_unity);
    cudaFree(d_q_half_step);
    cudaFree(d_q_junction);

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

    // Destroy streams
    for(int gpu=0; gpu<N_GPUS; gpu++)
    {
        cudaStreamDestroy(streams[gpu][0]);
        cudaStreamDestroy(streams[gpu][1]);
    }
}

void CudaPseudoDiscrete::update_bond_function()
{
    try
    {
        // For pseudo-spectral: advance_propagator()
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
        throw_without_line_number(exc.what());
    }
}
void CudaPseudoDiscrete::compute_statistics(
    std::map<std::string, const double*> w_input,
    std::map<std::string, const double*> q_init, std::string device)
{
    try
    {
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

        // Compute exp_dw
        for(const auto& item: w_input)
        {
            std::string monomer_type = item.first;
            const double *w = item.second;

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
                    (d_exp_dw[gpu][monomer_type], d_exp_dw[gpu][monomer_type], 1.0, -1*ds, M);
                
            }
            // Synchronize all GPUs
            for(int gpu=0; gpu<N_GPUS; gpu++)
            {
                gpu_error_check(cudaSetDevice(gpu));
                gpu_error_check(cudaDeviceSynchronize());
            }
        }

        // For each time span
        auto& branch_schedule = sc->get_schedule();
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
                if (d_propagator.find(key) == d_propagator.end())
                    throw_with_line_number("Could not find key '" + key + "'. ");
                #endif
                double **_d_propagator = d_propagator[key];

                // Calculate one block end
                if(n_segment_from == 1 && deps.size() == 0) // if it is leaf node
                {
                     // Q_init
                    if (key[0] == '{')
                    {
                        std::string g = Molecules::get_q_input_idx_from_key(key);
                        if (q_init.find(g) == q_init.end())
                            throw_with_line_number("Could not find q_init[\"" + g + "\"].");
                        gpu_error_check(cudaMemcpy(_d_propagator[0], q_init[g], sizeof(double)*M, cudaMemcpyInputToDevice));
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
                // If it is not leaf node
                else if (n_segment_from == 1 && deps.size() > 0)
                {
                    // If it is superposed
                    if (key[0] == '[')
                    {
                        // Initialize to zero
                        gpu_error_check(cudaMemset(_d_propagator[0], 0, sizeof(double)*M));

                        for(size_t d=0; d<deps.size(); d++)
                        {
                            std::string sub_dep = std::get<0>(deps[d]);
                            int sub_n_segment   = std::get<1>(deps[d]);
                            int sub_n_repeated  = std::get<2>(deps[d]);

                            // Check sub key
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
                        advance_one_propagator(0,
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

                        // Combine branches
                        gpu_error_check(cudaMemcpy(d_q_junction, d_q_unity, sizeof(double)*M, cudaMemcpyDeviceToDevice));

                        for(size_t d=0; d<deps.size(); d++)
                        {
                            std::string sub_dep = std::get<0>(deps[d]);
                            int sub_n_segment   = std::get<1>(deps[d]);

                            // Check sub key
                            #ifndef NDEBUG
                            if (d_propagator.find(sub_dep) == d_propagator.end())
                                throw_with_line_number("Could not find sub key '" + sub_dep + "'. ");
                            if (!propagator_finished[sub_dep][sub_n_segment-1])
                                throw_with_line_number("Could not compute '" + key +  "', since '"+ sub_dep + std::to_string(sub_n_segment) + "' is not prepared.");
                            #endif

                            advance_propagator_half_bond_step(0,
                                d_propagator[sub_dep][sub_n_segment-1],
                                d_q_half_step, d_boltz_bond_half[0][molecules->get_essential_propagator_code(sub_dep).monomer_type]);

                            multi_real<<<N_BLOCKS, N_THREADS>>>(d_q_junction, d_q_junction, d_q_half_step, 1.0, M);
                        }
                        gpu_error_check(cudaMemcpy(d_propagator_junction[key], d_q_junction, sizeof(double)*M, cudaMemcpyDeviceToDevice));

                        // Add half bond
                        advance_propagator_half_bond_step(0, d_q_junction, _d_propagator[0], d_boltz_bond_half[0][monomer_type]);

                        // Add full segment
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

                    advance_one_propagator(0, 
                        _d_propagator[n-1],
                        _d_propagator[n],
                        d_boltz_bond[0][monomer_type],
                        d_exp_dw[0][monomer_type]);

                    #ifndef NDEBUG
                    propagator_finished[key][n] = true;
                    #endif
                }
                cudaDeviceSynchronize();
            }

            // Synchronize all GPUs
            for(int gpu=0; gpu<N_GPUS; gpu++)
            {
                gpu_error_check(cudaSetDevice(gpu));
                gpu_error_check(cudaDeviceSynchronize());
            }

            // Advance propagator successively
            if(parallel_job->size()==1)
            {
                gpu_error_check(cudaSetDevice(0));
                auto& key = std::get<0>((*parallel_job)[0]);
                int n_segment_from = std::get<1>((*parallel_job)[0]);
                int n_segment_to = std::get<2>((*parallel_job)[0]);
                auto monomer_type = molecules->get_essential_propagator_code(key).monomer_type;
                double **_d_propagator_key = d_propagator[key];

                for(int n=n_segment_from; n<n_segment_to; n++)
                {
                    #ifndef NDEBUG
                    if (!propagator_finished[key][n-1])
                        throw_with_line_number("unfinished, key: " + key + ", " + std::to_string(n-1));
                    #endif

                    advance_one_propagator(0, 
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

                const int N_JOBS = 2;
                std::string keys[N_JOBS];
                int n_segment_froms[N_JOBS];
                int n_segment_tos[N_JOBS];
                std::string monomer_types[N_JOBS];
                double **_d_propagator_keys[N_JOBS];
                
                for(int j=0; j<N_JOBS; j++)
                {
                    keys[j] = std::get<0>((*parallel_job)[j]);
                    n_segment_froms[j] = std::get<1>((*parallel_job)[j]);
                    n_segment_tos[j] = std::get<2>((*parallel_job)[j]);
                    monomer_types[j] = molecules->get_essential_propagator_code(keys[j]).monomer_type;
                    _d_propagator_keys[j] = d_propagator[keys[j]];
                }

                if (N_GPUS > 1)
                {
                    int prev, next;
                    prev = 0;
                    next = 1;

                    // Copy propagator of key1 from device0 to device1
                    gpu_error_check(cudaMemcpy(
                        d_propagator_device_1[prev],
                        _d_propagator_keys[1][n_segment_froms[1]-1],
                        sizeof(double)*M, cudaMemcpyDeviceToDevice));

                    for(int n=0; n<n_segment_tos[0]-n_segment_froms[0]; n++)
                    {
                        #ifndef NDEBUG
                        if (!propagator_finished[keys[0]][n-1+n_segment_froms[0]])
                            throw_with_line_number("unfinished, key: " + keys[0] + ", " + std::to_string(n-1+n_segment_froms[0]));
                        if (!propagator_finished[keys[1]][n-1+n_segment_froms[1]])
                            throw_with_line_number("unfinished, key: " + keys[1] + ", " + std::to_string(n-1+n_segment_froms[1]));
                        #endif

                        // DEVICE 0,1, STREAM 0: calculate propagators 
                        advance_two_propagators_two_gpus(
                            _d_propagator_keys[0][n-1+n_segment_froms[0]],
                            d_propagator_device_1[prev],
                            _d_propagator_keys[0][n+n_segment_froms[0]],
                            d_propagator_device_1[next],
                            d_boltz_bond[0][monomer_types[0]],
                            d_boltz_bond[1][monomer_types[1]],
                            d_exp_dw[0][monomer_types[0]],
                            d_exp_dw[1][monomer_types[1]]);

                        // DEVICE 1, STREAM 1: copy memory from device 1 to device 0
                        if (n > 0)
                        {
                            gpu_error_check(cudaMemcpyAsync(
                                _d_propagator_keys[1][n-1+n_segment_froms[1]],
                                d_propagator_device_1[prev],
                                sizeof(double)*M, cudaMemcpyDeviceToDevice, streams[1][1]));
                        }

                        // Synchronize all GPUs
                        for(int gpu=0; gpu<N_GPUS; gpu++)
                        {
                            gpu_error_check(cudaSetDevice(gpu));
                            gpu_error_check(cudaDeviceSynchronize());
                        }

                        std::swap(prev, next);

                        #ifndef NDEBUG
                        propagator_finished[keys[0]][n+n_segment_froms[0]] = true;
                        propagator_finished[keys[1]][n+n_segment_froms[1]] = true;
                        #endif
                    }
                    gpu_error_check(cudaMemcpy(
                        _d_propagator_keys[1][n_segment_tos[1]-1],
                        d_propagator_device_1[prev],
                        sizeof(double)*M, cudaMemcpyDeviceToDevice));
                }
                else
                {
                    gpu_error_check(cudaSetDevice(0));
                    for(int n=0; n<n_segment_tos[0]-n_segment_froms[0]; n++)
                    {
                        #ifndef NDEBUG
                        if (!propagator_finished[keys[0]][n-1+n_segment_froms[0]])
                            throw_with_line_number("unfinished, key: " + keys[0] + ", " + std::to_string(n-n_segment_froms[0]));
                        if (!propagator_finished[keys[1]][n-1+n_segment_froms[1]])
                            throw_with_line_number("unfinished, key: " + keys[1] + ", " + std::to_string(n-n_segment_froms[1]));
                        #endif

                        advance_two_propagators(
                            _d_propagator_keys[0][n-1+n_segment_froms[0]],
                            _d_propagator_keys[1][n-1+n_segment_froms[1]],
                            _d_propagator_keys[0][n+n_segment_froms[0]],
                            _d_propagator_keys[1][n+n_segment_froms[1]],
                            d_boltz_bond[0][monomer_types[0]],
                            d_boltz_bond[0][monomer_types[1]],
                            d_exp_dw[0][monomer_types[0]],
                            d_exp_dw[0][monomer_types[1]]);

                        #ifndef NDEBUG
                        propagator_finished[keys[0]][n+n_segment_froms[0]] = true;
                        propagator_finished[keys[1]][n+n_segment_froms[1]] = true;
                        #endif
                    }
                    gpu_error_check(cudaDeviceSynchronize());
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
            int p                    = std::get<0>(segment_info);
            double *d_propagator_v   = std::get<1>(segment_info);
            double *d_propagator_u   = std::get<2>(segment_info);
            std::string monomer_type = std::get<3>(segment_info);
            int n_superposed         = std::get<4>(segment_info);

            single_partitions[p] = cb->inner_product_inverse_weight_device(
                d_propagator_v,  // q
                d_propagator_u, // q^dagger
                d_exp_dw[0][monomer_type])/n_superposed/cb->get_volume();
        }

        // Calculate segment concentrations
        for(const auto& block: d_block_phi)
        {
            const auto& key = block.first;
            int p                = std::get<0>(key);
            std::string dep_v    = std::get<1>(key);
            std::string dep_u    = std::get<2>(key);

            int n_repeated;
            int n_segment_allocated = molecules->get_essential_block(key).n_segment_allocated;
            int n_segment_offset    = molecules->get_essential_block(key).n_segment_offset;
            int n_segment_original  = molecules->get_essential_block(key).n_segment_original;
            std::string monomer_type = molecules->get_essential_block(key).monomer_type;

            // Contains no '['
            if (dep_u.find('[') == std::string::npos)
                n_repeated = molecules->get_essential_block(key).v_u.size();
            else
                n_repeated = 1;

            // Check keys
            #ifndef NDEBUG
            if (d_propagator.find(dep_v) == d_propagator.end())
                throw_with_line_number("Could not find dep_v key'" + dep_v + "'. ");
            if (d_propagator.find(dep_u) == d_propagator.end())
                throw_with_line_number("Could not find dep_u key'" + dep_u + "'. ");
            #endif

            // Calculate phi of one block (possibly multiple blocks when using superposition)
            calculate_phi_one_block(
                block.second,              // Phi
                d_propagator[dep_v],       // dependency v
                d_propagator[dep_u],       // dependency u
                d_exp_dw[0][monomer_type], // exp_dw
                n_segment_allocated,
                n_segment_offset,
                n_segment_original);
            
            // Normalize concentration
            Polymer& pc = molecules->get_polymer(p);
            double norm = molecules->get_ds()*pc.get_volume_fraction()/pc.get_alpha()/single_partitions[p]*n_repeated;
            lin_comb<<<N_BLOCKS, N_THREADS>>>(block.second, norm, block.second, 0.0, block.second, M);
        }
        gpu_error_check(cudaSetDevice(0));
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaPseudoDiscrete::advance_one_propagator(
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

        // Execute a Forward FFT
        cufftExecD2Z(plan_for_one[GPU], d_q_in, d_qk_in_1_one[GPU]);

        // Multiply exp(-k^2 ds/6) in fourier space
        multi_complex_real<<<N_BLOCKS, N_THREADS, 0, streams[GPU][0]>>>(d_qk_in_1_one[GPU], d_boltz_bond, M_COMPLEX);

        // Execute a backward FFT
        cufftExecZ2D(plan_bak_one[GPU], d_qk_in_1_one[GPU], d_q_out);

        // Evaluate exp(-w*ds) in real space
        multi_real<<<N_BLOCKS, N_THREADS, 0, streams[GPU][0]>>>(d_q_out, d_q_out, d_exp_dw, 1.0/((double)M), M);
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaPseudoDiscrete::advance_two_propagators(
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

        gpu_error_check(cudaMemcpyAsync(&d_q_step_1_two[0][0], d_q_in_1, sizeof(double)*M, cudaMemcpyDeviceToDevice, streams[0][0]));
        gpu_error_check(cudaMemcpyAsync(&d_q_step_1_two[0][M], d_q_in_2, sizeof(double)*M, cudaMemcpyDeviceToDevice, streams[0][0]));

        // Execute a Forward FFT
        cufftExecD2Z(plan_for_two[0], d_q_step_1_two[0], d_qk_in_1_two[0]);

        // Multiply exp(-k^2 ds/6) in fourier space
        complex_real_multi_bond_two<<<N_BLOCKS, N_THREADS, 0, streams[0][0]>>>(
            &d_qk_in_1_two[0][0],         d_boltz_bond_1, 
            &d_qk_in_1_two[0][M_COMPLEX], d_boltz_bond_2, M_COMPLEX);

        // Execute a backward FFT
        cufftExecZ2D(plan_bak_two[0], d_qk_in_1_two[0], d_q_step_1_two[0]);

        // Evaluate exp(-w*ds) in real space
        real_multi_exp_dw_two<<<N_BLOCKS, N_THREADS, 0, streams[0][0]>>>(
            d_q_out_1, &d_q_step_1_two[0][0], d_exp_dw_1,
            d_q_out_2, &d_q_step_1_two[0][M], d_exp_dw_2, 1.0/((double)M), M);

    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaPseudoDiscrete::advance_two_propagators_two_gpus(
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

        // Execute a Forward FFT
        gpu_error_check(cudaSetDevice(0));
        cufftExecD2Z(plan_for_one[0], d_q_in_1, d_qk_in_1_one[0]);
        gpu_error_check(cudaSetDevice(1));
        cufftExecD2Z(plan_for_one[1], d_q_in_2, d_qk_in_1_one[1]);

        // Multiply exp(-k^2 ds/6) in fourier space
        gpu_error_check(cudaSetDevice(0));
        multi_complex_real<<<N_BLOCKS, N_THREADS, 0, streams[0][0]>>>(d_qk_in_1_one[0], d_boltz_bond_1, M_COMPLEX);
        gpu_error_check(cudaSetDevice(1));
        multi_complex_real<<<N_BLOCKS, N_THREADS, 0, streams[1][0]>>>(d_qk_in_1_one[1], d_boltz_bond_2, M_COMPLEX);

        // Execute a backward FFT
        gpu_error_check(cudaSetDevice(0));
        cufftExecZ2D(plan_bak_one[0], d_qk_in_1_one[0], d_q_out_1);
        gpu_error_check(cudaSetDevice(1));
        cufftExecZ2D(plan_bak_one[1], d_qk_in_1_one[1], d_q_out_2);

        // Evaluate exp(-w*ds) in real space
        gpu_error_check(cudaSetDevice(0));
        multi_real<<<N_BLOCKS, N_THREADS, 0, streams[0][0]>>>(d_q_out_1, d_q_out_1, d_exp_dw_1, 1.0/((double)M), M);
        gpu_error_check(cudaSetDevice(1));
        multi_real<<<N_BLOCKS, N_THREADS, 0, streams[1][0]>>>(d_q_out_2, d_q_out_2, d_exp_dw_2, 1.0/((double)M), M);
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaPseudoDiscrete::advance_propagator_half_bond_step(const int GPU, double *d_q_in, double *d_q_out, double *d_boltz_bond_half)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = cb->get_n_grid();
        const int M_COMPLEX = this->n_complex_grid;

        // 3D fourier discrete transform, forward and inplace
        cufftExecD2Z(plan_for_one[GPU], d_q_in, d_qk_in_1_one[GPU]);
        // Multiply exp(-k^2 ds/12) in fourier space, in all 3 directions
        multi_complex_real<<<N_BLOCKS, N_THREADS, 0, streams[GPU][0]>>>(d_qk_in_1_one[GPU], d_boltz_bond_half, 1.0/((double)M), M_COMPLEX);
        // 3D fourier discrete transform, backward and inplace
        cufftExecZ2D(plan_bak_one[GPU], d_qk_in_1_one[GPU], d_q_out);
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
        gpu_error_check(cudaSetDevice(0));

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
void CudaPseudoDiscrete::get_total_concentration(std::string monomer_type, double *phi)
{
    try
    {
        gpu_error_check(cudaSetDevice(0));

        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();
        const int M = cb->get_n_grid();

        // Initialize to zero
        gpu_error_check(cudaMemset(d_phi, 0, sizeof(double)*M));

        // For each block
        for(const auto& d_block: d_block_phi)
        {
            const auto& key = d_block.first;
            std::string dep_v = std::get<1>(key);
            int n_segment_allocated = molecules->get_essential_block(key).n_segment_allocated;
            if (Molecules::get_monomer_type_from_key(dep_v) == monomer_type && n_segment_allocated != 0)
                lin_comb<<<N_BLOCKS, N_THREADS>>>(d_phi, 1.0, d_phi, 1.0, d_block.second, M);
        }
        gpu_error_check(cudaMemcpy(phi, d_phi, sizeof(double)*M, cudaMemcpyDeviceToHost));
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaPseudoDiscrete::get_total_concentration(int p, std::string monomer_type, double *phi)
{
    try
    {
        gpu_error_check(cudaSetDevice(0));

        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = cb->get_n_grid();
        const int P = molecules->get_n_polymer_types();

        if (p < 0 || p > P-1)
            throw_with_line_number("Index (" + std::to_string(p) + ") must be in range [0, " + std::to_string(P-1) + "]");

        // Initialize to zero
        gpu_error_check(cudaMemset(d_phi, 0, sizeof(double)*M));

        // For each block
        for(const auto& d_block: d_block_phi)
        {
            const auto& key = d_block.first;
            int polymer_idx = std::get<0>(key);
            std::string dep_v = std::get<1>(key);
            int n_segment_allocated = molecules->get_essential_block(key).n_segment_allocated;
            if (polymer_idx == p && Molecules::get_monomer_type_from_key(dep_v) == monomer_type && n_segment_allocated != 0)
                lin_comb<<<N_BLOCKS, N_THREADS>>>(d_phi, 1.0, d_phi, 1.0, d_block.second, M);
        }
        gpu_error_check(cudaMemcpy(phi, d_phi, sizeof(double)*M, cudaMemcpyDeviceToHost));
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaPseudoDiscrete::get_block_concentration(int p, double *phi)
{
    try
    {
        gpu_error_check(cudaSetDevice(0));

        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = cb->get_n_grid();
        const int P = molecules->get_n_polymer_types();

        if (p < 0 || p > P-1)
            throw_with_line_number("Index (" + std::to_string(p) + ") must be in range [0, " + std::to_string(P-1) + "]");

        if (molecules->is_using_superposition())
            throw_with_line_number("Disable 'superposition' option to obtain concentration of each block.");

        // Initialize to zero
        gpu_error_check(cudaMemset(d_phi, 0, sizeof(double)*M));

        Polymer& pc = molecules->get_polymer(p);
        std::vector<Block>& blocks = pc.get_blocks();

        for(size_t b=0; b<blocks.size(); b++)
        {
            std::string dep_v = pc.get_propagator_key(blocks[b].v, blocks[b].u);
            std::string dep_u = pc.get_propagator_key(blocks[b].u, blocks[b].v);
            if (dep_v < dep_u)
                dep_v.swap(dep_u);

            lin_comb<<<N_BLOCKS, N_THREADS>>>(d_phi, 0.0, d_phi, 1.0, d_block_phi[std::make_tuple(p, dep_v, dep_u)], M);
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
        const int N_GPUS = CudaCommon::get_instance().get_n_gpus();

        const int DIM  = cb->get_dim();
        const int M    = cb->get_n_grid();
        const int M_COMPLEX = this->n_complex_grid;

        auto bond_lengths = molecules->get_bond_lengths();
        std::vector<double> stress(DIM);
        std::map<std::tuple<int, std::string, std::string>, std::array<double,3>> block_dq_dl[MAX_GPUS];
        double stress_sum_out[MAX_GPUS][3];

        // Compute stress for each block
        for(const auto& d_block: d_block_phi)
        {
            const auto& key = d_block.first;
            int p                = std::get<0>(key);
            std::string dep_v    = std::get<1>(key);
            std::string dep_u    = std::get<2>(key);

            const int N           = molecules->get_essential_block(key).n_segment_allocated;
            const int N_OFFSET    = molecules->get_essential_block(key).n_segment_offset;
            const int N_ORIGINAL  = molecules->get_essential_block(key).n_segment_original;
            std::string monomer_type = molecules->get_essential_block(key).monomer_type;

            // Contains no '['
            int n_repeated;
            if (dep_u.find('[') == std::string::npos)
                n_repeated = molecules->get_essential_block(key).v_u.size();
            else
                n_repeated = 1;

            double bond_length_sq[MAX_GPUS][2];       // one for prev, the other for next
            double *d_boltz_bond_now[MAX_GPUS][2];    // one for prev, the other for next
            double **d_q_1 = d_propagator[dep_v];     // Propagator q
            double **d_q_2 = d_propagator[dep_u];     // Propagator q^dagger

            std::array<double,3> _block_dq_dl[MAX_GPUS];
            for(int gpu=0; gpu<N_GPUS; gpu++)
                for(int d=0; d<3; d++)
                    _block_dq_dl[gpu][d] = 0.0;

            // Check block_stress_info
            const auto& _block_stress_info_key = block_stress_info[key];
            if(_block_stress_info_key.size() != (unsigned int) (N+1))
            {
                throw_with_line_number("Mismatch of block_stress_info("
                    + std::to_string(p) + "," + dep_v + "," + dep_u + ") "
                    + std::to_string(_block_stress_info_key.size()) + ".size() with N+1 (" + std::to_string(N+1) + ")");
            }

            // Variables for block_stress_info
            double *d_propagator_v;
            double *d_propagator_u;
            bool is_half_bond_length;

            int prev, next;
            prev = 0;
            next = 1;

            // Copy memory from device to device
            for(int gpu=0; gpu<N_GPUS; gpu++)
            {
                // Index
                int idx = gpu;
                gpu_error_check(cudaSetDevice(gpu));
                if (idx <= N)
                {
                    d_propagator_v = std::get<0>(_block_stress_info_key[idx]);
                    d_propagator_u = std::get<1>(_block_stress_info_key[idx]);
                    is_half_bond_length = std::get<2>(_block_stress_info_key[idx]);

                    if (d_propagator_v != nullptr)
                    {
                        gpu_error_check(cudaMemcpy(&d_stress_q[gpu][prev][0], d_propagator_v, sizeof(double)*M, cudaMemcpyDeviceToDevice));
                        gpu_error_check(cudaMemcpy(&d_stress_q[gpu][prev][M], d_propagator_u, sizeof(double)*M, cudaMemcpyDeviceToDevice));

                        if(is_half_bond_length)
                        {
                            bond_length_sq[gpu][prev] = 0.5*bond_lengths[monomer_type]*bond_lengths[monomer_type];
                            d_boltz_bond_now[gpu][prev] = d_boltz_bond_half[gpu][monomer_type];
                        }
                        else
                        {
                            bond_length_sq[gpu][prev] = bond_lengths[monomer_type]*bond_lengths[monomer_type];
                            d_boltz_bond_now[gpu][prev] = d_boltz_bond[gpu][monomer_type];
                        }
                    }
                }
            }

            // Compute stress
            for(int n=0; n<=N; n+=N_GPUS)
            {
                // STREAM 1: copy memory from device to device
                for(int gpu=0; gpu<N_GPUS; gpu++)
                {
                    const int idx = n + gpu;
                    const int idx_next = idx + N_GPUS;
                    
                    gpu_error_check(cudaSetDevice(gpu));
                    if (idx_next <= N)
                    {
                        d_propagator_v = std::get<0>(_block_stress_info_key[idx_next]);
                        d_propagator_u = std::get<1>(_block_stress_info_key[idx_next]);
                        is_half_bond_length = std::get<2>(_block_stress_info_key[idx_next]);

                        if (d_propagator_v != nullptr)
                        {
                            gpu_error_check(cudaMemcpyAsync(&d_stress_q[gpu][next][0], d_propagator_v, sizeof(double)*M, cudaMemcpyDeviceToDevice, streams[gpu][1]));
                            gpu_error_check(cudaMemcpyAsync(&d_stress_q[gpu][next][M], d_propagator_u, sizeof(double)*M, cudaMemcpyDeviceToDevice, streams[gpu][1]));

                            if(is_half_bond_length)
                            {
                                bond_length_sq[gpu][next] = 0.5*bond_lengths[monomer_type]*bond_lengths[monomer_type];
                                d_boltz_bond_now[gpu][next] = d_boltz_bond_half[gpu][monomer_type];
                            }
                            else
                            {
                                bond_length_sq[gpu][next] = bond_lengths[monomer_type]*bond_lengths[monomer_type];
                                d_boltz_bond_now[gpu][next] = d_boltz_bond[gpu][monomer_type];
                            }
                        }
                    }
                }

                // STREAM 0: execute kernels
                for(int gpu=0; gpu<N_GPUS; gpu++)
                {
                    const int idx = n + gpu;
                    gpu_error_check(cudaSetDevice(gpu));
                    if (idx <= N)
                    {
                        d_propagator_v = std::get<0>(_block_stress_info_key[idx]);
                        d_propagator_u = std::get<1>(_block_stress_info_key[idx]);
                        is_half_bond_length = std::get<2>(_block_stress_info_key[idx]);

                        if (d_propagator_v != nullptr)
                        {
                            // Execute a Forward FFT
                            cufftExecD2Z(plan_for_two[gpu], d_stress_q[gpu][prev], d_qk_in_1_two[gpu]);
                            // Multiply two propagators in the fourier spaces
                            multi_complex_conjugate<<<N_BLOCKS, N_THREADS, 0, streams[gpu][0]>>>(d_q_multi[gpu], &d_qk_in_1_two[gpu][0], &d_qk_in_1_two[gpu][M_COMPLEX], M_COMPLEX);
                            multi_real<<<N_BLOCKS, N_THREADS, 0, streams[gpu][0]>>>(d_q_multi[gpu], d_q_multi[gpu], d_boltz_bond_now[gpu][prev], bond_length_sq[gpu][prev], M_COMPLEX);
                        }
                    }
                }

                // STREAM 0: reduction sum
                for(int gpu=0; gpu<N_GPUS; gpu++)
                {
                    const int idx = n + gpu;
                    gpu_error_check(cudaSetDevice(gpu));
                    if (idx <= N)
                    {
                        d_propagator_v = std::get<0>(_block_stress_info_key[idx]);
                        d_propagator_u = std::get<1>(_block_stress_info_key[idx]);
                        is_half_bond_length = std::get<2>(_block_stress_info_key[idx]);
                        if (d_propagator_v != nullptr)
                        {
                            if ( DIM == 3 )
                            {
                                // x direction
                                multi_real<<<N_BLOCKS, N_THREADS, 0, streams[gpu][0]>>>(d_stress_sum[gpu], d_q_multi[gpu], d_fourier_basis_x[gpu], 1.0, M_COMPLEX);
                                cub::DeviceReduce::Sum(d_temp_storage[gpu], temp_storage_bytes[gpu], d_stress_sum[gpu], d_stress_sum_out[gpu], M_COMPLEX, streams[gpu][0]);
                                gpu_error_check(cudaMemcpyAsync(&stress_sum_out[gpu][0],d_stress_sum_out[gpu],sizeof(double),cudaMemcpyDeviceToHost, streams[gpu][0]));

                                // y direction
                                multi_real<<<N_BLOCKS, N_THREADS, 0, streams[gpu][0]>>>(d_stress_sum[gpu], d_q_multi[gpu], d_fourier_basis_y[gpu], 1.0, M_COMPLEX);
                                cub::DeviceReduce::Sum(d_temp_storage[gpu], temp_storage_bytes[gpu], d_stress_sum[gpu], d_stress_sum_out[gpu], M_COMPLEX, streams[gpu][0]);
                                gpu_error_check(cudaMemcpyAsync(&stress_sum_out[gpu][1],d_stress_sum_out[gpu],sizeof(double),cudaMemcpyDeviceToHost, streams[gpu][0]));

                                // z direction
                                multi_real<<<N_BLOCKS, N_THREADS, 0, streams[gpu][0]>>>(d_stress_sum[gpu], d_q_multi[gpu], d_fourier_basis_z[gpu], 1.0, M_COMPLEX);
                                cub::DeviceReduce::Sum(d_temp_storage[gpu], temp_storage_bytes[gpu], d_stress_sum[gpu], d_stress_sum_out[gpu], M_COMPLEX, streams[gpu][0]);
                                gpu_error_check(cudaMemcpyAsync(&stress_sum_out[gpu][2],d_stress_sum_out[gpu],sizeof(double),cudaMemcpyDeviceToHost, streams[gpu][0]));
                            }
                            if ( DIM == 2 )
                            {
                                // y direction
                                multi_real<<<N_BLOCKS, N_THREADS, 0, streams[gpu][0]>>>(d_stress_sum[gpu], d_q_multi[gpu], d_fourier_basis_y[gpu], 1.0, M_COMPLEX);
                                cub::DeviceReduce::Sum(d_temp_storage[gpu], temp_storage_bytes[gpu], d_stress_sum[gpu], d_stress_sum_out[gpu], M_COMPLEX, streams[gpu][0]);
                                gpu_error_check(cudaMemcpyAsync(&stress_sum_out[gpu][0],d_stress_sum_out[gpu],sizeof(double),cudaMemcpyDeviceToHost, streams[gpu][0]));

                                // z direction
                                multi_real<<<N_BLOCKS, N_THREADS, 0, streams[gpu][0]>>>(d_stress_sum[gpu], d_q_multi[gpu], d_fourier_basis_z[gpu], 1.0, M_COMPLEX);
                                cub::DeviceReduce::Sum(d_temp_storage[gpu], temp_storage_bytes[gpu], d_stress_sum[gpu], d_stress_sum_out[gpu], M_COMPLEX, streams[gpu][0]);
                                gpu_error_check(cudaMemcpyAsync(&stress_sum_out[gpu][1],d_stress_sum_out[gpu],sizeof(double),cudaMemcpyDeviceToHost, streams[gpu][0]));
                            }
                            if ( DIM == 1 )
                            {
                                // z direction
                                multi_real<<<N_BLOCKS, N_THREADS, 0, streams[gpu][0]>>>(d_stress_sum[gpu], d_q_multi[gpu], d_fourier_basis_z[gpu], 1.0, M_COMPLEX);
                                cub::DeviceReduce::Sum(d_temp_storage[gpu], temp_storage_bytes[gpu], d_stress_sum[gpu], d_stress_sum_out[gpu], M_COMPLEX, streams[gpu][0]);
                                gpu_error_check(cudaMemcpyAsync(&stress_sum_out[gpu][0],d_stress_sum_out[gpu],sizeof(double),cudaMemcpyDeviceToHost, streams[gpu][0]));
                            }
                            // Synchronize streams and add results
                            gpu_error_check(cudaStreamSynchronize(streams[gpu][0]));
                            for(int d=0; d<DIM; d++)
                                _block_dq_dl[gpu][d] += stress_sum_out[gpu][d]*n_repeated;
                        }
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
        for(const auto& d_block: d_block_phi)
        {
            const auto& key = d_block.first;
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
void CudaPseudoDiscrete::get_chain_propagator(double *q_out, int polymer, int v, int u, int n)
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
        if (n < 1 || n > N)
            throw_with_line_number("n (" + std::to_string(n) + ") must be in range [1, " + std::to_string(N) + "]");

        gpu_error_check(cudaMemcpy(q_out, d_propagator[dep][n-1], sizeof(double)*M,cudaMemcpyDeviceToHost));
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
