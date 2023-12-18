#include <complex>
#include <thrust/reduce.h>
#include "CudaComputationReduceMemoryContinuous.h"
#include "CudaComputationBox.h"
#include "SimpsonRule.h"

CudaComputationReduceMemoryContinuous::CudaComputationReduceMemoryContinuous(
    ComputationBox *cb,
    Molecules *molecules,
    PropagatorAnalyzer *propagator_analyzer)
    : PropagatorComputation(cb, molecules, propagator_analyzer)
{
    try{
        const int M = cb->get_n_grid();
        const int N_GPUS = CudaCommon::get_instance().get_n_gpus();

        // Create streams
        for(int gpu=0; gpu<N_GPUS; gpu++)
        {
            gpu_error_check(cudaSetDevice(gpu));
            gpu_error_check(cudaStreamCreate(&streams[gpu][0])); // for kernel execution
            gpu_error_check(cudaStreamCreate(&streams[gpu][1])); // for memcpy
        }
        this->propagator_solver = new CudaSolverPseudo(cb, molecules, streams, true);

        // Allocate memory for propagators
        gpu_error_check(cudaSetDevice(0));
        if( propagator_analyzer->get_computation_propagator_codes().size() == 0)
            throw_with_line_number("There is no propagator code. Add polymers first.");
        for(const auto& item: propagator_analyzer->get_computation_propagator_codes())
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
        if( propagator_analyzer->get_computation_blocks().size() == 0)
            throw_with_line_number("There is no block. Add polymers first.");
        for(const auto& item: propagator_analyzer->get_computation_blocks())
        {
            phi_block[item.first] = nullptr;
            // Allocate pinned memory
            gpu_error_check(cudaMallocHost((void**)&phi_block[item.first], sizeof(double)*M));
        }

        // Total partition functions for each polymer
        single_polymer_partitions = new double[molecules->get_n_polymer_types()];

        // Remember one segment for each polymer chain to compute total partition function
        int current_p = 0;
        for(const auto& block: phi_block)
        {
            const auto& key = block.first;
            int p                = std::get<0>(key);
            std::string dep_v    = std::get<1>(key);
            std::string dep_u    = std::get<2>(key);

            // Skip if already found one segment
            if (p != current_p)
                continue;

            int n_aggregated;
            int n_segment_offset    = propagator_analyzer->get_computation_block(key).n_segment_offset;
            int n_segment_original  = propagator_analyzer->get_computation_block(key).n_segment_original;

            // Contains no '['
            if (dep_u.find('[') == std::string::npos)
                n_aggregated = 1;
            else
                n_aggregated = propagator_analyzer->get_computation_block(key).v_u.size();

            single_partition_segment.push_back(std::make_tuple(
                p,
                propagator[dep_v][n_segment_original-n_segment_offset],   // q
                propagator[dep_u][0],                                   // q_dagger
                n_aggregated                    // how many propagators are aggregated
                ));
            current_p++;
        }

        // Total partition functions for each solvent
        single_solvent_partitions = new double[molecules->get_n_solvent_types()];

        // Concentrations for each solvent
        for(int s=0;s<molecules->get_n_solvent_types();s++)
            phi_solvent.push_back(new double[M]);

        // Create scheduler for computation of propagator
        sc = new Scheduler(propagator_analyzer->get_computation_propagator_codes(), N_SCHEDULER_STREAMS); 

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
            gpu_error_check(cudaMalloc((void**)&d_q_mask[gpu], sizeof(double)*M));
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
            gpu_error_check(cudaMalloc((void**)&d_stress_q[gpu][0],     sizeof(double)*2*M)); // prev
            gpu_error_check(cudaMalloc((void**)&d_stress_q[gpu][1],     sizeof(double)*2*M)); // next
        }

        propagator_solver->update_laplacian_operator();
        gpu_error_check(cudaSetDevice(0));
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
CudaComputationReduceMemoryContinuous::~CudaComputationReduceMemoryContinuous()
{
    const int N_GPUS = CudaCommon::get_instance().get_n_gpus();
    
    delete sc;

    delete[] single_polymer_partitions;
    delete[] single_solvent_partitions;

    for(const auto& item: propagator)
    {
        for(int i=0; i<propagator_size[item.first]; i++)
            cudaFreeHost(item.second[i]);
        delete[] item.second;
    }
    for(const auto& item: phi_block)
        cudaFreeHost(item.second);
    for(const auto& item: phi_solvent)
        delete[] item;

    #ifndef NDEBUG
    for(const auto& item: propagator_finished)
        delete[] item.second;
    #endif

    for(int gpu=0; gpu<N_GPUS; gpu++)
    {
        cudaFree(d_q_one[gpu][0]);
        cudaFree(d_q_one[gpu][1]);
        cudaFree(d_q_mask[gpu]);
    }
    cudaFree(d_q_unity);

    // For pseudo-spectral: advance_one_propagator()
    cudaFree(d_propagator_sub_dep[0]);
    cudaFree(d_propagator_sub_dep[1]);

    // For stress calculation: compute_stress()
    for(int gpu=0; gpu<N_GPUS; gpu++)
    {
        cudaFree(d_stress_q[gpu][0]);
        cudaFree(d_stress_q[gpu][1]);
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

void CudaComputationReduceMemoryContinuous::update_laplacian_operator()
{
    try{
        propagator_solver->update_laplacian_operator();
    }
    catch(std::exception& exc)
    {
        throw_with_line_number(exc.what());
    }
}
void CudaComputationReduceMemoryContinuous::compute_statistics(
    std::string device,
    std::map<std::string, const double*> w_input,
    std::map<std::string, const double*> q_init)
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

        for(const auto& item: propagator_analyzer->get_computation_propagator_codes())
        {
            if( w_input.find(item.second.monomer_type) == w_input.end())
                throw_with_line_number("monomer_type \"" + item.second.monomer_type + "\" is not in w_input.");
        }

        // Copy mask to d_q_mask
        for(int gpu=0; gpu<N_GPUS; gpu++)
        {
            gpu_error_check(cudaSetDevice(gpu));
            if (cb->get_mask() != nullptr)
            {
                gpu_error_check(cudaMemcpy(d_q_mask[gpu], cb->get_mask(), sizeof(double)*M, cudaMemcpyInputToDevice));
            }
            else
            {
                d_q_mask[gpu] = nullptr;
            }
        }

        // Update dw or d_exp_dw
        propagator_solver->update_dw(device, w_input);

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
                auto& deps = propagator_analyzer->get_computation_propagator_code(key).deps;
                auto monomer_type = propagator_analyzer->get_computation_propagator_code(key).monomer_type;

                // Check key
                #ifndef NDEBUG
                if (propagator.find(key) == propagator.end())
                    throw_with_line_number("Could not find key '" + key + "'. ");
                #endif
                double *_propagator_0 = propagator[key][0];

                // If it is leaf node
                if(deps.size() == 0)
                {
                    // q_init
                    if (key[0] == '{')
                    {
                        std::string g = PropagatorCode::get_q_input_idx_from_key(key);
                        if (q_init.find(g) == q_init.end())
                            throw_with_line_number( "Could not find q_init[\"" + g + "\"].");
                        gpu_error_check(cudaMemcpy(d_q_one[0][0], q_init[g], sizeof(double)*M, cudaMemcpyInputToDevice));
                    }
                    else
                    {
                        gpu_error_check(cudaMemcpy(d_q_one[0][0], d_q_unity, sizeof(double)*M, cudaMemcpyDeviceToDevice));
                    }

                    // Multiply mask
                    if (d_q_mask[0] != nullptr)
                        multi_real<<<N_BLOCKS, N_THREADS>>>(d_q_one[0][0], d_q_one[0][0], d_q_mask[0], 1.0, M);

                    gpu_error_check(cudaMemcpy(_propagator_0, d_q_one[0][0], sizeof(double)*M, cudaMemcpyDeviceToHost));

                    #ifndef NDEBUG
                    propagator_finished[key][0] = true;
                    #endif
                }
                // If it is not leaf node
                else if (n_segment_from == 1 && deps.size() > 0)
                {
                    // If it is aggregated
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

                        // Multiply mask
                        if (d_q_mask[0] != nullptr)
                            multi_real<<<N_BLOCKS, N_THREADS>>>(d_q_one[0][0], d_q_one[0][0], d_q_mask[0], 1.0, M);

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

                        // Multiply mask
                        if (d_q_mask[0] != nullptr)
                            multi_real<<<N_BLOCKS, N_THREADS>>>(d_q_one[0][0], d_q_one[0][0], d_q_mask[0], 1.0, M);

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
                    monomer_types[j] = propagator_analyzer->get_computation_propagator_code(keys[j]).monomer_type;
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
                    propagator_solver->advance_two_propagators_continuous_two_gpus(
                        d_q_one[0][prev], d_q_one[1][prev],
                        d_q_one[0][next], d_q_one[1][next],
                        monomer_types[0], monomer_types[1],
                        d_q_mask);

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
                    monomer_types[j] = propagator_analyzer->get_computation_propagator_code(keys[j]).monomer_type;
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
                        propagator_solver->advance_one_propagator_continuous(0, 
                            d_q_one[0][prev],
                            d_q_one[0][next],
                            monomer_types[j],
                            d_q_mask[0]);

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
            int n_aggregated     = std::get<3>(segment_info);

            single_polymer_partitions[p]= cb->inner_product(
                propagator_v, propagator_u)/n_aggregated/cb->get_volume();
        }

        // Calculate segment concentrations
        for(const auto& block: phi_block)
        {
            const auto& key = block.first;
            int p                = std::get<0>(key);
            std::string dep_v    = std::get<1>(key);
            std::string dep_u    = std::get<2>(key);

            int n_repeated;
            int n_segment_allocated = propagator_analyzer->get_computation_block(key).n_segment_allocated;
            int n_segment_offset    = propagator_analyzer->get_computation_block(key).n_segment_offset;
            int n_segment_original  = propagator_analyzer->get_computation_block(key).n_segment_original;

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
                n_repeated = propagator_analyzer->get_computation_block(key).v_u.size();
            else
                n_repeated = 1;

            // Normalization constant
            Polymer& pc = molecules->get_polymer(p);
            double norm = molecules->get_ds()*pc.get_volume_fraction()/pc.get_alpha()/single_polymer_partitions[p]*n_repeated;

            // Calculate phi of one block (possibly multiple blocks when using aggregation)
            calculate_phi_one_block(
                block.second,       // phi
                propagator[dep_v],  // dependency v
                propagator[dep_u],  // dependency u
                n_segment_allocated,
                n_segment_offset,
                n_segment_original,
                norm);
        }
        // Calculate partition functions and concentrations of solvents
        for(size_t s=0; s<molecules->get_n_solvent_types(); s++)
        {
            double volume_fraction = std::get<0>(molecules->get_solvent(s));
            std::string monomer_type = std::get<1>(molecules->get_solvent(s));
            double *_d_exp_dw = propagator_solver->d_exp_dw[0][monomer_type];

            single_solvent_partitions[s] = cb->inner_product_device(_d_exp_dw, _d_exp_dw)/cb->get_volume();
            multi_real<<<N_BLOCKS, N_THREADS>>>(d_phi, _d_exp_dw, _d_exp_dw, volume_fraction/single_solvent_partitions[s], M);
            gpu_error_check(cudaMemcpy(phi_solvent[s], d_phi, sizeof(double)*M, cudaMemcpyDeviceToHost));
        }
        gpu_error_check(cudaSetDevice(0));
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaComputationReduceMemoryContinuous::calculate_phi_one_block(
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
double CudaComputationReduceMemoryContinuous::get_total_partition(int polymer)
{
    try
    {
        return single_polymer_partitions[polymer];
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaComputationReduceMemoryContinuous::get_total_concentration(std::string monomer_type, double *phi)
{
    try
    {
        const int M = cb->get_n_grid();
        // Initialize array
        for(int i=0; i<M; i++)
            phi[i] = 0.0;

        // For each block
        for(const auto& block: phi_block)
        {
            std::string dep_v = std::get<1>(block.first);
            int n_segment_allocated = propagator_analyzer->get_computation_block(block.first).n_segment_allocated;
            if (PropagatorCode::get_monomer_type_from_key(dep_v) == monomer_type && n_segment_allocated != 0)
            {
                for(int i=0; i<M; i++)
                    phi[i] += block.second[i]; 
            }
        }
        // For each solvent
        for(int s=0;s<molecules->get_n_solvent_types();s++)
        {
            if (std::get<1>(molecules->get_solvent(s)) == monomer_type)
            {
                double *phi_solvent_ = phi_solvent[s];
                for(int i=0; i<M; i++)
                    phi[i] += phi_solvent_[i];
            }
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaComputationReduceMemoryContinuous::get_total_concentration(int p, std::string monomer_type, double *phi)
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
        for(const auto& block: phi_block)
        {
            int polymer_idx = std::get<0>(block.first);
            std::string dep_v = std::get<1>(block.first);
            int n_segment_allocated = propagator_analyzer->get_computation_block(block.first).n_segment_allocated;
            if (polymer_idx == p && PropagatorCode::get_monomer_type_from_key(dep_v) == monomer_type && n_segment_allocated != 0)
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
void CudaComputationReduceMemoryContinuous::get_block_concentration(int p, double *phi)
{
    try
    {
        const int M = cb->get_n_grid();
        const int P = molecules->get_n_polymer_types();

        if (p < 0 || p > P-1)
            throw_with_line_number("Index (" + std::to_string(p) + ") must be in range [0, " + std::to_string(P-1) + "]");

        if (propagator_analyzer->is_aggregated())
            throw_with_line_number("Disable 'aggregation' option to invoke 'get_block_concentration'.");

        Polymer& pc = molecules->get_polymer(p);
        std::vector<Block>& blocks = pc.get_blocks();

        for(size_t b=0; b<blocks.size(); b++)
        {
            std::string dep_v = pc.get_propagator_key(blocks[b].v, blocks[b].u);
            std::string dep_u = pc.get_propagator_key(blocks[b].u, blocks[b].v);
            if (dep_v < dep_u)
                dep_v.swap(dep_u);

            double* _essential_phi_block = phi_block[std::make_tuple(p, dep_v, dep_u)];
            for(int i=0; i<M; i++)
                phi[i+b*M] = _essential_phi_block[i]; 
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
double CudaComputationReduceMemoryContinuous::get_solvent_partition(int s)
{
    try
    {
        return single_solvent_partitions[s];
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaComputationReduceMemoryContinuous::get_solvent_concentration(int s, double *phi)
{
    try
    {
        const int M = cb->get_n_grid();
        const int S = molecules->get_n_solvent_types();

        if (s < 0 || s > S-1)
            throw_with_line_number("Index (" + std::to_string(s) + ") must be in range [0, " + std::to_string(S-1) + "]");

        double *phi_solvent_ = phi_solvent[s];
        for(int i=0; i<M; i++)
            phi[i] = phi_solvent_[i];
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
std::vector<double> CudaComputationReduceMemoryContinuous::compute_stress()
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

        std::vector<double> stress(DIM);
        std::map<std::tuple<int, std::string, std::string>, std::array<double,3>> block_dq_dl[MAX_GPUS];

        // Compute stress for each block
        for(const auto& block: phi_block)
        {
            const auto& key = block.first;
            int p                = std::get<0>(key);
            std::string dep_v    = std::get<1>(key);
            std::string dep_u    = std::get<2>(key);

            const int N           = propagator_analyzer->get_computation_block(key).n_segment_allocated;
            const int N_OFFSET    = propagator_analyzer->get_computation_block(key).n_segment_offset;
            const int N_ORIGINAL  = propagator_analyzer->get_computation_block(key).n_segment_original;
            std::string monomer_type = propagator_analyzer->get_computation_block(key).monomer_type;

            // If there is no segment
            if(N == 0)
                continue;

            // Contains no '['
            int n_repeated;
            if (dep_u.find('[') == std::string::npos)
                n_repeated = propagator_analyzer->get_computation_block(key).v_u.size();
            else
                n_repeated = 1;

            std::vector<double> s_coeff = SimpsonRule::get_coeff(N);
            double** q_1 = propagator[dep_v];    // dependency v
            double** q_2 = propagator[dep_u];    // dependency u

            std::array<double,3> _block_dq_dl[MAX_GPUS];
            std::vector<double> segment_stress[MAX_GPUS];
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
                // Execute a forward FFT
                for(int gpu=0; gpu<N_GPUS; gpu++)
                {
                    const int idx = n + gpu;
                    gpu_error_check(cudaSetDevice(gpu));
                    if (idx <= N)
                        propagator_solver->compute_single_segment_stress_fourier(gpu, d_stress_q[gpu][prev]);

                }
                // Multiply two propagators in the fourier spaces
                for(int gpu=0; gpu<N_GPUS; gpu++)
                {
                    const int idx = n + gpu;
                    gpu_error_check(cudaSetDevice(gpu));
                    if (idx <= N)
                    {
                        segment_stress[gpu] = propagator_solver->compute_single_segment_stress_continuous(gpu, monomer_type);
                        for(int d=0; d<DIM; d++)
                            _block_dq_dl[gpu][d] += s_coeff[idx]*segment_stress[gpu][d]*n_repeated;
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
        for(const auto& block: phi_block)
        {
            const auto& key = block.first;
            int p             = std::get<0>(key);
            std::string dep_v = std::get<1>(key);
            std::string dep_u = std::get<2>(key);
            Polymer& pc  = molecules->get_polymer(p);

            for(int gpu=0; gpu<N_GPUS; gpu++)
                for(int d=0; d<DIM; d++)
                    stress[d] += block_dq_dl[gpu][key][d]*pc.get_volume_fraction()/pc.get_alpha()/single_polymer_partitions[p];
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
void CudaComputationReduceMemoryContinuous::get_chain_propagator(double *q_out, int polymer, int v, int u, int n)
{
    // This method should be invoked after invoking compute_statistics()

    // Get chain propagator for a selected polymer, block and direction.
    // This is made for debugging and testing.
    try
    {
        const int M = cb->get_n_grid();
        Polymer& pc = molecules->get_polymer(polymer);
        std::string dep = pc.get_propagator_key(v,u);

        if (propagator_analyzer->get_computation_propagator_codes().find(dep) == propagator_analyzer->get_computation_propagator_codes().end())
            throw_with_line_number("Could not find the propagator code '" + dep + "'. Disable 'aggregation' option to obtain propagator_analyzer.");

        const int N = propagator_analyzer->get_computation_propagator_codes()[dep].max_n_segment;
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
bool CudaComputationReduceMemoryContinuous::check_total_partition()
{
    const int M = cb->get_n_grid();
    int n_polymer_types = molecules->get_n_polymer_types();
    std::vector<std::vector<double>> total_partitions;
    for(int p=0;p<n_polymer_types;p++)
    {
        std::vector<double> total_partitions_p;
        total_partitions.push_back(total_partitions_p);
    }

    gpu_error_check(cudaSetDevice(0));
    for(const auto& block: phi_block)
    {
        const auto& key = block.first;
        int p                = std::get<0>(key);
        std::string dep_v    = std::get<1>(key);
        std::string dep_u    = std::get<2>(key);

        int n_aggregated;
        int n_segment_allocated = propagator_analyzer->get_computation_block(key).n_segment_allocated;
        int n_segment_offset    = propagator_analyzer->get_computation_block(key).n_segment_offset;
        int n_segment_original  = propagator_analyzer->get_computation_block(key).n_segment_original;

        // std::cout<< p << ", " << dep_v << ", " << dep_u << ": " << n_segment_original << ", " << n_segment_offset << ", " << n_segment_allocated << std::endl;

        // Contains no '['
        if (dep_u.find('[') == std::string::npos)
            n_aggregated = 1;
        else
            n_aggregated = propagator_analyzer->get_computation_block(key).v_u.size();

        for(int n=0;n<=n_segment_allocated;n++)
        {
            double total_partition = cb->inner_product(
                propagator[dep_v][n_segment_original-n_segment_offset-n],   // q
                propagator[dep_u][n])/n_aggregated/cb->get_volume();

            // std::cout<< p << ", " << n << ": " << total_partition << std::endl;
            total_partitions[p].push_back(total_partition);
        }
    }

    // Find minimum and maximum of total_partitions
    std::cout<< "Polymer id: maximum,  minimum, and difference of total partitions" << std::endl;
    for(size_t p=0;p<total_partitions.size();p++)
    {
        double max_partition = -1e20;
        double min_partition =  1e20;
        for(size_t n=0;n<total_partitions[p].size();n++)
        {
            if (total_partitions[p][n] > max_partition)
                max_partition = total_partitions[p][n];
            if (total_partitions[p][n] < min_partition)
                min_partition = total_partitions[p][n];
        }
        double diff_partition = abs(max_partition - min_partition);

        std::cout<< "\t" << p << ": " << max_partition << ", " << min_partition << ", " << diff_partition << std::endl;
        if (diff_partition > 1e-7)
            return false;
    }
    return true;
}