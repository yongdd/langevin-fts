#include <complex>
#include <thrust/reduce.h>
#include "CudaSolverContinuous.h"
#include "CudaComputationBox.h"
#include "CudaPseudo.h"
#include "SimpsonRule.h"

CudaSolverContinuous::CudaSolverContinuous(
    ComputationBox *cb,
    Molecules *molecules,
    PropagatorsAnalyzer *propagators_analyzer)
    : Solver(cb, molecules, propagators_analyzer)
{
    try{
        const int M = cb->get_n_grid();
        const int N_GPUS = CudaCommon::get_instance().get_n_gpus();
        this->propagators_analyzer = propagators_analyzer;

        // Create streams
        for(int gpu=0; gpu<N_GPUS; gpu++)
        {
            gpu_error_check(cudaSetDevice(gpu));
            gpu_error_check(cudaStreamCreate(&streams[gpu][0])); // for kernel execution
            gpu_error_check(cudaStreamCreate(&streams[gpu][1])); // for memcpy
        }
        this->propagator_solver = new CudaPseudo(cb, molecules, streams, false);

        // Allocate memory for propagators
        gpu_error_check(cudaSetDevice(0));
        if( propagators_analyzer->get_essential_propagator_codes().size() == 0)
            throw_with_line_number("There is no propagator code. Add polymers first.");
        for(const auto& item: propagators_analyzer->get_essential_propagator_codes())
        {
            std::string key = item.first;
            int max_n_segment = item.second.max_n_segment;
            
            propagator_size[key] = max_n_segment+1;
            d_propagator[key] = new double*[max_n_segment+1];
            for(int i=0; i<propagator_size[key]; i++)
                gpu_error_check(cudaMalloc((void**)&d_propagator[key][i], sizeof(double)*M));

            #ifndef NDEBUG
            propagator_finished[key] = new bool[max_n_segment+1];
            for(int i=0; i<=max_n_segment;i++)
                propagator_finished[key][i] = false;
            #endif
        }

        // Allocate memory for concentrations
        if( propagators_analyzer->get_essential_blocks().size() == 0)
            throw_with_line_number("There is no block. Add polymers first.");
        for(const auto& item: propagators_analyzer->get_essential_blocks())
        {
            d_phi_block[item.first] = nullptr;
            gpu_error_check(cudaMalloc((void**)&d_phi_block[item.first], sizeof(double)*M));
        }
        
        // Total partition functions for each polymer
        single_polymer_partitions = new double[molecules->get_n_polymer_types()];

        // Remember one segment for each polymer chain to compute total partition function
        int current_p = 0;
        for(const auto& block: d_phi_block)
        {
            const auto& key = block.first;
            int p                = std::get<0>(key);
            std::string dep_v    = std::get<1>(key);
            std::string dep_u    = std::get<2>(key);

            // Skip if already found one segment
            if (p != current_p)
                continue;

            int n_aggregated;
            int n_segment_offset    = propagators_analyzer->get_essential_block(key).n_segment_offset;
            int n_segment_original  = propagators_analyzer->get_essential_block(key).n_segment_original;

            // Contains no '['
            if (dep_u.find('[') == std::string::npos)
                n_aggregated = 1;
            else
                n_aggregated = propagators_analyzer->get_essential_block(key).v_u.size();

            single_partition_segment.push_back(std::make_tuple(
                p,
                d_propagator[dep_v][n_segment_original-n_segment_offset], // q
                d_propagator[dep_u][0],                                   // q_dagger
                n_aggregated                    // how many propagators are aggregated
                ));
            current_p++;
        }

        // Total partition functions for each solvent
        single_solvent_partitions = new double[molecules->get_n_solvent_types()];

        // Concentrations for each solvent
        for(int s=0;s<molecules->get_n_solvent_types();s++)
        {
            double *d_phi_;
            gpu_error_check(cudaMalloc((void**)&d_phi_, sizeof(double)*M));
            d_phi_solvent.push_back(d_phi_);
        }

        // Create scheduler for computation of propagator
        sc = new Scheduler(propagators_analyzer->get_essential_propagator_codes(), N_SCHEDULER_STREAMS); 

        // Allocate memory for pseudo-spectral: advance_propagator()
        for(int gpu=0; gpu<N_GPUS; gpu++)
        {
            gpu_error_check(cudaSetDevice(gpu));
            gpu_error_check(cudaMalloc((void**)&d_q_mask[gpu], sizeof(double)*M));
        }
        if (N_GPUS > 1)
        {
            gpu_error_check(cudaSetDevice(1));
            gpu_error_check(cudaMalloc((void**)&d_propagator_device_1[0], sizeof(double)*M));  // prev
            gpu_error_check(cudaMalloc((void**)&d_propagator_device_1[1], sizeof(double)*M));  // next
        }

        gpu_error_check(cudaSetDevice(0));
        gpu_error_check(cudaMalloc((void**)&d_phi,           sizeof(double)*M));
        
        double q_unity[M];
        for(int i=0; i<M; i++)
            q_unity[i] = 1.0;
        gpu_error_check(cudaMalloc((void**)&d_q_unity, sizeof(double)*M));
        gpu_error_check(cudaMemcpy(d_q_unity, q_unity, sizeof(double)*M, cudaMemcpyHostToDevice));

        // Allocate memory for stress calculation: compute_stress()
        for(int gpu=0; gpu<N_GPUS; gpu++)
        {
            gpu_error_check(cudaSetDevice(gpu));
            gpu_error_check(cudaMalloc((void**)&d_stress_q[gpu][0],     sizeof(double)*2*M)); // prev
            gpu_error_check(cudaMalloc((void**)&d_stress_q[gpu][1],     sizeof(double)*2*M)); // next
        }

        propagator_solver->update_bond_function();
        gpu_error_check(cudaSetDevice(0));
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
CudaSolverContinuous::~CudaSolverContinuous()
{
    const int N_GPUS = CudaCommon::get_instance().get_n_gpus();
    
    delete sc;

    delete[] single_polymer_partitions;
    delete[] single_solvent_partitions;

    for(const auto& item: d_propagator)
    {
        for(int i=0; i<propagator_size[item.first]; i++)
            cudaFree(item.second[i]);
        delete[] item.second;
    }
    for(const auto& item: d_phi_block)
        cudaFree(item.second);
    for(const auto& item: d_phi_solvent)
        cudaFree(item);

    #ifndef NDEBUG
    for(const auto& item: propagator_finished)
        delete[] item.second;
    #endif

    cudaFree(d_phi);
    cudaFree(d_q_unity);

    // For pseudo-spectral: advance_propagator()
    for(int gpu=0; gpu<N_GPUS; gpu++)
    {
        cudaFree(d_q_mask[gpu]);
    }

    if (N_GPUS > 1)
    {
        cudaFree(d_propagator_device_1[0]);
        cudaFree(d_propagator_device_1[1]);
    }

    // For stress calculation: compute_stress()
    for(int gpu=0; gpu<N_GPUS; gpu++)
    {
        cudaFree(d_stress_q[gpu][0]);
        cudaFree(d_stress_q[gpu][1]);
    }
    
    // Destroy streams
    for(int gpu=0; gpu<N_GPUS; gpu++)
    {
        cudaStreamDestroy(streams[gpu][0]);
        cudaStreamDestroy(streams[gpu][1]);
    }
}

void CudaSolverContinuous::update_bond_function()
{
    try{
        propagator_solver->update_bond_function();
    }
    catch(std::exception& exc)
    {
        throw_with_line_number(exc.what());
    }
}
void CudaSolverContinuous::compute_statistics(
    std::string device,
    std::map<std::string, const double*> w_input,
    std::map<std::string, const double*> q_init,
    double* q_mask)
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

        for(const auto& item: propagators_analyzer->get_essential_propagator_codes())
        {
            if( w_input.find(item.second.monomer_type) == w_input.end())
                throw_with_line_number("monomer_type \"" + item.second.monomer_type + "\" is not in w_input.");
        }

        // Copy q_mask to d_q_mask
        for(int gpu=0; gpu<N_GPUS; gpu++)
        {
            gpu_error_check(cudaSetDevice(gpu));
            if (q_mask != nullptr)
            {
                gpu_error_check(cudaMemcpy(d_q_mask[gpu], q_mask, sizeof(double)*M, cudaMemcpyInputToDevice));
            }
            else
            {
                d_q_mask[gpu] = nullptr;
            }
        }

        // Update dw or d_exp_dw
        propagator_solver->update_dw(device, w_input);

        gpu_error_check(cudaSetDevice(0));
        if(q_mask == nullptr)
        {
            this->accessible_volume = cb->get_volume();
        }
        else
        {
            this->accessible_volume = cb->integral_device(d_q_mask[0]);
        }
       
        auto& branch_schedule = sc->get_schedule();
        // // display all jobs
        // int time_span_count=0;
        // For (auto parallel_job = branch_schedule.begin(); parallel_job != branch_schedule.end(); parallel_job++)
        // {
        //     std::cout << "jobs: " << time_span_count << std::endl;
        //     for(int job=0; job<parallel_job->size(); job++)
        //     {
        //         auto& key = std::get<0>((*parallel_job)[job]);
        //         int n_segment_from = std::get<1>((*parallel_job)[job]);
        //         int n_segment_to = std::get<2>((*parallel_job)[job]);
        //         std::cout << "key, n_segment_from, n_segment_to: " + key + ", " + std::to_string(n_segment_from) + ", " + std::to_string(n_segment_to) + ". " << std::endl;
        //     }
        //     time_span_count++;
        // }
        // Time_span_count=0;

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
                auto& deps = propagators_analyzer->get_essential_propagator_code(key).deps;
                auto monomer_type = propagators_analyzer->get_essential_propagator_code(key).monomer_type;

                // Check key
                #ifndef NDEBUG
                if (d_propagator.find(key) == d_propagator.end())
                    throw_with_line_number("Could not find key '" + key + "'. ");
                #endif

                double **_d_propagator = d_propagator[key];

                // If it is leaf node
                if(n_segment_from == 1 && deps.size() == 0)
                {
                     // q_init
                    if (key[0] == '{')
                    {
                        std::string g = PropagatorCode::get_q_input_idx_from_key(key);
                        if (q_init.find(g) == q_init.end())
                            throw_with_line_number("Could not find q_init[\"" + g + "\"].");
                        gpu_error_check(cudaMemcpy(_d_propagator[0], q_init[g],
                            sizeof(double)*M, cudaMemcpyInputToDevice));
                    }
                    else
                    {
                        gpu_error_check(cudaMemcpy(_d_propagator[0], d_q_unity,
                            sizeof(double)*M, cudaMemcpyDeviceToDevice));
                    }

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
                        gpu_error_check(cudaMemset(_d_propagator[0], 0, sizeof(double)*M));

                        // Add all propagators at junction if necessary 
                        for(size_t d=0; d<deps.size(); d++)
                        {
                            std::string sub_dep = std::get<0>(deps[d]);
                            int sub_n_segment   = std::get<1>(deps[d]);
                            int sub_n_repeated  = std::get<2>(deps[d]);

                            // Check sub key
                            #ifndef NDEBUG
                            if (d_propagator.find(sub_dep) == d_propagator.end())
                                throw_with_line_number("Could not find sub key '" + sub_dep + "'. ");
                            if (!propagator_finished[sub_dep][sub_n_segment])
                                throw_with_line_number("Could not compute '" + key +  "', since '"+ sub_dep + std::to_string(sub_n_segment) + "' is not prepared.");
                            #endif

                            lin_comb<<<N_BLOCKS, N_THREADS>>>(
                                _d_propagator[0], 1.0, _d_propagator[0],
                                sub_n_repeated, d_propagator[sub_dep][sub_n_segment], M);
                        }

                        #ifndef NDEBUG
                        propagator_finished[key][0] = true;
                        #endif
                    }
                    else
                    {
                        // Initialize to one
                        gpu_error_check(cudaMemcpy(_d_propagator[0], d_q_unity,
                            sizeof(double)*M, cudaMemcpyDeviceToDevice));

                        // Multiply all propagators at junction if necessary 
                        for(size_t d=0; d<deps.size(); d++)
                        {
                            std::string sub_dep = std::get<0>(deps[d]);
                            int sub_n_segment   = std::get<1>(deps[d]);

                            // Check sub key
                            #ifndef NDEBUG
                            if (d_propagator.find(sub_dep) == d_propagator.end())
                                throw_with_line_number("Could not find sub key '" + sub_dep + "'. ");
                            if (!propagator_finished[sub_dep][sub_n_segment])
                                throw_with_line_number("Could not compute '" + key +  "', since '"+ sub_dep + std::to_string(sub_n_segment) + "' is not prepared.");
                            #endif

                            multi_real<<<N_BLOCKS, N_THREADS>>>(
                                _d_propagator[0], _d_propagator[0],
                                d_propagator[sub_dep][sub_n_segment], 1.0, M);
                        }
                        
                        #ifndef NDEBUG
                        propagator_finished[key][0] = true;
                        #endif
                    }
                }

                // Multiply mask
                if (d_q_mask[0] != nullptr)
                    multi_real<<<N_BLOCKS, N_THREADS>>>(_d_propagator[0], _d_propagator[0], d_q_mask[0], 1.0, M);

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
            if(non_zero_segment_jobs.size()==1)
            {
                gpu_error_check(cudaSetDevice(0));
                auto& key = std::get<0>(non_zero_segment_jobs[0]);
                int n_segment_from = std::get<1>(non_zero_segment_jobs[0]);
                int n_segment_to = std::get<2>(non_zero_segment_jobs[0]);
                auto monomer_type = propagators_analyzer->get_essential_propagator_code(key).monomer_type;
                double **_d_propagator_key = d_propagator[key];

                for(int n=n_segment_from; n<=n_segment_to; n++)
                {
                    #ifndef NDEBUG
                    if (!propagator_finished[key][n-1])
                        throw_with_line_number("unfinished, key: " + key + ", " + std::to_string(n-1));
                    #endif

                    propagator_solver->advance_one_propagator_continuous(0, 
                        _d_propagator_key[n-1],
                        _d_propagator_key[n],
                        monomer_type, d_q_mask[0]);

                    #ifndef NDEBUG
                    propagator_finished[key][n] = true;
                    #endif
                }
            }
            else if(non_zero_segment_jobs.size()==2)
            {
                const int N_JOBS = 2;
                std::string keys[N_JOBS];
                int n_segment_froms[N_JOBS];
                int n_segment_tos[N_JOBS];
                std::string monomer_types[N_JOBS];
                double **_d_propagator_keys[N_JOBS];
                
                for(int j=0; j<N_JOBS; j++)
                {
                    keys[j] = std::get<0>(non_zero_segment_jobs[j]);
                    n_segment_froms[j] = std::get<1>(non_zero_segment_jobs[j]);
                    n_segment_tos[j] = std::get<2>(non_zero_segment_jobs[j]);
                    monomer_types[j] = propagators_analyzer->get_essential_propagator_code(keys[j]).monomer_type;
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

                    for(int n=0; n<=n_segment_tos[0]-n_segment_froms[0]; n++)
                    {
                        #ifndef NDEBUG
                        if (!propagator_finished[keys[0]][n-1+n_segment_froms[0]])
                            throw_with_line_number("unfinished, key: " + keys[0] + ", " + std::to_string(n-1+n_segment_froms[0]));
                        if (!propagator_finished[keys[1]][n-1+n_segment_froms[1]])
                            throw_with_line_number("unfinished, key: " + keys[1] + ", " + std::to_string(n-1+n_segment_froms[1]));
                        #endif

                        // DEVICE 0,1, STREAM 0: calculate propagators 
                        propagator_solver->advance_two_propagators_continuous_two_gpus(
                            _d_propagator_keys[0][n-1+n_segment_froms[0]],
                            d_propagator_device_1[prev],
                            _d_propagator_keys[0][n+n_segment_froms[0]],
                            d_propagator_device_1[next],
                            monomer_types[0], monomer_types[1], d_q_mask);

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
                        _d_propagator_keys[1][n_segment_tos[1]],
                        d_propagator_device_1[prev],
                        sizeof(double)*M, cudaMemcpyDeviceToDevice));
                }
                else
                {
                    gpu_error_check(cudaSetDevice(0));
                    for(int n=0; n<=n_segment_tos[0]-n_segment_froms[0]; n++)
                    {
                        #ifndef NDEBUG
                        if (!propagator_finished[keys[0]][n-1+n_segment_froms[0]])
                            throw_with_line_number("unfinished, key: " + keys[0] + ", " + std::to_string(n-1+n_segment_froms[0]));
                        if (!propagator_finished[keys[1]][n-1+n_segment_froms[1]])
                            throw_with_line_number("unfinished, key: " + keys[1] + ", " + std::to_string(n-1+n_segment_froms[1]));
                        #endif

                        propagator_solver->advance_two_propagators_continuous(
                            _d_propagator_keys[0][n-1+n_segment_froms[0]],
                            _d_propagator_keys[1][n-1+n_segment_froms[1]],
                            _d_propagator_keys[0][n+n_segment_froms[0]],
                            _d_propagator_keys[1][n+n_segment_froms[1]],
                            monomer_types[0], monomer_types[1],
                            d_q_mask[0]);

                        #ifndef NDEBUG
                        propagator_finished[keys[0]][n+n_segment_froms[0]] = true;
                        propagator_finished[keys[1]][n+n_segment_froms[1]] = true;
                        #endif
                    }
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
            int p                  = std::get<0>(segment_info);
            double *d_propagator_v = std::get<1>(segment_info);
            double *d_propagator_u = std::get<2>(segment_info);
            int n_aggregated       = std::get<3>(segment_info);

            single_polymer_partitions[p] = cb->inner_product_device(
                d_propagator_v, d_propagator_u)/n_aggregated/this->accessible_volume;
        }

        // Calculate segment concentrations
        for(const auto& d_block: d_phi_block)
        {
            const auto& key = d_block.first;
            int p                = std::get<0>(key);
            std::string dep_v    = std::get<1>(key);
            std::string dep_u    = std::get<2>(key);

            int n_repeated;
            int n_segment_allocated = propagators_analyzer->get_essential_block(key).n_segment_allocated;
            int n_segment_offset    = propagators_analyzer->get_essential_block(key).n_segment_offset;
            int n_segment_original  = propagators_analyzer->get_essential_block(key).n_segment_original;

            // If there is no segment
            if(n_segment_allocated == 0)
            {
                gpu_error_check(cudaMemset(d_block.second, 0, sizeof(double)*M));
                continue;
            }

            // Check keys
            #ifndef NDEBUG
            if (d_propagator.find(dep_v) == d_propagator.end())
                throw_with_line_number("Could not find dep_v key'" + dep_v + "'. ");
            if (d_propagator.find(dep_u) == d_propagator.end())
                throw_with_line_number("Could not find dep_u key'" + dep_u + "'. ");
            #endif

            // Contains no '['
            if (dep_u.find('[') == std::string::npos)
                n_repeated = propagators_analyzer->get_essential_block(key).v_u.size();
            else
                n_repeated = 1;

            // Calculate phi of one block (possibly multiple blocks when using aggregation)
            calculate_phi_one_block(
                d_block.second,       // phi
                d_propagator[dep_v],  // dependency v
                d_propagator[dep_u],  // dependency u
                n_segment_allocated,
                n_segment_offset,
                n_segment_original);

            // Normalize concentration
            Polymer& pc = molecules->get_polymer(p);
            double norm = molecules->get_ds()*pc.get_volume_fraction()/pc.get_alpha()/single_polymer_partitions[p]*n_repeated;
            lin_comb<<<N_BLOCKS, N_THREADS>>>(d_block.second, norm, d_block.second, 0.0, d_block.second, M);
        }

        // Calculate partition functions and concentrations of solvents
        for(size_t s=0; s<molecules->get_n_solvent_types(); s++)
        {
            double *d_phi_ = d_phi_solvent[s];
            double volume_fraction = std::get<0>(molecules->get_solvent(s));
            std::string monomer_type = std::get<1>(molecules->get_solvent(s));
            double *_d_exp_dw = propagator_solver->d_exp_dw[0][monomer_type];

            single_solvent_partitions[s] = cb->inner_product_device(_d_exp_dw, _d_exp_dw)/this->accessible_volume;
            multi_real<<<N_BLOCKS, N_THREADS>>>(d_phi_,_d_exp_dw, _d_exp_dw, volume_fraction/single_solvent_partitions[s], M);
        }
        gpu_error_check(cudaSetDevice(0));
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaSolverContinuous::calculate_phi_one_block(
    double *d_phi, double **d_q_1, double **d_q_2, const int N, const int N_OFFSET, const int N_ORIGINAL)
{
    try
    {
        gpu_error_check(cudaSetDevice(0));

        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = cb->get_n_grid();
        std::vector<double> simpson_rule_coeff = SimpsonRule::get_coeff(N);

        // Compute segment concentration
        multi_real<<<N_BLOCKS, N_THREADS>>>(d_phi, d_q_1[N_ORIGINAL-N_OFFSET], d_q_2[0], simpson_rule_coeff[0], M);
        for(int n=1; n<=N; n++)
        {
            add_multi_real<<<N_BLOCKS, N_THREADS>>>(d_phi, d_q_1[N_ORIGINAL-N_OFFSET-n], d_q_2[n], simpson_rule_coeff[n], M);
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
double CudaSolverContinuous::get_total_partition(int polymer)
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
void CudaSolverContinuous::get_total_concentration(std::string monomer_type, double *phi)
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
        for(const auto& d_block: d_phi_block)
        {
            const auto& key = d_block.first;
            std::string dep_v = std::get<1>(key);
            int n_segment_allocated = propagators_analyzer->get_essential_block(key).n_segment_allocated;
            if (PropagatorCode::get_monomer_type_from_key(dep_v) == monomer_type && n_segment_allocated != 0)
                lin_comb<<<N_BLOCKS, N_THREADS>>>(d_phi, 1.0, d_phi, 1.0, d_block.second, M);
        }

        // For each solvent
        for(int s=0;s<molecules->get_n_solvent_types();s++)
        {
            if (std::get<1>(molecules->get_solvent(s)) == monomer_type)
                lin_comb<<<N_BLOCKS, N_THREADS>>>(d_phi, 1.0, d_phi, 1.0, d_phi_solvent[s], M);
        }
        gpu_error_check(cudaMemcpy(phi, d_phi, sizeof(double)*M, cudaMemcpyDeviceToHost));
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaSolverContinuous::get_total_concentration(int p, std::string monomer_type, double *phi)
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
        for(const auto& d_block: d_phi_block)
        {
            const auto& key = d_block.first;
            int polymer_idx = std::get<0>(key);
            std::string dep_v = std::get<1>(key);
            int n_segment_allocated = propagators_analyzer->get_essential_block(key).n_segment_allocated;
            if (polymer_idx == p && PropagatorCode::get_monomer_type_from_key(dep_v) == monomer_type && n_segment_allocated != 0)
                lin_comb<<<N_BLOCKS, N_THREADS>>>(d_phi, 1.0, d_phi, 1.0, d_block.second, M);
        }
        gpu_error_check(cudaMemcpy(phi, d_phi, sizeof(double)*M, cudaMemcpyDeviceToHost));
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaSolverContinuous::get_block_concentration(int p, double *phi)
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

        if (propagators_analyzer->is_using_propagator_aggregation())
            throw_with_line_number("Disable 'aggregation' option to invoke 'get_block_concentration'.");

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

            lin_comb<<<N_BLOCKS, N_THREADS>>>(d_phi, 0.0, d_phi, 1.0, d_phi_block[std::make_tuple(p, dep_v, dep_u)], M);
            gpu_error_check(cudaMemcpy(&phi[b*M], d_phi, sizeof(double)*M, cudaMemcpyDeviceToHost));
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
double CudaSolverContinuous::get_solvent_partition(int s)
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
void CudaSolverContinuous::get_solvent_concentration(int s, double *phi)
{
    try
    {
        gpu_error_check(cudaSetDevice(0));
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = cb->get_n_grid();
        const int S = molecules->get_n_solvent_types();

        if (s < 0 || s > S-1)
            throw_with_line_number("Index (" + std::to_string(s) + ") must be in range [0, " + std::to_string(S-1) + "]");

        gpu_error_check(cudaMemcpy(phi, d_phi_solvent[s], sizeof(double)*M, cudaMemcpyDeviceToHost));
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
std::vector<double> CudaSolverContinuous::compute_stress()
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
        for(const auto& block: d_phi_block)
        {
            const auto& key = block.first;
            int p                = std::get<0>(key);
            std::string dep_v    = std::get<1>(key);
            std::string dep_u    = std::get<2>(key);

            const int N           = propagators_analyzer->get_essential_block(key).n_segment_allocated;
            const int N_OFFSET    = propagators_analyzer->get_essential_block(key).n_segment_offset;
            const int N_ORIGINAL  = propagators_analyzer->get_essential_block(key).n_segment_original;
            std::string monomer_type = propagators_analyzer->get_essential_block(key).monomer_type;

            // If there is no segment
            if(N == 0)
                continue;

            // Contains no '['
            int n_repeated;
            if (dep_u.find('[') == std::string::npos)
                n_repeated = propagators_analyzer->get_essential_block(key).v_u.size();
            else
                n_repeated = 1;

            std::vector<double> s_coeff = SimpsonRule::get_coeff(N);
            double** d_q_1 = d_propagator[dep_v];    // dependency v
            double** d_q_2 = d_propagator[dep_u];    // dependency u

            std::array<double,3> _block_dq_dl[MAX_GPUS];
            std::vector<double> segment_stress[MAX_GPUS];
            for(int gpu=0; gpu<N_GPUS; gpu++)
                for(int d=0; d<3; d++)
                    _block_dq_dl[gpu][d] = 0.0;

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
                    gpu_error_check(cudaMemcpy(&d_stress_q[gpu][prev][0], d_q_1[N_ORIGINAL-N_OFFSET-idx],
                            sizeof(double)*M,cudaMemcpyDeviceToDevice));
                    gpu_error_check(cudaMemcpy(&d_stress_q[gpu][prev][M], d_q_2[idx],
                            sizeof(double)*M,cudaMemcpyDeviceToDevice));
                }
            }

            // Compute
            for(int n=0; n<=N; n+=N_GPUS)
            {
                // STREAM 1: copy data from device to device
                for(int gpu=0; gpu<N_GPUS; gpu++)
                {
                    // Index
                    const int idx = n + gpu;
                    const int idx_next = idx + N_GPUS;

                    gpu_error_check(cudaSetDevice(gpu));
                    if (idx_next <= N)
                    {
                        gpu_error_check(cudaMemcpyAsync(&d_stress_q[gpu][next][0], d_q_1[N_ORIGINAL-N_OFFSET-idx_next],
                                sizeof(double)*M,cudaMemcpyDeviceToDevice, streams[gpu][1]));
                        gpu_error_check(cudaMemcpyAsync(&d_stress_q[gpu][next][M], d_q_2[idx_next],
                                sizeof(double)*M,cudaMemcpyDeviceToDevice, streams[gpu][1]));
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
                // STREAM 0: execute kernels
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
        for(const auto& block: d_phi_block)
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
void CudaSolverContinuous::get_chain_propagator(double *q_out, int polymer, int v, int u, int n)
{
    // This method should be invoked after invoking compute_statistics()

    // Get chain propagator for a selected polymer, block and direction.
    // This is made for debugging and testing.
    try
    {
        const int M = cb->get_n_grid();
        Polymer& pc = molecules->get_polymer(polymer);
        std::string dep = pc.get_propagator_key(v,u);

        if (propagators_analyzer->get_essential_propagator_codes().find(dep) == propagators_analyzer->get_essential_propagator_codes().end())
            throw_with_line_number("Could not find the propagator code '" + dep + "'. Disable 'aggregation' option to obtain propagators_analyzer.");

        const int N = propagators_analyzer->get_essential_propagator_codes()[dep].max_n_segment;
        if (n < 0 || n > N)
            throw_with_line_number("n (" + std::to_string(n) + ") must be in range [0, " + std::to_string(N) + "]");

        gpu_error_check(cudaMemcpy(q_out, d_propagator[dep][n], sizeof(double)*M,cudaMemcpyDeviceToHost));
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
bool CudaSolverContinuous::check_total_partition()
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
    for(const auto& block: d_phi_block)
    {
        const auto& key = block.first;
        int p                = std::get<0>(key);
        std::string dep_v    = std::get<1>(key);
        std::string dep_u    = std::get<2>(key);

        int n_aggregated;
        int n_segment_allocated = propagators_analyzer->get_essential_block(key).n_segment_allocated;
        int n_segment_offset    = propagators_analyzer->get_essential_block(key).n_segment_offset;
        int n_segment_original  = propagators_analyzer->get_essential_block(key).n_segment_original;

        // std::cout<< p << ", " << dep_v << ", " << dep_u << ": " << n_segment_original << ", " << n_segment_offset << ", " << n_segment_allocated << std::endl;

        // Contains no '['
        if (dep_u.find('[') == std::string::npos)
            n_aggregated = 1;
        else
            n_aggregated = propagators_analyzer->get_essential_block(key).v_u.size();

        for(int n=0;n<=n_segment_allocated;n++)
        {
            double total_partition = cb->inner_product_device(
                d_propagator[dep_v][n_segment_original-n_segment_offset-n],
                d_propagator[dep_u][n])/n_aggregated/this->accessible_volume;

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