#include <complex>
#include <iostream>
#include <chrono>
#include <omp.h>

#include "CudaComputationBox.h"
#include "CudaComputationReduceMemoryDiscrete.h"
#include "CudaSolverPseudoDiscrete.h"

CudaComputationReduceMemoryDiscrete::CudaComputationReduceMemoryDiscrete(
    ComputationBox<double>* cb,
    Molecules *molecules,
    PropagatorComputationOptimizer *propagator_computation_optimizer)
    : PropagatorComputation(cb, molecules, propagator_computation_optimizer)
{
    try
    {
        #ifndef NDEBUG
        std::cout << "--------- Discrete Chain Solver, GPU Memoery Saving Version ---------" << std::endl;
        #endif

        const int M = this->cb->get_total_grid();
        const int N_GPUS = CudaCommon::get_instance().get_n_gpus();

        // The number of parallel streams for propagator computation
        const char *ENV_OMP_NUM_THREADS = getenv("OMP_NUM_THREADS");
        std::string env_omp_num_threads(ENV_OMP_NUM_THREADS ? ENV_OMP_NUM_THREADS  : "");
        if (env_omp_num_threads.empty())
            n_streams = MAX_STREAMS;
        else
            n_streams =  std::min(std::stoi(env_omp_num_threads), MAX_STREAMS);
        #ifndef NDEBUG
        std::cout << "The number of CPU threads: " << n_streams << std::endl;
        #endif

        // Copy streams
        for(int i=0; i<n_streams; i++)
        {
            gpu_error_check(cudaSetDevice(i % N_GPUS));
            gpu_error_check(cudaStreamCreate(&streams[i][0])); // for kernel execution
            gpu_error_check(cudaStreamCreate(&streams[i][1])); // for memcpy
        }

        this->propagator_solver = new CudaSolverPseudoDiscrete(cb, molecules, n_streams, streams, true);

        // Allocate memory for propagators
        if( this->propagator_computation_optimizer->get_computation_propagators().size() == 0)
            throw_with_line_number("There is no propagator code. Add polymers first.");
        for(const auto& item: this->propagator_computation_optimizer->get_computation_propagators())
        {
             // There are N segments
             // Example (N==5)
             // O--O--O--O--O
             // 1  2  3  4  5

             // Legend)
             // -- : full bond
             // O  : full segment

            std::string key = item.first;
            int max_n_segment = item.second.max_n_segment+1; 
            propagator_size[key] = max_n_segment;

            // Allocate memory for q(r,1/2)
            propagator_half_steps[key] = new double*[max_n_segment];
            propagator_half_steps[key][0] = nullptr;
            if (item.second.deps.size() > 0)
                gpu_error_check(cudaMallocHost((void**)&propagator_half_steps[key][0], sizeof(double)*M));

            // Allocate memory for q(r,s+1/2)
            for(int i=1; i<propagator_size[key]; i++)
            {
                propagator_half_steps[key][i] = nullptr;
                if (item.second.junction_ends.find(i) != item.second.junction_ends.end())
                    gpu_error_check(cudaMallocHost((void**)&propagator_half_steps[key][i], sizeof(double)*M));
            }

            // Allocate memory for q(r,s)
            // Index 0 will be not used
            propagator[key] = new double*[max_n_segment];
            propagator[key][0] = nullptr;
            for(int i=1; i<propagator_size[key]; i++)
                gpu_error_check(cudaMallocHost((void**)&propagator[key][i], sizeof(double)*M));

            #ifndef NDEBUG
            propagator_finished[key] = new bool[max_n_segment];
            for(int i=0; i<max_n_segment;i++)
                propagator_finished[key][i] = false;
            for (int n: item.second.junction_ends)
                propagator_half_steps_finished[key][n] = false;
            #endif
        }

        // Allocate memory for concentrations
        if( this->propagator_computation_optimizer->get_computation_blocks().size() == 0)
            throw_with_line_number("There is no block. Add polymers first.");
        for(const auto& item: this->propagator_computation_optimizer->get_computation_blocks())
        {
            phi_block[item.first] = nullptr;
            gpu_error_check(cudaMallocHost((void**)&phi_block[item.first], sizeof(double)*M));
        }

        // Remember one segment for each polymer chain to compute total partition function
        int current_p = 0;
        for(const auto& block: phi_block)
        {
            const auto& key = block.first;
            int p                 = std::get<0>(key);
            std::string key_left  = std::get<1>(key);
            std::string key_right = std::get<2>(key);

            // Skip if already found one segment
            if (p != current_p)
                continue;

            int n_aggregated = this->propagator_computation_optimizer->get_computation_block(key).v_u.size()/
                               this->propagator_computation_optimizer->get_computation_block(key).n_repeated;
            int n_segment_left = this->propagator_computation_optimizer->get_computation_block(key).n_segment_left;
            std::string monomer_type = this->propagator_computation_optimizer->get_computation_block(key).monomer_type;

            // Skip if n_segment_left is 0
            if (n_segment_left == 0)
                continue;

            single_partition_segment.push_back(std::make_tuple(
                p,
                propagator[key_left][n_segment_left],    // q
                propagator[key_right][1],                  // q_dagger
                monomer_type,       
                n_aggregated                               // how many propagators are aggregated
                ));
            current_p++;
        }

       // Find propagators and bond length for each segment to prepare stress computation
        for(const auto& block: phi_block)
        {
            const auto& key = block.first;
            int p                 = std::get<0>(key);
            std::string key_left  = std::get<1>(key);
            std::string key_right = std::get<2>(key);

            const int N_RIGHT = this->propagator_computation_optimizer->get_computation_block(key).n_segment_right;
            const int N_LEFT  = this->propagator_computation_optimizer->get_computation_block(key).n_segment_left;

            // If there is no segment
            if(N_RIGHT == 0)
                continue;

            double **q_1 = propagator[key_left];     // dependency v
            double **q_2 = propagator[key_right];    // dependency u

            auto& _block_stress_compuation_key = block_stress_computation_plan[key];

            // Find propagators and bond length
            for(int n=0; n<=N_RIGHT; n++)
            {
                double *propagator_left  = nullptr;
                double *propagator_right = nullptr;
                bool is_half_bond_length = false;

                // At v
                if (n == N_LEFT)
                {
                    if (this->propagator_computation_optimizer->get_computation_propagator(key_left).deps.size() == 0) // if v is leaf node, skip
                    {
                        _block_stress_compuation_key.push_back(std::make_tuple(propagator_left, propagator_right, is_half_bond_length));
                        continue;
                    }
                    
                    propagator_left  = propagator_half_steps[key_left][0];
                    propagator_right = q_2[N_RIGHT];
                    is_half_bond_length = true;
                }
                // At u
                else if (n == 0 && N_LEFT == N_RIGHT){
                    if (this->propagator_computation_optimizer->get_computation_propagator(key_right).deps.size() == 0) // if u is leaf node, skip
                    {
                        _block_stress_compuation_key.push_back(std::make_tuple(propagator_left, propagator_right, is_half_bond_length));
                        continue;
                    }

                    propagator_left  = q_1[N_LEFT];
                    propagator_right = propagator_half_steps[key_right][0];
                    is_half_bond_length = true;
                }
                // At aggregation junction
                else if (n == 0)
                {
                    _block_stress_compuation_key.push_back(std::make_tuple(propagator_left, propagator_right, is_half_bond_length));
                    continue;
                }
                // Within the blocks
                else
                {
                    propagator_left  = q_1[N_LEFT-n];
                    propagator_right = q_2[n];
                    is_half_bond_length = false;
                }
                _block_stress_compuation_key.push_back(std::make_tuple(propagator_left, propagator_right, is_half_bond_length));
            }
        }

        // Concentrations for each solvent
        for(int s=0;s<this->molecules->get_n_solvent_types();s++)
            phi_solvent.push_back(new double[M]);

        // Create scheduler for computation of propagator
        sc = new Scheduler(this->propagator_computation_optimizer->get_computation_propagators(), n_streams); 

        // Allocate memory for pseudo-spectral: advance_propagator()
        double q_unity[M];
        for(int i=0; i<M; i++)
            q_unity[i] = 1.0;
        for(int gpu=0; gpu<N_GPUS; gpu++)
        {
            gpu_error_check(cudaSetDevice(gpu));
            gpu_error_check(cudaMalloc((void**)&d_q_unity[gpu], sizeof(double)*M));
            gpu_error_check(cudaMemcpy(d_q_unity[gpu], q_unity, sizeof(double)*M, cudaMemcpyHostToDevice));
        }

        // Allocate memory for propagator computation
        for(int i=0; i<n_streams; i++)
        {
            gpu_error_check(cudaSetDevice(i % N_GPUS));
            gpu_error_check(cudaMalloc((void**)&d_q_one[i][0], sizeof(double)*M)); // for prev
            gpu_error_check(cudaMalloc((void**)&d_q_one[i][1], sizeof(double)*M)); // for next
            gpu_error_check(cudaMalloc((void**)&d_propagator_sub_dep[i][0], sizeof(double)*M)); // for prev
            gpu_error_check(cudaMalloc((void**)&d_propagator_sub_dep[i][1], sizeof(double)*M)); // for next
        }

        gpu_error_check(cudaSetDevice(0));
        // For concentration computation
        gpu_error_check(cudaMalloc((void**)&d_q_block_v[0], sizeof(double)*M)); // for prev
        gpu_error_check(cudaMalloc((void**)&d_q_block_v[1], sizeof(double)*M)); // for next
        gpu_error_check(cudaMalloc((void**)&d_q_block_u[0], sizeof(double)*M)); // for prev
        gpu_error_check(cudaMalloc((void**)&d_q_block_u[1], sizeof(double)*M)); // for next
        gpu_error_check(cudaMalloc((void**)&d_phi,          sizeof(double)*M));

        // Allocate memory for stress calculation: compute_stress()
        for(int i=0; i<n_streams; i++)
        {
            gpu_error_check(cudaSetDevice(i % N_GPUS));
            gpu_error_check(cudaMalloc((void**)&d_q_pair[i][0], sizeof(double)*2*M)); // prev
            gpu_error_check(cudaMalloc((void**)&d_q_pair[i][1], sizeof(double)*2*M)); // next
        }

        // Copy mask to d_q_mask
        for(int gpu=0; gpu<N_GPUS; gpu++)
        {
            gpu_error_check(cudaSetDevice(gpu));
            if (this->cb->get_mask() != nullptr)
            {
                gpu_error_check(cudaMalloc((void**)&d_q_mask [gpu], sizeof(double)*M));
                gpu_error_check(cudaMemcpy(d_q_mask[gpu], this->cb->get_mask(), sizeof(double)*M, cudaMemcpyHostToDevice));
            }
            else
                d_q_mask[gpu] = nullptr;
        }

        propagator_solver->update_laplacian_operator();
        gpu_error_check(cudaSetDevice(0));
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
CudaComputationReduceMemoryDiscrete::~CudaComputationReduceMemoryDiscrete()
{
    const int N_GPUS = CudaCommon::get_instance().get_n_gpus();

    delete propagator_solver;
    delete sc;

    for(const auto& item: propagator)
    {
        for(int i=0; i<propagator_size[item.first]; i++)
        {
            if(item.second[i] != nullptr)
                cudaFreeHost(item.second[i]);
        }
        delete[] item.second;
    }
    for(const auto& item: propagator_half_steps)
    {
        for(int i=0; i<propagator_size[item.first]; i++)
        {
            if(item.second[i] != nullptr)
                cudaFreeHost(item.second[i]);
        }
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

    // For pseudo-spectral: advance_one_propagator()
    for(int i=0; i<n_streams; i++)
    {
        cudaFree(d_q_one[i][0]); // for prev
        cudaFree(d_q_one[i][1]); // for next
        cudaFree(d_propagator_sub_dep[i][0]); // for prev
        cudaFree(d_propagator_sub_dep[i][1]); // for next
    }

    // For stress calculation: compute_stress()
    for(int i=0; i<n_streams; i++)
    {
        cudaFree(d_q_pair[i][0]);
        cudaFree(d_q_pair[i][1]);
    }

    // For concentration computation
    cudaFree(d_q_block_v[0]);
    cudaFree(d_q_block_v[1]);
    cudaFree(d_q_block_u[0]);
    cudaFree(d_q_block_u[1]);
    cudaFree(d_phi);

    // For pseudo-spectral: advance_propagator()
    for(int gpu=0; gpu<N_GPUS; gpu++)
    {
        if (d_q_mask[gpu] != nullptr)
            cudaFree(d_q_mask[gpu]);
        cudaFree(d_q_unity[gpu]);
    }
    
    // Destroy streams
    for(int i=0; i<n_streams; i++)
    {
        cudaStreamDestroy(streams[i][0]);
        cudaStreamDestroy(streams[i][1]);
    }
}

void CudaComputationReduceMemoryDiscrete::update_laplacian_operator()
{
    try
    {
        propagator_solver->update_laplacian_operator();
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

void CudaComputationReduceMemoryDiscrete::compute_statistics(
    std::map<std::string, const double*> w_input,
    std::map<std::string, const double*> q_init)
{
    this->compute_propagators(w_input, q_init);
    this->compute_concentrations();
}

void CudaComputationReduceMemoryDiscrete::compute_propagators(
    std::map<std::string, const double*> w_input,
    std::map<std::string, const double*> q_init)
{
    try{
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();
        const int N_GPUS = CudaCommon::get_instance().get_n_gpus();

        const int M = this->cb->get_total_grid();
        const double ds = this->molecules->get_ds();

        std::string device = "cpu";
        cudaMemcpyKind cudaMemcpyInputToDevice;
        if (device == "gpu")
            cudaMemcpyInputToDevice = cudaMemcpyDeviceToDevice;
        else if(device == "cpu")
            cudaMemcpyInputToDevice = cudaMemcpyHostToDevice;
        else
        {
            throw_with_line_number("Invalid device \"" + device + "\".");
        }

        for(const auto& item: this->propagator_computation_optimizer->get_computation_propagators())
        {
            if( w_input.find(item.second.monomer_type) == w_input.end())
                throw_with_line_number("monomer_type \"" + item.second.monomer_type + "\" is not in w_input.");
        }

        // Update dw or d_exp_dw
        propagator_solver->update_dw(device, w_input);

        // For each time span
        auto& branch_schedule = sc->get_schedule();
        for (auto parallel_job = branch_schedule.begin(); parallel_job != branch_schedule.end(); parallel_job++)
        {
            // // display all jobs
            // #ifndef NDEBUG
            // std::cout << "jobs:" << std::endl;
            // for(size_t job=0; job<parallel_job->size(); job++)
            // {
            //     auto& key = std::get<0>((*parallel_job)[job]);
            //     int n_segment_from = std::get<1>((*parallel_job)[job]);
            //     int n_segment_to = std::get<2>((*parallel_job)[job]);
            //     std::cout << "key, n_segment_from, n_segment_to: " + key + ", " + std::to_string(n_segment_from) + ", " + std::to_string(n_segment_to) + ". " << std::endl;
            //     std::cout << "half_steps: ";
            //     std::cout << "{";
            //     for (auto it = propagator_half_steps[key].begin(); it != propagator_half_steps[key].end(); ++it)
            //     {
            //         std::cout << it->first+1;
            //         if (std::next(it) != propagator_half_steps[key].end()) {
            //             std::cout << ", ";
            //         }
            //     }
            //     std::cout << "}, "<< std::endl;
            // }
            // auto start_time = std::chrono::duration_cast<std::chrono::microseconds>
            //     (std::chrono::system_clock::now().time_since_epoch()).count();
            // #endif

            // For each propagator
            #pragma omp parallel for num_threads(n_streams)
            for(size_t job=0; job<parallel_job->size(); job++)
            {
                const int STREAM = omp_get_thread_num();
                int gpu = omp_get_thread_num() % N_GPUS;
                gpu_error_check(cudaSetDevice(gpu));

                // printf("gpu, STREAM: %d, %d\n ", gpu, STREAM);

                auto& key = std::get<0>((*parallel_job)[job]);
                int n_segment_from = std::get<1>((*parallel_job)[job]);
                int n_segment_to = std::get<2>((*parallel_job)[job]);
                auto& deps = this->propagator_computation_optimizer->get_computation_propagator(key).deps;
                auto monomer_type = this->propagator_computation_optimizer->get_computation_propagator(key).monomer_type;

                // std::cout << "gpu, STREAM, key, n_segment_from, n_segment_to, monomer_type: "
                // if (STREAM == 0)
                //     std::cout << gpu << ", " << STREAM << ", " << n_segment_from << ", " << n_segment_to << ", " << monomer_type << ", " << key << ", "  << std::endl;    

                // #ifndef NDEBUG
                // #pragma omp critical
                // std::cout << job << " started, stream: " << STREAM << ", " <<
                //     std::chrono::duration_cast<std::chrono::microseconds>
                //     (std::chrono::system_clock::now().time_since_epoch()).count() - start_time << std::endl;
                // #endif

                // Check key
                #ifndef NDEBUG
                if (propagator.find(key) == propagator.end())
                    std::cout<< "Could not find key '" + key + "'. " << std::endl;
                #endif

                double **_propagator = propagator[key];
                double *_d_exp_dw = propagator_solver->d_exp_dw[gpu][monomer_type];

                // Calculate one block end
                if(n_segment_from == 0 && deps.size() == 0) // if it is leaf node
                {
                    // #ifndef NDEBUG
                    // #pragma omp critical
                    // std::cout << job << " init 1, " << 
                    //     std::chrono::duration_cast<std::chrono::microseconds>
                    //     (std::chrono::system_clock::now().time_since_epoch()).count() - start_time << std::endl;
                    // #endif

                    // q_init
                    if (key[0] == '{')
                    {
                        std::string g = PropagatorCode::get_q_input_idx_from_key(key);
                        if (q_init.find(g) == q_init.end())
                            std::cout<<  "Could not find q_init[\"" + g + "\"]." << std::endl;
                        gpu_error_check(cudaMemcpy(d_q_one[STREAM][0], q_init[g], sizeof(double)*M, cudaMemcpyInputToDevice));
                        ker_multi<double><<<N_BLOCKS, N_THREADS>>>(d_q_one[STREAM][0], d_q_one[STREAM][0], _d_exp_dw, 1.0, M);
                    }
                    else
                    {
                        gpu_error_check(cudaMemcpy(d_q_one[STREAM][0], _d_exp_dw, sizeof(double)*M, cudaMemcpyDeviceToDevice));
                    }

                    #ifndef NDEBUG
                    propagator_finished[key][1] = true;
                    #endif
                }
                else if (n_segment_from == 0 && deps.size() > 0) // if it is not leaf node
                {
                    // If it is aggregated
                    if (key[0] == '[')
                    {
                        // #ifndef NDEBUG
                        // #pragma omp critical
                        // std::cout << job << " init 2, " << 
                        //     std::chrono::duration_cast<std::chrono::microseconds>
                        //     (std::chrono::system_clock::now().time_since_epoch()).count() - start_time << std::endl;
                        // #endif

                        // Initialize to zero
                        gpu_error_check(cudaMemset(d_q_one[STREAM][0], 0, sizeof(double)*M));

                        int prev, next;
                        prev = 0;
                        next = 1;

                        // Copy memory from host to device
                        std::string sub_dep = std::get<0>(deps[0]);
                        int sub_n_segment   = std::get<1>(deps[0]);
                        int sub_n_repeated;

                        if (sub_n_segment == 0)
                        {
                            gpu_error_check(cudaMemcpy(d_propagator_sub_dep[STREAM][prev], propagator_half_steps[sub_dep][0], sizeof(double)*M, cudaMemcpyHostToDevice));
                        }
                        else
                        {
                            gpu_error_check(cudaMemcpy(d_propagator_sub_dep[STREAM][prev], propagator[sub_dep][sub_n_segment], sizeof(double)*M, cudaMemcpyHostToDevice));
                        }

                        for(size_t d=0; d<deps.size(); d++)
                        {
                            sub_dep         = std::get<0>(deps[d]);
                            sub_n_segment   = std::get<1>(deps[d]);
                            sub_n_repeated  = std::get<2>(deps[d]);
                            double **_propagator_sub_dep_next;

                            // STREAM 1: copy memory from host to device
                            if (d < deps.size()-1)
                            {
                                std::string sub_dep_next = std::get<0>(deps[d+1]);
                                int sub_n_segment_next   = std::get<1>(deps[d+1]);

                                if (sub_n_segment == 0)
                                {
                                    // Check sub key
                                    #ifndef NDEBUG
                                    if (propagator_half_steps.find(sub_dep_next) == propagator_half_steps.end())
                                        std::cout << "Could not find sub key '" + sub_dep_next + "'. " << std::endl;
                                    if (!propagator_half_steps_finished[sub_dep_next][0])
                                        std::cout << "Could not compute '" + key +  "', since '"+ sub_dep_next + std::to_string(0) + "' is not prepared." << std::endl;
                                    #endif

                                    _propagator_sub_dep_next = propagator_half_steps[sub_dep_next];
                                }
                                else
                                {
                                    // Check sub key
                                    #ifndef NDEBUG
                                    if (propagator.find(sub_dep_next) == propagator.end())
                                        std::cout<< "Could not find sub key '" + sub_dep_next + "'. " << std::endl;
                                    if (!propagator_finished[sub_dep_next][sub_n_segment_next])
                                        std::cout<< "Could not compute '" + key +  "', since '"+ sub_dep_next + std::to_string(sub_n_segment) + "' is not prepared." << std::endl;
                                    #endif

                                    _propagator_sub_dep_next = propagator[sub_dep_next];
                                }

                                gpu_error_check(cudaMemcpyAsync(d_propagator_sub_dep[STREAM][next],
                                                _propagator_sub_dep_next[sub_n_segment_next], sizeof(double)*M,
                                                cudaMemcpyHostToDevice, streams[STREAM][1]));
                            }

                            // STREAM 0: compute linear combination
                            ker_lin_comb<double><<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                                    d_q_one[STREAM][0], 1.0, d_q_one[STREAM][0],
                                    sub_n_repeated, d_propagator_sub_dep[STREAM][prev], M);

                            std::swap(prev, next);
                            cudaDeviceSynchronize();
                        }

                        // if sub_n_segment == 0
                        if (std::get<1>(deps[0]) == 0)
                        {
                            gpu_error_check(cudaMemcpyAsync(propagator_half_steps[key][0], d_q_one[STREAM][0], sizeof(double)*M, cudaMemcpyDeviceToHost, streams[STREAM][0]));

                            // Add half bond, STREAM 0
                            propagator_solver->advance_propagator_half_bond_step(
                                gpu, STREAM,
                                d_q_one[STREAM][0], d_q_one[STREAM][0], monomer_type);

                            // Add full segment
                            ker_multi<double><<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_q_one[STREAM][0], d_q_one[STREAM][0], _d_exp_dw, 1.0, M);
                        }
                        else
                        {
                            propagator_solver->advance_propagator(
                                gpu, STREAM,
                                d_q_one[STREAM][0], d_q_one[STREAM][0],
                                monomer_type, d_q_mask[gpu]);
                        }

                        #ifndef NDEBUG
                        propagator_finished[key][1] = true;
                        #endif
                    }
                    else
                    {
                        // #ifndef NDEBUG
                        // #pragma omp critical
                        // std::cout << job << " init 3, " << 
                        //     std::chrono::duration_cast<std::chrono::microseconds>
                        //     (std::chrono::system_clock::now().time_since_epoch()).count() - start_time << std::endl;
                        // #endif

                        // Combine branches
                        // Initialize to one
                        gpu_error_check(cudaMemcpy(d_q_one[STREAM][0], d_q_unity[gpu], sizeof(double)*M, cudaMemcpyDeviceToDevice));

                        int prev, next;
                        prev = 0;
                        next = 1;

                        // Copy memory from host to device
                        std::string sub_dep = std::get<0>(deps[0]);
                        int sub_n_segment   = std::get<1>(deps[0]);
                        gpu_error_check(cudaMemcpy(d_propagator_sub_dep[STREAM][prev], propagator_half_steps[sub_dep][sub_n_segment], sizeof(double)*M, cudaMemcpyHostToDevice));

                        for(size_t d=0; d<deps.size(); d++)
                        {
                            sub_dep       = std::get<0>(deps[d]);
                            sub_n_segment = std::get<1>(deps[d]);

                            // Check sub key
                            #ifndef NDEBUG
                            if (!propagator_half_steps_finished[sub_dep][sub_n_segment])
                                std::cout<< "Could not compute '" + key +  "', since '"+ sub_dep + std::to_string(sub_n_segment) + "+1/2' is not prepared." << std::endl;
                            #endif

                            // STREAM 1: copy memory from host to device
                            if (d < deps.size()-1)
                            {
                                std::string sub_dep_next = std::get<0>(deps[d+1]);
                                int sub_n_segment_next   = std::get<1>(deps[d+1]);

                                gpu_error_check(cudaMemcpyAsync(d_propagator_sub_dep[STREAM][next],
                                                propagator_half_steps[sub_dep_next][sub_n_segment_next], sizeof(double)*M,
                                                cudaMemcpyHostToDevice, streams[STREAM][1]));
                            }

                            // STREAM 0: multiply 
                            ker_multi<double><<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                                d_q_one[STREAM][0], d_q_one[STREAM][0], d_propagator_sub_dep[STREAM][prev], 1.0, M);

                            std::swap(prev, next);
                            cudaDeviceSynchronize();
                        }
                        gpu_error_check(cudaMemcpy(propagator_half_steps[key][0], d_q_one[STREAM][0], sizeof(double)*M,cudaMemcpyDeviceToHost));

                        #ifndef NDEBUG
                        propagator_half_steps_finished[key][0] = true;
                        #endif

                        if (n_segment_to > 0)
                        {
                            // Add half bond
                            propagator_solver->advance_propagator_half_bond_step(
                                gpu, STREAM,
                                d_q_one[STREAM][0], d_q_one[STREAM][0], monomer_type);

                            // Add full segment
                            ker_multi<double><<<N_BLOCKS, N_THREADS>>>(d_q_one[STREAM][0], d_q_one[STREAM][0], _d_exp_dw, 1.0, M);

                            #ifndef NDEBUG
                            propagator_finished[key][1] = true;
                            #endif
                        }
                    }
                }

                if (n_segment_to == 0)
                {
                    gpu_error_check(cudaStreamSynchronize(streams[STREAM][0]));
                    gpu_error_check(cudaStreamSynchronize(streams[STREAM][1]));
                    continue;
                }

                if (n_segment_from == 0)
                {
                    // Multiply mask
                    if (d_q_mask[gpu] != nullptr)
                        ker_multi<double><<<N_BLOCKS, N_THREADS>>>(d_q_one[STREAM][0], d_q_one[STREAM][0], d_q_mask[gpu], 1.0, M);

                    // Copy data between device and host
                    gpu_error_check(cudaMemcpy(_propagator[1], d_q_one[STREAM][0], sizeof(double)*M, cudaMemcpyDeviceToHost));

                    // q(r, 1+1/2)
                    if (propagator_half_steps[key][1] != nullptr)
                    {
                        #ifndef NDEBUG
                        if (propagator_half_steps_finished[key][1])
                            std::cout << "already half_step finished: " + key + ", " + std::to_string(1) << std::endl;
                        #endif

                        propagator_solver->advance_propagator_half_bond_step(
                            gpu, STREAM,
                            d_q_one[STREAM][0],
                            d_q_one[STREAM][1],
                            monomer_type);

                        gpu_error_check(cudaMemcpy(
                            propagator_half_steps[key][1],
                            d_q_one[STREAM][1],
                            sizeof(double)*M, cudaMemcpyDeviceToHost));

                        #ifndef NDEBUG
                        propagator_half_steps_finished[key][1] = true;
                        #endif
                    }
                    n_segment_from++;
                }
                else
                {
                    // Copy data between device and host
                    gpu_error_check(cudaMemcpy(d_q_one[STREAM][0], _propagator[n_segment_from], sizeof(double)*M, cudaMemcpyHostToDevice));
                }

                int prev, next;
                prev = 0;
                next = 1;

                // Create events
                cudaEvent_t kernel_done;
                cudaEvent_t memcpy_done;
                gpu_error_check(cudaEventCreate(&kernel_done));
                gpu_error_check(cudaEventCreate(&memcpy_done));

                // q(r,s)
                for(int n=n_segment_from; n<n_segment_to; n++)
                {
                    #ifndef NDEBUG
                    if (!propagator_finished[key][n])
                        std::cout << "unfinished, key: " + key + ", " + std::to_string(n) << std::endl;
                    if (propagator_finished[key][n+1])
                        std::cout << "already finished: " + key + ", " + std::to_string(n+1) << std::endl;
                    #endif

                    // #ifndef NDEBUG
                    // #pragma omp critical
                    // std::cout << job << " q_s, " << n << ", " << 
                    //     std::chrono::duration_cast<std::chrono::microseconds>
                    //     (std::chrono::system_clock::now().time_since_epoch()).count() - start_time << std::endl;
                    // #endif

                    // STREAM 0: calculate propagators
                    propagator_solver->advance_propagator(
                        gpu, STREAM, 
                        d_q_one[STREAM][prev],
                        d_q_one[STREAM][next],
                        monomer_type, d_q_mask[gpu]);
                    gpu_error_check(cudaEventRecord(kernel_done, streams[STREAM][0]));

                    // STREAM 1: copy memory from device to host
                    if (n > n_segment_from)
                    {
                        gpu_error_check(cudaMemcpyAsync(
                           _propagator[n],
                            d_q_one[STREAM][prev],
                            sizeof(double)*M, cudaMemcpyDeviceToHost, streams[STREAM][1]));
                        gpu_error_check(cudaEventRecord(memcpy_done, streams[STREAM][1]));
                    }

                    // Wait until computation and memory copy are done
                    gpu_error_check(cudaStreamWaitEvent(streams[STREAM][1], kernel_done, 0));
                    gpu_error_check(cudaStreamWaitEvent(streams[STREAM][0], memcpy_done, 0));

                    std::swap(prev, next);

                    #ifndef NDEBUG
                    propagator_finished[key][n+1] = true;
                    #endif
                }
                // Copy memory from device to host
                gpu_error_check(cudaMemcpyAsync(
                    _propagator[n_segment_to],
                    d_q_one[STREAM][prev],
                    sizeof(double)*M, cudaMemcpyDeviceToHost, streams[STREAM][1]));

                gpu_error_check(cudaEventRecord(memcpy_done, streams[STREAM][1]));
                gpu_error_check(cudaStreamWaitEvent(streams[STREAM][0], memcpy_done, 0));

                gpu_error_check(cudaEventDestroy(kernel_done));
                gpu_error_check(cudaEventDestroy(memcpy_done));

                // q(r, s+1/2)
                for(int n=n_segment_from; n<n_segment_to; n++)
                {
                    if (propagator_half_steps[key][n+1] != nullptr)
                    {
                        // #ifndef NDEBUG
                        // #pragma omp critical
                        // std::cout << job << " q_s+1/2, " << n << ", " << 
                        //     std::chrono::duration_cast<std::chrono::microseconds>
                        //     (std::chrono::system_clock::now().time_since_epoch()).count() - start_time << std::endl;
                        // #endif

                        #ifndef NDEBUG
                        if (propagator_half_steps_finished[key][n+1])
                            std::cout << "already half_step finished: " + key + ", " + std::to_string(n+1) << std::endl;
                        #endif

                        gpu_error_check(cudaMemcpyAsync(
                            d_q_one[STREAM][0], _propagator[n+1],
                            sizeof(double)*M, cudaMemcpyHostToDevice, streams[STREAM][0]));

                        propagator_solver->advance_propagator_half_bond_step(
                            gpu, STREAM,
                            d_q_one[STREAM][0],
                            d_q_one[STREAM][1],
                            monomer_type);

                        gpu_error_check(cudaMemcpyAsync(
                            propagator_half_steps[key][n+1],
                            d_q_one[STREAM][1],
                            sizeof(double)*M, cudaMemcpyDeviceToHost, streams[STREAM][0]));

                        #ifndef NDEBUG
                        propagator_half_steps_finished[key][n+1] = true;
                        #endif
                    }
                }
                gpu_error_check(cudaStreamSynchronize(streams[STREAM][0]));
                gpu_error_check(cudaStreamSynchronize(streams[STREAM][1]));
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
            double *propagator_left  = std::get<1>(segment_info);
            double *propagator_right = std::get<2>(segment_info);
            std::string monomer_type = std::get<3>(segment_info);
            int n_aggregated         = std::get<4>(segment_info);
            double *_d_exp_dw = propagator_solver->d_exp_dw[0][monomer_type];

            // Copy propagators from host to device
            gpu_error_check(cudaMemcpy(d_q_block_v[0], propagator_left,  sizeof(double)*M, cudaMemcpyHostToDevice));
            gpu_error_check(cudaMemcpy(d_q_block_u[0], propagator_right, sizeof(double)*M, cudaMemcpyHostToDevice));

            this->single_polymer_partitions[p] = this->cb->inner_product_inverse_weight_device(
                d_q_block_v[0],  // q
                d_q_block_u[0],  // q^dagger
                _d_exp_dw)/n_aggregated/this->cb->get_volume();
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

void CudaComputationReduceMemoryDiscrete::compute_concentrations()
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();
        const int N_GPUS = CudaCommon::get_instance().get_n_gpus();

        const int M = this->cb->get_total_grid();

        // Calculate segment concentrations
        for(const auto& block: phi_block)
        {
            const auto& key = block.first;
            int p                 = std::get<0>(key);
            std::string key_left  = std::get<1>(key);
            std::string key_right = std::get<2>(key);

            int n_segment_right      = this->propagator_computation_optimizer->get_computation_block(key).n_segment_right;
            int n_segment_left       = this->propagator_computation_optimizer->get_computation_block(key).n_segment_left;
            std::string monomer_type = this->propagator_computation_optimizer->get_computation_block(key).monomer_type;
            int n_repeated           = this->propagator_computation_optimizer->get_computation_block(key).n_repeated;
            double *_d_exp_dw = propagator_solver->d_exp_dw[0][monomer_type];

            // If there is no segment
            if(n_segment_right == 0)
            {
                for(int i=0; i<M;i++)
                    block.second[i] = 0.0;
                continue;
            }

            // Check keys
            #ifndef NDEBUG
            if (propagator.find(key_left) == propagator.end())
                throw_with_line_number("Could not find key_left key'" + key_left + "'. ");
            if (propagator.find(key_right) == propagator.end())
                throw_with_line_number("Could not find key_right key'" + key_right + "'. ");
            #endif

            // Normalize concentration
            Polymer& pc = this->molecules->get_polymer(p);
            double norm = this->molecules->get_ds()*pc.get_volume_fraction()/pc.get_alpha()/this->single_polymer_partitions[p]*n_repeated;

            // Calculate phi of one block (possibly multiple blocks when using aggregation)
            calculate_phi_one_block(
                block.second,               // phi
                propagator[key_left],       // dependency v
                propagator[key_right],      // dependency u
                _d_exp_dw,                  // exp_dw
                n_segment_right,
                n_segment_left,
                norm);
        }
        // Calculate partition functions and concentrations of solvents
        for(int s=0; s<this->molecules->get_n_solvent_types(); s++)
        {
            double volume_fraction   = std::get<0>(this->molecules->get_solvent(s));
            std::string monomer_type = std::get<1>(this->molecules->get_solvent(s));
            double *_d_exp_dw = propagator_solver->d_exp_dw[0][monomer_type];

            this->single_solvent_partitions[s] = this->cb->integral_device(_d_exp_dw)/this->cb->get_volume();
            ker_linear_scaling<double><<<N_BLOCKS, N_THREADS>>>(d_phi, _d_exp_dw, volume_fraction/this->single_solvent_partitions[s], 0.0, M);
            gpu_error_check(cudaMemcpy(phi_solvent[s], d_phi, sizeof(double)*M, cudaMemcpyDeviceToHost));
        }
        gpu_error_check(cudaSetDevice(0));
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaComputationReduceMemoryDiscrete::calculate_phi_one_block(
    double *phi, double **q_1, double **q_2, double *d_exp_dw, const int N_RIGHT, const int N_LEFT, const double NORM)
{
    try
    {
        gpu_error_check(cudaSetDevice(0));
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();
        const int M = this->cb->get_total_grid();

        int prev, next;
        prev = 0;
        next = 1;

        // Copy propagators from host to device
        gpu_error_check(cudaMemcpy(d_q_block_v[prev], q_1[N_LEFT], sizeof(double)*M, cudaMemcpyHostToDevice));
        gpu_error_check(cudaMemcpy(d_q_block_u[prev], q_2[1],      sizeof(double)*M, cudaMemcpyHostToDevice));

        // Initialize to zero
        gpu_error_check(cudaMemset(d_phi, 0, sizeof(double)*M));
 
        for(int n=1; n<=N_RIGHT; n++)
        {
            // STREAM 1: copy propagators from host to device
            if (n+1 <=N_RIGHT)
            {
                gpu_error_check(cudaMemcpyAsync(d_q_block_v[next], q_1[N_LEFT-(n+1)+1],
                    sizeof(double)*M, cudaMemcpyHostToDevice, streams[0][1]));
                gpu_error_check(cudaMemcpyAsync(d_q_block_u[next], q_2[(n+1)],
                    sizeof(double)*M, cudaMemcpyHostToDevice, streams[0][1]));
            }

            // STREAM 0: multiply two propagators
            ker_add_multi<double><<<N_BLOCKS, N_THREADS, 0, streams[0][0]>>>(d_phi, d_q_block_v[prev], d_q_block_u[prev], NORM, M);
            std::swap(prev, next);
            cudaDeviceSynchronize();
        }
        // divide exp_dw
        ker_divide<double><<<N_BLOCKS, N_THREADS>>>(d_phi, d_phi, d_exp_dw, 1.0, M);

        // Copy propagators from device to host
        gpu_error_check(cudaMemcpy(phi, d_phi, sizeof(double)*M, cudaMemcpyDeviceToHost));
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
double CudaComputationReduceMemoryDiscrete::get_total_partition(int polymer)
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
void CudaComputationReduceMemoryDiscrete::get_total_concentration(std::string monomer_type, double *phi)
{
    try
    {
        const int M = this->cb->get_total_grid();
        // Initialize array
        for(int i=0; i<M; i++)
            phi[i] = 0.0;

        // For each block
        for(const auto& block: phi_block)
        {
            std::string key_left = std::get<1>(block.first);
            int n_segment_right = this->propagator_computation_optimizer->get_computation_block(block.first).n_segment_right;
            if (PropagatorCode::get_monomer_type_from_key(key_left) == monomer_type && n_segment_right != 0)
            {
                for(int i=0; i<M; i++)
                    phi[i] += block.second[i]; 
            }
        }
        // For each solvent
        for(int s=0;s<this->molecules->get_n_solvent_types();s++)
        {
            if (std::get<1>(this->molecules->get_solvent(s)) == monomer_type)
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
void CudaComputationReduceMemoryDiscrete::get_total_concentration(int p, std::string monomer_type, double *phi)
{
    try
    {
        const int M = this->cb->get_total_grid();
        const int P = this->molecules->get_n_polymer_types();

        if (p < 0 || p > P-1)
            throw_with_line_number("Index (" + std::to_string(p) + ") must be in range [0, " + std::to_string(P-1) + "]");

        // Initialize array
        for(int i=0; i<M; i++)
            phi[i] = 0.0;

        // For each block
        for(const auto& block: phi_block)
        {
            int polymer_idx = std::get<0>(block.first);
            std::string key_left = std::get<1>(block.first);
            int n_segment_right = this->propagator_computation_optimizer->get_computation_block(block.first).n_segment_right;
            if (polymer_idx == p && PropagatorCode::get_monomer_type_from_key(key_left) == monomer_type && n_segment_right != 0)
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
void CudaComputationReduceMemoryDiscrete::get_total_concentration_gce(double fugacity, int p, std::string monomer_type, double *phi)
{
    try
    {
        const int M = this->cb->get_total_grid();
        const int P = this->molecules->get_n_polymer_types();

        if (p < 0 || p > P-1)
            throw_with_line_number("Index (" + std::to_string(p) + ") must be in range [0, " + std::to_string(P-1) + "]");

        // Initialize array
        for(int i=0; i<M; i++)
            phi[i] = 0.0;

        // For each block
        for(const auto& block: phi_block)
        {
            int polymer_idx = std::get<0>(block.first);
            std::string key_left = std::get<1>(block.first);
            int n_segment_right = this->propagator_computation_optimizer->get_computation_block(block.first).n_segment_right;
            if (polymer_idx == p && PropagatorCode::get_monomer_type_from_key(key_left) == monomer_type && n_segment_right != 0)
            {
                Polymer& pc = this->molecules->get_polymer(p);
                double norm = fugacity/pc.get_volume_fraction()*pc.get_alpha()*this->single_polymer_partitions[p];
                for(int i=0; i<M; i++)
                    phi[i] += block.second[i]*norm; 
            }
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaComputationReduceMemoryDiscrete::get_block_concentration(int p, double *phi)
{
    try
    {
        const int M = this->cb->get_total_grid();
        const int P = this->molecules->get_n_polymer_types();

        if (p < 0 || p > P-1)
            throw_with_line_number("Index (" + std::to_string(p) + ") must be in range [0, " + std::to_string(P-1) + "]");

        if (this->propagator_computation_optimizer->use_aggregation())
            throw_with_line_number("Disable 'aggregation' option to obtain concentration of each block.");

        Polymer& pc = this->molecules->get_polymer(p);
        std::vector<Block>& blocks = pc.get_blocks();

        for(size_t b=0; b<blocks.size(); b++)
        {
            std::string key_left  = pc.get_propagator_key(blocks[b].v, blocks[b].u);
            std::string key_right = pc.get_propagator_key(blocks[b].u, blocks[b].v);
            if (key_left < key_right)
                key_left.swap(key_right);

            double* _essential_phi_block = phi_block[std::make_tuple(p, key_left, key_right)];
            for(int i=0; i<M; i++)
                phi[i+b*M] = _essential_phi_block[i]; 
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
double CudaComputationReduceMemoryDiscrete::get_solvent_partition(int s)
{
    try
    {
        return this->single_solvent_partitions[s];
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaComputationReduceMemoryDiscrete::get_solvent_concentration(int s, double *phi)
{
    try
    {
        const int M = this->cb->get_total_grid();
        const int S = this->molecules->get_n_solvent_types();

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
void CudaComputationReduceMemoryDiscrete::compute_stress()
{
    // This method should be invoked after invoking compute_statistics().

    // To calculate stress, we multiply weighted fourier basis to q(k)*q^dagger(-k).
    // We only need the real part of stress calculation.
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();
        const int N_GPUS = CudaCommon::get_instance().get_n_gpus();

        const int DIM = this->cb->get_dim();
        const int M   = this->cb->get_total_grid();

        std::map<std::tuple<int, std::string, std::string>, std::array<double,3>> block_dq_dl[n_streams];

        // Reset stress map
        for(const auto& item: phi_block)
        {
            for(int i=0; i<n_streams; i++)
                for(int d=0; d<3; d++)
                    block_dq_dl[i][item.first][d] = 0.0;
        }

        // Compute stress for each block
        #pragma omp parallel for num_threads(n_streams)
        for(size_t b=0; b<phi_block.size();b++)
        {
            const int STREAM = omp_get_thread_num();
            int gpu = omp_get_thread_num() % N_GPUS;
            gpu_error_check(cudaSetDevice(gpu));

            auto block = phi_block.begin();
            advance(block, b);
            const auto& key   = block->first;

            int p                 = std::get<0>(key);
            std::string key_left  = std::get<1>(key);
            std::string key_right = std::get<2>(key);

            const int N_RIGHT        = this->propagator_computation_optimizer->get_computation_block(key).n_segment_right;
            const int N_LEFT         = this->propagator_computation_optimizer->get_computation_block(key).n_segment_left;
            std::string monomer_type = this->propagator_computation_optimizer->get_computation_block(key).monomer_type;
            int n_repeated           = this->propagator_computation_optimizer->get_computation_block(key).n_repeated;

            // If there is no segment
            if(N_RIGHT == 0)
                continue;

            double **q_1 = propagator[key_left];      // Propagator q
            double **q_2 = propagator[key_right];     // Propagator q^dagger

            std::array<double,3> _block_dq_dl = {0.0, 0.0, 0.0};

            // Check block_stress_computation_plan
            const auto& _block_stress_compuation_key = block_stress_computation_plan[key];
            if(_block_stress_compuation_key.size() != (unsigned int) (N_RIGHT+1))
            {
                throw_with_line_number("Mismatch of block_stress_computation_plan("
                    + std::to_string(p) + "," + key_left + "," + key_right + ") "
                    + std::to_string(_block_stress_compuation_key.size()) + ".size() with N+1 (" + std::to_string(N_RIGHT+1) + ")");
            }

            // Variables for block_stress_computation_plan
            double *propagator_left;
            double *propagator_right;

            double *d_segment_stress;
            double segment_stress[DIM];
            gpu_error_check(cudaMalloc((void**)&d_segment_stress, sizeof(double)*3));

            int prev, next;
            prev = 0;
            next = 1;

            // Create events
            cudaEvent_t kernel_done;
            cudaEvent_t memcpy_done;
            gpu_error_check(cudaEventCreate(&kernel_done));
            gpu_error_check(cudaEventCreate(&memcpy_done));

            // Copy memory from device to device
            propagator_left  = std::get<0>(_block_stress_compuation_key[0]);
            propagator_right = std::get<1>(_block_stress_compuation_key[0]);

            if (propagator_left != nullptr)
            {
                gpu_error_check(cudaMemcpyAsync(&d_q_pair[STREAM][prev][0], propagator_left,  sizeof(double)*M, cudaMemcpyHostToDevice, streams[STREAM][1]));
                gpu_error_check(cudaMemcpyAsync(&d_q_pair[STREAM][prev][M], propagator_right, sizeof(double)*M, cudaMemcpyHostToDevice, streams[STREAM][1]));
                gpu_error_check(cudaEventRecord(memcpy_done, streams[STREAM][1]));
            }
            gpu_error_check(cudaStreamWaitEvent(streams[STREAM][0], memcpy_done, 0));

            for(int n=0; n<=N_RIGHT; n++)
            {
                // STREAM 1: copy memory from device to device
                if (n+1 <= N_RIGHT)
                {
                    propagator_left  = std::get<0>(_block_stress_compuation_key[n+1]);
                    propagator_right = std::get<1>(_block_stress_compuation_key[n+1]);

                    if (propagator_left != nullptr)
                    {
                        gpu_error_check(cudaMemcpyAsync(&d_q_pair[STREAM][next][0], propagator_left,  sizeof(double)*M, cudaMemcpyHostToDevice, streams[STREAM][1]));
                        gpu_error_check(cudaMemcpyAsync(&d_q_pair[STREAM][next][M], propagator_right, sizeof(double)*M, cudaMemcpyHostToDevice, streams[STREAM][1]));
                        gpu_error_check(cudaEventRecord(memcpy_done, streams[STREAM][1]));
                    }
                }

                // STREAM 0: Compute stress
                propagator_left = std::get<0>(_block_stress_compuation_key[n]);
                bool is_half_bond_length = std::get<2>(_block_stress_compuation_key[n]);
                if (propagator_left != nullptr)
                {
                    propagator_solver->compute_single_segment_stress(
                        gpu, STREAM, d_q_pair[STREAM][prev], d_segment_stress,
                        monomer_type, is_half_bond_length);
                    gpu_error_check(cudaEventRecord(kernel_done, streams[STREAM][0]));
                }

                // Wait until computation and memory copy are done
                gpu_error_check(cudaStreamWaitEvent(streams[STREAM][1], kernel_done, 0));
                gpu_error_check(cudaStreamWaitEvent(streams[STREAM][0], memcpy_done, 0));

                if (propagator_left != nullptr)
                {
                    gpu_error_check(cudaMemcpy(segment_stress, d_segment_stress, sizeof(double)*DIM, cudaMemcpyDeviceToHost));
                    for(int d=0; d<DIM; d++)
                        _block_dq_dl[d] += segment_stress[d]*n_repeated;
                }
                std::swap(prev, next);
            }
            gpu_error_check(cudaStreamSynchronize(streams[STREAM][0]));
            gpu_error_check(cudaStreamSynchronize(streams[STREAM][1]));
            gpu_error_check(cudaEventDestroy(kernel_done));
            gpu_error_check(cudaEventDestroy(memcpy_done));

            // Copy stress data
            for(int d=0; d<DIM; d++)
                block_dq_dl[STREAM][key][d] += _block_dq_dl[d];
                
            cudaFree(d_segment_stress);
        }
        // Synchronize all GPUs
        for(int gpu=0; gpu<N_GPUS; gpu++)
        {
            gpu_error_check(cudaSetDevice(gpu));
            gpu_error_check(cudaDeviceSynchronize());
        }
        gpu_error_check(cudaSetDevice(0));
        
        // Compute total stress
        int n_polymer_types = this->molecules->get_n_polymer_types();
        for(int p=0; p<n_polymer_types; p++)
            for(int d=0; d<DIM; d++)
                this->dq_dl[p][d] = 0.0;
        for(const auto& block: phi_block)
        {
            const auto& key       = block.first;
            int p                 = std::get<0>(key);
            std::string key_left  = std::get<1>(key);
            std::string key_right = std::get<2>(key);
            Polymer& pc = this->molecules->get_polymer(p);

            for(int i=0; i<n_streams; i++)
                for(int d=0; d<DIM; d++)
                    this->dq_dl[p][d] += block_dq_dl[i][key][d];
        }
        for(int p=0; p<n_polymer_types; p++){
            for(int d=0; d<DIM; d++)
                this->dq_dl[p][d] /= -3.0*this->cb->get_lx(d)*M*M/this->molecules->get_ds();
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaComputationReduceMemoryDiscrete::get_chain_propagator(double *q_out, int polymer, int v, int u, int n)
{ 
    // This method should be invoked after invoking compute_statistics()

    // Get chain propagator for a selected polymer, block and direction.
    // This is made for debugging and testing.
    try
    {
        const int M = this->cb->get_total_grid();
        Polymer& pc = this->molecules->get_polymer(polymer);
        std::string dep = pc.get_propagator_key(v,u);

        if (this->propagator_computation_optimizer->get_computation_propagators().find(dep) == this->propagator_computation_optimizer->get_computation_propagators().end())
            throw_with_line_number("Could not find the propagator code '" + dep + "'. Disable 'aggregation' option to obtain propagator_computation_optimizer.");
            
        const int N_RIGHT = this->propagator_computation_optimizer->get_computation_propagator(dep).max_n_segment;
        if (n < 1 || n > N_RIGHT)
            throw_with_line_number("n (" + std::to_string(n) + ") must be in range [1, " + std::to_string(N_RIGHT) + "]");

        double* _propagator = propagator[dep][n];
        for(int i=0; i<M; i++)
            q_out[i] = _propagator[i];
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
bool CudaComputationReduceMemoryDiscrete::check_total_partition()
{
    const int M = this->cb->get_total_grid();
    int n_polymer_types = this->molecules->get_n_polymer_types();
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
        int p                 = std::get<0>(key);
        std::string key_left  = std::get<1>(key);
        std::string key_right = std::get<2>(key);

        int n_segment_right = this->propagator_computation_optimizer->get_computation_block(key).n_segment_right;
        int n_segment_left  = this->propagator_computation_optimizer->get_computation_block(key).n_segment_left;
        int n_repeated      = this->propagator_computation_optimizer->get_computation_block(key).n_repeated;
        int n_propagators   = this->propagator_computation_optimizer->get_computation_block(key).v_u.size();

        std::string monomer_type = this->propagator_computation_optimizer->get_computation_block(key).monomer_type;
        double *_d_exp_dw = propagator_solver->d_exp_dw[0][monomer_type];

        #ifndef NDEBUG
        std::cout<< p << ", " << key_left << ", " << key_right << ": " << n_segment_left << ", " << n_segment_right << ", " << n_propagators << ", " << this->propagator_computation_optimizer->get_computation_block(key).n_repeated << std::endl;
        #endif
        
        for(int n=1;n<=n_segment_right;n++)
        {
            // Copy propagators from host to device
            gpu_error_check(cudaMemcpy(d_q_block_v[0], propagator[key_left][n_segment_left-n+1], sizeof(double)*M, cudaMemcpyHostToDevice));
            gpu_error_check(cudaMemcpy(d_q_block_u[0], propagator[key_right][n], sizeof(double)*M, cudaMemcpyHostToDevice));

            double total_partition = this->cb->inner_product_inverse_weight_device(
                d_q_block_v[0],  // q
                d_q_block_u[0],  // q^dagger
                _d_exp_dw)*n_repeated/this->cb->get_volume();
            
            total_partition /= n_propagators;
            total_partitions[p].push_back(total_partition);

            #ifndef NDEBUG
            std::cout<< p << ", " << n << ": " << total_partition << std::endl;
            #endif
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
        double diff_partition = std::abs(max_partition - min_partition);

        std::cout<< "\t" << p << ": " << max_partition << ", " << min_partition << ", " << diff_partition << std::endl;
        if (diff_partition > 1e-7)
            return false;
    }
    return true;
}