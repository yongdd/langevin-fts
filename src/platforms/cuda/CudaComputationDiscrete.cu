#include <complex>
#include <iostream>
#include <chrono>
#include <omp.h>

#include "CudaComputationDiscrete.h"
#include "CudaComputationBox.h"
#include "CudaSolverPseudoDiscrete.h"

template <typename T>
CudaComputationDiscrete<T>::CudaComputationDiscrete(
    ComputationBox<T>* cb,
    Molecules *molecules,
    PropagatorComputationOptimizer *propagator_computation_optimizer)
    : PropagatorComputation<T>(cb, molecules, propagator_computation_optimizer)
{
    try
    {
        #ifndef NDEBUG
        std::cout << "--------- Discrete Chain Solver, GPU Version ---------" << std::endl;
        #endif

        const int M = this->cb->get_total_grid();

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
            gpu_error_check(cudaStreamCreate(&streams[i][0])); // for kernel execution
            gpu_error_check(cudaStreamCreate(&streams[i][1])); // for memcpy
        }

        this->propagator_solver = new CudaSolverPseudoDiscrete<T>(cb, molecules, n_streams, streams, false);

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
            d_propagator_half_steps[key] = new CuDeviceData<T>*[max_n_segment];
            d_propagator_half_steps[key][0] = nullptr;
            if (item.second.deps.size() > 0)
                gpu_error_check(cudaMalloc((void**)&d_propagator_half_steps[key][0], sizeof(T)*M));

            // Allocate memory for q(r,s+1/2)
            for(int i=1; i<propagator_size[key]; i++)
            {
                d_propagator_half_steps[key][i] = nullptr;
                if (item.second.junction_ends.find(i) != item.second.junction_ends.end())
                    gpu_error_check(cudaMalloc((void**)&d_propagator_half_steps[key][i], sizeof(T)*M));
            }

            // Allocate memory for q(r,s)
            // Index 0 will be not used
            d_propagator[key] = new CuDeviceData<T>*[max_n_segment];
            d_propagator[key][0] = nullptr;
            for(int i=1; i<propagator_size[key]; i++)
                gpu_error_check(cudaMalloc((void**)&d_propagator[key][i], sizeof(T)*M));

            #ifndef NDEBUG
            propagator_finished[key] = new bool[max_n_segment];
            for(int i=0; i<max_n_segment;i++)
                propagator_finished[key][i] = false;
            for (int n: item.second.junction_ends)
                propagator_half_steps_finished[key][n] = false;
            propagator_half_steps_finished[key][0] = false;
            #endif
        }

        // Allocate memory for concentrations
        if( this->propagator_computation_optimizer->get_computation_blocks().size() == 0)
            throw_with_line_number("There is no block. Add polymers first.");
        for(const auto& item: this->propagator_computation_optimizer->get_computation_blocks())
        {
            d_phi_block[item.first] = nullptr;
            gpu_error_check(cudaMalloc((void**)&d_phi_block[item.first], sizeof(T)*M));
        }

        // Remember one segment for each polymer chain to compute total partition function
        int current_p = 0;
        for(const auto& d_block: d_phi_block)
        {
            const auto& key = d_block.first;
            int p                 = std::get<0>(key);
            std::string key_left  = std::get<1>(key);
            std::string key_right = std::get<2>(key);

            // Skip if already found one segment
            if (p != current_p)
                continue;

            int n_aggregated         = this->propagator_computation_optimizer->get_computation_block(key).v_u.size()/
                                       this->propagator_computation_optimizer->get_computation_block(key).n_repeated;
            int n_segment_left       = this->propagator_computation_optimizer->get_computation_block(key).n_segment_left;
            std::string monomer_type = this->propagator_computation_optimizer->get_computation_block(key).monomer_type;

            // Skip if n_segment_left is 0
            if (n_segment_left == 0)
                continue;

            single_partition_segment.push_back(std::make_tuple(
                p,
                d_propagator[key_left][n_segment_left],    // q
                d_propagator[key_right][1],                  // q_dagger
                monomer_type,       
                n_aggregated                                 // how many propagators are aggregated
                ));
            current_p++;
        }

       // Find propagators and bond length for each segment to prepare stress computation
        for(const auto& block: d_phi_block)
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

            CuDeviceData<T> **d_q_1 = d_propagator[key_left];     // dependency v
            CuDeviceData<T> **d_q_2 = d_propagator[key_right];    // dependency u

            auto& _block_stress_compuation_key = block_stress_computation_plan[key];

            // Find propagators and bond length
            for(int n=0; n<=N_RIGHT; n++)
            {
                CuDeviceData<T> *d_propagator_left = nullptr;
                CuDeviceData<T> *d_propagator_right = nullptr;
                bool is_half_bond_length = false;

                // At v
                if (n == N_LEFT)
                {
                    if (this->propagator_computation_optimizer->get_computation_propagator(key_left).deps.size() == 0) // if v is leaf node, skip
                    {
                        _block_stress_compuation_key.push_back(std::make_tuple(d_propagator_left, d_propagator_right, is_half_bond_length));
                        continue;
                    }
                    
                    d_propagator_left  = d_propagator_half_steps[key_left][0];
                    d_propagator_right = d_q_2[N_RIGHT];
                    is_half_bond_length = true;
                }
                // At u
                else if (n == 0 && N_LEFT == N_RIGHT){
                    if (this->propagator_computation_optimizer->get_computation_propagator(key_right).deps.size() == 0) // if u is leaf node, skip
                    {
                        _block_stress_compuation_key.push_back(std::make_tuple(d_propagator_left, d_propagator_right, is_half_bond_length));
                        continue;
                    }

                    d_propagator_left  = d_q_1[N_LEFT];
                    d_propagator_right = d_propagator_half_steps[key_right][0];
                    is_half_bond_length = true;
                }
                // At aggregation junction
                else if (n == 0)
                {
                    _block_stress_compuation_key.push_back(std::make_tuple(d_propagator_left, d_propagator_right, is_half_bond_length));
                    continue;
                }
                // Within the blocks
                else
                {
                    d_propagator_left  = d_q_1[N_LEFT-n];
                    d_propagator_right = d_q_2[n];
                    is_half_bond_length = false;
                }
                _block_stress_compuation_key.push_back(std::make_tuple(d_propagator_left, d_propagator_right, is_half_bond_length));
            }
        }

        // Concentrations for each solvent
        for(int s=0;s<this->molecules->get_n_solvent_types();s++)
        {
            CuDeviceData<T> *d_phi_;
            gpu_error_check(cudaMalloc((void**)&d_phi_, sizeof(T)*M));
            d_phi_solvent.push_back(d_phi_);
        }

        // Create scheduler for computation of propagator
        sc = new Scheduler(this->propagator_computation_optimizer->get_computation_propagators(), n_streams); 

        // Allocate memory for pseudo-spectral: advance_propagator()
        gpu_error_check(cudaMalloc((void**)&d_q_unity, sizeof(T)*M));
        for(int i=0; i<M; i++)
        {
            CuDeviceData<T> q_unity;
            if constexpr (std::is_same<T, double>::value)
                q_unity = 1.0;
            else
                q_unity = make_cuDoubleComplex(1.0, 0.0);
            gpu_error_check(cudaMemcpy(&d_q_unity[i], &q_unity, sizeof(T), cudaMemcpyHostToDevice));
        }

        // Copy mask to d_q_mask
        if (this->cb->get_mask() != nullptr)
        {
            gpu_error_check(cudaMalloc((void**)&d_q_mask , sizeof(double)*M));
            gpu_error_check(cudaMemcpy(d_q_mask, this->cb->get_mask(), sizeof(double)*M, cudaMemcpyHostToDevice));
        }
        else
            d_q_mask = nullptr;
        gpu_error_check(cudaMalloc((void**)&d_phi, sizeof(T)*M));

        // Allocate memory for stress calculation: compute_stress()
        for(int i=0; i<n_streams; i++)
        {
            gpu_error_check(cudaMalloc((void**)&d_q_pair[i][0], sizeof(T)*2*M)); // prev
            gpu_error_check(cudaMalloc((void**)&d_q_pair[i][1], sizeof(T)*2*M)); // next
        }
        
        propagator_solver->update_laplacian_operator();
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
CudaComputationDiscrete<T>::~CudaComputationDiscrete()
{
    delete propagator_solver;
    delete sc;

    for(const auto& item: d_propagator)
    {
        for(int i=0; i<propagator_size[item.first]; i++)
        {
            if(item.second[i] != nullptr)
                cudaFree(item.second[i]);
        }
        delete[] item.second;
    }
    for(const auto& item: d_propagator_half_steps)
    {
        for(int i=0; i<propagator_size[item.first]; i++)
        {
            if(item.second[i] != nullptr)
                cudaFree(item.second[i]);
        }
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

    // For pseudo-spectral: advance_propagator()
    if (d_q_mask != nullptr)
        cudaFree(d_q_mask);
    cudaFree(d_q_unity);


    // For stress calculation: compute_stress()
    for(int i=0; i<n_streams; i++)
    {
        cudaFree(d_q_pair[i][0]);
        cudaFree(d_q_pair[i][1]);
    }

    // Destroy streams
    for(int i=0; i<n_streams; i++)
    {
        cudaStreamDestroy(streams[i][0]);
        cudaStreamDestroy(streams[i][1]);
    }
}
template <typename T>
void CudaComputationDiscrete<T>::update_laplacian_operator()
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
template <typename T>
void CudaComputationDiscrete<T>::compute_statistics(
    std::map<std::string, const T*> w_input,
    std::map<std::string, const T*> q_init)
{
    this->compute_propagators(w_input, q_init);
    this->compute_concentrations();
}
template <typename T>
void CudaComputationDiscrete<T>::compute_propagators(
    std::map<std::string, const T*> w_input,
    std::map<std::string, const T*> q_init)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

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
            //     for (auto it = d_propagator_half_steps[key].begin(); it != d_propagator_half_steps[key].end(); ++it)
            //     {
            //         std::cout << it->first+1;
            //         if (std::next(it) != d_propagator_half_steps[key].end()) {
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

                auto& key = std::get<0>((*parallel_job)[job]);
                int n_segment_from = std::get<1>((*parallel_job)[job]);
                int n_segment_to = std::get<2>((*parallel_job)[job]);
                auto& deps = this->propagator_computation_optimizer->get_computation_propagator(key).deps;
                auto monomer_type = this->propagator_computation_optimizer->get_computation_propagator(key).monomer_type;

                // #ifndef NDEBUG
                // #pragma omp critical
                // std::cout << job << " started, stream: " << STREAM << ", " <<
                //     std::chrono::duration_cast<std::chrono::microseconds>
                //     (std::chrono::system_clock::now().time_since_epoch()).count() - start_time << std::endl;
                // #endif

                // Check key
                #ifndef NDEBUG
                if (d_propagator.find(key) == d_propagator.end())
                    std::cout<< "Could not find key '" + key + "'. " << std::endl;
                #endif
                
                CuDeviceData<T> **_d_propagator = d_propagator[key];
                CuDeviceData<T> *_d_exp_dw = propagator_solver->d_exp_dw[monomer_type];

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
                            std::cout<< "Could not find q_init[\"" + g + "\"]." << std::endl;
                        gpu_error_check(cudaMemcpyAsync(_d_propagator[1], q_init[g], sizeof(T)*M, cudaMemcpyInputToDevice, streams[STREAM][0]));
                        ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(_d_propagator[1], _d_propagator[1], _d_exp_dw, 1.0, M);
                    }
                    else
                    {
                        gpu_error_check(cudaMemcpyAsync(_d_propagator[1], _d_exp_dw, sizeof(T)*M, cudaMemcpyDeviceToDevice, streams[STREAM][0]));
                    }
                    
                    #ifndef NDEBUG
                    propagator_finished[key][1] = true;
                    #endif
                }
                // If it is not leaf node
                else if (n_segment_from == 0 && deps.size() > 0)
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
                        gpu_error_check(cudaMemsetAsync(_d_propagator[1], 0, sizeof(T)*M, streams[STREAM][0]));

                        for(size_t d=0; d<deps.size(); d++)
                        {
                            std::string sub_dep = std::get<0>(deps[d]);
                            int sub_n_segment   = std::get<1>(deps[d]);
                            int sub_n_repeated  = std::get<2>(deps[d]);
                            CuDeviceData<T> **_propagator_sub_dep;

                            if (sub_n_segment == 0)
                            {
                                // Check sub key
                                #ifndef NDEBUG
                                if (d_propagator_half_steps.find(sub_dep) == d_propagator_half_steps.end())
                                    std::cout << "Could not find sub key '" + sub_dep + "'. " << std::endl;
                                if (!propagator_half_steps_finished[sub_dep][0])
                                    std::cout << "Could not compute '" + key +  "', since '"+ sub_dep + std::to_string(0) + "' is not prepared." << std::endl;
                                #endif

                                _propagator_sub_dep = d_propagator_half_steps[sub_dep];
                            }
                            else
                            {
                                // Check sub key
                                #ifndef NDEBUG
                                if (d_propagator.find(sub_dep) == d_propagator.end())
                                    std::cout<< "Could not find sub key '" + sub_dep + "'. " << std::endl;
                                if (!propagator_finished[sub_dep][sub_n_segment])
                                    std::cout<< "Could not compute '" + key +  "', since '"+ sub_dep + std::to_string(sub_n_segment) + "' is not prepared." << std::endl;
                                #endif

                                _propagator_sub_dep = d_propagator[sub_dep];
                            }

                            ker_lin_comb<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                                _d_propagator[1], 1.0, _d_propagator[1],
                                sub_n_repeated, _propagator_sub_dep[sub_n_segment], M);
                        }

                        // if sub_n_segment == 0
                        if (std::get<1>(deps[0]) == 0)
                        {
                            gpu_error_check(cudaMemcpyAsync(d_propagator_half_steps[key][0], _d_propagator[1], sizeof(T)*M, cudaMemcpyDeviceToDevice, streams[STREAM][0]));

                            // Add half bond, STREAM 0
                            propagator_solver->advance_propagator_half_bond_step(
                                STREAM,
                                _d_propagator[1], _d_propagator[1], monomer_type);

                            // Add full segment
                            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(_d_propagator[1], _d_propagator[1], _d_exp_dw, 1.0, M);

                        }
                        else
                        {
                            propagator_solver->advance_propagator(
                                STREAM,
                                _d_propagator[1],
                                _d_propagator[1],
                                monomer_type, d_q_mask);
                        }

                        #ifndef NDEBUG
                        propagator_finished[key][1] = true;
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

                        // #ifndef NDEBUG
                        // #pragma omp critical
                        // std::cout << job << " init 3, " << 
                        //     std::chrono::duration_cast<std::chrono::microseconds>
                        //     (std::chrono::system_clock::now().time_since_epoch()).count() - start_time << std::endl;
                        // #endif

                        // Combine branches
                        // Initialize to one
                        gpu_error_check(cudaMemcpyAsync(d_propagator_half_steps[key][0], d_q_unity, sizeof(T)*M, cudaMemcpyDeviceToDevice, streams[STREAM][0]));

                        for(size_t d=0; d<deps.size(); d++)
                        {
                            std::string sub_dep = std::get<0>(deps[d]);
                            int sub_n_segment   = std::get<1>(deps[d]);

                            // Check sub key
                            #ifndef NDEBUG
                            if (!propagator_half_steps_finished[sub_dep][sub_n_segment])
                                std::cout<< "Could not compute '" + key +  "', since '"+ sub_dep + std::to_string(sub_n_segment) + "+1/2' is not prepared." << std::endl;
                            #endif

                            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                                d_propagator_half_steps[key][0], d_propagator_half_steps[key][0],
                                d_propagator_half_steps[sub_dep][sub_n_segment], 1.0, M);
                        }

                        #ifndef NDEBUG
                        propagator_half_steps_finished[key][0] = true;
                        #endif

                        if (n_segment_to > 0)
                        {
                            // Add half bond, STREAM 0
                            propagator_solver->advance_propagator_half_bond_step(
                                STREAM,
                                d_propagator_half_steps[key][0], _d_propagator[1], monomer_type);

                            // Add full segment
                            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(_d_propagator[1], _d_propagator[1], _d_exp_dw, 1.0, M);

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
                    if (d_q_mask != nullptr)
                        ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(_d_propagator[1], _d_propagator[1], d_q_mask, 1.0, M);

                    // q(r, 1+1/2)
                    if (d_propagator_half_steps[key][1] != nullptr)
                    {
                        #ifndef NDEBUG
                        if (propagator_half_steps_finished[key][1])
                            std::cout << "already half_step finished: " + key + ", " + std::to_string(1) << std::endl;
                        #endif

                        propagator_solver->advance_propagator_half_bond_step(
                            STREAM, 
                            _d_propagator[1],
                            d_propagator_half_steps[key][1],
                            monomer_type);

                        #ifndef NDEBUG
                        propagator_half_steps_finished[key][1] = true;
                        #endif
                    }
                    n_segment_from++;
                }

                // q(r, s)
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

                    propagator_solver->advance_propagator(
                        STREAM, 
                        _d_propagator[n],
                        _d_propagator[n+1],
                        monomer_type, d_q_mask);

                    #ifndef NDEBUG
                    propagator_finished[key][n+1] = true;
                    #endif
                }
                
                // q(r, n+1/2)
                for(int n=n_segment_from; n<n_segment_to; n++)
                {
                    if (d_propagator_half_steps[key][n+1] != nullptr)
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

                        propagator_solver->advance_propagator_half_bond_step(
                            STREAM,
                            _d_propagator[n+1],
                            d_propagator_half_steps[key][n+1],
                            monomer_type);

                        #ifndef NDEBUG
                        propagator_half_steps_finished[key][n+1] = true;
                        #endif
                    }
                }
                gpu_error_check(cudaStreamSynchronize(streams[STREAM][0]));
                gpu_error_check(cudaStreamSynchronize(streams[STREAM][1]));
                
                // #ifndef NDEBUG
                // #pragma omp critical
                // std::cout << job << " finished, " << 
                //     std::chrono::duration_cast<std::chrono::microseconds>
                //     (std::chrono::system_clock::now().time_since_epoch()).count() - start_time << std::endl;
                // #endif
            }
            gpu_error_check(cudaDeviceSynchronize());
        }

        // Compute total partition function of each distinct polymers
        for(const auto& segment_info: single_partition_segment)
        {
            int p                               = std::get<0>(segment_info);
            CuDeviceData<T> *d_propagator_left  = std::get<1>(segment_info);
            CuDeviceData<T> *d_propagator_right = std::get<2>(segment_info);
            std::string monomer_type            = std::get<3>(segment_info);
            int n_aggregated                    = std::get<4>(segment_info);
            CuDeviceData<T> *_d_exp_dw          = propagator_solver->d_exp_dw[monomer_type];

            this->single_polymer_partitions[p] = dynamic_cast<CudaComputationBox<T>*>(this->cb)->inner_product_inverse_weight_device(
                d_propagator_left,   // q
                d_propagator_right,  // q^dagger
                _d_exp_dw)/(n_aggregated*this->cb->get_volume());
        }

    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
void CudaComputationDiscrete<T>::compute_concentrations()
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();
        const int M = this->cb->get_total_grid();

        // Calculate segment concentrations
        for(const auto& d_block: d_phi_block)
        {
            const auto& key = d_block.first;
            int p                 = std::get<0>(key);
            std::string key_left  = std::get<1>(key);
            std::string key_right = std::get<2>(key);

            int n_segment_right = this->propagator_computation_optimizer->get_computation_block(key).n_segment_right;
            int n_segment_left  = this->propagator_computation_optimizer->get_computation_block(key).n_segment_left;
            std::string monomer_type = this->propagator_computation_optimizer->get_computation_block(key).monomer_type;
            int n_repeated = this->propagator_computation_optimizer->get_computation_block(key).n_repeated;
            CuDeviceData<T> *_d_exp_dw = propagator_solver->d_exp_dw[monomer_type];

            // If there is no segment
            if(n_segment_right == 0)
            {
                gpu_error_check(cudaMemset(d_block.second, 0, sizeof(T)*M));
                continue;
            }

            // Check keys
            #ifndef NDEBUG
            if (d_propagator.find(key_left) == d_propagator.end())
                throw_with_line_number("Could not find key_left key'" + key_left + "'. ");
            if (d_propagator.find(key_right) == d_propagator.end())
                throw_with_line_number("Could not find key_right key'" + key_right + "'. ");
            #endif

            // Calculate phi of one block (possibly multiple blocks when using aggregation)
            calculate_phi_one_block(
                d_block.second,              // phi
                d_propagator[key_left],    // dependency v
                d_propagator[key_right],   // dependency u
                _d_exp_dw,                 // exp_dw
                n_segment_right,
                n_segment_left);
            
            // Normalize concentration
            Polymer& pc = this->molecules->get_polymer(p);
            CuDeviceData<T> norm;
            if constexpr (std::is_same<T, double>::value)
            {
                norm = this->molecules->get_ds()*pc.get_volume_fraction()/pc.get_alpha()*n_repeated;
                norm = norm/this->single_polymer_partitions[p];
            }
            else
            {
                norm = make_cuDoubleComplex(this->molecules->get_ds()*pc.get_volume_fraction()/pc.get_alpha()*n_repeated, 0.0);
                norm = cuCdiv(norm, stdToCuDoubleComplex(this->single_polymer_partitions[p]));
            }
            ker_lin_comb<<<N_BLOCKS, N_THREADS>>>(d_block.second, norm, d_block.second, 0.0, d_block.second, M);
        }

        // Calculate partition functions and concentrations of solvents
        for(int s=0; s<this->molecules->get_n_solvent_types(); s++)
        {
            CuDeviceData<T> *d_phi_ = d_phi_solvent[s];
            double volume_fraction   = std::get<0>(this->molecules->get_solvent(s));
            std::string monomer_type = std::get<1>(this->molecules->get_solvent(s));
            CuDeviceData<T> *_d_exp_dw = propagator_solver->d_exp_dw[monomer_type];

            this->single_solvent_partitions[s] = dynamic_cast<CudaComputationBox<T>*>(this->cb)->integral_device(_d_exp_dw)/this->cb->get_volume();

            CuDeviceData<T> norm;
            if constexpr (std::is_same<T, double>::value)
            {
                norm = volume_fraction;
                norm = norm/this->single_solvent_partitions[s];
            }
            else
            {
                norm = make_cuDoubleComplex(volume_fraction, 0.0);
                norm = cuCdiv(norm, stdToCuDoubleComplex(this->single_solvent_partitions[s]));
            }
            ker_linear_scaling<<<N_BLOCKS, N_THREADS>>>(d_phi_, _d_exp_dw, norm, 0.0, M);
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
void CudaComputationDiscrete<T>::calculate_phi_one_block(
    CuDeviceData<T> *d_phi, CuDeviceData<T> **d_q_1, CuDeviceData<T> **d_q_2, CuDeviceData<T> *d_exp_dw, const int N_RIGHT, const int N_LEFT)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();
        const int M = this->cb->get_total_grid();

        // Compute segment concentration
        ker_multi<<<N_BLOCKS, N_THREADS>>>(d_phi,d_q_1[N_LEFT], d_q_2[1], 1.0, M);
        for(int n=2; n<=N_RIGHT; n++)
        {
            ker_add_multi<<<N_BLOCKS, N_THREADS>>>(d_phi, d_q_1[N_LEFT-n+1], d_q_2[n], 1.0, M);
        }
        ker_divide<<<N_BLOCKS, N_THREADS>>>(d_phi, d_phi, d_exp_dw, 1.0, M);
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
T CudaComputationDiscrete<T>::get_total_partition(int polymer)
{
    try
    {
        return this->single_polymer_partitions[polymer];
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
void CudaComputationDiscrete<T>::get_total_concentration(std::string monomer_type, T *phi)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();
        const int M = this->cb->get_total_grid();

        // Initialize to zero
        gpu_error_check(cudaMemset(d_phi, 0, sizeof(T)*M));

        // For each block
        for(const auto& d_block: d_phi_block)
        {
            const auto& key = d_block.first;
            std::string key_left = std::get<1>(key);
            int n_segment_right = this->propagator_computation_optimizer->get_computation_block(key).n_segment_right;
            if (PropagatorCode::get_monomer_type_from_key(key_left) == monomer_type && n_segment_right != 0)
                ker_lin_comb<<<N_BLOCKS, N_THREADS>>>(d_phi, 1.0, d_phi, 1.0, d_block.second, M);
        }

        // For each solvent
        for(int s=0;s<this->molecules->get_n_solvent_types();s++)
        {
            if (std::get<1>(this->molecules->get_solvent(s)) == monomer_type)
                ker_lin_comb<<<N_BLOCKS, N_THREADS>>>(d_phi, 1.0, d_phi, 1.0, d_phi_solvent[s], M);
        }
        gpu_error_check(cudaMemcpy(phi, d_phi, sizeof(T)*M, cudaMemcpyDeviceToHost));
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
void CudaComputationDiscrete<T>::get_total_concentration(int p, std::string monomer_type, T *phi)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = this->cb->get_total_grid();
        const int P = this->molecules->get_n_polymer_types();

        if (p < 0 || p > P-1)
            throw_with_line_number("Index (" + std::to_string(p) + ") must be in range [0, " + std::to_string(P-1) + "]");

        // Initialize to zero
        gpu_error_check(cudaMemset(d_phi, 0, sizeof(T)*M));

        // For each block
        for(const auto& d_block: d_phi_block)
        {
            const auto& key = d_block.first;
            int polymer_idx = std::get<0>(key);
            std::string key_left = std::get<1>(key);
            int n_segment_right = this->propagator_computation_optimizer->get_computation_block(key).n_segment_right;
            if (polymer_idx == p && PropagatorCode::get_monomer_type_from_key(key_left) == monomer_type && n_segment_right != 0)
                ker_lin_comb<<<N_BLOCKS, N_THREADS>>>(d_phi, 1.0, d_phi, 1.0, d_block.second, M);
        }
        gpu_error_check(cudaMemcpy(phi, d_phi, sizeof(T)*M, cudaMemcpyDeviceToHost));
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
void CudaComputationDiscrete<T>::get_total_concentration_gce(double fugacity, int p, std::string monomer_type, T *phi)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = this->cb->get_total_grid();
        const int P = this->molecules->get_n_polymer_types();

        if (p < 0 || p > P-1)
            throw_with_line_number("Index (" + std::to_string(p) + ") must be in range [0, " + std::to_string(P-1) + "]");

        // Initialize to zero
        gpu_error_check(cudaMemset(d_phi, 0, sizeof(T)*M));

        // For each block
        for(const auto& d_block: d_phi_block)
        {
            const auto& key = d_block.first;
            int polymer_idx = std::get<0>(key);
            std::string key_left = std::get<1>(key);
            int n_segment_right = this->propagator_computation_optimizer->get_computation_block(key).n_segment_right;
            if (polymer_idx == p && PropagatorCode::get_monomer_type_from_key(key_left) == monomer_type && n_segment_right != 0)
            {
                Polymer& pc = this->molecules->get_polymer(p);
                
                CuDeviceData<T> norm;
                if constexpr (std::is_same<T, double>::value)
                    norm = fugacity/pc.get_volume_fraction()*pc.get_alpha()*this->single_polymer_partitions[p];
                else
                    norm = stdToCuDoubleComplex(fugacity/pc.get_volume_fraction()*pc.get_alpha()*this->single_polymer_partitions[p]);
                ker_lin_comb<<<N_BLOCKS, N_THREADS>>>(d_phi, norm, d_block.second, 1.0, d_phi, M);
            }
        }
        gpu_error_check(cudaMemcpy(phi, d_phi, sizeof(T)*M, cudaMemcpyDeviceToHost));
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
void CudaComputationDiscrete<T>::get_block_concentration(int p, T *phi)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = this->cb->get_total_grid();
        const int P = this->molecules->get_n_polymer_types();

        if (p < 0 || p > P-1)
            throw_with_line_number("Index (" + std::to_string(p) + ") must be in range [0, " + std::to_string(P-1) + "]");

        if (this->propagator_computation_optimizer->use_aggregation())
            throw_with_line_number("Disable 'aggregation' option to obtain concentration of each block.");

        // Initialize to zero
        gpu_error_check(cudaMemset(d_phi, 0, sizeof(T)*M));

        Polymer& pc = this->molecules->get_polymer(p);
        std::vector<Block>& blocks = pc.get_blocks();

        for(size_t b=0; b<blocks.size(); b++)
        {
            std::string key_left  = pc.get_propagator_key(blocks[b].v, blocks[b].u);
            std::string key_right = pc.get_propagator_key(blocks[b].u, blocks[b].v);
            if (key_left < key_right)
                key_left.swap(key_right);

            ker_lin_comb<<<N_BLOCKS, N_THREADS>>>(d_phi, 0.0, d_phi, 1.0, d_phi_block[std::make_tuple(p, key_left, key_right)], M);
            gpu_error_check(cudaMemcpy(&phi[b*M], d_phi, sizeof(T)*M, cudaMemcpyDeviceToHost));
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
T CudaComputationDiscrete<T>::get_solvent_partition(int s)
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
template <typename T>
void CudaComputationDiscrete<T>::get_solvent_concentration(int s, T *phi)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = this->cb->get_total_grid();
        const int S = this->molecules->get_n_solvent_types();

        if (s < 0 || s > S-1)
            throw_with_line_number("Index (" + std::to_string(s) + ") must be in range [0, " + std::to_string(S-1) + "]");

        gpu_error_check(cudaMemcpy(phi, d_phi_solvent[s], sizeof(T)*M, cudaMemcpyDeviceToHost));
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
void CudaComputationDiscrete<T>::compute_stress()
{
    // This method should be invoked after invoking compute_statistics().

    // To calculate stress, we multiply weighted fourier basis to q(k)*q^dagger(-k).
    // We only need the real part of stress calculation.
    try
    {
        // if constexpr (std::is_same<T, std::complex<double>>::value)
        //     throw_with_line_number("Currently, stress computation is not suppoted for complex number type.");

        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int DIM = this->cb->get_dim();
        const int M   = this->cb->get_total_grid();

        std::map<std::tuple<int, std::string, std::string>, std::array<T,3>> block_dq_dl[n_streams];

        // Reset stress map
        for(const auto& item: d_phi_block)
        {
            for(int i=0; i<n_streams; i++)
                for(int d=0; d<3; d++)
                    block_dq_dl[i][item.first][d] = 0.0;
        }

        // Compute stress for each block
        #pragma omp parallel for num_threads(n_streams)
        for(size_t b=0; b<d_phi_block.size();b++)
        {
            const int STREAM = omp_get_thread_num();
            auto block = d_phi_block.begin();
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

            CuDeviceData<T> **d_q_1 = d_propagator[key_left];      // Propagator q
            CuDeviceData<T> **d_q_2 = d_propagator[key_right];     // Propagator q^dagger

            std::array<T,3> _block_dq_dl;
            for(int i=0; i<3; i++)
                _block_dq_dl[i] = 0.0;

            // Check block_stress_computation_plan
            const auto& _block_stress_compuation_key = block_stress_computation_plan[key];
            if(_block_stress_compuation_key.size() != (unsigned int) (N_RIGHT+1))
            {
                throw_with_line_number("Mismatch of block_stress_computation_plan("
                    + std::to_string(p) + "," + key_left + "," + key_right + ") "
                    + std::to_string(_block_stress_compuation_key.size()) + ".size() with N+1 (" + std::to_string(N_RIGHT+1) + ")");
            }

            // Variables for block_stress_computation_plan
            CuDeviceData<T> *d_propagator_left;
            CuDeviceData<T> *d_propagator_right;

            int prev, next;
            prev = 0;
            next = 1;

            // Create events
            cudaEvent_t kernel_done;
            cudaEvent_t memcpy_done;
            gpu_error_check(cudaEventCreate(&kernel_done));
            gpu_error_check(cudaEventCreate(&memcpy_done));

            // Copy memory from device to device
            d_propagator_left  = std::get<0>(_block_stress_compuation_key[0]);
            d_propagator_right = std::get<1>(_block_stress_compuation_key[0]);

            if (d_propagator_left != nullptr)
            {
                gpu_error_check(cudaMemcpyAsync(&d_q_pair[STREAM][prev][0], d_propagator_left,  sizeof(T)*M, cudaMemcpyDeviceToDevice, streams[STREAM][1]));
                gpu_error_check(cudaMemcpyAsync(&d_q_pair[STREAM][prev][M], d_propagator_right, sizeof(T)*M, cudaMemcpyDeviceToDevice, streams[STREAM][1]));
                gpu_error_check(cudaEventRecord(memcpy_done, streams[STREAM][1]));
            }
            gpu_error_check(cudaStreamWaitEvent(streams[STREAM][0], memcpy_done, 0));

            for(int n=0; n<=N_RIGHT; n++)
            {
                // STREAM 1: copy memory from device to device
                if (n+1 <= N_RIGHT)
                {
                    d_propagator_left  = std::get<0>(_block_stress_compuation_key[n+1]);
                    d_propagator_right = std::get<1>(_block_stress_compuation_key[n+1]);

                    if (d_propagator_left != nullptr)
                    {
                        gpu_error_check(cudaMemcpyAsync(&d_q_pair[STREAM][next][0], d_propagator_left,  sizeof(T)*M, cudaMemcpyDeviceToDevice, streams[STREAM][1]));
                        gpu_error_check(cudaMemcpyAsync(&d_q_pair[STREAM][next][M], d_propagator_right, sizeof(T)*M, cudaMemcpyDeviceToDevice, streams[STREAM][1]));
                        gpu_error_check(cudaEventRecord(memcpy_done, streams[STREAM][1]));
                    }
                }

                // STREAM 0: Compute stress
                d_propagator_left = std::get<0>(_block_stress_compuation_key[n]);
                bool is_half_bond_length = std::get<2>(_block_stress_compuation_key[n]);
                std::vector<T> segment_stress;
                if (d_propagator_left != nullptr)
                {
                    segment_stress = propagator_solver->compute_single_segment_stress(
                        STREAM, d_q_pair[STREAM][prev], monomer_type, is_half_bond_length);
                    gpu_error_check(cudaEventRecord(kernel_done, streams[STREAM][0]));
                }

                // Wait until computation and memory copy are done
                gpu_error_check(cudaStreamWaitEvent(streams[STREAM][1], kernel_done, 0));
                gpu_error_check(cudaStreamWaitEvent(streams[STREAM][0], memcpy_done, 0));

                if (d_propagator_left != nullptr)
                {
                    for(int d=0; d<DIM; d++)
                        _block_dq_dl[d] += segment_stress[d]*static_cast<double>(n_repeated);
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
        }
        gpu_error_check(cudaDeviceSynchronize());

        // Compute total stress
        int n_polymer_types = this->molecules->get_n_polymer_types();
        for(int p=0; p<n_polymer_types; p++)
            for(int d=0; d<DIM; d++)
                this->dq_dl[p][d] = 0.0;
        for(const auto& d_block: d_phi_block)
        {
            const auto& key       = d_block.first;
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
template <typename T>
void CudaComputationDiscrete<T>::get_chain_propagator(T *q_out, int polymer, int v, int u, int n)
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

        gpu_error_check(cudaMemcpy(q_out, d_propagator[dep][n], sizeof(T)*M,cudaMemcpyDeviceToHost));
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
bool CudaComputationDiscrete<T>::check_total_partition()
{
    const int M = this->cb->get_total_grid();
    int n_polymer_types = this->molecules->get_n_polymer_types();
    std::vector<std::vector<T>> total_partitions;
    for(int p=0;p<n_polymer_types;p++)
    {
        std::vector<T> total_partitions_p;
        total_partitions.push_back(total_partitions_p);
    }

    for(const auto& block: d_phi_block)
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
        CuDeviceData<T> *_d_exp_dw = propagator_solver->d_exp_dw[monomer_type];

        #ifndef NDEBUG
        std::cout<< p << ", " << key_left << ", " << key_right << ": " << n_segment_left << ", " << n_segment_right << ", " << n_propagators << ", " << this->propagator_computation_optimizer->get_computation_block(key).n_repeated << std::endl;
        #endif

        for(int n=1;n<=n_segment_right;n++)
        {
            T total_partition = dynamic_cast<CudaComputationBox<T>*>(this->cb)->inner_product_inverse_weight_device(
                    d_propagator[key_left][(n_segment_left-n+1)],
                    d_propagator[key_right][n], _d_exp_dw);

            total_partition *= n_repeated/this->cb->get_volume()/n_propagators;
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
            if (std::abs(total_partitions[p][n]) > max_partition)
                max_partition = std::abs(total_partitions[p][n]);
            if (std::abs(total_partitions[p][n]) < min_partition)
                min_partition = std::abs(total_partitions[p][n]);
        }
        double diff_partition = std::abs(max_partition - min_partition);

        std::cout<< "\t" << p << ": " << max_partition << ", " << min_partition << ", " << diff_partition << std::endl;
        if (diff_partition > 1e-7)
            return false;
    }
    return true;
}

// Explicit template instantiation
template class CudaComputationDiscrete<double>;
template class CudaComputationDiscrete<std::complex<double>>;