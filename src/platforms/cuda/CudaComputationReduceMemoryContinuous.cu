#include <complex>
#include <omp.h>
#include "CudaComputationReduceMemoryContinuous.h"
#include "CudaComputationBox.h"
#include "CudaSolverPseudoContinuous.h"
#include "CudaSolverRealSpace.h"
#include "SimpsonRule.h"

template <typename T>
CudaComputationReduceMemoryContinuous<T>::CudaComputationReduceMemoryContinuous(
    ComputationBox<T>* cb,
    Molecules *molecules,
    PropagatorComputationOptimizer *propagator_computation_optimizer,
    std::string method)
    : PropagatorComputation<T>(cb, molecules, propagator_computation_optimizer)
{
    try{
        #ifndef NDEBUG
        std::cout << "--------- Continuous Chain Solver, GPU Memoery Saving Version ---------" << std::endl;
        #endif

        const int M = this->cb->get_total_grid();

        // The number of parallel streams is always 1 to reduce the memory usage
        n_streams = 1;
        #ifndef NDEBUG
        std::cout << "The number of CPU threads is always set to " << n_streams << "." << std::endl;
        #endif

        // Copy streams
        for(int i=0; i<n_streams; i++)
        {
            gpu_error_check(cudaStreamCreate(&streams[i][0])); // for kernel execution
            gpu_error_check(cudaStreamCreate(&streams[i][1])); // for memcpy
        }

        this->method = method;
        if(method == "pseudospectral")
            this->propagator_solver = new CudaSolverPseudoContinuous<T>(cb, molecules, n_streams, streams, false);
        else if(method == "realspace")
        {
            if constexpr (std::is_same<T, double>::value) 
                this->propagator_solver = new CudaSolverRealSpace(cb, molecules, n_streams, streams, false);
            else
                throw_with_line_number("Currently, the realspace method is only available for double precision.");
        }

        // Allocate memory for propagators
        if( this->propagator_computation_optimizer->get_computation_propagators().size() == 0)
            throw_with_line_number("There is no propagator code. Add polymers first.");
        for(const auto& item: this->propagator_computation_optimizer->get_computation_propagators())
        {
            std::string key = item.first;
            int max_n_segment = item.second.max_n_segment+1;

            #ifndef NDEBUG
            propagator_finished[key] = new bool[max_n_segment];
            for(int n=0; n<max_n_segment;n++)
                propagator_finished[key][n] = false;
            #endif
        }

        // Allocate memory for concentrations
        if( this->propagator_computation_optimizer->get_computation_blocks().size() == 0)
            throw_with_line_number("There is no block. Add polymers first.");
        for(const auto& item: this->propagator_computation_optimizer->get_computation_blocks())
        {
            phi_block[item.first] = nullptr;
            // Allocate pinned memory
            gpu_error_check(cudaMallocHost((void**)&phi_block[item.first], sizeof(T)*M));
        }

        // Allocate memory for check points
        if( this->propagator_computation_optimizer->get_computation_propagators().size() == 0)
            throw_with_line_number("There is no propagator code. Add polymers first.");
        for(const auto& item: this->propagator_computation_optimizer->get_computation_propagators())
        {
            std::string key = item.first;
            auto& deps = this->propagator_computation_optimizer->get_computation_propagator(key).deps;
            int max_n_segment = this->propagator_computation_optimizer->get_computation_propagator(key).max_n_segment;

            if(check_point_propagator.find(std::make_tuple(key, 0)) == check_point_propagator.end())
            {
                check_point_propagator[std::make_tuple(key, 0)] = nullptr;
                gpu_error_check(cudaMallocHost((void**)&check_point_propagator[std::make_tuple(key, 0)], sizeof(T)*M));
            }

            // Add 'max_n_segment' to pass the tests
            if(check_point_propagator.find(std::make_tuple(key, max_n_segment)) == check_point_propagator.end())
            {
                check_point_propagator[std::make_tuple(key, max_n_segment)] = nullptr;
                gpu_error_check(cudaMallocHost((void**)&check_point_propagator[std::make_tuple(key, max_n_segment)], sizeof(T)*M));
            }

            for(size_t d=0; d<deps.size(); d++)
            {
                std::string sub_dep = std::get<0>(deps[d]);
                int sub_n_segment   = std::get<1>(deps[d]);

                if(check_point_propagator.find(std::make_tuple(sub_dep, sub_n_segment)) == check_point_propagator.end())
                {
                    check_point_propagator[std::make_tuple(sub_dep, sub_n_segment)] = nullptr;
                    gpu_error_check(cudaMallocHost((void**)&check_point_propagator[std::make_tuple(sub_dep, sub_n_segment)], sizeof(T)*M));
                }
            }

            #ifndef NDEBUG
            propagator_finished[key] = new bool[max_n_segment];
            for(int i=0; i<max_n_segment;i++)
                propagator_finished[key][i] = false;
            #endif
        }

        // Find the total maximum n_segment and allocate temporary memory for recalculating propagators
        total_max_n_segment = 0;
        for(const auto& block: phi_block)
        {
            const auto& key = block.first;
            int n_segment_right = this->propagator_computation_optimizer->get_computation_block(key).n_segment_right;
            total_max_n_segment = std::max(total_max_n_segment, n_segment_right);
        }
        this->q_recal = new T*[total_max_n_segment+1];
        for(int n=0; n<=total_max_n_segment; n++)
        {
            this->q_recal[n] = nullptr;
            gpu_error_check(cudaMallocHost((void**)&this->q_recal[n], sizeof(T)*M));
        }
        // this->q_pair[0] = new T[M];
        // this->q_pair[1] = new T[M];

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

            if(check_point_propagator.find(std::make_tuple(key_left, n_segment_left)) == check_point_propagator.end())
            {
                check_point_propagator[std::make_tuple(key_left, n_segment_left)] = nullptr;
                gpu_error_check(cudaMallocHost((void**)&check_point_propagator[std::make_tuple(key_left, n_segment_left)], sizeof(T)*M));
                #ifndef NDEBUG
                std::cout << "Allocated, " + key_left + ", " << n_segment_left << std::endl;
                #endif
            }

            if(check_point_propagator.find(std::make_tuple(key_right, 0)) == check_point_propagator.end())
            {
                check_point_propagator[std::make_tuple(key_right, 0)] = nullptr;
                gpu_error_check(cudaMallocHost((void**)&check_point_propagator[std::make_tuple(key_right, 0)], sizeof(T)*M));
                #ifndef NDEBUG
                std::cout << "Allocated, " + key_right + ", " << 0 << std::endl;
                #endif
            }

            single_partition_segment.push_back(std::make_tuple(
                p,
                check_point_propagator[std::make_tuple(key_left, n_segment_left)],  // q
                check_point_propagator[std::make_tuple(key_right, 0)],              // q_dagger
                n_aggregated                                                        // how many propagators are aggregated
                ));
            current_p++;
        }
        // Concentrations for each solvent
        for(int s=0;s<this->molecules->get_n_solvent_types();s++)
            phi_solvent.push_back(new T[M]);

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

        // Allocate memory for propagator computation
        for(int i=0; i<n_streams; i++)
        {
            gpu_error_check(cudaMalloc((void**)&d_q_one[i][0], sizeof(T)*M)); // for prev
            gpu_error_check(cudaMalloc((void**)&d_q_one[i][1], sizeof(T)*M)); // for next
            gpu_error_check(cudaMalloc((void**)&d_propagator_sub_dep[i][0], sizeof(T)*M)); // for prev
            gpu_error_check(cudaMalloc((void**)&d_propagator_sub_dep[i][1], sizeof(T)*M)); // for next
        }

        // For concentration computation
        gpu_error_check(cudaMalloc((void**)&d_q_block_v[0], sizeof(T)*M)); // for prev
        gpu_error_check(cudaMalloc((void**)&d_q_block_v[1], sizeof(T)*M)); // for next
        gpu_error_check(cudaMalloc((void**)&d_q_block_u[0], sizeof(T)*M)); // for prev
        gpu_error_check(cudaMalloc((void**)&d_q_block_u[1], sizeof(T)*M)); // for next
        gpu_error_check(cudaMalloc((void**)&d_phi,          sizeof(T)*M));

        // Allocate memory for stress calculation: compute_stress()
        for(int i=0; i<n_streams; i++)
        {
            gpu_error_check(cudaMalloc((void**)&d_q_pair[i][0], sizeof(T)*2*M)); // prev
            gpu_error_check(cudaMalloc((void**)&d_q_pair[i][1], sizeof(T)*2*M)); // next
        }

        // Copy mask to d_q_mask
        if (this->cb->get_mask() != nullptr)
        {
            gpu_error_check(cudaMalloc((void**)&d_q_mask , sizeof(double)*M));
            gpu_error_check(cudaMemcpy(d_q_mask, this->cb->get_mask(), sizeof(double)*M, cudaMemcpyHostToDevice));
        }
        else
            d_q_mask = nullptr;

        propagator_solver->update_laplacian_operator();
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
CudaComputationReduceMemoryContinuous<T>::~CudaComputationReduceMemoryContinuous()
{
    delete propagator_solver;
    delete sc;

    for(int n=0; n<=total_max_n_segment; n++)
        cudaFreeHost(this->q_recal[n]);
    delete[] this->q_recal;

    for(const auto& item: check_point_propagator)
        cudaFreeHost(item.second);

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
    if (d_q_mask != nullptr)
        cudaFree(d_q_mask);
    cudaFree(d_q_unity);
    
    // Destroy streams
    for(int i=0; i<n_streams; i++)
    {
        cudaStreamDestroy(streams[i][0]);
        cudaStreamDestroy(streams[i][1]);
    }
}
template <typename T>
void CudaComputationReduceMemoryContinuous<T>::update_laplacian_operator()
{
    try{
        propagator_solver->update_laplacian_operator();
    }
    catch(std::exception& exc)
    {
        throw_with_line_number(exc.what());
    }
}
template <typename T>
void CudaComputationReduceMemoryContinuous<T>::compute_statistics(
    std::map<std::string, const T*> w_input,
    std::map<std::string, const T*> q_init)
{
    this->compute_propagators(w_input, q_init);
    this->compute_concentrations();
}
template <typename T>
void CudaComputationReduceMemoryContinuous<T>::compute_propagators(
    std::map<std::string, const T*> w_input,
    std::map<std::string, const T*> q_init)
{
    try{
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
        #ifndef NDEBUG
        for(const auto& item: this->propagator_computation_optimizer->get_computation_propagators())
        {
            std::string key = item.first;
            int max_n_segment = item.second.max_n_segment+1;
            for(int i=0; i<max_n_segment;i++)
                propagator_finished[key][i] = false;
        }
        #endif

        auto& branch_schedule = sc->get_schedule();
        for (auto parallel_job = branch_schedule.begin(); parallel_job != branch_schedule.end(); parallel_job++)
        {
            // For each propagator
            for(size_t job=0; job<parallel_job->size(); job++)
            {
                const int STREAM = omp_get_thread_num();

                // printf("gpu, STREAM: %d, %d\n ", gpu, STREAM);

                auto& key = std::get<0>((*parallel_job)[job]);
                int n_segment_from = std::get<1>((*parallel_job)[job]);
                int n_segment_to = std::get<2>((*parallel_job)[job]);
                auto& deps = this->propagator_computation_optimizer->get_computation_propagator(key).deps;
                auto monomer_type = this->propagator_computation_optimizer->get_computation_propagator(key).monomer_type;

                // if (STREAM == 0)
                //     std::cout << gpu << ", " << STREAM << ", " << n_segment_from << ", " << n_segment_to << ", " << monomer_type << ", " << key << ", "  << std::endl;    

                // Check key
                #ifndef NDEBUG
                if (check_point_propagator.find(std::make_tuple(key, n_segment_from)) == check_point_propagator.end())
                    std::cout << "Could not find key '" + key + "'. " << std::endl;
                #endif

                // If it is leaf node
                if(n_segment_from == 0 && deps.size() == 0)
                {
                    // q_init
                    if (key[0] == '{')
                    {
                        std::string g = PropagatorCode::get_q_input_idx_from_key(key);
                        if (q_init.find(g) == q_init.end())
                            std::cout<< "Could not find q_init[\"" + g + "\"]." << std::endl;
                        gpu_error_check(cudaMemcpy(d_q_one[STREAM][0], q_init[g], sizeof(T)*M, cudaMemcpyInputToDevice));
                    }
                    else
                    {
                        gpu_error_check(cudaMemcpy(d_q_one[STREAM][0], d_q_unity, sizeof(T)*M, cudaMemcpyDeviceToDevice));
                    }

                    #ifndef NDEBUG
                    propagator_finished[key][0] = true;
                    #endif
                }
                // If it is not leaf node
                else if (n_segment_from == 0 && deps.size() > 0)
                {
                    // If it is aggregated
                    if (key[0] == '[')
                    {
                        // Initialize to zero
                        gpu_error_check(cudaMemset(d_q_one[STREAM][0], 0, sizeof(T)*M));

                        int prev, next;
                        prev = 0;
                        next = 1;

                        // Copy memory from host to device
                        std::string sub_dep = std::get<0>(deps[0]);
                        int sub_n_segment   = std::get<1>(deps[0]);
                        int sub_n_repeated  = std::get<2>(deps[0]);

                        T* _q_sub = check_point_propagator[std::make_tuple(sub_dep, sub_n_segment)];
                        gpu_error_check(cudaMemcpy(d_propagator_sub_dep[STREAM][prev], _q_sub, sizeof(T)*M, cudaMemcpyHostToDevice));

                        for(size_t d=0; d<deps.size(); d++)
                        {
                            sub_dep         = std::get<0>(deps[d]);
                            sub_n_segment   = std::get<1>(deps[d]);
                            sub_n_repeated  = std::get<2>(deps[d]);

                            // Check sub key
                            #ifndef NDEBUG
                            if (check_point_propagator.find(std::make_tuple(sub_dep, sub_n_segment)) == check_point_propagator.end())
                                std::cout << "Could not find sub key '" + sub_dep + "[" + std::to_string(sub_n_segment) + "]. " << std::endl;
                            if (!propagator_finished[sub_dep][sub_n_segment])
                                std::cout<< "Could not compute '" + key +  "', since '"+ sub_dep + std::to_string(sub_n_segment) + "' is not prepared." << std::endl;
                            #endif

                            // MEMORY STREAM: copy memory from host to device
                            if (d < deps.size()-1)
                            {
                                std::string sub_dep_next = std::get<0>(deps[d+1]);
                                int sub_n_segment_next   = std::get<1>(deps[d+1]);

                                T* _q_sub = check_point_propagator[std::make_tuple(sub_dep_next, sub_n_segment_next)];
                                gpu_error_check(cudaMemcpyAsync(d_propagator_sub_dep[STREAM][next],
                                                _q_sub, sizeof(T)*M,
                                                cudaMemcpyHostToDevice, streams[STREAM][1]));
                            }

                            // KERNEL STREAM: compute linear combination
                            ker_lin_comb<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                                    d_q_one[STREAM][0], 1.0, d_q_one[STREAM][0],
                                    sub_n_repeated, d_propagator_sub_dep[STREAM][prev], M);

                            std::swap(prev, next);
                            cudaDeviceSynchronize();
                        }

                        #ifndef NDEBUG
                        propagator_finished[key][0] = true;
                        #endif
                    }
                    else if(key[0] == '(')
                    {
                        // Initialize to one
                        gpu_error_check(cudaMemcpy(d_q_one[STREAM][0], d_q_unity, sizeof(T)*M, cudaMemcpyDeviceToDevice));

                        int prev, next;
                        prev = 0;
                        next = 1;

                        // Copy memory from host to device
                        std::string sub_dep = std::get<0>(deps[0]);
                        int sub_n_segment   = std::get<1>(deps[0]);
                        int sub_n_repeated  = std::get<2>(deps[0]);

                        T* _q_sub = check_point_propagator[std::make_tuple(sub_dep, sub_n_segment)];
                        gpu_error_check(cudaMemcpy(d_propagator_sub_dep[STREAM][prev], _q_sub, sizeof(T)*M, cudaMemcpyHostToDevice));

                        for(size_t d=0; d<deps.size(); d++)
                        {
                            sub_dep         = std::get<0>(deps[d]);
                            sub_n_segment   = std::get<1>(deps[d]);
                            sub_n_repeated  = std::get<2>(deps[d]);

                            // Check sub key
                            #ifndef NDEBUG
                            if (check_point_propagator.find(std::make_tuple(sub_dep, sub_n_segment)) == check_point_propagator.end())
                                std::cout << "Could not find sub key '" + sub_dep + "[" + std::to_string(sub_n_segment) + "]. " << std::endl;
                            if (!propagator_finished[sub_dep][sub_n_segment])
                                std::cout<< "Could not compute '" + key +  "', since '"+ sub_dep + std::to_string(sub_n_segment) + "' is not prepared." << std::endl;
                            #endif

                            // MEMORY STREAM: copy memory from host to device
                            if (d < deps.size()-1)
                            {
                                std::string sub_dep_next = std::get<0>(deps[d+1]);
                                int sub_n_segment_next   = std::get<1>(deps[d+1]);

                                T* _q_sub = check_point_propagator[std::make_tuple(sub_dep_next, sub_n_segment_next)];
                                gpu_error_check(cudaMemcpyAsync(d_propagator_sub_dep[STREAM][next],
                                                _q_sub, sizeof(T)*M,
                                                cudaMemcpyHostToDevice, streams[STREAM][1]));
                            }

                            // KERNEL STREAM: multiply
                            for(int r=0; r<sub_n_repeated; r++)
                            {
                                ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                                    d_q_one[STREAM][0], d_q_one[STREAM][0], d_propagator_sub_dep[STREAM][prev], 1.0, M);
                            }

                            std::swap(prev, next);

                            gpu_error_check(cudaStreamSynchronize(streams[STREAM][0]));
                            gpu_error_check(cudaStreamSynchronize(streams[STREAM][1]));
                        }

                        #ifndef NDEBUG
                        propagator_finished[key][0] = true;
                        #endif
                    }
                }

                // Multiply mask
                if (n_segment_from == 0 && d_q_mask != nullptr)
                    ker_multi<<<N_BLOCKS, N_THREADS>>>(d_q_one[STREAM][0], d_q_one[STREAM][0], d_q_mask, 1.0, M);

                // Copy data between device and host
                if (n_segment_from == 0)
                {
                    T* _q_target = check_point_propagator[std::make_tuple(key, 0)];
                    gpu_error_check(cudaMemcpy(_q_target, d_q_one[STREAM][0], sizeof(T)*M, cudaMemcpyDeviceToHost));
                }
                else
                {
                    T* _q_from = check_point_propagator[std::make_tuple(key, n_segment_from)];
                    gpu_error_check(cudaMemcpy(d_q_one[STREAM][0], _q_from, sizeof(T)*M, cudaMemcpyHostToDevice));
                }

                int prev, next;
                prev = 0;
                next = 1;

                // Create events
                cudaEvent_t kernel_done;
                cudaEvent_t memcpy_done;
                gpu_error_check(cudaEventCreate(&kernel_done));
                gpu_error_check(cudaEventCreate(&memcpy_done));

                for(int n=n_segment_from; n<n_segment_to; n++)
                {
                    #ifndef NDEBUG
                    if (!propagator_finished[key][n])
                        std::cout << "unfinished, key: " + key + ", " + std::to_string(n) << std::endl;
                    if (propagator_finished[key][n+1])
                        std::cout << "already finished: " + key + ", " + std::to_string(n+1) << std::endl;
                    #endif

                    // KERNEL STREAM: calculate propagators
                    propagator_solver->advance_propagator(
                        STREAM, 
                        d_q_one[STREAM][prev],
                        d_q_one[STREAM][next],
                        monomer_type, d_q_mask);
                    gpu_error_check(cudaEventRecord(kernel_done, streams[STREAM][0]));

                    // MEMORY STREAM: copy memory from device to host
                    if (n > n_segment_from && check_point_propagator.find(std::make_tuple(key, n)) != check_point_propagator.end())
                    {
                        T* _q_target =  check_point_propagator[std::make_tuple(key, n)];
                        gpu_error_check(cudaMemcpyAsync(
                            _q_target,
                            d_q_one[STREAM][prev],
                            sizeof(T)*M, cudaMemcpyDeviceToHost, streams[STREAM][1]));
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

                // Copy memory from device 1 to device 0
                if (check_point_propagator.find(std::make_tuple(key, n_segment_to)) != check_point_propagator.end())
                {
                    T* _q_target =  check_point_propagator[std::make_tuple(key, n_segment_to)];
                    gpu_error_check(cudaMemcpyAsync(
                        _q_target, d_q_one[STREAM][prev],
                        sizeof(T)*M, cudaMemcpyDeviceToHost, streams[STREAM][1]));
                }

                gpu_error_check(cudaStreamSynchronize(streams[STREAM][0]));
                gpu_error_check(cudaStreamSynchronize(streams[STREAM][1]));
            
                gpu_error_check(cudaEventDestroy(kernel_done));
                gpu_error_check(cudaEventDestroy(memcpy_done));
            }
            gpu_error_check(cudaDeviceSynchronize());
        }

        // Compute total partition function of each distinct polymers
        for(const auto& segment_info: single_partition_segment)
        {
            int p               = std::get<0>(segment_info);
            T *propagator_left  = std::get<1>(segment_info);
            T *propagator_right = std::get<2>(segment_info);
            int n_aggregated    = std::get<3>(segment_info);

            this->single_polymer_partitions[p]= this->cb->inner_product(
                propagator_left, propagator_right)/(n_aggregated*this->cb->get_volume());
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
void CudaComputationReduceMemoryContinuous<T>::advance_propagator_single_segment(
    T* q_init, T *q_out, std::string monomer_type)
{
    try
    {
        const int M = this->cb->get_total_grid();
        const int STREAM = 0;
        gpu_error_check(cudaMemcpy(d_q_pair[STREAM][0], q_init, sizeof(T)*M, cudaMemcpyHostToDevice));

        propagator_solver->advance_propagator(
                        STREAM, d_q_pair[STREAM][0], d_q_pair[STREAM][1],
                        monomer_type, d_q_mask);
        gpu_error_check(cudaDeviceSynchronize());
        
        gpu_error_check(cudaMemcpy(q_out, d_q_pair[STREAM][1], sizeof(T)*M, cudaMemcpyDeviceToHost));
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
void CudaComputationReduceMemoryContinuous<T>::compute_concentrations()
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();
        const int M = this->cb->get_total_grid();

        // Calculate segment concentrations
        for(const auto& block: phi_block)
        {
            const auto& key = block.first;
            int p                 = std::get<0>(key);
            std::string key_left  = std::get<1>(key);
            std::string key_right = std::get<2>(key);

            int n_segment_right = this->propagator_computation_optimizer->get_computation_block(key).n_segment_right;
            int n_segment_left  = this->propagator_computation_optimizer->get_computation_block(key).n_segment_left;
            int n_repeated      = this->propagator_computation_optimizer->get_computation_block(key).n_repeated;
            std::string monomer_type = this->propagator_computation_optimizer->get_computation_block(key).monomer_type;

            // If there is no segment
            if(n_segment_right == 0)
            {
                for(int i=0; i<M;i++)
                    block.second[i] = 0.0;
                continue;
            }

            // Check keys
            #ifndef NDEBUG
            if (check_point_propagator.find(std::make_tuple(key_left, n_segment_left-n_segment_right)) == check_point_propagator.end())
                std::cout << "Check point at " + key_left + "[" + std::to_string(n_segment_left-n_segment_right) + "] is missing. ";
            if (check_point_propagator.find(std::make_tuple(key_right, 0)) == check_point_propagator.end())
                std::cout << "Check point at " + key_right + "[" + std::to_string(0) + "] is missing. ";
            #endif

            // Normalization constant
            Polymer& pc = this->molecules->get_polymer(p);
            T norm = (this->molecules->get_ds()*pc.get_volume_fraction()/pc.get_alpha()*n_repeated)/this->single_polymer_partitions[p];

            // Calculate phi of one block (possibly multiple blocks when using aggregation)
            calculate_phi_one_block(
                block.second,           // phi
                key_left,               // dependency v
                key_right,              // dependency u
                n_segment_left,
                n_segment_right,
                monomer_type,
                norm
            );
        }
        // Calculate partition functions and concentrations of solvents
        for(int s=0; s<this->molecules->get_n_solvent_types(); s++)
        {
            double volume_fraction = std::get<0>(this->molecules->get_solvent(s));
            std::string monomer_type = std::get<1>(this->molecules->get_solvent(s));
            CuDeviceData<T> *_d_exp_dw = propagator_solver->d_exp_dw[monomer_type];

            this->single_solvent_partitions[s] = dynamic_cast<CudaComputationBox<T>*>(this->cb)->inner_product_device(_d_exp_dw, _d_exp_dw)/this->cb->get_volume();

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

            ker_multi<<<N_BLOCKS, N_THREADS>>>(d_phi, _d_exp_dw, _d_exp_dw, norm, M);
            gpu_error_check(cudaMemcpy(phi_solvent[s], d_phi, sizeof(T)*M, cudaMemcpyDeviceToHost));
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
void CudaComputationReduceMemoryContinuous<T>::recalcaulte_propagator(T **q_out, std::string key, const int N_START, const int N_RIGHT, std::string monomer_type)
{
    // If a propagator is in check_point_propagator reuse it, otherwise compute it again with allocated memory space.
    try
    {
        const int M = this->cb->get_total_grid();
        int prev = 0;
        int next = 1;

        // Assign check_point_propagator to a pointer
        q_out[0] = check_point_propagator[std::make_tuple(key, N_START)];

        // Copy propagators from host to device
        gpu_error_check(cudaMemcpy(d_q_one[0][prev], q_out[0], sizeof(T)*M, cudaMemcpyHostToDevice));

        // Compute the q_out
        for(int n=0; n<=N_RIGHT; n++)
        {
            // MEMORY STREAM: copy memory from device to host
            if (n > 0)
            {
                // Use check_point_propagator if exists
                if(check_point_propagator.find(std::make_tuple(key, N_START+n)) != check_point_propagator.end())
                {
                    q_out[n] = check_point_propagator[std::make_tuple(key, N_START+n)];
                    #ifndef NDEBUG
                    std::cout << "Use check_point_propagator if exists: (phi, left) " << key << ", " << N_START+n << std::endl;
                    #endif
                }
                // Assign q_recal memory space, and copy the next propagator from computation result
                else if (n > 0)
                {
                    q_out[n] = this->q_recal[n];
                    gpu_error_check(cudaMemcpyAsync(
                        q_out[n], d_q_one[0][prev],
                        sizeof(T)*M, cudaMemcpyDeviceToHost, streams[0][1]));
                }
            }
            // KERNEL STREAM: calculate propagator
            if ((n+1 < N_RIGHT) ||
                // Compute the last, if the q_out[N_START+N_RIGHT] is not in check_point_propagator
                (n+1 == N_RIGHT && check_point_propagator.find(std::make_tuple(key, N_START+N_RIGHT)) == check_point_propagator.end()))
            {
                propagator_solver->advance_propagator(
                    0,
                    d_q_one[0][prev],   // q_out[n]
                    d_q_one[0][next],   // q_out[n+1]
                    monomer_type, d_q_mask);
            }
            std::swap(prev, next);
            cudaDeviceSynchronize();
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
void CudaComputationReduceMemoryContinuous<T>::calculate_phi_one_block(
    T *phi, std::string key_left, std::string key_right, const int N_LEFT, const int N_RIGHT, std::string monomer_type, const T NORM)
{
    try
    {
        // In this method, propagators are recalculcated from the check points
        // If a propagator is in check_point_propagator reuse it, otherwise compute it again with allocated memory space.
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();
        const int M = this->cb->get_total_grid();
        std::vector<double> simpson_rule_coeff = SimpsonRule::get_coeff(N_RIGHT);

        // An array of pointers for q_left
        T* q_left[total_max_n_segment];

        // Recalcaulte q_left from the check point
        recalcaulte_propagator(q_left, key_left, N_LEFT-N_RIGHT, N_RIGHT, monomer_type);

        int prev = 0;
        int next = 1;

        // Copy propagators from host to device
        gpu_error_check(cudaMemcpy(d_q_block_v[prev], q_left[N_RIGHT],
            sizeof(T)*M, cudaMemcpyHostToDevice));
        gpu_error_check(cudaMemcpy(d_q_block_u[prev], check_point_propagator[std::make_tuple(key_right, 0)],
            sizeof(T)*M, cudaMemcpyHostToDevice));

        // Initialize to zero
        gpu_error_check(cudaMemset(d_phi, 0, sizeof(T)*M));

        // Compute the q_right and segment concentration
        for(int n=0; n<=N_RIGHT; n++)
        {
            // MEMORY STREAM: (left) copy propagators from host to device
            if (n < N_RIGHT)
            {
                gpu_error_check(cudaMemcpyAsync(d_q_block_v[next], q_left[N_RIGHT-(n+1)],
                    sizeof(T)*M, cudaMemcpyHostToDevice, streams[0][1]));
            }

            // KERNEL STREAM: (right) calculate propagators
            if (n < N_RIGHT)
            {
                propagator_solver->advance_propagator(
                    0, 
                    d_q_block_u[prev],
                    d_q_block_u[next],
                    monomer_type, d_q_mask);
            }
            
            CuDeviceData<T> norm;
            if constexpr (std::is_same<T, double>::value)
                norm = NORM*simpson_rule_coeff[n];
            else
                norm = stdToCuDoubleComplex(NORM*simpson_rule_coeff[n]);

            // KERNEL STREAM: multiply two propagators
            ker_add_multi<<<N_BLOCKS, N_THREADS, 0, streams[0][0]>>>(d_phi, d_q_block_v[prev], d_q_block_u[prev], norm, M);
            std::swap(prev, next);
            cudaDeviceSynchronize();
        }

        // Copy propagators from device to host
        gpu_error_check(cudaMemcpy(phi, d_phi, sizeof(T)*M, cudaMemcpyDeviceToHost));
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
T CudaComputationReduceMemoryContinuous<T>::get_total_partition(int polymer)
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
void CudaComputationReduceMemoryContinuous<T>::get_total_concentration(std::string monomer_type, T *phi)
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
                T *phi_solvent_ = phi_solvent[s];
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
template <typename T>
void CudaComputationReduceMemoryContinuous<T>::get_total_concentration(int p, std::string monomer_type, T *phi)
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
template <typename T>
void CudaComputationReduceMemoryContinuous<T>::get_total_concentration_gce(double fugacity, int p, std::string monomer_type, T *phi)
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
                T norm = fugacity/pc.get_volume_fraction()*pc.get_alpha()*this->single_polymer_partitions[p];
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
template <typename T>
void CudaComputationReduceMemoryContinuous<T>::get_block_concentration(int p, T *phi)
{
    try
    {
        const int M = this->cb->get_total_grid();
        const int P = this->molecules->get_n_polymer_types();

        if (p < 0 || p > P-1)
            throw_with_line_number("Index (" + std::to_string(p) + ") must be in range [0, " + std::to_string(P-1) + "]");

        if (this->propagator_computation_optimizer->use_aggregation())
            throw_with_line_number("Disable 'aggregation' option to invoke 'get_block_concentration'.");

        Polymer& pc = this->molecules->get_polymer(p);
        std::vector<Block>& blocks = pc.get_blocks();

        for(size_t b=0; b<blocks.size(); b++)
        {
            std::string key_left  = pc.get_propagator_key(blocks[b].v, blocks[b].u);
            std::string key_right = pc.get_propagator_key(blocks[b].u, blocks[b].v);
            if (key_left < key_right)
                key_left.swap(key_right);

            T* _essential_phi_block = phi_block[std::make_tuple(p, key_left, key_right)];
            for(int i=0; i<M; i++)
                phi[i+b*M] = _essential_phi_block[i]; 
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
T CudaComputationReduceMemoryContinuous<T>::get_solvent_partition(int s)
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
void CudaComputationReduceMemoryContinuous<T>::get_solvent_concentration(int s, T *phi)
{
    try
    {
        const int M = this->cb->get_total_grid();
        const int S = this->molecules->get_n_solvent_types();

        if (s < 0 || s > S-1)
            throw_with_line_number("Index (" + std::to_string(s) + ") must be in range [0, " + std::to_string(S-1) + "]");

        T *phi_solvent_ = phi_solvent[s];
        for(int i=0; i<M; i++)
            phi[i] = phi_solvent_[i];
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
void CudaComputationReduceMemoryContinuous<T>::compute_stress()
{
    // This method should be invoked after invoking compute_statistics().

    // To calculate stress, we multiply weighted fourier basis to q(k)*q^dagger(-k).
    // We only need the real part of stress calculation.

    try
    {
        // if constexpr (std::is_same<T, std::complex<double>>::value)
        //     throw_with_line_number("Currently, stress computation is not suppoted for complex number type.");

        if (this->method == "realspace")
            throw_with_line_number("Currently, the real-space method does not support stress computation.");

        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();
        const int DIM = this->cb->get_dim();
        const int M   = this->cb->get_total_grid();

        std::map<std::tuple<int, std::string, std::string>, std::array<T,3>> block_dq_dl[n_streams];

        // Reset stress map
        for(const auto& item: phi_block)
        {
            for(int i=0; i<n_streams; i++)
                for(int d=0; d<3; d++)
                    block_dq_dl[i][item.first][d] = 0.0;
        }

        // Compute stress for each block
        for(size_t b=0; b<phi_block.size();b++)
        {
            const int STREAM = omp_get_thread_num();
            auto block = phi_block.begin();
            advance(block, b);
            const auto& key   = block->first;

            int p                 = std::get<0>(key);
            std::string key_left  = std::get<1>(key);
            std::string key_right = std::get<2>(key);

            const int N_RIGHT = this->propagator_computation_optimizer->get_computation_block(key).n_segment_right;
            const int N_LEFT  = this->propagator_computation_optimizer->get_computation_block(key).n_segment_left;
            std::string monomer_type = this->propagator_computation_optimizer->get_computation_block(key).monomer_type;
            int n_repeated = this->propagator_computation_optimizer->get_computation_block(key).n_repeated;

            // If there is no segment
            if(N_RIGHT == 0)
                continue;

            std::vector<double> s_coeff = SimpsonRule::get_coeff(N_RIGHT);

            std::array<T,3> _block_dq_dl;
            for(int i=0; i<3; i++)
                _block_dq_dl[i] = 0.0;

            CuDeviceData<T> *d_segment_stress;
            T segment_stress[DIM];
            gpu_error_check(cudaMalloc((void**)&d_segment_stress, sizeof(T)*3));

            // An array of pointers for q_left
            T* q_left[total_max_n_segment];

            // Recalcaulte q_left from the check point
            recalcaulte_propagator(q_left, key_left, N_LEFT-N_RIGHT, N_RIGHT, monomer_type);

            int prev = 0;
            int next = 1;

            // Copy propagators from host to device
            gpu_error_check(cudaMemcpy(&d_q_pair[0][prev][0], q_left[N_RIGHT], 
                sizeof(T)*M, cudaMemcpyHostToDevice));
            gpu_error_check(cudaMemcpy(&d_q_pair[0][prev][M], check_point_propagator[std::make_tuple(key_right, 0)],
                sizeof(T)*M, cudaMemcpyHostToDevice));

            // Compute the q_right and segment concentration
            for(int n=0; n<=N_RIGHT; n++)
            {
                // MEMORY STREAM: (left) copy propagators from host to device
                if (n < N_RIGHT)
                {
                    gpu_error_check(cudaMemcpyAsync(&d_q_pair[0][next][0], q_left[N_RIGHT-(n+1)],
                        sizeof(T)*M, cudaMemcpyHostToDevice, streams[0][1]));
                }

                // KERNEL STREAM: (right) calculate propagators
                if (n < N_RIGHT)
                {
                    propagator_solver->advance_propagator(
                        0, 
                        &d_q_pair[0][prev][M],
                        &d_q_pair[0][next][M],
                        monomer_type, d_q_mask);
                }
                // KERNEL STREAM: Compute stress
                propagator_solver->compute_single_segment_stress(
                    0, d_q_pair[0][prev], d_segment_stress,
                    monomer_type, false);

                std::swap(prev, next);
                cudaDeviceSynchronize();

                gpu_error_check(cudaMemcpy(segment_stress, d_segment_stress, sizeof(T)*DIM, cudaMemcpyDeviceToHost));
                for(int d=0; d<DIM; d++)
                    _block_dq_dl[d] += segment_stress[d]*(s_coeff[n]*n_repeated);
            }
            for(int d=0; d<DIM; d++)
                block_dq_dl[STREAM][key][d] += _block_dq_dl[d];

            cudaFree(d_segment_stress);
        }
        gpu_error_check(cudaDeviceSynchronize());

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
template <typename T>
void CudaComputationReduceMemoryContinuous<T>::get_chain_propagator(T *q_out, int polymer, int v, int u, int n)
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
        if (n < 0 || n > N_RIGHT)
            throw_with_line_number("n (" + std::to_string(n) + ") must be in range [0, " + std::to_string(N_RIGHT) + "]");

        if (check_point_propagator.find(std::make_tuple(dep, n)) == check_point_propagator.end())
            throw_with_line_number("The propagator " + dep + "[" + std::to_string(n) + "] is not stored'. Disable 'reduce_memory_usage' option to obtain propagator_computation_optimizer.");

        T *_q_from = check_point_propagator[std::make_tuple(dep, n)];
        for(int i=0; i<M; i++)
            q_out[i] = _q_from[i];
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
bool CudaComputationReduceMemoryContinuous<T>::check_total_partition()
{
    const int M = this->cb->get_total_grid();
    int n_polymer_types = this->molecules->get_n_polymer_types();
    std::vector<std::vector<T>> total_partitions;
    for(int p=0;p<n_polymer_types;p++)
    {
        std::vector<T> total_partitions_p;
        total_partitions.push_back(total_partitions_p);
    }

    for(const auto& block: phi_block)
    {
        const auto& key = block.first;
        int p                 = std::get<0>(key);
        std::string key_left  = std::get<1>(key);
        std::string key_right = std::get<2>(key);

        std::string monomer_type = this->propagator_computation_optimizer->get_computation_block(key).monomer_type;
        int N_RIGHT              = this->propagator_computation_optimizer->get_computation_block(key).n_segment_right;
        int N_LEFT               = this->propagator_computation_optimizer->get_computation_block(key).n_segment_left;
        int n_repeated           = this->propagator_computation_optimizer->get_computation_block(key).n_repeated;
        int n_propagators        = this->propagator_computation_optimizer->get_computation_block(key).v_u.size();

        #ifndef NDEBUG
        std::cout<< p << ", " << key_left << ", " << key_right << ": " << N_LEFT << ", " << N_RIGHT << ", " << n_propagators << ", " << n_repeated << std::endl;
        #endif

        // An array of pointers for q_left
        T* q_left[total_max_n_segment];

        // Recalcaulte q_left from the check point
        recalcaulte_propagator(q_left, key_left, N_LEFT-N_RIGHT, N_RIGHT, monomer_type);

        int prev = 0;
        int next = 1;

        // Copy propagators from host to device
        gpu_error_check(cudaMemcpy(d_q_block_v[prev], q_left[N_RIGHT], 
            sizeof(T)*M, cudaMemcpyHostToDevice));
        gpu_error_check(cudaMemcpy(d_q_block_u[prev], check_point_propagator[std::make_tuple(key_right, 0)],
            sizeof(T)*M, cudaMemcpyHostToDevice));

        // Compute the q_right and segment concentration
        for(int n=0; n<=N_RIGHT; n++)
        {
            // MEMORY STREAM: (left) copy propagators from host to device
            if (n < N_RIGHT)
            {
                gpu_error_check(cudaMemcpyAsync(d_q_block_v[next], q_left[N_RIGHT-(n+1)],
                    sizeof(T)*M, cudaMemcpyHostToDevice, streams[0][1]));
            }

            // KERNEL STREAM: (right) calculate propagators
            if (n < N_RIGHT)
            {
                propagator_solver->advance_propagator(
                    0, 
                    d_q_block_u[prev],
                    d_q_block_u[next],
                    monomer_type, d_q_mask);
            }
            // KERNEL STREAM: multiply two propagators
            T total_partition = dynamic_cast<CudaComputationBox<T>*>(this->cb)->inner_product_device(
                d_q_block_v[prev], d_q_block_u[prev]);

            total_partition *= n_repeated/this->cb->get_volume()/n_propagators;
            total_partitions[p].push_back(total_partition);

            #ifndef NDEBUG
            std::cout<< p << ", " << n << ": " << total_partition << std::endl;
            #endif

            std::swap(prev, next);
            cudaDeviceSynchronize();
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
template class CudaComputationReduceMemoryContinuous<double>;
template class CudaComputationReduceMemoryContinuous<std::complex<double>>;