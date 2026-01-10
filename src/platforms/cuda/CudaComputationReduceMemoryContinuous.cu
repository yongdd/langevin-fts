/**
 * @file CudaComputationReduceMemoryContinuous.cu
 * @brief Memory-efficient CUDA propagator computation for continuous chains.
 *
 * Implements checkpointing strategy that stores only selected propagator
 * values, recomputing intermediate values when needed. Trades computation
 * time (2-4x slower) for significantly reduced GPU memory usage.
 *
 * **Checkpointing Strategy:**
 *
 * - propagator_at_check_point: Stores propagators at key contour positions
 * - Check points include: s=0, block junctions, block ends
 * - Intermediate propagators recomputed from nearest checkpoint
 *
 * **Memory Layout:**
 *
 * - Uses pinned (page-locked) host memory for checkpoints
 * - Only 2 GPU arrays needed for current computation
 * - q_recal: Temporary host arrays for recomputation
 *
 * **Concentration Calculation:**
 *
 * For each block:
 * 1. Load checkpoint q(r, N_LEFT-N_RIGHT) from host
 * 2. Recompute forward propagators q(r, N_LEFT-n) for n=0..N_RIGHT
 * 3. Simultaneously advance backward propagator
 * 4. Accumulate concentration with Simpson's rule
 *
 * **Stream Usage:**
 *
 * Single stream (n_streams=1) to minimize memory:
 * - streams[0][0]: Kernel execution
 * - streams[0][1]: Host-device memory transfers
 *
 * **Template Instantiations:**
 *
 * - CudaComputationReduceMemoryContinuous<double>: Real field
 * - CudaComputationReduceMemoryContinuous<std::complex<double>>: Complex field
 *
 * @see CudaComputationContinuous for full memory version
 */

#include <complex>
#include <cmath>
#include <vector>
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

            if(propagator_at_check_point.find(std::make_tuple(key, 0)) == propagator_at_check_point.end())
            {
                propagator_at_check_point[std::make_tuple(key, 0)] = nullptr;
                gpu_error_check(cudaMallocHost((void**)&propagator_at_check_point[std::make_tuple(key, 0)], sizeof(T)*M));
            }

            for(size_t d=0; d<deps.size(); d++)
            {
                std::string sub_dep = std::get<0>(deps[d]);
                int sub_n_segment   = std::get<1>(deps[d]);

                if(propagator_at_check_point.find(std::make_tuple(sub_dep, sub_n_segment)) == propagator_at_check_point.end())
                {
                    propagator_at_check_point[std::make_tuple(sub_dep, sub_n_segment)] = nullptr;
                    gpu_error_check(cudaMallocHost((void**)&propagator_at_check_point[std::make_tuple(sub_dep, sub_n_segment)], sizeof(T)*M));
                }
            }

            #ifndef NDEBUG
            propagator_finished[key] = new bool[max_n_segment];
            for(int i=0; i<max_n_segment;i++)
                propagator_finished[key][i] = false;
            #endif
        }

        // Find the total sum of segments and allocate temporary memory for recalculating propagators
        total_max_n_segment = 0;
        for(const auto& block: phi_block)
        {
            const auto& key = block.first;
            int n_segment_right = this->propagator_computation_optimizer->get_computation_block(key).n_segment_right;
            total_max_n_segment += n_segment_right;
        }

        // Calculate checkpoint interval as 2*sqrt(N) for better memory-computation tradeoff
        checkpoint_interval = static_cast<int>(std::ceil(2.0*std::sqrt(static_cast<double>(total_max_n_segment))));
        if (checkpoint_interval < 1)
            checkpoint_interval = 1;

        #ifndef NDEBUG
        std::cout << "total_max_n_segment: " << total_max_n_segment << std::endl;
        std::cout << "checkpoint_interval: " << checkpoint_interval << std::endl;
        #endif

        // Allocate workspace for propagator recomputation
        // We skip intermediate values from checkpoint to left_start, then store from left_start to left_end
        // This requires only checkpoint_interval storage
        const int workspace_size = checkpoint_interval;
        this->q_recal.resize(workspace_size);
        for(int n=0; n<workspace_size; n++)
        {
            this->q_recal[n] = nullptr;
            gpu_error_check(cudaMallocHost((void**)&this->q_recal[n], sizeof(T)*M));
        }

        // Allocate checkpoints at sqrt(N) intervals for each propagator
        for(const auto& item: this->propagator_computation_optimizer->get_computation_propagators())
        {
            std::string key = item.first;
            int max_n_segment = item.second.max_n_segment;

            // Add checkpoints at sqrt(N) intervals
            for(int n=checkpoint_interval; n<max_n_segment; n+=checkpoint_interval)
            {
                if(propagator_at_check_point.find(std::make_tuple(key, n)) == propagator_at_check_point.end())
                {
                    propagator_at_check_point[std::make_tuple(key, n)] = nullptr;
                    gpu_error_check(cudaMallocHost((void**)&propagator_at_check_point[std::make_tuple(key, n)], sizeof(T)*M));
                    #ifndef NDEBUG
                    std::cout << "Allocated checkpoint, " + key + ", " << n << std::endl;
                    #endif
                }
            }
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

            if(propagator_at_check_point.find(std::make_tuple(key_left, n_segment_left)) == propagator_at_check_point.end())
            {
                propagator_at_check_point[std::make_tuple(key_left, n_segment_left)] = nullptr;
                gpu_error_check(cudaMallocHost((void**)&propagator_at_check_point[std::make_tuple(key_left, n_segment_left)], sizeof(T)*M));
                #ifndef NDEBUG
                std::cout << "Allocated, " + key_left + ", " << n_segment_left << std::endl;
                #endif
            }

            if(propagator_at_check_point.find(std::make_tuple(key_right, 0)) == propagator_at_check_point.end())
            {
                propagator_at_check_point[std::make_tuple(key_right, 0)] = nullptr;
                gpu_error_check(cudaMallocHost((void**)&propagator_at_check_point[std::make_tuple(key_right, 0)], sizeof(T)*M));
                #ifndef NDEBUG
                std::cout << "Allocated, " + key_right + ", " << 0 << std::endl;
                #endif
            }

            single_partition_segment.push_back(std::make_tuple(
                p,
                propagator_at_check_point[std::make_tuple(key_left, n_segment_left)],  // q
                propagator_at_check_point[std::make_tuple(key_right, 0)],              // q_dagger
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
            gpu_error_check(cudaMalloc((void**)&d_q_one[i][0], sizeof(T)*M));
            gpu_error_check(cudaMalloc((void**)&d_q_one[i][1], sizeof(T)*M));
        }

        // Allocate shared workspace (2×M each, contiguous for batch FFT in stress computation)
        // d_workspace[0]: primary buffer, d_workspace[1]: ping-pong buffer
        // Phase 1 (compute_propagators): uses d_propagator_sub_dep aliases
        // Phase 2 (compute_concentrations): uses q_left, q_right, d_phi (see memory layout in header)
        // Phase 3 (compute_stress): uses d_workspace directly (contiguous 2×M)
        gpu_error_check(cudaMalloc((void**)&d_workspace[0], sizeof(T)*2*M));
        gpu_error_check(cudaMalloc((void**)&d_workspace[1], sizeof(T)*2*M));

        // Set up aliases
        d_propagator_sub_dep[0][0] = d_workspace[0];      // first M of d_workspace[0]
        d_propagator_sub_dep[0][1] = d_workspace[0] + M;  // second M of d_workspace[0]
        d_phi = d_workspace[1] + M;                       // second M of d_workspace[1]

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

    // Free workspace (must match constructor allocation)
    const int workspace_size = checkpoint_interval;
    for(int n=0; n<workspace_size; n++)
        cudaFreeHost(this->q_recal[n]);

    for(const auto& item: propagator_at_check_point)
        cudaFreeHost(item.second);

    for(const auto& item: phi_block)
        cudaFreeHost(item.second);
    for(const auto& item: phi_solvent)
        delete[] item;

    #ifndef NDEBUG
    for(const auto& item: propagator_finished)
        delete[] item.second;
    #endif

    // Free GPU workspace
    // Only free actual allocations (d_propagator_sub_dep, d_phi are aliases into d_workspace)
    for(int i=0; i<n_streams; i++)
    {
        cudaFree(d_q_one[i][0]);
        cudaFree(d_q_one[i][1]);
    }
    cudaFree(d_workspace[0]);  // contains d_propagator_sub_dep[0][*], used for q_left, q_right[0] in concentration
    cudaFree(d_workspace[1]);  // contains q_right[1] ping-pong buffer and d_phi

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
                if (propagator_at_check_point.find(std::make_tuple(key, n_segment_from)) == propagator_at_check_point.end())
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

                        T* _q_sub = propagator_at_check_point[std::make_tuple(sub_dep, sub_n_segment)];
                        gpu_error_check(cudaMemcpy(d_propagator_sub_dep[STREAM][prev], _q_sub, sizeof(T)*M, cudaMemcpyHostToDevice));

                        for(size_t d=0; d<deps.size(); d++)
                        {
                            sub_dep         = std::get<0>(deps[d]);
                            sub_n_segment   = std::get<1>(deps[d]);
                            sub_n_repeated  = std::get<2>(deps[d]);

                            // Check sub key
                            #ifndef NDEBUG
                            if (propagator_at_check_point.find(std::make_tuple(sub_dep, sub_n_segment)) == propagator_at_check_point.end())
                                std::cout << "Could not find sub key '" + sub_dep + "[" + std::to_string(sub_n_segment) + "]. " << std::endl;
                            if (!propagator_finished[sub_dep][sub_n_segment])
                                std::cout<< "Could not compute '" + key +  "', since '"+ sub_dep + std::to_string(sub_n_segment) + "' is not prepared." << std::endl;
                            #endif

                            // MEMORY STREAM: copy memory from host to device
                            if (d < deps.size()-1)
                            {
                                std::string sub_dep_next = std::get<0>(deps[d+1]);
                                int sub_n_segment_next   = std::get<1>(deps[d+1]);

                                T* _q_sub = propagator_at_check_point[std::make_tuple(sub_dep_next, sub_n_segment_next)];
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

                        T* _q_sub = propagator_at_check_point[std::make_tuple(sub_dep, sub_n_segment)];
                        gpu_error_check(cudaMemcpy(d_propagator_sub_dep[STREAM][prev], _q_sub, sizeof(T)*M, cudaMemcpyHostToDevice));

                        for(size_t d=0; d<deps.size(); d++)
                        {
                            sub_dep         = std::get<0>(deps[d]);
                            sub_n_segment   = std::get<1>(deps[d]);
                            sub_n_repeated  = std::get<2>(deps[d]);

                            // Check sub key
                            #ifndef NDEBUG
                            if (propagator_at_check_point.find(std::make_tuple(sub_dep, sub_n_segment)) == propagator_at_check_point.end())
                                std::cout << "Could not find sub key '" + sub_dep + "[" + std::to_string(sub_n_segment) + "]. " << std::endl;
                            if (!propagator_finished[sub_dep][sub_n_segment])
                                std::cout<< "Could not compute '" + key +  "', since '"+ sub_dep + std::to_string(sub_n_segment) + "' is not prepared." << std::endl;
                            #endif

                            // MEMORY STREAM: copy memory from host to device
                            if (d < deps.size()-1)
                            {
                                std::string sub_dep_next = std::get<0>(deps[d+1]);
                                int sub_n_segment_next   = std::get<1>(deps[d+1]);

                                T* _q_sub = propagator_at_check_point[std::make_tuple(sub_dep_next, sub_n_segment_next)];
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
                    T* _q_target = propagator_at_check_point[std::make_tuple(key, 0)];
                    gpu_error_check(cudaMemcpy(_q_target, d_q_one[STREAM][0], sizeof(T)*M, cudaMemcpyDeviceToHost));
                }
                else
                {
                    T* _q_from = propagator_at_check_point[std::make_tuple(key, n_segment_from)];
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
                    if (n > n_segment_from && propagator_at_check_point.find(std::make_tuple(key, n)) != propagator_at_check_point.end())
                    {
                        T* _q_target =  propagator_at_check_point[std::make_tuple(key, n)];
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
                if (propagator_at_check_point.find(std::make_tuple(key, n_segment_to)) != propagator_at_check_point.end())
                {
                    T* _q_target =  propagator_at_check_point[std::make_tuple(key, n_segment_to)];
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
        gpu_error_check(cudaMemcpy(d_workspace[0], q_init, sizeof(T)*M, cudaMemcpyHostToDevice));

        propagator_solver->advance_propagator(
                        STREAM, d_workspace[0], d_workspace[1],
                        monomer_type, d_q_mask);
        gpu_error_check(cudaDeviceSynchronize());

        gpu_error_check(cudaMemcpy(q_out, d_workspace[1], sizeof(T)*M, cudaMemcpyDeviceToHost));
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
            if (propagator_at_check_point.find(std::make_tuple(key_left, n_segment_left-n_segment_right)) == propagator_at_check_point.end())
                std::cout << "Check point at " + key_left + "[" + std::to_string(n_segment_left-n_segment_right) + "] is missing. ";
            if (propagator_at_check_point.find(std::make_tuple(key_right, 0)) == propagator_at_check_point.end())
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
std::vector<T*> CudaComputationReduceMemoryContinuous<T>::recalcaulte_propagator(std::string key, const int N_START, const int N_RIGHT, std::string monomer_type)
{
    try
    {
        const int M = this->cb->get_total_grid();
        int prev = 0;
        int next = 1;

        // An array of pointers for q_out (use dynamic container to avoid VLA/stack overflow)
        std::vector<T*> q_out(total_max_n_segment + 1);

        // If a propagator is in propagator_at_check_point reuse it, otherwise compute it again with allocated memory space.
        for(int n=0; n<=N_RIGHT; n++)
        {
            auto it = propagator_at_check_point.find(std::make_tuple(key, N_START+n));
            if(it != propagator_at_check_point.end())
            {
                q_out[n] = it->second;
                #ifndef NDEBUG
                std::cout << "Use propagator_at_check_point if exists: (phi, left) " << key << ", " << N_START+n << std::endl;
                #endif
            }
            else
            {
                q_out[n] = this->q_recal[n];
            }
        }
        // Copy propagators from host to device
        gpu_error_check(cudaMemcpy(d_q_one[0][prev], q_out[0], sizeof(T)*M, cudaMemcpyHostToDevice));

        // Compute the q_out
        for(int n=0; n<=N_RIGHT; n++)
        {
            // MEMORY STREAM: copy memory from device to host
            if (n > 0)
            {
                // Use propagator_at_check_point if exists
                if(propagator_at_check_point.find(std::make_tuple(key, N_START+n)) == propagator_at_check_point.end())
                {
                    gpu_error_check(cudaMemcpyAsync(
                        q_out[n], d_q_one[0][prev],
                        sizeof(T)*M, cudaMemcpyDeviceToHost, streams[0][1]));
                }
            }
            // KERNEL STREAM: calculate propagator
            if ((n+1 < N_RIGHT) ||
                // Compute the last, if the q_out[N_START+N_RIGHT] is not in propagator_at_check_point
                (n+1 == N_RIGHT && propagator_at_check_point.find(std::make_tuple(key, N_START+N_RIGHT)) == propagator_at_check_point.end()))
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
        return q_out;
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
        // Uses block-based computation for memory efficiency
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();
        const int M = this->cb->get_total_grid();
        const int STREAM = 0;
        const int k = checkpoint_interval;

        // Set up local workspace pointers (see memory layout in header)
        CuDeviceData<T> *d_q_left = d_workspace[0];                           // first M of d_workspace[0]
        CuDeviceData<T> *d_q_right[2] = {d_workspace[0] + M, d_workspace[1]}; // ping-pong buffers

        std::vector<double> simpson_rule_coeff = SimpsonRule::get_coeff(N_RIGHT);

        // Initialize to zero
        gpu_error_check(cudaMemset(d_phi, 0, sizeof(T)*M));

        // Initialize q_right at segment 0
        gpu_error_check(cudaMemcpy(d_q_right[0], propagator_at_check_point[std::make_tuple(key_right, 0)],
            sizeof(T)*M, cudaMemcpyHostToDevice));
        int right_prev_idx = 0;
        int right_next_idx = 1;
        int current_n_right = 0;

        // Process n = 0 to N_RIGHT using block-based computation
        int num_blocks = (N_RIGHT + k) / k;

        for (int blk = 0; blk < num_blocks; blk++)
        {
            int n_start = blk * k;  // First n in this block
            int n_end = std::min((blk + 1) * k - 1, N_RIGHT);  // Last n in this block (non-overlapping)

            // q_left positions needed: segment (N_LEFT - n) for n in [n_start, n_end]
            // = segments [N_LEFT - n_end, N_LEFT - n_start]
            int left_start = N_LEFT - n_end;
            int left_end = N_LEFT - n_start;

            // For continuous chains, valid segments are 0 to N_LEFT
            int left_compute_start = std::max(left_start, 0);

            // Find checkpoint at or before left_compute_start
            int check_pos = (left_compute_start / k) * k;
            if (check_pos < 0) check_pos = 0;

            // Find the actual checkpoint
            auto it = propagator_at_check_point.find(std::make_tuple(key_left, check_pos));
            while (it == propagator_at_check_point.end() && check_pos > 0)
            {
                check_pos -= k;
                if (check_pos < 0) check_pos = 0;
                it = propagator_at_check_point.find(std::make_tuple(key_left, check_pos));
            }

            // Skip from checkpoint to left_start (don't store intermediate values)
            gpu_error_check(cudaMemcpy(d_q_one[STREAM][0], it->second, sizeof(T)*M, cudaMemcpyHostToDevice));
            for (int i = check_pos; i < left_start; i++)
            {
                propagator_solver->advance_propagator(STREAM, d_q_one[STREAM][0], d_q_one[STREAM][1], monomer_type, d_q_mask);
                gpu_error_check(cudaMemcpy(d_q_one[STREAM][0], d_q_one[STREAM][1], sizeof(T)*M, cudaMemcpyDeviceToDevice));
            }

            // Store q_left from left_start to left_end
            gpu_error_check(cudaMemcpy(q_recal[0], d_q_one[STREAM][0], sizeof(T)*M, cudaMemcpyDeviceToHost));
            for (int i = 1; i <= left_end - left_start; i++)
            {
                propagator_solver->advance_propagator(STREAM, d_q_one[STREAM][0], d_q_one[STREAM][1], monomer_type, d_q_mask);
                gpu_error_check(cudaMemcpy(q_recal[i], d_q_one[STREAM][1], sizeof(T)*M, cudaMemcpyDeviceToHost));
                gpu_error_check(cudaMemcpy(d_q_one[STREAM][0], d_q_one[STREAM][1], sizeof(T)*M, cudaMemcpyDeviceToDevice));
            }
            // Now q_recal[i] = q_left[left_start + i]

            // Process each n in [n_start, n_end]
            for (int n = n_start; n <= n_end; n++)
            {
                // Advance q_right to position n
                while (current_n_right < n)
                {
                    propagator_solver->advance_propagator(STREAM, d_q_right[right_prev_idx], d_q_right[right_next_idx], monomer_type, d_q_mask);
                    cudaDeviceSynchronize();
                    std::swap(right_prev_idx, right_next_idx);
                    current_n_right++;
                }

                // Get q_left position (segment N_LEFT - n)
                int left_pos = N_LEFT - n;
                int ws_idx = left_pos - left_start;

                // Copy q_left to device
                gpu_error_check(cudaMemcpy(d_q_left, q_recal[ws_idx], sizeof(T)*M, cudaMemcpyHostToDevice));

                CuDeviceData<T> norm;
                if constexpr (std::is_same<T, double>::value)
                    norm = NORM*simpson_rule_coeff[n];
                else
                    norm = stdToCuDoubleComplex(NORM*simpson_rule_coeff[n]);

                // Multiply and accumulate
                ker_add_multi<<<N_BLOCKS, N_THREADS>>>(d_phi, d_q_left, d_q_right[right_prev_idx], norm, M);
            }
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
    // Uses block-based computation to minimize memory usage (O(sqrt(N)) workspace).

    try
    {
        if (this->method == "realspace")
            throw_with_line_number("Currently, the real-space method does not support stress computation.");

        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();
        const int DIM = this->cb->get_dim();
        const int M   = this->cb->get_total_grid();
        const int k = checkpoint_interval;

        const int N_STRESS = 6;
        std::map<std::tuple<int, std::string, std::string>, std::array<T,N_STRESS>> block_dq_dl[n_streams];

        // Reset stress map
        for(const auto& item: phi_block)
        {
            for(int i=0; i<n_streams; i++)
                for(int d=0; d<N_STRESS; d++)
                    block_dq_dl[i][item.first][d] = 0.0;
        }

        // Compute stress for each phi_block
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

            std::array<T,N_STRESS> _block_dq_dl;
            for(int i=0; i<N_STRESS; i++)
                _block_dq_dl[i] = 0.0;

            // Number of stress components
            const bool is_ortho = this->cb->is_orthogonal();
            const int N_STRESS_DIM = (DIM == 3) ? (is_ortho ? 3 : 6) : ((DIM == 2) ? 3 : 1);
            CuDeviceData<T> *d_segment_stress;
            T segment_stress[6];
            gpu_error_check(cudaMalloc((void**)&d_segment_stress, sizeof(T)*N_STRESS_DIM));

            // Number of blocks
            const int num_blocks = (N_RIGHT + k) / k;

            // Pointers for q_right (ping-pong)
            int right_prev_idx = 0;
            int right_next_idx = 1;

            // Current position of q_right
            int current_n_right = -1;

            // Initialize q_right at position 0
            auto it_right_0 = propagator_at_check_point.find(std::make_tuple(key_right, 0));
            if(it_right_0 != propagator_at_check_point.end())
            {
                gpu_error_check(cudaMemcpy(&d_workspace[right_prev_idx][M], it_right_0->second, sizeof(T)*M, cudaMemcpyHostToDevice));
                current_n_right = 0;
            }

            // Process each block
            for(int blk = 0; blk < num_blocks; blk++)
            {
                const int n_start = blk * k;
                const int n_end = std::min((blk + 1) * k, N_RIGHT + 1) - 1;

                // q_left positions needed: [N_LEFT - n_end, N_LEFT - n_start]
                const int left_start = N_LEFT - n_end;
                const int left_end = N_LEFT - n_start;

                // Find the best checkpoint at or before left_start
                int check_pos = -1;
                for(int cp = 0; cp <= left_start; cp++)
                {
                    if(propagator_at_check_point.find(std::make_tuple(key_left, cp)) != propagator_at_check_point.end())
                        check_pos = cp;
                }

                if(check_pos < 0)
                    continue;

                // Recompute q_left from checkpoint to left_end using q_recal workspace
                const int steps_before = left_start - check_pos;
                const int storage_count = left_end - left_start + 1;

                auto it_checkpoint = propagator_at_check_point.find(std::make_tuple(key_left, check_pos));
                if(it_checkpoint == propagator_at_check_point.end())
                    continue;

                // Load checkpoint to device and compute q_left positions
                gpu_error_check(cudaMemcpy(d_q_one[0][0], it_checkpoint->second, sizeof(T)*M, cudaMemcpyHostToDevice));
                int left_ping = 0;
                int left_pong = 1;

                // Advance to left_start
                for(int step = 0; step < steps_before; step++)
                {
                    int actual_pos = check_pos + step + 1;
                    auto it = propagator_at_check_point.find(std::make_tuple(key_left, actual_pos));
                    if(it != propagator_at_check_point.end())
                    {
                        gpu_error_check(cudaMemcpy(d_q_one[0][left_pong], it->second, sizeof(T)*M, cudaMemcpyHostToDevice));
                    }
                    else
                    {
                        propagator_solver->advance_propagator(0, d_q_one[0][left_ping], d_q_one[0][left_pong], monomer_type, d_q_mask);
                        cudaDeviceSynchronize();
                    }
                    std::swap(left_ping, left_pong);
                }

                // Now d_q_one[0][left_ping] contains q_left[left_start]
                // Copy to q_recal[0] and compute remaining positions
                gpu_error_check(cudaMemcpy(q_recal[0], d_q_one[0][left_ping], sizeof(T)*M, cudaMemcpyDeviceToHost));

                for(int idx = 1; idx < storage_count; idx++)
                {
                    int actual_pos = left_start + idx;
                    auto it = propagator_at_check_point.find(std::make_tuple(key_left, actual_pos));
                    if(it != propagator_at_check_point.end())
                    {
                        for(int i=0; i<M; i++)
                            q_recal[idx][i] = it->second[i];
                        gpu_error_check(cudaMemcpy(d_q_one[0][left_pong], it->second, sizeof(T)*M, cudaMemcpyHostToDevice));
                    }
                    else
                    {
                        propagator_solver->advance_propagator(0, d_q_one[0][left_ping], d_q_one[0][left_pong], monomer_type, d_q_mask);
                        cudaDeviceSynchronize();
                        gpu_error_check(cudaMemcpy(q_recal[idx], d_q_one[0][left_pong], sizeof(T)*M, cudaMemcpyDeviceToHost));
                    }
                    std::swap(left_ping, left_pong);
                }

                // Process each n in [n_start, n_end]
                for(int n = n_start; n <= n_end; n++)
                {
                    // Advance q_right if needed
                    auto it_right = propagator_at_check_point.find(std::make_tuple(key_right, n));
                    if(it_right != propagator_at_check_point.end())
                    {
                        gpu_error_check(cudaMemcpy(&d_workspace[right_prev_idx][M], it_right->second, sizeof(T)*M, cudaMemcpyHostToDevice));
                        current_n_right = n;
                    }
                    else
                    {
                        while(current_n_right < n)
                        {
                            propagator_solver->advance_propagator(0, &d_workspace[right_prev_idx][M], &d_workspace[right_next_idx][M], monomer_type, d_q_mask);
                            cudaDeviceSynchronize();
                            std::swap(right_prev_idx, right_next_idx);
                            current_n_right++;
                        }
                    }

                    // Get q_left[N_LEFT - n] from q_recal
                    int left_pos = N_LEFT - n;
                    int storage_idx = left_pos - left_start;

                    // Copy q_left to device for stress computation
                    gpu_error_check(cudaMemcpy(&d_workspace[right_prev_idx][0], q_recal[storage_idx], sizeof(T)*M, cudaMemcpyHostToDevice));

                    // Compute stress (d_workspace[idx] is contiguous 2×M buffer)
                    propagator_solver->compute_single_segment_stress(0, d_workspace[right_prev_idx], d_segment_stress, monomer_type, false);
                    cudaDeviceSynchronize();

                    gpu_error_check(cudaMemcpy(segment_stress, d_segment_stress, sizeof(T)*N_STRESS_DIM, cudaMemcpyDeviceToHost));
                    for(int d=0; d<N_STRESS_DIM; d++)
                        _block_dq_dl[d] += segment_stress[d]*(s_coeff[n]*n_repeated);
                }
            }
            for(int d=0; d<N_STRESS_DIM; d++)
                block_dq_dl[STREAM][key][d] += _block_dq_dl[d];

            cudaFree(d_segment_stress);
        }
        gpu_error_check(cudaDeviceSynchronize());

        // Compute total stress
        // N_STRESS_TOTAL: 3D orthogonal->3, 3D non-orthogonal->6, 2D->3, 1D->1
        const bool is_ortho = this->cb->is_orthogonal();
        const int N_STRESS_TOTAL = (DIM == 3) ? (is_ortho ? 3 : 6) : ((DIM == 2) ? 3 : 1);
        int n_polymer_types = this->molecules->get_n_polymer_types();
        for(int p=0; p<n_polymer_types; p++)
            for(int d=0; d<N_STRESS_TOTAL; d++)
                this->dq_dl[p][d] = 0.0;
        for(const auto& block: phi_block)
        {
            const auto& key       = block.first;
            int p                 = std::get<0>(key);
            std::string key_left  = std::get<1>(key);
            std::string key_right = std::get<2>(key);
            Polymer& pc = this->molecules->get_polymer(p);

            for(int i=0; i<n_streams; i++)
                for(int d=0; d<N_STRESS_TOTAL; d++)
                    this->dq_dl[p][d] += block_dq_dl[i][key][d];
        }
        for(int p=0; p<n_polymer_types; p++)
        {
            // Diagonal components: xx, yy, zz
            for(int d=0; d<DIM; d++)
                this->dq_dl[p][d] /= -3.0*this->cb->get_lx(d)*M*M/this->molecules->get_ds();
            // Cross-term components for 3D: xy, xz, yz
            if (DIM == 3)
            {
                this->dq_dl[p][3] /= -3.0*std::sqrt(this->cb->get_lx(0)*this->cb->get_lx(1))*M*M/this->molecules->get_ds();
                this->dq_dl[p][4] /= -3.0*std::sqrt(this->cb->get_lx(0)*this->cb->get_lx(2))*M*M/this->molecules->get_ds();
                this->dq_dl[p][5] /= -3.0*std::sqrt(this->cb->get_lx(1)*this->cb->get_lx(2))*M*M/this->molecules->get_ds();
            }
            // Cross-term component for 2D: yz
            else if (DIM == 2)
            {
                this->dq_dl[p][2] /= -3.0*std::sqrt(this->cb->get_lx(0)*this->cb->get_lx(1))*M*M/this->molecules->get_ds();
            }
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
    // Uses block-based computation with reduced workspace (O(sqrt(N)))

    try
    {
        const int M = this->cb->get_total_grid();
        Polymer& pc = this->molecules->get_polymer(polymer);
        std::string dep = pc.get_propagator_key(v,u);

        if (this->propagator_computation_optimizer->get_computation_propagators().find(dep) == this->propagator_computation_optimizer->get_computation_propagators().end())
            throw_with_line_number("Could not find the propagator code '" + dep + "'. Disable 'aggregation' option to obtain propagator_computation_optimizer.");

        const int max_n_segment = this->propagator_computation_optimizer->get_computation_propagator(dep).max_n_segment;
        if (n < 0 || n > max_n_segment)
            throw_with_line_number("n (" + std::to_string(n) + ") must be in range [0, " + std::to_string(max_n_segment) + "]");

        // Check if the requested segment is at a checkpoint
        auto checkpoint_key = std::make_tuple(dep, n);
        if (propagator_at_check_point.find(checkpoint_key) != propagator_at_check_point.end())
        {
            // Directly copy from checkpoint
            T *_q_from = propagator_at_check_point[checkpoint_key];
            for(int i=0; i<M; i++)
                q_out[i] = _q_from[i];
        }
        else
        {
            // Find nearest checkpoint at or before position n
            std::string monomer_type = this->propagator_computation_optimizer->get_computation_propagator(dep).monomer_type;
            int check_pos = -1;
            for(int cp = 0; cp <= n; cp++)
            {
                if(propagator_at_check_point.find(std::make_tuple(dep, cp)) != propagator_at_check_point.end())
                    check_pos = cp;
            }

            if(check_pos < 0)
                throw_with_line_number("No checkpoint found at or before position " + std::to_string(n));

            // Load checkpoint and advance to position n using ping-pong buffers
            T* q_checkpoint = propagator_at_check_point[std::make_tuple(dep, check_pos)];
            gpu_error_check(cudaMemcpy(d_q_one[0][0], q_checkpoint, sizeof(T)*M, cudaMemcpyHostToDevice));

            int ping = 0;
            int pong = 1;
            for(int step = check_pos; step < n; step++)
            {
                // Check if next position is a checkpoint
                auto it = propagator_at_check_point.find(std::make_tuple(dep, step + 1));
                if(it != propagator_at_check_point.end())
                {
                    gpu_error_check(cudaMemcpy(d_q_one[0][pong], it->second, sizeof(T)*M, cudaMemcpyHostToDevice));
                }
                else
                {
                    propagator_solver->advance_propagator(0, d_q_one[0][ping], d_q_one[0][pong], monomer_type, d_q_mask);
                    cudaDeviceSynchronize();
                }
                std::swap(ping, pong);
            }

            // Copy result from device to output
            gpu_error_check(cudaMemcpy(q_out, d_q_one[0][ping], sizeof(T)*M, cudaMemcpyDeviceToHost));
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
bool CudaComputationReduceMemoryContinuous<T>::check_total_partition()
{
    // Uses block-based computation to minimize memory usage (O(sqrt(N)) workspace).
    const int M = this->cb->get_total_grid();
    const int k = checkpoint_interval;
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

        // Number of blocks
        const int num_blocks = (N_RIGHT + k) / k;

        // Set up local workspace pointers (see memory layout in header)
        CuDeviceData<T> *d_q_left = d_workspace[0];                           // first M of d_workspace[0]
        CuDeviceData<T> *d_q_right[2] = {d_workspace[0] + M, d_workspace[1]}; // ping-pong buffers

        // Pointers for q_right (ping-pong)
        int right_prev_idx = 0;
        int right_next_idx = 1;

        // Current position of q_right
        int current_n_right = -1;

        // Initialize q_right at position 0
        auto it_right_0 = propagator_at_check_point.find(std::make_tuple(key_right, 0));
        if(it_right_0 != propagator_at_check_point.end())
        {
            gpu_error_check(cudaMemcpy(d_q_right[right_prev_idx], it_right_0->second, sizeof(T)*M, cudaMemcpyHostToDevice));
            current_n_right = 0;
        }

        // Process each block
        for(int blk = 0; blk < num_blocks; blk++)
        {
            const int n_start = blk * k;
            const int n_end = std::min((blk + 1) * k, N_RIGHT + 1) - 1;

            // q_left positions needed: [N_LEFT - n_end, N_LEFT - n_start]
            const int left_start = N_LEFT - n_end;
            const int left_end = N_LEFT - n_start;

            // Find the best checkpoint at or before left_start
            int check_pos = -1;
            for(int cp = 0; cp <= left_start; cp++)
            {
                if(propagator_at_check_point.find(std::make_tuple(key_left, cp)) != propagator_at_check_point.end())
                    check_pos = cp;
            }

            if(check_pos < 0)
                continue;

            // Recompute q_left from checkpoint to left_end
            const int steps_before = left_start - check_pos;
            const int storage_count = left_end - left_start + 1;

            auto it_checkpoint = propagator_at_check_point.find(std::make_tuple(key_left, check_pos));
            if(it_checkpoint == propagator_at_check_point.end())
                continue;

            // Load checkpoint to device and compute q_left positions
            gpu_error_check(cudaMemcpy(d_q_one[0][0], it_checkpoint->second, sizeof(T)*M, cudaMemcpyHostToDevice));
            int left_ping = 0;
            int left_pong = 1;

            // Advance to left_start
            for(int step = 0; step < steps_before; step++)
            {
                int actual_pos = check_pos + step + 1;
                auto it = propagator_at_check_point.find(std::make_tuple(key_left, actual_pos));
                if(it != propagator_at_check_point.end())
                {
                    gpu_error_check(cudaMemcpy(d_q_one[0][left_pong], it->second, sizeof(T)*M, cudaMemcpyHostToDevice));
                }
                else
                {
                    propagator_solver->advance_propagator(0, d_q_one[0][left_ping], d_q_one[0][left_pong], monomer_type, d_q_mask);
                    cudaDeviceSynchronize();
                }
                std::swap(left_ping, left_pong);
            }

            // Copy to q_recal[0] and compute remaining positions
            gpu_error_check(cudaMemcpy(q_recal[0], d_q_one[0][left_ping], sizeof(T)*M, cudaMemcpyDeviceToHost));

            for(int idx = 1; idx < storage_count; idx++)
            {
                int actual_pos = left_start + idx;
                auto it = propagator_at_check_point.find(std::make_tuple(key_left, actual_pos));
                if(it != propagator_at_check_point.end())
                {
                    for(int i=0; i<M; i++)
                        q_recal[idx][i] = it->second[i];
                    gpu_error_check(cudaMemcpy(d_q_one[0][left_pong], it->second, sizeof(T)*M, cudaMemcpyHostToDevice));
                }
                else
                {
                    propagator_solver->advance_propagator(0, d_q_one[0][left_ping], d_q_one[0][left_pong], monomer_type, d_q_mask);
                    cudaDeviceSynchronize();
                    gpu_error_check(cudaMemcpy(q_recal[idx], d_q_one[0][left_pong], sizeof(T)*M, cudaMemcpyDeviceToHost));
                }
                std::swap(left_ping, left_pong);
            }

            // Process each n in [n_start, n_end]
            for(int n = n_start; n <= n_end; n++)
            {
                // Advance q_right if needed
                auto it_right = propagator_at_check_point.find(std::make_tuple(key_right, n));
                if(it_right != propagator_at_check_point.end())
                {
                    gpu_error_check(cudaMemcpy(d_q_right[right_prev_idx], it_right->second, sizeof(T)*M, cudaMemcpyHostToDevice));
                    current_n_right = n;
                }
                else
                {
                    while(current_n_right < n)
                    {
                        propagator_solver->advance_propagator(0, d_q_right[right_prev_idx], d_q_right[right_next_idx], monomer_type, d_q_mask);
                        cudaDeviceSynchronize();
                        std::swap(right_prev_idx, right_next_idx);
                        current_n_right++;
                    }
                }

                // Get q_left[N_LEFT - n] from q_recal
                int left_pos = N_LEFT - n;
                int storage_idx = left_pos - left_start;

                // Copy q_left to device for inner product
                gpu_error_check(cudaMemcpy(d_q_left, q_recal[storage_idx], sizeof(T)*M, cudaMemcpyHostToDevice));

                // Compute partition at this position
                T total_partition = dynamic_cast<CudaComputationBox<T>*>(this->cb)->inner_product_device(
                    d_q_left, d_q_right[right_prev_idx]);

                total_partition *= n_repeated/this->cb->get_volume()/n_propagators;
                total_partitions[p].push_back(total_partition);

                #ifndef NDEBUG
                std::cout<< p << ", " << n << ": " << total_partition << std::endl;
                #endif
            }
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
        if (diff_partition > PARTITION_TOLERANCE)
            return false;
    }
    return true;
}
// Explicit template instantiation
template class CudaComputationReduceMemoryContinuous<double>;
template class CudaComputationReduceMemoryContinuous<std::complex<double>>;