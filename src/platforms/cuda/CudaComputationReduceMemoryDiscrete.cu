/**
 * @file CudaComputationReduceMemoryDiscrete.cu
 * @brief Memory-efficient CUDA propagator computation for discrete chains.
 *
 * Implements checkpointing strategy for discrete chain model by storing
 * propagators only at checkpoint positions in pinned host memory and
 * recomputing intermediate values on-the-fly.
 *
 * **Checkpointing Strategy:**
 *
 * - propagator_at_check_point[(key, n)]: Full segment propagators at checkpoints
 * - propagator_half_steps_at_check_point[(key, n)]: Junction propagators
 * - Intermediate values are recomputed from nearest checkpoint when needed
 *
 * **Memory Savings:**
 *
 * Standard version stores O(N) propagators per chain.
 * Checkpoint version stores O(√N) checkpoints, recomputes O(√N) segments.
 * Net effect: ~10x memory reduction for long chains.
 *
 * **Computation Flow:**
 *
 * 1. Compute all propagators, storing only at checkpoints
 * 2. For concentration: recalculate from checkpoints as needed
 * 3. For stress: recalculate both forward and backward propagators
 *
 * @see CpuComputationReduceMemoryDiscrete for CPU version
 * @see CudaComputationDiscrete for full GPU memory version
 */

#include <complex>
#include <cmath>
#include <cstring>
#include <iostream>
#include <chrono>
#include <omp.h>

#include "CudaComputationBox.h"
#include "CudaComputationReduceMemoryDiscrete.h"
#include "CudaSolverPseudoDiscrete.h"

template <typename T>
CudaComputationReduceMemoryDiscrete<T>::CudaComputationReduceMemoryDiscrete(
    ComputationBox<T>* cb,
    Molecules *molecules,
    PropagatorComputationOptimizer *propagator_computation_optimizer)
    : PropagatorComputation<T>(cb, molecules, propagator_computation_optimizer)
{
    try
    {
        #ifndef NDEBUG
        std::cout << "--------- Discrete Chain Solver, GPU Memory Saving Version (Checkpointing) ---------" << std::endl;
        #endif

        const int M = this->cb->get_total_grid();

        // Use single stream to minimize memory usage
        n_streams = 1;
        #ifndef NDEBUG
        std::cout << "The number of CPU threads: " << n_streams << std::endl;
        #endif

        // Create streams
        for(int i=0; i<n_streams; i++)
        {
            gpu_error_check(cudaStreamCreate(&streams[i][0])); // for kernel execution
            gpu_error_check(cudaStreamCreate(&streams[i][1])); // for memcpy
        }

        this->propagator_solver = new CudaSolverPseudoDiscrete<T>(cb, molecules, n_streams, streams, true);

        // Find the total sum of segments for workspace allocation
        total_max_n_segment = 0;

        // Allocate memory for propagators at checkpoint positions only
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
            int max_n_segment = item.second.max_n_segment;

            // Accumulate total segment count across all propagators
            total_max_n_segment += max_n_segment;

            // Allocate checkpoint propagators in pinned memory
            // Store at segment 1 (first valid segment for discrete chains)
            gpu_error_check(cudaMallocHost((void**)&propagator_at_check_point[std::make_tuple(key, 1)], sizeof(T)*M));

            // Store at final segment (max_n_segment) - for partition function and concentration
            gpu_error_check(cudaMallocHost((void**)&propagator_at_check_point[std::make_tuple(key, max_n_segment)], sizeof(T)*M));

            // Also store at dependency points (junction ends) if they exist
            for (int n : item.second.junction_ends)
            {
                if (n != 1 && n != max_n_segment)  // Avoid duplicates
                {
                    if (propagator_at_check_point.find(std::make_tuple(key, n)) == propagator_at_check_point.end())
                        gpu_error_check(cudaMallocHost((void**)&propagator_at_check_point[std::make_tuple(key, n)], sizeof(T)*M));
                }
            }

            // Allocate half-bond propagators at junction points
            // Store at index 0 if there are dependencies (junction start)
            if (item.second.deps.size() > 0)
            {
                gpu_error_check(cudaMallocHost((void**)&propagator_half_steps_at_check_point[std::make_tuple(key, 0)], sizeof(T)*M));
            }

            // Store half-bond at junction ends
            for (int n : item.second.junction_ends)
            {
                gpu_error_check(cudaMallocHost((void**)&propagator_half_steps_at_check_point[std::make_tuple(key, n)], sizeof(T)*M));
            }

            #ifndef NDEBUG
            propagator_finished[key] = new bool[max_n_segment+1];
            for(int i=0; i<=max_n_segment; i++)
                propagator_finished[key][i] = false;
            for (int n: item.second.junction_ends)
                propagator_half_steps_finished[key][n] = false;
            if (item.second.deps.size() > 0)
                propagator_half_steps_finished[key][0] = false;
            #endif
        }

        // Calculate checkpoint interval as 2*sqrt(N) for better memory-computation tradeoff
        checkpoint_interval = static_cast<int>(std::ceil(2.0*std::sqrt(static_cast<double>(total_max_n_segment))));
        if (checkpoint_interval < 1)
            checkpoint_interval = 1;

        // Allocate checkpoints at sqrt(N) intervals for each propagator
        for(const auto& item: this->propagator_computation_optimizer->get_computation_propagators())
        {
            std::string key = item.first;
            int max_n_segment = item.second.max_n_segment;

            // Allocate checkpoints at every checkpoint_interval
            for(int n = checkpoint_interval; n < max_n_segment; n += checkpoint_interval)
            {
                if(propagator_at_check_point.find(std::make_tuple(key, n)) == propagator_at_check_point.end())
                {
                    gpu_error_check(cudaMallocHost((void**)&propagator_at_check_point[std::make_tuple(key, n)], sizeof(T)*M));
                }
            }
        }

        // Allocate workspace for recalculation (only need k since we skip intermediate values)
        const int workspace_size = checkpoint_interval;
        for(int i=0; i<workspace_size; i++)
            gpu_error_check(cudaMallocHost((void**)&q_recal.emplace_back(), sizeof(T)*M));

        // Allocate ping-pong buffers
        gpu_error_check(cudaMallocHost((void**)&q_pair[0], sizeof(T)*M));
        gpu_error_check(cudaMallocHost((void**)&q_pair[1], sizeof(T)*M));

        // Allocate memory for concentrations
        if( this->propagator_computation_optimizer->get_computation_blocks().size() == 0)
            throw_with_line_number("There is no block. Add polymers first.");
        for(const auto& item: this->propagator_computation_optimizer->get_computation_blocks())
        {
            phi_block[item.first] = nullptr;
            gpu_error_check(cudaMallocHost((void**)&phi_block[item.first], sizeof(T)*M));
            std::memset(phi_block[item.first], 0, sizeof(T)*M);  // Zero-initialize
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
                propagator_at_check_point[std::make_tuple(key_left, n_segment_left)],    // q
                propagator_at_check_point[std::make_tuple(key_right, 1)],                // q_dagger
                monomer_type,
                n_aggregated                                                              // how many propagators are aggregated
                ));
            current_p++;
        }

        // Concentrations for each solvent
        for(int s=0;s<this->molecules->get_n_solvent_types();s++)
            phi_solvent.push_back(new T[M]);

        // Create scheduler for computation of propagator
        sc = new Scheduler(this->propagator_computation_optimizer->get_computation_propagators(), n_streams);

        // Allocate GPU memory for unity array
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
        // d_workspace[0]: primary buffer, d_workspace[1]: second buffer
        // Phase 1 (compute_propagators): uses d_propagator_sub_dep aliases
        // Phase 2 (compute_concentrations): uses q_left, q_right, d_phi (see memory layout in header)
        // Phase 3 (compute_stress): uses d_workspace directly (contiguous 2×M)
        gpu_error_check(cudaMalloc((void**)&d_workspace[0], sizeof(T)*2*M));
        gpu_error_check(cudaMalloc((void**)&d_workspace[1], sizeof(T)*2*M));

        // Set up aliases
        d_propagator_sub_dep[0][0] = d_workspace[0];        // first M of d_workspace[0]
        d_propagator_sub_dep[0][1] = d_workspace[0] + M;    // second M of d_workspace[0]
        d_phi                      = d_workspace[1];        // first M of d_workspace[1]

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
CudaComputationReduceMemoryDiscrete<T>::~CudaComputationReduceMemoryDiscrete()
{
    delete propagator_solver;
    delete sc;

    // Free checkpoint propagators
    for(const auto& item: propagator_at_check_point)
        cudaFreeHost(item.second);
    for(const auto& item: propagator_half_steps_at_check_point)
        cudaFreeHost(item.second);

    // Free recalculation workspace (must match constructor allocation)
    const int workspace_size = checkpoint_interval;
    for(int n=0; n<workspace_size; n++)
        cudaFreeHost(q_recal[n]);
    cudaFreeHost(q_pair[0]);
    cudaFreeHost(q_pair[1]);

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
    cudaFree(d_workspace[0]);  // contains d_propagator_sub_dep[0][*], used for q_left, q_right in concentration
    cudaFree(d_workspace[1]);  // contains d_phi

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
void CudaComputationReduceMemoryDiscrete<T>::update_laplacian_operator()
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
void CudaComputationReduceMemoryDiscrete<T>::compute_statistics(
    std::map<std::string, const T*> w_input,
    std::map<std::string, const T*> q_init)
{
    this->compute_propagators(w_input, q_init);
    this->compute_concentrations();
}

template <typename T>
void CudaComputationReduceMemoryDiscrete<T>::compute_propagators(
    std::map<std::string, const T*> w_input,
    std::map<std::string, const T*> q_init)
{
    try{
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();
        const int M = this->cb->get_total_grid();

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

        // Reset debug flags
        #ifndef NDEBUG
        for(const auto& item: this->propagator_computation_optimizer->get_computation_propagators())
        {
            std::string key = item.first;
            int max_n_segment = item.second.max_n_segment;
            for(int i=0; i<=max_n_segment; i++)
                propagator_finished[key][i] = false;
            for (int n: item.second.junction_ends)
                propagator_half_steps_finished[key][n] = false;
            if (item.second.deps.size() > 0)
                propagator_half_steps_finished[key][0] = false;
        }
        #endif

        auto& branch_schedule = sc->get_schedule();
        for (auto parallel_job = branch_schedule.begin(); parallel_job != branch_schedule.end(); parallel_job++)
        {
            // For each propagator (single-threaded since n_streams=1)
            for(size_t job=0; job<parallel_job->size(); job++)
            {
                const int STREAM = 0;

                auto& key = std::get<0>((*parallel_job)[job]);
                int n_segment_from = std::get<1>((*parallel_job)[job]);
                int n_segment_to = std::get<2>((*parallel_job)[job]);
                auto& deps = this->propagator_computation_optimizer->get_computation_propagator(key).deps;
                auto& junction_ends = this->propagator_computation_optimizer->get_computation_propagator(key).junction_ends;
                auto monomer_type = this->propagator_computation_optimizer->get_computation_propagator(key).monomer_type;

                CuDeviceData<T> *_d_exp_dw = propagator_solver->d_exp_dw[monomer_type];

                // Calculate one block end
                if(n_segment_from == 0 && deps.size() == 0) // if it is leaf node
                {
                    // q_init
                    if (key[0] == '{')
                    {
                        std::string g = PropagatorCode::get_q_input_idx_from_key(key);
                        if (q_init.find(g) == q_init.end())
                            throw_with_line_number("Could not find q_init[\"" + g + "\"].");
                        gpu_error_check(cudaMemcpy(d_q_one[STREAM][0], q_init[g], sizeof(T)*M, cudaMemcpyInputToDevice));
                        ker_multi<<<N_BLOCKS, N_THREADS>>>(d_q_one[STREAM][0], d_q_one[STREAM][0], _d_exp_dw, 1.0, M);
                    }
                    else
                    {
                        gpu_error_check(cudaMemcpy(d_q_one[STREAM][0], _d_exp_dw, sizeof(T)*M, cudaMemcpyDeviceToDevice));
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
                        // Initialize to zero
                        gpu_error_check(cudaMemset(d_q_one[STREAM][0], 0, sizeof(T)*M));

                        for(size_t d=0; d<deps.size(); d++)
                        {
                            std::string sub_dep = std::get<0>(deps[d]);
                            int sub_n_segment   = std::get<1>(deps[d]);
                            int sub_n_repeated  = std::get<2>(deps[d]);

                            T* source_ptr;
                            if (sub_n_segment == 0)
                            {
                                source_ptr = propagator_half_steps_at_check_point[std::make_tuple(sub_dep, 0)];
                            }
                            else
                            {
                                source_ptr = propagator_at_check_point[std::make_tuple(sub_dep, sub_n_segment)];
                            }

                            gpu_error_check(cudaMemcpy(d_propagator_sub_dep[STREAM][0], source_ptr, sizeof(T)*M, cudaMemcpyHostToDevice));
                            ker_lin_comb<<<N_BLOCKS, N_THREADS>>>(
                                    d_q_one[STREAM][0], 1.0, d_q_one[STREAM][0],
                                    sub_n_repeated, d_propagator_sub_dep[STREAM][0], M);
                        }

                        // If n_segments of all deps are 0 (junction point)
                        if (std::get<1>(deps[0]) == 0)
                        {
                            gpu_error_check(cudaMemcpy(propagator_half_steps_at_check_point[std::make_tuple(key, 0)], d_q_one[STREAM][0], sizeof(T)*M, cudaMemcpyDeviceToHost));

                            // Add half bond
                            propagator_solver->advance_propagator_half_bond_step(
                                STREAM,
                                d_q_one[STREAM][0], d_q_one[STREAM][0], monomer_type);

                            // Add full segment
                            ker_multi<<<N_BLOCKS, N_THREADS>>>(d_q_one[STREAM][0], d_q_one[STREAM][0], _d_exp_dw, 1.0, M);

                            #ifndef NDEBUG
                            propagator_half_steps_finished[key][0] = true;
                            #endif
                        }
                        else
                        {
                            propagator_solver->advance_propagator(
                                STREAM,
                                d_q_one[STREAM][0], d_q_one[STREAM][0],
                                monomer_type, d_q_mask);
                        }

                        #ifndef NDEBUG
                        propagator_finished[key][1] = true;
                        #endif
                    }
                    else if(key[0] == '(')
                    {
                        // Combine branches - Initialize to one
                        gpu_error_check(cudaMemcpy(d_q_one[STREAM][0], d_q_unity, sizeof(T)*M, cudaMemcpyDeviceToDevice));

                        for(size_t d=0; d<deps.size(); d++)
                        {
                            std::string sub_dep   = std::get<0>(deps[d]);
                            int sub_n_segment     = std::get<1>(deps[d]);
                            int sub_n_repeated    = std::get<2>(deps[d]);

                            T* source_ptr = propagator_half_steps_at_check_point[std::make_tuple(sub_dep, sub_n_segment)];
                            gpu_error_check(cudaMemcpy(d_propagator_sub_dep[STREAM][0], source_ptr, sizeof(T)*M, cudaMemcpyHostToDevice));

                            for(int r=0; r<sub_n_repeated; r++)
                            {
                                ker_multi<<<N_BLOCKS, N_THREADS>>>(
                                    d_q_one[STREAM][0], d_q_one[STREAM][0], d_propagator_sub_dep[STREAM][0], 1.0, M);
                            }
                        }
                        gpu_error_check(cudaMemcpy(propagator_half_steps_at_check_point[std::make_tuple(key, 0)], d_q_one[STREAM][0], sizeof(T)*M, cudaMemcpyDeviceToHost));

                        #ifndef NDEBUG
                        propagator_half_steps_finished[key][0] = true;
                        #endif

                        if (n_segment_to > 0)
                        {
                            // Add half bond
                            propagator_solver->advance_propagator_half_bond_step(
                                STREAM,
                                d_q_one[STREAM][0], d_q_one[STREAM][0], monomer_type);

                            // Add full segment
                            ker_multi<<<N_BLOCKS, N_THREADS>>>(d_q_one[STREAM][0], d_q_one[STREAM][0], _d_exp_dw, 1.0, M);

                            #ifndef NDEBUG
                            propagator_finished[key][1] = true;
                            #endif
                        }
                    }
                }

                if (n_segment_to == 0)
                {
                    gpu_error_check(cudaDeviceSynchronize());
                    continue;
                }

                if (n_segment_from == 0)
                {
                    // Multiply mask
                    if (d_q_mask != nullptr)
                        ker_multi<<<N_BLOCKS, N_THREADS>>>(d_q_one[STREAM][0], d_q_one[STREAM][0], d_q_mask, 1.0, M);

                    // Store at checkpoint (segment 1)
                    gpu_error_check(cudaMemcpy(propagator_at_check_point[std::make_tuple(key, 1)], d_q_one[STREAM][0], sizeof(T)*M, cudaMemcpyDeviceToHost));

                    // q(r, 1+1/2) if at junction end
                    if (junction_ends.find(1) != junction_ends.end())
                    {
                        propagator_solver->advance_propagator_half_bond_step(
                            STREAM,
                            d_q_one[STREAM][0],
                            d_q_one[STREAM][1],
                            monomer_type);

                        gpu_error_check(cudaMemcpy(
                            propagator_half_steps_at_check_point[std::make_tuple(key, 1)],
                            d_q_one[STREAM][1],
                            sizeof(T)*M, cudaMemcpyDeviceToHost));

                        #ifndef NDEBUG
                        propagator_half_steps_finished[key][1] = true;
                        #endif
                    }
                    n_segment_from++;
                }
                else
                {
                    // Load from checkpoint
                    gpu_error_check(cudaMemcpy(d_q_one[STREAM][0], propagator_at_check_point[std::make_tuple(key, n_segment_from)], sizeof(T)*M, cudaMemcpyHostToDevice));
                }

                int prev = 0;
                int next = 1;

                // q(r,s)
                for(int n=n_segment_from; n<n_segment_to; n++)
                {
                    // Calculate propagators
                    propagator_solver->advance_propagator(
                        STREAM,
                        d_q_one[STREAM][prev],
                        d_q_one[STREAM][next],
                        monomer_type, d_q_mask);

                    std::swap(prev, next);

                    #ifndef NDEBUG
                    propagator_finished[key][n+1] = true;
                    #endif

                    // Store at checkpoint if this is a checkpoint position
                    auto checkpoint_key = std::make_tuple(key, n+1);
                    if (propagator_at_check_point.find(checkpoint_key) != propagator_at_check_point.end())
                    {
                        gpu_error_check(cudaMemcpy(propagator_at_check_point[checkpoint_key], d_q_one[STREAM][prev], sizeof(T)*M, cudaMemcpyDeviceToHost));
                    }

                    // q(r, s+1/2) if at junction end
                    if (junction_ends.find(n+1) != junction_ends.end())
                    {
                        propagator_solver->advance_propagator_half_bond_step(
                            STREAM,
                            d_q_one[STREAM][prev],
                            d_q_one[STREAM][next],
                            monomer_type);

                        gpu_error_check(cudaMemcpy(
                            propagator_half_steps_at_check_point[std::make_tuple(key, n+1)],
                            d_q_one[STREAM][next],
                            sizeof(T)*M, cudaMemcpyDeviceToHost));

                        #ifndef NDEBUG
                        propagator_half_steps_finished[key][n+1] = true;
                        #endif
                    }
                }
                gpu_error_check(cudaDeviceSynchronize());
            }
        }

        // Compute total partition function of each distinct polymers
        // Set up local workspace pointers (see memory layout in header)
        CuDeviceData<T> *d_q_left = d_workspace[0];          // first M of d_workspace[0]
        CuDeviceData<T> *d_q_right = d_workspace[0] + M;     // second M of d_workspace[0]

        for(const auto& segment_info: single_partition_segment)
        {
            int p                    = std::get<0>(segment_info);
            T *propagator_left  = std::get<1>(segment_info);
            T *propagator_right = std::get<2>(segment_info);
            std::string monomer_type = std::get<3>(segment_info);
            int n_aggregated         = std::get<4>(segment_info);
            CuDeviceData<T> *_d_exp_dw = propagator_solver->d_exp_dw[monomer_type];

            // Copy propagators from host to device
            gpu_error_check(cudaMemcpy(d_q_left, propagator_left,  sizeof(T)*M, cudaMemcpyHostToDevice));
            gpu_error_check(cudaMemcpy(d_q_right, propagator_right, sizeof(T)*M, cudaMemcpyHostToDevice));

            this->single_polymer_partitions[p] = dynamic_cast<CudaComputationBox<T>*>(this->cb)->inner_product_inverse_weight_device(
                d_q_left,   // q
                d_q_right,  // q^dagger
                _d_exp_dw)/(n_aggregated*this->cb->get_volume());
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

template <typename T>
std::vector<T*> CudaComputationReduceMemoryDiscrete<T>::recalculate_propagator(
    std::string key, const int N_START, const int N_RANGE, std::string monomer_type)
{
    // Recalculate propagators from segment N_START+1 to N_START+N_RANGE
    // Returns q_out where q_out[n] points to propagator at segment N_START+n
    // If propagator exists in checkpoint, reuse it; otherwise compute it

    const int M = this->cb->get_total_grid();
    const int STREAM = 0;

    // An array of pointers for q_out
    std::vector<T*> q_out(total_max_n_segment + 1);

    // Compute the q_out
    for(int n=1; n<=N_RANGE; n++)
    {
        int segment_idx = N_START + n;

        // Use propagator_at_check_point if exists
        auto it = propagator_at_check_point.find(std::make_tuple(key, segment_idx));
        if(it != propagator_at_check_point.end())
        {
            q_out[n] = it->second;
        }
        // Assign q_recal memory space, and compute the next propagator
        else if (n > 1)
        {
            q_out[n] = this->q_recal[n];

            // Copy previous propagator to device
            gpu_error_check(cudaMemcpy(d_q_one[STREAM][0], q_out[n-1], sizeof(T)*M, cudaMemcpyHostToDevice));

            // Advance propagator
            propagator_solver->advance_propagator(
                STREAM,
                d_q_one[STREAM][0],
                d_q_one[STREAM][1],
                monomer_type, d_q_mask);

            // Copy result back to host
            gpu_error_check(cudaMemcpy(q_out[n], d_q_one[STREAM][1], sizeof(T)*M, cudaMemcpyDeviceToHost));
        }
    }

    return q_out;
}

template <typename T>
void CudaComputationReduceMemoryDiscrete<T>::advance_propagator_single_segment(
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
void CudaComputationReduceMemoryDiscrete<T>::compute_concentrations()
{
    try
    {
        const int M = this->cb->get_total_grid();
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

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

            // If there is no segment
            if(n_segment_right == 0)
            {
                for(int i=0; i<M; i++)
                    block.second[i] = 0.0;
                continue;
            }

            // Calculate phi of one block (possibly multiple blocks when using aggregation)
            calculate_phi_one_block(
                block.second,               // phi
                key_left,                   // left propagator key
                key_right,                  // right propagator key
                n_segment_left,
                n_segment_right,
                monomer_type);
        }

        // Calculate partition functions and concentrations of solvents
        for(int s=0; s<this->molecules->get_n_solvent_types(); s++)
        {
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

            ker_linear_scaling<<<N_BLOCKS, N_THREADS>>>(d_phi, _d_exp_dw, norm, 0.0, M);
            gpu_error_check(cudaMemcpy(phi_solvent[s], d_phi, sizeof(T)*M, cudaMemcpyDeviceToHost));
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

template <typename T>
void CudaComputationReduceMemoryDiscrete<T>::calculate_phi_one_block(
    T *phi, std::string key_left, std::string key_right,
    const int N_LEFT, const int N_RIGHT, std::string monomer_type)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();
        const int M = this->cb->get_total_grid();
        const int STREAM = 0;
        const int k = checkpoint_interval;

        // Set up local workspace pointers (see memory layout in header)
        CuDeviceData<T> *d_q_left = d_workspace[0];          // first M of d_workspace[0]
        CuDeviceData<T> *d_q_right = d_workspace[0] + M;     // second M of d_workspace[0]

        // Get block info for normalization
        auto block_key = std::make_tuple(0, key_left, key_right);
        for(const auto& item: phi_block)
        {
            if(std::get<1>(item.first) == key_left && std::get<2>(item.first) == key_right)
            {
                block_key = item.first;
                break;
            }
        }

        int p = std::get<0>(block_key);
        int n_repeated = this->propagator_computation_optimizer->get_computation_block(block_key).n_repeated;
        CuDeviceData<T> *_d_exp_dw = propagator_solver->d_exp_dw[monomer_type];

        // Get monomer types for recalculation
        std::string monomer_type_left = this->propagator_computation_optimizer->get_computation_propagator(key_left).monomer_type;
        std::string monomer_type_right = this->propagator_computation_optimizer->get_computation_propagator(key_right).monomer_type;

        // Normalize concentration
        Polymer& pc = this->molecules->get_polymer(p);
        T norm = (this->molecules->get_ds()*pc.get_volume_fraction()/pc.get_alpha()*n_repeated)/this->single_polymer_partitions[p];

        CuDeviceData<T> cuda_norm;
        if constexpr (std::is_same<T, double>::value)
            cuda_norm = norm;
        else
            cuda_norm = stdToCuDoubleComplex(norm);

        // Initialize phi to zero
        gpu_error_check(cudaMemset(d_phi, 0, sizeof(T)*M));

        // q_right ping-pong
        T *q_right_prev = nullptr;
        T *q_right_next = nullptr;
        int prev_right = 0;
        int next_right = 1;

        // Process n = 1 to N_RIGHT using block-based computation
        int num_blocks = (N_RIGHT + k - 1) / k;

        for (int blk = 0; blk < num_blocks; blk++)
        {
            int n_start = blk * k + 1;  // First n in this block
            int n_end = std::min((blk + 1) * k, N_RIGHT);  // Last n in this block

            // q_left positions needed: segment (N_LEFT - n + 1) for n in [n_start, n_end]
            // = segments [N_LEFT - n_end + 1, N_LEFT - n_start + 1]
            int left_start = N_LEFT - n_end + 1;
            int left_end = N_LEFT - n_start + 1;

            // Handle discrete chain case: left_start must be >= 1
            int left_compute_start = std::max(left_start, 1);

            // Find checkpoint at or before left_compute_start
            int check_pos = (left_compute_start / k) * k;
            if (check_pos < 1) check_pos = 1;

            // Find the actual checkpoint
            auto it = propagator_at_check_point.find(std::make_tuple(key_left, check_pos));
            while (it == propagator_at_check_point.end() && check_pos > 1)
            {
                check_pos -= k;
                if (check_pos < 1) check_pos = 1;
                it = propagator_at_check_point.find(std::make_tuple(key_left, check_pos));
            }

            // Skip from checkpoint to left_compute_start (don't store intermediate values)
            gpu_error_check(cudaMemcpy(d_q_one[STREAM][0], it->second, sizeof(T)*M, cudaMemcpyHostToDevice));
            for (int i = check_pos; i < left_compute_start; i++)
            {
                propagator_solver->advance_propagator(STREAM, d_q_one[STREAM][0], d_q_one[STREAM][1], monomer_type_left, d_q_mask);
                gpu_error_check(cudaMemcpy(d_q_one[STREAM][0], d_q_one[STREAM][1], sizeof(T)*M, cudaMemcpyDeviceToDevice));
            }

            // Store q_left from left_compute_start to left_end
            gpu_error_check(cudaMemcpy(q_recal[0], d_q_one[STREAM][0], sizeof(T)*M, cudaMemcpyDeviceToHost));
            for (int i = 1; i <= left_end - left_compute_start; i++)
            {
                propagator_solver->advance_propagator(STREAM, d_q_one[STREAM][0], d_q_one[STREAM][1], monomer_type_left, d_q_mask);
                gpu_error_check(cudaMemcpy(q_recal[i], d_q_one[STREAM][1], sizeof(T)*M, cudaMemcpyDeviceToHost));
                gpu_error_check(cudaMemcpy(d_q_one[STREAM][0], d_q_one[STREAM][1], sizeof(T)*M, cudaMemcpyDeviceToDevice));
            }
            // Now q_recal[i] = q_left[left_compute_start + i]

            // Process each n in [n_start, n_end]
            for (int n = n_start; n <= n_end; n++)
            {
                // Compute q_right[n] incrementally
                auto it_right = propagator_at_check_point.find(std::make_tuple(key_right, n));
                if (it_right != propagator_at_check_point.end())
                {
                    q_right_next = it_right->second;
                }
                else if (n > 1)
                {
                    q_right_next = this->q_pair[prev_right];
                    gpu_error_check(cudaMemcpy(d_q_one[STREAM][0], q_right_prev, sizeof(T)*M, cudaMemcpyHostToDevice));
                    propagator_solver->advance_propagator(STREAM, d_q_one[STREAM][0], d_q_one[STREAM][1], monomer_type_right, d_q_mask);
                    gpu_error_check(cudaMemcpy(q_right_next, d_q_one[STREAM][1], sizeof(T)*M, cudaMemcpyDeviceToHost));
                    std::swap(prev_right, next_right);
                }

                // Get q_left position (segment N_LEFT - n + 1)
                int left_pos = N_LEFT - n + 1;
                int ws_idx = left_pos - left_compute_start;

                // Add contribution: q_left * q_right
                gpu_error_check(cudaMemcpy(d_q_left, q_recal[ws_idx], sizeof(T)*M, cudaMemcpyHostToDevice));
                gpu_error_check(cudaMemcpy(d_q_right, q_right_next, sizeof(T)*M, cudaMemcpyHostToDevice));

                ker_add_multi<<<N_BLOCKS, N_THREADS>>>(d_phi, d_q_left, d_q_right, cuda_norm, M);

                q_right_prev = q_right_next;
                q_right_next = nullptr;
            }
        }

        // Divide by exp_dw
        ker_divide<<<N_BLOCKS, N_THREADS>>>(d_phi, d_phi, _d_exp_dw, 1.0, M);

        // Copy result to host
        gpu_error_check(cudaMemcpy(phi, d_phi, sizeof(T)*M, cudaMemcpyDeviceToHost));
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

template <typename T>
T CudaComputationReduceMemoryDiscrete<T>::get_total_partition(int polymer)
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
void CudaComputationReduceMemoryDiscrete<T>::get_total_concentration(std::string monomer_type, T *phi)
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
void CudaComputationReduceMemoryDiscrete<T>::get_total_concentration(int p, std::string monomer_type, T *phi)
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
void CudaComputationReduceMemoryDiscrete<T>::get_total_concentration_gce(double fugacity, int p, std::string monomer_type, T *phi)
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
void CudaComputationReduceMemoryDiscrete<T>::get_block_concentration(int p, T *phi)
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
T CudaComputationReduceMemoryDiscrete<T>::get_solvent_partition(int s)
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
void CudaComputationReduceMemoryDiscrete<T>::get_solvent_concentration(int s, T *phi)
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
void CudaComputationReduceMemoryDiscrete<T>::compute_stress()
{
    // This method should be invoked after invoking compute_statistics().
    // Uses block-based computation for memory efficiency.
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();
        const int DIM = this->cb->get_dim();
        const int M   = this->cb->get_total_grid();
        const int STREAM = 0;
        const int k = checkpoint_interval;

        // Reset stress
        // N_STRESS: 3D->6, 2D->3, 1D->1
        const int N_STRESS = (DIM == 3) ? 6 : ((DIM == 2) ? 3 : 1);
        int n_polymer_types = this->molecules->get_n_polymer_types();
        for(int p=0; p<n_polymer_types; p++)
            for(int d=0; d<N_STRESS; d++)
                this->dq_dl[p][d] = 0.0;

        CuDeviceData<T> *d_segment_stress;
        T segment_stress[N_STRESS];
        gpu_error_check(cudaMalloc((void**)&d_segment_stress, sizeof(T)*N_STRESS));

        // Compute stress for each block
        for(const auto& block: phi_block)
        {
            const auto& key   = block.first;

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

            // Get monomer types for recalculation
            std::string monomer_type_left = this->propagator_computation_optimizer->get_computation_propagator(key_left).monomer_type;
            std::string monomer_type_right = this->propagator_computation_optimizer->get_computation_propagator(key_right).monomer_type;

            std::array<T,6> _block_dq_dl = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

            T *q_segment_1;
            T *q_segment_2;
            bool is_half_bond_length;

            // q_right ping-pong
            T *q_right_prev = nullptr;
            T *q_right_next = nullptr;
            int prev_right = 0;
            int next_right = 1;

            // Handle special case: stress at u junction (n=0) for key_right[0] == '('
            if (key_right[0] == '(')
            {
                if (this->propagator_computation_optimizer->get_computation_propagator(key_right).deps.size() > 0)
                {
                    // Need q_left[N_LEFT]
                    int left_check = (N_LEFT / k) * k;
                    if (left_check < 1) left_check = 1;

                    // Find nearest checkpoint
                    auto it = propagator_at_check_point.find(std::make_tuple(key_left, left_check));
                    while (it == propagator_at_check_point.end() && left_check > 1)
                    {
                        left_check -= k;
                        if (left_check < 1) left_check = 1;
                        it = propagator_at_check_point.find(std::make_tuple(key_left, left_check));
                    }

                    // Recalculate from checkpoint to N_LEFT
                    gpu_error_check(cudaMemcpy(d_q_one[STREAM][0], it->second, sizeof(T)*M, cudaMemcpyHostToDevice));
                    for (int i = left_check; i < N_LEFT; i++)
                    {
                        propagator_solver->advance_propagator(STREAM, d_q_one[STREAM][0], d_q_one[STREAM][1], monomer_type_left, d_q_mask);
                        gpu_error_check(cudaMemcpy(d_q_one[STREAM][0], d_q_one[STREAM][1], sizeof(T)*M, cudaMemcpyDeviceToDevice));
                    }
                    gpu_error_check(cudaMemcpy(q_recal[0], d_q_one[STREAM][0], sizeof(T)*M, cudaMemcpyDeviceToHost));

                    q_segment_1 = q_recal[0];
                    q_segment_2 = propagator_half_steps_at_check_point[std::make_tuple(key_right, 0)];
                    is_half_bond_length = true;

                    gpu_error_check(cudaMemcpy(&d_workspace[0][0], q_segment_1, sizeof(T)*M, cudaMemcpyHostToDevice));
                    gpu_error_check(cudaMemcpy(&d_workspace[0][M], q_segment_2, sizeof(T)*M, cudaMemcpyHostToDevice));

                    // Compute stress (d_workspace[0] is contiguous 2×M buffer)
                    propagator_solver->compute_single_segment_stress(
                        STREAM, d_workspace[0], d_segment_stress, monomer_type, is_half_bond_length);

                    gpu_error_check(cudaMemcpy(segment_stress, d_segment_stress, sizeof(T)*N_STRESS, cudaMemcpyDeviceToHost));

                    for(int d=0; d<N_STRESS; d++)
                        _block_dq_dl[d] += segment_stress[d]*static_cast<double>(n_repeated);
                }
            }

            // Process n = 1 to N_RIGHT using block-based computation
            int num_blocks = (N_RIGHT + k - 1) / k;

            for (int blk = 0; blk < num_blocks; blk++)
            {
                int n_start = blk * k + 1;  // First n in this block
                int n_end = std::min((blk + 1) * k, N_RIGHT);  // Last n in this block

                // q_left positions needed: [N_LEFT - n_end, N_LEFT - n_start] = [left_start, left_end]
                int left_start = N_LEFT - n_end;
                int left_end = N_LEFT - n_start;

                // Handle discrete chain case: left_start must be >= 1
                int left_compute_start = std::max(left_start, 1);

                // Find checkpoint at or before left_compute_start
                int check_pos = (left_compute_start / k) * k;
                if (check_pos < 1) check_pos = 1;

                // Find the actual checkpoint
                auto it = propagator_at_check_point.find(std::make_tuple(key_left, check_pos));
                while (it == propagator_at_check_point.end() && check_pos > 1)
                {
                    check_pos -= k;
                    if (check_pos < 1) check_pos = 1;
                    it = propagator_at_check_point.find(std::make_tuple(key_left, check_pos));
                }

                // Skip from checkpoint to left_compute_start (don't store intermediate values)
                gpu_error_check(cudaMemcpy(d_q_one[STREAM][0], it->second, sizeof(T)*M, cudaMemcpyHostToDevice));
                for (int i = check_pos; i < left_compute_start; i++)
                {
                    propagator_solver->advance_propagator(STREAM, d_q_one[STREAM][0], d_q_one[STREAM][1], monomer_type_left, d_q_mask);
                    gpu_error_check(cudaMemcpy(d_q_one[STREAM][0], d_q_one[STREAM][1], sizeof(T)*M, cudaMemcpyDeviceToDevice));
                }

                // Store q_left from left_compute_start to left_end
                gpu_error_check(cudaMemcpy(q_recal[0], d_q_one[STREAM][0], sizeof(T)*M, cudaMemcpyDeviceToHost));
                for (int i = 1; i <= left_end - left_compute_start; i++)
                {
                    propagator_solver->advance_propagator(STREAM, d_q_one[STREAM][0], d_q_one[STREAM][1], monomer_type_left, d_q_mask);
                    gpu_error_check(cudaMemcpy(q_recal[i], d_q_one[STREAM][1], sizeof(T)*M, cudaMemcpyDeviceToHost));
                    gpu_error_check(cudaMemcpy(d_q_one[STREAM][0], d_q_one[STREAM][1], sizeof(T)*M, cudaMemcpyDeviceToDevice));
                }
                // Now q_recal[i] = q_left[left_compute_start + i]

                // Process each n in [n_start, n_end]
                for (int n = n_start; n <= n_end; n++)
                {
                    int left_pos = N_LEFT - n;

                    // Compute q_right[n] incrementally
                    auto it_right = propagator_at_check_point.find(std::make_tuple(key_right, n));
                    if (it_right != propagator_at_check_point.end())
                    {
                        q_right_next = it_right->second;
                    }
                    else if (n > 1)
                    {
                        q_right_next = this->q_pair[prev_right];
                        gpu_error_check(cudaMemcpy(d_q_one[STREAM][0], q_right_prev, sizeof(T)*M, cudaMemcpyHostToDevice));
                        propagator_solver->advance_propagator(STREAM, d_q_one[STREAM][0], d_q_one[STREAM][1], monomer_type_right, d_q_mask);
                        gpu_error_check(cudaMemcpy(q_right_next, d_q_one[STREAM][1], sizeof(T)*M, cudaMemcpyDeviceToHost));
                        std::swap(prev_right, next_right);
                    }

                    // Handle junction at v (n == N_LEFT) - check BEFORE left_pos < 1
                    if (n == N_LEFT)
                    {
                        if (this->propagator_computation_optimizer->get_computation_propagator(key_left).deps.size() == 0)
                        {
                            q_right_prev = q_right_next;
                            q_right_next = nullptr;
                            continue;
                        }
                        q_segment_1 = propagator_half_steps_at_check_point[std::make_tuple(key_left, 0)];
                        q_segment_2 = q_right_next;
                        is_half_bond_length = true;
                    }
                    // Skip if left_pos < 1 for discrete chains (interior bonds only)
                    else if (left_pos < 1)
                    {
                        q_right_prev = q_right_next;
                        q_right_next = nullptr;
                        continue;
                    }
                    else
                    {
                        int ws_idx = left_pos - left_compute_start;
                        q_segment_1 = q_recal[ws_idx];
                        q_segment_2 = q_right_next;
                        is_half_bond_length = false;
                    }

                    // Copy propagator pair to device
                    gpu_error_check(cudaMemcpy(&d_workspace[0][0], q_segment_1, sizeof(T)*M, cudaMemcpyHostToDevice));
                    gpu_error_check(cudaMemcpy(&d_workspace[0][M], q_segment_2, sizeof(T)*M, cudaMemcpyHostToDevice));

                    // Compute stress (d_workspace[0] is contiguous 2×M buffer)
                    propagator_solver->compute_single_segment_stress(
                        STREAM, d_workspace[0], d_segment_stress, monomer_type, is_half_bond_length);

                    gpu_error_check(cudaMemcpy(segment_stress, d_segment_stress, sizeof(T)*N_STRESS, cudaMemcpyDeviceToHost));

                    for(int d=0; d<N_STRESS; d++)
                        _block_dq_dl[d] += segment_stress[d]*static_cast<double>(n_repeated);

                    q_right_prev = q_right_next;
                    q_right_next = nullptr;
                }
            }

            // Accumulate stress for this polymer
            for(int d=0; d<N_STRESS; d++)
                this->dq_dl[p][d] += _block_dq_dl[d];
        }

        // Normalize stress
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

        cudaFree(d_segment_stress);
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

template <typename T>
void CudaComputationReduceMemoryDiscrete<T>::get_chain_propagator(T *q_out, int polymer, int v, int u, int n)
{
    // This method should be invoked after invoking compute_statistics()
    // Get chain propagator for a selected polymer, block and direction.
    try
    {
        const int M = this->cb->get_total_grid();
        const int STREAM = 0;
        const int k = checkpoint_interval;

        Polymer& pc = this->molecules->get_polymer(polymer);
        std::string dep = pc.get_propagator_key(v,u);

        if (this->propagator_computation_optimizer->get_computation_propagators().find(dep) == this->propagator_computation_optimizer->get_computation_propagators().end())
            throw_with_line_number("Could not find the propagator code '" + dep + "'. Disable 'aggregation' option to obtain propagator.");

        const int N_RIGHT = this->propagator_computation_optimizer->get_computation_propagator(dep).max_n_segment;
        if (n < 1 || n > N_RIGHT)
            throw_with_line_number("n (" + std::to_string(n) + ") must be in range [1, " + std::to_string(N_RIGHT) + "]");

        // Check if the requested segment is at a checkpoint
        auto checkpoint_key = std::make_tuple(dep, n);
        auto it = propagator_at_check_point.find(checkpoint_key);
        if (it != propagator_at_check_point.end())
        {
            // Directly copy from checkpoint
            T* _propagator = it->second;
            for(int i=0; i<M; i++)
                q_out[i] = _propagator[i];
        }
        else
        {
            // Find the nearest checkpoint at or before n
            std::string monomer_type = this->propagator_computation_optimizer->get_computation_propagator(dep).monomer_type;

            int check_pos = (n / k) * k;
            if (check_pos < 1) check_pos = 1;

            // Find the actual checkpoint
            auto it_check = propagator_at_check_point.find(std::make_tuple(dep, check_pos));
            while (it_check == propagator_at_check_point.end() && check_pos > 1)
            {
                check_pos -= k;
                if (check_pos < 1) check_pos = 1;
                it_check = propagator_at_check_point.find(std::make_tuple(dep, check_pos));
            }

            // Use ping-pong buffers to advance from checkpoint to n
            T* q_prev = q_pair[0];
            T* q_next = q_pair[1];
            int prev_idx = 0;
            int next_idx = 1;

            // Copy checkpoint to q_prev
            for(int i=0; i<M; i++)
                q_prev[i] = it_check->second[i];

            // Advance from check_pos to n
            for (int step = check_pos; step < n; step++)
            {
                gpu_error_check(cudaMemcpy(d_q_one[STREAM][0], q_prev, sizeof(T)*M, cudaMemcpyHostToDevice));
                propagator_solver->advance_propagator(STREAM, d_q_one[STREAM][0], d_q_one[STREAM][1], monomer_type, d_q_mask);
                gpu_error_check(cudaMemcpy(q_next, d_q_one[STREAM][1], sizeof(T)*M, cudaMemcpyDeviceToHost));

                std::swap(q_prev, q_next);
                std::swap(prev_idx, next_idx);
            }

            // q_prev now contains the propagator at segment n
            for(int i=0; i<M; i++)
                q_out[i] = q_prev[i];
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

template <typename T>
bool CudaComputationReduceMemoryDiscrete<T>::check_total_partition()
{
    // Uses block-based computation for memory efficiency
    const int M = this->cb->get_total_grid();
    const int STREAM = 0;
    const int k = checkpoint_interval;

    // Set up local workspace pointers (see memory layout in header)
    CuDeviceData<T> *d_q_left = d_workspace[0];          // first M of d_workspace[0]
    CuDeviceData<T> *d_q_right = d_workspace[0] + M;     // second M of d_workspace[0]

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

        const int N_RIGHT = this->propagator_computation_optimizer->get_computation_block(key).n_segment_right;
        const int N_LEFT  = this->propagator_computation_optimizer->get_computation_block(key).n_segment_left;
        int n_repeated    = this->propagator_computation_optimizer->get_computation_block(key).n_repeated;
        int n_propagators = this->propagator_computation_optimizer->get_computation_block(key).v_u.size();

        std::string monomer_type = this->propagator_computation_optimizer->get_computation_block(key).monomer_type;
        CuDeviceData<T> *_d_exp_dw = propagator_solver->d_exp_dw[monomer_type];

        // Get monomer types for recalculation
        std::string monomer_type_left = this->propagator_computation_optimizer->get_computation_propagator(key_left).monomer_type;
        std::string monomer_type_right = this->propagator_computation_optimizer->get_computation_propagator(key_right).monomer_type;

        // q_right ping-pong
        T *q_right_prev = nullptr;
        T *q_right_next = nullptr;
        int prev_right = 0;
        int next_right = 1;

        // Process n = 1 to N_RIGHT using block-based computation
        int num_blocks = (N_RIGHT + k - 1) / k;

        for (int blk = 0; blk < num_blocks; blk++)
        {
            int n_start = blk * k + 1;  // First n in this block
            int n_end = std::min((blk + 1) * k, N_RIGHT);  // Last n in this block

            // q_left positions needed: segment (N_LEFT - n + 1) for n in [n_start, n_end]
            // = segments [N_LEFT - n_end + 1, N_LEFT - n_start + 1]
            int left_start = N_LEFT - n_end + 1;
            int left_end = N_LEFT - n_start + 1;

            // Handle discrete chain case: left_start must be >= 1
            int left_compute_start = std::max(left_start, 1);

            // Find checkpoint at or before left_compute_start
            int check_pos = (left_compute_start / k) * k;
            if (check_pos < 1) check_pos = 1;

            // Find the actual checkpoint
            auto it = propagator_at_check_point.find(std::make_tuple(key_left, check_pos));
            while (it == propagator_at_check_point.end() && check_pos > 1)
            {
                check_pos -= k;
                if (check_pos < 1) check_pos = 1;
                it = propagator_at_check_point.find(std::make_tuple(key_left, check_pos));
            }

            // Skip from checkpoint to left_compute_start (don't store intermediate values)
            gpu_error_check(cudaMemcpy(d_q_one[STREAM][0], it->second, sizeof(T)*M, cudaMemcpyHostToDevice));
            for (int i = check_pos; i < left_compute_start; i++)
            {
                propagator_solver->advance_propagator(STREAM, d_q_one[STREAM][0], d_q_one[STREAM][1], monomer_type_left, d_q_mask);
                gpu_error_check(cudaMemcpy(d_q_one[STREAM][0], d_q_one[STREAM][1], sizeof(T)*M, cudaMemcpyDeviceToDevice));
            }

            // Store q_left from left_compute_start to left_end
            gpu_error_check(cudaMemcpy(q_recal[0], d_q_one[STREAM][0], sizeof(T)*M, cudaMemcpyDeviceToHost));
            for (int i = 1; i <= left_end - left_compute_start; i++)
            {
                propagator_solver->advance_propagator(STREAM, d_q_one[STREAM][0], d_q_one[STREAM][1], monomer_type_left, d_q_mask);
                gpu_error_check(cudaMemcpy(q_recal[i], d_q_one[STREAM][1], sizeof(T)*M, cudaMemcpyDeviceToHost));
                gpu_error_check(cudaMemcpy(d_q_one[STREAM][0], d_q_one[STREAM][1], sizeof(T)*M, cudaMemcpyDeviceToDevice));
            }
            // Now q_recal[i] = q_left[left_compute_start + i]

            // Process each n in [n_start, n_end]
            for (int n = n_start; n <= n_end; n++)
            {
                // Compute q_right[n] incrementally
                auto it_right = propagator_at_check_point.find(std::make_tuple(key_right, n));
                if (it_right != propagator_at_check_point.end())
                {
                    q_right_next = it_right->second;
                }
                else if (n > 1)
                {
                    q_right_next = this->q_pair[prev_right];
                    gpu_error_check(cudaMemcpy(d_q_one[STREAM][0], q_right_prev, sizeof(T)*M, cudaMemcpyHostToDevice));
                    propagator_solver->advance_propagator(STREAM, d_q_one[STREAM][0], d_q_one[STREAM][1], monomer_type_right, d_q_mask);
                    gpu_error_check(cudaMemcpy(q_right_next, d_q_one[STREAM][1], sizeof(T)*M, cudaMemcpyDeviceToHost));
                    std::swap(prev_right, next_right);
                }

                // Get q_left position (segment N_LEFT - n + 1)
                int left_pos = N_LEFT - n + 1;
                int ws_idx = left_pos - left_compute_start;

                // Copy propagators from host to device
                gpu_error_check(cudaMemcpy(d_q_left, q_recal[ws_idx], sizeof(T)*M, cudaMemcpyHostToDevice));
                gpu_error_check(cudaMemcpy(d_q_right, q_right_next, sizeof(T)*M, cudaMemcpyHostToDevice));

                T total_partition = dynamic_cast<CudaComputationBox<T>*>(this->cb)->inner_product_inverse_weight_device(
                    d_q_left,   // q
                    d_q_right,  // q^dagger
                    _d_exp_dw);

                total_partition *= n_repeated/this->cb->get_volume()/n_propagators;
                total_partitions[p].push_back(total_partition);

                #ifndef NDEBUG
                std::cout<< p << ", " << n << ": " << total_partition << std::endl;
                #endif

                q_right_prev = q_right_next;
                q_right_next = nullptr;
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
template class CudaComputationReduceMemoryDiscrete<double>;
template class CudaComputationReduceMemoryDiscrete<std::complex<double>>;
