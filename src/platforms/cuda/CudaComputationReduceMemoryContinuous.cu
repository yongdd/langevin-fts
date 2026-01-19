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
 * - this->propagator_at_check_point: Stores propagators at key contour positions
 * - Check points include: s=0, block junctions, block ends
 * - Intermediate propagators recomputed from nearest checkpoint
 *
 * **Memory Layout:**
 *
 * - Uses pinned (page-locked) host memory for checkpoints
 * - Only 2 GPU arrays needed for current computation
 * - this->q_recal: Temporary host arrays for recomputation
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
 * Single stream (this->n_streams=1) to minimize memory:
 * - this->streams[0][0]: Kernel execution
 * - this->streams[0][1]: Host-device memory transfers
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
#include <cstring>
#include <vector>
#include <omp.h>
#include "CudaComputationReduceMemoryContinuous.h"
#include "CudaComputationBox.h"
#include "CudaSolverPseudoRQM4.h"
#include "CudaSolverPseudoRK2.h"
#include "CudaSolverPseudoETDRK4.h"
#include "CudaSolverCNADI.h"
#include "SimpsonRule.h"
#include "PropagatorCode.h"

template <typename T>
CudaComputationReduceMemoryContinuous<T>::CudaComputationReduceMemoryContinuous(
    ComputationBox<T>* cb,
    Molecules *molecules,
    PropagatorComputationOptimizer *propagator_computation_optimizer,
    std::string method,
    std::string numerical_method)
    : CudaComputationReduceMemoryBase<T>(cb, molecules, propagator_computation_optimizer)
{
    try{
        #ifndef NDEBUG
        std::cout << "--------- Continuous Chain Solver, GPU Memoery Saving Version ---------" << std::endl;
        #endif

        const int M = this->cb->get_total_grid();

        // The number of parallel streams is always 1 to reduce the memory usage
        this->n_streams = 1;
        #ifndef NDEBUG
        std::cout << "The number of CPU threads is always set to " << this->n_streams << "." << std::endl;
        #endif

        // Copy streams
        for(int i=0; i<this->n_streams; i++)
        {
            gpu_error_check(cudaStreamCreate(&this->streams[i][0])); // for kernel execution
            gpu_error_check(cudaStreamCreate(&this->streams[i][1])); // for memcpy
        }

        this->method = method;
        if(method == "pseudospectral")
        {
            if (numerical_method == "" || numerical_method == "rqm4")
                this->propagator_solver = new CudaSolverPseudoRQM4<T>(cb, molecules, this->n_streams, this->streams, false);
            else if (numerical_method == "rk2")
                this->propagator_solver = new CudaSolverPseudoRK2<T>(cb, molecules, this->n_streams, this->streams, false);
            else if (numerical_method == "etdrk4")
                this->propagator_solver = new CudaSolverPseudoETDRK4<T>(cb, molecules, this->n_streams, this->streams, false);
            else
                throw_with_line_number("Unknown pseudo-spectral method: '" + numerical_method + "'. Use 'rqm4', 'rk2', or 'etdrk4'.");
        }
        else if(method == "realspace")
        {
            if constexpr (std::is_same<T, double>::value)
            {
                // Local Richardson (cn-adi4-lr) or 2nd order (cn-adi2)
                bool use_4th_order = (numerical_method == "cn-adi4-lr");
                this->propagator_solver = new CudaSolverCNADI(cb, molecules, this->n_streams, this->streams, false, use_4th_order);
            }
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
            this->propagator_finished[key] = new bool[max_n_segment];
            for(int n=0; n<max_n_segment;n++)
                this->propagator_finished[key][n] = false;
            #endif
        }

        // Allocate memory for concentrations
        if( this->propagator_computation_optimizer->get_computation_blocks().size() == 0)
            throw_with_line_number("There is no block. Add polymers first.");
        for(const auto& item: this->propagator_computation_optimizer->get_computation_blocks())
        {
            this->alloc_checkpoint_memory(&this->phi_block[item.first], M);
            std::memset(this->phi_block[item.first], 0, sizeof(T)*M);
        }

        // Allocate memory for check points
        if( this->propagator_computation_optimizer->get_computation_propagators().size() == 0)
            throw_with_line_number("There is no propagator code. Add polymers first.");
        for(const auto& item: this->propagator_computation_optimizer->get_computation_propagators())
        {
            std::string key = item.first;
            auto& deps = this->propagator_computation_optimizer->get_computation_propagator(key).deps;
            int max_n_segment = this->propagator_computation_optimizer->get_computation_propagator(key).max_n_segment;

            if(this->propagator_at_check_point.find(std::make_tuple(key, 0)) == this->propagator_at_check_point.end())
            {
                this->alloc_checkpoint_memory(&this->propagator_at_check_point[std::make_tuple(key, 0)], M);
            }

            for(size_t d=0; d<deps.size(); d++)
            {
                std::string sub_dep = std::get<0>(deps[d]);
                int sub_n_segment   = std::get<1>(deps[d]);

                if(this->propagator_at_check_point.find(std::make_tuple(sub_dep, sub_n_segment)) == this->propagator_at_check_point.end())
                {
                    this->alloc_checkpoint_memory(&this->propagator_at_check_point[std::make_tuple(sub_dep, sub_n_segment)], M);
                }
            }

            #ifndef NDEBUG
            this->propagator_finished[key] = new bool[max_n_segment];
            for(int i=0; i<max_n_segment;i++)
                this->propagator_finished[key][i] = false;
            #endif
        }

        // Find the total sum of segments and allocate temporary memory for recalculating propagators
        this->total_max_n_segment = 0;
        for(const auto& block: this->phi_block)
        {
            const auto& key = block.first;
            int n_segment_right = this->propagator_computation_optimizer->get_computation_block(key).n_segment_right;
            this->total_max_n_segment += n_segment_right;
        }

        // Calculate checkpoint interval as 2*sqrt(N) for better memory-computation tradeoff
        this->checkpoint_interval = static_cast<int>(std::ceil(2.0*std::sqrt(static_cast<double>(this->total_max_n_segment))));
        if (this->checkpoint_interval < 1)
            this->checkpoint_interval = 1;

        #ifndef NDEBUG
        std::cout << "this->total_max_n_segment: " << this->total_max_n_segment << std::endl;
        std::cout << "this->checkpoint_interval: " << this->checkpoint_interval << std::endl;
        #endif

        // Allocate workspace for propagator recomputation
        // We skip intermediate values from checkpoint to left_start, then store from left_start to left_end
        // This requires only this->checkpoint_interval storage
        const int workspace_size = this->checkpoint_interval;
        this->q_recal.resize(workspace_size);
        for(int n=0; n<workspace_size; n++)
        {
            this->alloc_checkpoint_memory(&this->q_recal[n], M);
        }

        // Allocate checkpoints at sqrt(N) intervals for each propagator
        for(const auto& item: this->propagator_computation_optimizer->get_computation_propagators())
        {
            std::string key = item.first;
            int max_n_segment = item.second.max_n_segment;

            // Add checkpoints at sqrt(N) intervals
            for(int n=this->checkpoint_interval; n<max_n_segment; n+=this->checkpoint_interval)
            {
                if(this->propagator_at_check_point.find(std::make_tuple(key, n)) == this->propagator_at_check_point.end())
                {
                    this->alloc_checkpoint_memory(&this->propagator_at_check_point[std::make_tuple(key, n)], M);
                    #ifndef NDEBUG
                    std::cout << "Allocated checkpoint, " + key + ", " << n << std::endl;
                    #endif
                }
            }
        }

        // Remember one segment for each polymer chain to compute total partition function
        int current_p = 0;
        for(const auto& block: this->phi_block)
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

            if(this->propagator_at_check_point.find(std::make_tuple(key_left, n_segment_left)) == this->propagator_at_check_point.end())
            {
                this->alloc_checkpoint_memory(&this->propagator_at_check_point[std::make_tuple(key_left, n_segment_left)], M);
                #ifndef NDEBUG
                std::cout << "Allocated, " + key_left + ", " << n_segment_left << std::endl;
                #endif
            }

            if(this->propagator_at_check_point.find(std::make_tuple(key_right, 0)) == this->propagator_at_check_point.end())
            {
                this->alloc_checkpoint_memory(&this->propagator_at_check_point[std::make_tuple(key_right, 0)], M);
                #ifndef NDEBUG
                std::cout << "Allocated, " + key_right + ", " << 0 << std::endl;
                #endif
            }

            single_partition_segment.push_back(std::make_tuple(
                p,
                this->propagator_at_check_point[std::make_tuple(key_left, n_segment_left)],  // q
                this->propagator_at_check_point[std::make_tuple(key_right, 0)],              // q_dagger
                n_aggregated                                                        // how many propagators are aggregated
                ));
            current_p++;
        }
        // Concentrations for each solvent
        for(int s=0;s<this->molecules->get_n_solvent_types();s++)
            this->phi_solvent.push_back(new T[M]);

        // Create scheduler for computation of propagator
        this->sc = new Scheduler(this->propagator_computation_optimizer->get_computation_propagators(), this->n_streams); 

        // Allocate memory for pseudo-spectral: advance_propagator()
        gpu_error_check(cudaMalloc((void**)&this->d_q_unity, sizeof(T)*M));
        for(int i=0; i<M; i++)
        {
            CuDeviceData<T> q_unity;
            if constexpr (std::is_same<T, double>::value)
                q_unity = 1.0;
            else
                q_unity = make_cuDoubleComplex(1.0, 0.0);
            gpu_error_check(cudaMemcpy(&this->d_q_unity[i], &q_unity, sizeof(T), cudaMemcpyHostToDevice));
        }

        // Allocate memory for propagator computation
        for(int i=0; i<this->n_streams; i++)
        {
            gpu_error_check(cudaMalloc((void**)&this->d_q_one[i][0], sizeof(T)*M));
            gpu_error_check(cudaMalloc((void**)&this->d_q_one[i][1], sizeof(T)*M));
        }

        // Allocate shared workspace (2×M each, contiguous for batch FFT in stress computation)
        // this->d_workspace[0]: primary buffer, this->d_workspace[1]: ping-pong buffer
        // Phase 1 (compute_propagators): uses this->d_propagator_sub_dep aliases
        // Phase 2 (compute_concentrations): uses q_left, q_right, this->d_phi (see memory layout in header)
        // Phase 3 (compute_stress): uses this->d_workspace directly (contiguous 2×M)
        gpu_error_check(cudaMalloc((void**)&this->d_workspace[0], sizeof(T)*2*M));
        gpu_error_check(cudaMalloc((void**)&this->d_workspace[1], sizeof(T)*2*M));

        // Set up aliases
        this->d_propagator_sub_dep[0][0] = this->d_workspace[0];      // first M of this->d_workspace[0]
        this->d_propagator_sub_dep[0][1] = this->d_workspace[0] + M;  // second M of this->d_workspace[0]
        this->d_phi = this->d_workspace[1] + M;                       // second M of this->d_workspace[1]

        // Copy mask to this->d_q_mask
        if (this->cb->get_mask() != nullptr)
        {
            gpu_error_check(cudaMalloc((void**)&this->d_q_mask , sizeof(double)*M));
            gpu_error_check(cudaMemcpy(this->d_q_mask, this->cb->get_mask(), sizeof(double)*M, cudaMemcpyHostToDevice));
        }
        else
            this->d_q_mask = nullptr;

        this->propagator_solver->update_laplacian_operator();
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
CudaComputationReduceMemoryContinuous<T>::~CudaComputationReduceMemoryContinuous()
{
    delete this->propagator_solver;
    delete this->sc;

    // Free checkpoint memory (pinned host memory)
    for(int n=0; n<this->checkpoint_interval; n++)
        this->free_checkpoint_memory(this->q_recal[n]);
    for(const auto& item: this->propagator_at_check_point)
        this->free_checkpoint_memory(item.second);
    for(const auto& item: this->phi_block)
        this->free_checkpoint_memory(item.second);
    for(const auto& item: this->phi_solvent)
        delete[] item;

    #ifndef NDEBUG
    for(const auto& item: this->propagator_finished)
        delete[] item.second;
    #endif

    // Free GPU workspace
    // Only free actual allocations (this->d_propagator_sub_dep, this->d_phi are aliases into this->d_workspace)
    for(int i=0; i<this->n_streams; i++)
    {
        cudaFree(this->d_q_one[i][0]);
        cudaFree(this->d_q_one[i][1]);
    }
    cudaFree(this->d_workspace[0]);  // contains this->d_propagator_sub_dep[0][*], used for q_left, q_right[0] in concentration
    cudaFree(this->d_workspace[1]);  // contains q_right[1] ping-pong buffer and this->d_phi

    // For pseudo-spectral: advance_propagator()
    if (this->d_q_mask != nullptr)
        cudaFree(this->d_q_mask);
    cudaFree(this->d_q_unity);
    
    // Destroy streams
    for(int i=0; i<this->n_streams; i++)
    {
        cudaStreamDestroy(this->streams[i][0]);
        cudaStreamDestroy(this->streams[i][1]);
    }
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
        const double ds = this->molecules->get_global_ds();

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
        this->propagator_solver->update_dw(device, w_input);

        // For each time span
        #ifndef NDEBUG
        for(const auto& item: this->propagator_computation_optimizer->get_computation_propagators())
        {
            std::string key = item.first;
            int max_n_segment = item.second.max_n_segment+1;
            for(int i=0; i<max_n_segment;i++)
                this->propagator_finished[key][i] = false;
        }
        #endif

        auto& branch_schedule = this->sc->get_schedule();
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
                if (this->propagator_at_check_point.find(std::make_tuple(key, n_segment_from)) == this->propagator_at_check_point.end())
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
                        gpu_error_check(cudaMemcpy(this->d_q_one[STREAM][0], q_init[g], sizeof(T)*M, cudaMemcpyInputToDevice));
                    }
                    else
                    {
                        gpu_error_check(cudaMemcpy(this->d_q_one[STREAM][0], this->d_q_unity, sizeof(T)*M, cudaMemcpyDeviceToDevice));
                    }

                    #ifndef NDEBUG
                    this->propagator_finished[key][0] = true;
                    #endif
                }
                // If it is not leaf node
                else if (n_segment_from == 0 && deps.size() > 0)
                {
                    // If it is aggregated
                    if (key[0] == '[')
                    {
                        // Initialize to zero
                        gpu_error_check(cudaMemset(this->d_q_one[STREAM][0], 0, sizeof(T)*M));

                        int prev, next;
                        prev = 0;
                        next = 1;

                        // Copy memory from checkpoint to device
                        std::string sub_dep = std::get<0>(deps[0]);
                        int sub_n_segment   = std::get<1>(deps[0]);
                        int sub_n_repeated  = std::get<2>(deps[0]);

                        T* _q_sub = this->propagator_at_check_point[std::make_tuple(sub_dep, sub_n_segment)];
                        cudaMemcpyKind memcpy_from_checkpoint = cudaMemcpyHostToDevice;
                        gpu_error_check(cudaMemcpy(this->d_propagator_sub_dep[STREAM][prev], _q_sub, sizeof(T)*M, memcpy_from_checkpoint));

                        for(size_t d=0; d<deps.size(); d++)
                        {
                            sub_dep         = std::get<0>(deps[d]);
                            sub_n_segment   = std::get<1>(deps[d]);
                            sub_n_repeated  = std::get<2>(deps[d]);

                            // Check sub key
                            #ifndef NDEBUG
                            if (this->propagator_at_check_point.find(std::make_tuple(sub_dep, sub_n_segment)) == this->propagator_at_check_point.end())
                                std::cout << "Could not find sub key '" + sub_dep + "[" + std::to_string(sub_n_segment) + "]. " << std::endl;
                            if (!this->propagator_finished[sub_dep][sub_n_segment])
                                std::cout<< "Could not compute '" + key +  "', since '"+ sub_dep + std::to_string(sub_n_segment) + "' is not prepared." << std::endl;
                            #endif

                            // MEMORY STREAM: copy memory from checkpoint to device
                            if (d < deps.size()-1)
                            {
                                std::string sub_dep_next = std::get<0>(deps[d+1]);
                                int sub_n_segment_next   = std::get<1>(deps[d+1]);

                                T* _q_sub = this->propagator_at_check_point[std::make_tuple(sub_dep_next, sub_n_segment_next)];
                                gpu_error_check(cudaMemcpyAsync(this->d_propagator_sub_dep[STREAM][next],
                                                _q_sub, sizeof(T)*M,
                                                memcpy_from_checkpoint, this->streams[STREAM][1]));
                            }

                            // KERNEL STREAM: compute linear combination
                            ker_lin_comb<<<N_BLOCKS, N_THREADS, 0, this->streams[STREAM][0]>>>(
                                    this->d_q_one[STREAM][0], 1.0, this->d_q_one[STREAM][0],
                                    sub_n_repeated, this->d_propagator_sub_dep[STREAM][prev], M);

                            std::swap(prev, next);
                            cudaDeviceSynchronize();
                        }

                        #ifndef NDEBUG
                        this->propagator_finished[key][0] = true;
                        #endif
                    }
                    else if(key[0] == '(')
                    {
                        // Initialize to one
                        gpu_error_check(cudaMemcpy(this->d_q_one[STREAM][0], this->d_q_unity, sizeof(T)*M, cudaMemcpyDeviceToDevice));

                        int prev, next;
                        prev = 0;
                        next = 1;

                        // Copy memory from checkpoint to device
                        std::string sub_dep = std::get<0>(deps[0]);
                        int sub_n_segment   = std::get<1>(deps[0]);
                        int sub_n_repeated  = std::get<2>(deps[0]);

                        T* _q_sub = this->propagator_at_check_point[std::make_tuple(sub_dep, sub_n_segment)];
                        cudaMemcpyKind memcpy_from_checkpoint = cudaMemcpyHostToDevice;
                        gpu_error_check(cudaMemcpy(this->d_propagator_sub_dep[STREAM][prev], _q_sub, sizeof(T)*M, memcpy_from_checkpoint));

                        for(size_t d=0; d<deps.size(); d++)
                        {
                            sub_dep         = std::get<0>(deps[d]);
                            sub_n_segment   = std::get<1>(deps[d]);
                            sub_n_repeated  = std::get<2>(deps[d]);

                            // Check sub key
                            #ifndef NDEBUG
                            if (this->propagator_at_check_point.find(std::make_tuple(sub_dep, sub_n_segment)) == this->propagator_at_check_point.end())
                                std::cout << "Could not find sub key '" + sub_dep + "[" + std::to_string(sub_n_segment) + "]. " << std::endl;
                            if (!this->propagator_finished[sub_dep][sub_n_segment])
                                std::cout<< "Could not compute '" + key +  "', since '"+ sub_dep + std::to_string(sub_n_segment) + "' is not prepared." << std::endl;
                            #endif

                            // MEMORY STREAM: copy memory from checkpoint to device
                            if (d < deps.size()-1)
                            {
                                std::string sub_dep_next = std::get<0>(deps[d+1]);
                                int sub_n_segment_next   = std::get<1>(deps[d+1]);

                                T* _q_sub = this->propagator_at_check_point[std::make_tuple(sub_dep_next, sub_n_segment_next)];
                                gpu_error_check(cudaMemcpyAsync(this->d_propagator_sub_dep[STREAM][next],
                                                _q_sub, sizeof(T)*M,
                                                memcpy_from_checkpoint, this->streams[STREAM][1]));
                            }

                            // KERNEL STREAM: multiply
                            for(int r=0; r<sub_n_repeated; r++)
                            {
                                ker_multi<<<N_BLOCKS, N_THREADS, 0, this->streams[STREAM][0]>>>(
                                    this->d_q_one[STREAM][0], this->d_q_one[STREAM][0], this->d_propagator_sub_dep[STREAM][prev], 1.0, M);
                            }

                            std::swap(prev, next);

                            gpu_error_check(cudaStreamSynchronize(this->streams[STREAM][0]));
                            gpu_error_check(cudaStreamSynchronize(this->streams[STREAM][1]));
                        }

                        #ifndef NDEBUG
                        this->propagator_finished[key][0] = true;
                        #endif
                    }
                }

                // Multiply mask
                if (n_segment_from == 0 && this->d_q_mask != nullptr)
                    ker_multi<<<N_BLOCKS, N_THREADS>>>(this->d_q_one[STREAM][0], this->d_q_one[STREAM][0], this->d_q_mask, 1.0, M);

                // Copy data between device and checkpoint
                cudaMemcpyKind memcpy_to_checkpoint = cudaMemcpyDeviceToHost;
                cudaMemcpyKind memcpy_from_checkpoint_sync = cudaMemcpyHostToDevice;
                if (n_segment_from == 0)
                {
                    T* _q_target = this->propagator_at_check_point[std::make_tuple(key, 0)];
                    gpu_error_check(cudaMemcpy(_q_target, this->d_q_one[STREAM][0], sizeof(T)*M, memcpy_to_checkpoint));
                }
                else
                {
                    T* _q_from = this->propagator_at_check_point[std::make_tuple(key, n_segment_from)];
                    gpu_error_check(cudaMemcpy(this->d_q_one[STREAM][0], _q_from, sizeof(T)*M, memcpy_from_checkpoint_sync));
                }

                int prev, next;
                prev = 0;
                next = 1;

                // Get ds_index from the propagator key
                int ds_index = PropagatorCode::get_ds_index_from_key(key);
                if (ds_index < 1) ds_index = 1;  // Default to global ds

                // Reset solver internal state when starting a new propagator
                // (needed for Global Richardson method)
                if (n_segment_from == 0)
                    this->propagator_solver->reset_internal_state(STREAM);

                // Create events
                cudaEvent_t kernel_done;
                cudaEvent_t memcpy_done;
                gpu_error_check(cudaEventCreate(&kernel_done));
                gpu_error_check(cudaEventCreate(&memcpy_done));

                for(int n=n_segment_from; n<n_segment_to; n++)
                {
                    #ifndef NDEBUG
                    if (!this->propagator_finished[key][n])
                        std::cout << "unfinished, key: " + key + ", " + std::to_string(n) << std::endl;
                    if (this->propagator_finished[key][n+1])
                        std::cout << "already finished: " + key + ", " + std::to_string(n+1) << std::endl;
                    #endif

                    // KERNEL STREAM: calculate propagators
                    this->propagator_solver->advance_propagator(
                        STREAM,
                        this->d_q_one[STREAM][prev],
                        this->d_q_one[STREAM][next],
                        monomer_type, this->d_q_mask, ds_index);
                    gpu_error_check(cudaEventRecord(kernel_done, this->streams[STREAM][0]));

                    // MEMORY STREAM: copy memory from device to checkpoint
                    if (n > n_segment_from && this->propagator_at_check_point.find(std::make_tuple(key, n)) != this->propagator_at_check_point.end())
                    {
                        T* _q_target =  this->propagator_at_check_point[std::make_tuple(key, n)];
                        gpu_error_check(cudaMemcpyAsync(
                            _q_target,
                            this->d_q_one[STREAM][prev],
                            sizeof(T)*M, memcpy_to_checkpoint, this->streams[STREAM][1]));
                        gpu_error_check(cudaEventRecord(memcpy_done, this->streams[STREAM][1]));
                    }

                    // Wait until computation and memory copy are done
                    gpu_error_check(cudaStreamWaitEvent(this->streams[STREAM][1], kernel_done, 0));
                    gpu_error_check(cudaStreamWaitEvent(this->streams[STREAM][0], memcpy_done, 0));

                    std::swap(prev, next);

                    #ifndef NDEBUG
                    this->propagator_finished[key][n+1] = true;
                    #endif
                }

                // Copy memory from device to checkpoint
                if (this->propagator_at_check_point.find(std::make_tuple(key, n_segment_to)) != this->propagator_at_check_point.end())
                {
                    T* _q_target =  this->propagator_at_check_point[std::make_tuple(key, n_segment_to)];
                    gpu_error_check(cudaMemcpyAsync(
                        _q_target, this->d_q_one[STREAM][prev],
                        sizeof(T)*M, memcpy_to_checkpoint, this->streams[STREAM][1]));
                }

                gpu_error_check(cudaStreamSynchronize(this->streams[STREAM][0]));
                gpu_error_check(cudaStreamSynchronize(this->streams[STREAM][1]));
            
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

            // Checkpoints are in pinned host memory, use host inner product
            this->single_polymer_partitions[p] = this->cb->inner_product(
                propagator_left, propagator_right)/(n_aggregated*this->cb->get_volume());
        }
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
        for(const auto& block: this->phi_block)
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
            if (this->propagator_at_check_point.find(std::make_tuple(key_left, n_segment_left-n_segment_right)) == this->propagator_at_check_point.end())
                std::cout << "Check point at " + key_left + "[" + std::to_string(n_segment_left-n_segment_right) + "] is missing. ";
            if (this->propagator_at_check_point.find(std::make_tuple(key_right, 0)) == this->propagator_at_check_point.end())
                std::cout << "Check point at " + key_right + "[" + std::to_string(0) + "] is missing. ";
            #endif

            // Normalization constant
            Polymer& pc = this->molecules->get_polymer(p);
            T norm = (pc.get_volume_fraction()/pc.get_n_segment_total()*n_repeated)/this->single_polymer_partitions[p];

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
            CuDeviceData<T> *_d_exp_dw = this->propagator_solver->d_exp_dw[1][monomer_type];

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

            ker_multi<<<N_BLOCKS, N_THREADS>>>(this->d_phi, _d_exp_dw, _d_exp_dw, norm, M);
            gpu_error_check(cudaMemcpy(this->phi_solvent[s], this->d_phi, sizeof(T)*M, cudaMemcpyDeviceToHost));
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

        // Get ds_index from the propagator key
        int ds_index = PropagatorCode::get_ds_index_from_key(key);
        if (ds_index < 1) ds_index = 1;  // Default to global ds

        // Reset solver internal state when starting propagator recalculation
        // (needed for Global Richardson method)
        this->propagator_solver->reset_internal_state(0);

        // An array of pointers for q_out (use dynamic container to avoid VLA/stack overflow)
        std::vector<T*> q_out(this->total_max_n_segment + 1);

        // If a propagator is in this->propagator_at_check_point reuse it, otherwise compute it again with allocated memory space.
        for(int n=0; n<=N_RIGHT; n++)
        {
            auto it = this->propagator_at_check_point.find(std::make_tuple(key, N_START+n));
            if(it != this->propagator_at_check_point.end())
            {
                q_out[n] = it->second;
                #ifndef NDEBUG
                std::cout << "Use this->propagator_at_check_point if exists: (phi, left) " << key << ", " << N_START+n << std::endl;
                #endif
            }
            else
            {
                q_out[n] = this->q_recal[n];
            }
        }
        // Copy propagators from host to device
        gpu_error_check(cudaMemcpy(this->d_q_one[0][prev], q_out[0], sizeof(T)*M, cudaMemcpyHostToDevice));

        // Compute the q_out
        for(int n=0; n<=N_RIGHT; n++)
        {
            // MEMORY STREAM: copy memory from device to host
            if (n > 0)
            {
                // Use this->propagator_at_check_point if exists
                if(this->propagator_at_check_point.find(std::make_tuple(key, N_START+n)) == this->propagator_at_check_point.end())
                {
                    gpu_error_check(cudaMemcpyAsync(
                        q_out[n], this->d_q_one[0][prev],
                        sizeof(T)*M, cudaMemcpyDeviceToHost, this->streams[0][1]));
                }
            }
            // KERNEL STREAM: calculate propagator
            if ((n+1 < N_RIGHT) ||
                // Compute the last, if the q_out[N_START+N_RIGHT] is not in this->propagator_at_check_point
                (n+1 == N_RIGHT && this->propagator_at_check_point.find(std::make_tuple(key, N_START+N_RIGHT)) == this->propagator_at_check_point.end()))
            {
                this->propagator_solver->advance_propagator(
                    0,
                    this->d_q_one[0][prev],   // q_out[n]
                    this->d_q_one[0][next],   // q_out[n+1]
                    monomer_type, this->d_q_mask, ds_index);
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
        const int k = this->checkpoint_interval;

        // Get ds_index from the propagator key (use key_left, both should have same ds_index)
        int ds_index = PropagatorCode::get_ds_index_from_key(key_left);
        if (ds_index < 1) ds_index = 1;  // Default to global ds

        // Set up local workspace pointers (see memory layout in header)
        CuDeviceData<T> *d_q_left = this->d_workspace[0];                           // first M of this->d_workspace[0]
        CuDeviceData<T> *d_q_right[2] = {this->d_workspace[0] + M, this->d_workspace[1]}; // ping-pong buffers

        std::vector<double> simpson_rule_coeff = SimpsonRule::get_coeff(N_RIGHT);

        // Determine memory copy direction based on checkpoint storage location
        cudaMemcpyKind memcpy_from_checkpoint = cudaMemcpyHostToDevice;
        cudaMemcpyKind memcpy_to_recal = cudaMemcpyDeviceToHost;
        cudaMemcpyKind memcpy_from_recal = cudaMemcpyHostToDevice;

        // Initialize to zero
        gpu_error_check(cudaMemset(this->d_phi, 0, sizeof(T)*M));

        // Initialize q_right at segment 0
        gpu_error_check(cudaMemcpy(d_q_right[0], this->propagator_at_check_point[std::make_tuple(key_right, 0)],
            sizeof(T)*M, memcpy_from_checkpoint));
        int right_prev_idx = 0;
        int right_next_idx = 1;
        int current_n_right = 0;

        // Reset solver internal state for q_right (starts from segment 0)
        // Note: For Global Richardson, this only works correctly for q_right
        // which starts from 0. For q_left (starting from checkpoint), accuracy
        // may be reduced compared to recomputing from segment 0.
        this->propagator_solver->reset_internal_state(STREAM);

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
            auto it = this->propagator_at_check_point.find(std::make_tuple(key_left, check_pos));
            while (it == this->propagator_at_check_point.end() && check_pos > 0)
            {
                check_pos -= k;
                if (check_pos < 0) check_pos = 0;
                it = this->propagator_at_check_point.find(std::make_tuple(key_left, check_pos));
            }

            // Skip from checkpoint to left_start (don't store intermediate values)
            gpu_error_check(cudaMemcpy(this->d_q_one[STREAM][0], it->second, sizeof(T)*M, memcpy_from_checkpoint));
            for (int i = check_pos; i < left_start; i++)
            {
                this->propagator_solver->advance_propagator(STREAM, this->d_q_one[STREAM][0], this->d_q_one[STREAM][1], monomer_type, this->d_q_mask, ds_index);
                gpu_error_check(cudaMemcpy(this->d_q_one[STREAM][0], this->d_q_one[STREAM][1], sizeof(T)*M, cudaMemcpyDeviceToDevice));
            }

            // Store q_left from left_start to left_end
            gpu_error_check(cudaMemcpy(this->q_recal[0], this->d_q_one[STREAM][0], sizeof(T)*M, memcpy_to_recal));
            for (int i = 1; i <= left_end - left_start; i++)
            {
                this->propagator_solver->advance_propagator(STREAM, this->d_q_one[STREAM][0], this->d_q_one[STREAM][1], monomer_type, this->d_q_mask, ds_index);
                gpu_error_check(cudaMemcpy(this->q_recal[i], this->d_q_one[STREAM][1], sizeof(T)*M, memcpy_to_recal));
                gpu_error_check(cudaMemcpy(this->d_q_one[STREAM][0], this->d_q_one[STREAM][1], sizeof(T)*M, cudaMemcpyDeviceToDevice));
            }
            // Now this->q_recal[i] = q_left[left_start + i]

            // Process each n in [n_start, n_end]
            for (int n = n_start; n <= n_end; n++)
            {
                // Advance q_right to position n
                while (current_n_right < n)
                {
                    this->propagator_solver->advance_propagator(STREAM, d_q_right[right_prev_idx], d_q_right[right_next_idx], monomer_type, this->d_q_mask, ds_index);
                    cudaDeviceSynchronize();
                    std::swap(right_prev_idx, right_next_idx);
                    current_n_right++;
                }

                // Get q_left position (segment N_LEFT - n)
                int left_pos = N_LEFT - n;
                int ws_idx = left_pos - left_start;

                // Copy q_left to device
                gpu_error_check(cudaMemcpy(d_q_left, this->q_recal[ws_idx], sizeof(T)*M, memcpy_from_recal));

                CuDeviceData<T> norm;
                if constexpr (std::is_same<T, double>::value)
                    norm = NORM*simpson_rule_coeff[n];
                else
                    norm = stdToCuDoubleComplex(NORM*simpson_rule_coeff[n]);

                // Multiply and accumulate
                ker_add_multi<<<N_BLOCKS, N_THREADS>>>(this->d_phi, d_q_left, d_q_right[right_prev_idx], norm, M);
            }
        }

        // Copy result from device to phi_block storage
        cudaMemcpyKind memcpy_to_phi = cudaMemcpyDeviceToHost;
        gpu_error_check(cudaMemcpy(phi, this->d_phi, sizeof(T)*M, memcpy_to_phi));
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
        const int k = this->checkpoint_interval;

        const int N_STRESS = 6;
        std::map<std::tuple<int, std::string, std::string>, std::array<T,N_STRESS>> block_dq_dl[this->n_streams];

        // Reset stress map
        for(const auto& item: this->phi_block)
        {
            for(int i=0; i<this->n_streams; i++)
                for(int d=0; d<N_STRESS; d++)
                    block_dq_dl[i][item.first][d] = 0.0;
        }

        // Compute stress for each this->phi_block
        for(size_t b=0; b<this->phi_block.size();b++)
        {
            const int STREAM = omp_get_thread_num();
            auto block = this->phi_block.begin();
            advance(block, b);
            const auto& key   = block->first;

            int p                 = std::get<0>(key);
            std::string key_left  = std::get<1>(key);
            std::string key_right = std::get<2>(key);

            const int N_RIGHT = this->propagator_computation_optimizer->get_computation_block(key).n_segment_right;
            const int N_LEFT  = this->propagator_computation_optimizer->get_computation_block(key).n_segment_left;
            std::string monomer_type = this->propagator_computation_optimizer->get_computation_block(key).monomer_type;
            int n_repeated = this->propagator_computation_optimizer->get_computation_block(key).n_repeated;

            // Get ds_index from the propagator key
            int ds_index = PropagatorCode::get_ds_index_from_key(key_left);
            if (ds_index < 1) ds_index = 1;  // Default to global ds

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

            // Determine memory copy direction based on checkpoint storage location
            cudaMemcpyKind memcpy_from_checkpoint = cudaMemcpyHostToDevice;
            cudaMemcpyKind memcpy_to_recal = cudaMemcpyDeviceToHost;
            cudaMemcpyKind memcpy_from_recal = cudaMemcpyHostToDevice;

            // Initialize q_right at position 0
            auto it_right_0 = this->propagator_at_check_point.find(std::make_tuple(key_right, 0));
            if(it_right_0 != this->propagator_at_check_point.end())
            {
                gpu_error_check(cudaMemcpy(&this->d_workspace[right_prev_idx][M], it_right_0->second, sizeof(T)*M, memcpy_from_checkpoint));
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
                    if(this->propagator_at_check_point.find(std::make_tuple(key_left, cp)) != this->propagator_at_check_point.end())
                        check_pos = cp;
                }

                if(check_pos < 0)
                    continue;

                // Recompute q_left from checkpoint to left_end using this->q_recal workspace
                const int steps_before = left_start - check_pos;
                const int storage_count = left_end - left_start + 1;

                auto it_checkpoint = this->propagator_at_check_point.find(std::make_tuple(key_left, check_pos));
                if(it_checkpoint == this->propagator_at_check_point.end())
                    continue;

                // Load checkpoint to device and compute q_left positions
                gpu_error_check(cudaMemcpy(this->d_q_one[0][0], it_checkpoint->second, sizeof(T)*M, memcpy_from_checkpoint));
                int left_ping = 0;
                int left_pong = 1;

                // Advance to left_start
                for(int step = 0; step < steps_before; step++)
                {
                    int actual_pos = check_pos + step + 1;
                    auto it = this->propagator_at_check_point.find(std::make_tuple(key_left, actual_pos));
                    if(it != this->propagator_at_check_point.end())
                    {
                        gpu_error_check(cudaMemcpy(this->d_q_one[0][left_pong], it->second, sizeof(T)*M, memcpy_from_checkpoint));
                    }
                    else
                    {
                        this->propagator_solver->advance_propagator(0, this->d_q_one[0][left_ping], this->d_q_one[0][left_pong], monomer_type, this->d_q_mask, ds_index);
                        cudaDeviceSynchronize();
                    }
                    std::swap(left_ping, left_pong);
                }

                // Now this->d_q_one[0][left_ping] contains q_left[left_start]
                // Copy to this->q_recal[0] and compute remaining positions
                gpu_error_check(cudaMemcpy(this->q_recal[0], this->d_q_one[0][left_ping], sizeof(T)*M, memcpy_to_recal));

                for(int idx = 1; idx < storage_count; idx++)
                {
                    int actual_pos = left_start + idx;
                    auto it = this->propagator_at_check_point.find(std::make_tuple(key_left, actual_pos));
                    if(it != this->propagator_at_check_point.end())
                    {
                        // Copy from pinned host memory to recal buffer
                        for(int i=0; i<M; i++)
                            this->q_recal[idx][i] = it->second[i];
                        gpu_error_check(cudaMemcpy(this->d_q_one[0][left_pong], it->second, sizeof(T)*M, memcpy_from_checkpoint));
                    }
                    else
                    {
                        this->propagator_solver->advance_propagator(0, this->d_q_one[0][left_ping], this->d_q_one[0][left_pong], monomer_type, this->d_q_mask, ds_index);
                        cudaDeviceSynchronize();
                        gpu_error_check(cudaMemcpy(this->q_recal[idx], this->d_q_one[0][left_pong], sizeof(T)*M, memcpy_to_recal));
                    }
                    std::swap(left_ping, left_pong);
                }

                // Process each n in [n_start, n_end]
                for(int n = n_start; n <= n_end; n++)
                {
                    // Advance q_right if needed
                    auto it_right = this->propagator_at_check_point.find(std::make_tuple(key_right, n));
                    if(it_right != this->propagator_at_check_point.end())
                    {
                        gpu_error_check(cudaMemcpy(&this->d_workspace[right_prev_idx][M], it_right->second, sizeof(T)*M, memcpy_from_checkpoint));
                        current_n_right = n;
                    }
                    else
                    {
                        while(current_n_right < n)
                        {
                            this->propagator_solver->advance_propagator(0, &this->d_workspace[right_prev_idx][M], &this->d_workspace[right_next_idx][M], monomer_type, this->d_q_mask, ds_index);
                            cudaDeviceSynchronize();
                            std::swap(right_prev_idx, right_next_idx);
                            current_n_right++;
                        }
                    }

                    // Get q_left[N_LEFT - n] from this->q_recal
                    int left_pos = N_LEFT - n;
                    int storage_idx = left_pos - left_start;

                    // Copy q_left to device for stress computation
                    gpu_error_check(cudaMemcpy(&this->d_workspace[right_prev_idx][0], this->q_recal[storage_idx], sizeof(T)*M, memcpy_from_recal));

                    // Compute stress (this->d_workspace[idx] is contiguous 2×M buffer)
                    this->propagator_solver->compute_single_segment_stress(0, this->d_workspace[right_prev_idx], d_segment_stress, monomer_type, false);
                    cudaDeviceSynchronize();

                    gpu_error_check(cudaMemcpy(segment_stress, d_segment_stress, sizeof(T)*N_STRESS_DIM, cudaMemcpyDeviceToHost));
                    for(int d=0; d<N_STRESS_DIM; d++)
                        _block_dq_dl[d] += segment_stress[d]*(s_coeff[n]*n_repeated);
                }
            }

            // Multiply by local_ds for this block
            Polymer& pc_stress = this->molecules->get_polymer(p);
            const auto& v_u_stress = this->propagator_computation_optimizer->get_computation_block(key).v_u;
            int v_stress = std::get<0>(v_u_stress[0]);
            int u_stress = std::get<1>(v_u_stress[0]);
            double contour_length = pc_stress.get_block(v_stress, u_stress).contour_length;
            const ContourLengthMapping& mapping_stress = this->molecules->get_contour_length_mapping();
            double local_ds = mapping_stress.get_local_ds(contour_length);
            for(int d=0; d<N_STRESS_DIM; d++)
                _block_dq_dl[d] *= local_ds;

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
        for(const auto& block: this->phi_block)
        {
            const auto& key       = block.first;
            int p                 = std::get<0>(key);
            std::string key_left  = std::get<1>(key);
            std::string key_right = std::get<2>(key);
            Polymer& pc = this->molecules->get_polymer(p);

            for(int i=0; i<this->n_streams; i++)
                for(int d=0; d<N_STRESS_TOTAL; d++)
                    this->dq_dl[p][d] += block_dq_dl[i][key][d];
        }
        // Normalize stress components
        // Note: local_ds is already multiplied per-block in the stress loop above
        for(int p=0; p<n_polymer_types; p++)
        {
            // Diagonal components: xx, yy, zz
            for(int d=0; d<DIM; d++)
                this->dq_dl[p][d] /= -3.0*this->cb->get_lx(d)*M*M;
            // Cross-term components for 3D: xy, xz, yz
            if (DIM == 3)
            {
                this->dq_dl[p][3] /= -3.0*std::sqrt(this->cb->get_lx(0)*this->cb->get_lx(1))*M*M;
                this->dq_dl[p][4] /= -3.0*std::sqrt(this->cb->get_lx(0)*this->cb->get_lx(2))*M*M;
                this->dq_dl[p][5] /= -3.0*std::sqrt(this->cb->get_lx(1)*this->cb->get_lx(2))*M*M;
            }
            // Cross-term component for 2D: yz
            else if (DIM == 2)
            {
                this->dq_dl[p][2] /= -3.0*std::sqrt(this->cb->get_lx(0)*this->cb->get_lx(1))*M*M;
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

        // Determine memory copy direction based on checkpoint storage location
        cudaMemcpyKind memcpy_from_checkpoint = cudaMemcpyHostToDevice;

        // Check if the requested segment is at a checkpoint
        auto checkpoint_key = std::make_tuple(dep, n);
        if (this->propagator_at_check_point.find(checkpoint_key) != this->propagator_at_check_point.end())
        {
            // Directly copy from pinned host checkpoint
            T *_q_from = this->propagator_at_check_point[checkpoint_key];
            for(int i=0; i<M; i++)
                q_out[i] = _q_from[i];
        }
        else
        {
            // Find nearest checkpoint at or before position n
            std::string monomer_type = this->propagator_computation_optimizer->get_computation_propagator(dep).monomer_type;

            // Get ds_index from the propagator key
            int ds_index = PropagatorCode::get_ds_index_from_key(dep);
            if (ds_index < 1) ds_index = 1;  // Default to global ds

            int check_pos = -1;
            for(int cp = 0; cp <= n; cp++)
            {
                if(this->propagator_at_check_point.find(std::make_tuple(dep, cp)) != this->propagator_at_check_point.end())
                    check_pos = cp;
            }

            if(check_pos < 0)
                throw_with_line_number("No checkpoint found at or before position " + std::to_string(n));

            // Load checkpoint and advance to position n using ping-pong buffers
            T* q_checkpoint = this->propagator_at_check_point[std::make_tuple(dep, check_pos)];
            gpu_error_check(cudaMemcpy(this->d_q_one[0][0], q_checkpoint, sizeof(T)*M, memcpy_from_checkpoint));

            int ping = 0;
            int pong = 1;
            for(int step = check_pos; step < n; step++)
            {
                // Check if next position is a checkpoint
                auto it = this->propagator_at_check_point.find(std::make_tuple(dep, step + 1));
                if(it != this->propagator_at_check_point.end())
                {
                    gpu_error_check(cudaMemcpy(this->d_q_one[0][pong], it->second, sizeof(T)*M, memcpy_from_checkpoint));
                }
                else
                {
                    this->propagator_solver->advance_propagator(0, this->d_q_one[0][ping], this->d_q_one[0][pong], monomer_type, this->d_q_mask, ds_index);
                    cudaDeviceSynchronize();
                }
                std::swap(ping, pong);
            }

            // Copy result from device to output (always DeviceToHost since q_out is host memory)
            gpu_error_check(cudaMemcpy(q_out, this->d_q_one[0][ping], sizeof(T)*M, cudaMemcpyDeviceToHost));
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
    const int k = this->checkpoint_interval;
    int n_polymer_types = this->molecules->get_n_polymer_types();

    std::vector<std::vector<T>> total_partitions;
    for(int p=0;p<n_polymer_types;p++)
    {
        std::vector<T> total_partitions_p;
        total_partitions.push_back(total_partitions_p);
    }

    for(const auto& block: this->phi_block)
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

        // Get ds_index from the propagator key
        int ds_index = PropagatorCode::get_ds_index_from_key(key_left);
        if (ds_index < 1) ds_index = 1;  // Default to global ds

        #ifndef NDEBUG
        std::cout<< p << ", " << key_left << ", " << key_right << ": " << N_LEFT << ", " << N_RIGHT << ", " << n_propagators << ", " << n_repeated << std::endl;
        #endif

        // Number of blocks
        const int num_blocks = (N_RIGHT + k) / k;

        // Set up local workspace pointers (see memory layout in header)
        CuDeviceData<T> *d_q_left = this->d_workspace[0];                           // first M of this->d_workspace[0]
        CuDeviceData<T> *d_q_right[2] = {this->d_workspace[0] + M, this->d_workspace[1]}; // ping-pong buffers

        // Determine memory copy direction based on checkpoint storage location
        cudaMemcpyKind memcpy_from_checkpoint = cudaMemcpyHostToDevice;
        cudaMemcpyKind memcpy_to_recal = cudaMemcpyDeviceToHost;
        cudaMemcpyKind memcpy_from_recal = cudaMemcpyHostToDevice;

        // Pointers for q_right (ping-pong)
        int right_prev_idx = 0;
        int right_next_idx = 1;

        // Current position of q_right
        int current_n_right = -1;

        // Initialize q_right at position 0
        auto it_right_0 = this->propagator_at_check_point.find(std::make_tuple(key_right, 0));
        if(it_right_0 != this->propagator_at_check_point.end())
        {
            gpu_error_check(cudaMemcpy(d_q_right[right_prev_idx], it_right_0->second, sizeof(T)*M, memcpy_from_checkpoint));
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
                if(this->propagator_at_check_point.find(std::make_tuple(key_left, cp)) != this->propagator_at_check_point.end())
                    check_pos = cp;
            }

            if(check_pos < 0)
                continue;

            // Recompute q_left from checkpoint to left_end
            const int steps_before = left_start - check_pos;
            const int storage_count = left_end - left_start + 1;

            auto it_checkpoint = this->propagator_at_check_point.find(std::make_tuple(key_left, check_pos));
            if(it_checkpoint == this->propagator_at_check_point.end())
                continue;

            // Load checkpoint to device and compute q_left positions
            gpu_error_check(cudaMemcpy(this->d_q_one[0][0], it_checkpoint->second, sizeof(T)*M, memcpy_from_checkpoint));
            int left_ping = 0;
            int left_pong = 1;

            // Advance to left_start
            for(int step = 0; step < steps_before; step++)
            {
                int actual_pos = check_pos + step + 1;
                auto it = this->propagator_at_check_point.find(std::make_tuple(key_left, actual_pos));
                if(it != this->propagator_at_check_point.end())
                {
                    gpu_error_check(cudaMemcpy(this->d_q_one[0][left_pong], it->second, sizeof(T)*M, memcpy_from_checkpoint));
                }
                else
                {
                    this->propagator_solver->advance_propagator(0, this->d_q_one[0][left_ping], this->d_q_one[0][left_pong], monomer_type, this->d_q_mask, ds_index);
                    cudaDeviceSynchronize();
                }
                std::swap(left_ping, left_pong);
            }

            // Copy to this->q_recal[0] and compute remaining positions
            gpu_error_check(cudaMemcpy(this->q_recal[0], this->d_q_one[0][left_ping], sizeof(T)*M, memcpy_to_recal));

            for(int idx = 1; idx < storage_count; idx++)
            {
                int actual_pos = left_start + idx;
                auto it = this->propagator_at_check_point.find(std::make_tuple(key_left, actual_pos));
                if(it != this->propagator_at_check_point.end())
                {
                    // Copy from pinned host memory to recal buffer
                    for(int i=0; i<M; i++)
                        this->q_recal[idx][i] = it->second[i];
                    gpu_error_check(cudaMemcpy(this->d_q_one[0][left_pong], it->second, sizeof(T)*M, memcpy_from_checkpoint));
                }
                else
                {
                    this->propagator_solver->advance_propagator(0, this->d_q_one[0][left_ping], this->d_q_one[0][left_pong], monomer_type, this->d_q_mask, ds_index);
                    cudaDeviceSynchronize();
                    gpu_error_check(cudaMemcpy(this->q_recal[idx], this->d_q_one[0][left_pong], sizeof(T)*M, memcpy_to_recal));
                }
                std::swap(left_ping, left_pong);
            }

            // Process each n in [n_start, n_end]
            for(int n = n_start; n <= n_end; n++)
            {
                // Advance q_right if needed
                auto it_right = this->propagator_at_check_point.find(std::make_tuple(key_right, n));
                if(it_right != this->propagator_at_check_point.end())
                {
                    gpu_error_check(cudaMemcpy(d_q_right[right_prev_idx], it_right->second, sizeof(T)*M, memcpy_from_checkpoint));
                    current_n_right = n;
                }
                else
                {
                    while(current_n_right < n)
                    {
                        this->propagator_solver->advance_propagator(0, d_q_right[right_prev_idx], d_q_right[right_next_idx], monomer_type, this->d_q_mask, ds_index);
                        cudaDeviceSynchronize();
                        std::swap(right_prev_idx, right_next_idx);
                        current_n_right++;
                    }
                }

                // Get q_left[N_LEFT - n] from this->q_recal
                int left_pos = N_LEFT - n;
                int storage_idx = left_pos - left_start;

                // Copy q_left to device for inner product
                gpu_error_check(cudaMemcpy(d_q_left, this->q_recal[storage_idx], sizeof(T)*M, memcpy_from_recal));

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