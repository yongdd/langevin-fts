/**
 * @file CudaComputationContinuous.cu
 * @brief CUDA propagator computation for continuous Gaussian chains.
 *
 * Orchestrates GPU-based propagator computation using multiple CUDA streams
 * for concurrent propagator advancement. Implements the continuous chain
 * model with Simpson's rule integration for concentration calculation.
 *
 * **Multi-Stream Architecture:**
 *
 * - Up to MAX_STREAMS concurrent propagator computations
 * - Each stream has dedicated cuFFT plans and workspace
 * - OpenMP threads map to CUDA streams for parallel scheduling
 *
 * **Propagator Storage:**
 *
 * - this->d_propagator[key][n]: Full propagator history on GPU
 * - All contour steps stored for concentration calculation
 * - Memory-intensive but maximum performance
 *
 * **Concentration Calculation:**
 *
 * Uses Simpson's rule for contour integration:
 * φ(r) = (ds/Q) * Σ_n w_n * q(r,n) * q†(r,N-n)
 * where w_n are Simpson coefficients (1, 4, 2, 4, ..., 1)/3
 *
 * **Supported Methods:**
 *
 * - "pseudospectral": FFT-based solver
 * - "realspace": Finite difference solver (beta)
 *
 * **Template Instantiations:**
 *
 * - CudaComputationContinuous<double>: Real field simulations
 * - CudaComputationContinuous<std::complex<double>>: L-FTS simulations
 *
 * @see CudaSolverPseudoRQM4 for propagator advancement
 * @see Scheduler for computation ordering
 */

#include <complex>
#include <cmath>
#include <omp.h>
#include <cuComplex.h>
#include <cufft.h>

#include "CudaComputationContinuous.h"
#include "CudaComputationBox.h"
#include "CudaSolverPseudoRQM4.h"
#include "CudaSolverPseudoRK2.h"
#include "CudaSolverPseudoETDRK4.h"
#include "CudaSolverCNADI.h"
#include "SimpsonRule.h"
#include "PropagatorCode.h"
#include "SpaceGroup.h"

template <typename T>
CudaComputationContinuous<T>::CudaComputationContinuous(
    ComputationBox<T>* cb,
    Molecules *molecules,
    PropagatorComputationOptimizer *propagator_computation_optimizer,
    std::string method,
    std::string numerical_method,
    SpaceGroup* space_group)
    : CudaComputationBase<T>(cb, molecules, propagator_computation_optimizer),
      d_full_to_reduced_map_(nullptr),
      d_reduced_basis_indices_(nullptr)
{
    try{
        #ifndef NDEBUG
        std::cout << "--------- Continuous Chain Solver, GPU Version ---------" << std::endl;
        #endif

        // Initialize reduced basis buffers to nullptr
        for (int i = 0; i < MAX_STREAMS; i++)
        {
            d_q_full_[i][0] = nullptr;
            d_q_full_[i][1] = nullptr;
        }

        // Set space group first so that get_n_basis() returns the correct size
        if (space_group != nullptr) {
            PropagatorComputation<T>::set_space_group(space_group);
            // Allocate full grid buffers for FFT operations
            const int M = this->cb->get_total_grid();
            for (int i = 0; i < MAX_STREAMS; i++)
            {
                gpu_error_check(cudaMalloc((void**)&d_q_full_[i][0], sizeof(T)*M));
                gpu_error_check(cudaMalloc((void**)&d_q_full_[i][1], sizeof(T)*M));
            }
            // Allocate and copy mapping arrays
            const int N_reduced = space_group->get_n_irreducible();
            gpu_error_check(cudaMalloc((void**)&d_reduced_basis_indices_, sizeof(int)*N_reduced));
            gpu_error_check(cudaMemcpy(d_reduced_basis_indices_, space_group->get_reduced_basis_indices().data(),
                                       sizeof(int)*N_reduced, cudaMemcpyHostToDevice));
            gpu_error_check(cudaMalloc((void**)&d_full_to_reduced_map_, sizeof(int)*M));
            gpu_error_check(cudaMemcpy(d_full_to_reduced_map_, space_group->get_full_to_reduced_map().data(),
                                       sizeof(int)*M, cudaMemcpyHostToDevice));
            // Set base class members for use in CudaComputationBase methods
            this->d_full_to_reduced_map_base_ = d_full_to_reduced_map_;
            this->d_reduced_basis_indices_base_ = d_reduced_basis_indices_;
            // Allocate full grid buffer for base class methods (expand for output)
            gpu_error_check(cudaMalloc((void**)&this->d_phi_full_buffer_, sizeof(T)*M));
        }

        const int N = this->cb->get_n_basis();  // n_irreducible (with space group) or total_grid

        // The number of parallel streams for propagator computation
        const char *ENV_OMP_NUM_THREADS = getenv("OMP_NUM_THREADS");
        std::string env_omp_num_threads(ENV_OMP_NUM_THREADS ? ENV_OMP_NUM_THREADS  : "");
        if (env_omp_num_threads.empty())
            this->n_streams = MAX_STREAMS;
        else
            this->n_streams =  std::min(std::stoi(env_omp_num_threads), MAX_STREAMS);
        #ifndef NDEBUG
        std::cout << "The number of CPU threads: " << this->n_streams << std::endl;
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
            
            this->propagator_size[key] = max_n_segment;
            this->d_propagator[key] = new CuDeviceData<T>*[max_n_segment];
            for(int i=0; i<this->propagator_size[key]; i++)
                gpu_error_check(cudaMalloc((void**)&this->d_propagator[key][i], sizeof(T)*N));

            #ifndef NDEBUG
            this->propagator_finished[key] = new bool[max_n_segment];
            for(int i=0; i<max_n_segment;i++)
                this->propagator_finished[key][i] = false;
            #endif
        }

        // Allocate memory for concentrations
        if( this->propagator_computation_optimizer->get_computation_blocks().size() == 0)
            throw_with_line_number("There is no block. Add polymers first.");
        for(const auto& item: this->propagator_computation_optimizer->get_computation_blocks())
        {
            this->d_phi_block[item.first] = nullptr;
            gpu_error_check(cudaMalloc((void**)&this->d_phi_block[item.first], sizeof(T)*N));
            gpu_error_check(cudaMemset(this->d_phi_block[item.first], 0, sizeof(T)*N));  // Zero-initialize
        }

        // Remember one segment for each polymer chain to compute total partition function
        int current_p = 0;
        for(const auto& block: this->d_phi_block)
        {
            const auto& key = block.first;
            int p                 = std::get<0>(key);
            std::string key_left  = std::get<1>(key);
            std::string key_right = std::get<2>(key);

            // Skip if already found one segment
            if (p != current_p)
                continue;

            int n_aggregated   = this->propagator_computation_optimizer->get_computation_block(key).v_u.size()/
                                 this->propagator_computation_optimizer->get_computation_block(key).n_repeated;
            int n_segment_left = this->propagator_computation_optimizer->get_computation_block(key).n_segment_left;

            single_partition_segment.push_back(std::make_tuple(
                p,
                this->d_propagator[key_left][n_segment_left],   // q
                this->d_propagator[key_right][0],               // q_dagger
                n_aggregated                              // how many propagators are aggregated
                ));
            current_p++;
        }

        // Concentrations for each solvent
        for(int s=0;s<this->molecules->get_n_solvent_types();s++)
        {
            CuDeviceData<T> *d_phi_;
            gpu_error_check(cudaMalloc((void**)&d_phi_, sizeof(T)*N));
            this->d_phi_solvent.push_back(d_phi_);
        }

        // Create scheduler for computation of propagator
        this->sc = new Scheduler(this->propagator_computation_optimizer->get_computation_propagators(), this->n_streams); 

        // Allocate memory for pseudo-spectral: advance_propagator()
        // These need full grid (M) for FFT operations
        const int M = this->cb->get_total_grid();
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

        // Copy mask to this->d_q_mask (in reduced basis if space_group is set)
        if (this->cb->get_mask() != nullptr)
        {
            gpu_error_check(cudaMalloc((void**)&this->d_q_mask, sizeof(double)*N));
            if (this->space_group_ != nullptr)
            {
                // Convert mask to reduced basis on host, then copy to device
                std::vector<double> mask_reduced(N);
                this->space_group_->to_reduced_basis(this->cb->get_mask(), mask_reduced.data(), 1);
                gpu_error_check(cudaMemcpy(this->d_q_mask, mask_reduced.data(), sizeof(double)*N, cudaMemcpyHostToDevice));
            }
            else
            {
                gpu_error_check(cudaMemcpy(this->d_q_mask, this->cb->get_mask(), sizeof(double)*N, cudaMemcpyHostToDevice));
            }
        }
        else
            this->d_q_mask = nullptr;
        gpu_error_check(cudaMalloc((void**)&this->d_phi, sizeof(T)*M));

        // Allocate memory for stress calculation: compute_stress()
        for(int i=0; i<this->n_streams; i++)
        {
            gpu_error_check(cudaMalloc((void**)&this->d_q_pair[i][0], sizeof(T)*2*M)); // prev
            gpu_error_check(cudaMalloc((void**)&this->d_q_pair[i][1], sizeof(T)*2*M)); // next
        }

        this->propagator_solver->update_laplacian_operator();

        // Set space group on solver for reduced basis expand/reduce
        if (space_group != nullptr) {
            this->propagator_solver->set_space_group(
                space_group,
                d_reduced_basis_indices_,
                d_full_to_reduced_map_,
                space_group->get_n_irreducible());
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
CudaComputationContinuous<T>::~CudaComputationContinuous()
{
    delete this->propagator_solver;
    delete this->sc;

    for(const auto& item: this->d_propagator)
    {
        for(int i=0; i<this->propagator_size[item.first]; i++)
            cudaFree(item.second[i]);
        delete[] item.second;
    }
    for(const auto& item: this->d_phi_block)
        cudaFree(item.second);
    for(const auto& item: this->d_phi_solvent)
        cudaFree(item);

    #ifndef NDEBUG
    for(const auto& item: this->propagator_finished)
        delete[] item.second;
    #endif

    cudaFree(this->d_phi);

    // For pseudo-spectral: advance_propagator()
    if (this->d_q_mask != nullptr)
        cudaFree(this->d_q_mask);
    cudaFree(this->d_q_unity);

    // For reduced basis
    if (d_full_to_reduced_map_ != nullptr)
        cudaFree(d_full_to_reduced_map_);
    if (d_reduced_basis_indices_ != nullptr)
        cudaFree(d_reduced_basis_indices_);
    if (this->d_phi_full_buffer_ != nullptr)
        cudaFree(this->d_phi_full_buffer_);
    for (int i = 0; i < MAX_STREAMS; i++)
    {
        if (d_q_full_[i][0] != nullptr)
            cudaFree(d_q_full_[i][0]);
        if (d_q_full_[i][1] != nullptr)
            cudaFree(d_q_full_[i][1]);
    }
    for (const auto& item : d_w)
        cudaFree(item.second);

    // For stress calculation: compute_stress()
    for(int i=0; i<this->n_streams; i++)
    {
        cudaFree(this->d_q_pair[i][0]);
        cudaFree(this->d_q_pair[i][1]);
    }

    // Destroy streams
    for(int i=0; i<this->n_streams; i++)
    {
        cudaStreamDestroy(this->streams[i][0]);
        cudaStreamDestroy(this->streams[i][1]);
    }
}
template <typename T>
void CudaComputationContinuous<T>::compute_statistics(
    std::map<std::string, const T*> w_input,
    std::map<std::string, const T*> q_init)
{
    this->compute_propagators(w_input, q_init);
    this->compute_concentrations();
}
template <typename T>
void CudaComputationContinuous<T>::compute_propagators(
    std::map<std::string, const T*> w_input,
    std::map<std::string, const T*> q_init)
{
    try{
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = this->cb->get_total_grid();
        const bool use_reduced_basis = (this->space_group_ != nullptr);
        const int N = use_reduced_basis ? this->space_group_->get_n_irreducible() : M;

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

        // Store w on device for solvent computation (reduced basis when space_group is set)
        for(const auto& item: w_input)
        {
            std::string monomer_type = item.first;
            const T* w = item.second;

            if (d_w.find(monomer_type) == d_w.end())
                gpu_error_check(cudaMalloc((void**)&d_w[monomer_type], sizeof(CuDeviceData<T>)*N));
            cudaMemcpy(d_w[monomer_type], w, sizeof(CuDeviceData<T>)*N, cudaMemcpyInputToDevice);
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
            #pragma omp parallel for num_threads(this->n_streams)
            for(size_t job=0; job<parallel_job->size(); job++)
            {
                const int STREAM = omp_get_thread_num();
                // printf("gpu, STREAM: %d, %d\n ", gpu, STREAM);

                auto& key = std::get<0>((*parallel_job)[job]);
                int n_segment_from = std::get<1>((*parallel_job)[job]);
                int n_segment_to = std::get<2>((*parallel_job)[job]);
                auto& deps = this->propagator_computation_optimizer->get_computation_propagator(key).deps;
                auto monomer_type = this->propagator_computation_optimizer->get_computation_propagator(key).monomer_type;

                // // Display job info
                // #ifndef NDEBUG
                // std::cout << job << " started" << std::endl;
                // #endif

                // Check key
                #ifndef NDEBUG
                if (this->d_propagator.find(key) == this->d_propagator.end())
                    std::cout << "Could not find key '" + key + "'. " << std::endl;
                #endif

                CuDeviceData<T> **_d_propagator = this->d_propagator[key];

                // If it is leaf node
                if(n_segment_from == 0 && deps.size() == 0)
                {
                    // q_init is in reduced basis when space_group is set, otherwise full grid
                    if (key[0] == '{')
                    {
                        std::string g = PropagatorCode::get_q_input_idx_from_key(key);
                        if (q_init.find(g) == q_init.end())
                            std::cout << "Could not find q_init[\"" + g + "\"]." << std::endl;
                        gpu_error_check(cudaMemcpyAsync(_d_propagator[0], q_init[g],
                            sizeof(T)*N, cudaMemcpyInputToDevice, this->streams[STREAM][0]));
                    }
                    else
                    {
                        // d_q_unity has all 1.0, copy first N elements (works for both full grid and reduced basis)
                        gpu_error_check(cudaMemcpyAsync(_d_propagator[0], this->d_q_unity,
                            sizeof(T)*N, cudaMemcpyDeviceToDevice, this->streams[STREAM][0]));
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
                        gpu_error_check(cudaMemsetAsync(_d_propagator[0], 0, sizeof(T)*N, this->streams[STREAM][0]));

                        // Add all propagators at junction if necessary
                        for(size_t d=0; d<deps.size(); d++)
                        {
                            std::string sub_dep = std::get<0>(deps[d]);
                            int sub_n_segment   = std::get<1>(deps[d]);
                            int sub_n_repeated  = std::get<2>(deps[d]);

                            // Check sub key
                            #ifndef NDEBUG
                            if (this->d_propagator.find(sub_dep) == this->d_propagator.end())
                                std::cout << "Could not find sub key '" + sub_dep + "'. " << std::endl;
                            if (!this->propagator_finished[sub_dep][sub_n_segment])
                                std::cout << "Could not compute '" + key +  "', since '"+ sub_dep + std::to_string(sub_n_segment) + "' is not prepared." << std::endl;
                            #endif

                            ker_lin_comb<<<N_BLOCKS, N_THREADS, 0, this->streams[STREAM][0]>>>(
                                _d_propagator[0], 1.0, _d_propagator[0],
                                sub_n_repeated, this->d_propagator[sub_dep][sub_n_segment], N);
                            gpu_error_check(cudaPeekAtLastError());
                        }

                        #ifndef NDEBUG
                        this->propagator_finished[key][0] = true;
                        #endif
                    }
                    else if(key[0] == '(')
                    {
                        // Initialize to one (d_q_unity is all 1.0, copy first N elements for both cases)
                        gpu_error_check(cudaMemcpyAsync(_d_propagator[0], this->d_q_unity,
                            sizeof(T)*N, cudaMemcpyDeviceToDevice, this->streams[STREAM][0]));

                        // Multiply all propagators at junction (in reduced basis or full grid)
                        for(size_t d=0; d<deps.size(); d++)
                        {
                            std::string sub_dep = std::get<0>(deps[d]);
                            int sub_n_segment   = std::get<1>(deps[d]);
                            int sub_n_repeated  = std::get<2>(deps[d]);

                            // Check sub key
                            #ifndef NDEBUG
                            if (this->d_propagator.find(sub_dep) == this->d_propagator.end())
                                std::cout << "Could not find sub key '" + sub_dep + "'. " << std::endl;
                            if (!this->propagator_finished[sub_dep][sub_n_segment])
                                std::cout << "Could not compute '" + key +  "', since '"+ sub_dep + std::to_string(sub_n_segment) + "' is not prepared." << std::endl;
                            #endif

                            for(int r=0; r<sub_n_repeated; r++)
                            {
                                ker_multi<<<N_BLOCKS, N_THREADS, 0, this->streams[STREAM][0]>>>(
                                    _d_propagator[0], _d_propagator[0],
                                    this->d_propagator[sub_dep][sub_n_segment], 1.0, N);
                                gpu_error_check(cudaPeekAtLastError());
                            }
                        }

                        #ifndef NDEBUG
                        this->propagator_finished[key][0] = true;
                        #endif
                    }
                }

                // Apply mask (already in reduced basis)
                if (n_segment_from == 0 && this->d_q_mask != nullptr)
                {
                    ker_multi<<<N_BLOCKS, N_THREADS, 0, this->streams[STREAM][0]>>>(
                        _d_propagator[0], _d_propagator[0], this->d_q_mask, 1.0, N);
                    gpu_error_check(cudaPeekAtLastError());
                }

                // Get ds_index from the propagator key
                int ds_index = PropagatorCode::get_ds_index_from_key(key);

                // Reset solver internal state when starting a new propagator
                // (needed for Global Richardson method)
                if (n_segment_from == 0)
                    this->propagator_solver->reset_internal_state(STREAM);

                for(int n=n_segment_from; n<n_segment_to; n++)
                {
                    #ifndef NDEBUG
                    if (!this->propagator_finished[key][n])
                        std::cout << "unfinished, key: " + key + ", " + std::to_string(n) << std::endl;
                    if (this->propagator_finished[key][n+1])
                        std::cout << "already finished: " + key + ", " + std::to_string(n+1) << std::endl;
                    #endif

                    // Solver handles expand/reduce internally when space_group is set
                    this->propagator_solver->advance_propagator(
                        STREAM,
                        _d_propagator[n],
                        _d_propagator[n+1],
                        monomer_type, this->d_q_mask, ds_index);

                    #ifndef NDEBUG
                    this->propagator_finished[key][n+1] = true;
                    #endif
                }

                gpu_error_check(cudaStreamSynchronize(this->streams[STREAM][0]));
                gpu_error_check(cudaStreamSynchronize(this->streams[STREAM][1]));

                // // Display job info
                // #ifndef NDEBUG
                // std::cout << job << " finished" << std::endl;
                // #endif
            }
            gpu_error_check(cudaDeviceSynchronize());
        }

        // Compute total partition function of each distinct polymers
        for(const auto& segment_info: single_partition_segment)
        {
            int p                 = std::get<0>(segment_info);
            CuDeviceData<T> *d_propagator_left  = std::get<1>(segment_info);
            CuDeviceData<T> *d_propagator_right = std::get<2>(segment_info);
            int n_aggregated      = std::get<3>(segment_info);

            // inner_product_device handles reduced-basis weighting internally.
            this->single_polymer_partitions[p] =
                dynamic_cast<CudaComputationBox<T>*>(this->cb)->inner_product_device(
                    d_propagator_left, d_propagator_right)
                /(n_aggregated*this->cb->get_volume());
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
void CudaComputationContinuous<T>::advance_propagator_single_segment(
    T* q_init, T *q_out, int p, int v, int u)
{
    try
    {
        // Get block info from polymer
        const Block& block = this->molecules->get_polymer(p).get_block(v, u);
        std::string monomer_type = block.monomer_type;

        // Get ds_index from ContourLengthMapping
        const ContourLengthMapping& mapping = this->molecules->get_contour_length_mapping();
        int ds_index = mapping.get_ds_index(block.contour_length);

        const int M = this->cb->get_total_grid();
        const int STREAM = 0;
        gpu_error_check(cudaMemcpy(this->d_q_pair[STREAM][0], q_init, sizeof(T)*M, cudaMemcpyHostToDevice));

        this->propagator_solver->advance_propagator(
                        STREAM, this->d_q_pair[STREAM][0], this->d_q_pair[STREAM][1],
                        monomer_type, this->d_q_mask, ds_index);
        gpu_error_check(cudaDeviceSynchronize());

        gpu_error_check(cudaMemcpy(q_out, this->d_q_pair[STREAM][1], sizeof(T)*M, cudaMemcpyDeviceToHost));
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
void CudaComputationContinuous<T>::compute_concentrations()
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();
        const int M = this->cb->get_total_grid();
        const bool use_reduced_basis = (this->space_group_ != nullptr);
        const int N_grid = use_reduced_basis ? this->space_group_->get_n_irreducible() : M;

        // Calculate segment concentrations
        for(const auto& d_block: this->d_phi_block)
        {
            const auto& key = d_block.first;
            int p                 = std::get<0>(key);
            std::string key_left  = std::get<1>(key);
            std::string key_right = std::get<2>(key);

            int n_segment_right = this->propagator_computation_optimizer->get_computation_block(key).n_segment_right;
            int n_segment_left  = this->propagator_computation_optimizer->get_computation_block(key).n_segment_left;
            int n_repeated = this->propagator_computation_optimizer->get_computation_block(key).n_repeated;

            // If there is no segment
            if(n_segment_right == 0)
            {
                gpu_error_check(cudaMemset(d_block.second, 0, sizeof(T)*N_grid));
                continue;
            }

            // Check keys
            #ifndef NDEBUG
            if (this->d_propagator.find(key_left) == this->d_propagator.end())
                throw_with_line_number("Could not find key_left key'" + key_left + "'. ");
            if (this->d_propagator.find(key_right) == this->d_propagator.end())
                throw_with_line_number("Could not find key_right key'" + key_right + "'. ");
            #endif

            // Calculate phi of one block (possibly multiple blocks when using aggregation)
            calculate_phi_one_block(
                d_block.second,           // phi
                this->d_propagator[key_left],   // dependency v
                this->d_propagator[key_right],  // dependency u
                n_segment_left,
                n_segment_right
            );

            // Get local_ds from ds_index encoded in key (DK+M format)
            Polymer& pc = this->molecules->get_polymer(p);
            int ds_index = PropagatorCode::get_ds_index_from_key(key_right);
            const ContourLengthMapping& mapping = this->molecules->get_contour_length_mapping();
            double local_ds = mapping.get_ds_from_index(ds_index);

            // Normalize concentration: local_ds * volume_fraction / alpha
            T _norm = (local_ds*pc.get_volume_fraction()/pc.get_alpha()*n_repeated)/this->single_polymer_partitions[p];
            CuDeviceData<T> norm;
            if constexpr (std::is_same<T, double>::value)
                norm = _norm;
            else
                norm = stdToCuDoubleComplex(_norm);
            ker_lin_comb<<<N_BLOCKS, N_THREADS>>>(d_block.second, norm, d_block.second, 0.0, d_block.second, N_grid);
            gpu_error_check(cudaPeekAtLastError());
        }

        // Calculate partition functions and concentrations of solvents
        double ds = this->molecules->get_contour_length_mapping().get_global_ds();
        for(int s=0; s<this->molecules->get_n_solvent_types(); s++)
        {
            CuDeviceData<T> *d_phi_ = this->d_phi_solvent[s];
            double volume_fraction   = std::get<0>(this->molecules->get_solvent(s));
            std::string monomer_type = std::get<1>(this->molecules->get_solvent(s));

            // Compute phi = exp(-w*ds)
            ker_exp<<<N_BLOCKS, N_THREADS>>>(d_phi_, d_w[monomer_type], 1.0, -ds, N_grid);
            gpu_error_check(cudaPeekAtLastError());

            // Partition function using mean_device
            this->single_solvent_partitions[s] = dynamic_cast<CudaComputationBox<T>*>(this->cb)->mean_device(d_phi_);

            // Normalize: phi *= volume_fraction / partition
            CuDeviceData<T> norm;
            if constexpr (std::is_same<T, double>::value)
                norm = volume_fraction / this->single_solvent_partitions[s];
            else
                norm = cuCdiv(make_cuDoubleComplex(volume_fraction, 0.0), stdToCuDoubleComplex(this->single_solvent_partitions[s]));

            ker_lin_comb<<<N_BLOCKS, N_THREADS>>>(d_phi_, norm, d_phi_, 0.0, d_phi_, N_grid);
            gpu_error_check(cudaPeekAtLastError());
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
void CudaComputationContinuous<T>::calculate_phi_one_block(
    CuDeviceData<T> *d_phi, CuDeviceData<T> **d_q_1, CuDeviceData<T> **d_q_2, const int N_LEFT, const int N_RIGHT)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = this->cb->get_total_grid();
        const bool use_reduced_basis = (this->space_group_ != nullptr);
        const int N_grid = use_reduced_basis ? this->space_group_->get_n_irreducible() : M;

        std::vector<double> simpson_rule_coeff = SimpsonRule::get_coeff(N_RIGHT);

        // Compute segment concentration
        ker_multi<<<N_BLOCKS, N_THREADS>>>(d_phi, d_q_1[N_LEFT], d_q_2[0], simpson_rule_coeff[0], N_grid);
        for(int n=1; n<=N_RIGHT; n++)
        {
            ker_add_multi<<<N_BLOCKS, N_THREADS>>>(d_phi, d_q_1[N_LEFT-n], d_q_2[n], simpson_rule_coeff[n], N_grid);
        }
        gpu_error_check(cudaPeekAtLastError());
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
void CudaComputationContinuous<T>::compute_stress()
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

        // Check for non-periodic BC - stress computation not supported
        auto bc_vec = this->cb->get_boundary_conditions();
        for (const auto& bc : bc_vec)
        {
            if (bc != BoundaryCondition::PERIODIC)
            {
                throw_with_line_number("Stress computation with non-periodic boundary conditions "
                    "is not supported yet. Use periodic boundary conditions.");
            }
        }

        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int DIM = this->cb->get_dim();
        const int M   = this->cb->get_total_grid();
        const int N_grid = this->cb->get_n_basis();

        const int N_STRESS = 6;
        std::map<std::tuple<int, std::string, std::string>, std::array<T,N_STRESS>> block_dq_dl[this->n_streams];

        // Reset stress map
        for(const auto& item: this->d_phi_block)
        {
            for(int i=0; i<this->n_streams; i++)
                for(int d=0; d<N_STRESS; d++)
                    block_dq_dl[i][item.first][d] = 0.0;
        }

        // Compute stress for each block
        #pragma omp parallel for num_threads(this->n_streams)
        for(size_t b=0; b<this->d_phi_block.size();b++)
        {
            const int STREAM = omp_get_thread_num();

            auto block = this->d_phi_block.begin();
            advance(block, b);
            const auto& key   = block->first;

            // printf("start, b, gpu, STREAM: %2d, %2d, %2d\n", b, gpu, STREAM);

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

            // std::cout << p << ", " << key_left << ", " << key_right << ", " << N << ", " << N_LEFT << std::endl;

            std::vector<double> s_coeff = SimpsonRule::get_coeff(N_RIGHT);
            CuDeviceData<T>** d_q_1 = this->d_propagator[key_left];     // dependency v
            CuDeviceData<T>** d_q_2 = this->d_propagator[key_right];    // dependency u

            // Check if propagators at endpoints are leaf nodes (initial conditions)
            bool left_is_leaf = (this->propagator_computation_optimizer->get_computation_propagator(key_left).deps.size() == 0);
            bool right_is_leaf = (this->propagator_computation_optimizer->get_computation_propagator(key_right).deps.size() == 0);

            std::array<T,N_STRESS> _block_dq_dl;
            for(int i=0; i<N_STRESS; i++)
                _block_dq_dl[i] = 0.0;

            // Number of stress components: 3D orthogonal->3, 3D non-orthogonal->6, 2D->3, 1D->1
            const bool is_ortho = this->cb->is_orthogonal();
            const int N_STRESS_DIM = (DIM == 3) ? (is_ortho ? 3 : 6) : ((DIM == 2) ? 3 : 1);
            CuDeviceData<T> *d_segment_stress;
            T segment_stress[6];  // Max size to avoid VLA issues
            gpu_error_check(cudaMalloc((void**)&d_segment_stress, sizeof(T)*N_STRESS_DIM));

            int prev, next;
            prev = 0;
            next = 1;

            // Create events
            cudaEvent_t kernel_done;
            cudaEvent_t memcpy_done;
            gpu_error_check(cudaEventCreate(&kernel_done));
            gpu_error_check(cudaEventCreate(&memcpy_done));

            gpu_error_check(cudaMemcpyAsync(&this->d_q_pair[STREAM][prev][0], d_q_1[N_LEFT],
                    sizeof(T)*N_grid, cudaMemcpyDeviceToDevice, this->streams[STREAM][1]));
            gpu_error_check(cudaMemcpyAsync(&this->d_q_pair[STREAM][prev][N_grid], d_q_2[0],
                    sizeof(T)*N_grid, cudaMemcpyDeviceToDevice, this->streams[STREAM][1]));

            gpu_error_check(cudaEventRecord(memcpy_done, this->streams[STREAM][1]));
            gpu_error_check(cudaStreamWaitEvent(this->streams[STREAM][0], memcpy_done, 0));

            for(int n=0; n<=N_RIGHT; n++)
            {
                // STREAM 1: Copy data
                if (n+1 <= N_RIGHT)
                {
                    gpu_error_check(cudaMemcpyAsync(&this->d_q_pair[STREAM][next][0], d_q_1[N_LEFT-n-1],
                            sizeof(T)*N_grid, cudaMemcpyDeviceToDevice, this->streams[STREAM][1]));
                    gpu_error_check(cudaMemcpyAsync(&this->d_q_pair[STREAM][next][N_grid], d_q_2[n+1],
                            sizeof(T)*N_grid, cudaMemcpyDeviceToDevice, this->streams[STREAM][1]));
                    gpu_error_check(cudaEventRecord(memcpy_done, this->streams[STREAM][1]));
                }

                // STREAM 0: Compute stress
                this->propagator_solver->compute_single_segment_stress(
                    STREAM, this->d_q_pair[STREAM][prev], d_segment_stress,
                    monomer_type, false);   
                gpu_error_check(cudaEventRecord(kernel_done, this->streams[STREAM][0]));

                // Wait until computation and memory copy are done
                gpu_error_check(cudaStreamWaitEvent(this->streams[STREAM][1], kernel_done, 0));
                gpu_error_check(cudaStreamWaitEvent(this->streams[STREAM][0], memcpy_done, 0));

                gpu_error_check(cudaMemcpy(segment_stress, d_segment_stress, sizeof(T)*N_STRESS_DIM, cudaMemcpyDeviceToHost));

                // Skip endpoint terms where propagator is initial condition (q=1)
                // For non-periodic BC (DCT/DST), the initial condition q=1 doesn't transform
                // correctly, causing errors in stress computation.
                bool skip_this_n = false;
                if (n == 0 && right_is_leaf)
                    skip_this_n = true;
                if (n == N_RIGHT && left_is_leaf && N_LEFT == N_RIGHT)
                    skip_this_n = true;

                if (!skip_this_n)
                {
                    for(int d=0; d<N_STRESS_DIM; d++)
                        _block_dq_dl[d] += segment_stress[d]*(s_coeff[n]*n_repeated);
                }

                // std::cout << key_left << ", "  << key_right << ", " << n << ", " << segment_stress[0] << ", " << segment_stress[1] << ", " << segment_stress[2] << std::endl;

                std::swap(prev, next);
            }
            gpu_error_check(cudaStreamSynchronize(this->streams[STREAM][0]));
            gpu_error_check(cudaStreamSynchronize(this->streams[STREAM][1]));
            gpu_error_check(cudaEventDestroy(kernel_done));
            gpu_error_check(cudaEventDestroy(memcpy_done));

            // std::cout << p << ", " << key_left << ", " << key_right << ", " << N << ", " << N_LEFT << std::endl;
            // std::cout << "STREAM, _block_dq_dl[0] " << STREAM  << ", " << _block_dq_dl[0] << std::endl;

            // Multiply by local_ds for this block (get ds_index from key)
            int ds_index = PropagatorCode::get_ds_index_from_key(key_right);
            const ContourLengthMapping& mapping_stress = this->molecules->get_contour_length_mapping();
            double local_ds = mapping_stress.get_ds_from_index(ds_index);
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
        for(const auto& d_block: this->d_phi_block)
        {
            const auto& key       = d_block.first;
            int p                 = std::get<0>(key);
            std::string key_left  = std::get<1>(key);
            std::string key_right = std::get<2>(key);

            for(int i=0; i<this->n_streams; i++)
                for(int d=0; d<N_STRESS_TOTAL; d++)
                    this->dq_dl[p][d] += block_dq_dl[i][key][d];
        }
        // ============ DEFORMATION VECTOR APPROACH ============
        // The Pseudo class now computes v⊗v components directly, where
        // v = 2π g⁻¹ m is the deformation vector (units: 1/L²).
        //
        // The accumulated sums are already in the deformation vector basis:
        //   V₁₁ = Σ(kernel × v₁²), V₂₂ = Σ(kernel × v₂²), etc.
        //
        // The metric tensor g = hᵀh has simple derivatives:
        //   g₁₁ = L₁², g₂₂ = L₂², g₃₃ = L₃²
        //   g₁₂ = L₁L₂cosγ, g₁₃ = L₁L₃cosβ, g₂₃ = L₂L₃cosα
        //
        // Lattice parameter derivatives:
        //   ∂H/∂L₁ ∝ L₁V₁₁ + L₂cosγ·V₁₂ + L₃cosβ·V₁₃
        //   ∂H/∂γ  ∝ -L₁L₂sinγ·V₁₂
        //
        // @see docs/StressTensorCalculation.md for derivation

        // Get lattice parameters
        double L1 = this->cb->get_lx(0);
        double L2 = (DIM >= 2) ? this->cb->get_lx(1) : 1.0;
        double L3 = (DIM >= 3) ? this->cb->get_lx(2) : 1.0;

        // Get angles (radians)
        std::vector<double> angles = this->cb->get_angles();
        double cos_a = std::cos(angles[0]);  // alpha: between b and c
        double cos_b = std::cos(angles[1]);  // beta: between a and c
        double cos_g = std::cos(angles[2]);  // gamma: between a and b
        double sin_a = std::sin(angles[0]);
        double sin_b = std::sin(angles[1]);
        double sin_g = std::sin(angles[2]);

        // Normalization factor (from Boltzmann factor derivative)
        // Note: local_ds is already multiplied per-block in the stress loop above
        double norm = -3.0 * M * M;

        for(int p=0; p<n_polymer_types; p++)
        {
            // Get v⊗v sums (already in deformation vector basis from Pseudo)
            T V_11 = this->dq_dl[p][0];
            T V_22 = (DIM >= 2) ? this->dq_dl[p][1] : T(0.0);
            T V_33 = T(0.0), V_12 = T(0.0), V_13 = T(0.0), V_23 = T(0.0);

            if (DIM == 3) {
                V_33 = this->dq_dl[p][2];
                V_12 = this->dq_dl[p][3];
                V_13 = this->dq_dl[p][4];
                V_23 = this->dq_dl[p][5];
            } else if (DIM == 2) {
                V_12 = this->dq_dl[p][2];
            }

            // Compute lattice parameter derivatives using metric tensor formulas
            // ∂g/∂L₁: ∂g₁₁/∂L₁ = 2L₁, ∂g₁₂/∂L₁ = L₂cosγ, ∂g₁₃/∂L₁ = L₃cosβ
            // ∂H/∂L₁ = Σᵢⱼ Vᵢⱼ ∂gᵢⱼ/∂L₁ (factor of 2 from diagonal is absorbed in normalization)
            this->dq_dl[p][0] = (L1*V_11 + L2*cos_g*V_12 + L3*cos_b*V_13) / norm;
            if (DIM >= 2) {
                this->dq_dl[p][1] = (L2*V_22 + L1*cos_g*V_12 + L3*cos_a*V_23) / norm;
            }
            if (DIM >= 3) {
                this->dq_dl[p][2] = (L3*V_33 + L1*cos_b*V_13 + L2*cos_a*V_23) / norm;
            }

            // Compute angle derivatives using metric tensor formulas
            // ∂g₁₂/∂γ = -L₁L₂sinγ → ∂H/∂γ = -L₁L₂sinγ·V₁₂
            // ∂g₁₃/∂β = -L₁L₃sinβ → ∂H/∂β = -L₁L₃sinβ·V₁₃
            // ∂g₂₃/∂α = -L₂L₃sinα → ∂H/∂α = -L₂L₃sinα·V₂₃
            if (DIM == 3) {
                this->dq_dl[p][3] = -L1 * L2 * sin_g * V_12 / norm;  // dH/dγ
                this->dq_dl[p][4] = -L1 * L3 * sin_b * V_13 / norm;  // dH/dβ
                this->dq_dl[p][5] = -L2 * L3 * sin_a * V_23 / norm;  // dH/dα
            } else if (DIM == 2) {
                this->dq_dl[p][2] = -L1 * L2 * sin_g * V_12 / norm;  // dH/dγ
            }
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
void CudaComputationContinuous<T>::get_chain_propagator(T *q_out, int polymer, int v, int u, int n)
{
    // This method should be invoked after invoking compute_statistics()

    // Get chain propagator for a selected polymer, block and direction.
    // This is made for debugging and testing.
    try
    {
        const int N = this->cb->get_n_basis();  // n_irreducible (with space group) or total_grid

        Polymer& pc = this->molecules->get_polymer(polymer);
        std::string dep = pc.get_propagator_key(v,u);

        if (this->propagator_computation_optimizer->get_computation_propagators().find(dep) == this->propagator_computation_optimizer->get_computation_propagators().end())
            throw_with_line_number("Could not find the propagator code '" + dep + "'. Disable 'aggregation' option to obtain propagator_computation_optimizer.");

        const int N_RIGHT = this->propagator_computation_optimizer->get_computation_propagator(dep).max_n_segment;
        if (n < 0 || n > N_RIGHT)
            throw_with_line_number("n (" + std::to_string(n) + ") must be in range [0, " + std::to_string(N_RIGHT) + "]");

        gpu_error_check(cudaMemcpy(q_out, this->d_propagator[dep][n], sizeof(T)*N, cudaMemcpyDeviceToHost));
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
bool CudaComputationContinuous<T>::check_total_partition()
{
    const int M = this->cb->get_total_grid();
    int n_polymer_types = this->molecules->get_n_polymer_types();
    std::vector<std::vector<T>> total_partitions;
    for(int p=0;p<n_polymer_types;p++)
    {
        std::vector<T> total_partitions_p;
        total_partitions.push_back(total_partitions_p);
    }
    for(const auto& block: this->d_phi_block)
    {
        const auto& key = block.first;
        int p                 = std::get<0>(key);
        std::string key_left  = std::get<1>(key);
        std::string key_right = std::get<2>(key);

        int n_segment_right = this->propagator_computation_optimizer->get_computation_block(key).n_segment_right;
        int n_segment_left  = this->propagator_computation_optimizer->get_computation_block(key).n_segment_left;
        int n_repeated      = this->propagator_computation_optimizer->get_computation_block(key).n_repeated;
        int n_propagators   = this->propagator_computation_optimizer->get_computation_block(key).v_u.size();

        #ifndef NDEBUG
        std::cout<< p << ", " << key_left << ", " << key_right << ": " << n_segment_left << ", " << n_segment_right << ", " << n_propagators << ", " << this->propagator_computation_optimizer->get_computation_block(key).n_repeated << std::endl;
        #endif

        for(int n=0;n<=n_segment_right;n++)
        {
            T total_partition = dynamic_cast<CudaComputationBox<T>*>(this->cb)->inner_product_device(
                this->d_propagator[key_left][n_segment_left-n],
                this->d_propagator[key_right][n]);

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
        if (diff_partition > PARTITION_TOLERANCE)
            return false;
    }
    return true;
}

// Explicit template instantiation
template class CudaComputationContinuous<double>;
template class CudaComputationContinuous<std::complex<double>>;
