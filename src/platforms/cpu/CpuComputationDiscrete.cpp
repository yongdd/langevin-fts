/**
 * @file CpuComputationDiscrete.cpp
 * @brief CPU implementation of propagator computation for discrete chains.
 *
 * Implements the PropagatorComputation interface for freely-jointed
 * (discrete) chain models using OpenMP parallelization. Discrete chains
 * require half-bond steps at junction points.
 *
 * **Discrete Chain Model:**
 *
 * Unlike continuous chains, discrete chains have:
 * - Full segments at regular positions
 * - Half-bond steps at junction points
 * - Different concentration calculation (no Simpson's rule)
 *
 * **Memory Layout:**
 *
 * - this->propagator[key]: Full segment propagators q(r,n)
 * - propagator_half_steps[key]: Half-bond step values at junctions
 *
 * **Parallelization:**
 *
 * Uses OpenMP for parallel propagator computation across independent
 * branches of the polymer graph.
 *
 * **Template Instantiations:**
 *
 * - CpuComputationDiscrete<double>: Real fields
 * - CpuComputationDiscrete<std::complex<double>>: Complex fields
 *
 * @see CpuSolverPseudoDiscrete for pseudo-spectral solver
 */

#include <cmath>
#include <numbers>
#include <chrono>

#include "CpuComputationDiscrete.h"
#include "CpuSolverPseudoDiscrete.h"
#include "SimpsonRule.h"

/**
 * @brief Construct CPU propagator computation for discrete chains.
 *
 * Allocates propagator arrays including half-step storage at junctions.
 * Sets up parallel execution scheduler.
 *
 * @param cb                             Computation box for grid operations
 * @param molecules                      Polymer/solvent species definitions
 * @param propagator_computation_optimizer Optimized computation schedule
 */
template <typename T>
CpuComputationDiscrete<T>::CpuComputationDiscrete(
    ComputationBox<T>* cb,
    Molecules *molecules,
    PropagatorComputationOptimizer *propagator_computation_optimizer,
    FFTBackend backend,
    SpaceGroup* space_group)
    : CpuComputationBase<T>(cb, molecules, propagator_computation_optimizer)
{
    try
    {
        #ifndef NDEBUG
        std::cout << "--------- Discrete Chain Solver, CPU Version ---------" << std::endl;
        #endif

        // Set space group first so that get_n_basis() returns the correct size
        if (space_group != nullptr)
            PropagatorComputation<T>::set_space_group(space_group);

        const int N = this->cb->get_n_basis();  // n_irreducible (with space group) or total_grid
        this->propagator_solver = new CpuSolverPseudoDiscrete<T>(cb, molecules, backend);

        // Set space group on solver for internal expand/reduce handling
        if (space_group != nullptr)
            this->propagator_solver->set_space_group(space_group);

        // The number of parallel streams for propagator computation
        const char *ENV_OMP_NUM_THREADS = getenv("OMP_NUM_THREADS");
        std::string env_omp_num_threads(ENV_OMP_NUM_THREADS ? ENV_OMP_NUM_THREADS  : "");
        if (env_omp_num_threads.empty())
            this->n_streams = 4;
        else
            this->n_streams = std::stoi(env_omp_num_threads);
        #ifndef NDEBUG
        std::cout << "The number of CPU threads: " << this->n_streams << std::endl;
        #endif

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
            this->propagator_size[key] = max_n_segment;

            // Allocate memory for q(r,1/2)
            propagator_half_steps[key] = new T*[max_n_segment];
            if (item.second.deps.size() > 0)
                propagator_half_steps[key][0] = new T[N];
            else
                propagator_half_steps[key][0] = nullptr;

            // Allocate memory for q(r,s+1/2)
            for(int i=1; i<this->propagator_size[key]; i++)
            {
                if (!item.second.junction_ends.contains(i))
                    propagator_half_steps[key][i] = nullptr;
                else
                    propagator_half_steps[key][i] = new T[N];
            }

            // Allocate memory for q(r,s)
            // Index 0 will be not used
            this->propagator[key] = new T*[max_n_segment];
            this->propagator[key][0] = nullptr;
            for(int i=1; i<this->propagator_size[key]; i++)
                this->propagator[key][i] = new T[N];

            #ifndef NDEBUG
            this->propagator_finished[key] = new bool[max_n_segment];
            for(int i=0; i<max_n_segment;i++)
                this->propagator_finished[key][i] = false;
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
            this->phi_block[item.first] = new T[N]();  // Zero-initialize
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
            std::string monomer_type = this->propagator_computation_optimizer->get_computation_block(key).monomer_type;

            // Skip if n_segment_left is 0
            if (n_segment_left == 0)
                continue;

            single_partition_segment.push_back(std::make_tuple(
                p,
                this->propagator[key_left][n_segment_left],    // q
                this->propagator[key_right][1],                  // q_dagger
                monomer_type,       
                n_aggregated                               // how many propagators are aggregated
                ));
            current_p++;
        }

        // Concentrations for each solvent
        for(int s=0;s<this->molecules->get_n_solvent_types();s++)
            this->phi_solvent.push_back(new T[N]);

        // Create scheduler for computation of propagator
        this->sc = new Scheduler(this->propagator_computation_optimizer->get_computation_propagators(), this->n_streams);

        this->update_laplacian_operator();
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
CpuComputationDiscrete<T>::~CpuComputationDiscrete()
{
    delete this->propagator_solver;
    delete this->sc;

    for(const auto& item: this->propagator)
    {
        for(int i=0; i<this->propagator_size[item.first]; i++)
        {
            if(item.second[i] != nullptr)
                delete[] item.second[i];
        }
        delete[] item.second;
    }
    for(const auto& item: propagator_half_steps)
    {
        for(int i=0; i<this->propagator_size[item.first]; i++)
        {
            if(item.second[i] != nullptr)
                delete[] item.second[i];
        }
        delete[] item.second;
    }
    
    for(const auto& item: this->phi_block)
        delete[] item.second;
    for(const auto& item: this->phi_solvent)
        delete[] item; 

    #ifndef NDEBUG
    for(const auto& item: this->propagator_finished)
        delete[] item.second;
    #endif
}
template <typename T>
void CpuComputationDiscrete<T>::compute_statistics(
    std::map<std::string, const T*> w_input,
    std::map<std::string, const T*> q_init)
{
    this->compute_propagators(w_input, q_init);
    this->compute_concentrations();
}
template <typename T>
void CpuComputationDiscrete<T>::compute_propagators(
    std::map<std::string, const T*> w_input,
    std::map<std::string, const T*> q_init)
{
    try
    {
        const int M = this->cb->get_total_grid();
        const int N = this->cb->get_n_basis();  // n_irreducible (with space group) or total_grid
        const bool use_reduced_basis = (this->space_group_ != nullptr);

        for(const auto& item: this->propagator_computation_optimizer->get_computation_propagators())
        {
            if( !w_input.contains(item.second.monomer_type))
                throw_with_line_number("monomer_type \"" + item.second.monomer_type + "\" is not in w_input.");
        }

        #ifndef NDEBUG
        this->time_complexity = 0;
        #endif

        // Update dw or exp_dw (w_input is always on full grid)
        this->propagator_solver->update_dw(w_input);

        // Compute exp_dw in reduced basis for later use in compute_concentrations
        if (use_reduced_basis)
        {
            for (const auto& [monomer_type, exp_dw_full] : this->propagator_solver->exp_dw[0])
            {
                if (exp_dw_reduced_.find(monomer_type) == exp_dw_reduced_.end())
                    exp_dw_reduced_[monomer_type].resize(N);
                if constexpr (std::is_same_v<T, double>)
                    this->space_group_->to_reduced_basis(exp_dw_full.data(), exp_dw_reduced_[monomer_type].data(), 1);
            }
        }

        // For each time span
        #ifndef NDEBUG
        for(const auto& item: this->propagator_computation_optimizer->get_computation_propagators())
        {
            std::string key = item.first;
            int max_n_segment = item.second.max_n_segment+1;
            for(int i=0; i<max_n_segment;i++)
                this->propagator_finished[key][i] = false;
            for (int n: item.second.junction_ends)
                propagator_half_steps_finished[key][n] = false;
            propagator_half_steps_finished[key][0] = false;
        }
        #endif

        auto& branch_schedule = this->sc->get_schedule();
        for (auto parallel_job = branch_schedule.begin(); parallel_job != branch_schedule.end(); parallel_job++)
        {
            // display all jobs
            #ifndef NDEBUG
            std::cout << "jobs:" << std::endl;
            for(size_t job=0; job<parallel_job->size(); job++)
            {
                auto& key = std::get<0>((*parallel_job)[job]);
                int n_segment_from = std::get<1>((*parallel_job)[job]);
                int n_segment_to = std::get<2>((*parallel_job)[job]);
                std::cout << "key, n_segment_from, n_segment_to: " + key + ", " + std::to_string(n_segment_from) + ", " + std::to_string(n_segment_to) + ". " << std::endl;
                std::cout << "half_steps: ";
                std::cout << "{";
                for (int i=0; i<this->propagator_size[key]; i++)
                {
                    if (propagator_half_steps[key][i] != nullptr)
                        std::cout << i << ", ";
                }
                std::cout << "}, "<< std::endl;
            }
            auto start_time = std::chrono::duration_cast<std::chrono::microseconds>
                (std::chrono::system_clock::now().time_since_epoch()).count();
            #endif

            // For each propagator
            #pragma omp parallel for num_threads(this->n_streams)
            for(size_t job=0; job<parallel_job->size(); job++)
            {
                auto& key = std::get<0>((*parallel_job)[job]);
                int n_segment_from = std::get<1>((*parallel_job)[job]);
                int n_segment_to = std::get<2>((*parallel_job)[job]);
                auto& deps = this->propagator_computation_optimizer->get_computation_propagator(key).deps;
                auto monomer_type = this->propagator_computation_optimizer->get_computation_propagator(key).monomer_type;

                #ifndef NDEBUG
                #pragma omp critical
                std::cout << job << " started, " <<
                    std::chrono::duration_cast<std::chrono::microseconds>
                    (std::chrono::system_clock::now().time_since_epoch()).count() - start_time << std::endl;
                #endif

                // Check key
                #ifndef NDEBUG
                if (!this->propagator.contains(key))
                    std::cout << "Could not find key '" << key << "'. " << std::endl;
                #endif

                T **_propagator = this->propagator[key];
                const T *_exp_dw = this->propagator_solver->exp_dw[0][monomer_type].data();

                // Thread-local buffers for expand/reduce operations when using space group
                thread_local std::vector<T> q_full_in_local;
                thread_local std::vector<T> q_full_out_local;
                thread_local std::vector<T> q_full_work_local;
                if (use_reduced_basis && q_full_in_local.size() < static_cast<size_t>(M)) {
                    q_full_in_local.resize(M);
                    q_full_out_local.resize(M);
                    q_full_work_local.resize(M);
                }

                // Calculate one block end
                if (n_segment_from == 0 && deps.size() == 0) // if it is leaf node
                {
                    #ifndef NDEBUG
                    #pragma omp critical
                    std::cout << job << " init 1, " <<
                        std::chrono::duration_cast<std::chrono::microseconds>
                        (std::chrono::system_clock::now().time_since_epoch()).count() - start_time << std::endl;
                    #endif

                    // q_init is in reduced basis when space_group is set, otherwise full grid
                    const T* _exp_dw_effective = use_reduced_basis
                        ? exp_dw_reduced_[monomer_type].data() : _exp_dw;

                    if (key[0] == '{')
                    {
                        std::string g = PropagatorCode::get_q_input_idx_from_key(key);
                        if (!q_init.contains(g))
                            throw_with_line_number("Could not find q_init[\"" + g + "\"]. Pass q_init to run() for grafted polymers.");
                        for(int i=0; i<N; i++)
                            _propagator[1][i] = q_init[g][i] * _exp_dw_effective[i];
                    }
                    else
                    {
                        for(int i=0; i<N; i++)
                            _propagator[1][i] = _exp_dw_effective[i];
                    }

                    #ifndef NDEBUG
                    this->propagator_finished[key][1] = true;
                    #endif
                }
                else if (n_segment_from == 0 && deps.size() > 0) // if it is not leaf node
                {
                    // If it is aggregated
                    if (key[0] == '[')
                    {
                        // Sum propagators in reduced basis (or full grid if no space_group)
                        // Solver handles expand/FFT/reduce internally
                        for(int i=0; i<N; i++)
                            _propagator[1][i] = 0.0;

                        for(size_t d=0; d<deps.size(); d++)
                        {
                            std::string sub_dep = std::get<0>(deps[d]);
                            int sub_n_segment   = std::get<1>(deps[d]);
                            int sub_n_repeated  = std::get<2>(deps[d]);
                            T **_propagator_sub_dep;

                            if (sub_n_segment == 0)
                            {
                                #ifndef NDEBUG
                                if (!propagator_half_steps.contains(sub_dep))
                                    std::cout << "Could not find sub key '" + sub_dep + "'. " << std::endl;
                                if (!propagator_half_steps_finished[sub_dep][0])
                                    std::cout << "Could not compute '" + key +  "', since '"+ sub_dep + std::to_string(0) + "+1/2' is not prepared." << std::endl;
                                #endif
                                _propagator_sub_dep = propagator_half_steps[sub_dep];
                            }
                            else
                            {
                                #ifndef NDEBUG
                                if (!this->propagator.contains(sub_dep))
                                    std::cout << "Could not find sub key '" + sub_dep + "'. " << std::endl;
                                if (!this->propagator_finished[sub_dep][sub_n_segment])
                                    std::cout << "Could not compute '" + key +  "', since '"+ sub_dep + std::to_string(sub_n_segment) + "' is not prepared." << std::endl;
                                #endif
                                _propagator_sub_dep = this->propagator[sub_dep];
                            }
                            for(int i=0; i<N; i++)
                                _propagator[1][i] += _propagator_sub_dep[sub_n_segment][i] * static_cast<double>(sub_n_repeated);
                        }

                        #ifndef NDEBUG
                        #pragma omp critical
                        this->time_complexity++;
                        #endif

                        // Throw an error message if zero and nonzero sub_n_segments are mixed
                        #ifndef NDEBUG
                        #pragma omp critical
                        for(size_t d=1; d<deps.size(); d++)
                        {
                            if((std::get<1>(deps[d-1]) != 0 && std::get<1>(deps[d]) == 0) ||
                               (std::get<1>(deps[d-1]) == 0 && std::get<1>(deps[d]) != 0))
                                std::cout << "Zero and nonzero sub_n_segments are mixed." << std::endl;
                        }
                        #endif

                        // If n_segments of all deps are 0
                        if (std::get<1>(deps[0]) == 0)
                        {
                            T *_propagator_half_step = propagator_half_steps[key][0];
                            for(int i=0; i<N; i++)
                                _propagator_half_step[i] = _propagator[1][i];

                            // Add half bond (solver handles expand/FFT/reduce)
                            this->propagator_solver->advance_propagator_half_bond_step(
                                _propagator[1], _propagator[1], monomer_type);

                            // Add full segment (in reduced basis)
                            for(int i=0; i<N; i++)
                                _propagator[1][i] *= _exp_dw[i];
                        }
                        else
                        {
                            // Discrete chains always use ds_index=0 (global ds)
                            // Solver handles expand/FFT/reduce
                            this->propagator_solver->advance_propagator(
                                _propagator[1],
                                _propagator[1],
                                monomer_type,
                                this->cb->get_mask(), 0);
                        }

                        #ifndef NDEBUG
                        this->propagator_finished[key][1] = true;
                        #endif
                    }
                    else if(key[0] == '(')
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

                        #ifndef NDEBUG
                        #pragma omp critical
                        std::cout << job << " init 3, " <<
                            std::chrono::duration_cast<std::chrono::microseconds>
                            (std::chrono::system_clock::now().time_since_epoch()).count() - start_time << std::endl;
                        #endif

                        // Junction case: multiply propagators at junction point
                        // This can be done directly in reduced basis since symmetric fields' products are also symmetric
                        T *_q_junction_start = propagator_half_steps[key][0];

                        for(int i=0; i<N; i++)
                            _q_junction_start[i] = 1.0;

                        for(size_t d=0; d<deps.size(); d++)
                        {
                            std::string sub_dep = std::get<0>(deps[d]);
                            int sub_n_segment   = std::get<1>(deps[d]);
                            int sub_n_repeated  = std::get<2>(deps[d]);

                            #ifndef NDEBUG
                            if (!propagator_half_steps_finished[sub_dep][sub_n_segment])
                                std::cout << "Could not compute '" + key +  "', since '"+ sub_dep + std::to_string(sub_n_segment) + "+1/2' is not prepared." << std::endl;
                            #endif

                            T *_propagator_half_step = propagator_half_steps[sub_dep][sub_n_segment];
                            for(int i=0; i<N; i++)
                                _q_junction_start[i] *= pow(_propagator_half_step[i], sub_n_repeated);
                        }

                        #ifndef NDEBUG
                        propagator_half_steps_finished[key][0] = true;
                        #endif

                        if (n_segment_to > 0)
                        {
                            #ifndef NDEBUG
                            #pragma omp critical
                            this->time_complexity++;
                            #endif

                            // Add half bond (solver handles expand/reduce internally)
                            this->propagator_solver->advance_propagator_half_bond_step(
                                _q_junction_start, _propagator[1], monomer_type);

                            // Add full segment (apply exp_dw in appropriate basis)
                            if (use_reduced_basis)
                            {
                                const T* _exp_dw_reduced = exp_dw_reduced_[monomer_type].data();
                                for(int i=0; i<N; i++)
                                    _propagator[1][i] *= _exp_dw_reduced[i];
                            }
                            else
                            {
                                for(int i=0; i<N; i++)
                                    _propagator[1][i] *= _exp_dw[i];
                            }

                            #ifndef NDEBUG
                            this->propagator_finished[key][1] = true;
                            #endif
                        }
                    }
                }

                if (n_segment_to == 0)
                    continue;

                if (n_segment_from == 0)
                {
                    // Apply mask
                    if (this->cb->get_mask() != nullptr)
                    {
                        const double* q_mask = this->cb->get_mask();
                        for(int i=0; i<N; i++)
                            _propagator[1][i] *= q_mask[i];
                    }

                    // q(r, 1+1/2)
                    if (propagator_half_steps[key][1] != nullptr)
                    {
                        #ifndef NDEBUG
                        if (propagator_half_steps_finished[key][1])
                            std::cout << "already half_step finished: " + key + ", " + std::to_string(1) << std::endl;
                        #endif

                        #ifndef NDEBUG
                        #pragma omp critical
                        this->time_complexity++;
                        #endif

                        this->propagator_solver->advance_propagator_half_bond_step(
                            _propagator[1],
                            propagator_half_steps[key][1],
                            monomer_type);

                        #ifndef NDEBUG
                        propagator_half_steps_finished[key][1] = true;
                        #endif
                    }
                    n_segment_from++;
                }

                // Advance propagator successively
                // Solver handles expand/reduce internally when space_group is set
                // q(r, s)
                for(int n=n_segment_from; n<n_segment_to; n++)
                {
                    #ifndef NDEBUG
                    if (!this->propagator_finished[key][n])
                        std::cout << "unfinished, key: " + key + ", " + std::to_string(n) << std::endl;
                    if (this->propagator_finished[key][n+1])
                        std::cout << "already finished: " + key + ", " + std::to_string(n+1) << std::endl;
                    #endif

                    #ifndef NDEBUG
                    #pragma omp critical
                    std::cout << job << " q_s, " << n << ", " <<
                        std::chrono::duration_cast<std::chrono::microseconds>
                        (std::chrono::system_clock::now().time_since_epoch()).count() - start_time << std::endl;
                    #endif

                    #ifndef NDEBUG
                    #pragma omp critical
                    this->time_complexity++;
                    #endif

                    // Discrete chains always use ds_index=0 (global ds)
                    this->propagator_solver->advance_propagator(
                        _propagator[n], _propagator[n+1],
                        monomer_type, this->cb->get_mask(), 0);

                    #ifndef NDEBUG
                    this->propagator_finished[key][n+1] = true;
                    #endif
                }

                // q(r, s+1/2)
                for(int n=n_segment_from; n<n_segment_to; n++)
                {
                    if (propagator_half_steps[key][n+1] != nullptr)
                    {
                        #ifndef NDEBUG
                        #pragma omp critical
                        std::cout << job << " q_s+1/2, " << n << ", " <<
                            std::chrono::duration_cast<std::chrono::microseconds>
                            (std::chrono::system_clock::now().time_since_epoch()).count() - start_time << std::endl;
                        #endif

                        #ifndef NDEBUG
                        if (propagator_half_steps_finished[key][n+1])
                            std::cout << "already half_step finished: " + key + ", " + std::to_string(n+1) << std::endl;
                        #endif

                        #ifndef NDEBUG
                        #pragma omp critical
                        this->time_complexity++;
                        #endif

                        this->propagator_solver->advance_propagator_half_bond_step(
                            _propagator[n+1],
                            propagator_half_steps[key][n+1],
                            monomer_type);

                        #ifndef NDEBUG
                        propagator_half_steps_finished[key][n+1] = true;
                        #endif
                    }
                }

                #ifndef NDEBUG
                #pragma omp critical
                std::cout << job << " finished, " <<
                    std::chrono::duration_cast<std::chrono::microseconds>
                    (std::chrono::system_clock::now().time_since_epoch()).count() - start_time << std::endl;
                #endif
            }
        }

        #ifndef NDEBUG
        std::cout << "time_complexity: " << this->time_complexity << std::endl;
        #endif

        // Compute total partition function of each distinct polymers
        // ComputationBox::inner_product_inverse_weight supports reduced basis when space_group is set
        for(const auto& segment_info: single_partition_segment)
        {
            int p                    = std::get<0>(segment_info);
            T *propagator_left       = std::get<1>(segment_info);
            T *propagator_right      = std::get<2>(segment_info);
            std::string monomer_type = std::get<3>(segment_info);
            int n_aggregated         = std::get<4>(segment_info);

            // Use appropriate basis for exp_dw (reduced or full)
            const T *_exp_dw = use_reduced_basis
                ? exp_dw_reduced_[monomer_type].data()
                : this->propagator_solver->exp_dw[0][monomer_type].data();

            this->single_polymer_partitions[p] = this->cb->inner_product_inverse_weight(
                propagator_left, propagator_right, _exp_dw) / (n_aggregated * this->cb->get_volume());
        }

    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
void CpuComputationDiscrete<T>::advance_propagator_single_segment(
    T* q_init, T *q_out, int p, int v, int u)
{
    try
    {
        // Get block info from polymer
        const Block& block = this->molecules->get_polymer(p).get_block(v, u);
        std::string monomer_type = block.monomer_type;

        // Discrete chains always use ds_index=0 (global ds)
        this->propagator_solver->advance_propagator(q_init, q_out, monomer_type, this->cb->get_mask(), 0);
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
void CpuComputationDiscrete<T>::compute_concentrations()
{
    try
    {
        const int M = this->cb->get_total_grid();
        const int N = this->cb->get_n_basis();
        const bool use_reduced_basis = (this->space_group_ != nullptr);

        // Calculate segment concentrations
        #pragma omp parallel for num_threads(this->n_streams)
        for(size_t b=0; b<this->phi_block.size();b++)
        {
            auto block = this->phi_block.begin();
            advance(block, b);
            const auto& key = block->first;

            int p                 = std::get<0>(key);
            std::string key_left  = std::get<1>(key);
            std::string key_right = std::get<2>(key);

            int n_segment_right = this->propagator_computation_optimizer->get_computation_block(key).n_segment_right;
            int n_segment_left  = this->propagator_computation_optimizer->get_computation_block(key).n_segment_left;
            std::string monomer_type = this->propagator_computation_optimizer->get_computation_block(key).monomer_type;
            int n_repeated = this->propagator_computation_optimizer->get_computation_block(key).n_repeated;

            // If there is no segment
            if(n_segment_right == 0)
            {
                for(int i=0; i<N; i++)
                    block->second[i] = 0.0;
                continue;
            }

            // Check keys
            #ifndef NDEBUG
            if (!this->propagator.contains(key_left))
                std::cout << "Could not find key_left key'" + key_left + "'. " << std::endl;
            if (!this->propagator.contains(key_right))
                std::cout << "Could not find key_right key'" + key_right + "'. " << std::endl;
            #endif

            // Get exp_dw (use pre-computed reduced basis version if available)
            const T *_exp_dw = use_reduced_basis
                ? exp_dw_reduced_[monomer_type].data()
                : this->propagator_solver->exp_dw[0][monomer_type].data();

            // Calculate phi of one block (in reduced basis when space_group is set)
            calculate_phi_one_block(
                block->second,
                this->propagator[key_left],
                this->propagator[key_right],
                _exp_dw,
                n_segment_left,
                n_segment_right
            );

            // Get local_ds from ds_index encoded in key
            Polymer& pc = this->molecules->get_polymer(p);
            int ds_index = PropagatorCode::get_ds_index_from_key(key_right);
            const ContourLengthMapping& mapping = this->molecules->get_contour_length_mapping();
            double local_ds = mapping.get_ds_from_index(ds_index);

            // Normalize concentration
            T norm = (local_ds*pc.get_volume_fraction()/pc.get_alpha()*n_repeated)/this->single_polymer_partitions[p];
            for(int i=0; i<N; i++)
                block->second[i] *= norm;
        }

        // Calculate partition functions and concentrations of solvents
        for(int s=0; s<this->molecules->get_n_solvent_types(); s++)
        {
            T *_phi = this->phi_solvent[s];
            double volume_fraction = std::get<0>(this->molecules->get_solvent(s));
            std::string monomer_type = std::get<1>(this->molecules->get_solvent(s));

            // Partition function needs full grid for cb->mean
            const T *_exp_dw_full = this->propagator_solver->exp_dw[0][monomer_type].data();
            this->single_solvent_partitions[s] = this->cb->mean(_exp_dw_full);

            // Use appropriate basis for phi calculation
            const T *_exp_dw = use_reduced_basis
                ? exp_dw_reduced_[monomer_type].data()
                : _exp_dw_full;
            T norm = volume_fraction / this->single_solvent_partitions[s];
            for(int i=0; i<N; i++)
                _phi[i] = _exp_dw[i] * norm;
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
void CpuComputationDiscrete<T>::calculate_phi_one_block(
    T *phi, T **q_1, T **q_2, const T *exp_dw, const int N_LEFT, const int N_RIGHT)
{
    try
    {
        const int N = this->cb->get_n_basis();  // reduced or full grid
        // Compute segment concentration
        for(int i=0; i<N; i++)
            phi[i] = q_1[N_LEFT][i]*q_2[1][i];
        for(int n=2; n<=N_RIGHT; n++)
        {
            for(int i=0; i<N; i++)
                phi[i] += q_1[N_LEFT-n+1][i]*q_2[n][i];
        }
        for(int i=0; i<N; i++)
            phi[i] /= exp_dw[i];
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
void CpuComputationDiscrete<T>::compute_stress()
{
    // This method should be invoked after invoking compute_statistics().

    // To calculate stress, we multiply weighted fourier basis to q(k)*q^dagger(-k).
    // We only need the real part of stress calculation.

    try
    {
        // if constexpr (std::is_same<T, std::complex<double>>::value)
        //     throw_with_line_number("Currently, stress computation is not suppoted for complex number type.");

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

        const int DIM = this->cb->get_dim();
        const int N_STRESS = 6;  // Full stress tensor: xx, yy, zz, xy, xz, yz
        const int M = this->cb->get_total_grid();

        // Space group: propagators in reduced basis need expansion to full grid for FFT
        const bool use_reduced_basis = (this->space_group_ != nullptr);

        std::map<std::tuple<int, std::string, std::string>, std::array<T,6>> block_dq_dl;

        // Reset stress map
        for(const auto& item: this->phi_block)
        {
            for(int d=0; d<N_STRESS; d++)
                block_dq_dl[item.first][d] = 0.0;
        }

        // Compute stress for each block
        #pragma omp parallel for num_threads(this->n_streams)
        for(size_t b=0; b<this->phi_block.size();b++)
        {
            auto block = this->phi_block.begin();
            advance(block, b);
            const auto& key   = block->first;

            std::string key_left  = std::get<1>(key);
            std::string key_right = std::get<2>(key);

            const int N_RIGHT = this->propagator_computation_optimizer->get_computation_block(key).n_segment_right;
            const int N_LEFT  = this->propagator_computation_optimizer->get_computation_block(key).n_segment_left;
            std::string monomer_type = this->propagator_computation_optimizer->get_computation_block(key).monomer_type;
            int n_repeated = this->propagator_computation_optimizer->get_computation_block(key).n_repeated;

            T **q_1 = this->propagator[key_left];     // dependency v
            T **q_2 = this->propagator[key_right];    // dependency u

            std::array<T,6> _block_dq_dl = block_dq_dl[key];

            // Compute stress at each chain bond
            for(int n=0; n<=N_RIGHT; n++)
            {
                T *q_segment_1;
                T *q_segment_2;
                bool is_half_bond_length;

                // At v (left endpoint)
                if (n == N_LEFT)
                {
                    // Skip if v is a leaf node
                    if (this->propagator_computation_optimizer->get_computation_propagator(key_left).deps.size() == 0)
                        continue;
                    q_segment_1 = propagator_half_steps[key_left][0];
                    q_segment_2 = q_2[N_RIGHT];
                    is_half_bond_length = true;
                }
                // At u (right endpoint) - junction case
                else if (n == 0 && key_right[0] == '(')
                {
                    // Skip if u is a leaf node
                    if (this->propagator_computation_optimizer->get_computation_propagator(key_right).deps.size() == 0)
                        continue;
                    q_segment_1 = q_1[N_LEFT];
                    q_segment_2 = propagator_half_steps[key_right][0];
                    is_half_bond_length = true;
                }
                // At aggregation junction - skip
                else if (n == 0)
                {
                    continue;
                }
                // Within the block
                else
                {
                    q_segment_1 = q_1[N_LEFT-n];
                    q_segment_2 = q_2[n];
                    is_half_bond_length = false;
                }

                // Solver handles expand/reduce internally when space_group is set
                std::vector<T> segment_stress = this->propagator_solver->compute_single_segment_stress(
                    q_segment_1, q_segment_2, monomer_type, is_half_bond_length);

                for(int d=0; d<N_STRESS; d++)
                    _block_dq_dl[d] += segment_stress[d]*((T)n_repeated);
            }

            // Multiply by local_ds for this block
            int ds_index = PropagatorCode::get_ds_index_from_key(key_right);
            const ContourLengthMapping& mapping = this->molecules->get_contour_length_mapping();
            double local_ds = mapping.get_ds_from_index(ds_index);
            for(int d=0; d<N_STRESS; d++)
                _block_dq_dl[d] *= local_ds;

            block_dq_dl[key] = _block_dq_dl;
        }

        // Compute total stress
        int n_polymer_types = this->molecules->get_n_polymer_types();
        for(int p=0; p<n_polymer_types; p++)
            for(int d=0; d<N_STRESS; d++)
                this->dq_dl[p][d] = 0.0;
        for(const auto& block: this->phi_block)
        {
            const auto& key = block.first;
            int p = std::get<0>(key);
            for(int d=0; d<N_STRESS; d++)
                this->dq_dl[p][d] += block_dq_dl[key][d];
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
            // The metric tensor g has: g₁₁ = L₁², g₁₂ = L₁L₂cosγ, etc.
            // ∂|k|²/∂L₁ = Vᵢⱼ ∂gᵢⱼ/∂L₁, where ∂g₁₁/∂L₁ = 2L₁, ∂g₁₂/∂L₁ = L₂cosγ, etc.
            // The factor of 2 from the metric is already absorbed in the normalization
            this->dq_dl[p][0] = (L1*V_11 + L2*cos_g*V_12 + L3*cos_b*V_13) / norm;
            if (DIM >= 2) {
                this->dq_dl[p][1] = (L2*V_22 + L1*cos_g*V_12 + L3*cos_a*V_23) / norm;
            }
            if (DIM >= 3) {
                this->dq_dl[p][2] = (L3*V_33 + L1*cos_b*V_13 + L2*cos_a*V_23) / norm;
            }

            // Compute angle derivatives using metric tensor formulas
            // ∂g₁₂/∂γ = -L₁L₂sinγ, so ∂H/∂γ ∝ -L₁L₂sinγ·V₁₂
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
void CpuComputationDiscrete<T>::get_chain_propagator(T *q_out, int polymer, int v, int u, int n)
{
    // This method should be invoked after invoking compute_statistics()

    // Get chain propagator for a selected polymer, block and direction.
    // This is made for debugging and testing.
    try
    {
        const int N = this->cb->get_n_basis();  // n_irreducible (with space group) or total_grid
        Polymer& pc = this->molecules->get_polymer(polymer);
        std::string dep = pc.get_propagator_key(v,u);

        if (!this->propagator_computation_optimizer->get_computation_propagators().contains(dep))
            throw_with_line_number("Could not find the propagator code '" + dep + "'. Disable 'aggregation' option to obtain propagator_computation_optimizer.");

        const int N_RIGHT = this->propagator_computation_optimizer->get_computation_propagator(dep).max_n_segment;
        if (n < 1 || n > N_RIGHT)
            throw_with_line_number("n (" + std::to_string(n) + ") must be in range [1, " + std::to_string(N_RIGHT) + "]");

        T **partition = this->propagator[dep];
        for(int i=0; i<N; i++)
            q_out[i] = partition[n][i];
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
bool CpuComputationDiscrete<T>::check_total_partition()
{
    const bool use_reduced_basis = (this->space_group_ != nullptr);
    int n_polymer_types = this->molecules->get_n_polymer_types();
    std::vector<std::vector<T>> total_partitions;
    for(int p=0;p<n_polymer_types;p++)
    {
        std::vector<T> total_partitions_p;
        total_partitions.push_back(total_partitions_p);
    }

    // ComputationBox::inner_product_inverse_weight supports reduced basis when space_group is set
    for(const auto& block: this->phi_block)
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
        const T *_exp_dw = use_reduced_basis
            ? exp_dw_reduced_[monomer_type].data()
            : this->propagator_solver->exp_dw[0][monomer_type].data();

        #ifndef NDEBUG
        std::cout<< p << ", " << key_left << ", " << key_right << ": " << n_segment_left << ", " << n_segment_right << ", " << n_propagators << ", " << this->propagator_computation_optimizer->get_computation_block(key).n_repeated << std::endl;
        #endif

        for(int n=1;n<=n_segment_right;n++)
        {
            T total_partition = this->cb->inner_product_inverse_weight(
                this->propagator[key_left][n_segment_left-n+1],
                this->propagator[key_right][n], _exp_dw) * (n_repeated / this->cb->get_volume());

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
#include "TemplateInstantiations.h"
INSTANTIATE_CLASS(CpuComputationDiscrete);