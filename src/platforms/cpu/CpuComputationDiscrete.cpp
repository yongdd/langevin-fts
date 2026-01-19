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
    PropagatorComputationOptimizer *propagator_computation_optimizer)
    : CpuComputationBase<T>(cb, molecules, propagator_computation_optimizer)
{
    try
    {
        #ifndef NDEBUG
        std::cout << "--------- Discrete Chain Solver, CPU Version ---------" << std::endl;
        #endif

        const int M = this->cb->get_total_grid();
        this->propagator_solver = new CpuSolverPseudoDiscrete<T>(cb, molecules);

        // The number of parallel streams for propagator computation
        const char *ENV_OMP_NUM_THREADS = getenv("OMP_NUM_THREADS");
        std::string env_omp_num_threads(ENV_OMP_NUM_THREADS ? ENV_OMP_NUM_THREADS  : "");
        if (env_omp_num_threads.empty())
            this->n_streams = 8;
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
                propagator_half_steps[key][0] = new T[M];
            else
                propagator_half_steps[key][0] = nullptr;

            // Allocate memory for q(r,s+1/2)
            for(int i=1; i<this->propagator_size[key]; i++)
            {
                if (!item.second.junction_ends.contains(i))
                    propagator_half_steps[key][i] = nullptr;
                else
                    propagator_half_steps[key][i] = new T[M];
            }

            // Allocate memory for q(r,s)
            // Index 0 will be not used
            this->propagator[key] = new T*[max_n_segment];
            this->propagator[key][0] = nullptr;
            for(int i=1; i<this->propagator_size[key]; i++)
                this->propagator[key][i] = new T[M];

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
            this->phi_block[item.first] = new T[M]();  // Zero-initialize
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
            this->phi_solvent.push_back(new T[M]);

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

        for(const auto& item: this->propagator_computation_optimizer->get_computation_propagators())
        {
            if( !w_input.contains(item.second.monomer_type))
                throw_with_line_number("monomer_type \"" + item.second.monomer_type + "\" is not in w_input.");
        }

        #ifndef NDEBUG
        this->time_complexity = 0;
        #endif

        // Update dw or exp_dw
        this->propagator_solver->update_dw(w_input);

        // Assign a pointer for mask
        const double *q_mask = this->cb->get_mask();
        
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
                const T *_exp_dw = this->propagator_solver->exp_dw[1][monomer_type].data();

                // Calculate one block end
                if (n_segment_from == 0 && deps.size() == 0) // if it is leaf node
                {
                    #ifndef NDEBUG
                    #pragma omp critical
                    std::cout << job << " init 1, " << 
                        std::chrono::duration_cast<std::chrono::microseconds>
                        (std::chrono::system_clock::now().time_since_epoch()).count() - start_time << std::endl;
                    #endif

                     // q_init
                    if (key[0] == '{')
                    {
                        std::string g = PropagatorCode::get_q_input_idx_from_key(key);
                        if (!q_init.contains(g))
                            throw_with_line_number("Could not find q_init[\"" + g + "\"]. Pass q_init to run() for grafted polymers.");
                        for(int i=0; i<M; i++)
                            _propagator[1][i] = q_init[g][i]*_exp_dw[i];
                    }
                    else
                    {
                        for(int i=0; i<M; i++)
                            _propagator[1][i] = _exp_dw[i];
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
                        for(int i=0; i<M; i++)
                            _propagator[1][i] = 0.0;

                        for(size_t d=0; d<deps.size(); d++)
                        {
                            std::string sub_dep = std::get<0>(deps[d]);
                            int sub_n_segment   = std::get<1>(deps[d]);
                            int sub_n_repeated  = std::get<2>(deps[d]);
                            T **_propagator_sub_dep;

                            if (sub_n_segment == 0)
                            {
                                // Check sub key
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
                                // Check sub key
                                #ifndef NDEBUG
                                if (!this->propagator.contains(sub_dep))
                                    std::cout << "Could not find sub key '" + sub_dep + "'. " << std::endl;
                                if (!this->propagator_finished[sub_dep][sub_n_segment])
                                    std::cout << "Could not compute '" + key +  "', since '"+ sub_dep + std::to_string(sub_n_segment) + "' is not prepared." << std::endl;
                                #endif

                                _propagator_sub_dep = this->propagator[sub_dep];
                            }
                            for(int i=0; i<M; i++)
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
                            for(int i=0; i<M; i++)
                                _propagator_half_step[i] = _propagator[1][i];

                            // Add half bond
                            this->propagator_solver->advance_propagator_half_bond_step(
                                _propagator[1], _propagator[1], monomer_type);

                            // Add full segment
                            for(int i=0; i<M; i++)
                                _propagator[1][i] *= _exp_dw[i];
                        }
                        else
                        {
                            this->propagator_solver->advance_propagator(
                                _propagator[1],
                                _propagator[1],
                                monomer_type,
                                q_mask);
                        }

                        #ifndef NDEBUG
                        this->propagator_finished[key][1] = true;
                        #endif
                        // std::cout << "finished, key, n: " + key + ", 0" << std::endl;
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

                        // Combine branches
                        T *_q_junction_start = propagator_half_steps[key][0];
                        for(int i=0; i<M; i++)
                            _q_junction_start[i] = 1.0;
                        for(size_t d=0; d<deps.size(); d++)
                        {
                            std::string sub_dep = std::get<0>(deps[d]);
                            int sub_n_segment   = std::get<1>(deps[d]);
                            int sub_n_repeated  = std::get<2>(deps[d]);

                            // Check sub key
                            #ifndef NDEBUG
                            if (!propagator_half_steps_finished[sub_dep][sub_n_segment])
                                std::cout << "Could not compute '" + key +  "', since '"+ sub_dep + std::to_string(sub_n_segment) + "+1/2' is not prepared." << std::endl;
                            #endif

                            T *_propagator_half_step = propagator_half_steps[sub_dep][sub_n_segment];
                            for(int i=0; i<M; i++)
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

                            // Add half bond
                            this->propagator_solver->advance_propagator_half_bond_step(
                                _q_junction_start, _propagator[1], monomer_type);

                            // Add full segment
                            for(int i=0; i<M; i++)
                                _propagator[1][i] *= _exp_dw[i];
                            
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
                    // Multiply mask
                    if (q_mask != nullptr)
                    {
                        for(int i=0; i<M; i++)
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

                    this->propagator_solver->advance_propagator(
                        _propagator[n], _propagator[n+1],
                        monomer_type, q_mask);

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
        for(const auto& segment_info: single_partition_segment)
        {
            int p                    = std::get<0>(segment_info);
            T *propagator_left       = std::get<1>(segment_info);
            T *propagator_right      = std::get<2>(segment_info);
            std::string monomer_type = std::get<3>(segment_info);
            int n_aggregated         = std::get<4>(segment_info);
            const T *_exp_dw    = this->propagator_solver->exp_dw[1][monomer_type].data();

            this->single_polymer_partitions[p]= this->cb->inner_product_inverse_weight(
                propagator_left, propagator_right, _exp_dw)/(n_aggregated*this->cb->get_volume());
        }

    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
void CpuComputationDiscrete<T>::advance_propagator_single_segment(
    T* q_init, T *q_out, std::string monomer_type)
{
    try
    {
        // Assign a pointer for mask
        const double *q_mask = this->cb->get_mask();
        this->propagator_solver->advance_propagator(q_init, q_out, monomer_type, q_mask);
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
            const T *_exp_dw = this->propagator_solver->exp_dw[1][monomer_type].data();

            // If there is no segment
            if(n_segment_right == 0)
            {
                for(int i=0; i<M;i++)
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

            // Calculate phi of one block (possibly multiple blocks when using aggregation)
            calculate_phi_one_block(
                block->second,          // phi
                this->propagator[key_left],   // dependency v
                this->propagator[key_right],  // dependency u
                _exp_dw,                // exp_dw
                n_segment_left,
                n_segment_right
            );
            
            // Normalize concentration
            Polymer& pc = this->molecules->get_polymer(p);
            T norm = (pc.get_volume_fraction()/pc.get_n_segment_total()*n_repeated)/this->single_polymer_partitions[p];
            for(int i=0; i<M; i++)
                block->second[i] *= norm;
        }

        // Calculate partition functions and concentrations of solvents
        for(int s=0; s<this->molecules->get_n_solvent_types(); s++)
        {
            T *_phi = this->phi_solvent[s];
            double volume_fraction = std::get<0>(this->molecules->get_solvent(s));
            std::string monomer_type = std::get<1>(this->molecules->get_solvent(s));
            const T *_exp_dw = this->propagator_solver->exp_dw[1][monomer_type].data();

            this->single_solvent_partitions[s] = this->cb->integral(_exp_dw)/this->cb->get_volume();
            for(int i=0; i<M; i++)
                _phi[i] = _exp_dw[i]*volume_fraction/this->single_solvent_partitions[s];
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
        const int M = this->cb->get_total_grid();
        // Compute segment concentration
        for(int i=0; i<M; i++)
            phi[i] = q_1[N_LEFT][i]*q_2[1][i];
        for(int n=2; n<=N_RIGHT; n++)
        {
            for(int i=0; i<M; i++)
                phi[i] += q_1[N_LEFT-n+1][i]*q_2[n][i];
        }
        for(int i=0; i<M; i++)
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

        const int N_STRESS = 6;  // Full stress tensor: xx, yy, zz, xy, xz, yz
        const int DIM = this->cb->get_dim();
        const int M   = this->cb->get_total_grid();

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

            #ifndef NDEBUG
            std::cout << "key_left, key_right, N_LEFT, N_RIGHT: "
                 << key_left << ", " << key_right << ", " << N_LEFT << ", " << N_RIGHT << std::endl;
            std::cout << this->propagator_computation_optimizer->get_computation_propagator(key_left).deps.size() << ", "
                 << this->propagator_computation_optimizer->get_computation_propagator(key_right).deps.size() << std::endl;
            #endif

            // // If there is no segment
            // if(N_RIGHT == 0)
            //     continue;

            T **q_1 = this->propagator[key_left];     // dependency v
            T **q_2 = this->propagator[key_right];    // dependency u

            T *q_segment_1;
            T *q_segment_2;

            bool is_half_bond_length;
            std::array<T,6> _block_dq_dl = block_dq_dl[key];

            // Example: N==5
            // n:          0  1  2  3  4  5
            // Direction:
            // Case 1)      O--O--O--O--O    (chain ends at both ends, skip at 0 and N)
            // Case 2)     -O--O--O--O--O-   (junctions at both ends, half bond lengths at both ends)
            // Case 3)    --O--O--O--O--O    (aggregation junction at one end, skip at that end)
            // Case 4)      O--O--O--O--O--  (aggregation junction at the other end, compute with full bond length)

            // Compute stress at each chain bond
            for(int n=0; n<=N_RIGHT; n++)
            {
                // block_dq_dl[key][0] = 0.0;
                // At v
                if (n == N_LEFT)
                {
                    // std::cout << "case 1: " << propagator_junction_start[key_left][0] << ", " << q_2[(N-1)*M] << std::endl;
                    if (this->propagator_computation_optimizer->get_computation_propagator(key_left).deps.size() == 0) // if v is leaf node, skip
                        continue;
                    q_segment_1 = propagator_half_steps[key_left][0];
                    q_segment_2 = q_2[N_RIGHT];
                    is_half_bond_length = true;
                }
                // At u
                else if (n == 0 && key_right[0] == '(')
                {
                    // std::cout << "case 2: " << q_1[(N_LEFT-1)*M] << ", " << propagator_junction_start[key_right][0] << std::endl;
                    if (this->propagator_computation_optimizer->get_computation_propagator(key_right).deps.size() == 0) // if u is leaf node, skip
                        continue;
                    q_segment_1 = q_1[N_LEFT];
                    q_segment_2 = propagator_half_steps[key_right][0];
                    is_half_bond_length = true;
                }
                // At aggregation junction
                else if (n == 0)
                {
                    // std::cout << "case 4" << std::endl;
                    continue;
                }
                // Within the blocks
                else
                {
                    // std::cout << "case 5: " << q_1[(N_LEFT-n-1)*M] << ", " << q_2[(n-1)*M] << std::endl;

                    // double temp_sum1=0;
                    // double temp_sum2=0;
                    // For (int i=0;i<M;i++)
                    // {
                    //     temp_sum1 += q_1[(N_LEFT-n-1)*M+1];
                    //     temp_sum2 += q_2[(n-1)*M+1];
                    // }
                    // std::cout << "\t" << temp_sum1 << ", " << temp_sum2 << std::endl;
                    q_segment_1 = q_1[N_LEFT-n];
                    q_segment_2 = q_2[n];
                    is_half_bond_length = false;

                    // std::cout << "\t" << bond_length_sq << ", " << boltz_bond_now[10] << std::endl;
                }
                // Compute 
                std::vector<T> segment_stress = this->propagator_solver->compute_single_segment_stress(
                    q_segment_1, q_segment_2, monomer_type, is_half_bond_length);

                #ifndef NDEBUG
                std::cout << b << " " << key_left << ", " << key_right << "," << n << ",x: " << segment_stress[0]*((T)n_repeated) << ", " << is_half_bond_length << std::endl;
                #endif

                for(int d=0; d<N_STRESS; d++)
                    _block_dq_dl[d] += segment_stress[d]*((T)n_repeated);

                // std::cout << "n: " << n << ", " << is_half_bond_length << ", " << segment_stress[0] << std::endl;
                // std::cout << "n: " << n << ", " << block_dq_dl[key][0] << std::endl;
            }

            // Multiply by local_ds for this block
            int p_idx = std::get<0>(key);
            Polymer& pc_stress = this->molecules->get_polymer(p_idx);
            const auto& v_u_stress = this->propagator_computation_optimizer->get_computation_block(key).v_u;
            int v_stress = std::get<0>(v_u_stress[0]);
            int u_stress = std::get<1>(v_u_stress[0]);
            double contour_length = pc_stress.get_block(v_stress, u_stress).contour_length;
            const ContourLengthMapping& mapping_stress = this->molecules->get_contour_length_mapping();
            double local_ds = mapping_stress.get_local_ds(contour_length);
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
            const auto& key       = block.first;
            int p                 = std::get<0>(key);
            std::string key_left  = std::get<1>(key);
            std::string key_right = std::get<2>(key);

            for(int d=0; d<N_STRESS; d++)
                this->dq_dl[p][d] += block_dq_dl[key][d];
        }
        // ============ DEFORMATION VECTOR APPROACH ============
        // The deformation vector v = h⁻¹k transforms k⊗k sums to a natural basis
        // where the metric tensor g = hᵀh has simple derivatives:
        //   g₁₁ = L₁², g₂₂ = L₂², g₃₃ = L₃²
        //   g₁₂ = L₁L₂cosγ, g₁₃ = L₁L₃cosβ, g₂₃ = L₂L₃cosα
        //
        // |k|² = vᵀgv, so derivatives become:
        //   ∂H/∂L₁ ∝ 2(L₁V₁₁ + L₂cosγ·V₁₂ + L₃cosβ·V₁₃)
        //   ∂H/∂γ  ∝ -2L₁L₂sinγ·V₁₂
        //
        // V = h⁻¹ S⁽ᵏ⁾ h⁻ᵀ (congruence transformation)

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
            // Get Cartesian k⊗k sums: S⁽ᵏ⁾
            T S_xx = this->dq_dl[p][0];
            T S_yy = this->dq_dl[p][1];
            T S_zz = 0.0, S_xy = 0.0, S_xz = 0.0, S_yz = 0.0;

            if (DIM == 3) {
                S_zz = this->dq_dl[p][2];
                S_xy = this->dq_dl[p][3];
                S_xz = this->dq_dl[p][4];
                S_yz = this->dq_dl[p][5];
            } else if (DIM == 2) {
                S_xy = this->dq_dl[p][2];
            }

            // Compute h⁻¹ for the lattice matrix h = [a|b|c]
            // h = | L₁    L₂cosγ   c_x |
            //     | 0     L₂sinγ   c_y |
            //     | 0     0        c_z |
            // h⁻¹ is upper triangular
            double h_inv_11 = 1.0 / L1;
            double h_inv_12 = -cos_g / (L1 * sin_g);
            double h_inv_22 = 1.0 / (L2 * sin_g);

            // For 3D, c vector components and h⁻¹ elements
            double h_inv_13 = 0.0, h_inv_23 = 0.0, h_inv_33 = 1.0;
            if (DIM == 3) {
                double c_x = L3 * cos_b;
                double c_y = L3 * (cos_a - cos_b * cos_g) / sin_g;
                double c_z = L3 * std::sqrt(1.0 - cos_a*cos_a - cos_b*cos_b - cos_g*cos_g
                                            + 2.0*cos_a*cos_b*cos_g) / sin_g;
                h_inv_33 = 1.0 / c_z;
                h_inv_23 = -c_y / (L2 * sin_g * c_z);
                h_inv_13 = (c_y * cos_g / sin_g - c_x) / (L1 * c_z);
            }

            // Transform k⊗k sums to v⊗v sums: V = h⁻¹ S⁽ᵏ⁾ h⁻ᵀ
            // For symmetric S, V is also symmetric
            T V_11, V_22, V_33, V_12, V_13, V_23;

            if (DIM == 1) {
                // V = h⁻¹ S h⁻ᵀ for 1×1 case
                // h = [L₁], h⁻¹ = [1/L₁], V₁₁ = (1/L₁)² S_xx
                V_11 = h_inv_11*h_inv_11*S_xx;
            } else if (DIM == 2) {
                // V = h⁻¹ S h⁻ᵀ for 2×2 case
                V_11 = h_inv_11*h_inv_11*S_xx + 2*h_inv_11*h_inv_12*S_xy + h_inv_12*h_inv_12*S_yy;
                V_22 = h_inv_22*h_inv_22*S_yy;
                V_12 = h_inv_11*h_inv_22*S_xy + h_inv_12*h_inv_22*S_yy;
            } else if (DIM == 3) {
                // V = h⁻¹ S h⁻ᵀ for 3×3 case (h⁻¹ is upper triangular, h⁻ᵀ is lower triangular)
                // First compute T = S h⁻ᵀ, then V = h⁻¹ T
                // h⁻ᵀ[k,j] = h⁻¹[j,k], nonzero only when k >= j
                T T_11 = S_xx*h_inv_11 + S_xy*h_inv_12 + S_xz*h_inv_13;
                T T_12 = S_xy*h_inv_22 + S_xz*h_inv_23;
                T T_13 = S_xz*h_inv_33;
                T T_21 = S_xy*h_inv_11 + S_yy*h_inv_12 + S_yz*h_inv_13;
                T T_22 = S_yy*h_inv_22 + S_yz*h_inv_23;
                T T_23 = S_yz*h_inv_33;
                T T_31 = S_xz*h_inv_11 + S_yz*h_inv_12 + S_zz*h_inv_13;
                T T_32 = S_yz*h_inv_22 + S_zz*h_inv_23;
                T T_33 = S_zz*h_inv_33;

                // V = h⁻¹ T
                V_11 = h_inv_11*T_11 + h_inv_12*T_21 + h_inv_13*T_31;
                V_12 = h_inv_11*T_12 + h_inv_12*T_22 + h_inv_13*T_32;
                V_13 = h_inv_11*T_13 + h_inv_12*T_23 + h_inv_13*T_33;
                V_22 = h_inv_22*T_22 + h_inv_23*T_32;
                V_23 = h_inv_22*T_23 + h_inv_23*T_33;
                V_33 = h_inv_33*T_33;
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
        const int M = this->cb->get_total_grid();
        Polymer& pc = this->molecules->get_polymer(polymer);
        std::string dep = pc.get_propagator_key(v,u);

        if (!this->propagator_computation_optimizer->get_computation_propagators().contains(dep))
            throw_with_line_number("Could not find the propagator code '" + dep + "'. Disable 'aggregation' option to obtain propagator_computation_optimizer.");
            
        const int N_RIGHT = this->propagator_computation_optimizer->get_computation_propagator(dep).max_n_segment;
        if (n < 1 || n > N_RIGHT)
            throw_with_line_number("n (" + std::to_string(n) + ") must be in range [1, " + std::to_string(N_RIGHT) + "]");

        T **partition = this->propagator[dep];
        for(int i=0; i<M; i++)
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
    // const int M = this->cb->get_total_grid();
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

        int n_segment_right = this->propagator_computation_optimizer->get_computation_block(key).n_segment_right;
        int n_segment_left  = this->propagator_computation_optimizer->get_computation_block(key).n_segment_left;
        int n_repeated      = this->propagator_computation_optimizer->get_computation_block(key).n_repeated;
        int n_propagators   = this->propagator_computation_optimizer->get_computation_block(key).v_u.size();

        std::string monomer_type = this->propagator_computation_optimizer->get_computation_block(key).monomer_type;
        const T *_exp_dw = this->propagator_solver->exp_dw[1][monomer_type].data();

        #ifndef NDEBUG
        std::cout<< p << ", " << key_left << ", " << key_right << ": " << n_segment_left << ", " << n_segment_right << ", " << n_propagators << ", " << this->propagator_computation_optimizer->get_computation_block(key).n_repeated << std::endl;
        #endif

        for(int n=1;n<=n_segment_right;n++)
        {
            T total_partition = this->cb->inner_product_inverse_weight(
                this->propagator[key_left][n_segment_left-n+1],
                this->propagator[key_right][n], _exp_dw)*(n_repeated/this->cb->get_volume());
            
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