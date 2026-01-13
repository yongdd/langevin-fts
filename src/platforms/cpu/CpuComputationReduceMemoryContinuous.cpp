/**
 * @file CpuComputationReduceMemoryContinuous.cpp
 * @brief Memory-efficient CPU propagator computation with checkpointing.
 *
 * Implements propagator computation for continuous chains with reduced
 * memory footprint. Instead of storing all propagator slices, stores
 * only checkpoints and recomputes intermediate values as needed.
 *
 * **Memory Reduction Strategy:**
 *
 * - Store propagators only at dependency points (junctions, chain ends)
 * - Recompute intermediate propagators from nearest checkpoint
 * - Trades computation time for reduced memory usage
 *
 * **Use Case:**
 *
 * Useful for long chains or large grids where storing all propagator
 * slices would exceed available memory. Memory usage is O(N_checkpoints)
 * instead of O(N_segments).
 *
 * **Performance:**
 *
 * Approximately 2-4x slower than full storage mode due to recomputation.
 * Single-threaded to minimize memory usage during recomputation.
 *
 * **Template Instantiations:**
 *
 * - CpuComputationReduceMemoryContinuous<double>: Real fields
 * - CpuComputationReduceMemoryContinuous<std::complex<double>>: Complex fields
 *
 * @see CpuComputationContinuous for full storage version
 */

#include <cmath>
#include <numbers>
#include <omp.h>

#include "CpuComputationReduceMemoryContinuous.h"
#include "CpuSolverPseudoRQM4.h"
#include "CpuSolverPseudoETDRK4.h"
#include "CpuSolverCNADI.h"
#include "CpuSolverRichardsonGlobal.h"
#include "SimpsonRule.h"
#include "PropagatorCode.h"

/**
 * @brief Construct memory-efficient propagator computation.
 *
 * Allocates checkpoint storage and temporary arrays for recomputation.
 * Uses single thread to minimize memory during propagator advancement.
 *
 * @param cb                             Computation box for grid operations
 * @param molecules                      Polymer/solvent species definitions
 * @param propagator_computation_optimizer Optimized computation schedule
 * @param method                         "pseudospectral" or "realspace"
 * @param numerical_method               Numerical algorithm:
 *                                       - For pseudospectral: "rqm4" or "etdrk4"
 *                                       - For realspace: "cn-adi2" or "cn-adi4"
 */
template <typename T>
CpuComputationReduceMemoryContinuous<T>::CpuComputationReduceMemoryContinuous(
    ComputationBox<T>* cb,
    Molecules *molecules,
    PropagatorComputationOptimizer *propagator_computation_optimizer,
    std::string method,
    std::string numerical_method)
    : PropagatorComputation<T>(cb, molecules, propagator_computation_optimizer)
{
    try
    {
        #ifndef NDEBUG
        std::cout << "--------- Continuous Chain Solver, CPU Version ---------" << std::endl;
        #endif

        const int M = this->cb->get_total_grid();

        this->method = method;
        if(method == "pseudospectral")
        {
            if (numerical_method == "" || numerical_method == "rqm4")
                this->propagator_solver = new CpuSolverPseudoRQM4<T>(cb, molecules);
            else if (numerical_method == "etdrk4")
                this->propagator_solver = new CpuSolverPseudoETDRK4<T>(cb, molecules);
            else
                throw_with_line_number("Unknown pseudo-spectral method: '" + numerical_method + "'. Use 'rqm4' or 'etdrk4'.");
        }
        else if(method == "realspace")
        {
            if constexpr (std::is_same<T, double>::value)
            {
                if (numerical_method == "cn-adi4-g")
                {
                    // Global Richardson: two independent evolutions combined at the end
                    this->propagator_solver = new CpuSolverRichardsonGlobal(cb, molecules);
                }
                else
                {
                    // Per-step Richardson (cn-adi4) or 2nd order (cn-adi2)
                    bool use_4th_order = (numerical_method == "cn-adi4");
                    this->propagator_solver = new CpuSolverCNADI(cb, molecules, use_4th_order);
                }
            }
            else
                throw_with_line_number("Currently, the realspace method is only available for double precision.");
        }
        // The number of parallel streams is always 1 to reduce the memory usage
        n_streams = 1;
        #ifndef NDEBUG
        std::cout << "The number of CPU threads is always set to " << n_streams << "." << std::endl;
        #endif

        // Allocate memory for concentrations
        if( this->propagator_computation_optimizer->get_computation_blocks().size() == 0)
            throw_with_line_number("There is no block. Add polymers first.");
        for(const auto& item: this->propagator_computation_optimizer->get_computation_blocks())
        {
            phi_block[item.first] = new T[M]();  // Zero-initialize
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
                propagator_at_check_point[std::make_tuple(key, 0)] = new T[M];

            for(size_t d=0; d<deps.size(); d++)
            {
                std::string sub_dep = std::get<0>(deps[d]);
                int sub_n_segment   = std::get<1>(deps[d]);

                if(propagator_at_check_point.find(std::make_tuple(sub_dep, sub_n_segment)) == propagator_at_check_point.end())
                    propagator_at_check_point[std::make_tuple(sub_dep, sub_n_segment)] = new T[M];
            }

            #ifndef NDEBUG
            propagator_finished[key] = new bool[max_n_segment];
            for(int i=0; i<max_n_segment;i++)
                propagator_finished[key][i] = false;
            #endif
        }

        // Iterate through each element in the map
        #ifndef NDEBUG
        for (const auto& pair : propagator_at_check_point) {
            std::cout << "Key: " << std::get<0>(pair.first) << ", Value: " << std::get<1>(pair.first) << std::endl;
        }
        #endif

        // Find the total sum of segments for checkpoint interval calculation
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
        // - q_recal[0..checkpoint_interval-1]: workspace for storage (size = checkpoint_interval)
        // - q_pair[0-1]: ping-pong for q_right advancement
        // - q_skip[0-1]: ping-pong for skip phase (used in both compute_concentrations and compute_stress)
        const int workspace_size = checkpoint_interval;
        this->q_recal.resize(workspace_size);
        for(int n=0; n<workspace_size; n++)
            this->q_recal[n] = new T[M];
        this->q_pair[0] = new T[M];
        this->q_pair[1] = new T[M];
        this->q_skip[0] = new T[M];
        this->q_skip[1] = new T[M];

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
                    propagator_at_check_point[std::make_tuple(key, n)] = new T[M];
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

            int n_aggregated   = this->propagator_computation_optimizer->get_computation_block(key).v_u.size()/
                                 this->propagator_computation_optimizer->get_computation_block(key).n_repeated;
            int n_segment_left = this->propagator_computation_optimizer->get_computation_block(key).n_segment_left;

            if(propagator_at_check_point.find(std::make_tuple(key_left, n_segment_left)) == propagator_at_check_point.end())
            {
                propagator_at_check_point[std::make_tuple(key_left, n_segment_left)] = new T[M];
                #ifndef NDEBUG
                std::cout << "Allocated, " + key_left + ", " << n_segment_left << std::endl;
                #endif
            }

            if(propagator_at_check_point.find(std::make_tuple(key_right, 0)) == propagator_at_check_point.end())
            {
                propagator_at_check_point[std::make_tuple(key_right, 0)] = new T[M];
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
        sc = new Scheduler(this->propagator_computation_optimizer->get_computation_propagators(), 1); 

        propagator_solver->update_laplacian_operator();
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
CpuComputationReduceMemoryContinuous<T>::~CpuComputationReduceMemoryContinuous()
{
    delete propagator_solver;
    delete sc;

    // Free workspace (must match constructor allocation)
    for(size_t n=0; n<q_recal.size(); n++)
        delete[] this->q_recal[n];

    delete[] this->q_pair[0];
    delete[] this->q_pair[1];
    delete[] this->q_skip[0];
    delete[] this->q_skip[1];

    for(const auto& item: propagator_at_check_point)
        delete[] item.second;
    for(const auto& item: phi_block)
        delete[] item.second;
    for(const auto& item: phi_solvent)
        delete[] item;

    #ifndef NDEBUG
    for(const auto& item: propagator_finished)
        delete[] item.second;
    #endif
}
template <typename T>
void CpuComputationReduceMemoryContinuous<T>::update_laplacian_operator()
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
void CpuComputationReduceMemoryContinuous<T>::compute_statistics(
    std::map<std::string, const T*> w_input,
    std::map<std::string, const T*> q_init)
{
    this->compute_propagators(w_input, q_init);
    this->compute_concentrations();
}
template <typename T>
void CpuComputationReduceMemoryContinuous<T>::compute_propagators(
    std::map<std::string, const T*> w_input,
    std::map<std::string, const T*> q_init)
{
    try
    {
        const int M = this->cb->get_total_grid();

        for(const auto& item: this->propagator_computation_optimizer->get_computation_propagators())
        {
            if( w_input.find(item.second.monomer_type) == w_input.end())
                throw_with_line_number("monomer_type \"" + item.second.monomer_type + "\" is not in w_input.");
        }

        // Update dw or exp_dw
        propagator_solver->update_dw(w_input);

        // Assign a pointer for mask
        const double *q_mask = this->cb->get_mask();

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
            // Display all jobs
            #ifndef NDEBUG
            std::cout << "jobs:" << std::endl;
            for(size_t job=0; job<parallel_job->size(); job++)
            {
                auto& key          = std::get<0>((*parallel_job)[job]);
                int n_segment_from = std::get<1>((*parallel_job)[job]);
                int n_segment_to   = std::get<2>((*parallel_job)[job]);
                std::cout << "key, n_segment_from, n_segment_to: " + key + ", " + std::to_string(n_segment_from) + ", " + std::to_string(n_segment_to) + ". " << std::endl;
            }
            #endif

            // For each propagator
            for(size_t job=0; job<parallel_job->size(); job++)
            {
                auto& key = std::get<0>((*parallel_job)[job]);
                int n_segment_from = std::get<1>((*parallel_job)[job]);
                int n_segment_to   = std::get<2>((*parallel_job)[job]);
                auto& deps = this->propagator_computation_optimizer->get_computation_propagator(key).deps;
                auto monomer_type = this->propagator_computation_optimizer->get_computation_propagator(key).monomer_type;

                // Display job info
                #ifndef NDEBUG
                std::cout << job << " started" << std::endl;
                #endif

                // Check key
                #ifndef NDEBUG
                if (propagator_at_check_point.find(std::make_tuple(key, n_segment_from)) == propagator_at_check_point.end())
                    std::cout << "Could not find key '" + key + "'. " << std::endl;
                #endif

                int prev = 0;
                int next = 1;

                // If it is leaf node
                if(n_segment_from == 0 && deps.size() == 0) 
                {
                     // q_init
                    if (key[0] == '{')
                    {
                        std::string g = PropagatorCode::get_q_input_idx_from_key(key);
                        if (q_init.find(g) == q_init.end())
                            std::cout << "Could not find q_init[\"" + g + "\"]." << std::endl;
                        for(int i=0; i<M; i++)
                            q_pair[0][i] = q_init[g][i];
                    }
                    else
                    {
                        for(int i=0; i<M; i++)
                            q_pair[0][i] = 1.0;
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
                        for(int i=0; i<M; i++)
                            q_pair[0][i] = 0.0;
                        
                        // Add all propagators at junction if necessary 
                        for(size_t d=0; d<deps.size(); d++)
                        {
                            std::string sub_dep = std::get<0>(deps[d]);
                            int sub_n_segment   = std::get<1>(deps[d]);
                            int sub_n_repeated  = std::get<2>(deps[d]);

                            // Check sub key
                            #ifndef NDEBUG
                            if (propagator_at_check_point.find(std::make_tuple(sub_dep, sub_n_segment)) == propagator_at_check_point.end())
                                std::cout << "Could not find sub key '" + sub_dep + "[" + std::to_string(sub_n_segment) + "]. " << std::endl;
                            if (!propagator_finished[sub_dep][sub_n_segment])
                                std::cout << "Could not compute '" + key +  "', since '"+ sub_dep + std::to_string(sub_n_segment) + "' is not prepared." << std::endl;
                            #endif

                            T* _q_sub = propagator_at_check_point[std::make_tuple(sub_dep, sub_n_segment)];
                            for(int i=0; i<M; i++)
                                q_pair[0][i] += _q_sub[i]*static_cast<double>(sub_n_repeated);
                        }
                        #ifndef NDEBUG
                        propagator_finished[key][0] = true;
                        #endif

                        // std::cout << "finished, key, n: " + key + ", 0" << std::endl;
                    }
                    else if(key[0] == '(')
                    {
                        for(int i=0; i<M; i++)
                            q_pair[0][i] = 1.0;
                        
                        // Multiply all propagators at junction if necessary 
                        for(size_t d=0; d<deps.size(); d++)
                        {
                            std::string sub_dep = std::get<0>(deps[d]);
                            int sub_n_segment   = std::get<1>(deps[d]);
                            int sub_n_repeated  = std::get<2>(deps[d]);

                            // Check sub key
                            #ifndef NDEBUG
                            if (propagator_at_check_point.find(std::make_tuple(sub_dep, sub_n_segment)) == propagator_at_check_point.end())
                                std::cout << "Could not find sub key '" + sub_dep + "[" + std::to_string(sub_n_segment) + "]. " << std::endl;
                            if (!propagator_finished[sub_dep][sub_n_segment])
                                std::cout << "Could not compute '" + key +  "', since '"+ sub_dep + std::to_string(sub_n_segment) + "' is not prepared." << std::endl;
                            #endif

                            T* _q_sub = propagator_at_check_point[std::make_tuple(sub_dep, sub_n_segment)];
                            if (_q_sub == nullptr)
                                std::cout <<"_q_sub is a null pointer: " + sub_dep + "[" + std::to_string(sub_n_segment) + "]." << std::endl;
                            for(int i=0; i<M; i++)
                                q_pair[0][i] *= pow(_q_sub[i], sub_n_repeated);
                        }

                        #ifndef NDEBUG
                        propagator_finished[key][0] = true;
                        #endif

                        // std::cout << "finished, key, n: " + key + ", 0" << std::endl;
                    }
                }
        
                // Multiply mask
                if (n_segment_from == 0 && q_mask != nullptr)
                {
                    for(int i=0; i<M; i++)
                        q_pair[0][i] *= q_mask[i];
                }

                // Copy q_pair[0] to the propagator_at_check_point
                if(n_segment_from == 0)
                {
                    T* _q_target =  propagator_at_check_point[std::make_tuple(key, 0)];
                    for(int i=0; i<M; i++)
                        _q_target[i] = q_pair[0][i];
                }
                else
                {
                    T* _q_from = propagator_at_check_point[std::make_tuple(key, n_segment_from)];
                    for(int i=0; i<M; i++)
                        q_pair[0][i] = _q_from[i];
                }

                // Advance propagator successively
                // Get ds_index from the key
                int ds_index = PropagatorCode::get_ds_index_from_key(key);
                if (ds_index < 1) ds_index = 1;  // Default to global ds

                for(int n=n_segment_from; n<n_segment_to; n++)
                {
                    #ifndef NDEBUG
                    if (!propagator_finished[key][n])
                        std::cout << "unfinished, key: " + key + ", " + std::to_string(n) << std::endl;
                    if (propagator_finished[key][n+1])
                        std::cout << "already finished: " + key + ", " + std::to_string(n) << std::endl;
                    #endif

                    propagator_solver->advance_propagator(
                            q_pair[prev],
                            q_pair[next],
                            monomer_type, q_mask, ds_index);

                    #ifndef NDEBUG
                    propagator_finished[key][n+1] = true;
                    #endif

                    // Copy q_pair[next] to the propagator_at_check_point
                    if (propagator_at_check_point.find(std::make_tuple(key, n+1)) != propagator_at_check_point.end())
                    {
                        T* _q_target =  propagator_at_check_point[std::make_tuple(key, n+1)];
                        for(int i=0; i<M; i++)
                            _q_target[i] = q_pair[next][i];
                    }
                    std::swap(prev, next);
                }

                // Display job info
                #ifndef NDEBUG
                std::cout << job << " finished" << std::endl;
                #endif
            }
        }

        #ifndef NDEBUG
        std::cout << "total partition function 0" << std::endl;
        #endif

        // Compute total partition function of each distinct polymers
        for(const auto& segment_info: single_partition_segment)
        {
            int p      = std::get<0>(segment_info);
            T *q_left  = std::get<1>(segment_info);
            T *q_right = std::get<2>(segment_info);
            int n_aggregated = std::get<3>(segment_info);

            this->single_polymer_partitions[p]= this->cb->inner_product(
                q_left, q_right)/(n_aggregated*this->cb->get_volume());
        }

        #ifndef NDEBUG
        std::cout << "total partition function 1" << std::endl;
        #endif
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
void CpuComputationReduceMemoryContinuous<T>::advance_propagator_single_segment(
    T* q_init, T *q_out, std::string monomer_type)
{
    try
    {
        // const int M = this->cb->get_total_grid();
        // Assign a pointer for mask
        const double *q_mask = this->cb->get_mask();
        propagator_solver->advance_propagator(q_init, q_out, monomer_type, q_mask);

    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
void CpuComputationReduceMemoryContinuous<T>::compute_concentrations()
{
    try
    {
        #ifndef NDEBUG
        std::cout << "compute_concentrations 0" << std::endl;
        #endif

        const int M = this->cb->get_total_grid();

        // Calculate segment concentrations
        for(size_t b=0; b<phi_block.size();b++)
        {
            auto block = phi_block.begin();
            advance(block, b);
            const auto& key = block->first;

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
                    block->second[i] = 0.0;
                continue;
            }

            // Check keys
            #ifndef NDEBUG
            if (propagator_at_check_point.find(std::make_tuple(key_left, n_segment_left-n_segment_right)) == propagator_at_check_point.end())
                std::cout << "Check point at " + key_left + "[" + std::to_string(n_segment_left-n_segment_right) + "] is missing. ";
            if (propagator_at_check_point.find(std::make_tuple(key_right, 0)) == propagator_at_check_point.end())
                std::cout << "Check point at " + key_right + "[" + std::to_string(0) + "] is missing. ";
            #endif

            //     throw_with_line_number("Check point at " + key_left + "[" + std::to_string(n_segment_left-n_segment_right) + "] is missing. ");
            //     throw_with_line_number("Check point at " + key_right + "[" + std::to_string(0) + "] is missing. ");

            // Calculate phi of one block (possibly multiple blocks when using aggregation)
            calculate_phi_one_block(
                block->second,          // phi                       
                key_left,               // dependency v
                key_right,              // dependency u
                n_segment_left,
                n_segment_right,
                monomer_type
            );

            // Normalize concentration
            Polymer& pc = this->molecules->get_polymer(p);
            T norm = (this->molecules->get_ds()*pc.get_volume_fraction()/pc.get_alpha()*n_repeated)/this->single_polymer_partitions[p];
            for(int i=0; i<M; i++)
                block->second[i] *= norm;
        }

        #ifndef NDEBUG
        std::cout << " concentrations of solvents 0" << std::endl;
        #endif

        // Calculate partition functions and concentrations of solvents
        for(int s=0; s<this->molecules->get_n_solvent_types(); s++)
        {
            double volume_fraction   = std::get<0>(this->molecules->get_solvent(s));
            std::string monomer_type = std::get<1>(this->molecules->get_solvent(s));
            
            T *_phi = phi_solvent[s];
            T *_exp_dw = propagator_solver->exp_dw[1][monomer_type].data();

            this->single_solvent_partitions[s] = this->cb->inner_product(_exp_dw, _exp_dw)/this->cb->get_volume();
            for(int i=0; i<M; i++)
                _phi[i] = _exp_dw[i]*_exp_dw[i]*volume_fraction/this->single_solvent_partitions[s];
        }

    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
std::vector<T*> CpuComputationReduceMemoryContinuous<T>::recalcaulte_propagator(std::string key, const int N_START, const int N_RIGHT, std::string monomer_type)
{
    // Recompute propagators from checkpoints for positions [N_START, N_START+N_RIGHT].
    // This function is designed to work within one checkpoint interval (N_RIGHT <= checkpoint_interval).
    // For larger ranges, callers should use block-based computation.
    // Returns pointers where:
    //   - Checkpoint positions point to stored checkpoint arrays
    //   - Non-checkpoint positions point to q_recal workspace [2..checkpoint_interval+2]
    try
    {
        const double *q_mask = this->cb->get_mask();
        int ds_index = PropagatorCode::get_ds_index_from_key(key);
        if (ds_index < 1) ds_index = 1;  // Default to global ds

        // Output array of pointers (will contain mix of checkpoint pointers and q_recal pointers)
        std::vector<T*> q_out(N_RIGHT + 1, nullptr);

        // First pass: link all checkpoint positions
        for(int n=0; n<=N_RIGHT; n++)
        {
            auto it = propagator_at_check_point.find(std::make_tuple(key, N_START+n));
            if(it != propagator_at_check_point.end())
            {
                q_out[n] = it->second;
                #ifndef NDEBUG
                std::cout << "recalculate_propagator: checkpoint at " << key << "[" << N_START+n << "]" << std::endl;
                #endif
            }
        }

        // Find the first checkpoint
        int prev_checkpoint = -1;
        for(int n=0; n<=N_RIGHT; n++)
        {
            if(q_out[n] != nullptr)
            {
                prev_checkpoint = n;
                break;
            }
        }

        if(prev_checkpoint < 0)
        {
            throw_with_line_number("No checkpoint found for key " + key);
        }

        // Second pass: compute non-checkpoint positions sequentially
        // Use indices 2 onwards for storage (0-1 reserved for ping-pong)
        T* q_prev = q_out[prev_checkpoint];
        int ws_idx = 2;

        for(int n = prev_checkpoint + 1; n <= N_RIGHT; n++)
        {
            if(q_out[n] != nullptr)
            {
                // This is a checkpoint, use it as new starting point
                q_prev = q_out[n];
                // Continue with current ws_idx (don't reset)
            }
            else
            {
                // Compute and store in workspace
                if(ws_idx >= static_cast<int>(q_recal.size()))
                {
                    throw_with_line_number("Workspace overflow in recalcaulte_propagator at ws_idx=" +
                        std::to_string(ws_idx) + " (size=" + std::to_string(q_recal.size()) + "). " +
                        "N_RIGHT=" + std::to_string(N_RIGHT) + ", n=" + std::to_string(n) + ". " +
                        "This function should only be called with N_RIGHT <= checkpoint_interval.");
                }
                propagator_solver->advance_propagator(q_prev, q_recal[ws_idx], monomer_type, q_mask, ds_index);
                q_out[n] = q_recal[ws_idx];
                q_prev = q_recal[ws_idx];
                ws_idx++;
            }
        }

        return q_out;
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
void CpuComputationReduceMemoryContinuous<T>::calculate_phi_one_block(
    T *phi,
    std::string key_left, std::string key_right,
    const int N_LEFT, const int N_RIGHT,
    std::string monomer_type)
{
    try
    {
        // Block-based computation to minimize memory usage.
        // For n in [n_start, n_end], q_left positions [N_LEFT - n_end, N_LEFT - n_start] form a forward range.
        // We process in blocks of size checkpoint_interval, recomputing q_left from nearest checkpoint.
        //
        // q_recal layout: [0-1] ping-pong buffers for intermediate steps, [2..] storage for needed positions

        const int M = this->cb->get_total_grid();
        std::vector<double> simpson_rule_coeff = SimpsonRule::get_coeff(N_RIGHT);

        // Assign a pointer for mask
        const double *q_mask = this->cb->get_mask();

        // Get ds_index from keys (left and right should have the same ds_index for the same block)
        int ds_index_left = PropagatorCode::get_ds_index_from_key(key_left);
        if (ds_index_left < 1) ds_index_left = 1;
        int ds_index_right = PropagatorCode::get_ds_index_from_key(key_right);
        if (ds_index_right < 1) ds_index_right = 1;

        // Initialize phi to zero
        for(int i=0; i<M; i++)
            phi[i] = 0.0;

        // Number of blocks
        const int k = checkpoint_interval;
        const int num_blocks = (N_RIGHT + k) / k;

        // Pointers for q_right (ping-pong)
        T *q_right_prev = nullptr;
        T *q_right_curr = nullptr;
        int right_prev_idx = 0;
        int right_next_idx = 1;

        // Current position of q_right
        int current_n_right = -1;

        // Process each block
        for(int block = 0; block < num_blocks; block++)
        {
            const int n_start = block * k;
            const int n_end = std::min((block + 1) * k, N_RIGHT + 1) - 1;  // inclusive

            // q_left positions needed: [N_LEFT - n_end, N_LEFT - n_start] (forward range)
            const int left_start = N_LEFT - n_end;    // First position needed
            const int left_end = N_LEFT - n_start;    // Last position needed

            // Find checkpoint at or before left_start
            int check_pos = 0;  // Start with 0 (always exists)
            for(int cp = checkpoint_interval; cp <= left_start; cp += checkpoint_interval)
            {
                if(propagator_at_check_point.find(std::make_tuple(key_left, cp)) != propagator_at_check_point.end())
                    check_pos = cp;
            }

            #ifndef NDEBUG
            std::cout << "Block " << block << ": n=[" << n_start << "," << n_end << "], "
                      << "left=[" << left_start << "," << left_end << "], check_pos=" << check_pos << std::endl;
            #endif

            // Recompute q_left from check_pos to left_end
            // Skip phase uses q_skip[0-1] for ping-pong, storage uses q_recal[0+]
            const int steps_before = left_start - check_pos;  // Steps before we start storing
            const int storage_count = left_end - left_start + 1;  // Number of values to store

            // Load checkpoint
            T* q_checkpoint = propagator_at_check_point[std::make_tuple(key_left, check_pos)];
            T* q_prev;
            T* q_curr;

            if(steps_before == 0)
            {
                // Checkpoint is exactly at left_start, store directly in q_recal[0]
                for(int i=0; i<M; i++)
                    q_recal[0][i] = q_checkpoint[i];
                q_prev = q_recal[0];
            }
            else
            {
                // Start ping-pong from checkpoint using q_skip[0-1]
                for(int i=0; i<M; i++)
                    q_skip[0][i] = q_checkpoint[i];
                q_prev = q_skip[0];
                int ping_pong = 1;

                // Compute steps before left_start using q_skip ping-pong
                for(int step = 1; step < steps_before; step++)
                {
                    int actual_pos = check_pos + step;
                    q_curr = q_skip[ping_pong];

                    auto it = propagator_at_check_point.find(std::make_tuple(key_left, actual_pos));
                    if(it != propagator_at_check_point.end())
                    {
                        for(int i=0; i<M; i++)
                            q_curr[i] = it->second[i];
                    }
                    else
                    {
                        propagator_solver->advance_propagator(q_prev, q_curr, monomer_type, q_mask, ds_index_left);
                    }
                    q_prev = q_curr;
                    ping_pong = 1 - ping_pong;
                }

                // Compute the step that reaches left_start, store in q_recal[0]
                int actual_pos = check_pos + steps_before;
                auto it = propagator_at_check_point.find(std::make_tuple(key_left, actual_pos));
                if(it != propagator_at_check_point.end())
                {
                    for(int i=0; i<M; i++)
                        q_recal[0][i] = it->second[i];
                }
                else
                {
                    propagator_solver->advance_propagator(q_prev, q_recal[0], monomer_type, q_mask, ds_index_left);
                }
                q_prev = q_recal[0];
            }

            // Compute remaining positions [left_start+1, left_end] and store in q_recal[1+]
            for(int idx = 1; idx < storage_count; idx++)
            {
                int actual_pos = left_start + idx;
                q_curr = q_recal[idx];

                auto it = propagator_at_check_point.find(std::make_tuple(key_left, actual_pos));
                if(it != propagator_at_check_point.end())
                {
                    for(int i=0; i<M; i++)
                        q_curr[i] = it->second[i];
                }
                else
                {
                    propagator_solver->advance_propagator(q_prev, q_curr, monomer_type, q_mask, ds_index_left);
                }
                q_prev = q_curr;
            }

            // Now q_recal[idx] contains q_left[left_start + idx] for idx in [0, storage_count-1]

            // Process each n in [n_start, n_end]
            for(int n = n_start; n <= n_end; n++)
            {
                // Get q_right[n]
                auto it_right = propagator_at_check_point.find(std::make_tuple(key_right, n));
                if(it_right != propagator_at_check_point.end())
                {
                    q_right_curr = it_right->second;
                    q_right_prev = q_right_curr;
                    current_n_right = n;
                }
                else
                {
                    // Advance from current position to n
                    while(current_n_right < n)
                    {
                        T* q_out = this->q_pair[right_next_idx];
                        propagator_solver->advance_propagator(q_right_prev, q_out, monomer_type, q_mask, ds_index_right);
                        q_right_prev = q_out;
                        q_right_curr = q_out;
                        std::swap(right_prev_idx, right_next_idx);
                        current_n_right++;
                    }
                }

                // Get q_left[N_LEFT - n]
                int left_pos = N_LEFT - n;
                int storage_idx = left_pos - left_start;
                T* q_left_n = q_recal[storage_idx];

                // Accumulate phi
                for(int i=0; i<M; i++)
                    phi[i] += simpson_rule_coeff[n] * q_left_n[i] * q_right_curr[i];
            }
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
T CpuComputationReduceMemoryContinuous<T>::get_total_partition(int polymer)
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
void CpuComputationReduceMemoryContinuous<T>::get_total_concentration(std::string monomer_type, T *phi)
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
void CpuComputationReduceMemoryContinuous<T>::get_total_concentration(int p, std::string monomer_type, T *phi)
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
void CpuComputationReduceMemoryContinuous<T>::get_total_concentration_gce(double fugacity, int p, std::string monomer_type, T *phi)
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
void CpuComputationReduceMemoryContinuous<T>::get_block_concentration(int p, T *phi)
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
T CpuComputationReduceMemoryContinuous<T>::get_solvent_partition(int s)
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
void CpuComputationReduceMemoryContinuous<T>::get_solvent_concentration(int s, T *phi)
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
void CpuComputationReduceMemoryContinuous<T>::compute_stress()
{
    // This method should be invoked after invoking compute_statistics().

    // To calculate stress, we multiply weighted fourier basis to q(k)*q^dagger(-k).
    // We only need the real part of stress calculation.

    // Uses block-based computation to minimize memory usage (O(sqrt(N)) workspace).
    try
    {
        if (this->method == "realspace")
            throw_with_line_number("Currently, the real-space method does not support stress computation.");

        const int N_STRESS = 6;  // Full stress tensor: xx, yy, zz, xy, xz, yz
        const int DIM = this->cb->get_dim();
        const int M    = this->cb->get_total_grid();
        const int k = checkpoint_interval;

        std::map<std::tuple<int, std::string, std::string>, std::array<T,6>> block_dq_dl;

        // Assign a pointer for mask
        const double *q_mask = this->cb->get_mask();

        // Reset stress map
        for(const auto& item: phi_block)
        {
            for(int d=0; d<N_STRESS; d++)
                block_dq_dl[item.first][d] = 0.0;
        }

        // Compute stress for each block
        for(size_t b=0; b<phi_block.size();b++)
        {
            auto block = phi_block.begin();
            advance(block, b);
            const auto& key   = block->first;

            std::string key_left  = std::get<1>(key);
            std::string key_right = std::get<2>(key);

            const int N_RIGHT = this->propagator_computation_optimizer->get_computation_block(key).n_segment_right;
            const int N_LEFT  = this->propagator_computation_optimizer->get_computation_block(key).n_segment_left;
            std::string monomer_type = this->propagator_computation_optimizer->get_computation_block(key).monomer_type;
            int n_repeated = this->propagator_computation_optimizer->get_computation_block(key).n_repeated;

            // Get ds_index from keys
            int ds_index_left = PropagatorCode::get_ds_index_from_key(key_left);
            if (ds_index_left < 1) ds_index_left = 1;
            int ds_index_right = PropagatorCode::get_ds_index_from_key(key_right);
            if (ds_index_right < 1) ds_index_right = 1;

            // If there is no segment
            if(N_RIGHT == 0)
                continue;

            std::vector<double> s_coeff = SimpsonRule::get_coeff(N_RIGHT);
            std::array<T,6> _block_dq_dl = block_dq_dl[key];

            // Number of blocks
            const int num_blocks = (N_RIGHT + k) / k;

            // Pointers for q_right (ping-pong)
            T *q_right_prev = nullptr;
            T *q_right_curr = nullptr;
            int right_prev_idx = 0;
            int right_next_idx = 1;

            // Current position of q_right
            int current_n_right = -1;

            // Process each block
            for(int blk = 0; blk < num_blocks; blk++)
            {
                const int n_start = blk * k;
                const int n_end = std::min((blk + 1) * k, N_RIGHT + 1) - 1;  // inclusive

                // q_left positions needed: [N_LEFT - n_end, N_LEFT - n_start] (forward range)
                const int left_start = N_LEFT - n_end;    // First position needed
                const int left_end = N_LEFT - n_start;    // Last position needed

                // Find the best checkpoint at or before left_start
                int check_pos = -1;
                for(int cp = 0; cp <= left_start; cp++)
                {
                    if(propagator_at_check_point.find(std::make_tuple(key_left, cp)) != propagator_at_check_point.end())
                        check_pos = cp;
                }

                if(check_pos < 0)
                {
                    // No checkpoint found, skip this block
                    continue;
                }

                // Recompute q_left from check_pos to left_end using q_recal workspace
                const int steps_before = left_start - check_pos;
                const int storage_count = left_end - left_start + 1;

                // Load checkpoint - use find() to be safe
                auto it_checkpoint = propagator_at_check_point.find(std::make_tuple(key_left, check_pos));
                if(it_checkpoint == propagator_at_check_point.end())
                    continue;
                T* q_checkpoint = it_checkpoint->second;
                T* q_prev_left;
                T* q_curr_left;

                // Skip phase uses q_skip[0-1] for ping-pong, storage uses q_recal[0+]
                if(steps_before == 0)
                {
                    for(int i=0; i<M; i++)
                        q_recal[0][i] = q_checkpoint[i];
                    q_prev_left = q_recal[0];
                }
                else
                {
                    for(int i=0; i<M; i++)
                        q_skip[0][i] = q_checkpoint[i];
                    q_prev_left = q_skip[0];
                    int ping_pong = 1;

                    for(int step = 1; step < steps_before; step++)
                    {
                        int actual_pos = check_pos + step;
                        q_curr_left = q_skip[ping_pong];

                        auto it = propagator_at_check_point.find(std::make_tuple(key_left, actual_pos));
                        if(it != propagator_at_check_point.end())
                        {
                            for(int i=0; i<M; i++)
                                q_curr_left[i] = it->second[i];
                        }
                        else
                        {
                            propagator_solver->advance_propagator(q_prev_left, q_curr_left, monomer_type, q_mask, ds_index_left);
                        }
                        q_prev_left = q_curr_left;
                        ping_pong = 1 - ping_pong;
                    }

                    int actual_pos = check_pos + steps_before;
                    auto it = propagator_at_check_point.find(std::make_tuple(key_left, actual_pos));
                    if(it != propagator_at_check_point.end())
                    {
                        for(int i=0; i<M; i++)
                            q_recal[0][i] = it->second[i];
                    }
                    else
                    {
                        propagator_solver->advance_propagator(q_prev_left, q_recal[0], monomer_type, q_mask, ds_index_left);
                    }
                    q_prev_left = q_recal[0];
                }

                // Compute remaining positions [left_start+1, left_end] and store in q_recal[1+]
                for(int idx = 1; idx < storage_count; idx++)
                {
                    int actual_pos = left_start + idx;
                    q_curr_left = q_recal[idx];

                    auto it = propagator_at_check_point.find(std::make_tuple(key_left, actual_pos));
                    if(it != propagator_at_check_point.end())
                    {
                        for(int i=0; i<M; i++)
                            q_curr_left[i] = it->second[i];
                    }
                    else
                    {
                        propagator_solver->advance_propagator(q_prev_left, q_curr_left, monomer_type, q_mask, ds_index_left);
                    }
                    q_prev_left = q_curr_left;
                }

                // Process each n in [n_start, n_end]
                for(int n = n_start; n <= n_end; n++)
                {
                    // Get q_right[n]
                    auto it_right = propagator_at_check_point.find(std::make_tuple(key_right, n));
                    if(it_right != propagator_at_check_point.end())
                    {
                        q_right_curr = it_right->second;
                        q_right_prev = q_right_curr;
                        current_n_right = n;
                    }
                    else
                    {
                        while(current_n_right < n)
                        {
                            T* q_out = this->q_pair[right_next_idx];
                            propagator_solver->advance_propagator(q_right_prev, q_out, monomer_type, q_mask, ds_index_right);
                            q_right_prev = q_out;
                            q_right_curr = q_out;
                            std::swap(right_prev_idx, right_next_idx);
                            current_n_right++;
                        }
                    }

                    // Get q_left[N_LEFT - n]
                    int left_pos = N_LEFT - n;
                    int storage_idx = left_pos - left_start;
                    T* q_left_n = q_recal[storage_idx];

                    // Compute stress contribution
                    std::vector<T> segment_stress = propagator_solver->compute_single_segment_stress(
                        q_left_n, q_right_curr, monomer_type, false);

                    for(int d=0; d<N_STRESS; d++)
                        _block_dq_dl[d] += segment_stress[d]*(s_coeff[n]*n_repeated);
                }
            }
            block_dq_dl[key] = _block_dq_dl;
        }

        // Compute total stress
        int n_polymer_types = this->molecules->get_n_polymer_types();
        for(int p=0; p<n_polymer_types; p++)
            for(int d=0; d<N_STRESS; d++)
                this->dq_dl[p][d] = 0.0;
        for(const auto& block: phi_block)
        {
            const auto& key       = block.first;
            int p                 = std::get<0>(key);
            for(int d=0; d<N_STRESS; d++)
                this->dq_dl[p][d] += block_dq_dl[key][d];
        }
        // Normalize stress components
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
void CpuComputationReduceMemoryContinuous<T>::get_chain_propagator(T *q_out, int polymer, int v, int u, int n)
{
    // This method should be invoked after invoking compute_statistics()

    // Get chain propagator for a selected polymer, block and direction.
    // Uses O(sqrt(N)) workspace by computing from nearest checkpoint.
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
            int check_pos = 0;  // Start with 0 (always exists)
            for(int cp = checkpoint_interval; cp < n; cp += checkpoint_interval)
            {
                if(propagator_at_check_point.find(std::make_tuple(dep, cp)) != propagator_at_check_point.end())
                    check_pos = cp;
            }

            // Recalculate from checkpoint to position n
            std::string monomer_type = this->propagator_computation_optimizer->get_computation_propagator(dep).monomer_type;
            const double *q_mask = this->cb->get_mask();
            int ds_index = PropagatorCode::get_ds_index_from_key(dep);
            if (ds_index < 1) ds_index = 1;

            // Load checkpoint
            T* q_checkpoint = propagator_at_check_point[std::make_tuple(dep, check_pos)];

            // Use ping-pong buffers to compute to position n
            for(int i=0; i<M; i++)
                q_recal[0][i] = q_checkpoint[i];

            T* q_prev = q_recal[0];
            int ping_pong = 1;

            for(int pos = check_pos + 1; pos <= n; pos++)
            {
                // Check if this position is a checkpoint
                auto it = propagator_at_check_point.find(std::make_tuple(dep, pos));
                if(it != propagator_at_check_point.end())
                {
                    q_prev = it->second;
                }
                else
                {
                    T* q_curr = q_recal[ping_pong];
                    propagator_solver->advance_propagator(q_prev, q_curr, monomer_type, q_mask, ds_index);
                    q_prev = q_curr;
                    ping_pong = 1 - ping_pong;
                }
            }

            // Copy result to output
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
bool CpuComputationReduceMemoryContinuous<T>::check_total_partition()
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

    // Assign a pointer for mask
    const double *q_mask = this->cb->get_mask();

    for(const auto& block: phi_block)
    {
        const auto& key = block.first;
        int p                 = std::get<0>(key);
        std::string key_left  = std::get<1>(key);
        std::string key_right = std::get<2>(key);

        std::string monomer_type = this->propagator_computation_optimizer->get_computation_block(key).monomer_type;
        const int N_RIGHT        = this->propagator_computation_optimizer->get_computation_block(key).n_segment_right;
        const int N_LEFT         = this->propagator_computation_optimizer->get_computation_block(key).n_segment_left;
        int n_repeated           = this->propagator_computation_optimizer->get_computation_block(key).n_repeated;
        int n_propagators        = this->propagator_computation_optimizer->get_computation_block(key).v_u.size();

        // Get ds_index from keys
        int ds_index_left = PropagatorCode::get_ds_index_from_key(key_left);
        if (ds_index_left < 1) ds_index_left = 1;
        int ds_index_right = PropagatorCode::get_ds_index_from_key(key_right);
        if (ds_index_right < 1) ds_index_right = 1;

        #ifndef NDEBUG
        std::cout<< p << ", " << key_left << ", " << key_right << ": " << N_LEFT << ", " << N_RIGHT << ", " << n_propagators << ", " << n_repeated << std::endl;
        #endif

        // Number of blocks
        const int num_blocks = (N_RIGHT + k) / k;

        // Pointers for q_right (ping-pong)
        T *q_right_prev = nullptr;
        T *q_right_curr = nullptr;
        int right_prev_idx = 0;
        int right_next_idx = 1;

        // Current position of q_right
        int current_n_right = -1;

        // Process each block
        for(int blk = 0; blk < num_blocks; blk++)
        {
            const int n_start = blk * k;
            const int n_end = std::min((blk + 1) * k, N_RIGHT + 1) - 1;  // inclusive

            // q_left positions needed: [N_LEFT - n_end, N_LEFT - n_start] (forward range)
            const int left_start = N_LEFT - n_end;    // First position needed
            const int left_end = N_LEFT - n_start;    // Last position needed

            // Find the best checkpoint at or before left_start
            int check_pos = -1;
            for(int cp = 0; cp <= left_start; cp++)
            {
                if(propagator_at_check_point.find(std::make_tuple(key_left, cp)) != propagator_at_check_point.end())
                    check_pos = cp;
            }

            if(check_pos < 0)
            {
                // No checkpoint found, skip this block
                continue;
            }

            // Recompute q_left from check_pos to left_end using q_recal workspace
            const int steps_before = left_start - check_pos;
            const int storage_count = left_end - left_start + 1;

            // Load checkpoint - use find() to be safe
            auto it_checkpoint = propagator_at_check_point.find(std::make_tuple(key_left, check_pos));
            if(it_checkpoint == propagator_at_check_point.end())
                continue;
            T* q_checkpoint = it_checkpoint->second;
            T* q_prev_left;
            T* q_curr_left;

            // Skip phase uses q_skip[0-1] for ping-pong, storage uses q_recal[0+]
            if(steps_before == 0)
            {
                for(int i=0; i<M; i++)
                    q_recal[0][i] = q_checkpoint[i];
                q_prev_left = q_recal[0];
            }
            else
            {
                for(int i=0; i<M; i++)
                    q_skip[0][i] = q_checkpoint[i];
                q_prev_left = q_skip[0];
                int ping_pong = 1;

                for(int step = 1; step < steps_before; step++)
                {
                    int actual_pos = check_pos + step;
                    q_curr_left = q_skip[ping_pong];

                    auto it = propagator_at_check_point.find(std::make_tuple(key_left, actual_pos));
                    if(it != propagator_at_check_point.end())
                    {
                        for(int i=0; i<M; i++)
                            q_curr_left[i] = it->second[i];
                    }
                    else
                    {
                        propagator_solver->advance_propagator(q_prev_left, q_curr_left, monomer_type, q_mask, ds_index_left);
                    }
                    q_prev_left = q_curr_left;
                    ping_pong = 1 - ping_pong;
                }

                int actual_pos = check_pos + steps_before;
                auto it = propagator_at_check_point.find(std::make_tuple(key_left, actual_pos));
                if(it != propagator_at_check_point.end())
                {
                    for(int i=0; i<M; i++)
                        q_recal[0][i] = it->second[i];
                }
                else
                {
                    propagator_solver->advance_propagator(q_prev_left, q_recal[0], monomer_type, q_mask, ds_index_left);
                }
                q_prev_left = q_recal[0];
            }

            // Compute remaining positions [left_start+1, left_end] and store in q_recal[1+]
            for(int idx = 1; idx < storage_count; idx++)
            {
                int actual_pos = left_start + idx;
                q_curr_left = q_recal[idx];

                auto it = propagator_at_check_point.find(std::make_tuple(key_left, actual_pos));
                if(it != propagator_at_check_point.end())
                {
                    for(int i=0; i<M; i++)
                        q_curr_left[i] = it->second[i];
                }
                else
                {
                    propagator_solver->advance_propagator(q_prev_left, q_curr_left, monomer_type, q_mask, ds_index_left);
                }
                q_prev_left = q_curr_left;
            }

            // Process each n in [n_start, n_end]
            for(int n = n_start; n <= n_end; n++)
            {
                // Get q_right[n]
                auto it_right = propagator_at_check_point.find(std::make_tuple(key_right, n));
                if(it_right != propagator_at_check_point.end())
                {
                    q_right_curr = it_right->second;
                    q_right_prev = q_right_curr;
                    current_n_right = n;
                }
                else
                {
                    while(current_n_right < n)
                    {
                        T* q_out = this->q_pair[right_next_idx];
                        propagator_solver->advance_propagator(q_right_prev, q_out, monomer_type, q_mask, ds_index_right);
                        q_right_prev = q_out;
                        q_right_curr = q_out;
                        std::swap(right_prev_idx, right_next_idx);
                        current_n_right++;
                    }
                }

                // Get q_left[N_LEFT - n]
                int left_pos = N_LEFT - n;
                int storage_idx = left_pos - left_start;
                T* q_left_n = q_recal[storage_idx];

                // Compute partition at this position
                T total_partition = this->cb->inner_product(
                    q_left_n, q_right_curr)*(n_repeated/this->cb->get_volume());

                total_partition /= n_propagators;
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
#include "TemplateInstantiations.h"
INSTANTIATE_CLASS(CpuComputationReduceMemoryContinuous);