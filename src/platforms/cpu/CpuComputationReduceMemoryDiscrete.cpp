/**
 * @file CpuComputationReduceMemoryDiscrete.cpp
 * @brief Memory-efficient CPU propagator computation with checkpointing for discrete chains.
 *
 * Implements propagator computation for discrete chains with reduced
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
 * **Discrete Chain Specifics:**
 *
 * - Handles half-bond steps at junction points
 * - Uses discrete sum for concentration (not Simpson's rule)
 * - Proper Boltzmann factor weighting for partition function
 *
 * **Performance:**
 *
 * Approximately 2-4x slower than full storage mode due to recomputation.
 * Single-threaded to minimize memory usage during recomputation.
 *
 * **Template Instantiations:**
 *
 * - CpuComputationReduceMemoryDiscrete<double>: Real fields
 * - CpuComputationReduceMemoryDiscrete<std::complex<double>>: Complex fields
 *
 * @see CpuComputationDiscrete for full storage version
 */

#include <cmath>
#include <numbers>
#include <omp.h>

#include "CpuComputationReduceMemoryDiscrete.h"
#include "CpuSolverPseudoDiscrete.h"

/**
 * @brief Construct memory-efficient propagator computation for discrete chains.
 *
 * Allocates checkpoint storage and temporary arrays for recomputation.
 * Uses single thread to minimize memory during propagator advancement.
 *
 * @param cb                             Computation box for grid operations
 * @param molecules                      Polymer/solvent species definitions
 * @param propagator_computation_optimizer Optimized computation schedule
 */
template <typename T>
CpuComputationReduceMemoryDiscrete<T>::CpuComputationReduceMemoryDiscrete(
    ComputationBox<T>* cb,
    Molecules *molecules,
    PropagatorComputationOptimizer *propagator_computation_optimizer)
    : PropagatorComputation<T>(cb, molecules, propagator_computation_optimizer)
{
    try
    {
        #ifndef NDEBUG
        std::cout << "--------- Discrete Chain Solver (Reduce Memory), CPU Version ---------" << std::endl;
        #endif

        const int M = this->cb->get_total_grid();

        this->propagator_solver = new CpuSolverPseudoDiscrete<T>(cb, molecules);

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
            phi_block[item.first] = new T[M];
        }

        // Allocate memory for check points
        if( this->propagator_computation_optimizer->get_computation_propagators().size() == 0)
            throw_with_line_number("There is no propagator code. Add polymers first.");
        for(const auto& item: this->propagator_computation_optimizer->get_computation_propagators())
        {
            std::string key = item.first;
            auto& deps = this->propagator_computation_optimizer->get_computation_propagator(key).deps;
            int max_n_segment = this->propagator_computation_optimizer->get_computation_propagator(key).max_n_segment;

            // Allocate check point at segment 1 (first segment for discrete chains)
            if(propagator_at_check_point.find(std::make_tuple(key, 1)) == propagator_at_check_point.end())
                propagator_at_check_point[std::make_tuple(key, 1)] = new T[M];

            // Add 'max_n_segment' to pass the tests
            if(propagator_at_check_point.find(std::make_tuple(key, max_n_segment)) == propagator_at_check_point.end())
                propagator_at_check_point[std::make_tuple(key, max_n_segment)] = new T[M];

            // Allocate half-bond step at segment 0 if there are dependencies (for junction start)
            if (deps.size() > 0)
            {
                if(propagator_half_steps_at_check_point.find(std::make_tuple(key, 0)) == propagator_half_steps_at_check_point.end())
                    propagator_half_steps_at_check_point[std::make_tuple(key, 0)] = new T[M];
            }

            // Allocate check points at dependency positions
            for(size_t d=0; d<deps.size(); d++)
            {
                std::string sub_dep = std::get<0>(deps[d]);
                int sub_n_segment   = std::get<1>(deps[d]);

                if (sub_n_segment == 0)
                {
                    // Allocate half-bond step at junction
                    if(propagator_half_steps_at_check_point.find(std::make_tuple(sub_dep, 0)) == propagator_half_steps_at_check_point.end())
                        propagator_half_steps_at_check_point[std::make_tuple(sub_dep, 0)] = new T[M];
                }
                else
                {
                    if(propagator_at_check_point.find(std::make_tuple(sub_dep, sub_n_segment)) == propagator_at_check_point.end())
                        propagator_at_check_point[std::make_tuple(sub_dep, sub_n_segment)] = new T[M];
                }
            }

            // Allocate half-bond steps at junction_ends
            for (int n: item.second.junction_ends)
            {
                if(propagator_half_steps_at_check_point.find(std::make_tuple(key, n)) == propagator_half_steps_at_check_point.end())
                    propagator_half_steps_at_check_point[std::make_tuple(key, n)] = new T[M];
            }

            #ifndef NDEBUG
            propagator_finished[key] = new bool[max_n_segment+1];
            for(int i=0; i<=max_n_segment;i++)
                propagator_finished[key][i] = false;
            for (int n: item.second.junction_ends)
                propagator_half_steps_finished[key][n] = false;
            propagator_half_steps_finished[key][0] = false;
            #endif
        }

        // Iterate through each element in the map
        #ifndef NDEBUG
        std::cout << "propagator_at_check_point:" << std::endl;
        for (const auto& pair : propagator_at_check_point) {
            std::cout << "Key: " << std::get<0>(pair.first) << ", Value: " << std::get<1>(pair.first) << std::endl;
        }
        std::cout << "propagator_half_steps_at_check_point:" << std::endl;
        for (const auto& pair : propagator_half_steps_at_check_point) {
            std::cout << "Key: " << std::get<0>(pair.first) << ", Value: " << std::get<1>(pair.first) << std::endl;
        }
        #endif

        // Find the total maximum n_segment and allocate temporary memory for recalculating propagators
        // Need max of both n_segment_left and n_segment_right for stress calculation
        total_max_n_segment = 0;
        for(const auto& block: phi_block)
        {
            const auto& key = block.first;
            int n_segment_right = this->propagator_computation_optimizer->get_computation_block(key).n_segment_right;
            int n_segment_left = this->propagator_computation_optimizer->get_computation_block(key).n_segment_left;
            total_max_n_segment = std::max(total_max_n_segment, std::max(n_segment_right, n_segment_left));
        }
        this->q_recal.resize(total_max_n_segment+1);
        for(int n=0; n<=total_max_n_segment; n++)
            this->q_recal[n] = new T[M];
        this->q_pair[0] = new T[M];
        this->q_pair[1] = new T[M];

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
            std::string monomer_type = this->propagator_computation_optimizer->get_computation_block(key).monomer_type;

            // Skip if n_segment_left is 0
            if (n_segment_left == 0)
                continue;

            if(propagator_at_check_point.find(std::make_tuple(key_left, n_segment_left)) == propagator_at_check_point.end())
            {
                propagator_at_check_point[std::make_tuple(key_left, n_segment_left)] = new T[M];
                #ifndef NDEBUG
                std::cout << "Allocated, " + key_left + ", " << n_segment_left << std::endl;
                #endif
            }

            if(propagator_at_check_point.find(std::make_tuple(key_right, 1)) == propagator_at_check_point.end())
            {
                propagator_at_check_point[std::make_tuple(key_right, 1)] = new T[M];
                #ifndef NDEBUG
                std::cout << "Allocated, " + key_right + ", " << 1 << std::endl;
                #endif
            }

            single_partition_segment.push_back(std::make_tuple(
                p,
                propagator_at_check_point[std::make_tuple(key_left, n_segment_left)],  // q
                propagator_at_check_point[std::make_tuple(key_right, 1)],              // q_dagger
                monomer_type,
                n_aggregated                                                           // how many propagators are aggregated
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
CpuComputationReduceMemoryDiscrete<T>::~CpuComputationReduceMemoryDiscrete()
{
    delete propagator_solver;
    delete sc;

    for(int n=0; n<=total_max_n_segment; n++)
        delete[] this->q_recal[n];

    delete[] this->q_pair[0];
    delete[] this->q_pair[1];

    for(const auto& item: propagator_at_check_point)
        delete[] item.second;
    for(const auto& item: propagator_half_steps_at_check_point)
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
void CpuComputationReduceMemoryDiscrete<T>::update_laplacian_operator()
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
void CpuComputationReduceMemoryDiscrete<T>::compute_statistics(
    std::map<std::string, const T*> w_input,
    std::map<std::string, const T*> q_init)
{
    this->compute_propagators(w_input, q_init);
    this->compute_concentrations();
}
template <typename T>
void CpuComputationReduceMemoryDiscrete<T>::compute_propagators(
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
            for (int n: item.second.junction_ends)
                propagator_half_steps_finished[key][n] = false;
            propagator_half_steps_finished[key][0] = false;
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
                const T *_exp_dw = propagator_solver->exp_dw[monomer_type];

                // Display job info
                #ifndef NDEBUG
                std::cout << job << " started" << std::endl;
                #endif

                int prev = 0;
                int next = 1;

                // Calculate one block end
                if (n_segment_from == 0 && deps.size() == 0) // if it is leaf node
                {
                    // q_init
                    if (key[0] == '{')
                    {
                        std::string g = PropagatorCode::get_q_input_idx_from_key(key);
                        if (q_init.find(g) == q_init.end())
                            std::cout << "Could not find q_init[\"" + g + "\"]." << std::endl;
                        for(int i=0; i<M; i++)
                            q_pair[0][i] = q_init[g][i]*_exp_dw[i];
                    }
                    else
                    {
                        for(int i=0; i<M; i++)
                            q_pair[0][i] = _exp_dw[i];
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
                        for(int i=0; i<M; i++)
                            q_pair[0][i] = 0.0;

                        for(size_t d=0; d<deps.size(); d++)
                        {
                            std::string sub_dep = std::get<0>(deps[d]);
                            int sub_n_segment   = std::get<1>(deps[d]);
                            int sub_n_repeated  = std::get<2>(deps[d]);

                            T* _q_sub;
                            if (sub_n_segment == 0)
                            {
                                // Check sub key
                                #ifndef NDEBUG
                                if (propagator_half_steps_at_check_point.find(std::make_tuple(sub_dep, 0)) == propagator_half_steps_at_check_point.end())
                                    std::cout << "Could not find sub key '" + sub_dep + "[0]+1/2'. " << std::endl;
                                if (!propagator_half_steps_finished[sub_dep][0])
                                    std::cout << "Could not compute '" + key +  "', since '"+ sub_dep + "0+1/2' is not prepared." << std::endl;
                                #endif

                                _q_sub = propagator_half_steps_at_check_point[std::make_tuple(sub_dep, 0)];
                            }
                            else
                            {
                                // Check sub key
                                #ifndef NDEBUG
                                if (propagator_at_check_point.find(std::make_tuple(sub_dep, sub_n_segment)) == propagator_at_check_point.end())
                                    std::cout << "Could not find sub key '" + sub_dep + "[" + std::to_string(sub_n_segment) + "]. " << std::endl;
                                if (!propagator_finished[sub_dep][sub_n_segment])
                                    std::cout << "Could not compute '" + key +  "', since '"+ sub_dep + std::to_string(sub_n_segment) + "' is not prepared." << std::endl;
                                #endif

                                _q_sub = propagator_at_check_point[std::make_tuple(sub_dep, sub_n_segment)];
                            }
                            for(int i=0; i<M; i++)
                                q_pair[0][i] += _q_sub[i]*static_cast<double>(sub_n_repeated);
                        }

                        // If n_segments of all deps are 0
                        if (std::get<1>(deps[0]) == 0)
                        {
                            T *_propagator_half_step = propagator_half_steps_at_check_point[std::make_tuple(key, 0)];
                            for(int i=0; i<M; i++)
                                _propagator_half_step[i] = q_pair[0][i];

                            #ifndef NDEBUG
                            propagator_half_steps_finished[key][0] = true;
                            #endif

                            // Add half bond
                            propagator_solver->advance_propagator_half_bond_step(
                                q_pair[0], q_pair[0], monomer_type);

                            // Add full segment
                            for(int i=0; i<M; i++)
                                q_pair[0][i] *= _exp_dw[i];
                        }
                        else
                        {
                            propagator_solver->advance_propagator(
                                q_pair[0],
                                q_pair[0],
                                monomer_type,
                                q_mask);
                        }

                        #ifndef NDEBUG
                        propagator_finished[key][1] = true;
                        #endif
                    }
                    else if(key[0] == '(')
                    {
                        // Combine branches
                        T *_q_junction_start = propagator_half_steps_at_check_point[std::make_tuple(key, 0)];
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

                            T *_propagator_half_step = propagator_half_steps_at_check_point[std::make_tuple(sub_dep, sub_n_segment)];
                            for(int i=0; i<M; i++)
                                _q_junction_start[i] *= pow(_propagator_half_step[i], sub_n_repeated);
                        }

                        #ifndef NDEBUG
                        propagator_half_steps_finished[key][0] = true;
                        #endif

                        if (n_segment_to > 0)
                        {
                            // Add half bond
                            propagator_solver->advance_propagator_half_bond_step(
                                _q_junction_start, q_pair[0], monomer_type);

                            // Add full segment
                            for(int i=0; i<M; i++)
                                q_pair[0][i] *= _exp_dw[i];

                            #ifndef NDEBUG
                            propagator_finished[key][1] = true;
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
                            q_pair[0][i] *= q_mask[i];
                    }

                    // Store q(r, 1) at check point
                    if (propagator_at_check_point.find(std::make_tuple(key, 1)) != propagator_at_check_point.end())
                    {
                        T* _q_target = propagator_at_check_point[std::make_tuple(key, 1)];
                        for(int i=0; i<M; i++)
                            _q_target[i] = q_pair[0][i];
                    }

                    // q(r, 1+1/2) at junction_ends
                    auto it = propagator_half_steps_at_check_point.find(std::make_tuple(key, 1));
                    if (it != propagator_half_steps_at_check_point.end())
                    {
                        propagator_solver->advance_propagator_half_bond_step(
                            q_pair[0],
                            it->second,
                            monomer_type);

                        #ifndef NDEBUG
                        propagator_half_steps_finished[key][1] = true;
                        #endif
                    }
                    n_segment_from++;
                }
                else
                {
                    // Load from check point
                    T* _q_from = propagator_at_check_point[std::make_tuple(key, n_segment_from)];
                    for(int i=0; i<M; i++)
                        q_pair[0][i] = _q_from[i];
                }

                // Advance propagator successively
                for(int n=n_segment_from; n<n_segment_to; n++)
                {
                    #ifndef NDEBUG
                    if (!propagator_finished[key][n])
                        std::cout << "unfinished, key: " + key + ", " + std::to_string(n) << std::endl;
                    if (propagator_finished[key][n+1])
                        std::cout << "already finished: " + key + ", " + std::to_string(n+1) << std::endl;
                    #endif

                    propagator_solver->advance_propagator(
                            q_pair[prev],
                            q_pair[next],
                            monomer_type, q_mask);

                    #ifndef NDEBUG
                    propagator_finished[key][n+1] = true;
                    #endif

                    // Copy q_pair[next] to the propagator_at_check_point
                    if (propagator_at_check_point.find(std::make_tuple(key, n+1)) != propagator_at_check_point.end())
                    {
                        T* _q_target = propagator_at_check_point[std::make_tuple(key, n+1)];
                        for(int i=0; i<M; i++)
                            _q_target[i] = q_pair[next][i];
                    }

                    // q(r, n+1+1/2) at junction_ends
                    auto it = propagator_half_steps_at_check_point.find(std::make_tuple(key, n+1));
                    if (it != propagator_half_steps_at_check_point.end())
                    {
                        propagator_solver->advance_propagator_half_bond_step(
                            q_pair[next],
                            it->second,
                            monomer_type);

                        #ifndef NDEBUG
                        propagator_half_steps_finished[key][n+1] = true;
                        #endif
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
            int p                    = std::get<0>(segment_info);
            T *propagator_left       = std::get<1>(segment_info);
            T *propagator_right      = std::get<2>(segment_info);
            std::string monomer_type = std::get<3>(segment_info);
            int n_aggregated         = std::get<4>(segment_info);
            const T *_exp_dw         = propagator_solver->exp_dw[monomer_type];

            this->single_polymer_partitions[p]= this->cb->inner_product_inverse_weight(
                propagator_left, propagator_right, _exp_dw)/(n_aggregated*this->cb->get_volume());
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
void CpuComputationReduceMemoryDiscrete<T>::advance_propagator_single_segment(
    T* q_init, T *q_out, std::string monomer_type)
{
    try
    {
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
void CpuComputationReduceMemoryDiscrete<T>::compute_concentrations()
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
            T *_exp_dw = propagator_solver->exp_dw[monomer_type];

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
std::vector<T*> CpuComputationReduceMemoryDiscrete<T>::recalculate_propagator(std::string key, const int N_START, const int N_RIGHT, std::string monomer_type)
{
    // If a propagator is in propagator_at_check_point reuse it, otherwise compute it again with allocated memory space.
    // This function returns q_out where q_out[n] corresponds to position N_START+n for n=0..N_RIGHT.
    try
    {
        // Assign a pointer for mask
        const double *q_mask = this->cb->get_mask();

        // An array of pointers for q_out - sized to accommodate N_RIGHT elements
        std::vector<T*> q_out(N_RIGHT + 1);

        // Try to get starting position from checkpoint
        auto it_start = propagator_at_check_point.find(std::make_tuple(key, N_START));
        if (it_start != propagator_at_check_point.end())
        {
            q_out[0] = it_start->second;
        }

        // Compute the q_out for n=1..N_RIGHT
        for(int n=1; n<=N_RIGHT; n++)
        {
            // Use propagator_at_check_point if exists
            auto it = propagator_at_check_point.find(std::make_tuple(key, N_START+n));
            if(it != propagator_at_check_point.end())
            {
                q_out[n] = it->second;
                #ifndef NDEBUG
                std::cout << "Use propagator_at_check_point if exists: (phi, left) " << key << ", " << N_START+n << std::endl;
                #endif
            }
            // Assign q_recal memory space, and compute the next propagator from the previous position
            else if (q_out[n-1] != nullptr)
            {
                q_out[n] = this->q_recal[n];
                propagator_solver->advance_propagator(q_out[n-1], q_out[n], monomer_type, q_mask);
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
void CpuComputationReduceMemoryDiscrete<T>::calculate_phi_one_block(
    T *phi,
    std::string key_left, std::string key_right,
    const int N_LEFT, const int N_RIGHT,
    std::string monomer_type)
{
    try
    {
        // In this method, propagators are recalculated from the check points.
        // If a propagator is in propagator_at_check_point reuse it, otherwise compute it again with allocated memory space.
        const int M = this->cb->get_total_grid();
        const T *_exp_dw = propagator_solver->exp_dw[monomer_type];

        // Assign a pointer for mask
        const double *q_mask = this->cb->get_mask();

        // Recalculate q_left from the check point
        std::vector<T*> q_left = recalculate_propagator(key_left, N_LEFT-N_RIGHT, N_RIGHT, monomer_type);

        // Pointers for q_right
        T *q_prev = nullptr;
        T *q_next = nullptr;
        int prev = 0;
        int next = 1;

        // Compute the q_right and segment concentration (discrete sum, not Simpson's rule)
        for(int i=0; i<M; i++)
            phi[i] = 0.0;
        for(int n=1; n<=N_RIGHT; n++)
        {
            // Use propagator_at_check_point if exists
            auto it = propagator_at_check_point.find(std::make_tuple(key_right, n));
            if(it != propagator_at_check_point.end())
            {
                q_next = it->second;
                #ifndef NDEBUG
                std::cout << "Use propagator_at_check_point if exists: (phi, right) " << key_right << ", " <<  n << std::endl;
                #endif
            }
            // Assign this->q_pair memory space, and compute the next propagator from the check point
            else if (n > 1)
            {
                q_next = this->q_pair[prev];
                propagator_solver->advance_propagator(q_prev, q_next, monomer_type, q_mask);
                std::swap(prev, next);
            }

            // Add contribution: q_left[N_LEFT-n+1] * q_right[n]
            for(int i=0; i<M; i++)
                phi[i] += q_left[N_RIGHT-n+1][i]*q_next[i];

            q_prev = q_next;
            q_next = nullptr;
        }

        // Divide by exp_dw for proper normalization
        for(int i=0; i<M; i++)
            phi[i] /= _exp_dw[i];
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
T CpuComputationReduceMemoryDiscrete<T>::get_total_partition(int polymer)
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
void CpuComputationReduceMemoryDiscrete<T>::get_total_concentration(std::string monomer_type, T *phi)
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
void CpuComputationReduceMemoryDiscrete<T>::get_total_concentration(int p, std::string monomer_type, T *phi)
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
void CpuComputationReduceMemoryDiscrete<T>::get_total_concentration_gce(double fugacity, int p, std::string monomer_type, T *phi)
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
void CpuComputationReduceMemoryDiscrete<T>::get_block_concentration(int p, T *phi)
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
T CpuComputationReduceMemoryDiscrete<T>::get_solvent_partition(int s)
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
void CpuComputationReduceMemoryDiscrete<T>::get_solvent_concentration(int s, T *phi)
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
void CpuComputationReduceMemoryDiscrete<T>::compute_stress()
{
    // This method should be invoked after invoking compute_statistics().

    // To calculate stress, we multiply weighted fourier basis to q(k)*q^dagger(-k).
    // We only need the real part of stress calculation.

    // If a propagator is in propagator_at_check_point reuse it, otherwise compute it again with allocated memory space.
    try
    {
        const int N_STRESS = 6;  // Full stress tensor: xx, yy, zz, xy, xz, yz
        const int DIM = this->cb->get_dim();
        const int M    = this->cb->get_total_grid();

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

            // Recalculate q_left from position 0 to N_LEFT
            // For stress n=1..N_RIGHT, we need position N_LEFT-n, which is q_left[N_LEFT-n]
            // Get the monomer type for the left propagator
            std::string monomer_type_left = this->propagator_computation_optimizer->get_computation_propagator(key_left).monomer_type;
            std::vector<T*> q_left = recalculate_propagator(key_left, 0, N_LEFT, monomer_type_left);

            // For leaf propagators, position 1 might not be computable from recalculate_propagator
            // because position 0 isn't stored. Compute position 1 from exp_dw if needed.
            if (q_left[1] == nullptr && key_left[0] != '(' && key_left[0] != '[')
            {
                // Position 0 is exp_dw for leaf propagators
                const double *q_mask = this->cb->get_mask();
                const T *_exp_dw = propagator_solver->exp_dw[monomer_type_left];

                // Compute position 1 = advance(position 0)
                // Use q_recal[0] for temporary storage of position 0
                for (int i = 0; i < M; i++)
                    this->q_recal[0][i] = _exp_dw[i];

                q_left[1] = this->q_recal[1];
                propagator_solver->advance_propagator(this->q_recal[0], q_left[1], monomer_type_left, q_mask);

                // Compute subsequent positions if needed
                for (int n = 2; n <= N_LEFT; n++)
                {
                    if (q_left[n] == nullptr && q_left[n-1] != nullptr)
                    {
                        q_left[n] = this->q_recal[n];
                        propagator_solver->advance_propagator(q_left[n-1], q_left[n], monomer_type_left, q_mask);
                    }
                }
            }

            // Pointers for q_right
            T *q_prev = nullptr;
            T *q_next = nullptr;

            int prev = 0;
            int next = 1;

            // Initialize q_prev from checkpoint at step 0 if available (for aggregated propagators)
            auto it_init = propagator_at_check_point.find(std::make_tuple(key_right, 0));
            if (it_init != propagator_at_check_point.end())
            {
                q_prev = it_init->second;
            }

            // Compute the q_right and stress
            std::array<T,6> _block_dq_dl = block_dq_dl[key];

            T *q_segment_1;
            T *q_segment_2;
            bool is_half_bond_length;

            for(int n=0; n<=N_RIGHT; n++)
            {
                // Use propagator_at_check_point if exists
                auto it = propagator_at_check_point.find(std::make_tuple(key_right, n));
                if(n > 0 && it != propagator_at_check_point.end())
                {
                    q_next = it->second;
                    #ifndef NDEBUG
                    std::cout << "Use propagator_at_check_point if exists: (stress, right) " << key_right << ", " <<  n << std::endl;
                    #endif
                }
                // Assign this->q_pair memory space, and compute the next propagator from the check point
                else if (n >= 1 && q_prev != nullptr)
                {
                    q_next = this->q_pair[prev];
                    propagator_solver->advance_propagator(q_prev, q_next, monomer_type, q_mask);
                    std::swap(prev, next);
                }

                // At v
                if (n == N_LEFT)
                {
                    if (this->propagator_computation_optimizer->get_computation_propagator(key_left).deps.size() == 0) // if v is leaf node, skip
                        continue;
                    q_segment_1 = propagator_half_steps_at_check_point[std::make_tuple(key_left, 0)];
                    q_segment_2 = q_next;
                    is_half_bond_length = true;
                }
                // At u
                else if (n == 0 && key_right[0] == '(')
                {
                    if (this->propagator_computation_optimizer->get_computation_propagator(key_right).deps.size() == 0) // if u is leaf node, skip
                        continue;
                    // For "At u", we need position N_LEFT
                    q_segment_1 = q_left[N_LEFT];
                    q_segment_2 = propagator_half_steps_at_check_point[std::make_tuple(key_right, 0)];
                    is_half_bond_length = true;
                }
                // At aggregation junction
                else if (n == 0)
                {
                    continue;
                }
                // Within the blocks
                else
                {
                    // q_left[N_LEFT-n] corresponds to position N_LEFT-n
                    // If this position isn't available, skip
                    if (N_LEFT-n < 0 || N_LEFT-n > (int)q_left.size() || q_left[N_LEFT-n] == nullptr)
                    {
                        continue;
                    }
                    q_segment_1 = q_left[N_LEFT-n];
                    q_segment_2 = q_next;
                    is_half_bond_length = false;
                }

                // Check for nullptr - skip if either segment is unavailable
                if (q_segment_1 == nullptr || q_segment_2 == nullptr)
                {
                    continue;
                }

                std::vector<T> segment_stress = propagator_solver->compute_single_segment_stress(
                    q_segment_1, q_segment_2, monomer_type, is_half_bond_length);

                for(int d=0; d<N_STRESS; d++)
                    _block_dq_dl[d] += segment_stress[d]*((T)n_repeated);

                q_prev = q_next;
                q_next = nullptr;
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
            std::string key_left  = std::get<1>(key);
            std::string key_right = std::get<2>(key);
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
void CpuComputationReduceMemoryDiscrete<T>::get_chain_propagator(T *q_out, int polymer, int v, int u, int n)
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

        if (propagator_at_check_point.find(std::make_tuple(dep, n)) == propagator_at_check_point.end())
            throw_with_line_number("The propagator " + dep + "[" + std::to_string(n) + "] is not stored'. Disable 'reduce_memory_usage' option to obtain propagator_computation_optimizer.");

        T *_q_from = propagator_at_check_point[std::make_tuple(dep, n)];
        for(int i=0; i<M; i++)
            q_out[i] = _q_from[i];
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
bool CpuComputationReduceMemoryDiscrete<T>::check_total_partition()
{
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

        const T *_exp_dw = propagator_solver->exp_dw[monomer_type];

        #ifndef NDEBUG
        std::cout<< p << ", " << key_left << ", " << key_right << ": " << N_LEFT << ", " << N_RIGHT << ", " << n_propagators << ", " << n_repeated << std::endl;
        #endif

        // Recalculate q_left from the check point
        std::vector<T*> q_left = recalculate_propagator(key_left, N_LEFT-N_RIGHT, N_RIGHT, monomer_type);

        // Pointers for q_right
        T *q_prev = nullptr;
        T *q_next = nullptr;

        int prev = 0;
        int next = 1;

        // Compute the q_right and total partition
        for(int n=1; n<=N_RIGHT; n++)
        {
            // Use propagator_at_check_point if exists
            auto it = propagator_at_check_point.find(std::make_tuple(key_right, n));
            if(it != propagator_at_check_point.end())
            {
                q_next = it->second;
                #ifndef NDEBUG
                std::cout << "Use propagator_at_check_point if exists: (check_total_partition, right) " << key_right << ", " <<  n << std::endl;
                #endif
            }
            // Assign this->q_pair memory space, and compute the next propagator from the check point
            else if (n > 1)
            {
                q_next = this->q_pair[prev];
                propagator_solver->advance_propagator(q_prev, q_next, monomer_type, q_mask);
                std::swap(prev, next);
            }

            T total_partition = this->cb->inner_product_inverse_weight(
                q_left[N_RIGHT-n+1], q_next, _exp_dw)*(n_repeated/this->cb->get_volume());

            total_partition /= n_propagators;
            total_partitions[p].push_back(total_partition);

            #ifndef NDEBUG
            std::cout<< p << ", " << n << ": " << total_partition << std::endl;
            #endif

            q_prev = q_next;
            q_next = nullptr;
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

// template class CpuComputationReduceMemoryDiscrete<float>;
// template class CpuComputationReduceMemoryDiscrete<std::complex<float>>;
template class CpuComputationReduceMemoryDiscrete<double>;
template class CpuComputationReduceMemoryDiscrete<std::complex<double>>;
