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
    PropagatorComputationOptimizer *propagator_computation_optimizer,
    FFTBackend backend,
    SpaceGroup* space_group)
    : CpuComputationReduceMemoryBase<T>(cb, molecules, propagator_computation_optimizer)
{
    try
    {
        #ifndef NDEBUG
        std::cout << "--------- Discrete Chain Solver (Reduce Memory), CPU Version ---------" << std::endl;
        #endif

        // Set space group first so that get_n_basis() returns the correct size
        if (space_group != nullptr) {
            PropagatorComputation<T>::set_space_group(space_group);
        }

        const int N = this->cb->get_n_basis();  // n_irreducible (with space group) or total_grid

        this->propagator_solver = new CpuSolverPseudoDiscrete<T>(cb, molecules, backend);

        // The number of parallel streams is always 1 to reduce the memory usage
        this->n_streams = 1;
        #ifndef NDEBUG
        std::cout << "The number of CPU threads is always set to " << this->n_streams << "." << std::endl;
        #endif

        // Allocate memory for concentrations
        if( this->propagator_computation_optimizer->get_computation_blocks().size() == 0)
            throw_with_line_number("There is no block. Add polymers first.");
        for(const auto& item: this->propagator_computation_optimizer->get_computation_blocks())
        {
            this->phi_block[item.first] = new T[N]();  // Zero-initialize
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
            if(this->propagator_at_check_point.find(std::make_tuple(key, 1)) == this->propagator_at_check_point.end())
                this->propagator_at_check_point[std::make_tuple(key, 1)] = new T[N];

            // Allocate half-bond step at segment 0 if there are dependencies (for junction start)
            if (deps.size() > 0)
            {
                if(propagator_half_steps_at_check_point.find(std::make_tuple(key, 0)) == propagator_half_steps_at_check_point.end())
                    propagator_half_steps_at_check_point[std::make_tuple(key, 0)] = new T[N];
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
                        propagator_half_steps_at_check_point[std::make_tuple(sub_dep, 0)] = new T[N];
                }
                else
                {
                    if(this->propagator_at_check_point.find(std::make_tuple(sub_dep, sub_n_segment)) == this->propagator_at_check_point.end())
                        this->propagator_at_check_point[std::make_tuple(sub_dep, sub_n_segment)] = new T[N];
                }
            }

            // Allocate half-bond steps at junction_ends
            for (int n: item.second.junction_ends)
            {
                if(propagator_half_steps_at_check_point.find(std::make_tuple(key, n)) == propagator_half_steps_at_check_point.end())
                    propagator_half_steps_at_check_point[std::make_tuple(key, n)] = new T[N];
            }

            #ifndef NDEBUG
            this->propagator_finished[key] = new bool[max_n_segment+1];
            for(int i=0; i<=max_n_segment;i++)
                this->propagator_finished[key][i] = false;
            for (int n: item.second.junction_ends)
                propagator_half_steps_finished[key][n] = false;
            propagator_half_steps_finished[key][0] = false;
            #endif
        }

        // Iterate through each element in the map
        #ifndef NDEBUG
        std::cout << "this->propagator_at_check_point:" << std::endl;
        for (const auto& pair : this->propagator_at_check_point) {
            std::cout << "Key: " << std::get<0>(pair.first) << ", Value: " << std::get<1>(pair.first) << std::endl;
        }
        std::cout << "propagator_half_steps_at_check_point:" << std::endl;
        for (const auto& pair : propagator_half_steps_at_check_point) {
            std::cout << "Key: " << std::get<0>(pair.first) << ", Value: " << std::get<1>(pair.first) << std::endl;
        }
        #endif

        // Find the total sum of segments and max segment per block
        // Sum max of both n_segment_left and n_segment_right for each block
        this->total_max_n_segment = 0;
        int max_n_segment_per_block = 0;
        for(const auto& block: this->phi_block)
        {
            const auto& key = block.first;
            int n_segment_right = this->propagator_computation_optimizer->get_computation_block(key).n_segment_right;
            int n_segment_left = this->propagator_computation_optimizer->get_computation_block(key).n_segment_left;
            int block_max = std::max(n_segment_right, n_segment_left);
            this->total_max_n_segment += block_max;
            max_n_segment_per_block = std::max(max_n_segment_per_block, block_max);
        }

        // Calculate checkpoint interval as 2*sqrt(N) for better memory-computation tradeoff
        this->checkpoint_interval = static_cast<int>(std::ceil(2.0*std::sqrt(static_cast<double>(this->total_max_n_segment))));
        if (this->checkpoint_interval < 1)
            this->checkpoint_interval = 1;

        #ifndef NDEBUG
        std::cout << "Checkpoint interval: " << this->checkpoint_interval << std::endl;
        #endif

        // Allocate workspace for propagator recomputation
        // - this->q_recal[0..this->checkpoint_interval-1]: workspace for storage (size = this->checkpoint_interval)
        // - this->q_pair[0-1]: ping-pong for q_right advancement
        // - this->q_skip[0-1]: ping-pong for skip phase (used in both compute_concentrations and compute_stress)
        const int workspace_size = this->checkpoint_interval;
        this->q_recal.resize(workspace_size);
        for(int n=0; n<workspace_size; n++)
            this->q_recal[n] = new T[N];
        this->q_pair[0] = new T[N];
        this->q_pair[1] = new T[N];
        this->q_skip[0] = new T[N];
        this->q_skip[1] = new T[N];

        // Allocate checkpoints at sqrt(N) intervals for each propagator
        for(const auto& item: this->propagator_computation_optimizer->get_computation_propagators())
        {
            std::string key = item.first;
            int max_n_segment = item.second.max_n_segment;
            // Add checkpoints at this->checkpoint_interval positions
            for(int n=this->checkpoint_interval; n<max_n_segment; n+=this->checkpoint_interval)
            {
                if(this->propagator_at_check_point.find(std::make_tuple(key, n)) == this->propagator_at_check_point.end())
                    this->propagator_at_check_point[std::make_tuple(key, n)] = new T[N];
            }
        }

        #ifndef NDEBUG
        std::cout << "After adding sqrt(N) checkpoints, this->propagator_at_check_point:" << std::endl;
        for (const auto& pair : this->propagator_at_check_point) {
            std::cout << "Key: " << std::get<0>(pair.first) << ", Value: " << std::get<1>(pair.first) << std::endl;
        }
        #endif

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

            int n_aggregated   = this->propagator_computation_optimizer->get_computation_block(key).v_u.size()/
                                 this->propagator_computation_optimizer->get_computation_block(key).n_repeated;
            int n_segment_left = this->propagator_computation_optimizer->get_computation_block(key).n_segment_left;
            std::string monomer_type = this->propagator_computation_optimizer->get_computation_block(key).monomer_type;

            // Skip if n_segment_left is 0
            if (n_segment_left == 0)
                continue;

            if(this->propagator_at_check_point.find(std::make_tuple(key_left, n_segment_left)) == this->propagator_at_check_point.end())
            {
                this->propagator_at_check_point[std::make_tuple(key_left, n_segment_left)] = new T[N];
                #ifndef NDEBUG
                std::cout << "Allocated, " + key_left + ", " << n_segment_left << std::endl;
                #endif
            }

            if(this->propagator_at_check_point.find(std::make_tuple(key_right, 1)) == this->propagator_at_check_point.end())
            {
                this->propagator_at_check_point[std::make_tuple(key_right, 1)] = new T[N];
                #ifndef NDEBUG
                std::cout << "Allocated, " + key_right + ", " << 1 << std::endl;
                #endif
            }

            single_partition_segment.push_back(std::make_tuple(
                p,
                this->propagator_at_check_point[std::make_tuple(key_left, n_segment_left)],  // q
                this->propagator_at_check_point[std::make_tuple(key_right, 1)],              // q_dagger
                monomer_type,
                n_aggregated                                                           // how many propagators are aggregated
                ));

            current_p++;
        }
        // Concentrations for each solvent
        for(int s=0;s<this->molecules->get_n_solvent_types();s++)
            this->phi_solvent.push_back(new T[N]);

        // Create scheduler for computation of propagator
        this->sc = new Scheduler(this->propagator_computation_optimizer->get_computation_propagators(), 1);

        this->propagator_solver->update_laplacian_operator();
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
CpuComputationReduceMemoryDiscrete<T>::~CpuComputationReduceMemoryDiscrete()
{
    delete this->propagator_solver;
    delete this->sc;

    // Free workspace (must match constructor allocation)
    for(size_t n=0; n<this->q_recal.size(); n++)
        delete[] this->q_recal[n];

    delete[] this->q_pair[0];
    delete[] this->q_pair[1];
    delete[] this->q_skip[0];
    delete[] this->q_skip[1];

    for(const auto& item: this->propagator_at_check_point)
        delete[] item.second;
    for(const auto& item: propagator_half_steps_at_check_point)
        delete[] item.second;
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
void CpuComputationReduceMemoryDiscrete<T>::compute_propagators(
    std::map<std::string, const T*> w_input,
    std::map<std::string, const T*> q_init)
{
    try
    {
        const int M = this->cb->get_total_grid();
        const bool use_reduced_basis = (this->space_group_ != nullptr);
        const int N_cp = use_reduced_basis ? this->space_group_->get_n_irreducible() : M;

        for(const auto& item: this->propagator_computation_optimizer->get_computation_propagators())
        {
            if( w_input.find(item.second.monomer_type) == w_input.end())
                throw_with_line_number("monomer_type \"" + item.second.monomer_type + "\" is not in w_input.");
        }

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
                const T *_exp_dw = this->propagator_solver->exp_dw[0][monomer_type].data();

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
                            throw_with_line_number("Could not find q_init[\"" + g + "\"]. Pass q_init to run() for grafted polymers.");
                        if (use_reduced_basis)
                        {
                            // Expand q_init to full grid, then multiply by exp_dw
                            const auto& full_to_reduced = this->space_group_->get_full_to_reduced_map();
                            for(int i=0; i<M; i++)
                                this->q_pair[0][i] = q_init[g][full_to_reduced[i]]*_exp_dw[i];
                        }
                        else
                        {
                            for(int i=0; i<M; i++)
                                this->q_pair[0][i] = q_init[g][i]*_exp_dw[i];
                        }
                    }
                    else
                    {
                        for(int i=0; i<M; i++)
                            this->q_pair[0][i] = _exp_dw[i];
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
                            this->q_pair[0][i] = 0.0;

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
                                if (this->propagator_at_check_point.find(std::make_tuple(sub_dep, sub_n_segment)) == this->propagator_at_check_point.end())
                                    std::cout << "Could not find sub key '" + sub_dep + "[" + std::to_string(sub_n_segment) + "]. " << std::endl;
                                if (!this->propagator_finished[sub_dep][sub_n_segment])
                                    std::cout << "Could not compute '" + key +  "', since '"+ sub_dep + std::to_string(sub_n_segment) + "' is not prepared." << std::endl;
                                #endif

                                _q_sub = this->propagator_at_check_point[std::make_tuple(sub_dep, sub_n_segment)];
                            }

                            // Expand from reduced basis to full grid if needed
                            if (use_reduced_basis)
                            {
                                const auto& full_to_reduced = this->space_group_->get_full_to_reduced_map();
                                for(int i=0; i<M; i++)
                                    this->q_pair[0][i] += _q_sub[full_to_reduced[i]]*static_cast<double>(sub_n_repeated);
                            }
                            else
                            {
                                for(int i=0; i<M; i++)
                                    this->q_pair[0][i] += _q_sub[i]*static_cast<double>(sub_n_repeated);
                            }
                        }

                        // If n_segments of all deps are 0
                        if (std::get<1>(deps[0]) == 0)
                        {
                            T *_propagator_half_step = propagator_half_steps_at_check_point[std::make_tuple(key, 0)];
                            // Reduce from full grid to reduced basis if needed
                            if (use_reduced_basis)
                            {
                                const auto& reduced_basis_indices = this->space_group_->get_reduced_basis_indices();
                                for(int i=0; i<N_cp; i++)
                                    _propagator_half_step[i] = this->q_pair[0][reduced_basis_indices[i]];
                            }
                            else
                            {
                                for(int i=0; i<M; i++)
                                    _propagator_half_step[i] = this->q_pair[0][i];
                            }

                            #ifndef NDEBUG
                            propagator_half_steps_finished[key][0] = true;
                            #endif

                            // Add half bond
                            this->propagator_solver->advance_propagator_half_bond_step(
                                this->q_pair[0], this->q_pair[0], monomer_type);

                            // Add full segment
                            for(int i=0; i<M; i++)
                                this->q_pair[0][i] *= _exp_dw[i];
                        }
                        else
                        {
                            // Discrete chains always use ds_index=0 (global ds)
                            this->propagator_solver->advance_propagator(
                                this->q_pair[0],
                                this->q_pair[0],
                                monomer_type,
                                q_mask, 0);
                        }

                        #ifndef NDEBUG
                        this->propagator_finished[key][1] = true;
                        #endif
                    }
                    else if(key[0] == '(')
                    {
                        // Combine branches - compute in full grid first
                        for(int i=0; i<M; i++)
                            this->q_pair[0][i] = 1.0;

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
                            // Expand from reduced basis to full grid if needed
                            if (use_reduced_basis)
                            {
                                const auto& full_to_reduced = this->space_group_->get_full_to_reduced_map();
                                for(int i=0; i<M; i++)
                                    this->q_pair[0][i] *= pow(_propagator_half_step[full_to_reduced[i]], sub_n_repeated);
                            }
                            else
                            {
                                for(int i=0; i<M; i++)
                                    this->q_pair[0][i] *= pow(_propagator_half_step[i], sub_n_repeated);
                            }
                        }

                        // Store junction start in reduced basis
                        T *_q_junction_start = propagator_half_steps_at_check_point[std::make_tuple(key, 0)];
                        if (use_reduced_basis)
                        {
                            const auto& reduced_basis_indices = this->space_group_->get_reduced_basis_indices();
                            for(int i=0; i<N_cp; i++)
                                _q_junction_start[i] = this->q_pair[0][reduced_basis_indices[i]];
                        }
                        else
                        {
                            for(int i=0; i<M; i++)
                                _q_junction_start[i] = this->q_pair[0][i];
                        }

                        #ifndef NDEBUG
                        propagator_half_steps_finished[key][0] = true;
                        #endif

                        if (n_segment_to > 0)
                        {
                            // Add half bond (q_pair[0] already has expanded data)
                            this->propagator_solver->advance_propagator_half_bond_step(
                                this->q_pair[0], this->q_pair[0], monomer_type);

                            // Add full segment
                            for(int i=0; i<M; i++)
                                this->q_pair[0][i] *= _exp_dw[i];

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
                            this->q_pair[0][i] *= q_mask[i];
                    }

                    // Store q(r, 1) at check point
                    if (this->propagator_at_check_point.find(std::make_tuple(key, 1)) != this->propagator_at_check_point.end())
                    {
                        T* _q_target = this->propagator_at_check_point[std::make_tuple(key, 1)];
                        // Reduce from full grid to reduced basis if needed
                        if (use_reduced_basis)
                        {
                            const auto& reduced_basis_indices = this->space_group_->get_reduced_basis_indices();
                            for(int i=0; i<N_cp; i++)
                                _q_target[i] = this->q_pair[0][reduced_basis_indices[i]];
                        }
                        else
                        {
                            for(int i=0; i<M; i++)
                                _q_target[i] = this->q_pair[0][i];
                        }
                    }

                    // q(r, 1+1/2) at junction_ends
                    auto it = propagator_half_steps_at_check_point.find(std::make_tuple(key, 1));
                    if (it != propagator_half_steps_at_check_point.end())
                    {
                        // First compute half bond step in full grid workspace
                        this->propagator_solver->advance_propagator_half_bond_step(
                            this->q_pair[0],
                            this->q_pair[1],
                            monomer_type);

                        // Then store in reduced basis if needed
                        if (use_reduced_basis)
                        {
                            const auto& reduced_basis_indices = this->space_group_->get_reduced_basis_indices();
                            for(int i=0; i<N_cp; i++)
                                it->second[i] = this->q_pair[1][reduced_basis_indices[i]];
                        }
                        else
                        {
                            for(int i=0; i<M; i++)
                                it->second[i] = this->q_pair[1][i];
                        }

                        #ifndef NDEBUG
                        propagator_half_steps_finished[key][1] = true;
                        #endif
                    }
                    n_segment_from++;
                }
                else
                {
                    // Load from check point - expand from reduced basis to full grid if needed
                    T* _q_from = this->propagator_at_check_point[std::make_tuple(key, n_segment_from)];
                    if (use_reduced_basis)
                    {
                        const auto& full_to_reduced = this->space_group_->get_full_to_reduced_map();
                        for(int i=0; i<M; i++)
                            this->q_pair[0][i] = _q_from[full_to_reduced[i]];
                    }
                    else
                    {
                        for(int i=0; i<M; i++)
                            this->q_pair[0][i] = _q_from[i];
                    }
                }

                // Advance propagator successively
                for(int n=n_segment_from; n<n_segment_to; n++)
                {
                    #ifndef NDEBUG
                    if (!this->propagator_finished[key][n])
                        std::cout << "unfinished, key: " + key + ", " + std::to_string(n) << std::endl;
                    if (this->propagator_finished[key][n+1])
                        std::cout << "already finished: " + key + ", " + std::to_string(n+1) << std::endl;
                    #endif

                    // Discrete chains always use ds_index=0 (global ds)
                    this->propagator_solver->advance_propagator(
                            this->q_pair[prev],
                            this->q_pair[next],
                            monomer_type, q_mask, 0);

                    #ifndef NDEBUG
                    this->propagator_finished[key][n+1] = true;
                    #endif

                    // Copy this->q_pair[next] to the this->propagator_at_check_point
                    if (this->propagator_at_check_point.find(std::make_tuple(key, n+1)) != this->propagator_at_check_point.end())
                    {
                        T* _q_target = this->propagator_at_check_point[std::make_tuple(key, n+1)];
                        // Reduce from full grid to reduced basis if needed
                        if (use_reduced_basis)
                        {
                            const auto& reduced_basis_indices = this->space_group_->get_reduced_basis_indices();
                            for(int i=0; i<N_cp; i++)
                                _q_target[i] = this->q_pair[next][reduced_basis_indices[i]];
                        }
                        else
                        {
                            for(int i=0; i<M; i++)
                                _q_target[i] = this->q_pair[next][i];
                        }
                    }

                    // q(r, n+1+1/2) at junction_ends
                    auto it = propagator_half_steps_at_check_point.find(std::make_tuple(key, n+1));
                    if (it != propagator_half_steps_at_check_point.end())
                    {
                        // First compute half bond step, store temporarily
                        this->propagator_solver->advance_propagator_half_bond_step(
                            this->q_pair[next],
                            this->q_skip[0],  // Use q_skip as temp buffer
                            monomer_type);

                        // Then store in reduced basis if needed
                        if (use_reduced_basis)
                        {
                            const auto& reduced_basis_indices = this->space_group_->get_reduced_basis_indices();
                            for(int i=0; i<N_cp; i++)
                                it->second[i] = this->q_skip[0][reduced_basis_indices[i]];
                        }
                        else
                        {
                            for(int i=0; i<M; i++)
                                it->second[i] = this->q_skip[0][i];
                        }

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
            const T *_exp_dw         = this->propagator_solver->exp_dw[0][monomer_type].data();

            if (use_reduced_basis)
            {
                // Propagators are in reduced basis, need weighted inner product
                const auto& orbit_counts = this->space_group_->get_orbit_counts();
                const auto& reduced_basis_indices = this->space_group_->get_reduced_basis_indices();
                double dv_base = this->cb->get_volume() / M;
                T sum = 0.0;
                for(int i=0; i<N_cp; i++)
                {
                    // inner_product_inverse_weight divides by exp_dw at each point
                    int full_idx = reduced_basis_indices[i];
                    sum += propagator_left[i] * propagator_right[i] / _exp_dw[full_idx] * static_cast<double>(orbit_counts[i]);
                }
                sum *= static_cast<T>(dv_base);
                this->single_polymer_partitions[p] = sum / (n_aggregated * this->cb->get_volume());
            }
            else
            {
                this->single_polymer_partitions[p] = this->cb->inner_product_inverse_weight(
                    propagator_left, propagator_right, _exp_dw)/(n_aggregated*this->cb->get_volume());
            }
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
void CpuComputationReduceMemoryDiscrete<T>::compute_concentrations()
{
    try
    {
        #ifndef NDEBUG
        std::cout << "compute_concentrations 0" << std::endl;
        #endif

        const int M = this->cb->get_total_grid();
        const bool use_reduced_basis = (this->space_group_ != nullptr);
        const int N_phi = use_reduced_basis ? this->space_group_->get_n_irreducible() : M;

        // Calculate segment concentrations
        for(size_t b=0; b<this->phi_block.size();b++)
        {
            auto block = this->phi_block.begin();
            std::advance(block, b);
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
                for(int i=0; i<N_phi;i++)
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

            // Get local_ds from ds_index encoded in key (DK+M format)
            Polymer& pc = this->molecules->get_polymer(p);
            int ds_index = PropagatorCode::get_ds_index_from_key(key_right);
            const ContourLengthMapping& mapping = this->molecules->get_contour_length_mapping();
            double local_ds = mapping.get_ds_from_index(ds_index);

            // Normalize concentration: local_ds * volume_fraction / alpha
            T norm = (local_ds*pc.get_volume_fraction()/pc.get_alpha()*n_repeated)/this->single_polymer_partitions[p];
            for(int i=0; i<N_phi; i++)
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

            T *_phi = this->phi_solvent[s];
            T *_exp_dw = this->propagator_solver->exp_dw[0][monomer_type].data();

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
    // If a propagator is in this->propagator_at_check_point reuse it, otherwise compute it again with allocated memory space.
    // This function returns q_out where q_out[n] corresponds to position N_START+n for n=0..N_RIGHT.
    try
    {
        // Assign a pointer for mask
        const double *q_mask = this->cb->get_mask();

        // An array of pointers for q_out - sized to accommodate N_RIGHT elements
        std::vector<T*> q_out(N_RIGHT + 1);

        // Try to get starting position from checkpoint
        auto it_start = this->propagator_at_check_point.find(std::make_tuple(key, N_START));
        if (it_start != this->propagator_at_check_point.end())
        {
            q_out[0] = it_start->second;
        }

        // Compute the q_out for n=1..N_RIGHT
        for(int n=1; n<=N_RIGHT; n++)
        {
            // Use this->propagator_at_check_point if exists
            auto it = this->propagator_at_check_point.find(std::make_tuple(key, N_START+n));
            if(it != this->propagator_at_check_point.end())
            {
                q_out[n] = it->second;
                #ifndef NDEBUG
                std::cout << "Use this->propagator_at_check_point if exists: (phi, left) " << key << ", " << N_START+n << std::endl;
                #endif
            }
            // Assign q_recal memory space, and compute the next propagator from the previous position
            else if (q_out[n-1] != nullptr)
            {
                q_out[n] = this->q_recal[n];
                // Discrete chains always use ds_index=0 (global ds)
                this->propagator_solver->advance_propagator(q_out[n-1], q_out[n], monomer_type, q_mask, 0);
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
        // Block-based computation with skip-then-store pattern to minimize memory usage.
        // For n in [n_start, n_end], q_left positions [left_start, left_end] form a forward range.
        // We process in blocks of size this->checkpoint_interval, recomputing q_left from nearest checkpoint.
        //
        // this->q_skip[0-1]: ping-pong for skip phase, this->q_recal[0..]: storage for block values

        const int M = this->cb->get_total_grid();
        const T *_exp_dw = this->propagator_solver->exp_dw[0][monomer_type].data();
        const double *q_mask = this->cb->get_mask();
        const int k = this->checkpoint_interval;
        const bool use_reduced_basis = (this->space_group_ != nullptr);
        const int N_phi = use_reduced_basis ? this->space_group_->get_n_irreducible() : M;

        // Temporary buffer for phi in full grid (if using reduced basis)
        std::vector<T> phi_full;
        T* phi_work;
        if (use_reduced_basis)
        {
            phi_full.resize(M, 0.0);
            phi_work = phi_full.data();
        }
        else
        {
            phi_work = phi;
        }

        // Initialize phi to zero
        for(int i=0; i<M; i++)
            phi_work[i] = 0.0;

        // For the concentration calculation:
        //   phi += sum_{n=1}^{N_RIGHT} q_left[N_LEFT-n+1] * q_right[n]

        // Number of blocks for n = 1 to N_RIGHT
        const int num_blocks = (N_RIGHT + k - 1) / k;

        // q_right ping-pong buffers
        T *q_right_prev = nullptr;
        T *q_right_curr = nullptr;
        int right_prev = 0;
        int right_next = 1;

        // Process each block of n values
        for(int block = 0; block < num_blocks; block++)
        {
            int n_start = block * k + 1;  // First n in this block (1-indexed for discrete)
            int n_end = std::min((block + 1) * k, N_RIGHT);  // Last n in this block

            // For n in [n_start, n_end], we need q_left at positions:
            // (N_LEFT - n + 1) for n in [n_start, n_end]
            // which gives positions [N_LEFT - n_end + 1, N_LEFT - n_start + 1] (a forward range)
            const int left_start = N_LEFT - n_end + 1;
            const int left_end = N_LEFT - n_start + 1;

            // Find checkpoint at or before left_start
            int check_pos = 1;  // Default to position 1 (first segment for discrete)
            for(int cp = k; cp <= left_start; cp += k)
            {
                if(this->propagator_at_check_point.find(std::make_tuple(key_left, cp)) != this->propagator_at_check_point.end())
                    check_pos = cp;
            }

            // Skip phase uses this->q_skip[0-1] for ping-pong, storage uses this->q_recal[0+]
            const int steps_before = left_start - check_pos;
            const int storage_count = left_end - left_start + 1;

            // Load checkpoint
            auto it_checkpoint = this->propagator_at_check_point.find(std::make_tuple(key_left, check_pos));
            if (it_checkpoint == this->propagator_at_check_point.end())
            {
                throw_with_line_number("Checkpoint not found at position " + std::to_string(check_pos));
            }
            T* q_checkpoint = it_checkpoint->second;
            T* q_prev;

            if(steps_before == 0)
            {
                // Checkpoint is exactly at left_start, store directly in this->q_recal[0]
                // Expand from reduced basis to full grid if needed
                if (use_reduced_basis)
                {
                    const auto& full_to_reduced = this->space_group_->get_full_to_reduced_map();
                    for(int i=0; i<M; i++)
                        this->q_recal[0][i] = q_checkpoint[full_to_reduced[i]];
                }
                else
                {
                    for(int i=0; i<M; i++)
                        this->q_recal[0][i] = q_checkpoint[i];
                }
                q_prev = this->q_recal[0];
            }
            else
            {
                // Start ping-pong from checkpoint using this->q_skip[0-1]
                // Expand from reduced basis to full grid if needed
                if (use_reduced_basis)
                {
                    const auto& full_to_reduced = this->space_group_->get_full_to_reduced_map();
                    for(int i=0; i<M; i++)
                        this->q_skip[0][i] = q_checkpoint[full_to_reduced[i]];
                }
                else
                {
                    for(int i=0; i<M; i++)
                        this->q_skip[0][i] = q_checkpoint[i];
                }
                q_prev = this->q_skip[0];
                int ping_pong = 1;

                // Skip to left_start using q_skip ping-pong buffers
                for(int step = 1; step < steps_before; step++)
                {
                    int actual_pos = check_pos + step;
                    T* q_curr = this->q_skip[ping_pong];

                    auto it = this->propagator_at_check_point.find(std::make_tuple(key_left, actual_pos));
                    if(it != this->propagator_at_check_point.end())
                    {
                        // Expand from reduced basis to full grid if needed
                        if (use_reduced_basis)
                        {
                            const auto& full_to_reduced = this->space_group_->get_full_to_reduced_map();
                            for(int i=0; i<M; i++)
                                q_curr[i] = it->second[full_to_reduced[i]];
                        }
                        else
                        {
                            for(int i=0; i<M; i++)
                                q_curr[i] = it->second[i];
                        }
                    }
                    else
                    {
                        // Discrete chains always use ds_index=0 (global ds)
                        this->propagator_solver->advance_propagator(q_prev, q_curr, monomer_type, q_mask, 0);
                    }
                    q_prev = q_curr;
                    ping_pong = 1 - ping_pong;
                }

                // Compute the step that reaches left_start, store in this->q_recal[0]
                int actual_pos = check_pos + steps_before;
                auto it = this->propagator_at_check_point.find(std::make_tuple(key_left, actual_pos));
                if(it != this->propagator_at_check_point.end())
                {
                    // Expand from reduced basis to full grid if needed
                    if (use_reduced_basis)
                    {
                        const auto& full_to_reduced = this->space_group_->get_full_to_reduced_map();
                        for(int i=0; i<M; i++)
                            this->q_recal[0][i] = it->second[full_to_reduced[i]];
                    }
                    else
                    {
                        for(int i=0; i<M; i++)
                            this->q_recal[0][i] = it->second[i];
                    }
                }
                else
                {
                    // Discrete chains always use ds_index=0 (global ds)
                    this->propagator_solver->advance_propagator(q_prev, this->q_recal[0], monomer_type, q_mask, 0);
                }
                q_prev = this->q_recal[0];
            }

            // Compute remaining positions [left_start+1, left_end] and store in this->q_recal[1+]
            for(int idx = 1; idx < storage_count; idx++)
            {
                int actual_pos = left_start + idx;
                T* q_curr = this->q_recal[idx];

                auto it = this->propagator_at_check_point.find(std::make_tuple(key_left, actual_pos));
                if(it != this->propagator_at_check_point.end())
                {
                    // Expand from reduced basis to full grid if needed
                    if (use_reduced_basis)
                    {
                        const auto& full_to_reduced = this->space_group_->get_full_to_reduced_map();
                        for(int i=0; i<M; i++)
                            q_curr[i] = it->second[full_to_reduced[i]];
                    }
                    else
                    {
                        for(int i=0; i<M; i++)
                            q_curr[i] = it->second[i];
                    }
                }
                else
                {
                    // Discrete chains always use ds_index=0 (global ds)
                    this->propagator_solver->advance_propagator(q_prev, q_curr, monomer_type, q_mask, 0);
                }
                q_prev = q_curr;
            }

            // Process each n in this block
            for(int n = n_start; n <= n_end; n++)
            {
                // Get q_right[n]
                auto it_right = this->propagator_at_check_point.find(std::make_tuple(key_right, n));
                if (it_right != this->propagator_at_check_point.end())
                {
                    // Expand from reduced basis to full grid if needed
                    if (use_reduced_basis)
                    {
                        const auto& full_to_reduced = this->space_group_->get_full_to_reduced_map();
                        for(int i=0; i<M; i++)
                            this->q_pair[right_prev][i] = it_right->second[full_to_reduced[i]];
                        q_right_curr = this->q_pair[right_prev];
                        std::swap(right_prev, right_next);
                    }
                    else
                    {
                        q_right_curr = it_right->second;
                    }
                }
                else if (n > 1 && q_right_prev != nullptr)
                {
                    q_right_curr = this->q_pair[right_prev];
                    // Discrete chains always use ds_index=0 (global ds)
                    this->propagator_solver->advance_propagator(q_right_prev, q_right_curr, monomer_type, q_mask, 0);
                    std::swap(right_prev, right_next);
                }
                else if (n == 1)
                {
                    auto it_n1 = this->propagator_at_check_point.find(std::make_tuple(key_right, 1));
                    if (it_n1 != this->propagator_at_check_point.end())
                    {
                        // Expand from reduced basis to full grid if needed
                        if (use_reduced_basis)
                        {
                            const auto& full_to_reduced = this->space_group_->get_full_to_reduced_map();
                            for(int i=0; i<M; i++)
                                this->q_pair[right_prev][i] = it_n1->second[full_to_reduced[i]];
                            q_right_curr = this->q_pair[right_prev];
                            std::swap(right_prev, right_next);
                        }
                        else
                        {
                            q_right_curr = it_n1->second;
                        }
                    }
                }

                // Get q_left at position (N_LEFT - n + 1)
                // storage_idx = (N_LEFT - n + 1) - left_start = n_end - n
                int storage_idx = n_end - n;  // Index into this->q_recal[storage_idx]

                T* q_left_ptr = this->q_recal[storage_idx];

                // Add contribution: q_left * q_right
                if (q_left_ptr != nullptr && q_right_curr != nullptr)
                {
                    for(int i=0; i<M; i++)
                        phi_work[i] += q_left_ptr[i] * q_right_curr[i];
                }

                q_right_prev = q_right_curr;
            }
        }

        // Divide by exp_dw for proper normalization
        for(int i=0; i<M; i++)
            phi_work[i] /= _exp_dw[i];

        // Reduce from full grid to reduced basis if needed
        if (use_reduced_basis)
        {
            const auto& reduced_basis_indices = this->space_group_->get_reduced_basis_indices();
            for(int i=0; i<N_phi; i++)
                phi[i] = phi_work[reduced_basis_indices[i]];
        }
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
void CpuComputationReduceMemoryDiscrete<T>::compute_stress()
{
    // This method should be invoked after invoking compute_statistics().

    // Uses block-based computation to minimize memory usage (O(sqrt(N)) workspace).
    try
    {
        const int N_STRESS = 6;  // Full stress tensor: xx, yy, zz, xy, xz, yz
        const int DIM = this->cb->get_dim();
        const int M    = this->cb->get_total_grid();
        const int k = this->checkpoint_interval;

        std::map<std::tuple<int, std::string, std::string>, std::array<T,6>> block_dq_dl;

        // Assign a pointer for mask
        const double *q_mask = this->cb->get_mask();

        // Reset stress map
        for(const auto& item: this->phi_block)
        {
            for(int d=0; d<N_STRESS; d++)
                block_dq_dl[item.first][d] = 0.0;
        }

        // Compute stress for each block
        for(size_t b=0; b<this->phi_block.size();b++)
        {
            auto block = this->phi_block.begin();
            std::advance(block, b);
            const auto& key   = block->first;

            std::string key_left  = std::get<1>(key);
            std::string key_right = std::get<2>(key);

            const int N_RIGHT = this->propagator_computation_optimizer->get_computation_block(key).n_segment_right;
            const int N_LEFT  = this->propagator_computation_optimizer->get_computation_block(key).n_segment_left;
            std::string monomer_type = this->propagator_computation_optimizer->get_computation_block(key).monomer_type;
            int n_repeated = this->propagator_computation_optimizer->get_computation_block(key).n_repeated;

            // If there is no segment
            if(N_RIGHT == 0)
                continue;

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
                // For discrete chains, valid positions are >= 1
                const int left_start_raw = N_LEFT - n_end;
                const int left_end_raw = N_LEFT - n_start;
                const int left_start = std::max(1, left_start_raw);  // Clamp to valid range
                const int left_end = std::max(1, left_end_raw);      // Clamp to valid range

                // Find the best checkpoint at or before left_start
                int check_pos = -1;
                bool has_valid_q_left = (left_start <= left_end);

                if(has_valid_q_left)
                {
                    for(int cp = 1; cp <= left_start; cp++)  // Discrete starts from 1
                    {
                        if(this->propagator_at_check_point.find(std::make_tuple(key_left, cp)) != this->propagator_at_check_point.end())
                            check_pos = cp;
                    }
                }

                // Recompute q_left from check_pos to left_end using q_recal workspace
                // (Only if we have a valid checkpoint and valid positions)
                int storage_count = 0;
                if(check_pos >= 0 && has_valid_q_left)
                {
                    const int steps_before = left_start - check_pos;
                    storage_count = left_end - left_start + 1;

                    // Load checkpoint - use find() to be safe
                    auto it_checkpoint = this->propagator_at_check_point.find(std::make_tuple(key_left, check_pos));
                    if(it_checkpoint != this->propagator_at_check_point.end())
                    {
                        T* q_checkpoint = it_checkpoint->second;
                        T* q_prev_left;
                        T* q_curr_left;

                        // Skip phase uses this->q_skip[0-1] for ping-pong, storage uses this->q_recal[0+]
                        if(steps_before == 0)
                        {
                            for(int i=0; i<M; i++)
                                this->q_recal[0][i] = q_checkpoint[i];
                            q_prev_left = this->q_recal[0];
                        }
                        else
                        {
                            for(int i=0; i<M; i++)
                                this->q_skip[0][i] = q_checkpoint[i];
                            q_prev_left = this->q_skip[0];
                            int ping_pong = 1;

                            for(int step = 1; step < steps_before; step++)
                            {
                                int actual_pos = check_pos + step;
                                q_curr_left = this->q_skip[ping_pong];

                                auto it = this->propagator_at_check_point.find(std::make_tuple(key_left, actual_pos));
                                if(it != this->propagator_at_check_point.end())
                                {
                                    for(int i=0; i<M; i++)
                                        q_curr_left[i] = it->second[i];
                                }
                                else
                                {
                                    // Discrete chains always use ds_index=0 (global ds)
                                    this->propagator_solver->advance_propagator(q_prev_left, q_curr_left, monomer_type, q_mask, 0);
                                }
                                q_prev_left = q_curr_left;
                                ping_pong = 1 - ping_pong;
                            }

                            int actual_pos = check_pos + steps_before;
                            auto it = this->propagator_at_check_point.find(std::make_tuple(key_left, actual_pos));
                            if(it != this->propagator_at_check_point.end())
                            {
                                for(int i=0; i<M; i++)
                                    this->q_recal[0][i] = it->second[i];
                            }
                            else
                            {
                                // Discrete chains always use ds_index=0 (global ds)
                                this->propagator_solver->advance_propagator(q_prev_left, this->q_recal[0], monomer_type, q_mask, 0);
                            }
                            q_prev_left = this->q_recal[0];
                        }

                        // Compute remaining positions [left_start+1, left_end] and store in this->q_recal[1+]
                        for(int idx = 1; idx < storage_count; idx++)
                        {
                            int actual_pos = left_start + idx;
                            q_curr_left = this->q_recal[idx];

                            auto it = this->propagator_at_check_point.find(std::make_tuple(key_left, actual_pos));
                            if(it != this->propagator_at_check_point.end())
                            {
                                for(int i=0; i<M; i++)
                                    q_curr_left[i] = it->second[i];
                            }
                            else
                            {
                                // Discrete chains always use ds_index=0 (global ds)
                                this->propagator_solver->advance_propagator(q_prev_left, q_curr_left, monomer_type, q_mask, 0);
                            }
                            q_prev_left = q_curr_left;
                        }
                    }
                    else
                    {
                        storage_count = 0;  // No valid storage
                    }
                }

                // Process each n in [n_start, n_end]
                for(int n = n_start; n <= n_end; n++)
                {
                    // Get q_right[n]
                    auto it_right = this->propagator_at_check_point.find(std::make_tuple(key_right, n));
                    if(it_right != this->propagator_at_check_point.end())
                    {
                        q_right_curr = it_right->second;
                        q_right_prev = q_right_curr;
                        current_n_right = n;
                    }
                    else if(n > 0)
                    {
                        while(current_n_right < n)
                        {
                            T* q_out = this->q_pair[right_next_idx];
                            // Discrete chains always use ds_index=0 (global ds)
                            this->propagator_solver->advance_propagator(q_right_prev, q_out, monomer_type, q_mask, 0);
                            q_right_prev = q_out;
                            q_right_curr = q_out;
                            std::swap(right_prev_idx, right_next_idx);
                            current_n_right++;
                        }
                    }

                    T *q_segment_1 = nullptr;
                    T *q_segment_2 = nullptr;
                    bool is_half_bond_length = false;

                    // At v (junction at n == N_LEFT)
                    if (n == N_LEFT)
                    {
                        if (this->propagator_computation_optimizer->get_computation_propagator(key_left).deps.size() == 0)
                            continue;
                        auto it_half = propagator_half_steps_at_check_point.find(std::make_tuple(key_left, 0));
                        if(it_half != propagator_half_steps_at_check_point.end())
                            q_segment_1 = it_half->second;
                        q_segment_2 = q_right_curr;
                        is_half_bond_length = true;
                    }
                    // At u (junction at n == 0)
                    else if (n == 0 && key_right[0] == '(')
                    {
                        if (this->propagator_computation_optimizer->get_computation_propagator(key_right).deps.size() == 0)
                            continue;
                        // Get q_left at position N_LEFT from q_recal if available
                        int left_pos = N_LEFT;
                        if(left_pos >= 1 && left_pos >= left_start && left_pos <= left_end && storage_count > 0)
                        {
                            int storage_idx = left_pos - left_start;
                            q_segment_1 = this->q_recal[storage_idx];
                        }
                        else
                        {
                            // Try to get from checkpoint directly
                            auto it_left = this->propagator_at_check_point.find(std::make_tuple(key_left, N_LEFT));
                            if(it_left != this->propagator_at_check_point.end())
                                q_segment_1 = it_left->second;
                        }
                        auto it_half = propagator_half_steps_at_check_point.find(std::make_tuple(key_right, 0));
                        if(it_half != propagator_half_steps_at_check_point.end())
                            q_segment_2 = it_half->second;
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
                        int left_pos = N_LEFT - n;
                        if(left_pos >= 1 && left_pos >= left_start && left_pos <= left_end && storage_count > 0)
                        {
                            int storage_idx = left_pos - left_start;
                            q_segment_1 = this->q_recal[storage_idx];
                        }
                        q_segment_2 = q_right_curr;
                        is_half_bond_length = false;
                    }

                    // Check for nullptr
                    if (q_segment_1 == nullptr || q_segment_2 == nullptr)
                        continue;

                    std::vector<T> segment_stress = this->propagator_solver->compute_single_segment_stress(
                        q_segment_1, q_segment_2, monomer_type, is_half_bond_length);

                    for(int d=0; d<N_STRESS; d++)
                        _block_dq_dl[d] += segment_stress[d]*((T)n_repeated);
                }
            }

            // Multiply by local_ds for this block (get ds_index from key)
            int ds_index = PropagatorCode::get_ds_index_from_key(key_right);
            const ContourLengthMapping& mapping_stress = this->molecules->get_contour_length_mapping();
            double local_ds = mapping_stress.get_ds_from_index(ds_index);
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
            for(int d=0; d<N_STRESS; d++)
                this->dq_dl[p][d] += block_dq_dl[key][d];
        }
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
void CpuComputationReduceMemoryDiscrete<T>::get_chain_propagator(T *q_out, int polymer, int v, int u, int n)
{
    // This method should be invoked after invoking compute_statistics()

    // Get chain propagator for a selected polymer, block and direction.
    // Uses O(sqrt(N)) workspace by computing from nearest checkpoint.
    try
    {
        const int M = this->cb->get_total_grid();
        const int N = this->cb->get_n_basis();  // n_irreducible (with space group) or total_grid
        Polymer& pc = this->molecules->get_polymer(polymer);
        std::string dep = pc.get_propagator_key(v,u);

        if (this->propagator_computation_optimizer->get_computation_propagators().find(dep) == this->propagator_computation_optimizer->get_computation_propagators().end())
            throw_with_line_number("Could not find the propagator code '" + dep + "'. Disable 'aggregation' option to obtain propagator_computation_optimizer.");

        const int N_RIGHT = this->propagator_computation_optimizer->get_computation_propagator(dep).max_n_segment;
        if (n < 1 || n > N_RIGHT)
            throw_with_line_number("n (" + std::to_string(n) + ") must be in range [1, " + std::to_string(N_RIGHT) + "]");

        // Check if the requested segment is at a checkpoint
        auto checkpoint_key = std::make_tuple(dep, n);
        if (this->propagator_at_check_point.find(checkpoint_key) != this->propagator_at_check_point.end())
        {
            // Directly copy from checkpoint
            T *_q_from = this->propagator_at_check_point[checkpoint_key];
            for(int i=0; i<N; i++)
                q_out[i] = _q_from[i];
        }
        else
        {
            // Find nearest checkpoint at or before position n
            int check_pos = 1;  // Start with 1 (discrete chains start at 1)
            for(int cp = this->checkpoint_interval; cp < n; cp += this->checkpoint_interval)
            {
                if(this->propagator_at_check_point.find(std::make_tuple(dep, cp)) != this->propagator_at_check_point.end())
                    check_pos = cp;
            }

            // Recalculate from checkpoint to position n
            std::string monomer_type = this->propagator_computation_optimizer->get_computation_propagator(dep).monomer_type;
            const double *q_mask = this->cb->get_mask();

            // Load checkpoint
            T* q_checkpoint = this->propagator_at_check_point[std::make_tuple(dep, check_pos)];

            // Use ping-pong buffers to compute to position n
            for(int i=0; i<M; i++)
                this->q_recal[0][i] = q_checkpoint[i];

            T* q_prev = this->q_recal[0];
            int ping_pong = 1;

            for(int pos = check_pos + 1; pos <= n; pos++)
            {
                // Check if this position is a checkpoint
                auto it = this->propagator_at_check_point.find(std::make_tuple(dep, pos));
                if(it != this->propagator_at_check_point.end())
                {
                    q_prev = it->second;
                }
                else
                {
                    T* q_curr = this->q_recal[ping_pong];
                    // Discrete chains always use ds_index=0 (global ds)
                    this->propagator_solver->advance_propagator(q_prev, q_curr, monomer_type, q_mask, 0);
                    q_prev = q_curr;
                    ping_pong = 1 - ping_pong;
                }
            }

            // Copy result to output
            for(int i=0; i<N; i++)
                q_out[i] = q_prev[i];
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
bool CpuComputationReduceMemoryDiscrete<T>::check_total_partition()
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

    // Assign a pointer for mask
    const double *q_mask = this->cb->get_mask();

    for(const auto& block: this->phi_block)
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

        const T *_exp_dw = this->propagator_solver->exp_dw[0][monomer_type].data();

        #ifndef NDEBUG
        std::cout<< p << ", " << key_left << ", " << key_right << ": " << N_LEFT << ", " << N_RIGHT << ", " << n_propagators << ", " << n_repeated << std::endl;
        #endif

        // If there is no segment
        if(N_RIGHT == 0)
            continue;

        // Number of blocks (for n = 1 to N_RIGHT)
        const int num_blocks = (N_RIGHT + k - 1) / k;

        // Pointers for q_right (ping-pong)
        T *q_right_prev = nullptr;
        T *q_right_curr = nullptr;
        int right_prev_idx = 0;
        int right_next_idx = 1;

        // Current position of q_right
        int current_n_right = 0;

        // Process each block
        for(int blk = 0; blk < num_blocks; blk++)
        {
            const int n_start = blk * k + 1;  // Discrete starts from 1
            const int n_end = std::min((blk + 1) * k, N_RIGHT);  // inclusive

            // q_left positions needed: [N_LEFT - n_end + 1, N_LEFT - n_start + 1] (forward range)
            // For discrete: we need positions >= 1
            const int left_start_raw = N_LEFT - n_end + 1;
            const int left_end_raw = N_LEFT - n_start + 1;
            const int left_start = std::max(1, left_start_raw);
            const int left_end = std::max(1, left_end_raw);

            // Find the best checkpoint at or before left_start
            int check_pos = -1;
            bool has_valid_q_left = (left_start <= left_end);

            if(has_valid_q_left)
            {
                for(int cp = 1; cp <= left_start; cp++)  // Discrete starts from 1
                {
                    if(this->propagator_at_check_point.find(std::make_tuple(key_left, cp)) != this->propagator_at_check_point.end())
                        check_pos = cp;
                }
            }

            // Recompute q_left from check_pos to left_end using q_recal workspace
            int storage_count = 0;
            if(check_pos >= 0 && has_valid_q_left)
            {
                const int steps_before = left_start - check_pos;
                storage_count = left_end - left_start + 1;

                // Load checkpoint
                auto it_checkpoint = this->propagator_at_check_point.find(std::make_tuple(key_left, check_pos));
                if(it_checkpoint != this->propagator_at_check_point.end())
                {
                    T* q_checkpoint = it_checkpoint->second;
                    T* q_prev_left;
                    T* q_curr_left;

                    // Skip phase uses this->q_skip[0-1] for ping-pong, storage uses this->q_recal[0+]
                    if(steps_before == 0)
                    {
                        for(int i=0; i<M; i++)
                            this->q_recal[0][i] = q_checkpoint[i];
                        q_prev_left = this->q_recal[0];
                    }
                    else
                    {
                        for(int i=0; i<M; i++)
                            this->q_skip[0][i] = q_checkpoint[i];
                        q_prev_left = this->q_skip[0];
                        int ping_pong = 1;

                        for(int step = 1; step < steps_before; step++)
                        {
                            int actual_pos = check_pos + step;
                            q_curr_left = this->q_skip[ping_pong];

                            auto it = this->propagator_at_check_point.find(std::make_tuple(key_left, actual_pos));
                            if(it != this->propagator_at_check_point.end())
                            {
                                for(int i=0; i<M; i++)
                                    q_curr_left[i] = it->second[i];
                            }
                            else
                            {
                                // Discrete chains always use ds_index=0 (global ds)
                                this->propagator_solver->advance_propagator(q_prev_left, q_curr_left, monomer_type, q_mask, 0);
                            }
                            q_prev_left = q_curr_left;
                            ping_pong = 1 - ping_pong;
                        }

                        int actual_pos = check_pos + steps_before;
                        auto it = this->propagator_at_check_point.find(std::make_tuple(key_left, actual_pos));
                        if(it != this->propagator_at_check_point.end())
                        {
                            for(int i=0; i<M; i++)
                                this->q_recal[0][i] = it->second[i];
                        }
                        else
                        {
                            // Discrete chains always use ds_index=0 (global ds)
                            this->propagator_solver->advance_propagator(q_prev_left, this->q_recal[0], monomer_type, q_mask, 0);
                        }
                        q_prev_left = this->q_recal[0];
                    }

                    // Compute remaining positions [left_start+1, left_end] and store in this->q_recal[1+]
                    for(int idx = 1; idx < storage_count; idx++)
                    {
                        int actual_pos = left_start + idx;
                        q_curr_left = this->q_recal[idx];

                        auto it = this->propagator_at_check_point.find(std::make_tuple(key_left, actual_pos));
                        if(it != this->propagator_at_check_point.end())
                        {
                            for(int i=0; i<M; i++)
                                q_curr_left[i] = it->second[i];
                        }
                        else
                        {
                            // Discrete chains always use ds_index=0 (global ds)
                            this->propagator_solver->advance_propagator(q_prev_left, q_curr_left, monomer_type, q_mask, 0);
                        }
                        q_prev_left = q_curr_left;
                    }
                }
                else
                {
                    storage_count = 0;
                }
            }

            // Process each n in [n_start, n_end]
            for(int n = n_start; n <= n_end; n++)
            {
                // Get q_right[n]
                auto it_right = this->propagator_at_check_point.find(std::make_tuple(key_right, n));
                if(it_right != this->propagator_at_check_point.end())
                {
                    q_right_curr = it_right->second;
                    q_right_prev = q_right_curr;
                    current_n_right = n;
                    #ifndef NDEBUG
                    std::cout << "Use this->propagator_at_check_point if exists: (check_total_partition, right) " << key_right << ", " <<  n << std::endl;
                    #endif
                }
                else if(current_n_right > 0)
                {
                    while(current_n_right < n)
                    {
                        T* q_out = this->q_pair[right_next_idx];
                        // Discrete chains always use ds_index=0 (global ds)
                        this->propagator_solver->advance_propagator(q_right_prev, q_out, monomer_type, q_mask, 0);
                        q_right_prev = q_out;
                        q_right_curr = q_out;
                        std::swap(right_prev_idx, right_next_idx);
                        current_n_right++;
                    }
                }

                // Get q_left[N_LEFT - n + 1]
                int left_pos = N_LEFT - n + 1;
                T* q_left_n = nullptr;
                if(left_pos >= 1 && left_pos >= left_start && left_pos <= left_end && storage_count > 0)
                {
                    int storage_idx = left_pos - left_start;
                    q_left_n = this->q_recal[storage_idx];
                }

                // Skip if we don't have valid pointers
                if(q_left_n == nullptr || q_right_curr == nullptr)
                    continue;

                T total_partition = this->cb->inner_product_inverse_weight(
                    q_left_n, q_right_curr, _exp_dw)*(n_repeated/this->cb->get_volume());

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
INSTANTIATE_CLASS(CpuComputationReduceMemoryDiscrete);
