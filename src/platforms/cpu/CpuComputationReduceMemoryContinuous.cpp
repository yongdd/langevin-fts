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
#include "CpuSolverPseudoRK2.h"
#include "CpuSolverPseudoETDRK4.h"
#include "CpuSolverCNADI.h"
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
 * @param this->method                         "pseudospectral" or "realspace"
 * @param numerical_method               Numerical algorithm:
 *                                       - For pseudospectral: "rqm4" or "etdrk4"
 *                                       - For realspace: "cn-adi2" or "cn-adi4-lr"
 */
template <typename T>
CpuComputationReduceMemoryContinuous<T>::CpuComputationReduceMemoryContinuous(
    ComputationBox<T>* cb,
    Molecules *molecules,
    PropagatorComputationOptimizer *propagator_computation_optimizer,
    std::string method,
    std::string numerical_method)
    : CpuComputationReduceMemoryBase<T>(cb, molecules, propagator_computation_optimizer)
{
    try
    {
        #ifndef NDEBUG
        std::cout << "--------- Continuous Chain Solver, CPU Version ---------" << std::endl;
        #endif

        const int M = this->cb->get_total_grid();

        this->method = method;
        if(this->method == "pseudospectral")
        {
            if (numerical_method == "" || numerical_method == "rqm4")
                this->propagator_solver = new CpuSolverPseudoRQM4<T>(cb, molecules);
            else if (numerical_method == "rk2")
                this->propagator_solver = new CpuSolverPseudoRK2<T>(cb, molecules);
            else if (numerical_method == "etdrk4")
                this->propagator_solver = new CpuSolverPseudoETDRK4<T>(cb, molecules);
            else
                throw_with_line_number("Unknown pseudo-spectral this->method: '" + numerical_method + "'. Use 'rqm4', 'rk2', or 'etdrk4'.");
        }
        else if(this->method == "realspace")
        {
            if constexpr (std::is_same<T, double>::value)
            {
                // Local Richardson (cn-adi4-lr) or 2nd order (cn-adi2)
                bool use_4th_order = (numerical_method == "cn-adi4-lr");
                this->propagator_solver = new CpuSolverCNADI(cb, molecules, use_4th_order);
            }
            else
                throw_with_line_number("Currently, the realspace this->method is only available for double precision.");
        }
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
            this->phi_block[item.first] = new T[M]();  // Zero-initialize
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
                this->propagator_at_check_point[std::make_tuple(key, 0)] = new T[M];

            for(size_t d=0; d<deps.size(); d++)
            {
                std::string sub_dep = std::get<0>(deps[d]);
                int sub_n_segment   = std::get<1>(deps[d]);

                if(this->propagator_at_check_point.find(std::make_tuple(sub_dep, sub_n_segment)) == this->propagator_at_check_point.end())
                    this->propagator_at_check_point[std::make_tuple(sub_dep, sub_n_segment)] = new T[M];
            }

            #ifndef NDEBUG
            this->propagator_finished[key] = new bool[max_n_segment];
            for(int i=0; i<max_n_segment;i++)
                this->propagator_finished[key][i] = false;
            #endif
        }

        // Iterate through each element in the map
        #ifndef NDEBUG
        for (const auto& pair : this->propagator_at_check_point) {
            std::cout << "Key: " << std::get<0>(pair.first) << ", Value: " << std::get<1>(pair.first) << std::endl;
        }
        #endif

        // Find the total sum of segments for checkpoint interval calculation
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
        // - this->q_recal[0..this->checkpoint_interval-1]: workspace for storage (size = this->checkpoint_interval)
        // - this->q_pair[0-1]: ping-pong for q_right advancement
        // - this->q_skip[0-1]: ping-pong for skip phase (used in both compute_concentrations and compute_stress)
        const int workspace_size = this->checkpoint_interval;
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
            for(int n=this->checkpoint_interval; n<max_n_segment; n+=this->checkpoint_interval)
            {
                if(this->propagator_at_check_point.find(std::make_tuple(key, n)) == this->propagator_at_check_point.end())
                {
                    this->propagator_at_check_point[std::make_tuple(key, n)] = new T[M];
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

            int n_aggregated   = this->propagator_computation_optimizer->get_computation_block(key).v_u.size()/
                                 this->propagator_computation_optimizer->get_computation_block(key).n_repeated;
            int n_segment_left = this->propagator_computation_optimizer->get_computation_block(key).n_segment_left;

            if(this->propagator_at_check_point.find(std::make_tuple(key_left, n_segment_left)) == this->propagator_at_check_point.end())
            {
                this->propagator_at_check_point[std::make_tuple(key_left, n_segment_left)] = new T[M];
                #ifndef NDEBUG
                std::cout << "Allocated, " + key_left + ", " << n_segment_left << std::endl;
                #endif
            }

            if(this->propagator_at_check_point.find(std::make_tuple(key_right, 0)) == this->propagator_at_check_point.end())
            {
                this->propagator_at_check_point[std::make_tuple(key_right, 0)] = new T[M];
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
        this->sc = new Scheduler(this->propagator_computation_optimizer->get_computation_propagators(), 1); 

        this->propagator_solver->update_laplacian_operator();
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
CpuComputationReduceMemoryContinuous<T>::~CpuComputationReduceMemoryContinuous()
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

                // Display job info
                #ifndef NDEBUG
                std::cout << job << " started" << std::endl;
                #endif

                // Check key
                #ifndef NDEBUG
                if (this->propagator_at_check_point.find(std::make_tuple(key, n_segment_from)) == this->propagator_at_check_point.end())
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
                            throw_with_line_number("Could not find q_init[\"" + g + "\"]. Pass q_init to run() for grafted polymers.");
                        for(int i=0; i<M; i++)
                            this->q_pair[0][i] = q_init[g][i];
                    }
                    else
                    {
                        for(int i=0; i<M; i++)
                            this->q_pair[0][i] = 1.0;
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
                        for(int i=0; i<M; i++)
                            this->q_pair[0][i] = 0.0;
                        
                        // Add all propagators at junction if necessary 
                        for(size_t d=0; d<deps.size(); d++)
                        {
                            std::string sub_dep = std::get<0>(deps[d]);
                            int sub_n_segment   = std::get<1>(deps[d]);
                            int sub_n_repeated  = std::get<2>(deps[d]);

                            // Check sub key
                            #ifndef NDEBUG
                            if (this->propagator_at_check_point.find(std::make_tuple(sub_dep, sub_n_segment)) == this->propagator_at_check_point.end())
                                std::cout << "Could not find sub key '" + sub_dep + "[" + std::to_string(sub_n_segment) + "]. " << std::endl;
                            if (!this->propagator_finished[sub_dep][sub_n_segment])
                                std::cout << "Could not compute '" + key +  "', since '"+ sub_dep + std::to_string(sub_n_segment) + "' is not prepared." << std::endl;
                            #endif

                            T* _q_sub = this->propagator_at_check_point[std::make_tuple(sub_dep, sub_n_segment)];
                            for(int i=0; i<M; i++)
                                this->q_pair[0][i] += _q_sub[i]*static_cast<double>(sub_n_repeated);
                        }
                        #ifndef NDEBUG
                        this->propagator_finished[key][0] = true;
                        #endif

                        // std::cout << "finished, key, n: " + key + ", 0" << std::endl;
                    }
                    else if(key[0] == '(')
                    {
                        for(int i=0; i<M; i++)
                            this->q_pair[0][i] = 1.0;
                        
                        // Multiply all propagators at junction if necessary 
                        for(size_t d=0; d<deps.size(); d++)
                        {
                            std::string sub_dep = std::get<0>(deps[d]);
                            int sub_n_segment   = std::get<1>(deps[d]);
                            int sub_n_repeated  = std::get<2>(deps[d]);

                            // Check sub key
                            #ifndef NDEBUG
                            if (this->propagator_at_check_point.find(std::make_tuple(sub_dep, sub_n_segment)) == this->propagator_at_check_point.end())
                                std::cout << "Could not find sub key '" + sub_dep + "[" + std::to_string(sub_n_segment) + "]. " << std::endl;
                            if (!this->propagator_finished[sub_dep][sub_n_segment])
                                std::cout << "Could not compute '" + key +  "', since '"+ sub_dep + std::to_string(sub_n_segment) + "' is not prepared." << std::endl;
                            #endif

                            T* _q_sub = this->propagator_at_check_point[std::make_tuple(sub_dep, sub_n_segment)];
                            if (_q_sub == nullptr)
                                std::cout <<"_q_sub is a null pointer: " + sub_dep + "[" + std::to_string(sub_n_segment) + "]." << std::endl;
                            for(int i=0; i<M; i++)
                                this->q_pair[0][i] *= pow(_q_sub[i], sub_n_repeated);
                        }

                        #ifndef NDEBUG
                        this->propagator_finished[key][0] = true;
                        #endif

                        // std::cout << "finished, key, n: " + key + ", 0" << std::endl;
                    }
                }
        
                // Multiply mask
                if (n_segment_from == 0 && q_mask != nullptr)
                {
                    for(int i=0; i<M; i++)
                        this->q_pair[0][i] *= q_mask[i];
                }

                // Copy this->q_pair[0] to the this->propagator_at_check_point
                if(n_segment_from == 0)
                {
                    T* _q_target =  this->propagator_at_check_point[std::make_tuple(key, 0)];
                    for(int i=0; i<M; i++)
                        _q_target[i] = this->q_pair[0][i];
                }
                else
                {
                    T* _q_from = this->propagator_at_check_point[std::make_tuple(key, n_segment_from)];
                    for(int i=0; i<M; i++)
                        this->q_pair[0][i] = _q_from[i];
                }

                // Advance propagator successively
                // Get ds_index from the key
                int ds_index = PropagatorCode::get_ds_index_from_key(key);

                for(int n=n_segment_from; n<n_segment_to; n++)
                {
                    #ifndef NDEBUG
                    if (!this->propagator_finished[key][n])
                        std::cout << "unfinished, key: " + key + ", " + std::to_string(n) << std::endl;
                    if (this->propagator_finished[key][n+1])
                        std::cout << "already finished: " + key + ", " + std::to_string(n) << std::endl;
                    #endif

                    this->propagator_solver->advance_propagator(
                            this->q_pair[prev],
                            this->q_pair[next],
                            monomer_type, q_mask, ds_index);

                    #ifndef NDEBUG
                    this->propagator_finished[key][n+1] = true;
                    #endif

                    // Copy this->q_pair[next] to the this->propagator_at_check_point
                    if (this->propagator_at_check_point.find(std::make_tuple(key, n+1)) != this->propagator_at_check_point.end())
                    {
                        T* _q_target =  this->propagator_at_check_point[std::make_tuple(key, n+1)];
                        for(int i=0; i<M; i++)
                            _q_target[i] = this->q_pair[next][i];
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
void CpuComputationReduceMemoryContinuous<T>::compute_concentrations()
{
    try
    {
        #ifndef NDEBUG
        std::cout << "compute_concentrations 0" << std::endl;
        #endif

        const int M = this->cb->get_total_grid();

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
                for(int i=0; i<M;i++)
                    block->second[i] = 0.0;
                continue;
            }

            // Check keys
            #ifndef NDEBUG
            if (this->propagator_at_check_point.find(std::make_tuple(key_left, n_segment_left-n_segment_right)) == this->propagator_at_check_point.end())
                std::cout << "Check point at " + key_left + "[" + std::to_string(n_segment_left-n_segment_right) + "] is missing. ";
            if (this->propagator_at_check_point.find(std::make_tuple(key_right, 0)) == this->propagator_at_check_point.end())
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

            // Get local_ds from ds_index encoded in key (DK+M format)
            Polymer& pc = this->molecules->get_polymer(p);
            int ds_index = PropagatorCode::get_ds_index_from_key(key_right);
            const ContourLengthMapping& mapping = this->molecules->get_contour_length_mapping();
            double local_ds = mapping.get_ds_from_index(ds_index);

            // Normalize concentration: local_ds * volume_fraction / alpha
            T norm = (local_ds*pc.get_volume_fraction()/pc.get_alpha()*n_repeated)/this->single_polymer_partitions[p];
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

            T *_phi = this->phi_solvent[s];
            T *_exp_dw = this->propagator_solver->exp_dw[1][monomer_type].data();

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
    // This function is designed to work within one checkpoint interval (N_RIGHT <= this->checkpoint_interval).
    // For larger ranges, callers should use block-based computation.
    // Returns pointers where:
    //   - Checkpoint positions point to stored checkpoint arrays
    //   - Non-checkpoint positions point to q_recal workspace [2..this->checkpoint_interval+2]
    try
    {
        const double *q_mask = this->cb->get_mask();
        int ds_index = PropagatorCode::get_ds_index_from_key(key);

        // Output array of pointers (will contain mix of checkpoint pointers and q_recal pointers)
        std::vector<T*> q_out(N_RIGHT + 1, nullptr);

        // First pass: link all checkpoint positions
        for(int n=0; n<=N_RIGHT; n++)
        {
            auto it = this->propagator_at_check_point.find(std::make_tuple(key, N_START+n));
            if(it != this->propagator_at_check_point.end())
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
                if(ws_idx >= static_cast<int>(this->q_recal.size()))
                {
                    throw_with_line_number("Workspace overflow in recalcaulte_propagator at ws_idx=" +
                        std::to_string(ws_idx) + " (size=" + std::to_string(this->q_recal.size()) + "). " +
                        "N_RIGHT=" + std::to_string(N_RIGHT) + ", n=" + std::to_string(n) + ". " +
                        "This function should only be called with N_RIGHT <= this->checkpoint_interval.");
                }
                this->propagator_solver->advance_propagator(q_prev, this->q_recal[ws_idx], monomer_type, q_mask, ds_index);
                q_out[n] = this->q_recal[ws_idx];
                q_prev = this->q_recal[ws_idx];
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
        // We process in blocks of size this->checkpoint_interval, recomputing q_left from nearest checkpoint.
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
        const int k = this->checkpoint_interval;
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
            for(int cp = this->checkpoint_interval; cp <= left_start; cp += this->checkpoint_interval)
            {
                if(this->propagator_at_check_point.find(std::make_tuple(key_left, cp)) != this->propagator_at_check_point.end())
                    check_pos = cp;
            }

            #ifndef NDEBUG
            std::cout << "Block " << block << ": n=[" << n_start << "," << n_end << "], "
                      << "left=[" << left_start << "," << left_end << "], check_pos=" << check_pos << std::endl;
            #endif

            // Recompute q_left from check_pos to left_end
            // Skip phase uses this->q_skip[0-1] for ping-pong, storage uses this->q_recal[0+]
            const int steps_before = left_start - check_pos;  // Steps before we start storing
            const int storage_count = left_end - left_start + 1;  // Number of values to store

            // Load checkpoint
            T* q_checkpoint = this->propagator_at_check_point[std::make_tuple(key_left, check_pos)];
            T* q_prev;
            T* q_curr;

            if(steps_before == 0)
            {
                // Checkpoint is exactly at left_start, store directly in this->q_recal[0]
                for(int i=0; i<M; i++)
                    this->q_recal[0][i] = q_checkpoint[i];
                q_prev = this->q_recal[0];
            }
            else
            {
                // Start ping-pong from checkpoint using this->q_skip[0-1]
                for(int i=0; i<M; i++)
                    this->q_skip[0][i] = q_checkpoint[i];
                q_prev = this->q_skip[0];
                int ping_pong = 1;

                // Compute steps before left_start using q_skip ping-pong
                for(int step = 1; step < steps_before; step++)
                {
                    int actual_pos = check_pos + step;
                    q_curr = this->q_skip[ping_pong];

                    auto it = this->propagator_at_check_point.find(std::make_tuple(key_left, actual_pos));
                    if(it != this->propagator_at_check_point.end())
                    {
                        for(int i=0; i<M; i++)
                            q_curr[i] = it->second[i];
                    }
                    else
                    {
                        this->propagator_solver->advance_propagator(q_prev, q_curr, monomer_type, q_mask, ds_index_left);
                    }
                    q_prev = q_curr;
                    ping_pong = 1 - ping_pong;
                }

                // Compute the step that reaches left_start, store in this->q_recal[0]
                int actual_pos = check_pos + steps_before;
                auto it = this->propagator_at_check_point.find(std::make_tuple(key_left, actual_pos));
                if(it != this->propagator_at_check_point.end())
                {
                    for(int i=0; i<M; i++)
                        this->q_recal[0][i] = it->second[i];
                }
                else
                {
                    this->propagator_solver->advance_propagator(q_prev, this->q_recal[0], monomer_type, q_mask, ds_index_left);
                }
                q_prev = this->q_recal[0];
            }

            // Compute remaining positions [left_start+1, left_end] and store in this->q_recal[1+]
            for(int idx = 1; idx < storage_count; idx++)
            {
                int actual_pos = left_start + idx;
                q_curr = this->q_recal[idx];

                auto it = this->propagator_at_check_point.find(std::make_tuple(key_left, actual_pos));
                if(it != this->propagator_at_check_point.end())
                {
                    for(int i=0; i<M; i++)
                        q_curr[i] = it->second[i];
                }
                else
                {
                    this->propagator_solver->advance_propagator(q_prev, q_curr, monomer_type, q_mask, ds_index_left);
                }
                q_prev = q_curr;
            }

            // Now this->q_recal[idx] contains q_left[left_start + idx] for idx in [0, storage_count-1]

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
                else
                {
                    // Advance from current position to n
                    while(current_n_right < n)
                    {
                        T* q_out = this->q_pair[right_next_idx];
                        this->propagator_solver->advance_propagator(q_right_prev, q_out, monomer_type, q_mask, ds_index_right);
                        q_right_prev = q_out;
                        q_right_curr = q_out;
                        std::swap(right_prev_idx, right_next_idx);
                        current_n_right++;
                    }
                }

                // Get q_left[N_LEFT - n]
                int left_pos = N_LEFT - n;
                int storage_idx = left_pos - left_start;
                T* q_left_n = this->q_recal[storage_idx];

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
void CpuComputationReduceMemoryContinuous<T>::compute_stress()
{
    // This this->method should be invoked after invoking compute_statistics().

    // To calculate stress, we multiply weighted fourier basis to q(k)*q^dagger(-k).
    // We only need the real part of stress calculation.

    // Uses block-based computation to minimize memory usage (O(sqrt(N)) workspace).
    try
    {
        if (this->method == "realspace")
            throw_with_line_number("Currently, the real-space this->method does not support stress computation.");

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
                    if(this->propagator_at_check_point.find(std::make_tuple(key_left, cp)) != this->propagator_at_check_point.end())
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
                auto it_checkpoint = this->propagator_at_check_point.find(std::make_tuple(key_left, check_pos));
                if(it_checkpoint == this->propagator_at_check_point.end())
                    continue;
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
                            this->propagator_solver->advance_propagator(q_prev_left, q_curr_left, monomer_type, q_mask, ds_index_left);
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
                        this->propagator_solver->advance_propagator(q_prev_left, this->q_recal[0], monomer_type, q_mask, ds_index_left);
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
                        this->propagator_solver->advance_propagator(q_prev_left, q_curr_left, monomer_type, q_mask, ds_index_left);
                    }
                    q_prev_left = q_curr_left;
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
                    else
                    {
                        while(current_n_right < n)
                        {
                            T* q_out = this->q_pair[right_next_idx];
                            this->propagator_solver->advance_propagator(q_right_prev, q_out, monomer_type, q_mask, ds_index_right);
                            q_right_prev = q_out;
                            q_right_curr = q_out;
                            std::swap(right_prev_idx, right_next_idx);
                            current_n_right++;
                        }
                    }

                    // Get q_left[N_LEFT - n]
                    int left_pos = N_LEFT - n;
                    int storage_idx = left_pos - left_start;
                    T* q_left_n = this->q_recal[storage_idx];

                    // Compute stress contribution
                    std::vector<T> segment_stress = this->propagator_solver->compute_single_segment_stress(
                        q_left_n, q_right_curr, monomer_type, false);

                    for(int d=0; d<N_STRESS; d++)
                        _block_dq_dl[d] += segment_stress[d]*(s_coeff[n]*n_repeated);
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
void CpuComputationReduceMemoryContinuous<T>::get_chain_propagator(T *q_out, int polymer, int v, int u, int n)
{
    // This this->method should be invoked after invoking compute_statistics()

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
        if (this->propagator_at_check_point.find(checkpoint_key) != this->propagator_at_check_point.end())
        {
            // Directly copy from checkpoint
            T *_q_from = this->propagator_at_check_point[checkpoint_key];
            for(int i=0; i<M; i++)
                q_out[i] = _q_from[i];
        }
        else
        {
            // Find nearest checkpoint at or before position n
            int check_pos = 0;  // Start with 0 (always exists)
            for(int cp = this->checkpoint_interval; cp < n; cp += this->checkpoint_interval)
            {
                if(this->propagator_at_check_point.find(std::make_tuple(dep, cp)) != this->propagator_at_check_point.end())
                    check_pos = cp;
            }

            // Recalculate from checkpoint to position n
            std::string monomer_type = this->propagator_computation_optimizer->get_computation_propagator(dep).monomer_type;
            const double *q_mask = this->cb->get_mask();
            int ds_index = PropagatorCode::get_ds_index_from_key(dep);

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
                    this->propagator_solver->advance_propagator(q_prev, q_curr, monomer_type, q_mask, ds_index);
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
                if(this->propagator_at_check_point.find(std::make_tuple(key_left, cp)) != this->propagator_at_check_point.end())
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
            auto it_checkpoint = this->propagator_at_check_point.find(std::make_tuple(key_left, check_pos));
            if(it_checkpoint == this->propagator_at_check_point.end())
                continue;
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
                        this->propagator_solver->advance_propagator(q_prev_left, q_curr_left, monomer_type, q_mask, ds_index_left);
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
                    this->propagator_solver->advance_propagator(q_prev_left, this->q_recal[0], monomer_type, q_mask, ds_index_left);
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
                    this->propagator_solver->advance_propagator(q_prev_left, q_curr_left, monomer_type, q_mask, ds_index_left);
                }
                q_prev_left = q_curr_left;
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
                else
                {
                    while(current_n_right < n)
                    {
                        T* q_out = this->q_pair[right_next_idx];
                        this->propagator_solver->advance_propagator(q_right_prev, q_out, monomer_type, q_mask, ds_index_right);
                        q_right_prev = q_out;
                        q_right_curr = q_out;
                        std::swap(right_prev_idx, right_next_idx);
                        current_n_right++;
                    }
                }

                // Get q_left[N_LEFT - n]
                int left_pos = N_LEFT - n;
                int storage_idx = left_pos - left_start;
                T* q_left_n = this->q_recal[storage_idx];

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