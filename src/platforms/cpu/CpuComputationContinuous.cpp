/**
 * @file CpuComputationContinuous.cpp
 * @brief CPU implementation of propagator computation for continuous chains.
 *
 * Implements the PropagatorComputation interface for Gaussian (continuous)
 * chain models using OpenMP parallelization. Supports both pseudo-spectral
 * and real-space solvers.
 *
 * **Propagator Algorithm:**
 *
 * Solves the modified diffusion equation:
 *     ∂q/∂s = (b²/6)∇²q - w(r)q
 *
 * using either pseudo-spectral (4th-order Richardson) or real-space
 * (Crank-Nicolson) methods.
 *
 * **Parallelization:**
 *
 * Uses OpenMP for parallel propagator computation. Number of threads
 * determined by OMP_NUM_THREADS environment variable (default: 8).
 *
 * **Concentration Calculation:**
 *
 * Uses Simpson's rule integration along the contour:
 *     φ(r) = (1/Q) ∫ q(r,s) × q†(r,1-s) ds
 *
 * **Template Instantiations:**
 *
 * - CpuComputationContinuous<double>: Real fields
 * - CpuComputationContinuous<std::complex<double>>: Complex fields
 *
 * @see CpuSolverPseudoRQM4 for pseudo-spectral solver
 * @see CpuSolverCNADI for real-space solver
 */

#include <cmath>
#include <numbers>
#include <omp.h>

#include "CpuComputationContinuous.h"
#include "CpuSolverPseudoRQM4.h"
#include "CpuSolverPseudoETDRK4.h"
#include "CpuSolverCNADI.h"
#include "SimpsonRule.h"
#include "PropagatorCode.h"

/**
 * @brief Construct CPU propagator computation for continuous chains.
 *
 * Allocates propagator arrays, sets up parallelization, and creates
 * the appropriate solver (pseudo-spectral or real-space).
 *
 * @param cb                             Computation box for grid operations
 * @param molecules                      Polymer/solvent species definitions
 * @param propagator_computation_optimizer Optimized computation schedule
 * @param method                         "pseudospectral" or "realspace"
 * @param numerical_method               Numerical algorithm (e.g., "rqm4", "etdrk4", "cn-adi2", "cn-adi4")
 */
template <typename T>
CpuComputationContinuous<T>::CpuComputationContinuous(
    ComputationBox<T>* cb,
    Molecules *molecules,
    PropagatorComputationOptimizer *propagator_computation_optimizer,
    std::string method,
    std::string numerical_method)
    : CpuComputationBase<T>(cb, molecules, propagator_computation_optimizer)
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
                // Per-step Richardson (cn-adi4) or 2nd order (cn-adi2)
                bool use_4th_order = (numerical_method == "cn-adi4");
                this->propagator_solver = new CpuSolverCNADI(cb, molecules, use_4th_order);
            }
            else
                throw_with_line_number("Currently, the realspace method is only available for double precision.");
        }
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
            std::string key = item.first;
            int max_n_segment = item.second.max_n_segment+1;

            this->propagator_size[key] = max_n_segment;
            this->propagator[key] = new T*[max_n_segment];
            for(int i=0; i<this->propagator_size[key]; i++)
                this->propagator[key][i] = new T[M];

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

            int n_aggregated   = this->propagator_computation_optimizer->get_computation_block(key).v_u.size()/
                                 this->propagator_computation_optimizer->get_computation_block(key).n_repeated;
            int n_segment_left = this->propagator_computation_optimizer->get_computation_block(key).n_segment_left;

            single_partition_segment.push_back(std::make_tuple(
                p,
                this->propagator[key_left][n_segment_left],  // q
                this->propagator[key_right][0],              // q_dagger
                n_aggregated                           // how many propagators are aggregated
                ));
            current_p++;
        }
        // Concentrations for each solvent
        for(int s=0;s<this->molecules->get_n_solvent_types();s++)
            this->phi_solvent.push_back(new T[M]);

        // Create scheduler for computation of propagator
        this->sc = new Scheduler(this->propagator_computation_optimizer->get_computation_propagators(), this->n_streams); 

        this->propagator_solver->update_laplacian_operator();
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
CpuComputationContinuous<T>::~CpuComputationContinuous()
{
    delete this->propagator_solver;
    delete this->sc;

    for(const auto& item: this->propagator)
    {
        for(int i=0; i<this->propagator_size[item.first]; i++)
            delete[] item.second[i];
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
void CpuComputationContinuous<T>::compute_statistics(
    std::map<std::string, const T*> w_input,
    std::map<std::string, const T*> q_init)
{
    this->compute_propagators(w_input, q_init);
    this->compute_concentrations();
}
template <typename T>
void CpuComputationContinuous<T>::compute_propagators(
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
            // display all jobs
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
            #pragma omp parallel for num_threads(this->n_streams)
            for(size_t job=0; job<parallel_job->size(); job++)
            {
                auto& key = std::get<0>((*parallel_job)[job]);
                int n_segment_from = std::get<1>((*parallel_job)[job]);
                int n_segment_to   = std::get<2>((*parallel_job)[job]);
                auto& deps = this->propagator_computation_optimizer->get_computation_propagator(key).deps;
                auto monomer_type = this->propagator_computation_optimizer->get_computation_propagator(key).monomer_type;

                // // Display job info
                // #ifndef NDEBUG
                // std::cout << job << " started" << std::endl;
                // #endif

                // Check key
                #ifndef NDEBUG
                if (!this->propagator.contains(key))
                    std::cout << "Could not find key '" + key + "'. " << std::endl;
                #endif

                T **_propagator = this->propagator[key];

                // If it is leaf node
                if(n_segment_from == 0 && deps.size() == 0) 
                {
                     // q_init
                    if (key[0] == '{')
                    {
                        std::string g = PropagatorCode::get_q_input_idx_from_key(key);
                        if (!q_init.contains(g))
                            std::cout << "Could not find q_init[\"" + g + "\"]." << std::endl;
                        for(int i=0; i<M; i++)
                            _propagator[0][i] = q_init[g][i];
                    }
                    else
                    {
                        for(int i=0; i<M; i++)
                            _propagator[0][i] = 1.0;
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
                            _propagator[0][i] = 0.0;
                        
                        // Add all propagators at junction if necessary 
                        for(size_t d=0; d<deps.size(); d++)
                        {
                            std::string sub_dep = std::get<0>(deps[d]);
                            int sub_n_segment   = std::get<1>(deps[d]);
                            int sub_n_repeated  = std::get<2>(deps[d]);

                            // Check sub key
                            #ifndef NDEBUG
                            if (!this->propagator.contains(sub_dep))
                                std::cout << "Could not find sub key '" + sub_dep + "'. " << std::endl;
                            if (!this->propagator_finished[sub_dep][sub_n_segment])
                                std::cout << "Could not compute '" + key +  "', since '"+ sub_dep + std::to_string(sub_n_segment) + "' is not prepared." << std::endl;
                            #endif

                            T **_propagator_sub_dep = this->propagator[sub_dep];
                            for(int i=0; i<M; i++)
                                _propagator[0][i] += _propagator_sub_dep[sub_n_segment][i]*static_cast<double>(sub_n_repeated);
                        }
                        #ifndef NDEBUG
                        this->propagator_finished[key][0] = true;
                        #endif
                        // std::cout << "finished, key, n: " + key + ", 0" << std::endl;
                    }
                    else if(key[0] == '(')
                    {
                        for(int i=0; i<M; i++)
                            _propagator[0][i] = 1.0;
                        
                        // Multiply all propagators at junction if necessary 
                        for(size_t d=0; d<deps.size(); d++)
                        {
                            std::string sub_dep = std::get<0>(deps[d]);
                            int sub_n_segment   = std::get<1>(deps[d]);
                            int sub_n_repeated  = std::get<2>(deps[d]);

                            // Check sub key
                            #ifndef NDEBUG
                            if (!this->propagator.contains(sub_dep))
                                std::cout << "Could not find sub key '" + sub_dep + "'. " << std::endl;
                            if (!this->propagator_finished[sub_dep][sub_n_segment])
                                std::cout << "Could not compute '" + key +  "', since '"+ sub_dep + std::to_string(sub_n_segment) + "' is not prepared." << std::endl;
                            #endif

                            T **_propagator_sub_dep = this->propagator[sub_dep];
                            for(int i=0; i<M; i++)
                                _propagator[0][i] *= pow(_propagator_sub_dep[sub_n_segment][i], sub_n_repeated);
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
                        _propagator[0][i] *= q_mask[i];
                }

                // Get ds_index from the key
                int ds_index = PropagatorCode::get_ds_index_from_key(key);
                if (ds_index < 1) ds_index = 1;  // Default to global ds

                // Reset solver internal state when starting a new propagator
                // (needed for Global Richardson method)
                if (n_segment_from == 0)
                    this->propagator_solver->reset_internal_state();

                // Advance propagator successively
                for(int n=n_segment_from; n<n_segment_to; n++)
                {
                    #ifndef NDEBUG
                    if (!this->propagator_finished[key][n])
                        std::cout << "unfinished, key: " + key + ", " + std::to_string(n) << std::endl;
                    if (this->propagator_finished[key][n+1])
                        std::cout << "already finished: " + key + ", " + std::to_string(n) << std::endl;
                    #endif

                    this->propagator_solver->advance_propagator(
                            _propagator[n],
                            _propagator[n+1],
                            monomer_type, q_mask, ds_index);

                    #ifndef NDEBUG
                    this->propagator_finished[key][n+1] = true;
                    #endif
                }
                // // Display job info
                // #ifndef NDEBUG
                // std::cout << job << " finished" << std::endl;
                // #endif
            }
        }

        // for(const auto& block: this->phi_block)
        // {
        //     int p                = std::get<0>(block.first);
        //     std::string key_left    = std::get<1>(block.first);
        //     std::string key_right    = std::get<2>(block.first);
        //     int n_segment        = std::get<3>(block.first);

        //     // Check keys
        //     if (!this->propagator.contains(key_left))
        //         throw_with_line_number("Could not find key_left key'" + key_left + "'. ");
        //     if (!this->propagator.contains(key_right))
        //         throw_with_line_number("Could not find key_right key'" + key_right + "'. ");

        //     for(int i=0; i<=n_segment; i++)
        //     {
        //         if (!this->propagator_finished[key_left][i])
        //             throw_with_line_number("unfinished, key_left, n'" + key_left + ", " + std::to_string(i) + "'. ");
        //     }

        //     for(int i=0; i<=n_segment; i++)
        //     {
        //         if (!this->propagator_finished[key_right][i])
        //             throw_with_line_number("unfinished, key_right, n'" + key_right + ", " + std::to_string(i) + "'. ");
        //     }
        // }

        // Compute total partition function of each distinct polymers
        for(const auto& segment_info: single_partition_segment)
        {
            int p                    = std::get<0>(segment_info);
            T *propagator_left  = std::get<1>(segment_info);
            T *propagator_right = std::get<2>(segment_info);
            int n_aggregated         = std::get<3>(segment_info);

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
void CpuComputationContinuous<T>::advance_propagator_single_segment(
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
void CpuComputationContinuous<T>::compute_concentrations()
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
            int n_repeated      = this->propagator_computation_optimizer->get_computation_block(key).n_repeated;

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
                n_segment_left,
                n_segment_right
            );

            // Normalize concentration
            Polymer& pc = this->molecules->get_polymer(p);
            T norm = (this->molecules->get_ds()*pc.get_volume_fraction()/pc.get_alpha()*n_repeated)/this->single_polymer_partitions[p];
            for(int i=0; i<M; i++)
                block->second[i] *= norm;
        }

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
void CpuComputationContinuous<T>::calculate_phi_one_block(
    T *phi, T **q_1, T **q_2, const int N_LEFT, const int N_RIGHT)
{
    try
    {
        const int M = this->cb->get_total_grid();
        std::vector<double> simpson_rule_coeff = SimpsonRule::get_coeff(N_RIGHT);

        // Compute segment concentration
        for(int i=0; i<M; i++)
            phi[i] = simpson_rule_coeff[0]*q_1[N_LEFT][i]*q_2[0][i];
        for(int n=1; n<=N_RIGHT; n++)
        {
            for(int i=0; i<M; i++)
                phi[i] += simpson_rule_coeff[n]*q_1[N_LEFT-n][i]*q_2[n][i];
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
void CpuComputationContinuous<T>::compute_stress()
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

            // If there is no segment
            if(N_RIGHT == 0)
                continue;

            T **q_1 = this->propagator[key_left];     // dependency v
            T **q_2 = this->propagator[key_right];    // dependency u

            std::vector<double> s_coeff = SimpsonRule::get_coeff(N_RIGHT);
            std::array<T,6> _block_dq_dl = block_dq_dl[key];

            // Compute
            for(int n=0; n<=N_RIGHT; n++)
            {
                std::vector<T> segment_stress = this->propagator_solver->compute_single_segment_stress(
                    q_1[N_LEFT-n], q_2[n], monomer_type, false);

                // std::cout << key_left << ", "  << key_right << ", " << n << ", " << segment_stress[0] << ", " << segment_stress[1] << ", " << segment_stress[2] << std::endl;

                for(int d=0; d<N_STRESS; d++)
                    _block_dq_dl[d] += segment_stress[d]*(s_coeff[n]*n_repeated);
            }
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
void CpuComputationContinuous<T>::get_chain_propagator(T *q_out, int polymer, int v, int u, int n)
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
        if (n < 0 || n > N_RIGHT)
            throw_with_line_number("n (" + std::to_string(n) + ") must be in range [0, " + std::to_string(N_RIGHT) + "]");

        T **_partition = this->propagator[dep];
        for(int i=0; i<M; i++)
            q_out[i] = _partition[n][i];
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
bool CpuComputationContinuous<T>::check_total_partition()
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

        #ifndef NDEBUG
        std::cout<< p << ", " << key_left << ", " << key_right << ": " << n_segment_left << ", " << n_segment_right << ", " << n_propagators << ", " << this->propagator_computation_optimizer->get_computation_block(key).n_repeated << std::endl;
        #endif

        for(int n=0;n<=n_segment_right;n++)
        {
            T total_partition = this->cb->inner_product(
                this->propagator[key_left][n_segment_left-n],
                this->propagator[key_right][n])*(n_repeated/this->cb->get_volume());

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
INSTANTIATE_CLASS(CpuComputationContinuous);