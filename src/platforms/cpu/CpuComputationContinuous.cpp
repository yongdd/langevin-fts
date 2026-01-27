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
#include "SpaceGroup.h"
#include "CpuSolverPseudoRQM4.h"
#include "CpuSolverPseudoRK2.h"
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
 * @param numerical_method               Numerical algorithm (e.g., "rqm4", "rk2", "cn-adi2")
 */
template <typename T>
CpuComputationContinuous<T>::CpuComputationContinuous(
    ComputationBox<T>* cb,
    Molecules *molecules,
    PropagatorComputationOptimizer *propagator_computation_optimizer,
    std::string method,
    std::string numerical_method,
    FFTBackend backend,
    SpaceGroup* space_group)
    : CpuComputationBase<T>(cb, molecules, propagator_computation_optimizer)
{
    try
    {
        #ifndef NDEBUG
        std::cout << "--------- Continuous Chain Solver, CPU Version ---------" << std::endl;
        #endif

        // Set space group first so that get_n_basis() returns the correct size
        if (space_group != nullptr)
            PropagatorComputation<T>::set_space_group(space_group);

        const int N = this->cb->get_n_basis();  // n_irreducible (with space group) or total_grid

        this->method = method;
        if(method == "pseudospectral")
        {
            if (numerical_method == "" || numerical_method == "rqm4")
                this->propagator_solver = new CpuSolverPseudoRQM4<T>(cb, molecules, backend);
            else if (numerical_method == "rk2")
                this->propagator_solver = new CpuSolverPseudoRK2<T>(cb, molecules, backend);
            else
                throw_with_line_number("Unknown pseudo-spectral method: '" + numerical_method + "'. Use 'rqm4' or 'rk2'.");
        }
        else if(method == "realspace")
        {
            if constexpr (std::is_same<T, double>::value)
            {
                if (numerical_method != "" && numerical_method != "cn-adi2")
                {
                    throw_with_line_number("Unknown realspace method: '" + numerical_method + "'. Use 'cn-adi2'.");
                }
                this->propagator_solver = new CpuSolverCNADI(cb, molecules);
            }
            else
                throw_with_line_number("Currently, the realspace method is only available for double precision.");
        }

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
            std::string key = item.first;
            int max_n_segment = item.second.max_n_segment+1;

            this->propagator_size[key] = max_n_segment;
            this->propagator[key] = new T*[max_n_segment];
            for(int i=0; i<this->propagator_size[key]; i++)
                this->propagator[key][i] = new T[N];

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
            this->phi_solvent.push_back(new T[N]);

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
        const int N = (this->space_group_ != nullptr) ? this->space_group_->get_n_irreducible() : M;
        const bool use_reduced_basis = (this->space_group_ != nullptr);

        for(const auto& item: this->propagator_computation_optimizer->get_computation_propagators())
        {
            if( !w_input.contains(item.second.monomer_type))
                throw_with_line_number("monomer_type \"" + item.second.monomer_type + "\" is not in w_input.");
        }

        // Store w for solvent computation (reduced basis when space_group is set)
        for(const auto& item: w_input)
        {
            std::string monomer_type = item.first;
            const T* _w = item.second;
            this->w[monomer_type].resize(N);
            for(int i=0; i<N; i++)
                this->w[monomer_type][i] = _w[i];
        }

        // Update dw or exp_dw
        this->propagator_solver->update_dw(w_input);

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
                    // q_init is in reduced basis when space_group is set, otherwise full grid
                    if (key[0] == '{')
                    {
                        std::string g = PropagatorCode::get_q_input_idx_from_key(key);
                        if (!q_init.contains(g))
                            throw_with_line_number("Could not find q_init[\"" + g + "\"]. Pass q_init to run() for grafted polymers.");
                        for(int i=0; i<N; i++)
                            _propagator[0][i] = q_init[g][i];
                    }
                    else
                    {
                        for(int i=0; i<N; i++)
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
                        for(int i=0; i<N; i++)
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
                            for(int i=0; i<N; i++)
                                _propagator[0][i] += _propagator_sub_dep[sub_n_segment][i]*static_cast<double>(sub_n_repeated);
                        }
                        #ifndef NDEBUG
                        this->propagator_finished[key][0] = true;
                        #endif
                        // std::cout << "finished, key, n: " + key + ", 0" << std::endl;
                    }
                    else if(key[0] == '(')
                    {
                        for(int i=0; i<N; i++)
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
                            for(int i=0; i<N; i++)
                                _propagator[0][i] *= pow(_propagator_sub_dep[sub_n_segment][i], sub_n_repeated);
                        }

                        #ifndef NDEBUG
                        this->propagator_finished[key][0] = true;
                        #endif
                        // std::cout << "finished, key, n: " + key + ", 0" << std::endl;
                    }
                }
        
                // Apply mask
                if (n_segment_from == 0 && this->cb->get_mask() != nullptr)
                {
                    const double* q_mask = this->cb->get_mask();
                    for(int i=0; i<N; i++)
                        _propagator[0][i] *= q_mask[i];
                }

                // Get ds_index from the key
                int ds_index = PropagatorCode::get_ds_index_from_key(key);

                // Reset solver internal state when starting a new propagator
                // (needed for Global Richardson method)
                if (n_segment_from == 0)
                    this->propagator_solver->reset_internal_state();

                // Advance propagator successively
                // Solver handles expand/reduce internally when space_group is set
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
                            monomer_type, this->cb->get_mask(), ds_index);

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

        this->propagator_solver->advance_propagator(q_init, q_out, monomer_type, this->cb->get_mask(), ds_index);
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
        const int N = (this->space_group_ != nullptr) ? this->space_group_->get_n_irreducible() : M;

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
                for(int i=0; i<N;i++)
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

            // Get local_ds from ds_index encoded in key (DK+M format)
            Polymer& pc = this->molecules->get_polymer(p);
            int ds_index = PropagatorCode::get_ds_index_from_key(key_right);
            const ContourLengthMapping& mapping = this->molecules->get_contour_length_mapping();
            double local_ds = mapping.get_ds_from_index(ds_index);

            // Normalize concentration: local_ds * volume_fraction / alpha
            T norm = (local_ds*pc.get_volume_fraction()/pc.get_alpha()*n_repeated)/this->single_polymer_partitions[p];
            for(int i=0; i<N; i++)
                block->second[i] *= norm;
        }

        // Calculate partition functions and concentrations of solvents
        double ds = this->molecules->get_contour_length_mapping().get_global_ds();
        for(int s=0; s<this->molecules->get_n_solvent_types(); s++)
        {
            double volume_fraction   = std::get<0>(this->molecules->get_solvent(s));
            std::string monomer_type = std::get<1>(this->molecules->get_solvent(s));

            T *_phi = this->phi_solvent[s];
            const T* _w = w[monomer_type].data();

            // Compute phi = exp(-w*ds)
            for(int i=0; i<N; i++)
                _phi[i] = exp(-_w[i] * ds);

            // Partition function and normalization
            this->single_solvent_partitions[s] = this->cb->mean(_phi);
            T norm = volume_fraction / this->single_solvent_partitions[s];
            for(int i=0; i<N; i++)
                _phi[i] *= norm;
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
        const int N = this->cb->get_n_basis();  // n_irreducible (with space group) or total_grid
        std::vector<double> simpson_rule_coeff = SimpsonRule::get_coeff(N_RIGHT);

        // Compute segment concentration
        for(int i=0; i<N; i++)
            phi[i] = simpson_rule_coeff[0]*q_1[N_LEFT][i]*q_2[0][i];
        for(int n=1; n<=N_RIGHT; n++)
        {
            for(int i=0; i<N; i++)
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

            // If there is no segment
            if(N_RIGHT == 0)
                continue;

            T **q_1 = this->propagator[key_left];     // dependency v
            T **q_2 = this->propagator[key_right];    // dependency u

            std::vector<double> s_coeff = SimpsonRule::get_coeff(N_RIGHT);
            std::array<T,6> _block_dq_dl = block_dq_dl[key];

            // Check if propagators at endpoints are leaf nodes (initial conditions)
            bool left_is_leaf = (this->propagator_computation_optimizer->get_computation_propagator(key_left).deps.size() == 0);
            bool right_is_leaf = (this->propagator_computation_optimizer->get_computation_propagator(key_right).deps.size() == 0);

            // Compute stress contributions using Simpson's rule
            // Skip endpoint terms where propagator is initial condition (q=1)
            for(int n=0; n<=N_RIGHT; n++)
            {
                // At n=0: q_2[0] is initial condition if key_right is leaf
                if (n == 0 && right_is_leaf)
                    continue;

                // At n=N_RIGHT: q_1[0] is initial condition if key_left is leaf
                if (n == N_RIGHT && left_is_leaf && N_LEFT == N_RIGHT)
                    continue;

                // Solver handles expand/reduce internally when space_group is set
                std::vector<T> segment_stress = this->propagator_solver->compute_single_segment_stress(
                    q_1[N_LEFT-n], q_2[n], monomer_type, false);

                for(int d=0; d<N_STRESS; d++)
                    _block_dq_dl[d] += segment_stress[d]*(s_coeff[n]*n_repeated);
            }

            // Multiply by local_ds for this block (get ds_index from key)
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
void CpuComputationContinuous<T>::get_chain_propagator(T *q_out, int polymer, int v, int u, int n)
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
        if (n < 0 || n > N_RIGHT)
            throw_with_line_number("n (" + std::to_string(n) + ") must be in range [0, " + std::to_string(N_RIGHT) + "]");

        T **_partition = this->propagator[dep];

        for(int i=0; i<N; i++)
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
