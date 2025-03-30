#include <cmath>
#include <omp.h>

#include "CpuComputationContinuous.h"
#include "CpuSolverPseudoContinuous.h"
#include "CpuSolverReal.h"
#include "SimpsonRule.h"

CpuComputationContinuous::CpuComputationContinuous(
    ComputationBox *cb,
    Molecules *molecules,
    PropagatorComputationOptimizer *propagator_computation_optimizer,
    std::string method)
    : PropagatorComputation(cb, molecules, propagator_computation_optimizer)
{
    try
    {
        #ifndef NDEBUG
        std::cout << "--------- Continuous Chain Solver, CPU Version ---------" << std::endl;
        #endif

        const int M = cb->get_total_grid();
        if(method == "pseudospectral")
            this->propagator_solver = new CpuSolverPseudoContinuous<double>(cb, molecules);
        else if(method == "realspace")
            this->propagator_solver = new CpuSolverReal(cb, molecules);

        // The number of parallel streams for propagator computation
        const char *ENV_OMP_NUM_THREADS = getenv("OMP_NUM_THREADS");
        std::string env_omp_num_threads(ENV_OMP_NUM_THREADS ? ENV_OMP_NUM_THREADS  : "");
        if (env_omp_num_threads.empty())
            n_streams = 4;
        else
            n_streams = std::stoi(env_omp_num_threads);
        #ifndef NDEBUG
        std::cout << "The number of CPU threads: " << n_streams << std::endl;
        #endif

        // Allocate memory for propagators
        if( propagator_computation_optimizer->get_computation_propagators().size() == 0)
            throw_with_line_number("There is no propagator code. Add polymers first.");
        for(const auto& item: propagator_computation_optimizer->get_computation_propagators())
        {
            std::string key = item.first;
            int max_n_segment = item.second.max_n_segment+1;

            propagator_size[key] = max_n_segment;
            propagator[key] = new double*[max_n_segment];
            for(int i=0; i<propagator_size[key]; i++)
                propagator[key][i] = new double[M];

            #ifndef NDEBUG
            propagator_finished[key] = new bool[max_n_segment];
            for(int i=0; i<max_n_segment;i++)
                propagator_finished[key][i] = false;
            #endif
        }

        // Allocate memory for concentrations
        if( propagator_computation_optimizer->get_computation_blocks().size() == 0)
            throw_with_line_number("There is no block. Add polymers first.");
        for(const auto& item: propagator_computation_optimizer->get_computation_blocks())
        {
            phi_block[item.first] = new double[M];
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

            int n_aggregated = propagator_computation_optimizer->get_computation_block(key).v_u.size()/
                               propagator_computation_optimizer->get_computation_block(key).n_repeated;
            int n_segment_left = propagator_computation_optimizer->get_computation_block(key).n_segment_left;

            single_partition_segment.push_back(std::make_tuple(
                p,
                propagator[key_left][n_segment_left],  // q
                propagator[key_right][0],              // q_dagger
                n_aggregated                           // how many propagators are aggregated
                ));
            current_p++;
        }
        // Concentrations for each solvent
        for(int s=0;s<molecules->get_n_solvent_types();s++)
            phi_solvent.push_back(new double[M]);

        // Create scheduler for computation of propagator
        sc = new Scheduler(propagator_computation_optimizer->get_computation_propagators(), n_streams); 

        propagator_solver->update_laplacian_operator();
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
CpuComputationContinuous::~CpuComputationContinuous()
{
    delete propagator_solver;
    delete sc;

    for(const auto& item: propagator)
    {
        for(int i=0; i<propagator_size[item.first]; i++)
            delete[] item.second[i];
        delete[] item.second;
    }

    for(const auto& item: phi_block)
        delete[] item.second;
    for(const auto& item: phi_solvent)
        delete[] item; 

    #ifndef NDEBUG
    for(const auto& item: propagator_finished)
        delete[] item.second;
    #endif
}
void CpuComputationContinuous::update_laplacian_operator()
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

void CpuComputationContinuous::compute_statistics(
    std::map<std::string, const double*> w_input,
    std::map<std::string, const double*> q_init)
{
    this->compute_propagators(w_input, q_init);
    this->compute_concentrations();
}

void CpuComputationContinuous::compute_propagators(
    std::map<std::string, const double*> w_input,
    std::map<std::string, const double*> q_init)
{
    try
    {
        const int M = cb->get_total_grid();

        for(const auto& item: propagator_computation_optimizer->get_computation_propagators())
        {
            if( w_input.find(item.second.monomer_type) == w_input.end())
                throw_with_line_number("monomer_type \"" + item.second.monomer_type + "\" is not in w_input.");
        }

        // Update dw or exp_dw
        propagator_solver->update_dw(w_input);

        // Assign a pointer for mask
        const double *q_mask = cb->get_mask();

        // For each time span
        auto& branch_schedule = sc->get_schedule();
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
            }
            #endif

            // For each propagator
            #pragma omp parallel for num_threads(n_streams)
            for(size_t job=0; job<parallel_job->size(); job++)
            {
                auto& key = std::get<0>((*parallel_job)[job]);
                int n_segment_from = std::get<1>((*parallel_job)[job]);
                int n_segment_to   = std::get<2>((*parallel_job)[job]);
                auto& deps = propagator_computation_optimizer->get_computation_propagator(key).deps;
                auto monomer_type = propagator_computation_optimizer->get_computation_propagator(key).monomer_type;

                // // Display job info
                // #ifndef NDEBUG
                // std::cout << job << " started" << std::endl;
                // #endif

                // Check key
                #ifndef NDEBUG
                if (propagator.find(key) == propagator.end())
                    std::cout << "Could not find key '" + key + "'. " << std::endl;
                #endif

                double **_propagator = propagator[key];

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
                            _propagator[0][i] = q_init[g][i];
                    }
                    else
                    {
                        for(int i=0; i<M; i++)
                            _propagator[0][i] = 1.0;
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
                            _propagator[0][i] = 0.0;
                        
                        // Add all propagators at junction if necessary 
                        for(size_t d=0; d<deps.size(); d++)
                        {
                            std::string sub_dep = std::get<0>(deps[d]);
                            int sub_n_segment   = std::get<1>(deps[d]);
                            int sub_n_repeated  = std::get<2>(deps[d]);

                            // Check sub key
                            #ifndef NDEBUG
                            if (propagator.find(sub_dep) == propagator.end())
                                std::cout << "Could not find sub key '" + sub_dep + "'. " << std::endl;
                            if (!propagator_finished[sub_dep][sub_n_segment])
                                std::cout << "Could not compute '" + key +  "', since '"+ sub_dep + std::to_string(sub_n_segment) + "' is not prepared." << std::endl;
                            #endif

                            double **_propagator_sub_dep = propagator[sub_dep];
                            for(int i=0; i<M; i++)
                                _propagator[0][i] += _propagator_sub_dep[sub_n_segment][i]*sub_n_repeated;
                        }
                        #ifndef NDEBUG
                        propagator_finished[key][0] = true;
                        #endif
                        // std::cout << "finished, key, n: " + key + ", 0" << std::endl;
                    }
                    else
                    {
                        for(int i=0; i<M; i++)
                            _propagator[0][i] = 1.0;
                        
                        // Multiply all propagators at junction if necessary 
                        for(size_t d=0; d<deps.size(); d++)
                        {
                            std::string sub_dep = std::get<0>(deps[d]);
                            int sub_n_segment   = std::get<1>(deps[d]);

                            // Check sub key
                            #ifndef NDEBUG
                            if (propagator.find(sub_dep) == propagator.end())
                                std::cout << "Could not find sub key '" + sub_dep + "'. " << std::endl;
                            if (!propagator_finished[sub_dep][sub_n_segment])
                                std::cout << "Could not compute '" + key +  "', since '"+ sub_dep + std::to_string(sub_n_segment) + "' is not prepared." << std::endl;
                            #endif

                            double **_propagator_sub_dep = propagator[sub_dep];
                            for(int i=0; i<M; i++)
                                _propagator[0][i] *= _propagator_sub_dep[sub_n_segment][i];
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
                        _propagator[0][i] *= q_mask[i];
                }

                // Advance propagator successively
                for(int n=n_segment_from; n<n_segment_to; n++)
                {
                    #ifndef NDEBUG
                    if (!propagator_finished[key][n])
                        std::cout << "unfinished, key: " + key + ", " + std::to_string(n) << std::endl;
                    if (propagator_finished[key][n+1])
                        std::cout << "already finished: " + key + ", " + std::to_string(n) << std::endl;
                    #endif

                    propagator_solver->advance_propagator(
                            _propagator[n],
                            _propagator[n+1],
                            monomer_type, q_mask);

                    #ifndef NDEBUG
                    propagator_finished[key][n+1] = true;
                    #endif
                }
                // // Display job info
                // #ifndef NDEBUG
                // std::cout << job << " finished" << std::endl;
                // #endif
            }
        }

        // for(const auto& block: phi_block)
        // {
        //     int p                = std::get<0>(block.first);
        //     std::string key_left    = std::get<1>(block.first);
        //     std::string key_right    = std::get<2>(block.first);
        //     int n_segment        = std::get<3>(block.first);

        //     // Check keys
        //     if (propagator.find(key_left) == propagator.end())
        //         throw_with_line_number("Could not find key_left key'" + key_left + "'. ");
        //     if (propagator.find(key_right) == propagator.end())
        //         throw_with_line_number("Could not find key_right key'" + key_right + "'. ");

        //     for(int i=0; i<=n_segment; i++)
        //     {
        //         if (!propagator_finished[key_left][i])
        //             throw_with_line_number("unfinished, key_left, n'" + key_left + ", " + std::to_string(i) + "'. ");
        //     }

        //     for(int i=0; i<=n_segment; i++)
        //     {
        //         if (!propagator_finished[key_right][i])
        //             throw_with_line_number("unfinished, key_right, n'" + key_right + ", " + std::to_string(i) + "'. ");
        //     }
        // }

        // Compute total partition function of each distinct polymers
        for(const auto& segment_info: single_partition_segment)
        {
            int p                    = std::get<0>(segment_info);
            double *propagator_left  = std::get<1>(segment_info);
            double *propagator_right = std::get<2>(segment_info);
            int n_aggregated         = std::get<3>(segment_info);

            single_polymer_partitions[p]= cb->inner_product(
                propagator_left, propagator_right)/n_aggregated/cb->get_volume();
        }

    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

void CpuComputationContinuous::compute_concentrations()
{
    try
    {
        const int M = cb->get_total_grid();

        // Calculate segment concentrations
        #pragma omp parallel for num_threads(n_streams)
        for(size_t b=0; b<phi_block.size();b++)
        {
            auto block = phi_block.begin();
            advance(block, b);
            const auto& key = block->first;

            int p                 = std::get<0>(key);
            std::string key_left  = std::get<1>(key);
            std::string key_right = std::get<2>(key);

            int n_segment_right = propagator_computation_optimizer->get_computation_block(key).n_segment_right;
            int n_segment_left  = propagator_computation_optimizer->get_computation_block(key).n_segment_left;
            int n_repeated = propagator_computation_optimizer->get_computation_block(key).n_repeated;

            // If there is no segment
            if(n_segment_right == 0)
            {
                for(int i=0; i<M;i++)
                    block->second[i] = 0.0;
                continue;
            }

            // Check keys
            #ifndef NDEBUG
            if (propagator.find(key_left) == propagator.end())
                std::cout << "Could not find key_left key'" + key_left + "'. " << std::endl;
            if (propagator.find(key_right) == propagator.end())
                std::cout << "Could not find key_right key'" + key_right + "'. " << std::endl;
            #endif

            // Calculate phi of one block (possibly multiple blocks when using aggregation)
            calculate_phi_one_block(
                block->second,          // phi
                propagator[key_left],   // dependency v
                propagator[key_right],  // dependency u
                n_segment_right,
                n_segment_left);

            // Normalize concentration
            Polymer& pc = molecules->get_polymer(p);
            double norm = molecules->get_ds()*pc.get_volume_fraction()/pc.get_alpha()/single_polymer_partitions[p]*n_repeated;
            for(int i=0; i<M; i++)
                block->second[i] *= norm;
        }

        // Calculate partition functions and concentrations of solvents
        for(int s=0; s<molecules->get_n_solvent_types(); s++)
        {
            double volume_fraction = std::get<0>(molecules->get_solvent(s));
            std::string monomer_type = std::get<1>(molecules->get_solvent(s));
            
            double *_phi = phi_solvent[s];
            double *_exp_dw = propagator_solver->exp_dw[monomer_type];

            single_solvent_partitions[s] = cb->inner_product(_exp_dw, _exp_dw)/cb->get_volume();
            for(int i=0; i<M; i++)
                _phi[i] = _exp_dw[i]*_exp_dw[i]*volume_fraction/single_solvent_partitions[s];
        }

    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CpuComputationContinuous::calculate_phi_one_block(
    double *phi, double **q_1, double **q_2, const int N_RIGHT, const int N_LEFT)
{
    try
    {
        const int M = cb->get_total_grid();
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
double CpuComputationContinuous::get_total_partition(int polymer)
{
    try
    {
        return single_polymer_partitions[polymer];
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CpuComputationContinuous::get_total_concentration(std::string monomer_type, double *phi)
{
    try
    {
        const int M = cb->get_total_grid();
        // Initialize array
        for(int i=0; i<M; i++)
            phi[i] = 0.0;

        // For each block
        for(const auto& block: phi_block)
        {
            std::string key_left = std::get<1>(block.first);
            int n_segment_right = propagator_computation_optimizer->get_computation_block(block.first).n_segment_right;
            if (PropagatorCode::get_monomer_type_from_key(key_left) == monomer_type && n_segment_right != 0)
            {
                for(int i=0; i<M; i++)
                    phi[i] += block.second[i]; 
            }
        }

        // For each solvent
        for(int s=0;s<molecules->get_n_solvent_types();s++)
        {
            if (std::get<1>(molecules->get_solvent(s)) == monomer_type)
            {
                double *phi_solvent_ = phi_solvent[s];
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
void CpuComputationContinuous::get_total_concentration(int p, std::string monomer_type, double *phi)
{
    try
    {
        const int M = cb->get_total_grid();
        const int P = molecules->get_n_polymer_types();

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
            int n_segment_right = propagator_computation_optimizer->get_computation_block(block.first).n_segment_right;
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
void CpuComputationContinuous::get_total_concentration_gce(double fugacity, int p, std::string monomer_type, double *phi)
{
    try
    {
        const int M = cb->get_total_grid();
        const int P = molecules->get_n_polymer_types();

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
            int n_segment_right = propagator_computation_optimizer->get_computation_block(block.first).n_segment_right;
            if (polymer_idx == p && PropagatorCode::get_monomer_type_from_key(key_left) == monomer_type && n_segment_right != 0)
            {
                Polymer& pc = molecules->get_polymer(p);
                double norm = fugacity/pc.get_volume_fraction()*pc.get_alpha()*single_polymer_partitions[p];
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
void CpuComputationContinuous::get_block_concentration(int p, double *phi)
{
    try
    {
        const int M = cb->get_total_grid();
        const int P = molecules->get_n_polymer_types();

        if (p < 0 || p > P-1)
            throw_with_line_number("Index (" + std::to_string(p) + ") must be in range [0, " + std::to_string(P-1) + "]");

        if (propagator_computation_optimizer->is_aggregated())
            throw_with_line_number("Disable 'aggregation' option to invoke 'get_block_concentration'.");

        Polymer& pc = molecules->get_polymer(p);
        std::vector<Block>& blocks = pc.get_blocks();

        for(size_t b=0; b<blocks.size(); b++)
        {
            std::string key_left  = pc.get_propagator_key(blocks[b].v, blocks[b].u);
            std::string key_right = pc.get_propagator_key(blocks[b].u, blocks[b].v);
            if (key_left < key_right)
                key_left.swap(key_right);

            double* _essential_phi_block = phi_block[std::make_tuple(p, key_left, key_right)];
            for(int i=0; i<M; i++)
                phi[i+b*M] = _essential_phi_block[i]; 
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
double CpuComputationContinuous::get_solvent_partition(int s)
{
    try
    {
        return single_solvent_partitions[s];
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CpuComputationContinuous::get_solvent_concentration(int s, double *phi)
{
    try
    {
        const int M = cb->get_total_grid();
        const int S = molecules->get_n_solvent_types();

        if (s < 0 || s > S-1)
            throw_with_line_number("Index (" + std::to_string(s) + ") must be in range [0, " + std::to_string(S-1) + "]");

        double *phi_solvent_ = phi_solvent[s];
        for(int i=0; i<M; i++)
            phi[i] = phi_solvent_[i];
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CpuComputationContinuous::compute_stress()
{
    // This method should be invoked after invoking compute_statistics().

    // To calculate stress, we multiply weighted fourier basis to q(k)*q^dagger(-k).
    // We only need the real part of stress calculation.

    try
    {
        const int DIM  = cb->get_dim();
        const int M    = cb->get_total_grid();

        std::map<std::tuple<int, std::string, std::string>, std::array<double,3>> block_dq_dl;

        // Reset stress map
        for(const auto& item: phi_block)
        {
            for(int d=0; d<3; d++)
                block_dq_dl[item.first][d] = 0.0;
        }

        // Compute stress for each block
        #pragma omp parallel for num_threads(n_streams)
        for(size_t b=0; b<phi_block.size();b++)
        {
            auto block = phi_block.begin();
            advance(block, b);
            const auto& key   = block->first;

            std::string key_left  = std::get<1>(key);
            std::string key_right = std::get<2>(key);

            const int N_RIGHT = propagator_computation_optimizer->get_computation_block(key).n_segment_right;
            const int N_LEFT  = propagator_computation_optimizer->get_computation_block(key).n_segment_left;
            std::string monomer_type = propagator_computation_optimizer->get_computation_block(key).monomer_type;
            int n_repeated = propagator_computation_optimizer->get_computation_block(key).n_repeated;

            // If there is no segment
            if(N_RIGHT == 0)
                continue;

            double **q_1 = propagator[key_left];     // dependency v
            double **q_2 = propagator[key_right];    // dependency u

            std::vector<double> s_coeff = SimpsonRule::get_coeff(N_RIGHT);
            std::array<double,3> _block_dq_dl = block_dq_dl[key];

            // Compute
            for(int n=0; n<=N_RIGHT; n++)
            {
                std::vector<double> segment_stress = propagator_solver->compute_single_segment_stress(
                    q_1[N_LEFT-n], q_2[n], monomer_type, false);
                for(int d=0; d<DIM; d++)
                    _block_dq_dl[d] += segment_stress[d]*s_coeff[n]*n_repeated;
            }
            block_dq_dl[key] = _block_dq_dl;
        }

        // Compute total stress
        int n_polymer_types = molecules->get_n_polymer_types();
        for(int p=0; p<n_polymer_types; p++)
            for(int d=0; d<DIM; d++)
                dq_dl[p][d] = 0.0;
        for(const auto& block: phi_block)
        {
            const auto& key       = block.first;
            int p                 = std::get<0>(key);
            std::string key_left  = std::get<1>(key);
            std::string key_right = std::get<2>(key);
            Polymer& pc = molecules->get_polymer(p);

            for(int d=0; d<DIM; d++)
                dq_dl[p][d] += block_dq_dl[key][d];
        }
        for(int p=0; p<n_polymer_types; p++){
            for(int d=0; d<DIM; d++)
                dq_dl[p][d] /= -3.0*cb->get_lx(d)*M*M/molecules->get_ds();
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CpuComputationContinuous::get_chain_propagator(double *q_out, int polymer, int v, int u, int n)
{
    // This method should be invoked after invoking compute_statistics()

    // Get chain propagator for a selected polymer, block and direction.
    // This is made for debugging and testing.
    try
    {
        const int M = cb->get_total_grid();
        Polymer& pc = molecules->get_polymer(polymer);
        std::string dep = pc.get_propagator_key(v,u);

        if (propagator_computation_optimizer->get_computation_propagators().find(dep) == propagator_computation_optimizer->get_computation_propagators().end())
            throw_with_line_number("Could not find the propagator code '" + dep + "'. Disable 'aggregation' option to obtain propagator_computation_optimizer.");

        const int N_RIGHT = propagator_computation_optimizer->get_computation_propagator(dep).max_n_segment;
        if (n < 0 || n > N_RIGHT)
            throw_with_line_number("n (" + std::to_string(n) + ") must be in range [0, " + std::to_string(N_RIGHT) + "]");

        double **_partition = propagator[dep];
        for(int i=0; i<M; i++)
            q_out[i] = _partition[n][i];
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
bool CpuComputationContinuous::check_total_partition()
{
    // const int M = cb->get_total_grid();
    int n_polymer_types = molecules->get_n_polymer_types();
    std::vector<std::vector<double>> total_partitions;
    for(int p=0;p<n_polymer_types;p++)
    {
        std::vector<double> total_partitions_p;
        total_partitions.push_back(total_partitions_p);
    }

    for(const auto& block: phi_block)
    {
        const auto& key = block.first;
        int p                 = std::get<0>(key);
        std::string key_left  = std::get<1>(key);
        std::string key_right = std::get<2>(key);

        int n_segment_right = propagator_computation_optimizer->get_computation_block(key).n_segment_right;
        int n_segment_left  = propagator_computation_optimizer->get_computation_block(key).n_segment_left;
        int n_repeated      = propagator_computation_optimizer->get_computation_block(key).n_repeated;
        int n_propagators   = propagator_computation_optimizer->get_computation_block(key).v_u.size();

        #ifndef NDEBUG
        std::cout<< p << ", " << key_left << ", " << key_right << ": " << n_segment_left << ", " << n_segment_right << ", " << n_propagators << ", " << propagator_computation_optimizer->get_computation_block(key).n_repeated << std::endl;
        #endif

        for(int n=0;n<=n_segment_right;n++)
        {
            double total_partition = cb->inner_product(
                propagator[key_left][n_segment_left-n],
                propagator[key_right][n])*n_repeated/cb->get_volume();

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
            if (total_partitions[p][n] > max_partition)
                max_partition = total_partitions[p][n];
            if (total_partitions[p][n] < min_partition)
                min_partition = total_partitions[p][n];
        }
        double diff_partition = std::abs(max_partition - min_partition);

        std::cout<< "\t" << p << ": " << max_partition << ", " << min_partition << ", " << diff_partition << std::endl;
        if (diff_partition > 1e-7)
            return false;
    }
    return true;
}