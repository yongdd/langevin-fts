#include <cmath>

#include "CpuComputationContinuous.h"
#include "CpuSolverPseudo.h"
#include "CpuSolverReal.h"
#include "SimpsonRule.h"

CpuComputationContinuous::CpuComputationContinuous(
    ComputationBox *cb,
    Molecules *molecules,
    PropagatorAnalyzer *propagator_analyzer,
    std::string method)
    : PropagatorComputation(cb, molecules, propagator_analyzer)
{
    try
    {
        const int M = cb->get_n_grid();
        if(method == "pseudospectral")
            this->propagator_solver = new CpuSolverPseudo(cb, molecules);
        else if(method == "realspace")
            this->propagator_solver = new CpuSolverReal(cb, molecules);

        // Allocate memory for propagators
        if( propagator_analyzer->get_computation_propagator_codes().size() == 0)
            throw_with_line_number("There is no propagator code. Add polymers first.");
        for(const auto& item: propagator_analyzer->get_computation_propagator_codes())
        {
            std::string dep = item.first;
            int max_n_segment = item.second.max_n_segment;
            propagator[dep] = new double[M*(max_n_segment+1)];

            #ifndef NDEBUG
            propagator_finished[dep] = new bool[max_n_segment+1];
            for(int i=0; i<=max_n_segment;i++)
                propagator_finished[dep][i] = false;
            #endif
        }

        // Allocate memory for concentrations
        if( propagator_analyzer->get_computation_blocks().size() == 0)
            throw_with_line_number("There is no block. Add polymers first.");
        for(const auto& item: propagator_analyzer->get_computation_blocks())
        {
            phi_block[item.first] = new double[M];
        }

        // Total partition functions for each polymer
        single_polymer_partitions = new double[molecules->get_n_polymer_types()];

        // Remember one segment for each polymer chain to compute total partition function
        int current_p = 0;
        for(const auto& block: phi_block)
        {
            const auto& key = block.first;
            int p                = std::get<0>(key);
            std::string dep_v    = std::get<1>(key);
            std::string dep_u    = std::get<2>(key);

            // Skip if already found one segment
            if (p != current_p)
                continue;

            int n_aggregated = propagator_analyzer->get_computation_block(key).v_u.size()/
                               propagator_analyzer->get_computation_block(key).n_repeated;
            int n_segment_offset = propagator_analyzer->get_computation_block(key).n_segment_offset;

            single_partition_segment.push_back(std::make_tuple(
                p,
                &propagator[dep_v][(n_segment_offset)*M],   // q
                &propagator[dep_u][0],                      // q_dagger
                n_aggregated                                // how many propagators are aggregated
                ));
            current_p++;
        }

        // Total partition functions for each solvent
        single_solvent_partitions = new double[molecules->get_n_solvent_types()];

        // Concentrations for each solvent
        for(int s=0;s<molecules->get_n_solvent_types();s++)
            phi_solvent.push_back(new double[M]);

        // Create scheduler for computation of propagator
        sc = new Scheduler(propagator_analyzer->get_computation_propagator_codes(), N_SCHEDULER_STREAMS); 

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

    delete[] single_polymer_partitions;
    delete[] single_solvent_partitions;

    for(const auto& item: propagator)
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
    try
    {
        const int M = cb->get_n_grid();

        for(const auto& item: propagator_analyzer->get_computation_propagator_codes())
        {
            std::string monomer_type = item.second.monomer_type.substr(0, 1);
            if( w_input.find(monomer_type) == w_input.end())
                throw_with_line_number("monomer_type \"" + monomer_type + "\" is not in w_input.");
        }

        // If( q_init.size() > 0)
        //     throw_with_line_number("Currently, \'q_init\' is not supported.");

        // Update dw or exp_dw
        propagator_solver->update_dw(w_input);

        // Assign a pointer for mask
        const double *q_mask = cb->get_mask();

        auto& branch_schedule = sc->get_schedule();
        // // display all jobs
        // For (auto parallel_job = branch_schedule.begin(); parallel_job != branch_schedule.end(); parallel_job++)
        // {
        //     std::cout << "jobs:" << std::endl;
        //     for(int job=0; job<parallel_job->size(); job++)
        //     {
        //         auto& key = std::get<0>((*parallel_job)[job]);
        //         int n_segment_from = std::get<1>((*parallel_job)[job]);
        //         int n_segment_to = std::get<2>((*parallel_job)[job]);
        //         std::cout << "key, n_segment_from, n_segment_to: " + key + ", " + std::to_string(n_segment_from) + ", " + std::to_string(n_segment_to) + ". " << std::endl;
        //     }
        // }

        // For each time span
        for (auto parallel_job = branch_schedule.begin(); parallel_job != branch_schedule.end(); parallel_job++)
        {
            // For each propagator
            #pragma omp parallel for
            for(size_t job=0; job<parallel_job->size(); job++)
            {
                auto& key = std::get<0>((*parallel_job)[job]);
                int n_segment_from = std::get<1>((*parallel_job)[job]);
                int n_segment_to = std::get<2>((*parallel_job)[job]);
                auto& deps = propagator_analyzer->get_computation_propagator_code(key).deps;
                auto monomer_type = propagator_analyzer->get_computation_propagator_code(key).monomer_type.substr(0, 1);

                // Check key
                #ifndef NDEBUG
                if (propagator.find(key) == propagator.end())
                    std::cout << "Could not find key '" + key + "'. " << std::endl;
                #endif

                double *_propagator = propagator[key];

                // If it is leaf node
                if(n_segment_from == 1 && deps.size() == 0) 
                {
                     // q_init
                    if (key[0] == '{')
                    {
                        std::string g = PropagatorCode::get_q_input_idx_from_key(key);
                        if (q_init.find(g) == q_init.end())
                            std::cout << "Could not find q_init[\"" + g + "\"]." << std::endl;
                        for(int i=0; i<M; i++)
                            _propagator[i] = q_init[g][i];
                    }
                    else
                    {
                        for(int i=0; i<M; i++)
                            _propagator[i] = 1.0;
                    }

                    #ifndef NDEBUG
                    propagator_finished[key][0] = true;
                    #endif
                }
                // If it is not leaf node
                else if (n_segment_from == 1 && deps.size() > 0) 
                {
                    // If it is aggregated
                    if (key[0] == '[')
                    {
                        for(int i=0; i<M; i++)
                            _propagator[i] = 0.0;
                        
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

                            double *_propagator_sub_dep = propagator[sub_dep];
                            for(int i=0; i<M; i++)
                                _propagator[i] += _propagator_sub_dep[sub_n_segment*M+i]*sub_n_repeated;
                        }
                        #ifndef NDEBUG
                        propagator_finished[key][0] = true;
                        #endif
                        // std::cout << "finished, key, n: " + key + ", 0" << std::endl;
                    }
                    else
                    {
                        for(int i=0; i<M; i++)
                            _propagator[i] = 1.0;
                        
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

                            double *_propagator_sub_dep = propagator[sub_dep];
                            for(int i=0; i<M; i++)
                                _propagator[i] *= _propagator_sub_dep[sub_n_segment*M+i];
                        }

                        #ifndef NDEBUG
                        propagator_finished[key][0] = true;
                        #endif
                        // std::cout << "finished, key, n: " + key + ", 0" << std::endl;
                    }
                }
        
                // Multiply mask
                if (n_segment_from == 1 && q_mask != nullptr)
                {
                    for(int i=0; i<M; i++)
                        _propagator[i] *= q_mask[i];
                }

                // Advance propagator successively
                for(int n=n_segment_from; n<=n_segment_to; n++)
                {
                    #ifndef NDEBUG
                    if (!propagator_finished[key][n-1])
                        std::cout << "unfinished, key: " + key + ", " + std::to_string(n-1) << std::endl;
                    #endif
                    
                    propagator_solver->advance_propagator_continuous(
                            &_propagator[(n-1)*M],
                            &_propagator[n*M],
                            monomer_type, q_mask);

                    #ifndef NDEBUG
                    propagator_finished[key][n] = true;
                    #endif
                    // std::cout << "finished, key, n: " + key + ", " << std::to_string(n) << std::endl;
                }
            }
        }

        // for(const auto& block: phi_block)
        // {
        //     int p                = std::get<0>(block.first);
        //     std::string dep_v    = std::get<1>(block.first);
        //     std::string dep_u    = std::get<2>(block.first);
        //     int n_segment        = std::get<3>(block.first);

        //     // Check keys
        //     if (propagator.find(dep_v) == propagator.end())
        //         throw_with_line_number("Could not find dep_v key'" + dep_v + "'. ");
        //     if (propagator.find(dep_u) == propagator.end())
        //         throw_with_line_number("Could not find dep_u key'" + dep_u + "'. ");

        //     for(int i=0; i<=n_segment; i++)
        //     {
        //         if (!propagator_finished[dep_v][i])
        //             throw_with_line_number("unfinished, dep_v, n'" + dep_v + ", " + std::to_string(i) + "'. ");
        //     }

        //     for(int i=0; i<=n_segment; i++)
        //     {
        //         if (!propagator_finished[dep_u][i])
        //             throw_with_line_number("unfinished, dep_u, n'" + dep_u + ", " + std::to_string(i) + "'. ");
        //     }
        // }

        // Compute total partition function of each distinct polymers
        for(const auto& segment_info: single_partition_segment)
        {
            int p                = std::get<0>(segment_info);
            double *propagator_v = std::get<1>(segment_info);
            double *propagator_u = std::get<2>(segment_info);
            int n_aggregated     = std::get<3>(segment_info);

            single_polymer_partitions[p]= cb->inner_product(
                propagator_v, propagator_u)/n_aggregated/cb->get_volume();
        }

        // Calculate segment concentrations
        #pragma omp parallel for
        for(size_t b=0; b<phi_block.size();b++)
        {
            auto block = phi_block.begin();
            advance(block, b);
            const auto& key = block->first;

            int p                = std::get<0>(key);
            std::string dep_v    = std::get<1>(key);
            std::string dep_u    = std::get<2>(key);

            int n_segment_compute = propagator_analyzer->get_computation_block(key).n_segment_compute;
            int n_segment_offset  = propagator_analyzer->get_computation_block(key).n_segment_offset;
            int n_repeated = propagator_analyzer->get_computation_block(key).n_repeated;

            // If there is no segment
            if(n_segment_compute == 0)
            {
                for(int i=0; i<M;i++)
                    block->second[i] = 0.0;
                continue;
            }

            // Check keys
            #ifndef NDEBUG
            if (propagator.find(dep_v) == propagator.end())
                std::cout << "Could not find dep_v key'" + dep_v + "'. " << std::endl;
            if (propagator.find(dep_u) == propagator.end())
                std::cout << "Could not find dep_u key'" + dep_u + "'. " << std::endl;
            #endif

            // Calculate phi of one block (possibly multiple blocks when using aggregation)
            calculate_phi_one_block(
                block->second,             // phi
                propagator[dep_v],  // dependency v
                propagator[dep_u],  // dependency u
                n_segment_compute,
                n_segment_offset);

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
            std::string monomer_type = std::get<1>(molecules->get_solvent(s)).substr(0, 1);
            
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
    double *phi, double *q_1, double *q_2, const int N, const int N_OFFSET)
{
    try
    {
        const int M = cb->get_n_grid();
        std::vector<double> simpson_rule_coeff = SimpsonRule::get_coeff(N);

        // Compute segment concentration
        for(int i=0; i<M; i++)
            phi[i] = simpson_rule_coeff[0]*q_1[i+(N_OFFSET)*M]*q_2[i];
        for(int n=1; n<=N; n++)
        {
            for(int i=0; i<M; i++)
                phi[i] += simpson_rule_coeff[n]*q_1[i+(N_OFFSET-n)*M]*q_2[i+n*M];
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
        const int M = cb->get_n_grid();
        // Initialize array
        for(int i=0; i<M; i++)
            phi[i] = 0.0;

        // For each block
        for(const auto& block: phi_block)
        {
            std::string dep_v = std::get<1>(block.first);
            int n_segment_compute = propagator_analyzer->get_computation_block(block.first).n_segment_compute;
            if (PropagatorCode::get_monomer_type_from_key(dep_v).substr(0,1) == monomer_type && n_segment_compute != 0)
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
        const int M = cb->get_n_grid();
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
            std::string dep_v = std::get<1>(block.first);
            int n_segment_compute = propagator_analyzer->get_computation_block(block.first).n_segment_compute;
            if (polymer_idx == p && PropagatorCode::get_monomer_type_from_key(dep_v).substr(0,1) == monomer_type && n_segment_compute != 0)
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
void CpuComputationContinuous::get_block_concentration(int p, double *phi)
{
    try
    {
        const int M = cb->get_n_grid();
        const int P = molecules->get_n_polymer_types();

        if (p < 0 || p > P-1)
            throw_with_line_number("Index (" + std::to_string(p) + ") must be in range [0, " + std::to_string(P-1) + "]");

        if (propagator_analyzer->is_aggregated())
            throw_with_line_number("Disable 'aggregation' option to invoke 'get_block_concentration'.");

        Polymer& pc = molecules->get_polymer(p);
        std::vector<Block>& blocks = pc.get_blocks();

        for(size_t b=0; b<blocks.size(); b++)
        {
            std::string dep_v = pc.get_propagator_key(blocks[b].v, blocks[b].u);
            std::string dep_u = pc.get_propagator_key(blocks[b].u, blocks[b].v);
            if (dep_v < dep_u)
                dep_v.swap(dep_u);

            double* _essential_phi_block = phi_block[std::make_tuple(p, dep_v, dep_u)];
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
        const int M = cb->get_n_grid();
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
std::vector<double> CpuComputationContinuous::compute_stress()
{
    // This method should be invoked after invoking compute_statistics().

    // To calculate stress, we multiply weighted fourier basis to q(k)*q^dagger(-k).
    // We only need the real part of stress calculation.

    try
    {
        const int DIM  = cb->get_dim();
        const int M    = cb->get_n_grid();

        std::vector<double> stress(DIM);
        std::map<std::tuple<int, std::string, std::string>, std::array<double,3>> block_dq_dl;

        // Reset stress map
        for(const auto& item: phi_block)
        {
            for(int d=0; d<3; d++)
                block_dq_dl[item.first][d] = 0.0;
        }

        // Compute stress for each block
        #pragma omp parallel for
        for(size_t b=0; b<phi_block.size();b++)
        {
            auto block = phi_block.begin();
            advance(block, b);
            const auto& key   = block->first;

            std::string dep_v = std::get<1>(key);
            std::string dep_u = std::get<2>(key);

            const int N        = propagator_analyzer->get_computation_block(key).n_segment_compute;
            const int N_OFFSET = propagator_analyzer->get_computation_block(key).n_segment_offset;
            std::string monomer_type = propagator_analyzer->get_computation_block(key).monomer_type.substr(0,1);
            int n_repeated = propagator_analyzer->get_computation_block(key).n_repeated;

            // If there is no segment
            if(N == 0)
                continue;

            double *q_1 = propagator[dep_v];    // dependency v
            double *q_2 = propagator[dep_u];    // dependency u

            std::vector<double> s_coeff = SimpsonRule::get_coeff(N);
            std::array<double,3> _block_dq_dl = block_dq_dl[key];

            // Compute
            for(int n=0; n<=N; n++)
            {
                std::vector<double> segment_stress = propagator_solver->compute_single_segment_stress_continuous(
                    &q_1[(N_OFFSET-n)*M], &q_2[n*M], monomer_type);
                for(int d=0; d<DIM; d++)
                    _block_dq_dl[d] += segment_stress[d]*s_coeff[n]*n_repeated;
            }
            block_dq_dl[key] = _block_dq_dl;
        }

        // Compute total stress
        for(int d=0; d<DIM; d++)
            stress[d] = 0.0;
        for(const auto& block: phi_block)
        {
            const auto& key   = block.first;
            int p             = std::get<0>(key);
            std::string dep_v = std::get<1>(key);
            std::string dep_u = std::get<2>(key);
            Polymer& pc  = molecules->get_polymer(p);

            for(int d=0; d<DIM; d++)
                stress[d] += block_dq_dl[key][d]*pc.get_volume_fraction()/pc.get_alpha()/single_polymer_partitions[p];
        }

        for(int d=0; d<DIM; d++)
            stress[d] /= -3.0*cb->get_lx(d)*M*M/molecules->get_ds();

        return stress;
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
        const int M = cb->get_n_grid();
        Polymer& pc = molecules->get_polymer(polymer);
        std::string dep = pc.get_propagator_key(v,u);

        if (propagator_analyzer->get_computation_propagator_codes().find(dep) == propagator_analyzer->get_computation_propagator_codes().end())
            throw_with_line_number("Could not find the propagator code '" + dep + "'. Disable 'aggregation' option to obtain propagator_analyzer.");

        const int N = propagator_analyzer->get_computation_propagator_codes()[dep].max_n_segment;
        if (n < 0 || n > N)
            throw_with_line_number("n (" + std::to_string(n) + ") must be in range [0, " + std::to_string(N) + "]");

        double* _partition = propagator[dep];
        for(int i=0; i<M; i++)
            q_out[i] = _partition[n*M+i];
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
bool CpuComputationContinuous::check_total_partition()
{
    const int M = cb->get_n_grid();
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
        int p                = std::get<0>(key);
        std::string dep_v    = std::get<1>(key);
        std::string dep_u    = std::get<2>(key);

        int n_segment_compute = propagator_analyzer->get_computation_block(key).n_segment_compute;
        int n_segment_offset  = propagator_analyzer->get_computation_block(key).n_segment_offset;
        int n_repeated        = propagator_analyzer->get_computation_block(key).n_repeated;
        int n_propagators     = propagator_analyzer->get_computation_block(key).v_u.size();

        // std::cout<< p << ", " << dep_v << ", " << dep_u << ": " << n_segment_offset << ", " << n_segment_compute << ", " << n_propagators << ", " << propagator_analyzer->get_computation_block(key).n_repeated << std::endl;

        for(int n=0;n<=n_segment_compute;n++)
        {
            double total_partition = cb->inner_product(
                &propagator[dep_v][(n_segment_offset-n)*M],
                &propagator[dep_u][n*M])*n_repeated/cb->get_volume();

            total_partition /= n_propagators;

            // std::cout<< p << ", " << n << ": " << total_partition << std::endl;
            total_partitions[p].push_back(total_partition);
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