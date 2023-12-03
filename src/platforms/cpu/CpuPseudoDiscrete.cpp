#include <cmath>
#include "CpuPseudoDiscrete.h"
#include "SimpsonRule.h"

CpuPseudoDiscrete::CpuPseudoDiscrete(
    ComputationBox *cb,
    Molecules *molecules,
    PropagatorsAnalyzer *propagators_analyzer,
    FFT *fft)
    : Solver(cb, molecules, propagators_analyzer)
{
    try
    {
        const int M = cb->get_n_grid();
        const int M_COMPLEX = this->n_complex_grid;
        this->propagators_analyzer = propagators_analyzer;
        this->fft = fft;

        // Allocate memory for propagators
        if( propagators_analyzer->get_essential_propagator_codes().size() == 0)
            throw_with_line_number("There is no propagator code. Add polymers first.");
        for(const auto& item: propagators_analyzer->get_essential_propagator_codes())
        {
            std::string dep = item.first;
            int max_n_segment = item.second.max_n_segment;
            propagator[dep] = new double[M*max_n_segment];

            #ifndef NDEBUG
            propagator_finished[dep] = new bool[max_n_segment];
            for(int i=0; i<max_n_segment;i++)
                propagator_finished[dep][i] = false;
            #endif

             // There are N segments
             // Example (N==5)
             // O--O--O--O--O
             // 0  1  2  3  4 

             // Legend)
             // -- : full bond
             // O  : full segment
        }

        // Allocate memory for propagator_junction, which contain partition function at junction of discrete chain
        for(const auto& item: propagators_analyzer->get_essential_propagator_codes())
        {
            propagator_junction[item.first] = new double[M];
        }

        // Allocate memory for concentrations
        if( propagators_analyzer->get_essential_blocks().size() == 0)
            throw_with_line_number("There is no block. Add polymers first.");
        for(const auto& item: propagators_analyzer->get_essential_blocks())
        {
            block_phi[item.first] = new double[M];
        }

        // Create boltz_bond, boltz_bond_half, and exp_dw
        for(const auto& item: molecules->get_bond_lengths())
        {
            std::string monomer_type = item.first;
            boltz_bond     [monomer_type] = new double[M_COMPLEX];
            boltz_bond_half[monomer_type] = new double[M_COMPLEX]; 
            exp_dw         [monomer_type] = new double[M];
        }

        // Allocate memory for stress calculation: compute_stress()
        fourier_basis_x = new double[M_COMPLEX];
        fourier_basis_y = new double[M_COMPLEX];
        fourier_basis_z = new double[M_COMPLEX];

        // Total partition functions for each polymer
        single_partitions = new double[molecules->get_n_polymer_types()];

        // Remember one segment for each polymer chain to compute total partition function
        int current_p = 0;
        for(const auto& block: block_phi)
        {
            const auto& key = block.first;
            int p                = std::get<0>(key);
            std::string dep_v    = std::get<1>(key);
            std::string dep_u    = std::get<2>(key);

            // Skip if already found one segment
            if (p != current_p)
                continue;

            int n_aggregated;
            int n_segment_offset    = propagators_analyzer->get_essential_block(key).n_segment_offset;
            int n_segment_original  = propagators_analyzer->get_essential_block(key).n_segment_original;
            std::string monomer_type = propagators_analyzer->get_essential_block(key).monomer_type;

            // Contains no '['
            if (dep_u.find('[') == std::string::npos)
                n_aggregated = 1;
            else
                n_aggregated = propagators_analyzer->get_essential_block(key).v_u.size();

            single_partition_segment.push_back(std::make_tuple(
                p,
                &propagator[dep_v][(n_segment_original-n_segment_offset-1)*M],  // q
                &propagator[dep_u][0],                                        // q_dagger
                monomer_type,       
                n_aggregated                   // how many propagators are aggregated
                ));
            current_p++;
        }

        // Create scheduler for computation of propagator
        sc = new Scheduler(propagators_analyzer->get_essential_propagator_codes(), N_SCHEDULER_STREAMS); 

        update_bond_function();
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
CpuPseudoDiscrete::~CpuPseudoDiscrete()
{
    delete fft;
    delete sc;

    delete[] fourier_basis_x;
    delete[] fourier_basis_y;
    delete[] fourier_basis_z;
        
    delete[] single_partitions;

    for(const auto& item: boltz_bond)
        delete[] item.second;
    for(const auto& item: boltz_bond_half)
        delete[] item.second;
    for(const auto& item: exp_dw)
        delete[] item.second;
    for(const auto& item: propagator)
        delete[] item.second;
    for(const auto& item: block_phi)
        delete[] item.second;
    for(const auto& item: propagator_junction)
        delete[] item.second;

    #ifndef NDEBUG
    for(const auto& item: propagator_finished)
        delete[] item.second;
    #endif
}
void CpuPseudoDiscrete::update_bond_function()
{
    try
    {
        for(const auto& item: molecules->get_bond_lengths())
        {
            std::string monomer_type = item.first;
            double bond_length_sq = item.second*item.second;
            get_boltz_bond(boltz_bond     [monomer_type], bond_length_sq,   cb->get_nx(), cb->get_dx(), molecules->get_ds());
            get_boltz_bond(boltz_bond_half[monomer_type], bond_length_sq/2, cb->get_nx(), cb->get_dx(), molecules->get_ds());

            // For stress calculation: compute_stress()
            get_weighted_fourier_basis(fourier_basis_x, fourier_basis_y, fourier_basis_z, cb->get_nx(), cb->get_dx());
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CpuPseudoDiscrete::compute_statistics(
    std::map<std::string, const double*> w_input,
    std::map<std::string, const double*> q_init,
    double* q_mask)
{
    try
    {
        const int M = cb->get_n_grid();
        const double ds = molecules->get_ds();

        for(const auto& item: propagators_analyzer->get_essential_propagator_codes())
        {
            if( w_input.find(item.second.monomer_type) == w_input.end())
                throw_with_line_number("monomer_type \"" + item.second.monomer_type + "\" is not in w_input.");
        }

        for(const auto& item: w_input)
        {
            if( exp_dw.find(item.first) == exp_dw.end())
                throw_with_line_number("monomer_type \"" + item.first + "\" is not in exp_dw.");     
        }

        for(const auto& item: w_input)
        {
            std::string monomer_type = item.first;
            const double *w = item.second;
            for(int i=0; i<M; i++)
                exp_dw[monomer_type][i] = exp(-w[i]*ds);
        }

        // For each time span
        auto& branch_schedule = sc->get_schedule();
        for (auto parallel_job = branch_schedule.begin(); parallel_job != branch_schedule.end(); parallel_job++)
        {
            // For each propagator
            #pragma omp parallel for
            for(size_t job=0; job<parallel_job->size(); job++)
            {
                auto& key = std::get<0>((*parallel_job)[job]);
                int n_segment_from = std::get<1>((*parallel_job)[job]);
                int n_segment_to = std::get<2>((*parallel_job)[job]);
                auto& deps = propagators_analyzer->get_essential_propagator_code(key).deps;
                auto monomer_type = propagators_analyzer->get_essential_propagator_code(key).monomer_type;

                // Check key
                #ifndef NDEBUG
                if (propagator.find(key) == propagator.end())
                    std::cout << "Could not find key '" << key << "'. " << std::endl;
                #endif

                double *_propagator = propagator[key];

                // Calculate one block end
                if (n_segment_from == 1 && deps.size() == 0) // if it is leaf node
                {
                     // q_init
                    if (key[0] == '{')
                    {
                        std::string g = PropagatorCode::get_q_input_idx_from_key(key);
                        if (q_init.find(g) == q_init.end())
                            std::cout << "Could not find q_init[\"" + g + "\"]." << std::endl;
                        for(int i=0; i<M; i++)
                            _propagator[i] = q_init[g][i]*exp_dw[monomer_type][i];
                    }
                    else
                    {
                        for(int i=0; i<M; i++)
                            _propagator[i] = exp_dw[monomer_type][i];
                    }

                    #ifndef NDEBUG
                    propagator_finished[key][0] = true;
                    #endif
                }
                else if (n_segment_from == 1 && deps.size() > 0) // if it is not leaf node
                {
                    // If it is aggregated
                    if (key[0] == '[')
                    {
                        for(int i=0; i<M; i++)
                            _propagator[i] = 0.0;
                        for(size_t d=0; d<deps.size(); d++)
                        {
                            std::string sub_dep = std::get<0>(deps[d]);
                            int sub_n_segment   = std::get<1>(deps[d]);
                            int sub_n_repeated  = std::get<2>(deps[d]);

                            // Check sub key
                            #ifndef NDEBUG
                            if (propagator.find(sub_dep) == propagator.end())
                                std::cout << "Could not find sub key '" + sub_dep + "'. " << std::endl;
                            if (!propagator_finished[sub_dep][sub_n_segment-1])
                                std::cout << "Could not compute '" + key +  "', since '"+ sub_dep + std::to_string(sub_n_segment) + "' is not prepared." << std::endl;
                            #endif

                            double *_propagator_sub_dep = propagator[sub_dep];
                            for(int i=0; i<M; i++)
                                _propagator[i] += _propagator_sub_dep[(sub_n_segment-1)*M+i]*sub_n_repeated;
                        }

                        advance_propagator(&_propagator[0],
                            &_propagator[0],
                            boltz_bond[monomer_type],
                            exp_dw[monomer_type],
                            q_mask);

                        #ifndef NDEBUG
                        propagator_finished[key][0] = true;
                        #endif
                        // std::cout << "finished, key, n: " + key + ", 0" << std::endl;
                    }
                    else
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

                        // Combine branches
                        double q_junction[M];
                        for(int i=0; i<M; i++)
                            q_junction[i] = 1.0;
                        for(size_t d=0; d<deps.size(); d++)
                        {
                            std::string sub_dep = std::get<0>(deps[d]);
                            int sub_n_segment   = std::get<1>(deps[d]);
                            double q_half_step[M];

                            // Check sub key
                            #ifndef NDEBUG
                            if (propagator.find(sub_dep) == propagator.end())
                                std::cout << "Could not find sub key '" + sub_dep + "'. " << std::endl;
                            if (!propagator_finished[sub_dep][sub_n_segment-1])
                                std::cout << "Could not compute '" + key +  "', since '"+ sub_dep + std::to_string(sub_n_segment) + "' is not prepared." << std::endl;
                            #endif

                            advance_propagator_half_bond_step(&propagator[sub_dep][(sub_n_segment-1)*M],
                                q_half_step, boltz_bond_half[propagators_analyzer->get_essential_propagator_code(sub_dep).monomer_type]);

                            for(int i=0; i<M; i++)
                                q_junction[i] *= q_half_step[i];
                        }
                        double *_q_junction_cache = propagator_junction[key];
                        for(int i=0; i<M; i++)
                            _q_junction_cache[i] = q_junction[i];

                        // Add half bond
                        advance_propagator_half_bond_step(q_junction, &_propagator[0], boltz_bond_half[monomer_type]);

                        // Add full segment
                        for(int i=0; i<M; i++)
                            _propagator[i] *= exp_dw[monomer_type][i];
                        
                        #ifndef NDEBUG
                        propagator_finished[key][0] = true;
                        #endif
                    }
                }
                else
                {
                    n_segment_from--;
                }

                // Multiply mask
                if (q_mask != nullptr)
                {
                    for(int i=0; i<M; i++)
                        _propagator[i] *= q_mask[i];
                }

                // Advance propagator successively
                for(int n=n_segment_from; n<n_segment_to; n++)
                {
                    #ifndef NDEBUG
                    if (!propagator_finished[key][n-1])
                        std::cout << "unfinished, key: " + key + ", " + std::to_string(n);
                    #endif

                    advance_propagator(&_propagator[(n-1)*M],
                            &_propagator[n*M],
                            boltz_bond[monomer_type],
                            exp_dw[monomer_type],
                            q_mask);

                    #ifndef NDEBUG
                    propagator_finished[key][n] = true;
                    #endif

                    // std::cout << "finished, key, n: " + key + ", " << std::to_string(n) << std::endl;
                }
            }
        }

        // Compute total partition function of each distinct polymers
        for(const auto& segment_info: single_partition_segment)
        {
            int p                    = std::get<0>(segment_info);
            double *propagator_v     = std::get<1>(segment_info);
            double *propagator_u     = std::get<2>(segment_info);
            std::string monomer_type = std::get<3>(segment_info);
            int n_aggregated         = std::get<4>(segment_info);

            single_partitions[p]= cb->inner_product_inverse_weight(
                propagator_v, propagator_u, exp_dw[monomer_type])/n_aggregated/cb->get_volume();
        }

        // Calculate segment concentrations
        #pragma omp parallel for
        for(size_t b=0; b<block_phi.size();b++)
        {
            auto block = block_phi.begin();
            advance(block, b);
            const auto& key = block->first;

            int p                = std::get<0>(key);
            std::string dep_v    = std::get<1>(key);
            std::string dep_u    = std::get<2>(key);

            int n_repeated;
            int n_segment_allocated = propagators_analyzer->get_essential_block(key).n_segment_allocated;
            int n_segment_offset    = propagators_analyzer->get_essential_block(key).n_segment_offset;
            int n_segment_original  = propagators_analyzer->get_essential_block(key).n_segment_original;
            std::string monomer_type = propagators_analyzer->get_essential_block(key).monomer_type;

            // Contains no '['
            if (dep_u.find('[') == std::string::npos)
                n_repeated = propagators_analyzer->get_essential_block(key).v_u.size();
            else
                n_repeated = 1;

            // Check keys
            #ifndef NDEBUG
            if (propagator.find(dep_v) == propagator.end())
                std::cout << "Could not find dep_v key'" + dep_v + "'. " << std::endl;
            if (propagator.find(dep_u) == propagator.end())
                std::cout << "Could not find dep_u key'" + dep_u + "'. " << std::endl;
            #endif

            // Calculate phi of one block (possibly multiple blocks when using aggregation)
            calculate_phi_one_block(
                block->second,            // Phi
                propagator[dep_v],  // dependency v
                propagator[dep_u],  // dependency u
                exp_dw[monomer_type],     // Exp_dw
                n_segment_allocated,
                n_segment_offset,
                n_segment_original);
            
            // Normalize concentration
            Polymer& pc = molecules->get_polymer(p);
            double norm = molecules->get_ds()*pc.get_volume_fraction()/pc.get_alpha()/single_partitions[p]*n_repeated;
            for(int i=0; i<M; i++)
                block->second[i] *= norm;
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CpuPseudoDiscrete::advance_propagator(double *q_in, double *q_out,
                                 double *boltz_bond, double *exp_dw,
                                 double *q_mask)
{
    try
    {
        const int M = cb->get_n_grid();
        const int M_COMPLEX = this->n_complex_grid;
        std::complex<double> k_q_in[M_COMPLEX];

        // 3D fourier discrete transform, forward and inplace
        fft->forward(q_in,k_q_in);
        // Multiply exp(-k^2 ds/6) in fourier space, in all 3 directions
        for(int i=0; i<M_COMPLEX; i++)
            k_q_in[i] *= boltz_bond[i];
        // 3D fourier discrete transform, backward and inplace
        fft->backward(k_q_in,q_out);
        // Normalization calculation and evaluate exp(-w*ds) in real space
        for(int i=0; i<M; i++)
            q_out[i] *= exp_dw[i];
        
        // Multiply mask
        if (q_mask != nullptr)
        {
            for(int i=0; i<M; i++)
                q_out[i] *= q_mask[i];
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

void CpuPseudoDiscrete::advance_propagator_half_bond_step(double *q_in, double *q_out, double *boltz_bond_half)
{
    try
    {
        // Const int M = cb->get_n_grid();
        const int M_COMPLEX = this->n_complex_grid;
        std::complex<double> k_q_in[M_COMPLEX];

        // 3D fourier discrete transform, forward and inplace
        fft->forward(q_in,k_q_in);
        // Multiply exp(-k^2 ds/12) in fourier space, in all 3 directions
        for(int i=0; i<M_COMPLEX; i++)
            k_q_in[i] *= boltz_bond_half[i];
        // 3D fourier discrete transform, backward and inplace
        fft->backward(k_q_in,q_out);
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CpuPseudoDiscrete::calculate_phi_one_block(
    double *phi, double *q_1, double *q_2, double *exp_dw, const int N, const int N_OFFSET, const int N_ORIGINAL)
{
    try
    {
        const int M = cb->get_n_grid();
        // Compute segment concentration
        for(int i=0; i<M; i++)
            phi[i] = q_1[i+(N_ORIGINAL-N_OFFSET-1)*M]*q_2[i];
        for(int n=1; n<N; n++)
        {
            for(int i=0; i<M; i++)
                phi[i] += q_1[i+(N_ORIGINAL-N_OFFSET-n-1)*M]*q_2[i+n*M];
        }
        for(int i=0; i<M; i++)
            phi[i] /= exp_dw[i];
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
double CpuPseudoDiscrete::get_total_partition(int polymer)
{
    try
    {
        return single_partitions[polymer];
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CpuPseudoDiscrete::get_total_concentration(std::string monomer_type, double *phi)
{
    try
    {
        const int M = cb->get_n_grid();
        // Initialize array
        for(int i=0; i<M; i++)
            phi[i] = 0.0;

        // For each block
        for(const auto& block: block_phi)
        {
            std::string dep_v = std::get<1>(block.first);
            int n_segment_allocated = propagators_analyzer->get_essential_block(block.first).n_segment_allocated;
            if (PropagatorCode::get_monomer_type_from_key(dep_v) == monomer_type && n_segment_allocated != 0)
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
void CpuPseudoDiscrete::get_total_concentration(int p, std::string monomer_type, double *phi)
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
        for(const auto& block: block_phi)
        {
            int polymer_idx = std::get<0>(block.first);
            std::string dep_v = std::get<1>(block.first);
            int n_segment_allocated = propagators_analyzer->get_essential_block(block.first).n_segment_allocated;
            if (polymer_idx == p && PropagatorCode::get_monomer_type_from_key(dep_v) == monomer_type && n_segment_allocated != 0)
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
void CpuPseudoDiscrete::get_block_concentration(int p, double *phi)
{
    try
    {
        const int M = cb->get_n_grid();
        const int P = molecules->get_n_polymer_types();

        if (p < 0 || p > P-1)
            throw_with_line_number("Index (" + std::to_string(p) + ") must be in range [0, " + std::to_string(P-1) + "]");

        if (propagators_analyzer->is_using_propagator_aggregation())
            throw_with_line_number("Disable 'aggregation' option to obtain concentration of each block.");

        Polymer& pc = molecules->get_polymer(p);
        std::vector<Block>& blocks = pc.get_blocks();

        for(size_t b=0; b<blocks.size(); b++)
        {
            std::string dep_v = pc.get_propagator_key(blocks[b].v, blocks[b].u);
            std::string dep_u = pc.get_propagator_key(blocks[b].u, blocks[b].v);
            if (dep_v < dep_u)
                dep_v.swap(dep_u);

            double* _essential_block_phi = block_phi[std::make_tuple(p, dep_v, dep_u)];
            for(int i=0; i<M; i++)
                phi[i+b*M] = _essential_block_phi[i]; 
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
std::vector<double> CpuPseudoDiscrete::compute_stress()
{
    // This method should be invoked after invoking compute_statistics().

    // To calculate stress, we multiply weighted fourier basis to q(k)*q^dagger(-k).
    // We only need the real part of stress calculation.

    try
    {
        const int DIM  = cb->get_dim();
        const int M    = cb->get_n_grid();
        const int M_COMPLEX = this->n_complex_grid;

        auto bond_lengths = molecules->get_bond_lengths();
        std::vector<double> stress(DIM);
        std::map<std::tuple<int, std::string, std::string>, std::array<double,3>> block_dq_dl;

        // Reset stress map
        for(const auto& item: block_phi)
        {
            for(int d=0; d<3; d++)
                block_dq_dl[item.first][d] = 0.0;
        }

        // Compute stress for each block
        #pragma omp parallel for
        for(size_t b=0; b<block_phi.size();b++)
        {
            auto block = block_phi.begin();
            advance(block, b);
            const auto& key   = block->first;

            std::string dep_v = std::get<1>(key);
            std::string dep_u = std::get<2>(key);

            const int N           = propagators_analyzer->get_essential_block(key).n_segment_allocated;
            const int N_OFFSET    = propagators_analyzer->get_essential_block(key).n_segment_offset;
            const int N_ORIGINAL  = propagators_analyzer->get_essential_block(key).n_segment_original;
            std::string monomer_type = propagators_analyzer->get_essential_block(key).monomer_type;

            // Contains no '['
            int n_repeated;
            if (dep_u.find('[') == std::string::npos)
                n_repeated = propagators_analyzer->get_essential_block(key).v_u.size();
            else
                n_repeated = 1;

            std::complex<double> qk_1[M_COMPLEX];
            std::complex<double> qk_2[M_COMPLEX];

            double *q_1 = propagator[dep_v];    // dependency v
            double *q_2 = propagator[dep_u];    // dependency u

            double coeff;
            double bond_length_sq;
            double *boltz_bond_now;

            // std::cout << "dep_v, dep_u, N_ORIGINAL, N_OFFSET, N: "
            //      << dep_v << ", " << dep_u << ", " << N_ORIGINAL << ", "<< N_OFFSET << ", " << N << std::endl;

            std::array<double,3> _block_dq_dl = block_dq_dl[key];

            // Compute stress at each chain bond
            for(int n=0; n<=N; n++)
            {
                // Block_dq_dl[key][0] = 0.0;
                // At v
                if (n + N_OFFSET == N_ORIGINAL)
                {
                    // std::cout << "case 1: " << propagator_junction[dep_v][0] << ", " << q_2[(N-1)*M] << std::endl;
                    if (propagators_analyzer->get_essential_propagator_code(dep_v).deps.size() == 0) // if v is leaf node, skip
                        continue;
                    fft->forward(propagator_junction[dep_v], qk_1);
                    fft->forward(&q_2[(N-1)*M], qk_2);
                    bond_length_sq = 0.5*bond_lengths[monomer_type]*bond_lengths[monomer_type];
                    boltz_bond_now = boltz_bond_half[monomer_type];
                }
                // At u
                else if (n + N_OFFSET == 0){
                    // std::cout << "case 2: " << q_1[(N_ORIGINAL-N_OFFSET-1)*M] << ", " << propagator_junction[dep_u][0] << std::endl;
                    if (propagators_analyzer->get_essential_propagator_code(dep_u).deps.size() == 0) // if u is leaf node, skip
                        continue;
                    fft->forward(&q_1[(N_ORIGINAL-1)*M], qk_1);
                    fft->forward(propagator_junction[dep_u], qk_2);
                    bond_length_sq = 0.5*bond_lengths[monomer_type]*bond_lengths[monomer_type];
                    boltz_bond_now = boltz_bond_half[monomer_type];
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
                    // std::cout << "case 5: " << q_1[(N_ORIGINAL-N_OFFSET-n-1)*M] << ", " << q_2[(n-1)*M] << std::endl;

                    // double temp_sum1=0;
                    // double temp_sum2=0;
                    // For (int i=0;i<M;i++)
                    // {
                    //     temp_sum1 += q_1[(N_ORIGINAL-N_OFFSET-n-1)*M+1];
                    //     temp_sum2 += q_2[(n-1)*M+1];
                    // }
                    // std::cout << "\t" << temp_sum1 << ", " << temp_sum2 << std::endl;

                    fft->forward(&q_1[(N_ORIGINAL-N_OFFSET-n-1)*M], qk_1);
                    fft->forward(&q_2[(n-1)*M], qk_2);
                    bond_length_sq = bond_lengths[monomer_type]*bond_lengths[monomer_type];
                    boltz_bond_now = boltz_bond[monomer_type];

                    // std::cout << "\t" << bond_length_sq << ", " << boltz_bond_now[10] << std::endl;
                }
                // Compute 
                if ( DIM == 3 )
                {
                    for(int i=0; i<M_COMPLEX; i++)
                    {
                        coeff = bond_length_sq*boltz_bond_now[i]*(qk_1[i]*std::conj(qk_2[i])).real()*n_repeated;
                        _block_dq_dl[0] += coeff*fourier_basis_x[i];
                        _block_dq_dl[1] += coeff*fourier_basis_y[i];
                        _block_dq_dl[2] += coeff*fourier_basis_z[i];
                    }
                }
                else if ( DIM == 2 )
                {
                    for(int i=0; i<M_COMPLEX; i++)
                    {
                        coeff = bond_length_sq*boltz_bond_now[i]*(qk_1[i]*std::conj(qk_2[i])).real()*n_repeated;
                        _block_dq_dl[0] += coeff*fourier_basis_y[i];
                        _block_dq_dl[1] += coeff*fourier_basis_z[i];
                    }
                }
                else if ( DIM == 1 )
                {
                    for(int i=0; i<M_COMPLEX; i++)
                    {
                        coeff = bond_length_sq*boltz_bond_now[i]*(qk_1[i]*std::conj(qk_2[i])).real()*n_repeated;
                        _block_dq_dl[0] += coeff*fourier_basis_z[i];
                    }
                }
                // std::cout << "n: " << n << ", " << block_dq_dl[key][0] << std::endl;
            }
            block_dq_dl[key] = _block_dq_dl;
        }

        // Compute total stress
        for(int d=0; d<DIM; d++)
            stress[d] = 0.0;
        for(const auto& block: block_phi)
        {
            const auto& key      = block.first;
            int p                = std::get<0>(key);
            std::string dep_v    = std::get<1>(key);
            std::string dep_u    = std::get<2>(key);
            Polymer& pc = molecules->get_polymer(p);

            for(int d=0; d<DIM; d++)
                stress[d] += block_dq_dl[key][d]*pc.get_volume_fraction()/pc.get_alpha()/single_partitions[p];
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
void CpuPseudoDiscrete::get_chain_propagator(double *q_out, int polymer, int v, int u, int n)
{ 
    // This method should be invoked after invoking compute_statistics()

    // Get chain propagator for a selected polymer, block and direction.
    // This is made for debugging and testing.
    try
    {
        const int M = cb->get_n_grid();
        Polymer& pc = molecules->get_polymer(polymer);
        std::string dep = pc.get_propagator_key(v,u);

        if (propagators_analyzer->get_essential_propagator_codes().find(dep) == propagators_analyzer->get_essential_propagator_codes().end())
            throw_with_line_number("Could not find the propagator code '" + dep + "'. Disable 'aggregation' option to obtain propagators_analyzer.");
            
        const int N = propagators_analyzer->get_essential_propagator_codes()[dep].max_n_segment;
        if (n < 1 || n > N)
            throw_with_line_number("n (" + std::to_string(n) + ") must be in range [1, " + std::to_string(N) + "]");

        double* partition = propagator[dep];
        for(int i=0; i<M; i++)
            q_out[i] = partition[(n-1)*M+i];
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
bool CpuPseudoDiscrete::check_total_partition()
{
    const int M = cb->get_n_grid();
    int n_polymer_types = molecules->get_n_polymer_types();
    std::vector<std::vector<double>> total_partitions;
    for(int p=0;p<n_polymer_types;p++)
    {
        std::vector<double> total_partitions_p;
        total_partitions.push_back(total_partitions_p);
    }

    for(const auto& block: block_phi)
    {
        const auto& key = block.first;
        int p                = std::get<0>(key);
        std::string dep_v    = std::get<1>(key);
        std::string dep_u    = std::get<2>(key);

        int n_aggregated;
        int n_segment_allocated = propagators_analyzer->get_essential_block(key).n_segment_allocated;
        int n_segment_offset    = propagators_analyzer->get_essential_block(key).n_segment_offset;
        int n_segment_original  = propagators_analyzer->get_essential_block(key).n_segment_original;
        std::string monomer_type = propagators_analyzer->get_essential_block(key).monomer_type;

        // std::cout<< p << ", " << dep_v << ", " << dep_u << ": " << n_segment_original << ", " << n_segment_offset << ", " << n_segment_allocated << std::endl;

        // Contains no '['
        if (dep_u.find('[') == std::string::npos)
            n_aggregated = 1;
        else
            n_aggregated = propagators_analyzer->get_essential_block(key).v_u.size();

        for(int n=0;n<n_segment_allocated;n++)
        {
            double total_partition = cb->inner_product_inverse_weight(
                &propagator[dep_v][(n_segment_original-n_segment_offset-n-1)*M],
                &propagator[dep_u][n*M], exp_dw[monomer_type])/n_aggregated/cb->get_volume();

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
        double diff_partition = abs(max_partition - min_partition);

        std::cout<< "\t" << p << ": " << max_partition << ", " << min_partition << ", " << diff_partition << std::endl;
        if (diff_partition > 1e-7)
            return false;
    }
    return true;
}