#include <cmath>
#include "CpuPseudoDiscrete.h"
#include "SimpsonQuadrature.h"

CpuPseudoDiscrete::CpuPseudoDiscrete(
    ComputationBox *cb,
    Mixture *mx,
    FFT *fft)
    : Pseudo(cb, mx)
{
    try
    {
        const int M = cb->get_n_grid();
        const int M_COMPLEX = this->n_complex_grid;
        this->fft = fft;

        // allocate memory for partition functions
        if( mx->get_unique_branches().size() == 0)
            throw_with_line_number("There is no unique branch. Add polymers first.");
        for(const auto& item: mx->get_unique_branches())
        {
            std::string dep = item.first;
            int max_n_segment = item.second.max_n_segment;
            unique_partition[dep] = new double[M*max_n_segment];
            unique_partition_finished[dep] = new bool[max_n_segment];
            for(int i=0; i<max_n_segment;i++)
                unique_partition_finished[dep][i] = false;
             // There are N segments
             // Illustration (N==5)
             // O--O--O--O--O
             // 0  1  2  3  4 

             // Legend)
             // -- : full bond
             // O  : full segment
        }

        // allocate memory for unique_q_junctions, which contain partition function at junction of discrete chain
        for(const auto& item: mx->get_unique_branches())
        {
            unique_q_junctions[item.first] = new double[M];
        }

        // allocate memory for concentrations
        if( mx->get_unique_blocks().size() == 0)
            throw_with_line_number("There is no unique block. Add polymers first.");
        for(const auto& item: mx->get_unique_blocks())
        {
            unique_phi[item.first] = new double[M];
        }

        // create boltz_bond, boltz_bond_half, and exp_dw
        for(const auto& item: mx->get_bond_lengths())
        {
            std::string monomer_type = item.first;
            boltz_bond     [monomer_type] = new double[M_COMPLEX];
            boltz_bond_half[monomer_type] = new double[M_COMPLEX]; 
            exp_dw         [monomer_type] = new double[M];
        }

        // allocate memory for stress calculation: compute_stress()
        fourier_basis_x = new double[M_COMPLEX];
        fourier_basis_y = new double[M_COMPLEX];
        fourier_basis_z = new double[M_COMPLEX];

        // total partition functions for each polymer
        single_partitions = new double[mx->get_n_polymers()];

        // create scheduler for computation of partial partition function
        sc = new Scheduler(mx->get_unique_branches(), N_STREAM); 

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
    for(const auto& item: unique_partition)
        delete[] item.second;
    for(const auto& item: unique_phi)
        delete[] item.second;
    for(const auto& item: unique_q_junctions)
        delete[] item.second;
    for(const auto& item: unique_partition_finished)
        delete[] item.second;
}
void CpuPseudoDiscrete::update_bond_function()
{
    try
    {
        for(const auto& item: mx->get_bond_lengths())
        {
            std::string monomer_type = item.first;
            double bond_length_sq = item.second*item.second;
            get_boltz_bond(boltz_bond     [monomer_type], bond_length_sq,   cb->get_nx(), cb->get_dx(), mx->get_ds());
            get_boltz_bond(boltz_bond_half[monomer_type], bond_length_sq/2, cb->get_nx(), cb->get_dx(), mx->get_ds());

            // for stress calculation: compute_stress()
            get_weighted_fourier_basis(fourier_basis_x, fourier_basis_y, fourier_basis_z, cb->get_nx(), cb->get_dx());
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CpuPseudoDiscrete::compute_statistics(
    std::map<std::string, double*> w_input,
    std::map<std::string, double*> q_init)
{
    try
    {
        const int M = cb->get_n_grid();
        const double ds = mx->get_ds();

        for(const auto& item: mx->get_unique_branches())
        {
            if( w_input.find(item.second.monomer_type) == w_input.end())
                throw_with_line_number("monomer_type \"" + item.second.monomer_type + "\" is not in w_input.");
        }

        for(const auto& item: w_input)
        {
            if( exp_dw.find(item.first) == exp_dw.end())
                throw_with_line_number("monomer_type \"" + item.first + "\" is not in exp_dw.");     
        }

        // if( q_init.size() > 0)
        //     throw_with_line_number("Currently, \'q_init\' is not supported.");

        for(const auto& item: w_input)
        {
            std::string monomer_type = item.first;
            double *w = item.second;
            for(int i=0; i<M; i++)
                exp_dw[monomer_type][i] = exp(-w[i]*ds);
        }

        // for each time span
        auto& branch_schedule = sc->get_schedule();
        for (auto parallel_job = branch_schedule.begin(); parallel_job != branch_schedule.end(); parallel_job++)
        {
            // for each job
            #pragma omp parallel for
            for(size_t job=0; job<parallel_job->size(); job++)
            {
                auto& key = std::get<0>((*parallel_job)[job]);
                int n_segment_from = std::get<1>((*parallel_job)[job]);
                int n_segment_to = std::get<2>((*parallel_job)[job]);
                auto& deps = mx->get_unique_branch(key).deps;
                auto monomer_type = mx->get_unique_branch(key).monomer_type;

                // check key
                if (unique_partition.find(key) == unique_partition.end())
                    std::cout << "Could not find key '" << key << "'. " << std::endl;
                double *_unique_partition = unique_partition[key];

                // calculate one block end
                if(n_segment_from == 1 && deps.size() == 0) // if it is leaf node
                {
                     // q_init
                    if (key[0] == '{')
                    {
                        std::string g = Mixture::get_q_input_idx_from_key(key);
                        if (q_init.find(g) == q_init.end())
                            std::cout << "Could not find q_init[\"" + g + "\"]." << std::endl;
                        for(int i=0; i<M; i++)
                            _unique_partition[i] = q_init[g][i]*exp_dw[monomer_type][i];
                    }
                    else
                    {
                        for(int i=0; i<M; i++)
                            _unique_partition[i] = exp_dw[monomer_type][i];
                    }
                    unique_partition_finished[key][0] = true;

                }
                else if (n_segment_from == 1 && deps.size() > 0) // if it is not leaf node
                {
                    // if it is superposition
                    if (key[0] == '[')
                    {
                        for(int i=0; i<M; i++)
                            _unique_partition[i] = 0.0;
                        for(size_t d=0; d<deps.size(); d++)
                        {
                            std::string sub_dep = std::get<0>(deps[d]);
                            int sub_n_segment   = std::get<1>(deps[d]);
                            int sub_n_repeated  = std::get<2>(deps[d]);

                            // check sub key
                            if (unique_partition.find(sub_dep) == unique_partition.end())
                                std::cout << "Could not find sub key '" + sub_dep + "'. " << std::endl;
                            if (!unique_partition_finished[sub_dep][sub_n_segment-1])
                                std::cout << "Could not compute '" + key +  "', since '"+ sub_dep + std::to_string(sub_n_segment) + "' is not prepared." << std::endl;

                            double *_unique_partition_sub_dep = unique_partition[sub_dep];
                            for(int i=0; i<M; i++)
                                _unique_partition[i] += _unique_partition_sub_dep[(sub_n_segment-1)*M+i]*sub_n_repeated;
                        }
                        one_step(&_unique_partition[0],
                            &_unique_partition[0],
                            boltz_bond[monomer_type],
                            exp_dw[monomer_type]);

                        unique_partition_finished[key][0] = true;
                        // std::cout << "finished, key, n: " + key + ", 0" << std::endl;
                    }
                    else
                    {
                        // Illustration (four branches)
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

                        // combine branches
                        double q_junction[M];
                        for(int i=0; i<M; i++)
                            q_junction[i] = 1.0;
                        for(size_t d=0; d<deps.size(); d++)
                        {
                            std::string sub_dep = std::get<0>(deps[d]);
                            int sub_n_segment   = std::get<1>(deps[d]);
                            double q_half_step[M];

                            // check sub key
                            if (unique_partition.find(sub_dep) == unique_partition.end())
                                std::cout << "Could not find sub key '" + sub_dep + "'. " << std::endl;
                            if (!unique_partition_finished[sub_dep][sub_n_segment-1])
                                std::cout << "Could not compute '" + key +  "', since '"+ sub_dep + std::to_string(sub_n_segment) + "' is not prepared." << std::endl;

                            half_bond_step(&unique_partition[sub_dep][(sub_n_segment-1)*M],
                                q_half_step, boltz_bond_half[mx->get_unique_branch(sub_dep).monomer_type]);

                            for(int i=0; i<M; i++)
                                q_junction[i] *= q_half_step[i];
                        }
                        double *_unique_q_junctions = unique_q_junctions[key];
                        for(int i=0; i<M; i++)
                            _unique_q_junctions[i] = q_junction[i];

                        // add half bond
                        half_bond_step(q_junction, &_unique_partition[0], boltz_bond_half[monomer_type]);

                        // add full segment
                        for(int i=0; i<M; i++)
                            _unique_partition[i] *= exp_dw[monomer_type][i];
                        unique_partition_finished[key][0] = true;
                    }
                }
                else
                {
                    n_segment_from--;
                }

                // apply the propagator successively
                for(int n=n_segment_from; n<n_segment_to; n++)
                {
                    if (!unique_partition_finished[key][n-1])
                        std::cout << "unfinished, key: " + key + ", " + std::to_string(n);

                    one_step(&_unique_partition[(n-1)*M],
                            &_unique_partition[n*M],
                            boltz_bond[monomer_type],
                            exp_dw[monomer_type]);
                    unique_partition_finished[key][n] = true;

                    // std::cout << "finished, key, n: " + key + ", " << std::to_string(n) << std::endl;
                }
            }
        }

        // compute total partition function of each distinct polymers
        int current_p = 0;
        for(const auto& block: unique_phi)
        {
            int p                = std::get<0>(block.first);
            std::string dep_v    = std::get<1>(block.first);
            std::string dep_u    = std::get<2>(block.first);

            // already computed
            if (p != current_p)
                continue;

            int n_superposed;
            // int n_segment_allocated = mx->get_unique_block(block.first).n_segment_allocated;
            int n_segment_offset    = mx->get_unique_block(block.first).n_segment_offset;
            int n_segment_original  = mx->get_unique_block(block.first).n_segment_original;
            std::string monomer_type = mx->get_unique_block(block.first).monomer_type;

            // contains no '['
            if (dep_u.find('[') == std::string::npos)
                n_superposed = 1;
            else
                n_superposed = mx->get_unique_block(block.first).v_u.size();

            // check keys
            if (unique_partition.find(dep_v) == unique_partition.end())
                std::cout << "Could not find dep_v key'" + dep_v + "'. " << std::endl;
            if (unique_partition.find(dep_u) == unique_partition.end())
                std::cout << "Could not find dep_u key'" + dep_u + "'. " << std::endl;

            single_partitions[p]= cb->inner_product_inverse_weight(
                &unique_partition[dep_v][(n_segment_original-n_segment_offset-1)*M],  // q
                &unique_partition[dep_u][0],                                          // q^dagger
                exp_dw[monomer_type])/n_superposed/cb->get_volume();

            // std::cout << p <<", "<< dep_v <<", "<< dep_u <<", "<< n_segment <<", " << single_partitions[p] << std::endl;
            // std::cout << p <<", "<< n_segment <<", "<< n_segment_offset <<", "<< single_partitions[p] << std::endl;
            current_p++;
        }

        // calculate segment concentrations
        #pragma omp parallel for
        for(int b=0; b<unique_phi.size();b++)
        {
            auto block = unique_phi.begin();
            advance(block, b);
            const auto& key = block->first;

            int p                = std::get<0>(key);
            std::string dep_v    = std::get<1>(key);
            std::string dep_u    = std::get<2>(key);

            int n_repeated;
            int n_segment_allocated = mx->get_unique_block(key).n_segment_allocated;
            int n_segment_offset    = mx->get_unique_block(key).n_segment_offset;
            int n_segment_original  = mx->get_unique_block(key).n_segment_original;
            std::string monomer_type = mx->get_unique_block(key).monomer_type;

            // contains no '['
            if (dep_u.find('[') == std::string::npos)
                n_repeated = mx->get_unique_block(key).v_u.size();
            else
                n_repeated = 1;

            // check keys
            if (unique_partition.find(dep_v) == unique_partition.end())
                std::cout << "Could not find dep_v key'" + dep_v + "'. " << std::endl;
            if (unique_partition.find(dep_u) == unique_partition.end())
                std::cout << "Could not find dep_u key'" + dep_u + "'. " << std::endl;

            // calculate phi of one block (possibly multiple blocks when using superposition)
            calculate_phi_one_block(
                block->second,            // phi
                unique_partition[dep_v],  // dependency v
                unique_partition[dep_u],  // dependency u
                exp_dw[monomer_type],     // exp_dw
                n_segment_allocated,
                n_segment_offset,
                n_segment_original);
            
            // normalize concentration
            PolymerChain& pc = mx->get_polymer(p);
            double norm = mx->get_ds()*pc.get_volume_fraction()/pc.get_alpha()/single_partitions[p]*n_repeated;
            for(int i=0; i<M; i++)
                block->second[i] *= norm;
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CpuPseudoDiscrete::one_step(double *q_in, double *q_out,
                                 double *boltz_bond, double *exp_dw)
{
    try
    {
        const int M = cb->get_n_grid();
        const int M_COMPLEX = this->n_complex_grid;
        std::complex<double> k_q_in[M_COMPLEX];

        // 3D fourier discrete transform, forward and inplace
        fft->forward(q_in,k_q_in);
        // multiply e^(-k^2 ds/6) in fourier space, in all 3 directions
        for(int i=0; i<M_COMPLEX; i++)
            k_q_in[i] *= boltz_bond[i];
        // 3D fourier discrete transform, backward and inplace
        fft->backward(k_q_in,q_out);
        // normalization calculation and evaluate e^(-w*ds) in real space
        for(int i=0; i<M; i++)
            q_out[i] *= exp_dw[i];
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

void CpuPseudoDiscrete::half_bond_step(double *q_in, double *q_out, double *boltz_bond_half)
{
    try
    {
        // const int M = cb->get_n_grid();
        const int M_COMPLEX = this->n_complex_grid;
        std::complex<double> k_q_in[M_COMPLEX];

        // 3D fourier discrete transform, forward and inplace
        fft->forward(q_in,k_q_in);
        // multiply e^(-k^2 ds/12) in fourier space, in all 3 directions
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
void CpuPseudoDiscrete::get_monomer_concentration(std::string monomer_type, double *phi)
{
    try
    {
        const int M = cb->get_n_grid();
        // initialize array
        for(int i=0; i<M; i++)
            phi[i] = 0.0;

        // for each block
        for(const auto& block: unique_phi)
        {
            std::string dep_v = std::get<1>(block.first);
            int n_segment_allocated = mx->get_unique_block(block.first).n_segment_allocated;
            if (Mixture::get_monomer_type_from_key(dep_v) == monomer_type && n_segment_allocated != 0)
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
void CpuPseudoDiscrete::get_polymer_concentration(int p, double *phi)
{
    try
    {
        const int M = cb->get_n_grid();
        const int P = mx->get_n_polymers();

        if (p < 0 || p > P-1)
            throw_with_line_number("Index (" + std::to_string(p) + ") must be in range [0, " + std::to_string(P-1) + "]");

        if (mx->is_using_superposition())
            throw_with_line_number("Disable 'use_superposition' to obtain concentration of each block.");

        PolymerChain& pc = mx->get_polymer(p);
        std::vector<PolymerChainBlock>& blocks = pc.get_blocks();

        for(size_t b=0; b<blocks.size(); b++)
        {
            std::string dep_v = pc.get_dep(blocks[b].v, blocks[b].u);
            std::string dep_u = pc.get_dep(blocks[b].u, blocks[b].v);
            if (dep_v < dep_u)
                dep_v.swap(dep_u);

            double* _unique_phi = unique_phi[std::make_tuple(p, dep_v, dep_u)];
            for(int i=0; i<M; i++)
                phi[i+b*M] = _unique_phi[i]; 
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

        auto bond_lengths = mx->get_bond_lengths();
        std::vector<double> stress(cb->get_dim());
        std::map<std::tuple<int, std::string, std::string>, std::array<double,3>> unique_dq_dl;

        // reset stress map
        for(const auto& item: unique_phi)
        {
            for(int d=0; d<3; d++)
                unique_dq_dl[item.first][d] = 0.0;
        }

        // compute stress for each unique block
        #pragma omp parallel for
        for(int b=0; b<unique_phi.size();b++)
        {
            auto block = unique_phi.begin();
            advance(block, b);
            const auto& key   = block->first;

            std::string dep_v = std::get<1>(key);
            std::string dep_u = std::get<2>(key);

            const int N           = mx->get_unique_block(key).n_segment_allocated;
            const int N_OFFSET    = mx->get_unique_block(key).n_segment_offset;
            const int N_ORIGINAL  = mx->get_unique_block(key).n_segment_original;
            std::string monomer_type = mx->get_unique_block(key).monomer_type;

            // contains no '['
            int n_repeated;
            if (dep_u.find('[') == std::string::npos)
                n_repeated = mx->get_unique_block(key).v_u.size();
            else
                n_repeated = 1;

            std::complex<double> qk_1[M_COMPLEX];
            std::complex<double> qk_2[M_COMPLEX];

            double *q_1 = unique_partition[dep_v];    // dependency v
            double *q_2 = unique_partition[dep_u];    // dependency u

            double coeff;
            double bond_length_sq;
            double *boltz_bond_now;

            // std::cout << "dep_v, dep_u, N_ORIGINAL, N_OFFSET, N: "
            //      << dep_v << ", " << dep_u << ", " << N_ORIGINAL << ", "<< N_OFFSET << ", " << N << std::endl;

            std::array<double,3> _unique_dq_dl = unique_dq_dl[key];

            // compute stress at each chain bond
            for(int n=0; n<=N; n++)
            {
                // unique_dq_dl[key][0] = 0.0;
                // at v
                if (n + N_OFFSET == N_ORIGINAL)
                {
                    // std::cout << "case 1: " << unique_q_junctions[dep_v][0] << ", " << q_2[(N-1)*M] << std::endl;
                    if (mx->get_unique_branch(dep_v).deps.size() == 0) // if v is leaf node, skip
                        continue;
                    fft->forward(unique_q_junctions[dep_v], qk_1);
                    fft->forward(&q_2[(N-1)*M], qk_2);
                    bond_length_sq = 0.5*bond_lengths[monomer_type]*bond_lengths[monomer_type];
                    boltz_bond_now = boltz_bond_half[monomer_type];
                }
                // at u
                else if (n + N_OFFSET == 0){
                    // std::cout << "case 2: " << q_1[(N_ORIGINAL-N_OFFSET-1)*M] << ", " << unique_q_junctions[dep_u][0] << std::endl;
                    if (mx->get_unique_branch(dep_u).deps.size() == 0) // if u is leaf node, skip
                        continue;
                    fft->forward(&q_1[(N_ORIGINAL-1)*M], qk_1);
                    fft->forward(unique_q_junctions[dep_u], qk_2);
                    bond_length_sq = 0.5*bond_lengths[monomer_type]*bond_lengths[monomer_type];
                    boltz_bond_now = boltz_bond_half[monomer_type];
                }
                // at superposition junction
                else if (n == 0)
                {
                    // std::cout << "case 4" << std::endl;
                    continue;
                }
                // within the blocks
                else
                {
                    // std::cout << "case 5: " << q_1[(N_ORIGINAL-N_OFFSET-n-1)*M] << ", " << q_2[(n-1)*M] << std::endl;

                    // double temp_sum1=0;
                    // double temp_sum2=0;
                    // for (int i=0;i<M;i++)
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
                // compute 
                if ( DIM == 3 )
                {
                    for(int i=0; i<M_COMPLEX; i++)
                    {
                        coeff = bond_length_sq*boltz_bond_now[i]*(qk_1[i]*std::conj(qk_2[i])).real()*n_repeated;
                        _unique_dq_dl[0] += coeff*fourier_basis_x[i];
                        _unique_dq_dl[1] += coeff*fourier_basis_y[i];
                        _unique_dq_dl[2] += coeff*fourier_basis_z[i];
                    }
                }
                else if ( DIM == 2 )
                {
                    for(int i=0; i<M_COMPLEX; i++)
                    {
                        coeff = bond_length_sq*boltz_bond_now[i]*(qk_1[i]*std::conj(qk_2[i])).real()*n_repeated;
                        _unique_dq_dl[0] += coeff*fourier_basis_y[i];
                        _unique_dq_dl[1] += coeff*fourier_basis_z[i];
                    }
                }
                else if ( DIM == 1 )
                {
                    for(int i=0; i<M_COMPLEX; i++)
                    {
                        coeff = bond_length_sq*boltz_bond_now[i]*(qk_1[i]*std::conj(qk_2[i])).real()*n_repeated;
                        _unique_dq_dl[0] += coeff*fourier_basis_z[i];
                    }
                }
                // std::cout << "n: " << n << ", " << unique_dq_dl[key][0] << std::endl;
            }
            unique_dq_dl[key] = _unique_dq_dl;
        }

        // compute total stress
        for(int d=0; d<cb->get_dim(); d++)
            stress[d] = 0.0;
        for(const auto& block: unique_phi)
        {
            const auto& key      = block.first;
            int p                = std::get<0>(key);
            std::string dep_v    = std::get<1>(key);
            std::string dep_u    = std::get<2>(key);
            PolymerChain& pc = mx->get_polymer(p);

            for(int d=0; d<cb->get_dim(); d++)
                stress[d] += unique_dq_dl[key][d]*pc.get_volume_fraction()/pc.get_alpha()/single_partitions[p];
        }
        for(int d=0; d<cb->get_dim(); d++)
            stress[d] /= -3.0*cb->get_lx(d)*M*M/mx->get_ds();

        return stress;
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CpuPseudoDiscrete::get_partial_partition(double *q_out, int polymer, int v, int u, int n)
{ 
    // This method should be invoked after invoking compute_statistics()

    // Get partial partition functions
    // This is made for debugging and testing
    try
    {
        const int M = cb->get_n_grid();
        PolymerChain& pc = mx->get_polymer(polymer);
        std::string dep = pc.get_dep(v,u);

        if (mx->get_unique_branches().find(dep) == mx->get_unique_branches().end())
            throw_with_line_number("Could not find the branches '" + dep + "'. Disable 'use_superposition' to obtain partial partition functions.");
            
        const int N = mx->get_unique_branches()[dep].max_n_segment;
        if (n < 1 || n > N)
            throw_with_line_number("n (" + std::to_string(n) + ") must be in range [1, " + std::to_string(N) + "]");

        double* partition = unique_partition[dep];
        for(int i=0; i<M; i++)
            q_out[i] = partition[(n-1)*M+i];
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
