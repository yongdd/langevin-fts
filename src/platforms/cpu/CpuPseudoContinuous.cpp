#include <cmath>

#include "CpuPseudoContinuous.h"
#include "SimpsonQuadrature.h"

CpuPseudoContinuous::CpuPseudoContinuous(
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
            unique_partition[dep] = new double[M*(max_n_segment+1)];
            unique_partition_finished[dep] = new bool[max_n_segment+1];
            for(int i=0; i<=max_n_segment;i++)
                unique_partition_finished[dep][i] = false;
        }

        // allocate memory for concentrations
        if( mx->get_unique_blocks().size() == 0)
            throw_with_line_number("There is no unique block. Add polymers first.");
        for(const auto& item: mx->get_unique_blocks())
        {
            unique_phi[item.first] = new double[M];
        }

        // create boltz_bond, boltz_bond_half, exp_dw, and exp_dw_half
        for(const auto& item: mx->get_bond_lengths())
        {
            std::string monomer_type = item.first;
            boltz_bond     [monomer_type] = new double[M_COMPLEX];
            boltz_bond_half[monomer_type] = new double[M_COMPLEX]; 
            exp_dw         [monomer_type] = new double[M];
            exp_dw_half    [monomer_type] = new double[M]; 
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
CpuPseudoContinuous::~CpuPseudoContinuous()
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
    for(const auto& item: exp_dw_half)
        delete[] item.second;
    for(const auto& item: unique_partition)
        delete[] item.second;
    for(const auto& item: unique_phi)
        delete[] item.second;
    for(const auto& item: unique_partition_finished)
        delete[] item.second;
}
void CpuPseudoContinuous::update_bond_function()
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
void CpuPseudoContinuous::compute_statistics(
    std::map<std::string, double*> q_init,
    std::map<std::string, double*> w_input)
{
    try
    {
        const int M = cb->get_n_grid();
        const double ds = mx->get_ds();

        for(const auto& item: mx->get_unique_branches())
        {
            if( w_input.count(item.second.monomer_type) == 0)
                throw_with_line_number("\"" + item.second.monomer_type + "\" monomer_type is not in w_input.");
        }

        if( q_init.size() > 0)
            throw_with_line_number("Currently, \'q_init\' is not supported.");

        for(const auto& item: w_input)
        {
            std::string monomer_type = item.first;
            double *w = item.second;
            for(int i=0; i<M; i++)
            { 
                exp_dw     [monomer_type][i] = exp(-w[i]*ds*0.5);
                exp_dw_half[monomer_type][i] = exp(-w[i]*ds*0.25);
            }
        }

        auto& branch_schedule = sc->get_schedule();
        // // display all jobs
        // for (auto parallel_job = branch_schedule.begin(); parallel_job != branch_schedule.end(); parallel_job++)
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

        // for each time span
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
                    std::cout << "Could not find key '" + key + "'. " << std::endl;

                // calculate one block end
                if(n_segment_from == 1 && deps.size() == 0) // if it is leaf node
                {
                    for(int i=0; i<M; i++)
                        unique_partition[key][i] = 1.0; //* q_init
                    unique_partition_finished[key][0] = true;
                }
                else if (n_segment_from == 1 && deps.size() > 0) // if it is not leaf node
                {
                    // if it is superposition
                    if (key[0] == '[')
                    {
                        for(int i=0; i<M; i++)
                            unique_partition[key][i] = 0.0;
                        for(size_t d=0; d<deps.size(); d++)
                        {
                            std::string sub_dep = std::get<0>(deps[d]);
                            int sub_n_segment   = std::get<1>(deps[d]);
                            int sub_n_repeated  = std::get<2>(deps[d]);

                            // check sub key
                            if (unique_partition.find(sub_dep) == unique_partition.end())
                                std::cout << "Could not find sub key '" + sub_dep + "'. " << std::endl;
                            if (!unique_partition_finished[sub_dep][sub_n_segment])
                                std::cout << "Could not compute '" + key +  "', since '"+ sub_dep + std::to_string(sub_n_segment) + "' is not prepared." << std::endl;

                            for(int i=0; i<M; i++)
                                unique_partition[key][i] += unique_partition[sub_dep][sub_n_segment*M+i]*sub_n_repeated;
                        }
                        unique_partition_finished[key][0] = true;
                        // std::cout << "finished, key, n: " + key + ", 0" << std::endl;
                    }
                    else
                    { 
                        for(int i=0; i<M; i++)
                            unique_partition[key][i] = 1.0;
                        for(size_t d=0; d<deps.size(); d++)
                        {
                            std::string sub_dep = std::get<0>(deps[d]);
                            int sub_n_segment   = std::get<1>(deps[d]);

                            // check sub key
                            if (unique_partition.find(sub_dep) == unique_partition.end())
                                std::cout << "Could not find sub key '" + sub_dep + "'. " << std::endl;
                            if (!unique_partition_finished[sub_dep][sub_n_segment])
                                std::cout << "Could not compute '" + key +  "', since '"+ sub_dep + std::to_string(sub_n_segment) + "' is not prepared." << std::endl;

                            for(int i=0; i<M; i++)
                                unique_partition[key][i] *= unique_partition[sub_dep][sub_n_segment*M+i];
                        }
                        unique_partition_finished[key][0] = true;
                        // std::cout << "finished, key, n: " + key + ", 0" << std::endl;
                    }
                }
        
                // apply the propagator successively
                for(int n=n_segment_from; n<=n_segment_to; n++)
                {
                    if (!unique_partition_finished[key][n-1])
                        std::cout << "unfinished, key: " + key + ", " + std::to_string(n-1) << std::endl;

                    one_step(&unique_partition[key][(n-1)*M],
                            &unique_partition[key][n*M],
                            boltz_bond[monomer_type],
                            boltz_bond_half[monomer_type],
                            exp_dw[monomer_type],
                            exp_dw_half[monomer_type]);
                    unique_partition_finished[key][n] = true;

                    // std::cout << "finished, key, n: " + key + ", " << std::to_string(n) << std::endl;
                }
            }
        }

        // for(const auto& block: unique_phi)
        // {
        //     int p                = std::get<0>(block.first);
        //     std::string dep_v    = std::get<1>(block.first);
        //     std::string dep_u    = std::get<2>(block.first);
        //     int n_segment        = std::get<3>(block.first);
        //     int n_segment_offset = std::get<4>(block.first);

        //     // check keys
        //     if (unique_partition.find(dep_v) == unique_partition.end())
        //         throw_with_line_number("Could not find dep_v key'" + dep_v + "'. ");
        //     if (unique_partition.find(dep_u) == unique_partition.end())
        //         throw_with_line_number("Could not find dep_u key'" + dep_u + "'. ");

        //     for(int i=0; i<=n_segment+n_segment_offset; i++)
        //     {
        //         if (!unique_partition_finished[dep_v][i])
        //             throw_with_line_number("unfinished, dep_v, n'" + dep_v + ", " + std::to_string(i) + "'. ");
        //     }

        //     for(int i=0; i<=n_segment; i++)
        //     {
        //         if (!unique_partition_finished[dep_u][i])
        //             throw_with_line_number("unfinished, dep_u, n'" + dep_u + ", " + std::to_string(i) + "'. ");
        //     }
        // }

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
            // int n_segment          = std::get<3>(block.first);
            int original_n_segment = mx->get_unique_block(block.first).n_segment;

            // contains no '['
            if (dep_u.find('[') == std::string::npos)
                n_superposed = 1;
            else
                n_superposed = mx->get_unique_block(block.first).v_u.size();

            // check keys
            if (unique_partition.find(dep_v) == unique_partition.end())
                throw_with_line_number("Could not find dep_v key'" + dep_v + "'. ");
            if (unique_partition.find(dep_u) == unique_partition.end())
                throw_with_line_number("Could not find dep_u key'" + dep_u + "'. ");

            single_partitions[p]= cb->inner_product(
                &unique_partition[dep_v][original_n_segment*M],             // q
                &unique_partition[dep_u][0])/n_superposed/cb->get_volume(); // q^dagger
            
            // std::cout << p <<", "<< dep_v <<", "<< dep_u <<", "<< n_segment <<", " << single_partitions[p] << std::endl;
            // std::cout << p <<", "<< n_segment <<", "<< n_segment_offset <<", "<< single_partitions[p] << std::endl;
            current_p++;
        }

        // calculate segment concentrations
        // for(const auto& item: mx->get_unique_blocks())
        // {
        // #pragma omp parallel for
        for(const auto& block: unique_phi)
        {
            int p                = std::get<0>(block.first);
            std::string dep_v    = std::get<1>(block.first);
            std::string dep_u    = std::get<2>(block.first);

            int n_repeated;
            int n_segment          = std::get<3>(block.first);
            // int original_n_segment = mx->get_unique_block(block.first).n_segment;

            // if there is no segment
            if(n_segment == 0)
            {
                for(int i=0; i<M;i++)
                    block.second[i] = 0.0;
                continue;
            }

            // contains no '['
            if (dep_u.find('[') == std::string::npos)
                n_repeated = mx->get_unique_block(block.first).v_u.size();
            else
                n_repeated = 1;

            // check keys
            if (unique_partition.find(dep_v) == unique_partition.end())
                throw_with_line_number("Could not find dep_v key'" + dep_v + "'. ");
            if (unique_partition.find(dep_u) == unique_partition.end())
                throw_with_line_number("Could not find dep_u key'" + dep_u + "'. ");

            // calculate phi of one block (possibly multiple blocks when using superposition)
            calculate_phi_one_block(
                block.second,             // phi
                unique_partition[dep_v],  // dependency v
                unique_partition[dep_u],  // dependency u
                n_segment);               // n_segment

            // normalize concentration
            PolymerChain& pc = mx->get_polymer(p);
            double norm = mx->get_ds()*pc.get_volume_fraction()/pc.get_alpha()/single_partitions[p]*n_repeated;
            for(int i=0; i<M; i++)
                block.second[i] *= norm;
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CpuPseudoContinuous::one_step(double *q_in, double *q_out,
                                 double *boltz_bond, double *boltz_bond_half,
                                 double *exp_dw, double *exp_dw_half)
{
    try
    {
        const int M = cb->get_n_grid();
        const int M_COMPLEX = this->n_complex_grid;
        double q_out1[M], q_out2[M];
        std::complex<double> k_q_in1[M_COMPLEX], k_q_in2[M_COMPLEX];

        #pragma omp parallel sections num_threads(2)
        {
            #pragma omp section
            {
                // step 1
                for(int i=0; i<M; i++)
                    q_out1[i] = exp_dw[i]*q_in[i];
                // 3D fourier discrete transform, forward and inplace
                fft->forward(q_out1,k_q_in1);
                // multiply e^(-k^2 ds/6) in fourier space, in all 3 directions
                for(int i=0; i<M_COMPLEX; i++)
                    k_q_in1[i] *= boltz_bond[i];
                // 3D fourier discrete transform, backward and inplace
                fft->backward(k_q_in1,q_out1);
                // normalization calculation and evaluate e^(-w*ds/2) in real space
                for(int i=0; i<M; i++)
                    q_out1[i] *= exp_dw[i];
            }
            #pragma omp section
            {
                // step 2
                // evaluate e^(-w*ds/4) in real space
                for(int i=0; i<M; i++)
                    q_out2[i] = exp_dw_half[i]*q_in[i];
                // 3D fourier discrete transform, forward and inplace
                fft->forward(q_out2,k_q_in2);
                // multiply e^(-k^2 ds/12) in fourier space, in all 3 directions
                for(int i=0; i<M_COMPLEX; i++)
                    k_q_in2[i] *= boltz_bond_half[i];
                // 3D fourier discrete transform, backward and inplace
                fft->backward(k_q_in2,q_out2);
                // normalization calculation and evaluate e^(-w*ds/2) in real space
                for(int i=0; i<M; i++)
                    q_out2[i] *= exp_dw[i];
                // 3D fourier discrete transform, forward and inplace
                fft->forward(q_out2,k_q_in2);
                // multiply e^(-k^2 ds/12) in fourier space, in all 3 directions
                for(int i=0; i<M_COMPLEX; i++)
                    k_q_in2[i] *= boltz_bond_half[i];
                // 3D fourier discrete transform, backward and inplace
                fft->backward(k_q_in2,q_out2);
                // normalization calculation and evaluate e^(-w*ds/4) in real space
                for(int i=0; i<M; i++)
                    q_out2[i] *= exp_dw_half[i];
            }
        }
        for(int i=0; i<M; i++)
            q_out[i] = (4.0*q_out2[i] - q_out1[i])/3.0;
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CpuPseudoContinuous::calculate_phi_one_block(
    double *phi, double *q_1, double *q_2, const int N)
{
    try
    {
        const int M = cb->get_n_grid();
        std::vector<double> simpson_rule_coeff = SimpsonQuadrature::get_coeff(N);

        // Compute segment concentration
        for(int i=0; i<M; i++)
            phi[i] = simpson_rule_coeff[0]*q_1[i]*q_2[i+N*M];
        for(int n=1; n<=N; n++)
        {
            for(int i=0; i<M; i++)
                phi[i] += simpson_rule_coeff[n]*q_1[i+n*M]*q_2[i+(N-n)*M];
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
double CpuPseudoContinuous::get_total_partition(int polymer)
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
void CpuPseudoContinuous::get_monomer_concentration(std::string monomer_type, double *phi)
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
            int n_segment     = std::get<3>(block.first);
            if (Mixture::key_to_species(dep_v) == monomer_type && n_segment != 0)
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
void CpuPseudoContinuous::get_polymer_concentration(int p, double *phi)
{
    try
    {
        const int M = cb->get_n_grid();
        const int P = mx->get_n_polymers();

        if (p < 0 || p > P-1)
            throw_with_line_number("Index (" + std::to_string(p) + ") must be in range [0, " + std::to_string(P-1) + "]");

        if (mx->is_using_superposition())
            throw_with_line_number("Disable 'use_superposition' to invoke 'get_polymer_concentration'.");

        PolymerChain& pc = mx->get_polymer(p);
        std::vector<PolymerChainBlock>& blocks = pc.get_blocks();

        for(size_t b=0; b<blocks.size(); b++)
        {
            std::string dep_v = pc.get_dep(blocks[b].v, blocks[b].u);
            std::string dep_u = pc.get_dep(blocks[b].u, blocks[b].v);
            if (dep_v < dep_u)
                dep_v.swap(dep_u);
            double* _unique_phi = unique_phi[std::make_tuple(p, dep_v, dep_u, blocks[b].n_segment, 0)];

            for(int i=0; i<M; i++)
                phi[i+b*M] = _unique_phi[i]; 
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
std::vector<double> CpuPseudoContinuous::compute_stress()
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
        std::map<std::tuple<int, std::string, std::string, int, int>, std::array<double,3>> unique_dq_dl;

        // compute stress for unique block
        for(const auto& block: unique_phi)
        {
            const auto& key      = block.first;
            std::string dep_v    = std::get<1>(key);
            std::string dep_u    = std::get<2>(key);
            const int N          = std::get<3>(key);
            std::string monomer_type = mx->get_unique_block(key).monomer_type;

            // if there is no segment
            if(N == 0)
                continue;

            // contains no '['
            int n_repeated;
            if (dep_u.find('[') == std::string::npos)
                n_repeated = mx->get_unique_block(block.first).v_u.size();
            else
                n_repeated = 1;

            std::complex<double> qk_1[M_COMPLEX];
            std::complex<double> qk_2[M_COMPLEX];

            double* q_1 = unique_partition[dep_v];    // dependency v
            double* q_2 = unique_partition[dep_u];    // dependency u

            double coeff;
            std::vector<double> s_coeff = SimpsonQuadrature::get_coeff(N);
            double bond_length_sq = bond_lengths[monomer_type]*bond_lengths[monomer_type];
    
            // reset
            for(int d=0; d<3; d++)
                unique_dq_dl[key][d] = 0.0;

            // compute
            for(int n=0; n<=N; n++)
            {
                fft->forward(&q_1[n*M],     qk_1);
                fft->forward(&q_2[(N-n)*M], qk_2);

                if ( DIM == 3 )
                {
                    for(int i=0; i<M_COMPLEX; i++){
                        coeff = s_coeff[n]*bond_length_sq*(qk_1[i]*std::conj(qk_2[i])).real()*n_repeated;
                        unique_dq_dl[key][0] += coeff*fourier_basis_x[i];
                        unique_dq_dl[key][1] += coeff*fourier_basis_y[i];
                        unique_dq_dl[key][2] += coeff*fourier_basis_z[i];
                    }
                }
                if ( DIM == 2 )
                {
                    for(int i=0; i<M_COMPLEX; i++){
                        coeff = s_coeff[n]*bond_length_sq*(qk_1[i]*std::conj(qk_2[i])).real()*n_repeated;
                        unique_dq_dl[key][0] += coeff*fourier_basis_y[i];
                        unique_dq_dl[key][1] += coeff*fourier_basis_z[i];
                    }
                }
                if ( DIM == 1 )
                {
                    for(int i=0; i<M_COMPLEX; i++){
                        coeff = s_coeff[n]*bond_length_sq*(qk_1[i]*std::conj(qk_2[i])).real()*n_repeated;
                        unique_dq_dl[key][0] += coeff*fourier_basis_z[i];
                    }
                }
            }
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
            std::string monomer_type = mx->get_unique_block(key).monomer_type;
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
void CpuPseudoContinuous::get_partial_partition(double *q_out, int polymer, int v, int u, int n)
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
        if (n < 0 || n > N)
            throw_with_line_number("n (" + std::to_string(n) + ") must be in range [0, " + std::to_string(N) + "]");

        double* partition = unique_partition[dep];
        for(int i=0; i<M; i++)
            q_out[i] = partition[n*M+i];
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
