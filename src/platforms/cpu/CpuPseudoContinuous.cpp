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

        update();
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
void CpuPseudoContinuous::update()
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
    std::map<std::string, double*> w_block)
{
    try
    {
        const int M = cb->get_n_grid();
        const double ds = mx->get_ds();

        for(const auto& item: mx->get_unique_branches())
        {
            if( w_block.count(item.second.monomer_type) == 0)
                throw_with_line_number("\"" + item.second.monomer_type + "\" monomer_type is not in w_block.");
        }

        if( q_init.size() > 0)
            throw_with_line_number("Currently, \'q_init\' is not supported.");

        for(const auto& item: w_block)
        {
            std::string monomer_type = item.first;
            double *w = item.second;
            for(int i=0; i<M; i++)
            { 
                exp_dw     [monomer_type][i] = exp(-w[i]*ds*0.5);
                exp_dw_half[monomer_type][i] = exp(-w[i]*ds*0.25);
            }
        }

        // for each time span
        auto& branch_schedule = sc->get_schedule();
        for (auto parallel_job = branch_schedule.begin(); parallel_job != branch_schedule.end(); parallel_job++)
        {
            // for each job
            #pragma omp parallel for
            for(int job=0; job<parallel_job->size(); job++)
            {
                auto& key = std::get<0>((*parallel_job)[job]);
                int n_segment_from = std::get<1>((*parallel_job)[job]);
                int n_segment_to = std::get<2>((*parallel_job)[job]);
                auto& deps = mx->get_unique_branch(key).deps;
                auto monomer_type = mx->get_unique_branch(key).monomer_type;

                // calculate one block end
                if(n_segment_from == 1 && deps.size() == 0) // if it is leaf node
                {
                    for(int i=0; i<M; i++)
                        unique_partition[key][i] = 1.0; //* q_init
                    unique_partition_finished[key][0] = true;
                }
                else if (n_segment_from == 1 && deps.size() > 0) // if it is not leaf node
                {
                    for(int i=0; i<M; i++)
                        unique_partition[key][i] = 1.0;
                    for(int p=0; p<deps.size(); p++)
                    {
                        std::string sub_dep = deps[p].first;
                        int sub_n_segment   = deps[p].second;

                        if (!unique_partition_finished[sub_dep][sub_n_segment])
                            std::cout << "unfinished, sub_dep: " << sub_dep << ", " << sub_n_segment << std::endl;

                        for(int i=0; i<M; i++)
                            unique_partition[key][i] *= unique_partition[sub_dep][sub_n_segment*M+i];
                        unique_partition_finished[key][0] = true;
                    }
                }
        
                // apply the propagator successively
                for(int n=n_segment_from; n<=n_segment_to; n++)
                {
                    if (!unique_partition_finished[key][n-1])
                        std::cout << "unfinished, key: " << key << ", " << n << std::endl;

                    one_step(&unique_partition[key][(n-1)*M],
                            &unique_partition[key][n*M],
                            boltz_bond[monomer_type],
                            boltz_bond_half[monomer_type],
                            exp_dw[monomer_type],
                            exp_dw_half[monomer_type]);
                    unique_partition_finished[key][n] = true;
                }
            }
        }

        // calculate segment concentrations
        // for(const auto& item: mx->get_unique_blocks())
        // {
        #pragma omp parallel for
        for(int b=0; b<mx->get_unique_blocks().size();b++)
        {
            auto item = mx->get_unique_blocks().begin();
            advance(item, b);

            auto& key = item->first;
            calculate_phi_one_type(
                unique_phi[key],                     // phi
                unique_partition[std::get<0>(key)],  // dependency v
                unique_partition[std::get<1>(key)],  // dependency u
                std::get<2>(key));                    // n_segment
        }

        // for each distinct polymers
        #pragma omp parallel for
        for(int p=0; p<mx->get_n_polymers(); p++)
        {
            PolymerChain& pc = mx->get_polymer(p);
            std::vector<PolymerChainBlock>& blocks = pc.get_blocks();

            // calculate the single chain partition function at block 0
            std::string dep_v = pc.get_dep(blocks[0].v, blocks[0].u);
            std::string dep_u = pc.get_dep(blocks[0].u, blocks[0].v);
            int n_segment = blocks[0].n_segment;
            single_partitions[p]= cb->inner_product(
                &unique_partition[dep_v][n_segment*M],  // q
                &unique_partition[dep_u][0]);           // q^dagger
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
void CpuPseudoContinuous::calculate_phi_one_type(
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

        // for each distinct polymers 
        for(int p=0; p<mx->get_n_polymers(); p++)
        {
            PolymerChain& pc = mx->get_polymer(p);
            std::vector<PolymerChainBlock>& blocks = pc.get_blocks();

            for(int b=0; b<blocks.size(); b++)
            {
                if (blocks[b].monomer_type == monomer_type)
                {
                    std::string dep_v = pc.get_dep(blocks[b].v, blocks[b].u);
                    std::string dep_u = pc.get_dep(blocks[b].u, blocks[b].v);
                    if (dep_v > dep_u)
                        dep_v.swap(dep_u);
                    double* _unique_phi = unique_phi[std::make_tuple(dep_v, dep_u, blocks[b].n_segment)];

                    // normalize the concentration
                    double norm = cb->get_volume()*mx->get_ds()*pc.get_volume_fraction()/pc.get_alpha()/single_partitions[p];
                    for(int i=0; i<M; i++)
                        phi[i] += norm * _unique_phi[i]; 
                }
            }
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CpuPseudoContinuous::get_polymer_concentration(int polymer, double *phi)
{
    try
    {
        const int M = cb->get_n_grid();
        const int P = mx->get_n_polymers();

        if (polymer < 0 || polymer > P-1)
            throw_with_line_number("Index (" + std::to_string(polymer) + ") must be in range [0, " + std::to_string(P-1) + "]");

        PolymerChain& pc = mx->get_polymer(polymer);
        std::vector<PolymerChainBlock>& blocks = pc.get_blocks();

        for(int b=0; b<blocks.size(); b++)
        {
            std::string dep_v = pc.get_dep(blocks[b].v, blocks[b].u);
            std::string dep_u = pc.get_dep(blocks[b].u, blocks[b].v);
            if (dep_v > dep_u)
                dep_v.swap(dep_u);
            double* _unique_phi = unique_phi[std::make_tuple(dep_v, dep_u, blocks[b].n_segment)];

            // normalize the concentration
            double norm = cb->get_volume()*mx->get_ds()*pc.get_volume_fraction()/pc.get_alpha()/single_partitions[polymer];
            for(int i=0; i<M; i++)
                phi[i+b*M] = norm * _unique_phi[i]; 
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
std::array<double,3> CpuPseudoContinuous::compute_stress()
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
        std::array<double,3> stress;
        std::map<std::tuple<std::string, std::string, int>, std::array<double,3>> unique_dq_dl;

        // compute stress for Unique key pairs
        // for(const auto& item: mx->get_unique_blocks())
        // {
        //     auto& key = item.first;
        //     std::string dep_v = std::get<0>(key);
        //     std::string dep_u = std::get<1>(key);
        //     const int N       = std::get<2>(key);
        //     std::string monomer_type = item.second.monomer_type;
        // #pragma omp parallel for
        for(int b=0; b<mx->get_unique_blocks().size();b++)
        {
            auto item = mx->get_unique_blocks().begin();
            advance(item, b);

            auto& key = item->first;
            std::string dep_v = std::get<0>(key);
            std::string dep_u = std::get<1>(key);
            const int N       = std::get<2>(key);
            std::string monomer_type = item->second.monomer_type;

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
                        coeff = s_coeff[n]*bond_length_sq*(qk_1[i]*std::conj(qk_2[i])).real();
                        unique_dq_dl[key][0] += coeff*fourier_basis_x[i];
                        unique_dq_dl[key][1] += coeff*fourier_basis_y[i];
                        unique_dq_dl[key][2] += coeff*fourier_basis_z[i];
                    }
                }
                if ( DIM == 2 )
                {
                    for(int i=0; i<M_COMPLEX; i++){
                        coeff = s_coeff[n]*bond_length_sq*(qk_1[i]*std::conj(qk_2[i])).real();
                        unique_dq_dl[key][1] += coeff*fourier_basis_y[i];
                        unique_dq_dl[key][2] += coeff*fourier_basis_z[i];
                    }
                }
                if ( DIM == 1 )
                {
                    for(int i=0; i<M_COMPLEX; i++){
                        coeff = s_coeff[n]*bond_length_sq*(qk_1[i]*std::conj(qk_2[i])).real();
                        unique_dq_dl[key][2] += coeff*fourier_basis_z[i];
                    }
                }
            }
        }
        // compute total stress
        for(int d=0; d<3; d++)
            stress[d] = 0.0;
        for(int p=0; p < mx->get_n_polymers(); p++)
        {
            PolymerChain& pc = mx->get_polymer(p);
            std::vector<PolymerChainBlock>& blocks = pc.get_blocks();
            for(int b=0; b<blocks.size(); b++)
            {
                std::string dep_v = pc.get_dep(blocks[b].v, blocks[b].u);
                std::string dep_u = pc.get_dep(blocks[b].u, blocks[b].v);
                if (dep_v > dep_u)
                    dep_v.swap(dep_u);
                for(int d=0; d<3; d++)
                    stress[d] += unique_dq_dl[std::make_tuple(dep_v, dep_u, blocks[b].n_segment)][d]*pc.get_volume_fraction()/pc.get_alpha()/single_partitions[p];
            }
        }
        for(int d=0; d<3; d++)
            stress[d] /= -3.0*cb->get_lx(d)*M*M/mx->get_ds()/cb->get_volume();

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
