#include <cmath>
#include "CpuPseudoBranchedContinuous.h"
#include "SimpsonQuadrature.h"

CpuPseudoBranchedContinuous::CpuPseudoBranchedContinuous(
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
        for(const auto& item: mx->get_reduced_branches())
        {
            std::string dep = item.first;
            int max_n_segment = item.second.max_n_segment;
            reduced_partition[dep] = new double[M*(max_n_segment+1)];
        }

        // allocate memory for concentrations
        for(const auto& item: mx->get_reduced_blocks())
        {
            reduced_phi[item.first] = new double[M];
        }

        // create boltz_bond, boltz_bond_half, exp_dw, and exp_dw_half
        for(const auto& item: mx->get_bond_lengths())
        {
            std::string species = item.first;
            boltz_bond     [species] = new double[M_COMPLEX];
            boltz_bond_half[species] = new double[M_COMPLEX]; 
            exp_dw         [species] = new double[M];
            exp_dw_half    [species] = new double[M]; 
        }

        // allocate memory for stress calculation: dq_dl()
        fourier_basis_x = new double[M_COMPLEX];
        fourier_basis_y = new double[M_COMPLEX];
        fourier_basis_z = new double[M_COMPLEX];

        update();
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
CpuPseudoBranchedContinuous::~CpuPseudoBranchedContinuous()
{
    delete fft;

    delete fourier_basis_x;
    delete fourier_basis_y;
    delete fourier_basis_z;

    for(const auto& item: boltz_bond)
        delete[] item.second;
    for(const auto& item: boltz_bond_half)
        delete[] item.second;
    for(const auto& item: exp_dw)
        delete[] item.second;
    for(const auto& item: exp_dw_half)
        delete[] item.second;
    for(const auto& item: reduced_partition)
        delete[] item.second;
    for(const auto& item: reduced_phi)
        delete[] item.second;
}
void CpuPseudoBranchedContinuous::update()
{
    try
    {
        for(const auto& item: mx->get_bond_lengths())
        {
            std::string species = item.first;
            double bond_length_sq = item.second*item.second;
            get_boltz_bond(boltz_bond     [species], bond_length_sq,   cb->get_nx(), cb->get_dx(), mx->get_ds());
            get_boltz_bond(boltz_bond_half[species], bond_length_sq/2, cb->get_nx(), cb->get_dx(), mx->get_ds());

            // for stress calculation: dq_dl()
            get_weighted_fourier_basis(fourier_basis_x, fourier_basis_y, fourier_basis_z, cb->get_nx(), cb->get_dx());
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
std::vector<double> CpuPseudoBranchedContinuous::compute_statistics(
    std::map<std::string, double*> q_init,
    std::map<std::string, double*> w_block,
    std::vector<double *> phi)
{
    try
    {
        const int M = cb->get_n_grid();
        const double ds = mx->get_ds();

        for(const auto& item: mx->get_reduced_branches())
        {
            if( w_block.count(item.second.species) == 0)
                throw_with_line_number("\"" + item.second.species + "\" species is not in w_block.");
        }

        if( q_init.size() > 0)
            throw_with_line_number("Currently, \'q_init\' is not supported for branched polymers.");

        for(const auto& item: w_block)
        {
            std::string species = item.first;
            double *w = item.second;
            for(int i=0; i<M; i++)
            { 
                exp_dw     [species][i] = exp(-w[i]*ds*0.5);
                exp_dw_half[species][i] = exp(-w[i]*ds*0.25);
            }
        }

        for(const auto& item: mx->get_reduced_branches())
        {
            auto& key = item.first;
            // calculate one block end
            if (item.second.deps.size() > 0) // if it is not leaf node
            {
                for(int i=0; i<M; i++)
                    reduced_partition[key][i] = 1.0;
                for(int p=0; p<item.second.deps.size(); p++)
                {
                    std::string sub_dep = item.second.deps[p].first;
                    int sub_n_segment   = item.second.deps[p].second;
                    for(int i=0; i<M; i++)
                        reduced_partition[key][i] *= reduced_partition[sub_dep][sub_n_segment*M+i];
                }
            }
            else // if it is leaf node
            {
                for(int i=0; i<M; i++)
                    reduced_partition[key][i] = 1.0; //* q_init
            }

            // apply the propagator successively
            for(int n=1; n<=item.second.max_n_segment; n++)
            {
                one_step(&reduced_partition[key][(n-1)*M],
                         &reduced_partition[key][n*M],
                         boltz_bond[item.second.species],
                         boltz_bond_half[item.second.species],
                         exp_dw[item.second.species],
                         exp_dw_half[item.second.species]);
            }
        }

        // calculate segment concentrations
        for(const auto& item: mx->get_reduced_blocks())
        {
            auto& key = item.first;
            calculate_phi_one_type(
                reduced_phi[key],                     // phi
                reduced_partition[std::get<0>(key)],  // dependency v
                reduced_partition[std::get<1>(key)],  // dependency u
                std::get<2>(key));                    // n_segment
        }

        // for each distinct polymers 
        std::vector<double> single_partitions(mx->get_n_distinct_polymers());
        for(int p=0; p<mx->get_n_distinct_polymers(); p++)
        {
            PolymerChain *pc = mx->get_polymer_chain(p);
            std::vector<PolymerChainBlock>& blocks = pc->get_blocks();

            // calculate the single chain partition function at block 0
            std::string dep_v = pc->get_dep(blocks[0].v, blocks[0].u);
            std::string dep_u = pc->get_dep(blocks[0].u, blocks[0].v);
            int n_segment = blocks[0].n_segment;
            single_partitions[p]= cb->inner_product(
                &reduced_partition[dep_v][n_segment*M],  // q
                &reduced_partition[dep_u][0]);           // q^dagger

            // copy phi
            double* phi_p = phi[p];
            for(int b=0; b<blocks.size(); b++)
            {
                std::string dep_v = pc->get_dep(blocks[b].v, blocks[b].u);
                std::string dep_u = pc->get_dep(blocks[b].u, blocks[b].v);
                if (dep_v > dep_u)
                    dep_v.swap(dep_u);
                double* _reduced_phi = reduced_phi[std::make_tuple(dep_v, dep_u, blocks[b].n_segment)];
                // normalize the concentration
                double norm = cb->get_volume()*mx->get_ds()/single_partitions[p];
                for(int i=0; i<M; i++)
                    phi_p[i+b*M] = norm * _reduced_phi[i]; 
            }
        }
        return single_partitions;
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CpuPseudoBranchedContinuous::one_step(double *q_in, double *q_out,
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
void CpuPseudoBranchedContinuous::calculate_phi_one_type(
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
std::vector<std::array<double,3>> CpuPseudoBranchedContinuous::dq_dl()
{
    // This method should be invoked after invoking compute_statistics().

    // To calculate stress, we multiply weighted fourier basis to q(k)*q^dagger(-k).
    // We only need the real part of stress calculation.

    try
    {
        const int DIM  = cb->get_dim();
        const int M    = cb->get_n_grid();
        const int M_COMPLEX = this->n_complex_grid;
        double coeff;

        std::complex<double> qk_1[M_COMPLEX];
        std::complex<double> qk_2[M_COMPLEX];

        std::map<std::string, double>& bond_lengths = mx->get_bond_lengths();
        std::vector<std::array<double,3>> dq_dl(mx->get_n_distinct_polymers());
        std::map<std::tuple<std::string, std::string, int>, std::array<double,3>> reduced_dq_dl;

        // compute stress for reduced key pairs
        for(const auto& item: mx->get_reduced_blocks())
        {
            auto& key = item.first;
            std::string dep_v = std::get<0>(key);
            std::string dep_u = std::get<1>(key);
            const int N       = std::get<2>(key);
            std::string species = item.second.species;

            std::vector<double> s_coeff = SimpsonQuadrature::get_coeff(N);
            double bond_length_sq = bond_lengths[species]*bond_lengths[species];
            double* q_1 = reduced_partition[dep_v];    // dependency v
            double* q_2 = reduced_partition[dep_u];    // dependency u

            // reset
            for(int d=0; d<3; d++)
                reduced_dq_dl[key][d] = 0.0;

            // compute
            for(int n=0; n<=N; n++)
            {
                fft->forward(&q_1[n*M],     qk_1);
                fft->forward(&q_2[(N-n)*M], qk_2);

                if ( DIM == 3 )
                {
                    for(int i=0; i<M_COMPLEX; i++){
                        coeff = s_coeff[n]*bond_length_sq*(qk_1[i]*std::conj(qk_2[i])).real();
                        reduced_dq_dl[key][0] += coeff*fourier_basis_x[i];
                        reduced_dq_dl[key][1] += coeff*fourier_basis_y[i];
                        reduced_dq_dl[key][2] += coeff*fourier_basis_z[i];
                    }
                }
                if ( DIM == 2 )
                {
                    for(int i=0; i<M_COMPLEX; i++){
                        coeff = s_coeff[n]*bond_length_sq*(qk_1[i]*std::conj(qk_2[i])).real();
                        reduced_dq_dl[key][1] += coeff*fourier_basis_y[i];
                        reduced_dq_dl[key][2] += coeff*fourier_basis_y[i];
                    }
                }
                if ( DIM == 1 )
                {
                    for(int i=0; i<M_COMPLEX; i++){
                        coeff = s_coeff[n]*bond_length_sq*(qk_1[i]*std::conj(qk_2[i])).real();
                        reduced_dq_dl[key][2] += coeff*fourier_basis_z[i];
                    }
                }
            }
        }

        // compute total stress for each distinct polymers 
        for(int p=0; p < mx->get_n_distinct_polymers(); p++)
        {
            for(int d=0; d<3; d++)
                dq_dl[p][d] = 0.0;
            PolymerChain *pc = mx->get_polymer_chain(p);
            std::vector<PolymerChainBlock>& blocks = pc->get_blocks();
            for(int b=0; b<blocks.size(); b++)
            {
                std::string dep_v = pc->get_dep(blocks[b].v, blocks[b].u);
                std::string dep_u = pc->get_dep(blocks[b].u, blocks[b].v);
                if (dep_v > dep_u)
                    dep_v.swap(dep_u);
                for(int d=0; d<3; d++)
                    dq_dl[p][d] += reduced_dq_dl[std::make_tuple(dep_v, dep_u, blocks[b].n_segment)][d];
            }
            for(int d=0; d<3; d++)
                dq_dl[p][d] /= 3.0*cb->get_lx(d)*M*M/mx->get_ds()/cb->get_volume();
        }

        return dq_dl;
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CpuPseudoBranchedContinuous::get_partition(double *q_out, int polymer, int v, int u, int n)
{
    // This method should be invoked after invoking compute_statistics()

    // Get partial partition functions
    // This is made for debugging and testing
    try
    {
        const int M = cb->get_n_grid();
        PolymerChain *pc = mx->get_polymer_chain(polymer);
        std::string dep = pc->get_dep(v,u);
        const int N = mx->get_reduced_branches()[dep].max_n_segment;
        if (n < 0 || n > N)
            throw_with_line_number("n (" + std::to_string(n) + ") must be in range [0, " + std::to_string(N) + "]");

        double* partition = reduced_partition[dep];
        for(int i=0; i<M; i++)
            q_out[i] = partition[n*M+i];
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
