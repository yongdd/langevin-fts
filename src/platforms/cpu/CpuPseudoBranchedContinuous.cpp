#include <cmath>
#include "CpuPseudoBranchedContinuous.h"
#include "SimpsonQuadrature.h"

CpuPseudoBranchedContinuous::CpuPseudoBranchedContinuous(
    ComputationBox *cb,
    PolymerChain *pc,
    FFT *fft)
    : Pseudo(cb, pc)
{
    try
    {
        const int M = cb->get_n_grid();
        this->fft = fft;

        // create reduced_edges, which contains partition function
        for(const auto& item: pc->get_reduced_branches_max_segment()){
            std::string dep = item.first;
            reduced_edges[dep].max_n_segment = item.second;
            reduced_edges[dep].partition     = new double[M*(item.second+1)];
            reduced_edges[dep].species       = pc->key_to_species(dep);
            reduced_edges[dep].deps          = pc->key_to_deps(dep);
        }

        // create reduced_blocks, which contains concentration
        std::vector<polymer_chain_block>& blocks = pc->get_blocks();
        for(int i=0; i<blocks.size(); i++){
            std::string dep_v = pc->get_dep(blocks[i].v, blocks[i].u);
            std::string dep_u = pc->get_dep(blocks[i].u, blocks[i].v);
            if (dep_v > dep_u)
                dep_v.swap(dep_u);
            std::pair<std::string, std::string> key = std::make_pair(dep_v, dep_u);
            reduced_blocks[key].n_segment = blocks[i].n_segment;
            reduced_blocks[key].species   = blocks[i].species;
        }
        for(const auto& item: reduced_blocks){
            //std::cout << "reduced_blocks: " << item.first.first << ", " << item.first.second << std::endl;
            reduced_blocks[item.first].phi = new double[M];
        }

        // create boltz_bond, boltz_bond_half, exp_dw, and exp_dw_half
        for(const auto& item: pc->get_dict_bond_lengths()){
            std::string species = item.first;
            boltz_bond     [species] = new double[n_complex_grid];
            boltz_bond_half[species] = new double[n_complex_grid]; 
            exp_dw         [species] = new double[M];
            exp_dw_half    [species] = new double[M]; 
        }
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
    for(const auto& item: boltz_bond)
        delete[] item.second;
    for(const auto& item: boltz_bond_half)
        delete[] item.second;
    for(const auto& item: exp_dw)
        delete[] item.second;
    for(const auto& item: exp_dw_half)
        delete[] item.second;
    for(const auto& item: reduced_edges)
        delete[] item.second.partition;
    for(const auto& item: reduced_blocks)
        delete[] item.second.phi;
}
void CpuPseudoBranchedContinuous::update()
{
    try
    {
        for(const auto& item: pc->get_dict_bond_lengths()){
            std::string species = item.first;
            double bond_length_sq = item.second*item.second;
            get_boltz_bond(boltz_bond     [species], bond_length_sq,   cb->get_nx(), cb->get_dx(), pc->get_ds());
            get_boltz_bond(boltz_bond_half[species], bond_length_sq/2, cb->get_nx(), cb->get_dx(), pc->get_ds());
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CpuPseudoBranchedContinuous::compute_statistics(
    std::map<std::string, double*> q_init,
    std::map<std::string, double*> w_block,
    double *phi, double &single_partition)
{
    try
    {
        const int M = cb->get_n_grid();
        const double ds = pc->get_ds();

        for(const auto& item: reduced_edges)
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

        for(const auto& item: reduced_edges)
        {
            // calculate one block end
            if (item.second.deps.size() > 0) // if it is not leaf node
            {
                for(int i=0; i<M; i++)
                    item.second.partition[i] = 1.0;
                for(int p=0; p<item.second.deps.size(); p++)
                {
                    std::string sub_dep = item.second.deps[p].first;
                    int sub_n_segment   = item.second.deps[p].second;
                    for(int i=0; i<M; i++)
                        item.second.partition[i] *= reduced_edges[sub_dep].partition[sub_n_segment*M+i];
                }
            }
            else // if it is leaf node
            {
                for(int i=0; i<M; i++)
                    item.second.partition[i] = 1.0; //* q_init
            }

            // apply the propagator successively
            for(int n=1; n<=item.second.max_n_segment; n++)
            {
                one_step(&item.second.partition[(n-1)*M],
                         &item.second.partition[n*M],
                         boltz_bond[item.second.species],
                         boltz_bond_half[item.second.species],
                         exp_dw[item.second.species],
                         exp_dw_half[item.second.species]);
                // std::cout << "n: " << n << std::endl;    
                // for(int i=0; i<M; i++)
                //     std::cout << item.second.partition[(n-1)*M+i] << ", ";
            }
        }

        // calculates the single chain partition function
        std::string dep_v = reduced_blocks.begin()->first.first;
        std::string dep_u = reduced_blocks.begin()->first.second;
        int n_segment = reduced_blocks.begin()->second.n_segment;
        single_partition = cb->inner_product(
            &reduced_edges[dep_v].partition[n_segment*M],  // q
            &reduced_edges[dep_u].partition[0]);           // q^dagger

        // segment concentration.
        for(const auto& item: reduced_blocks)
        {
            calculate_phi_one_type(
                item.second.phi,                          // phi
                reduced_edges[item.first.first].partition,    // dependency v
                reduced_edges[item.first.second].partition,   // dependency u
                item.second.n_segment);                   // n_segment
        }

        // normalize the concentration
        for(const auto& item: reduced_blocks)
        {
            for(int i=0; i<M; i++)
                item.second.phi[i] *= cb->get_volume()*pc->get_ds()/single_partition;
        }

        // copy phi
        std::vector<polymer_chain_block>& blocks = pc->get_blocks();
        for(int n=0; n<blocks.size(); n++)
        {
            std::string dep_v = pc->get_dep(blocks[n].v, blocks[n].u);
            std::string dep_u = pc->get_dep(blocks[n].u, blocks[n].v);
            if (dep_v > dep_u)
                dep_v.swap(dep_u);
            for(int i=0; i<M; i++)
                phi[i+n*M] = reduced_blocks[std::make_pair(dep_v, dep_u)].phi[i];
        }
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
std::array<double,3> CpuPseudoBranchedContinuous::dq_dl()
{
    // This method should be invoked after invoking compute_statistics().

    // To calculate stress, we multiply weighted fourier basis to q(k)*q^dagger(-k).
    // We only need the real part of stress calculation.

    try
    {
        const int DIM  = cb->get_dim();
        const int M    = cb->get_n_grid();
        const int M_COMPLEX = this->n_complex_grid;

        std::array<double,3> dq_dl;
        std::complex<double> k_q_1[M_COMPLEX];
        std::complex<double> k_q_2[M_COMPLEX];

        double fourier_basis_x[M_COMPLEX];
        double fourier_basis_y[M_COMPLEX];
        double fourier_basis_z[M_COMPLEX];

        get_weighted_fourier_basis(fourier_basis_x, fourier_basis_y, fourier_basis_z, cb->get_nx(), cb->get_dx());
        std::map<std::string, double>& dict_bond_lengths = pc->get_dict_bond_lengths();
        std::map<std::pair<std::string, std::string>, std::array<double,3>> reduced_dq_dl;

        // compute stress for optimal key pairs
        for(const auto& item: reduced_blocks)
        {
            const int N = item.second.n_segment;
            auto key = item.first;
            std::string dep_v = key.first;
            std::string dep_u = key.second; 
            std::string species = item.second.species;

            std::vector<double> simpson_rule_coeff = SimpsonQuadrature::get_coeff(N);
            double bond_length_sq = dict_bond_lengths[species]*dict_bond_lengths[species];
            double* q_1 = reduced_edges[dep_v].partition;    // dependency v
            double* q_2 = reduced_edges[dep_u].partition;   // dependency u

            // reset
            for(int d=0; d<3; d++)
                reduced_dq_dl[key][d] = 0.0;

            // compute
            for(int n=0; n<=N; n++)
            {
                double coeff;

                fft->forward(&q_1[n*M],    k_q_1);
                fft->forward(&q_2[(N-n)*M],k_q_2);

                if ( DIM == 3 )
                {
                    for(int i=0; i<M_COMPLEX; i++){
                        coeff = simpson_rule_coeff[n]*bond_length_sq*(k_q_1[i]*std::conj(k_q_2[i])).real();
                        reduced_dq_dl[key][0] += coeff*fourier_basis_x[i];
                        reduced_dq_dl[key][1] += coeff*fourier_basis_y[i];
                        reduced_dq_dl[key][2] += coeff*fourier_basis_z[i];
                    }
                }
                if ( DIM == 2 )
                {
                    for(int i=0; i<M_COMPLEX; i++){
                        coeff = simpson_rule_coeff[n]*bond_length_sq*(k_q_1[i]*std::conj(k_q_2[i])).real();
                        reduced_dq_dl[key][1] += coeff*fourier_basis_y[i];
                        reduced_dq_dl[key][2] += coeff*fourier_basis_y[i];
                    }
                }
                if ( DIM == 1 )
                {
                    for(int i=0; i<M_COMPLEX; i++){
                        coeff = simpson_rule_coeff[n]*bond_length_sq*(k_q_1[i]*std::conj(k_q_2[i])).real();
                        reduced_dq_dl[key][2] += coeff*fourier_basis_z[i];
                    }
                }
            }
        }

        // compute total stress
        std::vector<polymer_chain_block>& blocks = pc->get_blocks();
        for(int d=0; d<3; d++)
            dq_dl[d] = 0.0;
        for(int n=0; n<blocks.size(); n++)
        {
            std::string dep_v = pc->get_dep(blocks[n].v, blocks[n].u);
            std::string dep_u = pc->get_dep(blocks[n].u, blocks[n].v);
            if (dep_v > dep_u)
                dep_v.swap(dep_u);
            for(int d=0; d<3; d++)
                dq_dl[d] += reduced_dq_dl[std::make_pair(dep_v, dep_u)][d];
        }
        for(int d=0; d<3; d++)
            dq_dl[d] /= 3.0*cb->get_lx(d)*M*M/pc->get_ds()/cb->get_volume();

        return dq_dl;
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CpuPseudoBranchedContinuous::get_partition(double *q_out, int v, int u, int n)
{
    // This method should be invoked after invoking compute_statistics()

    // Get partial partition functions
    // This is made for debugging and testing
    try
    {
        const int M = cb->get_n_grid();
        std::string dep = pc->get_dep(v,u);
        const int N = reduced_edges[dep].max_n_segment;
        if (n < 0 || n > N)
            throw_with_line_number("n (" + std::to_string(n) + ") must be in range [0, " + std::to_string(N) + "]");

        double* partition = reduced_edges[dep].partition;
        for(int i=0; i<M; i++)
            q_out[i] = partition[n*M+i];
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
