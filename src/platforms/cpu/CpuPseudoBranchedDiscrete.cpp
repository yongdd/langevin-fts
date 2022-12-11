#include <cmath>
#include "CpuPseudoBranchedDiscrete.h"
#include "SimpsonQuadrature.h"

CpuPseudoBranchedDiscrete::CpuPseudoBranchedDiscrete(
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
            reduced_partition[dep] = new double[M*max_n_segment];

             // There are N segments
             // Illustration (N==5)
             // O--O--O--O--O
             // 0  1  2  3  4 

             // Legend)
             // -- : full bond
             // O  : full segment
        }

        // allocate memory for reduced_q_junctions, which contain partition function at junction of discrete chain
        for(const auto& item: mx->get_reduced_branches())
        {
            reduced_q_junctions[item.first] = new double[M];
        }

        // allocate memory for concentrations
        for(const auto& item: mx->get_reduced_blocks())
        {
            reduced_phi[item.first] = new double[M];
        }

        // create boltz_bond, boltz_bond_half, and exp_dw
        for(const auto& item: mx->get_bond_lengths())
        {
            std::string species = item.first;
            boltz_bond     [species] = new double[M_COMPLEX];
            boltz_bond_half[species] = new double[M_COMPLEX]; 
            exp_dw         [species] = new double[M];
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
CpuPseudoBranchedDiscrete::~CpuPseudoBranchedDiscrete()
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
    for(const auto& item: reduced_partition)
        delete[] item.second;
    for(const auto& item: reduced_phi)
        delete[] item.second;
    for(const auto& item: reduced_q_junctions)
        delete[] item.second;
}
void CpuPseudoBranchedDiscrete::update()
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
std::vector<double> CpuPseudoBranchedDiscrete::compute_statistics(
    std::map<std::string, double*> q_init,
    std::map<std::string, double*> w_block,
    std::vector<double *> phi)
{
    try
    {
        const int M = cb->get_n_grid();
        const int M_COMPLEX = this->n_complex_grid;
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
                exp_dw[species][i] = exp(-w[i]*ds);
        }

        for(const auto& item: mx->get_reduced_branches())
        {
            auto& key = item.first;
            // calculate one block end
            if (item.second.deps.size() > 0) // if it is not leaf node
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
                for(int p=0; p<item.second.deps.size(); p++)
                {
                    std::string sub_dep = item.second.deps[p].first;
                    int sub_n_segment   = item.second.deps[p].second;
                    double q_half_step[M];

                    half_bond_step(&reduced_partition[sub_dep][(sub_n_segment-1)*M],
                        q_half_step, boltz_bond_half[mx->get_reduced_branch(sub_dep).species]);

                    for(int i=0; i<M; i++)
                        q_junction[i] *= q_half_step[i];
                }
                // std::cout << "key (added): " << item.first << std::endl;
                for(int i=0; i<M; i++)
                    reduced_q_junctions[item.first][i] = q_junction[i];

                // add half bond
                half_bond_step(q_junction, &reduced_partition[key][0], boltz_bond_half[item.second.species]);

                // add full segment
                for(int i=0; i<M; i++)
                    reduced_partition[key][i] *= exp_dw[item.second.species][i];
            }
            else  // if it is leaf node
            {
                for(int i=0; i<M; i++)
                    reduced_partition[key][i] = exp_dw[item.second.species][i]; //* q_init
            }

            // diffusion of each blocks
            for(int n=1; n<item.second.max_n_segment; n++)
            {
                one_step(&reduced_partition[key][(n-1)*M],
                         &reduced_partition[key][n*M],
                         boltz_bond[item.second.species],
                         exp_dw[item.second.species]);
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
                exp_dw[item.second.species],          // exp_dw
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
            single_partitions[p]= cb->inner_product_inverse_weight(
                &reduced_partition[dep_v][(n_segment-1)*M],  // q
                &reduced_partition[dep_u][0],                // q^dagger
                exp_dw[blocks[0].species]);        

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
void CpuPseudoBranchedDiscrete::one_step(double *q_in, double *q_out,
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

void CpuPseudoBranchedDiscrete::half_bond_step(double *q_in, double *q_out, double *boltz_bond_half)
{
    try
    {
        const int M = cb->get_n_grid();
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
void CpuPseudoBranchedDiscrete::calculate_phi_one_type(
    double *phi, double *q_1, double *q_2, double *exp_dw, const int N)
{
    try
    {
        const int M = cb->get_n_grid();
        // Compute segment concentration
        for(int i=0; i<M; i++)
            phi[i] = q_1[i]*q_2[i+(N-1)*M];
        for(int n=1; n<N; n++)
        {
            for(int i=0; i<M; i++)
                phi[i] += q_1[i+n*M]*q_2[i+(N-n-1)*M];
        }
        for(int i=0; i<M; i++)
            phi[i] /= exp_dw[i];
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
std::vector<std::array<double,3>> CpuPseudoBranchedDiscrete::dq_dl()
{
    // This method should be invoked after invoking compute_statistics().

    // To calculate stress, we multiply weighted fourier basis to q(k)*q^dagger(-k).
    // We only need the real part of stress calculation.

    try
    {
        const int DIM  = cb->get_dim();
        const int M    = cb->get_n_grid();
        const int M_COMPLEX = this->n_complex_grid;

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

            double* q_1 = reduced_partition[dep_v];    // dependency v
            double* q_2 = reduced_partition[dep_u];    // dependency u

            double coeff;
            double bond_length_sq;
            double * boltz_bond_now;

            // reset
            for(int d=0; d<3; d++)
                reduced_dq_dl[key][d] = 0.0;

            // std::cout << "dep_v: " << dep_v << std::endl;
            // std::cout << "dep_u: " << dep_u << std::endl;
            
            // compute stress
            for(int n=0; n<=N; n++)
            {
                // at v
                if (n == 0){
                    if (mx->get_reduced_branch(dep_v).deps.size() == 0) // if v is leaf node, skip
                        continue; 
                    fft->forward(reduced_q_junctions[dep_v], qk_1);
                    fft->forward(&q_2[(N-1)*M], qk_2);
                    bond_length_sq = 0.5*bond_lengths[species]*bond_lengths[species];
                    boltz_bond_now = boltz_bond_half[species];
                }
                // at u  
                else if (n == N)
                {
                    if (mx->get_reduced_branch(dep_u).deps.size() == 0) // if u is leaf node, skip
                        continue; 
                    fft->forward(&q_1[(N-1)*M], qk_1);
                    fft->forward(reduced_q_junctions[dep_u], qk_2);
                    bond_length_sq = 0.5*bond_lengths[species]*bond_lengths[species];
                    boltz_bond_now = boltz_bond_half[species];
                }
                // within the blocks
                else
                {
                    fft->forward(&q_1[(n-1)*M], qk_1);
                    fft->forward(&q_2[(N-n-1)*M], qk_2);
                    bond_length_sq = bond_lengths[species]*bond_lengths[species];
                    boltz_bond_now = boltz_bond[species];
                }
                // compute 
                if ( DIM == 3 )
                {
                    for(int i=0; i<M_COMPLEX; i++)
                    {
                        coeff = bond_length_sq*boltz_bond_now[i]*(qk_1[i]*std::conj(qk_2[i])).real();
                        reduced_dq_dl[key][0] += coeff*fourier_basis_x[i];
                        reduced_dq_dl[key][1] += coeff*fourier_basis_y[i];
                        reduced_dq_dl[key][2] += coeff*fourier_basis_z[i];
                    }
                }
                else if ( DIM == 2 )
                {
                    for(int i=0; i<M_COMPLEX; i++)
                    {
                        coeff = bond_length_sq*boltz_bond_now[i]*(qk_1[i]*std::conj(qk_2[i])).real();
                        reduced_dq_dl[key][1] += coeff*fourier_basis_y[i];
                        reduced_dq_dl[key][2] += coeff*fourier_basis_z[i];
                    }
                }
                else if ( DIM == 1 )
                {
                    for(int i=0; i<M_COMPLEX; i++)
                    {
                        coeff = bond_length_sq*boltz_bond_now[i]*(qk_1[i]*std::conj(qk_2[i])).real();
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
void CpuPseudoBranchedDiscrete::get_partition(double *q_out, int polymer, int v, int u, int n)
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
        if (n < 1 || n > N)
            throw_with_line_number("n (" + std::to_string(n) + ") must be in range [1, " + std::to_string(N) + "]");

        double* partition = reduced_partition[dep];
        for(int i=0; i<M; i++)
            q_out[i] = partition[(n-1)*M+i];
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
