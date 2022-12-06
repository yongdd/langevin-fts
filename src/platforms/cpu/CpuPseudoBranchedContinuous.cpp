#include <cmath>
#include "CpuPseudoBranchedContinuous.h"
#include "SimpsonQuadrature.h"

CpuPseudoBranchedContinuous::CpuPseudoBranchedContinuous(
    ComputationBox *cb,
    BranchedPolymerChain *bpc,
    PolymerChain *pc, FFT *fft)
    : Pseudo(cb, pc)
{
    try
    {
        const int M = cb->get_n_grid();
        this->bpc = bpc;
        this->fft = fft;

        // create opt_edges, which contains partition function
        for(const auto& item: bpc->get_opt_max_segments()){
            opt_edges[item.first].max_n_segment = item.second;
            opt_edges[item.first].partition     = new double[M*(item.second+1)];
            opt_edges[item.first].species       = bpc->key_to_species(item.first);
            opt_edges[item.first].deps          = bpc->key_to_deps(item.first);
        }

        // create opt_blocks, which contains concentration
        std::vector<polymer_chain_block>& blocks = bpc->get_blocks();
        for(int i=0; i<blocks.size(); i++){
            std::string dep_v = bpc->get_dep(blocks[i].v, blocks[i].u);
            std::string dep_u = bpc->get_dep(blocks[i].u, blocks[i].v);
            if (dep_v > dep_u)
                dep_v.swap(dep_u);
            std::pair<std::string, std::string> key = std::make_pair(dep_v, dep_u);
            opt_blocks[key].n_segment = blocks[i].n_segment;
            opt_blocks[key].species   = blocks[i].species;
        }
        for(const auto& item: opt_blocks){
            //std::cout << "opt_blocks: " << item.first.first << ", " << item.first.second << std::endl;
            opt_blocks[item.first].phi = new double[M];
        }

        // create boltz_bond, boltz_bond_half, exp_dw, and exp_dw_half
        for(const auto& item: bpc->get_dict_bond_lengths()){
            boltz_bond[item.first]      = new double[n_complex_grid];
            boltz_bond_half[item.first] = new double[n_complex_grid]; 
            exp_dw[item.first]      = new double[M];
            exp_dw_half[item.first] = new double[M]; 
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
    for(const auto& item: opt_edges)
        delete[] item.second.partition;
}
void CpuPseudoBranchedContinuous::update()
{
    try
    {
        for(const auto& item: bpc->get_dict_bond_lengths()){
            get_boltz_bond(boltz_bond[item.first],      item.second*item.second,   cb->get_nx(), cb->get_dx(), bpc->get_ds());
            get_boltz_bond(boltz_bond_half[item.first], item.second*item.second/2, cb->get_nx(), cb->get_dx(), bpc->get_ds());
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
        const int N    = pc->get_n_segment_total();
        const int N_B  = pc->get_n_block();
        const std::vector<int> N_SEG    = pc->get_n_segment();
        const int M_COMPLEX = this->n_complex_grid;
        const std::vector<int> seg_start= pc->get_block_start();

        std::array<double,3> dq_dl;
        std::complex<double> k_q_1[M_COMPLEX];
        std::complex<double> k_q_2[M_COMPLEX];

        // //double simpson_rule_coeff_a[N_A+1];
        // //double simpson_rule_coeff_b[N-N_A+1];
        // double simpson_rule_coeff[N];
        // double fourier_basis_x[M_COMPLEX];
        // double fourier_basis_y[M_COMPLEX];
        // double fourier_basis_z[M_COMPLEX];

        // get_weighted_fourier_basis(fourier_basis_x, fourier_basis_y, fourier_basis_z, cb->get_nx(), cb->get_dx());

        // for(int i=0; i<3; i++)
        //     dq_dl[i] = 0.0;
        // for(int b=0; b<N_B; b++)
        // {
        //     SimpsonQuadrature::init_coeff(simpson_rule_coeff, N_SEG[b]);
        //     for(int n=seg_start[b]; n<=seg_start[b+1]; n++)
        //     {
        //         fft->forward(&q_1[n*M],k_q_1);
        //         fft->forward(&q_2[n*M],k_q_2);

        //         if ( DIM >= 3 )
        //         {
        //             for(int i=0; i<M_COMPLEX; i++)
        //                 dq_dl[0] += simpson_rule_coeff[n-seg_start[b]]*pc->get_bond_length_sq(b)*(k_q_1[i]*std::conj(k_q_2[i])).real()*fourier_basis_x[i];
        //         }
        //         if ( DIM >= 2 )
        //         {
        //             for(int i=0; i<M_COMPLEX; i++)
        //                 dq_dl[1] += simpson_rule_coeff[n-seg_start[b]]*pc->get_bond_length_sq(b)*(k_q_1[i]*std::conj(k_q_2[i])).real()*fourier_basis_y[i];
        //         }
        //         if ( DIM >= 1 )
        //         {
        //             for(int i=0; i<M_COMPLEX; i++)
        //                 dq_dl[2] += simpson_rule_coeff[n-seg_start[b]]*pc->get_bond_length_sq(b)*(k_q_1[i]*std::conj(k_q_2[i])).real()*fourier_basis_z[i];
        //         }
        //     }
        // }
        
        // for(int d=0; d<3; d++)
        //     dq_dl[d] /= 3.0*cb->get_lx(d)*M*M/pc->get_ds()/cb->get_volume();

        return dq_dl;
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

void CpuPseudoBranchedContinuous::calculate_phi_one_type(
    double *phi, const int N_START, const int N_END)
{
    try
    {
        // const int M = cb->get_n_grid();
        // double simpson_rule_coeff[N_END-N_START+1];

        // SimpsonQuadrature::init_coeff(simpson_rule_coeff, N_END-N_START);

        // // Compute segment concentration
        // for(int i=0; i<M; i++)
        //     phi[i] = simpson_rule_coeff[0]*q_1[i+N_START*M]*q_2[i+N_START*M];
        // for(int n=N_START+1; n<=N_END; n++)
        // {
        //     for(int i=0; i<M; i++)
        //         phi[i] += simpson_rule_coeff[n-N_START]*q_1[i+n*M]*q_2[i+n*M];
        // }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

void CpuPseudoBranchedContinuous::compute_statistics(double *phi, double *q_1_init, double *q_2_init,
                                 std::map<std::string, double*> w_block, double &single_partition)
{
    try
    {
        const int  M        = cb->get_n_grid();
        //const int  N        = pc->get_n_segment_total();
        //const int  N_B      = pc->get_n_block();
        //const std::vector<int> N_SEG    = pc->get_n_segment();
        const double ds     = pc->get_ds();
        //const std::vector<int> seg_start= pc->get_block_start();

        for(const auto& item: opt_edges){
            if( w_block.count(item.second.species) == 0)
                throw_with_line_number("\"" + item.second.species + "\" species is not in w_block");
        }

        for(const auto& item: w_block){
            for(int i=0; i<M; i++){ 
                exp_dw     [item.first][i] = exp(-item.second[i]*ds*0.5);
                exp_dw_half[item.first][i] = exp(-item.second[i]*ds*0.25);
            }
        }

        for(const auto& item: opt_edges){
            for(int i=0; i<M; i++)
                item.second.partition[i] = 1.0;
            for(int p=0; p<item.second.deps.size(); p++){
                 std::string sub_dep = item.second.deps[p].first;
                 int sub_n_segment   = item.second.deps[p].second;
                 for(int i=0; i<M; i++)
                    item.second.partition[i] *= opt_edges[sub_dep].partition[sub_n_segment*M+i];
            }
            // diffusion of each blocks
            for(int n=1; n<=item.second.max_n_segment; n++){
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

        // segment concentration.
        for(int b=0; b<N_B; b++)
        {
            calculate_phi_one_type(&phi[b*M], seg_start[b], seg_start[b+1]);
        }

        // // calculates the single chain partition function
        // single_partition = cb->inner_product(&q_1[N*M],&q_2[N*M]);

        // // normalize the concentration
        // for(int b=0; b<N_B; b++)
        //     for(int i=0; i<M; i++)
        //         phi[b*M+i] *= cb->get_volume()*pc->get_ds()/single_partition;
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
void CpuPseudoBranchedContinuous::get_partition(double *q_1_out, int n1, double *q_2_out, int n2)
{
    return;
}
void CpuPseudoBranchedContinuous::get_partition(double *q_out, int v, int u, int n)
{
    // This method should be invoked after invoking compute_statistics()

    // Get partial partition functions
    // This is made for debugging and testing
    const int M = cb->get_n_grid();
    std::string dep = bpc->get_dep(v,u);
    const int N = opt_edges[dep].max_n_segment;
    if (n < 0 || n > N)
        throw_with_line_number("n (" + std::to_string(n) + ") must be in range [0, " + std::to_string(N) + "]");

    double* partition = opt_edges[dep].partition;
    for(int i=0; i<M; i++)
        q_out[i] = partition[n*M+i];
}
