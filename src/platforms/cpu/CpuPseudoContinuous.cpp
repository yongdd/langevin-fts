#include <cmath>
#include "CpuPseudoContinuous.h"
#include "SimpsonQuadrature.h"

CpuPseudoContinuous::CpuPseudoContinuous(
    ComputationBox *cb,
    PolymerChain *pc, FFT *fft)
    : Pseudo(cb, pc)
{
    try
    {
        const int M = cb->get_n_grid();
        const int N = pc->get_n_segment_total();
        this->fft = fft;

        // create boltz_bond, boltz_bond_half, exp_dw, and exp_dw_half
        for(const auto& item: pc->get_dict_bond_lengths()){
            std::string species = item.first;
            boltz_bond     [species] = new double[n_complex_grid];
            boltz_bond_half[species] = new double[n_complex_grid]; 
            exp_dw         [species] = new double[M];
            exp_dw_half    [species] = new double[M]; 
        }

        // partial partition functions, q and q^dagger
        this->q_1 = new double[M*(N+1)];
        this->q_2 = new double[M*(N+1)];

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
    for(const auto& item: boltz_bond)
        delete[] item.second;
    for(const auto& item: boltz_bond_half)
        delete[] item.second;
    for(const auto& item: exp_dw)
        delete[] item.second;
    for(const auto& item: exp_dw_half)
        delete[] item.second;
    delete[] q_1, q_2;
}
void CpuPseudoContinuous::update()
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
std::vector<int> CpuPseudoContinuous::get_block_start()
{
    std::vector<int> seg_start;
    seg_start.push_back(0);
    int seg_start_temp = 0;
    for(int i=0; i<pc->get_n_block(); i++){
        seg_start_temp += pc->get_n_segment(i);
        seg_start.push_back(seg_start_temp);
    }
    return seg_start;
}
std::array<double,3> CpuPseudoContinuous::dq_dl()
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
        const int M_COMPLEX = this->n_complex_grid;

        std::array<double,3> dq_dl;
        std::complex<double> k_q_1[M_COMPLEX];
        std::complex<double> k_q_2[M_COMPLEX];

        double fourier_basis_x[M_COMPLEX];
        double fourier_basis_y[M_COMPLEX];
        double fourier_basis_z[M_COMPLEX];

        get_weighted_fourier_basis(fourier_basis_x, fourier_basis_y, fourier_basis_z, cb->get_nx(), cb->get_dx());
        std::map<std::string, double>& dict_bond_lengths = pc->get_dict_bond_lengths();
        auto seg_start = get_block_start();

        for(int i=0; i<3; i++)
            dq_dl[i] = 0.0;

        for(int b=0; b<N_B; b++)
        {
            std::vector<double> simpson_rule_coeff = SimpsonQuadrature::get_coeff(pc->get_n_segment(b));
            std::string species = pc->get_block_species(b);
            double bond_length_sq = dict_bond_lengths[species]*dict_bond_lengths[species];

            for(int n=seg_start[b]; n<=seg_start[b+1]; n++)
            {
                double s_coeff = simpson_rule_coeff[n-seg_start[b]];

                fft->forward(&q_1[n*M],k_q_1);
                fft->forward(&q_2[n*M],k_q_2);
                
                if ( DIM >= 3 )
                {
                    for(int i=0; i<M_COMPLEX; i++)
                        dq_dl[0] += s_coeff*bond_length_sq*(k_q_1[i]*std::conj(k_q_2[i])).real()*fourier_basis_x[i];
                }
                if ( DIM >= 2 )
                {
                    for(int i=0; i<M_COMPLEX; i++)
                        dq_dl[1] += s_coeff*bond_length_sq*(k_q_1[i]*std::conj(k_q_2[i])).real()*fourier_basis_y[i];
                }
                if ( DIM >= 1 )
                {
                    for(int i=0; i<M_COMPLEX; i++)
                        dq_dl[2] += s_coeff*bond_length_sq*(k_q_1[i]*std::conj(k_q_2[i])).real()*fourier_basis_z[i];
                }
            }
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

void CpuPseudoContinuous::calculate_phi_one_type(
    double *phi, const int N_START, const int N_END)
{
    try
    {
        const int M = cb->get_n_grid();
        auto simpson_rule_coeff = SimpsonQuadrature::get_coeff(N_END-N_START);

        // Compute segment concentration
        for(int i=0; i<M; i++)
            phi[i] = simpson_rule_coeff[0]*q_1[i+N_START*M]*q_2[i+N_START*M];
        for(int n=N_START+1; n<=N_END; n++)
        {
            for(int i=0; i<M; i++)
                phi[i] += simpson_rule_coeff[n-N_START]*q_1[i+n*M]*q_2[i+n*M];
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

void CpuPseudoContinuous::compute_statistics(
    std::map<std::string, double*> q_init,
    std::map<std::string, double*> w_block,
    double *phi, double &single_partition)
{
    try
    {
        const int M   = cb->get_n_grid();
        const int N   = pc->get_n_segment_total();
        const int N_B = pc->get_n_block();
        const double ds = pc->get_ds();
        auto seg_start  = get_block_start();

        for(int b=0; b<N_B; b++)
        {
            if( w_block.count(pc->get_block_species(b)) == 0)
                throw_with_line_number("\"" + pc->get_block_species(b) + "\" species is not in w_block.");
        }

        for(const auto& item: w_block)
        {
            std::string species = item.first;
            double *w = item.second;
            for(int i=0; i<M; i++){ 
                exp_dw     [species][i] = exp(-w[i]*ds*0.5);
                exp_dw_half[species][i] = exp(-w[i]*ds*0.25);
            }
        }

        #pragma omp parallel sections num_threads(2)
        {
            #pragma omp section
            {
                // initial condition
                if (q_init.count("1") == 0)
                {
                    for(int i=0; i<M; i++)
                        q_1[i] = 1.0;
                }
                else
                {
                    for(int i=0; i<M; i++)
                        q_1[i] = q_init["1"][i];
                }
                // apply the propagator successively
                for(int b=0; b<N_B; b++)
                {
                    std::string species = pc->get_block_species(b);
                    for(int n=seg_start[b]+1; n<=seg_start[b+1]; n++)
                        one_step(
                            &q_1[(n-1)*M],&q_1[n*M], boltz_bond[species],
                            boltz_bond_half[species], exp_dw[species], exp_dw_half[species]);
                }
            }
            #pragma omp section
            {
                // initial condition
                if (q_init.count("2") == 0)
                {
                    for(int i=0; i<M; i++)
                        q_2[i+N*M] = 1.0;
                }
                else
                {
                    for(int i=0; i<M; i++)
                        q_2[i+N*M] = q_init["2"][i];
                }

                // apply the propagator successively
                for(int b=N_B-1; b>=0; b--)
                {
                    std::string species = pc->get_block_species(b);
                    for(int n=seg_start[b+1]; n>=seg_start[b]+1; n--)
                        one_step(
                            &q_2[n*M], &q_2[(n-1)*M], boltz_bond[species], 
                            boltz_bond_half[species], exp_dw[species], exp_dw_half[species]);
                }
            }
        }

        // segment concentration.
        for(int b=0; b<N_B; b++)
        {
            calculate_phi_one_type(&phi[b*M], seg_start[b], seg_start[b+1]);
        }

        // calculates the single chain partition function
        single_partition = cb->inner_product(&q_1[N*M],&q_2[N*M]);

        // normalize the concentration
        for(int b=0; b<N_B; b++)
            for(int i=0; i<M; i++)
                phi[b*M+i] *= cb->get_volume()*pc->get_ds()/single_partition;
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
void CpuPseudoContinuous::get_partition(double *q_out, int v, int u, int n)
{
    // This method should be invoked after invoking compute_statistics()

    // Get partial partition functions
    // This is made for debugging and testing
    try
    {
        const int M = cb->get_n_grid();
        const int b = pc->get_array_idx(v,u);
        const int N = pc->get_n_segment(b);
        auto block_start = get_block_start();

        if (n < 0 || n > N)
            throw_with_line_number("n (" + std::to_string(n) + ") must be in range [0, " + std::to_string(N) + "]");

        if (v < u)
        {
            for(int i=0; i<M; i++)
                q_out[i] = q_1[(block_start[b]+n)*M+i];
        }
        else
        {
            for(int i=0; i<M; i++)
                q_out[i] = q_2[(N-block_start[b]-n)*M+i];
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}