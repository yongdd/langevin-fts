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
        const int N_B = pc->get_n_block();
        const int N = pc->get_n_segment_total();

        this->n_block = N_B;
        this->fft = fft;
        this->boltz_bond = new double*[N_B];
        this->boltz_bond_half = new double*[N_B];
        for (int b=0; b<N_B; b++)
        {
            this->boltz_bond[b]= new double[n_complex_grid];
            this->boltz_bond_half[b] = new double[n_complex_grid];
        }
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
    const int N_B = n_block;

    delete fft;
    for (int b=0; b<N_B; b++)
    {
        delete[] boltz_bond[b];
        delete[] boltz_bond_half[b];
    }
    delete[] boltz_bond, boltz_bond_half;
    delete[] q_1, q_2;
}
void CpuPseudoContinuous::update()
{
    try
    {
        const int N_B = pc->get_n_block();

        for (int b=0; b<N_B; b++)
        {
        get_boltz_bond(boltz_bond[b],      pc->get_bond_length(b),   cb->get_nx(), cb->get_dx(), pc->get_ds());
        get_boltz_bond(boltz_bond_half[b], pc->get_bond_length(b)/2, cb->get_nx(), cb->get_dx(), pc->get_ds());
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
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
        const std::vector<int> N_SEG    = pc->get_n_segment();
        const int M_COMPLEX = this->n_complex_grid;
        const std::vector<int> seg_start= pc->get_block_start();

        std::array<double,3> dq_dl;
        std::complex<double> k_q_1[M_COMPLEX];
        std::complex<double> k_q_2[M_COMPLEX];

        //double simpson_rule_coeff_a[N_A+1];
        //double simpson_rule_coeff_b[N-N_A+1];
        double simpson_rule_coeff[N];
        double fourier_basis_x[M_COMPLEX];
        double fourier_basis_y[M_COMPLEX];
        double fourier_basis_z[M_COMPLEX];

        get_weighted_fourier_basis(fourier_basis_x, fourier_basis_y, fourier_basis_z, cb->get_nx(), cb->get_dx());

        for(int i=0; i<3; i++)
            dq_dl[i] = 0.0;
        for(int b=0; b<N_B; b++)
        {
            SimpsonQuadrature::init_coeff(simpson_rule_coeff, N_SEG[b]);
            for(int n=seg_start[b]; n<=seg_start[b+1]; n++)
            {
                fft->forward(&q_1[n*M],k_q_1);
                fft->forward(&q_2[n*M],k_q_2);

                if ( DIM >= 3 )
                {
                    for(int i=0; i<M_COMPLEX; i++)
                        dq_dl[0] += simpson_rule_coeff[n-seg_start[b]]*pc->get_bond_length(b)*(k_q_1[i]*std::conj(k_q_2[i])).real()*fourier_basis_x[i];
                }
                if ( DIM >= 2 )
                {
                    for(int i=0; i<M_COMPLEX; i++)
                        dq_dl[1] += simpson_rule_coeff[n-seg_start[b]]*pc->get_bond_length(b)*(k_q_1[i]*std::conj(k_q_2[i])).real()*fourier_basis_y[i];
                }
                if ( DIM >= 1 )
                {
                    for(int i=0; i<M_COMPLEX; i++)
                        dq_dl[2] += simpson_rule_coeff[n-seg_start[b]]*pc->get_bond_length(b)*(k_q_1[i]*std::conj(k_q_2[i])).real()*fourier_basis_z[i];
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
        double simpson_rule_coeff[N_END-N_START+1];

        SimpsonQuadrature::init_coeff(simpson_rule_coeff, N_END-N_START);

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

void CpuPseudoContinuous::compute_statistics(double *phi, double *q_1_init, double *q_2_init,
                                 std::map<std::string, double*> w_block, double &single_partition)
{
    try
    {
        const int  M        = cb->get_n_grid();
        const int  N        = pc->get_n_segment_total();
        const int  N_B      = pc->get_n_block();
        const std::vector<int> N_SEG    = pc->get_n_segment();
        const double ds     = pc->get_ds();
        const std::vector<int> seg_start= pc->get_block_start();

        double exp_dw[N_B][M];
        double exp_dw_half[N_B][M];

        for(int b=0; b<N_B; b++)
        {
            double *w_block_one = w_block[pc->get_type(b)];
            for(int i=0; i<M; i++)
            {
                exp_dw     [b][i] = exp(-w_block_one[i]*ds*0.5);
                exp_dw_half[b][i] = exp(-w_block_one[i]*ds*0.25);
            }
        }

        #pragma omp parallel sections num_threads(2)
        {
            #pragma omp section
            {
                for(int i=0; i<M; i++)
                    q_1[i] = q_1_init[i];
                // diffusion of each blocks
                for(int b=0; b<N_B; b++)
                {
                    for(int n=seg_start[b]+1; n<=seg_start[b+1]; n++)
                        one_step(&q_1[(n-1)*M],&q_1[n*M],
                                boltz_bond[b],boltz_bond_half[b],
                                exp_dw[b],exp_dw_half[b]);
                }
            }
            #pragma omp section
            {
                for(int i=0; i<M; i++)
                    q_2[i+N*M] = q_2_init[i];
                for(int b=N_B-1; b>=0; b--)
                {
                    for(int n=seg_start[b+1]; n>=seg_start[b]+1; n--)
                        one_step(&q_2[n*M],&q_2[(n-1)*M],
                                boltz_bond[b],boltz_bond_half[b],
                                exp_dw[b],exp_dw_half[b]);
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
                // 3D fourier discrete transform, backword and inplace
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
                // 3D fourier discrete transform, backword and inplace
                fft->backward(k_q_in2,q_out2);
                // normalization calculation and evaluate e^(-w*ds/2) in real space
                for(int i=0; i<M; i++)
                    q_out2[i] *= exp_dw[i];
                // 3D fourier discrete transform, forward and inplace
                fft->forward(q_out2,k_q_in2);
                // multiply e^(-k^2 ds/12) in fourier space, in all 3 directions
                for(int i=0; i<M_COMPLEX; i++)
                    k_q_in2[i] *= boltz_bond_half[i];
                // 3D fourier discrete transform, backword and inplace
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
void CpuPseudoContinuous::get_partition(double *q_1_out, int n1, double *q_2_out, int n2)
{
    // This method should be invoked after invoking compute_statistics().
    
    // Get partial partition functions
    // This is made for debugging and testing.
    const int M = cb->get_n_grid();
    const int N = pc->get_n_segment_total();

    if (n1 < 0 || n1 > N)
        throw_with_line_number("n1 (" + std::to_string(n1) + ") must be in range [0, " + std::to_string(N) + "]");
    if (n2 < 0 || n2 > N)
        throw_with_line_number("n2 (" + std::to_string(n2) + ") must be in range [0, " + std::to_string(N) + "]");

    for(int i=0; i<M; i++)
    {
        q_1_out[i] =q_1[n1*M+i];
        q_2_out[i] =q_2[n2*M+i];
    }
}
