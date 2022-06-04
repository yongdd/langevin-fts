#include <cmath>
#include "CpuPseudoGaussian.h"
#include "SimpsonQuadrature.h"

CpuPseudoGaussian::CpuPseudoGaussian(
    SimulationBox *sb,
    PolymerChain *pc, FFT *fft)
    : Pseudo(sb, pc)
{
    try
    {
        const int M = sb->get_n_grid();
        const int N = pc->get_n_contour();

        this->fft = fft;
        this->boltz_bond_a = new double[n_complex_grid];
        this->boltz_bond_b = new double[n_complex_grid];
        this->boltz_bond_a_half = new double[n_complex_grid];
        this->boltz_bond_b_half = new double[n_complex_grid];
        this->q_1 = new double[M*(N+1)];
        this->q_2 = new double[M*(N+1)];

        update();
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
CpuPseudoGaussian::~CpuPseudoGaussian()
{
    delete fft;
    delete[] boltz_bond_a, boltz_bond_a_half;
    delete[] boltz_bond_b, boltz_bond_b_half;
    delete[] q_1, q_2;
}
void CpuPseudoGaussian::update()
{
    try
    {
        double bond_length_a, bond_length_b;
        const double eps = pc->get_epsilon();
        const double f = pc->get_f();

        bond_length_a = eps*eps/(f*eps*eps + (1.0-f));
        bond_length_b = 1.0/(f*eps*eps + (1.0-f));

        get_boltz_bond(boltz_bond_a,      bond_length_a,   sb->get_nx(), sb->get_dx(), pc->get_ds());
        get_boltz_bond(boltz_bond_b,      bond_length_b,   sb->get_nx(), sb->get_dx(), pc->get_ds());
        get_boltz_bond(boltz_bond_a_half, bond_length_a/2, sb->get_nx(), sb->get_dx(), pc->get_ds());
        get_boltz_bond(boltz_bond_b_half, bond_length_b/2, sb->get_nx(), sb->get_dx(), pc->get_ds());
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

std::array<double,3> CpuPseudoGaussian::dq_dl()
{
    // This method should be invoked after invoking find_phi().

    // To calculate stress, we multiply weighted fourier basis to q(k)*q^dagger(-k).
    // We only need the real part of stress calculation.

    try
    {
        const int DIM  = sb->get_dim();
        const int M    = sb->get_n_grid();
        const int N    = pc->get_n_contour();
        const int N_A  = pc->get_n_contour_a();
        const int M_COMPLEX = this->n_complex_grid;

        const double eps = pc->get_epsilon();
        const double f = pc->get_f();
        const double bond_length_a = eps*eps/(f*eps*eps + (1.0-f));
        const double bond_length_b = 1.0/(f*eps*eps + (1.0-f));
        double bond_length;

        std::array<double,3> dq_dl;
        std::complex<double> k_q_1[M_COMPLEX];
        std::complex<double> k_q_2[M_COMPLEX];

        double simpson_rule_coeff_a[N_A+1];
        double simpson_rule_coeff_b[N-N_A+1];
        double fourier_basis_x[M_COMPLEX];
        double fourier_basis_y[M_COMPLEX];
        double fourier_basis_z[M_COMPLEX];

        get_weighted_fourier_basis(fourier_basis_x, fourier_basis_y, fourier_basis_z, sb->get_nx(), sb->get_dx());

        for(int i=0; i<3; i++)
            dq_dl[i] = 0.0;

        SimpsonQuadrature::init_coeff(simpson_rule_coeff_a, N_A);
        for(int n=0; n<=N_A; n++)
        {
            fft->forward(&q_1[n*M],k_q_1);
            fft->forward(&q_2[n*M],k_q_2);

            if ( DIM >= 3 )
            {
                for(int i=0; i<M_COMPLEX; i++)
                    dq_dl[0] += simpson_rule_coeff_a[n]*bond_length_a*(k_q_1[i]*std::conj(k_q_2[i])).real()*fourier_basis_x[i];
            }
            if ( DIM >= 2 )
            {
                for(int i=0; i<M_COMPLEX; i++)
                    dq_dl[1] += simpson_rule_coeff_a[n]*bond_length_a*(k_q_1[i]*std::conj(k_q_2[i])).real()*fourier_basis_y[i];
            }
            if ( DIM >= 1 )
            {
                for(int i=0; i<M_COMPLEX; i++)
                    dq_dl[2] += simpson_rule_coeff_a[n]*bond_length_a*(k_q_1[i]*std::conj(k_q_2[i])).real()*fourier_basis_z[i];
            }
        }
        
        SimpsonQuadrature::init_coeff(simpson_rule_coeff_b, N-N_A);
        for(int n=N_A; n<=N; n++)
        {
            fft->forward(&q_1[n*M],k_q_1);
            fft->forward(&q_2[n*M],k_q_2);
            if ( DIM >= 3 )
            {
                for(int i=0; i<M_COMPLEX; i++)
                    dq_dl[0] += simpson_rule_coeff_b[n-N_A]*bond_length_b*(k_q_1[i]*std::conj(k_q_2[i])).real()*fourier_basis_x[i];
            }
            if ( DIM >= 2 )
            {
                for(int i=0; i<M_COMPLEX; i++)
                    dq_dl[1] += simpson_rule_coeff_b[n-N_A]*bond_length_b*(k_q_1[i]*std::conj(k_q_2[i])).real()*fourier_basis_y[i];
            }
            if ( DIM >= 1 )
            {
                for(int i=0; i<M_COMPLEX; i++)
                    dq_dl[2] += simpson_rule_coeff_b[n-N_A]*bond_length_b*(k_q_1[i]*std::conj(k_q_2[i])).real()*fourier_basis_z[i];
            }
        }
        
        for(int d=0; d<3; d++)
            dq_dl[d] /= 3.0*sb->get_lx(d)*M*M*N/sb->get_volume();

        return dq_dl;
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

void CpuPseudoGaussian::calculate_phi_one_type(
    double *phi, const int N_START, const int N_END)
{
    try
    {
        const int M = sb->get_n_grid();
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

void CpuPseudoGaussian::find_phi(double *phi_a,  double *phi_b,
                                 double *q_1_init, double *q_2_init,
                                 double *w_a, double *w_b, double &single_partition)
{
    try
    {
        const int M     = sb->get_n_grid();
        const int N     = pc->get_n_contour();
        const int N_A   = pc->get_n_contour_a();
        //const int N_B   = pc->get_n_contour_b();
        const double ds = pc->get_ds();

        double exp_dw_a[M];
        double exp_dw_b[M];
        double exp_dw_a_half[M];
        double exp_dw_b_half[M];

        for(int i=0; i<M; i++)
        {
            exp_dw_a     [i] = exp(-w_a[i]*ds*0.5);
            exp_dw_b     [i] = exp(-w_b[i]*ds*0.5);
            exp_dw_a_half[i] = exp(-w_a[i]*ds*0.25);
            exp_dw_b_half[i] = exp(-w_b[i]*ds*0.25);
        }

        #pragma omp parallel sections num_threads(2)
        {
            #pragma omp section
            {
                for(int i=0; i<M; i++)
                    q_1[i] = q_1_init[i];
                // diffusion of A chain
                for(int n=1; n<=N_A; n++)
                    one_step(&q_1[(n-1)*M],&q_1[n*M],
                            boltz_bond_a,boltz_bond_a_half,
                            exp_dw_a,exp_dw_a_half);
                // diffusion of B chain
                for(int n=N_A+1; n<=N; n++)
                    one_step(&q_1[(n-1)*M],&q_1[n*M],
                            boltz_bond_b,boltz_bond_b_half,
                            exp_dw_b,exp_dw_b_half);
            }
            #pragma omp section
            {
                for(int i=0; i<M; i++)
                    q_2[i+N*M] = q_2_init[i];
                // diffusion of B chain
                for(int n=N; n>=N_A+1; n--)
                    one_step(&q_2[n*M],&q_2[(n-1)*M],
                            boltz_bond_b,boltz_bond_b_half,
                            exp_dw_b,exp_dw_b_half);
                // diffusion of A chain
                for(int n=N_A; n>=1; n--)
                    one_step(&q_2[n*M],&q_2[(n-1)*M],
                            boltz_bond_a,boltz_bond_a_half,
                            exp_dw_a,exp_dw_a_half);
            }
        }

        // segment concentration.
        // A block
        calculate_phi_one_type(phi_a, 0, N_A);
        // B block
        calculate_phi_one_type(phi_b, N_A, N);

        // calculates the single chain partition function
        single_partition = sb->inner_product(&q_1[N_A*M],&q_2[N_A*M]);

        // normalize the concentration
        for(int i=0; i<M; i++)
        {
            phi_a[i] *= sb->get_volume()/single_partition/N;
            phi_b[i] *= sb->get_volume()/single_partition/N;
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CpuPseudoGaussian::one_step(double *q_in, double *q_out,
                                 double *boltz_bond, double *boltz_bond_half,
                                 double *exp_dw, double *exp_dw_half)
{
    try
    {
        const int M = sb->get_n_grid();
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
void CpuPseudoGaussian::get_partition(double *q_1_out, int n1, double *q_2_out, int n2)
{
    // This method should be invoked after invoking find_phi().
    
    // Get partial partition functions
    // This is made for debugging and testing.
    const int M = sb->get_n_grid();
    const int N = pc->get_n_contour();

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
