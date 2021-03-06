#include <cmath>
#include "CpuPseudoDiscrete.h"

CpuPseudoDiscrete::CpuPseudoDiscrete(
    SimulationBox *sb,
    PolymerChain *pc, FFT *fft)
    : Pseudo(sb, pc)
{
    try
    {
        const int M = sb->get_n_grid();
        const int N = pc->get_n_segment();

        this->fft = fft;
        this->boltz_bond_a  = new double[n_complex_grid];
        this->boltz_bond_b  = new double[n_complex_grid];
        this->boltz_bond_ab = new double[n_complex_grid];
        this->q_1 = new double[M*N];
        this->q_2 = new double[M*N];

        update();
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
CpuPseudoDiscrete::~CpuPseudoDiscrete()
{
    delete fft;
    delete[] boltz_bond_a, boltz_bond_b, boltz_bond_ab;
    delete[] q_1;
    delete[] q_2;
}
void CpuPseudoDiscrete::update()
{
    try
    {
        double bond_length_a, bond_length_b, bond_length_ab;
        const double eps = pc->get_epsilon();
        const double f = pc->get_f();

        bond_length_a = eps*eps/(f*eps*eps + (1.0-f));
        bond_length_b = 1.0/(f*eps*eps + (1.0-f));
        bond_length_ab = 0.5*bond_length_a + 0.5*bond_length_b;

        get_boltz_bond(boltz_bond_a,  bond_length_a,  sb->get_nx(), sb->get_dx(), pc->get_ds());
        get_boltz_bond(boltz_bond_b,  bond_length_b,  sb->get_nx(), sb->get_dx(), pc->get_ds());
        get_boltz_bond(boltz_bond_ab, bond_length_ab, sb->get_nx(), sb->get_dx(), pc->get_ds());
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
std::array<double,3> CpuPseudoDiscrete::dq_dl()
{
    // This method should be invoked after invoking find_phi().

    // To calculate stress, we multiply weighted fourier basis to q(k)*q^dagger(-k).
    // We only need the real part of stress calculation.

    try
    {
        const int DIM  = sb->get_dim();
        const int M    = sb->get_n_grid();
        const int N    = pc->get_n_segment();
        const int N_A  = pc->get_n_segment_a();
        const int M_COMPLEX = this->n_complex_grid;

        const double eps = pc->get_epsilon();
        const double f = pc->get_f();
        const double bond_length_a = eps*eps/(f*eps*eps + (1.0-f));
        const double bond_length_b = 1.0/(f*eps*eps + (1.0-f));
        const double bond_length_ab = 0.5*bond_length_a + 0.5*bond_length_b;
        double temp, bond_length, *boltz_bond;

        std::array<double,3> dq_dl;
        std::complex<double> k_q_1[M_COMPLEX];
        std::complex<double> k_q_2[M_COMPLEX];

        double fourier_basis_x[M_COMPLEX];
        double fourier_basis_y[M_COMPLEX];
        double fourier_basis_z[M_COMPLEX];

        get_weighted_fourier_basis(fourier_basis_x, fourier_basis_y, fourier_basis_z, sb->get_nx(), sb->get_dx());

        for(int i=0; i<3; i++)
            dq_dl[i] = 0.0;

        for(int n=1; n<N; n++)
        {
            fft->forward(&q_1[(n-1)*M],k_q_1);
            fft->forward(&q_2[ n   *M],k_q_2);

            if ( n < N_A)
            {
                bond_length = bond_length_a;
                boltz_bond = boltz_bond_a;
            }
            else if ( n == N_A)
            {
                bond_length = bond_length_ab;
                boltz_bond = boltz_bond_ab;
            }
            else if ( n < N)
            {
                bond_length = bond_length_b;
                boltz_bond = boltz_bond_b;
            }

            if ( DIM >= 3 )
            {
                for(int i=0; i<M_COMPLEX; i++)
                    dq_dl[0] += bond_length*boltz_bond[i]*(k_q_1[i]*std::conj(k_q_2[i])).real()*fourier_basis_x[i];
            }
            if ( DIM >= 2 )
            {
                for(int i=0; i<M_COMPLEX; i++)
                    dq_dl[1] += bond_length*boltz_bond[i]*(k_q_1[i]*std::conj(k_q_2[i])).real()*fourier_basis_y[i];
            }
            if ( DIM >= 1 )
            {
                for(int i=0; i<M_COMPLEX; i++)
                    dq_dl[2] += bond_length*boltz_bond[i]*(k_q_1[i]*std::conj(k_q_2[i])).real()*fourier_basis_z[i];
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
void CpuPseudoDiscrete::find_phi(double *phi_a,  double *phi_b,
                                 double *q_1_init, double *q_2_init,
                                 double *w_a, double *w_b, double &single_partition)
{
    try
    {
        const int M    = sb->get_n_grid();
        const int N    = pc->get_n_segment();
        const int N_A  = pc->get_n_segment_a();
        //const int N_B  = pc->get_n_segment_b();
        const double ds = pc->get_ds();

        double h_a[M];
        double h_b[M];

        for(int i=0; i<M; i++)
        {
            h_a[i] = exp(-w_a[i]*ds);
            h_b[i] = exp(-w_b[i]*ds);
        }

        #pragma omp parallel sections num_threads(2)
        {
            #pragma omp section
            {
                for(int i=0; i<M; i++)
                    q_1[i] = h_a[i]*q_1_init[i];
                // diffusion of A segment
                for(int n=1; n<N_A; n++)
                    one_step(&q_1[(n-1)*M],&q_1[n*M],boltz_bond_a, h_a);
                // diffusion of B from A segment
                one_step(&q_1[(N_A-1)*M],&q_1[N_A*M],boltz_bond_ab,h_b);
                // diffusion of B segment
                for(int n=N_A+1; n<N; n++)
                    one_step(&q_1[(n-1)*M],&q_1[n*M],boltz_bond_b, h_b);
            }
            #pragma omp section
            {
                for(int i=0; i<M; i++)
                    q_2[i+(N-1)*M] = h_b[i]*q_2_init[i];
                // diffusion of B segment
                for(int n=N-1; n>N_A; n--)
                    one_step(&q_2[n*M],&q_2[(n-1)*M],boltz_bond_b, h_b);
                // diffusion of A from B segment
                one_step(&q_2[N_A*M],&q_2[(N_A-1)*M],boltz_bond_ab,h_a);
                // diffusion of A segment
                for(int n=N_A-1; n>0; n--)
                    one_step(&q_2[n*M],&q_2[(n-1)*M],boltz_bond_a, h_a);
            }
        }
        // Compute segment concentration A
        for(int i=0; i<M; i++)
            phi_a[i] = q_1[i]*q_2[i];
        for(int n=1; n<N_A; n++)
        {
            for(int i=0; i<M; i++)
                phi_a[i] += q_1[i+n*M]*q_2[i+n*M];
        }
        // Compute segment concentration B
        for(int i=0; i<M; i++)
            phi_b[i] = q_1[i+N_A*M]*q_2[i+N_A*M];
        for(int n=N_A+1; n<N; n++)
        {
            for(int i=0; i<M; i++)
                phi_b[i] += q_1[i+n*M]*q_2[i+n*M];
        }
        // calculates the single chain partition function
        single_partition = sb->inner_product(&q_1[(N-1)*M], q_1_init);

        // normalize the concentration
        for(int i=0; i<M; i++)
        {
            phi_a[i] *= sb->get_volume()/h_a[i]/single_partition/N;
            phi_b[i] *= sb->get_volume()/h_b[i]/single_partition/N;
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
        const int M = sb->get_n_grid();
        const int M_COMPLEX = this->n_complex_grid;

        std::complex<double> k_q_in[M_COMPLEX];
        // 3D fourier discrete transform, forward and inplace
        fft->forward(q_in,k_q_in);
        // multiply e^(-k^2 ds/6) in fourier space, in all 3 directions
        for(int i=0; i<M_COMPLEX; i++)
            k_q_in[i] *= boltz_bond[i];
        // 3D fourier discrete transform, backword and inplace
        fft->backward(k_q_in,q_out);
        // normalization calculation and evaluate e^(-w*ds/2) in real space
        for(int i=0; i<M; i++)
            q_out[i] *= exp_dw[i];
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CpuPseudoDiscrete::get_partition(double *q_1_out, int n1, double *q_2_out, int n2)
{
    // This method should be invoked after invoking find_phi().
    
    // Get partial partition functions
    // This is made for debugging and testing.

    const int M = sb->get_n_grid();
    const int N = pc->get_n_segment();

    if (n1 < 1 || n1 > N)
        throw_with_line_number("n1 (" + std::to_string(n1) + ") must be in range [1, " + std::to_string(N) + "]");
    if (n2 < 1 || n2 > N)
        throw_with_line_number("n2 (" + std::to_string(n2) + ") must be in range [1, " + std::to_string(N) + "]");

    for(int i=0; i<M; i++)
    {
        q_1_out[i] =q_1[(n1-1)*M+i];
        q_2_out[i] =q_2[(n2-1)*M+i];
    }
}
