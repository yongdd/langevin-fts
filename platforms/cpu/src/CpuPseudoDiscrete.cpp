#include <cmath>
#include "CpuPseudoDiscrete.h"

CpuPseudoDiscrete::CpuPseudoDiscrete(
    SimulationBox *sb,
    PolymerChain *pc, FFT *fft)
    : Pseudo(sb, pc)
{
    const int M = sb->get_n_grid();
    const int N = pc->get_n_contour();

    this->fft = fft;
    this->boltz_bond_a = new double[n_complex_grid];
    this->boltz_bond_b = new double[n_complex_grid];
    this->boltz_bond_ab = new double[n_complex_grid];
    this->q_1 = new double[M*N] {0.0};
    this->q_2 = new double[M*N] {0.0};
    
    update();
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
    double step_a, step_b, step_ab;
    const double eps = pc->get_epsilon();
    const double f = pc->get_f();
    
    step_a = eps*eps/(f*eps*eps + (1.0-f));
    step_b = 1.0/(f*eps*eps + (1.0-f));
    step_ab = 0.5*step_a + 0.5*step_b;
    
    set_boltz_bond(boltz_bond_a,  step_a,  sb->get_nx(), sb->get_dx(), pc->get_ds());
    set_boltz_bond(boltz_bond_b,  step_b,  sb->get_nx(), sb->get_dx(), pc->get_ds());
    set_boltz_bond(boltz_bond_ab, step_ab, sb->get_nx(), sb->get_dx(), pc->get_ds());
}
void CpuPseudoDiscrete::find_phi(double *phi_a,  double *phi_b,
                                 double *q_1_init, double *q_2_init,
                                 double *w_a, double *w_b, double &single_partition)
{
    const int M    = sb->get_n_grid();
    const int N    = pc->get_n_contour();
    const int N_A  = pc->get_n_contour_a();
    //const int N_B  = pc->get_n_contour_b();
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
void CpuPseudoDiscrete::one_step(double *q_in, double *q_out,
                                double *boltz_bond, double *exp_dw)
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

/* Get partial partition functions
* This is made for debugging and testing.
* Do NOT this at main progarams.
* */
void CpuPseudoDiscrete::get_partition(double *q_1_out, int n1, double *q_2_out, int n2)
{
    const int M = sb->get_n_grid();
    const int N = pc->get_n_contour();

    for(int i=0; i<M; i++)
    {
        q_1_out[i] =q_1[(n1-1)*M+i];
        q_2_out[i] =q_2[(n2-1)*M+i];
    }
}
