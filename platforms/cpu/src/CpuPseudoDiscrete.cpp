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
    this->q1 = new double[M*N] {0.0};
    this->q2 = new double[M*N] {0.0};
}
CpuPseudoDiscrete::~CpuPseudoDiscrete()
{
    delete fft;
    delete[] q1;
    delete[] q2;
}
void CpuPseudoDiscrete::find_phi(double *phia,  double *phib,
                                 double *q1_init, double *q2_init,
                                 double *wa, double *wb, double &QQ)
{
    const int M    = sb->get_n_grid();
    const int N    = pc->get_n_contour();
    const int N_A  = pc->get_n_contour_a();
    const int N_B  = pc->get_n_contour_b();
    const double ds = pc->get_ds();

    double h_a[M];
    double h_b[M];

    for(int i=0; i<M; i++)
    {
        h_a[i] = exp(-wa[i]*ds);
        h_b[i] = exp(-wb[i]*ds);
    }

    #pragma omp parallel sections num_threads(2)
    {
        #pragma omp section
        {
            for(int i=0; i<M; i++)
                q1[i] = h_a[i]*q1_init[i];
            // diffusion of A chain
            for(int n=1; n<N_A; n++)
                onestep(&q1[(n-1)*M],&q1[n*M],h_a);
            // diffusion of B chain
            for(int n=N_A; n<N; n++)
                onestep(&q1[(n-1)*M],&q1[n*M],h_b);
        }
        #pragma omp section
        {
            for(int i=0; i<M; i++)
                q2[i+(N-1)*M] = h_b[i]*q2_init[i];
            // diffusion of B chain
            for(int n=N-1; n>N_A; n--)
                onestep(&q2[n*M],&q2[(n-1)*M],h_b);
            // diffusion of A chain
            for(int n=N_A; n>0; n--)
                onestep(&q2[n*M],&q2[(n-1)*M],h_a);
        }
    }
    // Compute segment concentration A
    for(int i=0; i<M; i++)
        phia[i] = q1[i]*q2[i];
    for(int n=1; n<N_A; n++)
    {
        for(int i=0; i<M; i++)
            phia[i] += q1[i+n*M]*q2[i+n*M];
    }
    // Compute segment concentration B
    for(int i=0; i<M; i++)
        phib[i] = q1[i+N_A*M]*q2[i+N_A*M];
    for(int n=N_A+1; n<N; n++)
    {
        for(int i=0; i<M; i++)
            phib[i] += q1[i+n*M]*q2[i+n*M];
    }
    // calculates the total partition function
    QQ = sb->integral(&q1[(N-1)*M]);

    // normalize the concentration
    for(int i=0; i<M; i++)
    {
        phia[i] *= sb->get_volume()/h_a[i]/QQ/N;
        phib[i] *= sb->get_volume()/h_b[i]/QQ/N;
    }
}
void CpuPseudoDiscrete::onestep(double *qin, double *qout,
                                double *expdw)
{
    const int M = sb->get_n_grid();
    const int M_COMPLEX = this->n_complex_grid;
    
    std::complex<double> k_qin[M_COMPLEX];
    // 3D fourier discrete transform, forward and inplace
    fft->forward(qin,k_qin);
    // multiply e^(-k^2 ds/6) in fourier space, in all 3 directions
    for(int i=0; i<M_COMPLEX; i++)
        k_qin[i] *= expf[i];
    // 3D fourier discrete transform, backword and inplace
    fft->backward(k_qin,qout);
    // normalization calculation and evaluate e^(-w*ds/2) in real space
    for(int i=0; i<M; i++)
        qout[i] *= expdw[i];
}

/* Get partial partition functions
* This is made for debugging and testing.
* Do NOT this at main progarams.
* */
void CpuPseudoDiscrete::get_partition(double *q1_out,  double *q2_out, int n)
{
    const int M = sb->get_n_grid();
    const int N = pc->get_n_contour();

    for(int i=0; i<M; i++)
    {
        q1_out[i] =q1[(n-1)*M+i];
        q2_out[i] =q2[(N-n)*M+i];
    }
}
