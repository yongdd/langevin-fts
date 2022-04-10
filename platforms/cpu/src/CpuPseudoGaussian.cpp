#include <cmath>
#include "CpuPseudoGaussian.h"

CpuPseudoGaussian::CpuPseudoGaussian(
    SimulationBox *sb,
    PolymerChain *pc, FFT *fft)
    : Pseudo(sb, pc)
{
    const int M = sb->get_n_grid();
    const int N = pc->get_n_contour();  
    
    this->fft = fft;
    this->q1 = new double[M*(N+1)];
    this->q2 = new double[M*(N+1)];
}
CpuPseudoGaussian::~CpuPseudoGaussian()
{
    delete fft;
    delete[] q1;
    delete[] q2;
}
void CpuPseudoGaussian::find_phi(double *phia,  double *phib,
                      double *q1_init, double *q2_init,
                      double *wa, double *wb, double &QQ)
{
    const int M     = sb->get_n_grid();
    const int N     = pc->get_n_contour();
    const int N_A   = pc->get_n_contour_a();
    //const int N_B   = pc->get_n_contour_b();
    const double ds = pc->get_ds();
    
    double expdwa[M];
    double expdwb[M];
    double expdwa_half[M];
    double expdwb_half[M];

    for(int i=0; i<M; i++)
    {
        expdwa     [i] = exp(-wa[i]*ds*0.5);
        expdwb     [i] = exp(-wb[i]*ds*0.5);
        expdwa_half[i] = exp(-wa[i]*ds*0.25);
        expdwb_half[i] = exp(-wb[i]*ds*0.25);
    }

    #pragma omp parallel sections num_threads(2)
    {
        #pragma omp section
        {
            for(int i=0; i<M; i++)
                q1[i] = q1_init[i];
            // diffusion of A chain
            for(int n=1; n<=N_A; n++)
                one_step(&q1[(n-1)*M],&q1[n*M],expdwa,expdwa_half);
            // diffusion of B chain
            for(int n=N_A+1; n<=N; n++)
                one_step(&q1[(n-1)*M],&q1[n*M],expdwb,expdwb_half);
        }
        #pragma omp section
        {
            for(int i=0; i<M; i++)
                q2[i+N*M] = q2_init[i];
            // diffusion of B chain
            for(int n=N; n>=N_A+1; n--)
                one_step(&q2[n*M],&q2[(n-1)*M],expdwb,expdwb_half);
            // diffusion of A chain
            for(int n=N_A; n>=1; n--)
                one_step(&q2[n*M],&q2[(n-1)*M],expdwa,expdwa_half);
        }
    }
    // compute segment concentration with Simpson quadratrue.
    // segment concentration. only half contribution from the end
    for(int i=0; i<M; i++)
        phia[i] = q1[i]*q2[i]/2;
    for(int n=1; n<=N_A-1; n++)
    {
        for(int i=0; i<M; i++)
            phia[i] += q1[i+n*M]*q2[i+n*M];
    }
    // the junction is half A and half B
    for(int i=0; i<M; i++)
    {
        phib[i] = q1[i+N_A*M]*q2[i+N_A*M]/2;
        phia[i] += phib[i];
    }
    for(int n=N_A+1; n<=N-1; n++)
    {
        for(int i=0; i<M; i++)
            phib[i] += q1[i+n*M]*q2[i+n*M];
    }
    // only half contribution from the end
    for(int i=0; i<M; i++)
        phib[i] += q1[i+N*M]*q2[i+N*M]/2;
    // calculates the total partition function
    QQ = sb->inner_product(&q1[N_A*M],&q2[N_A*M]);

    // normalize the concentration
    for(int i=0; i<M; i++)
    {
        phia[i] *= sb->get_volume()/QQ/N;
        phib[i] *= sb->get_volume()/QQ/N;
    }
}
void CpuPseudoGaussian::one_step(double *qin, double *qout,
                     double *expdw, double *expdw_half)
{
    const int M = sb->get_n_grid();
    const int M_COMPLEX = this->n_complex_grid;
    double qout1[M], qout2[M];
    std::complex<double> k_qin1[M_COMPLEX], k_qin2[M_COMPLEX];

    #pragma omp parallel sections num_threads(2)
    {
        #pragma omp section
        {
            // step 1
            for(int i=0; i<M; i++)
                qout1[i] = expdw[i]*qin[i];
            // 3D fourier discrete transform, forward and inplace
            fft->forward(qout1,k_qin1);
            // multiply e^(-k^2 ds/6) in fourier space, in all 3 directions
            for(int i=0; i<M_COMPLEX; i++)
                k_qin1[i] *= expf[i];
            // 3D fourier discrete transform, backword and inplace
            fft->backward(k_qin1,qout1);
            // normalization calculation and evaluate e^(-w*ds/2) in real space
            for(int i=0; i<M; i++)
                qout1[i] *= expdw[i];
        }
        #pragma omp section
        {
            // step 2
            // evaluate e^(-w*ds/4) in real space
            for(int i=0; i<M; i++)
                qout2[i] = expdw_half[i]*qin[i];
            // 3D fourier discrete transform, forward and inplace
            fft->forward(qout2,k_qin2);
            // multiply e^(-k^2 ds/12) in fourier space, in all 3 directions
            for(int i=0; i<M_COMPLEX; i++)
                k_qin2[i] *= expf_half[i];
            // 3D fourier discrete transform, backword and inplace
            fft->backward(k_qin2,qout2);
            // normalization calculation and evaluate e^(-w*ds/2) in real space
            for(int i=0; i<M; i++)
                qout2[i] *= expdw[i];
            // 3D fourier discrete transform, forward and inplace
            fft->forward(qout2,k_qin2);
            // multiply e^(-k^2 ds/12) in fourier space, in all 3 directions
            for(int i=0; i<M_COMPLEX; i++)
                k_qin2[i] *= expf_half[i];
            // 3D fourier discrete transform, backword and inplace
            fft->backward(k_qin2,qout2);
            // normalization calculation and evaluate e^(-w*ds/4) in real space
            for(int i=0; i<M; i++)
                qout2[i] *= expdw_half[i];
        }
    }
    for(int i=0; i<M; i++)
        qout[i] = (4.0*qout2[i] - qout1[i])/3.0;
}

/* Get partial partition functions
* This is made for debugging and testing.
* Do NOT this at main progarams.
* */
void CpuPseudoGaussian::get_partition(double *q1_out, int n1, double *q2_out, int n2)
{
    const int M = sb->get_n_grid();
    const int N = pc->get_n_contour();
    
    for(int i=0; i<M; i++)
    {
        q1_out[i] =q1[n1*M+i];
        q2_out[i] =q2[(N-n2)*M+i];
    }
}
