#include <cmath>
#include "CpuPseudoDiscrete.h"

CpuPseudoDiscrete::CpuPseudoDiscrete(
    SimulationBox *sb,
    PolymerChain *pc, FFT *fft)
    : Pseudo(sb, pc)
{
    const int MM = sb->get_MM();
    const int NN = pc->get_NN();

    this->fft = fft;
    this->q1 = new double[MM*NN] {0.0};
    this->q2 = new double[MM*NN] {0.0};
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
    const int MM = sb->get_MM();
    const int NN = pc->get_NN();
    const int NN_A = pc->get_NN_A();
    const int NN_B = pc->get_NN_B();
    const double ds = pc->get_ds();

    double h_a[MM];
    double h_b[MM];

    for(int i=0; i<MM; i++)
    {
        h_a[i] = exp(-wa[i]*ds);
        h_b[i] = exp(-wb[i]*ds);
    }

    #pragma omp parallel sections num_threads(2)
    {
        #pragma omp section
        {
            for(int i=0; i<MM; i++)
                q1[i] = h_a[i]*q1_init[i];
            // diffusion of A chain
            for(int n=1; n<NN_A; n++)
                onestep(&q1[(n-1)*MM],&q1[n*MM],h_a);
            // diffusion of B chain
            for(int n=NN_A; n<NN; n++)
                onestep(&q1[(n-1)*MM],&q1[n*MM],h_b);
        }
        #pragma omp section
        {
            for(int i=0; i<MM; i++)
                q2[i+(NN-1)*MM] = h_b[i]*q2_init[i];
            // diffusion of B chain
            for(int n=NN-1; n>NN_A; n--)
                onestep(&q2[n*MM],&q2[(n-1)*MM],h_b);
            // diffusion of A chain
            for(int n=NN_A; n>0; n--)
                onestep(&q2[n*MM],&q2[(n-1)*MM],h_a);
        }
    }
    // Compute segment concentration A
    for(int i=0; i<MM; i++)
        phia[i] = q1[i]*q2[i];
    for(int n=1; n<NN_A; n++)
    {
        for(int i=0; i<MM; i++)
            phia[i] += q1[i+n*MM]*q2[i+n*MM];
    }
    // Compute segment concentration B
    for(int i=0; i<MM; i++)
        phib[i] = q1[i+NN_A*MM]*q2[i+NN_A*MM];
    for(int n=NN_A+1; n<NN; n++)
    {
        for(int i=0; i<MM; i++)
            phib[i] += q1[i+n*MM]*q2[i+n*MM];
    }
    // calculates the total partition function
    QQ = sb->integral(&q1[(NN-1)*MM]);

    // normalize the concentration
    for(int i=0; i<MM; i++)
    {
        phia[i] *= sb->get_volume()/h_a[i]/QQ/NN;
        phib[i] *= sb->get_volume()/h_b[i]/QQ/NN;
    }
}
void CpuPseudoDiscrete::onestep(double *qin, double *qout,
                                double *expdw)
{
    const int MM = sb->get_MM();
    std::complex<double> k_qin[MM_COMPLEX];

    // 3D fourier discrete transform, forward and inplace
    fft->forward(qin,k_qin);
    // multiply e^(-k^2 ds/6) in fourier space, in all 3 directions
    for(int i=0; i<MM_COMPLEX; i++)
        k_qin[i] *= expf[i];
    // 3D fourier discrete transform, backword and inplace
    fft->backward(k_qin,qout);
    // normalization calculation and evaluate e^(-w*ds/2) in real space
    for(int i=0; i<MM; i++)
        qout[i] *= expdw[i];
}

/* Get partial partition functions
* This is made for debugging and testing.
* Do NOT this at main progarams.
* */
void CpuPseudoDiscrete::get_partition(double *q1_out,  double *q2_out, int n)
{
    const int MM = sb->get_MM();
    const int NN = pc->get_NN();

    for(int i=0; i<MM; i++)
    {
        q1_out[i] =q1[(n-1)*MM+i];
        q2_out[i] =q2[(NN-n)*MM+i];
    }
}
