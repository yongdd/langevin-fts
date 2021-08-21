#include <cmath>
#include "CpuPseudoGaussian.h"

CpuPseudoGaussian::CpuPseudoGaussian(
    SimulationBox *sb,
    PolymerChain *pc, FFT *fft)
    : Pseudo(sb, pc)
{
    const int MM = sb->get_MM();
    const int NN = pc->get_NN();  
    
    this->fft = fft;
    this->q1 = new double[MM*(NN+1)];
    this->q2 = new double[MM*(NN+1)];
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
    const int MM = sb->get_MM();
    const int NN = pc->get_NN();
    const int NN_A = pc->get_NN_A();
    const int NN_B = pc->get_NN_B();
    const double ds = pc->get_ds();
    
    double expdwa[MM];
    double expdwb[MM];
    double expdwa_half[MM];
    double expdwb_half[MM];

    for(int i=0; i<MM; i++)
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
            for(int i=0; i<MM; i++)
                q1[i] = q1_init[i];
            // diffusion of A chain
            for(int n=1; n<=NN_A; n++)
                onestep(&q1[(n-1)*MM],&q1[n*MM],expdwa,expdwa_half);
            // diffusion of B chain
            for(int n=NN_A+1; n<=NN; n++)
                onestep(&q1[(n-1)*MM],&q1[n*MM],expdwb,expdwb_half);
        }
        #pragma omp section
        {
            for(int i=0; i<MM; i++)
                q2[i+NN*MM] = q2_init[i];
            // diffusion of B chain
            for(int n=NN; n>=NN_A+1; n--)
                onestep(&q2[n*MM],&q2[(n-1)*MM],expdwb,expdwb_half);
            // diffusion of A chain
            for(int n=NN_A; n>=1; n--)
                onestep(&q2[n*MM],&q2[(n-1)*MM],expdwa,expdwa_half);
        }
    }
    // compute segment concentration with Simpson quadratrue.
    // segment concentration. only half contribution from the end
    for(int i=0; i<MM; i++)
        phia[i] = q1[i]*q2[i]/2;
    for(int n=1; n<=NN_A-1; n++)
    {
        for(int i=0; i<MM; i++)
            phia[i] += q1[i+n*MM]*q2[i+n*MM];
    }
    // the junction is half A and half B
    for(int i=0; i<MM; i++)
    {
        phib[i] = q1[i+NN_A*MM]*q2[i+NN_A*MM]/2;
        phia[i] += phib[i];
    }
    for(int n=NN_A+1; n<=NN-1; n++)
    {
        for(int i=0; i<MM; i++)
            phib[i] += q1[i+n*MM]*q2[i+n*MM];
    }
    // only half contribution from the end
    for(int i=0; i<MM; i++)
        phib[i] += q1[i+NN*MM]*q2[i+NN*MM]/2;
    // calculates the total partition function
    QQ = sb->inner_product(&q1[NN_A*MM],&q2[NN_A*MM]);

    // normalize the concentration
    for(int i=0; i<MM; i++)
    {
        phia[i] *= sb->get_volume()/QQ/NN;
        phib[i] *= sb->get_volume()/QQ/NN;
    }
}
void CpuPseudoGaussian::onestep(double *qin, double *qout,
                     double *expdw, double *expdw_half)
{
    const int MM = sb->get_MM();
    double qout1[MM], qout2[MM];
    std::complex<double> k_qin1[MM_COMPLEX], k_qin2[MM_COMPLEX];

    #pragma omp parallel sections num_threads(2)
    {
        #pragma omp section
        {
            // step 1
            for(int i=0; i<MM; i++)
                qout1[i] = expdw[i]*qin[i];
            // 3D fourier discrete transform, forward and inplace
            fft->forward(qout1,k_qin1);
            // multiply e^(-k^2 ds/6) in fourier space, in all 3 directions
            for(int i=0; i<MM_COMPLEX; i++)
                k_qin1[i] *= expf[i];
            // 3D fourier discrete transform, backword and inplace
            fft->backward(k_qin1,qout1);
            // normalization calculation and evaluate e^(-w*ds/2) in real space
            for(int i=0; i<MM; i++)
                qout1[i] *= expdw[i];
        }
        #pragma omp section
        {
            // step 2
            // evaluate e^(-w*ds/4) in real space
            for(int i=0; i<MM; i++)
                qout2[i] = expdw_half[i]*qin[i];
            // 3D fourier discrete transform, forward and inplace
            fft->forward(qout2,k_qin2);
            // multiply e^(-k^2 ds/12) in fourier space, in all 3 directions
            for(int i=0; i<MM_COMPLEX; i++)
                k_qin2[i] *= expf_half[i];
            // 3D fourier discrete transform, backword and inplace
            fft->backward(k_qin2,qout2);
            // normalization calculation and evaluate e^(-w*ds/2) in real space
            for(int i=0; i<MM; i++)
                qout2[i] *= expdw[i];
            // 3D fourier discrete transform, forward and inplace
            fft->forward(qout2,k_qin2);
            // multiply e^(-k^2 ds/12) in fourier space, in all 3 directions
            for(int i=0; i<MM_COMPLEX; i++)
                k_qin2[i] *= expf_half[i];
            // 3D fourier discrete transform, backword and inplace
            fft->backward(k_qin2,qout2);
            // normalization calculation and evaluate e^(-w*ds/4) in real space
            for(int i=0; i<MM; i++)
                qout2[i] *= expdw_half[i];
        }
    }
    for(int i=0; i<MM; i++)
        qout[i] = (4.0*qout2[i] - qout1[i])/3.0;
}

/* Get partial partition functions
* This is made for debugging and testing.
* Do NOT this at main progarams.
* */
void CpuPseudoGaussian::get_partition(double *q1_out,  double *q2_out, int n)
{
    const int MM = sb->get_MM();
    const int NN = pc->get_NN();
    
    for(int i=0; i<MM; i++)
    {
        q1_out[i] =q1[n*MM+i];
        q2_out[i] =q2[(NN-n)*MM+i];
    }
}
