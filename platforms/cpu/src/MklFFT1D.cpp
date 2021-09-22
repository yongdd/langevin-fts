/* this module defines parameters and subroutines to conduct fast
* Fourier transform (FFT) using math kernel library(MKL). */
#include "MklFFT1D.h"

MklFFT1D::MklFFT1D(int nx)
{
    MKL_LONG NX = nx;
    this->MM = nx;
    
    // Execution status
    MKL_LONG status{0};

    status = DftiCreateDescriptor(&hand_forward,  DFTI_DOUBLE, DFTI_REAL, 1, NX );
    status = DftiSetValue(hand_forward, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    status = DftiSetValue(hand_forward, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
    status = DftiCommitDescriptor(hand_forward);

    status = DftiCreateDescriptor(&hand_backward, DFTI_DOUBLE, DFTI_REAL, 1, NX );
    status = DftiSetValue(hand_backward, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    status = DftiSetValue(hand_backward, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
    status = DftiCommitDescriptor(hand_backward);

    // compute a normalization factor
    this->fft_normal_factor = nx;
}
MklFFT1D::~MklFFT1D()
{
    int status;
    status = DftiFreeDescriptor(&hand_forward);
    status = DftiFreeDescriptor(&hand_backward);
}
void MklFFT1D::forward(double *rdata, std::complex<double> *cdata)
{
    int status;
    status = DftiComputeForward(hand_forward, rdata, cdata);
}
void MklFFT1D::backward(std::complex<double> *cdata, double *rdata)
{
    int status;
    status = DftiComputeBackward(hand_backward, cdata, rdata);
    for(int i=0; i<MM; i++)
        rdata[i] /= fft_normal_factor;
}