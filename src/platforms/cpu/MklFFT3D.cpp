/* this module defines parameters and subroutines to conduct fast
* Fourier transform (FFT) using math kernel library(MKL). */

#include "MklFFT3D.h"

MklFFT3D::MklFFT3D(std::array<int,3> nx)
{
    try
    {
        MKL_LONG NX[3] = {nx[0],nx[1],nx[2]};
        this->n_grid = nx[0]*nx[1]*nx[2];
        
        // Execution status
        MKL_LONG status{0};

        // Strides describe data layout in real and conjugate-even domain
        MKL_LONG rs[4] = {0, nx[1]*nx[2], nx[2], 1};
        MKL_LONG cs[4] = {0, nx[1]*(nx[2]/2+1), nx[2]/2+1, 1};

        status = DftiCreateDescriptor(&hand_forward,  DFTI_DOUBLE, DFTI_REAL, 3, NX );
        status = DftiSetValue(hand_forward, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
        status = DftiSetValue(hand_forward, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
        status = DftiSetValue(hand_forward, DFTI_INPUT_STRIDES, rs);
        status = DftiSetValue(hand_forward, DFTI_OUTPUT_STRIDES, cs);
        status = DftiCommitDescriptor(hand_forward);

        status = DftiCreateDescriptor(&hand_backward, DFTI_DOUBLE, DFTI_REAL, 3, NX );
        status = DftiSetValue(hand_backward, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
        status = DftiSetValue(hand_backward, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
        status = DftiSetValue(hand_backward, DFTI_INPUT_STRIDES, cs);
        status = DftiSetValue(hand_backward, DFTI_OUTPUT_STRIDES, rs);
        status = DftiCommitDescriptor(hand_backward);

        // compute a normalization factor
        this->fft_normal_factor = nx[0]*nx[1]*nx[2];
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
MklFFT3D::~MklFFT3D()
{
    int status;
    status = DftiFreeDescriptor(&hand_forward);
    status = DftiFreeDescriptor(&hand_backward);
}
void MklFFT3D::forward(double *rdata, std::complex<double> *cdata)
{
    int status;
    status = DftiComputeForward(hand_forward, rdata, cdata);
}
void MklFFT3D::backward(std::complex<double> *cdata, double *rdata)
{
    int status;
    status = DftiComputeBackward(hand_backward, cdata, rdata);
    for(int i=0; i<n_grid; i++)
        rdata[i] /= fft_normal_factor;
}
