/* this module defines parameters and subroutines to conduct fast
* Fourier transform (FFT) using math kernel library(MKL). */
#include <iostream>

#include "MklFFT1D.h"

MklFFT1D::MklFFT1D(int nx)
{
    try
    {
        MKL_LONG NX = nx;
        this->total_grid = nx;
        
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

        if (status !=0)
            std::cout << "MKL status: " << status << std::endl;

        // Compute a normalization factor
        this->fft_normal_factor = nx;
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
MklFFT1D::~MklFFT1D()
{
    int status;
    status = DftiFreeDescriptor(&hand_forward);
    status = DftiFreeDescriptor(&hand_backward);

    if (status !=0)
        std::cout << "MKL status: " << status << std::endl;
}
void MklFFT1D::forward(double *rdata, std::complex<double> *cdata)
{
    int status;
    status = DftiComputeForward(hand_forward, rdata, cdata);

    if (status !=0)
        std::cout << "MKL status: " << status << std::endl;
}
void MklFFT1D::backward(std::complex<double> *cdata, double *rdata)
{
    int status;
    status = DftiComputeBackward(hand_backward, cdata, rdata);

    if (status !=0)
        std::cout << "MKL status: " << status << std::endl;
        
    for(int i=0; i<total_grid; i++)
        rdata[i] /= fft_normal_factor;
}
