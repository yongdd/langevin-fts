/* this module defines parameters and subroutines to conduct fast
* Fourier transform (FFT) using math kernel library(MKL). */

#ifndef MKL_FFT_1D_H_
#define MKL_FFT_1D_H_

#include <array>
#include <complex>
#include "FFT.h"
#include "mkl_service.h"
#include "mkl_dfti.h"

class MklFFT1D : public FFT
{
private:
    double fft_normal_factor; //normalization factor FFT
    int n_grid; // the number of grids
    // Pointers for forward and backward transform
    DFTI_DESCRIPTOR_HANDLE hand_forward = NULL;
    DFTI_DESCRIPTOR_HANDLE hand_backward = NULL;
public:
    MklFFT1D(int nx);
    ~MklFFT1D();

    void forward (double *rdata, std::complex<double> *cdata) override;
    void backward(std::complex<double> *cdata, double *rdata) override;
};
#endif
