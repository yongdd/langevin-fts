/* this module defines parameters and subroutines to conduct fast
* Fourier transform (FFT) using math kernel library(MKL). */

#ifndef MKL_FFT_2D_H_
#define MKL_FFT_2D_H_

#include <array>
#include <complex>
#include "FFT.h"
#include "mkl_service.h"
#include "mkl_dfti.h"

class MklFFT2D : public FFT
{
private:
    double fft_normal_factor; //normalization factor FFT
    int total_grid; // the number of grids
    // Pointers for forward and backward transform
    DFTI_DESCRIPTOR_HANDLE hand_forward = NULL;
    DFTI_DESCRIPTOR_HANDLE hand_backward = NULL;
public:

    MklFFT2D(std::array<int,2> nx);
    MklFFT2D(int *nx) : MklFFT2D({nx[0],nx[1]}){};
    ~MklFFT2D();

    void forward (double *rdata, std::complex<double> *cdata) override;
    void backward(std::complex<double> *cdata, double *rdata) override;
};
#endif
