/* this module defines parameters and subroutines to conduct fast
* Fourier transform (FFT) using math kernel library(MKL). */

#ifndef MKL_FFT_3D_H_
#define MKL_FFT_3D_H_

#include <array>
#include <complex>
#include "FFT.h"
#include "mkl_service.h"
#include "mkl_dfti.h"

class MklFFT3D : public FFT
{
private:
    double fft_normal_factor; //nomalization factor FFT
    int n_grid; // the number of grids
    // pointers for forward and backward transform
    DFTI_DESCRIPTOR_HANDLE hand_forward = NULL;
    DFTI_DESCRIPTOR_HANDLE hand_backward = NULL;
public:

    MklFFT3D(std::array<int,3> nx);
    MklFFT3D(int *nx) : MklFFT3D({nx[0],nx[1],nx[2]}){};
    ~MklFFT3D();

    void forward (double *rdata, std::complex<double> *cdata) override;
    void backward(std::complex<double> *cdata, double *rdata) override;
};
#endif
