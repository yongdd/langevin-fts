/*-------------------------------------------------------------
* This class conduct fast Fourier transform (FFT) using the
* Fastest Fourier transform in the West (FFTW)
*------------------------------------------------------------*/

#ifndef FFTW_FFT_1D_H_
#define FFTW_FFT_1D_H_

#include <array>
#include <complex>
#include "FFT.h"
#include "fftw3.h"

class FftwFFT1D : public FFT
{
private:
    double fft_normal_factor; //nomalization factor FFT
    int n_grid; // the number of grids
    // pointers for forward and backward transform
    fftw_plan plan_forward, plan_backward;
public:

    FftwFFT1D(int nx);
    ~FftwFFT1D();

    void forward (double *rdata, std::complex<double> *cdata) override;
    void backward(std::complex<double> *cdata, double *rdata) override;
};
#endif
