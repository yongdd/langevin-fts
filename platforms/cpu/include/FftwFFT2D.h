/*-------------------------------------------------------------
* This class conduct fast Fourier transform (FFT) using the
* Fastest Fourier transform in the West (FFTW)
*------------------------------------------------------------*/

#ifndef FFTW_FFT_2D_H_
#define FFTW_FFT_2D_H_

#include <array>
#include <complex>
#include "FFT.h"
#include "fftw3.h"

class FftwFFT2D : public FFT
{
private:
    double fft_normal_factor; //nomalization factor FFT
    int MM; // the number of total grids
    // pointers for forward and backward transform
    fftw_plan plan_forward, plan_backward;
public:

    FftwFFT2D(std::array<int,2> nx);
    FftwFFT2D(int *nx) : FftwFFT2D({nx[0],nx[1]}){};
    ~FftwFFT2D();

    void forward (double *rdata, std::complex<double> *cdata) override;
    void backward(std::complex<double> *cdata, double *rdata) override;
};
#endif
