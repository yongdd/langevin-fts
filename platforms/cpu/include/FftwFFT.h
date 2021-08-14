/*-------------------------------------------------------------
* This class conduct fast Fourier transform (FFT) using the
* Fastest Fourier transform in the West (FFTW)
*------------------------------------------------------------*/

#ifndef FFTW_FFT_H_
#define FFTW_FFT_H_

#include <array>
#include <complex>
#include "FFT.h"
#include "fftw3.h"

class FftwFFT : public FFT
{
private:
    double fft_normal_factor; //nomalization factor FFT
    int MM; // the number of total grids
    // pointers for forward and backward transform
    fftw_plan plan_forward, plan_backward;
public:

    FftwFFT(std::array<int,3> nx);
    FftwFFT(int *nx) : FftwFFT({nx[0],nx[1],nx[2]}){};
    ~FftwFFT();

    void forward (double *rdata, std::complex<double> *cdata) override;
    void backward(std::complex<double> *cdata, double *rdata) override;
};
#endif
