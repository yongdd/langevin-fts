/* this module defines parameters and subroutines to conduct fast
* Fourier transform (FFT) using PocketFFT. */

#ifndef POCKET_FFT_1D_H_
#define POCKET_FFT_1D_H_

#include <complex>
#include "FFT.h"
#include "PocketFFT.h"

class PocketFFT1D : public FFT
{
private:
    double fft_normal_factor; //nomalization factor FFT
    int n_grid; // the number of grids

    pocketfft::shape_t shape;
    pocketfft::stride_t stride_in, stride_out;
    pocketfft::shape_t axes;

public:
    PocketFFT1D(int nx);
    ~PocketFFT1D();

    void forward (double *rdata, std::complex<double> *cdata) override;
    void backward(std::complex<double> *cdata, double *rdata) override;
};
#endif
