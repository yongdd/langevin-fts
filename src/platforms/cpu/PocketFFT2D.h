/* this module defines parameters and subroutines to conduct fast
* Fourier transform (FFT) using PocketFFT. */

#ifndef POCKET_FFT_2D_H_
#define POCKET_FFT_2D_H_

#include <complex>
#include "FFT.h"
#include "PocketFFT.h"

class PocketFFT2D : public FFT
{
private:
    double fft_normal_factor; //nomalization factor FFT
    int n_grid; // the number of grids

    pocketfft::shape_t shape;
    pocketfft::stride_t stride_in, stride_out;
    pocketfft::shape_t axes;

public:
    PocketFFT2D(std::array<int,2> nx);
    PocketFFT2D(int *nx) : PocketFFT2D({nx[0],nx[1]}){};
    ~PocketFFT2D();

    void forward (double *rdata, std::complex<double> *cdata) override;
    void backward(std::complex<double> *cdata, double *rdata) override;
};
#endif
