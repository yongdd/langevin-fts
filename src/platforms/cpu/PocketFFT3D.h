/* this module defines parameters and subroutines to conduct fast
* Fourier transform (FFT) using PocketFFT. */

#ifndef POCKET_FFT_3D_H_
#define POCKET_FFT_3D_H_

#include <complex>
#include "FFT.h"
#include "PocketFFT.h"

class PocketFFT3D : public FFT
{
private:
    double fft_normal_factor; //nomalization factor FFT
    int n_grid; // the number of grids

    pocketfft::shape_t shape;
    pocketfft::stride_t stride_in, stride_out;
    pocketfft::shape_t axes;

public:
    PocketFFT3D(std::array<int,3> nx);
    PocketFFT3D(int *nx) : PocketFFT3D({nx[0],nx[1],nx[2]}){};
    ~PocketFFT3D();

    void forward (double *rdata, std::complex<double> *cdata) override;
    void backward(std::complex<double> *cdata, double *rdata) override;
};
#endif
