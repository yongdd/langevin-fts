/* this module defines parameters and subroutines to conduct fast
* Fourier transform (FFT) using PocketFFT. */

#include <complex>
#include "PocketFFT.h"
#include "PocketFFT1D.h"

PocketFFT1D::PocketFFT1D(int nx)
{
    try
    {
        this->n_grid = nx;
        this->fft_normal_factor = nx;
        shape.push_back((long unsigned int) nx);
        stride_in.push_back(sizeof(double));
        stride_out.push_back(sizeof(std::complex<double>));
        axes.push_back(0);
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
PocketFFT1D::~PocketFFT1D()
{
}
void PocketFFT1D::forward(double *rdata, std::complex<double> *cdata)
{
    pocketfft::r2c(shape, stride_in, stride_out, axes, pocketfft::FORWARD, rdata, cdata, 1.);
}
void PocketFFT1D::backward(std::complex<double> *cdata, double *rdata)
{
    pocketfft::c2r(shape, stride_out, stride_in, axes, pocketfft::BACKWARD, cdata, rdata, 1./fft_normal_factor);
}
