/* this module defines parameters and subroutines to conduct fast
* Fourier transform (FFT) using PocketFFT. */

#include <complex>
#include "PocketFFT.h"
#include "PocketFFT3D.h"

PocketFFT3D::PocketFFT3D(std::array<int,3> nx)
{
    try
    {
        this->n_grid = nx[0]*nx[1]*nx[2];
        this->fft_normal_factor = nx[0]*nx[1]*nx[2];

        shape.push_back((long unsigned int) nx[0]);
        shape.push_back((long unsigned int) nx[1]);
        shape.push_back((long unsigned int) nx[2]);
        stride_in.push_back(sizeof(double)*nx[1]*nx[2]);
        stride_in.push_back(sizeof(double)*nx[2]);
        stride_in.push_back(sizeof(double));
        stride_out.push_back(sizeof(std::complex<double>)*(nx[2]/2+1)*nx[1]);
        stride_out.push_back(sizeof(std::complex<double>)*(nx[2]/2+1));
        stride_out.push_back(sizeof(std::complex<double>));
        axes.push_back(0);
        axes.push_back(1);
        axes.push_back(2);
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
PocketFFT3D::~PocketFFT3D()
{
}
void PocketFFT3D::forward(double *rdata, std::complex<double> *cdata)
{
    pocketfft::r2c(shape, stride_in, stride_out, axes, pocketfft::FORWARD, rdata, cdata, 1.);
}
void PocketFFT3D::backward(std::complex<double> *cdata, double *rdata)
{
    pocketfft::c2r(shape, stride_out, stride_in, axes, pocketfft::BACKWARD, cdata, rdata, 1./fft_normal_factor);
}
