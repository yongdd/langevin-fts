#include <complex>
#include "FftwFFT.h"

FftwFFT::FftwFFT(std::array<int,3> nx)
{
    this->MM = nx[0]*nx[1]*nx[2];

    // dummpy arrays for FFTW_Plan. need to find a better way
    double *data_in_dummpy = new double[this->MM];
    std::complex<double>* data_out_dummpy = new std::complex<double>[nx[0]*nx[1]*(nx[2]/2+1)];

    plan_forward =  fftw_plan_dft_r2c_3d(
                        nx[0],nx[1],nx[2]
                        ,data_in_dummpy,
                        reinterpret_cast<fftw_complex*> (data_out_dummpy),
                        FFTW_MEASURE);
    plan_backward = fftw_plan_dft_c2r_3d(
                        nx[0],nx[1],nx[2],
                        reinterpret_cast<fftw_complex *> (data_out_dummpy),
                        data_in_dummpy,
                        FFTW_MEASURE); //FFTW_MEASURE, FFTW_ESTIMATE

    delete[] data_in_dummpy;
    delete[] data_out_dummpy;

    // compute a normalization factor
    this->fft_normal_factor = nx[0]*nx[1]*nx[2];
}
FftwFFT::~FftwFFT()
{
    fftw_destroy_plan(plan_forward);
    fftw_destroy_plan(plan_backward);
}
void FftwFFT::forward(double *rdata, std::complex<double> *cdata)
{
    fftw_execute_dft_r2c(plan_forward,
                         rdata, reinterpret_cast<fftw_complex *>(cdata));
}
void FftwFFT::backward(std::complex<double> *cdata, double *rdata)
{
    fftw_execute_dft_c2r(plan_backward,
                         reinterpret_cast<fftw_complex *>(cdata), rdata);
    for(int i=0; i<MM; i++)
        rdata[i] /= fft_normal_factor;
}
