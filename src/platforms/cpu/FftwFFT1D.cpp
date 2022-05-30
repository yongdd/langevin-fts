#include <complex>
#include "FftwFFT1D.h"

FftwFFT1D::FftwFFT1D(int nx)
{
    try
    {
        this->n_grid = nx;

        // dummpy arrays for FFTW_Plan. need to find a better way
        double *data_in_dummpy = new double[this->n_grid];
        std::complex<double>* data_out_dummpy = new std::complex<double>[(nx/2+1)];

        plan_forward =  fftw_plan_dft_r2c_1d(
                            nx,
                            data_in_dummpy,
                            reinterpret_cast<fftw_complex*> (data_out_dummpy),
                            FFTW_MEASURE);
        plan_backward = fftw_plan_dft_c2r_1d(
                            nx,
                            reinterpret_cast<fftw_complex *> (data_out_dummpy),
                            data_in_dummpy,
                            FFTW_MEASURE); //FFTW_MEASURE, FFTW_ESTIMATE

        delete[] data_in_dummpy;
        delete[] data_out_dummpy;

        // compute a normalization factor
        this->fft_normal_factor = nx;
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
FftwFFT1D::~FftwFFT1D()
{
    fftw_destroy_plan(plan_forward);
    fftw_destroy_plan(plan_backward);
}
void FftwFFT1D::forward(double *rdata, std::complex<double> *cdata)
{
    fftw_execute_dft_r2c(plan_forward,
                         rdata, reinterpret_cast<fftw_complex *>(cdata));
}
void FftwFFT1D::backward(std::complex<double> *cdata, double *rdata)
{
    fftw_execute_dft_c2r(plan_backward,
                         reinterpret_cast<fftw_complex *>(cdata), rdata);
    for(int i=0; i<n_grid; i++)
        rdata[i] /= fft_normal_factor;
}
