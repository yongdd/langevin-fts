#include <iostream>
#include <cmath>
#include <complex>
#include <iomanip>
#include <algorithm>
#include "FftwFFT1D.h"

int main()
{
    const int II{5};
    const int MM{II};
    const int MM_COMPLEX{II/2+1};

    double error;
    double data_r[MM];
    std::complex<double> data_k[MM_COMPLEX];
    std::array<double,MM> diff_sq;
    std::array<double,MM_COMPLEX> diff_sq_cplx;
    //-------------- initialize ------------
    std::cout<< "Initializing" << std::endl;
    FftwFFT1D fft(II);
    double data_init[MM] =
    {
        0.183471406e+0,0.623968915e+0,0.731257661e+0,0.997228140e+0,0.961913696e+0,
    };
    std::complex<double> data_k_answer[MM_COMPLEX] =
    {
        {3.497839818,0}, {-0.7248383037,0.4777381112}, {-0.5654030903,-0.05431399883},     
    };
    //---------------- Forward --------------------
    std::cout<< "Running FFTW 1D" << std::endl;
    fft.forward(data_init,data_k);
    //std::cout << std::setprecision(10);
    //for(int i=0; i<MM_COMPLEX; i++)
    //    std::cout<< data_k[i] << ", " << std::endl;
    std::cout<< "If error is less than 1.0e-7, it is ok!" << std::endl;
    for(int i=0; i<MM_COMPLEX; i++){
        diff_sq_cplx[i]  = pow(std::abs(data_k[i].real() - data_k_answer[i].real()),2);
        diff_sq_cplx[i] += pow(std::abs(data_k[i].imag() - data_k_answer[i].imag()),2);
    }
    error = sqrt(*std::max_element(diff_sq_cplx.begin(),diff_sq_cplx.end()));
    std::cout<< "FFT Forward Error: " << error << std::endl;
    if(std::isnan(error) || error > 1e-7)
        return -1;

    //--------------- Backward --------------------
    fft.backward(data_k_answer,data_r);
    for(int i=0; i<MM; i++)
        diff_sq[i] = pow(std::abs(data_r[i] - data_init[i]),2);
    error = sqrt(*std::max_element(diff_sq.begin(),diff_sq.end()));
    std::cout<< "FFT Backward Error: " << error << std::endl;
    if(std::isnan(error) || error > 1e-7)
        return -1;
    return 0;
}
