/*******************************************************************************
 * WARNING TO AI ASSISTANTS (Claude, ChatGPT, Copilot, etc.):
 * DO NOT MODIFY TEST PARAMETERS WITHOUT EXPLICIT PERMISSION FROM THE USER.
 * - NEVER increase tolerance values (e.g., 1e-7 -> 1e-6)
 * - NEVER decrease field strength or standard deviation values
 * - NEVER change grid sizes, box dimensions, or polymer parameters
 * - NEVER weaken any test conditions to make tests pass
 * These parameters are carefully calibrated. If a test fails, report the
 * failure to the user rather than modifying the test to pass.
 ******************************************************************************/

#include <iostream>
#include <cmath>
#include <numbers>
#include <complex>
#include <algorithm>
#include <random>

#include "Exception.h"
#include "FFT.h"
#ifdef USE_CPU_FFTW
#include "FftwFFT.h"
#endif

int main()
{
    try{
        const int II{5};
        const int M{II};

        double error;
        std::complex<double> data_r[M], data_k[M];
        std::array<double, M> diff_sq_cplx;

        std::complex<double> data_init[M] =
        {
            {0.2733181541,-0.0187024081}, {-0.1507659497,0.6894673618}, 
            {-0.0350475991,0.2143790085}, {-0.0175411299,0.4013954178}, 
            {0.8336831753,0.4648317909},
        };
        std::complex<double> data_k_answer[M] =
        {
            {0.9036466507,1.7513711709}, {0.6306119942,0.7863804708}, 
            {0.0144763523,-0.2002702411}, {-0.6053249486,-1.3242603125}, 
            {0.4231807219,-1.1067331286},
        };

        //-------------- initialize ------------
        std::cout<< "Initializing" << std::endl;
        std::vector<FFT<std::complex<double>> *> fft_list;
#ifdef USE_CPU_FFTW
        fft_list.push_back(new FftwFFT<std::complex<double>, 1>({II}));
#endif
        // For each platform    
        for(auto fft : fft_list){
            for(int i=0; i<M; i++)
                data_r[i] = {0.0, 0.0};
            for(int i=0; i<M; i++)
                data_k[i] = {0.0, 0.0};

            //---------------- Forward --------------------
            std::cout<< "Running FFT<std::complex<double>> 1D" << std::endl;
            fft->forward(data_init, data_k);
            std::cout<< "If error is less than 1.0e-7, it is ok!" << std::endl;
            for(int i=0; i<M; i++){
                diff_sq_cplx[i]  = pow(std::abs(data_k[i].real() - data_k_answer[i].real()),2);
                diff_sq_cplx[i] += pow(std::abs(data_k[i].imag() - data_k_answer[i].imag()),2);
            }
            error = sqrt(*std::max_element(diff_sq_cplx.begin(),diff_sq_cplx.end()));
            std::cout<< "FFT<std::complex<double>> Forward Error: " << error << std::endl;
            if(!std::isfinite(error) || error > 1e-7)
                return -1;

            //--------------- Backward --------------------
            fft->backward(data_k_answer, data_r);
            for(int i=0; i<M; i++){
                diff_sq_cplx[i]  = pow(std::abs(data_r[i].real() - data_init[i].real()),2);
                diff_sq_cplx[i] += pow(std::abs(data_r[i].imag() - data_init[i].imag()),2);
            }
            error = sqrt(*std::max_element(diff_sq_cplx.begin(),diff_sq_cplx.end()));
            std::cout<< "FFT<std::complex<double>> Backward Error: " << error << std::endl;
            if(!std::isfinite(error) || error > 1e-7)
                return -1;
            
            delete fft;
        }
        return 0;
    }
    catch(std::exception& exc)
    {
        std::cout << exc.what() << std::endl;
        return -1;
    }
}