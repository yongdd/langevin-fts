#include <iostream>
#include <cmath>
#include <complex>
#include <algorithm>
#include <random>

#include "Exception.h"
#include "FFT.h"
#ifdef USE_CPU_MKL
#include "MklFFT1D.h"
#endif
#ifdef USE_CPU_POCKET_FFT
#include "PocketFFT1D.h"
#endif

int main()
{
    try{
        const int II{5};
        const int MM{II};
        const int MM_COMPLEX{II/2+1};

        double error;
        double data_r[MM];
        std::complex<double> data_k[MM_COMPLEX];

        std::array<double,MM> diff_sq;
        std::array<double,MM_COMPLEX> diff_sq_cplx;

        double data_init[MM] =
        {
            0.183471406e+0,0.623968915e+0,0.731257661e+0,0.997228140e+0,0.961913696e+0,
        };
        std::complex<double> data_k_answer[MM_COMPLEX] =
        {
            {3.497839818,0}, {-0.7248383037,0.4777381112}, {-0.5654030903,-0.05431399883},     
        };

        //-------------- initialize ------------
        std::cout<< "Initializing" << std::endl;
        std::vector<FFT*> fft_list;
        #ifdef USE_CPU_MKL
        fft_list.push_back(new MklFFT1D({II}));
        #endif
        #ifdef USE_CPU_POCKET_FFT
        fft_list.push_back(new PocketFFT1D({II}));
        #endif

        // For each platform    
        for(FFT* fft : fft_list){
            for(int i=0; i<MM; i++)
                data_r[i] = 0.0;
            for(int i=0; i<MM_COMPLEX; i++)
                data_k[i] = 0.0;

            //---------------- Forward --------------------
            std::cout<< "Running FFT 1D" << std::endl;
            fft->forward(data_init,data_k);
            std::cout<< "If error is less than 1.0e-7, it is ok!" << std::endl;
            for(int i=0; i<MM_COMPLEX; i++){
                diff_sq_cplx[i]  = pow(std::abs(data_k[i].real() - data_k_answer[i].real()),2);
                diff_sq_cplx[i] += pow(std::abs(data_k[i].imag() - data_k_answer[i].imag()),2);
                std::cout << data_k[i].real() << " " << data_k_answer[i].real() << std::endl;
                std::cout << data_k[i].imag() << " " << data_k_answer[i].imag() << std::endl;
            }
            error = sqrt(*std::max_element(diff_sq_cplx.begin(),diff_sq_cplx.end()));
            std::cout<< "FFT Forward Error: " << error << std::endl;
            //if(std::isnan(error) || error > 1e-7)
            //    return -1;

            //--------------- Backward --------------------
            fft->backward(data_k,data_r);
            for(int i=0; i<MM; i++){
                diff_sq[i] = pow(std::abs(data_r[i] - data_init[i]),2);
                std::cout << data_r[i] << " " << data_init[i] << std::endl;
            }
            error = sqrt(*std::max_element(diff_sq.begin(),diff_sq.end()));
            std::cout<< "FFT Backward Error: " << error << std::endl;
            if(std::isnan(error) || error > 1e-7)
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