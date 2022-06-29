#include <iostream>
#include <cmath>
#include <complex>
#include <algorithm>
#include <random>

#include "Exception.h"
#include "FFT.h"
#ifdef USE_CPU_MKL
#include "MklFFT2D.h"
#endif
#ifdef USE_CPU_POCKET_FFT
#include "PocketFFT2D.h"
#endif

int main()
{
    try
    {
        const int II{5};
        const int JJ{4};

        const int MM{II*JJ};
        const int MM_COMPLEX{II*(JJ/2+1)};

        double error;
        double data_r[MM];
        std::complex<double> data_k[MM_COMPLEX];

        std::array<double,MM> diff_sq;
        std::array<double,MM_COMPLEX> diff_sq_cplx;

        double data_init[MM] =
        {
            0.183471406e+0,0.623968915e+0,0.731257661e+0,0.997228140e+0,0.961913696e+0,
            0.792673860e-1,0.429684069e+0,0.290531312e+0,0.453270921e+0,0.199228629e+0,
            0.754931905e-1,0.226924328e+0,0.936407886e+0,0.979392715e+0,0.464957186e+0,
            0.742653949e+0,0.368019859e+0,0.885231224e+0,0.406191773e+0,0.653096157e+0,
        };
        std::complex<double> data_k_answer[MM_COMPLEX] =
        {
            {10.6881904,0},                {0.7954998885,0.143345017},
            {-0.6668551075,0},             {0.4954041066,1.798776899},
            {-0.5050260775,0.04850456904}, {-0.4504167737,-1.947311157},
            {0.5003159972,-1.738407521},   {-0.675930056,-0.09881542923},
            {-0.9823256426,-0.6471590658}, {0.5003159972,1.738407521},
            {-0.6941925918,0.7499083742},  {-0.9823256426,0.6471590658},
            {0.4954041066,-1.798776899},   {-1.659282438,1.023353594},
            {-0.4504167737,1.947311157},
        };

        //-------------- initialize ------------
        std::cout<< "Initializing" << std::endl;
        std::vector<FFT*> fft_list;
        #ifdef USE_CPU_MKL
        fft_list.push_back(new MklFFT2D({II,JJ}));
        #endif
        #ifdef USE_CPU_POCKET_FFT
        fft_list.push_back(new PocketFFT2D({II,JJ}));
        #endif

        // For each platform    
        for(FFT* fft : fft_list){
            for(int i=0; i<MM; i++)
                data_r[i] = 0.0;
            for(int i=0; i<MM_COMPLEX; i++)
                data_k[i] = 0.0;

            //---------------- Forward --------------------
            std::cout<< "Running FFT 2D" << std::endl;
            fft->forward(data_init,data_k);
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
            fft->backward(data_k_answer,data_r);
            for(int i=0; i<MM; i++)
                diff_sq[i] = pow(std::abs(data_r[i] - data_init[i]),2);
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