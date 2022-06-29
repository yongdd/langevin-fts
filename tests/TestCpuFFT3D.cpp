#include <iostream>
#include <cmath>
#include <complex>
#include <algorithm>
#include <random>

#include "Exception.h"
#include "FFT.h"
#ifdef USE_CPU_MKL
#include "MklFFT3D.h"
#endif
#ifdef USE_CPU_POCKET_FFT
#include "PocketFFT3D.h"
#endif

int main()
{
    try
    {
        const int II{5};
        const int JJ{4};
        const int KK{3};

        const int MM{II*JJ*KK};
        const int MM_COMPLEX{II*JJ*(KK/2+1)};

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
            0.567929080e-1,0.568028857e+0,0.144986181e+0,0.466158777e+0,0.573327733e+0,
            0.136324723e+0,0.819010407e+0,0.271218167e+0,0.626224101e+0,0.398109186e-1,
            0.860031651e+0,0.338153865e+0,0.688078522e+0,0.564682952e+0,0.222924187e+0,
            0.306816449e+0,0.316316038e+0,0.640568415e+0,0.702342408e+0,0.632135481e+0,
            0.649402777e+0,0.647100865e+0,0.370402133e+0,0.691313864e+0,0.447870566e+0,
            0.757298851e+0,0.586173682e+0,0.766745717e-1,0.504185402e+0,0.812016428e+0,
            0.217988206e+0,0.273487202e+0,0.937672578e+0,0.570540523e+0,0.409071185e+0,
            0.391548274e-1,0.663478965e+0,0.260755447e+0,0.503943226e+0,0.979481790e+0
        };
        std::complex<double> data_k_answer[MM_COMPLEX] =
        {
            {30.0601362322000,0.000000000000000e+0}, {0.353642310400000,-0.656637882635999},
            {1.84441281060000,-2.74233574840000}, {-2.46150775700102,4.133457522749440e-2},
            {0.817180262600000,0.000000000000000e+0}, {-0.825032729800000,0.726831889225593},
            {1.84441281060000,2.74233574840000}, {0.732077908401022,-1.00081656417251},
            {-0.458084677221565,0.153852624198044}, {-0.144717578762292,-2.28803500866730},
            {0.705009791908085,-2.01933903816772}, {2.964764771912920e-2,-2.25308567091476},
            {-0.973844201363205,-2.06176730216391}, {-0.997469264485567,-0.602564694796768},
            {1.40462480063246,1.01369298185703}, {1.75359066116106,0.326376682455054},
            {-1.44138430512843,-1.31255361296267}, {-0.744097894016966,-0.480832819344072},
            {-2.15508603929787,-1.93166708304686}, {-0.438393767508181,-1.10483993164215},
            {0.995576356313205,0.780345054198534}, {-3.86510390713435,4.55728094860600},
            {2.709703615732217e-2,-2.321947897161802e-2}, {0.208459929713295,2.48927791839840},
            {-1.44138430512843,1.31255361296267}, {1.68993207144876,-0.218131499149520},
            {2.709703615732217e-2,2.321947897161802e-2}, {-2.04216524355430,2.34581169104130},
            {0.995576356313205,-0.780345054198534}, {2.469425255987412e-2,1.01595227927158},
            {-2.15508603929787,1.93166708304686}, {-0.744464511100393,-5.438522180051475e-2},
            {-0.458084677221565,-0.153852624198044}, {-0.713266212819504,1.64663971056941},
            {1.40462480063246,-1.01369298185703}, {-2.32489174723348,-1.41241859006073},
            {-0.973844201363205,2.06176730216391}, {0.857829657610043,-1.36198877135770},
            {0.705009791908085,2.01933903816772}, {-0.231601465597141,0.142526551270715}
        };

        //-------------- initialize ------------
        std::cout<< "Initializing" << std::endl;
        std::vector<FFT*> fft_list;
        #ifdef USE_CPU_MKL
        fft_list.push_back(new MklFFT3D({II,JJ,KK}));
        #endif
        #ifdef USE_CPU_POCKET_FFT
        fft_list.push_back(new PocketFFT3D({II,JJ,KK}));
        #endif

        // For each platform    
        for(FFT* fft : fft_list){
            for(int i=0; i<MM; i++)
                data_r[i] = 0.0;
            for(int i=0; i<MM_COMPLEX; i++)
                data_k[i] = 0.0;

            //---------------- Forward --------------------
            std::cout<< "Running FFT 3D" << std::endl;
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
