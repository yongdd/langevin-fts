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
    try
    {
        const int II{5};
        const int JJ{4};

        const int M{II*JJ};

        double error;
        std::complex<double> data_r[M], data_k[M];
        std::array<double, M> diff_sq_cplx;

        std::complex<double> data_init[M] =
        {
            {0.7479851438,0.5113923184}, {-0.611924137,0.6688134066}, 
            {-0.2562048442,0.7156670162}, {-0.698469607,0.3809731847}, 
            {0.5071785071,-0.8416524347}, {-0.1415995644,0.9279043295}, 
            {0.1582071603,-0.2578680913}, {-0.9819601031,0.4541845031}, 
            {0.2625178548,-0.7306389533}, {0.0197493655,0.6360389997}, 
            {-0.6824309284,-0.0573651711}, {-0.588476528,-0.4983797445}, 
            {0.6192052822,-0.3934574836}, {0.582460527,0.9074145996}, 
            {0.4822692317,-0.4687287157}, {-0.6937104783,-0.9072499747}, 
            {-0.7862444092,0.3346206151}, {-0.7524654201,0.0442255985}, 
            {0.420599632,-0.7957530045}, {0.9435906198,0.9249318616}, 
        };
        std::complex<double> data_k_answer[M] =
        {
            {-1.4497226956,1.5550728596}, {4.0581392311,-1.3709348392}, 
            {4.3958879555,-5.5226406686}, {-1.6017349765,0.8595588959}, 
            {-1.1054099382,5.1776018019}, {-6.4498665458,-0.692090622}, 
            {0.0443508843,-1.1905937191}, {2.0370359645,-2.3202722334}, 
            {-0.6401006748,-0.5453888601}, {1.1391904384,-3.7814794759}, 
            {0.6683255063,0.3989231452}, {2.334602,1.8269050208}, 
            {0.0275726665,2.8851798905}, {5.9827303722,-0.6033125243}, 
            {1.7837980751,5.1801546701}, {-1.3263453051,-0.7671836679}, 
            {-0.9254065801,2.3117639377}, {1.7299575537,4.9937166227}, 
            {2.1185077968,2.0205202883}, {2.1381911477,-0.1876541545}, 
        };

        //-------------- initialize ------------
        std::cout<< "Initializing" << std::endl;
        std::vector<FFT<std::complex<double>>*> fft_list;
#ifdef USE_CPU_FFTW
        fft_list.push_back(new FftwFFT<std::complex<double>, 2>({II,JJ}));
#endif
        // For each platform    
        for(auto fft : fft_list){
            for(int i=0; i<M; i++)
                data_r[i] = {0.0, 0.0};
            for(int i=0; i<M; i++)
                data_k[i] = {0.0, 0.0};

            //---------------- Forward --------------------
            std::cout<< "Running FFT<std::complex<double>> 2D" << std::endl;
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