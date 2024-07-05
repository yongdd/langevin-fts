#include <iostream>
#include <cmath>
#include <complex>
#include <algorithm>
#include <random>
#include <array>
#include "CudaCommon.h"
#include <cufft.h>

int main()
{
    try{
        const int II{5};
        const int M{II};
        const int M_COMPLEX{II/2+1};

        double error;
        std::complex<double> data_k[M_COMPLEX];

        //std::array<double,M> diff_sq;
        std::array<double,M_COMPLEX> diff_sq_cplx;

        cufftHandle plan_for;
        cufftDoubleComplex *d_data_k;
        double **d_data_init;
        double data_init[2*M] =
        {
            0.961913696e+0,0.623968915e+0,0.183471406e+0,0.997228140e+0,0.731257661e+0,
            0.183471406e+0,0.623968915e+0,0.731257661e+0,0.997228140e+0,0.961913696e+0,
        };

        std::complex<double> data_k_answer[M_COMPLEX] =
        {
            {3.497839818,0}, {-0.7248383037,0.4777381112}, {-0.5654030903,-0.05431399883},     
        };

        //-------------- initialize ------------
        int n_grid[1] = {II};
        cufftPlanMany(&plan_for, 1, n_grid, NULL, 1, 0, NULL, 1, 0, CUFFT_D2Z, 1);

        d_data_init = new double*[2];
        gpu_error_check(cudaMalloc((void**)&d_data_init[0], sizeof(double)*M));
        gpu_error_check(cudaMalloc((void**)&d_data_init[1], sizeof(double)*M));
        gpu_error_check(cudaMalloc((void**)&d_data_k, sizeof(cufftDoubleComplex)*M_COMPLEX));

        //---------------- Forward --------------------
        std::cout<< "Running FFT 1D" << std::endl;
        cudaMemcpy(d_data_init[1], &data_init[M], sizeof(double)*M,cudaMemcpyHostToDevice);
        cufftExecD2Z(plan_for, d_data_init[1], d_data_k);
        cudaMemcpy(data_k, d_data_k, sizeof(cufftDoubleComplex)*M_COMPLEX,cudaMemcpyDeviceToHost);   

        // for(int i=0; i<M_COMPLEX; i++)
        // {
        //     std::cout << "(" << data_k[i].real() << "," << data_k[i].imag() << ")" << std::endl;
        // }

        std::cout<< "If error is less than 1.0e-7, it is ok!" << std::endl;
        for(int i=0; i<M_COMPLEX; i++){
            diff_sq_cplx[i]  = pow(std::abs(data_k[i].real() - data_k_answer[i].real()),2);
            diff_sq_cplx[i] += pow(std::abs(data_k[i].imag() - data_k_answer[i].imag()),2);
        }
        error = sqrt(*std::max_element(diff_sq_cplx.begin(),diff_sq_cplx.end()));
        std::cout<< "FFT Forward Error: " << error << std::endl;
        if(!std::isfinite(error) || error > 1e-7)
            return -1;

        cufftDestroy(plan_for);

        cudaFree(d_data_k);
        cudaFree(d_data_init[0]);
        cudaFree(d_data_init[1]);
        delete[] d_data_init;

        return 0;
    }
    catch(std::exception& exc)
    {
        std::cout << exc.what() << std::endl;
        return -1;
    }
}