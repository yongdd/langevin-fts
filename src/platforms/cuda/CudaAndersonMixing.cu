#include <iostream>
#include <algorithm>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include "CudaCommon.h"
#include "CudaSimulationBox.h"
#include "CudaCircularBuffer.h"
#include "CudaAndersonMixing.h"

CudaAndersonMixing::CudaAndersonMixing(
    int n_var, int max_hist, double start_error,
    double mix_min,   double mix_init)
    :AndersonMixing(n_var, max_hist, start_error, mix_min, mix_init)
{
    try
    {
        // number of anderson mixing steps, increases from 0 to max_hist
        n_anderson = -1;
        // record hisotry of w_out in GPU device memory
        d_cb_w_out_hist = new CudaCircularBuffer(max_hist+1, n_var);
        // record hisotry of w_deriv in GPU device memory
        d_cb_w_deriv_hist = new CudaCircularBuffer(max_hist+1, n_var);
        // record hisotry of inner_product product of w_deriv in CPU host memory
        cb_w_deriv_dots = new CircularBuffer(max_hist+1, max_hist+1);

        // define arrays for anderson mixing
        this->u_nm = new double*[max_hist];
        for(int i=0; i<max_hist; i++)
            this->u_nm[i] = new double[max_hist];
        this->v_n = new double[max_hist];
        this->a_n = new double[max_hist];
        this->w_deriv_dots = new double[max_hist+1];

        // fields arrays
        gpu_error_check(cudaMalloc((void**)&d_w_deriv, sizeof(double)*n_var));
        gpu_error_check(cudaMalloc((void**)&d_w,       sizeof(double)*n_var));
        gpu_error_check(cudaMalloc((void**)&d_sum, sizeof(double)*n_var));

        // reset_count
        reset_count();
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
CudaAndersonMixing::~CudaAndersonMixing()
{
    delete d_cb_w_out_hist;
    delete d_cb_w_deriv_hist;
    delete cb_w_deriv_dots;

    for (int i=0; i<max_hist; i++)
        delete[] u_nm[i];
    delete[] u_nm;
    delete[] v_n;
    delete[] a_n;
    delete[] w_deriv_dots;
    
    cudaFree(d_w_deriv);
    cudaFree(d_w);
    cudaFree(d_sum);
}
void CudaAndersonMixing::reset_count()
{
    try
    {
        /* initialize mixing parameter */
        mix = mix_init;
        /* number of anderson mixing steps, increases from 0 to max_hist */
        n_anderson = -1;

        d_cb_w_out_hist->reset();
        d_cb_w_deriv_hist->reset();
        cb_w_deriv_dots->reset();
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

void CudaAndersonMixing::caculate_new_fields(
    double *w,
    double *w_out,
    double *w_deriv,
    double old_error_level,
    double error_level)
{
    try
    {
        const int N_BLOCKS = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();
        
        double *d_w_out_hist1;
        double *d_w_out_hist2;

        gpu_error_check(cudaMemcpy(d_w_deriv, w_deriv, sizeof(double)*n_var, cudaMemcpyHostToDevice));

        thrust::device_ptr<double> temp_gpu_ptr(d_sum);
        //printf("mix: %f\n", mix);
        // condition to start anderson mixing
        if(error_level < start_error || n_anderson >= 0)
            n_anderson = n_anderson + 1;
        if( n_anderson >= 0 )
        {
            // number of histories to use for anderson mixing
            n_anderson = std::min(max_hist, n_anderson);
            
            // store the input and output field (the memory is used in a periodic way)
            d_cb_w_out_hist->insert(w_out);
            d_cb_w_deriv_hist->insert(w_deriv);

            // evaluate w_deriv inner_product products for calculating Unm and Vn in Thompson's paper
            for(int i=0; i<= n_anderson; i++)
            {
                multi_real<<<N_BLOCKS, N_THREADS>>>(d_sum, d_w_deriv, d_cb_w_deriv_hist->get_array(i), 1.0, n_var);
                w_deriv_dots[i] = thrust::reduce(temp_gpu_ptr, temp_gpu_ptr + n_var);
            }
            //print_array(max_hist+1, w_deriv_dots);
            cb_w_deriv_dots->insert(w_deriv_dots);
        }

        // conditions to apply the simple mixing method
        if( n_anderson <= 0 )
        {
            // dynamically change mixing parameter
            if (old_error_level < error_level)
                mix = std::max(mix*0.7, mix_min);
            else
                mix = mix*1.01;

            // make a simple mixing of input and output fields for the next iteration
            for(int i=0; i<n_var; i++)
                w[i] = (1.0-mix)*w[i] + mix*w_out[i];
        }
        else
        {
            // calculate Unm and Vn
            for(int i=0; i<n_anderson; i++)
            {
                v_n[i] = cb_w_deriv_dots->get(0, 0)
                        - cb_w_deriv_dots->get(0, i+1);

                for(int j=0; j<n_anderson; j++)
                {
                    u_nm[i][j] = cb_w_deriv_dots->get(0, 0)
                                - cb_w_deriv_dots->get(0, i+1)
                                - cb_w_deriv_dots->get(0, j+1)
                                + cb_w_deriv_dots->get(std::min(i+1, j+1),
                                                    std::abs(i-j));
                }
            }
            //print_array(max_hist, v_n);
            //exit(-1);
            find_an(u_nm, v_n, a_n, n_anderson);

            // calculate the new field
            d_w_out_hist1 = d_cb_w_out_hist->get_array(0);
            gpu_error_check(cudaMemcpy(d_w, d_w_out_hist1, sizeof(double)*n_var,cudaMemcpyDeviceToDevice));
            for(int i=0; i<n_anderson; i++)
            {
                d_w_out_hist2 = d_cb_w_out_hist->get_array(i+1);
                add_lin_comb<<<N_BLOCKS, N_THREADS>>>(d_w, a_n[i], d_w_out_hist2, -a_n[i], d_w_out_hist1, n_var);
            }
            gpu_error_check(cudaMemcpy(w, d_w, sizeof(double)*n_var,cudaMemcpyDeviceToHost));
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaAndersonMixing::print_array(int n, double *a)
{
    for(int i=0; i<n-1; i++)
    {
        std::cout << a[i] << ", ";
    }
    std::cout << a[n-1] << std::endl;
}
