#include <iostream>
#include <algorithm>
#include <thrust/reduce.h>
#include "CudaCommon.h"
#include "CudaComputationBox.h"
#include "CudaCircularBuffer.h"
#include "CudaAndersonMixing.h"

CudaAndersonMixing::CudaAndersonMixing(
    int n_var, int max_hist, double start_error,
    double mix_min,   double mix_init)
    :AndersonMixing(n_var, max_hist, start_error, mix_min, mix_init)
{
    try
    {
        // const int N_GPUS = CudaCommon::get_instance().get_n_gpus();
        // gpu_error_check(cudaSetDevice(0));

        // Number of anderson mixing steps, increases from 0 to max_hist
        n_anderson = -1;
        // Record history of w in GPU device memory
        d_cb_w_hist = new CudaCircularBuffer(max_hist+1, n_var);
        // Record history of w_deriv in GPU device memory
        d_cb_w_deriv_hist = new CudaCircularBuffer(max_hist+1, n_var);
        // Record history of inner_product product of w_deriv in CPU host memory
        cb_w_deriv_dots = new CircularBuffer(max_hist+1, max_hist+1);

        // define arrays for anderson mixing
        this->u_nm = new double*[max_hist];
        for(int i=0; i<max_hist; i++)
            this->u_nm[i] = new double[max_hist];
        this->v_n = new double[max_hist];
        this->a_n = new double[max_hist];
        this->w_deriv_dots = new double[max_hist+1];

        // Create streams
        gpu_error_check(cudaStreamCreate(&streams[0])); // for kernel execution
        gpu_error_check(cudaStreamCreate(&streams[1])); // for memcpy

        // Fields arrays
        gpu_error_check(cudaMalloc((void**)&d_w_current, sizeof(double)*n_var));
        gpu_error_check(cudaMalloc((void**)&d_w_new,   sizeof(double)*n_var));
        gpu_error_check(cudaMalloc((void**)&d_w_deriv, sizeof(double)*n_var));
        gpu_error_check(cudaMalloc((void**)&d_sum,     sizeof(double)*n_var));
        gpu_error_check(cudaMalloc((void**)&d_sum_out, sizeof(double)));

        // Allocate memory for cub reduction sum
        d_temp_storage = nullptr; 
        temp_storage_bytes = 0;
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_sum, d_sum_out, n_var, streams[0]);
        gpu_error_check(cudaMalloc(&d_temp_storage, temp_storage_bytes));

        // Reset_count
        reset_count();
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
CudaAndersonMixing::~CudaAndersonMixing()
{
    delete d_cb_w_hist;
    delete d_cb_w_deriv_hist;
    delete cb_w_deriv_dots;

    for (int i=0; i<max_hist; i++)
        delete[] u_nm[i];
    delete[] u_nm;
    delete[] v_n;
    delete[] a_n;
    delete[] w_deriv_dots;

    cudaFree(d_w_current);
    cudaFree(d_w_deriv);
    cudaFree(d_w_new);
    cudaFree(d_sum);
    cudaFree(d_sum_out);
    cudaFree(d_temp_storage);

    // Destroy streams
    cudaStreamDestroy(streams[0]);
    cudaStreamDestroy(streams[1]);
}
void CudaAndersonMixing::reset_count()
{
    try
    {
        /* initialize mixing parameter */
        mix = mix_init;
        /* number of anderson mixing steps, increases from 0 to max_hist */
        n_anderson = -1;

        d_cb_w_hist->reset();
        d_cb_w_deriv_hist->reset();
        cb_w_deriv_dots->reset();
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

void CudaAndersonMixing::calculate_new_fields(
    double *w_new,
    double *w_current,
    double *w_deriv,
    double old_error_level,
    double error_level)
{
    try
    {
        const int N_BLOCKS = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();
        
        double *d_w_hist1;
        double *d_w_hist2;
        double *d_w_deriv_hist1;
        double *d_w_deriv_hist2;

        gpu_error_check(cudaMemcpy(d_w_deriv, w_deriv,  sizeof(double)*n_var, cudaMemcpyHostToDevice));
        gpu_error_check(cudaMemcpy(d_w_current, w_current, sizeof(double)*n_var, cudaMemcpyHostToDevice));
        // If (N_GPUS > 1)
        //     gpu_error_check(cudaMemcpy(d_w_deriv[1], d_w_deriv[0], sizeof(double)*n_var, cudaMemcpyDeviceToDevice));

        //printf("mix: %f\n", mix);
        // Condition to start anderson mixing
        if(error_level < start_error || n_anderson >= 0)
            n_anderson = n_anderson + 1;
        if( n_anderson >= 0 )
        {
            // Number of histories to use for anderson mixing
            n_anderson = std::min(max_hist, n_anderson);
            
            // store the input and output field (the memory is used in a periodic way)
            d_cb_w_hist->insert(d_w_current);
            d_cb_w_deriv_hist->insert(d_w_deriv);

            // Evaluate w_deriv inner_product products for calculating Unm and Vn in Thompson's paper
            for(int i=0; i<=n_anderson; i++)
            {
                ker_multi<<<N_BLOCKS, N_THREADS>>>(d_sum, d_w_deriv, d_cb_w_deriv_hist->get_array(i), 1.0, n_var);
                cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_sum, d_sum_out, n_var);
                gpu_error_check(cudaMemcpy(&w_deriv_dots[i], d_sum_out, sizeof(double),cudaMemcpyDeviceToHost));
            }

            //print_array(max_hist+1, w_deriv_dots);
            cb_w_deriv_dots->insert(w_deriv_dots);
        }

        // Conditions to apply the simple mixing method
        if( n_anderson <= 0 )
        {
            // dynamically change mixing parameter
            if (old_error_level < error_level)
                mix = std::max(mix*0.7, mix_min);
            else
                mix = mix*1.01;

            // Make a simple mixing of input and output fields for the next iteration
            ker_lin_comb<<<N_BLOCKS, N_THREADS>>>(d_w_new, 1.0, d_w_current, mix, d_w_deriv, n_var);
            gpu_error_check(cudaMemcpy(w_new, d_w_new, sizeof(double)*n_var, cudaMemcpyDeviceToHost));
        }
        else
        {
            // Calculate Unm and Vn
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

            // Calculate the new field
            d_w_hist1 = d_cb_w_hist->get_array(0);
            d_w_deriv_hist1 = d_cb_w_deriv_hist->get_array(0);
            gpu_error_check(cudaMemcpy(d_w_new, d_w_hist1, sizeof(double)*n_var,cudaMemcpyDeviceToDevice));
            ker_lin_comb<<<N_BLOCKS, N_THREADS>>>(d_w_new, 1.0, d_w_hist1, 1.0, d_w_deriv_hist1, n_var);
            for(int i=0; i<n_anderson; i++)
            {
                d_w_hist2 = d_cb_w_hist->get_array(i+1);
                d_w_deriv_hist2 = d_cb_w_deriv_hist->get_array(i+1);
                ker_add_lin_comb<<<N_BLOCKS, N_THREADS>>>(d_w_new, a_n[i], d_w_hist2,       -a_n[i], d_w_hist1,       n_var);
                ker_add_lin_comb<<<N_BLOCKS, N_THREADS>>>(d_w_new, a_n[i], d_w_deriv_hist2, -a_n[i], d_w_deriv_hist1, n_var);
            }
            gpu_error_check(cudaMemcpy(w_new, d_w_new, sizeof(double)*n_var,cudaMemcpyDeviceToHost));
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
