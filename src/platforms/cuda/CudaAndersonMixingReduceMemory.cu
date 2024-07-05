#include <iostream>
#include <algorithm>
#include <thrust/reduce.h>
#include "CudaCommon.h"
#include "CudaComputationBox.h"
#include "PinnedCircularBuffer.h"
#include "CudaAndersonMixingReduceMemory.h"

CudaAndersonMixingReduceMemory::CudaAndersonMixingReduceMemory(
    int n_var, int max_hist, double start_error,
    double mix_min,   double mix_init)
    :AndersonMixing(n_var, max_hist, start_error, mix_min, mix_init)
{
    try
    {
        // Number of anderson mixing steps, increases from 0 to max_hist
        n_anderson = -1;
        // Record history of w in pinned host memory
        pinned_cb_w_hist = new PinnedCircularBuffer(max_hist+1, n_var);
        // Record history of w_deriv in pinned host memory
        pinned_cb_w_deriv_hist = new PinnedCircularBuffer(max_hist+1, n_var);
        // Record history of inner_product product of w_deriv in CPU host memory
        cb_w_deriv_dots = new CircularBuffer(max_hist+1, max_hist+1);

        // define arrays for anderson mixing
        this->u_nm = new double*[max_hist];
        for(int i=0; i<max_hist; i++)
            this->u_nm[i] = new double[max_hist];
        this->v_n = new double[max_hist];
        this->a_n = new double[max_hist];
        this->w_deriv_dots = new double[max_hist+1];

        // Temporary fields arrays
        gpu_error_check(cudaMalloc((void**)&d_w_hist1,  sizeof(double)*n_var));
        gpu_error_check(cudaMalloc((void**)&d_w_hist2,  sizeof(double)*n_var));
        gpu_error_check(cudaMalloc((void**)&d_w_deriv_hist1,  sizeof(double)*n_var));
        gpu_error_check(cudaMalloc((void**)&d_w_deriv_hist2,  sizeof(double)*n_var));

        gpu_error_check(cudaMalloc((void**)&d_w_new,   sizeof(double)*n_var));
        gpu_error_check(cudaMalloc((void**)&d_w_deriv, sizeof(double)*n_var));
        gpu_error_check(cudaMalloc((void**)&d_sum,     sizeof(double)*n_var));

        // Allocate memory for cub reduction sum
        gpu_error_check(cudaMalloc((void**)&d_sum_out, sizeof(double)));
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_sum, d_sum_out, n_var);
        gpu_error_check(cudaMalloc(&d_temp_storage, temp_storage_bytes));

        // Reset_count
        reset_count();
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
CudaAndersonMixingReduceMemory::~CudaAndersonMixingReduceMemory()
{
    delete pinned_cb_w_hist;
    delete pinned_cb_w_deriv_hist;
    delete cb_w_deriv_dots;

    for (int i=0; i<max_hist; i++)
        delete[] u_nm[i];
    delete[] u_nm;
    delete[] v_n;
    delete[] a_n;
    delete[] w_deriv_dots;
    
    cudaFree(d_w_deriv);
    cudaFree(d_w_new);
    cudaFree(d_sum);
    cudaFree(d_sum_out);
    cudaFree(d_temp_storage);

    cudaFree(d_w_hist1);
    cudaFree(d_w_hist2);
    cudaFree(d_w_deriv_hist1);
    cudaFree(d_w_deriv_hist2);

}
void CudaAndersonMixingReduceMemory::reset_count()
{
    try
    {
        /* initialize mixing parameter */
        mix = mix_init;
        /* number of anderson mixing steps, increases from 0 to max_hist */
        n_anderson = -1;

        pinned_cb_w_hist->reset();
        pinned_cb_w_deriv_hist->reset();

        cb_w_deriv_dots->reset();
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

void CudaAndersonMixingReduceMemory::calculate_new_fields(
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
        
        gpu_error_check(cudaMemcpy(d_w_deriv, w_deriv, sizeof(double)*n_var, cudaMemcpyHostToDevice));
        //printf("mix: %f\n", mix);
        // Condition to start anderson mixing
        if(error_level < start_error || n_anderson >= 0)
            n_anderson = n_anderson + 1;
        if( n_anderson >= 0 )
        {
            // Number of histories to use for anderson mixing
            n_anderson = std::min(max_hist, n_anderson);
            
            // store the input and output field (the memory is used in a periodic way)
            pinned_cb_w_hist->insert(w_current);
            pinned_cb_w_deriv_hist->insert(w_deriv);

            // Evaluate w_deriv inner_product products for calculating Unm and Vn in Thompson's paper
            for(int i=0; i<= n_anderson; i++)
            {
                gpu_error_check(cudaMemcpy(d_w_hist1, pinned_cb_w_deriv_hist->get_array(i), sizeof(double)*n_var, cudaMemcpyHostToDevice));
                multi_real<<<N_BLOCKS, N_THREADS>>>(d_sum, d_w_deriv, d_w_hist1, 1.0, n_var);
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
            for(int i=0; i<n_var; i++)
                w_new[i] = w_current[i] + mix*w_deriv[i];
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
            gpu_error_check(cudaMemcpy(d_w_hist1,       pinned_cb_w_hist->get_array(0),       sizeof(double)*n_var,cudaMemcpyHostToDevice));
            gpu_error_check(cudaMemcpy(d_w_deriv_hist1, pinned_cb_w_deriv_hist->get_array(0), sizeof(double)*n_var,cudaMemcpyHostToDevice));

            gpu_error_check(cudaMemcpy(d_w_new, d_w_hist1,  sizeof(double)*n_var,cudaMemcpyDeviceToDevice));
            lin_comb<<<N_BLOCKS, N_THREADS>>>(d_w_new, 1.0, d_w_hist1, 1.0, d_w_deriv_hist1, n_var);
            for(int i=0; i<n_anderson; i++)
            {
                gpu_error_check(cudaMemcpy(d_w_hist2,       pinned_cb_w_hist->get_array(i+1),       sizeof(double)*n_var,cudaMemcpyHostToDevice));
                gpu_error_check(cudaMemcpy(d_w_deriv_hist2, pinned_cb_w_deriv_hist->get_array(i+1), sizeof(double)*n_var,cudaMemcpyHostToDevice));
                add_lin_comb<<<N_BLOCKS, N_THREADS>>>(d_w_new, a_n[i], d_w_hist2,       -a_n[i], d_w_hist1,       n_var);
                add_lin_comb<<<N_BLOCKS, N_THREADS>>>(d_w_new, a_n[i], d_w_deriv_hist2, -a_n[i], d_w_deriv_hist1, n_var);
            }
            gpu_error_check(cudaMemcpy(w_new, d_w_new, sizeof(double)*n_var,cudaMemcpyDeviceToHost));
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaAndersonMixingReduceMemory::print_array(int n, double *a)
{
    for(int i=0; i<n-1; i++)
    {
        std::cout << a[i] << ", ";
    }
    std::cout << a[n-1] << std::endl;
}
