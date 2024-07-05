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
        const int N_GPUS = CudaCommon::get_instance().get_n_gpus();

        gpu_error_check(cudaSetDevice(0));
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
        for(int gpu=0; gpu<N_GPUS; gpu++)
        {
            gpu_error_check(cudaSetDevice(gpu));
            gpu_error_check(cudaStreamCreate(&streams[gpu][0])); // for kernel execution
            gpu_error_check(cudaStreamCreate(&streams[gpu][1])); // for memcpy
        }

        // Fields arrays
        gpu_error_check(cudaSetDevice(0));
        gpu_error_check(cudaMalloc((void**)&d_w_current, sizeof(double)*n_var));
        if (N_GPUS > 1)
        {
            gpu_error_check(cudaSetDevice(1));
            gpu_error_check(cudaMalloc((void**)&d_w_deriv_device_1[0], sizeof(double)*n_var));  // prev
            gpu_error_check(cudaMalloc((void**)&d_w_deriv_device_1[1], sizeof(double)*n_var));  // next
        }

        for(int gpu=0; gpu<N_GPUS; gpu++)
        {
            gpu_error_check(cudaSetDevice(gpu));
            gpu_error_check(cudaMalloc((void**)&d_w_new[gpu],   sizeof(double)*n_var));
            gpu_error_check(cudaMalloc((void**)&d_w_deriv[gpu], sizeof(double)*n_var));
            gpu_error_check(cudaMalloc((void**)&d_sum[gpu],     sizeof(double)*n_var));
            gpu_error_check(cudaMalloc((void**)&d_sum_out[gpu], sizeof(double)));
        }

        // Allocate memory for cub reduction sum
        for(int gpu=0; gpu<N_GPUS; gpu++)
        {
            gpu_error_check(cudaSetDevice(gpu));
            d_temp_storage[gpu] = nullptr; 
            temp_storage_bytes[gpu] = 0;
            cub::DeviceReduce::Sum(d_temp_storage[gpu], temp_storage_bytes[gpu], d_sum[gpu], d_sum_out[gpu], n_var, streams[gpu][0]);
            gpu_error_check(cudaMalloc(&d_temp_storage[gpu], temp_storage_bytes[gpu]));
        }        
        // Reset_count
        reset_count();
        
        gpu_error_check(cudaSetDevice(0));
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
CudaAndersonMixing::~CudaAndersonMixing()
{
    const int N_GPUS = CudaCommon::get_instance().get_n_gpus();

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
    for(int gpu=0; gpu<N_GPUS; gpu++)
    {
        cudaFree(d_w_deriv[gpu]);
        cudaFree(d_w_new[gpu]);
        cudaFree(d_sum[gpu]);
        cudaFree(d_sum_out[gpu]);
        cudaFree(d_temp_storage[gpu]);
    }

    if (N_GPUS > 1)
    {
        cudaFree(d_w_deriv_device_1[0]);
        cudaFree(d_w_deriv_device_1[1]);
    }

    // Destroy streams
    for(int gpu=0; gpu<N_GPUS; gpu++)
    {
        cudaStreamDestroy(streams[gpu][0]);
        cudaStreamDestroy(streams[gpu][1]);
    }
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
        const int N_GPUS = CudaCommon::get_instance().get_n_gpus();
        
        double *d_w_hist1;
        double *d_w_hist2;
        double *d_w_deriv_hist1;
        double *d_w_deriv_hist2;

        gpu_error_check(cudaSetDevice(0));
        gpu_error_check(cudaMemcpy(d_w_deriv[0], w_deriv,  sizeof(double)*n_var, cudaMemcpyHostToDevice));
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
            d_cb_w_deriv_hist->insert(d_w_deriv[0]);

            // If(N_GPUS > 1)
            // {
            //     int prev, next;
            //     prev = 0;
            //     next = 1;

            //     // Copy memory from device 0 to device 1
            //     const int idx_gpu_1 = 1;
            //     if(idx_gpu_1 <= n_anderson)
            //     {
            //         gpu_error_check(cudaMemcpy(d_w_deriv_device_1[prev], d_cb_w_deriv_hist->get_array(idx_gpu_1),
            //                 sizeof(double)*n_var,cudaMemcpyDeviceToDevice));
            //     }

            //     // Evaluate w_deriv inner_product products for calculating Unm and Vn in Thompson's paper
            //     int n_anderson_odd = n_anderson;
            //     if (n_anderson%2 == 0)
            //         n_anderson_odd -= 1;

            //     for(int i=0; i<= n_anderson_odd; i+=2)
            //     {
            //         const int idx_gpu_0 = i;
            //         const int idx_gpu_1 = i+1;
            //         const int idx_next_gpu_1 = i+3;

            //         // Multiply
            //         gpu_error_check(cudaSetDevice(0));
            //         multi_real<<<N_BLOCKS, N_THREADS, 0, streams[0][0]>>>(d_sum[0], d_w_deriv[0], d_cb_w_deriv_hist->get_array(idx_gpu_0), 1.0, n_var);
            //         gpu_error_check(cudaSetDevice(1));
            //         multi_real<<<N_BLOCKS, N_THREADS, 0, streams[1][0]>>>(d_sum[1], d_w_deriv[1], d_w_deriv_device_1[prev], 1.0, n_var);

            //         // DEVICE 1, STREAM 1: copy memory from device 0 to device 1
            //         if(idx_next_gpu_1 <= n_anderson_odd)
            //         {
            //             gpu_error_check(cudaMemcpyAsync(d_w_deriv_device_1[next], d_cb_w_deriv_hist->get_array(idx_next_gpu_1),
            //                     sizeof(double)*n_var,cudaMemcpyDeviceToDevice, streams[1][1]));
            //         }

            //         // Reduce sum
            //         gpu_error_check(cudaSetDevice(0));
            //         cub::DeviceReduce::Sum(d_temp_storage[0], temp_storage_bytes[0], d_sum[0], d_sum_out[0], n_var, streams[0][0]);
            //         gpu_error_check(cudaSetDevice(1));
            //         cub::DeviceReduce::Sum(d_temp_storage[1], temp_storage_bytes[1], d_sum[1], d_sum_out[1], n_var, streams[1][0]);

            //         gpu_error_check(cudaSetDevice(0));
            //         gpu_error_check(cudaMemcpyAsync(&w_deriv_dots[idx_gpu_0], d_sum_out[0], sizeof(double), cudaMemcpyDeviceToHost, streams[0][0]));
            //         gpu_error_check(cudaSetDevice(1));
            //         gpu_error_check(cudaMemcpyAsync(&w_deriv_dots[idx_gpu_1], d_sum_out[1], sizeof(double), cudaMemcpyDeviceToHost, streams[1][0]));

            //         // Synchronize all GPUs
            //         for(int gpu=0; gpu<N_GPUS; gpu++)
            //         {
            //             gpu_error_check(cudaSetDevice(gpu));
            //             gpu_error_check(cudaDeviceSynchronize());
            //         }
            //         std::swap(prev, next);
            //     }
            //     gpu_error_check(cudaSetDevice(0));
            //     if (n_anderson%2 == 0)
            //     {
            //         multi_real<<<N_BLOCKS, N_THREADS>>>(d_sum[0], d_w_deriv[0], d_cb_w_deriv_hist->get_array(n_anderson), 1.0, n_var);
            //         cub::DeviceReduce::Sum(d_temp_storage[0], temp_storage_bytes[0], d_sum[0], d_sum_out[0], n_var);
            //         gpu_error_check(cudaMemcpy(&w_deriv_dots[n_anderson], d_sum_out[0], sizeof(double),cudaMemcpyDeviceToHost));
            //     }
            // }
            // Else
            {
                gpu_error_check(cudaSetDevice(0));
                // Evaluate w_deriv inner_product products for calculating Unm and Vn in Thompson's paper
                for(int i=0; i<=n_anderson; i++)
                {
                    multi_real<<<N_BLOCKS, N_THREADS>>>(d_sum[0], d_w_deriv[0], d_cb_w_deriv_hist->get_array(i), 1.0, n_var);
                    cub::DeviceReduce::Sum(d_temp_storage[0], temp_storage_bytes[0], d_sum[0], d_sum_out[0], n_var);
                    gpu_error_check(cudaMemcpy(&w_deriv_dots[i], d_sum_out[0], sizeof(double),cudaMemcpyDeviceToHost));
                }
            }

            //print_array(max_hist+1, w_deriv_dots);
            cb_w_deriv_dots->insert(w_deriv_dots);
        }

        gpu_error_check(cudaSetDevice(0));
        // Conditions to apply the simple mixing method
        if( n_anderson <= 0 )
        {
            // dynamically change mixing parameter
            if (old_error_level < error_level)
                mix = std::max(mix*0.7, mix_min);
            else
                mix = mix*1.01;

            // Make a simple mixing of input and output fields for the next iteration
            lin_comb<<<N_BLOCKS, N_THREADS>>>(d_w_new[0], 1.0, d_w_current, mix, d_w_deriv[0], n_var);
            gpu_error_check(cudaMemcpy(w_new, d_w_new[0], sizeof(double)*n_var, cudaMemcpyDeviceToHost));
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
            gpu_error_check(cudaMemcpy(d_w_new[0], d_w_hist1, sizeof(double)*n_var,cudaMemcpyDeviceToDevice));
            lin_comb<<<N_BLOCKS, N_THREADS>>>(d_w_new[0], 1.0, d_w_hist1, 1.0, d_w_deriv_hist1, n_var);
            for(int i=0; i<n_anderson; i++)
            {
                d_w_hist2 = d_cb_w_hist->get_array(i+1);
                d_w_deriv_hist2 = d_cb_w_deriv_hist->get_array(i+1);
                add_lin_comb<<<N_BLOCKS, N_THREADS>>>(d_w_new[0], a_n[i], d_w_hist2,       -a_n[i], d_w_hist1,       n_var);
                add_lin_comb<<<N_BLOCKS, N_THREADS>>>(d_w_new[0], a_n[i], d_w_deriv_hist2, -a_n[i], d_w_deriv_hist1, n_var);
            }
            gpu_error_check(cudaMemcpy(w_new, d_w_new[0], sizeof(double)*n_var,cudaMemcpyDeviceToHost));
        }
        gpu_error_check(cudaSetDevice(0));
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
