#include <iostream>
#include <algorithm>
#include "CudaCommon.h"
#include "CudaSimulationBox.h"
#include "CudaCircularBuffer.h"
#include "CudaAndersonMixing.h"

CudaAndersonMixing::CudaAndersonMixing(
    SimulationBox *sb, int n_comp,
    int max_hist, double start_error,
    double mix_min,   double mix_init)
    :AndersonMixing(sb, n_comp,
                    max_hist, start_error,
                    mix_min,  mix_init)
{
    const int M = sb->get_n_grid();
    
    // number of anderson mixing steps, increases from 0 to max_hist
    n_anderson = -1;
    // record hisotry of w_out in GPU device memory
    d_cb_w_out_hist = new CudaCircularBuffer(max_hist+1, n_comp*M);
    // record hisotry of w_out-w in GPU device memory
    d_cb_w_diff_hist = new CudaCircularBuffer(max_hist+1, n_comp*M);
    // record hisotry of inner_product product of w_out-w in CPU host memory
    cb_w_diff_dots = new CircularBuffer(max_hist+1, max_hist+1);

    // define arrays for anderson mixing
    this->u_nm = new double*[max_hist];
    for(int i=0; i<max_hist; i++)
        this->u_nm[i] = new double[max_hist];
    this->v_n = new double[max_hist];
    this->a_n = new double[max_hist];
    this->w_diff_dots = new double[max_hist+1];

    // fields arrays
    cudaMalloc((void**)&d_w_diff, sizeof(double)*n_comp*M);
    cudaMalloc((void**)&d_w,      sizeof(double)*n_comp*M);

    // reset_count
    reset_count();
}
CudaAndersonMixing::~CudaAndersonMixing()
{
    delete d_cb_w_out_hist;
    delete d_cb_w_diff_hist;
    delete cb_w_diff_dots;

    for (int i=0; i<max_anderson; i++)
        delete[] u_nm[i];
    delete[] u_nm;
    delete[] v_n;
    delete[] a_n;
    delete[] w_diff_dots;
    
    cudaFree(d_w_diff);
    cudaFree(d_w);
}
void CudaAndersonMixing::reset_count()
{
    /* initialize mixing parameter */
    mix = mix_init;
    /* number of anderson mixing steps, increases from 0 to max_anderson */
    n_anderson = -1;

    d_cb_w_out_hist->reset();
    d_cb_w_diff_hist->reset();
    cb_w_diff_dots->reset();
}

void CudaAndersonMixing::caculate_new_fields(
    double *w,
    double *w_out,
    double *w_diff,
    double old_error_level,
    double error_level)
{
    const int N_BLOCKS = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();

    const int M = sb->get_n_grid();
    double* d_w_out_hist1;
    double* d_w_out_hist2;

    cudaMemcpy(d_w_diff, w_diff, sizeof(double)*n_comp*M, cudaMemcpyHostToDevice);

    //printf("mix: %f\n", mix);
    // condition to start anderson mixing
    if(error_level < start_anderson_error || n_anderson >= 0)
        n_anderson = n_anderson + 1;
    if( n_anderson >= 0 )
    {
        // number of histories to use for anderson mixing
        n_anderson = std::min(max_anderson, n_anderson);
        
        // store the input and output field (the memory is used in a periodic way)
        d_cb_w_out_hist->insert(w_out);
        d_cb_w_diff_hist->insert(w_diff);

        // evaluate w_diff inner_product products for calculating Unm and Vn in Thompson's paper
        for(int i=0; i<= n_anderson; i++)
        {
            w_diff_dots[i] = ((CudaSimulationBox *)sb)->mutiple_inner_product_gpu(n_comp, d_w_diff,
                                     d_cb_w_diff_hist->get_array(i));
        }
        //print_array(max_anderson+1, w_diff_dots);
        cb_w_diff_dots->insert(w_diff_dots);
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
        for(int i=0; i<n_comp*M; i++)
            w[i] = (1.0-mix)*w[i] + mix*w_out[i];
    }
    else
    {
        // calculate Unm and Vn
        for(int i=0; i<n_anderson; i++)
        {
            v_n[i] = cb_w_diff_dots->get(0, 0)
                     - cb_w_diff_dots->get(0, i+1);

            for(int j=0; j<n_anderson; j++)
            {
                u_nm[i][j] = cb_w_diff_dots->get(0, 0)
                             - cb_w_diff_dots->get(0, i+1)
                             - cb_w_diff_dots->get(0, j+1)
                             + cb_w_diff_dots->get(std::min(i+1, j+1),
                                                  std::abs(i-j));
            }
        }
        //print_array(max_anderson, v_n);
        //exit(-1);
        find_an(u_nm, v_n, a_n, n_anderson);

        // calculate the new field
        d_w_out_hist1 = d_cb_w_out_hist->get_array(0);
        cudaMemcpy(d_w, d_w_out_hist1, sizeof(double)*n_comp*M,cudaMemcpyDeviceToDevice);
        for(int i=0; i<n_anderson; i++)
        {
            d_w_out_hist2 = d_cb_w_out_hist->get_array(i+1);
            add_lin_comb<<<N_BLOCKS, N_THREADS>>>(d_w, a_n[i], d_w_out_hist2, -a_n[i], d_w_out_hist1, n_comp*M);
        }
        cudaMemcpy(w, d_w, sizeof(double)*n_comp*M,cudaMemcpyDeviceToHost);
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
