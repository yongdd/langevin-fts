#include <iostream>
#include <algorithm>
#include "CudaSimulationBox.h"
#include "CudaAndersonMixing.h"
#include "CudaCommon.h"

CudaCircularBuffer::CudaCircularBuffer(int length, int width)
{
    this->length = length;
    this->width = width;
    this->start = 0;
    this->n_items = 0;

    elems_d = new double*[length];
    for (int i=0; i<length; i++)
        cudaMalloc((void**)&elems_d[i], sizeof(double)*width);
}
CudaCircularBuffer::~CudaCircularBuffer()
{
    for (int i=0; i<length; i++)
        cudaFree(elems_d[i]);
    delete[] elems_d;
}
void CudaCircularBuffer::reset()
{
    start = 0;
    n_items = 0;
}
void CudaCircularBuffer::insert(double* new_arr)
{
    cudaMemcpy(elems_d[(start+n_items)%length], new_arr,
               sizeof(double)*width, cudaMemcpyHostToDevice);
    if (n_items == length)
        start = (start+1)%length;
    n_items = min(n_items+1, length);
}
double* CudaCircularBuffer::get_array(int n)
{
    return elems_d[(start+n)%length];
}

CudaAndersonMixing::CudaAndersonMixing(
    SimulationBox *sb, int num_components,
    int max_anderson, double start_anderson_error,
    double mix_min,   double mix_init)
    :AndersonMixing(sb, num_components,
                    max_anderson, start_anderson_error,
                    mix_min,  mix_init)
{
    this->N_BLOCKS = CudaCommon::get_instance().N_BLOCKS;
    this->N_THREADS = CudaCommon::get_instance().N_THREADS;

    // number of anderson mixing steps, increases from 0 to max_anderson
    n_anderson = -1;

    // record hisotry of wout in GPU device memory
    cb_wout_hist_d = new CudaCircularBuffer(max_anderson+1, TOTAL_MM);
    // record hisotry of wout-w in GPU device memory
    cb_wdiff_hist_d = new CudaCircularBuffer(max_anderson+1, TOTAL_MM);
    // record hisotry of inner_product product of wout-w in CPU host memory
    cb_wdiff_dots = new CircularBuffer(max_anderson+1, max_anderson+1);

    // define arrays for anderson mixing
    this->u_nm = new double*[max_anderson];
    for(int i=0; i<max_anderson; i++)
        this->u_nm[i] = new double[max_anderson];
    this->v_n = new double[max_anderson];
    this->a_n = new double[max_anderson];
    this->wdiff_dots = new double[max_anderson+1];

    // fields arrays
    cudaMalloc((void**)&w_diff_d, sizeof(double)*TOTAL_MM);
    cudaMalloc((void**)&w_d, sizeof(double)*TOTAL_MM);

    // reset_count
    reset_count();
}
CudaAndersonMixing::~CudaAndersonMixing()
{
    delete cb_wout_hist_d;
    delete cb_wdiff_hist_d;
    delete cb_wdiff_dots;

    for (int i=0; i<max_anderson; i++)
        delete[] u_nm[i];
    delete[] u_nm;
    delete[] v_n;
    delete[] a_n;
    delete[] wdiff_dots;
    cudaFree(w_diff_d);
    cudaFree(w_d);
}
void CudaAndersonMixing::reset_count()
{
    /* initialize mixing parameter */
    mix = mix_init;
    /* number of anderson mixing steps, increases from 0 to max_anderson */
    n_anderson = -1;

    cb_wout_hist_d->reset();
    cb_wdiff_hist_d->reset();
    cb_wdiff_dots->reset();
}

void CudaAndersonMixing::caculate_new_fields(
    double *w,
    double *w_out,
    double *w_diff,
    double old_error_level,
    double error_level)
{
    double* wout_hist1_d;
    double* wout_hist2_d;

    cudaMemcpy(w_diff_d, w_diff, sizeof(double)*TOTAL_MM, cudaMemcpyHostToDevice);

    //printf("mix: %f\n", mix);
    // condition to start anderson mixing
    if(error_level < start_anderson_error || n_anderson >= 0)
        n_anderson = n_anderson + 1;
    if( n_anderson >= 0 )
    {
        // number of histories to use for anderson mixing
        n_anderson = std::min(max_anderson, n_anderson);
        // store the input and output field (the memory is used in a periodic way)

        cb_wout_hist_d->insert(w_out);
        cb_wdiff_hist_d->insert(w_diff);

        // evaluate wdiff inner_product products for calculating Unm and Vn in Thompson's paper
        for(int i=0; i<= n_anderson; i++)
        {
            wdiff_dots[i] = ((CudaSimulationBox *)sb)->multi_inner_product_gpu(num_components, w_diff_d,
                                     cb_wdiff_hist_d->get_array(n_anderson-i));
        }
        //print_array(max_anderson+1, wdiff_dots);
        cb_wdiff_dots->insert(wdiff_dots);
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
        for(int i=0; i<TOTAL_MM; i++)
            w[i] = (1.0-mix)*w[i] + mix*w_out[i];
    }
    else
    {
        // calculate Unm and Vn
        for(int i=0; i<n_anderson; i++)
        {
            v_n[i] = cb_wdiff_dots->get_sym(n_anderson, n_anderson)
                     - cb_wdiff_dots->get_sym(n_anderson, n_anderson-i-1);

            for(int j=0; j<n_anderson; j++)
            {
                u_nm[i][j] = cb_wdiff_dots->get_sym(n_anderson, n_anderson)
                             - cb_wdiff_dots->get_sym(n_anderson, n_anderson-i-1)
                             - cb_wdiff_dots->get_sym(n_anderson-j-1, n_anderson)
                             + cb_wdiff_dots->get_sym(n_anderson-i-1, n_anderson-j-1);
            }
        }
        //print_array(max_anderson, v_n);
        //exit(-1);
        find_an(u_nm, v_n, a_n, n_anderson);

        // calculate the new field
        wout_hist1_d = cb_wout_hist_d->get_array(n_anderson);
        cudaMemcpy(w_d, wout_hist1_d, sizeof(double)*TOTAL_MM,cudaMemcpyDeviceToDevice);
        for(int i=0; i<n_anderson; i++)
        {
            wout_hist2_d = cb_wout_hist_d->get_array(n_anderson-i-1);
            addLinComb<<<N_BLOCKS, N_THREADS>>>(w_d, a_n[i], wout_hist2_d, -a_n[i], wout_hist1_d, TOTAL_MM);
        }
        cudaMemcpy(w, w_d, sizeof(double)*TOTAL_MM,cudaMemcpyDeviceToHost);
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
