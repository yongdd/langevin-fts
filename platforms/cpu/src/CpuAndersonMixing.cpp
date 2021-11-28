#include <iostream>
#include <algorithm>
#include "CpuAndersonMixing.h"

CpuAndersonMixing::CpuAndersonMixing(
    SimulationBox *sb, int n_comp,
    int max_anderson, double start_anderson_error,
    double mix_min,   double mix_init)
    :AndersonMixing(sb, n_comp,
                    max_anderson, start_anderson_error,
                    mix_min,  mix_init)
{
    const int M = sb->get_n_grid();
    // number of anderson mixing steps, increases from 0 to max_anderson
    n_anderson = -1;
    // record hisotry of wout in memory
    cb_wout_hist = new CircularBuffer(max_anderson+1, n_comp*M);
    // record hisotry of wout-w in memory
    cb_wdiff_hist = new CircularBuffer(max_anderson+1, n_comp*M);
    // record hisotry of inner_product product of wout-w in memory
    cb_wdiff_dots = new CircularBuffer(max_anderson+1, max_anderson+1);

    // define arrays for anderson mixing
    this->u_nm = new double*[max_anderson];
    for(int i=0; i<max_anderson; i++)
        this->u_nm[i] = new double[max_anderson];
    this->v_n = new double[max_anderson];
    this->a_n = new double[max_anderson];
    this->wdiff_dots = new double[max_anderson+1];

    /* reset_count */
    reset_count();

}
CpuAndersonMixing::~CpuAndersonMixing()
{
    delete cb_wout_hist;
    delete cb_wdiff_hist;
    delete cb_wdiff_dots;

    for (int i=0; i<max_anderson; i++)
        delete[] u_nm[i];
    delete[] u_nm;
    delete[] v_n;
    delete[] a_n;
    delete[] wdiff_dots;
}
void CpuAndersonMixing::reset_count()
{
    /* initialize mixing parameter */
    mix = mix_init;
    /* number of anderson mixing steps, increases from 0 to max_anderson */
    n_anderson = -1;

    cb_wout_hist->reset();
    cb_wdiff_hist->reset();
    cb_wdiff_dots->reset();
}
void CpuAndersonMixing::caculate_new_fields(
    double *w,
    double *w_out,
    double *w_diff,
    double old_error_level,
    double error_level)
{
    const int M = sb->get_n_grid();
    
    double* wout_hist1;
    double* wout_hist2;

    //printf("mix: %f\n", mix);
    /* condition to start anderson mixing */
    if(error_level < start_anderson_error || n_anderson >= 0)
        n_anderson = n_anderson + 1;
    if( n_anderson >= 0 )
    {
        /* number of histories to use for anderson mixing */
        n_anderson = std::min(max_anderson, n_anderson);
        /* store the input and output field (the memory is used in a periodic way) */

        cb_wout_hist->insert(w_out);
        cb_wdiff_hist->insert(w_diff);

        /* evaluate wdiff inner_product products for calculating Unm and Vn in Thompson's paper */
        for(int i=0; i<= n_anderson; i++)
        {
            wdiff_dots[i] = sb->multi_inner_product(n_comp, w_diff, cb_wdiff_hist->get_array(i));
        }
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
        for(int i=0; i<n_comp*M; i++)
        {
            w[i] = (1.0-mix)*w[i] + mix*w_out[i];
        }
    }
    else
    {
        // calculate Unm and Vn
        for(int i=0; i<n_anderson; i++)
        {
            v_n[i] = cb_wdiff_dots->get(0, 0)
                     - cb_wdiff_dots->get(0, i+1);

            for(int j=0; j<n_anderson; j++)
            {
                u_nm[i][j] = cb_wdiff_dots->get(0, 0)
                             - cb_wdiff_dots->get(0, i+1)
                             - cb_wdiff_dots->get(0, j+1)
                             + cb_wdiff_dots->get(std::min(i+1, j+1),
                                                  std::abs(i-j));
            }
        }
        find_an(u_nm, v_n, a_n, n_anderson);
        //print_array(max_anderson, v_n);
        //exit(-1);

        // calculate the new field
        wout_hist1 = cb_wout_hist->get_array(0);
        for(int i=0; i<n_comp*M; i++)
            w[i] = wout_hist1[i];
        for(int i=0; i<n_anderson; i++)
        {
            wout_hist2 = cb_wout_hist->get_array(i+1);
            for(int j=0; j<n_comp*M; j++)
                w[j] += a_n[i]*(wout_hist2[j] - wout_hist1[j]);
        }
    }
}
void CpuAndersonMixing::print_array(int n, double *a)
{
    for(int i=0; i<n-1; i++)
    {
        std::cout << a[i] << ", ";
    }
    std::cout << a[n-1] << std::endl;
}
