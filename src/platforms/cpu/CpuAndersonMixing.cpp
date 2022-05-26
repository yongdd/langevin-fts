#include <iostream>
#include <algorithm>
#include "CpuAndersonMixing.h"

CpuAndersonMixing::CpuAndersonMixing(int n_var, int max_hist, 
    double start_error, double mix_min, double mix_init)
    :AndersonMixing(n_var, max_hist, start_error,
                    mix_min,  mix_init)
{
    // number of anderson mixing steps, increases from 0 to max_hist
    n_anderson = -1;
    // record hisotry of w_out in memory
    cb_w_out_hist = new CircularBuffer(max_hist+1, n_var);
    // record hisotry of w_deriv in memory
    cb_w_deriv_hist = new CircularBuffer(max_hist+1, n_var);
    // record hisotry of (inner_product of w_deriv + inner_product of h_deriv) in memory
    cb_w_deriv_dots = new CircularBuffer(max_hist+1, max_hist+1);

    // define arrays for anderson mixing
    this->u_nm = new double*[max_hist];
    for(int i=0; i<max_hist; i++)
        this->u_nm[i] = new double[max_hist];
    this->v_n = new double[max_hist];
    this->a_n = new double[max_hist];
    this->w_deriv_dots = new double[max_hist+1];

    /* reset_count */
    reset_count();

}
CpuAndersonMixing::~CpuAndersonMixing()
{
    delete cb_w_out_hist;
    delete cb_w_deriv_hist;
    delete cb_w_deriv_dots;

    for (int i=0; i<max_hist; i++)
        delete[] u_nm[i];
    delete[] u_nm;
    delete[] v_n;
    delete[] a_n;
    delete[] w_deriv_dots;
}
void CpuAndersonMixing::reset_count()
{
    /* initialize mixing parameter */
    mix = mix_init;
    /* number of anderson mixing steps, increases from 0 to max_hist */
    n_anderson = -1;

    cb_w_out_hist->reset();
    cb_w_deriv_hist->reset();
    cb_w_deriv_dots->reset();
}
double CpuAndersonMixing::dot_product(double *a, double *b)
{
    double sum{0.0};
    for(int i; i<n_var; i++)
        sum += a[i]*b[i];
    return sum;
}

void CpuAndersonMixing::caculate_new_fields(
    double *w,
    double *w_out,
    double *w_deriv,
    double old_error_level,
    double error_level)
{
    double* w_out_hist1;
    double* w_out_hist2;

    //printf("mix: %f\n", mix);
    /* condition to start anderson mixing */
    if(error_level < start_error || n_anderson >= 0)
        n_anderson = n_anderson + 1;
    if( n_anderson >= 0 )
    {
        /* number of histories to use for anderson mixing */
        n_anderson = std::min(max_hist, n_anderson);
        /* store the input and output field (the memory is used in a periodic way) */

        cb_w_out_hist->insert(w_out);
        cb_w_deriv_hist->insert(w_deriv);

        /* evaluate w_deriv inner_product products for calculating Unm and Vn in Thompson's paper */
        for(int i=0; i<= n_anderson; i++)
        {
            w_deriv_dots[i] = dot_product(w_deriv, cb_w_deriv_hist->get_array(i));
        }
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
        {
            w[i] = (1.0-mix)*w[i] + mix*w_out[i];
        }
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
        find_an(u_nm, v_n, a_n, n_anderson);
        //print_array(max_hist, v_n);
        //exit(-1);

        // calculate the new field
        w_out_hist1 = cb_w_out_hist->get_array(0);
        for(int i=0; i<n_var; i++)
            w[i] = w_out_hist1[i];
        for(int i=0; i<n_anderson; i++)
        {
            w_out_hist2 = cb_w_out_hist->get_array(i+1);
            for(int j=0; j<n_var; j++)
                w[j] += a_n[i]*(w_out_hist2[j] - w_out_hist1[j]);
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
