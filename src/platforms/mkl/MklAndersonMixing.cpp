/*-------------------------------------------------------------
!  Anderson mixing module
-------------------------------------------------------------*/

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include "CircularBuffer.h"

class AnderosnMixing
{
private:
    /* a few previous field values are stored for anderson mixing */
    CircularBuffer *cb_wout_hist, *cb_wdiff_hist;
    /* arrays to calculate anderson mixing */
    CircularBuffer *cb_wdiffdots;
    double **u_nm, *v_n, *a_n, *wdiffdots;

    int num_components, MM, TOTAL_MM;
    double start_anderson_error, mix_min, mix, init_mix;
    int max_anderson, n_anderson;
    double *dv, *temp, *sum;

    double multi_dot(int n_comp, double *a, double *b);
    void find_an(double **u_nm, double *v_n, double *a_n, int n);
    
public:

    AnderosnMixing(int num_components, int MM, double dv,
                   int max_anderson, double start_anderson_error,
                   double mix_min, double init_mix);
    ~AnderosnMixing();

    void reset_count_();
    void caculate_new_fields(double *w, double *w_out, double *w_diff,
        double old_error_level, double error_level);

};
AnderosnMixing::AnderosnMixing(
    int num_components, int MM,
    double dv,
    int max_anderson,   double start_anderson_error,
    double mix_min,     double init_mix)
{
    this->num_components = num_components;
    this->MM = MM;
    this->TOTAL_MM = num_components * MM;
    /* anderson mixing begin if error level becomes less then start_anderson_error */
    this->start_anderson_error = start_anderson_error;
    /* max number of previous steps to calculate new field when using Anderson mixing */
    this->max_anderson = max_anderson;
    /* minimum mixing parameter */
    this->mix_min = mix_min;
    /* initialize mixing parameter */
    this->mix = init_mix;
    this->init_mix = init_mix;
    /* number of anderson mixing steps, increases from 0 to max_anderson */
    n_anderson = -1;

    /* record hisotry of wout in GPU device memory */
    cb_wout_hist = new CircularBuffer(max_anderson+1, TOTAL_MM);
    /* record hisotry of wout-w in GPU device memory */
    cb_wdiff_hist = new CircularBuffer(max_anderson+1, TOTAL_MM);
    /* record hisotry of dot product of wout-w in CPU host memory */
    cb_wdiffdots = new CircularBuffer(max_anderson+1, max_anderson+1);

    /* define arrays for anderson mixing */
    this->u_nm = new double*[max_anderson];
    for (int i=0; i<max_anderson; i++)
        this->u_nm[i] = new double[max_anderson];
    this->v_n = new double[max_anderson];
    this->a_n = new double[max_anderson];
    this->wdiffdots = new double[max_anderson+1];

    this->dv = new double[MM];
    this->temp = new double[MM];

    /* copy segment arrays */
    for(int i=0; i<MM; i++)
        this->dv[i] = dv[i];

    /* sum arrays */
    sum = new double[MM];

    /* reset_count */
    reset_count();

}
void AnderosnMixing::~AnderosnMixing()
{
    delete cb_wout_hist;
    delete cb_wdiff_hist;
    delete cb_wdiffdots;

    for (int i=0; i<max_anderson; i++)
        delete[] u_nm[i];
    delete[] u_nm;
    delete[] v_n;
    delete[] a_n;
    delete[] wdiffdots;

    delete[] seg;
    delete[] temp;
    delete[] sum;
}
void AnderosnMixing::reset_count()
{
    /* initialize mixing parameter */
    mix = init_mix;
    /* number of anderson mixing steps, increases from 0 to max_anderson */
    n_anderson = -1;

    cb_wout_hist->reset();
    cb_wdiff_hist->reset();
    cb_wdiffdots->reset();
}
void AnderosnMixing::caculate_new_fields(
    double *w,
    double *w_out,
    double *w_diff,
    double old_error_level,
    double error_level)
{

    //printf("mix: %f\n", mix);
    /* condition to start anderson mixing */
    if(error_level < start_anderson_error || n_anderson >= 0)
        n_anderson = n_anderson + 1;
    if( n_anderson >= 0 )
    {
        /* number of histories to use for anderson mixing */
        n_anderson = min(max_anderson, n_anderson);
        /* store the input and output field (the memory is used in a periodic way) */
        cb_wout_hist->insert(w_out);
        cb_wdiff_hist->insert(w_diff);
        /* evaluate wdiff dot products for calculating Unm and Vn in Thompson's paper */
        for(int i=0; i<= n_anderson; i++)
        {
            wdiffdots[i] = multi_dot(num_components,
                                     w_diff_d, cb_wdiff_hist->getArray(n_anderson-i));
        }
        cb_wdiffdots->insert(wdiffdots);
    }
    /* conditions to apply the simple mixing method */
    if( n_anderson <= 0 )
    {
        /* dynamically change mixing parameter */
        if (old_error_level < error_level)
            mix = max(mix*0.7, mix_min);
        else
            mix = mix*1.01;
        /* make a simple mixing of input and output fields for the next iteration */
        for(int i=0; i<TOTAL_MM; i++)
            w[i] = (1.0-mix)*w[i] + mix*w_out[i];
    }
    else
    {
        /* calculate Unm and Vn */
        for(int i=0; i<n_anderson; i++)
        {
            v_n[i] = cb_wdiffdots->get_sym(n_anderson, n_anderson)
                     - cb_wdiffdots->get_sym(n_anderson, n_anderson-i-1);
            for(int j=0; j<n_anderson; j++)
            {
                u_nm[i][j] = cb_wdiffdots->get_sym(n_anderson, n_anderson)
                             - cb_wdiffdots->get_sym(n_anderson, n_anderson-i-1)
                             - cb_wdiffdots->get_sym(n_anderson-j-1, n_anderson)
                             + cb_wdiffdots->get_sym(n_anderson-i-1, n_anderson-j-1);
            }
        }

        find_an(u_nm, v_n, a_n, n_anderson);
        /* calculate the new field */
        wout_hist1 = cb_wout_hist->get_array(n_anderson);
        for(int i=0; i<TOTAL_MM; i++)
            w[i] = wout_hist1[i];
        for(int i=0; i<n_anderson; i++)
        {
            wout_hist2 = cb_wout_hist->get_array(n_anderson-i-1);
            for(int j=0; j<TOTAL_MM; j++)
                w[j] += a_n[i](wout_hist2[j] - wout_hist1[j]);
        }
    }
}
double AnderosnMixing::multi_dot(int n_comp, double *a, double *b)
{
    double total{0.0};
    for(int n=0; n<n_comp; n++)
    {
        for(int i=0; i<MM; i++)
            total = total + seg[i]*a[i+n*MM]*b[i+n*MM];
    }
    return total;
}
void AnderosnMixing::find_an(double **u, double *v, double *a, int n)
{

    int i,j,k;
    double factor, tempsum;
    /* elimination process */
    for(i=0; i<n; i++)
    {
        for(j=i+1; j<n; j++)
        {
            factor = u[j][i]/u[i][i];
            v[j] = v[j] - v[i]*factor;
            for(k=i+1; k<n; k++)
            {
                u[j][k] = u[j][k] - u[i][k]*factor;
            }
        }
    }
    /* find the solution */
    a[n-1] = v[n-1]/u[n-1][n-1];
    for(i=n-2; i>=0; i--)
    {
        tempsum = 0.0;
        for(j=i+1; j<n; j++)
        {
            tempsum = tempsum + u[i][j]*a[j];
        }
        a[i] = (v[i] - tempsum)/u[i][i];
    }
}
