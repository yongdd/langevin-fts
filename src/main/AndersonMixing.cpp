#include "AndersonMixing.h"
AndersonMixing::AndersonMixing(
    SimulationBox *sb, int n_var,
    int max_hist, double start_error,
    double mix_min, double mix_init)
{
    this->sb = sb;
    /* the number of variables to be determined */
    this->n_var = n_var;
    /* anderson mixing begin if error level becomes less then start_anderson_error */
    this->start_error = start_error;
    /* max number of histories to calculate new field when using Anderson mixing */
    this->max_hist = max_hist;
    /* minimum mixing parameter */
    this->mix_min = mix_min;
    /* initialize mixing parameter */
    this->mix = mix_init;
    this->mix_init = mix_init;
}

void AndersonMixing::find_an(double **u, double *v, double *a, int n)
{
    int i,j,k;
    double factor, tempsum;
    // elimination process
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
    // find the solution
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
