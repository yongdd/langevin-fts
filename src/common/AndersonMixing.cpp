#include <complex>

#include "AndersonMixing.h"

template <typename T>
AndersonMixing<T>::AndersonMixing(int n_var, int max_hist,
    double start_error, double mix_min, double mix_init)
{
    // The number of variables to be determined
    this->n_var = n_var;
    // Anderson mixing begin if error level becomes less then start_anderson_error
    this->start_error = start_error;
    // Maximum number of histories to calculate new field when using Anderson mixing
    this->max_hist = max_hist;
    // Minimum mixing parameter
    this->mix_min = mix_min;
    // Initialize mixing parameter
    this->mix = mix_init;
    this->mix_init = mix_init;
}
template <typename T>
void AndersonMixing<T>::find_an(T **u, T *v, T *a, int n)
{
    int i,j,k;
    T factor, temp_sum;
    // Elimination process
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
    // Find the solution
    a[n-1] = v[n-1]/u[n-1][n-1];
    for(i=n-2; i>=0; i--)
    {
        temp_sum = 0.0;
        for(j=i+1; j<n; j++)
        {
            temp_sum = temp_sum + u[i][j]*a[j];
        }
        a[i] = (v[i] - temp_sum)/u[i][i];
    }
}

// Explicit template instantiation
template class AndersonMixing<double>;
template class AndersonMixing<std::complex<double>>;