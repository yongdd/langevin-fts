/**
 * @file AndersonMixing.cpp
 * @brief Implementation of AndersonMixing base class.
 *
 * Provides the constructor and Gaussian elimination solver for the
 * Anderson mixing algorithm. The actual mixing computation is implemented
 * in platform-specific derived classes (CpuAndersonMixing, CudaAndersonMixing).
 *
 * **Anderson Mixing Algorithm:**
 *
 * Given field history {w_i} and residual history {d_i}, find coefficients
 * {a_i} that minimize the linear combination of residuals:
 *
 *     min || Σ a_i d_i ||²  subject to Σ a_i = 1
 *
 * This leads to a least-squares problem solved by Gaussian elimination.
 *
 * **Template Instantiations:**
 *
 * - AndersonMixing<double>: Real field mixing
 * - AndersonMixing<std::complex<double>>: Complex field mixing
 */

#include <complex>

#include "AndersonMixing.h"

/**
 * @brief Construct Anderson mixing with given parameters.
 *
 * @param n_var       Number of field variables
 * @param max_hist    Maximum history length for mixing
 * @param start_error Error threshold to begin Anderson mixing
 * @param mix_min     Minimum mixing parameter (prevents instability)
 * @param mix_init    Initial mixing parameter
 */
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

/**
 * @brief Solve linear system Ua = v using Gaussian elimination.
 *
 * Implements forward elimination followed by back substitution to solve
 * the normal equations arising from the Anderson mixing least-squares problem.
 *
 * **Algorithm:**
 *
 * 1. Forward elimination: Transform U to upper triangular form
 * 2. Back substitution: Solve for coefficients a_i
 *
 * @param u Input matrix U (n × n), modified in place
 * @param v Input/output vector (size n), modified to intermediate values
 * @param a Output solution vector (size n)
 * @param n System size (number of history entries used)
 *
 * @warning Matrix u is modified during elimination. Caller should
 *          provide a copy if original values are needed.
 */
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