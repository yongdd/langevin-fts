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
#include <algorithm>
#include <cmath>

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

    // Regularize diagonal for numerical stability (Tikhonov)
    double max_diag = 0.0;
    for(i=0; i<n; i++)
        max_diag = std::max(max_diag, std::abs(u[i][i]));
    if(max_diag > 0.0)
    {
        double reg = max_diag * 1e-8;
        for(i=0; i<n; i++)
            u[i][i] += static_cast<T>(reg);
    }

    // Gaussian elimination with partial pivoting
    for(i=0; i<n; i++)
    {
        // Find pivot row
        int max_row = i;
        double max_val = std::abs(u[i][i]);
        for(j=i+1; j<n; j++)
        {
            if(std::abs(u[j][i]) > max_val)
            {
                max_val = std::abs(u[j][i]);
                max_row = j;
            }
        }
        // Swap rows if needed
        if(max_row != i)
        {
            std::swap(u[i], u[max_row]);
            T tmp = v[i]; v[i] = v[max_row]; v[max_row] = tmp;
        }
        // Skip near-zero pivot
        if(std::abs(u[i][i]) < 1e-30)
            continue;

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
    // Back substitution with zero-pivot protection
    for(i=n-1; i>=0; i--)
    {
        temp_sum = static_cast<T>(0.0);
        for(j=i+1; j<n; j++)
        {
            temp_sum = temp_sum + u[i][j]*a[j];
        }
        if(std::abs(u[i][i]) < 1e-30)
            a[i] = static_cast<T>(0.0);
        else
            a[i] = (v[i] - temp_sum)/u[i][i];
    }
}

// Explicit template instantiation
template class AndersonMixing<double>;
template class AndersonMixing<std::complex<double>>;