/**
 * @file CpuAndersonMixing.cpp
 * @brief CPU implementation of Anderson mixing algorithm.
 *
 * Provides CPU-optimized Anderson mixing for accelerating SCFT convergence.
 * Uses stored history of field and residual values to compute optimal
 * mixing coefficients via least-squares fitting.
 *
 * **Algorithm:**
 *
 * 1. Store field history w_i and residual history d_i in circular buffers
 * 2. Compute U matrix: U_ij = <d_i, d_j> (inner products of residuals)
 * 3. Solve least-squares problem for coefficients a_i
 * 4. Update field: w_new = Σ a_i (w_i + λ d_i)
 *
 * **Circular Buffer Storage:**
 *
 * Uses CircularBuffer to efficiently manage rolling history without
 * memory reallocation. Oldest entries are automatically overwritten.
 *
 * **Template Instantiations:**
 *
 * - CpuAndersonMixing<double>: Real field mixing
 * - CpuAndersonMixing<std::complex<double>>: Complex field mixing
 *
 * @see AndersonMixing for base class and algorithm description
 */

#include <iostream>
#include <algorithm>
#include "CpuAndersonMixing.h"

/**
 * @brief Construct CPU Anderson mixing with given parameters.
 *
 * @param n_var       Number of field variables
 * @param max_hist    Maximum history length
 * @param start_error Error threshold to begin Anderson mixing
 * @param mix_min     Minimum mixing parameter
 * @param mix_init    Initial mixing parameter
 */
template <typename T>
CpuAndersonMixing<T>::CpuAndersonMixing(int n_var, int max_hist,
    double start_error, double mix_min, double mix_init)
    :AndersonMixing<T>(n_var, max_hist, start_error,
                    mix_min,  mix_init)
{
    try
    {
        // Number of anderson mixing steps, increases from 0 to max_hist
        this->n_anderson = -1;
        // Record history of w in memory
        cb_w_hist = new CircularBuffer<T>(max_hist+1, n_var);
        // Record history of w_deriv in memory
        cb_w_deriv_hist = new CircularBuffer<T>(max_hist+1, n_var);
        // Record history of (inner_product of w_deriv + inner_product of h_deriv) in memory
        cb_w_deriv_dots = new CircularBuffer<T>(max_hist+1, max_hist+1);

        // define arrays for anderson mixing
        this->u_nm = new T*[max_hist];
        for(int i=0; i<max_hist; i++)
            this->u_nm[i] = new T[max_hist];
        this->v_n = new T[max_hist];
        this->a_n = new T[max_hist];
        this->w_deriv_dots = new T[max_hist+1];

        // Reset_count
        reset_count();
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
CpuAndersonMixing<T>::~CpuAndersonMixing()
{
    delete cb_w_hist;
    delete cb_w_deriv_hist;
    delete cb_w_deriv_dots;

    for (int i=0; i<this->max_hist; i++)
        delete[] u_nm[i];
    delete[] u_nm;
    delete[] v_n;
    delete[] a_n;
    delete[] w_deriv_dots;
}
template <typename T>
void CpuAndersonMixing<T>::reset_count()
{
    try
    {
        // Initialize mixing parameter
        this->mix = this->mix_init;
        // Number of anderson mixing steps, increases from 0 to max_hist
        this->n_anderson = -1;

        cb_w_hist->reset();
        cb_w_deriv_hist->reset();
        cb_w_deriv_dots->reset();
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
T CpuAndersonMixing<T>::dot_product(T *a, T *b)
{
    T sum{0.0};
    for(int i=0; i<this->n_var; i++)
        sum += a[i]*b[i];
    return sum;
}
template <typename T>
void CpuAndersonMixing<T>::calculate_new_fields(
    T *w_new,
    T *w_current,
    T *w_deriv,
    double old_error_level,
    double error_level)
{
    try
    {
        T *w_hist1;
        T *w_hist2;
        T *w_deriv_hist1;
        T *w_deriv_hist2;

        // Condition to start anderson mixing
        if(error_level < this->start_error || this->n_anderson >= 0)
            this->n_anderson = this->n_anderson + 1;
        if(this->n_anderson >= 0)
        {
            // Number of histories to use for anderson mixing
            this->n_anderson = std::min(this->max_hist, this->n_anderson);
            // store the input and output field (the memory is used in a periodic way)
            cb_w_hist->insert(w_current);
            cb_w_deriv_hist->insert(w_deriv);

            // Evaluate w_deriv inner_product products for calculating Unm and Vn in Thompson's paper
            for(int i=0; i<= this->n_anderson; i++)
            {
                w_deriv_dots[i] = dot_product(w_deriv, cb_w_deriv_hist->get_array(i));
            }
            cb_w_deriv_dots->insert(w_deriv_dots);
        }
        // Conditions to apply the simple mixing method
        if(this->n_anderson <= 0)
        {
            // dynamically change mixing parameter
            if (old_error_level < error_level)
                this->mix = std::max(this->mix*0.7, this->mix_min);
            else
                this->mix = this->mix*1.01;

            // Make a simple mixing of input and output fields for the next iteration
            for(int i=0; i<this->n_var; i++)
                w_new[i] = w_current[i] + this->mix*w_deriv[i];
        }
        else
        {
            // Calculate Unm and Vn
            for(int i=0; i<this->n_anderson; i++)
            {
                v_n[i] = cb_w_deriv_dots->get(0, 0)
                        - cb_w_deriv_dots->get(0, i+1);

                for(int j=0; j<this->n_anderson; j++)
                {
                    u_nm[i][j] = cb_w_deriv_dots->get(0, 0)
                               - cb_w_deriv_dots->get(0, i+1)
                               - cb_w_deriv_dots->get(0, j+1)
                               + cb_w_deriv_dots->get(std::min(i+1, j+1),
                                                      std::abs(i-j));
                }
            }
            this->find_an(u_nm, v_n, a_n, this->n_anderson);
            //std::cout << "v_n2" << std::endl;
            //print_array(n_anderson+1, v_n);
            //exit(-1);

            // Calculate the new field
            w_hist1 = cb_w_hist->get_array(0);
            w_deriv_hist1 = cb_w_deriv_hist->get_array(0);
            for(int i=0; i<this->n_var; i++)
                w_new[i] = w_hist1[i] + w_deriv_hist1[i];
            for(int i=0; i<this->n_anderson; i++)
            {
                w_hist2       = cb_w_hist->get_array(i+1);
                w_deriv_hist2 = cb_w_deriv_hist->get_array(i+1);
                for(int j=0; j<this->n_var; j++)
                    w_new[j] += a_n[i]*(w_hist2[j] + w_deriv_hist2[j] - w_hist1[j] - w_deriv_hist1[j]);
            }
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
// Print array for debugging
template <typename T>
void CpuAndersonMixing<T>::print_array(int n, T *a)
{
    for(int i=0; i<n-1; i++)
    {
        std::cout << a[i] << ", ";
    }
    std::cout << a[n-1] << std::endl;
}

// Explicit template instantiation
template class CpuAndersonMixing<double>;
template class CpuAndersonMixing<std::complex<double>>;