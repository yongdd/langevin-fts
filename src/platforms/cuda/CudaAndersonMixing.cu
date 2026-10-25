/**
 * @file CudaAndersonMixing.cu
 * @brief CUDA implementation of Anderson mixing algorithm.
 *
 * Provides GPU-accelerated Anderson mixing for SCFT convergence using
 * device memory and CUB parallel reductions.
 *
 * **GPU Optimization:**
 *
 * - Field history stored in CudaCircularBuffer (device memory)
 * - Inner products computed with CUB reduction kernels
 *
 * **Algorithm:**
 *
 * 1. Store w_i and d_i history on GPU
 * 2. Compute U matrix inner products using CUB reduction
 * 3. Solve least-squares on CPU (small system)
 * 4. Apply mixing coefficients with CUDA kernels
 *
 * **Template Instantiations:**
 *
 * - CudaAndersonMixing<double>: Real field mixing
 * - CudaAndersonMixing<std::complex<double>>: Complex field mixing
 *
 * @see CudaCircularBuffer for GPU history storage
 * @see AndersonMixing for algorithm description
 */

#include <iostream>
#include <algorithm>
#include <thrust/reduce.h>
#include "CudaCommon.h"
#include "CudaComputationBox.h"
#include "CudaCircularBuffer.h"
#include "CudaAndersonMixing.h"

/**
 * @brief Construct CUDA Anderson mixing with GPU buffers.
 *
 * @param n_var       Number of field variables
 * @param max_hist    Maximum history length
 * @param start_error Error threshold to begin Anderson mixing
 * @param mix_min     Minimum mixing parameter
 * @param mix_init    Initial mixing parameter
 */
template <typename T>
CudaAndersonMixing<T>::CudaAndersonMixing(
    int n_var, int max_hist, double start_error,
    double mix_min,   double mix_init)
    :AndersonMixing<T>(n_var, max_hist, start_error, mix_min, mix_init)
{
    try
    {

        // Number of anderson mixing steps, increases from 0 to this->max_hist
        this->n_anderson = -1;
        // Record history of w in GPU device memory
        d_cb_w_hist = std::make_unique<CudaCircularBuffer<T>>(this->max_hist+1, this->n_var);
        // Record history of w_deriv in GPU device memory
        d_cb_w_deriv_hist = std::make_unique<CudaCircularBuffer<T>>(this->max_hist+1, this->n_var);
        // Record history of inner_product product of w_deriv in CPU host memory
        cb_w_deriv_dots = std::make_unique<CircularBuffer<T>>(this->max_hist+1, this->max_hist+1);

        // define arrays for anderson mixing
        u_nm_storage.assign(static_cast<size_t>(this->max_hist) * this->max_hist, T());
        u_nm.resize(this->max_hist);
        for(int i=0; i<this->max_hist; i++)
            u_nm[i] = u_nm_storage.data() + static_cast<size_t>(i) * this->max_hist;
        v_n.assign(this->max_hist, T());
        a_n.assign(this->max_hist, T());
        w_deriv_dots.assign(this->max_hist+1, T());

        // Fields arrays
        d_w_current.allocate(this->n_var);
        d_w_new.allocate(this->n_var);
        d_w_deriv.allocate(this->n_var);
        d_sum.allocate(this->n_var);
        d_sum_out.allocate(1);

        // Allocate memory for cub reduction sum (the first call only queries the size)
        temp_storage_bytes = 0;
        if constexpr (std::is_same<T, double>::value)
            CUDA_CHECK(cub::DeviceReduce::Sum(nullptr, temp_storage_bytes, d_sum.get(), d_sum_out.get(), this->n_var));
        else
            CUDA_CHECK(cub::DeviceReduce::Reduce(nullptr, temp_storage_bytes, d_sum.get(), d_sum_out.get(), this->n_var, ComplexSumOp(), CuDeviceData<T>{0.0,0.0}));
        d_temp_storage.allocate(temp_storage_bytes);

        // Reset_count
        reset_count();
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
CudaAndersonMixing<T>::~CudaAndersonMixing() = default;
template <typename T>
void CudaAndersonMixing<T>::reset_count()
{
    try
    {
        /* initialize mixing parameter */
        this->mix = this->mix_init;
        /* number of anderson mixing steps, increases from 0 to this->max_hist */
        this->n_anderson = -1;

        d_cb_w_hist->reset();
        d_cb_w_deriv_hist->reset();
        cb_w_deriv_dots->reset();
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
void CudaAndersonMixing<T>::calculate_new_fields(
    T *w_new,
    T *w_current,
    T *w_deriv,
    double old_error_level,
    double error_level)
{
    try
    {
        const int N_BLOCKS = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();
        
        CuDeviceData<T> *d_w_hist1;
        CuDeviceData<T> *d_w_hist2;
        CuDeviceData<T> *d_w_deriv_hist1;
        CuDeviceData<T> *d_w_deriv_hist2;

        CUDA_CHECK(cudaMemcpy(d_w_deriv,   w_deriv,   sizeof(T)*this->n_var, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_w_current, w_current, sizeof(T)*this->n_var, cudaMemcpyHostToDevice));

        // Condition to start anderson mixing
        if(error_level < this->start_error || this->n_anderson >= 0)
            this->n_anderson = this->n_anderson + 1;
        if( this->n_anderson >= 0 )
        {
            // Number of histories to use for anderson mixing
            this->n_anderson = std::min(this->max_hist, this->n_anderson);
            
            // store the input and output field (the memory is used in a periodic way)
            d_cb_w_hist->insert(d_w_current);
            d_cb_w_deriv_hist->insert(d_w_deriv);

            // Evaluate w_deriv inner_product products for calculating Unm and Vn in Thompson's paper
            for(int i=0; i<=this->n_anderson; i++)
            {
                ker_multi<<<N_BLOCKS, N_THREADS>>>(d_sum, d_w_deriv, d_cb_w_deriv_hist->get_array(i), 1.0, this->n_var);
                CUDA_CHECK_LAUNCH();
                if constexpr (std::is_same<T, double>::value)
                    CUDA_CHECK(cub::DeviceReduce::Sum(d_temp_storage.get(), temp_storage_bytes, d_sum.get(), d_sum_out.get(), this->n_var));
                else
                    CUDA_CHECK(cub::DeviceReduce::Reduce(d_temp_storage.get(), temp_storage_bytes, d_sum.get(), d_sum_out.get(), this->n_var, ComplexSumOp(), CuDeviceData<T>{0.0,0.0}));
                CUDA_CHECK(cudaMemcpy(&w_deriv_dots[i], d_sum_out, sizeof(T),cudaMemcpyDeviceToHost));
            }

            cb_w_deriv_dots->insert(w_deriv_dots.data());
        }

        // Conditions to apply the simple mixing method
        if( this->n_anderson <= 0 )
        {
            // dynamically change mixing parameter
            if (old_error_level < error_level)
                this->mix = std::max(this->mix*0.7, this->mix_min);
            else
                this->mix = this->mix*1.01;

            // Make a simple mixing of input and output fields for the next iteration
            ker_lin_comb<<<N_BLOCKS, N_THREADS>>>(d_w_new, 1.0, d_w_current, this->mix, d_w_deriv, this->n_var);
            CUDA_CHECK_LAUNCH();
            CUDA_CHECK(cudaMemcpy(w_new, d_w_new, sizeof(T)*this->n_var, cudaMemcpyDeviceToHost));
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
            this->find_an(u_nm.data(), v_n.data(), a_n.data(), this->n_anderson);

            // Calculate the new field
            d_w_hist1 = d_cb_w_hist->get_array(0);
            d_w_deriv_hist1 = d_cb_w_deriv_hist->get_array(0);
            ker_lin_comb<<<N_BLOCKS, N_THREADS>>>(d_w_new, 1.0, d_w_hist1, 1.0, d_w_deriv_hist1, this->n_var);
            CUDA_CHECK_LAUNCH();
            for(int i=0; i<this->n_anderson; i++)
            {
                d_w_hist2 = d_cb_w_hist->get_array(i+1);
                d_w_deriv_hist2 = d_cb_w_deriv_hist->get_array(i+1);
                if constexpr (std::is_same<T, double>::value)
                {
                    ker_add_lin_comb<<<N_BLOCKS, N_THREADS>>>(d_w_new, a_n[i], d_w_hist2,       -a_n[i], d_w_hist1,       this->n_var);
                    CUDA_CHECK_LAUNCH();
                    ker_add_lin_comb<<<N_BLOCKS, N_THREADS>>>(d_w_new, a_n[i], d_w_deriv_hist2, -a_n[i], d_w_deriv_hist1, this->n_var);
                    CUDA_CHECK_LAUNCH();
                }
                else
                {
                    ker_add_lin_comb<<<N_BLOCKS, N_THREADS>>>(d_w_new, stdToCuDoubleComplex(a_n[i]), d_w_hist2,       stdToCuDoubleComplex(-a_n[i]), d_w_hist1,       this->n_var);
                    CUDA_CHECK_LAUNCH();
                    ker_add_lin_comb<<<N_BLOCKS, N_THREADS>>>(d_w_new, stdToCuDoubleComplex(a_n[i]), d_w_deriv_hist2, stdToCuDoubleComplex(-a_n[i]), d_w_deriv_hist1, this->n_var);
                    CUDA_CHECK_LAUNCH();
                }
            }
            CUDA_CHECK(cudaMemcpy(w_new, d_w_new, sizeof(T)*this->n_var,cudaMemcpyDeviceToHost));
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
void CudaAndersonMixing<T>::print_array(int n, T *a)
{
    #ifndef NDEBUG
    for(int i=0; i<n-1; i++)
    {
        std::cout << a[i] << ", ";
    }
    std::cout << a[n-1] << std::endl;
    #else
    (void)n; (void)a;  // Suppress unused parameter warnings
    #endif
}

// Explicit template instantiation
template class CudaAndersonMixing<double>;
template class CudaAndersonMixing<std::complex<double>>;