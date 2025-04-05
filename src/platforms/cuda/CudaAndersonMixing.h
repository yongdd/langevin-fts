/*-------------------------------------------------------------
* This is a derived CudaAndersonMixing class
*------------------------------------------------------------*/

#ifndef CUDA_ANDERSON_MIXING_H_
#define CUDA_ANDERSON_MIXING_H_

#include "CircularBuffer.h"
#include "AndersonMixing.h"
#include "CudaCommon.h"
#include "CudaCircularBuffer.h"

template <typename T>
class CudaAndersonMixing : public AndersonMixing<T>
{
private:
    // Two streams for each gpu
    cudaStream_t streams[2]; // one for kernel execution, the other for memcpy

    // A few previous field values are stored for anderson mixing in GPU
    CudaCircularBuffer<T> *d_cb_w_hist, *d_cb_w_deriv_hist;
    CircularBuffer<T> *cb_w_deriv_dots;
    T *w_deriv_dots;
    // A matrix and arrays for determining coefficients
    T **u_nm, *v_n, *a_n;
    // Temporary arrays
    CuDeviceData<T> *d_w_current;
    CuDeviceData<T> *d_w_new;
    CuDeviceData<T> *d_w_deriv;
    CuDeviceData<T> *d_sum;

    // Variables for cub reduction sum
    size_t temp_storage_bytes = 0;
    CuDeviceData<T> *d_temp_storage = nullptr;
    CuDeviceData<T> *d_sum_out;

    void print_array(int n, T *a);
public:

    CudaAndersonMixing(int n_var, int max_hist,
        double start_error, double mix_min, double mix_init);
    ~CudaAndersonMixing();

    void reset_count() override;
    void calculate_new_fields(
        T *w_new, T *w_current, T *w_deriv,
        double old_error_level, double error_level) override;

};
#endif
