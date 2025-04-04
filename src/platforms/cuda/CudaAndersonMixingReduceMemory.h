/*-------------------------------------------------------------
* This is a derived CudaAndersonMixingReducedMemory class
*------------------------------------------------------------*/

#ifndef CUDA_ANDERSON_MIXING_REDUCE_MEMORY_H_
#define CUDA_ANDERSON_MIXING_REDUCE_MEMORY_H_

#include "CircularBuffer.h"
#include "ComputationBox.h"
#include "AndersonMixing.h"
#include "CudaCommon.h"
#include "PinnedCircularBuffer.h"

template <typename T>
class CudaAndersonMixingReduceMemory : public AndersonMixing<T>
{
private:
    // A few previous field values are stored for anderson mixing in pinned host memory
    PinnedCircularBuffer<T> *pinned_cb_w_hist, *pinned_cb_w_deriv_hist;

    CircularBuffer<T> *cb_w_deriv_dots;
    T *w_deriv_dots;
    // A matrix and arrays for determining coefficients
    T **u_nm, *v_n, *a_n;
    // Temporary arrays
    CuDeviceData<T> *d_w_new;
    CuDeviceData<T> *d_w_deriv;
    CuDeviceData<T> *d_sum;
    
    CuDeviceData<T> *d_w_hist1;
    CuDeviceData<T> *d_w_hist2;
    CuDeviceData<T> *d_w_deriv_hist1;
    CuDeviceData<T> *d_w_deriv_hist2;

    // Variables for cub reduction sum
    size_t temp_storage_bytes = 0;
    CuDeviceData<T> *d_temp_storage = nullptr;
    CuDeviceData<T> *d_sum_out;

    void print_array(int n, T *a);
public:

    CudaAndersonMixingReduceMemory(int n_var, int max_hist,
        double start_error, double mix_min, double mix_init);
    ~CudaAndersonMixingReduceMemory();

    void reset_count() override;
    void calculate_new_fields(
        T *w_new, T *w_current, T *w_deriv,
        double old_error_level, double error_level) override;

};
#endif
