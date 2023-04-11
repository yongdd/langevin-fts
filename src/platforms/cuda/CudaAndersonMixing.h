/*-------------------------------------------------------------
* This is a derived CudaAndersonMixing class
*------------------------------------------------------------*/

#ifndef CUDA_ANDERSON_MIXING_H_
#define CUDA_ANDERSON_MIXING_H_

#include "CircularBuffer.h"
#include "AndersonMixing.h"
#include "CudaCommon.h"
#include "CudaCircularBuffer.h"

class CudaAndersonMixing : public AndersonMixing
{
private:
    // two streams for each gpu
    cudaStream_t streams[MAX_GPUS][2]; // one for kernel execution, the other for memcpy

    // a few previous field values are stored for anderson mixing in GPU
    CudaCircularBuffer *d_cb_w_hist, *d_cb_w_deriv_hist;
    CircularBuffer *cb_w_deriv_dots;
    double *w_deriv_dots;
    // a matrix and arrays for determining coefficients
    double **u_nm, *v_n, *a_n;
    // temporary arrays
    double *d_w_current;
    double *d_w_new[MAX_GPUS];
    double *d_w_deriv[MAX_GPUS];
    double *d_sum[MAX_GPUS];

    // temporary arrays for device_1
    double *d_w_deriv_device_1[2]; // one for prev, the other for next
    
    // variables for cub reduction sum
    size_t temp_storage_bytes[MAX_GPUS];
    double *d_temp_storage[MAX_GPUS];
    double *d_sum_out[MAX_GPUS];

    void print_array(int n, double *a);
public:

    CudaAndersonMixing(int n_var, int max_hist,
        double start_error, double mix_min, double mix_init);
    ~CudaAndersonMixing();

    void reset_count() override;
    void calculate_new_fields(
        double *w_new, double *w_current, double *w_deriv,
        double old_error_level, double error_level) override;

};
#endif
