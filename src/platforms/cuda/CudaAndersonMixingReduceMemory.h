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

class CudaAndersonMixingReduceMemory : public AndersonMixing
{
private:
    // a few previous field values are stored for anderson mixing in pinned host memory
    PinnedCircularBuffer *pinned_cb_w_hist, *pinned_cb_w_deriv_hist;

    CircularBuffer *cb_w_deriv_dots;
    double *w_deriv_dots;
    // a matrix and arrays for determining coefficients
    double **u_nm, *v_n, *a_n;
    // temporary arrays
    double *d_w_new;
    double *d_w_deriv;
    double *d_sum;
    
    double *d_w_hist1;
    double *d_w_hist2;
    double *d_w_deriv_hist1;
    double *d_w_deriv_hist2;

    // variables for cub reduction sum
    size_t temp_storage_bytes = 0;
    double *d_temp_storage = nullptr;
    double *d_sum_out;

    void print_array(int n, double *a);
public:

    CudaAndersonMixingReduceMemory(int n_var, int max_hist,
        double start_error, double mix_min, double mix_init);
    ~CudaAndersonMixingReduceMemory();

    void reset_count() override;
    void calculate_new_fields(
        double *w_new, double *w_current, double *w_deriv,
        double old_error_level, double error_level) override;

};
#endif
