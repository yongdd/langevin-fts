/*-------------------------------------------------------------
* This is a derived CudaAndersonMixing class
*------------------------------------------------------------*/

#ifndef CUDA_ANDERSON_MIXING_H_
#define CUDA_ANDERSON_MIXING_H_

#include "CircularBuffer.h"
#include "ComputationBox.h"
#include "AndersonMixing.h"
#include "CudaCommon.h"
#include "CudaCircularBuffer.h"

class CudaAndersonMixing : public AndersonMixing
{
private:
    // a few previous field values are stored for anderson mixing in GPU
    CudaCircularBuffer *d_cb_w_out_hist, *d_cb_w_deriv_hist;
    CircularBuffer *cb_w_deriv_dots;
    double *w_deriv_dots;
    // a matrix and arrays for determining coefficients
    double **u_nm, *v_n, *a_n;
    // temporary arrays
    double *d_w, *d_w_deriv, *d_sum;
    
    void print_array(int n, double *a);
public:

    CudaAndersonMixing(int n_var, int max_hist,
        double start_error, double mix_min, double mix_init);
    ~CudaAndersonMixing();

    void reset_count() override;
    void calculate_new_fields(
        double *w, double *w_out, double *w_deriv,
        double old_error_level, double error_level) override;

};
#endif
