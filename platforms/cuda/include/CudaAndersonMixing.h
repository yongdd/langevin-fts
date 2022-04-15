/*-------------------------------------------------------------
* This is a derived CudaAndersonMixing class
*------------------------------------------------------------*/

#ifndef CUDA_ANDERSON_MIXING_H_
#define CUDA_ANDERSON_MIXING_H_

#include "CircularBuffer.h"
#include "SimulationBox.h"
#include "AndersonMixing.h"
#include "CudaCommon.h"
#include "CudaCircularBuffer.h"

class CudaAndersonMixing : public AndersonMixing
{
private:
    // a few previous field values are stored for anderson mixing in GPU
    CudaCircularBuffer *d_cb_w_out_hist, *d_cb_w_diff_hist;
    // arrays to calculate anderson mixing
    CircularBuffer *cb_w_diff_dots;
    double **u_nm, *v_n, *a_n, *w_diff_dots;
    double *d_w, *d_w_diff;

    void print_array(int n, double *a);
public:

    CudaAndersonMixing(
        SimulationBox *sb, int n_comp,
        int max_hist, double start_error,
        double mix_min, double mix_init);
    ~CudaAndersonMixing();

    void reset_count() override;
    void caculate_new_fields(
        double *w, double *w_out, double *w_diff,
        double old_error_level, double error_level) override;

};
#endif
