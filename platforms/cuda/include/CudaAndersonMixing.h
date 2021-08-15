/*-------------------------------------------------------------
* This is a derived CudaAndersonMixing class
*------------------------------------------------------------*/

#ifndef CUDA_ANDERSON_MIXING_H_
#define CUDA_ANDERSON_MIXING_H_

#include "CircularBuffer.h"
#include "SimulationBox.h"
#include "AndersonMixing.h"
#include "CudaCommon.h"

/*-----------------------------------------------------------------
! A circular buffer stores data in the GPU memory.
!-----------------------------------------------------------------*/
class CudaCircularBuffer
{
private:
    int length; // maximum number of elements
    int width;  // size of each elements
    int start;  // index of oldest elements
    int n_items;   // index at which to write new element
    double** elems_d;

public:
    CudaCircularBuffer(int length, int width);
    ~CudaCircularBuffer();
    void reset();
    void insert(double* new_arr);
    double* get_array(int n);
};

class CudaAndersonMixing : public AndersonMixing
{
private:

    int N_BLOCKS;
    int N_THREADS;

    SimulationBox *sb;
    // a few previous field values are stored for anderson mixing in GPU
    CudaCircularBuffer *cb_wout_hist_d, *cb_wdiff_hist_d;
    // arrays to calculate anderson mixing
    CircularBuffer *cb_wdiffinner_products;
    double **u_nm, *v_n, *a_n, *wdiffinner_products;
    double *w_d, *w_diff_d;

    int num_components, MM, TOTAL_MM;
    double start_anderson_error, mix_min, mix, mix_init;
    int max_anderson, n_anderson;

    void print_array(int n, double *a);
public:

    CudaAndersonMixing(
        SimulationBox *sb, int num_components,
        int max_anderson, double start_anderson_error,
        double mix_min, double mix_init);
    ~CudaAndersonMixing();

    void reset_count() override;
    void caculate_new_fields(
        double *w, double *w_out, double *w_diff,
        double old_error_level, double error_level) override;

};
#endif
