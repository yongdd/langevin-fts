/*-------------------------------------------------------------
* This is a derived CpuAndersonMixing class
*------------------------------------------------------------*/

#ifndef CPU_ANDERSON_MIXING_H_
#define CPU_ANDERSON_MIXING_H_

#include "ComputationBox.h"
#include "CircularBuffer.h"
#include "AndersonMixing.h"

class CpuAndersonMixing : public AndersonMixing
{
private:
    // A few previous field values are stored
    CircularBuffer *cb_w_hist, *cb_w_deriv_hist;
    CircularBuffer *cb_w_deriv_dots;
    double *w_deriv_dots;
    // A matrix and arrays for determining coefficients
    double **u_nm, *v_n, *a_n;
    
    double dot_product(double *a, double *b);
    void print_array(int n, double *a);
public:

    CpuAndersonMixing(int n_var, int max_hist,
        double start_error, double mix_min, double mix_init);
    ~CpuAndersonMixing();
      
    void reset_count() override;
    void calculate_new_fields(
        double *w_new, double *w_current, double *w_deriv,
        double old_error_level, double error_level) override;
};
#endif
