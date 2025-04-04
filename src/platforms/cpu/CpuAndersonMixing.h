/*-------------------------------------------------------------
* This is a derived CpuAndersonMixing class
*------------------------------------------------------------*/

#ifndef CPU_ANDERSON_MIXING_H_
#define CPU_ANDERSON_MIXING_H_

#include "ComputationBox.h"
#include "CircularBuffer.h"
#include "AndersonMixing.h"

template <typename T>
class CpuAndersonMixing : public AndersonMixing<T>
{
private:
    // A few previous field values are stored
    CircularBuffer<T> *cb_w_hist, *cb_w_deriv_hist;
    CircularBuffer<T> *cb_w_deriv_dots;
    T *w_deriv_dots;
    // A matrix and arrays for determining coefficients
    T **u_nm, *v_n, *a_n;
    
    T dot_product(T *a, T *b);
    void print_array(int n, T *a);
public:

    CpuAndersonMixing(int n_var, int max_hist,
        double start_error, double mix_min, double mix_init);
    ~CpuAndersonMixing();
      
    void reset_count() override;
    void calculate_new_fields(
        T *w_new, T *w_current, T *w_deriv,
        double old_error_level, double error_level) override;
};
#endif
