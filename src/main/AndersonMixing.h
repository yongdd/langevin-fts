/*-------------------------------------------------------------
* This is an abstract AndersonMixing class
*------------------------------------------------------------*/

#ifndef ANDERSON_MIXING_H_
#define ANDERSON_MIXING_H_

#include <cassert>
#include "SimulationBox.h"

class AndersonMixing
{
protected:
    int n_var, max_hist, n_anderson;
    double start_error, mix_min, mix, mix_init;

    void find_an(double **u, double *v, double *a, int n);
public:
    AndersonMixing(int n_var, int max_hist,
     double start_error, double mix_min, double mix_init);

    virtual ~AndersonMixing(){};

    virtual void reset_count() {};
    virtual void caculate_new_fields(
        double *w, double *w_out, double *w_deriv,
        double old_error_level, double error_level)=0;

    // Methods for SWIG
    void caculate_new_fields(
        double *w_in, int len_w_in,
        double *w_out, int len_w_out,
        double *w_deriv, int len_w_deriv,
        double old_error_level, double error_level)
    {
        assert(len_w_in  == n_var);
        assert(len_w_out  == n_var);
        assert(len_w_deriv == n_var);
        caculate_new_fields(w_in, w_out, w_deriv, old_error_level, error_level);
    }
};
#endif
