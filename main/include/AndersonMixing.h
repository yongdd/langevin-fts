/*-------------------------------------------------------------
* This is an abstract AndersonMixing class
*------------------------------------------------------------*/

#ifndef ANDERSON_MIXING_H_
#define ANDERSON_MIXING_H_

#include "SimulationBox.h"

class AndersonMixing
{
protected:
    void find_an(double **u, double *v, double *a, int n);
public:
    
    virtual void reset_count() {};
    virtual void caculate_new_fields(
        double *w, double *w_out, double *w_diff,
        double old_error_level, double error_level) {};
    
    // Methods for SWIG
    void caculate_new_fields(
        double *w_in, int len_w_in,
        double *w_out, int len_wout,
        double *w_diff, int len_wdiff,
        double old_error_level, double error_level)
    {
        caculate_new_fields(w_in, w_out, w_diff, old_error_level, error_level);
    }
};
#endif
