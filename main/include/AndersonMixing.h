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
};
#endif
