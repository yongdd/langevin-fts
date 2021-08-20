/*-------------------------------------------------------------
* This is an abstract AndersonMixing class
*------------------------------------------------------------*/

#ifndef ANDERSON_MIXING_H_
#define ANDERSON_MIXING_H_

#include "SimulationBox.h"

class AndersonMixing
{
protected:
    SimulationBox *sb;
    
    int num_components, TOTAL_MM;
    double start_anderson_error, mix_min, mix, mix_init;
    int max_anderson, n_anderson;

    void find_an(double **u, double *v, double *a, int n);
public:
    AndersonMixing(SimulationBox *sb, int num_components,
                   int max_anderson, double start_anderson_error,
                   double mix_min,   double mix_init);
    virtual ~AndersonMixing(){};

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
