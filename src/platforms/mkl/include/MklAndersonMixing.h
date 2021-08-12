/*-------------------------------------------------------------
!  Anderson mixing module
-------------------------------------------------------------*/

#ifndef MKL_ANDERSON_MIXING_H_
#define MKL_ANDERSON_MIXING_H_

#include "CircularBuffer.h"

class AnderosnMixing
{
private:
    /* a few previous field values are stored for anderson mixing */
    CircularBuffer *cb_wout_hist, *cb_wdiff_hist;
    /* arrays to calculate anderson mixing */
    CircularBuffer *cb_wdiffdots;
    double **u_nm, *v_n, *a_n, *wdiffdots;

    int num_components, MM, TOTAL_MM;
    double start_anderson_error, mix_min, mix, init_mix;
    int max_anderson, n_anderson;
    double *dv, *temp, *sum;

    void print_array(int n, double *a);
    double multi_dot(int n_comp, double *a, double *b);
    void find_an(double **u_nm, double *v_n, double *a_n, int n);
    
public:

    AnderosnMixing(int num_components, int MM, double *dv,
                   int max_anderson, double start_anderson_error,
                   double mix_min, double init_mix);
    ~AnderosnMixing();

    void reset_count();
    void caculate_new_fields(double *w, double *w_out, double *w_diff,
                            double old_error_level, double error_level);

};
#endif
