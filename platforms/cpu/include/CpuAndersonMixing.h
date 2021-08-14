/*-------------------------------------------------------------
!  Anderson mixing module
-------------------------------------------------------------*/

#ifndef CPU_ANDERSON_MIXING_H_
#define CPU_ANDERSON_MIXING_H_

#include "CpuCircularBuffer.h"
#include "CpuSimulationBox.h"

class CpuAndersonMixing
{
private:
    CpuSimulationBox *sb;
    /* a few previous field values are stored for anderson mixing */
    CpuCircularBuffer *cb_wout_hist, *cb_wdiff_hist;
    /* arrays to calculate anderson mixing */
    CpuCircularBuffer *cb_wdiffdots;
    double **u_nm, *v_n, *a_n, *wdiffdots;

    int num_components, MM, TOTAL_MM;
    double start_anderson_error, mix_min, mix, mix_init;
    int max_anderson, n_anderson;

    void print_array(int n, double *a);
    double multi_dot(int n_comp, double *a, double *b);
    void find_an(double **u_nm, double *v_n, double *a_n, int n);
    
public:

    CpuAndersonMixing(CpuSimulationBox *sb, int num_components,
                   int max_anderson, double start_anderson_error,
                   double mix_min, double mix_init);
    ~CpuAndersonMixing();

    void reset_count();
    void caculate_new_fields(double *w, double *w_out, double *w_diff,
                            double old_error_level, double error_level);

};
#endif
