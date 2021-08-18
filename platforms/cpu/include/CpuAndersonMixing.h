/*-------------------------------------------------------------
* This is a derived CpuAndersonMixing class
*------------------------------------------------------------*/

#ifndef CPU_ANDERSON_MIXING_H_
#define CPU_ANDERSON_MIXING_H_

#include "SimulationBox.h"
#include "CircularBuffer.h"
#include "AndersonMixing.h"

class CpuAndersonMixing : public AndersonMixing
{
private:
    /* a few previous field values are stored for anderson mixing */
    CircularBuffer *cb_wout_hist, *cb_wdiff_hist;
    /* arrays to calculate anderson mixing */
    CircularBuffer *cb_wdiff_dots;
    double **u_nm, *v_n, *a_n, *wdiff_dots;
    
    void print_array(int n, double *a);
public:

    CpuAndersonMixing(
        SimulationBox *sb, int num_components,
        int max_anderson, double start_anderson_error,
        double mix_min, double mix_init);
    ~CpuAndersonMixing();
      
    void reset_count() override;
    void caculate_new_fields(
        double *w, double *w_out, double *w_diff,
        double old_error_level, double error_level) override;

};
#endif
