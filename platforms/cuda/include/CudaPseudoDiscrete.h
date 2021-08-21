/*-------------------------------------------------------------
* This is a derived CpuPseudo class
*------------------------------------------------------------*/

#ifndef CUDA_PSEUDO_H_
#define CUDA_PSEUDO_H_

#include <array>
#include <cufft.h>
#include "SimulationBox.h"
#include "Pseudo.h"
#include "CudaCommon.h"

class CudaPseudoDiscrete : public Pseudo
{
private:
    cufftHandle plan_for, plan_bak;

    double *temp_d, *temp_arr;
    double *q1_d, *q2_d;
    double *qstep_d;
    ftsComplex *kqin_d;

    double *expf_d;
    double *expdwa_d, *expdwb_d;
    double *phia_d, *phib_d;

    void onestep(double *qin1_d,   double *qin2_d,
                 double *qout1_d,  double *qout2_d,
                 double *expdw1_d, double *expdw2_d);
public:

    CudaPseudoDiscrete(SimulationBox *sb,
                       PolymerChain *pc);
    ~CudaPseudoDiscrete();

    void find_phi(double *phia,  double *phib,
                  double *q1_init, double *q2_init,
                  double *wa, double *wb, double &QQ) override;

    void get_partition(double *q1_out, double *q2_out, int n);
};

#endif
