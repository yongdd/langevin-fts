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
    // partition function and complementry partition are 
    // contiguously stored in q_d for every contour step.
    // In other words,
    // q       (r,1)   = q_d[0]          ~ q_d[MM-1]
    // q^dagger(r,N)   = q_d[MM]         ~ q_d[2*MM-1]
    // q       (r,2)   = q_d[2*MM]       ~ q_d[3*MM-1]
    // q^dagger(r,N-2) = q_d[3*MM]       ~ q_d[4*MM-1]
    // ......
    // q       (r,n)   = q_d[(2*n-2)*MM] ~ q_d[(2*n-1)*MM-1]
    // q^dagger(r,N-n) = q_d[(2*n-1)*MM] ~ q_d[(2*n  )*MM-1]
    // ......
    
    double *q_d;
    double *qstep_d;
    ftsComplex *kqin_d;

    double *expf_d;
    double *expdwa_d, *expdwb_d;
    double *phia_d, *phib_d;
                 
    void onestep(double *qin_d, double *qout_d,
                 double *expdw1_d, double *expdw2_d);
public:

    CudaPseudoDiscrete(SimulationBox *sb,
                       PolymerChain *pc);
    ~CudaPseudoDiscrete();

    void find_phi(double *phia,  double *phib,
                  double *q1_init, double *q2_init,
                  double *wa, double *wb, double &QQ) override;

    void get_partition(double *q1_out, double *q2_out, int n) override;
};

#endif
