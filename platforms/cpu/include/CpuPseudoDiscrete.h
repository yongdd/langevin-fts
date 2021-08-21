/*-------------------------------------------------------------
* This is an abstract CpuPseudoDiscrete class
*------------------------------------------------------------*/

#ifndef CPU_PSEUDO_DISCRETE_H_
#define CPU_PSEUDO_DISCRETE_H_

#include "SimulationBox.h"
#include "PolymerChain.h"
#include "Pseudo.h"
#include "FFT.h"

class CpuPseudoDiscrete : public Pseudo
{
protected:
    FFT *fft;
    double *q1, *q2;
    
    void onestep(double *qin, double *qout, double *expdw);
public:
    CpuPseudoDiscrete(SimulationBox *sb, PolymerChain *pc, FFT *ff);
    ~CpuPseudoDiscrete();

    void find_phi(
        double *phia,  double *phib,
        double *q1_init, double *q2_init,
        double *wa, double *wb, double &QQ) override;
    void get_partition(double *q1_out,  double *q2_out, int n);
};
#endif
