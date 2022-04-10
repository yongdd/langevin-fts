/*-------------------------------------------------------------
* This is an abstract CpuPseudoGaussian class
*------------------------------------------------------------*/

#ifndef CPU_PSEUDO_GAUSSIAN_H_
#define CPU_PSEUDO_GAUSSIAN_H_

#include "SimulationBox.h"
#include "PolymerChain.h"
#include "Pseudo.h"
#include "FFT.h"

class CpuPseudoGaussian : public Pseudo
{
protected:
    FFT *fft;
    double *q1, *q2;
    void one_step(double *qin, double *qout,
                 double *expdw, double *expdw_half);
public:
    CpuPseudoGaussian(SimulationBox *sb, PolymerChain *pc, FFT *ff);
    ~CpuPseudoGaussian();

    void find_phi(
        double *phia,  double *phib,
        double *q1_init, double *q2_init,
        double *wa, double *wb, double &QQ) override;
    void get_partition(double *q1_out, int n1, double *q2_out, int n2) override;
};
#endif
