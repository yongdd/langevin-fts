/*-------------------------------------------------------------
* This is an abstract CpuPseudo class
*------------------------------------------------------------*/

#ifndef CPU_PSEUDO_H_
#define CPU_PSEUDO_H_

#include "SimulationBox.h"
#include "PolymerChain.h"
#include "Pseudo.h"
#include "FFT.h"

class CpuPseudo : public Pseudo
{
protected:
    FFT *fft;
    double *q1, *q2;
    
    void onestep(double *qin, double *qout,
                 double *expdw, double *expdw_half);
public:
    CpuPseudo(SimulationBox *sb, PolymerChain *pc, FFT *ff);
    ~CpuPseudo();

    void find_phi(
        double *phia,  double *phib,
        double *q1_init, double *q2_init,
        double *wa, double *wb, double &QQ) override;
    void get_partition(double *q1_out,  double *q2_out, int n);
};
#endif
