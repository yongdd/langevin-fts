/*-------------------------------------------------------------
* This is a derived CpuPseudo class
*------------------------------------------------------------*/

#ifndef CPU_PSEUDO_H_
#define CPU_PSEUDO_H_

#include "SimulationBox.h"
#include "Pseudo.h"
#include "MklFFT.h"

class CpuPseudo : public Pseudo
{
private:

    SimulationBox *sb;
    MklFFT *fft;
    
    int MM, MM_COMPLEX;
    int NN, NNf;
    double *expf, *expf_half;
    double *q1, *q2;

    void init_gaussian_factor(int *nx, double *dx, double ds);
    void onestep(double *qin, double *qout,
                 double *expdw, double *expdw_half);

public:

    CpuPseudo(SimulationBox *sb, double ds, int NN, int NNf);
    ~CpuPseudo();

    void find_phi(double *phia,  double *phib,
                  double *q1_init, double *q2_init,
                  double *wa, double *wb,
                  double ds, double &QQ) override;

    void get_partition(double *q1_out,  double *q2_out, int n);
};
#endif
