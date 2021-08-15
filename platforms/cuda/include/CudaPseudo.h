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

class CudaPseudo : public Pseudo
{
private:

    int N_BLOCKS;
    int N_THREADS;

    cufftHandle plan_for, plan_bak;

    double *temp_d, *temp_arr;
    double *q1_d, *q2_d;
    double *qstep1_d, *qstep2_d;
    ftsComplex *kqin_d;

    double *expdwa_d, *expdwa_half_d;
    double *expdwb_d, *expdwb_half_d;
    double *phia_d, *phib_d;

    double *expf_d, *expf_half_d;

    void onestep(double *qin1_d, double *qout1_d,
                 double *qin2_d, double *qout2_d,
                 double *expdw1_d, double *expdw1_half_d,
                 double *expdw2_d, double *expdw2_half_d);

public:

    CudaPseudo(SimulationBox *sb,
               PolymerChain *pc);
    ~CudaPseudo();

    void find_phi(double *phia,  double *phib,
                  double *q1_init, double *q2_init,
                  double *wa, double *wb, double &QQ) override;

    void get_partition(double *q1_out,  double *q2_out, int n);
};

#endif
