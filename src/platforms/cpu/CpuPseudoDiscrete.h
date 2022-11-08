/*-------------------------------------------------------------
* This is an abstract CpuPseudoDiscrete class
*------------------------------------------------------------*/

#ifndef CPU_PSEUDO_DISCRETE_H_
#define CPU_PSEUDO_DISCRETE_H_

#include "ComputationBox.h"
#include "PolymerChain.h"
#include "Pseudo.h"
#include "FFT.h"

class CpuPseudoDiscrete : public Pseudo
{
private:
    FFT *fft;
    double *q_1, *q_2;
    //double *boltz_bond_a, *boltz_bond_b, *boltz_bond_ab;
    double **boltz_bond;
    double **boltz_bond_middle;
    void one_step(double *q_in, double *q_out, double *boltz_bond, double *exp_dw);
public:
    CpuPseudoDiscrete(ComputationBox *cb, PolymerChain *pc, FFT *ff);
    ~CpuPseudoDiscrete();

    void update() override;
    std::array<double,3> dq_dl() override;
    void find_phi(double *phi, double *q_1_init, double *q_2_init,
                  double *w_block, double &single_partition) override;
    void get_partition(double *q_1_out, int n1, double *q_2_out, int n2) override;
};
#endif
