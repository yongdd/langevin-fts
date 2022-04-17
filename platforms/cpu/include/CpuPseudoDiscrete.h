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
    double *q_1, *q_2;
    double *boltz_bond_a, *boltz_bond_b, *boltz_bond_ab;
    void one_step(double *q_in, double *q_out, double *boltz_bond, double *exp_dw);
public:
    CpuPseudoDiscrete(SimulationBox *sb, PolymerChain *pc, FFT *ff);
    ~CpuPseudoDiscrete();

    void update() override;
    std::array<double,3> dq_dl() override;
    void find_phi(
        double *phi_a,  double *phi_b,
        double *q_1_init, double *q_2_init,
        double *w_a, double *w_b, double &single_partition) override;
    void get_partition(double *q_1_out, int n1, double *q_2_out, int n2) override;
};
#endif
