/*-------------------------------------------------------------
* This is an abstract CpuPseudoContinuous class
*------------------------------------------------------------*/

#ifndef CPU_PSEUDO_CONTINUOUS_H_
#define CPU_PSEUDO_CONTINUOUS_H_

#include "SimulationBox.h"
#include "PolymerChain.h"
#include "Pseudo.h"
#include "FFT.h"

class CpuPseudoContinuous : public Pseudo
{
private:
    FFT *fft;
    double *q_1, *q_2;
    double *boltz_bond_a, *boltz_bond_a_half;
    double *boltz_bond_b, *boltz_bond_b_half;
    void one_step(double *q_in, double *q_out, 
                  double *boltz_bond, double *boltz_bond_half,
                  double *exp_dw, double *exp_dw_half);
    void calculate_phi_one_type(double *phi, const int N_START, const int N_END);
    void init_simpson_rule_coeff(double *coeff, const int N);
public:
    CpuPseudoContinuous(SimulationBox *sb, PolymerChain *pc, FFT *ff);
    ~CpuPseudoContinuous();
    
    void update() override;
    std::array<double,3> dq_dl() override;
    void find_phi(double *phi_a,  double *phi_b,
                  double *q_1_init, double *q_2_init,
                  double *w_a, double *w_b, double &single_partition) override;
    void get_partition(double *q_1_out, int n1, double *q_2_out, int n2) override;
};
#endif
