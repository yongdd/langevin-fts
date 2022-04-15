/*-------------------------------------------------------------
* This is an abstract CudaPseudoGaussian class
*------------------------------------------------------------*/

#ifndef CUDA_PSEUDO_GAUSSIAN_H_
#define CUDA_PSEUDO_GAUSSIAN_H_

#include <array>
#include <cufft.h>
#include "SimulationBox.h"
#include "Pseudo.h"
#include "CudaCommon.h"

class CudaPseudoGaussian : public Pseudo
{
private:
    cufftHandle plan_for, plan_bak;

    double *d_temp, *temp_arr;
    double *d_q_1, *d_q_2;
    double *d_q_step1, *d_q_step2;
    ftsComplex *d_k_q_in;

    double *d_exp_dw_a, *d_exp_dw_a_half;
    double *d_exp_dw_b, *d_exp_dw_b_half;
    double *d_phi_a,  *d_phi_b;
    
    double *d_boltz_bond_a, *d_boltz_bond_a_half;
    double *d_boltz_bond_b, *d_boltz_bond_b_half;

    void one_step(double *d_q_1_in, double *d_q_1_out,
                  double *d_q_2_in, double *d_q_2_out,
                  double *d_boltz_bond_1, double *d_boltz_bond_1_half,
                  double *d_boltz_bond_2, double *d_boltz_bond_2_half,
                  double *d_exp_dw_1, double *d_exp_dw_1_half,
                  double *d_exp_dw_2, double *d_exp_dw_2_half);
public:

    CudaPseudoGaussian(SimulationBox *sb, PolymerChain *pc);
    ~CudaPseudoGaussian();

    void update() override;

    void find_phi(double *phi_a,  double *phi_b,
                  double *q_1_init, double *q_2_init,
                  double *w_a, double *w_b, double &single_partition) override;

    void get_partition(double *q_1_out, int n1, double *q_2_out, int n2) override;
};

#endif
