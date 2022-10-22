/*-------------------------------------------------------------
* This is an abstract CudaPseudoContinuous class
*------------------------------------------------------------*/

#ifndef CUDA_PSEUDO_CONTINUOUS_H_
#define CUDA_PSEUDO_CONTINUOUS_H_

#include <array>
#include <cufft.h>
#include "SimulationBox.h"
#include "Pseudo.h"
#include "CudaCommon.h"

class CudaPseudoContinuous : public Pseudo
{
private:
    cufftHandle plan_for, plan_bak;

    double *d_temp, *temp_arr;
    double *d_q_1, *d_q_2;
    double *d_q_step1, *d_q_step2;
    ftsComplex *d_k_q_in;

    //double *d_exp_dw_a, *d_exp_dw_a_half;
    double **d_exp_dw, **d_exp_dw_half;
    //double *d_phi_a,  *d_phi_b;
    double *d_phi;
    
    //double *d_boltz_bond_a, *d_boltz_bond_a_half;
    double **d_boltz_bond, **d_boltz_bond_half;

    void one_step(double *d_q_1_in, double *d_q_1_out,
                  double *d_q_2_in, double *d_q_2_out,
                  double *d_boltz_bond_1, double *d_boltz_bond_1_half,
                  double *d_boltz_bond_2, double *d_boltz_bond_2_half,
                  double *d_exp_dw_1, double *d_exp_dw_1_half,
                  double *d_exp_dw_2, double *d_exp_dw_2_half);
    void calculate_phi_one_type(double *d_phi, const int N_START, const int N_END);
    void init_simpson_rule_coeff(double *coeff, const int N);
public:

    CudaPseudoContinuous(SimulationBox *sb, PolymerChain *pc);
    ~CudaPseudoContinuous();

    void update() override;
    std::array<double,3> dq_dl() override;
    void find_phi(double *phi, double *q_1_init, double *q_2_init,
                  double *w_block, double &single_partition) override;
    void get_partition(double *q_1_out, int n1, double *q_2_out, int n2) override;
};

#endif
