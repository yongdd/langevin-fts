/*-------------------------------------------------------------
* This is a derived CudaPseudoContinuous class
*------------------------------------------------------------*/

#ifndef CUDA_PSEUDO_CONTINUOUS_H_
#define CUDA_PSEUDO_CONTINUOUS_H_

#include <array>
#include <cufft.h>

#include "ComputationBox.h"
#include "BranchedPolymerChain.h"
#include "PseudoBranched.h"
#include "CudaCommon.h"

class CudaPseudoContinuous : public PseudoBranched
{
private:
    cufftHandle plan_for, plan_bak;

    double *d_q_1, *d_q_2;
    double *d_q_step1, *d_q_step2;
    ftsComplex *d_k_q_in;

    //double *d_phi_a,  *d_phi_b;
    double *d_phi;

    std::map<std::string, double*> d_boltz_bond;        // boltzmann factor for the single bond
    std::map<std::string, double*> d_boltz_bond_half;   // boltzmann factor for the half bond
    std::map<std::string, double*> d_exp_dw;            // boltzmann factor for the single segment
    std::map<std::string, double*> d_exp_dw_half;       // boltzmann factor for the half segment

    std::vector<int> get_block_start();
    void one_step(double *d_q_1_in, double *d_q_1_out,
                  double *d_q_2_in, double *d_q_2_out,
                  double *d_boltz_bond_1, double *d_boltz_bond_1_half,
                  double *d_boltz_bond_2, double *d_boltz_bond_2_half,
                  double *d_exp_dw_1, double *d_exp_dw_1_half,
                  double *d_exp_dw_2, double *d_exp_dw_2_half);

    void calculate_phi_one_type(double *d_phi, const int N_START, const int N_END);
public:

    CudaPseudoContinuous(ComputationBox *cb, BranchedPolymerChain *pc);
    ~CudaPseudoContinuous();

    void update() override;
    void compute_statistics(
        std::map<std::string, double*> q_init,
        std::map<std::string, double*> w_block,
        double *phi, double &single_partition) override;
    std::array<double,3> dq_dl() override;
    void get_partition(double *q_out, int v, int u, int n) override;
};

#endif
