/*-------------------------------------------------------------
* This is a derived CudaPseudoLinearDiscrete class
*------------------------------------------------------------*/

#ifndef CUDA_PSEUDO_LINEAR_DISCRETE_H_
#define CUDA_PSEUDO_LINEAR_DISCRETE_H_

#include <array>
#include <cufft.h>

#include "ComputationBox.h"
#include "PolymerChain.h"
#include "Pseudo.h"
#include "CudaCommon.h"

class CudaPseudoLinearDiscrete : public Pseudo
{
private:
    cufftHandle plan_for, plan_bak;

    // partition function and complementary partition are 
    // contiguously stored in q_d for every segment step.
    // In other words,
    // q       (r,1)   = q_d[0]          ~ q_d[MM-1]
    // q^dagger(r,N)   = q_d[MM]         ~ q_d[2*MM-1]
    // q       (r,2)   = q_d[2*MM]       ~ q_d[3*MM-1]
    // q^dagger(r,N-2) = q_d[3*MM]       ~ q_d[4*MM-1]
    // ......
    // q       (r,n)   = q_d[(2*n-2)*MM] ~ q_d[(2*n-1)*MM-1]
    // q^dagger(r,N-n) = q_d[(2*n-1)*MM] ~ q_d[(2*n  )*MM-1]
    // ......
    
    double *d_q, *d_phi;
    ftsComplex *d_k_q_in;

    double **d_boltz_bond, **d_boltz_bond_middle;       // boltzmann factor for the segment bond

    std::map<std::string, double*> d_exp_dw;            // boltzmann factor for the single segment
    std::map<std::string, double*> d_exp_dw_half;       // boltzmann factor for the half segment

    std::vector<int> get_block_start();
    void one_step(double *d_q_in,         double *d_q_out,
                  double *d_boltz_bond_1, double *d_boltz_bond_2,
                  double *d_exp_dw_1,     double *d_exp_dw_2);
public:

    CudaPseudoLinearDiscrete(ComputationBox *cb, PolymerChain *pc);
    ~CudaPseudoLinearDiscrete();

    void update() override;
    void compute_statistics(
        std::map<std::string, double*> q_init,
        std::map<std::string, double*> w_block,
        double *phi, double &single_partition) override;
    std::array<double,3> dq_dl() override;
    void get_partition(double *q_out, int v, int u, int n) override;
};

#endif
