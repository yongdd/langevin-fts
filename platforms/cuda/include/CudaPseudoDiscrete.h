/*-------------------------------------------------------------
* This is an abstract CudaPseudoDiscrete class
*------------------------------------------------------------*/

#ifndef CUDA_PSEUDO_DISCRETE_H_
#define CUDA_PSEUDO_DISCRETE_H_

#include <array>
#include <cufft.h>
#include "SimulationBox.h"
#include "Pseudo.h"
#include "CudaCommon.h"

class CudaPseudoDiscrete : public Pseudo
{
private:
    cufftHandle plan_for, plan_bak;

    // partition function and complementry partition are 
    // contiguously stored in q_d for every contour step.
    // In other words,
    // q       (r,1)   = q_d[0]          ~ q_d[MM-1]
    // q^dagger(r,N)   = q_d[MM]         ~ q_d[2*MM-1]
    // q       (r,2)   = q_d[2*MM]       ~ q_d[3*MM-1]
    // q^dagger(r,N-2) = q_d[3*MM]       ~ q_d[4*MM-1]
    // ......
    // q       (r,n)   = q_d[(2*n-2)*MM] ~ q_d[(2*n-1)*MM-1]
    // q^dagger(r,N-n) = q_d[(2*n-1)*MM] ~ q_d[(2*n  )*MM-1]
    // ......
    
    double *d_q;
    ftsComplex *d_k_q_in;

    double *d_boltz_bond_a, *d_boltz_bond_b, *d_boltz_bond_ab;
    double *d_phi_a, *d_exp_dw_a;
    double *d_phi_b, *d_exp_dw_b;
                 
    void one_step(double *d_q_in,         double *d_q_out,
                  double *d_boltz_bond_1, double *d_boltz_bond_2,
                  double *d_exp_dw_1,     double *d_exp_dw_2);
public:

    CudaPseudoDiscrete(SimulationBox *sb, PolymerChain *pc);
    ~CudaPseudoDiscrete();

    void update() override;
    std::array<double,3> dq_dl() override;
    void find_phi(double *phi_a,  double *phi_b,
                  double *q_1_init, double *q_2_init,
                  double *w_a, double *w_b, double &single_partition) override;
    void get_partition(double *q_1_out, int n1, double *q_2_out, int n2) override;
};

#endif
