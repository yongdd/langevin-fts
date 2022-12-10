/*-------------------------------------------------------------
* This is a derived CudaPseudoBranchedDiscrete class
*------------------------------------------------------------*/

#ifndef CUDA_PSEUDO_BRANCHED_DISCRETE_H_
#define CUDA_PSEUDO_BRANCHED_DISCRETE_H_

#include <array>
#include <cufft.h>

#include "ComputationBox.h"
#include "PolymerChain.h"
#include "Pseudo.h"
#include "CudaCommon.h"

class CudaPseudoBranchedDiscrete : public Pseudo
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

    // for pseudo-spectral: one-step()
    ftsComplex *d_qk_in;
    double *d_q_half_step, *d_q_junction;

    // for stress calculation: dq_dl()
    double *d_fourier_basis_x;
    double *d_fourier_basis_y;
    double *d_fourier_basis_z;
    ftsComplex *d_qk_1, *d_qk_2;
    double *d_q_multi, *d_stress_sum;

    // key: (dep) + species, value: branched_pseudo_edge
    std::map<std::string, branched_pseudo_edge, std::greater<std::string>> d_reduced_edges;

    // key: (dep_v, dep_u) (assert(dep_v <= dep_u)), value: branched_pseudo_block
    std::map<std::pair<std::string, std::string>, branched_pseudo_block> d_reduced_blocks; 

    // key: (dep), value: array pointer
    std::map<std::string, double*> d_reduced_q_junctions;
    
    std::map<std::string, double*> d_boltz_bond;        // boltzmann factor for the single bond
    std::map<std::string, double*> d_boltz_bond_half;   // boltzmann factor for the half bond
    std::map<std::string, double*> d_exp_dw;            // boltzmann factor for the single segment

    void one_step(double *d_q_in, double *d_q_out, double *d_boltz_bond, double *d_exp_dw);
    void half_bond_step(double *d_q_in, double *d_q_out, double *d_boltz_bond_half);
    void calculate_phi_one_type(double *d_phi, double *d_q_1, double *d_q_2, const int N);

public:
    CudaPseudoBranchedDiscrete(ComputationBox *cb, PolymerChain *pc);
    ~CudaPseudoBranchedDiscrete();

    void update() override;
    void compute_statistics(
        std::map<std::string, double*> q_init,
        std::map<std::string, double*> w_block,
        double *phi, double &single_partition) override;
    std::array<double,3> dq_dl() override;
    void get_partition(double *q_out, int v, int u, int n) override;
};

#endif
