/*-------------------------------------------------------------
* This is a derived CudaPseudoBranchedContinuous class
*------------------------------------------------------------*/

#ifndef CUDA_PSEUDO_BRANCHED_CONTINUOUS_H_
#define CUDA_PSEUDO_BRANCHED_CONTINUOUS_H_

#include <array>
#include <cufft.h>

#include "ComputationBox.h"
#include "PolymerChain.h"
#include "Mixture.h"
#include "Pseudo.h"
#include "CudaCommon.h"

class CudaPseudoBranchedContinuous : public Pseudo
{
private:
    cufftHandle plan_for, plan_bak;

    // for pseudo-spectral: one-step()
    double *d_q_step1, *d_q_step2;
    ftsComplex *d_qk_in;

    // for stress calculation: dq_dl()
    double *d_fourier_basis_x;
    double *d_fourier_basis_y;
    double *d_fourier_basis_z;
    ftsComplex *d_qk_1, *d_qk_2;
    double *d_q_multi, *d_stress_sum;

    // key: (dep) + species, value: partition function
    std::map<std::string, double *> d_reduced_partition;

    // key: (dep_v, dep_u) (assert(dep_v <= dep_u)), value: concentration
    std::map<std::tuple<std::string, std::string, int>, double *> d_reduced_phi;

    std::map<std::string, double*> d_boltz_bond;        // boltzmann factor for the single bond
    std::map<std::string, double*> d_boltz_bond_half;   // boltzmann factor for the half bond
    std::map<std::string, double*> d_exp_dw;            // boltzmann factor for the single segment
    std::map<std::string, double*> d_exp_dw_half;       // boltzmann factor for the half segment

    void one_step(double *d_q_in, double *d_q_out,
                  double *d_boltz_bond, double *d_boltz_bond_half,
                  double *d_exp_dw, double *d_exp_dw_half);
    void calculate_phi_one_type(double *d_phi, double *d_q_1, double *d_q_2, const int N);
public:

    CudaPseudoBranchedContinuous(ComputationBox *cb, Mixture *pc);
    ~CudaPseudoBranchedContinuous();

    void update() override;
    std::vector<double> compute_statistics(
        std::map<std::string, double*> q_init,
        std::map<std::string, double*> w_block,
        std::vector<double *> phi) override;
    std::vector<std::array<double,3>> dq_dl() override;
    void get_partition(double *q_out, int polymer, int v, int u, int n) override;
};

#endif
