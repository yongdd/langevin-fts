/*-------------------------------------------------------------
* This is a derived CudaPseudoDiscrete class
*------------------------------------------------------------*/

#ifndef CUDA_PSEUDO_DISCRETE_H_
#define CUDA_PSEUDO_DISCRETE_H_

#include <array>
#include <cufft.h>

#include "ComputationBox.h"
#include "PolymerChain.h"
#include "Mixture.h"
#include "Pseudo.h"
#include "CudaCommon.h"
#include "Scheduler.h"

class CudaPseudoDiscrete : public Pseudo
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

    // for pseudo-spectral: one_step()
    ftsComplex *d_qk_in_1;
    double *d_q_half_step, *d_q_junction;

    // for stress calculation: compute_stress()
    double *d_fourier_basis_x;
    double *d_fourier_basis_y;
    double *d_fourier_basis_z;
    ftsComplex *d_qk_1, *d_qk_2;
    double *d_q_multi, *d_stress_sum;
    
    // to compute concentration
    double *d_phi;

    // key: (dep) + species, value: partition functions
    std::map<std::string, double *> d_unique_partition;
    Scheduler *sc;          // scheduler for partial partition function
    const int N_STREAM = 2; // the number of job threads

    // key: (dep_v, dep_u) (assert(dep_v <= dep_u)), value: concentrations
    std::map<std::tuple<std::string, std::string, int>, double *> d_unique_phi;

    // key: (dep), value: array pointer
    std::map<std::string, double*> d_unique_q_junctions;
    
    std::map<std::string, double*> d_boltz_bond;        // boltzmann factor for the single bond
    std::map<std::string, double*> d_boltz_bond_half;   // boltzmann factor for the half bond
    std::map<std::string, double*> d_exp_dw;            // boltzmann factor for the single segment

    // total partition functions for each polymer
    double* single_partitions;

    void one_step(double *d_q_in, double *d_q_out, double *d_boltz_bond, double *d_exp_dw);
    void half_bond_step(double *d_q_in, double *d_q_out, double *d_boltz_bond_half);
    void calculate_phi_one_type(double *d_phi, double *d_q_1, double *d_q_2, double *d_exp_dw, const int N);

public:
    CudaPseudoDiscrete(ComputationBox *cb, Mixture *mx);
    ~CudaPseudoDiscrete();

    void update() override;
    void compute_statistics(
        std::map<std::string, double*> q_init,
        std::map<std::string, double*> w_block) override;
    double get_total_partition(int polymer) override;
    void get_species_concentration(std::string species, double *phi) override;
    void get_polymer_concentration(int polymer, double *phi) override;
    std::array<double,3> compute_stress() override;
    void get_partial_partition(double *q_out, int polymer, int v, int u, int n) override;
};

#endif
