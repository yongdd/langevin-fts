/*-------------------------------------------------------------
* This is a derived CudaPseudoReduceMemoryDiscrete class
*------------------------------------------------------------*/

#ifndef CUDA_PSEUDO_REDUCE_MEMORY_DISCRETE_H_
#define CUDA_PSEUDO_REDUCE_MEMORY_DISCRETE_H_

#include <array>
#include <cufft.h>

#include "ComputationBox.h"
#include "PolymerChain.h"
#include "Mixture.h"
#include "Pseudo.h"
#include "CudaCommon.h"
#include "Scheduler.h"

class CudaPseudoReduceMemoryDiscrete : public Pseudo
{
private:
    // for pseudo-spectral: one_step()
    cufftHandle plan_for, plan_bak;
    double *d_q_step1, *d_q_step2;
    ftsComplex *d_qk_in;
    double **d_q;
    double **d_propagator_sub_dep;
    double *d_q_half_step, *d_q_junction;

    // for stress calculation: compute_stress()
    cufftHandle plan_for_two;
    double **d_q_in_temp;
    ftsComplex *d_qk_in_2;

    double *d_fourier_basis_x;
    double *d_fourier_basis_y;
    double *d_fourier_basis_z;
    double *d_q_multi, *d_stress_sum;

    // three streams for overlapping kernel execution and data transfers 
    cudaStream_t *streams;

    // key: (dep) + monomer_type, value: partition functions
    std::map<std::string, double *> propagator;

    // check if computation of propagator is finished
    #ifndef NDEBUG
    std::map<std::string, bool *> propagator_finished;
    #endif

    // key: (polymer id, dep_v, dep_u) (assert(dep_v <= dep_u)), value: concentrations
    std::map<std::tuple<int, std::string, std::string>, double *> block_phi;

    // key: (dep), value: array pointer
    std::map<std::string, double*> q_junction_cache;
    
    std::map<std::string, double*> d_boltz_bond;        // boltzmann factor for the single bond
    std::map<std::string, double*> d_boltz_bond_half;   // boltzmann factor for the half bond
    std::map<std::string, double*> d_exp_dw;            // boltzmann factor for the single segment
    std::map<std::string, double*>   exp_dw;
    
    // total partition functions for each polymer
    double* single_partitions;

    void one_step(double *d_q_in, double *d_q_out, double *d_boltz_bond, double *d_exp_dw);
    void half_bond_step(double *d_q_in, double *d_q_out, double *d_boltz_bond_half);
    void calculate_phi_one_block(double *d_phi, double *q_1, double *q_2, double *exp_dw, const int N, const int N_OFFSET, const int N_ORIGINAL);

public:
    CudaPseudoReduceMemoryDiscrete(ComputationBox *cb, Mixture *mx);
    ~CudaPseudoReduceMemoryDiscrete();

    void update_bond_function() override;
    void compute_statistics(
        std::map<std::string, double*> w_input,
        std::map<std::string, double*> q_init) override;
    double get_total_partition(int polymer) override;
    void get_monomer_concentration(std::string monomer_type, double *phi) override;
    void get_polymer_concentration(int polymer, double *phi) override;
    std::vector<double> compute_stress() override;
    void get_chain_propagator(double *q_out, int polymer, int v, int u, int n) override;
};

#endif
