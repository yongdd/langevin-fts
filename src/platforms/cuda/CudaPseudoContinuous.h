/*-------------------------------------------------------------
* This is a derived CudaPseudoContinuous class
*------------------------------------------------------------*/

#ifndef CUDA_PSEUDO_CONTINUOUS_H_
#define CUDA_PSEUDO_CONTINUOUS_H_

#include <array>
#include <cufft.h>

#include "ComputationBox.h"
#include "PolymerChain.h"
#include "Mixture.h"
#include "Pseudo.h"
#include "CudaCommon.h"
#include "Scheduler.h"

class CudaPseudoContinuous : public Pseudo
{
private:
    cufftHandle plan_for_1, plan_bak_1;
    cufftHandle plan_for_2, plan_bak_2;

    // for pseudo-spectral: one_step()
    double *d_q_step1_1, *d_q_step2_1;
    ftsComplex *d_qk_in_1;

    double *d_q_step1_2, *d_q_step2_2;
    ftsComplex *d_qk_in_2;

    // for stress calculation: compute_stress()
    double *d_fourier_basis_x;
    double *d_fourier_basis_y;
    double *d_fourier_basis_z;
    ftsComplex *d_qk_1, *d_qk_2;
    double *d_q_multi, *d_stress_sum;

    // to compute concentration
    double *d_phi;

    // key: (dep) + monomer_type, value: partition function
    std::map<std::string, double **> d_esssential_propagator;
    std::map<std::string, int> d_esssential_propagator_size; // for deallocation

    // check if computation of esssential_propagator is finished
    #ifndef NDEBUG
    std::map<std::string, bool *> esssential_propagator_finished;
    #endif

    // scheduler for partial partition function
    Scheduler *sc;
    // the number of job threads
    const int N_STREAM = 2;

    // key: (polymer id, dep_v, dep_u) (assert(dep_v <= dep_u)), value: concentration
    std::map<std::tuple<int, std::string, std::string>, double *> d_essential_block_phi;

    std::map<std::string, double*> d_boltz_bond;        // boltzmann factor for the single bond
    std::map<std::string, double*> d_boltz_bond_half;   // boltzmann factor for the half bond
    std::map<std::string, double*> d_exp_dw;            // boltzmann factor for the single segment
    std::map<std::string, double*> d_exp_dw_half;       // boltzmann factor for the half segment

    // total partition functions for each polymer
    double* single_partitions;

    void one_step_1(double *d_q_in, double *d_q_out,
                  double *d_boltz_bond, double *d_boltz_bond_half,
                  double *d_exp_dw, double *d_exp_dw_half);

    void one_step_2(double *d_q_in_1, double *d_q_in_2,
                  double *d_q_out_1, double *d_q_out_2,
                  double *d_boltz_bond_1, double *d_boltz_bond_2, 
                  double *d_boltz_bond_half_1, double *d_boltz_bond_half_2,         
                  double *d_exp_dw_1, double *d_exp_dw_2,
                  double *d_exp_dw_half_1, double *d_exp_dw_half_2);

    void calculate_phi_one_block(double *d_phi, double **d_q_1, double **d_q_2, const int N, const int N_OFFSET, const int N_ORIGINAL);
public:

    CudaPseudoContinuous(ComputationBox *cb, Mixture *pc);
    ~CudaPseudoContinuous();

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
