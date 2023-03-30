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

    // for pseudo-spectral: one_step()
    double *d_q_unity; // all elements are 1 for initializing propagtors
    cufftHandle plan_for_1[MAX_GPUS], plan_bak_1[MAX_GPUS];
    cufftHandle plan_for_two[MAX_GPUS], plan_bak_two[MAX_GPUS];

    double *d_q_step1_1[MAX_GPUS], *d_q_step2_1[MAX_GPUS];
    double *d_q_step1_two, *d_q_step2_two;
    ftsComplex *d_qk_in_1[MAX_GPUS];
    ftsComplex *d_qk_in_two[MAX_GPUS];

    // for stress calculation: compute_stress()
    double *d_fourier_basis_x[MAX_GPUS];
    double *d_fourier_basis_y[MAX_GPUS];
    double *d_fourier_basis_z[MAX_GPUS];
    double *d_stress_q[MAX_GPUS][2];  // one for prev, the other for next
    double *d_stress_sum[MAX_GPUS];
    double *d_q_multi[MAX_GPUS];

    // to compute concentration
    double *d_phi;

    // one stream for each gpu
    cudaStream_t streams[MAX_GPUS][2]; // one for kernel execution, the other for memcpy

    // key: (dep) + monomer_type, value: partition function
    std::map<std::string, double **> d_propagator;
    std::map<std::string, int> propagator_size; // for deallocation

    // temporary arrays for device_1
    double *d_propagator_device_1[2]; // one for prev, the other for next

    // check if computation of propagator is finished
    #ifndef NDEBUG
    std::map<std::string, bool *> propagator_finished;
    #endif

    // the number of parallel jobs for propagator computation
    const int N_PARALLEL_STREAMS = 2;

    // scheduler for propagator
    Scheduler *sc;

    // key: (polymer id, dep_v, dep_u) (assert(dep_v <= dep_u)), value: concentration
    std::map<std::tuple<int, std::string, std::string>, double *> d_block_phi;

    std::map<std::string, double*> d_boltz_bond[MAX_GPUS];        // boltzmann factor for the single bond
    std::map<std::string, double*> d_boltz_bond_half[MAX_GPUS];   // boltzmann factor for the half bond
    std::map<std::string, double*> d_exp_dw[MAX_GPUS];            // boltzmann factor for the single segment
    std::map<std::string, double*> d_exp_dw_half[MAX_GPUS];       // boltzmann factor for the half segment

    // total partition functions for each polymer
    double* single_partitions;

    void one_step_1(const int GPU,
            double *d_q_in, double *d_q_out,
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
