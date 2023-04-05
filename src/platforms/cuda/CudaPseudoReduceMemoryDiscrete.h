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

    // two streams for each gpu
    cudaStream_t streams[MAX_GPUS][2]; // one for kernel execution, the other for memcpy

    // for pseudo-spectral: advance_propagator()
    double *d_q_unity; // all elements are 1 for initializing propagators
    cufftHandle plan_for_one[MAX_GPUS], plan_bak_one[MAX_GPUS];
    cufftHandle plan_for_two[MAX_GPUS], plan_bak_two[MAX_GPUS];

    double *d_q_step_1_two[MAX_GPUS];

    ftsComplex *d_qk_in_1_one[MAX_GPUS];
    ftsComplex *d_qk_in_1_two[MAX_GPUS];

    double *d_q_one[MAX_GPUS][2];     // one for prev, the other for next
    double *d_q_two[2];               // one for prev, the other for next
    double *d_propagator_sub_dep[2];  // one for prev, the other for next

    // for concentration computation
    double *d_q_block_v[2];    // one for prev, the other for next
    double *d_q_block_u[2];    // one for prev, the other for next
    double *d_phi;

    // for pseudo-spectral: advance_one_propagator()
    double *d_q_half_step, *d_q_junction;

    // for stress calculation: compute_stress()
    double *d_fourier_basis_x[MAX_GPUS];
    double *d_fourier_basis_y[MAX_GPUS];
    double *d_fourier_basis_z[MAX_GPUS];
    double *d_stress_q[MAX_GPUS][2];  // one for prev, the other for next
    double *d_q_multi[MAX_GPUS];

    // variables for cub reduction sum
    size_t temp_storage_bytes[MAX_GPUS];
    double *d_temp_storage[MAX_GPUS];
    double *d_stress_sum[MAX_GPUS];
    double *d_stress_sum_out[MAX_GPUS];

    // scheduler for propagator computation 
    Scheduler *sc;
    // the number of parallel streams
    const int N_SCHEDULER_STREAMS = 2;
    // key: (dep), value: array pointer
    std::map<std::string, double*> propagator_junction;
    // host pinned memory space to store propagator, key: (dep) + monomer_type, value: propagator
    std::map<std::string, double **> propagator;
    // map for deallocation of d_propagator
    std::map<std::string, int> propagator_size;
    // check if computation of propagator is finished
    #ifndef NDEBUG
    std::map<std::string, bool *> propagator_finished;
    #endif

    // total partition functions for each polymer
    double* single_partitions;
    // remember one segment for each polymer chain to compute total partition function
    // (polymer id, propagator forward, propagator backward, monomer_type, n_superposed)
    std::vector<std::tuple<int, double *, double *, std::string, int>> single_partition_segment;

    // key: (polymer id, dep_v, dep_u) (assert(dep_v <= dep_u)), value: concentrations
    std::map<std::tuple<int, std::string, std::string>, double *> block_phi;

    // remember propagators and bond length for each segment to prepare stress computation
    // key: (polymer id, dep_v, dep_u), value (propagator forward, propagator backward, is_half_bond)
    std::map<std::tuple<int, std::string, std::string>, std::vector<std::tuple<double *, double *, bool>>> block_stress_info;

    // gpu arrays for pseudo-spectral
    std::map<std::string, double*> d_boltz_bond[MAX_GPUS];        // boltzmann factor for the single bond
    std::map<std::string, double*> d_boltz_bond_half[MAX_GPUS];   // boltzmann factor for the half bond
    std::map<std::string, double*> d_exp_dw[MAX_GPUS];            // boltzmann factor for the single segment

    // advance one propagator by one segment step
    void advance_one_propagator(const int GPU, double *d_q_in, double *d_q_out, double *d_boltz_bond, double *d_exp_dw);

    // advance two propagators by one segment step
    void advance_two_propagators(
            double *d_q_in_two, double *d_q_out_two,
            double *d_boltz_bond_1, double *d_boltz_bond_2,  
            double *d_exp_dw_1, double *d_exp_dw_2);

    // advance two propagators by one segment step in two GPUs
    void advance_two_propagators_two_gpus(double *d_q_in_1, double *d_q_in_2,
            double *d_q_out_1, double *d_q_out_2,
            double *d_boltz_bond_1, double *d_boltz_bond_2,
            double *d_exp_dw_1, double *d_exp_dw_2);

    // advance propagator by half bond step
    void advance_propagator_half_bond_step(const int GPU, double *d_q_in, double *d_q_out, double *d_boltz_bond_half);

    // calculate concentration of one block
    void calculate_phi_one_block(double *phi, double **q_1, double **q_2, double *d_exp_dw, const int N, const int N_OFFSET, const int N_ORIGINAL, const double NORM);

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
