/*-------------------------------------------------------------
* This is a derived CudaComputationReduceMemoryDiscrete class
*------------------------------------------------------------*/

#ifndef CUDA_PSEUDO_REDUCE_MEMORY_DISCRETE_H_
#define CUDA_PSEUDO_REDUCE_MEMORY_DISCRETE_H_

#include <array>
#include <cufft.h>

#include "ComputationBox.h"
#include "Polymer.h"
#include "Molecules.h"
#include "PropagatorComputation.h"
#include "CudaCommon.h"
#include "CudaSolverPseudo.h"
#include "Scheduler.h"

class CudaComputationReduceMemoryDiscrete : public PropagatorComputation
{
private:
    // Pseudo-spectral PDE solver
    CudaSolverPseudo *propagator_solver;

    // The number of parallel streams for propagator computation
    int n_streams;

    // Two streams for each gpu
    cudaStream_t streams[MAX_STREAMS][2]; // one for kernel execution, the other for memcpy

    // All elements are 1 for initializing propagators
    double *d_q_unity[MAX_GPUS]; 

    // q_mask to make impenetrable region for nano particles
    double *d_q_mask[MAX_GPUS];

    // One for prev, the other for next
    double *d_q_pair[MAX_STREAMS][2];

    // For pseudo-spectral: advance_one propagator()
    double *d_q_one[MAX_STREAMS][2];               // one for prev, the other for next
    double *d_propagator_sub_dep[MAX_STREAMS][2];  // one for prev, the other for next

    // For concentration computation
    double *d_q_block_v[2];    // one for prev, the other for next
    double *d_q_block_u[2];    // one for prev, the other for next
    double *d_phi;

    // Scheduler for propagator computation 
    Scheduler *sc;

    // Temporary arrays for compute segment at junction
    double *d_q_half_step[MAX_STREAMS], *d_q_junction[MAX_STREAMS];

    // key: (dep), value: array pointer
    std::map<std::string, double*> propagator_junction;
    // Host pinned memory space to store propagator, key: (dep) + monomer_type, value: propagator
    std::map<std::string, double **> propagator;
    // Map for deallocation of d_propagator
    std::map<std::string, int> propagator_size;
    // Check if computation of propagator is finished
    #ifndef NDEBUG
    std::map<std::string, bool *> propagator_finished;
    #endif

    // Total partition functions for each polymer
    double* single_polymer_partitions;
    // Remember one segment for each polymer chain to compute total partition function
    // (polymer id, propagator forward, propagator backward, monomer_type, n_repeated)
    std::vector<std::tuple<int, double *, double *, std::string, int>> single_partition_segment;

    // key: (polymer id, key_left, key_right) (assert(key_left <= key_right)), value: concentrations
    std::map<std::tuple<int, std::string, std::string>, double *> phi_block;

    // Remember propagators and bond length for each segment to prepare stress computation
    // key: (polymer id, key_left, key_right), value (propagator forward, propagator backward, is_half_bond_length)
    std::map<std::tuple<int, std::string, std::string>, std::vector<std::tuple<double *, double *, bool>>> block_stress_computation_plan;

    // Total partition functions for each solvent
    double* single_solvent_partitions;

    // Solvent concentrations
    std::vector<double *> phi_solvent;

    // Calculate concentration of one block
    void calculate_phi_one_block(double *phi, double **q_1, double **q_2, double *d_exp_dw, const int N_RIGHT, const int N_LEFT, const double NORM);

    // Compute statistics with inputs from selected device arrays
    void compute_statistics(std::string device,
        std::map<std::string, const double*> w_input,
        std::map<std::string, const double*> q_init = {});
public:
    CudaComputationReduceMemoryDiscrete(ComputationBox *cb, Molecules *molecules, PropagatorAnalyzer *propagator_analyzer);
    ~CudaComputationReduceMemoryDiscrete();

    void update_laplacian_operator() override;
    void compute_statistics(
        std::map<std::string, const double*> w_block,
        std::map<std::string, const double*> q_init = {}) override
    {
        compute_statistics("cpu", w_block, q_init);
    };
    void compute_statistics_device(
        std::map<std::string, const double*> d_w_block,
        std::map<std::string, const double*> d_q_init = {}) override
    {
        compute_statistics("gpu", d_w_block, d_q_init);
    };
    double get_total_partition(int polymer) override;
    void get_total_concentration(std::string monomer_type, double *phi) override;
    void get_total_concentration(int polymer, std::string monomer_type, double *phi) override;
    void get_block_concentration(int polymer, double *phi) override;
    std::vector<double> compute_stress() override;
    void get_chain_propagator(double *q_out, int polymer, int v, int u, int n) override;

    double get_solvent_partition(int s) override;
    void get_solvent_concentration(int s, double *phi) override;

    // For tests
    bool check_total_partition() override;
};

#endif
