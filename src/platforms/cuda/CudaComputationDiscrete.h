/*-------------------------------------------------------------
* This is a derived CudaComputationDiscrete class
*------------------------------------------------------------*/

#ifndef CUDA_COMPUTATION_DISCRETE_H_
#define CUDA_COMPUTATION_DISCRETE_H_

#include <array>
#include <cufft.h>

#include "ComputationBox.h"
#include "Polymer.h"
#include "Molecules.h"
#include "PropagatorComputation.h"
#include "CudaCommon.h"
#include "CudaSolver.h"
#include "Scheduler.h"

class CudaComputationDiscrete : public PropagatorComputation
{
private:
    // Pseudo-spectral PDE solver
    CudaSolver<double> *propagator_solver;

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

    // Scheduler for propagator computation 
    Scheduler *sc;

    // Map for propagator q(r,s)
    std::map<std::string, double **> d_propagator;
    // Map for q(r,s+1/2, ; code)
    std::map<std::string, double **> d_propagator_half_steps;
    // Map for deallocation of propagator
    std::map<std::string, int> propagator_size;

    // Temporary arrays for device_1, one for prev, the other for next
    double *d_propagator_device[MAX_GPUS][2];
    // Check if computation of propagator is finished
    #ifndef NDEBUG
    std::map<std::string, bool *> propagator_finished;
    std::map<std::string, std::map<int, bool>> propagator_half_steps_finished;
    #endif

    // Remember one segment for each polymer chain to compute total partition function
    // (polymer id, propagator forward, propagator backward, monomer_type, n_repeated)
    std::vector<std::tuple<int, double *, double *, std::string, int>> single_partition_segment;

    // gpu memory space to store concentration, key: (polymer id, key_left, key_right) (assert(key_left <= key_right)), value: concentration
    std::map<std::tuple<int, std::string, std::string>, double *> d_phi_block;
    // Temp array for concentration computation
    double *d_phi;
    
    // Remember propagators and bond length for each segment to prepare stress computation
    // key: (polymer id, key_left, key_right), value (propagator forward, propagator backward, is_half_bond_length)
    std::map<std::tuple<int, std::string, std::string>, std::vector<std::tuple<double *, double *, bool>>> block_stress_computation_plan;

    // Solvent concentrations
    std::vector<double *> d_phi_solvent;

    // Calculate concentration of one block
    void calculate_phi_one_block(double *d_phi, double **d_q_1, double **d_q_2, double *d_exp_dw, const int N_RIGHT, const int N_LEFT);

public:
    CudaComputationDiscrete(ComputationBox *cb, Molecules *molecules, PropagatorComputationOptimizer *propagator_computation_optimizer);
    ~CudaComputationDiscrete();

    void update_laplacian_operator() override;

    void compute_propagators(
        std::map<std::string, const double*> w_block,
        std::map<std::string, const double*> q_init = {}) override;

    void compute_concentrations() override;

    // Compute statistics with inputs from selected device arrays
    void compute_statistics(
        std::map<std::string, const double*> w_input,
        std::map<std::string, const double*> q_init = {}) override;
    
    void compute_stress() override;
    double get_total_partition(int polymer) override;
    void get_chain_propagator(double *q_out, int polymer, int v, int u, int n) override;

    // Canonical ensemble
    void get_total_concentration(std::string monomer_type, double *phi) override;
    void get_total_concentration(int polymer, std::string monomer_type, double *phi) override;
    void get_block_concentration(int polymer, double *phi) override;

    double get_solvent_partition(int s) override;
    void get_solvent_concentration(int s, double *phi) override;

    // Grand canonical ensemble
    void get_total_concentration_gce(double fugacity, int polymer, std::string monomer_type, double *phi) override;

    // For tests
    bool check_total_partition() override;
};

#endif
