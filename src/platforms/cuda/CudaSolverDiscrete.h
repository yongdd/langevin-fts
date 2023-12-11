/*-------------------------------------------------------------
* This is a derived CudaSolverDiscrete class
*------------------------------------------------------------*/

#ifndef CUDA_PSEUDO_DISCRETE_H_
#define CUDA_PSEUDO_DISCRETE_H_

#include <array>
#include <cufft.h>

#include "ComputationBox.h"
#include "Polymer.h"
#include "Molecules.h"
#include "Solver.h"
#include "CudaCommon.h"
#include "CudaPseudo.h"
#include "Scheduler.h"

class CudaSolverDiscrete : public Solver
{
private:
    // Pseudo-spectral PDE solver
    CudaPseudo *propagator_solver;

    // Two streams for each gpu
    cudaStream_t streams[MAX_GPUS][2]; // one for kernel execution, the other for memcpy

    // All elements are 1 for initializing propagators
    double *d_q_unity; 

    // q_mask to make impenetrable region for nano particles
    double *d_q_mask[MAX_GPUS];

    // For stress calculation: compute_stress()
    double *d_stress_q[MAX_GPUS][2];  // one for prev, the other for next

    // Scheduler for propagator computation 
    Scheduler *sc;
    // The number of parallel streams
    const int N_SCHEDULER_STREAMS = 2;
    // key: (dep), value: array pointer
    std::map<std::string, double*> d_propagator_junction;
    // Temporary arrays for compute segment at junction
    double *d_q_half_step, *d_q_junction;
    // gpu memory space to store propagator, key: (dep) + monomer_type, value: propagator
    std::map<std::string, double **> d_propagator;
    // Map for deallocation of d_propagator
    std::map<std::string, int> propagator_size;
    // Temporary arrays for device_1, one for prev, the other for next
    double *d_propagator_device_1[2];
    // Check if computation of propagator is finished
    #ifndef NDEBUG  
    std::map<std::string, bool *> propagator_finished;
    #endif

    // Total partition functions for each polymer
    double* single_polymer_partitions;
    // Remember one segment for each polymer chain to compute total partition function
    // (polymer id, propagator forward, propagator backward, monomer_type, n_aggregated)
    std::vector<std::tuple<int, double *, double *, std::string, int>> single_partition_segment;

    // gpu memory space to store concentration, key: (polymer id, dep_v, dep_u) (assert(dep_v <= dep_u)), value: concentration
    std::map<std::tuple<int, std::string, std::string>, double *> d_phi_block;
    // Temp array for concentration computation
    double *d_phi;
    
    // Remember propagators and bond length for each segment to prepare stress computation
    // key: (polymer id, dep_v, dep_u), value (propagator forward, propagator backward, is_half_bond_length)
    std::map<std::tuple<int, std::string, std::string>, std::vector<std::tuple<double *, double *, bool>>> block_stress_info;

    // Total partition functions for each solvent
    double* single_solvent_partitions;

    // Solvent concentrations
    std::vector<double *> d_phi_solvent;

    // Accessible volume of polymers excluding mask region
    double accessible_volume;

    // Calculate concentration of one block
    void calculate_phi_one_block(double *d_phi, double **d_q_1, double **d_q_2, double *d_exp_dw, const int N, const int N_OFFSET, const int N_ORIGINAL);

    // Compute statistics with inputs from selected device arrays
    void compute_statistics(std::string device,
        std::map<std::string, const double*> w_input,
        std::map<std::string, const double*> q_init = {},
        double* q_mask=nullptr);
public:
    CudaSolverDiscrete(ComputationBox *cb, Molecules *molecules, PropagatorsAnalyzer *propagators_analyzer);
    ~CudaSolverDiscrete();

    void update_bond_function() override;
    void compute_statistics(
        std::map<std::string, const double*> w_block,
        std::map<std::string, const double*> q_init = {},
        double* q_mask=nullptr) override
    {
        compute_statistics("cpu", w_block, q_init, q_mask);
    };
    void compute_statistics_device(
        std::map<std::string, const double*> d_w_block,
        std::map<std::string, const double*> d_q_init = {},
        double* d_q_mask=nullptr) override
    {
        compute_statistics("gpu", d_w_block, d_q_init, d_q_mask);
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