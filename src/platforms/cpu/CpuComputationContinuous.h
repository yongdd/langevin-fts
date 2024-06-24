/*-------------------------------------------------------------
* This is a derived CpuComputationContinuous class
*------------------------------------------------------------*/

#ifndef CPU_PSEUDO_CONTINUOUS_H_
#define CPU_PSEUDO_CONTINUOUS_H_

#include <string>
#include <vector>
#include <map>

#include "ComputationBox.h"
#include "Polymer.h"
#include "Molecules.h"
#include "PropagatorAnalyzer.h"
#include "PropagatorComputation.h"
#include "CpuSolverPseudo.h"
#include "Scheduler.h"

class CpuComputationContinuous : public PropagatorComputation
{
private:
    // Pseudo-spectral PDE solver
    CpuSolver *propagator_solver;
    // Scheduler for propagator
    Scheduler *sc;
    // The number of parallel streams for propagator computation
    int n_streams;
    // key: (dep) + monomer_type, value: propagator
    std::map<std::string, double **> propagator; 
    // Map for deallocation of propagator
    std::map<std::string, int> propagator_size;
    // Check if computation of propagator is finished
    #ifndef NDEBUG
    std::map<std::string, bool *> propagator_finished;
    #endif

    // Total partition functions for each polymer
    double* single_polymer_partitions;

    // Remember one segment for each polymer chain to compute total partition function
    // (polymer id, propagator forward, propagator backward, n_repeated)
    std::vector<std::tuple<int, double *, double *, int>> single_partition_segment;

    // key: (polymer id, key_left, key_right) (assert(key_left <= key_right)), value: concentrations
    std::map<std::tuple<int, std::string, std::string>, double *> phi_block;

    // Total partition functions for each solvent
    double* single_solvent_partitions;

    // Solvent concentrations
    std::vector<double *> phi_solvent;

    // Calculate concentration of one block
    void calculate_phi_one_block(double *phi, double **q_1, double **q_2, const int N_RIGHT, const int N_LEFT);
public:
    CpuComputationContinuous(ComputationBox *cb, Molecules *molecules, PropagatorAnalyzer* propagator_analyzer, std::string method);
    ~CpuComputationContinuous();
    
    void update_laplacian_operator() override;
    void compute_statistics(
        std::map<std::string, const double*> w_block,
        std::map<std::string, const double*> q_init = {}) override;
    void compute_statistics_device(
        std::map<std::string, const double*> w_block,
        std::map<std::string, const double*> q_init = {}) override
    {
        compute_statistics(w_block, q_init);
    };
    double get_total_partition(int polymer) override;
    void get_total_concentration(std::string monomer_type, double *phi) override;
    void get_total_concentration(int polymer, std::string monomer_type, double *phi) override;
    void get_block_concentration(int polymer, double *phi) override;

    double get_solvent_partition(int s) override;
    void get_solvent_concentration(int s, double *phi) override;

    std::vector<double> compute_stress() override;
    void get_chain_propagator(double *q_out, int polymer, int v, int u, int n) override;

    // For tests
    bool check_total_partition() override;
};
#endif
