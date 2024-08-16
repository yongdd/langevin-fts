/*-------------------------------------------------------------
* This is a derived CpuComputationDiscrete class
*------------------------------------------------------------*/

#ifndef CPU_PSEUDO_DISCRETE_H_
#define CPU_PSEUDO_DISCRETE_H_

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

class CpuComputationDiscrete : public PropagatorComputation
{
private:
    // Pseudo-spectral integral solver
    CpuSolverPseudo *propagator_solver;
    // Scheduler for propagator
    Scheduler *sc;
    // The number of parallel streams for propagator computation
    int n_streams;
    // Map for propagator q(r,s; code)
    std::map<std::string, double **> propagator;
    // Map for q(r,1/2+s; code)
    std::map<std::string, double **> propagator_half_steps;
    // Map for deallocation of propagator
    std::map<std::string, int> propagator_size;
    // Check if computation of propagator is finished
    #ifndef NDEBUG
    std::map<std::string, bool *> propagator_finished;
    std::map<std::string, std::map<int, bool>> propagator_half_steps_finished;
    int time_complexity;
    #endif

    // Remember one segment for each polymer chain to compute total partition function
    // (polymer id, propagator forward, propagator backward, monomer_type, n_repeated)
    std::vector<std::tuple<int, double *, double *, std::string, int>> single_partition_segment;

    // key: (polymer id, dep_v, dep_u) (assert(dep_v <= dep_u)), value: concentrations
    std::map<std::tuple<int, std::string, std::string>, double *> phi_block;

    // Solvent concentrations
    std::vector<double *> phi_solvent;

    // Calculate concentration of one block
    void calculate_phi_one_block(double *phi, double **q_1, double **q_2, const double *exp_dw, const int N_RIGHT, const int N_LEFT);
public:
    CpuComputationDiscrete(ComputationBox *cb, Molecules *molecules, PropagatorAnalyzer* propagator_analyzer);
    ~CpuComputationDiscrete();
    
    void update_laplacian_operator() override;

    void compute_propagators(
        std::map<std::string, const double*> w_block,
        std::map<std::string, const double*> q_init = {}) override;

    void compute_concentrations() override;

    void compute_statistics(
        std::map<std::string, const double*> w_block,
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