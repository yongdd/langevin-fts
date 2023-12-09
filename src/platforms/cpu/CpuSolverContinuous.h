/*-------------------------------------------------------------
* This is a derived CpuSolverContinuous class
*------------------------------------------------------------*/

#ifndef CPU_PSEUDO_CONTINUOUS_H_
#define CPU_PSEUDO_CONTINUOUS_H_

#include <string>
#include <vector>
#include <map>

#include "ComputationBox.h"
#include "Polymer.h"
#include "Molecules.h"
#include "PropagatorsAnalyzer.h"
#include "Solver.h"
#include "CpuPseudo.h"
#include "Scheduler.h"

class CpuSolverContinuous : public Solver
{
private:
    // Pseudo-spectral PDE solver
    CpuPseudo *propagator_solver;
    // Scheduler for propagator
    Scheduler *sc;
    // The number of parallel streams for propagator computation
    const int N_SCHEDULER_STREAMS = 4;
    // key: (dep) + monomer_type, value: propagator
    std::map<std::string, double *> propagator;
    // Check if computation of propagator is finished
    #ifndef NDEBUG
    std::map<std::string, bool *> propagator_finished;
    #endif

    // Total partition functions for each polymer
    double* single_polymer_partitions;

    // Remember one segment for each polymer chain to compute total partition function
    // (polymer id, propagator forward, propagator backward, n_aggregated)
    std::vector<std::tuple<int, double *, double *, int>> single_partition_segment;

    // key: (polymer id, dep_v, dep_u) (assert(dep_v <= dep_u)), value: concentrations
    std::map<std::tuple<int, std::string, std::string>, double *> phi_block;

    // Total partition functions for each solvent
    double* single_solvent_partitions;

    // Solvent concentrations
    std::vector<double *> phi_solvent;

    // Accessible volume of polymers excluding mask region
    double accessible_volume;

    // Calculate concentration of one block
    void calculate_phi_one_block(double *phi, double *q_1, double *q_2, const int N, const int N_OFFSET, const int N_ORIGINAL);
public:
    CpuSolverContinuous(ComputationBox *cb, Molecules *molecules, PropagatorsAnalyzer* propagators_analyzer);
    ~CpuSolverContinuous();
    
    void update_bond_function() override;
    void compute_statistics(
        std::map<std::string, const double*> w_block,
        std::map<std::string, const double*> q_init = {},
        double* q_mask=nullptr) override;
    void compute_statistics_device(
        std::map<std::string, const double*> w_block,
        std::map<std::string, const double*> q_init = {},
        double* q_mask=nullptr) override
    {
        compute_statistics(w_block, q_init, q_mask);
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
