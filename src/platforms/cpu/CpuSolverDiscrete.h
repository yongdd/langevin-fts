/*-------------------------------------------------------------
* This is a derived CpuSolverDiscrete class
*------------------------------------------------------------*/

#ifndef CPU_PSEUDO_DISCRETE_H_
#define CPU_PSEUDO_DISCRETE_H_

#include <string>
#include <vector>
#include <map>

#include "ComputationBox.h"
#include "Polymer.h"
#include "Molecules.h"
#include "PropagatorsAnalyzer.h"
#include "Solver.h"
#include "FFT.h"
#include "Scheduler.h"

class CpuSolverDiscrete : public Solver
{
private:
    FFT *fft;

    // For stress calculation: compute_stress()
    double *fourier_basis_x;
    double *fourier_basis_y;
    double *fourier_basis_z;

    // Scheduler for propagator
    Scheduler *sc;
    // The number of parallel streams for propagator computation
    const int N_SCHEDULER_STREAMS = 4;
    // key: (dep), value: array pointer
    std::map<std::string, double*> propagator_junction;
    // key: (dep) + monomer_type, value: propagator
    std::map<std::string, double *> propagator;
    // Check if computation of propagator is finished
    #ifndef NDEBUG
    std::map<std::string, bool *> propagator_finished;
    #endif

    // Total partition functions for each polymer
    double* single_polymer_partitions;
    // Remember one segment for each polymer chain to compute total partition function
    // (polymer id, propagator forward, propagator backward, monomer_type, n_aggregated)
    std::vector<std::tuple<int, double *, double *, std::string, int>> single_partition_segment;

    // key: (polymer id, dep_v, dep_u) (assert(dep_v <= dep_u)), value: concentrations
    std::map<std::tuple<int, std::string, std::string>, double *> phi_block;

    // Total partition functions for each solvent
    double* single_solvent_partitions;

    // Solvent concentrations
    std::vector<double *> phi_solvent;

    // Accessible volume of polymers excluding mask region
    double accessible_volume;

    // Arrays for pseudo-spectral
    std::map<std::string, double*> boltz_bond;        // Boltzmann factor for the single bond
    std::map<std::string, double*> boltz_bond_half;   // Boltzmann factor for the half bond
    std::map<std::string, double*> exp_dw;            // Boltzmann factor for the single segment

    // Advance propagator by one segment step
    void advance_propagator(double *q_in, double *q_out, double *boltz_bond, double *exp_dw, double* q_mask);

    // Advance propagator by half bond step
    void advance_propagator_half_bond_step(double *q_in, double *q_out, double *boltz_bond_half);

    // Calculate concentration of one block
    void calculate_phi_one_block(double *phi, double *q_1, double *q_2, double *exp_dw, const int N, const int N_OFFSET, const int N_ORIGINAL);
public:
    CpuSolverDiscrete(ComputationBox *cb, Molecules *molecules, PropagatorsAnalyzer* propagators_analyzer, FFT *fft);
    ~CpuSolverDiscrete();
    
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