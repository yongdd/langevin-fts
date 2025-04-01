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
#include "PropagatorComputationOptimizer.h"
#include "PropagatorComputation.h"
#include "CpuSolver.h"
#include "Scheduler.h"

template <typename T>
class CpuComputationContinuous : public PropagatorComputation<T>
{
private:
    // Pseudo-spectral PDE solver
    CpuSolver<T> *propagator_solver;
    // Scheduler for propagator
    Scheduler *sc;
    // The number of parallel streams for propagator computation
    int n_streams;
    // key: (dep) + monomer_type, value: propagator
    std::map<std::string, T **> propagator; 
    // Map for deallocation of propagator
    std::map<std::string, int> propagator_size;
    // Check if computation of propagator is finished
    #ifndef NDEBUG
    std::map<std::string, bool *> propagator_finished;
    #endif

    // Remember one segment for each polymer chain to compute total partition function
    // (polymer id, propagator forward, propagator backward, n_repeated)
    std::vector<std::tuple<int, T *, T *, int>> single_partition_segment;

    // key: (polymer id, key_left, key_right) (assert(key_left <= key_right)), value: concentrations
    std::map<std::tuple<int, std::string, std::string>, T *> phi_block;

    // Solvent concentrations
    std::vector<T *> phi_solvent;

    // Calculate concentration of one block
    void calculate_phi_one_block(T *phi, T **q_1, T **q_2, const int N_RIGHT, const int N_LEFT);
public:
    CpuComputationContinuous(ComputationBox<T>* cb, Molecules *molecules, PropagatorComputationOptimizer* propagator_computation_optimizer, std::string method);
    ~CpuComputationContinuous();
    
    void update_laplacian_operator() override;

    void compute_propagators(
        std::map<std::string, const T*> w_block,
        std::map<std::string, const T*> q_init = {}) override;

    void compute_concentrations() override;

    void compute_statistics(
        std::map<std::string, const T*> w_block,
        std::map<std::string, const T*> q_init = {}) override;

    void compute_stress() override;
    T get_total_partition(int polymer) override;
    void get_chain_propagator(T *q_out, int polymer, int v, int u, int n) override;

    // Canonical ensemble
    void get_total_concentration(std::string monomer_type, T *phi) override;
    void get_total_concentration(int polymer, std::string monomer_type, T *phi) override;
    void get_block_concentration(int polymer, T *phi) override;

    T get_solvent_partition(int s) override;
    void get_solvent_concentration(int s, T *phi) override;

    // Grand canonical ensemble
    void get_total_concentration_gce(double fugacity, int polymer, std::string monomer_type, T *phi) override;

    // For tests
    bool check_total_partition() override;
};
#endif
