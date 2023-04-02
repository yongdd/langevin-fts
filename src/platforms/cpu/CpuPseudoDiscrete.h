/*-------------------------------------------------------------
* This is a derived CpuPseudoDiscrete class
*------------------------------------------------------------*/

#ifndef CPU_PSEUDO_DISCRETE_H_
#define CPU_PSEUDO_DISCRETE_H_

#include <string>
#include <vector>
#include <map>

#include "ComputationBox.h"
#include "PolymerChain.h"
#include "Mixture.h"
#include "Pseudo.h"
#include "FFT.h"
#include "Scheduler.h"

class CpuPseudoDiscrete : public Pseudo
{
private:
    FFT *fft;

    // for stress calculation: compute_stress()
    double *fourier_basis_x;
    double *fourier_basis_y;
    double *fourier_basis_z;

    // scheduler for propagator
    Scheduler *sc;
    // the number of parallel streams for propagator computation
    const int N_SCHEDULER_STREAMS = 4;
    // key: (dep), value: array pointer
    std::map<std::string, double*> q_junction_cache;
    // key: (dep) + monomer_type, value: propagator
    std::map<std::string, double *> propagator;
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

    // arrays for pseudo-spectral
    std::map<std::string, double*> boltz_bond;        // boltzmann factor for the single bond
    std::map<std::string, double*> boltz_bond_half;   // boltzmann factor for the half bond
    std::map<std::string, double*> exp_dw;            // boltzmann factor for the single segment

    // advance propagator by one segment step
    void one_step(double *q_in, double *q_out, double *boltz_bond, double *exp_dw);

    // advance propagator by half bond step
    void half_bond_step(double *q_in, double *q_out, double *boltz_bond_half);

    // calculate concentration of one block
    void calculate_phi_one_block(double *phi, double *q_1, double *q_2, double *exp_dw, const int N, const int N_OFFSET, const int N_ORIGINAL);
public:
    CpuPseudoDiscrete(ComputationBox *cb, Mixture *mx, FFT *fft);
    ~CpuPseudoDiscrete();
    
    void update_bond_function() override;
    void compute_statistics(
        std::map<std::string, double*> w_block,
        std::map<std::string, double*> q_init) override;
    double get_total_partition(int polymer) override;
    void get_monomer_concentration(std::string monomer_type, double *phi) override;
    void get_polymer_concentration(int polymer, double *phi) override;
    std::vector<double> compute_stress() override;
    void get_chain_propagator(double *q_out, int polymer, int v, int u, int n) override;
};
#endif
