/*-------------------------------------------------------------
* This is a derived CpuPseudoContinuous class
*------------------------------------------------------------*/

#ifndef CPU_PSEUDO_CONTINUOUS_H_
#define CPU_PSEUDO_CONTINUOUS_H_

#include <string>
#include <vector>
#include <map>

#include "ComputationBox.h"
#include "Polymer.h"
#include "Molecules.h"
#include "Pseudo.h"
#include "FFT.h"
#include "Scheduler.h"

class CpuPseudoContinuous : public Pseudo
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
    // key: (dep) + monomer_type, value: propagator
    std::map<std::string, double *> propagator;
    // Check if computation of propagator is finished
    #ifndef NDEBUG
    std::map<std::string, bool *> propagator_finished;
    #endif

    // Total partition functions for each polymer
    double* single_partitions;
    // Remember one segment for each polymer chain to compute total partition function
    // (polymer id, propagator forward, propagator backward, n_superposed)
    std::vector<std::tuple<int, double *, double *, int>> single_partition_segment;

    // key: (polymer id, dep_v, dep_u) (assert(dep_v <= dep_u)), value: concentrations
    std::map<std::tuple<int, std::string, std::string>, double *> block_phi;

    // Arrays for pseudo-spectral
    std::map<std::string, double*> boltz_bond;        // Boltzmann factor for the single bond
    std::map<std::string, double*> boltz_bond_half;   // Boltzmann factor for the half bond
    std::map<std::string, double*> exp_dw;            // Boltzmann factor for the single segment
    std::map<std::string, double*> exp_dw_half;       // Boltzmann factor for the half segment

    // Advance propagator by one contour step
    void advance_propagator(double *q_in, double *q_out, 
                  double *boltz_bond, double *boltz_bond_half,
                  double *exp_dw, double *exp_dw_half);

    // Calculate concentration of one block
    void calculate_phi_one_block(double *phi, double *q_1, double *q_2, const int N, const int N_OFFSET, const int N_ORIGINAL);
public:
    CpuPseudoContinuous(ComputationBox *cb, Molecules *pc, FFT *ff);
    ~CpuPseudoContinuous();
    
    void update_bond_function() override;
    void compute_statistics(
        std::map<std::string, const double*> w_block,
        std::map<std::string, const double*> q_init) override;
    void compute_statistics_device(
        std::map<std::string, const double*> w_block,
        std::map<std::string, const double*> q_init) override
    {
        compute_statistics(w_block, q_init);
    };
    double get_total_partition(int polymer) override;
    void get_total_concentration(std::string monomer_type, double *phi) override;
    void get_total_concentration(int polymer, std::string monomer_type, double *phi) override;
    void get_block_concentration(int polymer, double *phi) override;
    std::vector<double> compute_stress() override;
    void get_chain_propagator(double *q_out, int polymer, int v, int u, int n) override;
};
#endif
