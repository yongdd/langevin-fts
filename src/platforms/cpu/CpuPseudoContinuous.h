/*-------------------------------------------------------------
* This is a derived CpuPseudoContinuous class
*------------------------------------------------------------*/

#ifndef CPU_PSEUDO_CONTINUOUS_H_
#define CPU_PSEUDO_CONTINUOUS_H_

#include <string>
#include <vector>
#include <map>

#include "ComputationBox.h"
#include "PolymerChain.h"
#include "Mixture.h"
#include "Pseudo.h"
#include "FFT.h"
#include "Scheduler.h"

class CpuPseudoContinuous : public Pseudo
{
private:
    FFT *fft;

    // for stress calculation: compute_stress()
    double *fourier_basis_x;
    double *fourier_basis_y;
    double *fourier_basis_z;

    // key: (dep) + species, value: partition functions
    std::map<std::string, double *> unique_partition;
    std::map<std::string, bool *> unique_partition_finished;
    Scheduler *sc;          // scheduler for partial partition function
    const int N_STREAM = 4; // the number of job threads

    // key: (dep_v, dep_u) (assert(dep_v <= dep_u)), value: concentrations
    std::map<std::tuple<std::string, std::string, int>, double *> unique_phi;

    std::map<std::string, double*> boltz_bond;        // boltzmann factor for the single bond
    std::map<std::string, double*> boltz_bond_half;   // boltzmann factor for the half bond
    std::map<std::string, double*> exp_dw;            // boltzmann factor for the single segment
    std::map<std::string, double*> exp_dw_half;       // boltzmann factor for the half segment

    // total partition functions for each polymer
    double* single_partitions;

    void one_step(double *q_in, double *q_out, 
                  double *boltz_bond, double *boltz_bond_half,
                  double *exp_dw, double *exp_dw_half);
    void calculate_phi_one_type(double *phi, double *q_1, double *q_2, const int N);
public:
    CpuPseudoContinuous(ComputationBox *cb, Mixture *pc, FFT *ff);
    ~CpuPseudoContinuous();
    
    void update() override;
    void compute_statistics(
        std::map<std::string, double*> q_init,
        std::map<std::string, double*> w_block) override;
    double get_total_partition(int polymer) override;
    void get_species_concentration(std::string species, double *phi) override;
    void get_polymer_concentration(int polymer, double *phi) override;
    std::array<double,3> compute_stress() override;
    void get_partial_partition(double *q_out, int polymer, int v, int u, int n) override;
};
#endif
