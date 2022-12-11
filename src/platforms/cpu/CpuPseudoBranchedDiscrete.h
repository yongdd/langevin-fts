/*-------------------------------------------------------------
* This is a derived CpuPseudoBranchedDiscrete class
*------------------------------------------------------------*/

#ifndef CPU_PSEUDO_BRANCHED_DISCRETE_H_
#define CPU_PSEUDO_BRANCHED_DISCRETE_H_

#include <string>
#include <vector>
#include <map>

#include "ComputationBox.h"
#include "PolymerChain.h"
#include "Mixture.h"
#include "Pseudo.h"
#include "FFT.h"

class CpuPseudoBranchedDiscrete : public Pseudo
{
private:
    FFT *fft;

    // for stress calculation: dq_dl()
    double *fourier_basis_x;
    double *fourier_basis_y;
    double *fourier_basis_z;

    // key: (dep) + species, value: partition functions
    std::map<std::string, double *> reduced_partition;

    // key: (dep_v, dep_u) (assert(dep_v <= dep_u)), value: concentrations
    std::map<std::tuple<std::string, std::string, int>, double *> reduced_phi;

    // key: (dep), value: array pointer
    std::map<std::string, double*> reduced_q_junctions;
    
    std::map<std::string, double*> boltz_bond;        // boltzmann factor for the single bond
    std::map<std::string, double*> boltz_bond_half;   // boltzmann factor for the half bond
    std::map<std::string, double*> exp_dw;            // boltzmann factor for the single segment

    void one_step(double *q_in, double *q_out, double *boltz_bond, double *exp_dw);
    void half_bond_step(double *q_in, double *q_out, double *boltz_bond_half);
    void calculate_phi_one_type(double *phi, double *q_1, double *q_2, double *exp_dw, const int N);
public:
    CpuPseudoBranchedDiscrete(ComputationBox *cb, Mixture *mx, FFT *fft);
    ~CpuPseudoBranchedDiscrete();
    
    void update() override;

    std::vector<double> compute_statistics(
        std::map<std::string, double*> q_init,
        std::map<std::string, double*> w_block,
        std::vector<double *> phi) override;
    std::vector<std::array<double,3>> dq_dl() override;
    void get_partition(double *q_out, int polymer, int v, int u, int n) override;
};
#endif
