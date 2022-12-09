/*-------------------------------------------------------------
* This is a derived CpuPseudoBranchedContinuous class
*------------------------------------------------------------*/

#ifndef CPU_PSEUDO_BRANCHED_CONTINUOUS_H_
#define CPU_PSEUDO_BRANCHED_CONTINUOUS_H_

#include <string>
#include <vector>
#include <map>

#include "ComputationBox.h"
#include "PolymerChain.h"
#include "Pseudo.h"
#include "FFT.h"

class CpuPseudoBranchedContinuous : public Pseudo
{
private:
    FFT *fft;

    // key: (dep) + species, value: branched_pseudo_edge
    std::map<std::string, branched_pseudo_edge, std::greater<std::string>> reduced_edges;

    // key: (dep_v, dep_u) (assert(dep_v <= dep_u)), value: branched_pseudo_block
    std::map<std::pair<std::string, std::string>, branched_pseudo_block> reduced_blocks; 

    std::map<std::string, double*> boltz_bond;        // boltzmann factor for the single bond
    std::map<std::string, double*> boltz_bond_half;   // boltzmann factor for the half bond
    std::map<std::string, double*> exp_dw;            // boltzmann factor for the single segment
    std::map<std::string, double*> exp_dw_half;       // boltzmann factor for the half segment

    void one_step(double *q_in, double *q_out, 
                  double *boltz_bond, double *boltz_bond_half,
                  double *exp_dw, double *exp_dw_half);
    void calculate_phi_one_type(double *phi, double *q_1, double *q_2, const int N);
    void init_simpson_rule_coeff(double *coeff, const int N);
public:
    CpuPseudoBranchedContinuous(ComputationBox *cb, PolymerChain *pc, FFT *ff);
    ~CpuPseudoBranchedContinuous();
    
    void update() override;
    
    void compute_statistics(
        std::map<std::string, double*> q_init,
        std::map<std::string, double*> w_block,
        double *phi, double &single_partition) override;
    std::array<double,3> dq_dl() override;
    void get_partition(double *q_out, int v, int u, int n) override;
};
#endif
