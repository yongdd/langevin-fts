/*-------------------------------------------------------------
* This is a derived CpuPseudoContinuous class
*------------------------------------------------------------*/

#ifndef CPU_PSEUDO_CONTINUOUS_H_
#define CPU_PSEUDO_CONTINUOUS_H_

#include "ComputationBox.h"
#include "BranchedPolymerChain.h"
#include "PseudoBranched.h"
#include "FFT.h"

class CpuPseudoContinuous : public PseudoBranched
{
private:
    FFT *fft;
    double *q_1, *q_2;

    std::map<std::string, double*> boltz_bond;        // boltzmann factor for the single bond
    std::map<std::string, double*> boltz_bond_half;   // boltzmann factor for the half bond
    std::map<std::string, double*> exp_dw;            // boltzmann factor for the single segment
    std::map<std::string, double*> exp_dw_half;       // boltzmann factor for the half segment

    std::vector<int> get_block_start();
    void one_step(double *q_in, double *q_out, 
                  double *boltz_bond, double *boltz_bond_half,
                  double *exp_dw, double *exp_dw_half);
    void calculate_phi_one_type(double *phi, const int N_START, const int N_END);
public:
    CpuPseudoContinuous(ComputationBox *cb, BranchedPolymerChain *pc, FFT *ff);
    ~CpuPseudoContinuous();
    
    void update() override;
    void compute_statistics(
        std::map<std::string, double*> q_init,
        std::map<std::string, double*> w_block,
        double *phi, double &single_partition) override;
    std::array<double,3> dq_dl() override;
    void get_partition(double *q_out, int v, int u, int n) override;
};
#endif
