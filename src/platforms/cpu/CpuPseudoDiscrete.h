/*-------------------------------------------------------------
* This is a derived CpuPseudoDiscrete class
*------------------------------------------------------------*/

#ifndef CPU_PSEUDO_DISCRETE_H_
#define CPU_PSEUDO_DISCRETE_H_

#include "ComputationBox.h"
#include "PolymerChain.h"
#include "Pseudo.h"
#include "FFT.h"

class CpuPseudoDiscrete : public Pseudo
{
private:
    FFT *fft;
    double *q_1, *q_2;
    std::map<std::string, double*> exp_dw;            // boltzmann factor for the single segment

    double **boltz_bond;
    double **boltz_bond_middle;

    std::vector<int> get_block_start();
    void one_step(double *q_in, double *q_out, double *boltz_bond, double *exp_dw);
public:
    CpuPseudoDiscrete(ComputationBox *cb, PolymerChain *pc, FFT *ff);
    ~CpuPseudoDiscrete();

    void update() override;
    void compute_statistics(
        std::map<std::string, double*> q_init,
        std::map<std::string, double*> w_block,
        double *phi, double &single_partition) override;
    std::array<double,3> dq_dl() override;
    void get_partition(double *q_out, int v, int u, int n) override;
};
#endif
