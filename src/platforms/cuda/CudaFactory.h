/*----------------------------------------------------------
* class CudaFactory
*-----------------------------------------------------------*/

#ifndef CUDA_FACTORY_H_
#define CUDA_FACTORY_H_

#include "PolymerChain.h"
#include "ComputationBox.h"
#include "Pseudo.h"
#include "AndersonMixing.h"
#include "AbstractFactory.h"

class CudaFactory : public AbstractFactory
{
public :
    CudaFactory(std::string chain_model);
    PolymerChain* create_polymer_chain(
        std::vector<int> n_segment, 
        std::vector<double> bond_length,
        double ds) override;
    ComputationBox* create_computation_box(
        std::vector<int> nx,
        std::vector<double> lx) override;
    Pseudo* create_pseudo(
        ComputationBox *cb,
        PolymerChain *pc) override;
    AndersonMixing* create_anderson_mixing(
        int n_var, int max_hist, double start_error,
        double mix_min, double mix_init) override;
    void display_info() override;
};
#endif
