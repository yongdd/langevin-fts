/*----------------------------------------------------------
* class CudaFactory
*-----------------------------------------------------------*/

#ifndef CUDA_FACTORY_H_
#define CUDA_FACTORY_H_

#include "ComputationBox.h"
#include "PolymerChain.h"
#include "Pseudo.h"
#include "AndersonMixing.h"
#include "AbstractFactory.h"
#include "Array.h"

class CudaFactory : public AbstractFactory
{
public :
    CudaFactory(std::string chain_model, bool reduce_memory_usage);

    std::shared_ptr<Array> create_array(
        unsigned int size) override;

    std::shared_ptr<Array> create_array(
        double *data,
        unsigned int size) override;

    ComputationBox* create_computation_box(
        std::vector<int> nx,
        std::vector<double> lx) override;

    Mixture* create_mixture(
        double ds, std::map<std::string, double> bond_lengths, bool use_superposition) override;

    Pseudo* create_pseudo(
        ComputationBox *cb, Mixture *mx) override;

    AndersonMixing* create_anderson_mixing(
        int n_var, int max_hist, double start_error,
        double mix_min, double mix_init) override;

    void display_info() override;
};
#endif
