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

class CudaFactory : public AbstractFactory
{
public :
    CudaFactory(std::string chain_model);
    PolymerChain* create_polymer_chain(
        double ds,
        std::map<std::string, double> dict_segment_lengths,
        std::vector<std::string> block_species, 
        std::vector<double> contour_lengths,
        std::vector<int> v, std::vector<int> u,
        std::map<int, int> v_to_grafting_index) override;
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
