/*----------------------------------------------------------
* class MklFactory
*-----------------------------------------------------------*/

#ifndef MKL_FACTORY_H_
#define MKL_FACTORY_H_

#include "ComputationBox.h"
#include "BranchedPolymerChain.h"
#include "PseudoBranched.h"
#include "AndersonMixing.h"
#include "AbstractFactory.h"

class MklFactory : public AbstractFactory
{
public :
    MklFactory(std::string chain_model);
    BranchedPolymerChain* create_polymer_chain(
        double ds,
        std::map<std::string, double> dict_segment_lengths,
        std::vector<std::string> block_species, 
        std::vector<double> contour_lengths,
        std::vector<int> v, std::vector<int> u,
        std::map<int, int> v_to_grafting_index) override;
    ComputationBox* create_computation_box(
        std::vector<int> nx,
        std::vector<double> lx) override;
    PseudoBranched* create_pseudo(
        ComputationBox *cb,
        BranchedPolymerChain *pc) override;
    AndersonMixing* create_anderson_mixing(
        int n_var, int max_hist, double start_error,
        double mix_min, double mix_init) override;
    void display_info() override;
};
#endif
