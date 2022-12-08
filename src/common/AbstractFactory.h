/*----------------------------------------------------------
* class AbstractFactory
*-----------------------------------------------------------*/

#ifndef ABSTRACT_FACTORY_H_
#define ABSTRACT_FACTORY_H_

#include <string>
#include <array>

#include "ComputationBox.h"
#include "BranchedPolymerChain.h"
#include "PseudoBranched.h"
#include "AndersonMixing.h" 

// Design Pattern : Abstract Factory

class AbstractFactory
{
protected:
    std::string chain_model;
public :
    virtual ~AbstractFactory() {};
    virtual BranchedPolymerChain* create_polymer_chain(
        double ds,
        std::map<std::string, double> dict_segment_lengths,
        std::vector<std::string> block_species, 
        std::vector<double> contour_lengths,
        std::vector<int> v, std::vector<int> u,
        std::map<int, int> v_to_grafting_index) = 0;
    virtual ComputationBox* create_computation_box(
        std::vector<int> nx,
        std::vector<double> lx) = 0;
    std::string get_model_name() {return chain_model;};
    virtual PseudoBranched* create_pseudo(
        ComputationBox *cb,
        BranchedPolymerChain *pc) = 0; 
    virtual AndersonMixing* create_anderson_mixing(
        int n_var, int max_hist, double start_error,
        double mix_min, double mix_init) = 0;
    virtual void display_info() = 0;
};
#endif
