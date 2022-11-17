/*----------------------------------------------------------
* class AbstractFactory
*-----------------------------------------------------------*/

#ifndef ABSTRACT_FACTORY_H_
#define ABSTRACT_FACTORY_H_

#include <string>
#include <array>

#include "PolymerChain.h"
#include "ComputationBox.h"
#include "Pseudo.h"
#include "AndersonMixing.h" 

// Design Pattern : Abstract Factory

class AbstractFactory
{
protected:
    std::string chain_model;
public :
    virtual ~AbstractFactory() {};
    virtual PolymerChain* create_polymer_chain(
        std::vector<std::string> types,
        std::vector<double> block_lengths,
        std::map<std::string, double> dict_segment_lengths,
        double ds) = 0;
    virtual ComputationBox* create_computation_box(
        std::vector<int> nx,
        std::vector<double> lx) = 0;
    std::string get_model_name() {return chain_model;};
    virtual Pseudo* create_pseudo(
        ComputationBox *cb,
        PolymerChain *pc) = 0; 
    virtual AndersonMixing* create_anderson_mixing(
        int n_var, int max_hist, double start_error,
        double mix_min, double mix_init) = 0;
    virtual void display_info() = 0;
};
#endif
