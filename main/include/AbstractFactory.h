/*----------------------------------------------------------
* class AbstractFactory
*-----------------------------------------------------------*/

#ifndef ABSTRACT_FACTORY_H_
#define ABSTRACT_FACTORY_H_

#include <string>
#include <array>
#include "PolymerChain.h"
#include "SimulationBox.h"
#include "Pseudo.h"
#include "AndersonMixing.h"

// Design Pattern : Abstract Factory

class AbstractFactory
{
public :
    virtual ~AbstractFactory() {};
    virtual PolymerChain* create_polymer_chain(
        double f, int NN, double chi_n) = 0; 
    virtual SimulationBox* create_simulation_box(
        std::vector<int> nx,
        std::vector<double> lx) = 0;
    virtual Pseudo* create_pseudo(
        SimulationBox *sb,
        PolymerChain *pc,
        std::string str_model) = 0; 
    virtual AndersonMixing* create_anderson_mixing(
        SimulationBox *sb, int n_comp,
        double max_anderson, double start_anderson_error,
        double mix_min, double mix_init) = 0; 
};
#endif