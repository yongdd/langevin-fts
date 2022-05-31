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
        double f, int n_contour, double chi_n, 
        std::string model_name, double epsilon=1.0) = 0;
    virtual SimulationBox* create_simulation_box(
        std::vector<int> nx,
        std::vector<double> lx) = 0;
    virtual Pseudo* create_pseudo(
        SimulationBox *sb,
        PolymerChain *pc) = 0; 
    virtual AndersonMixing* create_anderson_mixing(
        int n_var, int max_hist, double start_error,
        double mix_min, double mix_init) = 0;
    virtual void display_info() = 0;
};
#endif
