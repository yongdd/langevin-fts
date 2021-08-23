/*----------------------------------------------------------
* class MklFactory
*-----------------------------------------------------------*/

#ifndef MKL_FACTORY_H_
#define MKL_FACTORY_H_

#include "PolymerChain.h"
#include "SimulationBox.h"
#include "Pseudo.h"
#include "AndersonMixing.h"
#include "AbstractFactory.h"

class MklFactory : public AbstractFactory
{
public :
    PolymerChain* create_polymer_chain(
        double f, int NN, double chi_n) override;
    SimulationBox* create_simulation_box(
        std::array<int,3> nx,
        std::array<double,3>  lx) override;
    Pseudo* create_pseudo(
        SimulationBox *sb,
        PolymerChain *pc,
        std::string str_model) override;
    AndersonMixing* create_anderson_mixing(
        SimulationBox *sb, int n_comp,
        double max_anderson, double start_anderson_error,
        double mix_min, double mix_init) override;
};
#endif
