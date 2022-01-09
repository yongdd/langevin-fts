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
        double f, int n_contour, double chi_n, std::string model_name) override;
    SimulationBox* create_simulation_box(
        std::vector<int> nx,
        std::vector<double> lx) override;
    Pseudo* create_pseudo(
        SimulationBox *sb,
        PolymerChain *pc) override;
    AndersonMixing* create_anderson_mixing(
        SimulationBox *sb, int n_comp,
        double max_hist, double start_error,
        double mix_min, double mix_init) override;
};
#endif
