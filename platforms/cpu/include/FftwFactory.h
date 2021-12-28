/*----------------------------------------------------------
* class FftwFactory
*-----------------------------------------------------------*/

#ifndef FFTW_FACTORY_H_
#define FFTW_FACTORY_H_

#include "PolymerChain.h"
#include "SimulationBox.h"
#include "Pseudo.h"
#include "AndersonMixing.h"
#include "AbstractFactory.h"

class FftwFactory : public AbstractFactory
{
public :
    PolymerChain* create_polymer_chain(
        double f, int n_contour, double chi_n) override;
    SimulationBox* create_simulation_box(
        std::vector<int> nx,
        std::vector<double> lx) override;
    Pseudo* create_pseudo(
        SimulationBox *sb,
        PolymerChain *pc,
        std::string str_model) override;
    AndersonMixing* create_anderson_mixing(
        SimulationBox *sb, int n_comp,
        double max_hist, double start_error,
        double mix_min, double mix_init) override;
};
#endif
