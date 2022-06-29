/*----------------------------------------------------------
* class MklFactory
*-----------------------------------------------------------*/

#ifndef POCKET_FACTORY_H_
#define POCKET_FACTORY_H_

#include "PolymerChain.h"
#include "SimulationBox.h"
#include "Pseudo.h"
#include "AndersonMixing.h"
#include "AbstractFactory.h"

class PocketFFTFactory : public AbstractFactory
{
public :
    PolymerChain* create_polymer_chain(
        double f, int n_segment, double chi_n,
        std::string model_name, double epsilon=1.0) override;
    SimulationBox* create_simulation_box(
        std::vector<int> nx,
        std::vector<double> lx) override;
    Pseudo* create_pseudo(
        SimulationBox *sb,
        PolymerChain *pc) override;
    AndersonMixing* create_anderson_mixing(
        int n_var, int max_hist, double start_error,
        double mix_min, double mix_init) override;
    void display_info() override;
};
#endif
