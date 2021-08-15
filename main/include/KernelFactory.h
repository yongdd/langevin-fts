/*----------------------------------------------------------
* class KernelFactory
*-----------------------------------------------------------*/

#ifndef KERNEL_FACTORY_H_
#define KERNEL_FACTORY_H_

#include "PolymerChain.h"
#include "SimulationBox.h"
#include "Pseudo.h"
#include "AndersonMixing.h"
#include "KernelFactory.h"

// Design Pattern : Abstract Factory (but not fully implmented yet)

class KernelFactory
{
private:
    std::string str_platform;
public:
    KernelFactory(std::string str_platform);
    PolymerChain* create_polymer_chain(double f, int NN, double chi_n);
    SimulationBox* create_simulation_box(
        int *nx, double *lx);
    Pseudo* create_pseudo(SimulationBox *sb, PolymerChain *pc);

    AndersonMixing* create_anderson_mixing(
        SimulationBox *sb, int n_comp,
        double max_anderson, double start_anderson_error,
        double mix_min, double mix_init);
};

#endif
