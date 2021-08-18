/*----------------------------------------------------------
* class KernelFactory
*-----------------------------------------------------------*/

#ifndef KERNEL_FACTORY_H_
#define KERNEL_FACTORY_H_

#include <array>
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
    KernelFactory();
    KernelFactory(std::string str_platform);
    PolymerChain* create_polymer_chain(double f, int NN, double chi_n);
    SimulationBox* create_simulation_box(std::array<int,3> nx, std::array<double,3> lx);
    SimulationBox* create_simulation_box(int *nx, double *lx)
    {
        return create_simulation_box({nx[0],nx[1],nx[2]}, {lx[0],lx[1],lx[2]});
    }
    Pseudo* create_pseudo(SimulationBox *sb, PolymerChain *pc);

    AndersonMixing* create_anderson_mixing(
        SimulationBox *sb, int n_comp,
        double max_anderson, double start_anderson_error,
        double mix_min, double mix_init);
};

#endif
