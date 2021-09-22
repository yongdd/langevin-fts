/*----------------------------------------------------------
* class MklFactory
*-----------------------------------------------------------*/

#include <iostream>
#include <array>
#include <vector>
#include <string>
#include <algorithm>

#include "MklFFT3D.h"
#include "MklFFT2D.h"
#include "MklFFT1D.h"
#include "CpuPseudoGaussian.h"
#include "CpuPseudoDiscrete.h"
#include "CpuAndersonMixing.h"
#include "MklFactory.h"

PolymerChain* MklFactory::create_polymer_chain(double f, int NN, double chi_n)
{
    return new PolymerChain(f, NN, chi_n);
}
SimulationBox* MklFactory::create_simulation_box(
    std::vector<int> nx, std::vector<double> lx)
{
    return new SimulationBox(nx, lx);
}
Pseudo* MklFactory::create_pseudo(SimulationBox *sb, PolymerChain *pc, std::string str_model)
{
    std::transform(str_model.begin(), str_model.end(), str_model.begin(),
                   [](unsigned char c)
    {
        return std::tolower(c);
    });

    if ( str_model == "gaussian" )
    {
        if (sb->get_dimension() == 3)
            return new CpuPseudoGaussian(sb, pc,
                new MklFFT3D({sb->get_nx(0),sb->get_nx(1),sb->get_nx(2)}));
        else if (sb->get_dimension() == 2)
            return new CpuPseudoGaussian(sb, pc,
                new MklFFT2D({sb->get_nx(0),sb->get_nx(1)}));
        else if (sb->get_dimension() == 1)
            return new CpuPseudoGaussian(sb, pc,
                new MklFFT1D(sb->get_nx(0)));
    }
    else if ( str_model == "discrete" )
    {
        if (sb->get_dimension() == 3)
            return new CpuPseudoDiscrete(sb, pc,
                new MklFFT3D({sb->get_nx(0),sb->get_nx(1),sb->get_nx(2)}));
        else if (sb->get_dimension() == 2)
            return new CpuPseudoDiscrete(sb, pc,
                new MklFFT2D({sb->get_nx(0),sb->get_nx(1)}));
        else if (sb->get_dimension() == 1)
            return new CpuPseudoGaussian(sb, pc,
                new MklFFT1D(sb->get_nx(0)));
    }
    return NULL;
}
AndersonMixing* MklFactory::create_anderson_mixing(
    SimulationBox *sb, int n_comp,
    double max_anderson, double start_anderson_error,
    double mix_min, double mix_init)
{
    return new CpuAndersonMixing(
               sb, n_comp, max_anderson,
               start_anderson_error, mix_min, mix_init);
}