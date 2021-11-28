/*----------------------------------------------------------
* class FftwFactory
*-----------------------------------------------------------*/

#include <array>
#include <vector>
#include <string>
#include <algorithm>

#include "FftwFFT3D.h"
#include "FftwFFT2D.h"
#include "FftwFFT1D.h"
#include "CpuPseudoGaussian.h"
#include "CpuPseudoDiscrete.h"
#include "CpuAndersonMixing.h"
#include "FftwFactory.h"

PolymerChain* FftwFactory::create_polymer_chain(double f, int n_contour, double chi_n)
{
    return new PolymerChain(f, n_contour, chi_n);
}
SimulationBox* FftwFactory::create_simulation_box(
    std::vector<int> nx, std::vector<double> lx)
{
    return new SimulationBox(nx, lx);
}
Pseudo* FftwFactory::create_pseudo(SimulationBox *sb, PolymerChain *pc, std::string str_model)
{
    std::transform(str_model.begin(), str_model.end(), str_model.begin(),
                   [](unsigned char c)
    {
        return std::tolower(c);
    });

    if ( str_model == "gaussian" )
    {
        if (sb->get_dim() == 3)
            return new CpuPseudoGaussian(sb, pc,
                new FftwFFT3D({sb->get_nx(0),sb->get_nx(1),sb->get_nx(2)}));
        else if (sb->get_dim() == 2)
            return new CpuPseudoGaussian(sb, pc,
                new FftwFFT2D({sb->get_nx(0),sb->get_nx(1)}));
        else if (sb->get_dim() == 1)
            return new CpuPseudoGaussian(sb, pc,
                new FftwFFT1D(sb->get_nx(0)));
    }
    else if ( str_model == "discrete" )
    {
        if (sb->get_dim() == 3)
            return new CpuPseudoDiscrete(sb, pc,
                new FftwFFT3D({sb->get_nx(0),sb->get_nx(1),sb->get_nx(2)}));
        else if (sb->get_dim() == 2)
            return new CpuPseudoDiscrete(sb, pc,
                new FftwFFT2D({sb->get_nx(0),sb->get_nx(1)}));
        else if (sb->get_dim() == 1)
            return new CpuPseudoGaussian(sb, pc,
                new FftwFFT1D(sb->get_nx(0)));
    }
    return NULL;
}
AndersonMixing* FftwFactory::create_anderson_mixing(
    SimulationBox *sb, int n_comp,
    double max_anderson, double start_anderson_error,
    double mix_min, double mix_init)
{
    return new CpuAndersonMixing(
               sb, n_comp, max_anderson,
               start_anderson_error, mix_min, mix_init);
}
