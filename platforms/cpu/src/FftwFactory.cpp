/*----------------------------------------------------------
* class FftwFactory
*-----------------------------------------------------------*/

#include <iostream>
#include <array>
#include <vector>
#include <string>

#include "FftwFFT3D.h"
#include "FftwFFT2D.h"
#include "FftwFFT1D.h"
#include "CpuPseudoGaussian.h"
#include "CpuPseudoDiscrete.h"
#include "CpuAndersonMixing.h"
#include "FftwFactory.h"

PolymerChain* FftwFactory::create_polymer_chain(
    double f, int NN, double chi_n, std::string model_name, double epsilon)
{
    return new PolymerChain(f, NN, chi_n, model_name, epsilon);
}
SimulationBox* FftwFactory::create_simulation_box(
    std::vector<int> nx, std::vector<double> lx)
{
    return new SimulationBox(nx, lx);
}
Pseudo* FftwFactory::create_pseudo(SimulationBox *sb, PolymerChain *pc)
{
    std::string model_name = pc->get_model_name();
    if ( model_name == "gaussian" )
    {
        if (sb->get_dim() == 3)
            return new CpuPseudoGaussian(sb, pc,
                new FftwFFT3D({sb->get_nx(0),sb->get_nx(1),sb->get_nx(2)}));
        else if (sb->get_dim() == 2)
            return new CpuPseudoGaussian(sb, pc,
                new FftwFFT2D({sb->get_nx(1),sb->get_nx(2)}));
        else if (sb->get_dim() == 1)
            return new CpuPseudoGaussian(sb, pc,
                new FftwFFT1D(sb->get_nx(2)));
    }
    else if ( model_name == "discrete" )
    {
        if (sb->get_dim() == 3)
            return new CpuPseudoDiscrete(sb, pc,
                new FftwFFT3D({sb->get_nx(0),sb->get_nx(1),sb->get_nx(2)}));
        else if (sb->get_dim() == 2)
            return new CpuPseudoDiscrete(sb, pc,
                new FftwFFT2D({sb->get_nx(1),sb->get_nx(2)}));
        else if (sb->get_dim() == 1)
            return new CpuPseudoDiscrete(sb, pc,
                new FftwFFT1D(sb->get_nx(2)));
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
void FftwFactory::display_info()
{
    std::cout << "cpu-fftw" << std::endl;
}
