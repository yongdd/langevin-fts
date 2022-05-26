/*----------------------------------------------------------
* class MklFactory
*-----------------------------------------------------------*/

#include <iostream>
#include <array>
#include <vector>
#include <string>

#include "MklFFT3D.h"
#include "MklFFT2D.h"
#include "MklFFT1D.h"
#include "CpuPseudoGaussian.h"
#include "CpuPseudoDiscrete.h"
#include "CpuAndersonMixing.h"
#include "MklFactory.h"

PolymerChain* MklFactory::create_polymer_chain(
    double f, int NN, double chi_n, std::string model_name, double epsilon)
{
    return new PolymerChain(f, NN, chi_n, model_name, epsilon);
}
SimulationBox* MklFactory::create_simulation_box(
    std::vector<int> nx, std::vector<double> lx)
{
    return new SimulationBox(nx, lx);
}
Pseudo* MklFactory::create_pseudo(SimulationBox *sb, PolymerChain *pc)
{
    std::string model_name = pc->get_model_name();
    if ( model_name == "gaussian" )
    {
        if (sb->get_dim() == 3)
            return new CpuPseudoGaussian(sb, pc,
                new MklFFT3D({sb->get_nx(0),sb->get_nx(1),sb->get_nx(2)}));
        else if (sb->get_dim() == 2)
            return new CpuPseudoGaussian(sb, pc,
                new MklFFT2D({sb->get_nx(1),sb->get_nx(2)}));
        else if (sb->get_dim() == 1)
            return new CpuPseudoGaussian(sb, pc,
                new MklFFT1D(sb->get_nx(2)));
    }
    else if ( model_name == "discrete" )
    {
        if (sb->get_dim() == 3)
            return new CpuPseudoDiscrete(sb, pc,
                new MklFFT3D({sb->get_nx(0),sb->get_nx(1),sb->get_nx(2)}));
        else if (sb->get_dim() == 2)
            return new CpuPseudoDiscrete(sb, pc,
                new MklFFT2D({sb->get_nx(1),sb->get_nx(2)}));
        else if (sb->get_dim() == 1)
            return new CpuPseudoDiscrete(sb, pc,
                new MklFFT1D(sb->get_nx(2)));
    }
    return NULL;
}
AndersonMixing* MklFactory::create_anderson_mixing(
    int n_var, int max_hist, double start_error,
    double mix_min, double mix_init)
{
    return new CpuAndersonMixing(
        n_var, max_hist, start_error, mix_min, mix_init);
}
void MklFactory::display_info()
{
    std::cout << "cpu-mkl" << std::endl;
}
