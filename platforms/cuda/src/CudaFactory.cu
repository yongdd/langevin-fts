/*----------------------------------------------------------
* class CudaFactory
*-----------------------------------------------------------*/

#include <array>
#include <vector>
#include <string>
#include <algorithm>

#include "CudaSimulationBox.h"
#include "CudaPseudoGaussian.h"
#include "CudaPseudoDiscrete.h"
#include "CudaAndersonMixing.h"
#include "CudaFactory.h"

PolymerChain* CudaFactory::create_polymer_chain(double f, int NN, double chi_n)
{
    return new PolymerChain(f, NN, chi_n);
}
SimulationBox* CudaFactory::create_simulation_box(
    std::array<int,3> nx, std::array<double,3>  lx)
{
    return new CudaSimulationBox(nx, lx);
}
Pseudo* CudaFactory::create_pseudo(SimulationBox *sb, PolymerChain *pc, std::string str_model)
{
    std::transform(str_model.begin(), str_model.end(), str_model.begin(),
    [](unsigned char c){ return std::tolower(c); });
    
    if ( str_model == "gaussian" )
        return new CudaPseudoGaussian(sb, pc);
    else if ( str_model == "discrete" )
        return new CudaPseudoDiscrete(sb, pc);
    return NULL;
}
AndersonMixing* CudaFactory::create_anderson_mixing(
    SimulationBox *sb, int n_comp,
    double max_anderson, double start_anderson_error,
    double mix_min, double mix_init)
{
    return new CudaAndersonMixing(
               sb, n_comp, max_anderson,
               start_anderson_error, mix_min, mix_init);
}
