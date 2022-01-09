/*----------------------------------------------------------
* class CudaFactory
*-----------------------------------------------------------*/

#include <array>
#include <vector>
#include <string>

#include "CudaSimulationBox.h"
#include "CudaPseudoGaussian.h"
#include "CudaPseudoDiscrete.h"
#include "CudaAndersonMixing.h"
#include "CudaFactory.h"

PolymerChain* CudaFactory::create_polymer_chain(double f, int NN, double chi_n, std::string model_name)
{
    return new PolymerChain(f, NN, chi_n, model_name);
}
SimulationBox* CudaFactory::create_simulation_box(
    std::vector<int> nx, std::vector<double>  lx)
{
    return new CudaSimulationBox(nx, lx);
}
Pseudo* CudaFactory::create_pseudo(SimulationBox *sb, PolymerChain *pc)
{
    std::string model_name = pc->get_model_name();
    if ( model_name == "gaussian" )
        return new CudaPseudoGaussian(sb, pc);
    else if ( model_name == "discrete" )
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
