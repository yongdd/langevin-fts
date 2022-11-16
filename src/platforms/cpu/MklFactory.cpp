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
#include "CpuPseudoContinuous.h"
#include "CpuPseudoDiscrete.h"
#include "CpuAndersonMixing.h"
#include "MklFactory.h"

MklFactory::MklFactory(std::string chain_model){
    this->chain_model = chain_model;
}
PolymerChain* MklFactory::create_polymer_chain(
    std::vector<int> n_segment,
    std::vector<double> bond_length,
    double ds)
{
    return new PolymerChain(n_segment,bond_length,ds,chain_model);
}
ComputationBox* MklFactory::create_computation_box(
    std::vector<int> nx, std::vector<double> lx)
{
    return new ComputationBox(nx, lx);
}
Pseudo* MklFactory::create_pseudo(ComputationBox *cb, PolymerChain *pc)
{
    std::string chain_model = pc->get_model_name();
    if ( chain_model == "continuous" )
    {
        if (cb->get_dim() == 3)
            return new CpuPseudoContinuous(cb, pc,
                new MklFFT3D({cb->get_nx(0),cb->get_nx(1),cb->get_nx(2)}));
        else if (cb->get_dim() == 2)
            return new CpuPseudoContinuous(cb, pc,
                new MklFFT2D({cb->get_nx(1),cb->get_nx(2)}));
        else if (cb->get_dim() == 1)
            return new CpuPseudoContinuous(cb, pc,
                new MklFFT1D(cb->get_nx(2)));
    }
    else if ( chain_model == "discrete" )
    {
        if (cb->get_dim() == 3)
            return new CpuPseudoDiscrete(cb, pc,
                new MklFFT3D({cb->get_nx(0),cb->get_nx(1),cb->get_nx(2)}));
        else if (cb->get_dim() == 2)
            return new CpuPseudoDiscrete(cb, pc,
                new MklFFT2D({cb->get_nx(1),cb->get_nx(2)}));
        else if (cb->get_dim() == 1)
            return new CpuPseudoDiscrete(cb, pc,
                new MklFFT1D(cb->get_nx(2)));
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
