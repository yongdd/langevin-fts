/*----------------------------------------------------------
* class MklFactory
*-----------------------------------------------------------*/

#include <iostream>
#include <array>
#include <vector>
#include <string>

#include "mkl.h"

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
BranchedPolymerChain* MklFactory::create_polymer_chain(
    double ds, 
    std::map<std::string, double> dict_segment_lengths,
    std::vector<std::string> block_species, 
    std::vector<double> contour_lengths,
    std::vector<int> v, std::vector<int> u,
    std::map<int, int> v_to_grafting_index)
{
    return new BranchedPolymerChain(
        chain_model, ds, dict_segment_lengths,
        block_species, contour_lengths, v, u,
        v_to_grafting_index);
}
ComputationBox* MklFactory::create_computation_box(
    std::vector<int> nx, std::vector<double> lx)
{
    return new ComputationBox(nx, lx);
}
PseudoBranched* MklFactory::create_pseudo(ComputationBox *cb, BranchedPolymerChain *pc)
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
    MKLVersion Version;
 
    mkl_get_version(&Version);
    std::cout<< "-------------------- MKL Version --------------------" << std::endl;
    printf("Major version:           %d\n",Version.MajorVersion);
    printf("Minor version:           %d\n",Version.MinorVersion);
    printf("Update version:          %d\n",Version.UpdateVersion);
    printf("Product status:          %s\n",Version.ProductStatus);
    printf("Build:                   %s\n",Version.Build);
    printf("Platform:                %s\n",Version.Platform);
    printf("Processor optimization:  %s\n",Version.Processor);
    printf("================================================================\n");
}
