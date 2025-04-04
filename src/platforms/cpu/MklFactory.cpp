/*----------------------------------------------------------
* class MklFactory
*-----------------------------------------------------------*/

#include <iostream>
#include <array>
#include <vector>
#include <string>

#include "mkl.h"

#include "CpuArray.h"
#include "CpuComputationBox.h"
#include "CpuComputationContinuous.h"
#include "CpuComputationDiscrete.h"
#include "CpuAndersonMixing.h"
#include "MklFactory.h"

template <typename T>
MklFactory<T>::MklFactory(bool reduce_memory_usage)
{
    // this->data_type = data_type;
    // if (this->data_type != "double" && this->data_type != "complex")
    //     throw_with_line_number("MklFactory only supports double and complex data types. Please check your input.");

    this->reduce_memory_usage = reduce_memory_usage;

    if (this->reduce_memory_usage)
        std::cout << "(warning) Reducing memory usage option only works for CUDA. This option will be ignored in MKL." << std::endl;
}
// template <typename T>
// Array* MklFactory<T>::create_array(
//     unsigned int size)
// {
//     return new CpuArray(size);
// }
// template <typename T>
// Array* MklFactory<T>::create_array(
//     double *data,
//     unsigned int size)
// {
//     return new CpuArray(data, size);
// }

template <typename T>
ComputationBox* MklFactory<T>::create_computation_box(
    std::vector<int> nx, std::vector<double> lx, std::vector<std::string> bc, const double *mask)
{
    return new CpuComputationBox(nx, lx, bc, mask);
}
template <typename T>
Molecules* MklFactory<T>::create_molecules_information(
    std::string chain_model, double ds, std::map<std::string, double> bond_lengths) 
{
    return new Molecules(chain_model, ds, bond_lengths);
}
template <typename T>
PropagatorComputation<T>* MklFactory<T>::create_pseudospectral_solver(ComputationBox* cb, Molecules *molecules, PropagatorComputationOptimizer* propagator_computation_optimizer)
{
    std::string chain_model = molecules->get_model_name();
    if ( chain_model == "continuous" )
    {
        return new CpuComputationContinuous<T>(cb, molecules, propagator_computation_optimizer, "pseudospectral");
    }
    else if ( chain_model == "discrete" )
    {
        return new CpuComputationDiscrete<T>(cb, molecules, propagator_computation_optimizer);
    }
    return NULL;
}
template <typename T>
PropagatorComputation<T>* MklFactory<T>::create_realspace_solver(ComputationBox* cb, Molecules *molecules, PropagatorComputationOptimizer* propagator_computation_optimizer)
{
    try
    {
        std::string chain_model = molecules->get_model_name();
        if ( chain_model == "continuous" )
        {
            return new CpuComputationContinuous<T>(cb, molecules, propagator_computation_optimizer, "realspace");
        }
        else if ( chain_model == "discrete" )
        {
            throw_with_line_number("The real-space solver does not support discrete chain model.");
        }
        return NULL;
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
AndersonMixing<T>* MklFactory<T>::create_anderson_mixing(
    int n_var, int max_hist, double start_error,
    double mix_min, double mix_init)
{
    return new CpuAndersonMixing<T>(
        n_var, max_hist, start_error, mix_min, mix_init);
}
template <typename T>
void MklFactory<T>::display_info()
{
    MKLVersion Version;
 
    mkl_get_version(&Version);
    std::cout<< "==================== MKL Version ====================" << std::endl;
    printf("Major version:           %d\n",Version.MajorVersion);
    printf("Minor version:           %d\n",Version.MinorVersion);
    printf("Update version:          %d\n",Version.UpdateVersion);
    printf("Product status:          %s\n",Version.ProductStatus);
    printf("Build:                   %s\n",Version.Build);
    printf("Platform:                %s\n",Version.Platform);
    printf("Processor optimization:  %s\n",Version.Processor);
    printf("================================================================\n");
}

// Explicit template instantiation
template class MklFactory<double>;
template class MklFactory<std::complex<double>>;