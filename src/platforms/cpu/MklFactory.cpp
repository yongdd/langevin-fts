/**
 * @file MklFactory.cpp
 * @brief Implementation of Intel MKL-based abstract factory.
 *
 * Provides the abstract factory implementation for creating CPU-based
 * computational objects using Intel Math Kernel Library. This factory
 * creates all platform-specific classes needed for SCFT/L-FTS simulations.
 *
 * **Created Objects:**
 *
 * - CpuComputationBox: Grid with MKL FFT support
 * - CpuAndersonMixing: CPU-optimized field mixing
 * - CpuComputationContinuous: Propagator computation (continuous chains)
 * - CpuComputationDiscrete: Propagator computation (discrete chains)
 *
 * **Memory Mode:**
 *
 * When reduce_memory_usage=true, creates CpuComputationReduceMemory* variants
 * that store only checkpoint propagators and recompute intermediate values.
 * This trades computation time for reduced memory footprint.
 *
 * **Template Instantiations:**
 *
 * - MklFactory<double>: Factory for real field simulations
 * - MklFactory<std::complex<double>>: Factory for complex field simulations
 *
 * @see AbstractFactory for factory interface
 * @see PlatformSelector for factory creation
 */

#include <iostream>
#include <array>
#include <vector>
#include <string>

#include "mkl.h"

#include "CpuArray.h"
#include "CpuComputationBox.h"
#include "CpuComputationContinuous.h"
#include "CpuComputationDiscrete.h"
#include "CpuComputationReduceMemoryContinuous.h"
#include "CpuComputationReduceMemoryDiscrete.h"
#include "CpuAndersonMixing.h"
#include "MklFactory.h"

/**
 * @brief Construct MKL factory with optional memory reduction and method selection.
 *
 * @param reduce_memory_usage Enable memory-saving mode (checkpointing)
 * @param pseudo_method       Pseudo-spectral method: "rqm4" or "etdrk4"
 * @param realspace_method    Real-space method: "cn-adi2" or "cn-adi4"
 */
template <typename T>
MklFactory<T>::MklFactory(bool reduce_memory_usage,
                          std::string pseudo_method,
                          std::string realspace_method)
{
    this->reduce_memory_usage = reduce_memory_usage;
    this->pseudo_method = pseudo_method;
    this->realspace_method = realspace_method;
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
ComputationBox<T>* MklFactory<T>::create_computation_box(
    std::vector<int> nx, std::vector<double> lx, std::vector<std::string> bc, const double *mask)
{
    return new CpuComputationBox<T>(nx, lx, bc, mask);
}
template <typename T>
ComputationBox<T>* MklFactory<T>::create_computation_box(
    std::vector<int> nx, std::vector<double> lx, std::vector<std::string> bc,
    std::vector<double> angles, const double *mask)
{
    return new CpuComputationBox<T>(nx, lx, bc, angles, mask);
}
template <typename T>
Molecules* MklFactory<T>::create_molecules_information(
    std::string chain_model, double ds, std::map<std::string, double> bond_lengths) 
{
    return new Molecules(chain_model, ds, bond_lengths);
}
template <typename T>
PropagatorComputation<T>* MklFactory<T>::create_pseudospectral_solver(ComputationBox<T>* cb, Molecules *molecules, PropagatorComputationOptimizer* propagator_computation_optimizer)
{
    std::string chain_model = molecules->get_model_name();

    if( chain_model == "continuous" && this->reduce_memory_usage == false)
        return new CpuComputationContinuous<T>(cb, molecules, propagator_computation_optimizer, "pseudospectral", this->pseudo_method);
    else if( chain_model == "continuous" && this->reduce_memory_usage == true)
        return new CpuComputationReduceMemoryContinuous<T>(cb, molecules, propagator_computation_optimizer, "pseudospectral", this->pseudo_method);
    else if( chain_model == "discrete" && this->reduce_memory_usage == false )
        return new CpuComputationDiscrete<T>(cb, molecules, propagator_computation_optimizer);
    else if( chain_model == "discrete" && this->reduce_memory_usage == true)
        return new CpuComputationReduceMemoryDiscrete<T>(cb, molecules, propagator_computation_optimizer);
    return nullptr;
}
template <typename T>
PropagatorComputation<T>* MklFactory<T>::create_realspace_solver(ComputationBox<T>* cb, Molecules *molecules, PropagatorComputationOptimizer* propagator_computation_optimizer)
{
    try
    {
        std::string chain_model = molecules->get_model_name();
        if ( chain_model == "continuous" )
        {
            return new CpuComputationContinuous<T>(cb, molecules, propagator_computation_optimizer, "realspace", this->realspace_method);
        }
        else if ( chain_model == "discrete" )
        {
            throw_with_line_number("The real-space solver does not support discrete chain model.");
        }
        return nullptr;
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