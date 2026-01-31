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
 * **Checkpointing Mode:**
 *
 * When reduce_memory=true, creates CpuComputationReduceMemory* variants
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
#include "FFT.h"  // For FFTBackend enum

/**
 * @brief Construct MKL factory with optional checkpointing.
 *
 * @param reduce_memory Enable checkpointing mode (reduces memory, increases compute)
 */
template <typename T>
MklFactory<T>::MklFactory(bool reduce_memory)
{
    this->reduce_memory = reduce_memory;
}

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
PropagatorComputation<T>* MklFactory<T>::create_propagator_computation(ComputationBox<T>* cb, Molecules *molecules, PropagatorComputationOptimizer* propagator_computation_optimizer, std::string numerical_method, SpaceGroup* space_group)
{
    try
    {
        std::string chain_model = molecules->get_model_name();

        // Discrete chain model has its own solver, numerical_method is not used
        if (chain_model == "discrete")
        {
            if (!this->reduce_memory)
                return new CpuComputationDiscrete<T>(cb, molecules, propagator_computation_optimizer, FFTBackend::MKL, space_group);
            else
                return new CpuComputationReduceMemoryDiscrete<T>(cb, molecules, propagator_computation_optimizer, FFTBackend::MKL, space_group);
        }

        // Continuous chain model: validate and use numerical_method
        std::string solver_type;
        if (numerical_method == "rqm4" || numerical_method == "rk2")
            solver_type = "pseudospectral";
        else if (numerical_method == "cn-adi2")
            solver_type = "realspace";
        else
            throw_with_line_number("Unknown numerical method: " + numerical_method);

        if (!this->reduce_memory)
            return new CpuComputationContinuous<T>(cb, molecules, propagator_computation_optimizer, solver_type, numerical_method, FFTBackend::MKL, space_group);
        else
            return new CpuComputationReduceMemoryContinuous<T>(cb, molecules, propagator_computation_optimizer, solver_type, numerical_method, FFTBackend::MKL, space_group);
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
