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
 * - Propagator computations (ComputationContinuous, ComputationDiscrete) with
 *   the CPU backend and MKL transforms
 *
 * **Checkpointing Mode:**
 *
 * When reduce_memory=true, creates the ComputationReduceMemory* variants
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

#include <climits>
#include "mkl.h"

#include "CpuComputationBox.h"
#include "CpuComputationBackend.h"
#include "ComputationContinuous.h"
#include "ComputationDiscrete.h"
#include "ComputationReduceMemoryContinuous.h"
#include "ComputationReduceMemoryDiscrete.h"
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
    mkl_set_num_threads(1);  // Disable MKL internal threading (parallelism at propagator level)
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
        // The solver: discrete chains have their own; continuous chains take the numerical method
        const bool discrete = (molecules->get_model_name() == "discrete");
        std::string solver_type = "pseudospectral";
        if (discrete)
            numerical_method = "";
        else if (numerical_method == "cn-adi2")
            solver_type = "realspace";
        else if (numerical_method != "rqm4" && numerical_method != "rk2")
            throw_with_line_number("Unknown numerical method: " + numerical_method);

        // Reduce-memory mode computes one propagator at a time
        const int max_n_streams = this->reduce_memory ? 1 : INT_MAX;
        auto create_backend = [=]() {
            return new CpuComputationBackend<T>(cb, molecules, solver_type, numerical_method, FFTBackend::MKL, space_group, max_n_streams);
        };

        if (discrete && !this->reduce_memory)
            return new ComputationDiscrete<T>(cb, molecules, propagator_computation_optimizer, space_group, create_backend);
        if (discrete)
            return new ComputationReduceMemoryDiscrete<T>(cb, molecules, propagator_computation_optimizer, space_group, create_backend);
        if (!this->reduce_memory)
            return new ComputationContinuous<T>(cb, molecules, propagator_computation_optimizer, solver_type, space_group, create_backend);
        return new ComputationReduceMemoryContinuous<T>(cb, molecules, propagator_computation_optimizer, solver_type, space_group, create_backend);
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
