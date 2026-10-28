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
 * - CpuComputationBackend with MKL transforms, around which
 *   AbstractFactory::create_propagator_computation() builds the computation
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

#include "CpuComputationBox.h"
#include "CpuComputationBackend.h"
#include "CpuAndersonMixing.h"
#include "CpuFourierTransform.h"
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
std::unique_ptr<ComputationBox<T>> MklFactory<T>::create_computation_box(
    std::vector<int> nx, std::vector<double> lx, std::vector<std::string> bc, const double *mask)
{
    return std::make_unique<CpuComputationBox<T>>(nx, lx, bc, mask);
}
template <typename T>
std::unique_ptr<ComputationBox<T>> MklFactory<T>::create_computation_box(
    std::vector<int> nx, std::vector<double> lx, std::vector<std::string> bc,
    std::vector<double> angles, const double *mask)
{
    return std::make_unique<CpuComputationBox<T>>(nx, lx, bc, angles, mask);
}
template <typename T>
std::unique_ptr<ComputationBackend<T>> MklFactory<T>::create_backend(
    ComputationBox<T>* cb, Molecules* molecules, const NumericalMethodInfo& method_info,
    SpaceGroup* space_group, int max_n_streams)
{
    return std::make_unique<CpuComputationBackend<T>>(cb, molecules, method_info, FFTBackend::MKL, space_group, max_n_streams);
}
template <typename T>
std::unique_ptr<AndersonMixing<T>> MklFactory<T>::create_anderson_mixing(
    int n_var, int max_hist, double start_error,
    double mix_min, double mix_init)
{
    return std::make_unique<CpuAndersonMixing<T>>(
        n_var, max_hist, start_error, mix_min, mix_init);
}
template <typename T>
std::unique_ptr<FourierTransform<T>> MklFactory<T>::create_fourier_transform(
    ComputationBox<T>* cb, std::vector<BoundaryCondition> bc)
{
    return std::make_unique<CpuFourierTransform<T>>(cb, bc, FFTBackend::MKL);
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
