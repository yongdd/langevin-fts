/**
 * @file FftwFactory.cpp
 * @brief Implementation of FFTW3-based abstract factory.
 *
 * Provides the abstract factory implementation for creating CPU-based
 * computational objects using FFTW3. This factory creates all platform-specific
 * classes needed for SCFT/L-FTS simulations.
 *
 * **Created Objects:**
 *
 * - CpuComputationBox: Grid with FFTW FFT support
 * - CpuAndersonMixing: CPU-optimized field mixing
 * - CpuComputationBackend with FFTW transforms, around which
 *   AbstractFactory::create_propagator_computation() builds the computation
 *
 * **Template Instantiations:**
 *
 * - FftwFactory<double>: Factory for real field simulations
 * - FftwFactory<std::complex<double>>: Factory for complex field simulations
 *
 * @see AbstractFactory for factory interface
 * @see PlatformSelector for factory creation
 */

#include <iostream>
#include <array>
#include <vector>
#include <string>

#include <fftw3.h>

#include "CpuComputationBox.h"
#include "CpuComputationBackend.h"
#include "CpuAndersonMixing.h"
#include "FftwFactory.h"
#include "FFT.h"  // For FFTBackend enum

/**
 * @brief Construct FFTW factory with optional checkpointing.
 *
 * @param reduce_memory Enable checkpointing mode (reduces memory, increases compute)
 */
template <typename T>
FftwFactory<T>::FftwFactory(bool reduce_memory)
{
    this->reduce_memory = reduce_memory;
}

template <typename T>
ComputationBox<T>* FftwFactory<T>::create_computation_box(
    std::vector<int> nx, std::vector<double> lx, std::vector<std::string> bc, const double *mask)
{
    return new CpuComputationBox<T>(nx, lx, bc, mask);
}

template <typename T>
ComputationBox<T>* FftwFactory<T>::create_computation_box(
    std::vector<int> nx, std::vector<double> lx, std::vector<std::string> bc,
    std::vector<double> angles, const double *mask)
{
    return new CpuComputationBox<T>(nx, lx, bc, angles, mask);
}

template <typename T>
std::unique_ptr<ComputationBackend<T>> FftwFactory<T>::create_backend(
    ComputationBox<T>* cb, Molecules* molecules, const NumericalMethodInfo& method_info,
    SpaceGroup* space_group, int max_n_streams)
{
    return std::make_unique<CpuComputationBackend<T>>(cb, molecules, method_info, FFTBackend::FFTW, space_group, max_n_streams);
}

template <typename T>
AndersonMixing<T>* FftwFactory<T>::create_anderson_mixing(
    int n_var, int max_hist, double start_error,
    double mix_min, double mix_init)
{
    return new CpuAndersonMixing<T>(
        n_var, max_hist, start_error, mix_min, mix_init);
}

template <typename T>
void FftwFactory<T>::display_info()
{
    std::cout << "==================== FFTW Version ====================" << std::endl;
    std::cout << "Version:                 " << fftw_version << std::endl;
    std::cout << "================================================================" << std::endl;
}

// Explicit template instantiation
template class FftwFactory<double>;
template class FftwFactory<std::complex<double>>;
