/**
 * @file CudaFactory.cu
 * @brief Implementation of CUDA-based abstract factory.
 *
 * Provides the abstract factory implementation for creating GPU-based
 * computational objects using CUDA. Creates all platform-specific classes
 * needed for SCFT/L-FTS simulations on NVIDIA GPUs.
 *
 * **Created Objects:**
 *
 * - CudaComputationBox: Grid with cuFFT support
 * - CudaAndersonMixing: GPU-accelerated field mixing
 * - Propagator computations with the CUDA backend (multi-stream)
 *
 * **Memory Modes:**
 *
 * - Normal: Full propagator storage for maximum speed
 * - Reduce memory: Checkpointing with recomputation (2-4x slower)
 *
 * **Device Information:**
 *
 * display_info() prints GPU capabilities including:
 * - Compute capability
 * - Memory sizes
 * - Block/thread limits
 *
 * **Template Instantiations:**
 *
 * - CudaFactory<double>: Real field factory
 * - CudaFactory<std::complex<double>>: Complex field factory
 *
 * @see AbstractFactory for factory interface
 * @see PlatformSelector for factory creation
 */

#include <iostream>
#include <array>
#include <vector>
#include <string>

#include "NumericalMethods.h"
#include <climits>
#include <type_traits>
#include "CudaComputationBox.h"
#include "CudaComputationBackend.h"
#include "ComputationContinuous.h"
#include "ComputationDiscrete.h"
#include "ComputationReduceMemoryContinuous.h"
#include "ComputationReduceMemoryDiscrete.h"
#include "CudaAndersonMixing.h"
#include "CudaAndersonMixingReduceMemory.h"
#include "CudaFactory.h"

/**
 * @brief Construct CUDA factory with optional checkpointing.
 *
 * @param reduce_memory Enable checkpointing mode
 */
template <typename T>
CudaFactory<T>::CudaFactory(bool reduce_memory)
{
    this->reduce_memory = reduce_memory;
}

template <typename T>
ComputationBox<T>* CudaFactory<T>::create_computation_box(
    std::vector<int> nx, std::vector<double> lx, std::vector<std::string> bc, const double* mask)
{
    return new CudaComputationBox<T>(nx, lx, bc, mask);
}
template <typename T>
ComputationBox<T>* CudaFactory<T>::create_computation_box(
    std::vector<int> nx, std::vector<double> lx, std::vector<std::string> bc,
    std::vector<double> angles, const double* mask)
{
    return new CudaComputationBox<T>(nx, lx, bc, angles, mask);
}
template <typename T>
Molecules* CudaFactory<T>::create_molecules_information(
    std::string chain_model, double ds, std::map<std::string, double> bond_lengths) 
{
    return new Molecules(chain_model, ds, bond_lengths);
}
template <typename T>
PropagatorComputation<T>* CudaFactory<T>::create_propagator_computation(ComputationBox<T>* cb, Molecules *molecules, PropagatorComputationOptimizer* propagator_computation_optimizer, std::string numerical_method, SpaceGroup* space_group)
{
    try
    {
        if constexpr (std::is_same_v<T, std::complex<double>>)
            if (space_group != nullptr)
                throw_with_line_number("Space group reduced basis is only supported for real fields.");

        // The solver: discrete chains have their own; continuous chains take the numerical method
        const bool discrete = (molecules->get_model_name() == "discrete");
        const auto& method_info = resolve_numerical_method(molecules->get_model_name(), numerical_method);
        validate_numerical_method(method_info, std::is_same_v<T, std::complex<double>>, cb->is_orthogonal());
        const std::string solver_type = method_info.solver_type;
        numerical_method = method_info.name;

        // Reduce-memory mode computes one propagator at a time
        const int max_n_streams = this->reduce_memory ? 1 : INT_MAX;
        auto create_backend = [=]() {
            return std::make_unique<CudaComputationBackend<T>>(cb, molecules, solver_type, numerical_method, space_group, max_n_streams);
        };

        typename ComputationBox<T>::SpaceGroupGuard group_change(cb, space_group);
        std::unique_ptr<PropagatorComputation<T>> computation;
        if (discrete && !this->reduce_memory)
            computation = std::make_unique<ComputationDiscrete<T>>(cb, molecules, propagator_computation_optimizer, space_group, create_backend);
        else if (discrete)
            computation = std::make_unique<ComputationReduceMemoryDiscrete<T>>(cb, molecules, propagator_computation_optimizer, space_group, create_backend);
        else if (!this->reduce_memory)
            computation = std::make_unique<ComputationContinuous<T>>(cb, molecules, propagator_computation_optimizer, solver_type, space_group, create_backend);
        else
            computation = std::make_unique<ComputationReduceMemoryContinuous<T>>(cb, molecules, propagator_computation_optimizer, solver_type, space_group, create_backend);
        group_change.commit();
        return computation.release();
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
AndersonMixing<T>* CudaFactory<T>::create_anderson_mixing(
    int n_var, int max_hist, double start_error,
    double mix_min, double mix_init)
{
    if(this->reduce_memory)
    {
        return new CudaAndersonMixingReduceMemory<T>(
            n_var, max_hist, start_error, mix_min, mix_init);
    }
    else
    {
        return new CudaAndersonMixing<T>(
            n_var, max_hist, start_error, mix_min, mix_init);
    }
}
template <typename T>
void CudaFactory<T>::display_info()
{
    int device;
    int devices_count;
    struct cudaDeviceProp prop;

    const int N_BLOCKS = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();
    // const int N_GPUS = CudaCommon::get_instance().get_n_gpus();

    // Get GPU info
    CUDA_CHECK(cudaGetDeviceCount(&devices_count));
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    std::cout<< "========== CUDA Setting and Device Information ==========" << std::endl;
    std::cout<< "N_BLOCKS, N_THREADS: " << N_BLOCKS << ", " << N_THREADS << std::endl;

    std::cout<< "DeviceCount (# of available GPUs): " << devices_count << std::endl;
    std::cout<< "Device " << device << ": \t\t\t\t" << prop.name << std::endl;

    std::cout<< "Compute capability version: \t\t" << prop.major << "." << prop.minor << std::endl;
    std::cout<< "Multiprocessor: \t\t\t" << prop.multiProcessorCount << std::endl;

    std::cout<< "Global memory: \t\t\t\t" << prop.totalGlobalMem/(1024*1024) << " MBytes" << std::endl;
    std::cout<< "Constant memory: \t\t\t" << prop.totalConstMem << " Bytes" << std::endl;
    std::cout<< "Shared memory per block: \t\t" << prop.sharedMemPerBlock << " Bytes" << std::endl;
    std::cout<< "Registers available per block: \t\t" << prop.regsPerBlock << std::endl;

    std::cout<< "Warp size: \t\t\t\t" << prop.warpSize << std::endl;
    std::cout<< "Maximum threads per block: \t\t" << prop.maxThreadsPerBlock << std::endl;
    std::cout<< "Max size of a thread block (x,y,z): \t(";
    std::cout<< prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << ")\n";
    std::cout<< "Max size of a grid size    (x,y,z): \t(";
    std::cout<< prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")\n";

#if defined(CUDART_VERSION) && (CUDART_VERSION >= 13000)
    if (prop.asyncEngineCount > 0)
        std::cout<< "Device overlap: \t\t\tYes" << std::endl;
    else
        std::cout<< "Device overlap: \t\t\tNo" << std::endl;
#else
    if (prop.deviceOverlap)
        std::cout<< "Device overlap: \t\t\tYes" << std::endl;
    else
        std::cout<< "Device overlap: \t\t\tNo" << std::endl;
#endif

    if (N_THREADS > prop.maxThreadsPerBlock)
        throw_with_line_number("'threads_per_block' cannot be greater than 'Maximum threads per block'");
    if (N_BLOCKS > prop.maxGridSize[0])
        throw_with_line_number("The number of blocks cannot be greater than 'Max size of a grid size (x)'");

    printf("================================================================\n");
}

// Explicit template instantiation
template class CudaFactory<double>;
template class CudaFactory<std::complex<double>>;
