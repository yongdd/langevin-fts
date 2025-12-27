/*----------------------------------------------------------
* class CudaFactory
*-----------------------------------------------------------*/
#include <iostream>
#include <array>
#include <vector>
#include <string>

#include "CudaComputationBox.h"
#include "CudaComputationContinuous.h"
#include "CudaComputationDiscrete.h"
#include "CudaComputationReduceMemoryContinuous.h"
#include "CudaComputationReduceMemoryDiscrete.h"
#include "CudaAndersonMixing.h"
#include "CudaAndersonMixingReduceMemory.h"
#include "CudaFactory.h"
#include "CudaArray.h"

template <typename T>
CudaFactory<T>::CudaFactory(bool reduce_memory_usage)
{
    // this->data_type = data_type;
    // if (this->data_type != "double" && this->data_type != "complex")
    //     throw_with_line_number("CudaFactory only supports double and complex data types. Please check your input.");

    this->reduce_memory_usage = reduce_memory_usage;
}

// Array* CudaFactory<T>::create_array(
//     unsigned int size)
// {
//     return new CudaArray(size);
// }

// Array* CudaFactory<T>::create_array(
//     double *data,
//     unsigned int size)
// {
//     return new CudaArray(data, size);
// }
template <typename T>
ComputationBox<T>* CudaFactory<T>::create_computation_box(
    std::vector<int> nx, std::vector<double> lx, std::vector<std::string> bc, const double* mask)
{
    return new CudaComputationBox<T>(nx, lx, bc, mask);
}
template <typename T>
Molecules* CudaFactory<T>::create_molecules_information(
    std::string chain_model, double ds, std::map<std::string, double> bond_lengths) 
{
    return new Molecules(chain_model, ds, bond_lengths);
}
template <typename T>
PropagatorComputation<T>* CudaFactory<T>::create_pseudospectral_solver(ComputationBox<T>* cb, Molecules *molecules, PropagatorComputationOptimizer* propagator_computation_optimizer)
{
    std::string chain_model = molecules->get_model_name();

    if( chain_model == "continuous" && this->reduce_memory_usage == false)
        return new CudaComputationContinuous<T>(cb, molecules, propagator_computation_optimizer, "pseudospectral");
    else if( chain_model == "continuous" && this->reduce_memory_usage == true)
        return new CudaComputationReduceMemoryContinuous<T>(cb, molecules, propagator_computation_optimizer, "pseudospectral");
    else if( chain_model == "discrete" && this->reduce_memory_usage == false )
        return new CudaComputationDiscrete<T>(cb, molecules, propagator_computation_optimizer);
    else if( chain_model == "discrete" && this->reduce_memory_usage == true)
        return new CudaComputationReduceMemoryDiscrete<T>(cb, molecules, propagator_computation_optimizer);
    return NULL;
}
template <typename T>
PropagatorComputation<T>* CudaFactory<T>::create_realspace_solver(ComputationBox<T>* cb, Molecules *molecules, PropagatorComputationOptimizer* propagator_computation_optimizer)
{
    try
    {
        std::string chain_model = molecules->get_model_name();
        if( chain_model == "continuous" && this->reduce_memory_usage == false)
            return new CudaComputationContinuous<T>(cb, molecules, propagator_computation_optimizer, "realspace");
        else if( chain_model == "continuous" && this->reduce_memory_usage == true)
            return new CudaComputationReduceMemoryContinuous<T>(cb, molecules, propagator_computation_optimizer, "realspace");
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
AndersonMixing<T>* CudaFactory<T>::create_anderson_mixing(
    int n_var, int max_hist, double start_error,
    double mix_min, double mix_init)
{
    if(this->reduce_memory_usage)
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
    gpu_error_check(cudaGetDeviceCount(&devices_count));
    gpu_error_check(cudaGetDevice(&device));
    gpu_error_check(cudaGetDeviceProperties(&prop, device));

    std::cout<< "========== CUDA Setting and Device Information ==========" << std::endl;
    std::cout<< "N_BLOCKS, N_THREADS: " << N_BLOCKS << ", " << N_THREADS << std::endl;
    // std::cout<< "N_GPUS (# of using GPUs): " << N_GPUS << std::endl;

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

    if(prop.deviceOverlap)
        std::cout<< "Device overlap: \t\t\tYes" << std::endl;
    else
        std::cout<< "Device overlap: \t\t\tNo" << std::endl;

    if (N_THREADS > prop.maxThreadsPerBlock)
        throw_with_line_number("'threads_per_block' cannot be greater than 'Maximum threads per block'");
    if (N_BLOCKS > prop.maxGridSize[0])
        throw_with_line_number("The number of blocks cannot be greater than 'Max size of a grid size (x)'");

    printf("================================================================\n");
}

// Explicit template instantiation
template class CudaFactory<double>;
template class CudaFactory<std::complex<double>>;