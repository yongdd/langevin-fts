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

CudaFactory::CudaFactory(bool reduce_memory_usage)
{
    this->reduce_memory_usage = reduce_memory_usage;
}

Array* CudaFactory::create_array(
    unsigned int size)
{
    return new CudaArray(size);
}

Array* CudaFactory::create_array(
    double *data,
    unsigned int size)
{
    return new CudaArray(data, size);
}
ComputationBox<double>* CudaFactory::create_computation_box(
    std::vector<int> nx, std::vector<double> lx, std::vector<std::string> bc, const double* mask)
{
    return new CudaComputationBox<double>(nx, lx, bc, mask);
}
Molecules* CudaFactory::create_molecules_information(
    std::string chain_model, double ds, std::map<std::string, double> bond_lengths) 
{
    return new Molecules(chain_model, ds, bond_lengths);
}
PropagatorComputation<double>* CudaFactory::create_pseudospectral_solver(ComputationBox<double>* cb, Molecules *molecules, PropagatorComputationOptimizer* propagator_computation_optimizer)
{
    std::string model_name = molecules->get_model_name();

    if( model_name == "continuous" && reduce_memory_usage == false)
        return new CudaComputationContinuous(cb, molecules, propagator_computation_optimizer, "pseudospectral");
    else if( model_name == "continuous" && reduce_memory_usage == true)
        return new CudaComputationReduceMemoryContinuous(cb, molecules, propagator_computation_optimizer, "pseudospectral");
    else if( model_name == "discrete" && reduce_memory_usage == false )
        return new CudaComputationDiscrete(cb, molecules, propagator_computation_optimizer);
    else if( model_name == "discrete" && reduce_memory_usage == true)
        return new CudaComputationReduceMemoryDiscrete(cb, molecules, propagator_computation_optimizer);
    return NULL;
}
PropagatorComputation<double>* CudaFactory::create_realspace_solver(ComputationBox<double>* cb, Molecules *molecules, PropagatorComputationOptimizer* propagator_computation_optimizer)
{
    try
    {
        std::string model_name = molecules->get_model_name();
        if( model_name == "continuous" && reduce_memory_usage == false)
            return new CudaComputationContinuous(cb, molecules, propagator_computation_optimizer, "realspace");
        else if( model_name == "continuous" && reduce_memory_usage == true)
            return new CudaComputationReduceMemoryContinuous(cb, molecules, propagator_computation_optimizer, "realspace");
        else if ( model_name == "discrete" )
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
AndersonMixing* CudaFactory::create_anderson_mixing(
    int n_var, int max_hist, double start_error,
    double mix_min, double mix_init)
{

    if(this->reduce_memory_usage)
    {
        return new CudaAndersonMixingReduceMemory(
            n_var, max_hist, start_error, mix_min, mix_init);
    }
    else
    {
        return new CudaAndersonMixing(
            n_var, max_hist, start_error, mix_min, mix_init);
    }
}
void CudaFactory::display_info()
{
    int device;
    int devices_count;
    struct cudaDeviceProp prop;

    const int N_BLOCKS = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();
    const int N_GPUS = CudaCommon::get_instance().get_n_gpus();

    // Get GPU info
    gpu_error_check(cudaGetDeviceCount(&devices_count));
    gpu_error_check(cudaGetDevice(&device));
    gpu_error_check(cudaGetDeviceProperties(&prop, device));

    std::cout<< "========== CUDA Setting and Device Information ==========" << std::endl;
    std::cout<< "N_BLOCKS, N_THREADS: " << N_BLOCKS << ", " << N_THREADS << std::endl;
    std::cout<< "N_GPUS (# of using GPUs): " << N_GPUS << std::endl;

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
