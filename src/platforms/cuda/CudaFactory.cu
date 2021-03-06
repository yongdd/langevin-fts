/*----------------------------------------------------------
* class CudaFactory
*-----------------------------------------------------------*/
#define THRUST_IGNORE_DEPRECATED_CPP_DIALECT
#define CUB_IGNORE_DEPRECATED_CPP_DIALECT

#include <iostream>
#include <array>
#include <vector>
#include <string>

#include "CudaSimulationBox.h"
#include "CudaPseudoContinuous.h"
#include "CudaPseudoDiscrete.h"
#include "CudaAndersonMixing.h"
#include "CudaFactory.h"

PolymerChain* CudaFactory::create_polymer_chain(
    double f, int NN, double chi_n, std::string model_name, double epsilon)
{
    return new PolymerChain(f, NN, chi_n, model_name, epsilon);
}
SimulationBox* CudaFactory::create_simulation_box(
    std::vector<int> nx, std::vector<double>  lx)
{
    return new CudaSimulationBox(nx, lx);
}
Pseudo* CudaFactory::create_pseudo(SimulationBox *sb, PolymerChain *pc)
{
    std::string model_name = pc->get_model_name();

    if( model_name == "continuous" )
        return new CudaPseudoContinuous(sb, pc);
    else if ( model_name == "discrete" )
        return new CudaPseudoDiscrete(sb, pc);
    return NULL;
}
AndersonMixing* CudaFactory::create_anderson_mixing(
    int n_var, int max_hist, double start_error,
    double mix_min, double mix_init)
{
    return new CudaAndersonMixing(
        n_var, max_hist, start_error, mix_min, mix_init);
}
void CudaFactory::display_info()
{
    int device;
    int devices_count;
    struct cudaDeviceProp prop;

    const int N_BLOCKS = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();

    // get GPU info
    gpu_error_check(cudaGetDeviceCount(&devices_count));
    gpu_error_check(cudaGetDevice(&device));
    gpu_error_check(cudaGetDeviceProperties(&prop, device));

    std::cout<< "---------- CUDA Setting and Device Information ----------" << std::endl;
    std::cout<< "N_BLOCKS, N_THREADS: " << N_BLOCKS << ", " << N_THREADS << std::endl;

    std::cout<< "DeviceCount: " << devices_count << std::endl;
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
}
