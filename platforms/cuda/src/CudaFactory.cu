/*----------------------------------------------------------
* class CudaFactory
*-----------------------------------------------------------*/

#include <iostream>
#include <array>
#include <vector>
#include <string>

#include "CudaSimulationBox.h"
#include "CudaPseudoGaussian.h"
#include "CudaPseudoDiscrete.h"
//#include "CudaPseudoDiscreteTwoGpu.h"
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
    //bool use_two_gpus = false;
    //const char *ENV_VAR = getenv("LFTS_USE_TWO_GPUS");
    //std::string env_var(ENV_VAR ? ENV_VAR : "");
    //if (env_var == "yes" || env_var == "y" ||
        //env_var == "YES" || env_var == "Y" ||
        //env_var == "on" || env_var == "true" ||
        //env_var == "ON" || env_var == "TRUE"){
        //use_two_gpus = true;
    //}
    //if(use_two_gpus && model_name == "discrete")
        //return new CudaPseudoDiscreteTwoGpu(sb, pc);
    //if(use_two_gpus && model_name == "gaussian")
        //return NULL;
    //else 
    
    if( model_name == "gaussian" )
        return new CudaPseudoGaussian(sb, pc);
    else if ( model_name == "discrete" )
        return new CudaPseudoDiscrete(sb, pc);
    return NULL;
}
AndersonMixing* CudaFactory::create_anderson_mixing(
    SimulationBox *sb, int n_comp,
    double max_anderson, double start_anderson_error,
    double mix_min, double mix_init)
{
    return new CudaAndersonMixing(
               sb, n_comp, max_anderson,
               start_anderson_error, mix_min, mix_init);
}
void CudaFactory::display_info()
{
    int device;
    int devices_count;
    struct cudaDeviceProp prop;
    cudaError_t err;

    const int N_BLOCKS = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();

    // get GPU info
    err = cudaGetDeviceCount(&devices_count);
    if (err != cudaSuccess)
    {
        std::cout<< cudaGetErrorString(err) << std::endl;
        exit (1);
    }
    err = cudaGetDevice(&device);
    if (err != cudaSuccess)
    {
        std::cout<< cudaGetErrorString(err) << std::endl;
        exit (1);
    }
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess)
    {
        std::cout<< cudaGetErrorString(err) << std::endl;
        exit (1);
    }

    std::cout<< "---------- CUDA Setting and Device Information ----------" << std::endl;
    std::cout<< "N_BLOCKS, N_THREADS: " << N_BLOCKS << ", " << N_THREADS << std::endl;

    std::cout<< "DeviceCount: " << devices_count << std::endl;
    printf( "Device %d: \t\t\t\t%s\n", device, prop.name );
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
    {
    std::cout<< "Device overlap: \t\t\tYes" << std::endl;
    }
    else
    {
    std::cout<< "Device overlap: \t\t\tNo" << std::endl;
    }

    if (N_THREADS > prop.maxThreadsPerBlock)
    {
        std::cout<< "'threads_per_block' cannot be greater than 'Maximum threads per block'" << std::endl;
        exit (1);
    }

    if (N_BLOCKS > prop.maxGridSize[0])
    {
        std::cout<< "The number of blocks cannot be greater than 'Max size of a grid size (x)'" << std::endl;
        exit (1);
    }
    
}
