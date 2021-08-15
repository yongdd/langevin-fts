
#include <iostream>
#include "CudaCommon.h"

CudaCommon::CudaCommon(int N_BLOCKS, int N_THREADS, int process_idx)
{
    int device;
    int devices_count;  
    struct cudaDeviceProp prop;
    cudaError_t err;

    this->N_BLOCKS = N_BLOCKS;
    this->N_THREADS = N_THREADS;

    // change GPU setting
    err = cudaGetDeviceCount(&devices_count);
    if (err != cudaSuccess)
    {
        std::cout<< cudaGetErrorString(err) << std::endl;
        exit (1);
    }
    err = cudaSetDevice(process_idx%devices_count);
    if (err != cudaSuccess)
    {
        std::cout<< cudaGetErrorString(err) << std::endl;
        exit (1);
    }
     // get GPU info
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
    printf( "Device %d : \t\t\t\t%s\n", device, prop.name );
    std::cout<< "Compute capability version : \t\t" << prop.major << "." << prop.minor << std::endl;
    std::cout<< "Multiprocessor : \t\t\t" << prop.multiProcessorCount << std::endl;

    std::cout<< "Global memory : \t\t\t" << prop.totalGlobalMem/(1024*1024) << " MBytes" << std::endl;
    std::cout<< "Constant memory : \t\t\t" << prop.totalConstMem << " Bytes" << std::endl;
    std::cout<< "Shared memory per block : \t\t" << prop.sharedMemPerBlock << " Bytes" << std::endl;
    std::cout<< "Registers available per block : \t" << prop.regsPerBlock << std::endl;

    std::cout<< "Warp size : \t\t\t\t" << prop.warpSize << std::endl;
    std::cout<< "Maximum threads per block : \t\t" << prop.maxThreadsPerBlock << std::endl;
    std::cout<< "Max size of a thread block (x,y,z) : \t(";
    std::cout<< prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << ")\n";
    std::cout<< "Max size of a grid size    (x,y,z) : \t(";
    std::cout<< prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")\n";

    //if(prop.deviceOverlap)
    //{
        //std::cout<< "Device overlap : \t\t\t Yes" << std::endl;
    //}
    //else
    //{
        //std::cout<< "Device overlap : \t\t\t No" << std::endl;
    //}

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
    if (prop.warpSize < 32)
    {
        std::cout<< "'Warp size' cannot be less than 32 due to synchronization in 'multi_inner_product_kernel'." << std::endl;
        exit (1);
    }

    if (N_THREADS > 1024)
    {
        std::cout<<"'threads_per_block' cannot be greater than 1024 because of 'multi_inner_product_kernel'." << std::endl;
        exit (1);
    }
}
void CudaCommon::initialize(int N_BLOCKS, int N_THREADS, int process_idx)
{
    CudaCommon::get_instance(N_BLOCKS, N_THREADS, process_idx);
}
__global__ void multiReal(double* dst,
                          double* src1,
                          double* src2,
                          double  a, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        dst[i] = a * src1[i] * src2[i];
        i += blockDim.x * gridDim.x;
    }
}

__global__ void addMultiReal(double* dst,
                             double* src1,
                             double* src2,
                             double  a, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        dst[i] = dst[i] + a * src1[i] * src2[i];
        i += blockDim.x * gridDim.x;
    }
}

__global__ void linComb(double* dst,
                        double a,
                        double* src1,
                        double b,
                        double* src2,
                        const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        dst[i] = a*src1[i] + b*src2[i];
        i += blockDim.x * gridDim.x;
    }
}

__global__ void addLinComb(double* dst,
                           double a,
                           double* src1,
                           double b,
                           double* src2,
                           const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        dst[i] = dst[i] + a*src1[i] + b*src2[i];
        i += blockDim.x * gridDim.x;
    }
}

__global__ void multiComplexReal(ftsComplex* a,
                                 double* b, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        a[i].x = a[i].x * b[i];
        a[i].y = a[i].y * b[i];
        i += blockDim.x * gridDim.x;
    }
}
