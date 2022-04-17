
#include <iostream>
#include <cstdlib>
#include <string>
#include "CudaCommon.h"

CudaCommon::CudaCommon()
{
    const char *ENV_N_BLOCKS  = getenv("LFTS_GPU_NUM_BLOCKS");
    const char *ENV_N_THREADS = getenv("LFTS_GPU_NUM_THREADS");

    std::string env_var_n_blocks (ENV_N_BLOCKS  ? ENV_N_BLOCKS  : "");
    std::string env_var_n_threads(ENV_N_THREADS ? ENV_N_THREADS : "");

    if (env_var_n_blocks.empty())
        this->n_blocks = 256;
    else
        this->n_blocks = std::stoi(env_var_n_blocks);

    if (env_var_n_threads.empty())
        this->n_threads = 256;
    else
        this->n_threads = std::stoi(env_var_n_threads);
}
void CudaCommon::set(int n_blocks, int n_threads, int process_idx)
{
    int devices_count;
    cudaError_t err;

    this->set_n_blocks(n_blocks);
    this->set_n_threads(n_threads);

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
}
int CudaCommon::get_n_blocks()
{
    return n_blocks;
}
int CudaCommon::get_n_threads()
{
    return n_threads;
}
void CudaCommon::set_n_blocks(int n_blocks)
{
    this->n_blocks = n_blocks;
}
void CudaCommon::set_n_threads(int n_threads)
{
    this->n_threads = n_threads;
}
void CudaCommon::set_idx(int process_idx)
{
    int devices_count;
    cudaError_t err;

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
}
__global__ void multi_real(double* dst,
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

__global__ void mutiple_multi_real(int n_comp,
                          double* dst,
                          double* src1,
                          double* src2,
                          double  a, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {  
        dst[i] = a * src1[i] * src2[i];
        for(int n = 1; n < n_comp; n++)
            dst[i] += a * src1[i+n*M] * src2[i+n*M];
        i += blockDim.x * gridDim.x;
    }
}

__global__ void divide_real(double* dst,
                          double* src1,
                          double* src2,
                          double  a, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        dst[i] = a * src1[i]/src2[i];
        i += blockDim.x * gridDim.x;
    }
}
__global__ void add_multi_real(double* dst,
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

__global__ void lin_comb(double* dst,
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

__global__ void add_lin_comb(double* dst,
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

__global__ void multi_complex_real(ftsComplex* dst,
                                 double* src, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        dst[i].x = dst[i].x * src[i];
        dst[i].y = dst[i].y * src[i];
        i += blockDim.x * gridDim.x;
    }
}
__global__ void multi_complex_conjugate(double* dst,
                                 ftsComplex* src1,
                                 ftsComplex* src2, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        dst[i] = src1[i].x * src2[i].x + src1[i].y * src2[i].y;
        i += blockDim.x * gridDim.x;
    }
}
