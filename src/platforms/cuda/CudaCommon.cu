#include <iostream>
#include <cstdlib>
#include <string>

#include "CudaCommon.h"

void throw_on_cuda_error(cudaError_t code, const char *file, int line, const char *func)
{
    if (code != cudaSuccess){
        std::string file_and_line("File: \"" + std::string(file) + "\", line: " + std::to_string(line) + ", function <" + std::string(func) + ">");
        throw thrust::system_error(code, thrust::cuda_category(), file_and_line);
    }
}

CudaCommon::CudaCommon()
{
    try{
        // Intialize NUM_BLOCKS and NUM_THREADS
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

        // The number of GPUs
        int devices_count;
        gpu_error_check(cudaGetDeviceCount(&devices_count));
        const char *ENV_N_GPUS = getenv("LFTS_NUM_GPUS");
        std::string env_var_n_gpus (ENV_N_GPUS  ? ENV_N_GPUS  : "");

        if (env_var_n_gpus.empty())
            n_gpus = 1;
        else
            n_gpus = std::min(std::min(std::stoi(env_var_n_gpus), devices_count), MAX_GPUS);

        // Check if can access peer GPUs
        if (n_gpus > 1)
        {
            int can_access_from_0_to_1;
            int can_access_from_1_to_0;
            gpu_error_check(cudaDeviceCanAccessPeer(&can_access_from_0_to_1, 0, 1));
            gpu_error_check(cudaDeviceCanAccessPeer(&can_access_from_1_to_0, 1, 0));

            if (can_access_from_0_to_1 == 1 && can_access_from_1_to_0 == 1)
            {
                gpu_error_check(cudaSetDevice(0));
                gpu_error_check(cudaDeviceEnablePeerAccess(1, 0));
                gpu_error_check(cudaSetDevice(1));
                gpu_error_check(cudaDeviceEnablePeerAccess(0, 0));
            }
            else
            {
                std::cout << "Could not establish peer access between GPUs." << std::endl;
                std::cout << "Only one GPU will be used." << std::endl;
                n_gpus = 1;
            }
        }
        gpu_error_check(cudaSetDevice(0));
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaCommon::set(int n_blocks, int n_threads, int process_idx)
{
    int devices_count;

    this->set_n_blocks(n_blocks);
    this->set_n_threads(n_threads);

    // Change GPU setting
    gpu_error_check(cudaGetDeviceCount(&devices_count));
    gpu_error_check(cudaSetDevice(process_idx%devices_count));
}
int CudaCommon::get_n_blocks()
{
    return n_blocks;
}
int CudaCommon::get_n_threads()
{
    return n_threads;
}
int CudaCommon::get_n_gpus()
{
    return n_gpus;
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

    // Change GPU setting
    gpu_error_check(cudaGetDeviceCount(&devices_count));
    gpu_error_check(cudaSetDevice(process_idx%devices_count));
}

__global__ void linear_scaling_real(
            double* dst, const double* src, double a, double b, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        dst[i] = a*src[i] + b;
        i += blockDim.x * gridDim.x;
    }
}

__global__ void exp_real(double* dst,
                        const double* src,
                        double a, 
                        double exp_b, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        dst[i] = a * exp(exp_b*src[i]);
        i += blockDim.x * gridDim.x;
    }
}

__global__ void exp_complex(ftsComplex* dst,
                            const ftsComplex* src,
                            double a, 
                            double exp_b, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        dst[i].x = a * exp(exp_b * src[i].x) * cos(exp_b * src[i].y);
        dst[i].y = a * exp(exp_b * src[i].x) * sin(exp_b * src[i].y);
        i += blockDim.x * gridDim.x;
    }
}
__global__ void multi_real(double* dst,
                          const double* src1,
                          const double* src2,
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
                          const double* src1,
                          const double* src2,
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
                          const double* src1,
                          const double* src2,
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
                             const double* src1,
                             const double* src2,
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
                        const double* src1,
                        double b,
                        const double* src2,
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
                           const double* src1,
                           double b,
                           const double* src2,
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
                                   const double* src, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        dst[i].x = dst[i].x * src[i];
        dst[i].y = dst[i].y * src[i];
        i += blockDim.x * gridDim.x;
    }
}

__global__ void multi_complex_real(ftsComplex* dst,
                                  const double* src, double a, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        dst[i].x = a * dst[i].x * src[i];
        dst[i].y = a * dst[i].y * src[i];
        i += blockDim.x * gridDim.x;
    }
}

__global__ void multi_complex_conjugate(double* dst,
                                 const ftsComplex* src1,
                                 const ftsComplex* src2, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        dst[i] = src1[i].x * src2[i].x + src1[i].y * src2[i].y;
        i += blockDim.x * gridDim.x;
    }
}

__global__ void real_multi_exp_dw_two(
                        double* dst1, const double* src1, const double* exp_dw1,
                        double* dst2, const double* src2, const double* exp_dw2,
                        double  a, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        dst1[i] = a * src1[i] * exp_dw1[i];
        dst2[i] = a * src2[i] * exp_dw2[i];
        i += blockDim.x * gridDim.x;
    }
}

__global__ void real_multi_exp_dw_four(
                        double* dst1, const double* src1, const double* exp_dw1,
                        double* dst2, const double* src2, const double* exp_dw2,
                        double* dst3, const double* src3, const double* exp_dw3,
                        double* dst4, const double* src4, const double* exp_dw4,
                        double  a, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        dst1[i] = a * src1[i] * exp_dw1[i];
        dst2[i] = a * src2[i] * exp_dw2[i];
        dst3[i] = a * src3[i] * exp_dw3[i];
        dst4[i] = a * src4[i] * exp_dw4[i];
        i += blockDim.x * gridDim.x;
    }
}

__global__ void complex_real_multi_bond_two(
                        ftsComplex* dst1, const double* boltz_bond1,
                        ftsComplex* dst2, const double* boltz_bond2,
                        const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        dst1[i].x = dst1[i].x * boltz_bond1[i];
        dst1[i].y = dst1[i].y * boltz_bond1[i];
        dst2[i].x = dst2[i].x * boltz_bond2[i];
        dst2[i].y = dst2[i].y * boltz_bond2[i];
        i += blockDim.x * gridDim.x;
    }
}

__global__ void complex_real_multi_bond_four(
                        ftsComplex* dst1, const double* boltz_bond1,
                        ftsComplex* dst2, const double* boltz_bond2,
                        ftsComplex* dst3, const double* boltz_bond3,
                        ftsComplex* dst4, const double* boltz_bond4,
                        const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        dst1[i].x = dst1[i].x * boltz_bond1[i];
        dst1[i].y = dst1[i].y * boltz_bond1[i];
        dst2[i].x = dst2[i].x * boltz_bond2[i];
        dst2[i].y = dst2[i].y * boltz_bond2[i];
        dst3[i].x = dst3[i].x * boltz_bond3[i];
        dst3[i].y = dst3[i].y * boltz_bond3[i];
        dst4[i].x = dst4[i].x * boltz_bond4[i];
        dst4[i].y = dst4[i].y * boltz_bond4[i];
        i += blockDim.x * gridDim.x;
    }
}