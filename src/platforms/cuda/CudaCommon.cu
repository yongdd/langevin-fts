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

        // // The number of GPUs
        // int devices_count;
        // gpu_error_check(cudaGetDeviceCount(&devices_count));
        // const char *ENV_N_GPUS = getenv("LFTS_NUM_GPUS");
        // std::string env_var_n_gpus (ENV_N_GPUS  ? ENV_N_GPUS  : "");

        // if (env_var_n_gpus.empty())
        //     n_gpus = 1;
        // else
        //     n_gpus = std::min(std::min(std::stoi(env_var_n_gpus), devices_count), MAX_GPUS);

        // // Check if can access peer GPUs
        // if (n_gpus > 1)
        // {
        //     int can_access_from_0_to_1;
        //     int can_access_from_1_to_0;
        //     gpu_error_check(cudaDeviceCanAccessPeer(&can_access_from_0_to_1, 0, 1));
        //     gpu_error_check(cudaDeviceCanAccessPeer(&can_access_from_1_to_0, 1, 0));

        //     if (can_access_from_0_to_1 == 1 && can_access_from_1_to_0 == 1)
        //     {
        //         gpu_error_check(cudaSetDevice(0));
        //         gpu_error_check(cudaDeviceEnablePeerAccess(1, 0));
        //         gpu_error_check(cudaSetDevice(1));
        //         gpu_error_check(cudaDeviceEnablePeerAccess(0, 0));
        //     }
        //     else
        //     {
        //         std::cout << "Could not establish peer access between GPUs." << std::endl;
        //         std::cout << "Only one GPU will be used." << std::endl;
        //         n_gpus = 1;
        //     }
        // }
        // gpu_error_check(cudaSetDevice(0));
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
// int CudaCommon::get_n_gpus()
// {
//     return n_gpus;
// }
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

template <typename T>
__global__ void ker_linear_scaling(T* dst, const T* src, double a, T b, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        dst[i] = a * src[i] + b;
        i += blockDim.x * gridDim.x;
    }
}

template <>
__global__ void ker_linear_scaling<ftsComplex>(ftsComplex* dst, const ftsComplex* src, double a, ftsComplex b, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        dst[i].x = a * src[i].x + b.x;
        dst[i].y = a * src[i].y + b.y;
        i += blockDim.x * gridDim.x;
    }
}

template <typename T>
__global__ void ker_exp(T* dst, const T* src, double a, double exp_b, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        dst[i] = a * exp(exp_b * src[i]);
        i += blockDim.x * gridDim.x;
    }
}

template <>
__global__ void ker_exp<ftsComplex>(ftsComplex* dst, const ftsComplex* src, double a, double exp_b, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        dst[i].x = a * exp(exp_b * src[i].x) * cos(exp_b * src[i].y);
        dst[i].y = a * exp(exp_b * src[i].x) * sin(exp_b * src[i].y);
        i += blockDim.x * gridDim.x;
    }
}

template <typename T>
__global__ void ker_multi(T* dst, const T* src1, const T* src2, double a, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        dst[i] = a * src1[i] * src2[i];
        i += blockDim.x * gridDim.x;
    }
}

template <>
__global__ void ker_multi<ftsComplex>(ftsComplex* dst, const ftsComplex* src1, const ftsComplex* src2, double a, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        dst[i].x = a * (src1[i].x * src2[i].x - src1[i].y * src2[i].y);
        dst[i].y = a * (src1[i].x * src2[i].y + src1[i].y * src2[i].x);
        i += blockDim.x * gridDim.x;
    }
}

template <typename T>
__global__ void ker_mutiple_multi(int n_comp, T* dst, const T* src1, const T* src2, double  a, const int M)
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

template <>
__global__ void ker_mutiple_multi<ftsComplex>(int n_comp, ftsComplex* dst, const ftsComplex* src1, const ftsComplex* src2, double a, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        ftsComplex result = {a * (src1[i].x * src2[i].x - src1[i].y * src2[i].y),
                             a * (src1[i].x * src2[i].y + src1[i].y * src2[i].x)};
        for (int n = 1; n < n_comp; n++)
        {
            int offset = i + n * M;
            result.x += a * (src1[offset].x * src2[offset].x - src1[offset].y * src2[offset].y);
            result.y += a * (src1[offset].x * src2[offset].y + src1[offset].y * src2[offset].x);
        }
        dst[i] = result;
        i += blockDim.x * gridDim.x;
    }
}

template <typename T>
__global__ void ker_divide(T* dst, const T* src1, const T* src2, double a, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        dst[i] = a * src1[i] / src2[i];
        i += blockDim.x * gridDim.x;
    }
}

template <>
__global__ void ker_divide<ftsComplex>(ftsComplex* dst, const ftsComplex* src1, const ftsComplex* src2, double a, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        double denom = src2[i].x * src2[i].x + src2[i].y * src2[i].y;
        dst[i].x = a * (src1[i].x * src2[i].x + src1[i].y * src2[i].y) / denom;
        dst[i].y = a * (src1[i].y * src2[i].x - src1[i].x * src2[i].y) / denom;
        i += blockDim.x * gridDim.x;
    }
}

template <typename T>
__global__ void ker_add_multi(T* dst, const T* src1, const T* src2, double a, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        dst[i] += a * src1[i] * src2[i];
        i += blockDim.x * gridDim.x;
    }
}

template <>
__global__ void ker_add_multi<ftsComplex>(ftsComplex* dst, const ftsComplex* src1, const ftsComplex* src2, double a, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        dst[i].x += a * (src1[i].x * src2[i].x - src1[i].y * src2[i].y);
        dst[i].y += a * (src1[i].x * src2[i].y + src1[i].y * src2[i].x);
        i += blockDim.x * gridDim.x;
    }
}

template <typename T>
__global__ void ker_lin_comb(T* dst, double a, const T* src1, double b, const T* src2, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        dst[i] = a * src1[i] + b * src2[i];
        i += blockDim.x * gridDim.x;
    }
}

template <>
__global__ void ker_lin_comb<ftsComplex>(ftsComplex* dst, double a, const ftsComplex* src1, double b, const ftsComplex* src2, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        dst[i].x = a * src1[i].x + b * src2[i].x;
        dst[i].y = a * src1[i].y + b * src2[i].y;
        i += blockDim.x * gridDim.x;
    }
}

template <typename T>
__global__ void ker_add_lin_comb(T* dst, double a, const T* src1, double b, const T* src2, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        dst[i] += a * src1[i] + b * src2[i];
        i += blockDim.x * gridDim.x;
    }
}

template <>
__global__ void ker_add_lin_comb<ftsComplex>(ftsComplex* dst, double a, const ftsComplex* src1, double b, const ftsComplex* src2, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        dst[i].x += a * src1[i].x + b * src2[i].x;
        dst[i].y += a * src1[i].y + b * src2[i].y;
        i += blockDim.x * gridDim.x;
    }
}

__global__ void ker_multi_complex_real(ftsComplex* dst, const double* src, double a, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        dst[i].x = a * dst[i].x * src[i];
        dst[i].y = a * dst[i].y * src[i];
        i += blockDim.x * gridDim.x;
    }
}

__global__ void ker_multi_complex_real(ftsComplex* dst, const ftsComplex* src1, const double* src2, double a, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        dst[i].x = a * src1[i].x * src2[i];
        dst[i].y = a * src1[i].y * src2[i];
        i += blockDim.x * gridDim.x;
    }
}

template <typename T>
__global__ void ker_multi_complex_conjugate(T* dst, const ftsComplex* src1, const ftsComplex* src2, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        dst[i] = src1[i].x * src2[i].x + src1[i].y * src2[i].y;
        i += blockDim.x * gridDim.x;
    }
}

template <>
__global__ void ker_multi_complex_conjugate<ftsComplex>(
    ftsComplex* dst, const ftsComplex* src1, const ftsComplex* src2, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        dst[i].x = src1[i].x * src2[i].x + src1[i].y * src2[i].y;
        dst[i].y = src1[i].x * src2[i].y - src1[i].y * src2[i].x;
        i += blockDim.x * gridDim.x;
    }
}

template <typename T>
__global__ void ker_multi_exp_dw_two(
    T* dst1, const T* src1, const T* exp_dw1,
    T* dst2, const T* src2, const T* exp_dw2,
    double a, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        dst1[i] = a * src1[i] * exp_dw1[i];
        dst2[i] = a * src2[i] * exp_dw2[i];
        i += blockDim.x * gridDim.x;
    }
}

template <>
__global__ void ker_multi_exp_dw_two<ftsComplex>(
    ftsComplex* dst1, const ftsComplex* src1, const ftsComplex* exp_dw1,
    ftsComplex* dst2, const ftsComplex* src2, const ftsComplex* exp_dw2,
    double a, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        dst1[i].x = a * (src1[i].x * exp_dw1[i].x - src1[i].y * exp_dw1[i].y);
        dst1[i].y = a * (src1[i].x * exp_dw1[i].y + src1[i].y * exp_dw1[i].x);

        dst2[i].x = a * (src2[i].x * exp_dw2[i].x - src2[i].y * exp_dw2[i].y);
        dst2[i].y = a * (src2[i].x * exp_dw2[i].y + src2[i].y * exp_dw2[i].x);

        i += blockDim.x * gridDim.x;
    }
}

template <typename T>
__global__ void ker_multi_exp_dw_four(
    T* dst1, const T* src1, const T* exp_dw1,
    T* dst2, const T* src2, const T* exp_dw2,
    T* dst3, const T* src3, const T* exp_dw3,
    T* dst4, const T* src4, const T* exp_dw4,
    double a, const int M)
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

template <>
__global__ void ker_multi_exp_dw_four<ftsComplex>(
    ftsComplex* dst1, const ftsComplex* src1, const ftsComplex* exp_dw1,
    ftsComplex* dst2, const ftsComplex* src2, const ftsComplex* exp_dw2,
    ftsComplex* dst3, const ftsComplex* src3, const ftsComplex* exp_dw3,
    ftsComplex* dst4, const ftsComplex* src4, const ftsComplex* exp_dw4,
    double a, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        dst1[i].x = a * (src1[i].x * exp_dw1[i].x - src1[i].y * exp_dw1[i].y);
        dst1[i].y = a * (src1[i].x * exp_dw1[i].y + src1[i].y * exp_dw1[i].x);

        dst2[i].x = a * (src2[i].x * exp_dw2[i].x - src2[i].y * exp_dw2[i].y);
        dst2[i].y = a * (src2[i].x * exp_dw2[i].y + src2[i].y * exp_dw2[i].x);

        dst3[i].x = a * (src3[i].x * exp_dw3[i].x - src3[i].y * exp_dw3[i].y);
        dst3[i].y = a * (src3[i].x * exp_dw3[i].y + src3[i].y * exp_dw3[i].x);

        dst4[i].x = a * (src4[i].x * exp_dw4[i].x - src4[i].y * exp_dw4[i].y);
        dst4[i].y = a * (src4[i].x * exp_dw4[i].y + src4[i].y * exp_dw4[i].x);

        i += blockDim.x * gridDim.x;
    }
}

__global__ void ker_complex_real_multi_bond_two(
    ftsComplex* dst1, const double* boltz_bond1,
    ftsComplex* dst2, const double* boltz_bond2,
    const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        dst1[i].x *= boltz_bond1[i];
        dst1[i].y *= boltz_bond1[i];
        dst2[i].x *= boltz_bond2[i];
        dst2[i].y *= boltz_bond2[i];
        i += blockDim.x * gridDim.x;
    }
}

__global__ void ker_complex_real_multi_bond_four(
    ftsComplex* dst1, const double* boltz_bond1,
    ftsComplex* dst2, const double* boltz_bond2,
    ftsComplex* dst3, const double* boltz_bond3,
    ftsComplex* dst4, const double* boltz_bond4,
    const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        dst1[i].x *= boltz_bond1[i];
        dst1[i].y *= boltz_bond1[i];
        dst2[i].x *= boltz_bond2[i];
        dst2[i].y *= boltz_bond2[i];
        dst3[i].x *= boltz_bond3[i];
        dst3[i].y *= boltz_bond3[i];
        dst4[i].x *= boltz_bond4[i];
        dst4[i].y *= boltz_bond4[i];
        i += blockDim.x * gridDim.x;
    }
}

// Explicit template instantiations for double and std::complex<double>
template __global__ void ker_linear_scaling<double>(double*, const double*, double, double, const int);
template __global__ void ker_linear_scaling<ftsComplex>(ftsComplex*, const ftsComplex*, double, ftsComplex, const int);

template __global__ void ker_exp<double>(double*, const double*, double, double, const int);
template __global__ void ker_exp<ftsComplex>(ftsComplex*, const ftsComplex*, double, double, const int);

template __global__ void ker_multi<double>(double*, const double*, const double*, double, const int);
template __global__ void ker_multi<ftsComplex>(ftsComplex*, const ftsComplex*, const ftsComplex*, double, const int);

template __global__ void ker_mutiple_multi<double>(int, double*, const double*, const double*, double, const int);
template __global__ void ker_mutiple_multi<ftsComplex>(int, ftsComplex*, const ftsComplex*, const ftsComplex*, double, const int);

template __global__ void ker_multi_complex_conjugate<double>(double*, const ftsComplex*, const ftsComplex*, const int);
template __global__ void ker_multi_complex_conjugate<ftsComplex>(ftsComplex*, const ftsComplex*, const ftsComplex*, const int);

template __global__ void ker_divide<double>(double*, const double*, const double*, double, const int);
template __global__ void ker_divide<ftsComplex>(ftsComplex*, const ftsComplex*, const ftsComplex*, double, const int);

template __global__ void ker_add_multi<double>(double*, const double*, const double*, double, const int);
template __global__ void ker_add_multi<ftsComplex>(ftsComplex*, const ftsComplex*, const ftsComplex*, double, const int);

template __global__ void ker_lin_comb<double>(double*, double, const double*, double, const double*, const int);
template __global__ void ker_lin_comb<ftsComplex>(ftsComplex*, double, const ftsComplex*, double, const ftsComplex*, const int);

template __global__ void ker_add_lin_comb<double>(double*, double, const double*, double, const double*, const int);
template __global__ void ker_add_lin_comb<ftsComplex>(ftsComplex*, double, const ftsComplex*, double, const ftsComplex*, const int);

template __global__ void ker_multi_exp_dw_two<double>(
    double*, const double*, const double*, double*, const double*, const double*, double, const int);
template __global__ void ker_multi_exp_dw_two<ftsComplex>(
    ftsComplex*, const ftsComplex*, const ftsComplex*, ftsComplex*, const ftsComplex*, const ftsComplex*, double, const int);

template __global__ void ker_multi_exp_dw_four<double>(
    double*, const double*, const double*, double*, const double*, const double*, double*, const double*, const double*, double*, const double*, const double*, double, const int);
template __global__ void ker_multi_exp_dw_four<ftsComplex>(
    ftsComplex*, const ftsComplex*, const ftsComplex*, ftsComplex*, const ftsComplex*, const ftsComplex*, ftsComplex*, const ftsComplex*, const ftsComplex*, ftsComplex*, const ftsComplex*, const ftsComplex*, double, const int);