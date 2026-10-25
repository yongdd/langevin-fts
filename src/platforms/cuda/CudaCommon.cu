/**
 * @file CudaCommon.cu
 * @brief CUDA common utilities and kernel implementations.
 *
 * Provides GPU configuration management and a comprehensive library of
 * CUDA kernels for element-wise array operations used throughout the
 * SCFT/L-FTS computations.
 *
 * **Configuration:**
 *
 * - CudaCommon singleton manages GPU blocks/threads configuration
 * - Environment variables: LFTS_GPU_NUM_BLOCKS, LFTS_GPU_NUM_THREADS
 * - Default: 1024 blocks, 256 threads per block
 *
 * **Kernel Categories:**
 *
 * - Linear scaling: ker_linear_scaling (dst = a*src + b)
 * - Exponential: ker_exp (dst = a*exp(b*src))
 * - Multiplication: ker_multi, ker_add_multi (element-wise products)
 * - Division: ker_divide (element-wise division)
 * - Linear combination: ker_lin_comb, ker_add_lin_comb
 * - Complex operations: ker_multi_complex_conjugate, ker_multi_complex_real
 * - Batched exp-dw: ker_multi_exp_dw_two, ker_multi_exp_dw_four
 *
 * **Type Conversion:**
 *
 * - cuDoubleToStdComplex: CUDA to std::complex conversion
 * - stdToCuDoubleComplex: std::complex to CUDA conversion
 * - reinterpret_map: Convert pointer maps between types
 *
 * @see CudaCommon for singleton interface
 */

#include <iostream>
#include <cstdlib>
#include <string>

#include "CudaCommon.h"

void report_cuda_error(cudaError_t code, const char *file, int line, const char *func)
{
    // A failed API call (e.g. cudaMalloc) also leaves its error in the per-thread
    // slot; clear it so the next launch check does not report it again. Sticky
    // errors of a broken context stay, as they must.
    (void)cudaGetLastError();
    std::string file_and_line("File: \"" + std::string(file) + "\", line: " + std::to_string(line) + ", function <" + std::string(func) + ">");
    throw thrust::system_error(code, thrust::cuda_category(), file_and_line);
}

void report_cuda_allocation_error(cudaError_t code, size_t bytes, const char *kind,
                                  const std::source_location& where, bool injected)
{
    if (!injected)
        (void)cudaGetLastError();   // as above (an injected failure made no CUDA call)
    std::string what = std::string(injected ? "injected failure (test hook) of the allocation"
                                            : "allocation")
        + " of " + std::to_string(bytes) + " bytes of " + kind + " memory; File: \""
        + where.file_name() + "\", line: " + std::to_string(where.line())
        + ", function <" + where.function_name() + ">";
    throw thrust::system_error(code, thrust::cuda_category(), what);
}

std::complex<double> cuDoubleToStdComplex(cuDoubleComplex z) {
    return std::complex<double>(z.x, z.y);
}

cuDoubleComplex stdToCuDoubleComplex(const std::complex<double>& z) {
    return make_cuDoubleComplex(z.real(), z.imag());
}

template<typename From, typename To>
std::map<std::string, const To*> reinterpret_map(const std::map<std::string, const From*>& input) {
    std::map<std::string, const To*> output;
    for (const auto& [key, ptr] : input) {
        output[key] = reinterpret_cast<const To*>(ptr);
    }
    return output;
}

namespace
{
    // The integer value of a launch environment variable, or default_value when it is unset
    int launch_variable(const char* name, int default_value)
    {
        const char* value = std::getenv(name);
        if (value == nullptr || *value == '\0')
            return default_value;
        std::size_t end = 0;
        int result = 0;
        try
        {
            result = std::stoi(value, &end);
        }
        catch (const std::exception&)
        {
            end = 0;
        }
        if (end == 0 || value[end] != '\0')
            throw_with_line_number(std::string(name) + " must be an integer, got \"" + value + "\".");
        return result;
    }
}

CudaCommon::CudaCommon()
{
    try{
        // 1024 blocks of 256 threads: 3-6% faster than 256 blocks at 64^3 on an A10
        // (grid-stride kernels and CUB reductions, so the results do not depend on it)
        this->n_blocks  = launch_variable("LFTS_GPU_NUM_BLOCKS", 1024);
        this->n_threads = launch_variable("LFTS_GPU_NUM_THREADS", 256);
        check_launch(this->n_blocks, this->n_threads);

    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
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
// The grid-stride kernels launched with n_blocks x n_threads rely on at most
// MAX_GRID_THREADS threads (see GridLimits.h), within the limits of the current device
void CudaCommon::check_launch(int n_blocks, int n_threads)
{
    if (n_blocks < 1 || n_threads < 1 || n_threads > 1024 ||
        static_cast<long long>(n_blocks) * n_threads > MAX_GRID_THREADS)
        throw_with_line_number("The CUDA launch of " + std::to_string(n_blocks) + " blocks x " +
            std::to_string(n_threads) + " threads is not supported: at most 1024 threads per block and " +
            std::to_string(MAX_GRID_THREADS) + " threads in total (LFTS_GPU_NUM_BLOCKS, LFTS_GPU_NUM_THREADS).");

    int device = 0, max_threads = 0, max_blocks = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&max_threads, cudaDevAttrMaxThreadsPerBlock, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&max_blocks, cudaDevAttrMaxGridDimX, device));
    if (n_threads > max_threads || n_blocks > max_blocks)
        throw_with_line_number("The CUDA launch of " + std::to_string(n_blocks) + " blocks x " +
            std::to_string(n_threads) + " threads exceeds the limits of device " + std::to_string(device) +
            ": " + std::to_string(max_threads) + " threads per block and " + std::to_string(max_blocks) +
            " blocks (LFTS_GPU_NUM_BLOCKS, LFTS_GPU_NUM_THREADS).");
}
void CudaCommon::set_n_blocks(int n_blocks)
{
    check_launch(n_blocks, this->n_threads);
    this->n_blocks = n_blocks;
}
void CudaCommon::set_n_threads(int n_threads)
{
    check_launch(this->n_blocks, n_threads);
    this->n_threads = n_threads;
}

__global__ void ker_copy_data_with_idx(cuDoubleComplex* dst, const cuDoubleComplex* src, const int* negative_k_idx, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        dst[i] = src[negative_k_idx[i]];
        i += blockDim.x * gridDim.x;
    }
}

__global__ void ker_linear_scaling(double* dst, const double* src, double a, double b, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        dst[i] = a * src[i] + b;
        i += blockDim.x * gridDim.x;
    }
}

__global__ void ker_linear_scaling(cuDoubleComplex* dst, const cuDoubleComplex* src, double a, cuDoubleComplex b, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        dst[i].x = a * src[i].x + b.x;
        dst[i].y = a * src[i].y + b.y;
        i += blockDim.x * gridDim.x;
    }
}

__global__ void ker_linear_scaling(cuDoubleComplex* dst, const cuDoubleComplex* src, cuDoubleComplex a, double b, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        // Use a temporary variable to avoid overwriting when dst and src share the same address
        cuDoubleComplex temp_dst;
        temp_dst.x = a.x * src[i].x - a.y * src[i].y + b;
        temp_dst.y = a.x * src[i].y + a.y * src[i].x;
        dst[i] = temp_dst;
        i += blockDim.x * gridDim.x;
    }
}

__global__ void ker_exp(double* dst, const double* src, double a, double exp_b, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        dst[i] = a * exp(exp_b * src[i]);
        i += blockDim.x * gridDim.x;
    }
}

__global__ void ker_exp(cuDoubleComplex* dst, const cuDoubleComplex* src, double a, double exp_b, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        // Use a temporary variable to avoid overwriting when dst and src share the same address
        cuDoubleComplex temp_dst;
        temp_dst.x = a * exp(exp_b * src[i].x) * cos(exp_b * src[i].y);
        temp_dst.y = a * exp(exp_b * src[i].x) * sin(exp_b * src[i].y);
        dst[i] = temp_dst;
        i += blockDim.x * gridDim.x;
    }
}

__global__ void ker_multi(double* dst, const double* src1, const double* src2, double a, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        dst[i] = a * src1[i] * src2[i];
        i += blockDim.x * gridDim.x;
    }
}

__global__ void ker_multi(cuDoubleComplex* dst, const cuDoubleComplex* src1, const double* src2, double a, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        dst[i].x = a * src1[i].x * src2[i];
        dst[i].y = a * src1[i].y * src2[i];
        i += blockDim.x * gridDim.x;
    }
}

__global__ void ker_multi(cuDoubleComplex* dst, const cuDoubleComplex* src1, const cuDoubleComplex* src2, double a, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        // Use a temporary variable to avoid overwriting when dst and src share the same address
        cuDoubleComplex temp_dst;
        temp_dst.x = a * (src1[i].x * src2[i].x - src1[i].y * src2[i].y);
        temp_dst.y = a * (src1[i].x * src2[i].y + src1[i].y * src2[i].x);
        dst[i] = temp_dst;
        i += blockDim.x * gridDim.x;
    }
}

__global__ void ker_multi(cuDoubleComplex* dst, const cuDoubleComplex* src1, const cuDoubleComplex* src2, cuDoubleComplex a, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        // Multiply src1[i] and src2[i]
        cuDoubleComplex temp;
        temp.x = src1[i].x * src2[i].x - src1[i].y * src2[i].y;
        temp.y = src1[i].x * src2[i].y + src1[i].y * src2[i].x;

        // Multiply the result by the complex number a
        dst[i].x = a.x * temp.x - a.y * temp.y;
        dst[i].y = a.x * temp.y + a.y * temp.x;

        i += blockDim.x * gridDim.x;
    }
}

__global__ void ker_mutiple_multi(int n_comp, double* dst, const double* src1, const double* src2, double  a, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {  
        dst[i] = a * src1[i] * src2[i];
        for(int n = 1; n < n_comp; n++)
        {
            // n*M may exceed int for large grids; offset in 64 bits
            const size_t offset = i + static_cast<size_t>(n) * M;
            dst[i] += a * src1[offset] * src2[offset];
        }
        i += blockDim.x * gridDim.x;
    }
}

__global__ void ker_mutiple_multi(int n_comp, cuDoubleComplex* dst, const cuDoubleComplex* src1, const cuDoubleComplex* src2, double a, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        cuDoubleComplex result = {a * (src1[i].x * src2[i].x - src1[i].y * src2[i].y),
                             a * (src1[i].x * src2[i].y + src1[i].y * src2[i].x)};
        for (int n = 1; n < n_comp; n++)
        {
            // n*M may exceed int for large grids; offset in 64 bits
            const size_t offset = i + static_cast<size_t>(n) * M;
            result.x += a * (src1[offset].x * src2[offset].x - src1[offset].y * src2[offset].y);
            result.y += a * (src1[offset].x * src2[offset].y + src1[offset].y * src2[offset].x);
        }
        dst[i] = result;
        i += blockDim.x * gridDim.x;
    }
}

__global__ void ker_divide(double* dst, const double* src1, const double* src2, double a, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        dst[i] = a * src1[i] / src2[i];
        i += blockDim.x * gridDim.x;
    }
}

__global__ void ker_divide(cuDoubleComplex* dst, const cuDoubleComplex* src1, const cuDoubleComplex* src2, double a, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        // Use a temporary variable to avoid overwriting when dst and src share the same address
        cuDoubleComplex temp_dst;
        double denom = src2[i].x * src2[i].x + src2[i].y * src2[i].y;
        temp_dst.x = a * (src1[i].x * src2[i].x + src1[i].y * src2[i].y) / denom;
        temp_dst.y = a * (src1[i].y * src2[i].x - src1[i].x * src2[i].y) / denom;
        dst[i] = temp_dst;
        i += blockDim.x * gridDim.x;
    }
}

__global__ void ker_add_multi(double* dst, const double* src1, const double* src2, double a, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        dst[i] += a * src1[i] * src2[i];
        i += blockDim.x * gridDim.x;
    }
}

__global__ void ker_add_multi(cuDoubleComplex* dst, const cuDoubleComplex* src1, const cuDoubleComplex* src2, double a, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        // Use a temporary variable to avoid overwriting when dst and src share the same address
        cuDoubleComplex temp_dst;
        temp_dst.x = a * (src1[i].x * src2[i].x - src1[i].y * src2[i].y);
        temp_dst.y = a * (src1[i].x * src2[i].y + src1[i].y * src2[i].x);
        dst[i].x += temp_dst.x;
        dst[i].y += temp_dst.y;
        i += blockDim.x * gridDim.x;
    }
}

__global__ void ker_add_multi(cuDoubleComplex* dst, const cuDoubleComplex* src1, const cuDoubleComplex* src2, cuDoubleComplex a, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        // Perform complex multiplication of src1[i] and src2[i]
        cuDoubleComplex temp;
        temp.x = src1[i].x * src2[i].x - src1[i].y * src2[i].y;
        temp.y = src1[i].x * src2[i].y + src1[i].y * src2[i].x;

        // Multiply the result by the complex number a
        cuDoubleComplex result;
        result.x = a.x * temp.x - a.y * temp.y;
        result.y = a.x * temp.y + a.y * temp.x;

        // Add the result to dst[i]
        dst[i].x += result.x;
        dst[i].y += result.y;
        
        i += blockDim.x * gridDim.x;
    }
}

__global__ void ker_lin_comb(double* dst, double a, const double* src1, double b, const double* src2, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        dst[i] = a * src1[i] + b * src2[i];
        i += blockDim.x * gridDim.x;
    }
}

__global__ void ker_lin_comb(cuDoubleComplex* dst, double a, const cuDoubleComplex* src1, double b, const cuDoubleComplex* src2, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        dst[i].x = a * src1[i].x + b * src2[i].x;
        dst[i].y = a * src1[i].y + b * src2[i].y;
        i += blockDim.x * gridDim.x;
    }
}

__global__ void ker_lin_comb(cuDoubleComplex* dst, cuDoubleComplex a, const cuDoubleComplex* src1, double b, const cuDoubleComplex* src2, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        // Use a temporary variable to avoid overwriting when dst and src share the same address
        cuDoubleComplex temp_dst;
        temp_dst.x = a.x * src1[i].x - a.y * src1[i].y + b * src2[i].x;
        temp_dst.y = a.x * src1[i].y + a.y * src1[i].x + b * src2[i].y;
        dst[i] = temp_dst;
        i += blockDim.x * gridDim.x;
    }
}

__global__ void ker_multi_lin_comb(double* dst, double a, const double* src1, const double* w1, double scale,
                                   double b, const double* src2, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        const double t = scale * src1[i] * w1[i];
        dst[i] = a * t + b * src2[i];
        i += blockDim.x * gridDim.x;
    }
}

__global__ void ker_multi_lin_comb(cuDoubleComplex* dst, double a, const cuDoubleComplex* src1, const cuDoubleComplex* w1, double scale,
                                   double b, const cuDoubleComplex* src2, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        // Same arithmetic as the complex ker_multi
        const double tx = scale * (src1[i].x * w1[i].x - src1[i].y * w1[i].y);
        const double ty = scale * (src1[i].x * w1[i].y + src1[i].y * w1[i].x);
        dst[i].x = a * tx + b * src2[i].x;
        dst[i].y = a * ty + b * src2[i].y;
        i += blockDim.x * gridDim.x;
    }
}

__global__ void ker_lin_comb_exp(
    double* dst,
    const double* src1, const double* exp1, double a,
    const double* src2, const double* exp2, double b,
    const double* mask, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        double val = a * (src1[i] * exp1[i]) + b * (src2[i] * exp2[i]);
        if (mask != nullptr)
            val *= mask[i];
        dst[i] = val;
        i += blockDim.x * gridDim.x;
    }
}

__global__ void ker_add_lin_comb(double* dst, double a, const double* src1, double b, const double* src2, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        dst[i] += a * src1[i] + b * src2[i];
        i += blockDim.x * gridDim.x;
    }
}

__global__ void ker_add_lin_comb(cuDoubleComplex* dst, double a, const cuDoubleComplex* src1, double b, const cuDoubleComplex* src2, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        dst[i].x += a * src1[i].x + b * src2[i].x;
        dst[i].y += a * src1[i].y + b * src2[i].y;
        i += blockDim.x * gridDim.x;
    }
}

__global__ void ker_add_lin_comb(cuDoubleComplex* dst, cuDoubleComplex a, const cuDoubleComplex* src1, cuDoubleComplex b, const cuDoubleComplex* src2, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        // Use a temporary variable to avoid overwriting when dst and src share the same address
        cuDoubleComplex temp_dst;
        temp_dst.x = a.x * src1[i].x - a.y * src1[i].y + b.x * src2[i].x - b.y * src2[i].y;
        temp_dst.y = a.x * src1[i].y + a.y * src1[i].x + b.x * src2[i].y + b.y * src2[i].x;
        dst[i].x += temp_dst.x;
        dst[i].y += temp_dst.y;
        i += blockDim.x * gridDim.x;
    }
}

__global__ void ker_multi_complex_real(cuDoubleComplex* dst, const double* src, double a, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        dst[i].x = a * dst[i].x * src[i];
        dst[i].y = a * dst[i].y * src[i];
        i += blockDim.x * gridDim.x;
    }
}

__global__ void ker_multi_complex_real(cuDoubleComplex* dst, const cuDoubleComplex* src1, const double* src2, double a, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        dst[i].x = a * src1[i].x * src2[i];
        dst[i].y = a * src1[i].y * src2[i];
        i += blockDim.x * gridDim.x;
    }
}

__global__ void ker_multi_complex_conjugate(double* dst, const cuDoubleComplex* src1, const cuDoubleComplex* src2, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        dst[i] = src1[i].x * src2[i].x + src1[i].y * src2[i].y;
        i += blockDim.x * gridDim.x;
    }
}

__global__ void ker_multi_complex_conjugate(
    cuDoubleComplex* dst, const cuDoubleComplex* src1, const cuDoubleComplex* src2, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        // Use a temporary variable to avoid overwriting when dst and src share the same address
        cuDoubleComplex temp_dst;
        temp_dst.x = src1[i].x * src2[i].x + src1[i].y * src2[i].y;
        temp_dst.y = src1[i].x * src2[i].y - src1[i].y * src2[i].x;
        dst[i] = temp_dst;
        i += blockDim.x * gridDim.x;
    }
}

__global__ void ker_multi_exp_dw_two(
    double* dst1, const double* src1, const double* exp_dw1,
    double* dst2, const double* src2, const double* exp_dw2,
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

__global__ void ker_multi_exp_dw_two(
    cuDoubleComplex* dst1, const cuDoubleComplex* src1, const cuDoubleComplex* exp_dw1,
    cuDoubleComplex* dst2, const cuDoubleComplex* src2, const cuDoubleComplex* exp_dw2,
    double a, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        // Use temporary variables to avoid overwriting when dst and src share the same address
        cuDoubleComplex temp_dst1, temp_dst2;

        temp_dst1.x = a * (src1[i].x * exp_dw1[i].x - src1[i].y * exp_dw1[i].y);
        temp_dst1.y = a * (src1[i].x * exp_dw1[i].y + src1[i].y * exp_dw1[i].x);

        temp_dst2.x = a * (src2[i].x * exp_dw2[i].x - src2[i].y * exp_dw2[i].y);
        temp_dst2.y = a * (src2[i].x * exp_dw2[i].y + src2[i].y * exp_dw2[i].x);

        dst1[i] = temp_dst1;
        dst2[i] = temp_dst2;

        i += blockDim.x * gridDim.x;
    }
}

__global__ void ker_multi_exp_dw_four(
    double* dst1, const double* src1, const double* exp_dw1,
    double* dst2, const double* src2, const double* exp_dw2,
    double* dst3, const double* src3, const double* exp_dw3,
    double* dst4, const double* src4, const double* exp_dw4,
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

__global__ void ker_multi_exp_dw_four(
    cuDoubleComplex* dst1, const cuDoubleComplex* src1, const cuDoubleComplex* exp_dw1,
    cuDoubleComplex* dst2, const cuDoubleComplex* src2, const cuDoubleComplex* exp_dw2,
    cuDoubleComplex* dst3, const cuDoubleComplex* src3, const cuDoubleComplex* exp_dw3,
    cuDoubleComplex* dst4, const cuDoubleComplex* src4, const cuDoubleComplex* exp_dw4,
    double a, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        // Use temporary variables to avoid overwriting when dst and src share the same address
        cuDoubleComplex temp_dst1, temp_dst2, temp_dst3, temp_dst4;

        temp_dst1.x = a * (src1[i].x * exp_dw1[i].x - src1[i].y * exp_dw1[i].y);
        temp_dst1.y = a * (src1[i].x * exp_dw1[i].y + src1[i].y * exp_dw1[i].x);

        temp_dst2.x = a * (src2[i].x * exp_dw2[i].x - src2[i].y * exp_dw2[i].y);
        temp_dst2.y = a * (src2[i].x * exp_dw2[i].y + src2[i].y * exp_dw2[i].x);

        temp_dst3.x = a * (src3[i].x * exp_dw3[i].x - src3[i].y * exp_dw3[i].y);
        temp_dst3.y = a * (src3[i].x * exp_dw3[i].y + src3[i].y * exp_dw3[i].x);

        temp_dst4.x = a * (src4[i].x * exp_dw4[i].x - src4[i].y * exp_dw4[i].y);
        temp_dst4.y = a * (src4[i].x * exp_dw4[i].y + src4[i].y * exp_dw4[i].x);

        dst1[i] = temp_dst1;
        dst2[i] = temp_dst2;
        dst3[i] = temp_dst3;
        dst4[i] = temp_dst4;

        i += blockDim.x * gridDim.x;
    }
}

__global__ void ker_complex_real_multi_bond_two(
    cuDoubleComplex* dst1, const double* boltz_bond1,
    cuDoubleComplex* dst2, const double* boltz_bond2,
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
    cuDoubleComplex* dst1, const double* boltz_bond1,
    cuDoubleComplex* dst2, const double* boltz_bond2,
    cuDoubleComplex* dst3, const double* boltz_bond3,
    cuDoubleComplex* dst4, const double* boltz_bond4,
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

// ============================================================
// Space group reduced basis kernels
// ============================================================

__global__ void ker_expand_reduced_basis(
    double* dst, const double* src, const int* full_to_reduced_map, const int total_grid)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < total_grid)
    {
        dst[i] = src[full_to_reduced_map[i]];
        i += blockDim.x * gridDim.x;
    }
}

__global__ void ker_expand_reduced_basis(
    cuDoubleComplex* dst, const cuDoubleComplex* src, const int* full_to_reduced_map, const int total_grid)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < total_grid)
    {
        dst[i] = src[full_to_reduced_map[i]];
        i += blockDim.x * gridDim.x;
    }
}

__global__ void ker_expand_reduced_basis_multi(
    double* dst, const double* src, const double* reduced_multiplier,
    const int* phys_to_reduced_map, const int n_phys)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < n_phys)
    {
        int ridx = phys_to_reduced_map[i];
        dst[i] = src[ridx] * reduced_multiplier[ridx];
        i += blockDim.x * gridDim.x;
    }
}

__global__ void ker_reduce_to_basis(
    double* dst, const double* src, const int* reduced_basis_indices, const int n_basis)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < n_basis)
    {
        dst[i] = src[reduced_basis_indices[i]];
        i += blockDim.x * gridDim.x;
    }
}

__global__ void ker_reduce_to_basis(
    cuDoubleComplex* dst, const cuDoubleComplex* src, const int* reduced_basis_indices, const int n_basis)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < n_basis)
    {
        dst[i] = src[reduced_basis_indices[i]];
        i += blockDim.x * gridDim.x;
    }
}

__global__ void ker_multi_map(
    double* dst, const double* reduced_multiplier, const int* phys_to_reduced_map, const int n_phys)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < n_phys)
    {
        dst[i] *= reduced_multiplier[phys_to_reduced_map[i]];
        i += blockDim.x * gridDim.x;
    }
}

__global__ void ker_multi_gather(
    double* dst, const double* src, const int* reduced_basis_indices, double norm, const int n_basis)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < n_basis)
    {
        double val = src[reduced_basis_indices[i]];
        dst[i] = val * val * norm;
        i += blockDim.x * gridDim.x;
    }
}

__global__ void ker_multi_gather(
    cuDoubleComplex* dst, const cuDoubleComplex* src, const int* reduced_basis_indices, cuDoubleComplex norm, const int n_basis)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < n_basis)
    {
        cuDoubleComplex val = src[reduced_basis_indices[i]];
        dst[i] = cuCmul(cuCmul(val, val), norm);
        i += blockDim.x * gridDim.x;
    }
}

__global__ void ker_multi_weight(
    double* dst, const double* src, const int* weights, const int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < n)
    {
        dst[i] = src[i] * static_cast<double>(weights[i]);
        i += blockDim.x * gridDim.x;
    }
}

__global__ void ker_multi_weight(
    cuDoubleComplex* dst, const cuDoubleComplex* src, const int* weights, const int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < n)
    {
        double w = static_cast<double>(weights[i]);
        dst[i] = make_cuDoubleComplex(src[i].x * w, src[i].y * w);
        i += blockDim.x * gridDim.x;
    }
}

// Explicit template instantiation

template std::map<std::string, const double*>
reinterpret_map<double, double>(const std::map<std::string, const double*>& input);

template std::map<std::string, const cuDoubleComplex*>
reinterpret_map<std::complex<double>, cuDoubleComplex>(const std::map<std::string, const std::complex<double>*>& input);
