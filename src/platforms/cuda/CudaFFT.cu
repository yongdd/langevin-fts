/**
 * @file CudaFFT.cu
 * @brief CUDA implementation of spectral transforms for all boundary conditions.
 *
 * Implements:
 * - cuFFT for periodic boundary conditions
 * - DCT-II/III for reflecting (Neumann) BCs
 * - DST-II/III for absorbing (Dirichlet) BCs
 *
 * **Transform Definitions:**
 *
 * DCT-II (forward): X[k] = sum_{n=0}^{N-1} x[n] * cos(pi*k*(n+0.5)/N)
 * DCT-III (inverse): x[n] = X[0]/N + (2/N) * sum_{k=1}^{N-1} X[k] * cos(pi*k*(n+0.5)/N)
 *
 * DST-II (forward): X[k] = sum_{n=0}^{N-1} x[n] * sin(pi*(k+1)*(n+0.5)/N)
 * DST-III (inverse): x[n] = (1/N) * [(-1)^n * X[N-1] + 2*sum_{k=0}^{N-2} X[k]*sin(pi*(2n+1)*(k+1)/(2N))]
 *
 * @see CudaFFT.h for class documentation
 */

#include <iostream>
#include <cmath>
#include <stdexcept>

#include "CudaFFT.h"
#include "CudaCommon.h"

//------------------------------------------------------------------------------
// CUDA Kernels for DCT/DST transforms
//------------------------------------------------------------------------------

/**
 * @brief DCT-II forward transform kernel.
 */
__global__ void ker_dct2_forward(
    double* dst, const double* src, const double* cos_table,
    int n, int stride, int num_transforms, int total_grid)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_transforms * stride * n) return;

    int batch = idx / (n * stride);
    int remainder = idx % (n * stride);
    int k = remainder / stride;
    int s = remainder % stride;

    int offset = batch * n * stride + s;

    double sum = 0.0;
    for (int j = 0; j < n; ++j)
    {
        sum += src[offset + j * stride] * cos_table[k * n + j];
    }
    dst[offset + k * stride] = sum;
}

/**
 * @brief DCT-III backward transform kernel.
 */
__global__ void ker_dct3_backward(
    double* dst, const double* src, const double* cos_table,
    int n, int stride, int num_transforms, int total_grid)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_transforms * stride * n) return;

    int batch = idx / (n * stride);
    int remainder = idx % (n * stride);
    int j = remainder / stride;
    int s = remainder % stride;

    int offset = batch * n * stride + s;

    double sum = src[offset] / n;
    for (int k = 1; k < n; ++k)
    {
        sum += (2.0 / n) * src[offset + k * stride] * cos_table[k * n + j];
    }
    dst[offset + j * stride] = sum;
}

/**
 * @brief DST-II forward transform kernel.
 */
__global__ void ker_dst2_forward(
    double* dst, const double* src, const double* sin_table,
    int n, int stride, int num_transforms, int total_grid)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_transforms * stride * n) return;

    int batch = idx / (n * stride);
    int remainder = idx % (n * stride);
    int k = remainder / stride;
    int s = remainder % stride;

    int offset = batch * n * stride + s;

    double sum = 0.0;
    for (int j = 0; j < n; ++j)
    {
        sum += src[offset + j * stride] * sin_table[k * n + j];
    }
    dst[offset + k * stride] = sum;
}

/**
 * @brief DST-III backward transform kernel.
 */
__global__ void ker_dst3_backward(
    double* dst, const double* src,
    int n, int stride, int num_transforms, int total_grid)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_transforms * stride * n) return;

    int batch = idx / (n * stride);
    int remainder = idx % (n * stride);
    int j = remainder / stride;
    int s = remainder % stride;

    int offset = batch * n * stride + s;

    const double PI = 3.14159265358979323846;

    double sign = (j % 2 == 0) ? 1.0 : -1.0;
    double sum = sign * src[offset + (n - 1) * stride];

    for (int k = 0; k < n - 1; ++k)
    {
        sum += 2.0 * src[offset + k * stride] * sin(PI * (2 * j + 1) * (k + 1) / (2.0 * n));
    }

    dst[offset + j * stride] = sum / n;
}

/**
 * @brief Copy data kernel.
 */
__global__ void ker_copy_fft_data(double* dst, const double* src, int M)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M)
        dst[idx] = src[idx];
}

/**
 * @brief Copy from complex to real (extract real part).
 */
__global__ void ker_complex_to_real_fft(double* dst, const cuDoubleComplex* src, int M)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M)
        dst[idx] = src[idx].x;
}

/**
 * @brief Copy from real to complex (imaginary = 0).
 */
__global__ void ker_real_to_complex_fft(cuDoubleComplex* dst, const double* src, int M)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M)
    {
        dst[idx].x = src[idx];
        dst[idx].y = 0.0;
    }
}

/**
 * @brief Copy interleaved complex to double array (for periodic BC double* interface).
 */
__global__ void ker_complex_to_interleaved(double* dst, const cuDoubleComplex* src, int M_COMPLEX)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M_COMPLEX)
    {
        dst[2 * idx] = src[idx].x;
        dst[2 * idx + 1] = src[idx].y;
    }
}

/**
 * @brief Copy interleaved double array to complex (for periodic BC double* interface).
 */
__global__ void ker_interleaved_to_complex(cuDoubleComplex* dst, const double* src, int M_COMPLEX)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M_COMPLEX)
    {
        dst[idx].x = src[2 * idx];
        dst[idx].y = src[2 * idx + 1];
    }
}

/**
 * @brief Scale real array.
 */
__global__ void ker_scale_fft(double* data, double scale, int M)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M)
        data[idx] *= scale;
}

//------------------------------------------------------------------------------
// Constructor (all periodic - backward compatible)
//------------------------------------------------------------------------------
template <typename T, int DIM>
CudaFFT<T, DIM>::CudaFFT(std::array<int, DIM> nx)
    : nx_(nx), d_work_buffer_(nullptr), d_temp_buffer_(nullptr), is_all_periodic_(true)
{
    // Set all BCs to periodic
    for (int d = 0; d < DIM; ++d)
        bc_[d] = BoundaryCondition::PERIODIC;

    try
    {
        total_grid_ = 1;
        for (int d = 0; d < DIM; ++d)
            total_grid_ *= nx_[d];

        // Periodic: last dimension halved for r2c
        if (DIM == 3)
            total_complex_grid_ = nx_[0] * nx_[1] * (nx_[2] / 2 + 1);
        else if (DIM == 2)
            total_complex_grid_ = nx_[0] * (nx_[1] / 2 + 1);
        else if (DIM == 1)
            total_complex_grid_ = nx_[0] / 2 + 1;

        // Allocate work buffers (for double* interface)
        gpu_error_check(cudaMalloc((void**)&d_work_buffer_, sizeof(cuDoubleComplex) * total_complex_grid_));
        gpu_error_check(cudaMalloc((void**)&d_temp_buffer_, sizeof(double) * total_grid_));

        d_sin_tables_.resize(DIM, nullptr);
        d_cos_tables_.resize(DIM, nullptr);

        initPeriodicFFT();
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

//------------------------------------------------------------------------------
// Constructor (with boundary conditions)
//------------------------------------------------------------------------------
template <typename T, int DIM>
CudaFFT<T, DIM>::CudaFFT(std::array<int, DIM> nx, std::array<BoundaryCondition, DIM> bc)
    : nx_(nx), bc_(bc), d_work_buffer_(nullptr), d_temp_buffer_(nullptr),
      plan_forward_(0), plan_backward_(0)
{
    try
    {
        total_grid_ = 1;
        for (int d = 0; d < DIM; ++d)
            total_grid_ *= nx_[d];

        // Check if all periodic
        is_all_periodic_ = true;
        for (int d = 0; d < DIM; ++d)
        {
            if (bc_[d] != BoundaryCondition::PERIODIC)
            {
                is_all_periodic_ = false;
                break;
            }
        }

        // Mixed periodic/non-periodic not supported
        if (!is_all_periodic_)
        {
            for (int d = 0; d < DIM; ++d)
            {
                if (bc_[d] == BoundaryCondition::PERIODIC)
                {
                    throw_with_line_number("Mixed periodic and non-periodic BCs not yet supported. "
                                          "Use all periodic or all non-periodic (reflecting/absorbing).");
                }
            }
        }

        // Compute complex grid size
        if (is_all_periodic_)
        {
            if (DIM == 3)
                total_complex_grid_ = nx_[0] * nx_[1] * (nx_[2] / 2 + 1);
            else if (DIM == 2)
                total_complex_grid_ = nx_[0] * (nx_[1] / 2 + 1);
            else if (DIM == 1)
                total_complex_grid_ = nx_[0] / 2 + 1;
        }
        else
        {
            total_complex_grid_ = total_grid_;
        }

        // Allocate work buffers
        if (is_all_periodic_)
        {
            gpu_error_check(cudaMalloc((void**)&d_work_buffer_, sizeof(cuDoubleComplex) * total_complex_grid_));
            gpu_error_check(cudaMalloc((void**)&d_temp_buffer_, sizeof(double) * total_grid_));
        }
        else
        {
            gpu_error_check(cudaMalloc((void**)&d_work_buffer_, sizeof(double) * total_grid_));
            gpu_error_check(cudaMalloc((void**)&d_temp_buffer_, sizeof(double) * total_grid_));
        }

        d_sin_tables_.resize(DIM, nullptr);
        d_cos_tables_.resize(DIM, nullptr);

        if (is_all_periodic_)
        {
            initPeriodicFFT();
        }
        else
        {
            precomputeTrigTables();
        }
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

//------------------------------------------------------------------------------
// Destructor
//------------------------------------------------------------------------------
template <typename T, int DIM>
CudaFFT<T, DIM>::~CudaFFT()
{
    if (d_work_buffer_ != nullptr)
        cudaFree(d_work_buffer_);
    if (d_temp_buffer_ != nullptr)
        cudaFree(d_temp_buffer_);

    for (int d = 0; d < DIM; ++d)
    {
        if (d_sin_tables_[d] != nullptr)
            cudaFree(d_sin_tables_[d]);
        if (d_cos_tables_[d] != nullptr)
            cudaFree(d_cos_tables_[d]);
    }

    if (plan_forward_ != 0)
        cufftDestroy(plan_forward_);
    if (plan_backward_ != 0)
        cufftDestroy(plan_backward_);
}

//------------------------------------------------------------------------------
// Initialize cuFFT plans for periodic BC
//------------------------------------------------------------------------------
template <typename T, int DIM>
void CudaFFT<T, DIM>::initPeriodicFFT()
{
    int total_grid[DIM];
    for (int d = 0; d < DIM; ++d)
        total_grid[d] = nx_[d];

    cufftType cufft_forward;
    cufftType cufft_backward;

    if constexpr (std::is_same<T, double>::value)
    {
        cufft_forward = CUFFT_D2Z;
        cufft_backward = CUFFT_Z2D;
    }
    else
    {
        cufft_forward = CUFFT_Z2Z;
        cufft_backward = CUFFT_Z2Z;
    }

    cufftPlanMany(&plan_forward_, DIM, total_grid, NULL, 1, 0, NULL, 1, 0, cufft_forward, 1);
    cufftPlanMany(&plan_backward_, DIM, total_grid, NULL, 1, 0, NULL, 1, 0, cufft_backward, 1);
}

//------------------------------------------------------------------------------
// Precompute trig tables and upload to GPU
//------------------------------------------------------------------------------
template <typename T, int DIM>
void CudaFFT<T, DIM>::precomputeTrigTables()
{
    const double PI = 3.14159265358979323846;

    for (int dim = 0; dim < DIM; ++dim)
    {
        int n = nx_[dim];

        if (bc_[dim] == BoundaryCondition::REFLECTING)
        {
            std::vector<double> cos_table(n * n);
            for (int k = 0; k < n; ++k)
            {
                for (int j = 0; j < n; ++j)
                {
                    cos_table[k * n + j] = std::cos(PI * k * (j + 0.5) / n);
                }
            }
            gpu_error_check(cudaMalloc((void**)&d_cos_tables_[dim], sizeof(double) * n * n));
            gpu_error_check(cudaMemcpy(d_cos_tables_[dim], cos_table.data(),
                                       sizeof(double) * n * n, cudaMemcpyHostToDevice));
        }
        else if (bc_[dim] == BoundaryCondition::ABSORBING)
        {
            std::vector<double> sin_table(n * n);
            for (int k = 0; k < n; ++k)
            {
                for (int j = 0; j < n; ++j)
                {
                    sin_table[k * n + j] = std::sin(PI * (k + 1) * (j + 0.5) / n);
                }
            }
            gpu_error_check(cudaMalloc((void**)&d_sin_tables_[dim], sizeof(double) * n * n));
            gpu_error_check(cudaMemcpy(d_sin_tables_[dim], sin_table.data(),
                                       sizeof(double) * n * n, cudaMemcpyHostToDevice));
        }
    }
}

//------------------------------------------------------------------------------
// Get strides for dimension-by-dimension transform
//------------------------------------------------------------------------------
template <typename T, int DIM>
void CudaFFT<T, DIM>::getStrides(int dim, int& stride, int& num_transforms) const
{
    stride = 1;
    for (int d = dim + 1; d < DIM; ++d)
        stride *= nx_[d];

    num_transforms = 1;
    for (int d = 0; d < dim; ++d)
        num_transforms *= nx_[d];
}

//------------------------------------------------------------------------------
// Apply DCT-II forward transform for one dimension
//------------------------------------------------------------------------------
template <typename T, int DIM>
void CudaFFT<T, DIM>::applyDCT2Forward(double* d_data, int dim, cudaStream_t stream)
{
    const int N_BLOCKS = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();

    int n = nx_[dim];
    int stride, num_transforms;
    getStrides(dim, stride, num_transforms);

    ker_dct2_forward<<<N_BLOCKS, N_THREADS, 0, stream>>>(
        d_temp_buffer_, d_data, d_cos_tables_[dim],
        n, stride, num_transforms, total_grid_);
    gpu_error_check(cudaPeekAtLastError());

    ker_copy_fft_data<<<N_BLOCKS, N_THREADS, 0, stream>>>(d_data, d_temp_buffer_, total_grid_);
    gpu_error_check(cudaPeekAtLastError());
}

//------------------------------------------------------------------------------
// Apply DCT-III backward transform for one dimension
//------------------------------------------------------------------------------
template <typename T, int DIM>
void CudaFFT<T, DIM>::applyDCT3Backward(double* d_data, int dim, cudaStream_t stream)
{
    const int N_BLOCKS = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();

    int n = nx_[dim];
    int stride, num_transforms;
    getStrides(dim, stride, num_transforms);

    ker_dct3_backward<<<N_BLOCKS, N_THREADS, 0, stream>>>(
        d_temp_buffer_, d_data, d_cos_tables_[dim],
        n, stride, num_transforms, total_grid_);
    gpu_error_check(cudaPeekAtLastError());

    ker_copy_fft_data<<<N_BLOCKS, N_THREADS, 0, stream>>>(d_data, d_temp_buffer_, total_grid_);
    gpu_error_check(cudaPeekAtLastError());
}

//------------------------------------------------------------------------------
// Apply DST-II forward transform for one dimension
//------------------------------------------------------------------------------
template <typename T, int DIM>
void CudaFFT<T, DIM>::applyDST2Forward(double* d_data, int dim, cudaStream_t stream)
{
    const int N_BLOCKS = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();

    int n = nx_[dim];
    int stride, num_transforms;
    getStrides(dim, stride, num_transforms);

    ker_dst2_forward<<<N_BLOCKS, N_THREADS, 0, stream>>>(
        d_temp_buffer_, d_data, d_sin_tables_[dim],
        n, stride, num_transforms, total_grid_);
    gpu_error_check(cudaPeekAtLastError());

    ker_copy_fft_data<<<N_BLOCKS, N_THREADS, 0, stream>>>(d_data, d_temp_buffer_, total_grid_);
    gpu_error_check(cudaPeekAtLastError());
}

//------------------------------------------------------------------------------
// Apply DST-III backward transform for one dimension
//------------------------------------------------------------------------------
template <typename T, int DIM>
void CudaFFT<T, DIM>::applyDST3Backward(double* d_data, int dim, cudaStream_t stream)
{
    const int N_BLOCKS = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();

    int n = nx_[dim];
    int stride, num_transforms;
    getStrides(dim, stride, num_transforms);

    ker_dst3_backward<<<N_BLOCKS, N_THREADS, 0, stream>>>(
        d_temp_buffer_, d_data,
        n, stride, num_transforms, total_grid_);
    gpu_error_check(cudaPeekAtLastError());

    ker_copy_fft_data<<<N_BLOCKS, N_THREADS, 0, stream>>>(d_data, d_temp_buffer_, total_grid_);
    gpu_error_check(cudaPeekAtLastError());
}

//------------------------------------------------------------------------------
// Forward transform (double* interface)
//------------------------------------------------------------------------------
template <typename T, int DIM>
void CudaFFT<T, DIM>::forward_stream(T* d_rdata, double* d_cdata, cudaStream_t stream)
{
    try
    {
        const int N_BLOCKS = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        if (is_all_periodic_)
        {
            // Use cuFFT, then copy to interleaved double array
            cuDoubleComplex* d_complex = reinterpret_cast<cuDoubleComplex*>(d_work_buffer_);

            if constexpr (std::is_same<T, double>::value)
            {
                cufftSetStream(plan_forward_, stream);
                cufftExecD2Z(plan_forward_, d_rdata, d_complex);
            }
            else
            {
                cufftSetStream(plan_forward_, stream);
                cufftExecZ2Z(plan_forward_, reinterpret_cast<cuDoubleComplex*>(d_rdata),
                             d_complex, CUFFT_FORWARD);
            }
            gpu_error_check(cudaPeekAtLastError());

            // Copy complex to interleaved double
            ker_complex_to_interleaved<<<N_BLOCKS, N_THREADS, 0, stream>>>(
                d_cdata, d_complex, total_complex_grid_);
            gpu_error_check(cudaPeekAtLastError());
        }
        else
        {
            // Non-periodic: apply DCT/DST dimension by dimension
            if constexpr (std::is_same<T, double>::value)
            {
                ker_copy_fft_data<<<N_BLOCKS, N_THREADS, 0, stream>>>(d_work_buffer_, d_rdata, total_grid_);
            }
            else
            {
                ker_complex_to_real_fft<<<N_BLOCKS, N_THREADS, 0, stream>>>(
                    d_work_buffer_, reinterpret_cast<cuDoubleComplex*>(d_rdata), total_grid_);
            }
            gpu_error_check(cudaPeekAtLastError());

            for (int dim = 0; dim < DIM; ++dim)
            {
                if (bc_[dim] == BoundaryCondition::REFLECTING)
                    applyDCT2Forward(d_work_buffer_, dim, stream);
                else if (bc_[dim] == BoundaryCondition::ABSORBING)
                    applyDST2Forward(d_work_buffer_, dim, stream);
            }

            ker_copy_fft_data<<<N_BLOCKS, N_THREADS, 0, stream>>>(d_cdata, d_work_buffer_, total_complex_grid_);
            gpu_error_check(cudaPeekAtLastError());
        }
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

//------------------------------------------------------------------------------
// Backward transform (double* interface)
//------------------------------------------------------------------------------
template <typename T, int DIM>
void CudaFFT<T, DIM>::backward_stream(double* d_cdata, T* d_rdata, cudaStream_t stream)
{
    try
    {
        const int N_BLOCKS = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        if (is_all_periodic_)
        {
            // Copy interleaved double to complex, then use cuFFT
            cuDoubleComplex* d_complex = reinterpret_cast<cuDoubleComplex*>(d_work_buffer_);

            ker_interleaved_to_complex<<<N_BLOCKS, N_THREADS, 0, stream>>>(
                d_complex, d_cdata, total_complex_grid_);
            gpu_error_check(cudaPeekAtLastError());

            if constexpr (std::is_same<T, double>::value)
            {
                cufftSetStream(plan_backward_, stream);
                cufftExecZ2D(plan_backward_, d_complex, d_rdata);

                // Normalize
                ker_scale_fft<<<N_BLOCKS, N_THREADS, 0, stream>>>(
                    d_rdata, 1.0 / total_grid_, total_grid_);
            }
            else
            {
                cufftSetStream(plan_backward_, stream);
                cufftExecZ2Z(plan_backward_, d_complex,
                             reinterpret_cast<cuDoubleComplex*>(d_rdata), CUFFT_INVERSE);

                // Normalize (scale both real and imaginary parts)
                ker_scale_fft<<<N_BLOCKS, N_THREADS, 0, stream>>>(
                    reinterpret_cast<double*>(d_rdata), 1.0 / total_grid_, total_grid_ * 2);
            }
            gpu_error_check(cudaPeekAtLastError());
        }
        else
        {
            // Non-periodic: apply inverse DCT/DST dimension by dimension
            ker_copy_fft_data<<<N_BLOCKS, N_THREADS, 0, stream>>>(d_work_buffer_, d_cdata, total_complex_grid_);
            gpu_error_check(cudaPeekAtLastError());

            for (int dim = DIM - 1; dim >= 0; --dim)
            {
                if (bc_[dim] == BoundaryCondition::REFLECTING)
                    applyDCT3Backward(d_work_buffer_, dim, stream);
                else if (bc_[dim] == BoundaryCondition::ABSORBING)
                    applyDST3Backward(d_work_buffer_, dim, stream);
            }

            if constexpr (std::is_same<T, double>::value)
            {
                ker_copy_fft_data<<<N_BLOCKS, N_THREADS, 0, stream>>>(d_rdata, d_work_buffer_, total_grid_);
            }
            else
            {
                ker_real_to_complex_fft<<<N_BLOCKS, N_THREADS, 0, stream>>>(
                    reinterpret_cast<cuDoubleComplex*>(d_rdata), d_work_buffer_, total_grid_);
            }
            gpu_error_check(cudaPeekAtLastError());
        }
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

//------------------------------------------------------------------------------
// Forward transform (complex* interface - periodic only)
//------------------------------------------------------------------------------
template <typename T, int DIM>
void CudaFFT<T, DIM>::forward_stream(T* d_rdata, std::complex<double>* d_cdata, cudaStream_t stream)
{
    try
    {
        if (!is_all_periodic_)
        {
            throw_with_line_number("Complex interface only available for periodic boundary conditions.");
        }

        cuDoubleComplex* d_cdata_cu = reinterpret_cast<cuDoubleComplex*>(d_cdata);

        if constexpr (std::is_same<T, double>::value)
        {
            cufftSetStream(plan_forward_, stream);
            cufftExecD2Z(plan_forward_, d_rdata, d_cdata_cu);
        }
        else
        {
            cufftSetStream(plan_forward_, stream);
            cufftExecZ2Z(plan_forward_, reinterpret_cast<cuDoubleComplex*>(d_rdata),
                         d_cdata_cu, CUFFT_FORWARD);
        }
        gpu_error_check(cudaPeekAtLastError());
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

//------------------------------------------------------------------------------
// Backward transform (complex* interface - periodic only)
//------------------------------------------------------------------------------
template <typename T, int DIM>
void CudaFFT<T, DIM>::backward_stream(std::complex<double>* d_cdata, T* d_rdata, cudaStream_t stream)
{
    try
    {
        if (!is_all_periodic_)
        {
            throw_with_line_number("Complex interface only available for periodic boundary conditions.");
        }

        const int N_BLOCKS = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        cuDoubleComplex* d_cdata_cu = reinterpret_cast<cuDoubleComplex*>(d_cdata);

        if constexpr (std::is_same<T, double>::value)
        {
            cufftSetStream(plan_backward_, stream);
            cufftExecZ2D(plan_backward_, d_cdata_cu, d_rdata);

            // Normalize
            ker_scale_fft<<<N_BLOCKS, N_THREADS, 0, stream>>>(
                d_rdata, 1.0 / total_grid_, total_grid_);
        }
        else
        {
            cufftSetStream(plan_backward_, stream);
            cufftExecZ2Z(plan_backward_, d_cdata_cu,
                         reinterpret_cast<cuDoubleComplex*>(d_rdata), CUFFT_INVERSE);

            // Normalize
            ker_scale_fft<<<N_BLOCKS, N_THREADS, 0, stream>>>(
                reinterpret_cast<double*>(d_rdata), 1.0 / total_grid_, total_grid_ * 2);
        }
        gpu_error_check(cudaPeekAtLastError());
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

// Explicit template instantiations
template class CudaFFT<double, 1>;
template class CudaFFT<double, 2>;
template class CudaFFT<double, 3>;
template class CudaFFT<std::complex<double>, 1>;
template class CudaFFT<std::complex<double>, 2>;
template class CudaFFT<std::complex<double>, 3>;
