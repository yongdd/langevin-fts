/**
 * @file CudaFFTMixedBC.cu
 * @brief CUDA implementation of DCT/DST transforms for mixed boundary conditions.
 *
 * Implements DCT-II/III and DST-II/III transforms using custom CUDA kernels.
 * These transforms are used for reflecting (Neumann) and absorbing (Dirichlet)
 * boundary conditions in pseudo-spectral methods.
 *
 * **Transform Definitions:**
 *
 * DCT-II (forward): X[k] = sum_{n=0}^{N-1} x[n] * cos(π*k*(n+0.5)/N)
 * DCT-III (inverse): x[n] = X[0]/N + (2/N) * sum_{k=1}^{N-1} X[k] * cos(π*k*(n+0.5)/N)
 *
 * DST-II (forward): X[k] = sum_{n=0}^{N-1} x[n] * sin(π*(k+1)*(n+0.5)/N)
 * DST-III (inverse): x[n] = (1/N) * [(-1)^n * X[N-1] + 2*sum_{k=0}^{N-2} X[k]*sin(π*(2n+1)*(k+1)/(2N))]
 *
 * @see CudaFFTMixedBC.h for class documentation
 */

#include <iostream>
#include <cmath>
#include <stdexcept>

#include "CudaFFTMixedBC.h"
#include "CudaCommon.h"

//------------------------------------------------------------------------------
// CUDA Kernels for DCT/DST transforms
//------------------------------------------------------------------------------

/**
 * @brief DCT-II forward transform kernel.
 *
 * Each thread computes one output coefficient X[k].
 */
__global__ void ker_dct2_forward(
    double* dst, const double* src, const double* cos_table,
    int n, int stride, int num_transforms, int total_grid)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_transforms * stride * n) return;

    // Decode which transform and which coefficient
    int batch = idx / (n * stride);
    int remainder = idx % (n * stride);
    int k = remainder / stride;
    int s = remainder % stride;

    int offset = batch * n * stride + s;

    // DCT-II: X[k] = sum_j x[j] * cos(π*k*(j+0.5)/n)
    double sum = 0.0;
    for (int j = 0; j < n; ++j)
    {
        sum += src[offset + j * stride] * cos_table[k * n + j];
    }
    dst[offset + k * stride] = sum;
}

/**
 * @brief DCT-III backward transform kernel.
 *
 * Each thread computes one output value x[j].
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

    // DCT-III: x[j] = X[0]/n + (2/n) * sum_{k=1}^{n-1} X[k] * cos(π*k*(j+0.5)/n)
    double sum = src[offset] / n;  // k=0 term
    for (int k = 1; k < n; ++k)
    {
        sum += (2.0 / n) * src[offset + k * stride] * cos_table[k * n + j];
    }
    dst[offset + j * stride] = sum;
}

/**
 * @brief DST-II forward transform kernel.
 *
 * Each thread computes one output coefficient X[k].
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

    // DST-II: X[k] = sum_j x[j] * sin(π*(k+1)*(j+0.5)/n)
    double sum = 0.0;
    for (int j = 0; j < n; ++j)
    {
        sum += src[offset + j * stride] * sin_table[k * n + j];
    }
    dst[offset + k * stride] = sum;
}

/**
 * @brief DST-III backward transform kernel.
 *
 * Each thread computes one output value x[j].
 * DST-III: x[j] = (1/N) * [(-1)^j * X[N-1] + 2*sum_{k=0}^{N-2} X[k]*sin(π*(2j+1)*(k+1)/(2N))]
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

    // Last term: (-1)^j * X[N-1]
    double sign = (j % 2 == 0) ? 1.0 : -1.0;
    double sum = sign * src[offset + (n - 1) * stride];

    // Sum over k=0 to n-2
    for (int k = 0; k < n - 1; ++k)
    {
        sum += 2.0 * src[offset + k * stride] * sin(PI * (2 * j + 1) * (k + 1) / (2.0 * n));
    }

    dst[offset + j * stride] = sum / n;
}

/**
 * @brief Copy data kernel.
 */
__global__ void ker_copy_data(double* dst, const double* src, int M)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M)
        dst[idx] = src[idx];
}

/**
 * @brief Copy from complex to real (extract real part).
 */
__global__ void ker_complex_to_real(double* dst, const cuDoubleComplex* src, int M)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M)
        dst[idx] = src[idx].x;
}

/**
 * @brief Copy from real to complex (imaginary = 0).
 */
__global__ void ker_real_to_complex(cuDoubleComplex* dst, const double* src, int M)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M)
    {
        dst[idx].x = src[idx];
        dst[idx].y = 0.0;
    }
}

//------------------------------------------------------------------------------
// Constructor
//------------------------------------------------------------------------------
template <typename T, int DIM>
CudaFFTMixedBC<T, DIM>::CudaFFTMixedBC(std::array<int, DIM> nx, std::array<BoundaryCondition, DIM> bc)
    : nx_(nx), bc_(bc), d_work_buffer_(nullptr), d_temp_buffer_(nullptr),
      has_periodic_dim_(false), periodic_dim_idx_(-1)
{
    try
    {
        // Compute total grid sizes
        total_grid_ = 1;
        for (int d = 0; d < DIM; ++d)
            total_grid_ *= nx_[d];

        // Check for periodic dimensions
        for (int d = 0; d < DIM; ++d)
        {
            if (bc_[d] == BoundaryCondition::PERIODIC)
            {
                has_periodic_dim_ = true;
                periodic_dim_idx_ = d;
                break;
            }
        }

        // For simplicity, require either all periodic or all non-periodic
        if (has_periodic_dim_)
        {
            for (int d = 0; d < DIM; ++d)
            {
                if (bc_[d] != BoundaryCondition::PERIODIC)
                {
                    throw_with_line_number("Mixed periodic and non-periodic BCs not yet supported. "
                                          "Use all periodic or all non-periodic (reflecting/absorbing).");
                }
            }
        }

        // For DCT/DST, complex grid size equals real grid size
        if (has_periodic_dim_)
        {
            // Standard r2c FFT: last dimension halved
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

        // Allocate work buffers on GPU
        gpu_error_check(cudaMalloc((void**)&d_work_buffer_, sizeof(double) * total_grid_));
        gpu_error_check(cudaMalloc((void**)&d_temp_buffer_, sizeof(double) * total_grid_));

        // Initialize trig table pointers
        d_sin_tables_.resize(DIM, nullptr);
        d_cos_tables_.resize(DIM, nullptr);

        // Precompute and upload trig tables
        if (!has_periodic_dim_)
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
CudaFFTMixedBC<T, DIM>::~CudaFFTMixedBC()
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
}

//------------------------------------------------------------------------------
// Precompute trig tables and upload to GPU
//------------------------------------------------------------------------------
template <typename T, int DIM>
void CudaFFTMixedBC<T, DIM>::precomputeTrigTables()
{
    const double PI = 3.14159265358979323846;

    for (int dim = 0; dim < DIM; ++dim)
    {
        int n = nx_[dim];

        if (bc_[dim] == BoundaryCondition::REFLECTING)
        {
            // DCT-II: cos(π*k*(j+0.5)/n) for k=0..n-1, j=0..n-1
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
            // DST-II: sin(π*(k+1)*(j+0.5)/n) for k=0..n-1, j=0..n-1
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
void CudaFFTMixedBC<T, DIM>::getStrides(int dim, int& stride, int& num_transforms) const
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
void CudaFFTMixedBC<T, DIM>::applyDCT2Forward(double* d_data, int dim, cudaStream_t stream)
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

    // Copy back
    ker_copy_data<<<N_BLOCKS, N_THREADS, 0, stream>>>(d_data, d_temp_buffer_, total_grid_);
    gpu_error_check(cudaPeekAtLastError());
}

//------------------------------------------------------------------------------
// Apply DCT-III backward transform for one dimension
//------------------------------------------------------------------------------
template <typename T, int DIM>
void CudaFFTMixedBC<T, DIM>::applyDCT3Backward(double* d_data, int dim, cudaStream_t stream)
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

    // Copy back
    ker_copy_data<<<N_BLOCKS, N_THREADS, 0, stream>>>(d_data, d_temp_buffer_, total_grid_);
    gpu_error_check(cudaPeekAtLastError());
}

//------------------------------------------------------------------------------
// Apply DST-II forward transform for one dimension
//------------------------------------------------------------------------------
template <typename T, int DIM>
void CudaFFTMixedBC<T, DIM>::applyDST2Forward(double* d_data, int dim, cudaStream_t stream)
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

    // Copy back
    ker_copy_data<<<N_BLOCKS, N_THREADS, 0, stream>>>(d_data, d_temp_buffer_, total_grid_);
    gpu_error_check(cudaPeekAtLastError());
}

//------------------------------------------------------------------------------
// Apply DST-III backward transform for one dimension
//------------------------------------------------------------------------------
template <typename T, int DIM>
void CudaFFTMixedBC<T, DIM>::applyDST3Backward(double* d_data, int dim, cudaStream_t stream)
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

    // Copy back
    ker_copy_data<<<N_BLOCKS, N_THREADS, 0, stream>>>(d_data, d_temp_buffer_, total_grid_);
    gpu_error_check(cudaPeekAtLastError());
}

//------------------------------------------------------------------------------
// Check if all periodic
//------------------------------------------------------------------------------
template <typename T, int DIM>
bool CudaFFTMixedBC<T, DIM>::is_all_periodic() const
{
    for (int d = 0; d < DIM; ++d)
    {
        if (bc_[d] != BoundaryCondition::PERIODIC)
            return false;
    }
    return true;
}

//------------------------------------------------------------------------------
// Forward transform
//------------------------------------------------------------------------------
template <typename T, int DIM>
void CudaFFTMixedBC<T, DIM>::forward(T* d_rdata, double* d_cdata, cudaStream_t stream)
{
    try
    {
        const int N_BLOCKS = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        if (has_periodic_dim_)
        {
            throw_with_line_number("Forward transform for periodic BC should use standard CuFFT");
        }

        // Copy input to work buffer
        if constexpr (std::is_same<T, double>::value)
        {
            ker_copy_data<<<N_BLOCKS, N_THREADS, 0, stream>>>(d_work_buffer_, d_rdata, total_grid_);
        }
        else
        {
            ker_complex_to_real<<<N_BLOCKS, N_THREADS, 0, stream>>>(
                d_work_buffer_, reinterpret_cast<cuDoubleComplex*>(d_rdata), total_grid_);
        }
        gpu_error_check(cudaPeekAtLastError());

        // Apply transforms dimension by dimension
        for (int dim = 0; dim < DIM; ++dim)
        {
            if (bc_[dim] == BoundaryCondition::REFLECTING)
            {
                applyDCT2Forward(d_work_buffer_, dim, stream);
            }
            else if (bc_[dim] == BoundaryCondition::ABSORBING)
            {
                applyDST2Forward(d_work_buffer_, dim, stream);
            }
        }

        // Copy to output
        ker_copy_data<<<N_BLOCKS, N_THREADS, 0, stream>>>(d_cdata, d_work_buffer_, total_complex_grid_);
        gpu_error_check(cudaPeekAtLastError());
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

//------------------------------------------------------------------------------
// Backward transform
//------------------------------------------------------------------------------
template <typename T, int DIM>
void CudaFFTMixedBC<T, DIM>::backward(double* d_cdata, T* d_rdata, cudaStream_t stream)
{
    try
    {
        const int N_BLOCKS = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        if (has_periodic_dim_)
        {
            throw_with_line_number("Backward transform for periodic BC should use standard CuFFT");
        }

        // Copy input to work buffer
        ker_copy_data<<<N_BLOCKS, N_THREADS, 0, stream>>>(d_work_buffer_, d_cdata, total_complex_grid_);
        gpu_error_check(cudaPeekAtLastError());

        // Apply inverse transforms dimension by dimension (reverse order)
        for (int dim = DIM - 1; dim >= 0; --dim)
        {
            if (bc_[dim] == BoundaryCondition::REFLECTING)
            {
                applyDCT3Backward(d_work_buffer_, dim, stream);
            }
            else if (bc_[dim] == BoundaryCondition::ABSORBING)
            {
                applyDST3Backward(d_work_buffer_, dim, stream);
            }
        }

        // Copy to output
        if constexpr (std::is_same<T, double>::value)
        {
            ker_copy_data<<<N_BLOCKS, N_THREADS, 0, stream>>>(d_rdata, d_work_buffer_, total_grid_);
        }
        else
        {
            ker_real_to_complex<<<N_BLOCKS, N_THREADS, 0, stream>>>(
                reinterpret_cast<cuDoubleComplex*>(d_rdata), d_work_buffer_, total_grid_);
        }
        gpu_error_check(cudaPeekAtLastError());
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

// Explicit template instantiations
template class CudaFFTMixedBC<double, 1>;
template class CudaFFTMixedBC<double, 2>;
template class CudaFFTMixedBC<double, 3>;
template class CudaFFTMixedBC<std::complex<double>, 1>;
template class CudaFFTMixedBC<std::complex<double>, 2>;
template class CudaFFTMixedBC<std::complex<double>, 3>;
