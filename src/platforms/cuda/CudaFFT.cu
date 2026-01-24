/**
 * @file CudaFFT.cu
 * @brief CUDA implementation of spectral transforms for all boundary conditions.
 *
 * Implements:
 * - cuFFT for periodic boundary conditions
 * - DCT-II/III for reflecting (Neumann) BCs via FFT (O(N log N))
 * - DST-II/III for absorbing (Dirichlet) BCs via FFT (O(N log N))
 *
 * **DCT/DST via FFT Algorithm:**
 *
 * Based on cuHelmholtz library:
 * M. Ren, Y. Gao, G. Wang, and X. Liu, "Discrete Sine and Cosine Transform
 * and Helmholtz Equation Solver on GPU," 2020 IEEE Intl. Conf. on Parallel &
 * Distributed Processing with Applications (ISPA), pp. 57-66, 2020.
 * DOI: 10.1109/ISPA-BDCloud-SocialCom-SustainCom51426.2020.00034
 * GitHub: https://github.com/rmingming/cuHelmholtz
 *
 * DCT-II: preprocess → cuFFT Z2D (inverse) → postprocess (twiddle multiply)
 * DCT-III: preprocess (twiddle multiply) → cuFFT D2Z (forward) → postprocess
 * DST-II/III: Similar structure with different twiddle factors
 *
 * @see CudaFFT.h for class documentation
 */

#include <iostream>
#include <cmath>
#include <stdexcept>

#include "CudaFFT.h"
#include "CudaCommon.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

//------------------------------------------------------------------------------
// CUDA Kernels for copy and scale operations
//------------------------------------------------------------------------------

__global__ void ker_copy_fft_data(double* dst, const double* src, int M)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M)
        dst[idx] = src[idx];
}

__global__ void ker_complex_to_real_fft(double* dst, const cuDoubleComplex* src, int M)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M)
        dst[idx] = src[idx].x;
}

__global__ void ker_real_to_complex_fft(cuDoubleComplex* dst, const double* src, int M)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M)
    {
        dst[idx].x = src[idx];
        dst[idx].y = 0.0;
    }
}

__global__ void ker_complex_to_interleaved(double* dst, const cuDoubleComplex* src, int M_COMPLEX)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M_COMPLEX)
    {
        dst[2 * idx] = src[idx].x;
        dst[2 * idx + 1] = src[idx].y;
    }
}

__global__ void ker_interleaved_to_complex(cuDoubleComplex* dst, const double* src, int M_COMPLEX)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M_COMPLEX)
    {
        dst[idx].x = src[2 * idx];
        dst[idx].y = src[2 * idx + 1];
    }
}

__global__ void ker_scale_fft(double* data, double scale, int M)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M)
        data[idx] *= scale;
}

//------------------------------------------------------------------------------
// DCT-II via FFT: O(N log N) implementation (cuHelmholtz algorithm)
// Each block handles one 1D transform using shared memory
//------------------------------------------------------------------------------

/**
 * @brief DCT-II preprocessing kernel.
 *
 * Converts N real values to (N/2+1) complex values for inverse FFT.
 * Based on cuHelmholtz dct2funcinplace.cu preOp_dct2_inplace.
 *
 * Memory layout:
 * - Input src: N real values per transform (with stride between elements)
 * - Output dst: (N+2) values per transform (interleaved complex)
 */
__global__ void ker_dct2_preprocess(
    double* dst, const double* src,
    int N, int stride, int num_batches)
{
    extern __shared__ double sh[];

    int batch = blockIdx.x;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    if (batch >= num_batches) return;

    // Calculate strided input offset for this batch
    int outer_idx = batch / stride;
    int inner_idx = batch % stride;
    int in_base = outer_idx * N * stride + inner_idx;

    const double* pin = src + in_base;
    // Output is CONTIGUOUS per batch: 2*(N/2+1) values for each batch
    int complex_per_batch = N / 2 + 1;
    double* pout = dst + batch * (2 * complex_per_batch);

    // Load strided input into shared memory (loop for N > num_threads)
    for (int k = tid; k < N; k += num_threads)
    {
        sh[k] = pin[k * stride];
    }
    __syncthreads();

    // Rearrange into CONTIGUOUS complex format (loop for large N)
    for (int k = tid; k <= N / 2; k += num_threads)
    {
        if (k == 0)
        {
            pout[0] = sh[0];
            pout[1] = 0.0;
        }
        else if (k == N / 2 && (N % 2 == 0))  // Nyquist only for even N
        {
            pout[N] = sh[N - 1];
            pout[N + 1] = 0.0;
        }
        else
        {
            double x_2k = sh[2 * k];
            double x_2k_1 = sh[2 * k - 1];
            pout[2 * k] = (x_2k + x_2k_1) / 2.0;
            pout[2 * k + 1] = -(x_2k_1 - x_2k) / 2.0;
        }
    }
}

/**
 * @brief DCT-II postprocessing kernel.
 *
 * Applies twiddle factors after inverse FFT.
 * Based on cuHelmholtz dct2funcinplace.cu postOp_dct2_inplace.
 */
__global__ void ker_dct2_postprocess(
    double* dst, const double* src, int N, int stride, int num_batches)
{
    extern __shared__ double sh[];

    int batch = blockIdx.x;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    if (batch >= num_batches) return;

    // Input is CONTIGUOUS per batch (N values from FFT output)
    const double* pin = src + batch * N;

    // Output is STRIDED
    int outer_idx = batch / stride;
    int inner_idx = batch % stride;
    int out_base = outer_idx * N * stride + inner_idx;
    double* pout = dst + out_base;

    // Load contiguous FFT output into shared memory
    for (int k = tid; k < N; k += num_threads)
    {
        sh[k] = pin[k];
    }
    __syncthreads();

    // Apply twiddle factors and write to strided output
    for (int k = tid; k <= N / 2; k += num_threads)
    {
        if (k == 0)
        {
            pout[0] = sh[0];  // DC component unchanged
        }
        else
        {
            double sina, cosa;
            sincos(k * M_PI / (2.0 * N), &sina, &cosa);

            double Ta = sh[k] + sh[N - k];
            double Tb = sh[k] - sh[N - k];

            pout[k * stride] = (Ta * cosa + Tb * sina) / 2.0;
            pout[(N - k) * stride] = (Ta * sina - Tb * cosa) / 2.0;
        }
    }
}

//------------------------------------------------------------------------------
// DCT-III via FFT: O(N log N) implementation (cuHelmholtz algorithm)
//------------------------------------------------------------------------------

/**
 * @brief DCT-III preprocessing kernel.
 *
 * Applies twiddle factors before forward FFT.
 * Based on cuHelmholtz dct3funcinplace.cu preOp_dct3_inplace.
 */
__global__ void ker_dct3_preprocess(
    double* dst, const double* src, int N, int stride, int num_batches)
{
    extern __shared__ double sh[];

    int batch = blockIdx.x;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    if (batch >= num_batches) return;

    // Input is STRIDED
    int outer_idx = batch / stride;
    int inner_idx = batch % stride;
    int in_base = outer_idx * N * stride + inner_idx;
    const double* pin = src + in_base;

    // Output is CONTIGUOUS per batch (N values for FFT input)
    double* pout = dst + batch * N;

    // Load strided input into shared memory
    for (int k = tid; k < N; k += num_threads)
    {
        sh[k] = pin[k * stride];
    }
    __syncthreads();

    // Copy DC component (unchanged) - only thread 0
    if (tid == 0)
    {
        pout[0] = sh[0];
    }

    // Apply twiddle factors for pairs (k+1, N-k-1) where k = 0 to N/2-1
    for (int k = tid; k < N / 2; k += num_threads)
    {
        double sina, cosa;
        sincos((k + 1) * M_PI / (2.0 * N), &sina, &cosa);

        double val_k = sh[k + 1];
        double val_nk = sh[N - k - 1];

        double Ta = val_k + val_nk;
        double Tb = val_k - val_nk;

        pout[k + 1] = Ta * sina + Tb * cosa;
        pout[N - k - 1] = Ta * cosa - Tb * sina;
    }
}

/**
 * @brief DCT-III postprocessing kernel.
 *
 * Rearranges output after forward FFT.
 * Based on cuHelmholtz dct3funcinplace.cu postOp_dct3_inplace.
 */
__global__ void ker_dct3_postprocess(
    double* dst, const double* src,
    int N, int stride, int num_batches)
{
    extern __shared__ double sh[];

    int batch = blockIdx.x;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    if (batch >= num_batches) return;

    // Input is CONTIGUOUS per batch: 2*(N/2+1) values from FFT output
    int complex_per_batch = N / 2 + 1;
    const double* psrc = src + batch * (2 * complex_per_batch);

    // Output is STRIDED
    int outer_idx = batch / stride;
    int inner_idx = batch % stride;
    int out_base = outer_idx * N * stride + inner_idx;
    double* pdst = dst + out_base;

    // Load contiguous complex FFT output into shared memory
    for (int k = tid; k <= N / 2; k += num_threads)
    {
        sh[2 * k] = psrc[2 * k];
        sh[2 * k + 1] = psrc[2 * k + 1];
    }
    __syncthreads();

    // Rearrange to strided real output (cuHelmholtz formula)
    for (int k = tid; k <= N / 2; k += num_threads)
    {
        if (k == 0)
        {
            pdst[0] = sh[0] / 2.0;
        }
        else
        {
            double re = sh[2 * k];
            double im = sh[2 * k + 1];

            pdst[(2 * k - 1) * stride] = (re - im) / 2.0;
            if (2 * k < N)
            {
                pdst[(2 * k) * stride] = (re + im) / 2.0;
            }
        }
    }
}

//------------------------------------------------------------------------------
// DST via DCT transformation kernels
// DST-II(x) can be computed via DCT-II by:
//   1. Transform input: y[n] = (-1)^n * x[N-1-n]
//   2. Apply DCT-II to get Z
//   3. Transform output: DST-II(x)[k] = Z[N-1-k]
// DST-III is similarly computed via DCT-III.
//------------------------------------------------------------------------------

/**
 * @brief Transform input for DST-II via DCT-II.
 * y[n] = (-1)^n * x[N-1-n]
 * Reads from STRIDED input, writes to CONTIGUOUS output (per batch).
 */
__global__ void ker_dst2_input_transform(
    double* dst, const double* src,
    int N, int stride, int num_batches)
{
    int batch = blockIdx.x;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    if (batch >= num_batches) return;

    // Input is STRIDED
    int outer_idx = batch / stride;
    int inner_idx = batch % stride;
    int in_base = outer_idx * N * stride + inner_idx;
    const double* pin = src + in_base;

    // Output is CONTIGUOUS per batch
    double* pout = dst + batch * N;

    for (int n = tid; n < N; n += num_threads)
    {
        double sign = (n % 2 == 0) ? 1.0 : -1.0;
        pout[n] = sign * pin[(N - 1 - n) * stride];
    }
}

/**
 * @brief Transform DCT-II output to DST-II output.
 * Simply reverse: DST-II(x)[k] = DCT-II(y)[N-1-k]
 * Reads from CONTIGUOUS input (per batch), writes to STRIDED output.
 */
__global__ void ker_dst2_output_transform(
    double* dst, const double* src,
    int N, int stride, int num_batches)
{
    int batch = blockIdx.x;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    if (batch >= num_batches) return;

    // Input is CONTIGUOUS per batch
    const double* pin = src + batch * N;

    // Output is STRIDED
    int outer_idx = batch / stride;
    int inner_idx = batch % stride;
    int out_base = outer_idx * N * stride + inner_idx;
    double* pout = dst + out_base;

    for (int k = tid; k < N; k += num_threads)
    {
        pout[k * stride] = pin[N - 1 - k];
    }
}

/**
 * @brief Transform input for DST-III via DCT-III.
 * Simply reverse: y[n] = x[N-1-n]
 * Reads from STRIDED input, writes to CONTIGUOUS output (per batch).
 */
__global__ void ker_dst3_input_transform(
    double* dst, const double* src,
    int N, int stride, int num_batches)
{
    int batch = blockIdx.x;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    if (batch >= num_batches) return;

    // Input is STRIDED
    int outer_idx = batch / stride;
    int inner_idx = batch % stride;
    int in_base = outer_idx * N * stride + inner_idx;
    const double* pin = src + in_base;

    // Output is CONTIGUOUS per batch
    double* pout = dst + batch * N;

    for (int n = tid; n < N; n += num_threads)
    {
        pout[n] = pin[(N - 1 - n) * stride];
    }
}

/**
 * @brief Transform DCT-III output to DST-III output.
 * y[n] = -(-1)^n * dct_out[N-1-n] = (n%2==0 ? -1 : 1) * dct_out[N-1-n]
 * Reads from CONTIGUOUS input (per batch), writes to STRIDED output.
 */
__global__ void ker_dst3_output_transform(
    double* dst, const double* src,
    int N, int stride, int num_batches)
{
    int batch = blockIdx.x;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    if (batch >= num_batches) return;

    // Input is CONTIGUOUS per batch
    const double* pin = src + batch * N;

    // Output is STRIDED
    int outer_idx = batch / stride;
    int inner_idx = batch % stride;
    int out_base = outer_idx * N * stride + inner_idx;
    double* pout = dst + out_base;

    for (int n = tid; n < N; n += num_threads)
    {
        double sign = (n % 2 == 0) ? -1.0 : 1.0;
        pout[n * stride] = sign * pin[N - 1 - n];
    }
}

//------------------------------------------------------------------------------
// Constructor (all periodic - backward compatible)
//------------------------------------------------------------------------------
template <typename T, int DIM>
CudaFFT<T, DIM>::CudaFFT(std::array<int, DIM> nx)
    : nx_(nx), d_work_buffer_(nullptr), d_temp_buffer_(nullptr), d_fft_buffer_(nullptr), is_all_periodic_(true)
{
    for (int d = 0; d < DIM; ++d)
        bc_[d] = BoundaryCondition::PERIODIC;

    try
    {
        total_grid_ = 1;
        for (int d = 0; d < DIM; ++d)
            total_grid_ *= nx_[d];

        if (DIM == 3)
            total_complex_grid_ = nx_[0] * nx_[1] * (nx_[2] / 2 + 1);
        else if (DIM == 2)
            total_complex_grid_ = nx_[0] * (nx_[1] / 2 + 1);
        else if (DIM == 1)
            total_complex_grid_ = nx_[0] / 2 + 1;

        gpu_error_check(cudaMalloc((void**)&d_work_buffer_, sizeof(cuDoubleComplex) * total_complex_grid_));
        gpu_error_check(cudaMalloc((void**)&d_temp_buffer_, sizeof(double) * total_grid_ * 2));

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
    : nx_(nx), bc_(bc), d_work_buffer_(nullptr), d_temp_buffer_(nullptr), d_fft_buffer_(nullptr),
      plan_forward_(0), plan_backward_(0)
{
    try
    {
        total_grid_ = 1;
        for (int d = 0; d < DIM; ++d)
            total_grid_ *= nx_[d];

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
        // For non-periodic, we need extra space for FFT (N+2 per dimension)
        size_t work_size = is_all_periodic_
            ? sizeof(cuDoubleComplex) * total_complex_grid_
            : sizeof(double) * total_grid_ * 2;  // Extra space for FFT

        gpu_error_check(cudaMalloc((void**)&d_work_buffer_, work_size));
        // For DST via DCT, we need extra buffer space for intermediate results
        gpu_error_check(cudaMalloc((void**)&d_temp_buffer_, sizeof(double) * total_grid_ * 4));
        // cuFFT requires input and output to be in SEPARATE allocations (not just different offsets)
        // d_fft_buffer_ is used for FFT output, separate from d_temp_buffer_
        gpu_error_check(cudaMalloc((void**)&d_fft_buffer_, sizeof(double) * total_grid_ * 2));

        d_sin_tables_.resize(DIM, nullptr);
        d_cos_tables_.resize(DIM, nullptr);

        if (is_all_periodic_)
        {
            initPeriodicFFT();
        }
        // For non-periodic, we don't need precomputed tables anymore
        // since we use FFT-based O(N log N) algorithm
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
    if (d_fft_buffer_ != nullptr)
        cudaFree(d_fft_buffer_);

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

    cufftPlanMany(&plan_forward_, DIM, total_grid, nullptr, 1, 0, nullptr, 1, 0, cufft_forward, 1);
    cufftPlanMany(&plan_backward_, DIM, total_grid, nullptr, 1, 0, nullptr, 1, 0, cufft_backward, 1);
}

//------------------------------------------------------------------------------
// Precompute trig tables (kept for backward compatibility, not used in new impl)
//------------------------------------------------------------------------------
template <typename T, int DIM>
void CudaFFT<T, DIM>::precomputeTrigTables()
{
    // This function is kept for backward compatibility
    // The new O(N log N) implementation computes twiddle factors on-the-fly
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
// Apply DCT-II forward transform for one dimension (O(N log N))
// Uses cuHelmholtz algorithm: preprocess → Z2D → postprocess
// Data flow: strided input → contiguous complex → FFT → contiguous real → strided output
//------------------------------------------------------------------------------
template <typename T, int DIM>
void CudaFFT<T, DIM>::applyDCT2Forward(double* d_data, int dim, cudaStream_t stream)
{
    int N = nx_[dim];
    int stride, num_outer;
    getStrides(dim, stride, num_outer);

    int num_batches = num_outer * stride;  // Total number of 1D transforms
    int threads = 256;  // Fixed thread count, kernel loops handle larger N
    size_t shmem = sizeof(double) * N;

    // Number of complex values per batch (floor(N/2)+1)
    int complex_per_batch = N / 2 + 1;

    // Use SEPARATE buffer for FFT output (cuFFT requires separate allocations)
    double* d_fft_out = d_fft_buffer_;

    // Step 1: Preprocess - strided input → contiguous complex
    ker_dct2_preprocess<<<num_batches, threads, shmem, stream>>>(
        d_temp_buffer_, d_data, N, stride, num_batches);
    gpu_error_check(cudaPeekAtLastError());

    // Step 2: Inverse FFT (Z2D) - contiguous batches
    cufftHandle plan;
    int n_arr[1] = {N};

    // Input: complex_per_batch complex values per batch, contiguous batches
    // Output: N real values per batch, contiguous batches
    cufftPlanMany(&plan, 1, n_arr,
                  nullptr, 1, complex_per_batch,  // inembed=nullptr, istride=1, idist
                  nullptr, 1, N,                   // onembed=nullptr, ostride=1, odist=N
                  CUFFT_Z2D, num_batches);

    cufftSetStream(plan, stream);
    cufftExecZ2D(plan, reinterpret_cast<cufftDoubleComplex*>(d_temp_buffer_), d_fft_out);
    cufftDestroy(plan);
    gpu_error_check(cudaPeekAtLastError());

    // Step 3: Postprocess - contiguous real → strided output with twiddles
    ker_dct2_postprocess<<<num_batches, threads, shmem, stream>>>(
        d_data, d_fft_out, N, stride, num_batches);
    gpu_error_check(cudaPeekAtLastError());
}

//------------------------------------------------------------------------------
// Apply DCT-III backward transform for one dimension (O(N log N))
// Uses cuHelmholtz algorithm: preprocess → D2Z → postprocess
// Data flow: strided input → contiguous real (with twiddles) → FFT → contiguous complex → strided output
//------------------------------------------------------------------------------
template <typename T, int DIM>
void CudaFFT<T, DIM>::applyDCT3Backward(double* d_data, int dim, cudaStream_t stream)
{
    int N = nx_[dim];
    int stride, num_outer;
    getStrides(dim, stride, num_outer);

    int num_batches = num_outer * stride;
    int threads_pre = 256;  // Fixed thread count
    int threads_post = 256;
    int complex_per_batch = N / 2 + 1;
    size_t shmem_pre = sizeof(double) * N;
    size_t shmem_post = sizeof(double) * (2 * complex_per_batch);

    // Use SEPARATE buffer for FFT output (cuFFT requires separate allocations)
    double* d_fft_out = d_fft_buffer_;

    // Step 1: Preprocess - strided input → contiguous real with twiddles
    ker_dct3_preprocess<<<num_batches, threads_pre, shmem_pre, stream>>>(
        d_temp_buffer_, d_data, N, stride, num_batches);
    gpu_error_check(cudaPeekAtLastError());

    // Step 2: Forward FFT (D2Z) - contiguous batches
    cufftHandle plan;
    int n_arr[1] = {N};

    // Input: N real values per batch, contiguous batches
    // Output: complex_per_batch complex values per batch, contiguous batches
    cufftPlanMany(&plan, 1, n_arr,
                  nullptr, 1, N,                   // inembed=nullptr, istride=1, idist=N
                  nullptr, 1, complex_per_batch,   // onembed=nullptr, ostride=1, odist
                  CUFFT_D2Z, num_batches);

    cufftSetStream(plan, stream);
    cufftExecD2Z(plan, d_temp_buffer_, reinterpret_cast<cufftDoubleComplex*>(d_fft_out));
    cufftDestroy(plan);
    gpu_error_check(cudaPeekAtLastError());

    // Step 3: Postprocess - contiguous complex → strided output
    ker_dct3_postprocess<<<num_batches, threads_post, shmem_post, stream>>>(
        d_data, d_fft_out, N, stride, num_batches);
    gpu_error_check(cudaPeekAtLastError());
}

//------------------------------------------------------------------------------
// Apply DST-II forward transform for one dimension (O(N log N))
// Uses DST via DCT transformation:
//   1. Transform input: y[n] = (-1)^n * x[N-1-n]
//   2. Apply DCT-II
//   3. Transform output: reverse the result
//------------------------------------------------------------------------------
template <typename T, int DIM>
void CudaFFT<T, DIM>::applyDST2Forward(double* d_data, int dim, cudaStream_t stream)
{
    int N = nx_[dim];
    int stride, num_outer;
    getStrides(dim, stride, num_outer);

    int num_batches = num_outer * stride;
    int threads = 256;  // Fixed thread count, kernel loops handle larger N
    size_t shmem = sizeof(double) * N;

    // Number of complex values per batch (consistent with DCT kernels)
    int complex_per_batch = N / 2 + 1;

    // Buffer layout:
    // d_temp_buffer_: transformed input then complex FFT input (reused)
    // d_fft_buffer_: FFT output (SEPARATE allocation required by cuFFT)
    double* d_complex_buffer = d_temp_buffer_ + num_batches * N;
    double* d_fft_out = d_fft_buffer_;  // Use separate buffer for FFT output

    // Step 1: Transform input for DST-II: y[n] = (-1)^n * x[N-1-n]
    // This also copies from strided to contiguous layout
    ker_dst2_input_transform<<<num_batches, threads, 0, stream>>>(
        d_temp_buffer_, d_data, N, stride, num_batches);
    gpu_error_check(cudaPeekAtLastError());

    // Step 2: Apply DCT-II to transformed input
    // DCT-II preprocess: contiguous input → contiguous complex
    // Note: ker_dct2_preprocess expects strided input, but after DST input transform
    // our data is contiguous per batch. We use stride=1 for the contiguous layout.
    ker_dct2_preprocess<<<num_batches, threads, shmem, stream>>>(
        d_complex_buffer, d_temp_buffer_, N, 1, num_batches);
    gpu_error_check(cudaPeekAtLastError());

    // DCT-II FFT (Z2D) - contiguous batches
    cufftHandle plan;
    int n_arr[1] = {N};

    cufftPlanMany(&plan, 1, n_arr,
                  nullptr, 1, complex_per_batch,  // idist: complex values per batch
                  nullptr, 1, N,                   // odist: real values per batch
                  CUFFT_Z2D, num_batches);

    cufftSetStream(plan, stream);
    cufftExecZ2D(plan, reinterpret_cast<cufftDoubleComplex*>(d_complex_buffer), d_fft_out);
    cufftDestroy(plan);
    gpu_error_check(cudaPeekAtLastError());

    // DCT-II postprocess: contiguous FFT output → contiguous result with twiddles
    // Using stride=1 for contiguous layout
    ker_dct2_postprocess<<<num_batches, threads, shmem, stream>>>(
        d_temp_buffer_, d_fft_out, N, 1, num_batches);
    gpu_error_check(cudaPeekAtLastError());

    // Step 3: Transform output: reverse to get DST-II result
    // This also copies from contiguous to strided layout
    ker_dst2_output_transform<<<num_batches, threads, 0, stream>>>(
        d_data, d_temp_buffer_, N, stride, num_batches);
    gpu_error_check(cudaPeekAtLastError());
}

//------------------------------------------------------------------------------
// Apply DST-III backward transform for one dimension (O(N log N))
// Uses DST via DCT transformation:
//   1. Transform input: reverse the input
//   2. Apply DCT-III
//   3. Transform output: y[n] = -(-1)^n * dct_out[N-1-n]
//------------------------------------------------------------------------------
template <typename T, int DIM>
void CudaFFT<T, DIM>::applyDST3Backward(double* d_data, int dim, cudaStream_t stream)
{
    int N = nx_[dim];
    int stride, num_outer;
    getStrides(dim, stride, num_outer);

    int num_batches = num_outer * stride;
    int threads_transform = 256;  // Fixed thread count, kernel loops handle larger N
    int threads_dct_pre = 256;
    int threads_dct_post = 256;
    int complex_per_batch = N / 2 + 1;
    size_t shmem_pre = sizeof(double) * N;
    size_t shmem_post = sizeof(double) * (2 * complex_per_batch);

    // Buffer layout:
    // d_temp_buffer_: reversed input then DCT-III preprocess output (reused)
    // d_fft_buffer_: FFT output (SEPARATE allocation required by cuFFT)
    double* d_dct_buffer = d_temp_buffer_ + num_batches * N;
    double* d_fft_out = d_fft_buffer_;  // Use separate buffer for FFT output

    // Step 1: Transform input for DST-III: reverse
    // This also copies from strided to contiguous layout
    ker_dst3_input_transform<<<num_batches, threads_transform, 0, stream>>>(
        d_temp_buffer_, d_data, N, stride, num_batches);
    gpu_error_check(cudaPeekAtLastError());

    // Step 2: Apply DCT-III to transformed input
    // DCT-III preprocess: contiguous input → contiguous output with twiddles
    // Use stride=1 for contiguous layout
    ker_dct3_preprocess<<<num_batches, threads_dct_pre, shmem_pre, stream>>>(
        d_dct_buffer, d_temp_buffer_, N, 1, num_batches);
    gpu_error_check(cudaPeekAtLastError());

    // DCT-III FFT (D2Z) - contiguous batches
    cufftHandle plan;
    int n_arr[1] = {N};

    cufftPlanMany(&plan, 1, n_arr,
                  nullptr, 1, N,                   // idist: real values per batch
                  nullptr, 1, complex_per_batch,   // odist: complex values per batch
                  CUFFT_D2Z, num_batches);

    cufftSetStream(plan, stream);
    cufftExecD2Z(plan, d_dct_buffer, reinterpret_cast<cufftDoubleComplex*>(d_fft_out));
    cufftDestroy(plan);
    gpu_error_check(cudaPeekAtLastError());

    // DCT-III postprocess: contiguous FFT output → contiguous result
    // Use stride=1 for contiguous layout
    ker_dct3_postprocess<<<num_batches, threads_dct_post, shmem_post, stream>>>(
        d_temp_buffer_, d_fft_out, N, 1, num_batches);
    gpu_error_check(cudaPeekAtLastError());

    // Step 3: Transform output: y[n] = -(-1)^n * dct_out[N-1-n]
    // This also copies from contiguous to strided layout
    ker_dst3_output_transform<<<num_batches, threads_transform, 0, stream>>>(
        d_data, d_temp_buffer_, N, stride, num_batches);
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
            cuDoubleComplex* d_complex = reinterpret_cast<cuDoubleComplex*>(d_work_buffer_);

            ker_interleaved_to_complex<<<N_BLOCKS, N_THREADS, 0, stream>>>(
                d_complex, d_cdata, total_complex_grid_);
            gpu_error_check(cudaPeekAtLastError());

            if constexpr (std::is_same<T, double>::value)
            {
                cufftSetStream(plan_backward_, stream);
                cufftExecZ2D(plan_backward_, d_complex, d_rdata);

                ker_scale_fft<<<N_BLOCKS, N_THREADS, 0, stream>>>(
                    d_rdata, 1.0 / total_grid_, total_grid_);
            }
            else
            {
                cufftSetStream(plan_backward_, stream);
                cufftExecZ2Z(plan_backward_, d_complex,
                             reinterpret_cast<cuDoubleComplex*>(d_rdata), CUFFT_INVERSE);

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

            // Normalize: DCT/DST round-trip scaling is N/2 per dimension
            // Need to divide by (N/2)^DIM = product of (nx_[d]/2)
            double scale = 1.0;
            for (int d = 0; d < DIM; ++d)
                scale *= 2.0 / nx_[d];

            if constexpr (std::is_same<T, double>::value)
            {
                ker_copy_fft_data<<<N_BLOCKS, N_THREADS, 0, stream>>>(d_rdata, d_work_buffer_, total_grid_);
                ker_scale_fft<<<N_BLOCKS, N_THREADS, 0, stream>>>(d_rdata, scale, total_grid_);
            }
            else
            {
                ker_real_to_complex_fft<<<N_BLOCKS, N_THREADS, 0, stream>>>(
                    reinterpret_cast<cuDoubleComplex*>(d_rdata), d_work_buffer_, total_grid_);
                ker_scale_fft<<<N_BLOCKS, N_THREADS, 0, stream>>>(
                    reinterpret_cast<double*>(d_rdata), scale, total_grid_ * 2);
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

            ker_scale_fft<<<N_BLOCKS, N_THREADS, 0, stream>>>(
                d_rdata, 1.0 / total_grid_, total_grid_);
        }
        else
        {
            cufftSetStream(plan_backward_, stream);
            cufftExecZ2Z(plan_backward_, d_cdata_cu,
                         reinterpret_cast<cuDoubleComplex*>(d_rdata), CUFFT_INVERSE);

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
