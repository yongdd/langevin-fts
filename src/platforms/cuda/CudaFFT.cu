/**
 * @file CudaFFT.cu
 * @brief CUDA implementation of spectral transforms for all boundary conditions.
 *
 * Implements:
 * - cuFFT for periodic boundary conditions
 * - Delegates to CudaRealTransform for non-periodic BCs (DCT/DST)
 *
 * @see CudaFFT.h for class documentation
 * @see CudaRealTransform for DCT/DST implementation
 */

#include <iostream>
#include <cmath>
#include <stdexcept>

#include "CudaFFT.h"
#include "CudaCommon.h"
#include "CudaRealTransform.h"

//------------------------------------------------------------------------------
// CUDA Kernels for copy and scale operations
//------------------------------------------------------------------------------

// NOTE: these glue kernels are launched with a fixed grid (n_blocks*n_threads),
// so they MUST use a grid-stride loop to cover all M elements. Without it, arrays
// larger than n_blocks*n_threads (e.g. 65536) are left partially uninitialized,
// which shows up as NaN in the non-periodic (DCT/DST) transform path.
__global__ void ker_copy_data(double* dst, const double* src, int M)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < M; idx += blockDim.x * gridDim.x)
        dst[idx] = src[idx];
}

__global__ void ker_complex_part_to_real(double* dst, const cuDoubleComplex* src, int part, int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        dst[i] = (part == 0) ? src[i].x : src[i].y;
        i += blockDim.x * gridDim.x;
    }
}

__global__ void ker_real_to_interleaved_part(double* dst, const double* src, int part, int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        dst[2 * i + part] = src[i];
        i += blockDim.x * gridDim.x;
    }
}

__global__ void ker_interleaved_part_to_real(double* dst, const double* src, int part, int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        dst[i] = src[2 * i + part];
        i += blockDim.x * gridDim.x;
    }
}

__global__ void ker_real_to_complex_part(cuDoubleComplex* dst, const double* src, int part, int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        if (part == 0) dst[i].x = src[i];
        else           dst[i].y = src[i];
        i += blockDim.x * gridDim.x;
    }
}

__global__ void ker_complex_to_real(double* dst, const cuDoubleComplex* src, int M)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < M; idx += blockDim.x * gridDim.x)
        dst[idx] = src[idx].x;
}

__global__ void ker_real_to_complex(cuDoubleComplex* dst, const double* src, int M)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < M; idx += blockDim.x * gridDim.x)
    {
        dst[idx].x = src[idx];
        dst[idx].y = 0.0;
    }
}

__global__ void ker_complex_to_interleaved(double* dst, const cuDoubleComplex* src, int M_COMPLEX)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < M_COMPLEX; idx += blockDim.x * gridDim.x)
    {
        dst[2 * idx] = src[idx].x;
        dst[2 * idx + 1] = src[idx].y;
    }
}

__global__ void ker_interleaved_to_complex(cuDoubleComplex* dst, const double* src, int M_COMPLEX)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < M_COMPLEX; idx += blockDim.x * gridDim.x)
    {
        dst[idx].x = src[2 * idx];
        dst[idx].y = src[2 * idx + 1];
    }
}

__global__ void ker_scale(double* data, double scale, int M)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < M; idx += blockDim.x * gridDim.x)
        data[idx] *= scale;
}

//------------------------------------------------------------------------------
// Constructor (all periodic - backward compatible)
//------------------------------------------------------------------------------
template <typename T, int DIM>
CudaFFT<T, DIM>::CudaFFT(std::array<int, DIM> nx)
    : nx_(nx), is_all_periodic_(true), plan_forward_(0), plan_backward_(0),
      d_work_buffer_(nullptr), rt_forward_(nullptr), rt_backward_(nullptr)
{
    for (int d = 0; d < DIM; ++d)
        bc_[d] = BoundaryCondition::PERIODIC;

    try
    {
        total_grid_ = 1;
        for (int d = 0; d < DIM; ++d)
            total_grid_ *= nx_[d];

        if constexpr (std::is_same<T, std::complex<double>>::value)
        {
            // c2c transform: full spectrum (no Hermitian symmetry)
            total_complex_grid_ = total_grid_;
        }
        else if (DIM == 3)
            total_complex_grid_ = nx_[0] * nx_[1] * (nx_[2] / 2 + 1);
        else if (DIM == 2)
            total_complex_grid_ = nx_[0] * (nx_[1] / 2 + 1);
        else if (DIM == 1)
            total_complex_grid_ = nx_[0] / 2 + 1;

        gpu_error_check(cudaMalloc((void**)&d_work_buffer_, sizeof(cuDoubleComplex) * total_complex_grid_));

        initPeriodicFFT();
    }
    catch (std::exception& exc)
    {
        release();
        throw_without_line_number(exc.what());
    }
    catch (...)
    {
        release();
        throw;
    }
}

//------------------------------------------------------------------------------
// Constructor (with boundary conditions)
//------------------------------------------------------------------------------
template <typename T, int DIM>
CudaFFT<T, DIM>::CudaFFT(std::array<int, DIM> nx, std::array<BoundaryCondition, DIM> bc)
    : nx_(nx), bc_(bc), plan_forward_(0), plan_backward_(0),
      d_work_buffer_(nullptr), rt_forward_(nullptr), rt_backward_(nullptr)
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
            if constexpr (std::is_same<T, std::complex<double>>::value)
            {
                // c2c transform: full spectrum (no Hermitian symmetry)
                total_complex_grid_ = total_grid_;
            }
            else if (DIM == 3)
                total_complex_grid_ = nx_[0] * nx_[1] * (nx_[2] / 2 + 1);
            else if (DIM == 2)
                total_complex_grid_ = nx_[0] * (nx_[1] / 2 + 1);
            else if (DIM == 1)
                total_complex_grid_ = nx_[0] / 2 + 1;

            gpu_error_check(cudaMalloc((void**)&d_work_buffer_, sizeof(cuDoubleComplex) * total_complex_grid_));
            initPeriodicFFT();
        }
        else
        {
            // Check for odd dimensions - DCT-2/DCT-3 only supports even N
            // The FCT algorithm (Makhoul 1980) requires even N for interleaved decomposition
            for (int d = 0; d < DIM; ++d)
            {
                if (nx_[d] % 2 != 0)
                {
                    throw_with_line_number("CudaFFT with non-periodic BC requires even grid sizes. "
                                          "Dimension " + std::to_string(d) + " has odd size " +
                                          std::to_string(nx_[d]) + ". DCT-2/DCT-3 implementation "
                                          "uses FCT algorithm which only supports even N.");
                }
            }

            total_complex_grid_ = total_grid_;

            // Allocate work buffer for non-periodic transforms
            gpu_error_check(cudaMalloc((void**)&d_work_buffer_, sizeof(double) * total_grid_));

            initNonPeriodicFFT();
        }
    }
    catch (std::exception& exc)
    {
        release();
        throw_without_line_number(exc.what());
    }
    catch (...)
    {
        release();
        throw;
    }
}

//------------------------------------------------------------------------------
// Destructor
//------------------------------------------------------------------------------
template <typename T, int DIM>
CudaFFT<T, DIM>::~CudaFFT()
{
    release();
}

//------------------------------------------------------------------------------
// Release (idempotent; tolerates a partly built object)
//------------------------------------------------------------------------------
template <typename T, int DIM>
void CudaFFT<T, DIM>::release()
{
    if (d_work_buffer_ != nullptr)
        cudaFree(d_work_buffer_);
    d_work_buffer_ = nullptr;

    if (plan_forward_created_)
        cufftDestroy(plan_forward_);
    if (plan_backward_created_)
        cufftDestroy(plan_backward_);
    plan_forward_created_ = false;
    plan_backward_created_ = false;

    // Delete CudaRealTransform objects (either may be null if construction failed)
    if constexpr (DIM == 1)
    {
        delete static_cast<CudaRealTransform1D*>(rt_forward_);
        delete static_cast<CudaRealTransform1D*>(rt_backward_);
    }
    else if constexpr (DIM == 2)
    {
        delete static_cast<CudaRealTransform2D*>(rt_forward_);
        delete static_cast<CudaRealTransform2D*>(rt_backward_);
    }
    else if constexpr (DIM == 3)
    {
        delete static_cast<CudaRealTransform3D*>(rt_forward_);
        delete static_cast<CudaRealTransform3D*>(rt_backward_);
    }
    rt_forward_ = nullptr;
    rt_backward_ = nullptr;
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

    cufft_error_check(cufftPlanMany(&plan_forward_, DIM, total_grid, nullptr, 1, 0, nullptr, 1, 0, cufft_forward, 1));
    plan_forward_created_ = true;
    cufft_error_check(cufftPlanMany(&plan_backward_, DIM, total_grid, nullptr, 1, 0, nullptr, 1, 0, cufft_backward, 1));
    plan_backward_created_ = true;
}

//------------------------------------------------------------------------------
// Initialize CudaRealTransform for non-periodic BC
//------------------------------------------------------------------------------
template <typename T, int DIM>
void CudaFFT<T, DIM>::initNonPeriodicFFT()
{
    // Determine transform types for each dimension
    // Forward: DCT-2 for REFLECTING, DST-2 for ABSORBING
    // Backward: DCT-3 for REFLECTING, DST-3 for ABSORBING

    if constexpr (DIM == 1)
    {
        CudaTransformType fwd_type = (bc_[0] == BoundaryCondition::REFLECTING) ? CUDA_DCT_2 : CUDA_DST_2;
        CudaTransformType bwd_type = (bc_[0] == BoundaryCondition::REFLECTING) ? CUDA_DCT_3 : CUDA_DST_3;

        rt_forward_ = new CudaRealTransform1D(nx_[0], fwd_type);
        rt_backward_ = new CudaRealTransform1D(nx_[0], bwd_type);
    }
    else if constexpr (DIM == 2)
    {
        CudaTransformType fwd_type_x = (bc_[0] == BoundaryCondition::REFLECTING) ? CUDA_DCT_2 : CUDA_DST_2;
        CudaTransformType fwd_type_y = (bc_[1] == BoundaryCondition::REFLECTING) ? CUDA_DCT_2 : CUDA_DST_2;
        CudaTransformType bwd_type_x = (bc_[0] == BoundaryCondition::REFLECTING) ? CUDA_DCT_3 : CUDA_DST_3;
        CudaTransformType bwd_type_y = (bc_[1] == BoundaryCondition::REFLECTING) ? CUDA_DCT_3 : CUDA_DST_3;

        rt_forward_ = new CudaRealTransform2D(nx_[0], nx_[1], fwd_type_x, fwd_type_y);
        rt_backward_ = new CudaRealTransform2D(nx_[0], nx_[1], bwd_type_x, bwd_type_y);
    }
    else if constexpr (DIM == 3)
    {
        CudaTransformType fwd_type_x = (bc_[0] == BoundaryCondition::REFLECTING) ? CUDA_DCT_2 : CUDA_DST_2;
        CudaTransformType fwd_type_y = (bc_[1] == BoundaryCondition::REFLECTING) ? CUDA_DCT_2 : CUDA_DST_2;
        CudaTransformType fwd_type_z = (bc_[2] == BoundaryCondition::REFLECTING) ? CUDA_DCT_2 : CUDA_DST_2;
        CudaTransformType bwd_type_x = (bc_[0] == BoundaryCondition::REFLECTING) ? CUDA_DCT_3 : CUDA_DST_3;
        CudaTransformType bwd_type_y = (bc_[1] == BoundaryCondition::REFLECTING) ? CUDA_DCT_3 : CUDA_DST_3;
        CudaTransformType bwd_type_z = (bc_[2] == BoundaryCondition::REFLECTING) ? CUDA_DCT_3 : CUDA_DST_3;

        rt_forward_ = new CudaRealTransform3D(nx_[0], nx_[1], nx_[2], fwd_type_x, fwd_type_y, fwd_type_z);
        rt_backward_ = new CudaRealTransform3D(nx_[0], nx_[1], nx_[2], bwd_type_x, bwd_type_y, bwd_type_z);
    }
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
            // Non-periodic: use CudaRealTransform (in-place). The DCT/DST is
            // real-linear, so complex fields (CL-FTS) transform their real and
            // imaginary parts independently; coefficients are stored
            // interleaved as complex pairs (d_cdata then holds
            // 2*total_complex_grid_ doubles, matching the periodic layout).
            if constexpr (std::is_same<T, double>::value)
            {
                // Real fields: copy the input into the output buffer (both hold
                // total_grid_ values) and transform it there, so the input is kept
                ker_copy_data<<<N_BLOCKS, N_THREADS, 0, stream>>>(d_cdata, d_rdata, total_grid_);
                gpu_error_check(cudaPeekAtLastError());
                if constexpr (DIM == 1)
                    static_cast<CudaRealTransform1D*>(rt_forward_)->execute(d_cdata, stream);
                else if constexpr (DIM == 2)
                    static_cast<CudaRealTransform2D*>(rt_forward_)->execute(d_cdata, stream);
                else if constexpr (DIM == 3)
                    static_cast<CudaRealTransform3D*>(rt_forward_)->execute(d_cdata, stream);
                return;
            }
            const int n_parts = 2;
            for (int part = 0; part < n_parts; ++part)
            {
                ker_complex_part_to_real<<<N_BLOCKS, N_THREADS, 0, stream>>>(
                    d_work_buffer_, reinterpret_cast<cuDoubleComplex*>(d_rdata), part, total_grid_);
                gpu_error_check(cudaPeekAtLastError());

                // Execute forward transform (DCT-2 or DST-2) on the given stream
                if constexpr (DIM == 1)
                    static_cast<CudaRealTransform1D*>(rt_forward_)->execute(d_work_buffer_, stream);
                else if constexpr (DIM == 2)
                    static_cast<CudaRealTransform2D*>(rt_forward_)->execute(d_work_buffer_, stream);
                else if constexpr (DIM == 3)
                    static_cast<CudaRealTransform3D*>(rt_forward_)->execute(d_work_buffer_, stream);

                // Copy to output (interleaved for complex fields)
                ker_real_to_interleaved_part<<<N_BLOCKS, N_THREADS, 0, stream>>>(
                    d_cdata, d_work_buffer_, part, total_complex_grid_);
                gpu_error_check(cudaPeekAtLastError());
            }
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

                ker_scale<<<N_BLOCKS, N_THREADS, 0, stream>>>(
                    d_rdata, 1.0 / total_grid_, total_grid_);
            }
            else
            {
                cufftSetStream(plan_backward_, stream);
                cufftExecZ2Z(plan_backward_, d_complex,
                             reinterpret_cast<cuDoubleComplex*>(d_rdata), CUFFT_INVERSE);

                ker_scale<<<N_BLOCKS, N_THREADS, 0, stream>>>(
                    reinterpret_cast<double*>(d_rdata), 1.0 / total_grid_, total_grid_ * 2);
            }
            gpu_error_check(cudaPeekAtLastError());
        }
        else
        {
            // Non-periodic: use CudaRealTransform (in-place). Complex fields
            // arrive as interleaved coefficient pairs (see forward_stream);
            // inverse-transform the two components independently.
            double scale = 1.0;
            for (int d = 0; d < DIM; ++d)
                scale *= 1.0 / (2.0 * nx_[d]);

            if constexpr (std::is_same<T, double>::value)
            {
                // Real fields: copy the coefficients into the output buffer and
                // transform there, so the coefficients are kept
                ker_copy_data<<<N_BLOCKS, N_THREADS, 0, stream>>>(d_rdata, d_cdata, total_grid_);
                gpu_error_check(cudaPeekAtLastError());
                if constexpr (DIM == 1)
                    static_cast<CudaRealTransform1D*>(rt_backward_)->execute(d_rdata, stream);
                else if constexpr (DIM == 2)
                    static_cast<CudaRealTransform2D*>(rt_backward_)->execute(d_rdata, stream);
                else if constexpr (DIM == 3)
                    static_cast<CudaRealTransform3D*>(rt_backward_)->execute(d_rdata, stream);
                // Normalize (DCT/DST round-trip scaling is 2*N per dimension)
                ker_scale<<<N_BLOCKS, N_THREADS, 0, stream>>>(d_rdata, scale, total_grid_);
                gpu_error_check(cudaPeekAtLastError());
                return;
            }
            const int n_parts = 2;
            for (int part = 0; part < n_parts; ++part)
            {
                // Copy input component to work buffer
                ker_interleaved_part_to_real<<<N_BLOCKS, N_THREADS, 0, stream>>>(
                    d_work_buffer_, d_cdata, part, total_complex_grid_);
                gpu_error_check(cudaPeekAtLastError());

                // Execute backward transform (DCT-3 or DST-3) on the given stream
                if constexpr (DIM == 1)
                    static_cast<CudaRealTransform1D*>(rt_backward_)->execute(d_work_buffer_, stream);
                else if constexpr (DIM == 2)
                    static_cast<CudaRealTransform2D*>(rt_backward_)->execute(d_work_buffer_, stream);
                else if constexpr (DIM == 3)
                    static_cast<CudaRealTransform3D*>(rt_backward_)->execute(d_work_buffer_, stream);

                // Normalize (DCT/DST round-trip scaling is 2*N per dimension)
                ker_scale<<<N_BLOCKS, N_THREADS, 0, stream>>>(d_work_buffer_, scale, total_grid_);

                // Copy to output component
                ker_real_to_complex_part<<<N_BLOCKS, N_THREADS, 0, stream>>>(
                    reinterpret_cast<cuDoubleComplex*>(d_rdata), d_work_buffer_, part, total_grid_);
                gpu_error_check(cudaPeekAtLastError());
            }
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

            ker_scale<<<N_BLOCKS, N_THREADS, 0, stream>>>(
                d_rdata, 1.0 / total_grid_, total_grid_);
        }
        else
        {
            cufftSetStream(plan_backward_, stream);
            cufftExecZ2Z(plan_backward_, d_cdata_cu,
                         reinterpret_cast<cuDoubleComplex*>(d_rdata), CUFFT_INVERSE);

            ker_scale<<<N_BLOCKS, N_THREADS, 0, stream>>>(
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
