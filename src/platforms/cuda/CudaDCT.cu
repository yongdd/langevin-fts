/**
 * @file CudaDCT.cu
 * @brief CUDA DCT (Discrete Cosine Transform) Types 1-4 implementation.
 *
 * DCT-1/2/3 use N-size FFT, DCT-4 uses 2N-size FFT:
 * - DCT-1: N-size Z2D FFT (cuHelmholtz style)
 * - DCT-2: N-size Z2D FFT (cuHelmholtz style)
 * - DCT-3: N-size D2Z FFT (cuHelmholtz style)
 * - DCT-4: 2N-size Z2Z FFT (pre/post phase rotation)
 *
 * Reference: https://github.com/rmingming/cuHelmholtz
 */

#include "CudaDCT.h"
#include <cmath>
#include <stdexcept>
#include <cstdio>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

//==============================================================================
// Helper functions
//==============================================================================
static int pow2roundup(int x)
{
    if (x < 0) return 0;
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x + 1;
}

//==============================================================================
// DCT-1 Kernels (FFTW_REDFT00) - cuHelmholtz style with N-size FFT
// Input: N+1 points, FFT size: N
// Reference: https://github.com/rmingming/cuHelmholtz/blob/master/cudasymmfft/dct1funcinplace.cu
//==============================================================================

/**
 * @brief DCT-1 preprocessing following cuHelmholtz exactly.
 *
 * 1. Load odd-indexed elements to shared memory
 * 2. Compute differences and store to odd positions
 * 3. Set pin[1]=0, pin[N+1]=0
 * 4. Parallel reduction to compute x1
 */
__global__ void kernel_dct1_preOp(
    double* pin,
    double* px1,
    int N,
    int nThread)
{
    extern __shared__ double sh_in[];
    int itx = threadIdx.x;

    // Load odd-indexed elements: pin[1], pin[3], ..., pin[N-1]
    if (itx < N / 2) {
        sh_in[itx] = pin[itx * 2 + 1];
    } else {
        sh_in[itx] = 0;
    }
    __syncthreads();

    // Compute differences and store to odd positions
    if (itx < N / 2) {
        if (itx == 0) {
            pin[N + 1] = 0;
            pin[1] = 0;
        } else {
            pin[itx * 2 + 1] = sh_in[itx - 1] - sh_in[itx];
        }
    }
    __syncthreads();

    // Parallel reduction to sum differences
    for (unsigned int s = nThread >> 1; s > 0; s >>= 1) {
        if (itx < s) {
            sh_in[itx] += sh_in[itx + s];
        }
        __syncthreads();
    }

    if (itx == 0) {
        px1[0] = sh_in[0];
    }
}

/**
 * @brief DCT-1 postprocessing following cuHelmholtz.
 * Multiply by 2 to match FFTW (cuHelmholtz gives 1/2 of FFTW).
 */
__global__ void kernel_dct1_postOp(
    double* pin,
    double* px1,
    int N)
{
    int itx = threadIdx.x;

    __syncthreads();

    if (itx < N / 2 + 1) {
        if (itx != 0) {
            double sitx = pin[itx];
            double sNitx = pin[N - itx];
            double Ta = (sitx + sNitx) * 0.5;
            double Tb = (sitx - sNitx) * 0.25 / sin(itx * M_PI / N);

            // Multiply by 2 to match FFTW
            pin[itx] = (Ta - Tb);      // was * 0.5
            pin[N - itx] = (Ta + Tb);  // was * 0.5
        } else {
            double sh0 = pin[0] * 0.5;
            // Multiply by 2 to match FFTW
            pin[0] = (sh0 + px1[0]) * 2.0;
            pin[N] = (sh0 - px1[0]) * 2.0;
        }
    }
}

//==============================================================================
// DCT-2 Kernels (FFTW_REDFT10)
// Following cuHelmholtz: preOp -> Z2D FFT -> postOp
//==============================================================================

__global__ void kernel_dct2_preOp(
    double* data,
    int N)
{
    extern __shared__ double sh_in[];
    int itx = threadIdx.x;

    if (itx < N / 2) {
        sh_in[itx] = data[itx];
        sh_in[itx + N / 2] = data[itx + N / 2];
    }
    __syncthreads();

    if (itx < N / 2 + 1) {
        if (itx == 0) {
            data[0] = sh_in[0];
            data[1] = 0;
        } else if (itx == N / 2) {
            data[N] = sh_in[N - 1];
            data[N + 1] = 0;
        } else {
            data[itx * 2] = (sh_in[itx * 2] + sh_in[itx * 2 - 1]) / 2;
            data[itx * 2 + 1] = -((sh_in[itx * 2 - 1] - sh_in[itx * 2]) / 2);
        }
    }
}

__global__ void kernel_dct2_postOp(
    double* data,
    int N)
{
    extern __shared__ double sh_in[];
    int itx = threadIdx.x;

    if (itx < N / 2) {
        sh_in[itx] = data[itx];
        sh_in[itx + N / 2] = data[itx + N / 2];
    }
    __syncthreads();

    if (itx < N / 2 + 1) {
        if (itx == 0) {
            data[0] = sh_in[0] * 2;
        } else {
            double sina, cosa;
            sincos(itx * M_PI / (2 * N), &sina, &cosa);
            double Ta = sh_in[itx] + sh_in[N - itx];
            double Tb = sh_in[itx] - sh_in[N - itx];

            data[itx] = (Ta * cosa + Tb * sina);
            data[N - itx] = (Ta * sina - Tb * cosa);
        }
    }
}

//==============================================================================
// DCT-3 Kernels (FFTW_REDFT01)
// Following cuHelmholtz: preOp -> D2Z FFT -> postOp
//==============================================================================

__global__ void kernel_dct3_preOp(
    double* data,
    int N)
{
    int itx = threadIdx.x;

    if (itx < N / 2) {
        double sina, cosa;
        sincos((itx + 1) * M_PI / (2 * N), &sina, &cosa);
        double Ta = data[itx + 1] + data[N - itx - 1];
        double Tb = data[itx + 1] - data[N - itx - 1];

        data[itx + 1] = Ta * sina + Tb * cosa;
        data[N - itx - 1] = Ta * cosa - Tb * sina;
    }
}

__global__ void kernel_dct3_postOp(
    double* data,
    int N)
{
    extern __shared__ double sh_in[];
    int itx = threadIdx.x;

    if (itx < N / 2 + 1) {
        sh_in[itx] = data[itx];
        sh_in[itx + N / 2 + 1] = data[itx + N / 2 + 1];
    }
    __syncthreads();

    if (itx < N / 2 + 1) {
        if (itx == 0) {
            data[0] = sh_in[0];
        } else {
            data[2 * itx - 1] = (sh_in[itx * 2] - sh_in[itx * 2 + 1]);
            if (itx * 2 < N) {
                data[2 * itx] = (sh_in[itx * 2] + sh_in[itx * 2 + 1]);
            }
        }
    }
}

//==============================================================================
// DCT-4 Kernels (FFTW_REDFT11)
// DCT-4[k] = sum_{n=0}^{N-1} x[n] * cos(π(2n+1)(2k+1)/(4N))
//
// Using 2N-size complex FFT (Z2Z):
// 1. preOp: z[n] = x[n] * exp(-iπ(2n+1)/(4N)) for n=0..N-1, z[n]=0 for n=N..2N-1
// 2. 2N Z2Z FFT
// 3. postOp: DCT-4[k] = 2 * Re(Z[k] * exp(-iπk/(2N)))
//==============================================================================

/**
 * @brief DCT-4 preOp: apply phase rotation and zero-pad to 2N.
 * z[n] = x[n] * exp(-iπ(2n+1)/(4N)) for n=0..N-1
 * z[n] = 0 for n=N..2N-1
 */
__global__ void kernel_dct4_preOp(
    const double* __restrict__ in,
    cufftDoubleComplex* __restrict__ out,
    int N)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int N2 = 2 * N;

    if (n < N2) {
        if (n < N) {
            double angle = -M_PI * (2 * n + 1) / (4.0 * N);
            out[n].x = in[n] * cos(angle);
            out[n].y = in[n] * sin(angle);
        } else {
            out[n].x = 0.0;
            out[n].y = 0.0;
        }
    }
}

/**
 * @brief DCT-4 postOp: apply phase rotation and extract real part.
 * DCT-4[k] = 2 * Re(Z[k] * exp(-iπk/(2N)))
 */
__global__ void kernel_dct4_postOp(
    const cufftDoubleComplex* __restrict__ in,
    double* __restrict__ out,
    int N)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < N) {
        double angle = -M_PI * k / (2.0 * N);
        double cosa = cos(angle);
        double sina = sin(angle);

        // Z[k] * exp(-iπk/(2N))
        double re = in[k].x * cosa - in[k].y * sina;

        // Multiply by 2 to match FFTW
        out[k] = 2.0 * re;
    }
}

//==============================================================================
// CudaDCT class implementation
//==============================================================================
CudaDCT::CudaDCT(int N, CudaDCTType type)
    : N_(N), type_(type), plan_(0), d_work_(nullptr), d_x1_(nullptr), initialized_(false)
{
    if (type_ == CUDA_DCT_1) {
        // DCT-1: N-size Z2D FFT (cuHelmholtz style)
        int N = N_ - 1;  // Actual DCT-1 size
        cudaMalloc(&d_work_, sizeof(double) * (N + 2));
        cudaMalloc(&d_x1_, sizeof(double) * 1);

        int n[1] = {N};
        int inembed[1] = {(N + 2) / 2};
        int onembed[1] = {N + 2};
        cufftPlanMany(&plan_, 1, n, inembed, 1, (N + 2) / 2,
                      onembed, 1, N + 2, CUFFT_Z2D, 1);
    } else if (type_ == CUDA_DCT_2) {
        cudaMalloc(&d_work_, sizeof(double) * (N_ + 2));
        cudaMalloc(&d_x1_, sizeof(cufftDoubleComplex) * (N_ / 2 + 1));

        int n[1] = {N_};
        int inembed[1] = {(N_ + 2) / 2};
        int onembed[1] = {N_ + 2};
        cufftPlanMany(&plan_, 1, n, inembed, 1, (N_ + 2) / 2,
                      onembed, 1, N_ + 2, CUFFT_Z2D, 1);
    } else if (type_ == CUDA_DCT_3) {
        cudaMalloc(&d_work_, sizeof(double) * (N_ + 2));
        cudaMalloc(&d_x1_, sizeof(cufftDoubleComplex) * (N_ / 2 + 1));

        int n[1] = {N_};
        int inembed[1] = {N_ + 2};
        int onembed[1] = {(N_ + 2) / 2};
        cufftPlanMany(&plan_, 1, n, inembed, 1, N_ + 2,
                      onembed, 1, (N_ + 2) / 2, CUFFT_D2Z, 1);
    } else if (type_ == CUDA_DCT_4) {
        // DCT-4: 2N-size Z2Z FFT
        cudaMalloc(&d_work_, sizeof(cufftDoubleComplex) * 2 * N_);  // Complex input buffer (2N)
        cudaMalloc(&d_x1_, sizeof(cufftDoubleComplex) * 2 * N_);    // Complex output buffer (2N)
        cufftPlan1d(&plan_, 2 * N_, CUFFT_Z2Z, 1);
    }

    initialized_ = true;
}

CudaDCT::~CudaDCT()
{
    if (plan_) cufftDestroy(plan_);
    if (d_work_) cudaFree(d_work_);
    if (d_x1_) cudaFree(d_x1_);
}

void CudaDCT::execute(double* d_data)
{
    int threads = 256;

    switch (type_) {
        case CUDA_DCT_1:
        {
            // cuHelmholtz style: preOp -> Z2D -> postOp
            int N = N_ - 1;
            int nThread = pow2roundup(N / 2);

            // Copy input to work buffer (N+2 size for FFT padding)
            cudaMemcpy(d_work_, d_data, sizeof(double) * (N + 1), cudaMemcpyDeviceToDevice);
            cudaMemset(d_work_ + N + 1, 0, sizeof(double));  // Zero padding

            // PreOp
            kernel_dct1_preOp<<<1, nThread, nThread * sizeof(double)>>>(
                d_work_, d_x1_, N, nThread);
            cudaDeviceSynchronize();

            // Z2D FFT
            cufftExecZ2D(plan_, reinterpret_cast<cufftDoubleComplex*>(d_work_), d_work_);
            cudaDeviceSynchronize();

            // PostOp
            kernel_dct1_postOp<<<1, N / 2 + 1>>>(d_work_, d_x1_, N);
            cudaDeviceSynchronize();

            // Copy result back
            cudaMemcpy(d_data, d_work_, sizeof(double) * (N + 1), cudaMemcpyDeviceToDevice);
            break;
        }

        case CUDA_DCT_2:
        {
            int nThread = N_ / 2 + 1;

            cudaMemcpy(d_work_, d_data, sizeof(double) * N_, cudaMemcpyDeviceToDevice);

            kernel_dct2_preOp<<<1, nThread, N_ * sizeof(double)>>>(d_work_, N_);
            cudaDeviceSynchronize();

            cufftExecZ2D(plan_, reinterpret_cast<cufftDoubleComplex*>(d_work_), d_work_);
            cudaDeviceSynchronize();

            kernel_dct2_postOp<<<1, nThread, N_ * sizeof(double)>>>(d_work_, N_);
            cudaDeviceSynchronize();

            cudaMemcpy(d_data, d_work_, sizeof(double) * N_, cudaMemcpyDeviceToDevice);
            break;
        }

        case CUDA_DCT_3:
        {
            int nThread = N_ / 2 + 1;

            cudaMemcpy(d_work_, d_data, sizeof(double) * N_, cudaMemcpyDeviceToDevice);

            kernel_dct3_preOp<<<1, N_ / 2>>>(d_work_, N_);
            cudaDeviceSynchronize();

            cufftExecD2Z(plan_, d_work_, reinterpret_cast<cufftDoubleComplex*>(d_work_));
            cudaDeviceSynchronize();

            kernel_dct3_postOp<<<1, nThread, (N_ + 2) * sizeof(double)>>>(d_work_, N_);
            cudaDeviceSynchronize();

            cudaMemcpy(d_data, d_work_, sizeof(double) * N_, cudaMemcpyDeviceToDevice);
            break;
        }

        case CUDA_DCT_4:
        {
            // DCT-4 using 2N-size Z2Z FFT
            int N2 = 2 * N_;
            int blocks_2N = (N2 + threads - 1) / threads;
            int blocks_N = (N_ + threads - 1) / threads;
            cufftDoubleComplex* d_complex_in = reinterpret_cast<cufftDoubleComplex*>(d_work_);
            cufftDoubleComplex* d_complex_out = reinterpret_cast<cufftDoubleComplex*>(d_x1_);

            // Step 1: preOp - phase rotation and zero-pad to 2N
            kernel_dct4_preOp<<<blocks_2N, threads>>>(d_data, d_complex_in, N_);
            cudaDeviceSynchronize();

            // Step 2: 2N Z2Z FFT
            cufftExecZ2Z(plan_, d_complex_in, d_complex_out, CUFFT_FORWARD);
            cudaDeviceSynchronize();

            // Step 3: postOp - phase rotation and extract real part
            kernel_dct4_postOp<<<blocks_N, threads>>>(d_complex_out, d_data, N_);
            cudaDeviceSynchronize();
            break;
        }
    }
}

int CudaDCT::get_size() const
{
    return N_;
}

double CudaDCT::get_normalization() const
{
    switch (type_) {
        case CUDA_DCT_1:
            return 1.0 / (2.0 * (N_ - 1));
        case CUDA_DCT_2:
        case CUDA_DCT_3:
            return 1.0 / (2.0 * N_);
        case CUDA_DCT_4:
            return 1.0 / (2.0 * N_);
        default:
            return 1.0;
    }
}

//==============================================================================
// CudaDCT3D class implementation
// 3D DCT-1 using separable 1D transforms with transposes
//==============================================================================

/**
 * @brief Apply 1D DCT-1 to all rows of a 3D array (along fastest dimension).
 */
__global__ void kernel_dct1_3d_preOp_z(
    double* data,
    double* x1_buffer,
    int Nx, int Ny, int Nz)
{
    // Each block handles one row (Nz+1 elements)
    int row_idx = blockIdx.x;
    int ix = row_idx / (Ny + 1);
    int iy = row_idx % (Ny + 1);

    extern __shared__ double sh[];
    int itx = threadIdx.x;

    int row_offset = (ix * (Ny + 1) + iy) * (Nz + 1);
    double* row = data + row_offset;

    // Load odd-indexed elements
    if (itx < Nz / 2) {
        sh[itx] = row[2 * itx + 1];
    }
    __syncthreads();

    // Compute differences
    double diff;
    if (itx == 0) {
        diff = row[0] - sh[0];
    } else if (itx < Nz / 2) {
        diff = sh[itx - 1] - sh[itx];
    }
    __syncthreads();

    if (itx < Nz / 2) {
        sh[itx] = diff;
    }
    __syncthreads();

    // Parallel reduction
    int nThread = blockDim.x;
    for (int stride = nThread / 2; stride > 0; stride >>= 1) {
        if (itx < stride && itx + stride < Nz / 2) {
            sh[itx] += sh[itx + stride];
        }
        __syncthreads();
    }

    if (itx == 0) {
        x1_buffer[row_idx] = sh[0] + row[Nz] - row[Nz - 1];
    }
}

// Using mirror extension for 3D (simpler and works well)
__global__ void kernel_dct1_3d_mirror_extend(
    const double* __restrict__ in,
    double* __restrict__ out,
    int Nx, int Ny, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int ext_total = 2 * Nx * 2 * Ny * 2 * Nz;

    if (idx < ext_total) {
        int Nz2 = 2 * Nz, Ny2 = 2 * Ny;

        int iz = idx % Nz2;
        int iy = (idx / Nz2) % Ny2;
        int ix = idx / (Nz2 * Ny2);

        int ix_in = (ix <= Nx) ? ix : (2 * Nx - ix);
        int iy_in = (iy <= Ny) ? iy : (2 * Ny - iy);
        int iz_in = (iz <= Nz) ? iz : (2 * Nz - iz);

        int in_idx = (ix_in * (Ny + 1) + iy_in) * (Nz + 1) + iz_in;
        out[idx] = in[in_idx];
    }
}

__global__ void kernel_dct1_3d_extract(
    const cufftDoubleComplex* __restrict__ in,
    double* __restrict__ out,
    int Nx, int Ny, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int out_total = (Nx + 1) * (Ny + 1) * (Nz + 1);

    if (idx < out_total) {
        int Ny1 = Ny + 1, Nz1 = Nz + 1;
        int Nz_half = Nz + 1;

        int iz = idx % Nz1;
        int iy = (idx / Nz1) % Ny1;
        int ix = idx / (Nz1 * Ny1);

        int freq_idx = (ix * 2 * Ny + iy) * Nz_half + iz;
        out[idx] = in[freq_idx].x;
    }
}

//==============================================================================
// 2D DCT-1 Kernels
//==============================================================================

__global__ void kernel_dct1_2d_mirror_extend(
    const double* __restrict__ in,
    double* __restrict__ out,
    int Nx, int Ny)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int ext_total = 2 * Nx * 2 * Ny;

    if (idx < ext_total) {
        int Ny2 = 2 * Ny;

        int iy = idx % Ny2;
        int ix = idx / Ny2;

        int ix_in = (ix <= Nx) ? ix : (2 * Nx - ix);
        int iy_in = (iy <= Ny) ? iy : (2 * Ny - iy);

        int in_idx = ix_in * (Ny + 1) + iy_in;
        out[idx] = in[in_idx];
    }
}

__global__ void kernel_dct1_2d_extract(
    const cufftDoubleComplex* __restrict__ in,
    double* __restrict__ out,
    int Nx, int Ny)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int out_total = (Nx + 1) * (Ny + 1);

    if (idx < out_total) {
        int Ny1 = Ny + 1;
        int Ny_half = Ny + 1;

        int iy = idx % Ny1;
        int ix = idx / Ny1;

        int freq_idx = ix * Ny_half + iy;
        out[idx] = in[freq_idx].x;
    }
}

//==============================================================================
// 2D DCT-2 Kernels (batched 1D DCT-2)
//==============================================================================

/**
 * @brief 2D DCT-2 preOp for Y dimension.
 * Applies DCT-2 preprocessing along Y for each row.
 */
__global__ void kernel_dct2_2d_preOp_y(
    const double* __restrict__ in,
    cufftDoubleComplex* __restrict__ out,
    int Nx, int Ny)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * (Ny / 2 + 1);

    if (idx < total) {
        int iy_half = idx % (Ny / 2 + 1);
        int ix = idx / (Ny / 2 + 1);

        int out_idx = ix * (Ny / 2 + 1) + iy_half;

        if (iy_half == 0) {
            out[out_idx].x = in[ix * Ny];
            out[out_idx].y = 0.0;
        } else if (iy_half == Ny / 2) {
            out[out_idx].x = in[ix * Ny + Ny - 1];
            out[out_idx].y = 0.0;
        } else {
            int in_idx = ix * Ny + 2 * iy_half;
            out[out_idx].x = (in[in_idx] + in[in_idx - 1]) / 2.0;
            out[out_idx].y = -((in[in_idx - 1] - in[in_idx]) / 2.0);
        }
    }
}

/**
 * @brief 2D DCT-2 postOp for Y dimension.
 */
__global__ void kernel_dct2_2d_postOp_y(
    const double* __restrict__ in,
    double* __restrict__ out,
    int Nx, int Ny)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny;

    if (idx < total) {
        int iy = idx % Ny;
        int ix = idx / Ny;

        if (iy == 0) {
            out[idx] = in[ix * Ny] * 2.0;
        } else {
            double sina, cosa;
            sincos(iy * M_PI / (2.0 * Ny), &sina, &cosa);
            double Ta = in[ix * Ny + iy] + in[ix * Ny + (Ny - iy)];
            double Tb = in[ix * Ny + iy] - in[ix * Ny + (Ny - iy)];
            out[idx] = Ta * cosa + Tb * sina;
        }
    }
}

/**
 * @brief 2D DCT-2 preOp for X dimension (transposed layout).
 */
__global__ void kernel_dct2_2d_preOp_x(
    const double* __restrict__ in,
    cufftDoubleComplex* __restrict__ out,
    int Nx, int Ny)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Ny * (Nx / 2 + 1);

    if (idx < total) {
        int ix_half = idx % (Nx / 2 + 1);
        int iy = idx / (Nx / 2 + 1);

        int out_idx = iy * (Nx / 2 + 1) + ix_half;

        if (ix_half == 0) {
            out[out_idx].x = in[iy];  // in[0][iy] in transposed
            out[out_idx].y = 0.0;
        } else if (ix_half == Nx / 2) {
            out[out_idx].x = in[(Nx - 1) * Ny + iy];
            out[out_idx].y = 0.0;
        } else {
            int ix_even = 2 * ix_half;
            int ix_odd = ix_even - 1;
            out[out_idx].x = (in[ix_even * Ny + iy] + in[ix_odd * Ny + iy]) / 2.0;
            out[out_idx].y = -((in[ix_odd * Ny + iy] - in[ix_even * Ny + iy]) / 2.0);
        }
    }
}

/**
 * @brief 2D DCT-2 postOp for X dimension (output to original layout).
 */
__global__ void kernel_dct2_2d_postOp_x(
    const double* __restrict__ in,
    double* __restrict__ out,
    int Nx, int Ny)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny;

    if (idx < total) {
        int iy = idx % Ny;
        int ix = idx / Ny;

        // in is in transposed layout [Ny][Nx], we read from [iy][ix]
        if (ix == 0) {
            out[idx] = in[iy * Nx] * 2.0;
        } else {
            double sina, cosa;
            sincos(ix * M_PI / (2.0 * Nx), &sina, &cosa);
            double Ta = in[iy * Nx + ix] + in[iy * Nx + (Nx - ix)];
            double Tb = in[iy * Nx + ix] - in[iy * Nx + (Nx - ix)];
            out[idx] = Ta * cosa + Tb * sina;
        }
    }
}

//==============================================================================
// 2D DCT-3 Kernels (batched 1D DCT-3)
//==============================================================================

/**
 * @brief 2D DCT-3 preOp for Y dimension.
 */
__global__ void kernel_dct3_2d_preOp_y(
    double* __restrict__ data,
    int Nx, int Ny)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * (Ny / 2);

    if (idx < total) {
        int iy = idx % (Ny / 2);
        int ix = idx / (Ny / 2);

        int base = ix * Ny;
        int iy1 = iy + 1;
        int iy2 = Ny - iy - 1;

        double sina, cosa;
        sincos(iy1 * M_PI / (2.0 * Ny), &sina, &cosa);
        double Ta = data[base + iy1] + data[base + iy2];
        double Tb = data[base + iy1] - data[base + iy2];

        data[base + iy1] = Ta * sina + Tb * cosa;
        data[base + iy2] = Ta * cosa - Tb * sina;
    }
}

/**
 * @brief 2D DCT-3 postOp for Y dimension.
 */
__global__ void kernel_dct3_2d_postOp_y(
    const cufftDoubleComplex* __restrict__ in,
    double* __restrict__ out,
    int Nx, int Ny)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny;

    if (idx < total) {
        int iy = idx % Ny;
        int ix = idx / Ny;

        int in_base = ix * (Ny / 2 + 1);

        if (iy == 0) {
            out[idx] = in[in_base].x;
        } else {
            int half_idx = (iy + 1) / 2;
            double re = in[in_base + half_idx].x;
            double im = in[in_base + half_idx].y;

            if (iy % 2 == 1) {
                out[idx] = re - im;
            } else {
                out[idx] = re + im;
            }
        }
    }
}

/**
 * @brief 2D DCT-3 preOp for X dimension with transpose.
 * Reads from [Nx][Ny] layout, applies DCT-3 preOp, writes to [Ny][Nx] transposed.
 */
__global__ void kernel_dct3_2d_preOp_x(
    const double* __restrict__ in,
    double* __restrict__ out,
    int Nx, int Ny)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny;

    if (idx < total) {
        int ix = idx % Nx;
        int iy = idx / Nx;

        // Read from [Nx][Ny] layout (in[ix][iy])
        // Write to [Ny][Nx] layout (out[iy][ix])
        out[iy * Nx + ix] = in[ix * Ny + iy];
    }
}

/**
 * @brief Apply DCT-3 preOp in-place on transposed [Ny][Nx] data.
 */
__global__ void kernel_dct3_2d_preOp_x_inplace(
    double* __restrict__ data,
    int Nx, int Ny)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Ny * (Nx / 2);

    if (idx < total) {
        int ix = idx % (Nx / 2);
        int iy = idx / (Nx / 2);

        // Data is in [Ny][Nx] layout, each row is an X-direction transform
        int ix1 = ix + 1;
        int ix2 = Nx - ix - 1;
        int base = iy * Nx;

        double sina, cosa;
        sincos(ix1 * M_PI / (2.0 * Nx), &sina, &cosa);
        double Ta = data[base + ix1] + data[base + ix2];
        double Tb = data[base + ix1] - data[base + ix2];

        data[base + ix1] = Ta * sina + Tb * cosa;
        data[base + ix2] = Ta * cosa - Tb * sina;
    }
}

/**
 * @brief 2D DCT-3 postOp for X dimension with transpose back.
 * Reads from [Ny][half_x] complex FFT output, writes to [Nx][Ny] layout.
 */
__global__ void kernel_dct3_2d_postOp_x(
    const cufftDoubleComplex* __restrict__ in,
    double* __restrict__ out,
    int Nx, int Ny)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny;

    if (idx < total) {
        // Output index in [Nx][Ny] layout
        int iy = idx % Ny;
        int ix = idx / Ny;

        // Input is in [Ny][Nx/2+1] layout
        int half_x = Nx / 2 + 1;
        int in_base = iy * half_x;

        double val;
        if (ix == 0) {
            val = in[in_base].x;
        } else {
            int half_idx = (ix + 1) / 2;
            double re = in[in_base + half_idx].x;
            double im = in[in_base + half_idx].y;

            if (ix % 2 == 1) {
                val = re - im;
            } else {
                val = re + im;
            }
        }

        // Write to [Nx][Ny] layout
        out[ix * Ny + iy] = val;
    }
}

//==============================================================================
// 2D DCT-4 Kernels
//==============================================================================

__global__ void kernel_dct4_2d_preOp_y(
    const double* __restrict__ in,
    cufftDoubleComplex* __restrict__ out,
    int Nx, int Ny)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * 2 * Ny;

    if (idx < total) {
        int iy = idx % (2 * Ny);
        int ix = idx / (2 * Ny);

        if (iy < Ny) {
            int in_idx = ix * Ny + iy;
            double angle = -M_PI * (2 * iy + 1) / (4.0 * Ny);
            out[idx].x = in[in_idx] * cos(angle);
            out[idx].y = in[in_idx] * sin(angle);
        } else {
            out[idx].x = 0.0;
            out[idx].y = 0.0;
        }
    }
}

__global__ void kernel_dct4_2d_postOp_y(
    const cufftDoubleComplex* __restrict__ in,
    double* __restrict__ out,
    int Nx, int Ny)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny;

    if (idx < total) {
        int iy = idx % Ny;
        int ix = idx / Ny;

        int in_idx = ix * 2 * Ny + iy;
        double angle = -M_PI * iy / (2.0 * Ny);
        double cosa = cos(angle);
        double sina = sin(angle);

        double re = in[in_idx].x * cosa - in[in_idx].y * sina;
        out[idx] = 2.0 * re;
    }
}

__global__ void kernel_dct4_2d_preOp_x(
    const double* __restrict__ in,
    cufftDoubleComplex* __restrict__ out,
    int Nx, int Ny)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Ny * 2 * Nx;

    if (idx < total) {
        int ix = idx % (2 * Nx);
        int iy = idx / (2 * Nx);

        if (ix < Nx) {
            int in_idx = ix * Ny + iy;
            double angle = -M_PI * (2 * ix + 1) / (4.0 * Nx);
            out[idx].x = in[in_idx] * cos(angle);
            out[idx].y = in[in_idx] * sin(angle);
        } else {
            out[idx].x = 0.0;
            out[idx].y = 0.0;
        }
    }
}

__global__ void kernel_dct4_2d_postOp_x(
    const cufftDoubleComplex* __restrict__ in,
    double* __restrict__ out,
    int Nx, int Ny)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny;

    if (idx < total) {
        int iy = idx % Ny;
        int ix = idx / Ny;

        int in_idx = iy * 2 * Nx + ix;
        double angle = -M_PI * ix / (2.0 * Nx);
        double cosa = cos(angle);
        double sina = sin(angle);

        double re = in[in_idx].x * cosa - in[in_idx].y * sina;
        out[idx] = 2.0 * re;
    }
}

//==============================================================================
// CudaDCT2D class implementation
//==============================================================================

CudaDCT2D::CudaDCT2D(int Nx, int Ny, CudaDCTType type)
    : Nx_(Nx), Ny_(Ny), type_x_(type), type_y_(type),
      plan_x_(0), plan_y_(0),
      d_work_(nullptr), d_temp_(nullptr), d_x1_(nullptr),
      initialized_(false)
{
    init();
}

CudaDCT2D::CudaDCT2D(int Nx, int Ny, CudaDCTType type_x, CudaDCTType type_y)
    : Nx_(Nx), Ny_(Ny), type_x_(type_x), type_y_(type_y),
      plan_x_(0), plan_y_(0),
      d_work_(nullptr), d_temp_(nullptr), d_x1_(nullptr),
      initialized_(false)
{
    // DCT-1 cannot be mixed with other types
    if ((type_x == CUDA_DCT_1 || type_y == CUDA_DCT_1) && type_x != type_y) {
        throw std::runtime_error("CudaDCT2D: DCT-1 cannot be mixed with other types");
    }
    init();
}

void CudaDCT2D::init()
{
    // DCT-1 uses optimized 2D FFT approach
    if (type_x_ == CUDA_DCT_1 && type_y_ == CUDA_DCT_1) {
        M_ = (Nx_ + 1) * (Ny_ + 1);

        int ext_size = 4 * Nx_ * Ny_;
        int freq_size = 2 * Nx_ * (Ny_ + 1);

        cudaMalloc(&d_work_, sizeof(double) * ext_size);
        cudaMalloc(&d_temp_, sizeof(cufftDoubleComplex) * freq_size);

        M_padded_ = ext_size;

        cufftPlan2d(&plan_x_, 2 * Nx_, 2 * Ny_, CUFFT_D2Z);
    } else {
        // Mixed or same type (DCT-2/3/4): use per-dimension processing
        M_ = Nx_ * Ny_;

        int half_y = Ny_ / 2 + 1;
        int half_x = Nx_ / 2 + 1;

        int max_freq_z2d = std::max(Nx_ * half_y, Ny_ * half_x);
        int max_freq_z2z = std::max(Nx_ * 2 * Ny_, Ny_ * 2 * Nx_);
        int max_freq = std::max(max_freq_z2d, max_freq_z2z);

        // For DCT-4, d_work_ is used as complex buffer, need 2x space
        bool has_dct4 = (type_x_ == CUDA_DCT_4 || type_y_ == CUDA_DCT_4);
        int work_size = has_dct4 ? 2 * max_freq_z2z : std::max(M_, max_freq_z2z);
        cudaMalloc(&d_work_, sizeof(double) * work_size);
        cudaMalloc(&d_temp_, sizeof(cufftDoubleComplex) * max_freq);
        cudaMalloc(&d_x1_, sizeof(double) * M_);

        M_padded_ = max_freq;

        // Create FFT plans for Y dimension
        if (type_y_ == CUDA_DCT_2) {
            int n_y[1] = {Ny_};
            int inembed_y[1] = {half_y};
            int onembed_y[1] = {Ny_};
            cufftPlanMany(&plan_y_, 1, n_y, inembed_y, 1, half_y,
                          onembed_y, 1, Ny_, CUFFT_Z2D, Nx_);
        } else if (type_y_ == CUDA_DCT_3) {
            int n_y[1] = {Ny_};
            int inembed_y[1] = {Ny_};
            int onembed_y[1] = {half_y};
            cufftPlanMany(&plan_y_, 1, n_y, inembed_y, 1, Ny_,
                          onembed_y, 1, half_y, CUFFT_D2Z, Nx_);
        } else if (type_y_ == CUDA_DCT_4) {
            cufftPlan1d(&plan_y_, 2 * Ny_, CUFFT_Z2Z, Nx_);
        }

        // Create FFT plans for X dimension
        if (type_x_ == CUDA_DCT_2) {
            int n_x[1] = {Nx_};
            int inembed_x[1] = {half_x};
            int onembed_x[1] = {Nx_};
            cufftPlanMany(&plan_x_, 1, n_x, inembed_x, 1, half_x,
                          onembed_x, 1, Nx_, CUFFT_Z2D, Ny_);
        } else if (type_x_ == CUDA_DCT_3) {
            int n_x[1] = {Nx_};
            int inembed_x[1] = {Nx_};
            int onembed_x[1] = {half_x};
            cufftPlanMany(&plan_x_, 1, n_x, inembed_x, 1, Nx_,
                          onembed_x, 1, half_x, CUFFT_D2Z, Ny_);
        } else if (type_x_ == CUDA_DCT_4) {
            cufftPlan1d(&plan_x_, 2 * Nx_, CUFFT_Z2Z, Ny_);
        }
    }

    initialized_ = true;
}

CudaDCT2D::~CudaDCT2D()
{
    if (plan_x_) cufftDestroy(plan_x_);
    if (plan_y_) cufftDestroy(plan_y_);
    if (d_work_) cudaFree(d_work_);
    if (d_temp_) cudaFree(d_temp_);
    if (d_x1_) cudaFree(d_x1_);
}

void CudaDCT2D::execute(double* d_data)
{
    int threads = 256;
    int blocks_out = (M_ + threads - 1) / threads;
    cufftDoubleComplex* d_freq = reinterpret_cast<cufftDoubleComplex*>(d_temp_);
    cufftDoubleComplex* d_complex_in = reinterpret_cast<cufftDoubleComplex*>(d_work_);
    cufftDoubleComplex* d_complex_out = reinterpret_cast<cufftDoubleComplex*>(d_temp_);

    // DCT-1 uses optimized 2D FFT approach
    if (type_x_ == CUDA_DCT_1 && type_y_ == CUDA_DCT_1) {
        int ext_size = 4 * Nx_ * Ny_;
        int blocks_ext = (ext_size + threads - 1) / threads;

        kernel_dct1_2d_mirror_extend<<<blocks_ext, threads>>>(d_data, d_work_, Nx_, Ny_);
        cudaDeviceSynchronize();

        cufftExecD2Z(plan_x_, d_work_, d_freq);
        cudaDeviceSynchronize();

        kernel_dct1_2d_extract<<<blocks_out, threads>>>(d_freq, d_data, Nx_, Ny_);
        cudaDeviceSynchronize();
        return;
    }

    // Per-dimension processing for DCT-2/3/4 (same or mixed types)

    // Step 1: Process Y dimension
    if (type_y_ == CUDA_DCT_2) {
        int half_y = Ny_ / 2 + 1;
        int total_preOp_y = Nx_ * half_y;
        int blocks_pre_y = (total_preOp_y + threads - 1) / threads;
        kernel_dct2_2d_preOp_y<<<blocks_pre_y, threads>>>(d_data, d_freq, Nx_, Ny_);
        cudaDeviceSynchronize();

        cufftExecZ2D(plan_y_, d_freq, d_work_);
        cudaDeviceSynchronize();

        kernel_dct2_2d_postOp_y<<<blocks_out, threads>>>(d_work_, d_x1_, Nx_, Ny_);
        cudaDeviceSynchronize();
    } else if (type_y_ == CUDA_DCT_3) {
        cudaMemcpy(d_work_, d_data, sizeof(double) * M_, cudaMemcpyDeviceToDevice);

        int total_preOp_y = Nx_ * (Ny_ / 2);
        int blocks_pre_y = (total_preOp_y + threads - 1) / threads;
        kernel_dct3_2d_preOp_y<<<blocks_pre_y, threads>>>(d_work_, Nx_, Ny_);
        cudaDeviceSynchronize();

        cufftExecD2Z(plan_y_, d_work_, d_freq);
        cudaDeviceSynchronize();

        kernel_dct3_2d_postOp_y<<<blocks_out, threads>>>(d_freq, d_x1_, Nx_, Ny_);
        cudaDeviceSynchronize();
    } else if (type_y_ == CUDA_DCT_4) {
        int total_y = Nx_ * 2 * Ny_;
        int blocks_y = (total_y + threads - 1) / threads;
        kernel_dct4_2d_preOp_y<<<blocks_y, threads>>>(d_data, d_complex_in, Nx_, Ny_);
        cudaDeviceSynchronize();

        cufftExecZ2Z(plan_y_, d_complex_in, d_complex_out, CUFFT_FORWARD);
        cudaDeviceSynchronize();

        kernel_dct4_2d_postOp_y<<<blocks_out, threads>>>(d_complex_out, d_x1_, Nx_, Ny_);
        cudaDeviceSynchronize();
    }

    // Step 2: Process X dimension
    if (type_x_ == CUDA_DCT_2) {
        int half_x = Nx_ / 2 + 1;
        int total_preOp_x = Ny_ * half_x;
        int blocks_pre_x = (total_preOp_x + threads - 1) / threads;
        kernel_dct2_2d_preOp_x<<<blocks_pre_x, threads>>>(d_x1_, d_freq, Nx_, Ny_);
        cudaDeviceSynchronize();

        cufftExecZ2D(plan_x_, d_freq, d_work_);
        cudaDeviceSynchronize();

        kernel_dct2_2d_postOp_x<<<blocks_out, threads>>>(d_work_, d_data, Nx_, Ny_);
        cudaDeviceSynchronize();
    } else if (type_x_ == CUDA_DCT_3) {
        kernel_dct3_2d_preOp_x<<<blocks_out, threads>>>(d_x1_, d_work_, Nx_, Ny_);
        cudaDeviceSynchronize();

        int total_preOp_x = Ny_ * (Nx_ / 2);
        int blocks_pre_x = (total_preOp_x + threads - 1) / threads;
        kernel_dct3_2d_preOp_x_inplace<<<blocks_pre_x, threads>>>(d_work_, Nx_, Ny_);
        cudaDeviceSynchronize();

        cufftExecD2Z(plan_x_, d_work_, d_freq);
        cudaDeviceSynchronize();

        kernel_dct3_2d_postOp_x<<<blocks_out, threads>>>(d_freq, d_data, Nx_, Ny_);
        cudaDeviceSynchronize();
    } else if (type_x_ == CUDA_DCT_4) {
        int total_x = Ny_ * 2 * Nx_;
        int blocks_x = (total_x + threads - 1) / threads;
        kernel_dct4_2d_preOp_x<<<blocks_x, threads>>>(d_x1_, d_complex_in, Nx_, Ny_);
        cudaDeviceSynchronize();

        cufftExecZ2Z(plan_x_, d_complex_in, d_complex_out, CUFFT_FORWARD);
        cudaDeviceSynchronize();

        kernel_dct4_2d_postOp_x<<<blocks_out, threads>>>(d_complex_out, d_data, Nx_, Ny_);
        cudaDeviceSynchronize();
    }
}

void CudaDCT2D::get_dims(int& nx, int& ny) const
{
    if (type_x_ == CUDA_DCT_1) {
        nx = Nx_ + 1;
        ny = Ny_ + 1;
    } else {
        nx = Nx_;
        ny = Ny_;
    }
}

void CudaDCT2D::get_types(CudaDCTType& type_x, CudaDCTType& type_y) const
{
    type_x = type_x_;
    type_y = type_y_;
}

double CudaDCT2D::get_normalization() const
{
    return 1.0 / (4.0 * Nx_ * Ny_);
}

//==============================================================================
// 3D DCT-2 Kernels (batched 1D DCT-2)
//==============================================================================

/**
 * @brief 3D DCT-2 preOp for Z dimension.
 */
__global__ void kernel_dct2_3d_preOp_z(
    const double* __restrict__ in,
    cufftDoubleComplex* __restrict__ out,
    int Nx, int Ny, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_z = Nz / 2 + 1;
    int total = Nx * Ny * half_z;

    if (idx < total) {
        int iz_half = idx % half_z;
        int iy = (idx / half_z) % Ny;
        int ix = idx / (half_z * Ny);

        int out_idx = (ix * Ny + iy) * half_z + iz_half;
        int in_base = (ix * Ny + iy) * Nz;

        if (iz_half == 0) {
            out[out_idx].x = in[in_base];
            out[out_idx].y = 0.0;
        } else if (iz_half == Nz / 2) {
            out[out_idx].x = in[in_base + Nz - 1];
            out[out_idx].y = 0.0;
        } else {
            int iz_even = 2 * iz_half;
            int iz_odd = iz_even - 1;
            out[out_idx].x = (in[in_base + iz_even] + in[in_base + iz_odd]) / 2.0;
            out[out_idx].y = -((in[in_base + iz_odd] - in[in_base + iz_even]) / 2.0);
        }
    }
}

/**
 * @brief 3D DCT-2 postOp for Z dimension.
 */
__global__ void kernel_dct2_3d_postOp_z(
    const double* __restrict__ in,
    double* __restrict__ out,
    int Nx, int Ny, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny * Nz;

    if (idx < total) {
        int iz = idx % Nz;
        int iy = (idx / Nz) % Ny;
        int ix = idx / (Nz * Ny);

        int in_base = (ix * Ny + iy) * Nz;

        if (iz == 0) {
            out[idx] = in[in_base] * 2.0;
        } else {
            double sina, cosa;
            sincos(iz * M_PI / (2.0 * Nz), &sina, &cosa);
            double Ta = in[in_base + iz] + in[in_base + (Nz - iz)];
            double Tb = in[in_base + iz] - in[in_base + (Nz - iz)];
            out[idx] = Ta * cosa + Tb * sina;
        }
    }
}

/**
 * @brief 3D DCT-2 preOp for Y dimension (data in XYZ layout).
 */
__global__ void kernel_dct2_3d_preOp_y(
    const double* __restrict__ in,
    cufftDoubleComplex* __restrict__ out,
    int Nx, int Ny, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_y = Ny / 2 + 1;
    int total = Nx * Nz * half_y;

    if (idx < total) {
        int iy_half = idx % half_y;
        int iz = (idx / half_y) % Nz;
        int ix = idx / (half_y * Nz);

        int out_idx = (ix * Nz + iz) * half_y + iy_half;

        if (iy_half == 0) {
            out[out_idx].x = in[(ix * Ny) * Nz + iz];
            out[out_idx].y = 0.0;
        } else if (iy_half == Ny / 2) {
            out[out_idx].x = in[(ix * Ny + Ny - 1) * Nz + iz];
            out[out_idx].y = 0.0;
        } else {
            int iy_even = 2 * iy_half;
            int iy_odd = iy_even - 1;
            double v_even = in[(ix * Ny + iy_even) * Nz + iz];
            double v_odd = in[(ix * Ny + iy_odd) * Nz + iz];
            out[out_idx].x = (v_even + v_odd) / 2.0;
            out[out_idx].y = -((v_odd - v_even) / 2.0);
        }
    }
}

/**
 * @brief 3D DCT-2 postOp for Y dimension.
 */
__global__ void kernel_dct2_3d_postOp_y(
    const double* __restrict__ in,
    double* __restrict__ out,
    int Nx, int Ny, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny * Nz;

    if (idx < total) {
        int iz = idx % Nz;
        int iy = (idx / Nz) % Ny;
        int ix = idx / (Nz * Ny);

        // in is in [Nx][Nz][Ny] layout
        int in_base = (ix * Nz + iz) * Ny;

        if (iy == 0) {
            out[idx] = in[in_base] * 2.0;
        } else {
            double sina, cosa;
            sincos(iy * M_PI / (2.0 * Ny), &sina, &cosa);
            double Ta = in[in_base + iy] + in[in_base + (Ny - iy)];
            double Tb = in[in_base + iy] - in[in_base + (Ny - iy)];
            out[idx] = Ta * cosa + Tb * sina;
        }
    }
}

/**
 * @brief 3D DCT-2 preOp for X dimension (data in XYZ layout).
 */
__global__ void kernel_dct2_3d_preOp_x(
    const double* __restrict__ in,
    cufftDoubleComplex* __restrict__ out,
    int Nx, int Ny, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_x = Nx / 2 + 1;
    int total = Ny * Nz * half_x;

    if (idx < total) {
        int ix_half = idx % half_x;
        int iz = (idx / half_x) % Nz;
        int iy = idx / (half_x * Nz);

        int out_idx = (iy * Nz + iz) * half_x + ix_half;

        if (ix_half == 0) {
            out[out_idx].x = in[iy * Nz + iz];
            out[out_idx].y = 0.0;
        } else if (ix_half == Nx / 2) {
            out[out_idx].x = in[((Nx - 1) * Ny + iy) * Nz + iz];
            out[out_idx].y = 0.0;
        } else {
            int ix_even = 2 * ix_half;
            int ix_odd = ix_even - 1;
            double v_even = in[(ix_even * Ny + iy) * Nz + iz];
            double v_odd = in[(ix_odd * Ny + iy) * Nz + iz];
            out[out_idx].x = (v_even + v_odd) / 2.0;
            out[out_idx].y = -((v_odd - v_even) / 2.0);
        }
    }
}

/**
 * @brief 3D DCT-2 postOp for X dimension.
 */
__global__ void kernel_dct2_3d_postOp_x(
    const double* __restrict__ in,
    double* __restrict__ out,
    int Nx, int Ny, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny * Nz;

    if (idx < total) {
        int iz = idx % Nz;
        int iy = (idx / Nz) % Ny;
        int ix = idx / (Nz * Ny);

        // in is in [Ny][Nz][Nx] layout
        int in_base = (iy * Nz + iz) * Nx;

        if (ix == 0) {
            out[idx] = in[in_base] * 2.0;
        } else {
            double sina, cosa;
            sincos(ix * M_PI / (2.0 * Nx), &sina, &cosa);
            double Ta = in[in_base + ix] + in[in_base + (Nx - ix)];
            double Tb = in[in_base + ix] - in[in_base + (Nx - ix)];
            out[idx] = Ta * cosa + Tb * sina;
        }
    }
}

//==============================================================================
// 3D DCT-3 Kernels (batched 1D DCT-3)
//==============================================================================

/**
 * @brief 3D DCT-3 preOp for Z dimension.
 */
__global__ void kernel_dct3_3d_preOp_z(
    double* __restrict__ data,
    int Nx, int Ny, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny * (Nz / 2);

    if (idx < total) {
        int iz = idx % (Nz / 2);
        int iy = (idx / (Nz / 2)) % Ny;
        int ix = idx / ((Nz / 2) * Ny);

        int base = (ix * Ny + iy) * Nz;
        int iz1 = iz + 1;
        int iz2 = Nz - iz - 1;

        double sina, cosa;
        sincos(iz1 * M_PI / (2.0 * Nz), &sina, &cosa);
        double Ta = data[base + iz1] + data[base + iz2];
        double Tb = data[base + iz1] - data[base + iz2];

        data[base + iz1] = Ta * sina + Tb * cosa;
        data[base + iz2] = Ta * cosa - Tb * sina;
    }
}

/**
 * @brief 3D DCT-3 postOp for Z dimension.
 */
__global__ void kernel_dct3_3d_postOp_z(
    const cufftDoubleComplex* __restrict__ in,
    double* __restrict__ out,
    int Nx, int Ny, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny * Nz;

    if (idx < total) {
        int iz = idx % Nz;
        int iy = (idx / Nz) % Ny;
        int ix = idx / (Nz * Ny);

        int half_z = Nz / 2 + 1;
        int in_base = (ix * Ny + iy) * half_z;

        if (iz == 0) {
            out[idx] = in[in_base].x;
        } else {
            int half_idx = (iz + 1) / 2;
            double re = in[in_base + half_idx].x;
            double im = in[in_base + half_idx].y;

            if (iz % 2 == 1) {
                out[idx] = re - im;
            } else {
                out[idx] = re + im;
            }
        }
    }
}

/**
 * @brief Transpose 3D array from [Nx][Ny][Nz] to [Nx][Nz][Ny].
 */
__global__ void kernel_transpose_xyz_to_xzy(
    const double* __restrict__ in,
    double* __restrict__ out,
    int Nx, int Ny, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny * Nz;

    if (idx < total) {
        int iz = idx % Nz;
        int iy = (idx / Nz) % Ny;
        int ix = idx / (Nz * Ny);

        // Input: [Nx][Ny][Nz] -> in[ix*Ny*Nz + iy*Nz + iz]
        // Output: [Nx][Nz][Ny] -> out[ix*Nz*Ny + iz*Ny + iy]
        out[ix * Nz * Ny + iz * Ny + iy] = in[idx];
    }
}

/**
 * @brief Transpose 3D array from [Nx][Ny][Nz] to [Ny][Nz][Nx].
 */
__global__ void kernel_transpose_xyz_to_yzx(
    const double* __restrict__ in,
    double* __restrict__ out,
    int Nx, int Ny, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny * Nz;

    if (idx < total) {
        int iz = idx % Nz;
        int iy = (idx / Nz) % Ny;
        int ix = idx / (Nz * Ny);

        // Input: [Nx][Ny][Nz] -> in[ix*Ny*Nz + iy*Nz + iz]
        // Output: [Ny][Nz][Nx] -> out[iy*Nz*Nx + iz*Nx + ix]
        out[iy * Nz * Nx + iz * Nx + ix] = in[idx];
    }
}

/**
 * @brief 3D DCT-3 preOp for Y dimension on [Nx][Nz][Ny] transposed layout.
 */
__global__ void kernel_dct3_3d_preOp_y_transposed(
    double* __restrict__ data,
    int Nx, int Ny, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Nz * (Ny / 2);

    if (idx < total) {
        int iy = idx % (Ny / 2);
        int iz = (idx / (Ny / 2)) % Nz;
        int ix = idx / ((Ny / 2) * Nz);

        int iy1 = iy + 1;
        int iy2 = Ny - iy - 1;
        // Data is in [Nx][Nz][Ny] layout
        int base = (ix * Nz + iz) * Ny;

        double sina, cosa;
        sincos(iy1 * M_PI / (2.0 * Ny), &sina, &cosa);
        double Ta = data[base + iy1] + data[base + iy2];
        double Tb = data[base + iy1] - data[base + iy2];

        data[base + iy1] = Ta * sina + Tb * cosa;
        data[base + iy2] = Ta * cosa - Tb * sina;
    }
}

/**
 * @brief 3D DCT-3 preOp for Y dimension.
 * @deprecated Use kernel_dct3_3d_preOp_y_transposed with proper transpose instead.
 */
__global__ void kernel_dct3_3d_preOp_y(
    double* __restrict__ data,
    int Nx, int Ny, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Nz * (Ny / 2);

    if (idx < total) {
        int iy = idx % (Ny / 2);
        int iz = (idx / (Ny / 2)) % Nz;
        int ix = idx / ((Ny / 2) * Nz);

        int iy1 = iy + 1;
        int iy2 = Ny - iy - 1;

        double sina, cosa;
        sincos(iy1 * M_PI / (2.0 * Ny), &sina, &cosa);
        double Ta = data[(ix * Ny + iy1) * Nz + iz] + data[(ix * Ny + iy2) * Nz + iz];
        double Tb = data[(ix * Ny + iy1) * Nz + iz] - data[(ix * Ny + iy2) * Nz + iz];

        data[(ix * Ny + iy1) * Nz + iz] = Ta * sina + Tb * cosa;
        data[(ix * Ny + iy2) * Nz + iz] = Ta * cosa - Tb * sina;
    }
}

/**
 * @brief 3D DCT-3 postOp for Y dimension.
 */
__global__ void kernel_dct3_3d_postOp_y(
    const cufftDoubleComplex* __restrict__ in,
    double* __restrict__ out,
    int Nx, int Ny, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny * Nz;

    if (idx < total) {
        int iz = idx % Nz;
        int iy = (idx / Nz) % Ny;
        int ix = idx / (Nz * Ny);

        int half_y = Ny / 2 + 1;
        // in is in [Nx][Nz][half_y] layout
        int in_base = (ix * Nz + iz) * half_y;

        if (iy == 0) {
            out[idx] = in[in_base].x;
        } else {
            int half_idx = (iy + 1) / 2;
            double re = in[in_base + half_idx].x;
            double im = in[in_base + half_idx].y;

            if (iy % 2 == 1) {
                out[idx] = re - im;
            } else {
                out[idx] = re + im;
            }
        }
    }
}

/**
 * @brief 3D DCT-3 preOp for X dimension on [Ny][Nz][Nx] transposed layout.
 */
__global__ void kernel_dct3_3d_preOp_x_transposed(
    double* __restrict__ data,
    int Nx, int Ny, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Ny * Nz * (Nx / 2);

    if (idx < total) {
        int ix = idx % (Nx / 2);
        int iz = (idx / (Nx / 2)) % Nz;
        int iy = idx / ((Nx / 2) * Nz);

        int ix1 = ix + 1;
        int ix2 = Nx - ix - 1;
        // Data is in [Ny][Nz][Nx] layout
        int base = (iy * Nz + iz) * Nx;

        double sina, cosa;
        sincos(ix1 * M_PI / (2.0 * Nx), &sina, &cosa);
        double Ta = data[base + ix1] + data[base + ix2];
        double Tb = data[base + ix1] - data[base + ix2];

        data[base + ix1] = Ta * sina + Tb * cosa;
        data[base + ix2] = Ta * cosa - Tb * sina;
    }
}

/**
 * @brief 3D DCT-3 preOp for X dimension.
 * @deprecated Use kernel_dct3_3d_preOp_x_transposed with proper transpose instead.
 */
__global__ void kernel_dct3_3d_preOp_x(
    double* __restrict__ data,
    int Nx, int Ny, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Ny * Nz * (Nx / 2);

    if (idx < total) {
        int ix = idx % (Nx / 2);
        int iz = (idx / (Nx / 2)) % Nz;
        int iy = idx / ((Nx / 2) * Nz);

        int ix1 = ix + 1;
        int ix2 = Nx - ix - 1;

        double sina, cosa;
        sincos(ix1 * M_PI / (2.0 * Nx), &sina, &cosa);
        double Ta = data[(ix1 * Ny + iy) * Nz + iz] + data[(ix2 * Ny + iy) * Nz + iz];
        double Tb = data[(ix1 * Ny + iy) * Nz + iz] - data[(ix2 * Ny + iy) * Nz + iz];

        data[(ix1 * Ny + iy) * Nz + iz] = Ta * sina + Tb * cosa;
        data[(ix2 * Ny + iy) * Nz + iz] = Ta * cosa - Tb * sina;
    }
}

/**
 * @brief 3D DCT-3 postOp for X dimension.
 */
__global__ void kernel_dct3_3d_postOp_x(
    const cufftDoubleComplex* __restrict__ in,
    double* __restrict__ out,
    int Nx, int Ny, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny * Nz;

    if (idx < total) {
        int iz = idx % Nz;
        int iy = (idx / Nz) % Ny;
        int ix = idx / (Nz * Ny);

        int half_x = Nx / 2 + 1;
        // in is in [Ny][Nz][half_x] layout
        int in_base = (iy * Nz + iz) * half_x;

        if (ix == 0) {
            out[idx] = in[in_base].x;
        } else {
            int half_idx = (ix + 1) / 2;
            double re = in[in_base + half_idx].x;
            double im = in[in_base + half_idx].y;

            if (ix % 2 == 1) {
                out[idx] = re - im;
            } else {
                out[idx] = re + im;
            }
        }
    }
}

//==============================================================================
// 3D DCT-4 Kernels
// Apply 1D DCT-4 along each dimension using batched 2N Z2Z FFT
//==============================================================================

/**
 * @brief 3D DCT-4 preOp for Z dimension: real input to complex with phase rotation.
 * z[n] = x[n] * exp(-iπ(2n+1)/(4N)) for n=0..N-1, z[n]=0 for n=N..2N-1
 */
__global__ void kernel_dct4_3d_preOp_z(
    const double* __restrict__ in,
    cufftDoubleComplex* __restrict__ out,
    int Nx, int Ny, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny * 2 * Nz;

    if (idx < total) {
        int iz = idx % (2 * Nz);
        int iy = (idx / (2 * Nz)) % Ny;
        int ix = idx / (2 * Nz * Ny);

        if (iz < Nz) {
            int in_idx = (ix * Ny + iy) * Nz + iz;
            double angle = -M_PI * (2 * iz + 1) / (4.0 * Nz);
            out[idx].x = in[in_idx] * cos(angle);
            out[idx].y = in[in_idx] * sin(angle);
        } else {
            out[idx].x = 0.0;
            out[idx].y = 0.0;
        }
    }
}

/**
 * @brief 3D DCT-4 postOp for Z dimension: complex to real with phase rotation.
 */
__global__ void kernel_dct4_3d_postOp_z(
    const cufftDoubleComplex* __restrict__ in,
    double* __restrict__ out,
    int Nx, int Ny, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny * Nz;

    if (idx < total) {
        int iz = idx % Nz;
        int iy = (idx / Nz) % Ny;
        int ix = idx / (Nz * Ny);

        int in_idx = (ix * Ny + iy) * 2 * Nz + iz;
        double angle = -M_PI * iz / (2.0 * Nz);
        double cosa = cos(angle);
        double sina = sin(angle);

        double re = in[in_idx].x * cosa - in[in_idx].y * sina;
        out[idx] = 2.0 * re;
    }
}

/**
 * @brief 3D DCT-4 preOp for Y dimension: transpose XYZ -> XZY and apply phase.
 */
__global__ void kernel_dct4_3d_preOp_y(
    const double* __restrict__ in,
    cufftDoubleComplex* __restrict__ out,
    int Nx, int Ny, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Nz * 2 * Ny;

    if (idx < total) {
        int iy = idx % (2 * Ny);
        int iz = (idx / (2 * Ny)) % Nz;
        int ix = idx / (2 * Ny * Nz);

        if (iy < Ny) {
            int in_idx = (ix * Ny + iy) * Nz + iz;
            double angle = -M_PI * (2 * iy + 1) / (4.0 * Ny);
            out[idx].x = in[in_idx] * cos(angle);
            out[idx].y = in[in_idx] * sin(angle);
        } else {
            out[idx].x = 0.0;
            out[idx].y = 0.0;
        }
    }
}

/**
 * @brief 3D DCT-4 postOp for Y dimension: apply phase and transpose XZY -> XYZ.
 */
__global__ void kernel_dct4_3d_postOp_y(
    const cufftDoubleComplex* __restrict__ in,
    double* __restrict__ out,
    int Nx, int Ny, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny * Nz;

    if (idx < total) {
        int iz = idx % Nz;
        int iy = (idx / Nz) % Ny;
        int ix = idx / (Nz * Ny);

        int in_idx = (ix * Nz + iz) * 2 * Ny + iy;
        double angle = -M_PI * iy / (2.0 * Ny);
        double cosa = cos(angle);
        double sina = sin(angle);

        double re = in[in_idx].x * cosa - in[in_idx].y * sina;
        out[idx] = 2.0 * re;
    }
}

/**
 * @brief 3D DCT-4 preOp for X dimension: transpose XYZ -> YZX and apply phase.
 */
__global__ void kernel_dct4_3d_preOp_x(
    const double* __restrict__ in,
    cufftDoubleComplex* __restrict__ out,
    int Nx, int Ny, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Ny * Nz * 2 * Nx;

    if (idx < total) {
        int ix = idx % (2 * Nx);
        int iz = (idx / (2 * Nx)) % Nz;
        int iy = idx / (2 * Nx * Nz);

        if (ix < Nx) {
            int in_idx = (ix * Ny + iy) * Nz + iz;
            double angle = -M_PI * (2 * ix + 1) / (4.0 * Nx);
            out[idx].x = in[in_idx] * cos(angle);
            out[idx].y = in[in_idx] * sin(angle);
        } else {
            out[idx].x = 0.0;
            out[idx].y = 0.0;
        }
    }
}

/**
 * @brief 3D DCT-4 postOp for X dimension: apply phase and transpose YZX -> XYZ.
 */
__global__ void kernel_dct4_3d_postOp_x(
    const cufftDoubleComplex* __restrict__ in,
    double* __restrict__ out,
    int Nx, int Ny, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny * Nz;

    if (idx < total) {
        int iz = idx % Nz;
        int iy = (idx / Nz) % Ny;
        int ix = idx / (Nz * Ny);

        int in_idx = (iy * Nz + iz) * 2 * Nx + ix;
        double angle = -M_PI * ix / (2.0 * Nx);
        double cosa = cos(angle);
        double sina = sin(angle);

        double re = in[in_idx].x * cosa - in[in_idx].y * sina;
        out[idx] = 2.0 * re;
    }
}

//==============================================================================
// CudaDCT3D class implementation
//==============================================================================

CudaDCT3D::CudaDCT3D(int Nx, int Ny, int Nz, CudaDCTType type)
    : Nx_(Nx), Ny_(Ny), Nz_(Nz), type_x_(type), type_y_(type), type_z_(type),
      plan_x_(0), plan_y_(0), plan_z_(0),
      d_work_(nullptr), d_temp_(nullptr), d_x1_(nullptr),
      initialized_(false)
{
    init();
}

CudaDCT3D::CudaDCT3D(int Nx, int Ny, int Nz, CudaDCTType type_x, CudaDCTType type_y, CudaDCTType type_z)
    : Nx_(Nx), Ny_(Ny), Nz_(Nz), type_x_(type_x), type_y_(type_y), type_z_(type_z),
      plan_x_(0), plan_y_(0), plan_z_(0),
      d_work_(nullptr), d_temp_(nullptr), d_x1_(nullptr),
      initialized_(false)
{
    // DCT-1 cannot be mixed with other types
    bool has_dct1 = (type_x == CUDA_DCT_1 || type_y == CUDA_DCT_1 || type_z == CUDA_DCT_1);
    bool all_dct1 = (type_x == CUDA_DCT_1 && type_y == CUDA_DCT_1 && type_z == CUDA_DCT_1);
    if (has_dct1 && !all_dct1) {
        throw std::runtime_error("CudaDCT3D: DCT-1 cannot be mixed with other types");
    }
    init();
}

void CudaDCT3D::init()
{
    // DCT-1 uses optimized 3D FFT approach
    if (type_x_ == CUDA_DCT_1 && type_y_ == CUDA_DCT_1 && type_z_ == CUDA_DCT_1) {
        M_ = (Nx_ + 1) * (Ny_ + 1) * (Nz_ + 1);

        int ext_size = 8 * Nx_ * Ny_ * Nz_;
        int freq_size = 2 * Nx_ * 2 * Ny_ * (Nz_ + 1);

        cudaMalloc(&d_work_, sizeof(double) * ext_size);
        cudaMalloc(&d_temp_, sizeof(cufftDoubleComplex) * freq_size);

        M_padded_ = ext_size;

        cufftPlan3d(&plan_x_, 2 * Nx_, 2 * Ny_, 2 * Nz_, CUFFT_D2Z);
    } else {
        // Mixed or same type (DCT-2/3/4): use per-dimension processing
        M_ = Nx_ * Ny_ * Nz_;

        int half_z = Nz_ / 2 + 1;
        int half_y = Ny_ / 2 + 1;
        int half_x = Nx_ / 2 + 1;

        int max_freq_z2d = std::max({Nx_ * Ny_ * half_z, Nx_ * Nz_ * half_y, Ny_ * Nz_ * half_x});
        int max_freq_z2z = std::max({Nx_ * Ny_ * 2 * Nz_, Nx_ * Nz_ * 2 * Ny_, Ny_ * Nz_ * 2 * Nx_});
        int max_freq = std::max(max_freq_z2d, max_freq_z2z);

        // For DCT-4, d_work_ is used as complex buffer, need 2x space
        bool has_dct4 = (type_x_ == CUDA_DCT_4 || type_y_ == CUDA_DCT_4 || type_z_ == CUDA_DCT_4);
        int work_size = has_dct4 ? 2 * max_freq_z2z : std::max(M_, max_freq_z2z);
        cudaMalloc(&d_work_, sizeof(double) * work_size);
        cudaMalloc(&d_temp_, sizeof(cufftDoubleComplex) * max_freq);
        cudaMalloc(&d_x1_, sizeof(double) * M_);

        M_padded_ = max_freq;

        // Create FFT plans for Z dimension
        if (type_z_ == CUDA_DCT_2) {
            int n_z[1] = {Nz_};
            int inembed_z[1] = {half_z};
            int onembed_z[1] = {Nz_};
            cufftPlanMany(&plan_z_, 1, n_z, inembed_z, 1, half_z,
                          onembed_z, 1, Nz_, CUFFT_Z2D, Nx_ * Ny_);
        } else if (type_z_ == CUDA_DCT_3) {
            int n_z[1] = {Nz_};
            int inembed_z[1] = {Nz_};
            int onembed_z[1] = {half_z};
            cufftPlanMany(&plan_z_, 1, n_z, inembed_z, 1, Nz_,
                          onembed_z, 1, half_z, CUFFT_D2Z, Nx_ * Ny_);
        } else if (type_z_ == CUDA_DCT_4) {
            cufftPlan1d(&plan_z_, 2 * Nz_, CUFFT_Z2Z, Nx_ * Ny_);
        }

        // Create FFT plans for Y dimension
        if (type_y_ == CUDA_DCT_2) {
            int n_y[1] = {Ny_};
            int inembed_y[1] = {half_y};
            int onembed_y[1] = {Ny_};
            cufftPlanMany(&plan_y_, 1, n_y, inembed_y, 1, half_y,
                          onembed_y, 1, Ny_, CUFFT_Z2D, Nx_ * Nz_);
        } else if (type_y_ == CUDA_DCT_3) {
            int n_y[1] = {Ny_};
            int inembed_y[1] = {Ny_};
            int onembed_y[1] = {half_y};
            cufftPlanMany(&plan_y_, 1, n_y, inembed_y, 1, Ny_,
                          onembed_y, 1, half_y, CUFFT_D2Z, Nx_ * Nz_);
        } else if (type_y_ == CUDA_DCT_4) {
            cufftPlan1d(&plan_y_, 2 * Ny_, CUFFT_Z2Z, Nx_ * Nz_);
        }

        // Create FFT plans for X dimension
        if (type_x_ == CUDA_DCT_2) {
            int n_x[1] = {Nx_};
            int inembed_x[1] = {half_x};
            int onembed_x[1] = {Nx_};
            cufftPlanMany(&plan_x_, 1, n_x, inembed_x, 1, half_x,
                          onembed_x, 1, Nx_, CUFFT_Z2D, Ny_ * Nz_);
        } else if (type_x_ == CUDA_DCT_3) {
            int n_x[1] = {Nx_};
            int inembed_x[1] = {Nx_};
            int onembed_x[1] = {half_x};
            cufftPlanMany(&plan_x_, 1, n_x, inembed_x, 1, Nx_,
                          onembed_x, 1, half_x, CUFFT_D2Z, Ny_ * Nz_);
        } else if (type_x_ == CUDA_DCT_4) {
            cufftPlan1d(&plan_x_, 2 * Nx_, CUFFT_Z2Z, Ny_ * Nz_);
        }
    }

    initialized_ = true;
}

CudaDCT3D::~CudaDCT3D()
{
    if (plan_x_) cufftDestroy(plan_x_);
    if (plan_y_) cufftDestroy(plan_y_);
    if (plan_z_) cufftDestroy(plan_z_);
    if (d_work_) cudaFree(d_work_);
    if (d_temp_) cudaFree(d_temp_);
    if (d_x1_) cudaFree(d_x1_);
}

void CudaDCT3D::execute(double* d_data)
{
    int threads = 256;
    int blocks_out = (M_ + threads - 1) / threads;
    cufftDoubleComplex* d_freq = reinterpret_cast<cufftDoubleComplex*>(d_temp_);
    cufftDoubleComplex* d_complex_in = reinterpret_cast<cufftDoubleComplex*>(d_work_);
    cufftDoubleComplex* d_complex_out = reinterpret_cast<cufftDoubleComplex*>(d_temp_);

    // DCT-1 uses optimized 3D FFT approach
    if (type_x_ == CUDA_DCT_1 && type_y_ == CUDA_DCT_1 && type_z_ == CUDA_DCT_1) {
        int ext_size = 8 * Nx_ * Ny_ * Nz_;
        int blocks_ext = (ext_size + threads - 1) / threads;

        kernel_dct1_3d_mirror_extend<<<blocks_ext, threads>>>(d_data, d_work_, Nx_, Ny_, Nz_);
        cudaDeviceSynchronize();

        cufftExecD2Z(plan_x_, d_work_, d_freq);
        cudaDeviceSynchronize();

        kernel_dct1_3d_extract<<<blocks_out, threads>>>(d_freq, d_data, Nx_, Ny_, Nz_);
        cudaDeviceSynchronize();
        return;
    }

    // Per-dimension processing for DCT-2/3/4 (same or mixed types)

    // Step 1: Process Z dimension
    if (type_z_ == CUDA_DCT_2) {
        int half_z = Nz_ / 2 + 1;
        int total_preOp_z = Nx_ * Ny_ * half_z;
        int blocks_pre_z = (total_preOp_z + threads - 1) / threads;
        kernel_dct2_3d_preOp_z<<<blocks_pre_z, threads>>>(d_data, d_freq, Nx_, Ny_, Nz_);
        cudaDeviceSynchronize();

        cufftExecZ2D(plan_z_, d_freq, d_work_);
        cudaDeviceSynchronize();

        kernel_dct2_3d_postOp_z<<<blocks_out, threads>>>(d_work_, d_x1_, Nx_, Ny_, Nz_);
        cudaDeviceSynchronize();
    } else if (type_z_ == CUDA_DCT_3) {
        cudaMemcpy(d_work_, d_data, sizeof(double) * M_, cudaMemcpyDeviceToDevice);

        int total_preOp_z = Nx_ * Ny_ * (Nz_ / 2);
        int blocks_pre_z = (total_preOp_z + threads - 1) / threads;
        kernel_dct3_3d_preOp_z<<<blocks_pre_z, threads>>>(d_work_, Nx_, Ny_, Nz_);
        cudaDeviceSynchronize();

        cufftExecD2Z(plan_z_, d_work_, d_freq);
        cudaDeviceSynchronize();

        kernel_dct3_3d_postOp_z<<<blocks_out, threads>>>(d_freq, d_x1_, Nx_, Ny_, Nz_);
        cudaDeviceSynchronize();
    } else if (type_z_ == CUDA_DCT_4) {
        int total_z = Nx_ * Ny_ * 2 * Nz_;
        int blocks_z = (total_z + threads - 1) / threads;
        kernel_dct4_3d_preOp_z<<<blocks_z, threads>>>(d_data, d_complex_in, Nx_, Ny_, Nz_);
        cudaDeviceSynchronize();

        cufftExecZ2Z(plan_z_, d_complex_in, d_complex_out, CUFFT_FORWARD);
        cudaDeviceSynchronize();

        kernel_dct4_3d_postOp_z<<<blocks_out, threads>>>(d_complex_out, d_x1_, Nx_, Ny_, Nz_);
        cudaDeviceSynchronize();
    }

    // Step 2: Process Y dimension
    if (type_y_ == CUDA_DCT_2) {
        int half_y = Ny_ / 2 + 1;
        int total_preOp_y = Nx_ * Nz_ * half_y;
        int blocks_pre_y = (total_preOp_y + threads - 1) / threads;
        kernel_dct2_3d_preOp_y<<<blocks_pre_y, threads>>>(d_x1_, d_freq, Nx_, Ny_, Nz_);
        cudaDeviceSynchronize();

        cufftExecZ2D(plan_y_, d_freq, d_work_);
        cudaDeviceSynchronize();

        kernel_dct2_3d_postOp_y<<<blocks_out, threads>>>(d_work_, d_x1_, Nx_, Ny_, Nz_);
        cudaDeviceSynchronize();
    } else if (type_y_ == CUDA_DCT_3) {
        kernel_transpose_xyz_to_xzy<<<blocks_out, threads>>>(d_x1_, d_work_, Nx_, Ny_, Nz_);
        cudaDeviceSynchronize();

        int total_preOp_y = Nx_ * Nz_ * (Ny_ / 2);
        int blocks_pre_y = (total_preOp_y + threads - 1) / threads;
        kernel_dct3_3d_preOp_y_transposed<<<blocks_pre_y, threads>>>(d_work_, Nx_, Ny_, Nz_);
        cudaDeviceSynchronize();

        cufftExecD2Z(plan_y_, d_work_, d_freq);
        cudaDeviceSynchronize();

        kernel_dct3_3d_postOp_y<<<blocks_out, threads>>>(d_freq, d_x1_, Nx_, Ny_, Nz_);
        cudaDeviceSynchronize();
    } else if (type_y_ == CUDA_DCT_4) {
        int total_y = Nx_ * Nz_ * 2 * Ny_;
        int blocks_y = (total_y + threads - 1) / threads;
        kernel_dct4_3d_preOp_y<<<blocks_y, threads>>>(d_x1_, d_complex_in, Nx_, Ny_, Nz_);
        cudaDeviceSynchronize();

        cufftExecZ2Z(plan_y_, d_complex_in, d_complex_out, CUFFT_FORWARD);
        cudaDeviceSynchronize();

        kernel_dct4_3d_postOp_y<<<blocks_out, threads>>>(d_complex_out, d_x1_, Nx_, Ny_, Nz_);
        cudaDeviceSynchronize();
    }

    // Step 3: Process X dimension
    if (type_x_ == CUDA_DCT_2) {
        int half_x = Nx_ / 2 + 1;
        int total_preOp_x = Ny_ * Nz_ * half_x;
        int blocks_pre_x = (total_preOp_x + threads - 1) / threads;
        kernel_dct2_3d_preOp_x<<<blocks_pre_x, threads>>>(d_x1_, d_freq, Nx_, Ny_, Nz_);
        cudaDeviceSynchronize();

        cufftExecZ2D(plan_x_, d_freq, d_work_);
        cudaDeviceSynchronize();

        kernel_dct2_3d_postOp_x<<<blocks_out, threads>>>(d_work_, d_data, Nx_, Ny_, Nz_);
        cudaDeviceSynchronize();
    } else if (type_x_ == CUDA_DCT_3) {
        kernel_transpose_xyz_to_yzx<<<blocks_out, threads>>>(d_x1_, d_work_, Nx_, Ny_, Nz_);
        cudaDeviceSynchronize();

        int total_preOp_x = Ny_ * Nz_ * (Nx_ / 2);
        int blocks_pre_x = (total_preOp_x + threads - 1) / threads;
        kernel_dct3_3d_preOp_x_transposed<<<blocks_pre_x, threads>>>(d_work_, Nx_, Ny_, Nz_);
        cudaDeviceSynchronize();

        cufftExecD2Z(plan_x_, d_work_, d_freq);
        cudaDeviceSynchronize();

        kernel_dct3_3d_postOp_x<<<blocks_out, threads>>>(d_freq, d_data, Nx_, Ny_, Nz_);
        cudaDeviceSynchronize();
    } else if (type_x_ == CUDA_DCT_4) {
        int total_x = Ny_ * Nz_ * 2 * Nx_;
        int blocks_x = (total_x + threads - 1) / threads;
        kernel_dct4_3d_preOp_x<<<blocks_x, threads>>>(d_x1_, d_complex_in, Nx_, Ny_, Nz_);
        cudaDeviceSynchronize();

        cufftExecZ2Z(plan_x_, d_complex_in, d_complex_out, CUFFT_FORWARD);
        cudaDeviceSynchronize();

        kernel_dct4_3d_postOp_x<<<blocks_out, threads>>>(d_complex_out, d_data, Nx_, Ny_, Nz_);
        cudaDeviceSynchronize();
    }
}

void CudaDCT3D::get_dims(int& nx, int& ny, int& nz) const
{
    if (type_x_ == CUDA_DCT_1) {
        nx = Nx_ + 1;
        ny = Ny_ + 1;
        nz = Nz_ + 1;
    } else {
        nx = Nx_;
        ny = Ny_;
        nz = Nz_;
    }
}

void CudaDCT3D::get_types(CudaDCTType& type_x, CudaDCTType& type_y, CudaDCTType& type_z) const
{
    type_x = type_x_;
    type_y = type_y_;
    type_z = type_z_;
}

double CudaDCT3D::get_normalization() const
{
    return 1.0 / (8.0 * Nx_ * Ny_ * Nz_);
}
