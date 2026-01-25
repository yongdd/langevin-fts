/**
 * @file CudaRealTransform.cu
 * @brief Unified CUDA Real Transform (DCT/DST) implementation.
 *
 * Combines DCT and DST types 1-4 for 1D, 2D, and 3D arrays.
 */

#include "CudaRealTransform.h"
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

static int getFFTSize(CudaTransformType type, int N)
{
    switch (type) {
        case CUDA_DCT_1:
            return N + 1;  // DCT-1 needs N+2 work buffer size, FFT size is N-1
        case CUDA_DCT_2:
        case CUDA_DCT_3:
            return N;
        case CUDA_DCT_4:
        case CUDA_DST_4:
            return 2 * N;
        case CUDA_DST_1:
            return N + 3;  // FCT/FST: FFT size N+1, buffer N+3
        case CUDA_DST_2:
        case CUDA_DST_3:
            return N + 2;  // FCT/FST: FFT size N, buffer N+2
        default:
            return N;
    }
}

static bool usesZ2Z(CudaTransformType type)
{
    return type == CUDA_DCT_4 || type == CUDA_DST_4;
}

static double getNormFactor(CudaTransformType type, int N)
{
    switch (type) {
        case CUDA_DCT_1:
            return 2.0 * (N - 1);
        case CUDA_DCT_2:
        case CUDA_DCT_3:
        case CUDA_DCT_4:
        case CUDA_DST_2:
        case CUDA_DST_3:
        case CUDA_DST_4:
            return 2.0 * N;
        case CUDA_DST_1:
            return 2.0 * (N + 1);
        default:
            return 1.0;
    }
}

//==============================================================================
// 1D DCT Kernels
//==============================================================================

__global__ void kernel_dct1_preOp(double* pin, double* px1, int N, int nThread)
{
    extern __shared__ double sh_in[];
    int itx = threadIdx.x;

    if (itx < N / 2) {
        sh_in[itx] = pin[itx * 2 + 1];
    } else {
        sh_in[itx] = 0;
    }
    __syncthreads();

    if (itx < N / 2) {
        if (itx == 0) {
            pin[N + 1] = 0;
            pin[1] = 0;
        } else {
            pin[itx * 2 + 1] = sh_in[itx - 1] - sh_in[itx];
        }
    }
    __syncthreads();

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

__global__ void kernel_dct1_postOp(double* pin, double* px1, int N)
{
    int itx = threadIdx.x;

    __syncthreads();

    if (itx < N / 2 + 1) {
        if (itx != 0) {
            double sitx = pin[itx];
            double sNitx = pin[N - itx];
            double Ta = (sitx + sNitx) * 0.5;
            double Tb = (sitx - sNitx) * 0.25 / sin(itx * M_PI / N);
            pin[itx] = (Ta - Tb);
            pin[N - itx] = (Ta + Tb);
        } else {
            double sh0 = pin[0] * 0.5;
            pin[0] = (sh0 + px1[0]) * 2.0;
            pin[N] = (sh0 - px1[0]) * 2.0;
        }
    }
}

__global__ void kernel_dct2_preOp(double* data, int N)
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

__global__ void kernel_dct2_postOp(double* data, int N)
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

__global__ void kernel_dct3_preOp(double* data, int N)
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

__global__ void kernel_dct3_postOp(double* data, int N)
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
        double re = in[k].x * cosa - in[k].y * sina;
        out[k] = 2.0 * re;
    }
}

//==============================================================================
// 1D DST Kernels (FCT/FST style, Makhoul 1980)
//==============================================================================

// DST-1: Uses Z2D FFT (FST algorithm)
// FST uses N elements where x[0]=0 is boundary, actual data at x[1..N-1]
// For FFTW n points: use FST with N = n+1 (odd)
// Note: FST was designed for even N (powers of 2), adapted for odd N here
__global__ void kernel_dst1_preOp(double* pin, int N)
{
    extern __shared__ double sh_in[];
    int itx = threadIdx.x;

    // Load pin[0..N-1] to sh_in
    // For odd N, need ceiling(N/2) threads loading 2 elements each
    int half = (N + 1) / 2;
    if (itx < half) {
        sh_in[itx] = pin[itx];
        if (itx + half < N) {
            sh_in[itx + half] = pin[itx + half];
        }
    }
    __syncthreads();

    // Create complex for Z2D FFT (N/2+1 complex values)
    // z[k] is stored at pin[2k, 2k+1]
    // General formula: z[k] = (sh_in[2k+1] - sh_in[2k-1], -sh_in[2k])
    // Boundary condition: sh_in[N] = 0
    if (itx < N / 2 + 1) {
        if (itx == 0) {
            pin[0] = 2.0 * sh_in[1];
            pin[1] = 0.0;
        } else if (itx == N / 2) {
            if (N % 2 == 0) {
                // Even N: Nyquist frequency, purely real
                pin[itx * 2] = -2.0 * sh_in[N - 1];
                pin[itx * 2 + 1] = 0.0;
            } else {
                // Odd N: general complex value
                // z[N/2] = (sh_in[N] - sh_in[N-2], -sh_in[N-1]) where sh_in[N] = 0
                pin[itx * 2] = -sh_in[itx * 2 - 1];      // = -sh_in[N-2]
                pin[itx * 2 + 1] = -sh_in[itx * 2];      // = -sh_in[N-1]
            }
        } else {
            pin[itx * 2] = sh_in[itx * 2 + 1] - sh_in[itx * 2 - 1];
            pin[itx * 2 + 1] = -sh_in[itx * 2];
        }
    }
}

__global__ void kernel_dst1_postOp(double* pin, int N)
{
    int itx = threadIdx.x;

    if (itx < N / 2 + 1) {
        if (itx != 0) {
            double sitx = pin[itx];
            double sNitx = pin[N - itx];
            double Ta = (sitx + sNitx) / (4.0 * sin(itx * M_PI / N));
            double Tb = (sitx - sNitx) / 2.0;
            __syncthreads();
            // FST divides by 2, remove for FFTW unnormalized
            pin[itx] = (Ta + Tb);
            pin[N - itx] = (Ta - Tb);
        } else {
            pin[0] = 0.0;
        }
    }
}

// DST-2: Uses Z2D FFT (FST algorithm)
// For FFTW N points: input at buffer[1..N], output at buffer[1..N]
__global__ void kernel_dst2_preOp(double* pin, int N)
{
    extern __shared__ double sh_in[];
    int itx = threadIdx.x;

    // Load from pin[itx+1] (FST reads from offset 1)
    if (itx < N / 2) {
        sh_in[itx] = pin[itx + 1];
        sh_in[itx + N / 2] = pin[itx + N / 2 + 1];
    }
    __syncthreads();

    // Create complex for Z2D FFT
    if (itx < N / 2 + 1) {
        if (itx == 0) {
            pin[0] = sh_in[0];
            pin[1] = 0.0;
        } else if (itx == N / 2) {
            pin[N] = -sh_in[N - 1];
            pin[N + 1] = 0.0;
        } else {
            pin[itx * 2] = (sh_in[itx * 2] - sh_in[itx * 2 - 1]) / 2.0;
            pin[itx * 2 + 1] = -((sh_in[itx * 2] + sh_in[itx * 2 - 1]) / 2.0);
        }
    }
}

__global__ void kernel_dst2_postOp(double* pin, int N)
{
    extern __shared__ double sh_in[];
    int itx = threadIdx.x;

    // Load FFT output
    if (itx < N / 2) {
        sh_in[itx] = pin[itx];
        sh_in[itx + N / 2] = pin[itx + N / 2];
    }
    __syncthreads();

    if (itx < N / 2 + 1) {
        if (itx != 0) {
            double sina, cosa;
            sincos(itx * M_PI / (2.0 * N), &sina, &cosa);
            double Ta = sh_in[itx] + sh_in[N - itx];
            double Tb = sh_in[itx] - sh_in[N - itx];
            pin[itx] = (Ta * sina + Tb * cosa);
            pin[N - itx] = (Ta * cosa - Tb * sina);
        } else {
            pin[0] = 0.0;
            pin[N] = sh_in[0] * 2.0;
        }
    }
}

// DST-3: Uses D2Z FFT (FST algorithm)
// For FFTW N points: input at buffer[0..N-1], output at buffer[1..N]
__global__ void kernel_dst3_preOp(double* pin, int N)
{
    int itx = threadIdx.x;

    if (itx < N / 2 + 1) {
        double sina, cosa;
        sincos(itx * M_PI / (2.0 * N), &sina, &cosa);
        double Ta = pin[itx] + pin[N - itx];
        double Tb = pin[itx] - pin[N - itx];
        __syncthreads();
        pin[itx] = Ta * cosa + Tb * sina;
        pin[N - itx] = Ta * sina - Tb * cosa;
    }
}

__global__ void kernel_dst3_postOp(double* pin, int N)
{
    extern __shared__ double sh_in[];
    int itx = threadIdx.x;

    // Load FFT output (complex: N/2+1 complex values)
    if (itx < N / 2 + 1) {
        sh_in[itx] = pin[itx];
        sh_in[itx + N / 2 + 1] = pin[itx + N / 2 + 1];
    }
    __syncthreads();

    // Negate imaginary parts (FST: sh_in[itx*2+1] = -sh_in[itx*2+1])
    if (itx < N / 2 + 1 && itx != 0) {
        sh_in[itx * 2 + 1] = -sh_in[itx * 2 + 1];
    }
    __syncthreads();

    // Extract output - multiply by 2 to match FFTW unnormalized convention (remove /2)
    if (itx < N / 2 + 1) {
        if (itx == 0) {
            pin[0] = 0.0;
            pin[1] = sh_in[0];
        } else {
            pin[2 * itx] = (sh_in[itx * 2 + 1] - sh_in[itx * 2]);
            if (itx * 2 + 1 < N + 1) {
                pin[2 * itx + 1] = (sh_in[itx * 2] + sh_in[itx * 2 + 1]);
            }
        }
    }
}

__global__ void kernel_dst4_preOp(
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

__global__ void kernel_dst4_postOp(
    const cufftDoubleComplex* __restrict__ in,
    double* __restrict__ out,
    int N)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < N) {
        double angle = -M_PI * k / (2.0 * N);
        double sina = sin(angle);
        double cosa = cos(angle);
        double im = in[k].x * sina + in[k].y * cosa;
        out[k] = -2.0 * im;
    }
}

//==============================================================================
// 2D DCT Kernels
//==============================================================================

// DCT-1 2D kernels
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

// DCT-2 2D kernels
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
            out[out_idx].x = in[iy];
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

// DCT-3 2D kernels
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

__global__ void kernel_dct3_2d_preOp_x_transpose(
    const double* __restrict__ in,
    double* __restrict__ out,
    int Nx, int Ny)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny;

    if (idx < total) {
        int ix = idx % Nx;
        int iy = idx / Nx;
        out[iy * Nx + ix] = in[ix * Ny + iy];
    }
}

__global__ void kernel_dct3_2d_preOp_x(
    double* __restrict__ data,
    int Nx, int Ny)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Ny * (Nx / 2);

    if (idx < total) {
        int ix = idx % (Nx / 2);
        int iy = idx / (Nx / 2);

        int base = iy * Nx;
        int ix1 = ix + 1;
        int ix2 = Nx - ix - 1;

        double sina, cosa;
        sincos(ix1 * M_PI / (2.0 * Nx), &sina, &cosa);
        double Ta = data[base + ix1] + data[base + ix2];
        double Tb = data[base + ix1] - data[base + ix2];

        data[base + ix1] = Ta * sina + Tb * cosa;
        data[base + ix2] = Ta * cosa - Tb * sina;
    }
}

__global__ void kernel_dct3_2d_postOp_x(
    const cufftDoubleComplex* __restrict__ in,
    double* __restrict__ out,
    int Nx, int Ny)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny;

    if (idx < total) {
        int iy = idx % Ny;
        int ix = idx / Ny;

        int in_base = iy * (Nx / 2 + 1);

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

// DCT-4 2D kernels
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
// 2D DST Kernels
//==============================================================================

// DST-1 2D kernels
__global__ void kernel_dst1_2d_mirror_extend(
    const double* __restrict__ in,
    double* __restrict__ out,
    int Nx, int Ny)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int ext_Nx = 2 * (Nx + 1);
    int ext_Ny = 2 * (Ny + 1);
    int ext_total = ext_Nx * ext_Ny;

    if (idx < ext_total) {
        int iy = idx % ext_Ny;
        int ix = idx / ext_Ny;

        double sign = 1.0;
        int ix_in, iy_in;

        if (ix == 0 || ix == Nx + 1) {
            out[idx] = 0.0;
            return;
        } else if (ix <= Nx) {
            ix_in = ix - 1;
        } else {
            ix_in = ext_Nx - ix - 1;
            sign *= -1.0;
        }

        if (iy == 0 || iy == Ny + 1) {
            out[idx] = 0.0;
            return;
        } else if (iy <= Ny) {
            iy_in = iy - 1;
        } else {
            iy_in = ext_Ny - iy - 1;
            sign *= -1.0;
        }

        out[idx] = sign * in[ix_in * Ny + iy_in];
    }
}

__global__ void kernel_dst1_2d_extract(
    const cufftDoubleComplex* __restrict__ in,
    double* __restrict__ out,
    int Nx, int Ny)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int out_total = Nx * Ny;

    if (idx < out_total) {
        int iy = idx % Ny;
        int ix = idx / Ny;

        int freq_half = Ny + 2;

        int freq_idx = (ix + 1) * freq_half + (iy + 1);
        out[idx] = -in[freq_idx].x;  // DST-1 from odd-odd extension: extract negated real part
    }
}

// DST-2 2D kernels (FST algorithm, Martucci 1994)
// DST-2 uses Z2D FFT with preOp/postOp
__global__ void kernel_dst2_2d_preOp_y(
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
            out[out_idx].x = -in[ix * Ny + Ny - 1];
            out[out_idx].y = 0.0;
        } else {
            int in_idx = ix * Ny + 2 * iy_half;
            // DST-2: z[k] = ((x[2k] - x[2k-1])/2, -((x[2k] + x[2k-1])/2))
            out[out_idx].x = (in[in_idx] - in[in_idx - 1]) / 2.0;
            out[out_idx].y = -((in[in_idx] + in[in_idx - 1]) / 2.0);
        }
    }
}

__global__ void kernel_dst2_2d_postOp_y(
    const double* __restrict__ in,
    double* __restrict__ out,
    int Nx, int Ny)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny;

    if (idx < total) {
        int iy = idx % Ny;
        int ix = idx / Ny;

        // FST index k = iy + 1 (mapping FFTW index iy to FST index k)
        // FST output[1..N] maps to FFTW output[0..N-1]
        int k = iy + 1;

        if (k == Ny) {
            // Last element: FFTW[Ny-1] = 2 * in[0]
            out[idx] = 2.0 * in[ix * Ny];
        } else if (k <= Ny / 2) {
            // First half: apply twiddle formula
            double sina, cosa;
            sincos(k * M_PI / (2.0 * Ny), &sina, &cosa);
            double Ta = in[ix * Ny + k] + in[ix * Ny + Ny - k];
            double Tb = in[ix * Ny + k] - in[ix * Ny + Ny - k];
            out[idx] = (Ta * sina + Tb * cosa);
        } else {
            // Second half: use symmetric formula
            int k_mirror = Ny - k;
            double sina, cosa;
            sincos(k_mirror * M_PI / (2.0 * Ny), &sina, &cosa);
            double Ta = in[ix * Ny + k_mirror] + in[ix * Ny + k];
            double Tb = in[ix * Ny + k_mirror] - in[ix * Ny + k];
            out[idx] = (Ta * cosa - Tb * sina);
        }
    }
}

__global__ void kernel_dst2_2d_preOp_x(
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
            out[out_idx].x = in[0 * Ny + iy];
            out[out_idx].y = 0.0;
        } else if (ix_half == Nx / 2) {
            out[out_idx].x = -in[(Nx - 1) * Ny + iy];
            out[out_idx].y = 0.0;
        } else {
            int ix_in = 2 * ix_half;
            // DST-2: z[k] = ((x[2k] - x[2k-1])/2, -((x[2k] + x[2k-1])/2))
            out[out_idx].x = (in[ix_in * Ny + iy] - in[(ix_in - 1) * Ny + iy]) / 2.0;
            out[out_idx].y = -((in[ix_in * Ny + iy] + in[(ix_in - 1) * Ny + iy]) / 2.0);
        }
    }
}

__global__ void kernel_dst2_2d_postOp_x(
    const double* __restrict__ in,
    double* __restrict__ out,
    int Nx, int Ny)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny;

    if (idx < total) {
        int iy = idx % Ny;
        int ix = idx / Ny;

        // FST index k = ix + 1 (mapping FFTW index ix to FST index k)
        // Input is in [Ny][Nx] transposed layout
        int k = ix + 1;

        if (k == Nx) {
            // Last element: FFTW[Nx-1] = 2 * in[0]
            out[idx] = 2.0 * in[iy * Nx];
        } else if (k <= Nx / 2) {
            // First half: apply twiddle formula
            double sina, cosa;
            sincos(k * M_PI / (2.0 * Nx), &sina, &cosa);
            double Ta = in[iy * Nx + k] + in[iy * Nx + Nx - k];
            double Tb = in[iy * Nx + k] - in[iy * Nx + Nx - k];
            out[idx] = (Ta * sina + Tb * cosa);
        } else {
            // Second half: use symmetric formula
            int k_mirror = Nx - k;
            double sina, cosa;
            sincos(k_mirror * M_PI / (2.0 * Nx), &sina, &cosa);
            double Ta = in[iy * Nx + k_mirror] + in[iy * Nx + k];
            double Tb = in[iy * Nx + k_mirror] - in[iy * Nx + k];
            out[idx] = (Ta * cosa - Tb * sina);
        }
    }
}

// DST-3 2D kernels (FST algorithm with padded buffer)
// Buffer layout per row: [0, x0, x1, ..., x_{N-1}, 0] (N+2 elements)
// This matches the 1D FST approach (Martucci 1994)

// Setup padded buffer for Y-direction DST-3 - matches 1D layout
// Data at positions 1..Ny, boundary at position 0 (zero)
__global__ void kernel_dst3_2d_setup_padded_y(
    const double* __restrict__ in,
    double* __restrict__ work,
    int Nx, int Ny)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = Ny + 2;  // Padded row stride
    int total = Nx * stride;

    if (idx < total) {
        int pos = idx % stride;
        int ix = idx / stride;

        if (pos == 0 || pos == Ny + 1) {
            // Boundary at position 0, padding at Ny+1
            work[idx] = 0.0;
        } else {
            // Data at positions 1..Ny
            work[idx] = in[ix * Ny + (pos - 1)];
        }
    }
}

// PreOp for Y-direction DST-3 - FCT/FST style (Makhoul 1980)
// Data at positions 0..Ny-1, boundary at position Ny (=0)
// Each block handles one row
__global__ void kernel_dst3_2d_preOp_y(
    double* __restrict__ data,
    int Nx, int Ny)
{
    int itx = threadIdx.x;
    int ix = blockIdx.x;
    double* pin = data + ix * (Ny + 2);

    // Pair pin[itx] with pin[Ny-itx] for itx = 0..Ny/2
    if (itx <= Ny / 2) {
        double sina, cosa;
        sincos(itx * M_PI / (2.0 * Ny), &sina, &cosa);

        double Ta = pin[itx] + pin[Ny - itx];
        double Tb = pin[itx] - pin[Ny - itx];

        pin[itx] = Ta * cosa + Tb * sina;
        pin[Ny - itx] = Ta * sina - Tb * cosa;
    }
}

// PostOp for Y-direction DST-3 - matches 1D implementation
// Each block handles one row, uses shared memory to avoid race conditions
// No /2 factor - matches FFTW unnormalized convention (same as 1D kernel)
__global__ void kernel_dst3_2d_postOp_y(
    double* __restrict__ data,
    int Nx, int Ny)
{
    extern __shared__ double sh_in[];

    int itx = threadIdx.x;
    int ix = blockIdx.x;
    double* pin = data + ix * (Ny + 2);

    // Load D2Z FFT output into shared memory
    if (itx < Ny / 2 + 1) {
        sh_in[itx] = pin[itx];
        sh_in[itx + Ny / 2 + 1] = pin[itx + Ny / 2 + 1];
    }
    __syncthreads();

    // Negate imaginary parts (odd indices starting from 3)
    if (itx != 0 && itx < Ny / 2 + 1) {
        sh_in[itx * 2 + 1] = -sh_in[itx * 2 + 1];
    }
    __syncthreads();

    // Compute final output - NO /2 factor to match FFTW convention
    if (itx < Ny / 2 + 1) {
        if (itx == 0) {
            pin[0] = 0.0;
            pin[1] = sh_in[0];
        } else {
            pin[2 * itx] = (sh_in[itx * 2 + 1] - sh_in[itx * 2]);
            if (itx * 2 + 1 < Ny + 1) {
                pin[2 * itx + 1] = (sh_in[itx * 2] + sh_in[itx * 2 + 1]);
            }
        }
    }
}

// Copy result back from padded buffer
__global__ void kernel_dst3_2d_copy_back_y(
    const double* __restrict__ work,
    double* __restrict__ out,
    int Nx, int Ny)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny;

    if (idx < total) {
        int iy = idx % Ny;
        int ix = idx / Ny;
        int stride = Ny + 2;

        out[idx] = work[ix * stride + iy + 1];  // Skip padding at position 0
    }
}

// X-direction kernels - matches 1D layout
// Data at positions 1..Nx, boundary at position 0 (zero)
__global__ void kernel_dst3_2d_setup_padded_x(
    const double* __restrict__ in,
    double* __restrict__ work,
    int Nx, int Ny)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = Nx + 2;
    int total = Ny * stride;

    if (idx < total) {
        int pos = idx % stride;
        int iy = idx / stride;

        if (pos == 0 || pos == Nx + 1) {
            // Boundary at position 0, padding at Nx+1
            work[idx] = 0.0;
        } else {
            // Data at positions 1..Nx, transpose from [Nx][Ny] to [Ny][Nx+2]
            work[idx] = in[(pos - 1) * Ny + iy];
        }
    }
}

// PreOp for X-direction DST-3 - FCT/FST style (Makhoul 1980)
// Each block handles one row
__global__ void kernel_dst3_2d_preOp_x(
    double* __restrict__ data,
    int Nx, int Ny)
{
    int itx = threadIdx.x;
    int iy = blockIdx.x;
    double* pin = data + iy * (Nx + 2);

    // Pair pin[itx] with pin[Nx-itx] for itx = 0..Nx/2
    if (itx <= Nx / 2) {
        double sina, cosa;
        sincos(itx * M_PI / (2.0 * Nx), &sina, &cosa);

        double Ta = pin[itx] + pin[Nx - itx];
        double Tb = pin[itx] - pin[Nx - itx];

        pin[itx] = Ta * cosa + Tb * sina;
        pin[Nx - itx] = Ta * sina - Tb * cosa;
    }
}

// PostOp for X-direction DST-3 - matches 1D implementation
// Each block handles one row, uses shared memory to avoid race conditions
// No /2 factor - matches FFTW unnormalized convention (same as 1D kernel)
__global__ void kernel_dst3_2d_postOp_x(
    double* __restrict__ data,
    int Nx, int Ny)
{
    extern __shared__ double sh_in[];

    int itx = threadIdx.x;
    int iy = blockIdx.x;
    double* pin = data + iy * (Nx + 2);

    // Load D2Z FFT output into shared memory
    if (itx < Nx / 2 + 1) {
        sh_in[itx] = pin[itx];
        sh_in[itx + Nx / 2 + 1] = pin[itx + Nx / 2 + 1];
    }
    __syncthreads();

    // Negate imaginary parts (odd indices starting from 3)
    if (itx != 0 && itx < Nx / 2 + 1) {
        sh_in[itx * 2 + 1] = -sh_in[itx * 2 + 1];
    }
    __syncthreads();

    // Compute final output - NO /2 factor to match FFTW convention
    if (itx < Nx / 2 + 1) {
        if (itx == 0) {
            pin[0] = 0.0;
            pin[1] = sh_in[0];
        } else {
            pin[2 * itx] = (sh_in[itx * 2 + 1] - sh_in[itx * 2]);
            if (itx * 2 + 1 < Nx + 1) {
                pin[2 * itx + 1] = (sh_in[itx * 2] + sh_in[itx * 2 + 1]);
            }
        }
    }
}

__global__ void kernel_dst3_2d_copy_back_x(
    const double* __restrict__ work,
    double* __restrict__ out,
    int Nx, int Ny)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny;

    if (idx < total) {
        int iy = idx % Ny;
        int ix = idx / Ny;
        int stride = Nx + 2;

        // Output needs to be in [Nx][Ny] layout
        out[idx] = work[iy * stride + ix + 1];
    }
}

// DST-4 2D kernels
__global__ void kernel_dst4_2d_preOp_y(
    const double* __restrict__ in,
    cufftDoubleComplex* __restrict__ out,
    int Nx, int Ny)
{
    int ix = blockIdx.x;
    int iy = threadIdx.x;
    int N2 = 2 * Ny;

    if (ix < Nx && iy < N2) {
        int out_idx = ix * N2 + iy;
        if (iy < Ny) {
            double angle = -M_PI * (2 * iy + 1) / (4.0 * Ny);
            out[out_idx].x = in[ix * Ny + iy] * cos(angle);
            out[out_idx].y = in[ix * Ny + iy] * sin(angle);
        } else {
            out[out_idx].x = 0.0;
            out[out_idx].y = 0.0;
        }
    }
}

__global__ void kernel_dst4_2d_postOp_y(
    const cufftDoubleComplex* __restrict__ in,
    double* __restrict__ out,
    int Nx, int Ny)
{
    int ix = blockIdx.x;
    int iy = threadIdx.x;
    int N2 = 2 * Ny;

    if (ix < Nx && iy < Ny) {
        double angle = -M_PI * iy / (2.0 * Ny);
        cufftDoubleComplex z = in[ix * N2 + iy];
        double im = z.x * sin(angle) + z.y * cos(angle);
        out[ix * Ny + iy] = -2.0 * im;
    }
}

__global__ void kernel_dst4_2d_preOp_x(
    const double* __restrict__ in,
    cufftDoubleComplex* __restrict__ out,
    int Nx, int Ny)
{
    int iy = blockIdx.x;
    int ix = threadIdx.x;
    int N2 = 2 * Nx;

    if (iy < Ny && ix < N2) {
        int out_idx = iy * N2 + ix;
        if (ix < Nx) {
            double angle = -M_PI * (2 * ix + 1) / (4.0 * Nx);
            out[out_idx].x = in[ix * Ny + iy] * cos(angle);
            out[out_idx].y = in[ix * Ny + iy] * sin(angle);
        } else {
            out[out_idx].x = 0.0;
            out[out_idx].y = 0.0;
        }
    }
}

__global__ void kernel_dst4_2d_postOp_x(
    const cufftDoubleComplex* __restrict__ in,
    double* __restrict__ out,
    int Nx, int Ny)
{
    int iy = blockIdx.x;
    int ix = threadIdx.x;
    int N2 = 2 * Nx;

    if (iy < Ny && ix < Nx) {
        double angle = -M_PI * ix / (2.0 * Nx);
        cufftDoubleComplex z = in[iy * N2 + ix];
        double im = z.x * sin(angle) + z.y * cos(angle);
        out[ix * Ny + iy] = -2.0 * im;
    }
}

//==============================================================================
// 3D DCT Kernels
// Data layout: [Nx][Ny][Nz] with linear index = (ix * Ny + iy) * Nz + iz
//==============================================================================

//------------------------------------------------------------------------------
// 3D DCT-1 Kernels (using mirror extension for 3D FFT)
//------------------------------------------------------------------------------

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

//------------------------------------------------------------------------------
// 3D DST-1 Kernels (using odd extension for 3D FFT)
//------------------------------------------------------------------------------

__global__ void kernel_dst1_3d_odd_extend(
    const double* __restrict__ in,
    double* __restrict__ out,
    int Nx, int Ny, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int Nx2 = 2 * (Nx + 1);
    int Ny2 = 2 * (Ny + 1);
    int Nz2 = 2 * (Nz + 1);
    int ext_total = Nx2 * Ny2 * Nz2;

    if (idx < ext_total) {
        int iz = idx % Nz2;
        int iy = (idx / Nz2) % Ny2;
        int ix = idx / (Nz2 * Ny2);

        // Determine sign and mapping for odd extension
        int sign_x = 1, sign_y = 1, sign_z = 1;
        int ix_in, iy_in, iz_in;

        if (ix == 0 || ix == Nx + 1) {
            out[idx] = 0.0;
            return;
        } else if (ix <= Nx) {
            ix_in = ix - 1;
        } else {
            ix_in = Nx2 - ix - 1;
            sign_x = -1;
        }

        if (iy == 0 || iy == Ny + 1) {
            out[idx] = 0.0;
            return;
        } else if (iy <= Ny) {
            iy_in = iy - 1;
        } else {
            iy_in = Ny2 - iy - 1;
            sign_y = -1;
        }

        if (iz == 0 || iz == Nz + 1) {
            out[idx] = 0.0;
            return;
        } else if (iz <= Nz) {
            iz_in = iz - 1;
        } else {
            iz_in = Nz2 - iz - 1;
            sign_z = -1;
        }

        int in_idx = (ix_in * Ny + iy_in) * Nz + iz_in;
        out[idx] = sign_x * sign_y * sign_z * in[in_idx];
    }
}

__global__ void kernel_dst1_3d_extract(
    const cufftDoubleComplex* __restrict__ in,
    double* __restrict__ out,
    int Nx, int Ny, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int out_total = Nx * Ny * Nz;

    if (idx < out_total) {
        int iz = idx % Nz;
        int iy = (idx / Nz) % Ny;
        int ix = idx / (Nz * Ny);

        int Nz_half = Nz + 2;  // (Nz+1) + 1 for Hermitian
        int freq_idx = ((ix + 1) * 2 * (Ny + 1) + (iy + 1)) * Nz_half + (iz + 1);
        out[idx] = in[freq_idx].y;  // DST-1 from 3D odd extension: extract imaginary part
    }
}

//------------------------------------------------------------------------------
// 3D DCT-2 Kernels
//------------------------------------------------------------------------------

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

//------------------------------------------------------------------------------
// 3D DCT-3 Kernels
//------------------------------------------------------------------------------

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

__global__ void kernel_dct3_3d_postOp_z(
    const cufftDoubleComplex* __restrict__ in,
    double* __restrict__ out,
    int Nx, int Ny, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny * Nz;
    int half_z = Nz / 2 + 1;

    if (idx < total) {
        int iz = idx % Nz;
        int iy = (idx / Nz) % Ny;
        int ix = idx / (Nz * Ny);

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

__global__ void kernel_dct3_3d_preOp_y_transpose(
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

        out[(ix * Nz + iz) * Ny + iy] = in[(ix * Ny + iy) * Nz + iz];
    }
}

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

        int base = (ix * Nz + iz) * Ny;
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

__global__ void kernel_dct3_3d_postOp_y(
    const cufftDoubleComplex* __restrict__ in,
    double* __restrict__ out,
    int Nx, int Ny, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny * Nz;
    int half_y = Ny / 2 + 1;

    if (idx < total) {
        int iz = idx % Nz;
        int iy = (idx / Nz) % Ny;
        int ix = idx / (Nz * Ny);

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

__global__ void kernel_dct3_3d_preOp_x_transpose(
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

        out[(iy * Nz + iz) * Nx + ix] = in[(ix * Ny + iy) * Nz + iz];
    }
}

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

        int base = (iy * Nz + iz) * Nx;
        int ix1 = ix + 1;
        int ix2 = Nx - ix - 1;

        double sina, cosa;
        sincos(ix1 * M_PI / (2.0 * Nx), &sina, &cosa);
        double Ta = data[base + ix1] + data[base + ix2];
        double Tb = data[base + ix1] - data[base + ix2];

        data[base + ix1] = Ta * sina + Tb * cosa;
        data[base + ix2] = Ta * cosa - Tb * sina;
    }
}

__global__ void kernel_dct3_3d_postOp_x(
    const cufftDoubleComplex* __restrict__ in,
    double* __restrict__ out,
    int Nx, int Ny, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny * Nz;
    int half_x = Nx / 2 + 1;

    if (idx < total) {
        int iz = idx % Nz;
        int iy = (idx / Nz) % Ny;
        int ix = idx / (Nz * Ny);

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

//------------------------------------------------------------------------------
// 3D DCT-4 Kernels
//------------------------------------------------------------------------------

__global__ void kernel_dct4_3d_preOp_z(
    const double* __restrict__ in,
    cufftDoubleComplex* __restrict__ out,
    int Nx, int Ny, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N2 = 2 * Nz;
    int total = Nx * Ny * N2;

    if (idx < total) {
        int iz = idx % N2;
        int iy = (idx / N2) % Ny;
        int ix = idx / (N2 * Ny);

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

__global__ void kernel_dct4_3d_postOp_z(
    const cufftDoubleComplex* __restrict__ in,
    double* __restrict__ out,
    int Nx, int Ny, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny * Nz;
    int N2 = 2 * Nz;

    if (idx < total) {
        int iz = idx % Nz;
        int iy = (idx / Nz) % Ny;
        int ix = idx / (Nz * Ny);

        int in_idx = (ix * Ny + iy) * N2 + iz;
        double angle = -M_PI * iz / (2.0 * Nz);
        double cosa = cos(angle);
        double sina = sin(angle);

        double re = in[in_idx].x * cosa - in[in_idx].y * sina;
        out[idx] = 2.0 * re;
    }
}

__global__ void kernel_dct4_3d_preOp_y(
    const double* __restrict__ in,
    cufftDoubleComplex* __restrict__ out,
    int Nx, int Ny, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N2 = 2 * Ny;
    int total = Nx * Nz * N2;

    if (idx < total) {
        int iy = idx % N2;
        int iz = (idx / N2) % Nz;
        int ix = idx / (N2 * Nz);

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

__global__ void kernel_dct4_3d_postOp_y(
    const cufftDoubleComplex* __restrict__ in,
    double* __restrict__ out,
    int Nx, int Ny, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny * Nz;
    int N2 = 2 * Ny;

    if (idx < total) {
        int iz = idx % Nz;
        int iy = (idx / Nz) % Ny;
        int ix = idx / (Nz * Ny);

        int in_idx = (ix * Nz + iz) * N2 + iy;
        double angle = -M_PI * iy / (2.0 * Ny);
        double cosa = cos(angle);
        double sina = sin(angle);

        double re = in[in_idx].x * cosa - in[in_idx].y * sina;
        out[idx] = 2.0 * re;
    }
}

__global__ void kernel_dct4_3d_preOp_x(
    const double* __restrict__ in,
    cufftDoubleComplex* __restrict__ out,
    int Nx, int Ny, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N2 = 2 * Nx;
    int total = Ny * Nz * N2;

    if (idx < total) {
        int ix = idx % N2;
        int iz = (idx / N2) % Nz;
        int iy = idx / (N2 * Nz);

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

__global__ void kernel_dct4_3d_postOp_x(
    const cufftDoubleComplex* __restrict__ in,
    double* __restrict__ out,
    int Nx, int Ny, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny * Nz;
    int N2 = 2 * Nx;

    if (idx < total) {
        int iz = idx % Nz;
        int iy = (idx / Nz) % Ny;
        int ix = idx / (Nz * Ny);

        int in_idx = (iy * Nz + iz) * N2 + ix;
        double angle = -M_PI * ix / (2.0 * Nx);
        double cosa = cos(angle);
        double sina = sin(angle);

        double re = in[in_idx].x * cosa - in[in_idx].y * sina;
        out[idx] = 2.0 * re;
    }
}

//==============================================================================
// 3D DST Kernels
//==============================================================================

//------------------------------------------------------------------------------
// 3D DST-2 Kernels
//------------------------------------------------------------------------------

// 3D DST-2 Z-direction kernels - FCT/FST style (Makhoul 1980)
__global__ void kernel_dst2_3d_preOp_z(
    const double* __restrict__ in,
    cufftDoubleComplex* __restrict__ out,
    int Nx, int Ny, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny * (Nz / 2 + 1);

    if (idx < total) {
        int iz_half = idx % (Nz / 2 + 1);
        int ixy = idx / (Nz / 2 + 1);
        int out_idx = ixy * (Nz / 2 + 1) + iz_half;

        if (iz_half == 0) {
            out[out_idx].x = in[ixy * Nz];
            out[out_idx].y = 0.0;
        } else if (iz_half == Nz / 2) {
            out[out_idx].x = -in[ixy * Nz + Nz - 1];
            out[out_idx].y = 0.0;
        } else {
            int in_idx = ixy * Nz + 2 * iz_half;
            out[out_idx].x = (in[in_idx] - in[in_idx - 1]) / 2.0;
            out[out_idx].y = -((in[in_idx] + in[in_idx - 1]) / 2.0);
        }
    }
}

__global__ void kernel_dst2_3d_postOp_z(
    const double* __restrict__ in,
    double* __restrict__ out,
    int Nx, int Ny, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny * Nz;

    if (idx < total) {
        int iz = idx % Nz;
        int ixy = idx / Nz;
        int k = iz + 1;

        if (k == Nz) {
            out[idx] = 2.0 * in[ixy * Nz];
        } else if (k <= Nz / 2) {
            double sina, cosa;
            sincos(k * M_PI / (2.0 * Nz), &sina, &cosa);
            double Ta = in[ixy * Nz + k] + in[ixy * Nz + Nz - k];
            double Tb = in[ixy * Nz + k] - in[ixy * Nz + Nz - k];
            out[idx] = (Ta * sina + Tb * cosa);
        } else {
            int k_mirror = Nz - k;
            double sina, cosa;
            sincos(k_mirror * M_PI / (2.0 * Nz), &sina, &cosa);
            double Ta = in[ixy * Nz + k_mirror] + in[ixy * Nz + k];
            double Tb = in[ixy * Nz + k_mirror] - in[ixy * Nz + k];
            out[idx] = (Ta * cosa - Tb * sina);
        }
    }
}

// 3D DST-2 Y-direction kernels - FCT/FST style (Makhoul 1980)
__global__ void kernel_dst2_3d_preOp_y(
    const double* __restrict__ in,
    cufftDoubleComplex* __restrict__ out,
    int Nx, int Ny, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Nz * (Ny / 2 + 1);

    if (idx < total) {
        int iy_half = idx % (Ny / 2 + 1);
        int ixz = idx / (Ny / 2 + 1);
        int iz = ixz % Nz;
        int ix = ixz / Nz;
        int out_idx = ixz * (Ny / 2 + 1) + iy_half;

        if (iy_half == 0) {
            out[out_idx].x = in[(ix * Ny) * Nz + iz];
            out[out_idx].y = 0.0;
        } else if (iy_half == Ny / 2) {
            out[out_idx].x = -in[(ix * Ny + Ny - 1) * Nz + iz];
            out[out_idx].y = 0.0;
        } else {
            int iy_in = 2 * iy_half;
            double v1 = in[(ix * Ny + iy_in) * Nz + iz];
            double v0 = in[(ix * Ny + iy_in - 1) * Nz + iz];
            out[out_idx].x = (v1 - v0) / 2.0;
            out[out_idx].y = -((v1 + v0) / 2.0);
        }
    }
}

__global__ void kernel_dst2_3d_postOp_y(
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
        int ixz = ix * Nz + iz;
        int k = iy + 1;

        if (k == Ny) {
            out[idx] = 2.0 * in[ixz * Ny];
        } else if (k <= Ny / 2) {
            double sina, cosa;
            sincos(k * M_PI / (2.0 * Ny), &sina, &cosa);
            double Ta = in[ixz * Ny + k] + in[ixz * Ny + Ny - k];
            double Tb = in[ixz * Ny + k] - in[ixz * Ny + Ny - k];
            out[idx] = (Ta * sina + Tb * cosa);
        } else {
            int k_mirror = Ny - k;
            double sina, cosa;
            sincos(k_mirror * M_PI / (2.0 * Ny), &sina, &cosa);
            double Ta = in[ixz * Ny + k_mirror] + in[ixz * Ny + k];
            double Tb = in[ixz * Ny + k_mirror] - in[ixz * Ny + k];
            out[idx] = (Ta * cosa - Tb * sina);
        }
    }
}

// 3D DST-2 X-direction kernels - FCT/FST style (Makhoul 1980)
__global__ void kernel_dst2_3d_preOp_x(
    const double* __restrict__ in,
    cufftDoubleComplex* __restrict__ out,
    int Nx, int Ny, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Ny * Nz * (Nx / 2 + 1);

    if (idx < total) {
        int ix_half = idx % (Nx / 2 + 1);
        int iyz = idx / (Nx / 2 + 1);
        int iz = iyz % Nz;
        int iy = iyz / Nz;
        int out_idx = iyz * (Nx / 2 + 1) + ix_half;

        if (ix_half == 0) {
            out[out_idx].x = in[(0 * Ny + iy) * Nz + iz];
            out[out_idx].y = 0.0;
        } else if (ix_half == Nx / 2) {
            out[out_idx].x = -in[((Nx - 1) * Ny + iy) * Nz + iz];
            out[out_idx].y = 0.0;
        } else {
            int ix_in = 2 * ix_half;
            double v1 = in[(ix_in * Ny + iy) * Nz + iz];
            double v0 = in[((ix_in - 1) * Ny + iy) * Nz + iz];
            out[out_idx].x = (v1 - v0) / 2.0;
            out[out_idx].y = -((v1 + v0) / 2.0);
        }
    }
}

__global__ void kernel_dst2_3d_postOp_x(
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
        int iyz = iy * Nz + iz;
        int k = ix + 1;

        if (k == Nx) {
            out[idx] = 2.0 * in[iyz * Nx];
        } else if (k <= Nx / 2) {
            double sina, cosa;
            sincos(k * M_PI / (2.0 * Nx), &sina, &cosa);
            double Ta = in[iyz * Nx + k] + in[iyz * Nx + Nx - k];
            double Tb = in[iyz * Nx + k] - in[iyz * Nx + Nx - k];
            out[idx] = (Ta * sina + Tb * cosa);
        } else {
            int k_mirror = Nx - k;
            double sina, cosa;
            sincos(k_mirror * M_PI / (2.0 * Nx), &sina, &cosa);
            double Ta = in[iyz * Nx + k_mirror] + in[iyz * Nx + k];
            double Tb = in[iyz * Nx + k_mirror] - in[iyz * Nx + k];
            out[idx] = (Ta * cosa - Tb * sina);
        }
    }
}

//------------------------------------------------------------------------------
// 3D DST-3 Kernels - FCT/FST style (Makhoul 1980)
//------------------------------------------------------------------------------

// Setup padded buffer for Z-direction: [0, x0, x1, ..., x_{Nz-1}, 0]
__global__ void kernel_dst3_3d_setup_z(
    const double* __restrict__ in,
    double* __restrict__ work,
    int Nx, int Ny, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = Nz + 2;
    int total = Nx * Ny * stride;

    if (idx < total) {
        int pos = idx % stride;
        int ixy = idx / stride;

        if (pos == 0 || pos == Nz + 1) {
            work[idx] = 0.0;
        } else {
            work[idx] = in[ixy * Nz + (pos - 1)];
        }
    }
}

// PreOp twiddle factor for Z-direction - each XY point processes independently
__global__ void kernel_dst3_3d_preOp_z(
    double* __restrict__ data,
    int Nx, int Ny, int Nz)
{
    int itx = threadIdx.x;
    int ixy = blockIdx.x;
    double* pin = data + ixy * (Nz + 2);

    if (itx <= Nz / 2) {
        double sina, cosa;
        sincos(itx * M_PI / (2.0 * Nz), &sina, &cosa);
        double Ta = pin[itx] + pin[Nz - itx];
        double Tb = pin[itx] - pin[Nz - itx];
        pin[itx] = Ta * cosa + Tb * sina;
        pin[Nz - itx] = Ta * sina - Tb * cosa;
    }
}

// PostOp for Z-direction - extract DST-3 result
__global__ void kernel_dst3_3d_postOp_z(
    double* __restrict__ data,
    int Nx, int Ny, int Nz)
{
    extern __shared__ double sh_in[];
    int itx = threadIdx.x;
    int ixy = blockIdx.x;
    double* pin = data + ixy * (Nz + 2);

    // Load FFT output into shared memory
    if (itx < Nz / 2 + 1) {
        sh_in[itx] = pin[itx];
        sh_in[itx + Nz / 2 + 1] = pin[itx + Nz / 2 + 1];
    }
    __syncthreads();

    // Negate imaginary parts
    if (itx != 0 && itx < Nz / 2 + 1) {
        sh_in[itx * 2 + 1] = -sh_in[itx * 2 + 1];
    }
    __syncthreads();

    // Compute output - NO /2 factor to match FFTW
    if (itx < Nz / 2 + 1) {
        if (itx == 0) {
            pin[0] = 0.0;
            pin[1] = sh_in[0];
        } else {
            pin[2 * itx] = (sh_in[itx * 2 + 1] - sh_in[itx * 2]);
            if (itx * 2 + 1 < Nz + 1) {
                pin[2 * itx + 1] = (sh_in[itx * 2] + sh_in[itx * 2 + 1]);
            }
        }
    }
}

// Copy result back from padded buffer
__global__ void kernel_dst3_3d_copy_z(
    const double* __restrict__ work,
    double* __restrict__ out,
    int Nx, int Ny, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny * Nz;

    if (idx < total) {
        int iz = idx % Nz;
        int ixy = idx / Nz;
        out[idx] = work[ixy * (Nz + 2) + iz + 1];
    }
}

// Setup padded buffer for Y-direction: [0, x0, x1, ..., x_{Ny-1}, 0]
__global__ void kernel_dst3_3d_setup_y(
    const double* __restrict__ in,
    double* __restrict__ work,
    int Nx, int Ny, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = Ny + 2;
    int total = Nx * Nz * stride;

    if (idx < total) {
        int pos = idx % stride;
        int ixz = idx / stride;
        int iz = ixz % Nz;
        int ix = ixz / Nz;

        if (pos == 0 || pos == Ny + 1) {
            work[idx] = 0.0;
        } else {
            work[idx] = in[(ix * Ny + (pos - 1)) * Nz + iz];
        }
    }
}

// PreOp twiddle factor for Y-direction
__global__ void kernel_dst3_3d_preOp_y(
    double* __restrict__ data,
    int Nx, int Ny, int Nz)
{
    int itx = threadIdx.x;
    int ixz = blockIdx.x;
    double* pin = data + ixz * (Ny + 2);

    if (itx <= Ny / 2) {
        double sina, cosa;
        sincos(itx * M_PI / (2.0 * Ny), &sina, &cosa);
        double Ta = pin[itx] + pin[Ny - itx];
        double Tb = pin[itx] - pin[Ny - itx];
        pin[itx] = Ta * cosa + Tb * sina;
        pin[Ny - itx] = Ta * sina - Tb * cosa;
    }
}

// PostOp for Y-direction
__global__ void kernel_dst3_3d_postOp_y(
    double* __restrict__ data,
    int Nx, int Ny, int Nz)
{
    extern __shared__ double sh_in[];
    int itx = threadIdx.x;
    int ixz = blockIdx.x;
    double* pin = data + ixz * (Ny + 2);

    if (itx < Ny / 2 + 1) {
        sh_in[itx] = pin[itx];
        sh_in[itx + Ny / 2 + 1] = pin[itx + Ny / 2 + 1];
    }
    __syncthreads();

    if (itx != 0 && itx < Ny / 2 + 1) {
        sh_in[itx * 2 + 1] = -sh_in[itx * 2 + 1];
    }
    __syncthreads();

    if (itx < Ny / 2 + 1) {
        if (itx == 0) {
            pin[0] = 0.0;
            pin[1] = sh_in[0];
        } else {
            pin[2 * itx] = (sh_in[itx * 2 + 1] - sh_in[itx * 2]);
            if (itx * 2 + 1 < Ny + 1) {
                pin[2 * itx + 1] = (sh_in[itx * 2] + sh_in[itx * 2 + 1]);
            }
        }
    }
}

// Copy result back for Y-direction
__global__ void kernel_dst3_3d_copy_y(
    const double* __restrict__ work,
    double* __restrict__ out,
    int Nx, int Ny, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny * Nz;

    if (idx < total) {
        int iz = idx % Nz;
        int iy = (idx / Nz) % Ny;
        int ix = idx / (Nz * Ny);
        int ixz = ix * Nz + iz;
        out[idx] = work[ixz * (Ny + 2) + iy + 1];
    }
}

// Setup padded buffer for X-direction: [0, x0, x1, ..., x_{Nx-1}, 0]
__global__ void kernel_dst3_3d_setup_x(
    const double* __restrict__ in,
    double* __restrict__ work,
    int Nx, int Ny, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = Nx + 2;
    int total = Ny * Nz * stride;

    if (idx < total) {
        int pos = idx % stride;
        int iyz = idx / stride;
        int iz = iyz % Nz;
        int iy = iyz / Nz;

        if (pos == 0 || pos == Nx + 1) {
            work[idx] = 0.0;
        } else {
            work[idx] = in[((pos - 1) * Ny + iy) * Nz + iz];
        }
    }
}

// PreOp twiddle factor for X-direction
__global__ void kernel_dst3_3d_preOp_x(
    double* __restrict__ data,
    int Nx, int Ny, int Nz)
{
    int itx = threadIdx.x;
    int iyz = blockIdx.x;
    double* pin = data + iyz * (Nx + 2);

    if (itx <= Nx / 2) {
        double sina, cosa;
        sincos(itx * M_PI / (2.0 * Nx), &sina, &cosa);
        double Ta = pin[itx] + pin[Nx - itx];
        double Tb = pin[itx] - pin[Nx - itx];
        pin[itx] = Ta * cosa + Tb * sina;
        pin[Nx - itx] = Ta * sina - Tb * cosa;
    }
}

// PostOp for X-direction
__global__ void kernel_dst3_3d_postOp_x(
    double* __restrict__ data,
    int Nx, int Ny, int Nz)
{
    extern __shared__ double sh_in[];
    int itx = threadIdx.x;
    int iyz = blockIdx.x;
    double* pin = data + iyz * (Nx + 2);

    if (itx < Nx / 2 + 1) {
        sh_in[itx] = pin[itx];
        sh_in[itx + Nx / 2 + 1] = pin[itx + Nx / 2 + 1];
    }
    __syncthreads();

    if (itx != 0 && itx < Nx / 2 + 1) {
        sh_in[itx * 2 + 1] = -sh_in[itx * 2 + 1];
    }
    __syncthreads();

    if (itx < Nx / 2 + 1) {
        if (itx == 0) {
            pin[0] = 0.0;
            pin[1] = sh_in[0];
        } else {
            pin[2 * itx] = (sh_in[itx * 2 + 1] - sh_in[itx * 2]);
            if (itx * 2 + 1 < Nx + 1) {
                pin[2 * itx + 1] = (sh_in[itx * 2] + sh_in[itx * 2 + 1]);
            }
        }
    }
}

// Copy result back for X-direction
__global__ void kernel_dst3_3d_copy_x(
    const double* __restrict__ work,
    double* __restrict__ out,
    int Nx, int Ny, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny * Nz;

    if (idx < total) {
        int iz = idx % Nz;
        int iy = (idx / Nz) % Ny;
        int ix = idx / (Nz * Ny);
        int iyz = iy * Nz + iz;
        out[idx] = work[iyz * (Nx + 2) + ix + 1];
    }
}

//------------------------------------------------------------------------------
// 3D DST-4 Kernels
//------------------------------------------------------------------------------

__global__ void kernel_dst4_3d_preOp_z(
    const double* __restrict__ in,
    cufftDoubleComplex* __restrict__ out,
    int Nx, int Ny, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N2 = 2 * Nz;
    int total = Nx * Ny * N2;

    if (idx < total) {
        int iz = idx % N2;
        int iy = (idx / N2) % Ny;
        int ix = idx / (N2 * Ny);

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

__global__ void kernel_dst4_3d_postOp_z(
    const cufftDoubleComplex* __restrict__ in,
    double* __restrict__ out,
    int Nx, int Ny, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny * Nz;
    int N2 = 2 * Nz;

    if (idx < total) {
        int iz = idx % Nz;
        int iy = (idx / Nz) % Ny;
        int ix = idx / (Nz * Ny);

        int in_idx = (ix * Ny + iy) * N2 + iz;
        double angle = -M_PI * iz / (2.0 * Nz);
        cufftDoubleComplex z = in[in_idx];
        double im = z.x * sin(angle) + z.y * cos(angle);
        out[idx] = -2.0 * im;
    }
}

__global__ void kernel_dst4_3d_preOp_y(
    const double* __restrict__ in,
    cufftDoubleComplex* __restrict__ out,
    int Nx, int Ny, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N2 = 2 * Ny;
    int total = Nx * Nz * N2;

    if (idx < total) {
        int iy = idx % N2;
        int iz = (idx / N2) % Nz;
        int ix = idx / (N2 * Nz);

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

__global__ void kernel_dst4_3d_postOp_y(
    const cufftDoubleComplex* __restrict__ in,
    double* __restrict__ out,
    int Nx, int Ny, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny * Nz;
    int N2 = 2 * Ny;

    if (idx < total) {
        int iz = idx % Nz;
        int iy = (idx / Nz) % Ny;
        int ix = idx / (Nz * Ny);

        int in_idx = (ix * Nz + iz) * N2 + iy;
        double angle = -M_PI * iy / (2.0 * Ny);
        cufftDoubleComplex z = in[in_idx];
        double im = z.x * sin(angle) + z.y * cos(angle);
        out[idx] = -2.0 * im;
    }
}

__global__ void kernel_dst4_3d_preOp_x(
    const double* __restrict__ in,
    cufftDoubleComplex* __restrict__ out,
    int Nx, int Ny, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N2 = 2 * Nx;
    int total = Ny * Nz * N2;

    if (idx < total) {
        int ix = idx % N2;
        int iz = (idx / N2) % Nz;
        int iy = idx / (N2 * Nz);

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

__global__ void kernel_dst4_3d_postOp_x(
    const cufftDoubleComplex* __restrict__ in,
    double* __restrict__ out,
    int Nx, int Ny, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny * Nz;
    int N2 = 2 * Nx;

    if (idx < total) {
        int iz = idx % Nz;
        int iy = (idx / Nz) % Ny;
        int ix = idx / (Nz * Ny);

        int in_idx = (iy * Nz + iz) * N2 + ix;
        double angle = -M_PI * ix / (2.0 * Nx);
        cufftDoubleComplex z = in[in_idx];
        double im = z.x * sin(angle) + z.y * cos(angle);
        out[idx] = -2.0 * im;
    }
}

//==============================================================================
// CudaRealTransform1D Implementation
//==============================================================================

CudaRealTransform1D::CudaRealTransform1D(int N, CudaTransformType type)
    : N_(N), type_(type), initialized_(false), d_work_(nullptr), d_x1_(nullptr)
{
    int fftSize = getFFTSize(type_, N_);
    bool z2z = usesZ2Z(type_);

    // Allocate buffers
    if (z2z) {
        cudaMalloc(&d_work_, sizeof(cufftDoubleComplex) * fftSize);
        cudaMalloc(&d_x1_, sizeof(cufftDoubleComplex) * fftSize);
    } else {
        int workSize = fftSize;
        int complexSize = fftSize / 2 + 1;
        cudaMalloc(&d_work_, sizeof(double) * workSize);
        cudaMalloc(&d_x1_, sizeof(cufftDoubleComplex) * complexSize);
    }

    // Create FFT plan
    if (z2z) {
        if (cufftPlan1d(&plan_, fftSize, CUFFT_Z2Z, 1) != CUFFT_SUCCESS) {
            throw std::runtime_error("Failed to create cuFFT Z2Z plan");
        }
    } else if (type_ == CUDA_DCT_1) {
        // DCT-1 uses Z2D (complex to real) FFT of size N-1 (where N_ = N+1 points)
        if (cufftPlan1d(&plan_, N_ - 1, CUFFT_Z2D, 1) != CUFFT_SUCCESS) {
            throw std::runtime_error("Failed to create cuFFT Z2D plan for DCT-1");
        }
    } else if (type_ == CUDA_DCT_2) {
        if (cufftPlan1d(&plan_, N_, CUFFT_Z2D, 1) != CUFFT_SUCCESS) {
            throw std::runtime_error("Failed to create cuFFT Z2D plan");
        }
    } else if (type_ == CUDA_DCT_3) {
        if (cufftPlan1d(&plan_, N_, CUFFT_D2Z, 1) != CUFFT_SUCCESS) {
            throw std::runtime_error("Failed to create cuFFT D2Z plan");
        }
    } else if (type_ == CUDA_DST_1) {
        // DST-1: Z2D FFT of size N+1 (FST algorithm)
        if (cufftPlan1d(&plan_, N_ + 1, CUFFT_Z2D, 1) != CUFFT_SUCCESS) {
            throw std::runtime_error("Failed to create cuFFT Z2D plan for DST-1");
        }
    } else if (type_ == CUDA_DST_2) {
        // DST-2: Z2D FFT of size N (FST algorithm)
        if (cufftPlan1d(&plan_, N_, CUFFT_Z2D, 1) != CUFFT_SUCCESS) {
            throw std::runtime_error("Failed to create cuFFT Z2D plan for DST-2");
        }
    } else if (type_ == CUDA_DST_3) {
        // DST-3: D2Z FFT of size N (FST algorithm)
        if (cufftPlan1d(&plan_, N_, CUFFT_D2Z, 1) != CUFFT_SUCCESS) {
            throw std::runtime_error("Failed to create cuFFT D2Z plan for DST-3");
        }
    } else {
        if (cufftPlan1d(&plan_, fftSize, CUFFT_D2Z, 1) != CUFFT_SUCCESS) {
            throw std::runtime_error("Failed to create cuFFT D2Z plan");
        }
    }

    initialized_ = true;
}

CudaRealTransform1D::~CudaRealTransform1D()
{
    if (initialized_) {
        if (d_work_) cudaFree(d_work_);
        if (d_x1_) cudaFree(d_x1_);
        cufftDestroy(plan_);
    }
}

void CudaRealTransform1D::execute(double* d_data)
{
    if (!initialized_) {
        throw std::runtime_error("CudaRealTransform1D not initialized");
    }

    int nThread = pow2roundup(N_);

    switch (type_) {
        case CUDA_DCT_1: {
            // FCT/FST style (Makhoul 1980): copy -> preOp -> Z2D FFT -> postOp -> copy back
            int N = N_ - 1;  // FFT size (N_ = N+1 points)
            int nT = pow2roundup(N / 2);

            // Copy input to work buffer and zero-pad
            cudaMemcpy(d_work_, d_data, sizeof(double) * (N + 1), cudaMemcpyDeviceToDevice);
            cudaMemset(d_work_ + N + 1, 0, sizeof(double));

            // PreOp modifies work buffer in place
            kernel_dct1_preOp<<<1, nT, nT * sizeof(double)>>>(d_work_, d_x1_, N, nT);

            // Z2D FFT (in-place: treat d_work_ as complex, output real)
            cufftExecZ2D(plan_, reinterpret_cast<cufftDoubleComplex*>(d_work_), d_work_);

            // PostOp
            kernel_dct1_postOp<<<1, N / 2 + 1>>>(d_work_, d_x1_, N);

            // Copy result back
            cudaMemcpy(d_data, d_work_, sizeof(double) * (N + 1), cudaMemcpyDeviceToDevice);
            break;
        }
        case CUDA_DCT_2: {
            kernel_dct2_preOp<<<1, nThread, sizeof(double) * nThread>>>(d_data, N_);
            cufftExecZ2D(plan_, (cufftDoubleComplex*)d_data, d_work_);
            kernel_dct2_postOp<<<1, nThread, sizeof(double) * nThread>>>(d_work_, N_);
            cudaMemcpy(d_data, d_work_, sizeof(double) * N_, cudaMemcpyDeviceToDevice);
            break;
        }
        case CUDA_DCT_3: {
            kernel_dct3_preOp<<<1, nThread>>>(d_data, N_);
            cufftExecD2Z(plan_, d_data, (cufftDoubleComplex*)d_work_);
            kernel_dct3_postOp<<<1, nThread, sizeof(double) * (nThread + 2)>>>(d_work_, N_);
            cudaMemcpy(d_data, d_work_, sizeof(double) * N_, cudaMemcpyDeviceToDevice);
            break;
        }
        case CUDA_DCT_4: {
            int N2 = 2 * N_;
            int blockSize = 256;
            int numBlocks = (N2 + blockSize - 1) / blockSize;
            cufftDoubleComplex* d_z_in = (cufftDoubleComplex*)d_work_;
            cufftDoubleComplex* d_z_out = (cufftDoubleComplex*)d_x1_;
            kernel_dct4_preOp<<<numBlocks, blockSize>>>(d_data, d_z_in, N_);
            cufftExecZ2Z(plan_, d_z_in, d_z_out, CUFFT_FORWARD);
            kernel_dct4_postOp<<<(N_ + blockSize - 1) / blockSize, blockSize>>>(d_z_out, d_data, N_);
            break;
        }
        case CUDA_DST_1: {
            // FCT/FST style (Makhoul 1980) with size M = N+1
            // Input mapping: buffer[1..N] = input[0..N-1], buffer[0] = 0
            int M = N_ + 1;  // FST size
            int half = (M + 1) / 2;  // threads needed for loading
            int nT = std::max(half, M / 2 + 1);  // threads needed for both loading and computing
            // Setup buffer: [0, x0, x1, ..., x_{N-1}, 0, 0]
            cudaMemset(d_work_, 0, sizeof(double) * (M + 2));
            cudaMemcpy(d_work_ + 1, d_data, sizeof(double) * N_, cudaMemcpyDeviceToDevice);
            // PreOp
            kernel_dst1_preOp<<<1, nT, M * sizeof(double)>>>(d_work_, M);
            // Z2D FFT of size M
            cufftExecZ2D(plan_, reinterpret_cast<cufftDoubleComplex*>(d_work_), d_work_);
            // PostOp
            kernel_dst1_postOp<<<1, M / 2 + 1>>>(d_work_, M);
            // Output mapping: result[0..N-1] = buffer[1..N]
            cudaMemcpy(d_data, d_work_ + 1, sizeof(double) * N_, cudaMemcpyDeviceToDevice);
            break;
        }
        case CUDA_DST_2: {
            // FCT/FST style (Makhoul 1980): input at buffer[1..N], output at buffer[1..N]
            // Setup buffer: [0, x0, x1, ..., x_{N-1}, 0]
            cudaMemset(d_work_, 0, sizeof(double) * (N_ + 2));
            cudaMemcpy(d_work_ + 1, d_data, sizeof(double) * N_, cudaMemcpyDeviceToDevice);
            // PreOp (reads from buffer[1..N]) - needs N/2+1 threads
            kernel_dst2_preOp<<<1, N_ / 2 + 1, N_ * sizeof(double)>>>(d_work_, N_);
            // Z2D FFT of size N
            cufftExecZ2D(plan_, reinterpret_cast<cufftDoubleComplex*>(d_work_), d_work_);
            // PostOp
            kernel_dst2_postOp<<<1, N_ / 2 + 1, N_ * sizeof(double)>>>(d_work_, N_);
            // Output mapping: result[0..N-1] = buffer[1..N]
            cudaMemcpy(d_data, d_work_ + 1, sizeof(double) * N_, cudaMemcpyDeviceToDevice);
            break;
        }
        case CUDA_DST_3: {
            // FCT/FST style (Makhoul 1980): input at buffer[1..N] (buffer[0]=0), output at buffer[1..N]
            // Setup buffer: [0, x0, x1, ..., x_{N-1}, 0]
            cudaMemset(d_work_, 0, sizeof(double) * (N_ + 2));
            cudaMemcpy(d_work_ + 1, d_data, sizeof(double) * N_, cudaMemcpyDeviceToDevice);
            // PreOp (reads buffer[0..N])
            kernel_dst3_preOp<<<1, N_ / 2 + 1>>>(d_work_, N_);
            // D2Z FFT of size N
            cufftExecD2Z(plan_, d_work_, reinterpret_cast<cufftDoubleComplex*>(d_work_));
            // PostOp
            kernel_dst3_postOp<<<1, N_ / 2 + 1, (N_ + 2) * sizeof(double)>>>(d_work_, N_);
            // Output mapping: result[0..N-1] = buffer[1..N]
            cudaMemcpy(d_data, d_work_ + 1, sizeof(double) * N_, cudaMemcpyDeviceToDevice);
            break;
        }
        case CUDA_DST_4: {
            int N2 = 2 * N_;
            int blockSize = 256;
            int numBlocks = (N2 + blockSize - 1) / blockSize;
            cufftDoubleComplex* d_z_in = (cufftDoubleComplex*)d_work_;
            cufftDoubleComplex* d_z_out = (cufftDoubleComplex*)d_x1_;
            kernel_dst4_preOp<<<numBlocks, blockSize>>>(d_data, d_z_in, N_);
            cufftExecZ2Z(plan_, d_z_in, d_z_out, CUFFT_FORWARD);
            kernel_dst4_postOp<<<(N_ + blockSize - 1) / blockSize, blockSize>>>(d_z_out, d_data, N_);
            break;
        }
        default:
            throw std::runtime_error("Unsupported transform type");
    }
}

int CudaRealTransform1D::get_size() const
{
    return N_;
}

double CudaRealTransform1D::get_normalization() const
{
    return getNormFactor(type_, N_);
}

//==============================================================================
// CudaRealTransform2D Implementation
//==============================================================================

CudaRealTransform2D::CudaRealTransform2D(int Nx, int Ny, CudaTransformType type)
    : CudaRealTransform2D(Nx, Ny, type, type)
{
}

CudaRealTransform2D::CudaRealTransform2D(int Nx, int Ny, CudaTransformType type_x, CudaTransformType type_y)
    : Nx_(Nx), Ny_(Ny), type_x_(type_x), type_y_(type_y), initialized_(false),
      d_work_(nullptr), d_temp_(nullptr), d_x1_(nullptr)
{
    // Type-1 cannot be mixed with other types
    if ((isType1(type_x_) || isType1(type_y_)) && type_x_ != type_y_) {
        throw std::runtime_error("Type-1 transforms (DCT-1/DST-1) cannot be mixed with other types");
    }

    init();
}

void CudaRealTransform2D::init()
{
    M_ = Nx_ * Ny_;

    // Calculate FFT sizes
    int fftSize_y = getFFTSize(type_y_, Ny_);
    int fftSize_x = getFFTSize(type_x_, Nx_);

    bool z2z_y = usesZ2Z(type_y_);
    bool z2z_x = usesZ2Z(type_x_);

    // Calculate buffer sizes
    size_t work_size_y = z2z_y ? sizeof(cufftDoubleComplex) * Nx_ * fftSize_y
                               : sizeof(double) * Nx_ * fftSize_y;
    size_t work_size_x = z2z_x ? sizeof(cufftDoubleComplex) * Ny_ * fftSize_x
                               : sizeof(double) * Ny_ * fftSize_x;

    size_t complex_size_y = z2z_y ? sizeof(cufftDoubleComplex) * Nx_ * fftSize_y
                                  : sizeof(cufftDoubleComplex) * Nx_ * (fftSize_y / 2 + 1);
    size_t complex_size_x = z2z_x ? sizeof(cufftDoubleComplex) * Ny_ * fftSize_x
                                  : sizeof(cufftDoubleComplex) * Ny_ * (fftSize_x / 2 + 1);

    // Allocate buffers
    cudaMalloc(&d_work_, std::max(work_size_y, work_size_x));
    cudaMalloc(&d_temp_, sizeof(double) * M_);
    cudaMalloc(&d_x1_, std::max(complex_size_y, complex_size_x));

    // Create FFT plans for Y dimension
    if (z2z_y) {
        if (cufftPlan1d(&plan_y_, fftSize_y, CUFFT_Z2Z, Nx_) != CUFFT_SUCCESS) {
            throw std::runtime_error("Failed to create cuFFT Z2Z plan for Y");
        }
    } else if (type_y_ == CUDA_DCT_2) {
        int n[1] = {Ny_};
        int inembed[1] = {Ny_ / 2 + 1};
        int onembed[1] = {Ny_};
        if (cufftPlanMany(&plan_y_, 1, n, inembed, 1, Ny_ / 2 + 1,
                          onembed, 1, Ny_, CUFFT_Z2D, Nx_) != CUFFT_SUCCESS) {
            throw std::runtime_error("Failed to create cuFFT Z2D plan for Y");
        }
    } else if (type_y_ == CUDA_DCT_3) {
        int n[1] = {Ny_};
        int inembed[1] = {Ny_};
        int onembed[1] = {Ny_ / 2 + 1};
        if (cufftPlanMany(&plan_y_, 1, n, inembed, 1, Ny_,
                          onembed, 1, Ny_ / 2 + 1, CUFFT_D2Z, Nx_) != CUFFT_SUCCESS) {
            throw std::runtime_error("Failed to create cuFFT D2Z plan for Y");
        }
    } else if (type_y_ == CUDA_DCT_1) {
        // DCT-1 2D
        M_padded_ = 2 * Nx_ * 2 * Ny_;
        cudaFree(d_work_);
        cudaMalloc(&d_work_, sizeof(double) * M_padded_);
        cudaFree(d_x1_);
        cudaMalloc(&d_x1_, sizeof(cufftDoubleComplex) * 2 * Nx_ * (Ny_ + 1));
        if (cufftPlan2d(&plan_y_, 2 * Nx_, 2 * Ny_, CUFFT_D2Z) != CUFFT_SUCCESS) {
            throw std::runtime_error("Failed to create cuFFT D2Z 2D plan for DCT-1");
        }
        plan_x_ = 0;
        initialized_ = true;
        return;
    } else if (type_y_ == CUDA_DST_1) {
        // DST-1 2D
        int ext_Nx = 2 * (Nx_ + 1);
        int ext_Ny = 2 * (Ny_ + 1);
        M_padded_ = ext_Nx * ext_Ny;
        cudaFree(d_work_);
        cudaMalloc(&d_work_, sizeof(double) * M_padded_);
        cudaFree(d_x1_);
        cudaMalloc(&d_x1_, sizeof(cufftDoubleComplex) * ext_Nx * (Ny_ + 2));
        if (cufftPlan2d(&plan_y_, ext_Nx, ext_Ny, CUFFT_D2Z) != CUFFT_SUCCESS) {
            throw std::runtime_error("Failed to create cuFFT D2Z 2D plan for DST-1");
        }
        plan_x_ = 0;
        initialized_ = true;
        return;
    } else if (type_y_ == CUDA_DST_2) {
        // DST-2: Z2D FFT (size Ny) - FCT/FST style (Makhoul 1980)
        int n[1] = {Ny_};
        int inembed[1] = {Ny_ / 2 + 1};
        int onembed[1] = {Ny_};
        if (cufftPlanMany(&plan_y_, 1, n, inembed, 1, Ny_ / 2 + 1,
                          onembed, 1, Ny_, CUFFT_Z2D, Nx_) != CUFFT_SUCCESS) {
            throw std::runtime_error("Failed to create cuFFT Z2D plan for Y (DST-2)");
        }
    } else if (type_y_ == CUDA_DST_3) {
        // DST-3: D2Z FFT (size Ny) with padded buffer - FCT/FST style (Makhoul 1980)
        // Buffer layout: [Nx][Ny+2], FFT operates on first Ny elements of each row
        int stride = Ny_ + 2;
        int n[1] = {Ny_};
        int inembed[1] = {stride};
        int onembed[1] = {stride / 2};  // Output is in-place, interleaved
        if (cufftPlanMany(&plan_y_, 1, n, inembed, 1, stride,
                          onembed, 1, stride / 2, CUFFT_D2Z, Nx_) != CUFFT_SUCCESS) {
            throw std::runtime_error("Failed to create cuFFT D2Z plan for Y (DST-3)");
        }
        // Reallocate work buffer for padded layout
        cudaFree(d_work_);
        cudaMalloc(&d_work_, sizeof(double) * Nx_ * stride);
    } else {
        // Fallback
        int n[1] = {fftSize_y};
        int inembed[1] = {fftSize_y};
        int onembed[1] = {fftSize_y / 2 + 1};
        if (cufftPlanMany(&plan_y_, 1, n, inembed, 1, fftSize_y,
                          onembed, 1, fftSize_y / 2 + 1, CUFFT_D2Z, Nx_) != CUFFT_SUCCESS) {
            throw std::runtime_error("Failed to create cuFFT D2Z plan for Y");
        }
    }

    // Create FFT plans for X dimension
    if (z2z_x) {
        if (cufftPlan1d(&plan_x_, fftSize_x, CUFFT_Z2Z, Ny_) != CUFFT_SUCCESS) {
            throw std::runtime_error("Failed to create cuFFT Z2Z plan for X");
        }
    } else if (type_x_ == CUDA_DCT_2) {
        int n[1] = {Nx_};
        int inembed[1] = {Nx_ / 2 + 1};
        int onembed[1] = {Nx_};
        if (cufftPlanMany(&plan_x_, 1, n, inembed, 1, Nx_ / 2 + 1,
                          onembed, 1, Nx_, CUFFT_Z2D, Ny_) != CUFFT_SUCCESS) {
            throw std::runtime_error("Failed to create cuFFT Z2D plan for X");
        }
    } else if (type_x_ == CUDA_DCT_3) {
        int n[1] = {Nx_};
        int inembed[1] = {Nx_};
        int onembed[1] = {Nx_ / 2 + 1};
        if (cufftPlanMany(&plan_x_, 1, n, inembed, 1, Nx_,
                          onembed, 1, Nx_ / 2 + 1, CUFFT_D2Z, Ny_) != CUFFT_SUCCESS) {
            throw std::runtime_error("Failed to create cuFFT D2Z plan for X");
        }
    } else if (type_x_ == CUDA_DST_2) {
        // DST-2: Z2D FFT (size Nx) - FCT/FST style (Makhoul 1980)
        int n[1] = {Nx_};
        int inembed[1] = {Nx_ / 2 + 1};
        int onembed[1] = {Nx_};
        if (cufftPlanMany(&plan_x_, 1, n, inembed, 1, Nx_ / 2 + 1,
                          onembed, 1, Nx_, CUFFT_Z2D, Ny_) != CUFFT_SUCCESS) {
            throw std::runtime_error("Failed to create cuFFT Z2D plan for X (DST-2)");
        }
    } else if (type_x_ == CUDA_DST_3) {
        // DST-3: D2Z FFT (size Nx) with padded buffer - FCT/FST style (Makhoul 1980)
        // Buffer layout: [Ny][Nx+2], FFT operates on first Nx elements of each row
        int stride = Nx_ + 2;
        int n[1] = {Nx_};
        int inembed[1] = {stride};
        int onembed[1] = {stride / 2};  // Output is in-place, interleaved
        if (cufftPlanMany(&plan_x_, 1, n, inembed, 1, stride,
                          onembed, 1, stride / 2, CUFFT_D2Z, Ny_) != CUFFT_SUCCESS) {
            throw std::runtime_error("Failed to create cuFFT D2Z plan for X (DST-3)");
        }
        // Ensure work buffer is large enough for X direction padded layout
        // d_work_ may have been allocated for Y direction, ensure it's big enough for X too
        size_t y_size = (size_t)Nx_ * (Ny_ + 2);
        size_t x_size = (size_t)Ny_ * (Nx_ + 2);
        if (x_size > y_size) {
            cudaFree(d_work_);
            cudaMalloc(&d_work_, sizeof(double) * x_size);
        }
    } else {
        // Fallback
        int n[1] = {fftSize_x};
        int inembed[1] = {fftSize_x};
        int onembed[1] = {fftSize_x / 2 + 1};
        if (cufftPlanMany(&plan_x_, 1, n, inembed, 1, fftSize_x,
                          onembed, 1, fftSize_x / 2 + 1, CUFFT_D2Z, Ny_) != CUFFT_SUCCESS) {
            throw std::runtime_error("Failed to create cuFFT D2Z plan for X");
        }
    }

    initialized_ = true;
}

CudaRealTransform2D::~CudaRealTransform2D()
{
    if (initialized_) {
        if (d_work_) cudaFree(d_work_);
        if (d_temp_) cudaFree(d_temp_);
        if (d_x1_) cudaFree(d_x1_);
        if (plan_y_) cufftDestroy(plan_y_);
        if (plan_x_) cufftDestroy(plan_x_);
    }
}

void CudaRealTransform2D::executeY_DCT1(double* d_data) {
    // Not implemented for mixed - Type-1 uses 2D FFT approach
}

void CudaRealTransform2D::executeY_DCT2(double* d_data)
{
    int blockSize = 256;
    int total_preOp = Nx_ * (Ny_ / 2 + 1);
    int numBlocks_preOp = (total_preOp + blockSize - 1) / blockSize;
    int total_postOp = Nx_ * Ny_;
    int numBlocks_postOp = (total_postOp + blockSize - 1) / blockSize;

    cufftDoubleComplex* d_complex = (cufftDoubleComplex*)d_x1_;

    kernel_dct2_2d_preOp_y<<<numBlocks_preOp, blockSize>>>(d_data, d_complex, Nx_, Ny_);
    cufftExecZ2D(plan_y_, d_complex, d_work_);
    kernel_dct2_2d_postOp_y<<<numBlocks_postOp, blockSize>>>(d_work_, d_temp_, Nx_, Ny_);
}

void CudaRealTransform2D::executeY_DCT3(double* d_data)
{
    int blockSize = 256;
    int total_preOp = Nx_ * (Ny_ / 2);
    int numBlocks_preOp = (total_preOp + blockSize - 1) / blockSize;
    int total_postOp = Nx_ * Ny_;
    int numBlocks_postOp = (total_postOp + blockSize - 1) / blockSize;

    cudaMemcpy(d_temp_, d_data, sizeof(double) * M_, cudaMemcpyDeviceToDevice);

    kernel_dct3_2d_preOp_y<<<numBlocks_preOp, blockSize>>>(d_temp_, Nx_, Ny_);

    cufftDoubleComplex* d_complex = (cufftDoubleComplex*)d_x1_;
    cufftExecD2Z(plan_y_, d_temp_, d_complex);

    kernel_dct3_2d_postOp_y<<<numBlocks_postOp, blockSize>>>(d_complex, d_temp_, Nx_, Ny_);
}

void CudaRealTransform2D::executeY_DCT4(double* d_data)
{
    int N2 = 2 * Ny_;
    cufftDoubleComplex* d_z_in = (cufftDoubleComplex*)d_work_;
    cufftDoubleComplex* d_z_out = (cufftDoubleComplex*)d_x1_;

    kernel_dct4_2d_preOp_y<<<Nx_, N2>>>(d_data, d_z_in, Nx_, Ny_);
    cufftExecZ2Z(plan_y_, d_z_in, d_z_out, CUFFT_FORWARD);
    kernel_dct4_2d_postOp_y<<<Nx_, Ny_>>>(d_z_out, d_temp_, Nx_, Ny_);
}

void CudaRealTransform2D::executeY_DST1(double* d_data) {
    // Not implemented for mixed - Type-1 uses 2D FFT approach
}

void CudaRealTransform2D::executeY_DST2(double* d_data)
{
    // FCT/FST style (Makhoul 1980): preOp -> Z2D FFT -> postOp
    int blockSize = 256;
    int total_preOp = Nx_ * (Ny_ / 2 + 1);
    int numBlocks_preOp = (total_preOp + blockSize - 1) / blockSize;
    int total_postOp = Nx_ * Ny_;
    int numBlocks_postOp = (total_postOp + blockSize - 1) / blockSize;

    cufftDoubleComplex* d_complex = (cufftDoubleComplex*)d_x1_;

    kernel_dst2_2d_preOp_y<<<numBlocks_preOp, blockSize>>>(d_data, d_complex, Nx_, Ny_);
    cufftExecZ2D(plan_y_, d_complex, d_work_);
    kernel_dst2_2d_postOp_y<<<numBlocks_postOp, blockSize>>>(d_work_, d_temp_, Nx_, Ny_);
}

void CudaRealTransform2D::executeY_DST3(double* d_data)
{
    // FCT/FST style (Makhoul 1980) with padded buffer:
    // 1. Setup padded buffer [Nx][Ny+2] with boundary zeros
    // 2. PreOp (in-place on padded buffer)
    // 3. D2Z FFT (in-place)
    // 4. PostOp (in-place with shared memory)
    // 5. Copy results back

    int blockSize = 256;
    int stride = Ny_ + 2;

    // Step 1: Setup padded buffer
    int total_setup = Nx_ * stride;
    int numBlocks_setup = (total_setup + blockSize - 1) / blockSize;
    kernel_dst3_2d_setup_padded_y<<<numBlocks_setup, blockSize>>>(d_data, d_work_, Nx_, Ny_);

    // Step 2: PreOp on padded buffer (each block handles one row)
    kernel_dst3_2d_preOp_y<<<Nx_, Ny_ / 2 + 1>>>(d_work_, Nx_, Ny_);

    // Step 3: D2Z FFT in-place on padded buffer
    cufftExecD2Z(plan_y_, d_work_, (cufftDoubleComplex*)d_work_);

    // Step 4: PostOp in-place (each block handles one row, needs shared memory)
    size_t sharedSize = sizeof(double) * (Ny_ + 2);
    kernel_dst3_2d_postOp_y<<<Nx_, Ny_ / 2 + 1, sharedSize>>>(d_work_, Nx_, Ny_);

    // Step 5: Copy results back to d_temp_
    int total_copy = Nx_ * Ny_;
    int numBlocks_copy = (total_copy + blockSize - 1) / blockSize;
    kernel_dst3_2d_copy_back_y<<<numBlocks_copy, blockSize>>>(d_work_, d_temp_, Nx_, Ny_);
}

void CudaRealTransform2D::executeY_DST4(double* d_data)
{
    int N2 = 2 * Ny_;
    cufftDoubleComplex* d_z_in = (cufftDoubleComplex*)d_work_;
    cufftDoubleComplex* d_z_out = (cufftDoubleComplex*)d_x1_;

    kernel_dst4_2d_preOp_y<<<Nx_, N2>>>(d_data, d_z_in, Nx_, Ny_);
    cufftExecZ2Z(plan_y_, d_z_in, d_z_out, CUFFT_FORWARD);
    kernel_dst4_2d_postOp_y<<<Nx_, Ny_>>>(d_z_out, d_temp_, Nx_, Ny_);
}

void CudaRealTransform2D::executeX_DCT1(double* d_data) {
    // Not implemented for mixed - Type-1 uses 2D FFT approach
}

void CudaRealTransform2D::executeX_DCT2(double* d_data)
{
    int blockSize = 256;
    int total_preOp = Ny_ * (Nx_ / 2 + 1);
    int numBlocks_preOp = (total_preOp + blockSize - 1) / blockSize;
    int total_postOp = Nx_ * Ny_;
    int numBlocks_postOp = (total_postOp + blockSize - 1) / blockSize;

    cufftDoubleComplex* d_complex = (cufftDoubleComplex*)d_x1_;

    kernel_dct2_2d_preOp_x<<<numBlocks_preOp, blockSize>>>(d_temp_, d_complex, Nx_, Ny_);
    cufftExecZ2D(plan_x_, d_complex, d_work_);
    kernel_dct2_2d_postOp_x<<<numBlocks_postOp, blockSize>>>(d_work_, d_data, Nx_, Ny_);
}

void CudaRealTransform2D::executeX_DCT3(double* d_data)
{
    int blockSize = 256;
    int total_transpose = Nx_ * Ny_;
    int numBlocks_transpose = (total_transpose + blockSize - 1) / blockSize;
    int total_preOp = Ny_ * (Nx_ / 2);
    int numBlocks_preOp = (total_preOp + blockSize - 1) / blockSize;
    int total_postOp = Nx_ * Ny_;
    int numBlocks_postOp = (total_postOp + blockSize - 1) / blockSize;

    kernel_dct3_2d_preOp_x_transpose<<<numBlocks_transpose, blockSize>>>(d_temp_, d_work_, Nx_, Ny_);
    kernel_dct3_2d_preOp_x<<<numBlocks_preOp, blockSize>>>(d_work_, Nx_, Ny_);

    cufftDoubleComplex* d_complex = (cufftDoubleComplex*)d_x1_;
    cufftExecD2Z(plan_x_, d_work_, d_complex);

    kernel_dct3_2d_postOp_x<<<numBlocks_postOp, blockSize>>>(d_complex, d_data, Nx_, Ny_);
}

void CudaRealTransform2D::executeX_DCT4(double* d_data)
{
    int N2 = 2 * Nx_;
    cufftDoubleComplex* d_z_in = (cufftDoubleComplex*)d_work_;
    cufftDoubleComplex* d_z_out = (cufftDoubleComplex*)d_x1_;

    kernel_dct4_2d_preOp_x<<<Ny_, N2>>>(d_temp_, d_z_in, Nx_, Ny_);
    cufftExecZ2Z(plan_x_, d_z_in, d_z_out, CUFFT_FORWARD);
    kernel_dct4_2d_postOp_x<<<Ny_, Nx_>>>(d_z_out, d_data, Nx_, Ny_);
}

void CudaRealTransform2D::executeX_DST1(double* d_data) {
    // Not implemented for mixed - Type-1 uses 2D FFT approach
}

void CudaRealTransform2D::executeX_DST2(double* d_data)
{
    // FCT/FST style (Makhoul 1980): preOp -> Z2D FFT -> postOp
    int blockSize = 256;
    int total_preOp = Ny_ * (Nx_ / 2 + 1);
    int numBlocks_preOp = (total_preOp + blockSize - 1) / blockSize;
    int total_postOp = Nx_ * Ny_;
    int numBlocks_postOp = (total_postOp + blockSize - 1) / blockSize;

    cufftDoubleComplex* d_complex = (cufftDoubleComplex*)d_x1_;

    kernel_dst2_2d_preOp_x<<<numBlocks_preOp, blockSize>>>(d_temp_, d_complex, Nx_, Ny_);
    cufftExecZ2D(plan_x_, d_complex, d_work_);
    kernel_dst2_2d_postOp_x<<<numBlocks_postOp, blockSize>>>(d_work_, d_data, Nx_, Ny_);
}

void CudaRealTransform2D::executeX_DST3(double* d_data)
{
    // FCT/FST style (Makhoul 1980) with padded buffer:
    // 1. Setup padded buffer [Ny][Nx+2] with boundary zeros and transpose from d_temp_
    // 2. PreOp (in-place on padded buffer)
    // 3. D2Z FFT (in-place)
    // 4. PostOp (in-place with shared memory)
    // 5. Copy results back with transpose

    int blockSize = 256;
    int stride = Nx_ + 2;

    // Step 1: Setup padded buffer with transpose
    int total_setup = Ny_ * stride;
    int numBlocks_setup = (total_setup + blockSize - 1) / blockSize;
    kernel_dst3_2d_setup_padded_x<<<numBlocks_setup, blockSize>>>(d_temp_, d_work_, Nx_, Ny_);

    // Step 2: PreOp on padded buffer (each block handles one row)
    kernel_dst3_2d_preOp_x<<<Ny_, Nx_ / 2 + 1>>>(d_work_, Nx_, Ny_);

    // Step 3: D2Z FFT in-place on padded buffer
    cufftExecD2Z(plan_x_, d_work_, (cufftDoubleComplex*)d_work_);

    // Step 4: PostOp in-place (each block handles one row, needs shared memory)
    size_t sharedSize = sizeof(double) * (Nx_ + 2);
    kernel_dst3_2d_postOp_x<<<Ny_, Nx_ / 2 + 1, sharedSize>>>(d_work_, Nx_, Ny_);

    // Step 5: Copy results back to d_data with transpose
    int total_copy = Nx_ * Ny_;
    int numBlocks_copy = (total_copy + blockSize - 1) / blockSize;
    kernel_dst3_2d_copy_back_x<<<numBlocks_copy, blockSize>>>(d_work_, d_data, Nx_, Ny_);
}

void CudaRealTransform2D::executeX_DST4(double* d_data)
{
    int N2 = 2 * Nx_;
    cufftDoubleComplex* d_z_in = (cufftDoubleComplex*)d_work_;
    cufftDoubleComplex* d_z_out = (cufftDoubleComplex*)d_x1_;

    kernel_dst4_2d_preOp_x<<<Ny_, N2>>>(d_temp_, d_z_in, Nx_, Ny_);
    cufftExecZ2Z(plan_x_, d_z_in, d_z_out, CUFFT_FORWARD);
    kernel_dst4_2d_postOp_x<<<Ny_, Nx_>>>(d_z_out, d_data, Nx_, Ny_);
}

void CudaRealTransform2D::execute(double* d_data)
{
    if (!initialized_) {
        throw std::runtime_error("CudaRealTransform2D not initialized");
    }

    // Handle Type-1 transforms (DCT-1 or DST-1) with 2D FFT approach
    if (type_y_ == CUDA_DCT_1) {
        int blockSize = 256;
        int ext_total = 2 * Nx_ * 2 * Ny_;
        int out_total = (Nx_ + 1) * (Ny_ + 1);

        kernel_dct1_2d_mirror_extend<<<(ext_total + blockSize - 1) / blockSize, blockSize>>>(
            d_data, d_work_, Nx_, Ny_);
        cufftExecD2Z(plan_y_, d_work_, (cufftDoubleComplex*)d_x1_);
        kernel_dct1_2d_extract<<<(out_total + blockSize - 1) / blockSize, blockSize>>>(
            (cufftDoubleComplex*)d_x1_, d_data, Nx_, Ny_);
        return;
    }

    if (type_y_ == CUDA_DST_1) {
        int blockSize = 256;
        int ext_Nx = 2 * (Nx_ + 1);
        int ext_Ny = 2 * (Ny_ + 1);
        int ext_total = ext_Nx * ext_Ny;
        int out_total = Nx_ * Ny_;

        kernel_dst1_2d_mirror_extend<<<(ext_total + blockSize - 1) / blockSize, blockSize>>>(
            d_data, d_work_, Nx_, Ny_);
        cufftExecD2Z(plan_y_, d_work_, (cufftDoubleComplex*)d_x1_);
        kernel_dst1_2d_extract<<<(out_total + blockSize - 1) / blockSize, blockSize>>>(
            (cufftDoubleComplex*)d_x1_, d_data, Nx_, Ny_);
        return;
    }

    // Y dimension transform
    switch (type_y_) {
        case CUDA_DCT_2: executeY_DCT2(d_data); break;
        case CUDA_DCT_3: executeY_DCT3(d_data); break;
        case CUDA_DCT_4: executeY_DCT4(d_data); break;
        case CUDA_DST_2: executeY_DST2(d_data); break;
        case CUDA_DST_3: executeY_DST3(d_data); break;
        case CUDA_DST_4: executeY_DST4(d_data); break;
        default:
            throw std::runtime_error("Unsupported transform type for Y dimension");
    }

    // X dimension transform
    switch (type_x_) {
        case CUDA_DCT_2: executeX_DCT2(d_data); break;
        case CUDA_DCT_3: executeX_DCT3(d_data); break;
        case CUDA_DCT_4: executeX_DCT4(d_data); break;
        case CUDA_DST_2: executeX_DST2(d_data); break;
        case CUDA_DST_3: executeX_DST3(d_data); break;
        case CUDA_DST_4: executeX_DST4(d_data); break;
        default:
            throw std::runtime_error("Unsupported transform type for X dimension");
    }
}

void CudaRealTransform2D::get_dims(int& nx, int& ny) const
{
    nx = Nx_;
    ny = Ny_;
}

void CudaRealTransform2D::get_types(CudaTransformType& type_x, CudaTransformType& type_y) const
{
    type_x = type_x_;
    type_y = type_y_;
}

double CudaRealTransform2D::get_normalization() const
{
    return getNormFactor(type_x_, Nx_) * getNormFactor(type_y_, Ny_);
}

//==============================================================================
// CudaRealTransform3D Implementation
//==============================================================================

CudaRealTransform3D::CudaRealTransform3D(int Nx, int Ny, int Nz, CudaTransformType type)
    : CudaRealTransform3D(Nx, Ny, Nz, type, type, type)
{
}

CudaRealTransform3D::CudaRealTransform3D(int Nx, int Ny, int Nz,
                                         CudaTransformType type_x,
                                         CudaTransformType type_y,
                                         CudaTransformType type_z)
    : Nx_(Nx), Ny_(Ny), Nz_(Nz), type_x_(type_x), type_y_(type_y), type_z_(type_z),
      initialized_(false), d_work_(nullptr), d_temp_(nullptr), d_x1_(nullptr)
{
    // Type-1 can only be used when all dimensions have the same Type-1
    bool hasType1 = isType1(type_x_) || isType1(type_y_) || isType1(type_z_);
    bool allSameType1 = (type_x_ == type_y_) && (type_y_ == type_z_) && isType1(type_x_);

    if (hasType1 && !allSameType1) {
        throw std::runtime_error("Type-1 transforms (DCT-1/DST-1) cannot be mixed with other types in 3D");
    }

    init();
}

void CudaRealTransform3D::init()
{
    // DCT-1: uses optimized 3D FFT approach with mirror extension
    if (type_x_ == CUDA_DCT_1 && type_y_ == CUDA_DCT_1 && type_z_ == CUDA_DCT_1) {
        M_ = (Nx_ + 1) * (Ny_ + 1) * (Nz_ + 1);

        int ext_size = 8 * Nx_ * Ny_ * Nz_;
        int freq_size = 2 * Nx_ * 2 * Ny_ * (Nz_ + 1);

        cudaMalloc(&d_work_, sizeof(double) * ext_size);
        cudaMalloc(&d_temp_, sizeof(cufftDoubleComplex) * freq_size);
        d_x1_ = nullptr;

        M_padded_ = ext_size;

        cufftPlan3d(&plan_x_, 2 * Nx_, 2 * Ny_, 2 * Nz_, CUFFT_D2Z);
        plan_y_ = 0;
        plan_z_ = 0;

        initialized_ = true;
        return;
    }

    // DST-1: uses optimized 3D FFT approach with odd extension
    if (type_x_ == CUDA_DST_1 && type_y_ == CUDA_DST_1 && type_z_ == CUDA_DST_1) {
        M_ = Nx_ * Ny_ * Nz_;

        int ext_size = 8 * (Nx_ + 1) * (Ny_ + 1) * (Nz_ + 1);
        int freq_size = 2 * (Nx_ + 1) * 2 * (Ny_ + 1) * (Nz_ + 2);

        cudaMalloc(&d_work_, sizeof(double) * ext_size);
        cudaMalloc(&d_temp_, sizeof(cufftDoubleComplex) * freq_size);
        d_x1_ = nullptr;

        M_padded_ = ext_size;

        cufftPlan3d(&plan_x_, 2 * (Nx_ + 1), 2 * (Ny_ + 1), 2 * (Nz_ + 1), CUFFT_D2Z);
        plan_y_ = 0;
        plan_z_ = 0;

        initialized_ = true;
        return;
    }

    // Types 2-4: use per-dimension processing
    M_ = Nx_ * Ny_ * Nz_;

    // Calculate FFT sizes
    int fftSize_z = getFFTSize(type_z_, Nz_);
    int fftSize_y = getFFTSize(type_y_, Ny_);
    int fftSize_x = getFFTSize(type_x_, Nx_);

    bool z2z_z = usesZ2Z(type_z_);
    bool z2z_y = usesZ2Z(type_y_);
    bool z2z_x = usesZ2Z(type_x_);

    // Calculate buffer sizes
    size_t work_size_z = z2z_z ? sizeof(cufftDoubleComplex) * Nx_ * Ny_ * fftSize_z
                               : sizeof(double) * Nx_ * Ny_ * fftSize_z;
    size_t work_size_y = z2z_y ? sizeof(cufftDoubleComplex) * Nx_ * Nz_ * fftSize_y
                               : sizeof(double) * Nx_ * Nz_ * fftSize_y;
    size_t work_size_x = z2z_x ? sizeof(cufftDoubleComplex) * Ny_ * Nz_ * fftSize_x
                               : sizeof(double) * Ny_ * Nz_ * fftSize_x;

    size_t max_work = std::max({work_size_z, work_size_y, work_size_x});

    size_t complex_size_z = z2z_z ? sizeof(cufftDoubleComplex) * Nx_ * Ny_ * fftSize_z
                                  : sizeof(cufftDoubleComplex) * Nx_ * Ny_ * (fftSize_z / 2 + 1);
    size_t complex_size_y = z2z_y ? sizeof(cufftDoubleComplex) * Nx_ * Nz_ * fftSize_y
                                  : sizeof(cufftDoubleComplex) * Nx_ * Nz_ * (fftSize_y / 2 + 1);
    size_t complex_size_x = z2z_x ? sizeof(cufftDoubleComplex) * Ny_ * Nz_ * fftSize_x
                                  : sizeof(cufftDoubleComplex) * Ny_ * Nz_ * (fftSize_x / 2 + 1);

    size_t max_complex = std::max({complex_size_z, complex_size_y, complex_size_x});

    // Allocate buffers
    cudaMalloc(&d_work_, max_work);
    cudaMalloc(&d_temp_, sizeof(double) * M_);
    cudaMalloc(&d_x1_, max_complex);

    // Create FFT plans for Z dimension (batch of Nx*Ny)
    int batch_z = Nx_ * Ny_;
    if (z2z_z) {
        if (cufftPlan1d(&plan_z_, fftSize_z, CUFFT_Z2Z, batch_z) != CUFFT_SUCCESS) {
            throw std::runtime_error("Failed to create cuFFT Z2Z plan for Z");
        }
    } else if (type_z_ == CUDA_DCT_2) {
        int n[1] = {Nz_};
        int inembed[1] = {Nz_ / 2 + 1};
        int onembed[1] = {Nz_};
        if (cufftPlanMany(&plan_z_, 1, n, inembed, 1, Nz_ / 2 + 1,
                          onembed, 1, Nz_, CUFFT_Z2D, batch_z) != CUFFT_SUCCESS) {
            throw std::runtime_error("Failed to create cuFFT Z2D plan for Z");
        }
    } else if (type_z_ == CUDA_DCT_3) {
        int n[1] = {Nz_};
        int inembed[1] = {Nz_};
        int onembed[1] = {Nz_ / 2 + 1};
        if (cufftPlanMany(&plan_z_, 1, n, inembed, 1, Nz_,
                          onembed, 1, Nz_ / 2 + 1, CUFFT_D2Z, batch_z) != CUFFT_SUCCESS) {
            throw std::runtime_error("Failed to create cuFFT D2Z plan for Z");
        }
    } else if (type_z_ == CUDA_DST_2) {
        // DST-2: Z2D FFT of size Nz - FCT/FST style (Makhoul 1980)
        int n[1] = {Nz_};
        int inembed[1] = {Nz_ / 2 + 1};
        int onembed[1] = {Nz_};
        if (cufftPlanMany(&plan_z_, 1, n, inembed, 1, Nz_ / 2 + 1,
                          onembed, 1, Nz_, CUFFT_Z2D, batch_z) != CUFFT_SUCCESS) {
            throw std::runtime_error("Failed to create cuFFT Z2D plan for Z (DST-2)");
        }
    } else if (type_z_ == CUDA_DST_3) {
        // DST-3: D2Z FFT of size Nz - FCT/FST style (Makhoul 1980) with padded buffer (N+2 elements)
        int n[1] = {Nz_};
        int inembed[1] = {Nz_ + 2};
        int onembed[1] = {Nz_ / 2 + 1};
        if (cufftPlanMany(&plan_z_, 1, n, inembed, 1, Nz_ + 2,
                          onembed, 1, Nz_ / 2 + 1, CUFFT_D2Z, batch_z) != CUFFT_SUCCESS) {
            throw std::runtime_error("Failed to create cuFFT D2Z plan for Z (DST-3)");
        }
    } else {
        int n[1] = {fftSize_z};
        int inembed[1] = {fftSize_z};
        int onembed[1] = {fftSize_z / 2 + 1};
        if (cufftPlanMany(&plan_z_, 1, n, inembed, 1, fftSize_z,
                          onembed, 1, fftSize_z / 2 + 1, CUFFT_D2Z, batch_z) != CUFFT_SUCCESS) {
            throw std::runtime_error("Failed to create cuFFT D2Z plan for Z");
        }
    }

    // Create FFT plans for Y dimension (batch of Nx*Nz)
    int batch_y = Nx_ * Nz_;
    if (z2z_y) {
        if (cufftPlan1d(&plan_y_, fftSize_y, CUFFT_Z2Z, batch_y) != CUFFT_SUCCESS) {
            throw std::runtime_error("Failed to create cuFFT Z2Z plan for Y");
        }
    } else if (type_y_ == CUDA_DCT_2) {
        int n[1] = {Ny_};
        int inembed[1] = {Ny_ / 2 + 1};
        int onembed[1] = {Ny_};
        if (cufftPlanMany(&plan_y_, 1, n, inembed, 1, Ny_ / 2 + 1,
                          onembed, 1, Ny_, CUFFT_Z2D, batch_y) != CUFFT_SUCCESS) {
            throw std::runtime_error("Failed to create cuFFT Z2D plan for Y");
        }
    } else if (type_y_ == CUDA_DCT_3) {
        int n[1] = {Ny_};
        int inembed[1] = {Ny_};
        int onembed[1] = {Ny_ / 2 + 1};
        if (cufftPlanMany(&plan_y_, 1, n, inembed, 1, Ny_,
                          onembed, 1, Ny_ / 2 + 1, CUFFT_D2Z, batch_y) != CUFFT_SUCCESS) {
            throw std::runtime_error("Failed to create cuFFT D2Z plan for Y");
        }
    } else if (type_y_ == CUDA_DST_2) {
        // DST-2: Z2D FFT of size Ny - FCT/FST style (Makhoul 1980)
        int n[1] = {Ny_};
        int inembed[1] = {Ny_ / 2 + 1};
        int onembed[1] = {Ny_};
        if (cufftPlanMany(&plan_y_, 1, n, inembed, 1, Ny_ / 2 + 1,
                          onembed, 1, Ny_, CUFFT_Z2D, batch_y) != CUFFT_SUCCESS) {
            throw std::runtime_error("Failed to create cuFFT Z2D plan for Y (DST-2)");
        }
    } else if (type_y_ == CUDA_DST_3) {
        // DST-3: D2Z FFT of size Ny - FCT/FST style (Makhoul 1980) with padded buffer (N+2 elements)
        int n[1] = {Ny_};
        int inembed[1] = {Ny_ + 2};
        int onembed[1] = {Ny_ / 2 + 1};
        if (cufftPlanMany(&plan_y_, 1, n, inembed, 1, Ny_ + 2,
                          onembed, 1, Ny_ / 2 + 1, CUFFT_D2Z, batch_y) != CUFFT_SUCCESS) {
            throw std::runtime_error("Failed to create cuFFT D2Z plan for Y (DST-3)");
        }
    } else {
        int n[1] = {fftSize_y};
        int inembed[1] = {fftSize_y};
        int onembed[1] = {fftSize_y / 2 + 1};
        if (cufftPlanMany(&plan_y_, 1, n, inembed, 1, fftSize_y,
                          onembed, 1, fftSize_y / 2 + 1, CUFFT_D2Z, batch_y) != CUFFT_SUCCESS) {
            throw std::runtime_error("Failed to create cuFFT D2Z plan for Y");
        }
    }

    // Create FFT plans for X dimension (batch of Ny*Nz)
    int batch_x = Ny_ * Nz_;
    if (z2z_x) {
        if (cufftPlan1d(&plan_x_, fftSize_x, CUFFT_Z2Z, batch_x) != CUFFT_SUCCESS) {
            throw std::runtime_error("Failed to create cuFFT Z2Z plan for X");
        }
    } else if (type_x_ == CUDA_DCT_2) {
        int n[1] = {Nx_};
        int inembed[1] = {Nx_ / 2 + 1};
        int onembed[1] = {Nx_};
        if (cufftPlanMany(&plan_x_, 1, n, inembed, 1, Nx_ / 2 + 1,
                          onembed, 1, Nx_, CUFFT_Z2D, batch_x) != CUFFT_SUCCESS) {
            throw std::runtime_error("Failed to create cuFFT Z2D plan for X");
        }
    } else if (type_x_ == CUDA_DCT_3) {
        int n[1] = {Nx_};
        int inembed[1] = {Nx_};
        int onembed[1] = {Nx_ / 2 + 1};
        if (cufftPlanMany(&plan_x_, 1, n, inembed, 1, Nx_,
                          onembed, 1, Nx_ / 2 + 1, CUFFT_D2Z, batch_x) != CUFFT_SUCCESS) {
            throw std::runtime_error("Failed to create cuFFT D2Z plan for X");
        }
    } else if (type_x_ == CUDA_DST_2) {
        // DST-2: Z2D FFT of size Nx - FCT/FST style (Makhoul 1980)
        int n[1] = {Nx_};
        int inembed[1] = {Nx_ / 2 + 1};
        int onembed[1] = {Nx_};
        if (cufftPlanMany(&plan_x_, 1, n, inembed, 1, Nx_ / 2 + 1,
                          onembed, 1, Nx_, CUFFT_Z2D, batch_x) != CUFFT_SUCCESS) {
            throw std::runtime_error("Failed to create cuFFT Z2D plan for X (DST-2)");
        }
    } else if (type_x_ == CUDA_DST_3) {
        // DST-3: D2Z FFT of size Nx - FCT/FST style (Makhoul 1980) with padded buffer (N+2 elements)
        int n[1] = {Nx_};
        int inembed[1] = {Nx_ + 2};
        int onembed[1] = {Nx_ / 2 + 1};
        if (cufftPlanMany(&plan_x_, 1, n, inembed, 1, Nx_ + 2,
                          onembed, 1, Nx_ / 2 + 1, CUFFT_D2Z, batch_x) != CUFFT_SUCCESS) {
            throw std::runtime_error("Failed to create cuFFT D2Z plan for X (DST-3)");
        }
    } else {
        int n[1] = {fftSize_x};
        int inembed[1] = {fftSize_x};
        int onembed[1] = {fftSize_x / 2 + 1};
        if (cufftPlanMany(&plan_x_, 1, n, inembed, 1, fftSize_x,
                          onembed, 1, fftSize_x / 2 + 1, CUFFT_D2Z, batch_x) != CUFFT_SUCCESS) {
            throw std::runtime_error("Failed to create cuFFT D2Z plan for X");
        }
    }

    initialized_ = true;
}

CudaRealTransform3D::~CudaRealTransform3D()
{
    if (initialized_) {
        if (d_work_) cudaFree(d_work_);
        if (d_temp_) cudaFree(d_temp_);
        if (d_x1_) cudaFree(d_x1_);
        if (plan_z_) cufftDestroy(plan_z_);
        if (plan_y_) cufftDestroy(plan_y_);
        if (plan_x_) cufftDestroy(plan_x_);
    }
}

//------------------------------------------------------------------------------
// 3D Z dimension transforms
//------------------------------------------------------------------------------

void CudaRealTransform3D::executeZ_DCT1(double* d_data) {
    throw std::runtime_error("DCT-1 not supported for 3D mixed transforms");
}

void CudaRealTransform3D::executeZ_DCT2(double* d_data)
{
    int blockSize = 256;
    int half_z = Nz_ / 2 + 1;
    int total_preOp = Nx_ * Ny_ * half_z;
    int numBlocks_preOp = (total_preOp + blockSize - 1) / blockSize;
    int numBlocks_postOp = (M_ + blockSize - 1) / blockSize;

    cufftDoubleComplex* d_complex = (cufftDoubleComplex*)d_x1_;

    kernel_dct2_3d_preOp_z<<<numBlocks_preOp, blockSize>>>(d_data, d_complex, Nx_, Ny_, Nz_);
    cufftExecZ2D(plan_z_, d_complex, d_work_);
    kernel_dct2_3d_postOp_z<<<numBlocks_postOp, blockSize>>>(d_work_, d_temp_, Nx_, Ny_, Nz_);
}

void CudaRealTransform3D::executeZ_DCT3(double* d_data)
{
    int blockSize = 256;
    int total_preOp = Nx_ * Ny_ * (Nz_ / 2);
    int numBlocks_preOp = (total_preOp + blockSize - 1) / blockSize;
    int numBlocks_postOp = (M_ + blockSize - 1) / blockSize;

    cudaMemcpy(d_temp_, d_data, sizeof(double) * M_, cudaMemcpyDeviceToDevice);

    kernel_dct3_3d_preOp_z<<<numBlocks_preOp, blockSize>>>(d_temp_, Nx_, Ny_, Nz_);

    cufftDoubleComplex* d_complex = (cufftDoubleComplex*)d_x1_;
    cufftExecD2Z(plan_z_, d_temp_, d_complex);

    kernel_dct3_3d_postOp_z<<<numBlocks_postOp, blockSize>>>(d_complex, d_temp_, Nx_, Ny_, Nz_);
}

void CudaRealTransform3D::executeZ_DCT4(double* d_data)
{
    int blockSize = 256;
    int N2 = 2 * Nz_;
    int total_preOp = Nx_ * Ny_ * N2;
    int numBlocks_preOp = (total_preOp + blockSize - 1) / blockSize;
    int numBlocks_postOp = (M_ + blockSize - 1) / blockSize;

    cufftDoubleComplex* d_z_in = (cufftDoubleComplex*)d_work_;
    cufftDoubleComplex* d_z_out = (cufftDoubleComplex*)d_x1_;

    kernel_dct4_3d_preOp_z<<<numBlocks_preOp, blockSize>>>(d_data, d_z_in, Nx_, Ny_, Nz_);
    cufftExecZ2Z(plan_z_, d_z_in, d_z_out, CUFFT_FORWARD);
    kernel_dct4_3d_postOp_z<<<numBlocks_postOp, blockSize>>>(d_z_out, d_temp_, Nx_, Ny_, Nz_);
}

void CudaRealTransform3D::executeZ_DST1(double* d_data) {
    throw std::runtime_error("DST-1 not supported for 3D mixed transforms");
}

void CudaRealTransform3D::executeZ_DST2(double* d_data)
{
    // FCT/FST style (Makhoul 1980): preOp -> Z2D FFT -> postOp
    int blockSize = 256;
    int total_preOp = Nx_ * Ny_ * (Nz_ / 2 + 1);
    int numBlocks_preOp = (total_preOp + blockSize - 1) / blockSize;
    int numBlocks_postOp = (M_ + blockSize - 1) / blockSize;

    cufftDoubleComplex* d_complex = (cufftDoubleComplex*)d_x1_;

    kernel_dst2_3d_preOp_z<<<numBlocks_preOp, blockSize>>>(d_data, d_complex, Nx_, Ny_, Nz_);
    cufftExecZ2D(plan_z_, d_complex, d_work_);
    kernel_dst2_3d_postOp_z<<<numBlocks_postOp, blockSize>>>(d_work_, d_temp_, Nx_, Ny_, Nz_);
}

void CudaRealTransform3D::executeZ_DST3(double* d_data)
{
    // FCT/FST style (Makhoul 1980): setup -> preOp -> D2Z FFT -> postOp -> copy
    int blockSize = 256;
    int batch = Nx_ * Ny_;
    int stride = Nz_ + 2;
    int total_setup = batch * stride;
    int numBlocks_setup = (total_setup + blockSize - 1) / blockSize;
    int numBlocks_copy = (M_ + blockSize - 1) / blockSize;

    // Setup padded buffer: [0, x0, x1, ..., x_{Nz-1}, 0] for each XY point
    kernel_dst3_3d_setup_z<<<numBlocks_setup, blockSize>>>(d_data, d_work_, Nx_, Ny_, Nz_);

    // PreOp twiddle factors - 1 block per XY point, Nz/2+1 threads per block
    kernel_dst3_3d_preOp_z<<<batch, Nz_ / 2 + 1>>>(d_work_, Nx_, Ny_, Nz_);

    // D2Z FFT
    cufftDoubleComplex* d_complex = (cufftDoubleComplex*)d_x1_;
    cufftExecD2Z(plan_z_, d_work_, d_complex);

    // PostOp - 1 block per XY point, with shared memory
    kernel_dst3_3d_postOp_z<<<batch, Nz_ / 2 + 1, sizeof(double) * (Nz_ + 2)>>>(
        (double*)d_complex, Nx_, Ny_, Nz_);

    // Copy result back from padded buffer
    kernel_dst3_3d_copy_z<<<numBlocks_copy, blockSize>>>((double*)d_complex, d_temp_, Nx_, Ny_, Nz_);
}

void CudaRealTransform3D::executeZ_DST4(double* d_data)
{
    int blockSize = 256;
    int N2 = 2 * Nz_;
    int total_preOp = Nx_ * Ny_ * N2;
    int numBlocks_preOp = (total_preOp + blockSize - 1) / blockSize;
    int numBlocks_postOp = (M_ + blockSize - 1) / blockSize;

    cufftDoubleComplex* d_z_in = (cufftDoubleComplex*)d_work_;
    cufftDoubleComplex* d_z_out = (cufftDoubleComplex*)d_x1_;

    kernel_dst4_3d_preOp_z<<<numBlocks_preOp, blockSize>>>(d_data, d_z_in, Nx_, Ny_, Nz_);
    cufftExecZ2Z(plan_z_, d_z_in, d_z_out, CUFFT_FORWARD);
    kernel_dst4_3d_postOp_z<<<numBlocks_postOp, blockSize>>>(d_z_out, d_temp_, Nx_, Ny_, Nz_);
}

//------------------------------------------------------------------------------
// 3D Y dimension transforms
//------------------------------------------------------------------------------

void CudaRealTransform3D::executeY_DCT1(double* d_data) {
    throw std::runtime_error("DCT-1 not supported for 3D mixed transforms");
}

void CudaRealTransform3D::executeY_DCT2(double* d_data)
{
    int blockSize = 256;
    int half_y = Ny_ / 2 + 1;
    int total_preOp = Nx_ * Nz_ * half_y;
    int numBlocks_preOp = (total_preOp + blockSize - 1) / blockSize;
    int numBlocks_postOp = (M_ + blockSize - 1) / blockSize;

    cufftDoubleComplex* d_complex = (cufftDoubleComplex*)d_x1_;

    kernel_dct2_3d_preOp_y<<<numBlocks_preOp, blockSize>>>(d_temp_, d_complex, Nx_, Ny_, Nz_);
    cufftExecZ2D(plan_y_, d_complex, d_work_);
    kernel_dct2_3d_postOp_y<<<numBlocks_postOp, blockSize>>>(d_work_, d_temp_, Nx_, Ny_, Nz_);
}

void CudaRealTransform3D::executeY_DCT3(double* d_data)
{
    int blockSize = 256;
    int numBlocks_M = (M_ + blockSize - 1) / blockSize;
    int total_preOp = Nx_ * Nz_ * (Ny_ / 2);
    int numBlocks_preOp = (total_preOp + blockSize - 1) / blockSize;

    kernel_dct3_3d_preOp_y_transpose<<<numBlocks_M, blockSize>>>(d_temp_, d_work_, Nx_, Ny_, Nz_);
    kernel_dct3_3d_preOp_y<<<numBlocks_preOp, blockSize>>>(d_work_, Nx_, Ny_, Nz_);

    cufftDoubleComplex* d_complex = (cufftDoubleComplex*)d_x1_;
    cufftExecD2Z(plan_y_, d_work_, d_complex);

    kernel_dct3_3d_postOp_y<<<numBlocks_M, blockSize>>>(d_complex, d_temp_, Nx_, Ny_, Nz_);
}

void CudaRealTransform3D::executeY_DCT4(double* d_data)
{
    int blockSize = 256;
    int N2 = 2 * Ny_;
    int total_preOp = Nx_ * Nz_ * N2;
    int numBlocks_preOp = (total_preOp + blockSize - 1) / blockSize;
    int numBlocks_postOp = (M_ + blockSize - 1) / blockSize;

    cufftDoubleComplex* d_z_in = (cufftDoubleComplex*)d_work_;
    cufftDoubleComplex* d_z_out = (cufftDoubleComplex*)d_x1_;

    kernel_dct4_3d_preOp_y<<<numBlocks_preOp, blockSize>>>(d_temp_, d_z_in, Nx_, Ny_, Nz_);
    cufftExecZ2Z(plan_y_, d_z_in, d_z_out, CUFFT_FORWARD);
    kernel_dct4_3d_postOp_y<<<numBlocks_postOp, blockSize>>>(d_z_out, d_temp_, Nx_, Ny_, Nz_);
}

void CudaRealTransform3D::executeY_DST1(double* d_data) {
    throw std::runtime_error("DST-1 not supported for 3D mixed transforms");
}

void CudaRealTransform3D::executeY_DST2(double* d_data)
{
    // FCT/FST style (Makhoul 1980): preOp (to complex) -> Z2D FFT -> postOp
    int blockSize = 256;
    int total_preOp = Nx_ * Nz_ * (Ny_ / 2 + 1);
    int numBlocks_preOp = (total_preOp + blockSize - 1) / blockSize;
    int numBlocks_postOp = (M_ + blockSize - 1) / blockSize;

    cufftDoubleComplex* d_complex = (cufftDoubleComplex*)d_x1_;

    kernel_dst2_3d_preOp_y<<<numBlocks_preOp, blockSize>>>(d_temp_, d_complex, Nx_, Ny_, Nz_);
    cufftExecZ2D(plan_y_, d_complex, d_work_);
    kernel_dst2_3d_postOp_y<<<numBlocks_postOp, blockSize>>>(d_work_, d_temp_, Nx_, Ny_, Nz_);
}

void CudaRealTransform3D::executeY_DST3(double* d_data)
{
    // FCT/FST style (Makhoul 1980): setup -> preOp -> D2Z FFT -> postOp -> copy
    int blockSize = 256;
    int batch = Nx_ * Nz_;
    int stride = Ny_ + 2;
    int total_setup = batch * stride;
    int numBlocks_setup = (total_setup + blockSize - 1) / blockSize;
    int numBlocks_copy = (M_ + blockSize - 1) / blockSize;

    // Setup padded buffer: [0, x0, x1, ..., x_{Ny-1}, 0] for each XZ point
    kernel_dst3_3d_setup_y<<<numBlocks_setup, blockSize>>>(d_temp_, d_work_, Nx_, Ny_, Nz_);

    // PreOp twiddle factors - 1 block per XZ point, Ny/2+1 threads per block
    kernel_dst3_3d_preOp_y<<<batch, Ny_ / 2 + 1>>>(d_work_, Nx_, Ny_, Nz_);

    // D2Z FFT
    cufftDoubleComplex* d_complex = (cufftDoubleComplex*)d_x1_;
    cufftExecD2Z(plan_y_, d_work_, d_complex);

    // PostOp - 1 block per XZ point, with shared memory
    kernel_dst3_3d_postOp_y<<<batch, Ny_ / 2 + 1, sizeof(double) * (Ny_ + 2)>>>(
        (double*)d_complex, Nx_, Ny_, Nz_);

    // Copy result back from padded buffer
    kernel_dst3_3d_copy_y<<<numBlocks_copy, blockSize>>>((double*)d_complex, d_temp_, Nx_, Ny_, Nz_);
}

void CudaRealTransform3D::executeY_DST4(double* d_data)
{
    int blockSize = 256;
    int N2 = 2 * Ny_;
    int total_preOp = Nx_ * Nz_ * N2;
    int numBlocks_preOp = (total_preOp + blockSize - 1) / blockSize;
    int numBlocks_postOp = (M_ + blockSize - 1) / blockSize;

    cufftDoubleComplex* d_z_in = (cufftDoubleComplex*)d_work_;
    cufftDoubleComplex* d_z_out = (cufftDoubleComplex*)d_x1_;

    kernel_dst4_3d_preOp_y<<<numBlocks_preOp, blockSize>>>(d_temp_, d_z_in, Nx_, Ny_, Nz_);
    cufftExecZ2Z(plan_y_, d_z_in, d_z_out, CUFFT_FORWARD);
    kernel_dst4_3d_postOp_y<<<numBlocks_postOp, blockSize>>>(d_z_out, d_temp_, Nx_, Ny_, Nz_);
}

//------------------------------------------------------------------------------
// 3D X dimension transforms
//------------------------------------------------------------------------------

void CudaRealTransform3D::executeX_DCT1(double* d_data) {
    throw std::runtime_error("DCT-1 not supported for 3D mixed transforms");
}

void CudaRealTransform3D::executeX_DCT2(double* d_data)
{
    int blockSize = 256;
    int half_x = Nx_ / 2 + 1;
    int total_preOp = Ny_ * Nz_ * half_x;
    int numBlocks_preOp = (total_preOp + blockSize - 1) / blockSize;
    int numBlocks_postOp = (M_ + blockSize - 1) / blockSize;

    cufftDoubleComplex* d_complex = (cufftDoubleComplex*)d_x1_;

    kernel_dct2_3d_preOp_x<<<numBlocks_preOp, blockSize>>>(d_temp_, d_complex, Nx_, Ny_, Nz_);
    cufftExecZ2D(plan_x_, d_complex, d_work_);
    kernel_dct2_3d_postOp_x<<<numBlocks_postOp, blockSize>>>(d_work_, d_data, Nx_, Ny_, Nz_);
}

void CudaRealTransform3D::executeX_DCT3(double* d_data)
{
    int blockSize = 256;
    int numBlocks_M = (M_ + blockSize - 1) / blockSize;
    int total_preOp = Ny_ * Nz_ * (Nx_ / 2);
    int numBlocks_preOp = (total_preOp + blockSize - 1) / blockSize;

    kernel_dct3_3d_preOp_x_transpose<<<numBlocks_M, blockSize>>>(d_temp_, d_work_, Nx_, Ny_, Nz_);
    kernel_dct3_3d_preOp_x<<<numBlocks_preOp, blockSize>>>(d_work_, Nx_, Ny_, Nz_);

    cufftDoubleComplex* d_complex = (cufftDoubleComplex*)d_x1_;
    cufftExecD2Z(plan_x_, d_work_, d_complex);

    kernel_dct3_3d_postOp_x<<<numBlocks_M, blockSize>>>(d_complex, d_data, Nx_, Ny_, Nz_);
}

void CudaRealTransform3D::executeX_DCT4(double* d_data)
{
    int blockSize = 256;
    int N2 = 2 * Nx_;
    int total_preOp = Ny_ * Nz_ * N2;
    int numBlocks_preOp = (total_preOp + blockSize - 1) / blockSize;
    int numBlocks_postOp = (M_ + blockSize - 1) / blockSize;

    cufftDoubleComplex* d_z_in = (cufftDoubleComplex*)d_work_;
    cufftDoubleComplex* d_z_out = (cufftDoubleComplex*)d_x1_;

    kernel_dct4_3d_preOp_x<<<numBlocks_preOp, blockSize>>>(d_temp_, d_z_in, Nx_, Ny_, Nz_);
    cufftExecZ2Z(plan_x_, d_z_in, d_z_out, CUFFT_FORWARD);
    kernel_dct4_3d_postOp_x<<<numBlocks_postOp, blockSize>>>(d_z_out, d_data, Nx_, Ny_, Nz_);
}

void CudaRealTransform3D::executeX_DST1(double* d_data) {
    throw std::runtime_error("DST-1 not supported for 3D mixed transforms");
}

void CudaRealTransform3D::executeX_DST2(double* d_data)
{
    // FCT/FST style (Makhoul 1980): preOp (to complex) -> Z2D FFT -> postOp
    int blockSize = 256;
    int total_preOp = Ny_ * Nz_ * (Nx_ / 2 + 1);
    int numBlocks_preOp = (total_preOp + blockSize - 1) / blockSize;
    int numBlocks_postOp = (M_ + blockSize - 1) / blockSize;

    cufftDoubleComplex* d_complex = (cufftDoubleComplex*)d_x1_;

    kernel_dst2_3d_preOp_x<<<numBlocks_preOp, blockSize>>>(d_temp_, d_complex, Nx_, Ny_, Nz_);
    cufftExecZ2D(plan_x_, d_complex, d_work_);
    kernel_dst2_3d_postOp_x<<<numBlocks_postOp, blockSize>>>(d_work_, d_data, Nx_, Ny_, Nz_);
}

void CudaRealTransform3D::executeX_DST3(double* d_data)
{
    // FCT/FST style (Makhoul 1980): setup -> preOp -> D2Z FFT -> postOp -> copy
    int blockSize = 256;
    int batch = Ny_ * Nz_;
    int stride = Nx_ + 2;
    int total_setup = batch * stride;
    int numBlocks_setup = (total_setup + blockSize - 1) / blockSize;
    int numBlocks_copy = (M_ + blockSize - 1) / blockSize;

    // Setup padded buffer: [0, x0, x1, ..., x_{Nx-1}, 0] for each YZ point
    kernel_dst3_3d_setup_x<<<numBlocks_setup, blockSize>>>(d_temp_, d_work_, Nx_, Ny_, Nz_);

    // PreOp twiddle factors - 1 block per YZ point, Nx/2+1 threads per block
    kernel_dst3_3d_preOp_x<<<batch, Nx_ / 2 + 1>>>(d_work_, Nx_, Ny_, Nz_);

    // D2Z FFT
    cufftDoubleComplex* d_complex = (cufftDoubleComplex*)d_x1_;
    cufftExecD2Z(plan_x_, d_work_, d_complex);

    // PostOp - 1 block per YZ point, with shared memory
    kernel_dst3_3d_postOp_x<<<batch, Nx_ / 2 + 1, sizeof(double) * (Nx_ + 2)>>>(
        (double*)d_complex, Nx_, Ny_, Nz_);

    // Copy result back from padded buffer
    kernel_dst3_3d_copy_x<<<numBlocks_copy, blockSize>>>((double*)d_complex, d_data, Nx_, Ny_, Nz_);
}

void CudaRealTransform3D::executeX_DST4(double* d_data)
{
    int blockSize = 256;
    int N2 = 2 * Nx_;
    int total_preOp = Ny_ * Nz_ * N2;
    int numBlocks_preOp = (total_preOp + blockSize - 1) / blockSize;
    int numBlocks_postOp = (M_ + blockSize - 1) / blockSize;

    cufftDoubleComplex* d_z_in = (cufftDoubleComplex*)d_work_;
    cufftDoubleComplex* d_z_out = (cufftDoubleComplex*)d_x1_;

    kernel_dst4_3d_preOp_x<<<numBlocks_preOp, blockSize>>>(d_temp_, d_z_in, Nx_, Ny_, Nz_);
    cufftExecZ2Z(plan_x_, d_z_in, d_z_out, CUFFT_FORWARD);
    kernel_dst4_3d_postOp_x<<<numBlocks_postOp, blockSize>>>(d_z_out, d_data, Nx_, Ny_, Nz_);
}

void CudaRealTransform3D::execute(double* d_data)
{
    if (!initialized_) {
        throw std::runtime_error("CudaRealTransform3D not initialized");
    }

    int threads = 256;

    // DCT-1: optimized 3D FFT approach with mirror extension
    if (type_x_ == CUDA_DCT_1 && type_y_ == CUDA_DCT_1 && type_z_ == CUDA_DCT_1) {
        int ext_size = 8 * Nx_ * Ny_ * Nz_;
        int blocks_ext = (ext_size + threads - 1) / threads;
        int blocks_out = (M_ + threads - 1) / threads;

        cufftDoubleComplex* d_freq = reinterpret_cast<cufftDoubleComplex*>(d_temp_);

        kernel_dct1_3d_mirror_extend<<<blocks_ext, threads>>>(d_data, d_work_, Nx_, Ny_, Nz_);
        cufftExecD2Z(plan_x_, d_work_, d_freq);
        kernel_dct1_3d_extract<<<blocks_out, threads>>>(d_freq, d_data, Nx_, Ny_, Nz_);
        return;
    }

    // DST-1: optimized 3D FFT approach with odd extension
    if (type_x_ == CUDA_DST_1 && type_y_ == CUDA_DST_1 && type_z_ == CUDA_DST_1) {
        int ext_size = 8 * (Nx_ + 1) * (Ny_ + 1) * (Nz_ + 1);
        int blocks_ext = (ext_size + threads - 1) / threads;
        int blocks_out = (M_ + threads - 1) / threads;

        cufftDoubleComplex* d_freq = reinterpret_cast<cufftDoubleComplex*>(d_temp_);

        kernel_dst1_3d_odd_extend<<<blocks_ext, threads>>>(d_data, d_work_, Nx_, Ny_, Nz_);
        cufftExecD2Z(plan_x_, d_work_, d_freq);
        kernel_dst1_3d_extract<<<blocks_out, threads>>>(d_freq, d_data, Nx_, Ny_, Nz_);
        return;
    }

    // Types 2-4: per-dimension processing

    // Z dimension transform
    switch (type_z_) {
        case CUDA_DCT_2: executeZ_DCT2(d_data); break;
        case CUDA_DCT_3: executeZ_DCT3(d_data); break;
        case CUDA_DCT_4: executeZ_DCT4(d_data); break;
        case CUDA_DST_2: executeZ_DST2(d_data); break;
        case CUDA_DST_3: executeZ_DST3(d_data); break;
        case CUDA_DST_4: executeZ_DST4(d_data); break;
        default:
            throw std::runtime_error("Unsupported transform type for Z dimension");
    }

    // Y dimension transform
    switch (type_y_) {
        case CUDA_DCT_2: executeY_DCT2(d_data); break;
        case CUDA_DCT_3: executeY_DCT3(d_data); break;
        case CUDA_DCT_4: executeY_DCT4(d_data); break;
        case CUDA_DST_2: executeY_DST2(d_data); break;
        case CUDA_DST_3: executeY_DST3(d_data); break;
        case CUDA_DST_4: executeY_DST4(d_data); break;
        default:
            throw std::runtime_error("Unsupported transform type for Y dimension");
    }

    // X dimension transform
    switch (type_x_) {
        case CUDA_DCT_2: executeX_DCT2(d_data); break;
        case CUDA_DCT_3: executeX_DCT3(d_data); break;
        case CUDA_DCT_4: executeX_DCT4(d_data); break;
        case CUDA_DST_2: executeX_DST2(d_data); break;
        case CUDA_DST_3: executeX_DST3(d_data); break;
        case CUDA_DST_4: executeX_DST4(d_data); break;
        default:
            throw std::runtime_error("Unsupported transform type for X dimension");
    }
}

void CudaRealTransform3D::get_dims(int& nx, int& ny, int& nz) const
{
    nx = Nx_;
    ny = Ny_;
    nz = Nz_;
}

void CudaRealTransform3D::get_types(CudaTransformType& type_x,
                                     CudaTransformType& type_y,
                                     CudaTransformType& type_z) const
{
    type_x = type_x_;
    type_y = type_y_;
    type_z = type_z_;
}

double CudaRealTransform3D::get_normalization() const
{
    return getNormFactor(type_x_, Nx_) * getNormFactor(type_y_, Ny_) * getNormFactor(type_z_, Nz_);
}
