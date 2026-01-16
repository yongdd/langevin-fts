/**
 * @file CudaSolverSDC.cu
 * @brief CUDA SDC solver implementation.
 *
 * Implements the Spectral Deferred Correction method for solving the
 * modified diffusion equation on GPU using Gauss-Lobatto quadrature nodes.
 */

#include <iostream>
#include <cmath>
#include <algorithm>
#include <utility>

#include "CudaSolverSDC.h"
#include "CudaSolverCNADI.h"
#include "FiniteDifference.h"

// Forward declarations of tridiagonal kernels from CudaSolverCNADI
extern __global__ void tridiagonal(
    const double* __restrict__ d_xl,
    const double* __restrict__ d_xd,
    const double* __restrict__ d_xh,
    double* __restrict__ d_c_star,
    const double* __restrict__ d_d,
    double* __restrict__ d_x,
    const int* __restrict__ d_offset,
    const int REPEAT, const int INTERVAL, const int M);

extern __global__ void tridiagonal_periodic(
    const double* __restrict__ d_xl,
    const double* __restrict__ d_xd,
    const double* __restrict__ d_xh,
    double* __restrict__ d_c_star,
    double* __restrict__ d_q_sparse,
    const double* __restrict__ d_d,
    double* __restrict__ d_x,
    const int* __restrict__ d_offset,
    const int REPEAT, const int INTERVAL, const int M);

// Forward declarations of Crank-Nicolson kernels from CudaSolverCNADI
extern __global__ void compute_crank_3d_step_1(
    BoundaryCondition bc_xl, BoundaryCondition bc_xh,
    BoundaryCondition bc_yl, BoundaryCondition bc_yh,
    BoundaryCondition bc_zl, BoundaryCondition bc_zh,
    const double *d_xl, const double *d_xd, const double *d_xh, const int I,
    const double *d_yl, const double *d_yd, const double *d_yh, const int J,
    const double *d_zl, const double *d_zd, const double *d_zh, const int K,
    double *d_q_out, const double *d_q_in, const int M);

extern __global__ void compute_crank_3d_step_2(
    BoundaryCondition bc_yl, BoundaryCondition bc_yh,
    const double *d_yl, const double *d_yd, const double *d_yh, const int J, const int K,
    double *d_q_out, const double *d_q_star, const double *d_q_in, const int M);

extern __global__ void compute_crank_3d_step_3(
    BoundaryCondition bc_zl, BoundaryCondition bc_zh,
    const double *d_zl, const double *d_zd, const double *d_zh, const int J, const int K,
    double *d_q_out, const double *d_q_dstar, const double *d_q_in, const int M);

extern __global__ void compute_crank_2d_step_1(
    BoundaryCondition bc_xl, BoundaryCondition bc_xh,
    BoundaryCondition bc_yl, BoundaryCondition bc_yh,
    const double *d_xl, const double *d_xd, const double *d_xh, const int I,
    const double *d_yl, const double *d_yd, const double *d_yh, const int J,
    double *d_q_out, const double *d_q_in, const int M);

extern __global__ void compute_crank_2d_step_2(
    BoundaryCondition bc_yl, BoundaryCondition bc_yh,
    const double *d_yl, const double *d_yd, const double *d_yh, const int J,
    double *d_q_out, const double *d_q_star, const double *d_q_in, const int M);

extern __global__ void compute_crank_1d(
    BoundaryCondition bc_xl, BoundaryCondition bc_xh,
    const double *d_xl, const double *d_xd, const double *d_xh,
    double *d_q_out, const double *d_q_in, const int M);

// Helper device functions
__device__ int sdc_max_of_two(int x, int y) { return (x > y) ? x : y; }
__device__ int sdc_min_of_two(int x, int y) { return (x < y) ? x : y; }

// Kernel implementations
__global__ void compute_F_kernel_3d(
    const double* d_q, const double* d_w, double* d_F,
    double alpha_x, double alpha_y, double alpha_z,
    int nx_I, int nx_J, int nx_K,
    BoundaryCondition bc_xl, BoundaryCondition bc_xh,
    BoundaryCondition bc_yl, BoundaryCondition bc_yh,
    BoundaryCondition bc_zl, BoundaryCondition bc_zh,
    int n_grid)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while(idx < n_grid)
    {
        int i = idx / (nx_J * nx_K);
        int j = (idx / nx_K) % nx_J;
        int k = idx % nx_K;

        // Get neighbor values with proper BC handling for x-direction
        // (cell-centered: antisymmetric ghost for absorbing BC)
        double q_im, q_ip;
        if(bc_xl == BoundaryCondition::PERIODIC)
            q_im = d_q[((nx_I + i - 1) % nx_I) * nx_J * nx_K + j * nx_K + k];
        else if(bc_xl == BoundaryCondition::ABSORBING && i == 0)
            q_im = -d_q[idx];  // Antisymmetric ghost
        else
            q_im = d_q[sdc_max_of_two(0, i - 1) * nx_J * nx_K + j * nx_K + k];

        if(bc_xh == BoundaryCondition::PERIODIC)
            q_ip = d_q[((i + 1) % nx_I) * nx_J * nx_K + j * nx_K + k];
        else if(bc_xh == BoundaryCondition::ABSORBING && i == nx_I - 1)
            q_ip = -d_q[idx];  // Antisymmetric ghost
        else
            q_ip = d_q[sdc_min_of_two(nx_I - 1, i + 1) * nx_J * nx_K + j * nx_K + k];

        // Get neighbor values with proper BC handling for y-direction
        // (cell-centered: antisymmetric ghost for absorbing BC)
        double q_jm, q_jp;
        if(bc_yl == BoundaryCondition::PERIODIC)
            q_jm = d_q[i * nx_J * nx_K + ((nx_J + j - 1) % nx_J) * nx_K + k];
        else if(bc_yl == BoundaryCondition::ABSORBING && j == 0)
            q_jm = -d_q[idx];  // Antisymmetric ghost
        else
            q_jm = d_q[i * nx_J * nx_K + sdc_max_of_two(0, j - 1) * nx_K + k];

        if(bc_yh == BoundaryCondition::PERIODIC)
            q_jp = d_q[i * nx_J * nx_K + ((j + 1) % nx_J) * nx_K + k];
        else if(bc_yh == BoundaryCondition::ABSORBING && j == nx_J - 1)
            q_jp = -d_q[idx];  // Antisymmetric ghost
        else
            q_jp = d_q[i * nx_J * nx_K + sdc_min_of_two(nx_J - 1, j + 1) * nx_K + k];

        // Get neighbor values with proper BC handling for z-direction
        // (cell-centered: antisymmetric ghost for absorbing BC)
        double q_km, q_kp;
        if(bc_zl == BoundaryCondition::PERIODIC)
            q_km = d_q[i * nx_J * nx_K + j * nx_K + (nx_K + k - 1) % nx_K];
        else if(bc_zl == BoundaryCondition::ABSORBING && k == 0)
            q_km = -d_q[idx];  // Antisymmetric ghost
        else
            q_km = d_q[i * nx_J * nx_K + j * nx_K + sdc_max_of_two(0, k - 1)];

        if(bc_zh == BoundaryCondition::PERIODIC)
            q_kp = d_q[i * nx_J * nx_K + j * nx_K + (k + 1) % nx_K];
        else if(bc_zh == BoundaryCondition::ABSORBING && k == nx_K - 1)
            q_kp = -d_q[idx];  // Antisymmetric ghost
        else
            q_kp = d_q[i * nx_J * nx_K + j * nx_K + sdc_min_of_two(nx_K - 1, k + 1)];

        double Lx = alpha_x * (q_im + q_ip - 2.0 * d_q[idx]);
        double Ly = alpha_y * (q_jm + q_jp - 2.0 * d_q[idx]);
        double Lz = alpha_z * (q_km + q_kp - 2.0 * d_q[idx]);
        d_F[idx] = Lx + Ly + Lz - d_w[idx] * d_q[idx];

        idx += stride;
    }
}

__global__ void compute_F_kernel_2d(
    const double* d_q, const double* d_w, double* d_F,
    double alpha_x, double alpha_y,
    int nx_I, int nx_J,
    BoundaryCondition bc_xl, BoundaryCondition bc_xh,
    BoundaryCondition bc_yl, BoundaryCondition bc_yh,
    int n_grid)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while(idx < n_grid)
    {
        int i = idx / nx_J;
        int j = idx % nx_J;

        // Get neighbor values with proper BC handling for x-direction
        // (cell-centered: antisymmetric ghost for absorbing BC)
        double q_im, q_ip;
        if(bc_xl == BoundaryCondition::PERIODIC)
            q_im = d_q[((nx_I + i - 1) % nx_I) * nx_J + j];
        else if(bc_xl == BoundaryCondition::ABSORBING && i == 0)
            q_im = -d_q[idx];  // Antisymmetric ghost
        else
            q_im = d_q[sdc_max_of_two(0, i - 1) * nx_J + j];

        if(bc_xh == BoundaryCondition::PERIODIC)
            q_ip = d_q[((i + 1) % nx_I) * nx_J + j];
        else if(bc_xh == BoundaryCondition::ABSORBING && i == nx_I - 1)
            q_ip = -d_q[idx];  // Antisymmetric ghost
        else
            q_ip = d_q[sdc_min_of_two(nx_I - 1, i + 1) * nx_J + j];

        // Get neighbor values with proper BC handling for y-direction
        // (cell-centered: antisymmetric ghost for absorbing BC)
        double q_jm, q_jp;
        if(bc_yl == BoundaryCondition::PERIODIC)
            q_jm = d_q[i * nx_J + (nx_J + j - 1) % nx_J];
        else if(bc_yl == BoundaryCondition::ABSORBING && j == 0)
            q_jm = -d_q[idx];  // Antisymmetric ghost
        else
            q_jm = d_q[i * nx_J + sdc_max_of_two(0, j - 1)];

        if(bc_yh == BoundaryCondition::PERIODIC)
            q_jp = d_q[i * nx_J + (j + 1) % nx_J];
        else if(bc_yh == BoundaryCondition::ABSORBING && j == nx_J - 1)
            q_jp = -d_q[idx];  // Antisymmetric ghost
        else
            q_jp = d_q[i * nx_J + sdc_min_of_two(nx_J - 1, j + 1)];

        double Lx = alpha_x * (q_im + q_ip - 2.0 * d_q[idx]);
        double Ly = alpha_y * (q_jm + q_jp - 2.0 * d_q[idx]);
        d_F[idx] = Lx + Ly - d_w[idx] * d_q[idx];

        idx += stride;
    }
}

__global__ void compute_F_kernel_1d(
    const double* d_q, const double* d_w, double* d_F,
    double alpha_x, int nx_I,
    BoundaryCondition bc_xl, BoundaryCondition bc_xh,
    int n_grid)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while(i < n_grid)
    {
        // Get neighbor values with proper BC handling
        // (cell-centered: antisymmetric ghost for absorbing BC)
        double q_im, q_ip;

        // Lower neighbor
        if(bc_xl == BoundaryCondition::PERIODIC)
            q_im = d_q[(nx_I + i - 1) % nx_I];
        else if(bc_xl == BoundaryCondition::ABSORBING && i == 0)
            q_im = -d_q[i];  // Antisymmetric ghost: q_{-1} = -q_0
        else
            q_im = d_q[sdc_max_of_two(0, i - 1)];  // Reflecting: symmetric ghost (clamp index)

        // Upper neighbor
        if(bc_xh == BoundaryCondition::PERIODIC)
            q_ip = d_q[(i + 1) % nx_I];
        else if(bc_xh == BoundaryCondition::ABSORBING && i == nx_I - 1)
            q_ip = -d_q[i];  // Antisymmetric ghost: q_{N} = -q_{N-1}
        else
            q_ip = d_q[sdc_min_of_two(nx_I - 1, i + 1)];  // Reflecting: symmetric ghost (clamp index)

        double Lx = alpha_x * (q_im + q_ip - 2.0 * d_q[i]);
        d_F[i] = Lx - d_w[i] * d_q[i];

        i += stride;
    }
}

__global__ void apply_exp_dw_kernel(
    double* d_out, const double* d_in, const double* d_exp_dw, int n_grid)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while(idx < n_grid)
    {
        d_out[idx] = d_exp_dw[idx] * d_in[idx];
        idx += stride;
    }
}

__global__ void copy_array_kernel(double* d_out, const double* d_in, int n_grid)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while(idx < n_grid)
    {
        d_out[idx] = d_in[idx];
        idx += stride;
    }
}

// Fused kernel: copy two arrays in a single pass (reduces kernel launch overhead)
__global__ void copy_two_arrays_kernel(
    double* d_out1, const double* d_in1,
    double* d_out2, const double* d_in2,
    int n_grid)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while(idx < n_grid)
    {
        d_out1[idx] = d_in1[idx];
        d_out2[idx] = d_in2[idx];
        idx += stride;
    }
}

// PCG kernels for sparse matrix solve
__global__ void sparse_matvec_kernel(
    const int* __restrict__ d_row_ptr,
    const int* __restrict__ d_col_idx,
    const double* __restrict__ d_values,
    const double* __restrict__ d_x,
    double* __restrict__ d_y,
    int n)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if(row >= n) return;

    double sum = 0.0;
    int row_start = d_row_ptr[row];
    int row_end = d_row_ptr[row + 1];
    for(int j = row_start; j < row_end; j++)
    {
        sum += d_values[j] * d_x[d_col_idx[j]];
    }
    d_y[row] = sum;
}

// Matrix-free matvec for 3D: y = A*x = (I - dtau*D*∇² + dtau*w)*x
// Computes y = (1 + dtau*w + 2*rx + 2*ry + 2*rz)*x - rx*(x[i±1]) - ry*(x[j±1]) - rz*(x[k±1])
// with boundary condition modifications
__global__ void matvec_free_kernel_3d(
    const double* __restrict__ d_x,
    const double* __restrict__ d_w,
    double* __restrict__ d_y,
    double rx, double ry, double rz,
    int nx_I, int nx_J, int nx_K,
    BoundaryCondition bc_xl, BoundaryCondition bc_xh,
    BoundaryCondition bc_yl, BoundaryCondition bc_yh,
    BoundaryCondition bc_zl, BoundaryCondition bc_zh,
    double dtau, int n_grid)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n_grid) return;

    int i = idx / (nx_J * nx_K);
    int j = (idx / nx_K) % nx_J;
    int k = idx % nx_K;

    double x_c = d_x[idx];

    // Base diagonal: 1 + 2*rx + 2*ry + 2*rz + dtau*w
    double diag = 1.0 + 2.0 * rx + 2.0 * ry + 2.0 * rz + dtau * d_w[idx];

    // X-direction neighbors
    // For non-periodic BCs at boundary: set neighbor to 0 (not ghost value)
    // The diagonal modification already accounts for the boundary effect
    double x_im, x_ip;
    if(bc_xl == BoundaryCondition::PERIODIC)
        x_im = d_x[((nx_I + i - 1) % nx_I) * nx_J * nx_K + j * nx_K + k];
    else if(bc_xl == BoundaryCondition::ABSORBING && i == 0) {
        x_im = 0.0;   // No neighbor contribution at boundary
        diag += rx;   // Boundary modification
    } else if(bc_xl == BoundaryCondition::REFLECTING && i == 0) {
        x_im = 0.0;   // No neighbor contribution at boundary
        diag -= rx;
    } else
        x_im = d_x[(i - 1) * nx_J * nx_K + j * nx_K + k];

    if(bc_xh == BoundaryCondition::PERIODIC)
        x_ip = d_x[((i + 1) % nx_I) * nx_J * nx_K + j * nx_K + k];
    else if(bc_xh == BoundaryCondition::ABSORBING && i == nx_I - 1) {
        x_ip = 0.0;
        diag += rx;
    } else if(bc_xh == BoundaryCondition::REFLECTING && i == nx_I - 1) {
        x_ip = 0.0;
        diag -= rx;
    } else
        x_ip = d_x[(i + 1) * nx_J * nx_K + j * nx_K + k];

    // Y-direction neighbors
    double x_jm, x_jp;
    if(bc_yl == BoundaryCondition::PERIODIC)
        x_jm = d_x[i * nx_J * nx_K + ((nx_J + j - 1) % nx_J) * nx_K + k];
    else if(bc_yl == BoundaryCondition::ABSORBING && j == 0) {
        x_jm = 0.0;
        diag += ry;
    } else if(bc_yl == BoundaryCondition::REFLECTING && j == 0) {
        x_jm = 0.0;
        diag -= ry;
    } else
        x_jm = d_x[i * nx_J * nx_K + (j - 1) * nx_K + k];

    if(bc_yh == BoundaryCondition::PERIODIC)
        x_jp = d_x[i * nx_J * nx_K + ((j + 1) % nx_J) * nx_K + k];
    else if(bc_yh == BoundaryCondition::ABSORBING && j == nx_J - 1) {
        x_jp = 0.0;
        diag += ry;
    } else if(bc_yh == BoundaryCondition::REFLECTING && j == nx_J - 1) {
        x_jp = 0.0;
        diag -= ry;
    } else
        x_jp = d_x[i * nx_J * nx_K + (j + 1) * nx_K + k];

    // Z-direction neighbors
    double x_km, x_kp;
    if(bc_zl == BoundaryCondition::PERIODIC)
        x_km = d_x[i * nx_J * nx_K + j * nx_K + (nx_K + k - 1) % nx_K];
    else if(bc_zl == BoundaryCondition::ABSORBING && k == 0) {
        x_km = 0.0;
        diag += rz;
    } else if(bc_zl == BoundaryCondition::REFLECTING && k == 0) {
        x_km = 0.0;
        diag -= rz;
    } else
        x_km = d_x[i * nx_J * nx_K + j * nx_K + k - 1];

    if(bc_zh == BoundaryCondition::PERIODIC)
        x_kp = d_x[i * nx_J * nx_K + j * nx_K + (k + 1) % nx_K];
    else if(bc_zh == BoundaryCondition::ABSORBING && k == nx_K - 1) {
        x_kp = 0.0;
        diag += rz;
    } else if(bc_zh == BoundaryCondition::REFLECTING && k == nx_K - 1) {
        x_kp = 0.0;
        diag -= rz;
    } else
        x_kp = d_x[i * nx_J * nx_K + j * nx_K + k + 1];

    // y = diag*x - rx*(x_im + x_ip) - ry*(x_jm + x_jp) - rz*(x_km + x_kp)
    d_y[idx] = diag * x_c - rx * (x_im + x_ip) - ry * (x_jm + x_jp) - rz * (x_km + x_kp);
}

// Matrix-free matvec for 2D: y = A*x = (I - dtau*D*∇² + dtau*w)*x
__global__ void matvec_free_kernel_2d(
    const double* __restrict__ d_x,
    const double* __restrict__ d_w,
    double* __restrict__ d_y,
    double rx, double ry,
    int nx_I, int nx_J,
    BoundaryCondition bc_xl, BoundaryCondition bc_xh,
    BoundaryCondition bc_yl, BoundaryCondition bc_yh,
    double dtau, int n_grid)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n_grid) return;

    int i = idx / nx_J;
    int j = idx % nx_J;

    double x_c = d_x[idx];

    // Base diagonal: 1 + 2*rx + 2*ry + dtau*w
    double diag = 1.0 + 2.0 * rx + 2.0 * ry + dtau * d_w[idx];

    // X-direction neighbors
    // For non-periodic BCs at boundary: set neighbor to 0 (not ghost value)
    // The diagonal modification already accounts for the boundary effect
    double x_im, x_ip;
    if(bc_xl == BoundaryCondition::PERIODIC)
        x_im = d_x[((nx_I + i - 1) % nx_I) * nx_J + j];
    else if(bc_xl == BoundaryCondition::ABSORBING && i == 0) {
        x_im = 0.0;
        diag += rx;
    } else if(bc_xl == BoundaryCondition::REFLECTING && i == 0) {
        x_im = 0.0;
        diag -= rx;
    } else
        x_im = d_x[(i - 1) * nx_J + j];

    if(bc_xh == BoundaryCondition::PERIODIC)
        x_ip = d_x[((i + 1) % nx_I) * nx_J + j];
    else if(bc_xh == BoundaryCondition::ABSORBING && i == nx_I - 1) {
        x_ip = 0.0;
        diag += rx;
    } else if(bc_xh == BoundaryCondition::REFLECTING && i == nx_I - 1) {
        x_ip = 0.0;
        diag -= rx;
    } else
        x_ip = d_x[(i + 1) * nx_J + j];

    // Y-direction neighbors
    double x_jm, x_jp;
    if(bc_yl == BoundaryCondition::PERIODIC)
        x_jm = d_x[i * nx_J + (nx_J + j - 1) % nx_J];
    else if(bc_yl == BoundaryCondition::ABSORBING && j == 0) {
        x_jm = 0.0;
        diag += ry;
    } else if(bc_yl == BoundaryCondition::REFLECTING && j == 0) {
        x_jm = 0.0;
        diag -= ry;
    } else
        x_jm = d_x[i * nx_J + j - 1];

    if(bc_yh == BoundaryCondition::PERIODIC)
        x_jp = d_x[i * nx_J + (j + 1) % nx_J];
    else if(bc_yh == BoundaryCondition::ABSORBING && j == nx_J - 1) {
        x_jp = 0.0;
        diag += ry;
    } else if(bc_yh == BoundaryCondition::REFLECTING && j == nx_J - 1) {
        x_jp = 0.0;
        diag -= ry;
    } else
        x_jp = d_x[i * nx_J + j + 1];

    // y = diag*x - rx*(x_im + x_ip) - ry*(x_jm + x_jp)
    d_y[idx] = diag * x_c - rx * (x_im + x_ip) - ry * (x_jm + x_jp);
}

// Compute diagonal inverse for Jacobi preconditioner (matrix-free version)
__global__ void compute_diag_inv_kernel_3d(
    const double* __restrict__ d_w,
    double* __restrict__ d_diag_inv,
    double rx, double ry, double rz,
    int nx_I, int nx_J, int nx_K,
    BoundaryCondition bc_xl, BoundaryCondition bc_xh,
    BoundaryCondition bc_yl, BoundaryCondition bc_yh,
    BoundaryCondition bc_zl, BoundaryCondition bc_zh,
    double dtau, int n_grid)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n_grid) return;

    int i = idx / (nx_J * nx_K);
    int j = (idx / nx_K) % nx_J;
    int k = idx % nx_K;

    double diag = 1.0 + 2.0 * rx + 2.0 * ry + 2.0 * rz + dtau * d_w[idx];

    // Boundary modifications
    if(bc_xl == BoundaryCondition::REFLECTING && i == 0) diag -= rx;
    else if(bc_xl == BoundaryCondition::ABSORBING && i == 0) diag += rx;
    if(bc_xh == BoundaryCondition::REFLECTING && i == nx_I - 1) diag -= rx;
    else if(bc_xh == BoundaryCondition::ABSORBING && i == nx_I - 1) diag += rx;
    if(bc_yl == BoundaryCondition::REFLECTING && j == 0) diag -= ry;
    else if(bc_yl == BoundaryCondition::ABSORBING && j == 0) diag += ry;
    if(bc_yh == BoundaryCondition::REFLECTING && j == nx_J - 1) diag -= ry;
    else if(bc_yh == BoundaryCondition::ABSORBING && j == nx_J - 1) diag += ry;
    if(bc_zl == BoundaryCondition::REFLECTING && k == 0) diag -= rz;
    else if(bc_zl == BoundaryCondition::ABSORBING && k == 0) diag += rz;
    if(bc_zh == BoundaryCondition::REFLECTING && k == nx_K - 1) diag -= rz;
    else if(bc_zh == BoundaryCondition::ABSORBING && k == nx_K - 1) diag += rz;

    d_diag_inv[idx] = 1.0 / diag;
}

__global__ void compute_diag_inv_kernel_2d(
    const double* __restrict__ d_w,
    double* __restrict__ d_diag_inv,
    double rx, double ry,
    int nx_I, int nx_J,
    BoundaryCondition bc_xl, BoundaryCondition bc_xh,
    BoundaryCondition bc_yl, BoundaryCondition bc_yh,
    double dtau, int n_grid)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n_grid) return;

    int i = idx / nx_J;
    int j = idx % nx_J;

    double diag = 1.0 + 2.0 * rx + 2.0 * ry + dtau * d_w[idx];

    // Boundary modifications
    if(bc_xl == BoundaryCondition::REFLECTING && i == 0) diag -= rx;
    else if(bc_xl == BoundaryCondition::ABSORBING && i == 0) diag += rx;
    if(bc_xh == BoundaryCondition::REFLECTING && i == nx_I - 1) diag -= rx;
    else if(bc_xh == BoundaryCondition::ABSORBING && i == nx_I - 1) diag += rx;
    if(bc_yl == BoundaryCondition::REFLECTING && j == 0) diag -= ry;
    else if(bc_yl == BoundaryCondition::ABSORBING && j == 0) diag += ry;
    if(bc_yh == BoundaryCondition::REFLECTING && j == nx_J - 1) diag -= ry;
    else if(bc_yh == BoundaryCondition::ABSORBING && j == nx_J - 1) diag += ry;

    d_diag_inv[idx] = 1.0 / diag;
}

// PCG initialization: r = b, z = diag_inv * r, p = z, x = 0
__global__ void pcg_init_kernel(
    double* __restrict__ d_x,
    double* __restrict__ d_r,
    double* __restrict__ d_z,
    double* __restrict__ d_p,
    const double* __restrict__ d_b,
    const double* __restrict__ d_diag_inv,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n) return;

    d_x[idx] = 0.0;
    d_r[idx] = d_b[idx];
    d_z[idx] = d_r[idx] * d_diag_inv[idx];
    d_p[idx] = d_z[idx];
}

// PCG warm-start initialization: x = x0 (already set), r = b - Ax0, z = diag_inv * r, p = z
// d_Ax0 contains A*x0 computed before calling this kernel
__global__ void pcg_init_warmstart_kernel(
    double* __restrict__ d_r,
    double* __restrict__ d_z,
    double* __restrict__ d_p,
    const double* __restrict__ d_b,
    const double* __restrict__ d_Ax0,
    const double* __restrict__ d_diag_inv,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n) return;

    double r_val = d_b[idx] - d_Ax0[idx];
    d_r[idx] = r_val;
    double z_val = r_val * d_diag_inv[idx];
    d_z[idx] = z_val;
    d_p[idx] = z_val;
}

// PCG update: x = x + alpha*p, r = r - alpha*Ap
__global__ void pcg_update_xr_kernel(
    double* __restrict__ d_x,
    double* __restrict__ d_r,
    const double* __restrict__ d_p,
    const double* __restrict__ d_Ap,
    double alpha,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n) return;

    d_x[idx] += alpha * d_p[idx];
    d_r[idx] -= alpha * d_Ap[idx];
}

// PCG preconditioner: z = diag_inv * r
__global__ void pcg_precond_kernel(
    double* __restrict__ d_z,
    const double* __restrict__ d_r,
    const double* __restrict__ d_diag_inv,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n) return;

    d_z[idx] = d_r[idx] * d_diag_inv[idx];
}

// PCG direction update: p = z + beta*p
__global__ void pcg_update_p_kernel(
    double* __restrict__ d_p,
    const double* __restrict__ d_z,
    double beta,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n) return;

    d_p[idx] = d_z[idx] + beta * d_p[idx];
}

// ============================================================
// Device-side scalar computation kernels (avoid GPU-CPU sync)
// Scalar layout: [0]=alpha, [1]=beta, [2]=rz_old, [3]=rz_new, [4]=pAp, [5]=r_norm_sq
// ============================================================

// Compute alpha = rz_old / pAp on GPU
__global__ void pcg_compute_alpha_kernel(double* __restrict__ d_scalars)
{
    if(threadIdx.x == 0 && blockIdx.x == 0)
        d_scalars[0] = d_scalars[2] / d_scalars[4];
}

// Compute beta = rz_new / rz_old, then update rz_old = rz_new
__global__ void pcg_compute_beta_kernel(double* __restrict__ d_scalars)
{
    if(threadIdx.x == 0 && blockIdx.x == 0)
    {
        d_scalars[1] = d_scalars[3] / d_scalars[2];
        d_scalars[2] = d_scalars[3];
    }
}

// Update x and r using alpha from device memory
__global__ void pcg_update_xr_dev_kernel(
    double* __restrict__ d_x,
    double* __restrict__ d_r,
    const double* __restrict__ d_p,
    const double* __restrict__ d_Ap,
    const double* __restrict__ d_scalars,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n) return;
    double alpha = d_scalars[0];
    d_x[idx] += alpha * d_p[idx];
    d_r[idx] -= alpha * d_Ap[idx];
}

// Update p using beta from device memory
__global__ void pcg_update_p_dev_kernel(
    double* __restrict__ d_p,
    const double* __restrict__ d_z,
    const double* __restrict__ d_scalars,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n) return;
    d_p[idx] = d_z[idx] + d_scalars[1] * d_p[idx];
}

// ============================================================
// Fused kernels for PCG optimization
// These kernels combine dot product reduction with scalar computation
// to reduce kernel launch overhead
// ============================================================

// Fused kernel: compute (p, Ap) and then alpha = rz_old / pAp
// Uses two-phase reduction: first reduce within blocks, then final block computes alpha
__global__ void pcg_fused_dot_alpha_kernel(
    const double* __restrict__ d_p,
    const double* __restrict__ d_Ap,
    double* __restrict__ d_scalars,  // scalars[0]=alpha, scalars[2]=rz_old, scalars[4]=pAp
    double* __restrict__ d_partial,  // partial sums from each block
    int n,
    int n_blocks_total)
{
    extern __shared__ double sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load and multiply
    double val = (idx < n) ? d_p[idx] * d_Ap[idx] : 0.0;
    sdata[tid] = val;
    __syncthreads();

    // Reduce within block
    for(int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if(tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // Write partial sum
    if(tid == 0)
        d_partial[blockIdx.x] = sdata[0];
}

// Second phase: reduce partial sums and compute alpha
__global__ void pcg_fused_reduce_alpha_kernel(
    double* __restrict__ d_scalars,
    const double* __restrict__ d_partial,
    int n_blocks)
{
    extern __shared__ double sdata[];

    int tid = threadIdx.x;

    // Load partial sums
    double val = (tid < n_blocks) ? d_partial[tid] : 0.0;
    sdata[tid] = val;
    __syncthreads();

    // Reduce
    for(int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if(tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // Thread 0 computes alpha = rz_old / pAp
    if(tid == 0)
    {
        d_scalars[4] = sdata[0];  // pAp
        d_scalars[0] = d_scalars[2] / sdata[0];  // alpha = rz_old / pAp
    }
}

// Fused kernel: compute (r, z) and then beta = rz_new / rz_old
__global__ void pcg_fused_dot_beta_kernel(
    const double* __restrict__ d_r,
    const double* __restrict__ d_z,
    double* __restrict__ d_scalars,  // scalars[1]=beta, scalars[2]=rz_old, scalars[3]=rz_new
    double* __restrict__ d_partial,
    int n,
    int n_blocks_total)
{
    extern __shared__ double sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load and multiply
    double val = (idx < n) ? d_r[idx] * d_z[idx] : 0.0;
    sdata[tid] = val;
    __syncthreads();

    // Reduce within block
    for(int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if(tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // Write partial sum
    if(tid == 0)
        d_partial[blockIdx.x] = sdata[0];
}

// Second phase: reduce partial sums and compute beta
__global__ void pcg_fused_reduce_beta_kernel(
    double* __restrict__ d_scalars,
    const double* __restrict__ d_partial,
    int n_blocks)
{
    extern __shared__ double sdata[];

    int tid = threadIdx.x;

    // Load partial sums
    double val = (tid < n_blocks) ? d_partial[tid] : 0.0;
    sdata[tid] = val;
    __syncthreads();

    // Reduce
    for(int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if(tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // Thread 0 computes beta = rz_new / rz_old, then updates rz_old
    if(tid == 0)
    {
        double rz_new = sdata[0];
        d_scalars[3] = rz_new;  // rz_new
        d_scalars[1] = rz_new / d_scalars[2];  // beta = rz_new / rz_old
        d_scalars[2] = rz_new;  // rz_old = rz_new for next iteration
    }
}

// Second phase for initialization: only compute rz_old (no beta computation)
__global__ void pcg_fused_reduce_rz_init_kernel(
    double* __restrict__ d_scalars,
    const double* __restrict__ d_partial,
    int n_blocks)
{
    extern __shared__ double sdata[];

    int tid = threadIdx.x;

    // Load partial sums
    double val = (tid < n_blocks) ? d_partial[tid] : 0.0;
    sdata[tid] = val;
    __syncthreads();

    // Reduce
    for(int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if(tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // Thread 0 stores rz_old only (no beta computation)
    if(tid == 0)
    {
        d_scalars[2] = sdata[0];  // rz_old
    }
}

// Fused kernel: update x, r, and apply preconditioner z in one pass
// Combines: x += alpha*p, r -= alpha*Ap, z = M^{-1}*r
// This reduces memory bandwidth by avoiding separate read/write of r
__global__ void pcg_fused_update_xrz_kernel(
    double* __restrict__ d_x,
    double* __restrict__ d_r,
    double* __restrict__ d_z,
    const double* __restrict__ d_p,
    const double* __restrict__ d_Ap,
    const double* __restrict__ d_diag_inv,
    const double* __restrict__ d_scalars,  // scalars[0]=alpha
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n) return;

    double alpha = d_scalars[0];
    d_x[idx] += alpha * d_p[idx];
    double r_new = d_r[idx] - alpha * d_Ap[idx];
    d_r[idx] = r_new;
    d_z[idx] = r_new * d_diag_inv[idx];
}

// ============================================================
// End fused kernels
// ============================================================

// ============================================================
// End device-side scalar kernels
// ============================================================

// Reduction kernels for dot product (using shared memory)
__global__ void dot_product_kernel(
    const double* __restrict__ d_a,
    const double* __restrict__ d_b,
    double* __restrict__ d_result,
    int n)
{
    extern __shared__ double sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load and multiply
    double val = (idx < n) ? d_a[idx] * d_b[idx] : 0.0;
    sdata[tid] = val;
    __syncthreads();

    // Reduce within block
    for(int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if(tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write block result
    if(tid == 0)
    {
        atomicAdd(d_result, sdata[0]);
    }
}

__global__ void sdc_spectral_integral_kernel(
    double* d_rhs,
    const double* d_X_m,
    const double* const* d_F_nodes,
    const double* d_S_row,
    int M_nodes,
    double ds,
    double dtau,
    const double* d_F_mp1,
    int n_grid)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n_grid) return;

    double integral = 0.0;
    for(int j = 0; j < M_nodes; j++)
    {
        integral += d_S_row[j] * d_F_nodes[j][idx] * ds;
    }

    d_rhs[idx] = d_X_m[idx] + integral - dtau * d_F_mp1[idx];
}

// Simpler kernel without pointer arrays
__global__ void sdc_rhs_kernel(
    double* d_rhs,
    const double* d_X_m,
    const double* d_integral,
    double dtau,
    const double* d_F_mp1,
    int n_grid)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while(idx < n_grid)
    {
        // For fully implicit SDC: rhs = X[m] + ∫F dt - dtau * F_old[m+1]
        // The implicit solve handles both diffusion and reaction: (I - dtau*D∇² + dtau*w)
        d_rhs[idx] = d_X_m[idx] + d_integral[idx] - dtau * d_F_mp1[idx];
        idx += stride;
    }
}

__global__ void accumulate_integral_kernel(
    double* d_integral,
    const double* d_F,
    double weight,
    int n_grid)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while(idx < n_grid)
    {
        d_integral[idx] += weight * d_F[idx];
        idx += stride;
    }
}

__global__ void apply_mask_kernel(double* d_q, const double* d_mask, int n_grid)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while(idx < n_grid)
    {
        d_q[idx] *= d_mask[idx];
        idx += stride;
    }
}

// Constructor
CudaSolverSDC::CudaSolverSDC(
    ComputationBox<double>* cb,
    Molecules *molecules,
    int n_streams,
    cudaStream_t streams[MAX_STREAMS][2],
    int M, int K)
{
    try{
        this->cb = cb;
        this->molecules = molecules;
        this->n_streams = n_streams;
        this->M = M;
        this->K = K;

        if(molecules->get_model_name() != "continuous")
            throw_with_line_number("SDC method only supports 'continuous' chain model.");

        if(!cb->is_orthogonal())
            throw_with_line_number("SDC method only supports orthogonal unit cells.");

        if(M < 2)
            throw_with_line_number("SDC requires at least M=2 Gauss-Lobatto nodes.");

        if(K < 0)
            throw_with_line_number("SDC requires K >= 0 correction iterations.");

        const int n_grid = cb->get_total_grid();
        const int DIM = cb->get_dim();
        this->dim = DIM;

        std::vector<int> nx(DIM);
        if(DIM == 3)
            nx = {cb->get_nx(0), cb->get_nx(1), cb->get_nx(2)};
        else if(DIM == 2)
            nx = {cb->get_nx(0), cb->get_nx(1), 1};
        else if(DIM == 1)
            nx = {cb->get_nx(0), 1, 1};

        // Copy streams
        for(int i = 0; i < n_streams; i++)
        {
            this->streams[i][0] = streams[i][0];
            this->streams[i][1] = streams[i][1];
        }

        // Compute Gauss-Lobatto nodes and integration matrix
        compute_gauss_lobatto_nodes();
        compute_integration_matrix();

        // Upload integration matrix to device (flattened)
        gpu_error_check(cudaMalloc((void**)&d_S, sizeof(double) * (M - 1) * M));
        std::vector<double> S_flat((M - 1) * M);
        for(int m = 0; m < M - 1; m++)
            for(int j = 0; j < M; j++)
                S_flat[m * M + j] = S[m][j];
        gpu_error_check(cudaMemcpy(d_S, S_flat.data(), sizeof(double) * (M - 1) * M, cudaMemcpyHostToDevice));

        // Allocate tridiagonal coefficients for each sub-interval
        d_xl.resize(M - 1);
        d_xd.resize(M - 1);
        d_xd_base.resize(M - 1);  // Base diagonal (diffusion only, for 1D)
        d_xh.resize(M - 1);
        d_yl.resize(M - 1);
        d_yd.resize(M - 1);
        d_yh.resize(M - 1);
        d_zl.resize(M - 1);
        d_zd.resize(M - 1);
        d_zh.resize(M - 1);
        dtau_sub.resize(M - 1);  // Sub-interval time steps

        for(int m = 0; m < M - 1; m++)
        {
            for(const auto& item: molecules->get_bond_lengths())
            {
                std::string monomer_type = item.first;

                gpu_error_check(cudaMalloc((void**)&d_xl[m][monomer_type], sizeof(double) * nx[0]));
                gpu_error_check(cudaMalloc((void**)&d_xd[m][monomer_type], sizeof(double) * nx[0]));
                gpu_error_check(cudaMalloc((void**)&d_xd_base[m][monomer_type], sizeof(double) * nx[0]));  // For 1D
                gpu_error_check(cudaMalloc((void**)&d_xh[m][monomer_type], sizeof(double) * nx[0]));

                gpu_error_check(cudaMalloc((void**)&d_yl[m][monomer_type], sizeof(double) * nx[1]));
                gpu_error_check(cudaMalloc((void**)&d_yd[m][monomer_type], sizeof(double) * nx[1]));
                gpu_error_check(cudaMalloc((void**)&d_yh[m][monomer_type], sizeof(double) * nx[1]));

                gpu_error_check(cudaMalloc((void**)&d_zl[m][monomer_type], sizeof(double) * nx[2]));
                gpu_error_check(cudaMalloc((void**)&d_zd[m][monomer_type], sizeof(double) * nx[2]));
                gpu_error_check(cudaMalloc((void**)&d_zh[m][monomer_type], sizeof(double) * nx[2]));
            }
        }

        // Allocate w field storage
        for(const auto& item: molecules->get_bond_lengths())
        {
            std::string monomer_type = item.first;
            gpu_error_check(cudaMalloc((void**)&d_w_field[monomer_type], sizeof(double) * n_grid));
        }

        // Allocate per-stream workspace
        for(int s = 0; s < n_streams; s++)
        {
            d_X[s].resize(M);
            d_F[s].resize(M);
            d_X_old[s].resize(M);
            d_F_old[s].resize(M);

            for(int m = 0; m < M; m++)
            {
                gpu_error_check(cudaMalloc((void**)&d_X[s][m], sizeof(double) * n_grid));
                gpu_error_check(cudaMalloc((void**)&d_F[s][m], sizeof(double) * n_grid));
                gpu_error_check(cudaMalloc((void**)&d_X_old[s][m], sizeof(double) * n_grid));
                gpu_error_check(cudaMalloc((void**)&d_F_old[s][m], sizeof(double) * n_grid));
            }

            gpu_error_check(cudaMalloc((void**)&d_temp[s], sizeof(double) * n_grid));
            gpu_error_check(cudaMalloc((void**)&d_rhs[s], sizeof(double) * n_grid));
            gpu_error_check(cudaMalloc((void**)&d_q_star[s], sizeof(double) * n_grid));
            gpu_error_check(cudaMalloc((void**)&d_q_dstar[s], sizeof(double) * n_grid));
            gpu_error_check(cudaMalloc((void**)&d_c_star[s], sizeof(double) * n_grid));
            gpu_error_check(cudaMalloc((void**)&d_q_sparse[s], sizeof(double) * n_grid));
            gpu_error_check(cudaMalloc((void**)&d_q_in_saved[s], sizeof(double) * n_grid));
        }

        // Allocate offset arrays for tridiagonal solver
        // Use std::vector instead of VLAs to avoid stack overflow and undefined behavior
        if(DIM == 3)
        {
            std::vector<int> offset_xy(nx[0] * nx[1]);
            std::vector<int> offset_yz(nx[1] * nx[2]);
            std::vector<int> offset_xz(nx[0] * nx[2]);
            int count;

            count = 0;
            for(int i = 0; i < nx[0]; i++)
                for(int j = 0; j < nx[1]; j++)
                    offset_xy[count++] = i * nx[1] * nx[2] + j * nx[2];

            count = 0;
            for(int j = 0; j < nx[1]; j++)
                for(int k = 0; k < nx[2]; k++)
                    offset_yz[count++] = j * nx[2] + k;

            count = 0;
            for(int i = 0; i < nx[0]; i++)
                for(int k = 0; k < nx[2]; k++)
                    offset_xz[count++] = i * nx[1] * nx[2] + k;

            gpu_error_check(cudaMalloc((void**)&d_offset_xy, sizeof(int) * nx[0] * nx[1]));
            gpu_error_check(cudaMalloc((void**)&d_offset_yz, sizeof(int) * nx[1] * nx[2]));
            gpu_error_check(cudaMalloc((void**)&d_offset_xz, sizeof(int) * nx[0] * nx[2]));

            gpu_error_check(cudaMemcpy(d_offset_xy, offset_xy.data(), sizeof(int) * nx[0] * nx[1], cudaMemcpyHostToDevice));
            gpu_error_check(cudaMemcpy(d_offset_yz, offset_yz.data(), sizeof(int) * nx[1] * nx[2], cudaMemcpyHostToDevice));
            gpu_error_check(cudaMemcpy(d_offset_xz, offset_xz.data(), sizeof(int) * nx[0] * nx[2], cudaMemcpyHostToDevice));
        }
        else if(DIM == 2)
        {
            std::vector<int> offset_x(nx[0]);
            std::vector<int> offset_y(nx[1]);

            for(int i = 0; i < nx[0]; i++)
                offset_x[i] = i * nx[1];
            for(int j = 0; j < nx[1]; j++)
                offset_y[j] = j;

            gpu_error_check(cudaMalloc((void**)&d_offset_x, sizeof(int) * nx[0]));
            gpu_error_check(cudaMalloc((void**)&d_offset_y, sizeof(int) * nx[1]));

            gpu_error_check(cudaMemcpy(d_offset_x, offset_x.data(), sizeof(int) * nx[0], cudaMemcpyHostToDevice));
            gpu_error_check(cudaMemcpy(d_offset_y, offset_y.data(), sizeof(int) * nx[1], cudaMemcpyHostToDevice));
        }
        else if(DIM == 1)
        {
            std::vector<int> offset(1, 0);
            gpu_error_check(cudaMalloc((void**)&d_offset, sizeof(int)));
            gpu_error_check(cudaMemcpy(d_offset, offset.data(), sizeof(int), cudaMemcpyHostToDevice));
        }

        // Initialize sparse matrices and PCG workspace for 2D/3D
        if(DIM >= 2)
        {
            // Initialize sparse matrices
            sparse_matrices.resize(M - 1);
            for(int m = 0; m < M - 1; m++)
            {
                for(const auto& item: molecules->get_bond_lengths())
                {
                    std::string monomer_type = item.first;
                    CudaSparseMatrixCSR& mat = sparse_matrices[m][monomer_type];
                    mat.d_row_ptr = nullptr;
                    mat.d_col_idx = nullptr;
                    mat.d_values = nullptr;
                    mat.d_diag_inv = nullptr;
                    mat.n = n_grid;
                    mat.nnz = 0;
                    mat.built = false;
                }
            }

            // Initialize matrix-free diagonal inverse storage
            d_diag_inv_free.resize(M - 1);
            diag_inv_built.resize(M - 1);
            for(int m = 0; m < M - 1; m++)
            {
                for(const auto& item: molecules->get_bond_lengths())
                {
                    std::string monomer_type = item.first;
                    d_diag_inv_free[m][monomer_type] = nullptr;
                    diag_inv_built[m][monomer_type] = false;
                }
            }

            // Calculate number of blocks for PCG
            pcg_n_blocks = (n_grid + PCG_BLOCK_SIZE - 1) / PCG_BLOCK_SIZE;

            // Allocate PCG workspace for each stream
            for(int s = 0; s < n_streams; s++)
            {
                gpu_error_check(cudaMalloc((void**)&d_pcg_r[s], sizeof(double) * n_grid));
                gpu_error_check(cudaMalloc((void**)&d_pcg_z[s], sizeof(double) * n_grid));
                gpu_error_check(cudaMalloc((void**)&d_pcg_p[s], sizeof(double) * n_grid));
                gpu_error_check(cudaMalloc((void**)&d_pcg_Ap[s], sizeof(double) * n_grid));
                // Device-side scalars: [alpha, beta, rz_old, rz_new, pAp, r_norm_sq]
                gpu_error_check(cudaMalloc((void**)&d_pcg_scalars[s], sizeof(double) * 6));
                // Partial reduction buffer for fused kernels
                gpu_error_check(cudaMalloc((void**)&d_pcg_partial[s], sizeof(double) * pcg_n_blocks));
            }

            // PCG parameters
            pcg_max_iter = 1000;
            pcg_tol = 1e-10;
        }

        // Initialize Laplacian operator
        update_laplacian_operator();
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

CudaSolverSDC::~CudaSolverSDC()
{
    // Free PCG resources (2D/3D only)
    if(dim >= 2)
    {
        for(int m = 0; m < M - 1; m++)
        {
            for(auto& item: sparse_matrices[m])
            {
                CudaSparseMatrixCSR& mat = item.second;
                if(mat.d_row_ptr) cudaFree(mat.d_row_ptr);
                if(mat.d_col_idx) cudaFree(mat.d_col_idx);
                if(mat.d_values) cudaFree(mat.d_values);
                if(mat.d_diag_inv) cudaFree(mat.d_diag_inv);
            }
            // Free matrix-free diagonal inverse storage
            for(auto& item: d_diag_inv_free[m])
            {
                if(item.second) cudaFree(item.second);
            }
        }

        for(int s = 0; s < n_streams; s++)
        {
            cudaFree(d_pcg_r[s]);
            cudaFree(d_pcg_z[s]);
            cudaFree(d_pcg_p[s]);
            cudaFree(d_pcg_Ap[s]);
            cudaFree(d_pcg_scalars[s]);
            cudaFree(d_pcg_partial[s]);
        }
    }

    // Free integration matrix
    cudaFree(d_S);

    // Free tridiagonal coefficients
    for(int m = 0; m < M - 1; m++)
    {
        for(const auto& item: d_xl[m])
            cudaFree(item.second);
        for(const auto& item: d_xd[m])
            cudaFree(item.second);
        for(const auto& item: d_xd_base[m])
            cudaFree(item.second);
        for(const auto& item: d_xh[m])
            cudaFree(item.second);
        for(const auto& item: d_yl[m])
            cudaFree(item.second);
        for(const auto& item: d_yd[m])
            cudaFree(item.second);
        for(const auto& item: d_yh[m])
            cudaFree(item.second);
        for(const auto& item: d_zl[m])
            cudaFree(item.second);
        for(const auto& item: d_zd[m])
            cudaFree(item.second);
        for(const auto& item: d_zh[m])
            cudaFree(item.second);
    }

    // Free w field storage
    for(const auto& item: d_w_field)
        cudaFree(item.second);

    // Free per-stream workspace
    for(int s = 0; s < n_streams; s++)
    {
        for(int m = 0; m < M; m++)
        {
            cudaFree(d_X[s][m]);
            cudaFree(d_F[s][m]);
            cudaFree(d_X_old[s][m]);
            cudaFree(d_F_old[s][m]);
        }
        cudaFree(d_temp[s]);
        cudaFree(d_rhs[s]);
        cudaFree(d_q_star[s]);
        cudaFree(d_q_dstar[s]);
        cudaFree(d_c_star[s]);
        cudaFree(d_q_sparse[s]);
        cudaFree(d_q_in_saved[s]);
    }

    // Free offset arrays
    if(dim == 3)
    {
        cudaFree(d_offset_xy);
        cudaFree(d_offset_yz);
        cudaFree(d_offset_xz);
    }
    else if(dim == 2)
    {
        cudaFree(d_offset_x);
        cudaFree(d_offset_y);
    }
    else if(dim == 1)
    {
        cudaFree(d_offset);
    }
}

void CudaSolverSDC::compute_gauss_lobatto_nodes()
{
    tau.resize(M);

    if(M == 2)
    {
        tau[0] = 0.0;
        tau[1] = 1.0;
    }
    else if(M == 3)
    {
        tau[0] = 0.0;
        tau[1] = 0.5;
        tau[2] = 1.0;
    }
    else if(M == 4)
    {
        tau[0] = 0.0;
        tau[1] = 0.5 - std::sqrt(5.0) / 10.0;
        tau[2] = 0.5 + std::sqrt(5.0) / 10.0;
        tau[3] = 1.0;
    }
    else if(M == 5)
    {
        tau[0] = 0.0;
        tau[1] = 0.5 - std::sqrt(21.0) / 14.0;
        tau[2] = 0.5;
        tau[3] = 0.5 + std::sqrt(21.0) / 14.0;
        tau[4] = 1.0;
    }
    else if(M == 6)
    {
        // Exact Gauss-Lobatto nodes for M=6
        double x1 = std::sqrt((7.0 + 2.0 * std::sqrt(7.0)) / 21.0);
        double x2 = std::sqrt((7.0 - 2.0 * std::sqrt(7.0)) / 21.0);
        tau[0] = 0.0;
        tau[1] = 0.5 * (1.0 - x1);
        tau[2] = 0.5 * (1.0 - x2);
        tau[3] = 0.5 * (1.0 + x2);
        tau[4] = 0.5 * (1.0 + x1);
        tau[5] = 1.0;
    }
    else if(M == 7)
    {
        // Exact Gauss-Lobatto nodes for M=7
        double x1 = std::sqrt((15.0 + 2.0 * std::sqrt(15.0)) / 33.0);
        double x2 = std::sqrt((15.0 - 2.0 * std::sqrt(15.0)) / 33.0);
        tau[0] = 0.0;
        tau[1] = 0.5 * (1.0 - x1);
        tau[2] = 0.5 * (1.0 - x2);
        tau[3] = 0.5;
        tau[4] = 0.5 * (1.0 + x2);
        tau[5] = 0.5 * (1.0 + x1);
        tau[6] = 1.0;
    }
    else
    {
        // General formula using Chebyshev nodes of second kind
        for(int j = 0; j < M; j++)
            tau[j] = 0.5 * (1.0 - std::cos(M_PI * j / (M - 1)));
    }
}

void CudaSolverSDC::compute_integration_matrix()
{
    S.resize(M - 1);
    for(int m = 0; m < M - 1; m++)
        S[m].resize(M);

    const int n_gauss = 16;
    const double nodes_16[] = {
        -0.9894009349916499, -0.9445750230732326, -0.8656312023878318, -0.7554044083550030,
        -0.6178762444026437, -0.4580167776572274, -0.2816035507792589, -0.0950125098376374,
         0.0950125098376374,  0.2816035507792589,  0.4580167776572274,  0.6178762444026437,
         0.7554044083550030,  0.8656312023878318,  0.9445750230732326,  0.9894009349916499
    };
    const double weights_16[] = {
        0.0271524594117541, 0.0622535239386479, 0.0951585116824928, 0.1246289712555339,
        0.1495959888165767, 0.1691565193950025, 0.1826034150449236, 0.1894506104550685,
        0.1894506104550685, 0.1826034150449236, 0.1691565193950025, 0.1495959888165767,
        0.1246289712555339, 0.0951585116824928, 0.0622535239386479, 0.0271524594117541
    };

    for(int m = 0; m < M - 1; m++)
    {
        double a = tau[m];
        double b = tau[m + 1];
        double h = b - a;

        for(int j = 0; j < M; j++)
        {
            double integral = 0.0;

            for(int k = 0; k < n_gauss; k++)
            {
                double t = a + 0.5 * h * (nodes_16[k] + 1.0);

                double L_j = 1.0;
                for(int i = 0; i < M; i++)
                {
                    if(i != j)
                        L_j *= (t - tau[i]) / (tau[j] - tau[i]);
                }

                integral += weights_16[k] * L_j;
            }

            S[m][j] = 0.5 * h * integral;
        }
    }
}

void CudaSolverSDC::update_laplacian_operator()
{
    try
    {
        const double ds = this->molecules->get_ds();
        const std::vector<int> nx = this->cb->get_nx();

        // Get dimensions safely - use 1 for unused dimensions
        int nx_x = nx[0];
        int nx_y = (dim >= 2) ? nx[1] : 1;
        int nx_z = (dim == 3) ? nx[2] : 1;

        // Store sub-interval time steps
        for(int m = 0; m < M - 1; m++)
        {
            dtau_sub[m] = (tau[m + 1] - tau[m]) * ds;
        }

        for(const auto& item: this->molecules->get_bond_lengths())
        {
            std::string monomer_type = item.first;
            double bond_length_sq = item.second * item.second;

            for(int m = 0; m < M - 1; m++)
            {
                double dtau = dtau_sub[m];

                // Compute coefficients on host - use std::vector to avoid VLA issues
                std::vector<double> h_xl(nx_x), h_xd(nx_x), h_xh(nx_x);
                std::vector<double> h_yl(nx_y), h_yd(nx_y), h_yh(nx_y);
                std::vector<double> h_zl(nx_z), h_zd(nx_z), h_zh(nx_z);

                // Use Backward Euler matrix (not Crank-Nicolson) for SDC
                FiniteDifference::get_backward_euler_matrix(
                    this->cb->get_boundary_conditions(),
                    this->cb->get_nx(), this->cb->get_dx(),
                    h_xl.data(), h_xd.data(), h_xh.data(),
                    h_yl.data(), h_yd.data(), h_yh.data(),
                    h_zl.data(), h_zd.data(), h_zh.data(),
                    bond_length_sq, dtau);

                // Copy to device
                gpu_error_check(cudaMemcpy(d_xl[m][monomer_type], h_xl.data(), sizeof(double) * nx_x, cudaMemcpyHostToDevice));
                gpu_error_check(cudaMemcpy(d_xd[m][monomer_type], h_xd.data(), sizeof(double) * nx_x, cudaMemcpyHostToDevice));
                gpu_error_check(cudaMemcpy(d_xh[m][monomer_type], h_xh.data(), sizeof(double) * nx_x, cudaMemcpyHostToDevice));

                // Store base diagonal (diffusion only) for 1D fully implicit update
                if(dim == 1)
                {
                    gpu_error_check(cudaMemcpy(d_xd_base[m][monomer_type], h_xd.data(), sizeof(double) * nx_x, cudaMemcpyHostToDevice));
                }

                if(dim >= 2)
                {
                    gpu_error_check(cudaMemcpy(d_yl[m][monomer_type], h_yl.data(), sizeof(double) * nx_y, cudaMemcpyHostToDevice));
                    gpu_error_check(cudaMemcpy(d_yd[m][monomer_type], h_yd.data(), sizeof(double) * nx_y, cudaMemcpyHostToDevice));
                    gpu_error_check(cudaMemcpy(d_yh[m][monomer_type], h_yh.data(), sizeof(double) * nx_y, cudaMemcpyHostToDevice));
                }

                if(dim == 3)
                {
                    gpu_error_check(cudaMemcpy(d_zl[m][monomer_type], h_zl.data(), sizeof(double) * nx_z, cudaMemcpyHostToDevice));
                    gpu_error_check(cudaMemcpy(d_zd[m][monomer_type], h_zd.data(), sizeof(double) * nx_z, cudaMemcpyHostToDevice));
                    gpu_error_check(cudaMemcpy(d_zh[m][monomer_type], h_zh.data(), sizeof(double) * nx_z, cudaMemcpyHostToDevice));
                }
            }
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

void CudaSolverSDC::update_dw(std::string device, std::map<std::string, const double*> w_input)
{
    const int n_grid = this->cb->get_total_grid();
    const std::vector<int> nx = this->cb->get_nx();

    for(const auto& item: w_input)
    {
        const std::string& monomer_type = item.first;
        const double* w = item.second;

        // Copy w to device
        if(device == "host") {
            gpu_error_check(cudaMemcpy(d_w_field[monomer_type], w, sizeof(double) * n_grid, cudaMemcpyHostToDevice));
        } else {
            gpu_error_check(cudaMemcpy(d_w_field[monomer_type], w, sizeof(double) * n_grid, cudaMemcpyDeviceToDevice));
        }

        // For 1D: Add dtau*w to the tridiagonal diagonal
        // This creates the fully implicit matrix: (I - dtau*D∇² + dtau*w)
        if(dim == 1)
        {
            // Get w on host for updating diagonal
            std::vector<double> h_w(n_grid);
            if(device == "host")
            {
                for(int i = 0; i < n_grid; i++)
                    h_w[i] = w[i];
            }
            else
            {
                gpu_error_check(cudaMemcpy(h_w.data(), w, sizeof(double) * n_grid, cudaMemcpyDeviceToHost));
            }

            // Get base diagonal on host
            std::vector<double> h_xd_base(nx[0]);
            std::vector<double> h_xd(nx[0]);

            for(int m = 0; m < M - 1; m++)
            {
                double dtau = dtau_sub[m];

                // Get base diagonal
                gpu_error_check(cudaMemcpy(h_xd_base.data(), d_xd_base[m][monomer_type],
                    sizeof(double) * nx[0], cudaMemcpyDeviceToHost));

                // Add dtau*w to diagonal: xd = xd_base + dtau * w
                for(int i = 0; i < nx[0]; i++)
                {
                    h_xd[i] = h_xd_base[i] + dtau * h_w[i];
                }

                // Copy updated diagonal to device
                gpu_error_check(cudaMemcpy(d_xd[m][monomer_type], h_xd.data(),
                    sizeof(double) * nx[0], cudaMemcpyHostToDevice));
            }
        }
        // For 2D/3D: Mark sparse matrices and diagonal inverse as needing rebuild
        // The w contribution will be included when the matrix is rebuilt
        else
        {
            for(int m = 0; m < M - 1; m++)
            {
                sparse_matrices[m][monomer_type].built = false;
                diag_inv_built[m][monomer_type] = false;
            }
        }
    }
}

void CudaSolverSDC::compute_F_device(int STREAM, const double* d_q, const double* d_w,
                                      double* d_F, std::string monomer_type)
{
    const int n_grid = this->cb->get_total_grid();
    const std::vector<int> nx = this->cb->get_nx();
    const std::vector<double> dx = this->cb->get_dx();
    const std::vector<BoundaryCondition> bc = this->cb->get_boundary_conditions();

    double bond_length = this->molecules->get_bond_lengths().at(monomer_type);
    double bond_length_sq = bond_length * bond_length;
    const double D = bond_length_sq / 6.0;

    const int N_BLOCKS = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();
    cudaStream_t stream = streams[STREAM][0];

    if(dim == 3)
    {
        double alpha_x = D / (dx[0] * dx[0]);
        double alpha_y = D / (dx[1] * dx[1]);
        double alpha_z = D / (dx[2] * dx[2]);

        compute_F_kernel_3d<<<N_BLOCKS, N_THREADS, 0, stream>>>(
            d_q, d_w, d_F,
            alpha_x, alpha_y, alpha_z,
            nx[0], nx[1], nx[2],
            bc[0], bc[1], bc[2], bc[3], bc[4], bc[5],
            n_grid);
    }
    else if(dim == 2)
    {
        double alpha_x = D / (dx[0] * dx[0]);
        double alpha_y = D / (dx[1] * dx[1]);

        compute_F_kernel_2d<<<N_BLOCKS, N_THREADS, 0, stream>>>(
            d_q, d_w, d_F,
            alpha_x, alpha_y,
            nx[0], nx[1],
            bc[0], bc[1], bc[2], bc[3],
            n_grid);
    }
    else
    {
        double alpha_x = D / (dx[0] * dx[0]);

        compute_F_kernel_1d<<<N_BLOCKS, N_THREADS, 0, stream>>>(
            d_q, d_w, d_F,
            alpha_x, nx[0],
            bc[0], bc[1],
            n_grid);
    }
}

void CudaSolverSDC::implicit_solve_step(int STREAM, int sub_interval,
                                        double* d_q_in, double* d_q_out, std::string monomer_type)
{
    std::vector<BoundaryCondition> bc = this->cb->get_boundary_conditions();

    if(dim >= 2)
    {
        // Use PCG sparse solver for 2D/3D (no splitting error)
        pcg_solve_step(STREAM, sub_interval, d_q_in, d_q_out, monomer_type);
    }
    else // dim == 1
    {
        // Use tridiagonal solver for 1D (exact)
        tridiagonal_solve_1d(STREAM, sub_interval, bc, d_q_in, d_q_out, monomer_type);
    }
}

void CudaSolverSDC::tridiagonal_solve_1d(int STREAM, int sub_interval,
    std::vector<BoundaryCondition> bc,
    double* d_q_in, double* d_q_out, std::string monomer_type)
{
    const int n_grid = this->cb->get_total_grid();
    const std::vector<int> nx = this->cb->get_nx();

    double *_d_xl = d_xl[sub_interval][monomer_type];
    double *_d_xd = d_xd[sub_interval][monomer_type];
    double *_d_xh = d_xh[sub_interval][monomer_type];

    // For Backward Euler: solve A * q_out = q_in directly
    // No CN transformation of RHS - just copy input to q_star
    gpu_error_check(cudaMemcpyAsync(d_q_star[STREAM], d_q_in, sizeof(double) * n_grid,
                                    cudaMemcpyDeviceToDevice, streams[STREAM][0]));

    if(bc[0] == BoundaryCondition::PERIODIC)
        tridiagonal_periodic<<<1, nx[0], sizeof(double) * 3 * nx[0], streams[STREAM][0]>>>(
            _d_xl, _d_xd, _d_xh, d_c_star[STREAM], d_q_sparse[STREAM],
            d_q_star[STREAM], d_q_out, d_offset, 1, 1, nx[0]);
    else
        tridiagonal<<<1, nx[0], sizeof(double) * 2 * nx[0], streams[STREAM][0]>>>(
            _d_xl, _d_xd, _d_xh, d_c_star[STREAM],
            d_q_star[STREAM], d_q_out, d_offset, 1, 1, nx[0]);
}

// Helper functions for max/min
static int host_max_of_two(int x, int y) { return (x > y) ? x : y; }
static int host_min_of_two(int x, int y) { return (x < y) ? x : y; }

void CudaSolverSDC::build_sparse_matrix(int sub_interval, std::string monomer_type)
{
    try
    {
        const int n_grid = this->cb->get_total_grid();
        const std::vector<int> nx = this->cb->get_nx();
        const std::vector<double> dx = this->cb->get_dx();
        const std::vector<BoundaryCondition> bc = this->cb->get_boundary_conditions();
        const double ds = this->molecules->get_ds();

        double bond_length = this->molecules->get_bond_lengths().at(monomer_type);
        double bond_length_sq = bond_length * bond_length;
        const double D = bond_length_sq / 6.0;
        double dtau = (tau[sub_interval + 1] - tau[sub_interval]) * ds;

        // Get w field on host for fully implicit scheme
        std::vector<double> h_w(n_grid);
        gpu_error_check(cudaMemcpy(h_w.data(), d_w_field.at(monomer_type),
            sizeof(double) * n_grid, cudaMemcpyDeviceToHost));

        CudaSparseMatrixCSR& mat = sparse_matrices[sub_interval][monomer_type];

        // Build A = I - dtau * D * ∇² + dtau * w in CSR format (0-based indexing for PCG)
        // Fully implicit scheme: includes both diffusion and reaction terms
        int stencil_size = (dim == 2) ? 5 : 7;
        int max_nnz = n_grid * stencil_size;

        std::vector<int> h_row_ptr(n_grid + 1);
        std::vector<int> h_col_idx;
        std::vector<double> h_values;
        std::vector<double> h_diag_inv(n_grid);  // For Jacobi preconditioner
        h_col_idx.reserve(max_nnz);
        h_values.reserve(max_nnz);

        if(dim == 2)
        {
            double rx = D * dtau / (dx[0] * dx[0]);
            double ry = D * dtau / (dx[1] * dx[1]);

            int nnz_count = 0;
            for(int i = 0; i < nx[0]; i++)
            {
                int im = (bc[0] == BoundaryCondition::PERIODIC) ? (nx[0] + i - 1) % nx[0] : host_max_of_two(0, i - 1);
                int ip = (bc[1] == BoundaryCondition::PERIODIC) ? (i + 1) % nx[0] : host_min_of_two(nx[0] - 1, i + 1);

                for(int j = 0; j < nx[1]; j++)
                {
                    int jm = (bc[2] == BoundaryCondition::PERIODIC) ? (nx[1] + j - 1) % nx[1] : host_max_of_two(0, j - 1);
                    int jp = (bc[3] == BoundaryCondition::PERIODIC) ? (j + 1) % nx[1] : host_min_of_two(nx[1] - 1, j + 1);

                    int row = i * nx[1] + j;
                    h_row_ptr[row] = nnz_count;

                    // Store entries in column order
                    std::vector<std::pair<int, double>> entries;

                    // Left neighbor - skip if absorbing BC at left boundary
                    if(!(bc[0] == BoundaryCondition::ABSORBING && i == 0))
                    {
                        int col_im = im * nx[1] + j;
                        if(col_im != row)
                            entries.push_back({col_im, -rx});
                    }

                    // Bottom neighbor - skip if absorbing BC at bottom boundary
                    if(!(bc[2] == BoundaryCondition::ABSORBING && j == 0))
                    {
                        int col_jm = i * nx[1] + jm;
                        if(col_jm != row)
                            entries.push_back({col_jm, -ry});
                    }

                    // Fully implicit: includes diffusion and reaction terms
                    double diag_val = 1.0 + 2.0 * rx + 2.0 * ry + dtau * h_w[row];
                    // Cell-centered boundary modifications using ghost cells:
                    // - Reflecting: symmetric ghost (q_{-1} = q_0) → diagonal decreases
                    // - Absorbing: antisymmetric ghost (q_{-1} = -q_0) → diagonal increases
                    if(bc[0] == BoundaryCondition::REFLECTING && i == 0)
                        diag_val -= rx;
                    else if(bc[0] == BoundaryCondition::ABSORBING && i == 0)
                        diag_val += rx;
                    if(bc[1] == BoundaryCondition::REFLECTING && i == nx[0] - 1)
                        diag_val -= rx;
                    else if(bc[1] == BoundaryCondition::ABSORBING && i == nx[0] - 1)
                        diag_val += rx;
                    if(bc[2] == BoundaryCondition::REFLECTING && j == 0)
                        diag_val -= ry;
                    else if(bc[2] == BoundaryCondition::ABSORBING && j == 0)
                        diag_val += ry;
                    if(bc[3] == BoundaryCondition::REFLECTING && j == nx[1] - 1)
                        diag_val -= ry;
                    else if(bc[3] == BoundaryCondition::ABSORBING && j == nx[1] - 1)
                        diag_val += ry;
                    entries.push_back({row, diag_val});

                    // Store inverse diagonal for Jacobi preconditioner
                    h_diag_inv[row] = 1.0 / diag_val;

                    // Top neighbor - skip if absorbing BC at top boundary
                    if(!(bc[3] == BoundaryCondition::ABSORBING && j == nx[1] - 1))
                    {
                        int col_jp = i * nx[1] + jp;
                        if(col_jp != row)
                            entries.push_back({col_jp, -ry});
                    }

                    // Right neighbor - skip if absorbing BC at right boundary
                    if(!(bc[1] == BoundaryCondition::ABSORBING && i == nx[0] - 1))
                    {
                        int col_ip = ip * nx[1] + j;
                        if(col_ip != row)
                            entries.push_back({col_ip, -rx});
                    }

                    std::sort(entries.begin(), entries.end());

                    for(const auto& e: entries)
                    {
                        h_col_idx.push_back(e.first);
                        h_values.push_back(e.second);
                        nnz_count++;
                    }
                }
            }
            h_row_ptr[n_grid] = nnz_count;
            mat.nnz = nnz_count;
        }
        else // dim == 3
        {
            double rx = D * dtau / (dx[0] * dx[0]);
            double ry = D * dtau / (dx[1] * dx[1]);
            double rz = D * dtau / (dx[2] * dx[2]);

            int nnz_count = 0;
            for(int i = 0; i < nx[0]; i++)
            {
                int im = (bc[0] == BoundaryCondition::PERIODIC) ? (nx[0] + i - 1) % nx[0] : host_max_of_two(0, i - 1);
                int ip = (bc[1] == BoundaryCondition::PERIODIC) ? (i + 1) % nx[0] : host_min_of_two(nx[0] - 1, i + 1);

                for(int j = 0; j < nx[1]; j++)
                {
                    int jm = (bc[2] == BoundaryCondition::PERIODIC) ? (nx[1] + j - 1) % nx[1] : host_max_of_two(0, j - 1);
                    int jp = (bc[3] == BoundaryCondition::PERIODIC) ? (j + 1) % nx[1] : host_min_of_two(nx[1] - 1, j + 1);

                    for(int k = 0; k < nx[2]; k++)
                    {
                        int km = (bc[4] == BoundaryCondition::PERIODIC) ? (nx[2] + k - 1) % nx[2] : host_max_of_two(0, k - 1);
                        int kp = (bc[5] == BoundaryCondition::PERIODIC) ? (k + 1) % nx[2] : host_min_of_two(nx[2] - 1, k + 1);

                        int row = i * nx[1] * nx[2] + j * nx[2] + k;
                        h_row_ptr[row] = nnz_count;

                        std::vector<std::pair<int, double>> entries;

                        // Left neighbor - skip if absorbing BC at left boundary
                        if(!(bc[0] == BoundaryCondition::ABSORBING && i == 0))
                        {
                            int col_im = im * nx[1] * nx[2] + j * nx[2] + k;
                            if(col_im != row)
                                entries.push_back({col_im, -rx});
                        }

                        // Back neighbor - skip if absorbing BC at back boundary
                        if(!(bc[2] == BoundaryCondition::ABSORBING && j == 0))
                        {
                            int col_jm = i * nx[1] * nx[2] + jm * nx[2] + k;
                            if(col_jm != row)
                                entries.push_back({col_jm, -ry});
                        }

                        // Bottom neighbor - skip if absorbing BC at bottom boundary
                        if(!(bc[4] == BoundaryCondition::ABSORBING && k == 0))
                        {
                            int col_km = i * nx[1] * nx[2] + j * nx[2] + km;
                            if(col_km != row)
                                entries.push_back({col_km, -rz});
                        }

                        // Fully implicit: includes diffusion and reaction terms
                        double diag_val = 1.0 + 2.0 * rx + 2.0 * ry + 2.0 * rz + dtau * h_w[row];
                        // Cell-centered boundary modifications using ghost cells:
                        // - Reflecting: symmetric ghost (q_{-1} = q_0) → diagonal decreases
                        // - Absorbing: antisymmetric ghost (q_{-1} = -q_0) → diagonal increases
                        if(bc[0] == BoundaryCondition::REFLECTING && i == 0)
                            diag_val -= rx;
                        else if(bc[0] == BoundaryCondition::ABSORBING && i == 0)
                            diag_val += rx;
                        if(bc[1] == BoundaryCondition::REFLECTING && i == nx[0] - 1)
                            diag_val -= rx;
                        else if(bc[1] == BoundaryCondition::ABSORBING && i == nx[0] - 1)
                            diag_val += rx;
                        if(bc[2] == BoundaryCondition::REFLECTING && j == 0)
                            diag_val -= ry;
                        else if(bc[2] == BoundaryCondition::ABSORBING && j == 0)
                            diag_val += ry;
                        if(bc[3] == BoundaryCondition::REFLECTING && j == nx[1] - 1)
                            diag_val -= ry;
                        else if(bc[3] == BoundaryCondition::ABSORBING && j == nx[1] - 1)
                            diag_val += ry;
                        if(bc[4] == BoundaryCondition::REFLECTING && k == 0)
                            diag_val -= rz;
                        else if(bc[4] == BoundaryCondition::ABSORBING && k == 0)
                            diag_val += rz;
                        if(bc[5] == BoundaryCondition::REFLECTING && k == nx[2] - 1)
                            diag_val -= rz;
                        else if(bc[5] == BoundaryCondition::ABSORBING && k == nx[2] - 1)
                            diag_val += rz;
                        entries.push_back({row, diag_val});

                        // Store inverse diagonal for Jacobi preconditioner
                        h_diag_inv[row] = 1.0 / diag_val;

                        // Top neighbor - skip if absorbing BC at top boundary
                        if(!(bc[5] == BoundaryCondition::ABSORBING && k == nx[2] - 1))
                        {
                            int col_kp = i * nx[1] * nx[2] + j * nx[2] + kp;
                            if(col_kp != row)
                                entries.push_back({col_kp, -rz});
                        }

                        // Front neighbor - skip if absorbing BC at front boundary
                        if(!(bc[3] == BoundaryCondition::ABSORBING && j == nx[1] - 1))
                        {
                            int col_jp = i * nx[1] * nx[2] + jp * nx[2] + k;
                            if(col_jp != row)
                                entries.push_back({col_jp, -ry});
                        }

                        // Right neighbor - skip if absorbing BC at right boundary
                        if(!(bc[1] == BoundaryCondition::ABSORBING && i == nx[0] - 1))
                        {
                            int col_ip = ip * nx[1] * nx[2] + j * nx[2] + k;
                            if(col_ip != row)
                                entries.push_back({col_ip, -rx});
                        }

                        std::sort(entries.begin(), entries.end());

                        for(const auto& e: entries)
                        {
                            h_col_idx.push_back(e.first);
                            h_values.push_back(e.second);
                            nnz_count++;
                        }
                    }
                }
            }
            h_row_ptr[n_grid] = nnz_count;
            mat.nnz = nnz_count;
        }

        // Allocate and copy to device
        gpu_error_check(cudaMalloc((void**)&mat.d_row_ptr, sizeof(int) * (n_grid + 1)));
        gpu_error_check(cudaMalloc((void**)&mat.d_col_idx, sizeof(int) * mat.nnz));
        gpu_error_check(cudaMalloc((void**)&mat.d_values, sizeof(double) * mat.nnz));
        gpu_error_check(cudaMalloc((void**)&mat.d_diag_inv, sizeof(double) * n_grid));

        gpu_error_check(cudaMemcpy(mat.d_row_ptr, h_row_ptr.data(), sizeof(int) * (n_grid + 1), cudaMemcpyHostToDevice));
        gpu_error_check(cudaMemcpy(mat.d_col_idx, h_col_idx.data(), sizeof(int) * mat.nnz, cudaMemcpyHostToDevice));
        gpu_error_check(cudaMemcpy(mat.d_values, h_values.data(), sizeof(double) * mat.nnz, cudaMemcpyHostToDevice));
        gpu_error_check(cudaMemcpy(mat.d_diag_inv, h_diag_inv.data(), sizeof(double) * n_grid, cudaMemcpyHostToDevice));

        mat.built = true;
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

void CudaSolverSDC::sparse_matvec(const CudaSparseMatrixCSR& mat, const double* d_x,
                                   double* d_y, cudaStream_t stream)
{
    int n_blocks = (mat.n + 255) / 256;
    sparse_matvec_kernel<<<n_blocks, 256, 0, stream>>>(
        mat.d_row_ptr, mat.d_col_idx, mat.d_values, d_x, d_y, mat.n);
}

void CudaSolverSDC::sparse_solve(int STREAM, int sub_interval,
                                  double* d_q_in, double* d_q_out, std::string monomer_type)
{
    try
    {
        CudaSparseMatrixCSR& mat = sparse_matrices[sub_interval][monomer_type];

        // Build matrix if not yet done
        if(!mat.built)
        {
            build_sparse_matrix(sub_interval, monomer_type);
        }

        const int n = mat.n;
        cudaStream_t stream = streams[STREAM][0];

        // OPTIMIZED PCG with:
        // 1. Warm-start using RHS as initial guess (matrix is close to identity)
        // 2. Fused kernels (dot product + scalar computation in single launch)
        // 3. Device-side scalars (avoid memory transfers)
        //
        // Scalar layout: [0]=alpha, [1]=beta, [2]=rz_old, [3]=rz_new, [4]=pAp, [5]=r_norm_sq
        double* d_scalars = d_pcg_scalars[STREAM];
        double* d_partial = d_pcg_partial[STREAM];

        // Warm-start: use RHS as initial guess (x0 = b)
        // Since A ≈ I for small dtau, this gives x0 ≈ solution
        // Provides ~28% speedup for 64^3 grids compared to cold-start (x0 = 0)
        gpu_error_check(cudaMemcpyAsync(d_q_out, d_q_in, sizeof(double) * n,
            cudaMemcpyDeviceToDevice, stream));

        // Compute Ax0 into d_pcg_Ap (reuse as temp buffer)
        sparse_matvec(mat, d_q_out, d_pcg_Ap[STREAM], stream);

        // Initialize: r = b - Ax0, z = M^{-1}*r, p = z
        pcg_init_warmstart_kernel<<<pcg_n_blocks, PCG_BLOCK_SIZE, 0, stream>>>(
            d_pcg_r[STREAM], d_pcg_z[STREAM], d_pcg_p[STREAM],
            d_q_in, d_pcg_Ap[STREAM], mat.d_diag_inv, n);

        // Compute the number of threads needed for final reduction (round up to power of 2)
        int reduce_threads = 1;
        while(reduce_threads < pcg_n_blocks) reduce_threads *= 2;
        if(reduce_threads > 1024) reduce_threads = 1024;

        // rz_old = (r, z) -> store in d_scalars[2] using fused reduction
        // Use the initialization kernel (not beta kernel) to avoid division by uninitialized memory
        pcg_fused_dot_beta_kernel<<<pcg_n_blocks, PCG_BLOCK_SIZE, PCG_BLOCK_SIZE * sizeof(double), stream>>>(
            d_pcg_r[STREAM], d_pcg_z[STREAM], d_scalars, d_partial, n, pcg_n_blocks);
        pcg_fused_reduce_rz_init_kernel<<<1, reduce_threads, reduce_threads * sizeof(double), stream>>>(
            d_scalars, d_partial, pcg_n_blocks);

        // PCG loop with convergence checking
        // Uses fused kernels to minimize memory bandwidth and kernel launch overhead
        for(int iter = 0; iter < pcg_max_iter; iter++)
        {
            // 1. Ap = A * p
            sparse_matvec(mat, d_pcg_p[STREAM], d_pcg_Ap[STREAM], stream);

            // 2-3. Fused: pAp = (p, Ap), then alpha = rz_old / pAp
            pcg_fused_dot_alpha_kernel<<<pcg_n_blocks, PCG_BLOCK_SIZE, PCG_BLOCK_SIZE * sizeof(double), stream>>>(
                d_pcg_p[STREAM], d_pcg_Ap[STREAM], d_scalars, d_partial, n, pcg_n_blocks);
            pcg_fused_reduce_alpha_kernel<<<1, reduce_threads, reduce_threads * sizeof(double), stream>>>(
                d_scalars, d_partial, pcg_n_blocks);

            // 4-5. Fused: x += alpha*p, r -= alpha*Ap, z = M^{-1}*r
            // This combines update and preconditioner in single pass to reduce memory bandwidth
            pcg_fused_update_xrz_kernel<<<pcg_n_blocks, PCG_BLOCK_SIZE, 0, stream>>>(
                d_q_out, d_pcg_r[STREAM], d_pcg_z[STREAM],
                d_pcg_p[STREAM], d_pcg_Ap[STREAM], mat.d_diag_inv, d_scalars, n);

            // 6-7. Fused: rz_new = (r, z), then beta = rz_new/rz_old, rz_old = rz_new
            pcg_fused_dot_beta_kernel<<<pcg_n_blocks, PCG_BLOCK_SIZE, PCG_BLOCK_SIZE * sizeof(double), stream>>>(
                d_pcg_r[STREAM], d_pcg_z[STREAM], d_scalars, d_partial, n, pcg_n_blocks);
            pcg_fused_reduce_beta_kernel<<<1, reduce_threads, reduce_threads * sizeof(double), stream>>>(
                d_scalars, d_partial, pcg_n_blocks);

            // Check convergence using rz_new (which approximates ||r||^2 for good preconditioner)
            // Only check periodically to reduce GPU-CPU sync overhead
            if((iter + 1) % PCG_CONV_CHECK_INTERVAL == 0 || iter == pcg_max_iter - 1)
            {
                double rz_new;
                gpu_error_check(cudaMemcpyAsync(&rz_new, &d_scalars[3],
                    sizeof(double), cudaMemcpyDeviceToHost, stream));
                gpu_error_check(cudaStreamSynchronize(stream));

                if(std::sqrt(std::abs(rz_new)) < pcg_tol)
                    break;
            }

            // 8. p = z + beta*p (reads beta from device memory)
            pcg_update_p_dev_kernel<<<pcg_n_blocks, PCG_BLOCK_SIZE, 0, stream>>>(
                d_pcg_p[STREAM], d_pcg_z[STREAM], d_scalars, n);
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

void CudaSolverSDC::matvec_free_solve(int STREAM, int sub_interval,
                                       double* d_q_in, double* d_q_out, std::string monomer_type)
{
    try
    {
        const int n_grid = this->cb->get_total_grid();
        const std::vector<int> nx = this->cb->get_nx();
        const std::vector<double> dx = this->cb->get_dx();
        const std::vector<BoundaryCondition> bc = this->cb->get_boundary_conditions();
        const double ds = this->molecules->get_ds();

        double bond_length = this->molecules->get_bond_lengths().at(monomer_type);
        double bond_length_sq = bond_length * bond_length;
        const double D = bond_length_sq / 6.0;
        double dtau = dtau_sub[sub_interval];

        cudaStream_t stream = streams[STREAM][0];

        // Compute diffusion coefficients
        double rx = D * dtau / (dx[0] * dx[0]);
        double ry = (dim >= 2) ? D * dtau / (dx[1] * dx[1]) : 0.0;
        double rz = (dim == 3) ? D * dtau / (dx[2] * dx[2]) : 0.0;

        // Get boundary conditions
        BoundaryCondition bc_xl = bc[0];
        BoundaryCondition bc_xh = bc[1];
        BoundaryCondition bc_yl = (dim >= 2) ? bc[2] : BoundaryCondition::PERIODIC;
        BoundaryCondition bc_yh = (dim >= 2) ? bc[3] : BoundaryCondition::PERIODIC;
        BoundaryCondition bc_zl = (dim == 3) ? bc[4] : BoundaryCondition::PERIODIC;
        BoundaryCondition bc_zh = (dim == 3) ? bc[5] : BoundaryCondition::PERIODIC;

        // Allocate and compute diagonal inverse if not already done
        // Note: Check for nullptr since constructor initializes entries to nullptr
        if(d_diag_inv_free[sub_interval][monomer_type] == nullptr)
        {
            gpu_error_check(cudaMalloc((void**)&d_diag_inv_free[sub_interval][monomer_type],
                sizeof(double) * n_grid));
            diag_inv_built[sub_interval][monomer_type] = false;
        }

        double* d_diag_inv = d_diag_inv_free[sub_interval][monomer_type];

        // Compute diagonal inverse (must be done when w changes)
        if(!diag_inv_built[sub_interval][monomer_type])
        {
            if(dim == 3)
            {
                compute_diag_inv_kernel_3d<<<pcg_n_blocks, PCG_BLOCK_SIZE, 0, stream>>>(
                    d_w_field[monomer_type], d_diag_inv,
                    rx, ry, rz, nx[0], nx[1], nx[2],
                    bc_xl, bc_xh, bc_yl, bc_yh, bc_zl, bc_zh,
                    dtau, n_grid);
            }
            else // dim == 2
            {
                compute_diag_inv_kernel_2d<<<pcg_n_blocks, PCG_BLOCK_SIZE, 0, stream>>>(
                    d_w_field[monomer_type], d_diag_inv,
                    rx, ry, nx[0], nx[1],
                    bc_xl, bc_xh, bc_yl, bc_yh,
                    dtau, n_grid);
            }
            diag_inv_built[sub_interval][monomer_type] = true;
        }

        // PCG solve using matrix-free matvec
        double* d_scalars = d_pcg_scalars[STREAM];
        double* d_partial = d_pcg_partial[STREAM];

        // Warm-start: use RHS as initial guess (x0 = b)
        gpu_error_check(cudaMemcpyAsync(d_q_out, d_q_in, sizeof(double) * n_grid,
            cudaMemcpyDeviceToDevice, stream));

        // Compute Ax0 using matrix-free matvec
        if(dim == 3)
        {
            matvec_free_kernel_3d<<<pcg_n_blocks, PCG_BLOCK_SIZE, 0, stream>>>(
                d_q_out, d_w_field[monomer_type], d_pcg_Ap[STREAM],
                rx, ry, rz, nx[0], nx[1], nx[2],
                bc_xl, bc_xh, bc_yl, bc_yh, bc_zl, bc_zh,
                dtau, n_grid);
        }
        else // dim == 2
        {
            matvec_free_kernel_2d<<<pcg_n_blocks, PCG_BLOCK_SIZE, 0, stream>>>(
                d_q_out, d_w_field[monomer_type], d_pcg_Ap[STREAM],
                rx, ry, nx[0], nx[1],
                bc_xl, bc_xh, bc_yl, bc_yh,
                dtau, n_grid);
        }

        // Initialize: r = b - Ax0, z = M^{-1}*r, p = z
        pcg_init_warmstart_kernel<<<pcg_n_blocks, PCG_BLOCK_SIZE, 0, stream>>>(
            d_pcg_r[STREAM], d_pcg_z[STREAM], d_pcg_p[STREAM],
            d_q_in, d_pcg_Ap[STREAM], d_diag_inv, n_grid);

        // Compute the number of threads needed for final reduction
        int reduce_threads = 1;
        while(reduce_threads < pcg_n_blocks) reduce_threads *= 2;
        if(reduce_threads > 1024) reduce_threads = 1024;

        // rz_old = (r, z)
        pcg_fused_dot_beta_kernel<<<pcg_n_blocks, PCG_BLOCK_SIZE, PCG_BLOCK_SIZE * sizeof(double), stream>>>(
            d_pcg_r[STREAM], d_pcg_z[STREAM], d_scalars, d_partial, n_grid, pcg_n_blocks);
        pcg_fused_reduce_rz_init_kernel<<<1, reduce_threads, reduce_threads * sizeof(double), stream>>>(
            d_scalars, d_partial, pcg_n_blocks);

        // PCG loop
        for(int iter = 0; iter < pcg_max_iter; iter++)
        {
            // 1. Ap = A * p (matrix-free)
            if(dim == 3)
            {
                matvec_free_kernel_3d<<<pcg_n_blocks, PCG_BLOCK_SIZE, 0, stream>>>(
                    d_pcg_p[STREAM], d_w_field[monomer_type], d_pcg_Ap[STREAM],
                    rx, ry, rz, nx[0], nx[1], nx[2],
                    bc_xl, bc_xh, bc_yl, bc_yh, bc_zl, bc_zh,
                    dtau, n_grid);
            }
            else // dim == 2
            {
                matvec_free_kernel_2d<<<pcg_n_blocks, PCG_BLOCK_SIZE, 0, stream>>>(
                    d_pcg_p[STREAM], d_w_field[monomer_type], d_pcg_Ap[STREAM],
                    rx, ry, nx[0], nx[1],
                    bc_xl, bc_xh, bc_yl, bc_yh,
                    dtau, n_grid);
            }

            // 2-3. Fused: pAp = (p, Ap), then alpha = rz_old / pAp
            pcg_fused_dot_alpha_kernel<<<pcg_n_blocks, PCG_BLOCK_SIZE, PCG_BLOCK_SIZE * sizeof(double), stream>>>(
                d_pcg_p[STREAM], d_pcg_Ap[STREAM], d_scalars, d_partial, n_grid, pcg_n_blocks);
            pcg_fused_reduce_alpha_kernel<<<1, reduce_threads, reduce_threads * sizeof(double), stream>>>(
                d_scalars, d_partial, pcg_n_blocks);

            // 4-5. Fused: x += alpha*p, r -= alpha*Ap, z = M^{-1}*r
            pcg_fused_update_xrz_kernel<<<pcg_n_blocks, PCG_BLOCK_SIZE, 0, stream>>>(
                d_q_out, d_pcg_r[STREAM], d_pcg_z[STREAM],
                d_pcg_p[STREAM], d_pcg_Ap[STREAM], d_diag_inv, d_scalars, n_grid);

            // 6-7. Fused: rz_new = (r, z), then beta = rz_new/rz_old, rz_old = rz_new
            pcg_fused_dot_beta_kernel<<<pcg_n_blocks, PCG_BLOCK_SIZE, PCG_BLOCK_SIZE * sizeof(double), stream>>>(
                d_pcg_r[STREAM], d_pcg_z[STREAM], d_scalars, d_partial, n_grid, pcg_n_blocks);
            pcg_fused_reduce_beta_kernel<<<1, reduce_threads, reduce_threads * sizeof(double), stream>>>(
                d_scalars, d_partial, pcg_n_blocks);

            // Check convergence periodically
            if((iter + 1) % PCG_CONV_CHECK_INTERVAL == 0 || iter == pcg_max_iter - 1)
            {
                double rz_new;
                gpu_error_check(cudaMemcpyAsync(&rz_new, &d_scalars[3],
                    sizeof(double), cudaMemcpyDeviceToHost, stream));
                gpu_error_check(cudaStreamSynchronize(stream));

                if(std::sqrt(std::abs(rz_new)) < pcg_tol)
                    break;
            }

            // 8. p = z + beta*p
            pcg_update_p_dev_kernel<<<pcg_n_blocks, PCG_BLOCK_SIZE, 0, stream>>>(
                d_pcg_p[STREAM], d_pcg_z[STREAM], d_scalars, n_grid);
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

void CudaSolverSDC::pcg_solve_step(int STREAM, int sub_interval,
                                    double* d_q_in, double* d_q_out, std::string monomer_type)
{
    // Matrix-free PCG: better memory efficiency and cache utilization
    // ~10% faster for 32^3 grids, ~2.4x faster for 48^3 grids
    matvec_free_solve(STREAM, sub_interval, d_q_in, d_q_out, monomer_type);
}

void CudaSolverSDC::advance_propagator(
    const int STREAM,
    double *d_q_in, double *d_q_out,
    std::string monomer_type, double *d_q_mask, int /*ds_index*/)
{
    try
    {
        const int n_grid = this->cb->get_total_grid();
        const double ds = this->molecules->get_ds();

        const int N_BLOCKS = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();
        cudaStream_t stream = streams[STREAM][0];

        // Initialize X[0] = q_in
        copy_array_kernel<<<N_BLOCKS, N_THREADS, 0, stream>>>(
            d_X[STREAM][0], d_q_in, n_grid);

        //=================================================================
        // Predictor: Fully implicit Backward Euler for both diffusion and reaction
        // Solve: (I - dtau*D∇² + dtau*w) X[m+1] = X[m]
        //=================================================================
        for(int m = 0; m < M - 1; m++)
        {
            // Fully implicit solve: matrix includes both diffusion and reaction
            implicit_solve_step(STREAM, m, d_X[STREAM][m], d_X[STREAM][m + 1], monomer_type);
        }

        //=================================================================
        // SDC Corrections (K iterations)
        //=================================================================
        double* d_w = d_w_field[monomer_type];

        for(int k_iter = 0; k_iter < K; k_iter++)
        {
            // Compute F at all GL nodes
            for(int m = 0; m < M; m++)
            {
                compute_F_device(STREAM, d_X[STREAM][m], d_w, d_F[STREAM][m], monomer_type);
            }

            // Store old values (fused kernel reduces launch overhead)
            for(int m = 0; m < M; m++)
            {
                copy_two_arrays_kernel<<<N_BLOCKS, N_THREADS, 0, stream>>>(
                    d_X_old[STREAM][m], d_X[STREAM][m],
                    d_F_old[STREAM][m], d_F[STREAM][m], n_grid);
            }

            // Reset X[0]
            copy_array_kernel<<<N_BLOCKS, N_THREADS, 0, stream>>>(
                d_X[STREAM][0], d_q_in, n_grid);

            // SDC correction sweep
            for(int m = 0; m < M - 1; m++)
            {
                double dtau = dtau_sub[m];

                // Initialize integral to zero
                gpu_error_check(cudaMemsetAsync(d_rhs[STREAM], 0, sizeof(double) * n_grid, stream));

                // Accumulate spectral integral: ∫ F dt = Σ S[m][j] * F[j] * ds
                for(int j = 0; j < M; j++)
                {
                    double weight = S[m][j] * ds;
                    accumulate_integral_kernel<<<N_BLOCKS, N_THREADS, 0, stream>>>(
                        d_rhs[STREAM], d_F_old[STREAM][j], weight, n_grid);
                }

                // Fully implicit RHS: X[m] + integral - dtau * F_old[m+1]
                // Subtract full F (both diffusion and reaction) since implicit solve handles both
                sdc_rhs_kernel<<<N_BLOCKS, N_THREADS, 0, stream>>>(
                    d_temp[STREAM], d_X[STREAM][m], d_rhs[STREAM], dtau, d_F_old[STREAM][m + 1], n_grid);

                // Fully implicit SDC: solve (I - dtau*D∇² + dtau*w) X[m+1] = rhs
                implicit_solve_step(STREAM, m, d_temp[STREAM], d_X[STREAM][m + 1], monomer_type);
            }
        }

        // Output is the final GL node
        copy_array_kernel<<<N_BLOCKS, N_THREADS, 0, stream>>>(
            d_q_out, d_X[STREAM][M - 1], n_grid);

        // Apply mask if provided
        if(d_q_mask != nullptr)
        {
            apply_mask_kernel<<<N_BLOCKS, N_THREADS, 0, stream>>>(
                d_q_out, d_q_mask, n_grid);
        }

        // Synchronize stream to ensure all operations complete before returning
        gpu_error_check(cudaStreamSynchronize(stream));
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

void CudaSolverSDC::compute_single_segment_stress(
    const int /*STREAM*/,
    double* /*d_q_pair*/, double* /*d_segment_stress*/,
    std::string /*monomer_type*/, bool /*is_half_bond_length*/)
{
    throw_with_line_number("SDC method does not support stress computation.");
}
