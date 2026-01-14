/**
 * @file CudaSolverSDC.cu
 * @brief CUDA SDC solver implementation.
 *
 * Implements the Spectral Deferred Correction method for solving the
 * modified diffusion equation on GPU using Gauss-Lobatto quadrature nodes.
 */

#include <iostream>
#include <cmath>

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
    if(idx >= n_grid) return;

    int i = idx / (nx_J * nx_K);
    int j = (idx / nx_K) % nx_J;
    int k = idx % nx_K;

    int im = (bc_xl == BoundaryCondition::PERIODIC) ? (nx_I + i - 1) % nx_I : sdc_max_of_two(0, i - 1);
    int ip = (bc_xh == BoundaryCondition::PERIODIC) ? (i + 1) % nx_I : sdc_min_of_two(nx_I - 1, i + 1);
    int jm = (bc_yl == BoundaryCondition::PERIODIC) ? (nx_J + j - 1) % nx_J : sdc_max_of_two(0, j - 1);
    int jp = (bc_yh == BoundaryCondition::PERIODIC) ? (j + 1) % nx_J : sdc_min_of_two(nx_J - 1, j + 1);
    int km = (bc_zl == BoundaryCondition::PERIODIC) ? (nx_K + k - 1) % nx_K : sdc_max_of_two(0, k - 1);
    int kp = (bc_zh == BoundaryCondition::PERIODIC) ? (k + 1) % nx_K : sdc_min_of_two(nx_K - 1, k + 1);

    int idx_im = im * nx_J * nx_K + j * nx_K + k;
    int idx_ip = ip * nx_J * nx_K + j * nx_K + k;
    int idx_jm = i * nx_J * nx_K + jm * nx_K + k;
    int idx_jp = i * nx_J * nx_K + jp * nx_K + k;
    int idx_km = i * nx_J * nx_K + j * nx_K + km;
    int idx_kp = i * nx_J * nx_K + j * nx_K + kp;

    double Lx = alpha_x * (d_q[idx_im] + d_q[idx_ip] - 2.0 * d_q[idx]);
    double Ly = alpha_y * (d_q[idx_jm] + d_q[idx_jp] - 2.0 * d_q[idx]);
    double Lz = alpha_z * (d_q[idx_km] + d_q[idx_kp] - 2.0 * d_q[idx]);
    d_F[idx] = Lx + Ly + Lz - d_w[idx] * d_q[idx];
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
    if(idx >= n_grid) return;

    int i = idx / nx_J;
    int j = idx % nx_J;

    int im = (bc_xl == BoundaryCondition::PERIODIC) ? (nx_I + i - 1) % nx_I : sdc_max_of_two(0, i - 1);
    int ip = (bc_xh == BoundaryCondition::PERIODIC) ? (i + 1) % nx_I : sdc_min_of_two(nx_I - 1, i + 1);
    int jm = (bc_yl == BoundaryCondition::PERIODIC) ? (nx_J + j - 1) % nx_J : sdc_max_of_two(0, j - 1);
    int jp = (bc_yh == BoundaryCondition::PERIODIC) ? (j + 1) % nx_J : sdc_min_of_two(nx_J - 1, j + 1);

    int idx_im = im * nx_J + j;
    int idx_ip = ip * nx_J + j;
    int idx_jm = i * nx_J + jm;
    int idx_jp = i * nx_J + jp;

    double Lx = alpha_x * (d_q[idx_im] + d_q[idx_ip] - 2.0 * d_q[idx]);
    double Ly = alpha_y * (d_q[idx_jm] + d_q[idx_jp] - 2.0 * d_q[idx]);
    d_F[idx] = Lx + Ly - d_w[idx] * d_q[idx];
}

__global__ void compute_F_kernel_1d(
    const double* d_q, const double* d_w, double* d_F,
    double alpha_x, int nx_I,
    BoundaryCondition bc_xl, BoundaryCondition bc_xh,
    int n_grid)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n_grid) return;

    int im = (bc_xl == BoundaryCondition::PERIODIC) ? (nx_I + i - 1) % nx_I : sdc_max_of_two(0, i - 1);
    int ip = (bc_xh == BoundaryCondition::PERIODIC) ? (i + 1) % nx_I : sdc_min_of_two(nx_I - 1, i + 1);

    double Lx = alpha_x * (d_q[im] + d_q[ip] - 2.0 * d_q[i]);
    d_F[i] = Lx - d_w[i] * d_q[i];
}

__global__ void apply_exp_dw_kernel(
    double* d_out, const double* d_in, const double* d_exp_dw, int n_grid)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n_grid) return;
    d_out[idx] = d_exp_dw[idx] * d_in[idx];
}

__global__ void copy_array_kernel(double* d_out, const double* d_in, int n_grid)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n_grid) return;
    d_out[idx] = d_in[idx];
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
    const double* d_w,
    const double* d_X_old_mp1,
    int n_grid)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n_grid) return;
    // For IMEX-SDC: rhs = X[m] + ∫F dt - dtau * D∇²q_old[m+1]
    // Since F = D∇²q - wq, we have D∇²q = F + wq
    // So: rhs = X[m] + ∫F dt - dtau * (F_old[m+1] + w * X_old[m+1])
    double diff_term = d_F_mp1[idx] + d_w[idx] * d_X_old_mp1[idx];
    d_rhs[idx] = d_X_m[idx] + d_integral[idx] - dtau * diff_term;
}

__global__ void accumulate_integral_kernel(
    double* d_integral,
    const double* d_F,
    double weight,
    int n_grid)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n_grid) return;
    d_integral[idx] += weight * d_F[idx];
}

__global__ void apply_mask_kernel(double* d_q, const double* d_mask, int n_grid)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n_grid) return;
    d_q[idx] *= d_mask[idx];
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
        d_xh.resize(M - 1);
        d_yl.resize(M - 1);
        d_yd.resize(M - 1);
        d_yh.resize(M - 1);
        d_zl.resize(M - 1);
        d_zd.resize(M - 1);
        d_zh.resize(M - 1);
        d_exp_dw_sub.resize(M - 1);

        for(int m = 0; m < M - 1; m++)
        {
            for(const auto& item: molecules->get_bond_lengths())
            {
                std::string monomer_type = item.first;

                gpu_error_check(cudaMalloc((void**)&d_xl[m][monomer_type], sizeof(double) * nx[0]));
                gpu_error_check(cudaMalloc((void**)&d_xd[m][monomer_type], sizeof(double) * nx[0]));
                gpu_error_check(cudaMalloc((void**)&d_xh[m][monomer_type], sizeof(double) * nx[0]));

                gpu_error_check(cudaMalloc((void**)&d_yl[m][monomer_type], sizeof(double) * nx[1]));
                gpu_error_check(cudaMalloc((void**)&d_yd[m][monomer_type], sizeof(double) * nx[1]));
                gpu_error_check(cudaMalloc((void**)&d_yh[m][monomer_type], sizeof(double) * nx[1]));

                gpu_error_check(cudaMalloc((void**)&d_zl[m][monomer_type], sizeof(double) * nx[2]));
                gpu_error_check(cudaMalloc((void**)&d_zd[m][monomer_type], sizeof(double) * nx[2]));
                gpu_error_check(cudaMalloc((void**)&d_zh[m][monomer_type], sizeof(double) * nx[2]));

                gpu_error_check(cudaMalloc((void**)&d_exp_dw_sub[m][monomer_type], sizeof(double) * n_grid));
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

        // Allocate offset arrays for ADI
        if(DIM == 3)
        {
            int offset_xy[nx[0] * nx[1]];
            int offset_yz[nx[1] * nx[2]];
            int offset_xz[nx[0] * nx[2]];
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

            gpu_error_check(cudaMemcpy(d_offset_xy, offset_xy, sizeof(int) * nx[0] * nx[1], cudaMemcpyHostToDevice));
            gpu_error_check(cudaMemcpy(d_offset_yz, offset_yz, sizeof(int) * nx[1] * nx[2], cudaMemcpyHostToDevice));
            gpu_error_check(cudaMemcpy(d_offset_xz, offset_xz, sizeof(int) * nx[0] * nx[2], cudaMemcpyHostToDevice));
        }
        else if(DIM == 2)
        {
            int offset_x[nx[0]];
            int offset_y[nx[1]];

            for(int i = 0; i < nx[0]; i++)
                offset_x[i] = i * nx[1];
            for(int j = 0; j < nx[1]; j++)
                offset_y[j] = j;

            gpu_error_check(cudaMalloc((void**)&d_offset_x, sizeof(int) * nx[0]));
            gpu_error_check(cudaMalloc((void**)&d_offset_y, sizeof(int) * nx[1]));

            gpu_error_check(cudaMemcpy(d_offset_x, offset_x, sizeof(int) * nx[0], cudaMemcpyHostToDevice));
            gpu_error_check(cudaMemcpy(d_offset_y, offset_y, sizeof(int) * nx[1], cudaMemcpyHostToDevice));
        }
        else if(DIM == 1)
        {
            int offset[1] = {0};
            gpu_error_check(cudaMalloc((void**)&d_offset, sizeof(int)));
            gpu_error_check(cudaMemcpy(d_offset, offset, sizeof(int), cudaMemcpyHostToDevice));
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
    // Free integration matrix
    cudaFree(d_S);

    // Free tridiagonal coefficients
    for(int m = 0; m < M - 1; m++)
    {
        for(const auto& item: d_xl[m])
            cudaFree(item.second);
        for(const auto& item: d_xd[m])
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
        for(const auto& item: d_exp_dw_sub[m])
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
    else
    {
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

        for(const auto& item: this->molecules->get_bond_lengths())
        {
            std::string monomer_type = item.first;
            double bond_length_sq = item.second * item.second;

            for(int m = 0; m < M - 1; m++)
            {
                double dtau = (tau[m + 1] - tau[m]) * ds;

                // Compute coefficients on host
                double h_xl[nx[0]], h_xd[nx[0]], h_xh[nx[0]];
                double h_yl[nx[1]], h_yd[nx[1]], h_yh[nx[1]];
                double h_zl[nx[2]], h_zd[nx[2]], h_zh[nx[2]];

                FiniteDifference::get_laplacian_matrix(
                    this->cb->get_boundary_conditions(),
                    this->cb->get_nx(), this->cb->get_dx(),
                    h_xl, h_xd, h_xh,
                    h_yl, h_yd, h_yh,
                    h_zl, h_zd, h_zh,
                    bond_length_sq, dtau);

                // Copy to device
                gpu_error_check(cudaMemcpy(d_xl[m][monomer_type], h_xl, sizeof(double) * nx[0], cudaMemcpyHostToDevice));
                gpu_error_check(cudaMemcpy(d_xd[m][monomer_type], h_xd, sizeof(double) * nx[0], cudaMemcpyHostToDevice));
                gpu_error_check(cudaMemcpy(d_xh[m][monomer_type], h_xh, sizeof(double) * nx[0], cudaMemcpyHostToDevice));

                if(dim >= 2)
                {
                    gpu_error_check(cudaMemcpy(d_yl[m][monomer_type], h_yl, sizeof(double) * nx[1], cudaMemcpyHostToDevice));
                    gpu_error_check(cudaMemcpy(d_yd[m][monomer_type], h_yd, sizeof(double) * nx[1], cudaMemcpyHostToDevice));
                    gpu_error_check(cudaMemcpy(d_yh[m][monomer_type], h_yh, sizeof(double) * nx[1], cudaMemcpyHostToDevice));
                }

                if(dim == 3)
                {
                    gpu_error_check(cudaMemcpy(d_zl[m][monomer_type], h_zl, sizeof(double) * nx[2], cudaMemcpyHostToDevice));
                    gpu_error_check(cudaMemcpy(d_zd[m][monomer_type], h_zd, sizeof(double) * nx[2], cudaMemcpyHostToDevice));
                    gpu_error_check(cudaMemcpy(d_zh[m][monomer_type], h_zh, sizeof(double) * nx[2], cudaMemcpyHostToDevice));
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
    const double ds = this->molecules->get_ds();

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

        // Compute exp(-w * dtau) for each sub-interval on host then copy
        std::vector<double> h_exp_dw(n_grid);
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

        for(int m = 0; m < M - 1; m++)
        {
            double dtau = (tau[m + 1] - tau[m]) * ds;

            for(int i = 0; i < n_grid; i++)
                h_exp_dw[i] = std::exp(-h_w[i] * dtau);

            gpu_error_check(cudaMemcpy(d_exp_dw_sub[m][monomer_type], h_exp_dw.data(),
                sizeof(double) * n_grid, cudaMemcpyHostToDevice));
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

    if(dim == 3)
    {
        double alpha_x = D / (dx[0] * dx[0]);
        double alpha_y = D / (dx[1] * dx[1]);
        double alpha_z = D / (dx[2] * dx[2]);

        compute_F_kernel_3d<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
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

        compute_F_kernel_2d<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
            d_q, d_w, d_F,
            alpha_x, alpha_y,
            nx[0], nx[1],
            bc[0], bc[1], bc[2], bc[3],
            n_grid);
    }
    else
    {
        double alpha_x = D / (dx[0] * dx[0]);

        compute_F_kernel_1d<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
            d_q, d_w, d_F,
            alpha_x, nx[0],
            bc[0], bc[1],
            n_grid);
    }
}

void CudaSolverSDC::adi_step(int STREAM, int sub_interval,
                              double* d_q_in, double* d_q_out, std::string monomer_type)
{
    std::vector<BoundaryCondition> bc = this->cb->get_boundary_conditions();

    if(dim == 3)
        adi_step_3d(STREAM, sub_interval, bc, d_q_in, d_q_out, monomer_type);
    else if(dim == 2)
        adi_step_2d(STREAM, sub_interval, bc, d_q_in, d_q_out, monomer_type);
    else if(dim == 1)
        adi_step_1d(STREAM, sub_interval, bc, d_q_in, d_q_out, monomer_type);
}

void CudaSolverSDC::adi_step_3d(int STREAM, int sub_interval,
    std::vector<BoundaryCondition> bc,
    double* d_q_in, double* d_q_out, std::string monomer_type)
{
    const int n_grid = this->cb->get_total_grid();
    const std::vector<int> nx = this->cb->get_nx();

    double *_d_xl = d_xl[sub_interval][monomer_type];
    double *_d_xd = d_xd[sub_interval][monomer_type];
    double *_d_xh = d_xh[sub_interval][monomer_type];
    double *_d_yl = d_yl[sub_interval][monomer_type];
    double *_d_yd = d_yd[sub_interval][monomer_type];
    double *_d_yh = d_yh[sub_interval][monomer_type];
    double *_d_zl = d_zl[sub_interval][monomer_type];
    double *_d_zd = d_zd[sub_interval][monomer_type];
    double *_d_zh = d_zh[sub_interval][monomer_type];

    const int N_BLOCKS = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();

    // Save input before X-sweep overwrites d_temp (which might be d_q_in)
    // This fixes the bug where Y/Z-sweeps used X-sweep result instead of original input
    gpu_error_check(cudaMemcpyAsync(d_q_in_saved[STREAM], d_q_in, sizeof(double) * n_grid,
                                    cudaMemcpyDeviceToDevice, streams[STREAM][0]));

    // Step 1: X-sweep
    compute_crank_3d_step_1<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
        bc[0], bc[1], bc[2], bc[3], bc[4], bc[5],
        _d_xl, _d_xd, _d_xh, nx[0],
        _d_yl, _d_yd, _d_yh, nx[1],
        _d_zl, _d_zd, _d_zh, nx[2],
        d_q_star[STREAM], d_q_in_saved[STREAM], n_grid);

    if(bc[0] == BoundaryCondition::PERIODIC)
        tridiagonal_periodic<<<nx[1] * nx[2], nx[0], sizeof(double) * 3 * nx[0], streams[STREAM][0]>>>(
            _d_xl, _d_xd, _d_xh, d_c_star[STREAM], d_q_sparse[STREAM],
            d_q_star[STREAM], d_temp[STREAM], d_offset_yz, nx[1] * nx[2], nx[1] * nx[2], nx[0]);
    else
        tridiagonal<<<nx[1] * nx[2], nx[0], sizeof(double) * 2 * nx[0], streams[STREAM][0]>>>(
            _d_xl, _d_xd, _d_xh, d_c_star[STREAM],
            d_q_star[STREAM], d_temp[STREAM], d_offset_yz, nx[1] * nx[2], nx[1] * nx[2], nx[0]);

    // Step 2: Y-sweep - use saved input for correction terms
    compute_crank_3d_step_2<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
        bc[2], bc[3],
        _d_yl, _d_yd, _d_yh, nx[1], nx[2],
        d_q_star[STREAM], d_temp[STREAM], d_q_in_saved[STREAM], n_grid);

    if(bc[2] == BoundaryCondition::PERIODIC)
        tridiagonal_periodic<<<nx[0] * nx[2], nx[1], sizeof(double) * 3 * nx[1], streams[STREAM][0]>>>(
            _d_yl, _d_yd, _d_yh, d_c_star[STREAM], d_q_sparse[STREAM],
            d_q_star[STREAM], d_q_dstar[STREAM], d_offset_xz, nx[0] * nx[2], nx[2], nx[1]);
    else
        tridiagonal<<<nx[0] * nx[2], nx[1], sizeof(double) * 2 * nx[1], streams[STREAM][0]>>>(
            _d_yl, _d_yd, _d_yh, d_c_star[STREAM],
            d_q_star[STREAM], d_q_dstar[STREAM], d_offset_xz, nx[0] * nx[2], nx[2], nx[1]);

    // Step 3: Z-sweep - use saved input for correction terms
    compute_crank_3d_step_3<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
        bc[4], bc[5],
        _d_zl, _d_zd, _d_zh, nx[1], nx[2],
        d_q_star[STREAM], d_q_dstar[STREAM], d_q_in_saved[STREAM], n_grid);

    if(bc[4] == BoundaryCondition::PERIODIC)
        tridiagonal_periodic<<<nx[0] * nx[1], nx[2], sizeof(double) * 3 * nx[2], streams[STREAM][0]>>>(
            _d_zl, _d_zd, _d_zh, d_c_star[STREAM], d_q_sparse[STREAM],
            d_q_star[STREAM], d_q_out, d_offset_xy, nx[0] * nx[1], 1, nx[2]);
    else
        tridiagonal<<<nx[0] * nx[1], nx[2], sizeof(double) * 2 * nx[2], streams[STREAM][0]>>>(
            _d_zl, _d_zd, _d_zh, d_c_star[STREAM],
            d_q_star[STREAM], d_q_out, d_offset_xy, nx[0] * nx[1], 1, nx[2]);
}

void CudaSolverSDC::adi_step_2d(int STREAM, int sub_interval,
    std::vector<BoundaryCondition> bc,
    double* d_q_in, double* d_q_out, std::string monomer_type)
{
    const int n_grid = this->cb->get_total_grid();
    const std::vector<int> nx = this->cb->get_nx();

    double *_d_xl = d_xl[sub_interval][monomer_type];
    double *_d_xd = d_xd[sub_interval][monomer_type];
    double *_d_xh = d_xh[sub_interval][monomer_type];
    double *_d_yl = d_yl[sub_interval][monomer_type];
    double *_d_yd = d_yd[sub_interval][monomer_type];
    double *_d_yh = d_yh[sub_interval][monomer_type];

    const int N_BLOCKS = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();

    // Save input before X-sweep overwrites d_temp (which might be d_q_in)
    // This fixes the bug where Y-sweep used X-sweep result instead of original input
    gpu_error_check(cudaMemcpyAsync(d_q_in_saved[STREAM], d_q_in, sizeof(double) * n_grid,
                                    cudaMemcpyDeviceToDevice, streams[STREAM][0]));

    // Step 1: X-sweep
    compute_crank_2d_step_1<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
        bc[0], bc[1], bc[2], bc[3],
        _d_xl, _d_xd, _d_xh, nx[0],
        _d_yl, _d_yd, _d_yh, nx[1],
        d_q_star[STREAM], d_q_in_saved[STREAM], n_grid);

    if(bc[0] == BoundaryCondition::PERIODIC)
        tridiagonal_periodic<<<nx[1], nx[0], sizeof(double) * 3 * nx[0], streams[STREAM][0]>>>(
            _d_xl, _d_xd, _d_xh, d_c_star[STREAM], d_q_sparse[STREAM],
            d_q_star[STREAM], d_temp[STREAM], d_offset_y, nx[1], nx[1], nx[0]);
    else
        tridiagonal<<<nx[1], nx[0], sizeof(double) * 2 * nx[0], streams[STREAM][0]>>>(
            _d_xl, _d_xd, _d_xh, d_c_star[STREAM],
            d_q_star[STREAM], d_temp[STREAM], d_offset_y, nx[1], nx[1], nx[0]);

    // Step 2: Y-sweep - use saved input for correction terms
    compute_crank_2d_step_2<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
        bc[2], bc[3],
        _d_yl, _d_yd, _d_yh, nx[1],
        d_q_star[STREAM], d_temp[STREAM], d_q_in_saved[STREAM], n_grid);

    if(bc[2] == BoundaryCondition::PERIODIC)
        tridiagonal_periodic<<<nx[0], nx[1], sizeof(double) * 3 * nx[1], streams[STREAM][0]>>>(
            _d_yl, _d_yd, _d_yh, d_c_star[STREAM], d_q_sparse[STREAM],
            d_q_star[STREAM], d_q_out, d_offset_x, nx[0], 1, nx[1]);
    else
        tridiagonal<<<nx[0], nx[1], sizeof(double) * 2 * nx[1], streams[STREAM][0]>>>(
            _d_yl, _d_yd, _d_yh, d_c_star[STREAM],
            d_q_star[STREAM], d_q_out, d_offset_x, nx[0], 1, nx[1]);
}

void CudaSolverSDC::adi_step_1d(int STREAM, int sub_interval,
    std::vector<BoundaryCondition> bc,
    double* d_q_in, double* d_q_out, std::string monomer_type)
{
    const int n_grid = this->cb->get_total_grid();
    const std::vector<int> nx = this->cb->get_nx();

    double *_d_xl = d_xl[sub_interval][monomer_type];
    double *_d_xd = d_xd[sub_interval][monomer_type];
    double *_d_xh = d_xh[sub_interval][monomer_type];

    const int N_BLOCKS = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();

    compute_crank_1d<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
        bc[0], bc[1],
        _d_xl, _d_xd, _d_xh,
        d_q_star[STREAM], d_q_in, n_grid);

    if(bc[0] == BoundaryCondition::PERIODIC)
        tridiagonal_periodic<<<1, nx[0], sizeof(double) * 3 * nx[0], streams[STREAM][0]>>>(
            _d_xl, _d_xd, _d_xh, d_c_star[STREAM], d_q_sparse[STREAM],
            d_q_star[STREAM], d_q_out, d_offset, 1, 1, nx[0]);
    else
        tridiagonal<<<1, nx[0], sizeof(double) * 2 * nx[0], streams[STREAM][0]>>>(
            _d_xl, _d_xd, _d_xh, d_c_star[STREAM],
            d_q_star[STREAM], d_q_out, d_offset, 1, 1, nx[0]);
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

        // Initialize X[0] = q_in
        copy_array_kernel<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
            d_X[STREAM][0], d_q_in, n_grid);

        //=================================================================
        // Predictor: Backward Euler for diffusion, explicit for reaction
        //=================================================================
        for(int m = 0; m < M - 1; m++)
        {
            // Apply reaction term: temp = exp(-w*dtau) * X[m]
            apply_exp_dw_kernel<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                d_temp[STREAM], d_X[STREAM][m], d_exp_dw_sub[m][monomer_type], n_grid);

            // Apply diffusion implicitly using ADI
            adi_step(STREAM, m, d_temp[STREAM], d_X[STREAM][m + 1], monomer_type);
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

            // Store old values
            for(int m = 0; m < M; m++)
            {
                copy_array_kernel<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                    d_X_old[STREAM][m], d_X[STREAM][m], n_grid);
                copy_array_kernel<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                    d_F_old[STREAM][m], d_F[STREAM][m], n_grid);
            }

            // Reset X[0]
            copy_array_kernel<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                d_X[STREAM][0], d_q_in, n_grid);

            // SDC correction sweep
            for(int m = 0; m < M - 1; m++)
            {
                double dtau = (tau[m + 1] - tau[m]) * ds;

                // Initialize integral to zero
                gpu_error_check(cudaMemsetAsync(d_rhs[STREAM], 0, sizeof(double) * n_grid, streams[STREAM][0]));

                // Accumulate spectral integral: ∫ F dt = Σ S[m][j] * F[j] * ds
                for(int j = 0; j < M; j++)
                {
                    double weight = S[m][j] * ds;
                    accumulate_integral_kernel<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                        d_rhs[STREAM], d_F_old[STREAM][j], weight, n_grid);
                }

                // Complete RHS: X[m] + integral - dtau * (F_old[m+1] + w * X_old[m+1])
                // This ensures we only subtract the diffusion term, not the reaction term
                sdc_rhs_kernel<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                    d_temp[STREAM], d_X[STREAM][m], d_rhs[STREAM], dtau, d_F_old[STREAM][m + 1],
                    d_w, d_X_old[STREAM][m + 1], n_grid);

                // Semi-implicit SDC: solve (I - dtau*D∇²) X[m+1] = rhs
                adi_step(STREAM, m, d_temp[STREAM], d_X[STREAM][m + 1], monomer_type);
            }
        }

        // Output is the final GL node
        copy_array_kernel<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
            d_q_out, d_X[STREAM][M - 1], n_grid);

        // Apply mask if provided
        if(d_q_mask != nullptr)
        {
            apply_mask_kernel<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                d_q_out, d_q_mask, n_grid);
        }
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
