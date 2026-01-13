/**
 * @file CudaSolverRichardsonGlobal.cu
 * @brief CUDA solver using Global Richardson extrapolation.
 *
 * Implements propagator advancement for continuous chains using finite
 * differences with Global Richardson extrapolation for 4th-order accuracy.
 * Unlike CN-ADI4 (per-step Richardson), this method maintains two independent
 * propagator evolutions and applies Richardson extrapolation to combine them.
 *
 * **Global Richardson vs CN-ADI4:**
 *
 * CN-ADI4 (per-step):
 * - Each step: q_out = (4*A(A(q_in, ds/2), ds/2) - A(q_in, ds)) / 3
 * - Next step uses q_out as input
 *
 * Global Richardson (this class):
 * - Maintains two independent states: q_full_internal, q_half_internal
 * - Each step:
 *   - q_full_internal_{n+1} = A(q_full_internal_n, ds)
 *   - q_half_internal_{n+1} = A(A(q_half_internal_n, ds/2), ds/2)
 *   - q_out = (4*q_half_internal - q_full_internal) / 3
 *
 * @see CudaSolverRichardsonGlobal.h for algorithm details
 * @see CudaSolverCNADI for per-step Richardson implementation
 */

#include <iostream>
#include <cmath>
#include <numbers>
#include "CudaSolverRichardsonGlobal.h"
#include "CudaSolverCNADI.h"  // For tridiagonal kernel declarations

CudaSolverRichardsonGlobal::CudaSolverRichardsonGlobal(
    ComputationBox<double>* cb,
    Molecules *molecules,
    int n_streams,
    cudaStream_t streams[MAX_STREAMS][2],
    bool reduce_memory_usage)
{
    try{
        this->cb = cb;
        this->molecules = molecules;
        this->chain_model = molecules->get_model_name();
        this->n_streams = n_streams;
        this->reduce_memory_usage = reduce_memory_usage;

        if(molecules->get_model_name() != "continuous")
            throw_with_line_number("Global Richardson method only supports 'continuous' chain model.");

        if(!cb->is_orthogonal())
            throw_with_line_number("Global Richardson method only supports orthogonal unit cells. "
                                   "Use pseudo-spectral method for non-orthogonal systems.");

        const int M = cb->get_total_grid();
        const int DIM = cb->get_dim();
        this->dim = DIM;
        std::vector<int> nx(DIM);
        if (DIM == 3)
            nx = {cb->get_nx(0), cb->get_nx(1), cb->get_nx(2)};
        else if (DIM == 2)
            nx = {cb->get_nx(0), cb->get_nx(1), 1};
        else if (DIM == 1)
            nx = {cb->get_nx(0), 1, 1};

        // Copy streams
        for(int i=0; i<n_streams; i++)
        {
            this->streams[i][0] = streams[i][0];
            this->streams[i][1] = streams[i][1];
        }

        // Global Richardson uses only global ds (ds_index=1)
        // Create exp_dw and exp_dw_half for ds_index=1 and each monomer type
        const int ds_idx = 1;
        for(const auto& item: molecules->get_bond_lengths())
        {
            std::string monomer_type = item.first;
            d_exp_dw     [ds_idx][monomer_type] = nullptr;
            d_exp_dw_half[ds_idx][monomer_type] = nullptr;

            gpu_error_check(cudaMalloc((void**)&d_exp_dw     [ds_idx][monomer_type], sizeof(double)*M));
            gpu_error_check(cudaMalloc((void**)&d_exp_dw_half[ds_idx][monomer_type], sizeof(double)*M));

            // Full step coefficients
            d_xl[monomer_type] = nullptr;
            d_xd[monomer_type] = nullptr;
            d_xh[monomer_type] = nullptr;
            gpu_error_check(cudaMalloc((void**)&d_xl[monomer_type], sizeof(double)*nx[0]));
            gpu_error_check(cudaMalloc((void**)&d_xd[monomer_type], sizeof(double)*nx[0]));
            gpu_error_check(cudaMalloc((void**)&d_xh[monomer_type], sizeof(double)*nx[0]));

            d_yl[monomer_type] = nullptr;
            d_yd[monomer_type] = nullptr;
            d_yh[monomer_type] = nullptr;
            gpu_error_check(cudaMalloc((void**)&d_yl[monomer_type], sizeof(double)*nx[1]));
            gpu_error_check(cudaMalloc((void**)&d_yd[monomer_type], sizeof(double)*nx[1]));
            gpu_error_check(cudaMalloc((void**)&d_yh[monomer_type], sizeof(double)*nx[1]));

            d_zl[monomer_type] = nullptr;
            d_zd[monomer_type] = nullptr;
            d_zh[monomer_type] = nullptr;
            gpu_error_check(cudaMalloc((void**)&d_zl[monomer_type], sizeof(double)*nx[2]));
            gpu_error_check(cudaMalloc((void**)&d_zd[monomer_type], sizeof(double)*nx[2]));
            gpu_error_check(cudaMalloc((void**)&d_zh[monomer_type], sizeof(double)*nx[2]));

            // Half step coefficients
            d_xl_half[monomer_type] = nullptr;
            d_xd_half[monomer_type] = nullptr;
            d_xh_half[monomer_type] = nullptr;
            gpu_error_check(cudaMalloc((void**)&d_xl_half[monomer_type], sizeof(double)*nx[0]));
            gpu_error_check(cudaMalloc((void**)&d_xd_half[monomer_type], sizeof(double)*nx[0]));
            gpu_error_check(cudaMalloc((void**)&d_xh_half[monomer_type], sizeof(double)*nx[0]));

            d_yl_half[monomer_type] = nullptr;
            d_yd_half[monomer_type] = nullptr;
            d_yh_half[monomer_type] = nullptr;
            gpu_error_check(cudaMalloc((void**)&d_yl_half[monomer_type], sizeof(double)*nx[1]));
            gpu_error_check(cudaMalloc((void**)&d_yd_half[monomer_type], sizeof(double)*nx[1]));
            gpu_error_check(cudaMalloc((void**)&d_yh_half[monomer_type], sizeof(double)*nx[1]));

            d_zl_half[monomer_type] = nullptr;
            d_zd_half[monomer_type] = nullptr;
            d_zh_half[monomer_type] = nullptr;
            gpu_error_check(cudaMalloc((void**)&d_zl_half[monomer_type], sizeof(double)*nx[2]));
            gpu_error_check(cudaMalloc((void**)&d_zd_half[monomer_type], sizeof(double)*nx[2]));
            gpu_error_check(cudaMalloc((void**)&d_zh_half[monomer_type], sizeof(double)*nx[2]));
        }

        // Allocate per-stream workspace arrays
        if(DIM == 3)
        {
            for(int i=0; i<n_streams; i++)
            {
                gpu_error_check(cudaMalloc((void**)&d_q_star[i], sizeof(double)*M));
                gpu_error_check(cudaMalloc((void**)&d_q_dstar[i], sizeof(double)*M));
                gpu_error_check(cudaMalloc((void**)&d_c_star[i], sizeof(double)*M));
                gpu_error_check(cudaMalloc((void**)&d_q_sparse[i], sizeof(double)*M));
                gpu_error_check(cudaMalloc((void**)&d_temp[i], sizeof(double)*M));
                gpu_error_check(cudaMalloc((void**)&d_q_full_internal[i], sizeof(double)*M));
                gpu_error_check(cudaMalloc((void**)&d_q_half_internal[i], sizeof(double)*M));
                gpu_error_check(cudaMalloc((void**)&d_q_work[i], sizeof(double)*M));
                is_initialized[i] = false;
            }

            int offset_xy[nx[0]*nx[1]];
            int offset_yz[nx[1]*nx[2]];
            int offset_xz[nx[0]*nx[2]];
            int count;

            count = 0;
            for(int i=0;i<nx[0];i++)
                for(int j=0;j<nx[1];j++)
                    offset_xy[count++] = i*nx[1]*nx[2] + j*nx[2];

            count = 0;
            for(int j=0;j<nx[1];j++)
                for(int k=0;k<nx[2];k++)
                    offset_yz[count++] = j*nx[2] + k;

            count = 0;
            for(int i=0;i<nx[0];i++)
                for(int k=0;k<nx[2];k++)
                    offset_xz[count++] = i*nx[1]*nx[2] + k;

            gpu_error_check(cudaMalloc((void**)&d_offset_xy, sizeof(int)*nx[0]*nx[1]));
            gpu_error_check(cudaMalloc((void**)&d_offset_yz, sizeof(int)*nx[1]*nx[2]));
            gpu_error_check(cudaMalloc((void**)&d_offset_xz, sizeof(int)*nx[0]*nx[2]));

            gpu_error_check(cudaMemcpy(d_offset_xy, offset_xy, sizeof(int)*nx[0]*nx[1], cudaMemcpyHostToDevice));
            gpu_error_check(cudaMemcpy(d_offset_yz, offset_yz, sizeof(int)*nx[1]*nx[2], cudaMemcpyHostToDevice));
            gpu_error_check(cudaMemcpy(d_offset_xz, offset_xz, sizeof(int)*nx[0]*nx[2], cudaMemcpyHostToDevice));
        }
        else if(DIM == 2)
        {
            for(int i=0; i<n_streams; i++)
            {
                gpu_error_check(cudaMalloc((void**)&d_q_star[i], sizeof(double)*M));
                gpu_error_check(cudaMalloc((void**)&d_c_star[i], sizeof(double)*M));
                gpu_error_check(cudaMalloc((void**)&d_q_sparse[i], sizeof(double)*M));
                gpu_error_check(cudaMalloc((void**)&d_temp[i], sizeof(double)*M));
                gpu_error_check(cudaMalloc((void**)&d_q_full_internal[i], sizeof(double)*M));
                gpu_error_check(cudaMalloc((void**)&d_q_half_internal[i], sizeof(double)*M));
                gpu_error_check(cudaMalloc((void**)&d_q_work[i], sizeof(double)*M));
                is_initialized[i] = false;
            }

            int offset_x[nx[0]];
            int offset_y[nx[1]];

            for(int i=0;i<nx[0];i++)
                offset_x[i] = i*nx[1];

            for(int j=0;j<nx[1];j++)
                offset_y[j] = j;

            gpu_error_check(cudaMalloc((void**)&d_offset_x, sizeof(int)*nx[0]));
            gpu_error_check(cudaMalloc((void**)&d_offset_y, sizeof(int)*nx[1]));

            gpu_error_check(cudaMemcpy(d_offset_x, offset_x, sizeof(int)*nx[0], cudaMemcpyHostToDevice));
            gpu_error_check(cudaMemcpy(d_offset_y, offset_y, sizeof(int)*nx[1], cudaMemcpyHostToDevice));
        }
        else if(DIM == 1)
        {
            for(int i=0; i<n_streams; i++)
            {
                gpu_error_check(cudaMalloc((void**)&d_q_star[i], sizeof(double)*M));
                gpu_error_check(cudaMalloc((void**)&d_c_star[i], sizeof(double)*M));
                gpu_error_check(cudaMalloc((void**)&d_q_sparse[i], sizeof(double)*M));
                gpu_error_check(cudaMalloc((void**)&d_q_full_internal[i], sizeof(double)*M));
                gpu_error_check(cudaMalloc((void**)&d_q_half_internal[i], sizeof(double)*M));
                gpu_error_check(cudaMalloc((void**)&d_q_work[i], sizeof(double)*M));
                is_initialized[i] = false;
            }

            gpu_error_check(cudaMalloc((void**)&d_offset, sizeof(int)));
            gpu_error_check(cudaMemset(d_offset, 0, sizeof(int)));
        }

        // Set max dynamic shared memory size for tridiagonal solver kernels
        int max_shmem = 96000;
        cudaFuncSetAttribute(tridiagonal, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shmem);
        cudaFuncSetAttribute(tridiagonal_periodic, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shmem);

        update_laplacian_operator();
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

CudaSolverRichardsonGlobal::~CudaSolverRichardsonGlobal()
{
    const int DIM = this->dim;

    // Free nested maps: d_exp_dw[ds_index][monomer_type]
    for(const auto& ds_entry: d_exp_dw)
        for(const auto& item: ds_entry.second)
            cudaFree(item.second);
    for(const auto& ds_entry: d_exp_dw_half)
        for(const auto& item: ds_entry.second)
            cudaFree(item.second);

    // Free full step coefficients
    for(const auto& item: d_xl)
        cudaFree(item.second);
    for(const auto& item: d_xd)
        cudaFree(item.second);
    for(const auto& item: d_xh)
        cudaFree(item.second);

    for(const auto& item: d_yl)
        cudaFree(item.second);
    for(const auto& item: d_yd)
        cudaFree(item.second);
    for(const auto& item: d_yh)
        cudaFree(item.second);

    for(const auto& item: d_zl)
        cudaFree(item.second);
    for(const auto& item: d_zd)
        cudaFree(item.second);
    for(const auto& item: d_zh)
        cudaFree(item.second);

    // Free half step coefficients
    for(const auto& item: d_xl_half)
        cudaFree(item.second);
    for(const auto& item: d_xd_half)
        cudaFree(item.second);
    for(const auto& item: d_xh_half)
        cudaFree(item.second);

    for(const auto& item: d_yl_half)
        cudaFree(item.second);
    for(const auto& item: d_yd_half)
        cudaFree(item.second);
    for(const auto& item: d_yh_half)
        cudaFree(item.second);

    for(const auto& item: d_zl_half)
        cudaFree(item.second);
    for(const auto& item: d_zd_half)
        cudaFree(item.second);
    for(const auto& item: d_zh_half)
        cudaFree(item.second);

    // Free per-stream workspace
    for(int i=0; i<n_streams; i++)
    {
        cudaFree(d_q_star[i]);
        cudaFree(d_c_star[i]);
        cudaFree(d_q_sparse[i]);
        cudaFree(d_q_full_internal[i]);
        cudaFree(d_q_half_internal[i]);
        cudaFree(d_q_work[i]);
    }

    if(DIM == 3)
    {
        for(int i=0; i<n_streams; i++)
        {
            cudaFree(d_q_dstar[i]);
            cudaFree(d_temp[i]);
        }
        cudaFree(d_offset_xy);
        cudaFree(d_offset_yz);
        cudaFree(d_offset_xz);
    }
    else if(DIM == 2)
    {
        for(int i=0; i<n_streams; i++)
            cudaFree(d_temp[i]);
        cudaFree(d_offset_x);
        cudaFree(d_offset_y);
    }
    else if(DIM == 1)
    {
        cudaFree(d_offset);
    }
}

void CudaSolverRichardsonGlobal::update_laplacian_operator()
{
    try
    {
        const double ds = this->molecules->get_ds();
        const int DIM = this->cb->get_dim();
        std::vector<int> nx(DIM);
        if (DIM == 3)
            nx = {cb->get_nx(0), cb->get_nx(1), cb->get_nx(2)};
        else if (DIM == 2)
            nx = {cb->get_nx(0), cb->get_nx(1), 1};
        else if (DIM == 1)
            nx = {cb->get_nx(0), 1, 1};

        for(const auto& item: this->molecules->get_bond_lengths())
        {
            std::string monomer_type = item.first;
            double bond_length_sq = item.second * item.second;

            // Host arrays for coefficients
            double xl_host[nx[0]], xd_host[nx[0]], xh_host[nx[0]];
            double yl_host[nx[1]], yd_host[nx[1]], yh_host[nx[1]];
            double zl_host[nx[2]], zd_host[nx[2]], zh_host[nx[2]];

            double xl_half_host[nx[0]], xd_half_host[nx[0]], xh_half_host[nx[0]];
            double yl_half_host[nx[1]], yd_half_host[nx[1]], yh_half_host[nx[1]];
            double zl_half_host[nx[2]], zd_half_host[nx[2]], zh_half_host[nx[2]];

            // Full step coefficients (ds)
            FiniteDifference::get_laplacian_matrix(
                this->cb->get_boundary_conditions(),
                this->cb->get_nx(), this->cb->get_dx(),
                xl_host, xd_host, xh_host,
                yl_host, yd_host, yh_host,
                zl_host, zd_host, zh_host,
                bond_length_sq, ds);

            // Half step coefficients (ds/2)
            FiniteDifference::get_laplacian_matrix(
                this->cb->get_boundary_conditions(),
                this->cb->get_nx(), this->cb->get_dx(),
                xl_half_host, xd_half_host, xh_half_host,
                yl_half_host, yd_half_host, yh_half_host,
                zl_half_host, zd_half_host, zh_half_host,
                bond_length_sq, ds * 0.5);

            // Copy to device - full step
            gpu_error_check(cudaMemcpy(d_xl[monomer_type], xl_host, sizeof(double)*nx[0], cudaMemcpyHostToDevice));
            gpu_error_check(cudaMemcpy(d_xd[monomer_type], xd_host, sizeof(double)*nx[0], cudaMemcpyHostToDevice));
            gpu_error_check(cudaMemcpy(d_xh[monomer_type], xh_host, sizeof(double)*nx[0], cudaMemcpyHostToDevice));

            gpu_error_check(cudaMemcpy(d_yl[monomer_type], yl_host, sizeof(double)*nx[1], cudaMemcpyHostToDevice));
            gpu_error_check(cudaMemcpy(d_yd[monomer_type], yd_host, sizeof(double)*nx[1], cudaMemcpyHostToDevice));
            gpu_error_check(cudaMemcpy(d_yh[monomer_type], yh_host, sizeof(double)*nx[1], cudaMemcpyHostToDevice));

            gpu_error_check(cudaMemcpy(d_zl[monomer_type], zl_host, sizeof(double)*nx[2], cudaMemcpyHostToDevice));
            gpu_error_check(cudaMemcpy(d_zd[monomer_type], zd_host, sizeof(double)*nx[2], cudaMemcpyHostToDevice));
            gpu_error_check(cudaMemcpy(d_zh[monomer_type], zh_host, sizeof(double)*nx[2], cudaMemcpyHostToDevice));

            // Copy to device - half step
            gpu_error_check(cudaMemcpy(d_xl_half[monomer_type], xl_half_host, sizeof(double)*nx[0], cudaMemcpyHostToDevice));
            gpu_error_check(cudaMemcpy(d_xd_half[monomer_type], xd_half_host, sizeof(double)*nx[0], cudaMemcpyHostToDevice));
            gpu_error_check(cudaMemcpy(d_xh_half[monomer_type], xh_half_host, sizeof(double)*nx[0], cudaMemcpyHostToDevice));

            gpu_error_check(cudaMemcpy(d_yl_half[monomer_type], yl_half_host, sizeof(double)*nx[1], cudaMemcpyHostToDevice));
            gpu_error_check(cudaMemcpy(d_yd_half[monomer_type], yd_half_host, sizeof(double)*nx[1], cudaMemcpyHostToDevice));
            gpu_error_check(cudaMemcpy(d_yh_half[monomer_type], yh_half_host, sizeof(double)*nx[1], cudaMemcpyHostToDevice));

            gpu_error_check(cudaMemcpy(d_zl_half[monomer_type], zl_half_host, sizeof(double)*nx[2], cudaMemcpyHostToDevice));
            gpu_error_check(cudaMemcpy(d_zd_half[monomer_type], zd_half_host, sizeof(double)*nx[2], cudaMemcpyHostToDevice));
            gpu_error_check(cudaMemcpy(d_zh_half[monomer_type], zh_half_host, sizeof(double)*nx[2], cudaMemcpyHostToDevice));
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

void CudaSolverRichardsonGlobal::update_dw(
    std::string device, std::map<std::string, const double*> w_input)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = this->cb->get_total_grid();
        const double ds = this->molecules->get_ds();

        // Global Richardson uses only global ds (ds_index=1)
        const int ds_idx = 1;

        cudaMemcpyKind cudaMemcpyInputToDevice;
        if (device == "gpu")
            cudaMemcpyInputToDevice = cudaMemcpyDeviceToDevice;
        else if(device == "cpu")
            cudaMemcpyInputToDevice = cudaMemcpyHostToDevice;
        else
        {
            throw_with_line_number("Invalid device \"" + device + "\".");
        }

        for(const auto& item: w_input)
        {
            std::string monomer_type = item.first;
            const double* w = item.second;

            if (d_exp_dw[ds_idx].find(monomer_type) == d_exp_dw[ds_idx].end())
                throw_with_line_number("monomer_type \"" + monomer_type + "\" is not in d_exp_dw[" + std::to_string(ds_idx) + "].");

            // Copy field configurations and compute Boltzmann factors
            gpu_error_check(cudaMemcpyAsync(
                d_exp_dw     [ds_idx][monomer_type], w,
                sizeof(double)*M, cudaMemcpyInputToDevice));
            gpu_error_check(cudaMemcpyAsync(
                d_exp_dw_half[ds_idx][monomer_type], w,
                sizeof(double)*M, cudaMemcpyInputToDevice));

            // Compute d_exp_dw = exp(-w*ds/2) for full step
            ker_exp<<<N_BLOCKS, N_THREADS>>>
                ((double*) d_exp_dw[ds_idx][monomer_type],
                 (double*) d_exp_dw[ds_idx][monomer_type], 1.0, -0.5*ds, M);

            // Compute d_exp_dw_half = exp(-w*ds/4) for half step
            ker_exp<<<N_BLOCKS, N_THREADS>>>
                ((double*) d_exp_dw_half[ds_idx][monomer_type],
                 (double*) d_exp_dw_half[ds_idx][monomer_type], 1.0, -0.25*ds, M);
        }
        gpu_error_check(cudaDeviceSynchronize());
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

void CudaSolverRichardsonGlobal::reset_internal_state(int STREAM)
{
    is_initialized[STREAM] = false;
}

void CudaSolverRichardsonGlobal::advance_full_step(
    const int STREAM, double* d_q_in, double* d_q_out, std::string monomer_type)
{
    const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();
    const int M = this->cb->get_total_grid();
    const int DIM = this->dim;

    const int ds_idx = 1;
    double *_d_exp_dw = d_exp_dw[ds_idx][monomer_type];

    // Apply exp(-w*ds/2) at start
    ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_q_out, d_q_in, _d_exp_dw, 1.0, M);
    gpu_error_check(cudaPeekAtLastError());

    // Diffusion with full-step coefficients
    if(DIM == 3)
        advance_propagator_3d_step(this->cb->get_boundary_conditions(), STREAM,
            d_q_out, d_q_out,
            d_xl[monomer_type], d_xd[monomer_type], d_xh[monomer_type],
            d_yl[monomer_type], d_yd[monomer_type], d_yh[monomer_type],
            d_zl[monomer_type], d_zd[monomer_type], d_zh[monomer_type]);
    else if(DIM == 2)
        advance_propagator_2d_step(this->cb->get_boundary_conditions(), STREAM,
            d_q_out, d_q_out,
            d_xl[monomer_type], d_xd[monomer_type], d_xh[monomer_type],
            d_yl[monomer_type], d_yd[monomer_type], d_yh[monomer_type]);
    else if(DIM == 1)
        advance_propagator_1d_step(this->cb->get_boundary_conditions(), STREAM,
            d_q_out, d_q_out,
            d_xl[monomer_type], d_xd[monomer_type], d_xh[monomer_type]);

    // Apply exp(-w*ds/2) at end
    ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_q_out, d_q_out, _d_exp_dw, 1.0, M);
    gpu_error_check(cudaPeekAtLastError());
}

void CudaSolverRichardsonGlobal::advance_two_half_steps(
    const int STREAM, double* d_q_in, double* d_q_out, std::string monomer_type)
{
    const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();
    const int M = this->cb->get_total_grid();
    const int DIM = this->dim;

    const int ds_idx = 1;
    double *_d_exp_dw = d_exp_dw[ds_idx][monomer_type];
    double *_d_exp_dw_half = d_exp_dw_half[ds_idx][monomer_type];

    // First half-step: exp(-w*ds/4) at start
    ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_q_out, d_q_in, _d_exp_dw_half, 1.0, M);
    gpu_error_check(cudaPeekAtLastError());

    // Diffusion with half-step coefficients
    if(DIM == 3)
        advance_propagator_3d_step(this->cb->get_boundary_conditions(), STREAM,
            d_q_out, d_q_out,
            d_xl_half[monomer_type], d_xd_half[monomer_type], d_xh_half[monomer_type],
            d_yl_half[monomer_type], d_yd_half[monomer_type], d_yh_half[monomer_type],
            d_zl_half[monomer_type], d_zd_half[monomer_type], d_zh_half[monomer_type]);
    else if(DIM == 2)
        advance_propagator_2d_step(this->cb->get_boundary_conditions(), STREAM,
            d_q_out, d_q_out,
            d_xl_half[monomer_type], d_xd_half[monomer_type], d_xh_half[monomer_type],
            d_yl_half[monomer_type], d_yd_half[monomer_type], d_yh_half[monomer_type]);
    else if(DIM == 1)
        advance_propagator_1d_step(this->cb->get_boundary_conditions(), STREAM,
            d_q_out, d_q_out,
            d_xl_half[monomer_type], d_xd_half[monomer_type], d_xh_half[monomer_type]);

    // Apply exp(-w*ds/2) at junction
    ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_q_out, d_q_out, _d_exp_dw, 1.0, M);
    gpu_error_check(cudaPeekAtLastError());

    // Second half-step: Diffusion with half-step coefficients
    if(DIM == 3)
        advance_propagator_3d_step(this->cb->get_boundary_conditions(), STREAM,
            d_q_out, d_q_out,
            d_xl_half[monomer_type], d_xd_half[monomer_type], d_xh_half[monomer_type],
            d_yl_half[monomer_type], d_yd_half[monomer_type], d_yh_half[monomer_type],
            d_zl_half[monomer_type], d_zd_half[monomer_type], d_zh_half[monomer_type]);
    else if(DIM == 2)
        advance_propagator_2d_step(this->cb->get_boundary_conditions(), STREAM,
            d_q_out, d_q_out,
            d_xl_half[monomer_type], d_xd_half[monomer_type], d_xh_half[monomer_type],
            d_yl_half[monomer_type], d_yd_half[monomer_type], d_yh_half[monomer_type]);
    else if(DIM == 1)
        advance_propagator_1d_step(this->cb->get_boundary_conditions(), STREAM,
            d_q_out, d_q_out,
            d_xl_half[monomer_type], d_xd_half[monomer_type], d_xh_half[monomer_type]);

    // Apply exp(-w*ds/4) at end
    ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_q_out, d_q_out, _d_exp_dw_half, 1.0, M);
    gpu_error_check(cudaPeekAtLastError());
}

void CudaSolverRichardsonGlobal::advance_propagator(
    const int STREAM,
    double *d_q_in, double *d_q_out,
    std::string monomer_type, double *d_q_mask, [[maybe_unused]] int ds_index)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();
        const int M = this->cb->get_total_grid();

        // Initialize internal states from d_q_in on first call after reset
        if (!is_initialized[STREAM])
        {
            gpu_error_check(cudaMemcpyAsync(d_q_full_internal[STREAM], d_q_in,
                            sizeof(double)*M, cudaMemcpyDeviceToDevice, streams[STREAM][0]));
            gpu_error_check(cudaMemcpyAsync(d_q_half_internal[STREAM], d_q_in,
                            sizeof(double)*M, cudaMemcpyDeviceToDevice, streams[STREAM][0]));
            is_initialized[STREAM] = true;
        }

        // Advance q_full_internal by one full step (using its own state)
        advance_full_step(STREAM, d_q_full_internal[STREAM], d_q_work[STREAM], monomer_type);
        gpu_error_check(cudaMemcpyAsync(d_q_full_internal[STREAM], d_q_work[STREAM],
                        sizeof(double)*M, cudaMemcpyDeviceToDevice, streams[STREAM][0]));

        // Advance q_half_internal by two half steps (using its own state)
        advance_two_half_steps(STREAM, d_q_half_internal[STREAM], d_q_work[STREAM], monomer_type);
        gpu_error_check(cudaMemcpyAsync(d_q_half_internal[STREAM], d_q_work[STREAM],
                        sizeof(double)*M, cudaMemcpyDeviceToDevice, streams[STREAM][0]));

        // Richardson extrapolation: q_out = (4*q_half - q_full) / 3
        ker_lin_comb<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
            d_q_out, 4.0/3.0, d_q_half_internal[STREAM], -1.0/3.0, d_q_full_internal[STREAM], M);
        gpu_error_check(cudaPeekAtLastError());

        // Multiply mask
        if (d_q_mask != nullptr)
        {
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_q_out, d_q_out, d_q_mask, 1.0, M);
            gpu_error_check(cudaPeekAtLastError());
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

// ============================================================================
// ADI step methods (reuse kernels from CudaSolverCNADI)
// ============================================================================

void CudaSolverRichardsonGlobal::advance_propagator_3d_step(
    std::vector<BoundaryCondition> bc,
    const int STREAM,
    double *d_q_in, double *d_q_out,
    double *_d_xl, double *_d_xd, double *_d_xh,
    double *_d_yl, double *_d_yd, double *_d_yh,
    double *_d_zl, double *_d_zd, double *_d_zh)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();
        const int M = this->cb->get_total_grid();
        const std::vector<int> nx = this->cb->get_nx();

        // Shared memory size for 3 arrays of doubles
        size_t shmem_x = 3 * nx[0] * sizeof(double);
        size_t shmem_y = 3 * nx[1] * sizeof(double);
        size_t shmem_z = 3 * nx[2] * sizeof(double);

        // Calculate q_star
        compute_crank_3d_step_1<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
            bc[0], bc[1], bc[2], bc[3], bc[4], bc[5],
            _d_xl, _d_xd, _d_xh, nx[0],
            _d_yl, _d_yd, _d_yh, nx[1],
            _d_zl, _d_zd, _d_zh, nx[2],
            d_temp[STREAM], d_q_in, M);
        gpu_error_check(cudaPeekAtLastError());

        if (bc[0] == BoundaryCondition::PERIODIC)
        {
            tridiagonal_periodic<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                _d_xl, _d_xd, _d_xh,
                d_c_star[STREAM], d_q_sparse[STREAM], d_temp[STREAM], d_q_star[STREAM],
                d_offset_yz, nx[1]*nx[2], nx[1]*nx[2], nx[0]);
        }
        else
        {
            tridiagonal<<<N_BLOCKS, N_THREADS, shmem_x, streams[STREAM][0]>>>(
                _d_xl, _d_xd, _d_xh,
                d_c_star[STREAM], d_temp[STREAM], d_q_star[STREAM],
                d_offset_yz, nx[1]*nx[2], nx[1]*nx[2], nx[0]);
        }
        gpu_error_check(cudaPeekAtLastError());

        // Calculate q_dstar
        compute_crank_3d_step_2<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
            bc[2], bc[3],
            _d_yl, _d_yd, _d_yh, nx[1], nx[2],
            d_temp[STREAM], d_q_star[STREAM], d_q_in, M);
        gpu_error_check(cudaPeekAtLastError());

        if (bc[2] == BoundaryCondition::PERIODIC)
        {
            tridiagonal_periodic<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                _d_yl, _d_yd, _d_yh,
                d_c_star[STREAM], d_q_sparse[STREAM], d_temp[STREAM], d_q_dstar[STREAM],
                d_offset_xz, nx[0]*nx[2], nx[2], nx[1]);
        }
        else
        {
            tridiagonal<<<N_BLOCKS, N_THREADS, shmem_y, streams[STREAM][0]>>>(
                _d_yl, _d_yd, _d_yh,
                d_c_star[STREAM], d_temp[STREAM], d_q_dstar[STREAM],
                d_offset_xz, nx[0]*nx[2], nx[2], nx[1]);
        }
        gpu_error_check(cudaPeekAtLastError());

        // Calculate q^(n+1)
        compute_crank_3d_step_3<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
            bc[4], bc[5],
            _d_zl, _d_zd, _d_zh, nx[1], nx[2],
            d_temp[STREAM], d_q_dstar[STREAM], d_q_in, M);
        gpu_error_check(cudaPeekAtLastError());

        if (bc[4] == BoundaryCondition::PERIODIC)
        {
            tridiagonal_periodic<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                _d_zl, _d_zd, _d_zh,
                d_c_star[STREAM], d_q_sparse[STREAM], d_temp[STREAM], d_q_out,
                d_offset_xy, nx[0]*nx[1], 1, nx[2]);
        }
        else
        {
            tridiagonal<<<N_BLOCKS, N_THREADS, shmem_z, streams[STREAM][0]>>>(
                _d_zl, _d_zd, _d_zh,
                d_c_star[STREAM], d_temp[STREAM], d_q_out,
                d_offset_xy, nx[0]*nx[1], 1, nx[2]);
        }
        gpu_error_check(cudaPeekAtLastError());
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

void CudaSolverRichardsonGlobal::advance_propagator_2d_step(
    std::vector<BoundaryCondition> bc,
    const int STREAM,
    double *d_q_in, double *d_q_out,
    double *_d_xl, double *_d_xd, double *_d_xh,
    double *_d_yl, double *_d_yd, double *_d_yh)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();
        const int M = this->cb->get_total_grid();
        const std::vector<int> nx = this->cb->get_nx();

        size_t shmem_x = 3 * nx[0] * sizeof(double);
        size_t shmem_y = 3 * nx[1] * sizeof(double);

        // Calculate q_star
        compute_crank_2d_step_1<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
            bc[0], bc[1], bc[2], bc[3],
            _d_xl, _d_xd, _d_xh, nx[0],
            _d_yl, _d_yd, _d_yh, nx[1],
            d_temp[STREAM], d_q_in, M);
        gpu_error_check(cudaPeekAtLastError());

        if (bc[0] == BoundaryCondition::PERIODIC)
        {
            tridiagonal_periodic<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                _d_xl, _d_xd, _d_xh,
                d_c_star[STREAM], d_q_sparse[STREAM], d_temp[STREAM], d_q_star[STREAM],
                d_offset_y, nx[1], nx[1], nx[0]);
        }
        else
        {
            tridiagonal<<<N_BLOCKS, N_THREADS, shmem_x, streams[STREAM][0]>>>(
                _d_xl, _d_xd, _d_xh,
                d_c_star[STREAM], d_temp[STREAM], d_q_star[STREAM],
                d_offset_y, nx[1], nx[1], nx[0]);
        }
        gpu_error_check(cudaPeekAtLastError());

        // Calculate q^(n+1)
        compute_crank_2d_step_2<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
            bc[2], bc[3],
            _d_yl, _d_yd, _d_yh, nx[1],
            d_temp[STREAM], d_q_star[STREAM], d_q_in, M);
        gpu_error_check(cudaPeekAtLastError());

        if (bc[2] == BoundaryCondition::PERIODIC)
        {
            tridiagonal_periodic<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                _d_yl, _d_yd, _d_yh,
                d_c_star[STREAM], d_q_sparse[STREAM], d_temp[STREAM], d_q_out,
                d_offset_x, nx[0], 1, nx[1]);
        }
        else
        {
            tridiagonal<<<N_BLOCKS, N_THREADS, shmem_y, streams[STREAM][0]>>>(
                _d_yl, _d_yd, _d_yh,
                d_c_star[STREAM], d_temp[STREAM], d_q_out,
                d_offset_x, nx[0], 1, nx[1]);
        }
        gpu_error_check(cudaPeekAtLastError());
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

void CudaSolverRichardsonGlobal::advance_propagator_1d_step(
    std::vector<BoundaryCondition> bc,
    const int STREAM,
    double *d_q_in, double *d_q_out,
    double *_d_xl, double *_d_xd, double *_d_xh)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();
        const int M = this->cb->get_total_grid();

        size_t shmem_x = 3 * M * sizeof(double);

        // Calculate q_star
        compute_crank_1d<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
            bc[0], bc[1],
            _d_xl, _d_xd, _d_xh,
            d_q_star[STREAM], d_q_in, M);
        gpu_error_check(cudaPeekAtLastError());

        if (bc[0] == BoundaryCondition::PERIODIC)
        {
            tridiagonal_periodic<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                _d_xl, _d_xd, _d_xh,
                d_c_star[STREAM], d_q_sparse[STREAM], d_q_star[STREAM], d_q_out,
                d_offset, 1, 1, M);
        }
        else
        {
            tridiagonal<<<N_BLOCKS, N_THREADS, shmem_x, streams[STREAM][0]>>>(
                _d_xl, _d_xd, _d_xh,
                d_c_star[STREAM], d_q_star[STREAM], d_q_out,
                d_offset, 1, 1, M);
        }
        gpu_error_check(cudaPeekAtLastError());
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

void CudaSolverRichardsonGlobal::compute_single_segment_stress(
    [[maybe_unused]] const int STREAM,
    [[maybe_unused]] double *d_q_pair, [[maybe_unused]] double *d_segment_stress,
    [[maybe_unused]] std::string monomer_type, [[maybe_unused]] bool is_half_bond_length)
{
    try
    {
        throw_with_line_number("Global Richardson method does not support stress computation.");
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
