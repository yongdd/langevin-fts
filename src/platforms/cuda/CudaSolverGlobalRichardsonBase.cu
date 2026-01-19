/**
 * @file CudaSolverGlobalRichardsonBase.cu
 * @brief Implementation of base CN-ADI2 solver for Global Richardson on GPU.
 *
 * Provides stateless CN-ADI2 propagator advancement with full and half step
 * support. This solver is used by CudaComputationGlobalRichardson which manages
 * two independent propagator chains and applies Richardson at the quadrature level.
 */

#include <iostream>
#include <cmath>
#include "CudaSolverGlobalRichardsonBase.h"
#include "CudaSolverCNADI.h"  // For tridiagonal kernel declarations

CudaSolverGlobalRichardsonBase::CudaSolverGlobalRichardsonBase(
    ComputationBox<double>* cb,
    Molecules* molecules,
    int n_streams,
    cudaStream_t streams[MAX_STREAMS][2])
{
    try
    {
        this->cb = cb;
        this->molecules = molecules;
        this->n_streams = n_streams;

        if (molecules->get_model_name() != "continuous")
            throw_with_line_number("Global Richardson method only supports 'continuous' chain model.");

        if (!cb->is_orthogonal())
            throw_with_line_number("Global Richardson method only supports orthogonal unit cells.");

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

        // Copy streams (nullptr allowed for reduce-memory mode)
        if (streams != nullptr)
        {
            for (int i = 0; i < n_streams; i++)
            {
                this->streams[i][0] = streams[i][0];
                this->streams[i][1] = streams[i][1];
            }
        }

        // Allocate coefficient arrays for each monomer type
        for (const auto& item : molecules->get_bond_lengths())
        {
            std::string monomer_type = item.first;

            // Boltzmann factors
            d_exp_dw_full[monomer_type] = nullptr;
            d_exp_dw_half[monomer_type] = nullptr;
            gpu_error_check(cudaMalloc((void**)&d_exp_dw_full[monomer_type], sizeof(double) * M));
            gpu_error_check(cudaMalloc((void**)&d_exp_dw_half[monomer_type], sizeof(double) * M));

            // Full step tridiagonal coefficients
            d_xl_full[monomer_type] = nullptr;
            d_xd_full[monomer_type] = nullptr;
            d_xh_full[monomer_type] = nullptr;
            gpu_error_check(cudaMalloc((void**)&d_xl_full[monomer_type], sizeof(double) * nx[0]));
            gpu_error_check(cudaMalloc((void**)&d_xd_full[monomer_type], sizeof(double) * nx[0]));
            gpu_error_check(cudaMalloc((void**)&d_xh_full[monomer_type], sizeof(double) * nx[0]));

            d_yl_full[monomer_type] = nullptr;
            d_yd_full[monomer_type] = nullptr;
            d_yh_full[monomer_type] = nullptr;
            gpu_error_check(cudaMalloc((void**)&d_yl_full[monomer_type], sizeof(double) * nx[1]));
            gpu_error_check(cudaMalloc((void**)&d_yd_full[monomer_type], sizeof(double) * nx[1]));
            gpu_error_check(cudaMalloc((void**)&d_yh_full[monomer_type], sizeof(double) * nx[1]));

            d_zl_full[monomer_type] = nullptr;
            d_zd_full[monomer_type] = nullptr;
            d_zh_full[monomer_type] = nullptr;
            gpu_error_check(cudaMalloc((void**)&d_zl_full[monomer_type], sizeof(double) * nx[2]));
            gpu_error_check(cudaMalloc((void**)&d_zd_full[monomer_type], sizeof(double) * nx[2]));
            gpu_error_check(cudaMalloc((void**)&d_zh_full[monomer_type], sizeof(double) * nx[2]));

            // Half step tridiagonal coefficients
            d_xl_half[monomer_type] = nullptr;
            d_xd_half[monomer_type] = nullptr;
            d_xh_half[monomer_type] = nullptr;
            gpu_error_check(cudaMalloc((void**)&d_xl_half[monomer_type], sizeof(double) * nx[0]));
            gpu_error_check(cudaMalloc((void**)&d_xd_half[monomer_type], sizeof(double) * nx[0]));
            gpu_error_check(cudaMalloc((void**)&d_xh_half[monomer_type], sizeof(double) * nx[0]));

            d_yl_half[monomer_type] = nullptr;
            d_yd_half[monomer_type] = nullptr;
            d_yh_half[monomer_type] = nullptr;
            gpu_error_check(cudaMalloc((void**)&d_yl_half[monomer_type], sizeof(double) * nx[1]));
            gpu_error_check(cudaMalloc((void**)&d_yd_half[monomer_type], sizeof(double) * nx[1]));
            gpu_error_check(cudaMalloc((void**)&d_yh_half[monomer_type], sizeof(double) * nx[1]));

            d_zl_half[monomer_type] = nullptr;
            d_zd_half[monomer_type] = nullptr;
            d_zh_half[monomer_type] = nullptr;
            gpu_error_check(cudaMalloc((void**)&d_zl_half[monomer_type], sizeof(double) * nx[2]));
            gpu_error_check(cudaMalloc((void**)&d_zd_half[monomer_type], sizeof(double) * nx[2]));
            gpu_error_check(cudaMalloc((void**)&d_zh_half[monomer_type], sizeof(double) * nx[2]));
        }

        // Allocate per-stream workspace arrays
        if (DIM == 3)
        {
            for (int i = 0; i < n_streams; i++)
            {
                gpu_error_check(cudaMalloc((void**)&d_q_star[i], sizeof(double) * M));
                gpu_error_check(cudaMalloc((void**)&d_q_dstar[i], sizeof(double) * M));
                gpu_error_check(cudaMalloc((void**)&d_c_star[i], sizeof(double) * M));
                gpu_error_check(cudaMalloc((void**)&d_q_sparse[i], sizeof(double) * M));
                gpu_error_check(cudaMalloc((void**)&d_temp[i], sizeof(double) * M));
            }

            int offset_xy[nx[0] * nx[1]];
            int offset_yz[nx[1] * nx[2]];
            int offset_xz[nx[0] * nx[2]];
            int count;

            count = 0;
            for (int i = 0; i < nx[0]; i++)
                for (int j = 0; j < nx[1]; j++)
                    offset_xy[count++] = i * nx[1] * nx[2] + j * nx[2];

            count = 0;
            for (int j = 0; j < nx[1]; j++)
                for (int k = 0; k < nx[2]; k++)
                    offset_yz[count++] = j * nx[2] + k;

            count = 0;
            for (int i = 0; i < nx[0]; i++)
                for (int k = 0; k < nx[2]; k++)
                    offset_xz[count++] = i * nx[1] * nx[2] + k;

            gpu_error_check(cudaMalloc((void**)&d_offset_xy, sizeof(int) * nx[0] * nx[1]));
            gpu_error_check(cudaMalloc((void**)&d_offset_yz, sizeof(int) * nx[1] * nx[2]));
            gpu_error_check(cudaMalloc((void**)&d_offset_xz, sizeof(int) * nx[0] * nx[2]));

            gpu_error_check(cudaMemcpy(d_offset_xy, offset_xy, sizeof(int) * nx[0] * nx[1], cudaMemcpyHostToDevice));
            gpu_error_check(cudaMemcpy(d_offset_yz, offset_yz, sizeof(int) * nx[1] * nx[2], cudaMemcpyHostToDevice));
            gpu_error_check(cudaMemcpy(d_offset_xz, offset_xz, sizeof(int) * nx[0] * nx[2], cudaMemcpyHostToDevice));
        }
        else if (DIM == 2)
        {
            for (int i = 0; i < n_streams; i++)
            {
                gpu_error_check(cudaMalloc((void**)&d_q_star[i], sizeof(double) * M));
                gpu_error_check(cudaMalloc((void**)&d_c_star[i], sizeof(double) * M));
                gpu_error_check(cudaMalloc((void**)&d_q_sparse[i], sizeof(double) * M));
                gpu_error_check(cudaMalloc((void**)&d_temp[i], sizeof(double) * M));
            }

            int offset_x[nx[0]];
            int offset_y[nx[1]];

            for (int i = 0; i < nx[0]; i++)
                offset_x[i] = i * nx[1];

            for (int j = 0; j < nx[1]; j++)
                offset_y[j] = j;

            gpu_error_check(cudaMalloc((void**)&d_offset_x, sizeof(int) * nx[0]));
            gpu_error_check(cudaMalloc((void**)&d_offset_y, sizeof(int) * nx[1]));

            gpu_error_check(cudaMemcpy(d_offset_x, offset_x, sizeof(int) * nx[0], cudaMemcpyHostToDevice));
            gpu_error_check(cudaMemcpy(d_offset_y, offset_y, sizeof(int) * nx[1], cudaMemcpyHostToDevice));
        }
        else if (DIM == 1)
        {
            for (int i = 0; i < n_streams; i++)
            {
                gpu_error_check(cudaMalloc((void**)&d_c_star[i], sizeof(double) * M));
                gpu_error_check(cudaMalloc((void**)&d_q_sparse[i], sizeof(double) * M));
                gpu_error_check(cudaMalloc((void**)&d_temp[i], sizeof(double) * M));
            }

            // For 1D case, there is only 1 system starting at offset 0
            int offset[1] = {0};
            gpu_error_check(cudaMalloc((void**)&d_offset, sizeof(int) * 1));
            gpu_error_check(cudaMemcpy(d_offset, offset, sizeof(int) * 1, cudaMemcpyHostToDevice));
        }

        update_laplacian_operator();
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

CudaSolverGlobalRichardsonBase::~CudaSolverGlobalRichardsonBase()
{
    const int DIM = this->dim;

    // Free Boltzmann factors
    for (const auto& item : d_exp_dw_full)
        cudaFree(item.second);
    for (const auto& item : d_exp_dw_half)
        cudaFree(item.second);

    // Free full step coefficients
    for (const auto& item : d_xl_full) cudaFree(item.second);
    for (const auto& item : d_xd_full) cudaFree(item.second);
    for (const auto& item : d_xh_full) cudaFree(item.second);
    for (const auto& item : d_yl_full) cudaFree(item.second);
    for (const auto& item : d_yd_full) cudaFree(item.second);
    for (const auto& item : d_yh_full) cudaFree(item.second);
    for (const auto& item : d_zl_full) cudaFree(item.second);
    for (const auto& item : d_zd_full) cudaFree(item.second);
    for (const auto& item : d_zh_full) cudaFree(item.second);

    // Free half step coefficients
    for (const auto& item : d_xl_half) cudaFree(item.second);
    for (const auto& item : d_xd_half) cudaFree(item.second);
    for (const auto& item : d_xh_half) cudaFree(item.second);
    for (const auto& item : d_yl_half) cudaFree(item.second);
    for (const auto& item : d_yd_half) cudaFree(item.second);
    for (const auto& item : d_yh_half) cudaFree(item.second);
    for (const auto& item : d_zl_half) cudaFree(item.second);
    for (const auto& item : d_zd_half) cudaFree(item.second);
    for (const auto& item : d_zh_half) cudaFree(item.second);

    // Free workspace arrays
    if (DIM == 3)
    {
        for (int i = 0; i < n_streams; i++)
        {
            cudaFree(d_q_star[i]);
            cudaFree(d_q_dstar[i]);
            cudaFree(d_c_star[i]);
            cudaFree(d_q_sparse[i]);
            cudaFree(d_temp[i]);
        }
        cudaFree(d_offset_xy);
        cudaFree(d_offset_yz);
        cudaFree(d_offset_xz);
    }
    else if (DIM == 2)
    {
        for (int i = 0; i < n_streams; i++)
        {
            cudaFree(d_q_star[i]);
            cudaFree(d_c_star[i]);
            cudaFree(d_q_sparse[i]);
            cudaFree(d_temp[i]);
        }
        cudaFree(d_offset_x);
        cudaFree(d_offset_y);
    }
    else if (DIM == 1)
    {
        for (int i = 0; i < n_streams; i++)
        {
            cudaFree(d_c_star[i]);
            cudaFree(d_q_sparse[i]);
            cudaFree(d_temp[i]);
        }
        cudaFree(d_offset);
    }
}

void CudaSolverGlobalRichardsonBase::update_laplacian_operator()
{
    try
    {
        const double ds = this->molecules->get_ds();
        std::vector<int> nx = this->cb->get_nx();

        for (const auto& item : this->molecules->get_bond_lengths())
        {
            std::string monomer_type = item.first;
            double bond_length_sq = item.second * item.second;

            // Host arrays for coefficient computation
            int max_nx = std::max({nx[0], (int)(nx.size() > 1 ? nx[1] : 1), (int)(nx.size() > 2 ? nx[2] : 1)});
            std::vector<double> xl(max_nx), xd(max_nx), xh(max_nx);
            std::vector<double> yl(max_nx), yd(max_nx), yh(max_nx);
            std::vector<double> zl(max_nx), zd(max_nx), zh(max_nx);

            // Full step coefficients (ds)
            FiniteDifference::get_laplacian_matrix(
                this->cb->get_boundary_conditions(),
                this->cb->get_nx(), this->cb->get_dx(),
                xl.data(), xd.data(), xh.data(),
                yl.data(), yd.data(), yh.data(),
                zl.data(), zd.data(), zh.data(),
                bond_length_sq, ds);

            gpu_error_check(cudaMemcpy(d_xl_full[monomer_type], xl.data(), sizeof(double) * nx[0], cudaMemcpyHostToDevice));
            gpu_error_check(cudaMemcpy(d_xd_full[monomer_type], xd.data(), sizeof(double) * nx[0], cudaMemcpyHostToDevice));
            gpu_error_check(cudaMemcpy(d_xh_full[monomer_type], xh.data(), sizeof(double) * nx[0], cudaMemcpyHostToDevice));

            if (nx.size() > 1)
            {
                gpu_error_check(cudaMemcpy(d_yl_full[monomer_type], yl.data(), sizeof(double) * nx[1], cudaMemcpyHostToDevice));
                gpu_error_check(cudaMemcpy(d_yd_full[monomer_type], yd.data(), sizeof(double) * nx[1], cudaMemcpyHostToDevice));
                gpu_error_check(cudaMemcpy(d_yh_full[monomer_type], yh.data(), sizeof(double) * nx[1], cudaMemcpyHostToDevice));
            }
            if (nx.size() > 2)
            {
                gpu_error_check(cudaMemcpy(d_zl_full[monomer_type], zl.data(), sizeof(double) * nx[2], cudaMemcpyHostToDevice));
                gpu_error_check(cudaMemcpy(d_zd_full[monomer_type], zd.data(), sizeof(double) * nx[2], cudaMemcpyHostToDevice));
                gpu_error_check(cudaMemcpy(d_zh_full[monomer_type], zh.data(), sizeof(double) * nx[2], cudaMemcpyHostToDevice));
            }

            // Half step coefficients (ds/2)
            FiniteDifference::get_laplacian_matrix(
                this->cb->get_boundary_conditions(),
                this->cb->get_nx(), this->cb->get_dx(),
                xl.data(), xd.data(), xh.data(),
                yl.data(), yd.data(), yh.data(),
                zl.data(), zd.data(), zh.data(),
                bond_length_sq, ds * 0.5);

            gpu_error_check(cudaMemcpy(d_xl_half[monomer_type], xl.data(), sizeof(double) * nx[0], cudaMemcpyHostToDevice));
            gpu_error_check(cudaMemcpy(d_xd_half[monomer_type], xd.data(), sizeof(double) * nx[0], cudaMemcpyHostToDevice));
            gpu_error_check(cudaMemcpy(d_xh_half[monomer_type], xh.data(), sizeof(double) * nx[0], cudaMemcpyHostToDevice));

            if (nx.size() > 1)
            {
                gpu_error_check(cudaMemcpy(d_yl_half[monomer_type], yl.data(), sizeof(double) * nx[1], cudaMemcpyHostToDevice));
                gpu_error_check(cudaMemcpy(d_yd_half[monomer_type], yd.data(), sizeof(double) * nx[1], cudaMemcpyHostToDevice));
                gpu_error_check(cudaMemcpy(d_yh_half[monomer_type], yh.data(), sizeof(double) * nx[1], cudaMemcpyHostToDevice));
            }
            if (nx.size() > 2)
            {
                gpu_error_check(cudaMemcpy(d_zl_half[monomer_type], zl.data(), sizeof(double) * nx[2], cudaMemcpyHostToDevice));
                gpu_error_check(cudaMemcpy(d_zd_half[monomer_type], zd.data(), sizeof(double) * nx[2], cudaMemcpyHostToDevice));
                gpu_error_check(cudaMemcpy(d_zh_half[monomer_type], zh.data(), sizeof(double) * nx[2], cudaMemcpyHostToDevice));
            }
        }
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

void CudaSolverGlobalRichardsonBase::update_dw(
    std::string device, std::map<std::string, const double*> w_input)
{
    try
    {
        const int N_BLOCKS = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();
        const int M = this->cb->get_total_grid();
        const double ds = this->molecules->get_ds();

        cudaMemcpyKind cudaMemcpyInputToDevice;
        if (device == "gpu")
            cudaMemcpyInputToDevice = cudaMemcpyDeviceToDevice;
        else if (device == "cpu")
            cudaMemcpyInputToDevice = cudaMemcpyHostToDevice;
        else
            throw_with_line_number("Invalid device \"" + device + "\".");

        for (const auto& item : w_input)
        {
            std::string monomer_type = item.first;
            const double* w = item.second;

            if (d_exp_dw_full.find(monomer_type) == d_exp_dw_full.end())
                throw_with_line_number("monomer_type \"" + monomer_type + "\" is not in d_exp_dw_full.");

            // Copy field and compute Boltzmann factors
            gpu_error_check(cudaMemcpyAsync(d_exp_dw_full[monomer_type], w,
                sizeof(double) * M, cudaMemcpyInputToDevice));
            gpu_error_check(cudaMemcpyAsync(d_exp_dw_half[monomer_type], w,
                sizeof(double) * M, cudaMemcpyInputToDevice));

            // Compute d_exp_dw_full = exp(-w*ds/2) for full step
            ker_exp<<<N_BLOCKS, N_THREADS>>>
                (d_exp_dw_full[monomer_type], d_exp_dw_full[monomer_type], 1.0, -0.5 * ds, M);

            // Compute d_exp_dw_half = exp(-w*ds/4) for half step
            ker_exp<<<N_BLOCKS, N_THREADS>>>
                (d_exp_dw_half[monomer_type], d_exp_dw_half[monomer_type], 1.0, -0.25 * ds, M);
        }
        gpu_error_check(cudaDeviceSynchronize());
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

void CudaSolverGlobalRichardsonBase::advance_full_step(
    const int STREAM,
    double* d_q_in, double* d_q_out,
    std::string monomer_type, double* d_q_mask)
{
    const int N_BLOCKS = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();
    const int M = this->cb->get_total_grid();
    const int DIM = this->dim;

    double* _d_exp_dw = d_exp_dw_full[monomer_type];

    // Apply exp(-w*ds/2) at start
    ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_q_out, d_q_in, _d_exp_dw, 1.0, M);
    gpu_error_check(cudaPeekAtLastError());

    // Diffusion with full-step coefficients
    if (DIM == 3)
        advance_propagator_3d_step(this->cb->get_boundary_conditions(), STREAM,
            d_q_out, d_q_out,
            d_xl_full[monomer_type], d_xd_full[monomer_type], d_xh_full[monomer_type],
            d_yl_full[monomer_type], d_yd_full[monomer_type], d_yh_full[monomer_type],
            d_zl_full[monomer_type], d_zd_full[monomer_type], d_zh_full[monomer_type]);
    else if (DIM == 2)
        advance_propagator_2d_step(this->cb->get_boundary_conditions(), STREAM,
            d_q_out, d_q_out,
            d_xl_full[monomer_type], d_xd_full[monomer_type], d_xh_full[monomer_type],
            d_yl_full[monomer_type], d_yd_full[monomer_type], d_yh_full[monomer_type]);
    else if (DIM == 1)
        advance_propagator_1d_step(this->cb->get_boundary_conditions(), STREAM,
            d_q_out, d_q_out,
            d_xl_full[monomer_type], d_xd_full[monomer_type], d_xh_full[monomer_type]);

    // Apply exp(-w*ds/2) at end
    ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_q_out, d_q_out, _d_exp_dw, 1.0, M);
    gpu_error_check(cudaPeekAtLastError());

    // Apply mask if provided
    if (d_q_mask != nullptr)
    {
        ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_q_out, d_q_out, d_q_mask, 1.0, M);
        gpu_error_check(cudaPeekAtLastError());
    }
}

void CudaSolverGlobalRichardsonBase::advance_half_step(
    const int STREAM,
    double* d_q_in, double* d_q_out,
    std::string monomer_type, double* d_q_mask)
{
    const int N_BLOCKS = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();
    const int M = this->cb->get_total_grid();
    const int DIM = this->dim;

    double* _d_exp_dw = d_exp_dw_half[monomer_type];

    // Apply exp(-w*ds/4) at start
    ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_q_out, d_q_in, _d_exp_dw, 1.0, M);
    gpu_error_check(cudaPeekAtLastError());

    // Diffusion with half-step coefficients
    if (DIM == 3)
        advance_propagator_3d_step(this->cb->get_boundary_conditions(), STREAM,
            d_q_out, d_q_out,
            d_xl_half[monomer_type], d_xd_half[monomer_type], d_xh_half[monomer_type],
            d_yl_half[monomer_type], d_yd_half[monomer_type], d_yh_half[monomer_type],
            d_zl_half[monomer_type], d_zd_half[monomer_type], d_zh_half[monomer_type]);
    else if (DIM == 2)
        advance_propagator_2d_step(this->cb->get_boundary_conditions(), STREAM,
            d_q_out, d_q_out,
            d_xl_half[monomer_type], d_xd_half[monomer_type], d_xh_half[monomer_type],
            d_yl_half[monomer_type], d_yd_half[monomer_type], d_yh_half[monomer_type]);
    else if (DIM == 1)
        advance_propagator_1d_step(this->cb->get_boundary_conditions(), STREAM,
            d_q_out, d_q_out,
            d_xl_half[monomer_type], d_xd_half[monomer_type], d_xh_half[monomer_type]);

    // Apply exp(-w*ds/4) at end
    ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_q_out, d_q_out, _d_exp_dw, 1.0, M);
    gpu_error_check(cudaPeekAtLastError());

    // Apply mask if provided
    if (d_q_mask != nullptr)
    {
        ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_q_out, d_q_out, d_q_mask, 1.0, M);
        gpu_error_check(cudaPeekAtLastError());
    }
}

// ============================================================================
// ADI step methods (reuse kernels from CudaSolverCNADI)
// ============================================================================

void CudaSolverGlobalRichardsonBase::advance_propagator_3d_step(
    std::vector<BoundaryCondition> bc,
    const int STREAM,
    double* d_q_in, double* d_q_out,
    double* _d_xl, double* _d_xd, double* _d_xh,
    double* _d_yl, double* _d_yd, double* _d_yh,
    double* _d_zl, double* _d_zd, double* _d_zh)
{
    try
    {
        const int N_BLOCKS = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();
        const int M = this->cb->get_total_grid();
        const std::vector<int> nx = this->cb->get_nx();

        size_t shmem_x = 3 * nx[0] * sizeof(double);
        size_t shmem_y = 3 * nx[1] * sizeof(double);
        size_t shmem_z = 3 * nx[2] * sizeof(double);

        // Calculate q_star (X-sweep)
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
                d_offset_yz, nx[1] * nx[2], nx[1] * nx[2], nx[0]);
        }
        else
        {
            tridiagonal<<<N_BLOCKS, N_THREADS, shmem_x, streams[STREAM][0]>>>(
                _d_xl, _d_xd, _d_xh,
                d_c_star[STREAM], d_temp[STREAM], d_q_star[STREAM],
                d_offset_yz, nx[1] * nx[2], nx[1] * nx[2], nx[0]);
        }
        gpu_error_check(cudaPeekAtLastError());

        // Calculate q_dstar (Y-sweep)
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
                d_offset_xz, nx[0] * nx[2], nx[2], nx[1]);
        }
        else
        {
            tridiagonal<<<N_BLOCKS, N_THREADS, shmem_y, streams[STREAM][0]>>>(
                _d_yl, _d_yd, _d_yh,
                d_c_star[STREAM], d_temp[STREAM], d_q_dstar[STREAM],
                d_offset_xz, nx[0] * nx[2], nx[2], nx[1]);
        }
        gpu_error_check(cudaPeekAtLastError());

        // Calculate q^(n+1) (Z-sweep)
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
                d_offset_xy, nx[0] * nx[1], 1, nx[2]);
        }
        else
        {
            tridiagonal<<<N_BLOCKS, N_THREADS, shmem_z, streams[STREAM][0]>>>(
                _d_zl, _d_zd, _d_zh,
                d_c_star[STREAM], d_temp[STREAM], d_q_out,
                d_offset_xy, nx[0] * nx[1], 1, nx[2]);
        }
        gpu_error_check(cudaPeekAtLastError());
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

void CudaSolverGlobalRichardsonBase::advance_propagator_2d_step(
    std::vector<BoundaryCondition> bc,
    const int STREAM,
    double* d_q_in, double* d_q_out,
    double* _d_xl, double* _d_xd, double* _d_xh,
    double* _d_yl, double* _d_yd, double* _d_yh)
{
    try
    {
        const int N_BLOCKS = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();
        const int M = this->cb->get_total_grid();
        const std::vector<int> nx = this->cb->get_nx();

        size_t shmem_x = 3 * nx[0] * sizeof(double);
        size_t shmem_y = 3 * nx[1] * sizeof(double);

        // Calculate q_star (X-sweep)
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

        // Calculate q^(n+1) (Y-sweep)
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
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

void CudaSolverGlobalRichardsonBase::advance_propagator_1d_step(
    std::vector<BoundaryCondition> bc,
    const int STREAM,
    double* d_q_in, double* d_q_out,
    double* _d_xl, double* _d_xd, double* _d_xh)
{
    try
    {
        const int N_BLOCKS = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();
        const int M = this->cb->get_total_grid();

        size_t shmem_x = 3 * M * sizeof(double);

        // Compute RHS
        compute_crank_1d<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
            bc[0], bc[1],
            _d_xl, _d_xd, _d_xh,
            d_temp[STREAM], d_q_in, M);
        gpu_error_check(cudaPeekAtLastError());

        // Solve tridiagonal system (1 system of size M for 1D case)
        if (bc[0] == BoundaryCondition::PERIODIC)
        {
            tridiagonal_periodic<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                _d_xl, _d_xd, _d_xh,
                d_c_star[STREAM], d_q_sparse[STREAM], d_temp[STREAM], d_q_out,
                d_offset, 1, 1, M);
        }
        else
        {
            tridiagonal<<<N_BLOCKS, N_THREADS, shmem_x, streams[STREAM][0]>>>(
                _d_xl, _d_xd, _d_xh,
                d_c_star[STREAM], d_temp[STREAM], d_q_out,
                d_offset, 1, 1, M);
        }
        gpu_error_check(cudaPeekAtLastError());
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
