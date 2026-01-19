/**
 * @file CudaSolverCNADI.cu
 * @brief CUDA solver using CN-ADI method.
 *
 * Implements propagator advancement for continuous chains using finite
 * differences instead of FFT. Supports non-periodic boundary conditions
 * (reflecting, absorbing) and uses ADI splitting with GPU-accelerated
 * tridiagonal solvers.
 *
 * **CN-ADI (Crank-Nicolson ADI):**
 *
 * The modified diffusion equation is split dimensionally:
 * - 3D: Three sequential tridiagonal solves (x, y, z directions)
 * - 2D: Two tridiagonal solves
 * - 1D: Single tridiagonal solve
 *
 * **Tridiagonal Solvers:**
 *
 * - tridiagonal(): Standard Thomas algorithm with shared memory
 * - tridiagonal_periodic(): Sherman-Morrison for periodic boundaries
 * - Both use dynamic shared memory for coefficient caching
 *
 * **Boundary Conditions:**
 *
 * - PERIODIC: Cyclic boundary with Sherman-Morrison correction
 * - REFLECTING: Zero flux (Neumann) boundary
 * - ABSORBING: Zero value (Dirichlet) boundary
 *
 * **Note:** Stress computation not yet supported for CN-ADI method.
 *
 * @see FiniteDifference for matrix construction
 * @see CudaComputationContinuous for integration
 */

#include <iostream>
#include <cmath>
#include <numbers>
#include "CudaSolverCNADI.h"

CudaSolverCNADI::CudaSolverCNADI(
    ComputationBox<double>* cb,
    Molecules *molecules,
    int n_streams,
    cudaStream_t streams[MAX_STREAMS][2],
    bool reduce_memory,
    bool use_4th_order)
{
    try{
        this->cb = cb;
        this->molecules = molecules;
        this->chain_model = molecules->get_model_name();
        this->n_streams = n_streams;
        this->reduce_memory = reduce_memory;
        this->use_4th_order = use_4th_order;

        if(molecules->get_model_name() != "continuous")
            throw_with_line_number("Real-space method only support 'continuous' chain model.");

        if(!cb->is_orthogonal())
            throw_with_line_number("Real-space method only supports orthogonal unit cells. "
                                   "Use pseudo-spectral method (chain_model='continuous') for non-orthogonal systems.");

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

        // Ensure ContourLengthMapping is finalized
        molecules->finalize_contour_length_mapping();

        // Get unique ds values from ContourLengthMapping
        const ContourLengthMapping& mapping = molecules->get_contour_length_mapping();
        int n_unique_ds = mapping.get_n_unique_ds();

        // Create exp_dw, exp_dw_half, and tridiagonal coefficients for each ds_index
        for (int ds_idx = 1; ds_idx <= n_unique_ds; ++ds_idx)
        {
            for(const auto& item: molecules->get_bond_lengths())
            {
                std::string monomer_type = item.first;
                d_exp_dw     [ds_idx][monomer_type] = nullptr;
                d_exp_dw_half[ds_idx][monomer_type] = nullptr;

                gpu_error_check(cudaMalloc((void**)&d_exp_dw     [ds_idx][monomer_type], sizeof(double)*M));
                gpu_error_check(cudaMalloc((void**)&d_exp_dw_half[ds_idx][monomer_type], sizeof(double)*M));

                d_xl[ds_idx][monomer_type] = nullptr;
                d_xd[ds_idx][monomer_type] = nullptr;
                d_xh[ds_idx][monomer_type] = nullptr;
                gpu_error_check(cudaMalloc((void**)&d_xl[ds_idx][monomer_type], sizeof(double)*nx[0]));
                gpu_error_check(cudaMalloc((void**)&d_xd[ds_idx][monomer_type], sizeof(double)*nx[0]));
                gpu_error_check(cudaMalloc((void**)&d_xh[ds_idx][monomer_type], sizeof(double)*nx[0]));

                d_yl[ds_idx][monomer_type] = nullptr;
                d_yd[ds_idx][monomer_type] = nullptr;
                d_yh[ds_idx][monomer_type] = nullptr;
                gpu_error_check(cudaMalloc((void**)&d_yl[ds_idx][monomer_type], sizeof(double)*nx[1]));
                gpu_error_check(cudaMalloc((void**)&d_yd[ds_idx][monomer_type], sizeof(double)*nx[1]));
                gpu_error_check(cudaMalloc((void**)&d_yh[ds_idx][monomer_type], sizeof(double)*nx[1]));

                d_zl[ds_idx][monomer_type] = nullptr;
                d_zd[ds_idx][monomer_type] = nullptr;
                d_zh[ds_idx][monomer_type] = nullptr;
                gpu_error_check(cudaMalloc((void**)&d_zl[ds_idx][monomer_type], sizeof(double)*nx[2]));
                gpu_error_check(cudaMalloc((void**)&d_zd[ds_idx][monomer_type], sizeof(double)*nx[2]));
                gpu_error_check(cudaMalloc((void**)&d_zh[ds_idx][monomer_type], sizeof(double)*nx[2]));

                // Half-step coefficients for CN-ADI4
                d_xl_half[ds_idx][monomer_type] = nullptr;
                d_xd_half[ds_idx][monomer_type] = nullptr;
                d_xh_half[ds_idx][monomer_type] = nullptr;
                gpu_error_check(cudaMalloc((void**)&d_xl_half[ds_idx][monomer_type], sizeof(double)*nx[0]));
                gpu_error_check(cudaMalloc((void**)&d_xd_half[ds_idx][monomer_type], sizeof(double)*nx[0]));
                gpu_error_check(cudaMalloc((void**)&d_xh_half[ds_idx][monomer_type], sizeof(double)*nx[0]));

                d_yl_half[ds_idx][monomer_type] = nullptr;
                d_yd_half[ds_idx][monomer_type] = nullptr;
                d_yh_half[ds_idx][monomer_type] = nullptr;
                gpu_error_check(cudaMalloc((void**)&d_yl_half[ds_idx][monomer_type], sizeof(double)*nx[1]));
                gpu_error_check(cudaMalloc((void**)&d_yd_half[ds_idx][monomer_type], sizeof(double)*nx[1]));
                gpu_error_check(cudaMalloc((void**)&d_yh_half[ds_idx][monomer_type], sizeof(double)*nx[1]));

                d_zl_half[ds_idx][monomer_type] = nullptr;
                d_zd_half[ds_idx][monomer_type] = nullptr;
                d_zh_half[ds_idx][monomer_type] = nullptr;
                gpu_error_check(cudaMalloc((void**)&d_zl_half[ds_idx][monomer_type], sizeof(double)*nx[2]));
                gpu_error_check(cudaMalloc((void**)&d_zd_half[ds_idx][monomer_type], sizeof(double)*nx[2]));
                gpu_error_check(cudaMalloc((void**)&d_zh_half[ds_idx][monomer_type], sizeof(double)*nx[2]));
            }
        }

        if(DIM == 3)
        {
            for(int i=0; i<n_streams; i++)
            {
                gpu_error_check(cudaMalloc((void**)&d_q_star[i], sizeof(double)*M));
                gpu_error_check(cudaMalloc((void**)&d_q_dstar[i], sizeof(double)*M));
                gpu_error_check(cudaMalloc((void**)&d_c_star[i], sizeof(double)*M));
                gpu_error_check(cudaMalloc((void**)&d_q_sparse[i], sizeof(double)*M));
                gpu_error_check(cudaMalloc((void**)&d_temp[i], sizeof(double)*M));
                gpu_error_check(cudaMalloc((void**)&d_q_full[i], sizeof(double)*M));
                gpu_error_check(cudaMalloc((void**)&d_q_half[i], sizeof(double)*M));
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
                gpu_error_check(cudaMalloc((void**)&d_q_full[i], sizeof(double)*M));
                gpu_error_check(cudaMalloc((void**)&d_q_half[i], sizeof(double)*M));
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
                gpu_error_check(cudaMalloc((void**)&d_q_full[i], sizeof(double)*M));
                gpu_error_check(cudaMalloc((void**)&d_q_half[i], sizeof(double)*M));
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
CudaSolverCNADI::~CudaSolverCNADI()
{
    const int DIM = this->dim;

    // Free nested maps: d_exp_dw[ds_index][monomer_type]
    for(const auto& ds_entry: d_exp_dw)
        for(const auto& item: ds_entry.second)
            cudaFree(item.second);
    for(const auto& ds_entry: d_exp_dw_half)
        for(const auto& item: ds_entry.second)
            cudaFree(item.second);

    // Free tridiagonal coefficients: d_xl[ds_index][monomer_type]
    for(const auto& ds_entry: d_xl)
        for(const auto& item: ds_entry.second)
            cudaFree(item.second);
    for(const auto& ds_entry: d_xd)
        for(const auto& item: ds_entry.second)
            cudaFree(item.second);
    for(const auto& ds_entry: d_xh)
        for(const auto& item: ds_entry.second)
            cudaFree(item.second);

    for(const auto& ds_entry: d_yl)
        for(const auto& item: ds_entry.second)
            cudaFree(item.second);
    for(const auto& ds_entry: d_yd)
        for(const auto& item: ds_entry.second)
            cudaFree(item.second);
    for(const auto& ds_entry: d_yh)
        for(const auto& item: ds_entry.second)
            cudaFree(item.second);

    for(const auto& ds_entry: d_zl)
        for(const auto& item: ds_entry.second)
            cudaFree(item.second);
    for(const auto& ds_entry: d_zd)
        for(const auto& item: ds_entry.second)
            cudaFree(item.second);
    for(const auto& ds_entry: d_zh)
        for(const auto& item: ds_entry.second)
            cudaFree(item.second);

    // Free half-step coefficients
    for(const auto& ds_entry: d_xl_half)
        for(const auto& item: ds_entry.second)
            cudaFree(item.second);
    for(const auto& ds_entry: d_xd_half)
        for(const auto& item: ds_entry.second)
            cudaFree(item.second);
    for(const auto& ds_entry: d_xh_half)
        for(const auto& item: ds_entry.second)
            cudaFree(item.second);

    for(const auto& ds_entry: d_yl_half)
        for(const auto& item: ds_entry.second)
            cudaFree(item.second);
    for(const auto& ds_entry: d_yd_half)
        for(const auto& item: ds_entry.second)
            cudaFree(item.second);
    for(const auto& ds_entry: d_yh_half)
        for(const auto& item: ds_entry.second)
            cudaFree(item.second);

    for(const auto& ds_entry: d_zl_half)
        for(const auto& item: ds_entry.second)
            cudaFree(item.second);
    for(const auto& ds_entry: d_zd_half)
        for(const auto& item: ds_entry.second)
            cudaFree(item.second);
    for(const auto& ds_entry: d_zh_half)
        for(const auto& item: ds_entry.second)
            cudaFree(item.second);

    if(DIM == 3)
    {
        for(int i=0; i<n_streams; i++)
        {
            cudaFree(d_q_star[i]);
            cudaFree(d_q_dstar[i]);
            cudaFree(d_c_star[i]);
            cudaFree(d_q_sparse[i]);
            cudaFree(d_temp[i]);
            cudaFree(d_q_full[i]);
            cudaFree(d_q_half[i]);
        }
        cudaFree(d_offset_xy);
        cudaFree(d_offset_yz);
        cudaFree(d_offset_xz);
    }
    else if(DIM == 2)
    {
        for(int i=0; i<n_streams; i++)
        {
            cudaFree(d_q_star[i]);
            cudaFree(d_c_star[i]);
            cudaFree(d_q_sparse[i]);
            cudaFree(d_temp[i]);
            cudaFree(d_q_full[i]);
            cudaFree(d_q_half[i]);
        }
        cudaFree(d_offset_x);
        cudaFree(d_offset_y);
    }
    else if(DIM == 1)
    {
        for(int i=0; i<n_streams; i++)
        {
            cudaFree(d_q_star[i]);
            cudaFree(d_c_star[i]);
            cudaFree(d_q_sparse[i]);
            cudaFree(d_q_full[i]);
            cudaFree(d_q_half[i]);
        }
        cudaFree(d_offset);
    }
}
void CudaSolverCNADI::update_laplacian_operator()
{
    try
    {
        const int M = this->cb->get_total_grid();
        const int DIM = this->cb->get_dim();
        std::vector<int> nx(DIM);
        if (DIM == 3)
            nx = {this->cb->get_nx(0), this->cb->get_nx(1), this->cb->get_nx(2)};
        else if (DIM == 2)
            nx = {this->cb->get_nx(0), this->cb->get_nx(1), 1};
        else if (DIM == 1)
            nx = {this->cb->get_nx(0), 1, 1};

        double xl[nx[0]], xd[nx[0]], xh[nx[0]];
        double yl[nx[1]], yd[nx[1]], yh[nx[1]];
        double zl[nx[2]], zd[nx[2]], zh[nx[2]];

        double xl_half[nx[0]], xd_half[nx[0]], xh_half[nx[0]];
        double yl_half[nx[1]], yd_half[nx[1]], yh_half[nx[1]];
        double zl_half[nx[2]], zd_half[nx[2]], zh_half[nx[2]];

        // Get unique ds values from ContourLengthMapping
        const ContourLengthMapping& mapping = this->molecules->get_contour_length_mapping();
        int n_unique_ds = mapping.get_n_unique_ds();

        // Compute coefficients for each unique ds value
        for (int ds_idx = 1; ds_idx <= n_unique_ds; ++ds_idx)
        {
            double local_ds = mapping.get_ds_from_index(ds_idx);

            for(const auto& item: this->molecules->get_bond_lengths())
            {
                std::string monomer_type = item.first;
                double bond_length_sq = item.second*item.second;

                // Full-step coefficients (local_ds)
                FiniteDifference::get_laplacian_matrix(
                    this->cb->get_boundary_conditions(),
                    this->cb->get_nx(), this->cb->get_dx(),
                    xl, xd, xh,
                    yl, yd, yh,
                    zl, zd, zh,
                    bond_length_sq, local_ds);

                gpu_error_check(cudaMemcpy(d_xl[ds_idx][monomer_type], xl, sizeof(double)*nx[0], cudaMemcpyHostToDevice));
                gpu_error_check(cudaMemcpy(d_xd[ds_idx][monomer_type], xd, sizeof(double)*nx[0], cudaMemcpyHostToDevice));
                gpu_error_check(cudaMemcpy(d_xh[ds_idx][monomer_type], xh, sizeof(double)*nx[0], cudaMemcpyHostToDevice));

                gpu_error_check(cudaMemcpy(d_yl[ds_idx][monomer_type], yl, sizeof(double)*nx[1], cudaMemcpyHostToDevice));
                gpu_error_check(cudaMemcpy(d_yd[ds_idx][monomer_type], yd, sizeof(double)*nx[1], cudaMemcpyHostToDevice));
                gpu_error_check(cudaMemcpy(d_yh[ds_idx][monomer_type], yh, sizeof(double)*nx[1], cudaMemcpyHostToDevice));

                gpu_error_check(cudaMemcpy(d_zl[ds_idx][monomer_type], zl, sizeof(double)*nx[2], cudaMemcpyHostToDevice));
                gpu_error_check(cudaMemcpy(d_zd[ds_idx][monomer_type], zd, sizeof(double)*nx[2], cudaMemcpyHostToDevice));
                gpu_error_check(cudaMemcpy(d_zh[ds_idx][monomer_type], zh, sizeof(double)*nx[2], cudaMemcpyHostToDevice));

                // Half-step coefficients for CN-ADI4 (local_ds/2)
                FiniteDifference::get_laplacian_matrix(
                    this->cb->get_boundary_conditions(),
                    this->cb->get_nx(), this->cb->get_dx(),
                    xl_half, xd_half, xh_half,
                    yl_half, yd_half, yh_half,
                    zl_half, zd_half, zh_half,
                    bond_length_sq, local_ds/2.0);

                gpu_error_check(cudaMemcpy(d_xl_half[ds_idx][monomer_type], xl_half, sizeof(double)*nx[0], cudaMemcpyHostToDevice));
                gpu_error_check(cudaMemcpy(d_xd_half[ds_idx][monomer_type], xd_half, sizeof(double)*nx[0], cudaMemcpyHostToDevice));
                gpu_error_check(cudaMemcpy(d_xh_half[ds_idx][monomer_type], xh_half, sizeof(double)*nx[0], cudaMemcpyHostToDevice));

                gpu_error_check(cudaMemcpy(d_yl_half[ds_idx][monomer_type], yl_half, sizeof(double)*nx[1], cudaMemcpyHostToDevice));
                gpu_error_check(cudaMemcpy(d_yd_half[ds_idx][monomer_type], yd_half, sizeof(double)*nx[1], cudaMemcpyHostToDevice));
                gpu_error_check(cudaMemcpy(d_yh_half[ds_idx][monomer_type], yh_half, sizeof(double)*nx[1], cudaMemcpyHostToDevice));

                gpu_error_check(cudaMemcpy(d_zl_half[ds_idx][monomer_type], zl_half, sizeof(double)*nx[2], cudaMemcpyHostToDevice));
                gpu_error_check(cudaMemcpy(d_zd_half[ds_idx][monomer_type], zd_half, sizeof(double)*nx[2], cudaMemcpyHostToDevice));
                gpu_error_check(cudaMemcpy(d_zh_half[ds_idx][monomer_type], zh_half, sizeof(double)*nx[2], cudaMemcpyHostToDevice));
            }
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaSolverCNADI::update_dw(std::string device, std::map<std::string, const double*> w_input)
{
    try{
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = this->cb->get_total_grid();

        cudaMemcpyKind cudaMemcpyInputToDevice;
        if (device == "gpu")
            cudaMemcpyInputToDevice = cudaMemcpyDeviceToDevice;
        else if(device == "cpu")
            cudaMemcpyInputToDevice = cudaMemcpyHostToDevice;
        else
        {
            throw_with_line_number("Invalid device \"" + device + "\".");
        }

        // Get unique ds values from ContourLengthMapping
        const ContourLengthMapping& mapping = this->molecules->get_contour_length_mapping();
        int n_unique_ds = mapping.get_n_unique_ds();

        // Compute exp_dw and exp_dw_half for each unique ds value
        for (int ds_idx = 1; ds_idx <= n_unique_ds; ++ds_idx)
        {
            double local_ds = mapping.get_ds_from_index(ds_idx);

            for(const auto& item: w_input)
            {
                std::string monomer_type = item.first;
                const double *w = item.second;

                if (d_exp_dw[ds_idx].find(monomer_type) == d_exp_dw[ds_idx].end())
                    throw_with_line_number("monomer_type \"" + monomer_type + "\" is not in d_exp_dw[" + std::to_string(ds_idx) + "].");

                // Copy field configurations from host to device
                gpu_error_check(cudaMemcpyAsync(
                    d_exp_dw     [ds_idx][monomer_type], w,
                    sizeof(double)*M, cudaMemcpyInputToDevice));
                gpu_error_check(cudaMemcpyAsync(
                    d_exp_dw_half[ds_idx][monomer_type], w,
                    sizeof(double)*M, cudaMemcpyInputToDevice));

                // Compute d_exp_dw and d_exp_dw_half
                ker_exp<<<N_BLOCKS, N_THREADS>>>
                    ((double*) d_exp_dw[ds_idx][monomer_type],
                     (double*) d_exp_dw[ds_idx][monomer_type],      1.0, -0.50*local_ds, M);
                ker_exp<<<N_BLOCKS, N_THREADS>>>
                    ((double*) d_exp_dw_half[ds_idx][monomer_type],
                     (double*) d_exp_dw_half[ds_idx][monomer_type], 1.0, -0.25*local_ds, M);
            }
        }
        gpu_error_check(cudaDeviceSynchronize());
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaSolverCNADI::advance_propagator(
    const int STREAM,
    double *d_q_in, double *d_q_out,
    std::string monomer_type, double *d_q_mask, int ds_index)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = this->cb->get_total_grid();
        const int DIM = this->cb->get_dim();

        // Use the provided ds_index for per-block local_ds
        double *_d_exp_dw = d_exp_dw[ds_index][monomer_type];

        if (use_4th_order)
        {
            // CN-ADI4: 4th order accuracy via Richardson extrapolation
            double *_d_exp_dw_half = d_exp_dw_half[ds_index][monomer_type];

            // ========================================
            // Full step: exp(-w*local_ds/2) * Diffusion(local_ds) * exp(-w*local_ds/2)
            // ========================================
            // Apply exp(-w*local_ds/2) at start
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_q_full[STREAM], d_q_in, _d_exp_dw, 1.0, M);
            gpu_error_check(cudaPeekAtLastError());

            // Diffusion with full-step coefficients
            if(DIM == 3)
                advance_propagator_3d(this->cb->get_boundary_conditions(), STREAM,
                    d_q_full[STREAM], d_q_full[STREAM], monomer_type, ds_index);
            else if(DIM == 2)
                advance_propagator_2d(this->cb->get_boundary_conditions(), STREAM,
                    d_q_full[STREAM], d_q_full[STREAM], monomer_type, ds_index);
            else if(DIM == 1)
                advance_propagator_1d(this->cb->get_boundary_conditions(), STREAM,
                    d_q_full[STREAM], d_q_full[STREAM], monomer_type, ds_index);

            // Apply exp(-w*local_ds/2) at end
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_q_full[STREAM], d_q_full[STREAM], _d_exp_dw, 1.0, M);
            gpu_error_check(cudaPeekAtLastError());

            // ========================================
            // Two half steps: exp(-w*local_ds/4) * Diffusion(local_ds/2) * exp(-w*local_ds/2) * Diffusion(local_ds/2) * exp(-w*local_ds/4)
            // ========================================
            // First half-step: exp(-w*local_ds/4) at start
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_q_half[STREAM], d_q_in, _d_exp_dw_half, 1.0, M);
            gpu_error_check(cudaPeekAtLastError());

            // Diffusion with half-step coefficients
            if(DIM == 3)
                advance_propagator_3d_step(this->cb->get_boundary_conditions(), STREAM,
                    d_q_half[STREAM], d_q_half[STREAM],
                    d_xl_half[ds_index][monomer_type], d_xd_half[ds_index][monomer_type], d_xh_half[ds_index][monomer_type],
                    d_yl_half[ds_index][monomer_type], d_yd_half[ds_index][monomer_type], d_yh_half[ds_index][monomer_type],
                    d_zl_half[ds_index][monomer_type], d_zd_half[ds_index][monomer_type], d_zh_half[ds_index][monomer_type]);
            else if(DIM == 2)
                advance_propagator_2d_step(this->cb->get_boundary_conditions(), STREAM,
                    d_q_half[STREAM], d_q_half[STREAM],
                    d_xl_half[ds_index][monomer_type], d_xd_half[ds_index][monomer_type], d_xh_half[ds_index][monomer_type],
                    d_yl_half[ds_index][monomer_type], d_yd_half[ds_index][monomer_type], d_yh_half[ds_index][monomer_type]);
            else if(DIM == 1)
                advance_propagator_1d_step(this->cb->get_boundary_conditions(), STREAM,
                    d_q_half[STREAM], d_q_half[STREAM],
                    d_xl_half[ds_index][monomer_type], d_xd_half[ds_index][monomer_type], d_xh_half[ds_index][monomer_type]);

            // Apply exp(-w*local_ds/2) at junction
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_q_half[STREAM], d_q_half[STREAM], _d_exp_dw, 1.0, M);
            gpu_error_check(cudaPeekAtLastError());

            // Second half-step: Diffusion with half-step coefficients
            if(DIM == 3)
                advance_propagator_3d_step(this->cb->get_boundary_conditions(), STREAM,
                    d_q_half[STREAM], d_q_half[STREAM],
                    d_xl_half[ds_index][monomer_type], d_xd_half[ds_index][monomer_type], d_xh_half[ds_index][monomer_type],
                    d_yl_half[ds_index][monomer_type], d_yd_half[ds_index][monomer_type], d_yh_half[ds_index][monomer_type],
                    d_zl_half[ds_index][monomer_type], d_zd_half[ds_index][monomer_type], d_zh_half[ds_index][monomer_type]);
            else if(DIM == 2)
                advance_propagator_2d_step(this->cb->get_boundary_conditions(), STREAM,
                    d_q_half[STREAM], d_q_half[STREAM],
                    d_xl_half[ds_index][monomer_type], d_xd_half[ds_index][monomer_type], d_xh_half[ds_index][monomer_type],
                    d_yl_half[ds_index][monomer_type], d_yd_half[ds_index][monomer_type], d_yh_half[ds_index][monomer_type]);
            else if(DIM == 1)
                advance_propagator_1d_step(this->cb->get_boundary_conditions(), STREAM,
                    d_q_half[STREAM], d_q_half[STREAM],
                    d_xl_half[ds_index][monomer_type], d_xd_half[ds_index][monomer_type], d_xh_half[ds_index][monomer_type]);

            // Apply exp(-w*local_ds/4) at end
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_q_half[STREAM], d_q_half[STREAM], _d_exp_dw_half, 1.0, M);
            gpu_error_check(cudaPeekAtLastError());

            // ========================================
            // CN-ADI4: Richardson extrapolation q_out = (4*q_half - q_full) / 3
            // Use ker_lin_comb (dst = a*src1 + b*src2), NOT ker_add_lin_comb (dst += a*src1 + b*src2)
            // ========================================
            ker_lin_comb<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                d_q_out, 4.0/3.0, d_q_half[STREAM], -1.0/3.0, d_q_full[STREAM], M);
            gpu_error_check(cudaPeekAtLastError());
        }
        else
        {
            // CN-ADI2: single full step only (2nd order accuracy)
            // ========================================
            // Full step: exp(-w*local_ds/2) * Diffusion(local_ds) * exp(-w*local_ds/2)
            // ========================================
            // Apply exp(-w*local_ds/2) at start
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_q_out, d_q_in, _d_exp_dw, 1.0, M);
            gpu_error_check(cudaPeekAtLastError());

            // Diffusion with full-step coefficients
            if(DIM == 3)
                advance_propagator_3d(this->cb->get_boundary_conditions(), STREAM,
                    d_q_out, d_q_out, monomer_type, ds_index);
            else if(DIM == 2)
                advance_propagator_2d(this->cb->get_boundary_conditions(), STREAM,
                    d_q_out, d_q_out, monomer_type, ds_index);
            else if(DIM == 1)
                advance_propagator_1d(this->cb->get_boundary_conditions(), STREAM,
                    d_q_out, d_q_out, monomer_type, ds_index);

            // Apply exp(-w*local_ds/2) at end
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_q_out, d_q_out, _d_exp_dw, 1.0, M);
            gpu_error_check(cudaPeekAtLastError());
        }

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
void CudaSolverCNADI::advance_propagator_3d(
    std::vector<BoundaryCondition> bc,
    const int STREAM,
    double *d_q_in, double *d_q_out, std::string monomer_type, int ds_index)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();
        const int M = this->cb->get_total_grid();
        const std::vector<int> nx = this->cb->get_nx();

        double *_d_xl = d_xl[ds_index][monomer_type];
        double *_d_xd = d_xd[ds_index][monomer_type];
        double *_d_xh = d_xh[ds_index][monomer_type];

        double *_d_yl = d_yl[ds_index][monomer_type];
        double *_d_yd = d_yd[ds_index][monomer_type];
        double *_d_yh = d_yh[ds_index][monomer_type];

        double *_d_zl = d_zl[ds_index][monomer_type];
        double *_d_zd = d_zd[ds_index][monomer_type];
        double *_d_zh = d_zh[ds_index][monomer_type];

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
void CudaSolverCNADI::advance_propagator_2d(
    std::vector<BoundaryCondition> bc,
    const int STREAM,
    double *d_q_in, double *d_q_out, std::string monomer_type, int ds_index)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();
        const int M = this->cb->get_total_grid();
        const std::vector<int> nx = this->cb->get_nx();

        double *_d_xl = d_xl[ds_index][monomer_type];
        double *_d_xd = d_xd[ds_index][monomer_type];
        double *_d_xh = d_xh[ds_index][monomer_type];

        double *_d_yl = d_yl[ds_index][monomer_type];
        double *_d_yd = d_yd[ds_index][monomer_type];
        double *_d_yh = d_yh[ds_index][monomer_type];

        // Shared memory size for 3 arrays of doubles
        size_t shmem_x = 3 * nx[0] * sizeof(double);
        size_t shmem_y = 3 * nx[1] * sizeof(double);

        // Calculate q_star
        compute_crank_2d_step_1<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
            bc[0], bc[1], bc[2], bc[3],
            _d_xl, _d_xd, _d_xh, nx[0],
            _d_yl, _d_yd, _d_yh, nx[1],
            d_temp[STREAM], d_q_in, M);
        gpu_error_check(cudaPeekAtLastError());

        // gpu_error_check(cudaMemcpy(d_q_out, d_q_star, sizeof(double)*M, cudaMemcpyDeviceToDevice));

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

        // Calculate q_dstar
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
void CudaSolverCNADI::advance_propagator_1d(
    std::vector<BoundaryCondition> bc,
    const int STREAM,
    double *d_q_in, double *d_q_out, std::string monomer_type, int ds_index)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();
        const int M = this->cb->get_total_grid();
        const std::vector<int> nx = this->cb->get_nx();

        double *_d_xl = d_xl[ds_index][monomer_type];
        double *_d_xd = d_xd[ds_index][monomer_type];
        double *_d_xh = d_xh[ds_index][monomer_type];

        // Shared memory size for 3 arrays of doubles
        size_t shmem_x = 3 * nx[0] * sizeof(double);

        compute_crank_1d<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
            bc[0], bc[1],
            _d_xl, _d_xd, _d_xh,
            d_q_star[STREAM], d_q_in, nx[0]);
        gpu_error_check(cudaPeekAtLastError());

        if (bc[0] == BoundaryCondition::PERIODIC)
            tridiagonal_periodic<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                _d_xl, _d_xd, _d_xh,
                d_c_star[STREAM], d_q_sparse[STREAM], d_q_star[STREAM], d_q_out,
                d_offset, 1, 1, nx[0]);
        else
            tridiagonal<<<N_BLOCKS, N_THREADS, shmem_x, streams[STREAM][0]>>>(
                _d_xl, _d_xd, _d_xh,
                d_c_star[STREAM], d_q_star[STREAM], d_q_out, d_offset, 1, 1, nx[0]);
        gpu_error_check(cudaPeekAtLastError());
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaSolverCNADI::advance_propagator_3d_step(
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
void CudaSolverCNADI::advance_propagator_2d_step(
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

        // Shared memory size for 3 arrays of doubles
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

        // Calculate q_dstar
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
void CudaSolverCNADI::advance_propagator_1d_step(
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
        const std::vector<int> nx = this->cb->get_nx();

        // Shared memory size for 3 arrays of doubles
        size_t shmem_x = 3 * nx[0] * sizeof(double);

        compute_crank_1d<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
            bc[0], bc[1],
            _d_xl, _d_xd, _d_xh,
            d_q_star[STREAM], d_q_in, nx[0]);
        gpu_error_check(cudaPeekAtLastError());

        if (bc[0] == BoundaryCondition::PERIODIC)
            tridiagonal_periodic<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                _d_xl, _d_xd, _d_xh,
                d_c_star[STREAM], d_q_sparse[STREAM], d_q_star[STREAM], d_q_out,
                d_offset, 1, 1, nx[0]);
        else
            tridiagonal<<<N_BLOCKS, N_THREADS, shmem_x, streams[STREAM][0]>>>(
                _d_xl, _d_xd, _d_xh,
                d_c_star[STREAM], d_q_star[STREAM], d_q_out, d_offset, 1, 1, nx[0]);
        gpu_error_check(cudaPeekAtLastError());
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaSolverCNADI::compute_single_segment_stress(
    [[maybe_unused]] const int STREAM,
    [[maybe_unused]] double *d_q_pair, [[maybe_unused]] double *d_segment_stress,
    [[maybe_unused]] std::string monomer_type, [[maybe_unused]] bool is_half_bond_length)
{
    try
    {
        throw_with_line_number("Currently, the CN-ADI method does not support stress computation.");
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

__device__ int d_max_of_two(int x, int y)
{
   return (x > y) ? x : y;
}
__device__ int d_min_of_two(int x, int y)
{
   return (x < y) ? x : y;
}

__global__ void compute_crank_3d_step_1(
    BoundaryCondition bc_xl, BoundaryCondition bc_xh,
    BoundaryCondition bc_yl, BoundaryCondition bc_yh,
    BoundaryCondition bc_zl, BoundaryCondition bc_zh,
    const double *d_xl, const double *d_xd, const double *d_xh, const int I,
    const double *d_yl, const double *d_yd, const double *d_yh, const int J,
    const double *d_zl, const double *d_zd, const double *d_zh, const int K,
    double *d_q_out, const double *d_q_in, const int M)
{
    int im, ip, jm, jp, km, kp;

    int n = blockIdx.x * blockDim.x + threadIdx.x;
    while(n < M)
    {
        int i = n / (J*K);
        int j = (n-i*J*K) / K;
        int k = n % K;

        if (bc_xl == BoundaryCondition::PERIODIC)
            im = (I+i-1) % I;
        else
            im = d_max_of_two(0,i-1);
        if (bc_xh == BoundaryCondition::PERIODIC)
            ip = (i+1) % I;
        else
            ip = d_min_of_two(I-1,i+1);

        if (bc_yl == BoundaryCondition::PERIODIC)
            jm = (J+j-1) % J;
        else
            jm = d_max_of_two(0,j-1);
        if (bc_yh == BoundaryCondition::PERIODIC)
            jp = (j+1) % J;
        else
            jp = d_min_of_two(J-1,j+1);

        if (bc_zl == BoundaryCondition::PERIODIC)
            km = (K+k-1) % K;
        else
            km = d_max_of_two(0,k-1);
        if (bc_zh == BoundaryCondition::PERIODIC)
            kp = (k+1) % K;
        else
            kp = d_min_of_two(K-1,k+1);

        int im_j_k = im*J*K + j*K + k;
        int ip_j_k = ip*J*K + j*K + k;
        int i_jm_k = i*J*K + jm*K + k;
        int i_jp_k = i*J*K + jp*K + k;
        int i_j_km = i*J*K + j*K + km;
        int i_j_kp = i*J*K + j*K + kp;

        d_q_out[n] = 2.0*((3.0-0.5*d_xd[i]-d_yd[j]-d_zd[k])*d_q_in[n]
                - d_zl[k]*d_q_in[i_j_km] - d_zh[k]*d_q_in[i_j_kp]
                - d_yl[j]*d_q_in[i_jm_k] - d_yh[j]*d_q_in[i_jp_k])
                - d_xl[i]*d_q_in[im_j_k] - d_xh[i]*d_q_in[ip_j_k];

        n += blockDim.x * gridDim.x;
    }
}

__global__ void compute_crank_3d_step_2(
    BoundaryCondition bc_yl, BoundaryCondition bc_yh,
    const double *d_yl, const double *d_yd, const double *d_yh, const int J, const int K,
    double *d_q_out, const double *d_q_star, const double *d_q_in, const int M)
{
    int jm, jp;

    int n = blockIdx.x * blockDim.x + threadIdx.x;
    while(n < M)
    {
        int i = n / (J*K);
        int j = (n-i*J*K) / K;
        int k = n % K;

        if (bc_yl == BoundaryCondition::PERIODIC)
            jm = (J+j-1) % J;
        else
            jm = d_max_of_two(0,j-1);
        if (bc_yh == BoundaryCondition::PERIODIC)
            jp = (j+1) % J;
        else
            jp = d_min_of_two(J-1,j+1);

        int i_jm_k = i*J*K + jm*K + k;
        int i_jp_k = i*J*K + jp*K + k;

        d_q_out[n] = d_q_star[n] + (d_yd[j]-1.0)*d_q_in[n]
            + d_yl[j]*d_q_in[i_jm_k] + d_yh[j]*d_q_in[i_jp_k];

        n += blockDim.x * gridDim.x;
    }
}

__global__ void compute_crank_3d_step_3(
    BoundaryCondition bc_zl, BoundaryCondition bc_zh,
    const double *d_zl, const double *d_zd, const double *d_zh, const int J, const int K,
    double *d_q_out, const double *d_q_dstar, const double *d_q_in, const int M)
{
    int km, kp;

    int n = blockIdx.x * blockDim.x + threadIdx.x;
    while(n < M)
    {
        int i = n / (J*K);
        int j = (n-i*J*K) / K;
        int k = n % K;

        if (bc_zl == BoundaryCondition::PERIODIC)
            km = (K+k-1) % K;
        else
            km = d_max_of_two(0,k-1);
        if (bc_zh == BoundaryCondition::PERIODIC)
            kp = (k+1) % K;
        else
            kp = d_min_of_two(K-1,k+1);

        int i_j_km = i*J*K + j*K + km;
        int i_j_kp = i*J*K + j*K + kp;

        d_q_out[n] = d_q_dstar[n] + (d_zd[k]-1.0)*d_q_in[n]
            + d_zl[k]*d_q_in[i_j_km] + d_zh[k]*d_q_in[i_j_kp];

        n += blockDim.x * gridDim.x;
    }
}

__global__ void compute_crank_2d_step_1(
    BoundaryCondition bc_xl, BoundaryCondition bc_xh,
    BoundaryCondition bc_yl, BoundaryCondition bc_yh,
    const double *d_xl, const double *d_xd, const double *d_xh, const int I,
    const double *d_yl, const double *d_yd, const double *d_yh, const int J,
    double *d_q_out, const double *d_q_in, const int M)
{
    int im, ip, jm, jp;

    int n = blockIdx.x * blockDim.x + threadIdx.x;
    while(n < M)
    {
        int i = n / J;
        int j = n % J;

        if (bc_xl == BoundaryCondition::PERIODIC)
            im = (I+i-1) % I;
        else
            im = d_max_of_two(0,i-1);
        if (bc_xh == BoundaryCondition::PERIODIC)
            ip = (i+1) % I;
        else
            ip = d_min_of_two(I-1,i+1);

        if (bc_yl == BoundaryCondition::PERIODIC)
            jm = (J+j-1) % J;
        else
            jm = d_max_of_two(0,j-1);
        if (bc_yh == BoundaryCondition::PERIODIC)
            jp = (j+1) % J;
        else
            jp = d_min_of_two(J-1,j+1);

        int i_jm = i*J + jm;
        int i_jp = i*J + jp;
        int im_j = im*J + j;
        int ip_j = ip*J + j;

        d_q_out[n] = 2.0*((2.0-0.5*d_xd[i]-d_yd[j])*d_q_in[n]
                   - d_yl[j]*d_q_in[i_jm] - d_yh[j]*d_q_in[i_jp])
                   - d_xl[i]*d_q_in[im_j] - d_xh[i]*d_q_in[ip_j];

        n += blockDim.x * gridDim.x;
    }
}

__global__ void compute_crank_2d_step_2(
    BoundaryCondition bc_yl, BoundaryCondition bc_yh,
    const double *d_yl, const double *d_yd, const double *d_yh, const int J,
    double *d_q_out, const double *d_q_star, const double *d_q_in, const int M)
{
    int jm, jp;

    int n = blockIdx.x * blockDim.x + threadIdx.x;
    while(n < M)
    {
        int i = n/J;
        int j = n%J;

        if (bc_yl == BoundaryCondition::PERIODIC)
            jm = (J+j-1) % J;
        else
            jm = d_max_of_two(0,j-1);
        if (bc_yh == BoundaryCondition::PERIODIC)
            jp = (j+1) % J;
        else
            jp = d_min_of_two(J-1,j+1);

        int i_jm = i*J + jm;
        int i_jp = i*J + jp;

        d_q_out[n] = d_q_star[n] + (d_yd[j]-1.0)*d_q_in[n]
            + d_yl[j]*d_q_in[i_jm] + d_yh[j]*d_q_in[i_jp];

        n += blockDim.x * gridDim.x;
    }
}

__global__ void compute_crank_1d(
    BoundaryCondition bc_xl, BoundaryCondition bc_xh,
    const double *d_xl, const double *d_xd, const double *d_xh,
    double *d_q_out, const double *d_q_in, const int M)
{
    int im, ip;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while(i < M)
    {
        if (bc_xl == BoundaryCondition::PERIODIC)
            im = (M+i-1) % M;
        else
            im = d_max_of_two(0,i-1);
        if (bc_xh == BoundaryCondition::PERIODIC)
            ip = (i+1) % M;
        else
            ip = d_min_of_two(M-1,i+1);

        // B part of Ax=B matrix equation
        d_q_out[i] = (2.0-d_xd[i])*d_q_in[i] - d_xl[i]*d_q_in[im] - d_xh[i]*d_q_in[ip];

        i += blockDim.x * gridDim.x;
    }
}

// This method solves CX=Y, where C is a tridiagonal matrix

// Use constant restrict for pointers to allow compiler optimizations
// and utilize the Read-Only Data Cache (__ldg).
#define FETCH(arr, i) __ldg(&arr[i])

__global__ void tridiagonal(
    const double* __restrict__ d_xl, 
    const double* __restrict__ d_xd, 
    const double* __restrict__ d_xh,
    double* __restrict__ d_c_star,   
    const double* __restrict__ d_d, 
    double* __restrict__ d_x,
    const int* __restrict__ d_offset, 
    const int REPEAT, const int INTERVAL, const int M)
{
    // Shared memory layout: [xl (M), xd (M), xh (M)]
    extern __shared__ double s_coeffs[];
    double* s_xl = s_coeffs;
    double* s_xd = s_coeffs + M;
    double* s_xh = s_coeffs + 2 * M;

    // Collaborative load of coefficients into shared memory
    for (int i = threadIdx.x; i < M; i += blockDim.x) {
        s_xl[i] = d_xl[i];
        s_xd[i] = d_xd[i];
        s_xh[i] = d_xh[i];
    }
    __syncthreads();

    int n = blockIdx.x * blockDim.x + threadIdx.x;
    while (n < REPEAT)
    {
        const int start_idx = d_offset[n];
        const double* _d_d = &d_d[start_idx];
        double*       _d_x = &d_x[start_idx];
        double*  _d_c_star = &d_c_star[start_idx];

        // Forward sweep using shared memory
        double inv_temp = 1.0 / s_xd[0];
        double c_prev = s_xh[0] * inv_temp;
        double x_prev = _d_d[0] * inv_temp;

        _d_c_star[0] = c_prev;
        _d_x[0] = x_prev;

        for (int i = 1; i < M; i++)
        {
            double xl_i = s_xl[i];
            double temp = s_xd[i] - xl_i * c_prev;
            inv_temp = 1.0 / temp;

            if (i < M - 1) {
                c_prev = s_xh[i] * inv_temp;
                _d_c_star[i * INTERVAL] = c_prev;
            }

            x_prev = (_d_d[i * INTERVAL] - xl_i * x_prev) * inv_temp;
            _d_x[i * INTERVAL] = x_prev;
        }

        // Backward substitution
        for (int i = M - 2; i >= 0; i--)
        {
            x_prev = _d_x[i * INTERVAL] - _d_c_star[i * INTERVAL] * x_prev;
            _d_x[i * INTERVAL] = x_prev;
        }
        n += blockDim.x * gridDim.x;
    }
}

// This method solves CX=Y, where C is a near-tridiagonal matrix with periodic boundary condition
__global__ void tridiagonal_periodic(
    const double* __restrict__ d_xl, 
    const double* __restrict__ d_xd, 
    const double* __restrict__ d_xh,
    double* __restrict__ d_c_star, 
    double* __restrict__ d_q_sparse, 
    const double* __restrict__ d_d, 
    double* __restrict__ d_x,
    const int* __restrict__ d_offset, 
    const int REPEAT, const int INTERVAL, const int M)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    while (n < REPEAT)
    {
        const int start_idx = d_offset[n];
        const double* _d_d = &d_d[start_idx];
        double*       _d_x = &d_x[start_idx];
        double*  _d_c_star = &d_c_star[start_idx];
        double* _d_q_sparse = &d_q_sparse[start_idx];

        // Forward sweep
        double inv_temp = 1.0 / (FETCH(d_xd, 0) - 1.0);
        double c_prev = FETCH(d_xh, 0) * inv_temp;
        double x_prev = _d_d[0] * inv_temp;
        double q_prev = inv_temp;

        _d_c_star[0] = c_prev;
        _d_x[0] = x_prev;
        _d_q_sparse[0] = q_prev;

        for (int i = 1; i < M - 1; i++)
        {
            double xl_i = FETCH(d_xl, i);
            inv_temp = 1.0 / (FETCH(d_xd, i) - xl_i * c_prev);
            
            c_prev = FETCH(d_xh, i) * inv_temp;
            x_prev = (_d_d[i * INTERVAL] - xl_i * x_prev) * inv_temp;
            q_prev = (-xl_i * q_prev) * inv_temp;

            _d_c_star[i * INTERVAL] = c_prev;
            _d_x[i * INTERVAL] = x_prev;
            _d_q_sparse[i * INTERVAL] = q_prev;
        }

        // Final element of forward sweep
        double xl_M_1 = FETCH(d_xl, M - 1);
        inv_temp = 1.0 / (FETCH(d_xd, M - 1) - FETCH(d_xh, M - 1) * FETCH(d_xl, 0) - xl_M_1 * c_prev);
        x_prev = (_d_d[(M - 1) * INTERVAL] - xl_M_1 * x_prev) * inv_temp;
        q_prev = (FETCH(d_xh, M - 1) - xl_M_1 * q_prev) * inv_temp;
        
        _d_x[(M - 1) * INTERVAL] = x_prev;
        _d_q_sparse[(M - 1) * INTERVAL] = q_prev;

        // Backward substitution
        // Note: c_star[M-1] is not used in back-substitution, we use registers for trailing values
        for (int i = M - 2; i >= 0; i--)
        {
            double c_star_i = _d_c_star[i * INTERVAL];
            x_prev = _d_x[i * INTERVAL] - c_star_i * x_prev;
            q_prev = _d_q_sparse[i * INTERVAL] - c_star_i * q_prev;
            
            _d_x[i * INTERVAL] = x_prev;
            _d_q_sparse[i * INTERVAL] = q_prev;
        }

        // Sherman-Morrison Correction
        double x_0 = _d_x[0];
        double q_0 = _d_q_sparse[0];
        double x_M_1 = _d_x[(M-1)*INTERVAL];
        double q_M_1 = _d_q_sparse[(M-1)*INTERVAL];
        double xl_0 = FETCH(d_xl, 0);

        double value = (x_0 + xl_0 * x_M_1) / (1.0 + q_0 + xl_0 * q_M_1);

        for (int i = 0; i < M; i++) {
            _d_x[i * INTERVAL] -= _d_q_sparse[i * INTERVAL] * value;
        }
        n += blockDim.x * gridDim.x;
    }
}