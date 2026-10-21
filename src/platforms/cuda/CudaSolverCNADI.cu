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
 * @see ComputationContinuous for integration
 */

#include <algorithm>
#include <iostream>
#include <cmath>
#include <numbers>
#include "CudaSolverCNADI.h"

// The line solvers below give one thread to each grid line, so a launch sized
// for whole grids (CudaCommon blocks x threads) would leave most blocks idle:
// a 64^3 grid has 4096 lines per direction. Launch one 64-thread block per 64 lines.
constexpr int LINE_THREADS = 64;
static inline int line_blocks(int n_lines) { return (n_lines + LINE_THREADS - 1)/LINE_THREADS; }

CudaSolverCNADI::CudaSolverCNADI(
    ComputationBox<double>* cb,
    Molecules *molecules,
    int n_streams,
    cudaStream_t streams[MAX_STREAMS][2],
    [[maybe_unused]] bool reduce_memory)
{
    try{
        this->cb = cb;
        this->molecules = molecules;
        this->n_streams = n_streams;
        this->space_group_ = nullptr;
        this->d_reduced_basis_indices_ = nullptr;
        this->d_full_to_reduced_map_ = nullptr;
        this->n_basis_ = 0;

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

        // Map slots are owning buffers: a failed allocation leaves the slot empty
        // Create exp_dw and tridiagonal coefficients for each ds_index
        for (int ds_idx = 0; ds_idx < n_unique_ds; ++ds_idx)
        {
            for(const auto& item: molecules->get_bond_lengths())
            {
                std::string monomer_type = item.first;
                d_exp_dw[ds_idx][monomer_type].allocate(M);

                d_xl[ds_idx][monomer_type].allocate(nx[0]);
                d_xd[ds_idx][monomer_type].allocate(nx[0]);
                d_xh[ds_idx][monomer_type].allocate(nx[0]);

                d_yl[ds_idx][monomer_type].allocate(nx[1]);
                d_yd[ds_idx][monomer_type].allocate(nx[1]);
                d_yh[ds_idx][monomer_type].allocate(nx[1]);

                d_zl[ds_idx][monomer_type].allocate(nx[2]);
                d_zd[ds_idx][monomer_type].allocate(nx[2]);
                d_zh[ds_idx][monomer_type].allocate(nx[2]);
            }
        }

        if(DIM == 3)
        {
            for(int i=0; i<n_streams; i++)
            {
                d_q_star[i].allocate(M);
                d_q_dstar[i].allocate(M);
                d_c_star[i].allocate(M);
                d_q_sparse[i].allocate(M);
                d_temp[i].allocate(M);
            }

            std::vector<int> offset_xy(nx[0]*nx[1]);
            std::vector<int> offset_yz(nx[1]*nx[2]);
            std::vector<int> offset_xz(nx[0]*nx[2]);
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

            d_offset_xy.allocate(nx[0]*nx[1]);
            d_offset_yz.allocate(nx[1]*nx[2]);
            d_offset_xz.allocate(nx[0]*nx[2]);

            CUDA_CHECK(cudaMemcpy(d_offset_xy, offset_xy.data(), sizeof(int)*nx[0]*nx[1], cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_offset_yz, offset_yz.data(), sizeof(int)*nx[1]*nx[2], cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_offset_xz, offset_xz.data(), sizeof(int)*nx[0]*nx[2], cudaMemcpyHostToDevice));
        }
        else if(DIM == 2)
        {
            for(int i=0; i<n_streams; i++)
            {
                d_q_star[i].allocate(M);
                d_c_star[i].allocate(M);
                d_q_sparse[i].allocate(M);
                d_temp[i].allocate(M);
            }

            std::vector<int> offset_x(nx[0]);
            std::vector<int> offset_y(nx[1]);

            for(int i=0;i<nx[0];i++)
                offset_x[i] = i*nx[1];

            for(int j=0;j<nx[1];j++)
                offset_y[j] = j;

            d_offset_x.allocate(nx[0]);
            d_offset_y.allocate(nx[1]);

            CUDA_CHECK(cudaMemcpy(d_offset_x, offset_x.data(), sizeof(int)*nx[0], cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_offset_y, offset_y.data(), sizeof(int)*nx[1], cudaMemcpyHostToDevice));
        }
        else if(DIM == 1)
        {
            for(int i=0; i<n_streams; i++)
            {
                d_q_star[i].allocate(M);
                d_c_star[i].allocate(M);
                d_q_sparse[i].allocate(M);
            }

            d_offset.allocate(1);
            CUDA_CHECK(cudaMemset(d_offset, 0, sizeof(int)));
        }

        // Allow the tridiagonal solver kernels up to 96000 bytes of dynamic shared
        // memory, or the device's opt-in limit when that is smaller (a launch that
        // needs more then fails with a reported launch error)
        int device = 0, optin_shmem = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaDeviceGetAttribute(&optin_shmem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
        const int max_shmem = std::min(96000, optin_shmem);
        CUDA_CHECK(cudaFuncSetAttribute(tridiagonal, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shmem));
        CUDA_CHECK(cudaFuncSetAttribute(tridiagonal_periodic, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shmem));

        update_laplacian_operator();
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());   // the owning members release what was acquired
    }
}
CudaSolverCNADI::~CudaSolverCNADI() = default;


void CudaSolverCNADI::set_space_group(
    SpaceGroup* sg,
    int* d_reduced_basis_indices,
    int* d_full_to_reduced_map,
    int n_basis)
{
    space_group_ = sg;
    d_reduced_basis_indices_ = d_reduced_basis_indices;
    d_full_to_reduced_map_ = d_full_to_reduced_map;
    n_basis_ = n_basis;

    const int M = cb->get_total_grid();
    if (space_group_ != nullptr)
    {
        for (int i = 0; i < n_streams; i++)
        {
            if (d_q_full_in_[i].empty())
                d_q_full_in_[i].allocate(M);
            if (d_q_full_out_[i].empty())
                d_q_full_out_[i].allocate(M);
        }
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

        std::vector<double> xl(nx[0]), xd(nx[0]), xh(nx[0]);
        std::vector<double> yl(nx[1]), yd(nx[1]), yh(nx[1]);
        std::vector<double> zl(nx[2]), zd(nx[2]), zh(nx[2]);

        // Get unique ds values from ContourLengthMapping
        const ContourLengthMapping& mapping = this->molecules->get_contour_length_mapping();
        int n_unique_ds = mapping.get_n_unique_ds();

        // Compute coefficients for each unique ds value
        for (int ds_idx = 0; ds_idx < n_unique_ds; ++ds_idx)
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
                    xl.data(), xd.data(), xh.data(),
                    yl.data(), yd.data(), yh.data(),
                    zl.data(), zd.data(), zh.data(),
                    bond_length_sq, local_ds);

                CUDA_CHECK(cudaMemcpy(d_xl.at(ds_idx).at(monomer_type), xl.data(), sizeof(double)*nx[0], cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_xd.at(ds_idx).at(monomer_type), xd.data(), sizeof(double)*nx[0], cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_xh.at(ds_idx).at(monomer_type), xh.data(), sizeof(double)*nx[0], cudaMemcpyHostToDevice));

                CUDA_CHECK(cudaMemcpy(d_yl.at(ds_idx).at(monomer_type), yl.data(), sizeof(double)*nx[1], cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_yd.at(ds_idx).at(monomer_type), yd.data(), sizeof(double)*nx[1], cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_yh.at(ds_idx).at(monomer_type), yh.data(), sizeof(double)*nx[1], cudaMemcpyHostToDevice));

                CUDA_CHECK(cudaMemcpy(d_zl.at(ds_idx).at(monomer_type), zl.data(), sizeof(double)*nx[2], cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_zd.at(ds_idx).at(monomer_type), zd.data(), sizeof(double)*nx[2], cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_zh.at(ds_idx).at(monomer_type), zh.data(), sizeof(double)*nx[2], cudaMemcpyHostToDevice));

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
        const bool use_reduced_basis = (space_group_ != nullptr);

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

        // Compute exp_dw for each unique ds value
        for (int ds_idx = 0; ds_idx < n_unique_ds; ++ds_idx)
        {
            double local_ds = mapping.get_ds_from_index(ds_idx);

            for(const auto& item: w_input)
            {
                std::string monomer_type = item.first;
                const double *w = item.second;

                if (d_exp_dw[ds_idx].find(monomer_type) == d_exp_dw[ds_idx].end())
                    throw_with_line_number("monomer_type \"" + monomer_type + "\" is not in d_exp_dw[" + std::to_string(ds_idx) + "].");

                if (use_reduced_basis && device == "gpu")
                {
                    ker_expand_reduced_basis<<<N_BLOCKS, N_THREADS>>>(
                        d_exp_dw.at(ds_idx).at(monomer_type), w, d_full_to_reduced_map_, M);
                    CUDA_CHECK_LAUNCH();
                }
                else
                {
                    const double* w_full = w;
                    std::vector<double> w_full_host;
                    if (use_reduced_basis)
                    {
                        w_full_host.resize(M);
                        space_group_->from_reduced_basis(w, w_full_host.data(), 1);
                        w_full = w_full_host.data();
                    }

                    // Copy field configurations from host to device
                    CUDA_CHECK(cudaMemcpyAsync(
                        d_exp_dw.at(ds_idx).at(monomer_type), w_full,
                        sizeof(double)*M, cudaMemcpyInputToDevice));
                }

                // Compute d_exp_dw
                ker_exp<<<N_BLOCKS, N_THREADS>>>
                    ((double*) d_exp_dw.at(ds_idx).at(monomer_type),
                     (double*) d_exp_dw.at(ds_idx).at(monomer_type),      1.0, -0.50*local_ds, M);
                CUDA_CHECK_LAUNCH();
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());
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
        const bool use_reduced_basis = (space_group_ != nullptr);

        double *d_q_in_full = d_q_in;
        double *d_q_out_full = d_q_out;

        if (use_reduced_basis)
        {
            ker_expand_reduced_basis<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                d_q_full_in_[STREAM], d_q_in, d_full_to_reduced_map_, M);
            CUDA_CHECK_LAUNCH();
            d_q_in_full = d_q_full_in_[STREAM];
            d_q_out_full = d_q_full_out_[STREAM];
        }

        // CN-ADI2: single full step only (2nd order accuracy)
        double *_d_exp_dw = d_exp_dw.at(ds_index).at(monomer_type);

        // Apply exp(-w*local_ds/2) at start
        ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_q_out_full, d_q_in_full, _d_exp_dw, 1.0, M);
        CUDA_CHECK_LAUNCH();

        // Diffusion with full-step coefficients
        if(DIM == 3)
            advance_propagator_3d(this->cb->get_boundary_conditions(), STREAM,
                d_q_out_full, d_q_out_full, monomer_type, ds_index);
        else if(DIM == 2)
            advance_propagator_2d(this->cb->get_boundary_conditions(), STREAM,
                d_q_out_full, d_q_out_full, monomer_type, ds_index);
        else if(DIM == 1)
            advance_propagator_1d(this->cb->get_boundary_conditions(), STREAM,
                d_q_out_full, d_q_out_full, monomer_type, ds_index);

        // Apply exp(-w*local_ds/2) at end
        ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_q_out_full, d_q_out_full, _d_exp_dw, 1.0, M);
        CUDA_CHECK_LAUNCH();

        // Multiply mask
        if (d_q_mask != nullptr)
        {
            double* d_q_mask_full = d_q_mask;
            if (use_reduced_basis)
            {
                ker_expand_reduced_basis<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                    d_q_full_in_[STREAM], d_q_mask, d_full_to_reduced_map_, M);
                CUDA_CHECK_LAUNCH();
                d_q_mask_full = d_q_full_in_[STREAM];
            }
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                d_q_out_full, d_q_out_full, d_q_mask_full, 1.0, M);
            CUDA_CHECK_LAUNCH();
        }

        if (use_reduced_basis)
        {
            ker_reduce_to_basis<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                d_q_out_full, d_q_out, d_reduced_basis_indices_, n_basis_);
            CUDA_CHECK_LAUNCH();
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

        double *_d_xl = d_xl.at(ds_index).at(monomer_type);
        double *_d_xd = d_xd.at(ds_index).at(monomer_type);
        double *_d_xh = d_xh.at(ds_index).at(monomer_type);

        double *_d_yl = d_yl.at(ds_index).at(monomer_type);
        double *_d_yd = d_yd.at(ds_index).at(monomer_type);
        double *_d_yh = d_yh.at(ds_index).at(monomer_type);

        double *_d_zl = d_zl.at(ds_index).at(monomer_type);
        double *_d_zd = d_zd.at(ds_index).at(monomer_type);
        double *_d_zh = d_zh.at(ds_index).at(monomer_type);

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
        CUDA_CHECK_LAUNCH();

        if (bc[0] == BoundaryCondition::PERIODIC)
        {
            tridiagonal_periodic<<<line_blocks(nx[1]*nx[2]), LINE_THREADS, 0, streams[STREAM][0]>>>(
                _d_xl, _d_xd, _d_xh,
                d_c_star[STREAM], d_q_sparse[STREAM], d_temp[STREAM], d_q_star[STREAM],
                d_offset_yz, nx[1]*nx[2], nx[1]*nx[2], nx[0]);
            CUDA_CHECK_LAUNCH();
        }
        else
        {
            tridiagonal<<<line_blocks(nx[1]*nx[2]), LINE_THREADS, shmem_x, streams[STREAM][0]>>>(
                _d_xl, _d_xd, _d_xh,
                d_c_star[STREAM], d_temp[STREAM], d_q_star[STREAM],
                d_offset_yz, nx[1]*nx[2], nx[1]*nx[2], nx[0]);
            CUDA_CHECK_LAUNCH();
        }
        CUDA_CHECK_LAUNCH();

        // Calculate q_dstar
        compute_crank_3d_step_2<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
            bc[2], bc[3], 
            _d_yl, _d_yd, _d_yh, nx[1], nx[2],
            d_temp[STREAM], d_q_star[STREAM], d_q_in, M);
        CUDA_CHECK_LAUNCH();

        if (bc[2] == BoundaryCondition::PERIODIC)
        {
            tridiagonal_periodic<<<line_blocks(nx[0]*nx[2]), LINE_THREADS, 0, streams[STREAM][0]>>>(
                _d_yl, _d_yd, _d_yh,
                d_c_star[STREAM], d_q_sparse[STREAM], d_temp[STREAM], d_q_dstar[STREAM],
                d_offset_xz, nx[0]*nx[2], nx[2], nx[1]);
            CUDA_CHECK_LAUNCH();
        }
        else
        {
            tridiagonal<<<line_blocks(nx[0]*nx[2]), LINE_THREADS, shmem_y, streams[STREAM][0]>>>(
                _d_yl, _d_yd, _d_yh,
                d_c_star[STREAM], d_temp[STREAM], d_q_dstar[STREAM],
                d_offset_xz, nx[0]*nx[2], nx[2], nx[1]);
            CUDA_CHECK_LAUNCH();
        }
        CUDA_CHECK_LAUNCH();

        // Calculate q^(n+1)
        compute_crank_3d_step_3<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
            bc[4], bc[5], 
            _d_zl, _d_zd, _d_zh, nx[1], nx[2],
            d_temp[STREAM], d_q_dstar[STREAM], d_q_in, M);
        CUDA_CHECK_LAUNCH();

        if (bc[4] == BoundaryCondition::PERIODIC)
        {
            tridiagonal_periodic<<<line_blocks(nx[0]*nx[1]), LINE_THREADS, 0, streams[STREAM][0]>>>(
                _d_zl, _d_zd, _d_zh,
                d_c_star[STREAM], d_q_sparse[STREAM], d_temp[STREAM], d_q_out,
                d_offset_xy, nx[0]*nx[1], 1, nx[2]);
            CUDA_CHECK_LAUNCH();
        }
        else
        {
            tridiagonal<<<line_blocks(nx[0]*nx[1]), LINE_THREADS, shmem_z, streams[STREAM][0]>>>(
                _d_zl, _d_zd, _d_zh,
                d_c_star[STREAM], d_temp[STREAM], d_q_out,
                d_offset_xy, nx[0]*nx[1], 1, nx[2]);
            CUDA_CHECK_LAUNCH();
        }
        CUDA_CHECK_LAUNCH();
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

        double *_d_xl = d_xl.at(ds_index).at(monomer_type);
        double *_d_xd = d_xd.at(ds_index).at(monomer_type);
        double *_d_xh = d_xh.at(ds_index).at(monomer_type);

        double *_d_yl = d_yl.at(ds_index).at(monomer_type);
        double *_d_yd = d_yd.at(ds_index).at(monomer_type);
        double *_d_yh = d_yh.at(ds_index).at(monomer_type);

        // Shared memory size for 3 arrays of doubles
        size_t shmem_x = 3 * nx[0] * sizeof(double);
        size_t shmem_y = 3 * nx[1] * sizeof(double);

        // Calculate q_star
        compute_crank_2d_step_1<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
            bc[0], bc[1], bc[2], bc[3],
            _d_xl, _d_xd, _d_xh, nx[0],
            _d_yl, _d_yd, _d_yh, nx[1],
            d_temp[STREAM], d_q_in, M);
        CUDA_CHECK_LAUNCH();

        // CUDA_CHECK(cudaMemcpy(d_q_out, d_q_star, sizeof(double)*M, cudaMemcpyDeviceToDevice));

        if (bc[0] == BoundaryCondition::PERIODIC)
        {
            tridiagonal_periodic<<<line_blocks(nx[1]), LINE_THREADS, 0, streams[STREAM][0]>>>(
                _d_xl, _d_xd, _d_xh,
                d_c_star[STREAM], d_q_sparse[STREAM], d_temp[STREAM], d_q_star[STREAM],
                d_offset_y, nx[1], nx[1], nx[0]);
            CUDA_CHECK_LAUNCH();
        }
        else
        {
            tridiagonal<<<line_blocks(nx[1]), LINE_THREADS, shmem_x, streams[STREAM][0]>>>(
                _d_xl, _d_xd, _d_xh,
                d_c_star[STREAM], d_temp[STREAM], d_q_star[STREAM],
                d_offset_y, nx[1], nx[1], nx[0]);
            CUDA_CHECK_LAUNCH();
        }
        CUDA_CHECK_LAUNCH();

        // Calculate q_dstar
        compute_crank_2d_step_2<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
            bc[2], bc[3],
            _d_yl, _d_yd, _d_yh, nx[1],
            d_temp[STREAM], d_q_star[STREAM], d_q_in, M);
        CUDA_CHECK_LAUNCH();

        if (bc[2] == BoundaryCondition::PERIODIC)
        {
            tridiagonal_periodic<<<line_blocks(nx[0]), LINE_THREADS, 0, streams[STREAM][0]>>>(
                _d_yl, _d_yd, _d_yh,
                d_c_star[STREAM], d_q_sparse[STREAM], d_temp[STREAM], d_q_out,
                d_offset_x, nx[0], 1, nx[1]);
            CUDA_CHECK_LAUNCH();
        }
        else
        {
            tridiagonal<<<line_blocks(nx[0]), LINE_THREADS, shmem_y, streams[STREAM][0]>>>(
                _d_yl, _d_yd, _d_yh,
                d_c_star[STREAM], d_temp[STREAM], d_q_out,
                d_offset_x, nx[0], 1, nx[1]);
            CUDA_CHECK_LAUNCH();
        }
        CUDA_CHECK_LAUNCH();
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

        double *_d_xl = d_xl.at(ds_index).at(monomer_type);
        double *_d_xd = d_xd.at(ds_index).at(monomer_type);
        double *_d_xh = d_xh.at(ds_index).at(monomer_type);

        // Shared memory size for 3 arrays of doubles
        size_t shmem_x = 3 * nx[0] * sizeof(double);

        compute_crank_1d<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
            bc[0], bc[1],
            _d_xl, _d_xd, _d_xh,
            d_q_star[STREAM], d_q_in, nx[0]);
        CUDA_CHECK_LAUNCH();

        if (bc[0] == BoundaryCondition::PERIODIC)
            tridiagonal_periodic<<<line_blocks(1), LINE_THREADS, 0, streams[STREAM][0]>>>(
                _d_xl, _d_xd, _d_xh,
                d_c_star[STREAM], d_q_sparse[STREAM], d_q_star[STREAM], d_q_out,
                d_offset, 1, 1, nx[0]);
        else
            tridiagonal<<<line_blocks(1), LINE_THREADS, shmem_x, streams[STREAM][0]>>>(
                _d_xl, _d_xd, _d_xh,
                d_c_star[STREAM], d_q_star[STREAM], d_q_out, d_offset, 1, 1, nx[0]);
        CUDA_CHECK_LAUNCH();
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
