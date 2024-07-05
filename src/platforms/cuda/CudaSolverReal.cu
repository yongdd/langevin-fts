#include <iostream>
#include <cmath>
#include "CudaSolverReal.h"

CudaSolverReal::CudaSolverReal(
    ComputationBox *cb,
    Molecules *molecules,
    int n_streams,
    cudaStream_t streams[MAX_STREAMS][2],
    bool reduce_gpu_memory_usage)
{
    try{
        this->cb = cb;
        this->molecules = molecules;
        this->chain_model = molecules->get_model_name();
        this->n_streams = n_streams;
        this->reduce_gpu_memory_usage = reduce_gpu_memory_usage;

        if(molecules->get_model_name() != "continuous")
            throw_with_line_number("Real-space method only support 'continuous' chain model.");     

        const int M = cb->get_n_grid();
        const int N_GPUS = CudaCommon::get_instance().get_n_gpus();
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

        // Create boltz_bond, boltz_bond_half, exp_dw, and exp_dw_half
        for(const auto& item: molecules->get_bond_lengths())
        {
            std::string monomer_type = item.first;
            for(int gpu=0; gpu<N_GPUS; gpu++)
            {
                gpu_error_check(cudaSetDevice(gpu));
                d_exp_dw     [gpu][monomer_type] = nullptr;
                d_exp_dw_half[gpu][monomer_type] = nullptr;

                gpu_error_check(cudaMalloc((void**)&d_exp_dw     [gpu][monomer_type], sizeof(double)*M));
                gpu_error_check(cudaMalloc((void**)&d_exp_dw_half[gpu][monomer_type], sizeof(double)*M));

                d_xl[gpu][monomer_type] = nullptr;
                d_xd[gpu][monomer_type] = nullptr;
                d_xh[gpu][monomer_type] = nullptr;
                gpu_error_check(cudaMalloc((void**)&d_xl[gpu][monomer_type], sizeof(double)*nx[0]));
                gpu_error_check(cudaMalloc((void**)&d_xd[gpu][monomer_type], sizeof(double)*nx[0]));
                gpu_error_check(cudaMalloc((void**)&d_xh[gpu][monomer_type], sizeof(double)*nx[0]));

                d_yl[gpu][monomer_type] = nullptr;
                d_yd[gpu][monomer_type] = nullptr;
                d_yh[gpu][monomer_type] = nullptr;
                gpu_error_check(cudaMalloc((void**)&d_yl[gpu][monomer_type], sizeof(double)*nx[1]));
                gpu_error_check(cudaMalloc((void**)&d_yd[gpu][monomer_type], sizeof(double)*nx[1]));
                gpu_error_check(cudaMalloc((void**)&d_yh[gpu][monomer_type], sizeof(double)*nx[1]));

                d_zl[gpu][monomer_type] = nullptr;
                d_zd[gpu][monomer_type] = nullptr;
                d_zh[gpu][monomer_type] = nullptr;
                gpu_error_check(cudaMalloc((void**)&d_zl[gpu][monomer_type], sizeof(double)*nx[2]));
                gpu_error_check(cudaMalloc((void**)&d_zd[gpu][monomer_type], sizeof(double)*nx[2]));
                gpu_error_check(cudaMalloc((void**)&d_zh[gpu][monomer_type], sizeof(double)*nx[2]));
            }
        }

        if(DIM == 3)
        {
            for(int i=0; i<n_streams; i++)
            {

                gpu_error_check(cudaSetDevice(i % N_GPUS));
                gpu_error_check(cudaMalloc((void**)&d_q_star[i], sizeof(double)*M));
                gpu_error_check(cudaMalloc((void**)&d_q_dstar[i], sizeof(double)*M));
                gpu_error_check(cudaMalloc((void**)&d_c_star[i], sizeof(double)*M));
                gpu_error_check(cudaMalloc((void**)&d_q_sparse[i], sizeof(double)*M));
                gpu_error_check(cudaMalloc((void**)&d_temp[i], sizeof(double)*M));
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

            for(int gpu=0; gpu<N_GPUS; gpu++)
            {
                gpu_error_check(cudaSetDevice(gpu));
                gpu_error_check(cudaMalloc((void**)&d_offset_xy[gpu], sizeof(int)*nx[0]*nx[1]));
                gpu_error_check(cudaMalloc((void**)&d_offset_yz[gpu], sizeof(int)*nx[1]*nx[2]));
                gpu_error_check(cudaMalloc((void**)&d_offset_xz[gpu], sizeof(int)*nx[0]*nx[2]));

                gpu_error_check(cudaMemcpy(d_offset_xy[gpu], offset_xy, sizeof(int)*nx[0]*nx[1], cudaMemcpyHostToDevice));
                gpu_error_check(cudaMemcpy(d_offset_yz[gpu], offset_yz, sizeof(int)*nx[1]*nx[2], cudaMemcpyHostToDevice));
                gpu_error_check(cudaMemcpy(d_offset_xz[gpu], offset_xz, sizeof(int)*nx[0]*nx[2], cudaMemcpyHostToDevice));
            }
        }
        else if(DIM == 2)
        {
            for(int i=0; i<n_streams; i++)
            {
                gpu_error_check(cudaSetDevice(i % N_GPUS));
                gpu_error_check(cudaMalloc((void**)&d_q_star[i], sizeof(double)*M));
                gpu_error_check(cudaMalloc((void**)&d_c_star[i], sizeof(double)*M));
                gpu_error_check(cudaMalloc((void**)&d_q_sparse[i], sizeof(double)*M));
                gpu_error_check(cudaMalloc((void**)&d_temp[i], sizeof(double)*M));
            }

            int offset_x[nx[0]];
            int offset_y[nx[1]];

            for(int i=0;i<nx[0];i++)
                offset_x[i] = i*nx[1];

            for(int j=0;j<nx[1];j++)
                offset_y[j] = j;

            for(int gpu=0; gpu<N_GPUS; gpu++)
            {
                gpu_error_check(cudaSetDevice(gpu));
                gpu_error_check(cudaMalloc((void**)&d_offset_x[gpu], sizeof(int)*nx[0]));
                gpu_error_check(cudaMalloc((void**)&d_offset_y[gpu], sizeof(int)*nx[1]));

                gpu_error_check(cudaMemcpy(d_offset_x[gpu], offset_x, sizeof(int)*nx[0], cudaMemcpyHostToDevice));
                gpu_error_check(cudaMemcpy(d_offset_y[gpu], offset_y, sizeof(int)*nx[1], cudaMemcpyHostToDevice));
            }
        }
        else if(DIM == 1)
        {
            for(int i=0; i<n_streams; i++)
            {
                gpu_error_check(cudaSetDevice(i % N_GPUS));
                gpu_error_check(cudaMalloc((void**)&d_q_star[i], sizeof(double)*M));
                gpu_error_check(cudaMalloc((void**)&d_c_star[i], sizeof(double)*M));
                gpu_error_check(cudaMalloc((void**)&d_q_sparse[i], sizeof(double)*M));
            }

            for(int gpu=0; gpu<N_GPUS; gpu++)
            {
                gpu_error_check(cudaSetDevice(gpu));
                gpu_error_check(cudaMalloc((void**)&d_offset[gpu], sizeof(int)));
                gpu_error_check(cudaMemset(d_offset[gpu], 0, sizeof(int)));
            }
        }

        update_laplacian_operator();
        gpu_error_check(cudaSetDevice(0));
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
CudaSolverReal::~CudaSolverReal()
{
    const int N_GPUS = CudaCommon::get_instance().get_n_gpus();
    const int DIM = this->dim;

    for(int gpu=0; gpu<N_GPUS; gpu++)
    {
        for(const auto& item: d_exp_dw[gpu])
            cudaFree(item.second);
        for(const auto& item: d_exp_dw_half[gpu])
            cudaFree(item.second);
        
        for(const auto& item: d_xl[gpu])
            cudaFree(item.second);
        for(const auto& item: d_xd[gpu])
            cudaFree(item.second);
        for(const auto& item: d_xh[gpu])
            cudaFree(item.second);

        for(const auto& item: d_yl[gpu])
            cudaFree(item.second);
        for(const auto& item: d_yd[gpu])
            cudaFree(item.second);
        for(const auto& item: d_yh[gpu])
            cudaFree(item.second);

        for(const auto& item: d_zl[gpu])
            cudaFree(item.second);
        for(const auto& item: d_zd[gpu])
            cudaFree(item.second);
        for(const auto& item: d_zh[gpu])
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
            }
            cudaFree(d_offset_xy[gpu]);
            cudaFree(d_offset_yz[gpu]);
            cudaFree(d_offset_xz[gpu]);
        }
        else if(DIM == 2)
        {
            for(int i=0; i<n_streams; i++)
            {
                cudaFree(d_q_star[i]);
                cudaFree(d_c_star[i]);
                cudaFree(d_q_sparse[i]);
                cudaFree(d_temp[i]);
            }
            cudaFree(d_offset_x[gpu]);
            cudaFree(d_offset_y[gpu]);
        }
        else if(DIM == 1)
        {
            for(int i=0; i<n_streams; i++)
            {
                cudaFree(d_q_star[i]);
                cudaFree(d_c_star[i]);
                cudaFree(d_q_sparse[i]);
            }
            cudaFree(d_offset[gpu]);
        }
    }
}
void CudaSolverReal::update_laplacian_operator()
{
    try
    {
        const int M = cb->get_n_grid();
        const int N_GPUS = CudaCommon::get_instance().get_n_gpus();
        const int DIM = cb->get_dim();
        std::vector<int> nx(DIM);
        if (DIM == 3)
            nx = {cb->get_nx(0), cb->get_nx(1), cb->get_nx(2)};
        else if (DIM == 2)
            nx = {cb->get_nx(0), cb->get_nx(1), 1};
        else if (DIM == 1)
            nx = {cb->get_nx(0), 1, 1};

        double xl[nx[0]], xd[nx[0]], xh[nx[0]];
        double yl[nx[1]], yd[nx[1]], yh[nx[1]];
        double zl[nx[2]], zd[nx[2]], zh[nx[2]];

        for(const auto& item: molecules->get_bond_lengths())
        {
            std::string monomer_type = item.first;
            double bond_length_sq = item.second*item.second;

            FiniteDifference::get_laplacian_matrix(
                cb->get_boundary_conditions(),
                cb->get_nx(), cb->get_dx(),
                xl, xd, xh,
                yl, yd, yh,
                zl, zd, zh,
                bond_length_sq, molecules->get_ds());

            for(int gpu=0; gpu<N_GPUS; gpu++)
            {
                gpu_error_check(cudaSetDevice(gpu));
                gpu_error_check(cudaMemcpy(d_xl[gpu][monomer_type], xl, sizeof(double)*nx[0], cudaMemcpyHostToDevice));
                gpu_error_check(cudaMemcpy(d_xd[gpu][monomer_type], xd, sizeof(double)*nx[0], cudaMemcpyHostToDevice));
                gpu_error_check(cudaMemcpy(d_xh[gpu][monomer_type], xh, sizeof(double)*nx[0], cudaMemcpyHostToDevice));

                gpu_error_check(cudaMemcpy(d_yl[gpu][monomer_type], yl, sizeof(double)*nx[1], cudaMemcpyHostToDevice));
                gpu_error_check(cudaMemcpy(d_yd[gpu][monomer_type], yd, sizeof(double)*nx[1], cudaMemcpyHostToDevice));
                gpu_error_check(cudaMemcpy(d_yh[gpu][monomer_type], yh, sizeof(double)*nx[1], cudaMemcpyHostToDevice));

                gpu_error_check(cudaMemcpy(d_zl[gpu][monomer_type], zl, sizeof(double)*nx[2], cudaMemcpyHostToDevice));
                gpu_error_check(cudaMemcpy(d_zd[gpu][monomer_type], zd, sizeof(double)*nx[2], cudaMemcpyHostToDevice));
                gpu_error_check(cudaMemcpy(d_zh[gpu][monomer_type], zh, sizeof(double)*nx[2], cudaMemcpyHostToDevice));
            }
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaSolverReal::update_dw(std::string device, std::map<std::string, const double*> w_input)
{
    try{
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();
        const int N_GPUS = CudaCommon::get_instance().get_n_gpus();

        const int M = cb->get_n_grid();
        const double ds = molecules->get_ds();

        for(const auto& item: w_input)
        {
            if( d_exp_dw[0].find(item.first) == d_exp_dw[0].end())
                throw_with_line_number("monomer_type \"" + item.first + "\" is not in d_exp_dw.");     
        }

        cudaMemcpyKind cudaMemcpyInputToDevice;
        if (device == "gpu")
            cudaMemcpyInputToDevice = cudaMemcpyDeviceToDevice;
        else if(device == "cpu")
            cudaMemcpyInputToDevice = cudaMemcpyHostToDevice;
        else
        {
            throw_with_line_number("Invalid device \"" + device + "\".");
        }

        // Compute exp_dw and exp_dw_half
        for(const auto& item: w_input)
        {
            std::string monomer_type = item.first;
            const double *w = item.second;

            // Copy field configurations from host to device
            for(int gpu=0; gpu<N_GPUS; gpu++)
            {
                gpu_error_check(cudaSetDevice(gpu));
                gpu_error_check(cudaMemcpyAsync(
                    d_exp_dw     [gpu][monomer_type], w,      
                    sizeof(double)*M, cudaMemcpyInputToDevice, streams[gpu][1]));
                gpu_error_check(cudaMemcpyAsync(
                    d_exp_dw_half[gpu][monomer_type], w,
                    sizeof(double)*M, cudaMemcpyInputToDevice, streams[gpu][1]));
            }

            // Compute d_exp_dw and d_exp_dw_half
            for(int gpu=0; gpu<N_GPUS; gpu++)
            {
                gpu_error_check(cudaSetDevice(gpu));
                exp_real<<<N_BLOCKS, N_THREADS, 0, streams[gpu][1]>>>
                    (d_exp_dw[gpu][monomer_type],      d_exp_dw[gpu][monomer_type],      1.0, -0.50*ds, M);
                exp_real<<<N_BLOCKS, N_THREADS, 0, streams[gpu][1]>>>
                    (d_exp_dw_half[gpu][monomer_type], d_exp_dw_half[gpu][monomer_type], 1.0, -0.25*ds, M);
            }

            // Synchronize all GPUs
            for(int gpu=0; gpu<N_GPUS; gpu++)
            {
                gpu_error_check(cudaSetDevice(gpu));
                gpu_error_check(cudaDeviceSynchronize());
            }
        }
        gpu_error_check(cudaSetDevice(0));
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaSolverReal::advance_propagator_continuous(
    const int GPU, const int STREAM,
    double *d_q_in, double *d_q_out,
    std::string monomer_type, double *d_q_mask) 
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = cb->get_n_grid();
        const int DIM = cb->get_dim();

        double *_d_exp_dw = d_exp_dw[GPU][monomer_type];

        // Evaluate exp(-w*ds/2) in real space
        multi_real<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_q_out, d_q_in, _d_exp_dw, 1.0, M);

        if(DIM == 3)           // input, output
            advance_propagator_3d(cb->get_boundary_conditions(), GPU, STREAM, d_q_out, d_q_out, monomer_type);
        else if(DIM == 2)
            advance_propagator_2d(cb->get_boundary_conditions(), GPU, STREAM, d_q_out, d_q_out, monomer_type);
        else if(DIM ==1 )
            advance_propagator_1d(cb->get_boundary_conditions(), GPU, STREAM, d_q_out, d_q_out, monomer_type);

        // Evaluate exp(-w*ds/2) in real space
        multi_real<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_q_out, d_q_out, _d_exp_dw, 1.0, M);

        // Multiply mask
        if (d_q_mask != nullptr)
            multi_real<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_q_out, d_q_out, d_q_mask, 1.0, M);
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaSolverReal::advance_propagator_3d(
    std::vector<BoundaryCondition> bc,
    const int GPU, const int STREAM,
    double *d_q_in, double *d_q_out, std::string monomer_type)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();
        const int M = cb->get_n_grid();
        const std::vector<int> nx = cb->get_nx();

        double *_d_xl = d_xl[GPU][monomer_type];
        double *_d_xd = d_xd[GPU][monomer_type];
        double *_d_xh = d_xh[GPU][monomer_type];

        double *_d_yl = d_yl[GPU][monomer_type];
        double *_d_yd = d_yd[GPU][monomer_type];
        double *_d_yh = d_yh[GPU][monomer_type];

        double *_d_zl = d_zl[GPU][monomer_type];
        double *_d_zd = d_zd[GPU][monomer_type];
        double *_d_zh = d_zh[GPU][monomer_type];

        // Calculate q_star
        compute_crank_3d_step_1<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
            bc[0], bc[1], bc[2], bc[3], bc[4], bc[5], 
            _d_xl, _d_xd, _d_xh, nx[0],
            _d_yl, _d_yd, _d_yh, nx[1],
            _d_zl, _d_zd, _d_zh, nx[2],
            d_temp[STREAM], d_q_in, M);

        if (bc[0] == BoundaryCondition::PERIODIC)
        {
            tridiagonal_periodic<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                _d_xl, _d_xd, _d_xh,
                d_c_star[STREAM], d_q_sparse[STREAM], d_temp[STREAM], d_q_star[STREAM],
                d_offset_yz[GPU], nx[1]*nx[2], nx[1]*nx[2], nx[0]);
        }
        else
        {
            tridiagonal<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                _d_xl, _d_xd, _d_xh,
                d_c_star[STREAM], d_temp[STREAM], d_q_star[STREAM],
                d_offset_yz[GPU], nx[1]*nx[2], nx[1]*nx[2], nx[0]);
        }

        // Calculate q_dstar
        compute_crank_3d_step_2<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
            bc[2], bc[3], 
            _d_yl, _d_yd, _d_yh, nx[1], nx[2],
            d_temp[STREAM], d_q_star[STREAM], d_q_in, M);

        if (bc[2] == BoundaryCondition::PERIODIC)
        {
            tridiagonal_periodic<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                _d_yl, _d_yd, _d_yh,
                d_c_star[STREAM], d_q_sparse[STREAM], d_temp[STREAM], d_q_dstar[STREAM],
                d_offset_xz[GPU], nx[0]*nx[2], nx[2], nx[1]);
        }
        else
        {
            tridiagonal<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                _d_yl, _d_yd, _d_yh,
                d_c_star[STREAM], d_temp[STREAM], d_q_dstar[STREAM],
                d_offset_xz[GPU], nx[0]*nx[2], nx[2], nx[1]);
        }

        // Calculate q^(n+1)
        compute_crank_3d_step_3<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
            bc[4], bc[5], 
            _d_zl, _d_zd, _d_zh, nx[1], nx[2],
            d_temp[STREAM], d_q_dstar[STREAM], d_q_in, M);

        if (bc[4] == BoundaryCondition::PERIODIC)
        {
            tridiagonal_periodic<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                _d_zl, _d_zd, _d_zh,
                d_c_star[STREAM], d_q_sparse[STREAM], d_temp[STREAM], d_q_out,
                d_offset_xy[GPU], nx[0]*nx[1], 1, nx[2]);
        }
        else
        {
            tridiagonal<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                _d_zl, _d_zd, _d_zh,
                d_c_star[STREAM], d_temp[STREAM], d_q_out,
                d_offset_xy[GPU], nx[0]*nx[1], 1, nx[2]);
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaSolverReal::advance_propagator_2d(
    std::vector<BoundaryCondition> bc,
    const int GPU, const int STREAM,
    double *d_q_in, double *d_q_out, std::string monomer_type)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();
        const int M = cb->get_n_grid();
        const std::vector<int> nx = cb->get_nx();

        double *_d_xl = d_xl[GPU][monomer_type];
        double *_d_xd = d_xd[GPU][monomer_type];
        double *_d_xh = d_xh[GPU][monomer_type];

        double *_d_yl = d_yl[GPU][monomer_type];
        double *_d_yd = d_yd[GPU][monomer_type];
        double *_d_yh = d_yh[GPU][monomer_type];

        // Calculate q_star
        compute_crank_2d_step_1<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
            bc[0], bc[1], bc[2], bc[3],
            _d_xl, _d_xd, _d_xh, nx[0],
            _d_yl, _d_yd, _d_yh, nx[1],
            d_temp[STREAM], d_q_in, M);

        // gpu_error_check(cudaMemcpy(d_q_out, d_q_star, sizeof(double)*M, cudaMemcpyDeviceToDevice));

        if (bc[0] == BoundaryCondition::PERIODIC)
        {
            tridiagonal_periodic<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                _d_xl, _d_xd, _d_xh,
                d_c_star[STREAM], d_q_sparse[STREAM], d_temp[STREAM], d_q_star[STREAM],
                d_offset_y[GPU], nx[1], nx[1], nx[0]);
        }
        else
        {
            tridiagonal<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                _d_xl, _d_xd, _d_xh,
                d_c_star[STREAM], d_temp[STREAM], d_q_star[STREAM],
                d_offset_y[GPU], nx[1], nx[1], nx[0]);
        }

        // Calculate q_dstar
        compute_crank_2d_step_2<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
            bc[2], bc[3],
            _d_yl, _d_yd, _d_yh, nx[1],
            d_temp[STREAM], d_q_star[STREAM], d_q_in, M);

        if (bc[2] == BoundaryCondition::PERIODIC)
        {
            tridiagonal_periodic<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                _d_yl, _d_yd, _d_yh,
                d_c_star[STREAM], d_q_sparse[STREAM], d_temp[STREAM], d_q_out,
                d_offset_x[GPU], nx[0], 1, nx[1]);
        }
        else
        {
            tridiagonal<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                _d_yl, _d_yd, _d_yh,
                d_c_star[STREAM], d_temp[STREAM], d_q_out,
                d_offset_x[GPU], nx[0], 1, nx[1]);
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaSolverReal::advance_propagator_1d(
    std::vector<BoundaryCondition> bc,
    const int GPU, const int STREAM,
    double *d_q_in, double *d_q_out, std::string monomer_type)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();
        const int M = cb->get_n_grid();
        const std::vector<int> nx = cb->get_nx();

        double *_d_xl = d_xl[GPU][monomer_type];
        double *_d_xd = d_xd[GPU][monomer_type];
        double *_d_xh = d_xh[GPU][monomer_type];

        compute_crank_1d<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
            bc[0], bc[1],
            _d_xl, _d_xd, _d_xh,
            d_q_star[STREAM], d_q_in, nx[0]);

        if (bc[0] == BoundaryCondition::PERIODIC)
            tridiagonal_periodic<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                _d_xl, _d_xd, _d_xh,
                d_c_star[STREAM], d_q_sparse[STREAM], d_q_star[STREAM], d_q_out,
                d_offset[GPU], 1, 1, nx[0]);
        else
            tridiagonal<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                _d_xl, _d_xd, _d_xh,
                d_c_star[STREAM], d_q_star[STREAM], d_q_out, d_offset[GPU], 1, 1, nx[0]);
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaSolverReal::compute_single_segment_stress_continuous(
        const int GPU, const int STREAM,
        double *d_q_pair, double *d_segment_stress, std::string monomer_type)
{
    try
    {
        throw_with_line_number("Currently, the real-space method does not support stress computation.");
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
__global__ void tridiagonal(
    const double *d_xl, const double *d_xd, const double *d_xh,
    double *d_c_star,  const double *d_d, double *d_x,
    const int *d_offset, const int REPEAT,
    const int INTERVAL, const int M)
{
    // d_xl: a
    // d_xd: b
    // d_xh: c

    double temp;

    int n = blockIdx.x * blockDim.x + threadIdx.x;
    while (n < REPEAT)
    {
        const double *_d_d = &d_d[d_offset[n]];
        double       *_d_x = &d_x[d_offset[n]];
        double  *_d_c_star = &d_c_star[d_offset[n]];

        // Forward sweep
        temp = d_xd[0];
        _d_c_star[0] = d_xh[0]/d_xd[0];
        _d_x[0] = _d_d[0]/d_xd[0];

        for(int i=1; i<M; i++)
        {
            _d_c_star[(i-1)*INTERVAL] = d_xh[i-1]/temp;
            temp = d_xd[i]-d_xl[i]*_d_c_star[(i-1)*INTERVAL];
            _d_x[i*INTERVAL] = (_d_d[i*INTERVAL]-d_xl[i]*_d_x[(i-1)*INTERVAL])/temp;
        }

        // Backward substitution
        for(int i=M-2;i>=0; i--)
            _d_x[i*INTERVAL] = _d_x[i*INTERVAL] - _d_c_star[i*INTERVAL]*_d_x[(i+1)*INTERVAL];
        
        n += blockDim.x * gridDim.x;
    }
}

// This method solves CX=Y, where C is a near-tridiagonal matrix with periodic boundary condition
__global__ void tridiagonal_periodic(
    const double *d_xl, const double *d_xd, const double *d_xh,
    double *d_c_star, double *d_q_sparse, 
    const double *d_d, double *d_x,
    const int *d_offset, const int REPEAT,
    const int INTERVAL, const int M)
{
    // xl: a
    // xd: b
    // xh: c
    // gamma = 1.0

    double temp, value;

    int n = blockIdx.x * blockDim.x + threadIdx.x;
    while (n < REPEAT)
    {
        const double *_d_d = &d_d[d_offset[n]];
        double       *_d_x = &d_x[d_offset[n]];
        double  *_d_c_star = &d_c_star[d_offset[n]];
        double *_d_q_sparse = &d_q_sparse[d_offset[n]];

        // Forward sweep
        temp = d_xd[0] - 1.0 ;
        _d_c_star[0] = d_xh[0]/temp;
        _d_x[0] = _d_d[0]/temp;
        _d_q_sparse[0] =  1.0/temp;

        for(int i=1; i<M-1; i++)
        {
            _d_c_star[(i-1)*INTERVAL] = d_xh[i-1]/temp;
            temp = d_xd[i]-d_xl[i]*_d_c_star[(i-1)*INTERVAL];
            _d_x[i*INTERVAL] = (_d_d[i*INTERVAL]-d_xl[i]*_d_x[(i-1)*INTERVAL])/temp;
            _d_q_sparse[i*INTERVAL] =   (-d_xl[i]*_d_q_sparse[(i-1)*INTERVAL])/temp;
        }

        _d_c_star[(M-2)*INTERVAL] = d_xh[M-2]/temp;
        temp = d_xd[M-1]-d_xh[M-1]*d_xl[0] - d_xl[M-1]*_d_c_star[(M-2)*INTERVAL];
        _d_x[(M-1)*INTERVAL] =    (_d_d[(M-1)*INTERVAL]-d_xl[M-1]*_d_x[(M-2)*INTERVAL])/temp;
        _d_q_sparse[(M-1)*INTERVAL] = (d_xh[M-1]-d_xl[M-1]*_d_q_sparse[(M-2)*INTERVAL])/temp;

        // Backward substitution
        for(int i=M-2;i>=0; i--)
        {
            _d_x[i*INTERVAL] = _d_x[i*INTERVAL] - _d_c_star[i*INTERVAL]*_d_x[(i+1)*INTERVAL];
            _d_q_sparse[i*INTERVAL] = _d_q_sparse[i*INTERVAL] - _d_c_star[i*INTERVAL]*_d_q_sparse[(i+1)*INTERVAL];
        }

        value = (_d_x[0]+d_xl[0]*_d_x[(M-1)*INTERVAL])/(1.0+_d_q_sparse[0]+d_xl[0]*_d_q_sparse[(M-1)*INTERVAL]);
        for(int i=0; i<M; i++)
            _d_x[i*INTERVAL] = _d_x[i*INTERVAL] - _d_q_sparse[i*INTERVAL]*value;

        n += blockDim.x * gridDim.x;
    }
}