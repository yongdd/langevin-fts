#define THRUST_IGNORE_DEPRECATED_CPP_DIALECT
#define CUB_IGNORE_DEPRECATED_CPP_DIALECT

#include <complex>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <iostream>
#include "CudaPseudoDiscrete.h"
#include "CudaComputationBox.h"

CudaPseudoDiscrete::CudaPseudoDiscrete(
    ComputationBox *cb,
    PolymerChain *pc)
    : Pseudo(cb, pc)
{
    try
    {
        const int M = cb->get_n_grid();
        const int N_B = pc->get_n_block();
        const int N = pc->get_n_segment_total();
        const int M_COMPLEX = this->n_complex_grid;

        // Create FFT plan
        const int BATCH{2};
        const int NRANK{cb->get_dim()};
        int n_grid[NRANK];

        this->n_block = N_B;
        if(cb->get_dim() == 3)
        {
            n_grid[0] = cb->get_nx(0);
            n_grid[1] = cb->get_nx(1);
            n_grid[2] = cb->get_nx(2);
        }
        else if(cb->get_dim() == 2)
        {
            n_grid[0] = cb->get_nx(1);
            n_grid[1] = cb->get_nx(2);
        }
        else if(cb->get_dim() == 1)
        {
            n_grid[0] = cb->get_nx(2);
        }
        cufftPlanMany(&plan_for, NRANK, n_grid, NULL, 1, 0, NULL, 1, 0, CUFFT_D2Z,BATCH);
        cufftPlanMany(&plan_bak, NRANK, n_grid, NULL, 1, 0, NULL, 1, 0, CUFFT_Z2D,BATCH);

        // Memory allocation
        this->d_exp_dw = new double*[N_B];
        this->d_boltz_bond = new double*[N_B];
        if (N_B>1)
            this->d_boltz_bond_middle = new double*[N_B-1];

        gpu_error_check(cudaMalloc((void**)&d_k_q_in, sizeof(ftsComplex)*2*M_COMPLEX));
        gpu_error_check(cudaMalloc((void**)&d_q,      sizeof(double)*2*M*N));
        gpu_error_check(cudaMalloc((void**)&d_phi, sizeof(double)*M*N_B));

        for (int b=0; b<N_B; b++)
        {
            gpu_error_check(cudaMalloc((void**)&d_boltz_bond[b], sizeof(double)*M_COMPLEX));
            gpu_error_check(cudaMalloc((void**)&d_exp_dw[b],   sizeof(double)*M));
        }
        for (int b=0; b<N_B-1; b++)
        {
            gpu_error_check(cudaMalloc((void**)&d_boltz_bond_middle[b], sizeof(double)*M_COMPLEX));
        }
        update();
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
CudaPseudoDiscrete::~CudaPseudoDiscrete()
{
    const int N_B = n_block;
    cufftDestroy(plan_for);
    cufftDestroy(plan_bak);

    cudaFree(d_k_q_in);
    cudaFree(d_q);
    cudaFree(d_phi);

    for (int b=0; b<N_B; b++)
    {
        cudaFree(d_boltz_bond[b]);
        cudaFree(d_exp_dw[b]);
    }
    for (int b=0; b<N_B-1; b++)
    {
        cudaFree(d_boltz_bond_middle[b]);
    }

    delete[] d_exp_dw, d_boltz_bond;
    if (N_B>1)
        delete[] d_boltz_bond_middle;
}

void CudaPseudoDiscrete::update()
{
    try
    {
        double bond_length_middle;
        const int N_B = pc->get_n_block();

        const int M_COMPLEX = this->n_complex_grid;
        double boltz_bond[N_B][M_COMPLEX], boltz_bond_middle[N_B][M_COMPLEX];

        get_boltz_bond(boltz_bond[0],  pc->get_bond_length(0),  cb->get_nx(), cb->get_dx(), pc->get_ds());
        gpu_error_check(cudaMemcpy(d_boltz_bond[0],  boltz_bond[0],  sizeof(double)*M_COMPLEX,cudaMemcpyHostToDevice));
        for (int b=0; b<N_B-1; b++)
        {
            bond_length_middle = 0.5*pc->get_bond_length(b) + 0.5*pc->get_bond_length(b+1);
            get_boltz_bond(boltz_bond[b+1],  pc->get_bond_length(b+1),  cb->get_nx(), cb->get_dx(), pc->get_ds());
            get_boltz_bond(boltz_bond_middle[b],  bond_length_middle,  cb->get_nx(), cb->get_dx(), pc->get_ds());
            gpu_error_check(cudaMemcpy(d_boltz_bond[b+1],  boltz_bond[b+1],  sizeof(double)*M_COMPLEX,cudaMemcpyHostToDevice));
            gpu_error_check(cudaMemcpy(d_boltz_bond_middle[b],  boltz_bond_middle[b],  sizeof(double)*M_COMPLEX,cudaMemcpyHostToDevice));
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

std::array<double,3> CudaPseudoDiscrete::dq_dl()
{
    // This method should be invoked after invoking compute_statistics().

    // To calculate stress, we multiply weighted fourier basis to q(k)*q^dagger(-k).
    // We only need the real part of stress calculation.
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int DIM  = cb->get_dim();
        const int M    = cb->get_n_grid();
        const int N_B = pc->get_n_block();
        const int N    = pc->get_n_segment_total();
        const int M_COMPLEX = this->n_complex_grid;
        const std::vector<int> seg_start= pc->get_block_start();

        // const double bond_length_a = eps*eps/(f*eps*eps + (1.0-f));
        // const double bond_length_b = 1.0/(f*eps*eps + (1.0-f));
        // const double bond_length_ab = 0.5*bond_length_a + 0.5*bond_length_b;
        double bond_length_now, *d_boltz_bond_now;

        std::array<double,3> dq_dl;
        double fourier_basis_x[M_COMPLEX];
        double fourier_basis_y[M_COMPLEX];
        double fourier_basis_z[M_COMPLEX];

        double *d_fourier_basis_x;
        double *d_fourier_basis_y;
        double *d_fourier_basis_z;
        double *d_q_in_2m, *d_q_multi, *d_stress_sum;

        get_weighted_fourier_basis(fourier_basis_x, fourier_basis_y, fourier_basis_z, cb->get_nx(), cb->get_dx());

        gpu_error_check(cudaMalloc((void**)&d_fourier_basis_x, sizeof(double)*M_COMPLEX));
        gpu_error_check(cudaMalloc((void**)&d_fourier_basis_y, sizeof(double)*M_COMPLEX));
        gpu_error_check(cudaMalloc((void**)&d_fourier_basis_z, sizeof(double)*M_COMPLEX));
        gpu_error_check(cudaMalloc((void**)&d_q_in_2m,         sizeof(double)*2*M));
        gpu_error_check(cudaMalloc((void**)&d_q_multi,         sizeof(double)*M_COMPLEX));
        gpu_error_check(cudaMalloc((void**)&d_stress_sum,      sizeof(double)*M_COMPLEX));

        gpu_error_check(cudaMemcpy(d_fourier_basis_x, fourier_basis_x, sizeof(double)*M_COMPLEX,cudaMemcpyHostToDevice));
        gpu_error_check(cudaMemcpy(d_fourier_basis_y, fourier_basis_y, sizeof(double)*M_COMPLEX,cudaMemcpyHostToDevice));
        gpu_error_check(cudaMemcpy(d_fourier_basis_z, fourier_basis_z, sizeof(double)*M_COMPLEX,cudaMemcpyHostToDevice));

        thrust::device_ptr<double> temp_gpu_ptr(d_stress_sum);

        for(int i=0; i<3; i++)
            dq_dl[i] = 0.0;

        // Bond between A segments
        for(int n=1; n<N; n++)
        {
            gpu_error_check(cudaMemcpy(&d_q_in_2m[0], &d_q[M*(2*n-2)],     sizeof(double)*M,cudaMemcpyDeviceToDevice));
            gpu_error_check(cudaMemcpy(&d_q_in_2m[M], &d_q[M*(2*(N-n)-1)], sizeof(double)*M,cudaMemcpyDeviceToDevice));
            cufftExecD2Z(plan_for, d_q_in_2m, d_k_q_in);
            if ( n < seg_start[1])
            {
                bond_length_now = pc->get_bond_length(0);
                d_boltz_bond_now = d_boltz_bond[0];
            }
            for(int b=1; b<N_B; b++)
            {
                if ( n == seg_start[b])
                {
                    bond_length_now = 0.5*pc->get_bond_length(b-1) + 0.5*pc->get_bond_length(b);
                    d_boltz_bond_now = d_boltz_bond_middle[b-1];
                }
                else if (n > seg_start[b] && n < seg_start[b+1])
                {
                    bond_length_now = pc->get_bond_length(b);
                    d_boltz_bond_now = d_boltz_bond[b];
                }
            }

            multi_complex_conjugate<<<N_BLOCKS, N_THREADS>>>(d_q_multi, &d_k_q_in[0], &d_k_q_in[M_COMPLEX], M_COMPLEX);
            multi_real<<<N_BLOCKS, N_THREADS>>>(d_q_multi, d_q_multi, d_boltz_bond_now, bond_length_now, M_COMPLEX);

            if ( DIM >= 3 )
            {
                multi_real<<<N_BLOCKS, N_THREADS>>>(d_stress_sum, d_q_multi, d_fourier_basis_x, 1.0,   M_COMPLEX);
                dq_dl[0] += thrust::reduce(temp_gpu_ptr, temp_gpu_ptr + M_COMPLEX);
            }
            if ( DIM >= 2 )
            {
                multi_real<<<N_BLOCKS, N_THREADS>>>(d_stress_sum, d_q_multi, d_fourier_basis_y, 1.0,   M_COMPLEX);
                dq_dl[1] += thrust::reduce(temp_gpu_ptr, temp_gpu_ptr + M_COMPLEX);
            }
            if ( DIM >= 1 )
            {
                multi_real<<<N_BLOCKS, N_THREADS>>>(d_stress_sum, d_q_multi, d_fourier_basis_z, 1.0,   M_COMPLEX);
                dq_dl[2] += thrust::reduce(temp_gpu_ptr, temp_gpu_ptr + M_COMPLEX);
            }
        }
        for(int d=0; d<3; d++)
            dq_dl[d] /= 3.0*cb->get_lx(d)*M*M/pc->get_ds()/cb->get_volume();

        cudaFree(d_fourier_basis_x);
        cudaFree(d_fourier_basis_y);
        cudaFree(d_fourier_basis_z);
        cudaFree(d_q_in_2m);
        cudaFree(d_q_multi);
        cudaFree(d_stress_sum);

        return dq_dl;
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

void CudaPseudoDiscrete::compute_statistics(double *phi, double *q_1_init, double *q_2_init,
                                  double *w_block, double &single_partition)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int  M        = cb->get_n_grid();
        const int  N        = pc->get_n_segment_total();
        const int  N_B      = pc->get_n_block();
        const std::vector<int> N_SEG    = pc->get_n_segment();
        const double ds     = pc->get_ds();
        const std::vector<int> seg_start= pc->get_block_start();

        double exp_dw[N_B][M];

        for(int b=0; b<N_B; b++)
        {
            for(int i=0; i<M; i++)
                exp_dw[b][i] = exp(-w_block[b*M+i]*ds);
        }
        // Copy array from host memory to device memory
        for(int b=0; b<N_B; b++)
            gpu_error_check(cudaMemcpy(d_exp_dw[b], exp_dw[b], sizeof(double)*M,cudaMemcpyHostToDevice));

        gpu_error_check(cudaMemcpy(&d_q[0], q_1_init, sizeof(double)*M, cudaMemcpyHostToDevice));
        gpu_error_check(cudaMemcpy(&d_q[M], q_2_init, sizeof(double)*M, cudaMemcpyHostToDevice));

        multi_real<<<N_BLOCKS, N_THREADS>>>(&d_q[0], &d_q[0], d_exp_dw[0], 1.0, M);
        multi_real<<<N_BLOCKS, N_THREADS>>>(&d_q[M], &d_q[M], d_exp_dw[N_B-1], 1.0, M);

        int b_curr_1, b_curr_2; // current block of q1/q2
        //bool is_middle_1, is_middle_2; // is current monomer the one between blocks
        double *boltz_bond_current_1, *boltz_bond_current_2; //boltzmann bond for current monomer
        for(int n=1; n<N; n++)
        {
            // find currently working block 
            int b_ = 0;
            while(b_ < N && seg_start[b_] <= n)  /////////////////////////reminder to revisit here to check error
            {
                b_curr_1 = b_;
                b_++;
            }
            b_ = N_B-1;
            while(b_ >= 0 && N - seg_start[b_+1] <= n)
            {
                b_curr_2 = b_;
                b_--;
            }
            boltz_bond_current_1 = d_boltz_bond[b_curr_1];
            boltz_bond_current_2 = d_boltz_bond[b_curr_2];

            if (seg_start[b_curr_1] == n)
                boltz_bond_current_1 = d_boltz_bond_middle[b_curr_1-1];
            if (N - seg_start[b_curr_2+1] == n)
                boltz_bond_current_2 = d_boltz_bond_middle[b_curr_2];

            one_step(&d_q[2*M*(n-1)], &d_q[2*M*n], boltz_bond_current_1,  boltz_bond_current_2,  d_exp_dw[b_curr_1], d_exp_dw[b_curr_2]);
        }

        //calculates the total partition function
        //d_phi is used as a temporary array
        gpu_error_check(cudaMemcpy(&d_phi[0], q_2_init, sizeof(double)*M, cudaMemcpyHostToDevice));
        single_partition = ((CudaComputationBox *)cb)->inner_product_gpu(&d_q[2*M*(N-1)],&d_phi[0]);

        // Calculate segment density
        for(int b=0; b<N_B; b++)
        {
            multi_real<<<N_BLOCKS, N_THREADS>>>(&d_phi[b*M], &d_q[2*M*seg_start[b]], &d_q[2*M*(N-seg_start[b]-1)+M], 1.0, M);
            for(int n=seg_start[b]+1; n<seg_start[b+1]; n++)
            {
                add_multi_real<<<N_BLOCKS, N_THREADS>>>(&d_phi[b*M], &d_q[2*M*n], &d_q[2*M*(N-n-1)+M], 1.0, M);
            }
        }

        // normalize the concentration
        for(int b=0; b<N_B; b++)
            divide_real<<<N_BLOCKS, N_THREADS>>>(&d_phi[b*M], &d_phi[b*M], d_exp_dw[b], cb->get_volume()*pc->get_ds()/single_partition, M);
        gpu_error_check(cudaMemcpy(phi, d_phi, sizeof(double)*N_B*M,cudaMemcpyDeviceToHost));
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

void CudaPseudoDiscrete::one_step(double *d_q_in,          double *d_q_out,
                                  double *d_boltz_bond_1, double *d_boltz_bond_2,
                                  double *d_exp_dw_1,      double *d_exp_dw_2)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = cb->get_n_grid();
        const int M_COMPLEX = this->n_complex_grid;

        //-------------- step 1 ----------
        // Execute a Forward FFT
        cufftExecD2Z(plan_for, d_q_in, d_k_q_in);

        // Multiply e^(-k^2 ds/6) in fourier space
        multi_complex_real<<<N_BLOCKS, N_THREADS>>>(&d_k_q_in[0],         d_boltz_bond_1, M_COMPLEX);
        multi_complex_real<<<N_BLOCKS, N_THREADS>>>(&d_k_q_in[M_COMPLEX], d_boltz_bond_2, M_COMPLEX);

        // Execute a backward FFT
        cufftExecZ2D(plan_bak, d_k_q_in, d_q_out);

        // Evaluate e^(-w*ds) in real space
        multi_real<<<N_BLOCKS, N_THREADS>>>(&d_q_out[0], &d_q_out[0], d_exp_dw_1, 1.0/((double)M), M);
        multi_real<<<N_BLOCKS, N_THREADS>>>(&d_q_out[M], &d_q_out[M], d_exp_dw_2, 1.0/((double)M), M);
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaPseudoDiscrete::get_partition(double *q_1_out, int n1, double *q_2_out, int n2)
{
    // This method should be invoked after invoking compute_statistics().
    
    // Get partial partition functions
    // This is made for debugging and testing.
    const int M = cb->get_n_grid();
    const int N = pc->get_n_segment_total();

    if (n1 < 1 || n1 > N)
        throw_with_line_number("n1 (" + std::to_string(n1) + ") must be in range [1, " + std::to_string(N) + "]");
    if (n2 < 1 || n2 > N)
        throw_with_line_number("n2 (" + std::to_string(n2) + ") must be in range [1, " + std::to_string(N) + "]");

    gpu_error_check(cudaMemcpy(q_1_out, &d_q[M*(2*n1-2)],       sizeof(double)*M,cudaMemcpyDeviceToHost));
    gpu_error_check(cudaMemcpy(q_2_out, &d_q[M*(2*(N-n2+1)-1)], sizeof(double)*M,cudaMemcpyDeviceToHost));
}
