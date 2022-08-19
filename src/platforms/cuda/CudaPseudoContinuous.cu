#define THRUST_IGNORE_DEPRECATED_CPP_DIALECT
#define CUB_IGNORE_DEPRECATED_CPP_DIALECT

#include <complex>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include "CudaPseudoContinuous.h"
#include "CudaSimulationBox.h"
#include "SimpsonQuadrature.h"

CudaPseudoContinuous::CudaPseudoContinuous(
    SimulationBox *sb,
    PolymerChain *pc)
    : Pseudo(sb, pc)
{
    try{
        const int M = sb->get_n_grid();
        const int N_B = pc->get_n_block();
        const int N = pc->get_n_segment_total();
        const int M_COMPLEX = this->n_complex_grid;

        // Create FFT plan
        const int BATCH{2};
        const int NRANK{sb->get_dim()};
        int n_grid[NRANK];

        this->n_block = N_B;
        if(sb->get_dim() == 3)
        {
            n_grid[0] = sb->get_nx(0);
            n_grid[1] = sb->get_nx(1);
            n_grid[2] = sb->get_nx(2);
        }
        else if(sb->get_dim() == 2)
        {
            n_grid[0] = sb->get_nx(1);
            n_grid[1] = sb->get_nx(2);
        }
        else if(sb->get_dim() == 1)
        {
            n_grid[0] = sb->get_nx(2);
        }
        cufftPlanMany(&plan_for, NRANK, n_grid, NULL, 1, 0, NULL, 1, 0, CUFFT_D2Z,BATCH);
        cufftPlanMany(&plan_bak, NRANK, n_grid, NULL, 1, 0, NULL, 1, 0, CUFFT_Z2D,BATCH);

        // Memory allocation
        this->d_exp_dw = new double*[N_B];
        this->d_exp_dw_half = new double*[N_B];
        this->d_boltz_bond = new double*[N_B];
        this->d_boltz_bond_half = new double*[N_B];

        gpu_error_check(cudaMalloc((void**)&d_q_step1, sizeof(double)*2*M));
        gpu_error_check(cudaMalloc((void**)&d_q_step2, sizeof(double)*2*M));
        gpu_error_check(cudaMalloc((void**)&d_k_q_in,   sizeof(ftsComplex)*2*M_COMPLEX));

        gpu_error_check(cudaMalloc((void**)&d_q_1, sizeof(double)*M*(N+1)));
        gpu_error_check(cudaMalloc((void**)&d_q_2, sizeof(double)*M*(N+1)));
        gpu_error_check(cudaMalloc((void**)&d_phi, sizeof(double)*M*N_B));

        for (int b=0; b<N_B; b++)
        {
            gpu_error_check(cudaMalloc((void**)&d_exp_dw[b],      sizeof(double)*M));
            gpu_error_check(cudaMalloc((void**)&d_exp_dw_half[b], sizeof(double)*M));

            gpu_error_check(cudaMalloc((void**)&d_boltz_bond[b],      sizeof(double)*M_COMPLEX));
            gpu_error_check(cudaMalloc((void**)&d_boltz_bond_half[b], sizeof(double)*M_COMPLEX));
        }
        update();
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
CudaPseudoContinuous::~CudaPseudoContinuous()
{
    const int N_B = n_block;

    cufftDestroy(plan_for);
    cufftDestroy(plan_bak);

    cudaFree(d_q_step1);
    cudaFree(d_q_step2);
    cudaFree(d_k_q_in);

    cudaFree(d_q_1);
    cudaFree(d_q_2);
    cudaFree(d_phi);
    for (int b=0; b<N_B; b++)
    {
        cudaFree(d_exp_dw[b]);
        cudaFree(d_exp_dw_half[b]);

        cudaFree(d_boltz_bond[b]);
        cudaFree(d_boltz_bond_half[b]);
    }

    delete[] d_exp_dw, d_exp_dw_half;  
    delete[] d_boltz_bond, d_boltz_bond_half;
}

void CudaPseudoContinuous::update()
{
    try{
        const int M_COMPLEX = this->n_complex_grid;
        const int N_B = pc->get_n_block();
        double boltz_bond[N_B][M_COMPLEX], boltz_bond_half[N_B][M_COMPLEX];

        for (int b=0; b<N_B; b++)
        {
        get_boltz_bond(boltz_bond[b],      pc->get_bond_length(b),   sb->get_nx(), sb->get_dx(), pc->get_ds());
        get_boltz_bond(boltz_bond_half[b], pc->get_bond_length(b)/2, sb->get_nx(), sb->get_dx(), pc->get_ds());

        gpu_error_check(cudaMemcpy(d_boltz_bond[b],      boltz_bond[b],      sizeof(double)*M_COMPLEX,cudaMemcpyHostToDevice));
        gpu_error_check(cudaMemcpy(d_boltz_bond_half[b], boltz_bond_half[b], sizeof(double)*M_COMPLEX,cudaMemcpyHostToDevice));

        }
    }
    catch(std::exception& exc)
    {
        throw_with_line_number(exc.what());
    }
}

std::array<double,3> CudaPseudoContinuous::dq_dl()
{
    // This method should be invoked after invoking find_phi().

    // To calculate stress, we multiply weighted fourier basis to q(k)*q^dagger(-k).
    // We only need the real part of stress calculation.

    try{
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int DIM  = sb->get_dim();
        const int M    = sb->get_n_grid();
        const int N    = pc->get_n_segment_total();
        const int N_B = pc->get_n_block();
        const std::vector<int> N_SEG    = pc->get_n_segment();
        const int M_COMPLEX = this->n_complex_grid;
        const std::vector<int> seg_start= pc->get_block_start();


        std::array<double,3> dq_dl;
        double simpson_rule_coeff[N];
        double fourier_basis_x[M_COMPLEX];
        double fourier_basis_y[M_COMPLEX];
        double fourier_basis_z[M_COMPLEX];

        double *d_fourier_basis_x;
        double *d_fourier_basis_y;
        double *d_fourier_basis_z;
        double *d_q_in_2m, *d_q_multi, *d_stress_sum;

        get_weighted_fourier_basis(fourier_basis_x, fourier_basis_y, fourier_basis_z, sb->get_nx(), sb->get_dx());

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
        for(int b=0; b<N_B; b++)
        {
            SimpsonQuadrature::init_coeff(simpson_rule_coeff, N_SEG[b]);
            for(int n=seg_start[b]; n<=seg_start[b+1]; n++)
            {
                gpu_error_check(cudaMemcpy(&d_q_in_2m[0], &d_q_1[n*M],     sizeof(double)*M,cudaMemcpyDeviceToDevice));
                gpu_error_check(cudaMemcpy(&d_q_in_2m[M], &d_q_2[(N-n)*M], sizeof(double)*M,cudaMemcpyDeviceToDevice));
                cufftExecD2Z(plan_for, d_q_in_2m, d_k_q_in);

                multi_complex_conjugate<<<N_BLOCKS, N_THREADS>>>(d_q_multi, &d_k_q_in[0], &d_k_q_in[M_COMPLEX], M_COMPLEX);
                if ( DIM >= 3 )
                {
                    multi_real<<<N_BLOCKS, N_THREADS>>>(d_stress_sum, d_q_multi, d_fourier_basis_x, pc->get_bond_length(b), M_COMPLEX);
                    dq_dl[0] += simpson_rule_coeff[n-seg_start[b]]*thrust::reduce(temp_gpu_ptr, temp_gpu_ptr + M_COMPLEX);
                }
                if ( DIM >= 2 )
                {
                    multi_real<<<N_BLOCKS, N_THREADS>>>(d_stress_sum, d_q_multi, d_fourier_basis_y, pc->get_bond_length(b), M_COMPLEX);
                    dq_dl[1] += simpson_rule_coeff[n-seg_start[b]]*thrust::reduce(temp_gpu_ptr, temp_gpu_ptr + M_COMPLEX);
                }
                if ( DIM >= 1 )
                {
                    multi_real<<<N_BLOCKS, N_THREADS>>>(d_stress_sum, d_q_multi, d_fourier_basis_z, pc->get_bond_length(b), M_COMPLEX);
                    dq_dl[2] += simpson_rule_coeff[n-seg_start[b]]*thrust::reduce(temp_gpu_ptr, temp_gpu_ptr + M_COMPLEX);
                }
            }
        }

        for(int d=0; d<3; d++)
            dq_dl[d] /= 3.0*sb->get_lx(d)*M*M*N/sb->get_volume();

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
void CudaPseudoContinuous::calculate_phi_one_type(
    double *d_phi, const int N_START, const int N_END)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = sb->get_n_grid();
        const int N = pc->get_n_segment_total();
        double simpson_rule_coeff[N_END-N_START+1];

        SimpsonQuadrature::init_coeff(simpson_rule_coeff, N_END-N_START);

        // Compute segment concentration
        multi_real<<<N_BLOCKS, N_THREADS>>>(d_phi, &d_q_1[M*N_START], &d_q_2[M*(N-N_START)], simpson_rule_coeff[0], M);
        for(int n=N_START+1; n<=N_END; n++)
        {
            add_multi_real<<<N_BLOCKS, N_THREADS>>>(d_phi, &d_q_1[M*n], &d_q_2[M*(N-n)], simpson_rule_coeff[n-N_START], M);
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

void CudaPseudoContinuous::find_phi(double *phi, double *q_1_init, double *q_2_init,
                                    double *w_block, double &single_partition)
{
    try{
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int  M        = sb->get_n_grid();
        const int  N        = pc->get_n_segment_total();
        const int  N_B      = pc->get_n_block();
        const std::vector<int> N_SEG    = pc->get_n_segment();
        const double ds     = pc->get_ds();
        const std::vector<int> seg_start= pc->get_block_start();                                 

        double exp_dw[N_B][M];
        double exp_dw_half[N_B][M];

        for(int b=0; b<N_B; b++)
        {
            for(int i=0; i<M; i++)
            {
                exp_dw      [b][i] = exp(-w_block[b*M+i]*ds*0.5);
                exp_dw_half [b][i] = exp(-w_block[b*M+i]*ds*0.25);
            }
        }

        // Copy array from host memory to device memory
        for(int b=0; b<N_B; b++)
        {
            gpu_error_check(cudaMemcpy(d_exp_dw[b], exp_dw[b], sizeof(double)*M,cudaMemcpyHostToDevice));
            gpu_error_check(cudaMemcpy(d_exp_dw_half[b], exp_dw_half[b], sizeof(double)*M,cudaMemcpyHostToDevice));
        }

        gpu_error_check(cudaMemcpy(&d_q_1[0], q_1_init, sizeof(double)*M,
                cudaMemcpyHostToDevice));
        gpu_error_check(cudaMemcpy(&d_q_2[0], q_2_init, sizeof(double)*M,
                cudaMemcpyHostToDevice));

        int b_curr_1, b_curr_2; // current block of q1/q2
        for(int n=0; n<N; n++)
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

            one_step(
                    &d_q_1[M*n], &d_q_1[M*(n+1)],
                    &d_q_2[M*n], &d_q_2[M*(n+1)],
                    d_boltz_bond[b_curr_1], d_boltz_bond_half[b_curr_1],
                    d_boltz_bond[b_curr_2], d_boltz_bond_half[b_curr_2],
                    d_exp_dw[b_curr_1], d_exp_dw_half[b_curr_1],
                    d_exp_dw[b_curr_2], d_exp_dw_half[b_curr_2]);
        }

        // calculates the total partition function
        //d_phi is used as a temporary array
        multi_real<<<N_BLOCKS, N_THREADS>>>(&d_phi[0], &d_q_1[M*N], &d_q_2[0], 0.5, M);//&d_q_1[M*N_A], &d_q_2[M*(N-N_A)], 0.5, M);
        single_partition = 2*((CudaSimulationBox *)sb)->integral_gpu(&d_phi[0]);

        // segment concentration.
        for(int b=0; b<N_B; b++)
        {
            calculate_phi_one_type(&d_phi[b*M], seg_start[b], seg_start[b+1]);
        }

        // normalize the concentration
        for(int b=0; b<N_B; b++)
            lin_comb<<<N_BLOCKS, N_THREADS>>>(&d_phi[b*M], (sb->get_volume())/single_partition/N, &d_phi[b*M], 0.0, &d_phi[b*M], M);

        for(int b=0; b<N_B; b++)
            gpu_error_check(cudaMemcpy(phi, d_phi, sizeof(double)*N_B*M,cudaMemcpyDeviceToHost));
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

// Advance two partial partition functions simultaneously using Richardson extrapolation.
// Note that cufft doesn't fully utilize GPU cores unless n_grid is sufficiently large.
// To increase GPU usage, FFT Batch is utilized.
void CudaPseudoContinuous::one_step(double *d_q_in1,        double *d_q_out1,
                                  double *d_q_in2,        double *d_q_out2,
                                  double *d_boltz_bond_1, double *d_boltz_bond_1_half,
                                  double *d_boltz_bond_2, double *d_boltz_bond_2_half,
                                  double *d_exp_dw_1,     double *d_exp_dw_1_half,
                                  double *d_exp_dw_2,     double *d_exp_dw_2_half)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = sb->get_n_grid();
        const int M_COMPLEX = this->n_complex_grid;

        //-------------- step 1 ----------
        // Evaluate e^(-w*ds/2) in real space
        multi_real<<<N_BLOCKS, N_THREADS>>>(&d_q_step1[0], d_q_in1, d_exp_dw_1, 1.0, M);
        multi_real<<<N_BLOCKS, N_THREADS>>>(&d_q_step1[M], d_q_in2, d_exp_dw_2, 1.0, M);

        // Execute a Forw_ard FFT
        cufftExecD2Z(plan_for, d_q_step1, d_k_q_in);

        // Multiply e^(-k^2 ds/6) in fourier space
        multi_complex_real<<<N_BLOCKS, N_THREADS>>>(&d_k_q_in[0],         d_boltz_bond_1, M_COMPLEX);
        multi_complex_real<<<N_BLOCKS, N_THREADS>>>(&d_k_q_in[M_COMPLEX], d_boltz_bond_2, M_COMPLEX);

        // Execute a backw_ard FFT
        cufftExecZ2D(plan_bak, d_k_q_in, d_q_step1);

        // Evaluate e^(-w*ds/2) in real space
        multi_real<<<N_BLOCKS, N_THREADS>>>(&d_q_step1[0], &d_q_step1[0], d_exp_dw_1, 1.0/((double)M), M);
        multi_real<<<N_BLOCKS, N_THREADS>>>(&d_q_step1[M], &d_q_step1[M], d_exp_dw_2, 1.0/((double)M), M);

        //-------------- step 2 ----------
        // Evaluate e^(-w*ds/4) in real space
        multi_real<<<N_BLOCKS, N_THREADS>>>(&d_q_step2[0], d_q_in1, d_exp_dw_1_half, 1.0, M);
        multi_real<<<N_BLOCKS, N_THREADS>>>(&d_q_step2[M], d_q_in2, d_exp_dw_2_half, 1.0, M);

        // Execute a Forw_ard FFT
        cufftExecD2Z(plan_for, d_q_step2, d_k_q_in);

        // Multiply e^(-k^2 ds/12) in fourier space
        multi_complex_real<<<N_BLOCKS, N_THREADS>>>(&d_k_q_in[0],         d_boltz_bond_1_half, M_COMPLEX);
        multi_complex_real<<<N_BLOCKS, N_THREADS>>>(&d_k_q_in[M_COMPLEX], d_boltz_bond_2_half, M_COMPLEX);

        // Execute a backw_ard FFT
        cufftExecZ2D(plan_bak, d_k_q_in, d_q_step2);

        // Evaluate e^(-w*ds/2) in real space
        multi_real<<<N_BLOCKS, N_THREADS>>>(&d_q_step2[0], &d_q_step2[0], d_exp_dw_1, 1.0/((double)M), M);
        multi_real<<<N_BLOCKS, N_THREADS>>>(&d_q_step2[M], &d_q_step2[M], d_exp_dw_2, 1.0/((double)M), M);
        // Execute a Forw_ard FFT
        cufftExecD2Z(plan_for, d_q_step2, d_k_q_in);

        // Multiply e^(-k^2 ds/12) in fourier space
        multi_complex_real<<<N_BLOCKS, N_THREADS>>>(&d_k_q_in[0],         d_boltz_bond_1_half, M_COMPLEX);
        multi_complex_real<<<N_BLOCKS, N_THREADS>>>(&d_k_q_in[M_COMPLEX], d_boltz_bond_2_half, M_COMPLEX);

        // Execute a backw_ard FFT
        cufftExecZ2D(plan_bak, d_k_q_in, d_q_step2);

        // Evaluate e^(-w*ds/4) in real space.
        multi_real<<<N_BLOCKS, N_THREADS>>>(&d_q_step2[0], &d_q_step2[0], d_exp_dw_1_half, 1.0/((double)M), M);
        multi_real<<<N_BLOCKS, N_THREADS>>>(&d_q_step2[M], &d_q_step2[M], d_exp_dw_2_half, 1.0/((double)M), M);
        //-------------- step 3 ----------
        lin_comb<<<N_BLOCKS, N_THREADS>>>(d_q_out1, 4.0/3.0, &d_q_step2[0], -1.0/3.0, &d_q_step1[0], M);
        lin_comb<<<N_BLOCKS, N_THREADS>>>(d_q_out2, 4.0/3.0, &d_q_step2[M], -1.0/3.0, &d_q_step1[M], M);
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaPseudoContinuous::get_partition(double *q_1_out, int n1, double *q_2_out, int n2)
{
    // This method should be invoked after invoking find_phi().
    
    // Get partial partition functions
    // This is made for debugging and testing.
    const int M = sb->get_n_grid();
    const int N = pc->get_n_segment_total();
        
    if (n1 < 0 || n1 > N)
        throw_with_line_number("n1 (" + std::to_string(n1) + ") must be in range [0, " + std::to_string(N) + "]");
    if (n2 < 0 || n2 > N)
        throw_with_line_number("n2 (" + std::to_string(n2) + ") must be in range [0, " + std::to_string(N) + "]");

    gpu_error_check(cudaMemcpy(q_1_out, &d_q_1[n1*M], sizeof(double)*M,cudaMemcpyDeviceToHost));
    gpu_error_check(cudaMemcpy(q_2_out, &d_q_2[(N-n2)*M], sizeof(double)*M,cudaMemcpyDeviceToHost));
}
