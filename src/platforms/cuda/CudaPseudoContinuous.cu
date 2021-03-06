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
        const int N = pc->get_n_segment();
        const int M_COMPLEX = this->n_complex_grid;

        // Create FFT plan
        const int BATCH{2};
        const int NRANK{sb->get_dim()};
        int n_grid[NRANK];

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
        gpu_error_check(cudaMalloc((void**)&d_q_step1, sizeof(double)*2*M));
        gpu_error_check(cudaMalloc((void**)&d_q_step2, sizeof(double)*2*M));
        gpu_error_check(cudaMalloc((void**)&d_k_q_in,   sizeof(ftsComplex)*2*M_COMPLEX));

        gpu_error_check(cudaMalloc((void**)&d_q_1, sizeof(double)*M*(N+1)));
        gpu_error_check(cudaMalloc((void**)&d_q_2, sizeof(double)*M*(N+1)));

        gpu_error_check(cudaMalloc((void**)&d_exp_dw_a,      sizeof(double)*M));
        gpu_error_check(cudaMalloc((void**)&d_exp_dw_b,      sizeof(double)*M));
        gpu_error_check(cudaMalloc((void**)&d_exp_dw_a_half, sizeof(double)*M));
        gpu_error_check(cudaMalloc((void**)&d_exp_dw_b_half, sizeof(double)*M));

        gpu_error_check(cudaMalloc((void**)&d_phi_a, sizeof(double)*M));
        gpu_error_check(cudaMalloc((void**)&d_phi_b, sizeof(double)*M));

        gpu_error_check(cudaMalloc((void**)&d_boltz_bond_a,      sizeof(double)*M_COMPLEX));
        gpu_error_check(cudaMalloc((void**)&d_boltz_bond_b,      sizeof(double)*M_COMPLEX));
        gpu_error_check(cudaMalloc((void**)&d_boltz_bond_a_half, sizeof(double)*M_COMPLEX));
        gpu_error_check(cudaMalloc((void**)&d_boltz_bond_b_half, sizeof(double)*M_COMPLEX));

        update();
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
CudaPseudoContinuous::~CudaPseudoContinuous()
{
    cufftDestroy(plan_for);
    cufftDestroy(plan_bak);

    cudaFree(d_q_step1);
    cudaFree(d_q_step2);
    cudaFree(d_k_q_in);

    cudaFree(d_q_1);
    cudaFree(d_q_2);

    cudaFree(d_exp_dw_a);
    cudaFree(d_exp_dw_b);
    cudaFree(d_exp_dw_a_half);
    cudaFree(d_exp_dw_b_half);
    cudaFree(d_phi_a);
    cudaFree(d_phi_b);

    cudaFree(d_boltz_bond_a);
    cudaFree(d_boltz_bond_b);
    cudaFree(d_boltz_bond_a_half);
    cudaFree(d_boltz_bond_b_half);
}

void CudaPseudoContinuous::update()
{
    try{
        double bond_length_a, bond_length_b;
        const double eps = pc->get_epsilon();
        const double f = pc->get_f();

        const int M_COMPLEX = this->n_complex_grid;
        double boltz_bond_a[M_COMPLEX], boltz_bond_a_half[M_COMPLEX];
        double boltz_bond_b[M_COMPLEX], boltz_bond_b_half[M_COMPLEX];

        bond_length_a = eps*eps/(f*eps*eps + (1.0-f));
        bond_length_b = 1.0/(f*eps*eps + (1.0-f));

        get_boltz_bond(boltz_bond_a,      bond_length_a,   sb->get_nx(), sb->get_dx(), pc->get_ds());
        get_boltz_bond(boltz_bond_b,      bond_length_b,   sb->get_nx(), sb->get_dx(), pc->get_ds());
        get_boltz_bond(boltz_bond_a_half, bond_length_a/2, sb->get_nx(), sb->get_dx(), pc->get_ds());
        get_boltz_bond(boltz_bond_b_half, bond_length_b/2, sb->get_nx(), sb->get_dx(), pc->get_ds());

        gpu_error_check(cudaMemcpy(d_boltz_bond_a,      boltz_bond_a,      sizeof(double)*M_COMPLEX,cudaMemcpyHostToDevice));
        gpu_error_check(cudaMemcpy(d_boltz_bond_b,      boltz_bond_b,      sizeof(double)*M_COMPLEX,cudaMemcpyHostToDevice));
        gpu_error_check(cudaMemcpy(d_boltz_bond_a_half, boltz_bond_a_half, sizeof(double)*M_COMPLEX,cudaMemcpyHostToDevice));
        gpu_error_check(cudaMemcpy(d_boltz_bond_b_half, boltz_bond_b_half, sizeof(double)*M_COMPLEX,cudaMemcpyHostToDevice));
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
        const int N    = pc->get_n_segment();
        const int N_A  = pc->get_n_segment_a();
        const int M_COMPLEX = this->n_complex_grid;

        const double eps = pc->get_epsilon();
        const double f = pc->get_f();
        const double bond_length_a = eps*eps/(f*eps*eps + (1.0-f));
        const double bond_length_b = 1.0/(f*eps*eps + (1.0-f));

        std::array<double,3> dq_dl;
        double simpson_rule_coeff_a[N_A+1];
        double simpson_rule_coeff_b[N-N_A+1];
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

        SimpsonQuadrature::init_coeff(simpson_rule_coeff_a, N_A);
        for(int n=0; n<=N_A; n++)
        {
            gpu_error_check(cudaMemcpy(&d_q_in_2m[0], &d_q_1[n*M],     sizeof(double)*M,cudaMemcpyDeviceToDevice));
            gpu_error_check(cudaMemcpy(&d_q_in_2m[M], &d_q_2[(N-n)*M], sizeof(double)*M,cudaMemcpyDeviceToDevice));
            cufftExecD2Z(plan_for, d_q_in_2m, d_k_q_in);

            multi_complex_conjugate<<<N_BLOCKS, N_THREADS>>>(d_q_multi, &d_k_q_in[0], &d_k_q_in[M_COMPLEX], M_COMPLEX);
            if ( DIM >= 3 )
            {
                multi_real<<<N_BLOCKS, N_THREADS>>>(d_stress_sum, d_q_multi, d_fourier_basis_x, bond_length_a, M_COMPLEX);
                dq_dl[0] += simpson_rule_coeff_a[n]*thrust::reduce(temp_gpu_ptr, temp_gpu_ptr + M_COMPLEX);
            }
            if ( DIM >= 2 )
            {
                multi_real<<<N_BLOCKS, N_THREADS>>>(d_stress_sum, d_q_multi, d_fourier_basis_y, bond_length_a, M_COMPLEX);
                dq_dl[1] += simpson_rule_coeff_a[n]*thrust::reduce(temp_gpu_ptr, temp_gpu_ptr + M_COMPLEX);
            }
            if ( DIM >= 1 )
            {
                multi_real<<<N_BLOCKS, N_THREADS>>>(d_stress_sum, d_q_multi, d_fourier_basis_z, bond_length_a, M_COMPLEX);
                dq_dl[2] += simpson_rule_coeff_a[n]*thrust::reduce(temp_gpu_ptr, temp_gpu_ptr + M_COMPLEX);
            }
        }

        SimpsonQuadrature::init_coeff(simpson_rule_coeff_b, N-N_A);
        for(int n=N_A; n<=N; n++)
        {
            gpu_error_check(cudaMemcpy(&d_q_in_2m[0], &d_q_1[n*M],     sizeof(double)*M,cudaMemcpyDeviceToDevice));
            gpu_error_check(cudaMemcpy(&d_q_in_2m[M], &d_q_2[(N-n)*M], sizeof(double)*M,cudaMemcpyDeviceToDevice));
            cufftExecD2Z(plan_for, d_q_in_2m, d_k_q_in);

            multi_complex_conjugate<<<N_BLOCKS, N_THREADS>>>(d_q_multi, &d_k_q_in[0], &d_k_q_in[M_COMPLEX], M_COMPLEX);
            if ( DIM >= 3 )
            {
                multi_real<<<N_BLOCKS, N_THREADS>>>(d_stress_sum, d_q_multi, d_fourier_basis_x, bond_length_b, M_COMPLEX);
                dq_dl[0] += simpson_rule_coeff_b[n-N_A]*thrust::reduce(temp_gpu_ptr, temp_gpu_ptr + M_COMPLEX);
            }
            if ( DIM >= 2 )
            {
                multi_real<<<N_BLOCKS, N_THREADS>>>(d_stress_sum, d_q_multi, d_fourier_basis_y, bond_length_b, M_COMPLEX);
                dq_dl[1] += simpson_rule_coeff_b[n-N_A]*thrust::reduce(temp_gpu_ptr, temp_gpu_ptr + M_COMPLEX);
            }
            if ( DIM >= 1 )
            {
                multi_real<<<N_BLOCKS, N_THREADS>>>(d_stress_sum, d_q_multi, d_fourier_basis_z, bond_length_b, M_COMPLEX);
                dq_dl[2] += simpson_rule_coeff_b[n-N_A]*thrust::reduce(temp_gpu_ptr, temp_gpu_ptr + M_COMPLEX);
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
        const int N = pc->get_n_segment();
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

void CudaPseudoContinuous::find_phi(double *phi_a,  double *phi_b,
                                  double *q_1_init, double *q_2_init,
                                  double *w_a, double *w_b, double &single_partition)
{
    try{
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = sb->get_n_grid();
        const int N = pc->get_n_segment();
        const int N_A = pc->get_n_segment_a();
        const int N_B = pc->get_n_segment_b();
        const double ds = pc->get_ds();

        double exp_dw_a[M];
        double exp_dw_b[M];
        double exp_dw_a_half[M];
        double exp_dw_b_half[M];

        for(int i=0; i<M; i++)
        {
            exp_dw_a     [i] = exp(-w_a[i]*ds*0.5);
            exp_dw_b     [i] = exp(-w_b[i]*ds*0.5);
            exp_dw_a_half[i] = exp(-w_a[i]*ds*0.25);
            exp_dw_b_half[i] = exp(-w_b[i]*ds*0.25);
        }

        // Copy array from host memory to device memory
        gpu_error_check(cudaMemcpy(d_exp_dw_a, exp_dw_a, sizeof(double)*M,cudaMemcpyHostToDevice));
        gpu_error_check(cudaMemcpy(d_exp_dw_b, exp_dw_b, sizeof(double)*M,cudaMemcpyHostToDevice));
        gpu_error_check(cudaMemcpy(d_exp_dw_a_half, exp_dw_a_half, sizeof(double)*M,cudaMemcpyHostToDevice));
        gpu_error_check(cudaMemcpy(d_exp_dw_b_half, exp_dw_b_half, sizeof(double)*M,cudaMemcpyHostToDevice));

        gpu_error_check(cudaMemcpy(&d_q_1[0], q_1_init, sizeof(double)*M,
                cudaMemcpyHostToDevice));
        gpu_error_check(cudaMemcpy(&d_q_2[0], q_2_init, sizeof(double)*M,
                cudaMemcpyHostToDevice));

        for(int n=0; n<N; n++)
        {
            if(n < N_A && n < N_B)
            {
                one_step(
                    &d_q_1[M*n], &d_q_1[M*(n+1)],
                    &d_q_2[M*n], &d_q_2[M*(n+1)],
                    d_boltz_bond_a, d_boltz_bond_a_half,
                    d_boltz_bond_b, d_boltz_bond_b_half,
                    d_exp_dw_a, d_exp_dw_a_half,
                    d_exp_dw_b, d_exp_dw_b_half);
            }
            else if(n < N_A &&  n >= N_B)
            {
                one_step(
                    &d_q_1[M*n], &d_q_1[M*(n+1)],
                    &d_q_2[M*n], &d_q_2[M*(n+1)],
                    d_boltz_bond_a, d_boltz_bond_a_half,
                    d_boltz_bond_a, d_boltz_bond_a_half,
                    d_exp_dw_a, d_exp_dw_a_half,
                    d_exp_dw_a, d_exp_dw_a_half);
            }
            else if(n >= N_A && n < N_B)
            {
                one_step(
                    &d_q_1[M*n], &d_q_1[M*(n+1)],
                    &d_q_2[M*n], &d_q_2[M*(n+1)],
                    d_boltz_bond_b, d_boltz_bond_b_half,
                    d_boltz_bond_b, d_boltz_bond_b_half,
                    d_exp_dw_b, d_exp_dw_b_half,
                    d_exp_dw_b, d_exp_dw_b_half);
            }
            else
            {
                one_step(
                    &d_q_1[M*n], &d_q_1[M*(n+1)],
                    &d_q_2[M*n], &d_q_2[M*(n+1)],
                    d_boltz_bond_b, d_boltz_bond_b_half,
                    d_boltz_bond_a, d_boltz_bond_a_half,
                    d_exp_dw_b, d_exp_dw_b_half,
                    d_exp_dw_a, d_exp_dw_a_half);
            }
        }

        // calculates the total partition function
        multi_real<<<N_BLOCKS, N_THREADS>>>(d_phi_a, &d_q_1[M*N_A], &d_q_2[M*(N_B)], 0.5, M);
        single_partition = 2*((CudaSimulationBox *)sb)->integral_gpu(d_phi_a);

        // segment concentration.
        // A block
        calculate_phi_one_type(d_phi_a, 0, N_A);
        // B block
        calculate_phi_one_type(d_phi_b, N_A, N);

        // normalize the concentration
        lin_comb<<<N_BLOCKS, N_THREADS>>>(d_phi_a, (sb->get_volume())/single_partition/N, d_phi_a, 0.0, d_phi_a, M);
        lin_comb<<<N_BLOCKS, N_THREADS>>>(d_phi_b, (sb->get_volume())/single_partition/N, d_phi_b, 0.0, d_phi_b, M);

        gpu_error_check(cudaMemcpy(phi_a, d_phi_a, sizeof(double)*M,cudaMemcpyDeviceToHost));
        gpu_error_check(cudaMemcpy(phi_b, d_phi_b, sizeof(double)*M,cudaMemcpyDeviceToHost));
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
    const int N = pc->get_n_segment();
        
    if (n1 < 0 || n1 > N)
        throw_with_line_number("n1 (" + std::to_string(n1) + ") must be in range [0, " + std::to_string(N) + "]");
    if (n2 < 0 || n2 > N)
        throw_with_line_number("n2 (" + std::to_string(n2) + ") must be in range [0, " + std::to_string(N) + "]");

    gpu_error_check(cudaMemcpy(q_1_out, &d_q_1[n1*M], sizeof(double)*M,cudaMemcpyDeviceToHost));
    gpu_error_check(cudaMemcpy(q_2_out, &d_q_2[(N-n2)*M], sizeof(double)*M,cudaMemcpyDeviceToHost));
}
