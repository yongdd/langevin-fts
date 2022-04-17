#define THRUST_IGNORE_DEPRECATED_CPP_DIALECT
#define CUB_IGNORE_DEPRECATED_CPP_DIALECT

#include <complex>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <iostream>
#include "CudaPseudoDiscrete.h"
#include "CudaSimulationBox.h"

CudaPseudoDiscrete::CudaPseudoDiscrete(
    SimulationBox *sb,
    PolymerChain *pc)
    : Pseudo(sb, pc)
{
    const int M = sb->get_n_grid();
    const int N = pc->get_n_contour();
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
    cudaMalloc((void**)&d_k_q_in, sizeof(ftsComplex)*2*M_COMPLEX);
    cudaMalloc((void**)&d_q,      sizeof(double)*2*M*N);

    cudaMalloc((void**)&d_boltz_bond_a, sizeof(double)*M_COMPLEX);
    cudaMalloc((void**)&d_boltz_bond_b, sizeof(double)*M_COMPLEX);
    cudaMalloc((void**)&d_boltz_bond_ab, sizeof(double)*M_COMPLEX);

    cudaMalloc((void**)&d_exp_dw_a,   sizeof(double)*M);
    cudaMalloc((void**)&d_exp_dw_b,   sizeof(double)*M);

    cudaMalloc((void**)&d_phi_a, sizeof(double)*M);
    cudaMalloc((void**)&d_phi_b, sizeof(double)*M);

    update();
}
CudaPseudoDiscrete::~CudaPseudoDiscrete()
{
    cufftDestroy(plan_for);
    cufftDestroy(plan_bak);

    cudaFree(d_k_q_in);
    cudaFree(d_q);

    cudaFree(d_boltz_bond_a);
    cudaFree(d_boltz_bond_b);
    cudaFree(d_boltz_bond_ab);

    cudaFree(d_exp_dw_a);
    cudaFree(d_exp_dw_b);

    cudaFree(d_phi_a);
    cudaFree(d_phi_b);
}

void CudaPseudoDiscrete::update()
{
    double bond_length_a, bond_length_b, bond_length_ab;
    const double eps = pc->get_epsilon();
    const double f = pc->get_f();

    const int M_COMPLEX = this->n_complex_grid;
    double boltz_bond_a[M_COMPLEX], boltz_bond_b[M_COMPLEX], boltz_bond_ab[M_COMPLEX];

    bond_length_a = eps*eps/(f*eps*eps + (1.0-f));
    bond_length_b = 1.0/(f*eps*eps + (1.0-f));
    bond_length_ab = 0.5*bond_length_a + 0.5*bond_length_b;

    get_boltz_bond(boltz_bond_a,  bond_length_a,  sb->get_nx(), sb->get_dx(), pc->get_ds());
    get_boltz_bond(boltz_bond_b,  bond_length_b,  sb->get_nx(), sb->get_dx(), pc->get_ds());
    get_boltz_bond(boltz_bond_ab, bond_length_ab, sb->get_nx(), sb->get_dx(), pc->get_ds());

    cudaMemcpy(d_boltz_bond_a,  boltz_bond_a,  sizeof(double)*M_COMPLEX,cudaMemcpyHostToDevice);
    cudaMemcpy(d_boltz_bond_b,  boltz_bond_b,  sizeof(double)*M_COMPLEX,cudaMemcpyHostToDevice);
    cudaMemcpy(d_boltz_bond_ab, boltz_bond_ab, sizeof(double)*M_COMPLEX,cudaMemcpyHostToDevice);
}

std::array<double,3> CudaPseudoDiscrete::dq_dl()
{
    // To calculate stress, we multiply weighted fourier basis to q(k)*q^dagger(-k).
    // Then, we can use results of real-to-complex Fourier transform as it is.
    // It is not problematic, since we only need the real part of stress calculation.

    const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();

    const int DIM  = sb->get_dim();
    const int M    = sb->get_n_grid();
    const int N    = pc->get_n_contour();
    const int N_A  = pc->get_n_contour_a();
    const int M_COMPLEX = this->n_complex_grid;

    const double eps = pc->get_epsilon();
    const double f = pc->get_f();
    const double bond_length_a = eps*eps/(f*eps*eps + (1.0-f));
    const double bond_length_b = 1.0/(f*eps*eps + (1.0-f));
    const double bond_length_ab = 0.5*bond_length_a + 0.5*bond_length_b;
    double bond_length, *d_boltz_bond;

    std::array<double,3> stress;
    double fourier_basis_x[M_COMPLEX] {0.0};
    double fourier_basis_y[M_COMPLEX] {0.0};
    double fourier_basis_z[M_COMPLEX] {0.0};

    double *d_fourier_basis_x;
    double *d_fourier_basis_y;
    double *d_fourier_basis_z;
    double *d_q_in_2m, *d_q_multi, *d_stress_sum;

    get_weighted_fourier_basis(fourier_basis_x, fourier_basis_y, fourier_basis_z, sb->get_nx(), sb->get_dx());

    cudaMalloc((void**)&d_fourier_basis_x, sizeof(double)*M_COMPLEX);
    cudaMalloc((void**)&d_fourier_basis_y, sizeof(double)*M_COMPLEX);
    cudaMalloc((void**)&d_fourier_basis_z, sizeof(double)*M_COMPLEX);
    cudaMalloc((void**)&d_q_in_2m,         sizeof(double)*2*M);
    cudaMalloc((void**)&d_q_multi,         sizeof(double)*M_COMPLEX);
    cudaMalloc((void**)&d_stress_sum,      sizeof(double)*M_COMPLEX);

    cudaMemcpy(d_fourier_basis_x, fourier_basis_x, sizeof(double)*M_COMPLEX,cudaMemcpyHostToDevice);
    cudaMemcpy(d_fourier_basis_y, fourier_basis_y, sizeof(double)*M_COMPLEX,cudaMemcpyHostToDevice);
    cudaMemcpy(d_fourier_basis_z, fourier_basis_z, sizeof(double)*M_COMPLEX,cudaMemcpyHostToDevice);

    thrust::device_ptr<double> temp_gpu_ptr(d_stress_sum);

    for(int i=0; i<3; i++)
        stress[i] = 0.0;

    // Bond between A segments
    for(int n=1; n<N; n++)
    {
        cudaMemcpy(&d_q_in_2m[0], &d_q[M*(2*n-2)],     sizeof(double)*M,cudaMemcpyDeviceToDevice);
        cudaMemcpy(&d_q_in_2m[M], &d_q[M*(2*(N-n)-1)], sizeof(double)*M,cudaMemcpyDeviceToDevice);
        cufftExecD2Z(plan_for, d_q_in_2m, d_k_q_in);

        if ( n < N_A)
        {
            bond_length = bond_length_a;
            d_boltz_bond = d_boltz_bond_a;
        }
        else if ( n == N_A)
        {
            bond_length = bond_length_ab;
            d_boltz_bond = d_boltz_bond_ab;
        }
        else if ( n < N)
        {
            bond_length = bond_length_b;
            d_boltz_bond = d_boltz_bond_b;
        }
        multi_complex_conjugate<<<N_BLOCKS, N_THREADS>>>(d_q_multi, &d_k_q_in[0], &d_k_q_in[M_COMPLEX], M_COMPLEX);
        multi_real<<<N_BLOCKS, N_THREADS>>>(d_q_multi, d_q_multi, d_boltz_bond, bond_length, M_COMPLEX);

        if ( DIM >= 3 )
        {
            multi_real<<<N_BLOCKS, N_THREADS>>>(d_stress_sum, d_q_multi, d_fourier_basis_x, 1.0,   M_COMPLEX);
            stress[0] += thrust::reduce(temp_gpu_ptr, temp_gpu_ptr + M_COMPLEX);
        }
        if ( DIM >= 2 )
        {
            multi_real<<<N_BLOCKS, N_THREADS>>>(d_stress_sum, d_q_multi, d_fourier_basis_y, 1.0,   M_COMPLEX);
            stress[1] += thrust::reduce(temp_gpu_ptr, temp_gpu_ptr + M_COMPLEX);
        }
        if ( DIM >= 1 )
        {
            multi_real<<<N_BLOCKS, N_THREADS>>>(d_stress_sum, d_q_multi, d_fourier_basis_z, 1.0,   M_COMPLEX);
            stress[2] += thrust::reduce(temp_gpu_ptr, temp_gpu_ptr + M_COMPLEX);
        }
    }
    for(int d=0; d<3; d++)
        stress[d] /= 3.0*sb->get_lx(d)*M*M*N/sb->get_volume();

    cudaFree(d_fourier_basis_x);
    cudaFree(d_fourier_basis_y);
    cudaFree(d_fourier_basis_z);
    cudaFree(d_q_in_2m);
    cudaFree(d_q_multi);
    cudaFree(d_stress_sum);

    return stress;
}

void CudaPseudoDiscrete::find_phi(double *phi_a,  double *phi_b,
                                  double *q_1_init, double *q_2_init,
                                  double *w_a, double *w_b, double &single_partition)
{
    const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();

    const int M = sb->get_n_grid();
    const int N = pc->get_n_contour();
    const int N_A = pc->get_n_contour_a();
    const int N_B = pc->get_n_contour_b();
    const double ds = pc->get_ds();

    double exp_dw_a[M];
    double exp_dw_b[M];

    for(int i=0; i<M; i++)
    {
        exp_dw_a[i] = exp(-w_a[i]*ds);
        exp_dw_b[i] = exp(-w_b[i]*ds);
    }

    // Copy array from host memory to device memory
    cudaMemcpy(d_exp_dw_a, exp_dw_a, sizeof(double)*M,cudaMemcpyHostToDevice);
    cudaMemcpy(d_exp_dw_b, exp_dw_b, sizeof(double)*M,cudaMemcpyHostToDevice);

    cudaMemcpy(&d_q[0], q_1_init, sizeof(double)*M, cudaMemcpyHostToDevice);
    cudaMemcpy(&d_q[M], q_2_init, sizeof(double)*M, cudaMemcpyHostToDevice);

    multi_real<<<N_BLOCKS, N_THREADS>>>(&d_q[0], &d_q[0], d_exp_dw_a, 1.0, M);
    multi_real<<<N_BLOCKS, N_THREADS>>>(&d_q[M], &d_q[M], d_exp_dw_b, 1.0, M);

    for(int n=1; n<N; n++)
    {
        if(n < N_A && n < N_B)
            one_step(&d_q[2*M*(n-1)], &d_q[2*M*n], d_boltz_bond_a,  d_boltz_bond_b,  d_exp_dw_a, d_exp_dw_b);
        else if(n < N_A &&  n == N_B)
            one_step(&d_q[2*M*(n-1)], &d_q[2*M*n], d_boltz_bond_a,  d_boltz_bond_ab, d_exp_dw_a, d_exp_dw_a);
        else if(n < N_A &&  n > N_B)
            one_step(&d_q[2*M*(n-1)], &d_q[2*M*n], d_boltz_bond_a,  d_boltz_bond_a,  d_exp_dw_a, d_exp_dw_a);

        else if(n == N_A && n < N_B)
            one_step(&d_q[2*M*(n-1)], &d_q[2*M*n], d_boltz_bond_ab, d_boltz_bond_b,  d_exp_dw_b, d_exp_dw_b);
        else if(n == N_A && n == N_B)
            one_step(&d_q[2*M*(n-1)], &d_q[2*M*n], d_boltz_bond_ab, d_boltz_bond_ab, d_exp_dw_b, d_exp_dw_a);
        else if(n == N_A && n > N_B)
            one_step(&d_q[2*M*(n-1)], &d_q[2*M*n], d_boltz_bond_ab, d_boltz_bond_a,  d_exp_dw_b, d_exp_dw_a);

        else if(n > N_A && n < N_B)
            one_step(&d_q[2*M*(n-1)], &d_q[2*M*n], d_boltz_bond_b,  d_boltz_bond_b,  d_exp_dw_b, d_exp_dw_b);
        else if(n > N_A && n == N_B)
            one_step(&d_q[2*M*(n-1)], &d_q[2*M*n], d_boltz_bond_b,  d_boltz_bond_ab, d_exp_dw_b, d_exp_dw_a);
        else if(n > N_A && n > N_B)
            one_step(&d_q[2*M*(n-1)], &d_q[2*M*n], d_boltz_bond_b,  d_boltz_bond_a,  d_exp_dw_b, d_exp_dw_a);
    }

    //calculates the total partition function
    //d_phi_a is used as a temporary array
    cudaMemcpy(d_phi_a, q_2_init, sizeof(double)*M, cudaMemcpyHostToDevice);
    single_partition = ((CudaSimulationBox *)sb)->inner_product_gpu(&d_q[2*M*(N-1)],d_phi_a);

    // Calculate segment density
    multi_real<<<N_BLOCKS, N_THREADS>>>(d_phi_a, &d_q[0], &d_q[2*M*(N-1)+M], 1.0, M);
    for(int n=1; n<N_A; n++)
    {
        add_multi_real<<<N_BLOCKS, N_THREADS>>>(d_phi_a, &d_q[2*M*n], &d_q[2*M*(N-n-1)+M], 1.0, M);
    }
    multi_real<<<N_BLOCKS, N_THREADS>>>(d_phi_b, &d_q[2*M*N_A], &d_q[2*M*(N_B-1)+M], 1.0, M);
    for(int n=N_A+1; n<N; n++)
    {
        add_multi_real<<<N_BLOCKS, N_THREADS>>>(d_phi_b, &d_q[2*M*n], &d_q[2*M*(N-n-1)+M], 1.0, M);
    }

    // normalize the concentration
    divide_real<<<N_BLOCKS, N_THREADS>>>(d_phi_a, d_phi_a, d_exp_dw_a, (sb->get_volume())/single_partition/N, M);
    divide_real<<<N_BLOCKS, N_THREADS>>>(d_phi_b, d_phi_b, d_exp_dw_b, (sb->get_volume())/single_partition/N, M);

    cudaMemcpy(phi_a, d_phi_a, sizeof(double)*M,cudaMemcpyDeviceToHost);
    cudaMemcpy(phi_b, d_phi_b, sizeof(double)*M,cudaMemcpyDeviceToHost);
}

void CudaPseudoDiscrete::one_step(double *d_q_in,          double *d_q_out,
                                  double *d_boltz_bond_1, double *d_boltz_bond_2,
                                  double *d_exp_dw_1,      double *d_exp_dw_2)
{
    const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();

    const int M = sb->get_n_grid();
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
// Get partial partition functions
// This is made for debugging and testing.
// Do NOT this at main progarams.
void CudaPseudoDiscrete::get_partition(double *q_1_out, int n1, double *q_2_out, int n2)
{
    const int M = sb->get_n_grid();
    const int N = pc->get_n_contour();
    cudaMemcpy(q_1_out, &d_q[M*(2*n1-2)],       sizeof(double)*M,cudaMemcpyDeviceToHost);
    cudaMemcpy(q_2_out, &d_q[M*(2*(N-n2+1)-1)], sizeof(double)*M,cudaMemcpyDeviceToHost);
}
