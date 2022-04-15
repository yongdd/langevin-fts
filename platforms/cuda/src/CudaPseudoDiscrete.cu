
#include <iostream>
#include <complex>
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
        n_grid[0] = sb->get_nx(0);
        n_grid[1] = sb->get_nx(1);
    }
    else if(sb->get_dim() == 1)
    {
        n_grid[0] = sb->get_nx(0);
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
    double step_a, step_b, step_ab;
    const double eps = pc->get_epsilon();
    const double f = pc->get_f();
    
    const int M_COMPLEX = this->n_complex_grid;
    double boltz_bond_a[M_COMPLEX], boltz_bond_b[M_COMPLEX], boltz_bond_ab[M_COMPLEX];
    
    step_a = eps*eps/(f*eps*eps + (1.0-f));
    step_b = 1.0/(f*eps*eps + (1.0-f));
    step_ab = 0.5*step_a + 0.5*step_b;
    
    set_boltz_bond(boltz_bond_a,  step_a,  sb->get_nx(), sb->get_dx(), pc->get_ds());
    set_boltz_bond(boltz_bond_b,  step_b,  sb->get_nx(), sb->get_dx(), pc->get_ds());
    set_boltz_bond(boltz_bond_ab, step_ab, sb->get_nx(), sb->get_dx(), pc->get_ds());
    
    cudaMemcpy(d_boltz_bond_a,  boltz_bond_a,  sizeof(double)*M_COMPLEX,cudaMemcpyHostToDevice);
    cudaMemcpy(d_boltz_bond_b,  boltz_bond_b,  sizeof(double)*M_COMPLEX,cudaMemcpyHostToDevice);
    cudaMemcpy(d_boltz_bond_ab, boltz_bond_ab, sizeof(double)*M_COMPLEX,cudaMemcpyHostToDevice);
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
