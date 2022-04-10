
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
    cudaMalloc((void**)&kqin_d,  sizeof(ftsComplex)*2*M_COMPLEX);
    cudaMalloc((void**)&q_d, sizeof(double)*2*M*N);

    cudaMalloc((void**)&expf_d,   sizeof(double)*M_COMPLEX);
    cudaMalloc((void**)&expdwa_d, sizeof(double)*M);
    cudaMalloc((void**)&expdwb_d, sizeof(double)*M);

    cudaMalloc((void**)&phia_d, sizeof(double)*M);
    cudaMalloc((void**)&phib_d, sizeof(double)*M);
    
    cudaMemcpy(expf_d, expf, sizeof(double)*M_COMPLEX,cudaMemcpyHostToDevice);
}
CudaPseudoDiscrete::~CudaPseudoDiscrete()
{
    cufftDestroy(plan_for);
    cufftDestroy(plan_bak);

    cudaFree(kqin_d);
    cudaFree(q_d);

    cudaFree(expf_d);
    cudaFree(expdwa_d);
    cudaFree(expdwb_d);

    cudaFree(phia_d);
    cudaFree(phib_d);
}

void CudaPseudoDiscrete::update()
{
    const int M_COMPLEX = this->n_complex_grid;
    
    Pseudo::update();
    cudaMemcpy(expf_d, expf, sizeof(double)*M_COMPLEX,cudaMemcpyHostToDevice);
}

void CudaPseudoDiscrete::find_phi(double *phia,  double *phib,
                                  double *q1_init, double *q2_init,
                                  double *wa, double *wb, double &QQ)
{
    const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();

    const int M = sb->get_n_grid();
    const int N = pc->get_n_contour();
    const int N_A = pc->get_n_contour_a();
    const int N_B = pc->get_n_contour_b();
    const double ds = pc->get_ds();

    double expdwa[M];
    double expdwb[M];

    for(int i=0; i<M; i++)
    {
        expdwa[i] = exp(-wa[i]*ds);
        expdwb[i] = exp(-wb[i]*ds);
    }

    // Copy array from host memory to device memory
    cudaMemcpy(expdwa_d, expdwa, sizeof(double)*M,cudaMemcpyHostToDevice);
    cudaMemcpy(expdwb_d, expdwb, sizeof(double)*M,cudaMemcpyHostToDevice);

    cudaMemcpy(&q_d[0], q1_init, sizeof(double)*M, cudaMemcpyHostToDevice);
    cudaMemcpy(&q_d[M], q2_init, sizeof(double)*M, cudaMemcpyHostToDevice);

    multi_real<<<N_BLOCKS, N_THREADS>>>(&q_d[0], &q_d[0], expdwa_d, 1.0, M);
    multi_real<<<N_BLOCKS, N_THREADS>>>(&q_d[M], &q_d[M], expdwb_d, 1.0, M);

    for(int n=1; n<N; n++)
    {
        if(n < N_A && n < N_B)
            one_step(&q_d[2*M*(n-1)], &q_d[2*M*n], expdwa_d, expdwb_d);
        else if(n < N_A &&  n >= N_B)
            one_step(&q_d[2*M*(n-1)], &q_d[2*M*n], expdwa_d, expdwa_d);
        else if(n >= N_A && n < N_B)
            one_step(&q_d[2*M*(n-1)], &q_d[2*M*n], expdwb_d, expdwb_d);
        else
            one_step(&q_d[2*M*(n-1)], &q_d[2*M*n], expdwb_d, expdwa_d);
    }

    //calculates the total partition function
    //phia_d is used as a temporary array
    cudaMemcpy(phia_d, q2_init, sizeof(double)*M, cudaMemcpyHostToDevice);
    QQ = ((CudaSimulationBox *)sb)->inner_product_gpu(&q_d[2*M*(N-1)],phia_d);
    
    // Calculate segment density
    multi_real<<<N_BLOCKS, N_THREADS>>>(phia_d, &q_d[0], &q_d[2*M*(N-1)+M], 1.0, M);
    for(int n=1; n<N_A; n++)
    {
        add_multi_real<<<N_BLOCKS, N_THREADS>>>(phia_d, &q_d[2*M*n], &q_d[2*M*(N-n-1)+M], 1.0, M);
    }
    multi_real<<<N_BLOCKS, N_THREADS>>>(phib_d, &q_d[2*M*N_A], &q_d[2*M*(N_B-1)+M], 1.0, M);
    for(int n=N_A+1; n<N; n++)
    {
        add_multi_real<<<N_BLOCKS, N_THREADS>>>(phib_d, &q_d[2*M*n], &q_d[2*M*(N-n-1)+M], 1.0, M);
    }

    // normalize the concentration
    divide_real<<<N_BLOCKS, N_THREADS>>>(phia_d, phia_d, expdwa_d, (sb->get_volume())/QQ/N, M);
    divide_real<<<N_BLOCKS, N_THREADS>>>(phib_d, phib_d, expdwb_d, (sb->get_volume())/QQ/N, M);

    cudaMemcpy(phia, phia_d, sizeof(double)*M,cudaMemcpyDeviceToHost);
    cudaMemcpy(phib, phib_d, sizeof(double)*M,cudaMemcpyDeviceToHost);
}

void CudaPseudoDiscrete::one_step(double *qin_d, double *qout_d,
                                 double *expdw1_d, double *expdw2_d)
{
    const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();
    
    const int M = sb->get_n_grid();
    const int M_COMPLEX = this->n_complex_grid;

    //-------------- step 1 ----------
    // Execute a Forward FFT
    cufftExecD2Z(plan_for, qin_d, kqin_d);

    // Multiply e^(-k^2 ds/6) in fourier space
    multi_complex_real<<<N_BLOCKS, N_THREADS>>>(&kqin_d[0],         expf_d, M_COMPLEX);
    multi_complex_real<<<N_BLOCKS, N_THREADS>>>(&kqin_d[M_COMPLEX], expf_d, M_COMPLEX);

    // Execute a backward FFT
    cufftExecZ2D(plan_bak, kqin_d, qout_d);

    // Evaluate e^(-w*ds) in real space
    multi_real<<<N_BLOCKS, N_THREADS>>>(&qout_d[0], &qout_d[0], expdw1_d, 1.0/((double)M), M);
    multi_real<<<N_BLOCKS, N_THREADS>>>(&qout_d[M], &qout_d[M], expdw2_d, 1.0/((double)M), M);
}
// Get partial partition functions
// This is made for debugging and testing.
// Do NOT this at main progarams.
void CudaPseudoDiscrete::get_partition(double *q1_out, int n1, double *q2_out, int n2)
{
    const int M = sb->get_n_grid();
    const int N = pc->get_n_contour();
    cudaMemcpy(q1_out, &q_d[M*(2*n1-2)], sizeof(double)*M,cudaMemcpyDeviceToHost);
    cudaMemcpy(q2_out, &q_d[M*(2*n2-1)], sizeof(double)*M,cudaMemcpyDeviceToHost);
}
