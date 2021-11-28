#include <complex>
#include "CudaPseudoGaussian.h"
#include "CudaSimulationBox.h"

CudaPseudoGaussian::CudaPseudoGaussian(
    SimulationBox *sb,
    PolymerChain *pc)
    : Pseudo(sb, pc)
{
    const int M = sb->get_n_grid();
    const int N = pc->get_n_contour();
    const int M_COMPLEX = this->n_complex_grid;

    if(sb->get_dim() == 3)
    {
        // create a 3D FFT plan
        const int NRANK{3};
        const int BATCH{2};
        int n_grid[NRANK] = {sb->get_nx(0),sb->get_nx(1),sb->get_nx(2)};

        cufftPlanMany(&plan_for, NRANK, n_grid, NULL, 1, 0, NULL, 1, 0, CUFFT_D2Z, BATCH);
        cufftPlanMany(&plan_bak, NRANK, n_grid, NULL, 1, 0, NULL, 1, 0, CUFFT_Z2D, BATCH);
    }
    else if(sb->get_dim() == 2)
    {
        // create a 2D FFT plan
        const int NRANK{2};
        const int BATCH{2};
        int n_grid[NRANK] = {sb->get_nx(0),sb->get_nx(1)};

        cufftPlanMany(&plan_for, NRANK, n_grid, NULL, 1, 0, NULL, 1, 0, CUFFT_D2Z, BATCH);
        cufftPlanMany(&plan_bak, NRANK, n_grid, NULL, 1, 0, NULL, 1, 0, CUFFT_Z2D, BATCH);
    }
    else if(sb->get_dim() == 1)
    {
        // create a 1D FFT plan
        const int NRANK{1};
        const int BATCH{2};
        int n_grid[NRANK] = {sb->get_nx(0)};

        cufftPlanMany(&plan_for, NRANK, n_grid, NULL, 1, 0, NULL, 1, 0, CUFFT_D2Z, BATCH);
        cufftPlanMany(&plan_bak, NRANK, n_grid, NULL, 1, 0, NULL, 1, 0, CUFFT_Z2D, BATCH);
    }
    
    cudaMalloc((void**)&temp_d,  sizeof(double)*M);

    cudaMalloc((void**)&qstep1_d, sizeof(double)*2*M);
    cudaMalloc((void**)&qstep2_d, sizeof(double)*2*M);
    cudaMalloc((void**)&kqin_d,   sizeof(ftsComplex)*2*M_COMPLEX);

    cudaMalloc((void**)&q1_d, sizeof(double)*M*(N+1));
    cudaMalloc((void**)&q2_d, sizeof(double)*M*(N+1));

    cudaMalloc((void**)&expdwa_d,      sizeof(double)*M);
    cudaMalloc((void**)&expdwb_d,      sizeof(double)*M);
    cudaMalloc((void**)&expdwa_half_d, sizeof(double)*M);
    cudaMalloc((void**)&expdwb_half_d, sizeof(double)*M);

    cudaMalloc((void**)&phia_d, sizeof(double)*M);
    cudaMalloc((void**)&phib_d, sizeof(double)*M);

    cudaMalloc((void**)&expf_d,      sizeof(double)*M_COMPLEX);
    cudaMalloc((void**)&expf_half_d, sizeof(double)*M_COMPLEX);

    this->temp_arr = new double[M];

    cudaMemcpy(expf_d,   expf,          sizeof(double)*M_COMPLEX,cudaMemcpyHostToDevice);
    cudaMemcpy(expf_half_d,  expf_half, sizeof(double)*M_COMPLEX,cudaMemcpyHostToDevice);
}
CudaPseudoGaussian::~CudaPseudoGaussian()
{
    cufftDestroy(plan_for);
    cufftDestroy(plan_bak);

    cudaFree(qstep1_d);
    cudaFree(qstep2_d);
    cudaFree(kqin_d);

    cudaFree(temp_d);
    cudaFree(q1_d);
    cudaFree(q2_d);

    cudaFree(expdwa_d);
    cudaFree(expdwb_d);
    cudaFree(expdwa_half_d);
    cudaFree(expdwb_half_d);
    cudaFree(phia_d);
    cudaFree(phib_d);

    cudaFree(expf_d);
    cudaFree(expf_half_d);
    
    delete[] temp_arr;
}
void CudaPseudoGaussian::find_phi(double *phia,  double *phib,
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
    double expdwa_half[M];
    double expdwb_half[M];

    for(int i=0; i<M; i++)
    {
        expdwa     [i] = exp(-wa[i]*ds*0.5);
        expdwb     [i] = exp(-wb[i]*ds*0.5);
        expdwa_half[i] = exp(-wa[i]*ds*0.25);
        expdwb_half[i] = exp(-wb[i]*ds*0.25);
    }

    // Copy array from host memory to device memory
    cudaMemcpy(expdwa_d, expdwa, sizeof(double)*M,cudaMemcpyHostToDevice);
    cudaMemcpy(expdwb_d, expdwb, sizeof(double)*M,cudaMemcpyHostToDevice);
    cudaMemcpy(expdwa_half_d, expdwa_half, sizeof(double)*M,cudaMemcpyHostToDevice);
    cudaMemcpy(expdwb_half_d, expdwb_half, sizeof(double)*M,cudaMemcpyHostToDevice);

    cudaMemcpy(&q1_d[0], q1_init, sizeof(double)*M,
               cudaMemcpyHostToDevice);
    cudaMemcpy(&q2_d[0], q2_init, sizeof(double)*M,
               cudaMemcpyHostToDevice);

    for(int n=0; n<N; n++)
    {
        if(n < N_A && n < N_B)
        {
            onestep(
                &q1_d[M*n], &q1_d[M*(n+1)],
                &q2_d[M*n], &q2_d[M*(n+1)],
                expdwa_d, expdwa_half_d,
                expdwb_d, expdwb_half_d);
        }
        else if(n < N_A &&  n >= N_B)
        {
            onestep(
                &q1_d[M*n], &q1_d[M*(n+1)],
                &q2_d[M*n], &q2_d[M*(n+1)],
                expdwa_d, expdwa_half_d,
                expdwa_d, expdwa_half_d);
        }
        else if(n >= N_A && n < N_B)
        {
            onestep(
                &q1_d[M*n], &q1_d[M*(n+1)],
                &q2_d[M*n], &q2_d[M*(n+1)],
                expdwb_d, expdwb_half_d,
                expdwb_d, expdwb_half_d);
        }
        else
        {
            onestep(
                &q1_d[M*n], &q1_d[M*(n+1)],
                &q2_d[M*n], &q2_d[M*(n+1)],
                expdwb_d, expdwb_half_d,
                expdwa_d, expdwa_half_d);
        }
    }

    // calculate Segment Density
    // segment concentration. only half contribution from the end
    multi_real<<<N_BLOCKS, N_THREADS>>>(phib_d, &q1_d[M*N], &q2_d[0], 0.5, M);
    // the B block segment
    for(int n=N-1; n>N_A; n--)
        add_multi_real<<<N_BLOCKS, N_THREADS>>>(phib_d, &q1_d[M*n], &q2_d[M*(N-n)], 1.0, M);

    // the junction is half A and half B
    multi_real<<<N_BLOCKS, N_THREADS>>>(phia_d, &q1_d[M*N_A], &q2_d[M*(N_B)], 0.5, M);
    lin_comb<<<N_BLOCKS, N_THREADS>>>(phib_d, 1.0, phib_d, 1.0, phia_d, M);

    // calculates the total partition function
    QQ = 2*((CudaSimulationBox *)sb)->integral_gpu(phia_d);

    // the A block segment
    for(int n=N_A-1; n>0; n--)
        add_multi_real<<<N_BLOCKS, N_THREADS>>>(phia_d, &q1_d[M*n], &q2_d[M*(N-n)], 1.0, M);
        
    // only half contribution from the end
    add_multi_real<<<N_BLOCKS, N_THREADS>>>(phia_d, &q1_d[0], &q2_d[M*N], 0.5, M);

    // normalize the concentration
    lin_comb<<<N_BLOCKS, N_THREADS>>>(phia_d, (sb->get_volume())/QQ/N, phia_d, 0.0, phia_d, M);
    lin_comb<<<N_BLOCKS, N_THREADS>>>(phib_d, (sb->get_volume())/QQ/N, phib_d, 0.0, phib_d, M);

    cudaMemcpy(phia, phia_d, sizeof(double)*M,cudaMemcpyDeviceToHost);
    cudaMemcpy(phib, phib_d, sizeof(double)*M,cudaMemcpyDeviceToHost);
}

// Advance two partial partition functions simultaneously using Richardson extrapolation.
// Note that cufft doesn't fully utilize GPU cores unless n_grid is sufficiently large.
// To increase GPU usage, we use FFT Batch.
void CudaPseudoGaussian::onestep(double *qin1_d, double *qout1_d,
                         double *qin2_d, double *qout2_d,
                         double *expdw1_d, double *expdw1_half_d,
                         double *expdw2_d, double *expdw2_half_d)
{
    const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();
    
    const int M = sb->get_n_grid();
    const int M_COMPLEX = this->n_complex_grid;
        
    //-------------- step 1 ---------- 
    // Evaluate e^(-w*ds/2) in real space
    multi_real<<<N_BLOCKS, N_THREADS>>>(&qstep1_d[0], qin1_d, expdw1_d, 1.0, M);
    multi_real<<<N_BLOCKS, N_THREADS>>>(&qstep1_d[M], qin2_d, expdw2_d, 1.0, M);

    // Execute a Forward FFT
    cufftExecD2Z(plan_for, qstep1_d, kqin_d);

    // Multiply e^(-k^2 ds/6) in fourier space
    multi_complex_real<<<N_BLOCKS, N_THREADS>>>(&kqin_d[0],         expf_d, M_COMPLEX);
    multi_complex_real<<<N_BLOCKS, N_THREADS>>>(&kqin_d[M_COMPLEX], expf_d, M_COMPLEX);

    // Execute a backward FFT
    cufftExecZ2D(plan_bak, kqin_d, qstep1_d);

    // Evaluate e^(-w*ds/2) in real space
    multi_real<<<N_BLOCKS, N_THREADS>>>(&qstep1_d[0], &qstep1_d[0], expdw1_d, 1.0/((double)M), M);
    multi_real<<<N_BLOCKS, N_THREADS>>>(&qstep1_d[M], &qstep1_d[M], expdw2_d, 1.0/((double)M), M);

    //-------------- step 2 ----------
    // Evaluate e^(-w*ds/4) in real space
    multi_real<<<N_BLOCKS, N_THREADS>>>(&qstep2_d[0], qin1_d, expdw1_half_d, 1.0, M);
    multi_real<<<N_BLOCKS, N_THREADS>>>(&qstep2_d[M], qin2_d, expdw2_half_d, 1.0, M);

    // Execute a Forward FFT
    cufftExecD2Z(plan_for, qstep2_d, kqin_d);

    // Multiply e^(-k^2 ds/12) in fourier space
    multi_complex_real<<<N_BLOCKS, N_THREADS>>>(&kqin_d[0],         expf_half_d, M_COMPLEX);
    multi_complex_real<<<N_BLOCKS, N_THREADS>>>(&kqin_d[M_COMPLEX], expf_half_d, M_COMPLEX);

    // Execute a backward FFT
    cufftExecZ2D(plan_bak, kqin_d, qstep2_d);

    // Evaluate e^(-w*ds/2) in real space
    multi_real<<<N_BLOCKS, N_THREADS>>>(&qstep2_d[0], &qstep2_d[0], expdw1_d, 1.0/((double)M), M);
    multi_real<<<N_BLOCKS, N_THREADS>>>(&qstep2_d[M], &qstep2_d[M], expdw2_d, 1.0/((double)M), M);
    // Execute a Forward FFT
    cufftExecD2Z(plan_for, qstep2_d, kqin_d);

    // Multiply e^(-k^2 ds/12) in fourier space
    multi_complex_real<<<N_BLOCKS, N_THREADS>>>(&kqin_d[0],         expf_half_d, M_COMPLEX);
    multi_complex_real<<<N_BLOCKS, N_THREADS>>>(&kqin_d[M_COMPLEX], expf_half_d, M_COMPLEX);

    // Execute a backward FFT
    cufftExecZ2D(plan_bak, kqin_d, qstep2_d);

    // Evaluate e^(-w*ds/4) in real space.
    multi_real<<<N_BLOCKS, N_THREADS>>>(&qstep2_d[0],  &qstep2_d[0], expdw1_half_d, 1.0/((double)M), M);
    multi_real<<<N_BLOCKS, N_THREADS>>>(&qstep2_d[M], &qstep2_d[M],  expdw2_half_d, 1.0/((double)M), M);
    //-------------- step 3 ----------
    lin_comb<<<N_BLOCKS, N_THREADS>>>(qout1_d, 4.0/3.0, &qstep2_d[0], -1.0/3.0, &qstep1_d[0],  M);
    lin_comb<<<N_BLOCKS, N_THREADS>>>(qout2_d, 4.0/3.0, &qstep2_d[M], -1.0/3.0, &qstep1_d[M], M);
}
// Get partial partition functions
// This is made for debugging and testing.
// Do NOT this at main progarams.

void CudaPseudoGaussian::get_partition(double *q1_out,  double *q2_out, int n)
{
    const int M = sb->get_n_grid();
    cudaMemcpy(q1_out, &q1_d[n*M], sizeof(double)*M,cudaMemcpyDeviceToHost);
    cudaMemcpy(q2_out, &q2_d[n*M], sizeof(double)*M,cudaMemcpyDeviceToHost);
}
