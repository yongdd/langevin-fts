
#include <iostream>
#include <complex>
#include "CudaPseudoDiscrete.h"
#include "CudaSimulationBox.h"

CudaPseudoDiscrete::CudaPseudoDiscrete(
    SimulationBox *sb,
    PolymerChain *pc)
    : Pseudo(sb, pc)
{
    const int NRANK{3};
    const int BATCH{2};

    int n_grid[NRANK] = {sb->get_nx(0),sb->get_nx(1),sb->get_nx(2)};

    const int MM = sb->get_MM();
    const int NN = pc->get_NN();

    cudaMalloc((void**)&temp_d,  sizeof(double)*MM);

    cudaMalloc((void**)&qstep_d, sizeof(double)*2*MM);
    cudaMalloc((void**)&kqin_d,  sizeof(ftsComplex)*2*MM_COMPLEX);

    cudaMalloc((void**)&q1_d, sizeof(double)*MM*NN);
    cudaMalloc((void**)&q2_d, sizeof(double)*MM*NN);

    cudaMalloc((void**)&expf_d,   sizeof(double)*MM_COMPLEX);
    cudaMalloc((void**)&expdwa_d, sizeof(double)*MM);
    cudaMalloc((void**)&expdwb_d, sizeof(double)*MM);

    cudaMalloc((void**)&phia_d, sizeof(double)*MM);
    cudaMalloc((void**)&phib_d, sizeof(double)*MM);

    this->temp_arr = new double[MM];

    cudaMemcpy(expf_d, expf, sizeof(double)*MM_COMPLEX,cudaMemcpyHostToDevice);

    /* Create a 3D FFT plan. */
    cufftPlanMany(&plan_for, NRANK, n_grid, NULL, 1, 0, NULL, 1, 0, CUFFT_D2Z, BATCH);
    cufftPlanMany(&plan_bak, NRANK, n_grid, NULL, 1, 0, NULL, 1, 0, CUFFT_Z2D, BATCH);
}
CudaPseudoDiscrete::~CudaPseudoDiscrete()
{
    cufftDestroy(plan_for);
    cufftDestroy(plan_bak);

    cudaFree(qstep_d);
    cudaFree(kqin_d);

    cudaFree(temp_d);
    cudaFree(q1_d);
    cudaFree(q2_d);

    cudaFree(expf_d);
    cudaFree(expdwa_d);
    cudaFree(expdwb_d);

    cudaFree(phia_d);
    cudaFree(phib_d);

    delete[] temp_arr;
}
void CudaPseudoDiscrete::find_phi(double *phia,  double *phib,
                                  double *q1_init, double *q2_init,
                                  double *wa, double *wb, double &QQ)
{
    const int N_BLOCKS = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();

    const int MM = sb->get_MM();
    const int NN = pc->get_NN();
    const int NN_A = pc->get_NN_A();
    const int NN_B = pc->get_NN_B();
    const double ds = pc->get_ds();

    double expdwa[MM];
    double expdwb[MM];

    for(int i=0; i<MM; i++)
    {
        expdwa[i] = exp(-wa[i]*ds);
        expdwb[i] = exp(-wb[i]*ds);
    }

    // Copy array from host memory to device memory
    cudaMemcpy(expdwa_d, expdwa, sizeof(double)*MM,cudaMemcpyHostToDevice);
    cudaMemcpy(expdwb_d, expdwb, sizeof(double)*MM,cudaMemcpyHostToDevice);

    cudaMemcpy(&q1_d[0], q1_init, sizeof(double)*MM, cudaMemcpyHostToDevice);
    cudaMemcpy(&q2_d[0], q2_init, sizeof(double)*MM, cudaMemcpyHostToDevice);

    multiReal<<<N_BLOCKS, N_THREADS>>>(&q1_d[0], &q1_d[0], expdwa_d, 1.0, MM);
    multiReal<<<N_BLOCKS, N_THREADS>>>(&q2_d[0], &q2_d[0], expdwb_d, 1.0, MM);

    for(int n=1; n<NN; n++)
    {
        if(n < NN_A && n < NN_B)
        {
            onestep(&q1_d[MM*(n-1)], &q2_d[MM*(n-1)],
                    &q1_d[MM*n],     &q2_d[MM*n],
                    expdwa_d,        expdwb_d);
        }
        else if(n < NN_A &&  n >= NN_B)
        {
            onestep(&q1_d[MM*(n-1)], &q2_d[MM*(n-1)],
                    &q1_d[MM*n],     &q2_d[MM*n],
                    expdwa_d,        expdwa_d);
        }
        else if(n >= NN_A && n < NN_B)
        {
            onestep(&q1_d[MM*(n-1)], &q2_d[MM*(n-1)],
                    &q1_d[MM*n],     &q2_d[MM*n],
                    expdwb_d,        expdwb_d);
        }
        else
        {
            onestep(&q1_d[MM*(n-1)], &q2_d[MM*(n-1)],
                    &q1_d[MM*n],     &q2_d[MM*n],
                    expdwb_d,        expdwa_d);
        }
    }

    //calculates the total partition function
    //phia_d is used as a temporary array
    cudaMemcpy(phia_d, q2_init, sizeof(double)*MM, cudaMemcpyHostToDevice);
    QQ = ((CudaSimulationBox *)sb)->inner_product_gpu(&q1_d[MM*(NN-1)],phia_d);
    
    // Calculate segment density
    multiReal<<<N_BLOCKS, N_THREADS>>>(phia_d, &q1_d[0], &q2_d[MM*(NN-1)], 1.0, MM);
    for(int n=1; n<NN_A; n++)
    {
        addMultiReal<<<N_BLOCKS, N_THREADS>>>(phia_d, &q1_d[MM*n], &q2_d[MM*(NN-n-1)], 1.0, MM);
    }
    multiReal<<<N_BLOCKS, N_THREADS>>>(phib_d, &q1_d[MM*NN_A], &q2_d[MM*(NN_B-1)], 1.0, MM);
    for(int n=NN_A+1; n<NN; n++)
    {
        addMultiReal<<<N_BLOCKS, N_THREADS>>>(phib_d, &q1_d[MM*n], &q2_d[MM*(NN-n-1)], 1.0, MM);
    }

    // normalize the concentration
    divideReal<<<N_BLOCKS, N_THREADS>>>(phia_d, phia_d, expdwa_d, (sb->get_volume())/QQ/NN, MM);
    divideReal<<<N_BLOCKS, N_THREADS>>>(phib_d, phib_d, expdwb_d, (sb->get_volume())/QQ/NN, MM);

    cudaMemcpy(phia, phia_d, sizeof(double)*MM,cudaMemcpyDeviceToHost);
    cudaMemcpy(phib, phib_d, sizeof(double)*MM,cudaMemcpyDeviceToHost);
}

// Advance two partial partition functions simultaneously using Richardson extrapolation.
// Note that cufft doesn't fully utilize GPU cores unless n_grid is sufficiently large.
// To increase GPU usage, we use FFT Batch.
void CudaPseudoDiscrete::onestep(double *qin1_d, double *qin2_d,
                                 double *qout1_d, double *qout2_d,
                                 double *expdw1_d, double *expdw2_d)
{
    const int N_BLOCKS = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();
    const int MM = sb->get_MM();

    //-------------- step 1 ----------
    cudaMemcpy(&qstep_d[0],  qin1_d, sizeof(double)*MM,cudaMemcpyDeviceToDevice);
    cudaMemcpy(&qstep_d[MM], qin2_d, sizeof(double)*MM,cudaMemcpyDeviceToDevice);

    // Execute a Forward 3D FFT
    cufftExecD2Z(plan_for, qstep_d, kqin_d);

    // Multiply e^(-k^2 ds/6) in fourier space
    multiComplexReal<<<N_BLOCKS, N_THREADS>>>(&kqin_d[0],          expf_d, MM_COMPLEX);
    multiComplexReal<<<N_BLOCKS, N_THREADS>>>(&kqin_d[MM_COMPLEX], expf_d, MM_COMPLEX);

    // Execute a backward 3D FFT
    cufftExecZ2D(plan_bak, kqin_d, qstep_d);

    // Evaluate e^(-w*ds) in real space
    multiReal<<<N_BLOCKS, N_THREADS>>>(qout1_d, &qstep_d[0],   expdw1_d, 1.0/((double)MM), MM);
    multiReal<<<N_BLOCKS, N_THREADS>>>(qout2_d, &qstep_d[MM],  expdw2_d, 1.0/((double)MM), MM);
}
// Get partial partition functions
// This is made for debugging and testing.
// Do NOT this at main progarams.

void CudaPseudoDiscrete::get_partition(double *q1_out,  double *q2_out, int n)
{
    const int MM = sb->get_MM();
    const int NN = pc->get_NN();
    cudaMemcpy(q1_out, &q1_d[(n-1)*MM],  sizeof(double)*MM,cudaMemcpyDeviceToHost);
    cudaMemcpy(q2_out, &q2_d[(n-1)*MM], sizeof(double)*MM,cudaMemcpyDeviceToHost);
}
