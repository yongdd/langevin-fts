#include <complex>
#include "CudaPseudo.h"
#include "CudaSimulationBox.h"

CudaPseudo::CudaPseudo(
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

    cudaMalloc((void**)&qstep1_d, sizeof(double)*2*MM);
    cudaMalloc((void**)&qstep2_d, sizeof(double)*2*MM);
    cudaMalloc((void**)&kqin_d,   sizeof(ftsComplex)*2*MM_COMPLEX);

    cudaMalloc((void**)&q1_d, sizeof(double)*MM*(NN+1));
    cudaMalloc((void**)&q2_d, sizeof(double)*MM*(NN+1));

    cudaMalloc((void**)&expdwa_d,      sizeof(double)*MM);
    cudaMalloc((void**)&expdwb_d,      sizeof(double)*MM);
    cudaMalloc((void**)&expdwa_half_d, sizeof(double)*MM);
    cudaMalloc((void**)&expdwb_half_d, sizeof(double)*MM);

    cudaMalloc((void**)&phia_d, sizeof(double)*MM);
    cudaMalloc((void**)&phib_d, sizeof(double)*MM);

    cudaMalloc((void**)&expf_d,      sizeof(double)*MM_COMPLEX);
    cudaMalloc((void**)&expf_half_d, sizeof(double)*MM_COMPLEX);

    this->temp_arr = new double[MM];

    cudaMemcpy(expf_d,   expf,          sizeof(double)*MM_COMPLEX,cudaMemcpyHostToDevice);
    cudaMemcpy(expf_half_d,  expf_half, sizeof(double)*MM_COMPLEX,cudaMemcpyHostToDevice);

    /* Create a 3D FFT plan. */
    cufftPlanMany(&plan_for, NRANK, n_grid, NULL, 1, 0, NULL, 1, 0, CUFFT_D2Z, BATCH);
    cufftPlanMany(&plan_bak, NRANK, n_grid, NULL, 1, 0, NULL, 1, 0, CUFFT_Z2D, BATCH);
}
CudaPseudo::~CudaPseudo()
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
void CudaPseudo::find_phi(double *phia,  double *phib,
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
    double expdwa_half[MM];
    double expdwb_half[MM];

    for(int i=0; i<MM; i++)
    {
        expdwa     [i] = exp(-wa[i]*ds*0.5);
        expdwb     [i] = exp(-wb[i]*ds*0.5);
        expdwa_half[i] = exp(-wa[i]*ds*0.25);
        expdwb_half[i] = exp(-wb[i]*ds*0.25);
    }

    // Copy array from host memory to device memory
    cudaMemcpy(expdwa_d, expdwa, sizeof(double)*MM,cudaMemcpyHostToDevice);
    cudaMemcpy(expdwb_d, expdwb, sizeof(double)*MM,cudaMemcpyHostToDevice);
    cudaMemcpy(expdwa_half_d, expdwa_half, sizeof(double)*MM,cudaMemcpyHostToDevice);
    cudaMemcpy(expdwb_half_d, expdwb_half, sizeof(double)*MM,cudaMemcpyHostToDevice);

    cudaMemcpy(&q1_d[0], q1_init, sizeof(double)*MM,
               cudaMemcpyHostToDevice);
    cudaMemcpy(&q2_d[0], q2_init, sizeof(double)*MM,
               cudaMemcpyHostToDevice);

    for(int n=0; n<NN; n++)
    {
        if(n < NN_A && n < NN_B)
        {
            onestep(
                &q1_d[MM*n], &q1_d[MM*(n+1)],
                &q2_d[MM*n], &q2_d[MM*(n+1)],
                expdwa_d, expdwa_half_d,
                expdwb_d, expdwb_half_d);
        }
        else if(n < NN_A &&  n >= NN_B)
        {
            onestep(
                &q1_d[MM*n], &q1_d[MM*(n+1)],
                &q2_d[MM*n], &q2_d[MM*(n+1)],
                expdwa_d, expdwa_half_d,
                expdwa_d, expdwa_half_d);
        }
        else if(n >= NN_A && n < NN_B)
        {
            onestep(
                &q1_d[MM*n], &q1_d[MM*(n+1)],
                &q2_d[MM*n], &q2_d[MM*(n+1)],
                expdwb_d, expdwb_half_d,
                expdwb_d, expdwb_half_d);
        }
        else
        {
            onestep(
                &q1_d[MM*n], &q1_d[MM*(n+1)],
                &q2_d[MM*n], &q2_d[MM*(n+1)],
                expdwb_d, expdwb_half_d,
                expdwa_d, expdwa_half_d);
        }
    }

    // Calculate Segment Density
    // dvment concentration. only half contribution from the end
    multiReal<<<N_BLOCKS, N_THREADS>>>(phib_d, &q1_d[MM*NN], &q2_d[0], 0.5, MM);
    //printf("NN, %5.3f\n", 1.0/3.0);
    // the B block dvment
    for(int n=NN-1; n>NN_A; n--)
    {
        //printf("%d, %5.3f\n", n, 2.0*((n % 2) +1)/3.0);
        addMultiReal<<<N_BLOCKS, N_THREADS>>>(phib_d, &q1_d[MM*n], &q2_d[MM*(NN-n)], 1.0, MM);
    }

    // the junction is half A and half B
    multiReal<<<N_BLOCKS, N_THREADS>>>(phia_d, &q1_d[MM*NN_A], &q2_d[MM*(NN_B)], 0.5, MM);
    linComb<<<N_BLOCKS, N_THREADS>>>(phib_d, 1.0, phib_d, 1.0, phia_d, MM);

    //printf("%d, %5.3f\n", NN_A, 1.0/3.0);
    //calculates the total partition function
    multiReal<<<N_BLOCKS, N_THREADS>>>(temp_d, phia_d, ((CudaSimulationBox *)sb)->dv_d, 2.0, MM);
    cudaMemcpy(temp_arr, temp_d, sizeof(double)*MM,cudaMemcpyDeviceToHost);
    QQ = 0.0;
    for(int i=0; i<MM; i++)
    {
        QQ = QQ + temp_arr[i];
    }

    // the A block dvment
    for(int n=NN_A-1; n>0; n--)
    {
        //printf("%d, %5.3f\n", n, 2.0*((n % 2) +1)/3.0);
        addMultiReal<<<N_BLOCKS, N_THREADS>>>(phia_d, &q1_d[MM*n], &q2_d[MM*(NN-n)], 1.0, MM);
    }
    // only half contribution from the end
    //printf("0, %5.3f\n", 1.0/3.0);
    addMultiReal<<<N_BLOCKS, N_THREADS>>>(phia_d, &q1_d[0], &q2_d[MM*NN], 0.5, MM);

    // normalize the concentration
    linComb<<<N_BLOCKS, N_THREADS>>>(phia_d, (sb->get_volume())/QQ/NN, phia_d, 0.0, phia_d, MM);
    linComb<<<N_BLOCKS, N_THREADS>>>(phib_d, (sb->get_volume())/QQ/NN, phib_d, 0.0, phib_d, MM);

    cudaMemcpy(phia, phia_d, sizeof(double)*MM,cudaMemcpyDeviceToHost);
    cudaMemcpy(phib, phib_d, sizeof(double)*MM,cudaMemcpyDeviceToHost);
}

// Advance two partial partition functions simultaneously using Richardson extrapolation.
// Note that cufft doesn't fully utilize GPU cores unless n_grid is sufficiently large.
// To increase GPU usage, we use FFT Batch.
void CudaPseudo::onestep(double *qin1_d, double *qout1_d,
                         double *qin2_d, double *qout2_d,
                         double *expdw1_d, double *expdw1_half_d,
                         double *expdw2_d, double *expdw2_half_d)
{
    const int N_BLOCKS = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();
    
    const int MM = sb->get_MM();
        
    //-------------- step 1 ---------- 
    // Evaluate e^(-w*ds/2) in real space
    multiReal<<<N_BLOCKS, N_THREADS>>>(&qstep1_d[0],  qin1_d, expdw1_d, 1.0, MM);
    multiReal<<<N_BLOCKS, N_THREADS>>>(&qstep1_d[MM], qin2_d, expdw2_d, 1.0, MM);

    // Execute a Forward 3D FFT
    cufftExecD2Z(plan_for, qstep1_d, kqin_d);

    // Multiply e^(-k^2 ds/6) in fourier space
    multiComplexReal<<<N_BLOCKS, N_THREADS>>>(&kqin_d[0],          expf_d, MM_COMPLEX);
    multiComplexReal<<<N_BLOCKS, N_THREADS>>>(&kqin_d[MM_COMPLEX], expf_d, MM_COMPLEX);

    // Execute a backward 3D FFT
    cufftExecZ2D(plan_bak, kqin_d, qstep1_d);

    // Evaluate e^(-w*ds/2) in real space
    multiReal<<<N_BLOCKS, N_THREADS>>>(&qstep1_d[0],  &qstep1_d[0],   expdw1_d, 1.0/((double)MM), MM);
    multiReal<<<N_BLOCKS, N_THREADS>>>(&qstep1_d[MM], &qstep1_d[MM],  expdw2_d, 1.0/((double)MM), MM);

    //-------------- step 2 ----------
    // Evaluate e^(-w*ds/4) in real space
    multiReal<<<N_BLOCKS, N_THREADS>>>(&qstep2_d[0],  qin1_d, expdw1_half_d, 1.0, MM);
    multiReal<<<N_BLOCKS, N_THREADS>>>(&qstep2_d[MM], qin2_d, expdw2_half_d, 1.0, MM);

    // Execute a Forward 3D FFT
    cufftExecD2Z(plan_for, qstep2_d, kqin_d);

    // Multiply e^(-k^2 ds/12) in fourier space
    multiComplexReal<<<N_BLOCKS, N_THREADS>>>(&kqin_d[0],          expf_half_d, MM_COMPLEX);
    multiComplexReal<<<N_BLOCKS, N_THREADS>>>(&kqin_d[MM_COMPLEX], expf_half_d, MM_COMPLEX);

    // Execute a backward 3D FFT
    cufftExecZ2D(plan_bak, kqin_d, qstep2_d);

    // Evaluate e^(-w*ds/2) in real space
    multiReal<<<N_BLOCKS, N_THREADS>>>(&qstep2_d[0],  &qstep2_d[0],  expdw1_d, 1.0/((double)MM), MM);
    multiReal<<<N_BLOCKS, N_THREADS>>>(&qstep2_d[MM], &qstep2_d[MM], expdw2_d, 1.0/((double)MM), MM);
    // Execute a Forward 3D FFT
    cufftExecD2Z(plan_for, qstep2_d, kqin_d);

    // Multiply e^(-k^2 ds/12) in fourier space
    multiComplexReal<<<N_BLOCKS, N_THREADS>>>(&kqin_d[0],          expf_half_d, MM_COMPLEX);
    multiComplexReal<<<N_BLOCKS, N_THREADS>>>(&kqin_d[MM_COMPLEX], expf_half_d, MM_COMPLEX);

    // Execute a backward 3D FFT
    cufftExecZ2D(plan_bak, kqin_d, qstep2_d);

    // Evaluate e^(-w*ds/4) in real space.
    multiReal<<<N_BLOCKS, N_THREADS>>>(&qstep2_d[0],  &qstep2_d[0],  expdw1_half_d, 1.0/((double)MM), MM);
    multiReal<<<N_BLOCKS, N_THREADS>>>(&qstep2_d[MM], &qstep2_d[MM], expdw2_half_d, 1.0/((double)MM), MM);
    //-------------- step 3 ----------
    linComb<<<N_BLOCKS, N_THREADS>>>(qout1_d, 4.0/3.0, &qstep2_d[0],  -1.0/3.0, &qstep1_d[0],  MM);
    linComb<<<N_BLOCKS, N_THREADS>>>(qout2_d, 4.0/3.0, &qstep2_d[MM], -1.0/3.0, &qstep1_d[MM], MM);
}
// Get partial partition functions
// This is made for debugging and testing.
// Do NOT this at main progarams.

void CudaPseudo::get_partition(double *q1_out,  double *q2_out, int n)
{
    const int MM = sb->get_MM();
    cudaMemcpy(q1_out, &q1_d[n*MM], sizeof(double)*MM,cudaMemcpyDeviceToHost);
    cudaMemcpy(q2_out, &q2_d[n*MM], sizeof(double)*MM,cudaMemcpyDeviceToHost);
}
