#define NRANK 3
#define BATCH 2

#include <stdio.h>
#include <cufft.h>
#include "common.h"

__global__ void multiReal(ftsReal* dst,
                          ftsReal* src1,
                          ftsReal* src2,
                          ftsReal  a, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        dst[i] = a * src1[i] * src2[i];
        i += blockDim.x * gridDim.x;
    }
}

__global__ void multiAddReal(ftsReal* dst,
                             ftsReal* src1,
                             ftsReal* src2,
                             ftsReal  a, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        dst[i] = dst[i] + a * src1[i] * src2[i];
        i += blockDim.x * gridDim.x;
    }
}

__global__ void multiFactor(ftsComplex* a,
                            ftsReal* b, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        a[i].x = a[i].x * b[i];
        a[i].y = a[i].y * b[i];
        i += blockDim.x * gridDim.x;
    }
}

__global__ void linComb(ftsReal* dst,
                        ftsReal a,
                        ftsReal* src1,
                        ftsReal b,
                        ftsReal* src2,
                        const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        dst[i] = a*src1[i] + b*src2[i];
        i += blockDim.x * gridDim.x;
    }
}

static int MM, MM_COMPLEX, NN, NNf;
static int N_BLOCKS;
static int N_THREADS;
static ftsReal VOLUME;

static cufftHandle plan_for, plan_bak;

static ftsReal *temp_d, *temp_arr;
static ftsReal *q1_d, *q2_d;
static ftsReal *qstep1_d, *qstep2_d;
static ftsComplex *kqin_d;

static ftsReal *expdwa_d, *expdwa_half_d;
static ftsReal *expdwb_d, *expdwb_half_d;
static ftsReal *phia_d, *phib_d;

static ftsReal *expf_d, *expf_half_d, *seg_d;

static void pseudo_onestep_kernel_overlap(ftsReal *qin1_d, ftsReal *qout1_d,
        ftsReal *qin2_d, ftsReal *qout2_d,
        ftsReal *expdw1_d, ftsReal *expdw1_half_d,
        ftsReal *expdw2_d, ftsReal *expdw2_half_d);

extern "C" void pseudo_cuda_initialize_(
    ftsReal *expf, ftsReal *expf_half,
    ftsReal *seg, ftsReal *pVOLUME,
    int *pII, int *pJJ, int *pKK,
    int *pNN, int *pNNf,
    int *pNUM_BLOCKS,
    int *pTHREADS_PER_BLOCK,
    int *pProcessIdx)
{
    int device;
    int devicesCount;
    struct cudaDeviceProp prop;

    MM         = (*pII) * (*pJJ) * (*pKK);
    MM_COMPLEX = ((*pII)/2+1) * (*pJJ) * (*pKK);
    NN = *pNN;
    NNf= *pNNf;

    VOLUME = *pVOLUME;

    N_BLOCKS = *pNUM_BLOCKS;
    N_THREADS = *pTHREADS_PER_BLOCK;

    printf( "MM: %d, NN: %d, NNf: %d, Volume: %f\n", MM, NN, NNf, VOLUME);
    printf( "N_BLOCKS: %d, N_THREADS: %d\n", N_BLOCKS, N_THREADS);

    cudaGetDeviceCount(&devicesCount);
    cudaSetDevice((*pProcessIdx)%devicesCount);

    printf( "DeviceCount: %d\n", devicesCount );
    printf( "ProcessIdx, DeviceID: %d, %d\n", *pProcessIdx, (*pProcessIdx)%devicesCount);

    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess)
    {
        printf("%s\n", cudaGetErrorString(err));
        exit (1);
    }
    cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess)
    {
        printf("%s\n", cudaGetErrorString(err));
        exit (1);
    }

    printf( "\n--- Current CUDA Device Query ---\n");
    printf( "Device %d : \t\t\t\t%s\n", device, prop.name );
    printf( "Compute capability version : \t\t%d.%d\n", prop.major, prop.minor );
    printf( "Multiprocessor : \t\t\t%d\n", prop.multiProcessorCount );

    printf( "Global memory : \t\t\t%ld MBytes\n", prop.totalGlobalMem/(1024*1024) );
    printf( "Constant memory : \t\t\t%ld Bytes\n", prop.totalConstMem );
    printf( "Shared memory per block : \t\t%ld Bytes\n", prop.sharedMemPerBlock );
    printf( "Registers available per block : \t%d\n", prop.regsPerBlock );

    printf( "Warp size : \t\t\t\t%d\n", prop.warpSize );
    printf( "Maximum threads per block : \t\t%d\n", prop.maxThreadsPerBlock );
    printf( "Max size of a thread block (x,y,z) : \t(%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2] );
    printf( "Max size of a grid size    (x,y,z) : \t(%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2] );

    if(prop.deviceOverlap)
    {
        printf( "Device overlap : \t\t\t Yes \n");
    }
    else
    {
        printf( "Device overlap : \t\t\t No \n");
    }

    if (N_THREADS > prop.maxThreadsPerBlock)
    {
        printf("'threads_per_block' cannot be greater than 'Maximum threads per block'\n");
        exit (1);
    }

    if (N_BLOCKS > prop.maxGridSize[0])
    {
        printf("The number of blocks cannot be greater than 'Max size of a grid size (x)'\n");
        exit (1);
    }
    printf( "\n" );

    int n_grid[NRANK] = {*pKK, *pJJ, *pII};

    cudaMalloc((void**)&temp_d,  sizeof(ftsReal)*MM);
    temp_arr = (ftsReal *) malloc(sizeof(ftsReal)*MM);

    cudaMalloc((void**)&qstep1_d, sizeof(ftsReal)*2*MM);
    cudaMalloc((void**)&qstep2_d, sizeof(ftsReal)*2*MM);
    cudaMalloc((void**)&kqin_d,   sizeof(ftsComplex)*2*MM_COMPLEX);

    cudaMalloc((void**)&q1_d, sizeof(ftsReal)*MM*(NN+1));
    cudaMalloc((void**)&q2_d, sizeof(ftsReal)*MM*(NN+1));

    cudaMalloc((void**)&expdwa_d,      sizeof(ftsReal)*MM);
    cudaMalloc((void**)&expdwb_d,      sizeof(ftsReal)*MM);
    cudaMalloc((void**)&expdwa_half_d, sizeof(ftsReal)*MM);
    cudaMalloc((void**)&expdwb_half_d, sizeof(ftsReal)*MM);

    cudaMalloc((void**)&phia_d, sizeof(ftsReal)*MM);
    cudaMalloc((void**)&phib_d, sizeof(ftsReal)*MM);

    cudaMalloc((void**)&expf_d,         sizeof(ftsReal)*MM_COMPLEX);
    cudaMalloc((void**)&expf_half_d,    sizeof(ftsReal)*MM_COMPLEX);
    cudaMalloc((void**)&seg_d,          sizeof(ftsReal)*MM);

    cudaMemcpy(expf_d,   expf,               sizeof(ftsReal)*MM_COMPLEX,cudaMemcpyHostToDevice);
    cudaMemcpy(expf_half_d,  expf_half,      sizeof(ftsReal)*MM_COMPLEX,cudaMemcpyHostToDevice);
    cudaMemcpy(seg_d,   seg,                 sizeof(ftsReal)*MM,cudaMemcpyHostToDevice);

    /* Create a 3D FFT plan. */
#if USE_SINGLE_PRECISION == 1
    cufftPlanMany(&plan_for, NRANK, n_grid, NULL, 1, 0, NULL, 1, 0, CUFFT_R2C, BATCH);
    cufftPlanMany(&plan_bak, NRANK, n_grid, NULL, 1, 0, NULL, 1, 0, CUFFT_C2R, BATCH);
#else
    cufftPlanMany(&plan_for, NRANK, n_grid, NULL, 1, 0, NULL, 1, 0, CUFFT_D2Z, BATCH);
    cufftPlanMany(&plan_bak, NRANK, n_grid, NULL, 1, 0, NULL, 1, 0, CUFFT_Z2D, BATCH);
#endif

}

extern "C" void pseudo_cuda_finalize_()
{

    cufftDestroy(plan_for);
    cufftDestroy(plan_bak);

    free(temp_arr);
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
    cudaFree(seg_d);
}

extern "C" void pseudo_cuda_run_(ftsReal *phia,   ftsReal *phib,
                                 ftsReal *QQ,
                                 ftsReal *q1_init,     ftsReal *q2_init,
                                 ftsReal *expdwa, ftsReal *expdwa_half,
                                 ftsReal *expdwb, ftsReal *expdwb_half)
{

    /* Copy array from host memory to device memory */
    cudaMemcpy(expdwa_d, expdwa, sizeof(ftsReal)*MM,cudaMemcpyHostToDevice);
    cudaMemcpy(expdwb_d, expdwb, sizeof(ftsReal)*MM,cudaMemcpyHostToDevice);
    cudaMemcpy(expdwa_half_d, expdwa_half, sizeof(ftsReal)*MM,cudaMemcpyHostToDevice);
    cudaMemcpy(expdwb_half_d, expdwb_half, sizeof(ftsReal)*MM,cudaMemcpyHostToDevice);

    cudaMemcpy(&q1_d[0], q1_init, sizeof(ftsReal)*MM,
               cudaMemcpyHostToDevice);
    cudaMemcpy(&q2_d[0], q2_init, sizeof(ftsReal)*MM,
               cudaMemcpyHostToDevice);

    for(int n=0; n<NN; n++)
    {
        if(n<NNf && n<NN-NNf)
        {
            pseudo_onestep_kernel_overlap(
                &q1_d[MM*n], &q1_d[MM*(n+1)],
                &q2_d[MM*n], &q2_d[MM*(n+1)],
                expdwa_d, expdwa_half_d,
                expdwb_d, expdwb_half_d);
        }
        else if(n<NNf && n>=NN-NNf)
        {
            pseudo_onestep_kernel_overlap(
                &q1_d[MM*n], &q1_d[MM*(n+1)],
                &q2_d[MM*n], &q2_d[MM*(n+1)],
                expdwa_d, expdwa_half_d,
                expdwa_d, expdwa_half_d);
        }
        else if(n>=NNf && n<NN-NNf)
        {
            pseudo_onestep_kernel_overlap(
                &q1_d[MM*n], &q1_d[MM*(n+1)],
                &q2_d[MM*n], &q2_d[MM*(n+1)],
                expdwb_d, expdwb_half_d,
                expdwb_d, expdwb_half_d);
        }
        else
        {
            pseudo_onestep_kernel_overlap(
                &q1_d[MM*n], &q1_d[MM*(n+1)],
                &q2_d[MM*n], &q2_d[MM*(n+1)],
                expdwb_d, expdwb_half_d,
                expdwa_d, expdwa_half_d);
        }
    }

    /* Calculate Segment Density */
    /* segment concentration. only half contribution from the end */
    multiReal<<<N_BLOCKS, N_THREADS>>>(phib_d, &q1_d[MM*NN], &q2_d[0], 0.5, MM);
    //printf("NN, %5.3f\n", 1.0/3.0);
    /* the B block segment */
    for(int n=NN-1; n>NNf; n--)
    {
        //printf("%d, %5.3f\n", n, 2.0*((n % 2) +1)/3.0);
        multiAddReal<<<N_BLOCKS, N_THREADS>>>(phib_d, &q1_d[MM*n], &q2_d[MM*(NN-n)], 1.0, MM);
    }

    /* the junction is half A and half B */
    multiReal<<<N_BLOCKS, N_THREADS>>>(phia_d, &q1_d[MM*NNf], &q2_d[MM*(NN-NNf)], 0.5, MM);
    linComb<<<N_BLOCKS, N_THREADS>>>(phib_d, 1.0, phib_d, 1.0, phia_d, MM);

    //printf("%d, %5.3f\n", NNf, 1.0/3.0);
    /* calculates the total partition function */
    multiReal<<<N_BLOCKS, N_THREADS>>>(temp_d, phia_d, seg_d, 2.0, MM);
    cudaMemcpy(temp_arr, temp_d, sizeof(ftsReal)*MM,cudaMemcpyDeviceToHost);
    *QQ = 0.0;
    for(int i=0; i<MM; i++)
    {
        *QQ = *QQ + temp_arr[i];
    }

    /* the A block segment */
    for(int n=NNf-1; n>0; n--)
    {
        //printf("%d, %5.3f\n", n, 2.0*((n % 2) +1)/3.0);
        multiAddReal<<<N_BLOCKS, N_THREADS>>>(phia_d, &q1_d[MM*n], &q2_d[MM*(NN-n)], 1.0, MM);
    }
    /* only half contribution from the end */
    //printf("0, %5.3f\n", 1.0/3.0);
    multiAddReal<<<N_BLOCKS, N_THREADS>>>(phia_d, &q1_d[0], &q2_d[MM*NN], 0.5, MM);

    /* normalize the concentration */
    linComb<<<N_BLOCKS, N_THREADS>>>(phia_d, (VOLUME)/(*QQ)/NN, phia_d, 0.0, phia_d, MM);
    linComb<<<N_BLOCKS, N_THREADS>>>(phib_d, (VOLUME)/(*QQ)/NN, phib_d, 0.0, phib_d, MM);

    cudaMemcpy(phia, phia_d, sizeof(ftsReal)*MM,cudaMemcpyDeviceToHost);
    cudaMemcpy(phib, phib_d, sizeof(ftsReal)*MM,cudaMemcpyDeviceToHost);
}

/* Advance two partial partition functions simultaneously using Richardson extrapolation.
* Note that cufft doesn't fully utilize GPU cores unless n_grid is sufficiently large.
* To increase GPU usage, we introduce kernel overlapping. */
static void pseudo_onestep_kernel_overlap(ftsReal *qin1_d, ftsReal *qout1_d,
        ftsReal *qin2_d, ftsReal *qout2_d,
        ftsReal *expdw1_d, ftsReal *expdw1_half_d,
        ftsReal *expdw2_d, ftsReal *expdw2_half_d)
{
    /*-------------- step 1 ---------- */
    /* Evaluate e^(-w*ds/2) in real space. */
    multiReal<<<N_BLOCKS, N_THREADS>>>(&qstep1_d[0],  qin1_d, expdw1_d, 1.0, MM);
    multiReal<<<N_BLOCKS, N_THREADS>>>(&qstep1_d[MM], qin2_d, expdw2_d, 1.0, MM);

    /* Execute a Forward 3D FFT . */
#if USE_SINGLE_PRECISION == 1
    cufftExecR2C(plan_for, qstep1_d, kqin_d);
#else
    cufftExecD2Z(plan_for, qstep1_d, kqin_d);
#endif
    /* Multiply e^(-k^2 ds/6) in fourier space */
    multiFactor<<<N_BLOCKS, N_THREADS>>>(&kqin_d[0],          expf_d, MM_COMPLEX);
    multiFactor<<<N_BLOCKS, N_THREADS>>>(&kqin_d[MM_COMPLEX], expf_d, MM_COMPLEX);
    /* Execute a backward 3D FFT . */
#if USE_SINGLE_PRECISION == 1
    cufftExecC2R(plan_bak, kqin_d, qstep1_d);
#else
    cufftExecZ2D(plan_bak, kqin_d, qstep1_d);
#endif
    /* Evaluate e^(-w*ds/2) in real space. */
    multiReal<<<N_BLOCKS, N_THREADS>>>(&qstep1_d[0],  &qstep1_d[0],   expdw1_d, 1.0/((ftsReal)MM), MM);
    multiReal<<<N_BLOCKS, N_THREADS>>>(&qstep1_d[MM], &qstep1_d[MM],  expdw2_d, 1.0/((ftsReal)MM), MM);

    /*-------------- step 2 ---------- */
    /* Evaluate e^(-w*ds/4) in real space. */
    multiReal<<<N_BLOCKS, N_THREADS>>>(&qstep2_d[0],  qin1_d, expdw1_half_d, 1.0, MM);
    multiReal<<<N_BLOCKS, N_THREADS>>>(&qstep2_d[MM], qin2_d, expdw2_half_d, 1.0, MM);

    /* Execute a Forward 3D FFT . */
#if USE_SINGLE_PRECISION == 1
    cufftExecR2C(plan_for, qstep2_d, kqin_d);
#else
    cufftExecD2Z(plan_for, qstep2_d, kqin_d);
#endif
    /* Multiply e^(-k^2 ds/12) in fourier space */
    multiFactor<<<N_BLOCKS, N_THREADS>>>(&kqin_d[0],          expf_half_d, MM_COMPLEX);
    multiFactor<<<N_BLOCKS, N_THREADS>>>(&kqin_d[MM_COMPLEX], expf_half_d, MM_COMPLEX);
    /* Execute a backward 3D FFT . */
#if USE_SINGLE_PRECISION == 1
    cufftExecC2R(plan_bak, kqin_d, qstep2_d);
#else
    cufftExecZ2D(plan_bak, kqin_d, qstep2_d);
#endif
    /* Evaluate e^(-w*ds/2) in real space. */
    multiReal<<<N_BLOCKS, N_THREADS>>>(&qstep2_d[0],  &qstep2_d[0],  expdw1_d, 1.0/((ftsReal)MM), MM);
    multiReal<<<N_BLOCKS, N_THREADS>>>(&qstep2_d[MM], &qstep2_d[MM], expdw2_d, 1.0/((ftsReal)MM), MM);
    /* Execute a Forward 3D FFT . */
#if USE_SINGLE_PRECISION == 1
    cufftExecR2C(plan_for, qstep2_d, kqin_d);
#else
    cufftExecD2Z(plan_for, qstep2_d, kqin_d);
#endif
    /* Multiply e^(-k^2 ds/12) in fourier space */
    multiFactor<<<N_BLOCKS, N_THREADS>>>(&kqin_d[0],          expf_half_d, MM_COMPLEX);
    multiFactor<<<N_BLOCKS, N_THREADS>>>(&kqin_d[MM_COMPLEX], expf_half_d, MM_COMPLEX);
    /* Execute a backward 3D FFT . */
#if USE_SINGLE_PRECISION == 1
    cufftExecC2R(plan_bak, kqin_d, qstep2_d);
#else
    cufftExecZ2D(plan_bak, kqin_d, qstep2_d);
#endif
    /* Evaluate e^(-w*ds/4) in real space. */
    multiReal<<<N_BLOCKS, N_THREADS>>>(&qstep2_d[0],  &qstep2_d[0],  expdw1_half_d, 1.0/((ftsReal)MM), MM);
    multiReal<<<N_BLOCKS, N_THREADS>>>(&qstep2_d[MM], &qstep2_d[MM], expdw2_half_d, 1.0/((ftsReal)MM), MM);
    /*-------------- step 3 ---------- */
    linComb<<<N_BLOCKS, N_THREADS>>>(qout1_d, 4.0/3.0, &qstep2_d[0],  -1.0/3.0, &qstep1_d[0],  MM);
    linComb<<<N_BLOCKS, N_THREADS>>>(qout2_d, 4.0/3.0, &qstep2_d[MM], -1.0/3.0, &qstep1_d[MM], MM);
}

/* Get partial partition functions of last step
* This is made for debugging and testing.
* Do NOT this at main progarams.
* */
extern "C" void pseudo_get_partition_(ftsReal *q1_out,  ftsReal *q2_out)
{
    cudaMemcpy(q1_out, &q1_d[NN*MM], sizeof(ftsReal)*MM,cudaMemcpyDeviceToHost);
    cudaMemcpy(q2_out, &q2_d[NN*MM], sizeof(ftsReal)*MM,cudaMemcpyDeviceToHost);
}
