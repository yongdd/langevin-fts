/*-------------------------------------------------------------
* This class defines simulation box parameters and provide
* methods that compute inner product in a given geometry.
*--------------------------------------------------------------*/
#include <iostream>
#include "CudaSimulationBox.h"
#include "CudaCommon.h"

template <unsigned int blockSize>
__device__ static void warp_reduce(volatile double* sdata, int tid)
{
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize>
__global__ static void multiplyReductionKernel(
    double *g_d, double *h_d, double *sum_d, unsigned int M)
{
    extern __shared__ double sdata[];
// each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int gridSize = blockDim.x*gridDim.x;
    sdata[tid] = 0.0;
    while (i < M)
    {
        sdata[tid] += g_d[i] * h_d[i];
        i += gridSize;
    }
    __syncthreads();

// do reduction in shared mem
    if (blockSize >= 1024)
    {
        if (tid < 512)
        {
            sdata[tid] += sdata[tid + 512];
        }
        __syncthreads();
    }
    if (blockSize >= 512)
    {
        if (tid < 256)
        {
            sdata[tid] += sdata[tid + 256];
        }
        __syncthreads();
    }
    if (blockSize >= 256)
    {
        if (tid < 128)
            sdata[tid] += sdata[tid + 128];
        __syncthreads();
    }

    if (blockSize >= 128)
    {
        if (tid < 64)
            sdata[tid] += sdata[tid + 64];
        __syncthreads();
    }
    if (tid < 32) warp_reduce<blockSize>(sdata, tid);
    // write result for this block to global mem
    if (tid == 0) sum_d[blockIdx.x] = sdata[0];
}

//----------------- Constructor -----------------------------
CudaSimulationBox::CudaSimulationBox(
    std::vector<int> nx, std::vector<double> lx)
    : SimulationBox(nx, lx)
{
    initialize();
}
void CudaSimulationBox::initialize()
{
    cudaMalloc((void**)&dv_d, sizeof(double)*MM);
    cudaMemcpy(dv_d, dv,      sizeof(double)*MM,cudaMemcpyHostToDevice);

    // temporal storage
    sum = new double[MM];
    cudaMalloc((void**)&sum_d, sizeof(double)*MM);
    cudaMalloc((void**)&multiple_d, sizeof(double)*MM);

    const int N_BLOCKS = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();

    if (N_BLOCKS > MM)
    {
        std::cout<< "'the number of grids'{" <<N_BLOCKS ;
        std::cout<< "} should not be less 'the number of grids'{";
        std::cout<< MM << "}." << std::endl;
        exit(-1);
    }
}
//----------------- Destructor -----------------------------
CudaSimulationBox::~CudaSimulationBox()
{
    delete[] sum;
    cudaFree(dv_d);
    cudaFree(sum_d);
    cudaFree(multiple_d);
}
//-----------------------------------------------------------
double CudaSimulationBox::integral_gpu(double *g_d)
{
    const int N_BLOCKS = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();

    switch(N_THREADS)
    {
    case 1024:
        multiplyReductionKernel<1024>
        <<<N_BLOCKS, N_THREADS, N_THREADS*sizeof(double)>>>
        (dv_d, g_d, sum_d, MM);
        break;
    case 512:
        multiplyReductionKernel<512>
        <<<N_BLOCKS, N_THREADS, N_THREADS*sizeof(double)>>>
        (dv_d, g_d, sum_d, MM);
        break;
    case 256:
        multiplyReductionKernel<256>
        <<<N_BLOCKS, N_THREADS, N_THREADS*sizeof(double)>>>
        (dv_d, g_d, sum_d, MM);
        break;
    case 128:
        multiplyReductionKernel<128>
        <<<N_BLOCKS, N_THREADS, N_THREADS*sizeof(double)>>>
        (dv_d, g_d, sum_d, MM);
        break;
    case 64:
        multiplyReductionKernel<64>
        <<<N_BLOCKS, N_THREADS, N_THREADS*sizeof(double)>>>
        (dv_d, g_d, sum_d, MM);
        break;
    case 32:
        multiplyReductionKernel<32>
        <<<N_BLOCKS, N_THREADS, N_THREADS*sizeof(double)>>>
        (dv_d, g_d, sum_d, MM);
        break;
    case 16:
        multiplyReductionKernel<16>
        <<<N_BLOCKS, N_THREADS, N_THREADS*sizeof(double)>>>
        (dv_d, g_d, sum_d, MM);
        break;
    case 8:
        multiplyReductionKernel<8>
        <<<N_BLOCKS, N_THREADS, N_THREADS*sizeof(double)>>>
        (dv_d, g_d, sum_d, MM);
        break;
    case 4:
        multiplyReductionKernel<4>
        <<<N_BLOCKS, N_THREADS, N_THREADS*sizeof(double)>>>
        (dv_d, g_d, sum_d, MM);
        break;
    case 2:
        multiplyReductionKernel<2>
        <<<N_BLOCKS, N_THREADS, N_THREADS*sizeof(double)>>>
        (dv_d, g_d, sum_d, MM);
        break;
    case 1:
        multiplyReductionKernel<1>
        <<<N_BLOCKS, N_THREADS, N_THREADS*sizeof(double)>>>
        (dv_d, g_d, sum_d, MM);
        break;
    }
    cudaMemcpy(sum, sum_d, sizeof(double)*N_BLOCKS,cudaMemcpyDeviceToHost);
    double total{0.0};
    for(int i=0; i<N_BLOCKS; i++)
    {
        total += sum[i];
    }
    return total;
}
//-----------------------------------------------------------
double CudaSimulationBox::inner_product_gpu(double *g_d, double *h_d)
{
    const int N_BLOCKS = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();

    multiReal<<<N_BLOCKS, N_THREADS>>>(multiple_d, g_d, h_d, 1.0, MM);
    return CudaSimulationBox::integral_gpu(multiple_d);
}
//-----------------------------------------------------------
double CudaSimulationBox::mutiple_inner_product_gpu(int n_comp, double *g_d, double *h_d)
{
    const int N_BLOCKS = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();

    mutipleMultiReal<<<N_BLOCKS, N_THREADS>>>(n_comp, multiple_d, g_d, h_d, 1.0, MM);
    return CudaSimulationBox::integral_gpu(multiple_d);
}
