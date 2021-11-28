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
__global__ static void multiply_reduction_kernel(
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
    cudaMalloc((void**)&dv_d, sizeof(double)*n_grid);
    cudaMemcpy(dv_d, dv,      sizeof(double)*n_grid,cudaMemcpyHostToDevice);

    // temporal storage
    sum = new double[n_grid];
    cudaMalloc((void**)&sum_d, sizeof(double)*n_grid);
    cudaMalloc((void**)&multiple_d, sizeof(double)*n_grid);

    const int N_BLOCKS = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();

    if (N_BLOCKS > n_grid)
    {
        std::cout<< "'the number of grids'{" <<N_BLOCKS ;
        std::cout<< "} should not be less 'the number of grids'{";
        std::cout<< n_grid << "}." << std::endl;
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
    const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();

    switch(N_THREADS)
    {
    case 1024:
        multiply_reduction_kernel<1024>
        <<<N_BLOCKS, N_THREADS, N_THREADS*sizeof(double)>>>
        (dv_d, g_d, sum_d, n_grid);
        break;
    case 512:
        multiply_reduction_kernel<512>
        <<<N_BLOCKS, N_THREADS, N_THREADS*sizeof(double)>>>
        (dv_d, g_d, sum_d, n_grid);
        break;
    case 256:
        multiply_reduction_kernel<256>
        <<<N_BLOCKS, N_THREADS, N_THREADS*sizeof(double)>>>
        (dv_d, g_d, sum_d, n_grid);
        break;
    case 128:
        multiply_reduction_kernel<128>
        <<<N_BLOCKS, N_THREADS, N_THREADS*sizeof(double)>>>
        (dv_d, g_d, sum_d, n_grid);
        break;
    case 64:
        multiply_reduction_kernel<64>
        <<<N_BLOCKS, N_THREADS, N_THREADS*sizeof(double)>>>
        (dv_d, g_d, sum_d, n_grid);
        break;
    case 32:
        multiply_reduction_kernel<32>
        <<<N_BLOCKS, N_THREADS, N_THREADS*sizeof(double)>>>
        (dv_d, g_d, sum_d, n_grid);
        break;
    case 16:
        multiply_reduction_kernel<16>
        <<<N_BLOCKS, N_THREADS, N_THREADS*sizeof(double)>>>
        (dv_d, g_d, sum_d, n_grid);
        break;
    case 8:
        multiply_reduction_kernel<8>
        <<<N_BLOCKS, N_THREADS, N_THREADS*sizeof(double)>>>
        (dv_d, g_d, sum_d, n_grid);
        break;
    case 4:
        multiply_reduction_kernel<4>
        <<<N_BLOCKS, N_THREADS, N_THREADS*sizeof(double)>>>
        (dv_d, g_d, sum_d, n_grid);
        break;
    case 2:
        multiply_reduction_kernel<2>
        <<<N_BLOCKS, N_THREADS, N_THREADS*sizeof(double)>>>
        (dv_d, g_d, sum_d, n_grid);
        break;
    case 1:
        multiply_reduction_kernel<1>
        <<<N_BLOCKS, N_THREADS, N_THREADS*sizeof(double)>>>
        (dv_d, g_d, sum_d, n_grid);
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

    multi_real<<<N_BLOCKS, N_THREADS>>>(multiple_d, g_d, h_d, 1.0, n_grid);
    return CudaSimulationBox::integral_gpu(multiple_d);
}
//-----------------------------------------------------------
double CudaSimulationBox::mutiple_inner_product_gpu(int n_comp, double *g_d, double *h_d)
{
    const int N_BLOCKS = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();

    mutiple_multi_real<<<N_BLOCKS, N_THREADS>>>(n_comp, multiple_d, g_d, h_d, 1.0, n_grid);
    return CudaSimulationBox::integral_gpu(multiple_d);
}
