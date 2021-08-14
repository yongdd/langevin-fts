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
__global__ static void multi_dot_kernel(
    int n_comp, double *dv_d, double *g_d, double *h_d, double *sum_d, unsigned int MM)
{
    extern __shared__ double sdata[];
// each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int gridSize = blockDim.x*gridDim.x;
    sdata[tid] = 0.0;
    while (i < MM)
    {
        for(int n = 0; n < n_comp; n++)
        {
            sdata[tid] += dv_d[i] * g_d[i+n*MM] * h_d[i+n*MM];
        }
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
    std::array<int,3> nx, std::array<double,3> lx)
{
    for(int i=0; i<3; i++)
    {
        this->nx[i] = nx[i];
        this->lx[i] = lx[i];
        this->dx[i] = lx[i]/nx[i];
    }
    // the number of total grids
    MM = nx[0]*nx[1]*nx[2];
    // weight factor for integral
    dv = new double[MM];
    for(int i=0; i<MM; i++)
        dv[i] = dx[0]*dx[1]*dx[2];
    cudaMalloc((void**)&dv_d, sizeof(double)*MM);
    cudaMemcpy(dv_d, dv,      sizeof(double)*MM,cudaMemcpyHostToDevice);

    // temporal storage
    sum = new double[MM];
    cudaMalloc((void**)&sum_d, sizeof(double)*MM);
    // system polymer
    volume = lx[0]*lx[1]*lx[2];

    N_BLOCKS = CudaCommon::get_instance().N_BLOCKS;
    N_THREADS = CudaCommon::get_instance().N_THREADS;
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
    delete[] dv;
    delete[] sum;
    cudaFree(dv_d);
    cudaFree(sum_d);
}
//-----------------------------------------------------------
// This method calculates integral g*h
double CudaSimulationBox::dot(double *g, double *h)
{
    double sum{0.0};
    for(int i=0; i<MM; i++)
        sum += dv[i] * g[i] * h[i];
    return sum;
}
double CudaSimulationBox::multi_dot(int n_comp, double *g, double *h)
{
    double sum{0.0};
    for(int n=0; n<n_comp; n++)
    {
        for(int i=0; i<MM; i++)
            sum += dv[i] * g[i+n*MM] * h[i+n*MM];
    }
    return sum;
}
double CudaSimulationBox::multi_dot_gpu(int n_comp, double *g_d, double *h_d)
{
    double total{0.0};

    switch(N_THREADS)
    {
    case 1024:
        multi_dot_kernel<1024><<<N_BLOCKS, N_THREADS, N_THREADS*sizeof(double)>>>(n_comp, dv_d, g_d, h_d, sum_d, MM);
        break;
    case 512:
        multi_dot_kernel<512><<<N_BLOCKS, N_THREADS, N_THREADS*sizeof(double)>>>(n_comp, dv_d, g_d, h_d, sum_d, MM);
        break;
    case 256:
        multi_dot_kernel<256><<<N_BLOCKS, N_THREADS, N_THREADS*sizeof(double)>>>(n_comp, dv_d, g_d, h_d, sum_d, MM);
        break;
    case 128:
        multi_dot_kernel<128><<<N_BLOCKS, N_THREADS, N_THREADS*sizeof(double)>>>(n_comp, dv_d, g_d, h_d, sum_d, MM);
        break;
    case 64:
        multi_dot_kernel<64><<<N_BLOCKS, N_THREADS, N_THREADS*sizeof(double)>>>(n_comp, dv_d, g_d, h_d, sum_d, MM);
        break;
    case 32:
        multi_dot_kernel<32><<<N_BLOCKS, N_THREADS, N_THREADS*sizeof(double)>>>(n_comp, dv_d, g_d, h_d, sum_d, MM);
        break;
    case 16:
        multi_dot_kernel<16><<<N_BLOCKS, N_THREADS, N_THREADS*sizeof(double)>>>(n_comp, dv_d, g_d, h_d, sum_d, MM);
        break;
    case 8:
        multi_dot_kernel<8><<<N_BLOCKS, N_THREADS, N_THREADS*sizeof(double)>>>(n_comp, dv_d, g_d, h_d, sum_d, MM);
        break;
    case 4:
        multi_dot_kernel<4><<<N_BLOCKS, N_THREADS, N_THREADS*sizeof(double)>>>(n_comp, dv_d, g_d, h_d, sum_d, MM);
        break;
    case 2:
        multi_dot_kernel<2><<<N_BLOCKS, N_THREADS, N_THREADS*sizeof(double)>>>(n_comp, dv_d, g_d, h_d, sum_d, MM);
        break;
    case 1:
        multi_dot_kernel<1><<<N_BLOCKS, N_THREADS, N_THREADS*sizeof(double)>>>(n_comp, dv_d, g_d, h_d, sum_d, MM);
        break;
    }
    cudaMemcpy(sum, sum_d, sizeof(double)*N_BLOCKS,cudaMemcpyDeviceToHost);
    for(int i=0; i<N_BLOCKS; i++)
    {
        total += sum[i];
    }
    return total;
}
//-----------------------------------------------------------
// This method makes the input a zero-meaned matrix
void CudaSimulationBox::zero_mean(double *w)
{
    double sum{0.0};
    for(int i=0; i<MM; i++)
        sum += dv[i]*w[i];
    for(int i=0; i<MM; i++)
        w[i] -= sum/volume;
}
