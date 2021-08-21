#ifndef CUDA_COMMON_H_
#define CUDA_COMMON_H_

#include <complex>
#include <cufft.h>

typedef cufftDoubleComplex ftsComplex;

// Design Pattern : Singleton (Scott Meyer)

class CudaCommon
{
private:

    int n_blocks;
    int n_threads;

    CudaCommon();
    ~CudaCommon();
    // Disable copy constructor
    CudaCommon(const CudaCommon &) = delete;
    CudaCommon& operator= (const CudaCommon &) = delete;
public:


    static CudaCommon& get_instance()
    {
        static CudaCommon* instance = new CudaCommon();
        return *instance;
    };
    static void display_info();
    static void set(int n_blocks, int n_threads, int process_idx);
    
    int get_n_blocks();
    int get_n_threads();
    
    void set_n_blocks(int n_blocks);
    void set_n_threads(int n_threads);
    void set_idx(int process_idx);
};

__global__ void multiReal(double* dst,
                          double* src1,
                          double* src2,
                          double  a, const int M);
                          
__global__ void mutipleMultiReal(int n_comp,
                          double* dst,
                          double* src1,
                          double* src2,
                          double  a, const int M);
                                    
__global__ void divideReal(double* dst,
                          double* src1,
                          double* src2,
                          double  a, const int M);
                          
__global__ void addMultiReal(double* dst,
                             double* src1,
                             double* src2,
                             double  a, const int M);

__global__ void linComb(double* dst,
                        double a,
                        double* src1,
                        double b,
                        double* src2,
                        const int M);

__global__ void addLinComb(double* dst,
                           double a,
                           double* src1,
                           double b,
                           double* src2,
                           const int M);

__global__ void multiComplexReal(ftsComplex* a,
                                 double* b, const int M);

template <unsigned int blockSize>
__device__ void warp_reduce(volatile double* sdata, int tid)
{
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}
template <unsigned int blockSize>
__global__ void multiplyReductionKernel(
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

#endif
