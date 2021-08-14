#ifndef CUDA_COMMON_H_
#define CUDA_COMMON_H_

#include <complex>
#include <cufft.h>

typedef cufftDoubleComplex ftsComplex;

// Design Pattern : Singleton (Scott Meyer)

class CudaCommon
{
private:
    CudaCommon(int N_BLOCKS, int N_THREADS, int process_idx);
    ~CudaCommon();
    // Disable copy constructor
    CudaCommon(const CudaCommon &) = delete;
    CudaCommon& operator= (const CudaCommon &) = delete;
public:
    int N_BLOCKS;
    int N_THREADS;

    static CudaCommon& get_instance(int N_BLOCKS=256, int N_THREADS=256, int process_idx=0)
    {
        static CudaCommon* instance = new CudaCommon(N_BLOCKS, N_THREADS, process_idx);
        return *instance;
    };
    static void initialize(int N_BLOCKS, int N_THREADS, int process_idx);
};

__global__ void multiReal(double* dst,
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

#endif
