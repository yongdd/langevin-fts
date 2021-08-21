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
#endif
