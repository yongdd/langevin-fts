#ifndef CUDA_COMMON_H_
#define CUDA_COMMON_H_

#include <complex>
#include <cufft.h>
#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>
#include "Exception.h"

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
        try{
            static CudaCommon* instance = new CudaCommon();
            return *instance;
        }
        catch(std::exception& exc)
        {
            throw_without_line_number(exc.what());
        }
    };
    void set(int n_blocks, int n_threads, int process_idx);
    
    int get_n_blocks();
    int get_n_threads();
    
    void set_n_blocks(int n_blocks);
    void set_n_threads(int n_threads);
    void set_idx(int process_idx);
};

#define gpu_error_check(code) throw_on_cuda_error((code), __FILE__, __LINE__, __func__);
void throw_on_cuda_error(cudaError_t code, const char *file, int line, const char *func);

__global__ void multi_real(double* dst,
                          double* src1,
                          double* src2,
                          double  a, const int M);
                          
__global__ void mutiple_multi_real(int n_comp,
                          double* dst,
                          double* src1,
                          double* src2,
                          double  a, const int M);
                                    
__global__ void divide_real(double* dst,
                          double* src1,
                          double* src2,
                          double  a, const int M);
                          
__global__ void add_multi_real(double* dst,
                             double* src1,
                             double* src2,
                             double  a, const int M);

__global__ void lin_comb(double* dst,
                        double a,
                        double* src1,
                        double b,
                        double* src2,
                        const int M);

__global__ void add_lin_comb(double* dst,
                           double a,
                           double* src1,
                           double b,
                           double* src2,
                           const int M);

__global__ void multi_complex_real(ftsComplex* dst,
                                 double* src, const int M);

__global__ void multi_complex_conjugate(double* dst,
                                 ftsComplex* src1,
                                 ftsComplex* src2, const int M);
#endif
