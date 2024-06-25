#ifndef CUDA_COMMON_H_
#define CUDA_COMMON_H_

#include <complex>
#include <cufft.h>
#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>
#include "Exception.h"

typedef cufftDoubleComplex ftsComplex;

// The maximum of GPUs
#define MAX_GPUS 2

// The maximum of computation streams
#define MAX_STREAMS 4

// Design Pattern : Singleton (Scott Meyer)

class CudaCommon
{
private:
    int n_blocks;
    int n_threads;

    int n_gpus;

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
    int get_n_gpus();
    
    void set_n_blocks(int n_blocks);
    void set_n_threads(int n_threads);
    void set_idx(int process_idx);
};

#define gpu_error_check(code) throw_on_cuda_error((code), __FILE__, __LINE__, __func__);
void throw_on_cuda_error(cudaError_t code, const char *file, int line, const char *func);

__global__ void linear_scaling_real(
            double* dst, const double* src, double a, double b, const int M);

__global__ void exp_real(double* dst,
                        const double* src,
                        double  a, double exp_b,
                        const int M);

__global__ void multi_real(double* dst,
                        const double* src1,
                        const double* src2,
                        double  a, const int M);
                          
__global__ void mutiple_multi_real(int n_comp,
                        double* dst,
                        const double* src1,
                        const double* src2,
                        double  a, const int M);
                                    
__global__ void divide_real(double* dst,
                        const double* src1,
                        const double* src2,
                        double  a, const int M);
                          
__global__ void add_multi_real(double* dst,
                        const double* src1,
                        const double* src2,
                        double  a, const int M);

__global__ void lin_comb(double* dst,
                        double a,
                        const double* src1,
                        double b,
                        const double* src2,
                        const int M);

__global__ void add_lin_comb(double* dst,
                        double a,
                        const double* src1,
                        double b,
                        const double* src2,
                        const int M);

__global__ void multi_complex_real(ftsComplex* dst,
                                 const double* src, const int M);

__global__ void multi_complex_real(ftsComplex* dst,
                                 const double* src, double a, const int M);

__global__ void multi_complex_conjugate(double* dst,
                                const ftsComplex* src1,
                                const ftsComplex* src2, const int M);

__global__ void real_multi_exp_dw_two(
                        double* dst1, const double* src1, const double* exp_dw1,
                        double* dst2, const double* src2, const double* exp_dw2,
                        double  a, const int M);

__global__ void real_multi_exp_dw_four(
                        double* dst1, const double* src1, const double* exp_dw1,
                        double* dst2, const double* src2, const double* exp_dw2,
                        double* dst3, const double* src3, const double* exp_dw3,
                        double* dst4, const double* src4, const double* exp_dw4,
                        double  a, const int M);

__global__ void complex_real_multi_bond_two(
                        ftsComplex* dst1, const double* boltz_bond1,
                        ftsComplex* dst2, const double* boltz_bond2,
                        const int M);

__global__ void complex_real_multi_bond_four(
                        ftsComplex* dst1, const double* boltz_bond1,
                        ftsComplex* dst2, const double* boltz_bond2,
                        ftsComplex* dst3, const double* boltz_bond3,
                        ftsComplex* dst4, const double* boltz_bond4,
                        const int M);
#endif
