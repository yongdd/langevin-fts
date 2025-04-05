#ifndef CUDA_COMMON_H_
#define CUDA_COMMON_H_

#include <complex>
#include <map>
#include <cufft.h>

#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>
#include "Exception.h"

// // The maximum of GPUs
// #define MAX_GPUS 2

// The maximum of computation streams
#define MAX_STREAMS 4

typedef cuDoubleComplex ftsComplex;

template<typename T>
using CuDeviceData = std::conditional_t<std::is_same_v<T, double>,               double,
                     std::conditional_t<std::is_same_v<T, std::complex<double>>, cuDoubleComplex, void>
>;

// Define custom reduction operator for cuDoubleComplex
struct ComplexSumOp {
    __device__ __forceinline__
    cuDoubleComplex operator()(const cuDoubleComplex& a, const cuDoubleComplex& b) const {
        cuDoubleComplex result;
        result.x = a.x + b.x;
        result.y = a.y + b.y;
        return result;
    }
};

std::complex<double> cuDoubleToStdComplex(cuDoubleComplex z);
cuDoubleComplex stdToCuDoubleComplex(const std::complex<double>& z);
template<typename From, typename To>
std::map<std::string, const To*> reinterpret_map(const std::map<std::string, const From*>& input);

// Design Pattern : Singleton (Scott Meyer)
class CudaCommon
{
private:
    int n_blocks;
    int n_threads;

    // int n_gpus;

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
    // int get_n_gpus();
    
    void set_n_blocks(int n_blocks);
    void set_n_threads(int n_threads);
    void set_idx(int process_idx);
};

#define gpu_error_check(code) throw_on_cuda_error((code), __FILE__, __LINE__, __func__);
void throw_on_cuda_error(cudaError_t code, const char *file, int line, const char *func);

__global__ void ker_linear_scaling(double* dst, const double* src, double a, double b, const int M);
__global__ void ker_linear_scaling(cuDoubleComplex* dst, const cuDoubleComplex* src, double a, cuDoubleComplex b, const int M);
__global__ void ker_linear_scaling(cuDoubleComplex* dst, const cuDoubleComplex* src, cuDoubleComplex a, double b, const int M);

__global__ void ker_exp(double* dst, const double* src, double a, double exp_b, const int M);
__global__ void ker_exp(cuDoubleComplex* dst, const cuDoubleComplex* src, double a, double exp_b, const int M);

__global__ void ker_multi(double* dst, const double* src1, const double* src2, double a, const int M);
__global__ void ker_multi(cuDoubleComplex* dst, const cuDoubleComplex* src1, const double* src2, double a, const int M);
__global__ void ker_multi(cuDoubleComplex* dst, const cuDoubleComplex* src1, const cuDoubleComplex* src2, double a, const int M);
__global__ void ker_multi(cuDoubleComplex* dst, const cuDoubleComplex* src1, const cuDoubleComplex* src2, cuDoubleComplex a, const int M);

__global__ void ker_mutiple_multi(int n_comp, double* dst, const double* src1, const double* src2, double  a, const int M);
__global__ void ker_mutiple_multi(int n_comp, cuDoubleComplex* dst, const cuDoubleComplex* src1, const cuDoubleComplex* src2, double  a, const int M);

__global__ void ker_multi_complex_conjugate(double* dst, const cuDoubleComplex* src1, const cuDoubleComplex* src2, const int M);
__global__ void ker_multi_complex_conjugate(cuDoubleComplex* dst, const cuDoubleComplex* src1, const cuDoubleComplex* src2, const int M);

__global__ void ker_divide(double* dst, const double* src1, const double* src2, double a, const int M);
__global__ void ker_divide(cuDoubleComplex* dst, const cuDoubleComplex* src1, const cuDoubleComplex* src2, double a, const int M);

__global__ void ker_add_multi(double* dst, const double* src1, const double* src2, double a, const int M);
__global__ void ker_add_multi(cuDoubleComplex* dst, const cuDoubleComplex* src1, const cuDoubleComplex* src2, double a, const int M);
__global__ void ker_add_multi(cuDoubleComplex* dst, const cuDoubleComplex* src1, const cuDoubleComplex* src2, cuDoubleComplex a, const int M);

__global__ void ker_lin_comb(double* dst, double a, const double* src1, double b, const double* src2, const int M);
__global__ void ker_lin_comb(cuDoubleComplex* dst, double a, const cuDoubleComplex* src1, double b, const cuDoubleComplex* src2, const int M);
__global__ void ker_lin_comb(cuDoubleComplex* dst, cuDoubleComplex a, const cuDoubleComplex* src1, double b, const cuDoubleComplex* src2, const int M);

__global__ void ker_add_lin_comb(double* dst, double a, const double* src1, double b, const double* src2, const int M);
__global__ void ker_add_lin_comb(cuDoubleComplex* dst, double a, const cuDoubleComplex* src1, double b, const cuDoubleComplex* src2, const int M);
__global__ void ker_add_lin_comb(cuDoubleComplex* dst, cuDoubleComplex a, const cuDoubleComplex* src1, cuDoubleComplex b, const cuDoubleComplex* src2, const int M);

__global__ void ker_multi_complex_real(cuDoubleComplex* dst, const double* src, double a, const int M);
__global__ void ker_multi_complex_real(cuDoubleComplex* dst, const cuDoubleComplex* src1, const double* src2, double a, const int M);


__global__ void ker_multi_exp_dw_two(
    double* dst1, const double* src1, const double* exp_dw1,
    double* dst2, const double* src2, const double* exp_dw2,
    double a, const int M);

__global__ void ker_multi_exp_dw_two(
    cuDoubleComplex* dst1, const cuDoubleComplex* src1, const cuDoubleComplex* exp_dw1,
    cuDoubleComplex* dst2, const cuDoubleComplex* src2, const cuDoubleComplex* exp_dw2,
    double a, const int M);

__global__ void ker_multi_exp_dw_four(
    double* dst1, const double* src1, const double* exp_dw1,
    double* dst2, const double* src2, const double* exp_dw2,
    double* dst3, const double* src3, const double* exp_dw3,
    double* dst4, const double* src4, const double* exp_dw4,
    double a, const int M);

__global__ void ker_multi_exp_dw_four(
    cuDoubleComplex* dst1, const cuDoubleComplex* src1, const cuDoubleComplex* exp_dw1,
    cuDoubleComplex* dst2, const cuDoubleComplex* src2, const cuDoubleComplex* exp_dw2,
    cuDoubleComplex* dst3, const cuDoubleComplex* src3, const cuDoubleComplex* exp_dw3,
    cuDoubleComplex* dst4, const cuDoubleComplex* src4, const cuDoubleComplex* exp_dw4,
    double a, const int M);

__global__ void ker_complex_real_multi_bond_two(
    cuDoubleComplex* dst1, const double* boltz_bond1,
    cuDoubleComplex* dst2, const double* boltz_bond2,
    const int M);

__global__ void ker_complex_real_multi_bond_four(
    cuDoubleComplex* dst1, const double* boltz_bond1,
    cuDoubleComplex* dst2, const double* boltz_bond2,
    cuDoubleComplex* dst3, const double* boltz_bond3,
    cuDoubleComplex* dst4, const double* boltz_bond4,
    const int M);

#endif
