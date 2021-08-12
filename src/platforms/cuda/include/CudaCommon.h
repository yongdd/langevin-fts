#ifndef CUDA_COMMON_H_
#define CUDA_COMMON_H_

#include <cufft.h>

typedef cufftDoubleComplex ftsComplex;

__global__ void multiReal(double* dst,
                          double* src1,
                          double* src2,
                          double  a, const int M);

__global__ void multiAddReal(double* dst,
                             double* src1,
                             double* src2,
                             double  a, const int M);

__global__ void multiFactor(ftsComplex* a,
                            double* b, const int M);

__global__ void linComb(double* dst,
                        double a,
                        double* src1,
                        double b,
                        double* src2,
                        const int M);

#endif
