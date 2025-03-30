/* This module defines parameters and subroutines to conduct fast
* Fourier transform (FFT) using the Math Kernel Library (MKL). */

#ifndef MKL_FFT_H_
#define MKL_FFT_H_

#include <array>
#include <complex>
#include "FFT.h"
#include "mkl_service.h"
#include "mkl_dfti.h"

template <typename T, int DIM>
class MklFFT : public FFT<T>
{
private:
    // Pointers for forward and backward transform
    DFTI_DESCRIPTOR_HANDLE hand_forward = NULL;
    DFTI_DESCRIPTOR_HANDLE hand_backward = NULL;

public:
    MklFFT(std::array<int, DIM> nx);
    ~MklFFT();

    void forward (T *rdata, std::complex<double> *cdata) override;
    void backward(std::complex<double> *cdata, T *rdata) override;
};

#endif