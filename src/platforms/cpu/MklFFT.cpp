/* This module defines parameters and subroutines to conduct fast
* Fourier transform (FFT) using the Math Kernel Library (MKL). */
#include <iostream>
#include <vector>
#include "MklFFT.h"

template <typename T, int DIM>
MklFFT<T, DIM>::MklFFT(std::array<int, DIM> nx)
{
    try {
        MKL_LONG NX[DIM];
        for (int i = 0; i < DIM; i++)
            NX[i] = nx[i];

        int total_grid = 1;
        for (int i = 0; i < DIM; i++)
            total_grid *= nx[i];

        // Execution status
        MKL_LONG status{0};

        // Strides describe data layout in real and conjugate-even domain
        MKL_LONG rs[DIM + 1] = {0};
        MKL_LONG cs[DIM + 1] = {0};
        rs[DIM] = 1;
        cs[DIM] = 1;

        // Determine if we're using real or complex data type
        constexpr bool is_real = std::is_same<T, double>::value;
        constexpr DFTI_CONFIG_VALUE dtype = is_real ? DFTI_REAL : DFTI_COMPLEX;

        // Calculate strides for real domain
        for (int i = DIM - 1; i > 0; i--)
            rs[i] = rs[i + 1] * nx[i];

        // Calculate strides for complex domain
        if (std::is_same<T, double>::value)
        {
            for (int i = DIM - 1; i > 0; i--)
                cs[i] = cs[i + 1] * (i == DIM - 1 ? nx[i] / 2 + 1 : nx[i]);
        }
        else
        {
            for (int i = DIM - 1; i > 0; i--)
                cs[i] = cs[i + 1] * nx[i];
        }

        if (DIM == 1)
        {
            // 1D FFT
            status = DftiCreateDescriptor(&hand_forward, DFTI_DOUBLE, dtype, 1, NX[0]);
            status = DftiCreateDescriptor(&hand_backward, DFTI_DOUBLE, dtype, 1, NX[0]);
        }
        else
        {
            // Multi-dimensional FFT
            status = DftiCreateDescriptor(&hand_forward, DFTI_DOUBLE, dtype, DIM, NX);
            status = DftiSetValue(hand_forward, DFTI_INPUT_STRIDES, rs);
            status = DftiSetValue(hand_forward, DFTI_OUTPUT_STRIDES, cs);

            status = DftiCreateDescriptor(&hand_backward, DFTI_DOUBLE, dtype, DIM, NX);
            status = DftiSetValue(hand_backward, DFTI_INPUT_STRIDES, cs);
            status = DftiSetValue(hand_backward, DFTI_OUTPUT_STRIDES, rs);
        }
        status = DftiSetValue(hand_forward, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
        status = DftiSetValue(hand_backward, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
        if (is_real)
        {
            status = DftiSetValue(hand_forward, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
            status = DftiSetValue(hand_backward, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
        }
        status = DftiCommitDescriptor(hand_forward);
        status = DftiSetValue(hand_backward, DFTI_BACKWARD_SCALE, 1.0/static_cast<double>(total_grid));
        status = DftiCommitDescriptor(hand_backward);

        if (status != 0)
            std::cerr << "MKL constructor, status: " << status << std::endl;

    }
    catch (std::exception &exc)
    {
        throw_without_line_number(exc.what());
    }
}

template <typename T, int DIM>
MklFFT<T, DIM>::~MklFFT()
{
    int status;
    status = DftiFreeDescriptor(&hand_forward);
    status = DftiFreeDescriptor(&hand_backward);

    if (status != 0)
        std::cerr << "MKL destructor, status: " << status << std::endl;
}

template <typename T, int DIM>
void MklFFT<T, DIM>::forward(T *rdata, std::complex<double> *cdata)
{
    // Real-to-complex transform
    int status = DftiComputeForward(hand_forward, rdata, cdata);

    if (status != 0)
    {
        throw_with_line_number("MKL forward, status: " + std::to_string(status));
    }
}

template <typename T, int DIM>
void MklFFT<T, DIM>::backward(std::complex<double> *cdata, T *rdata)
{
    // Complex-to-real transform
    int status = DftiComputeBackward(hand_backward, cdata, rdata);

    if (status != 0)
    {
        throw_with_line_number("MKL backward, status: " + std::to_string(status));
    }
}

// Explicit template instantiations for 1D, 2D, and 3D FFTs
template class MklFFT<double, 1>;
template class MklFFT<double, 2>;
template class MklFFT<double, 3>;
template class MklFFT<std::complex<double>, 1>;
template class MklFFT<std::complex<double>, 2>;
template class MklFFT<std::complex<double>, 3>;