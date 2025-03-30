/* This module defines parameters and subroutines to conduct fast
* Fourier transform (FFT) using the Math Kernel Library (MKL). */
#include <iostream>
#include "MklFFT.h"

template <typename T, int DIM>
MklFFT<T, DIM>::MklFFT(std::array<int, DIM> nx)
{
    try
    {
        MKL_LONG NX[DIM];
        for (int i = 0; i < DIM; i++)
            NX[i] = nx[i];

        this->total_grid = 1;
        for (int i = 0; i < DIM; i++)
            this->total_grid *= nx[i];

        // Execution status
        MKL_LONG status{0};

        // Strides describe data layout in real and conjugate-even domain
        MKL_LONG rs[DIM + 1] = {0};
        MKL_LONG cs[DIM + 1] = {0};
        rs[DIM] = 1;
        cs[DIM] = 1;

        for (int i = DIM - 1; i > 0; i--)
        {
            rs[i] = rs[i + 1] * nx[i];
            cs[i] = cs[i + 1] * (i == DIM - 1 ? nx[i] / 2 + 1 : nx[i]);
        }

        // Set precision based on the template type
        constexpr DFTI_CONFIG_VALUE precision = std::is_same<T, double>::value ? DFTI_DOUBLE : DFTI_SINGLE;

        if (DIM == 1)
        {
            status = DftiCreateDescriptor(&hand_forward, precision, DFTI_REAL, 1, NX[0]);
            status = DftiSetValue(hand_forward, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
            status = DftiSetValue(hand_forward, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
            status = DftiCommitDescriptor(hand_forward);
    
            status = DftiCreateDescriptor(&hand_backward, precision, DFTI_REAL, 1, NX[0]);
            status = DftiSetValue(hand_backward, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
            status = DftiSetValue(hand_backward, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
            status = DftiCommitDescriptor(hand_backward);
        }
        else
        {
            status = DftiCreateDescriptor(&hand_forward, precision, DFTI_REAL, DIM, NX);
            status = DftiSetValue(hand_forward, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
            status = DftiSetValue(hand_forward, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
            status = DftiSetValue(hand_forward, DFTI_INPUT_STRIDES, rs);
            status = DftiSetValue(hand_forward, DFTI_OUTPUT_STRIDES, cs);
            status = DftiCommitDescriptor(hand_forward);

            status = DftiCreateDescriptor(&hand_backward, precision, DFTI_REAL, DIM, NX);
            status = DftiSetValue(hand_backward, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
            status = DftiSetValue(hand_backward, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
            status = DftiSetValue(hand_backward, DFTI_INPUT_STRIDES, cs);
            status = DftiSetValue(hand_backward, DFTI_OUTPUT_STRIDES, rs);
            status = DftiCommitDescriptor(hand_backward);
        }

        if (status != 0)
            std::cout << "MKL constructor, status: " << status << std::endl;

        // Compute a normalization factor
        this->fft_normal_factor = static_cast<T>(this->total_grid);
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
        std::cout << "MKL destructor, status: " << status << std::endl;
}

template <typename T, int DIM>
void MklFFT<T, DIM>::forward(T *rdata, std::complex<T> *cdata)
{
    int status;
    status = DftiComputeForward(hand_forward, rdata, cdata);

    if (status != 0)
    {
        throw_with_line_number("MKL forward, status: " + std::to_string(status));
    }
}

template <typename T, int DIM>
void MklFFT<T, DIM>::backward(std::complex<T> *cdata, T *rdata)
{
    int status;
    status = DftiComputeBackward(hand_backward, cdata, rdata);

    if (status != 0)
    {
        throw_with_line_number("MKL backward, status: " + std::to_string(status));
    }

    for (int i = 0; i < total_grid; i++)
        rdata[i] /= fft_normal_factor;
}

// Explicit template instantiations for 1D, 2D, and 3D FFTs
template class MklFFT<double, 1>;
template class MklFFT<double, 2>;
template class MklFFT<double, 3>;