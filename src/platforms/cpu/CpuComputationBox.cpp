/**
 * @file CpuComputationBox.cpp
 * @brief CPU implementation of computation box with Intel MKL FFT.
 *
 * Provides template instantiations for CpuComputationBox. The actual
 * implementation is in the header file (template class). This file
 * contains only explicit instantiations to reduce compilation time.
 *
 * **FFT Operations:**
 *
 * - forward_fft(): Real-to-complex (r2c) or complex-to-complex (c2c)
 * - backward_fft(): Complex-to-real (c2r) or c2c inverse
 *
 * **Template Instantiations:**
 *
 * - CpuComputationBox<double>: Real fields with r2c FFT
 * - CpuComputationBox<std::complex<double>>: Complex fields with c2c FFT
 *
 * @see MklFFT for FFT implementation details
 * @see ComputationBox for base class interface
 */

#include <iostream>
#include <sstream>
#include <complex>

#include "CpuComputationBox.h"

// Explicit template instantiation
template class CpuComputationBox<double>;
template class CpuComputationBox<std::complex<double>>;