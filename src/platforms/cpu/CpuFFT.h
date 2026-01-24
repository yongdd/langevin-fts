/**
 * @file CpuFFT.h
 * @brief CPU FFT backend selection header.
 *
 * This header provides a compile-time selection between FFTW FFT
 * backends. Include this header instead of FftwFFT.h or FftwFFT.h directly
 * in solver code to enable backend switching via preprocessor defines.
 *
 * **Backend Selection:**
 *
 * - USE_CPU_FFTW: Use FFTW's DFTI for FFT
 * - USE_CPU_FFTW: Use FFTW3 for FFT
 *
 * Both backends provide identical interfaces through the FFT<T> base class
 * and support all boundary conditions (PERIODIC, REFLECTING, ABSORBING).
 *
 * @see FftwFFT for FFTW implementation
 * @see FftwFFT for FFTW implementation
 */

#ifndef CPU_FFT_H_
#define CPU_FFT_H_

#if defined(USE_CPU_FFTW)
#include "FftwFFT.h"

/**
 * @brief Type alias for FFTW-based FFT.
 * @tparam T Numeric type (double or std::complex<double>)
 * @tparam DIM Dimensionality (1, 2, or 3)
 */
template <typename T, int DIM>
using CpuFFT = FftwFFT<T, DIM>;

#elif defined(USE_CPU_FFTW)
#include "FftwFFT.h"

/**
 * @brief Type alias for FFTW-based FFT.
 * @tparam T Numeric type (double or std::complex<double>)
 * @tparam DIM Dimensionality (1, 2, or 3)
 */
template <typename T, int DIM>
using CpuFFT = FftwFFT<T, DIM>;

#else
#error "No CPU FFT backend selected. Define USE_CPU_FFTW or USE_CPU_FFTW."
#endif

#endif  // CPU_FFT_H_
