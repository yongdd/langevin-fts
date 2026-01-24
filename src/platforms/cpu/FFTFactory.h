/**
 * @file FFTFactory.h
 * @brief Factory functions for creating FFT objects.
 *
 * This header provides template factory functions that create FFT objects
 * using the FFTW backend. FFTW supports all boundary conditions including
 * periodic (FFT), reflecting (DCT), and absorbing (DST).
 *
 * **Usage:**
 *
 * @code
 * FFT<double>* fft = createFFT<double, 3>(nx, bc, FFTBackend::FFTW);
 * @endcode
 *
 * @see FFT for the abstract interface
 * @see FftwFFT for FFTW implementation
 */

#ifndef FFT_FACTORY_H_
#define FFT_FACTORY_H_

#include <array>
#include "FFT.h"
#include "ComputationBox.h"

#ifdef USE_CPU_FFTW
#include "FftwFFT.h"
#endif

/**
 * @brief Create an FFT object with the specified backend.
 *
 * Factory function that instantiates the appropriate FFT implementation.
 * Currently only FFTW backend is supported, which provides O(N log N)
 * algorithms for all boundary conditions:
 * - Periodic: FFT (DFT)
 * - Reflecting: DCT (Discrete Cosine Transform)
 * - Absorbing: DST (Discrete Sine Transform)
 *
 * @tparam T   Numeric type (double or std::complex<double>)
 * @tparam DIM Dimensionality (1, 2, or 3)
 *
 * @param nx      Grid dimensions
 * @param bc      Boundary conditions per dimension
 * @param backend FFT backend to use (only FFTW is supported)
 *
 * @return Pointer to newly created FFT object (caller owns memory)
 *
 * @throws Exception if the requested backend is not available
 */
template <typename T, int DIM>
FFT<T>* createFFT(std::array<int, DIM> nx, std::array<BoundaryCondition, DIM> bc, FFTBackend backend)
{
    (void)backend;  // Currently only FFTW is supported

#ifdef USE_CPU_FFTW
    return new FftwFFT<T, DIM>(nx, bc);
#else
    (void)nx;
    (void)bc;
    throw_with_line_number("FFTW backend is required. "
        "Compile with POLYMERFTS_USE_FFTW=ON to enable FFT support.");
#endif
}

/**
 * @brief Create an FFT object with default (periodic) boundary conditions.
 *
 * Convenience overload that creates an FFT with all periodic boundaries.
 *
 * @tparam T   Numeric type (double or std::complex<double>)
 * @tparam DIM Dimensionality (1, 2, or 3)
 *
 * @param nx      Grid dimensions
 * @param backend FFT backend to use (only FFTW is supported)
 *
 * @return Pointer to newly created FFT object (caller owns memory)
 */
template <typename T, int DIM>
FFT<T>* createFFT(std::array<int, DIM> nx, FFTBackend backend)
{
    std::array<BoundaryCondition, DIM> bc;
    bc.fill(BoundaryCondition::PERIODIC);
    return createFFT<T, DIM>(nx, bc, backend);
}

#endif  // FFT_FACTORY_H_
