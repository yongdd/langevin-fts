/**
 * @file FFTFactory.h
 * @brief Factory functions for creating FFT objects.
 *
 * This header provides template factory functions that create FFT objects
 * using the FFTW or MKL backend. Both support all boundary conditions including
 * periodic (FFT), reflecting (DCT), and absorbing (DST).
 *
 * **Usage:**
 *
 * @code
 * FFT<double>* fft = createFFT<double, 3>(nx, bc, FFTBackend::FFTW);
 * FFT<double>* fft_mkl = createFFT<double, 3>(nx, bc, FFTBackend::MKL);
 * @endcode
 *
 * @see FFT for the abstract interface
 * @see FftwFFT for FFTW implementation
 * @see MklFFT for MKL implementation
 */

#ifndef FFT_FACTORY_H_
#define FFT_FACTORY_H_

#include <array>
#include "FFT.h"
#include "ComputationBox.h"

#ifdef USE_CPU_FFTW
#include "FftwFFT.h"
#endif

#ifdef USE_CPU_MKL
#include "MklFFT.h"
#endif

/**
 * @brief Create an FFT object with the specified backend.
 *
 * Factory function that instantiates the appropriate FFT implementation.
 * Two backends are available:
 * - FFTW: O(N log N) for all boundary conditions (GPL license)
 * - MKL: O(N log N) for all boundary conditions (proprietary, via Intel oneAPI)
 *
 * @tparam T   Numeric type (double or std::complex<double>)
 * @tparam DIM Dimensionality (1, 2, or 3)
 *
 * @param nx      Grid dimensions
 * @param bc      Boundary conditions per dimension
 * @param backend FFT backend to use (FFTW or MKL)
 *
 * @return Pointer to newly created FFT object (caller owns memory)
 *
 * @throws Exception if the requested backend is not available
 */
template <typename T, int DIM>
FFT<T>* createFFT(std::array<int, DIM> nx, std::array<BoundaryCondition, DIM> bc, FFTBackend backend)
{
#ifdef USE_CPU_MKL
    if (backend == FFTBackend::MKL)
        return new MklFFT<T, DIM>(nx, bc);
#endif

#ifdef USE_CPU_FFTW
    if (backend == FFTBackend::FFTW)
        return new FftwFFT<T, DIM>(nx, bc);
#endif

    (void)nx;
    (void)bc;
    (void)backend;
    throw_with_line_number("Requested FFT backend is not available. "
        "Compile with POLYMERFTS_USE_FFTW=ON or POLYMERFTS_USE_MKL=ON.");
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
