/**
 * @file FFTFactory.h
 * @brief Factory functions for creating FFT objects with backend selection.
 *
 * This header provides template factory functions that create FFT objects
 * based on the selected backend (MKL or FFTW). This enables runtime
 * backend selection while keeping the solver code generic.
 *
 * **Usage:**
 *
 * @code
 * FFT<double>* fft = createFFT<double, 3>(nx, bc, FFTBackend::FFTW);
 * @endcode
 *
 * @see FFT for the abstract interface
 * @see MklFFT for MKL implementation
 * @see FftwFFT for FFTW implementation
 */

#ifndef FFT_FACTORY_H_
#define FFT_FACTORY_H_

#include <array>
#include "FFT.h"
#include "ComputationBox.h"

#ifdef USE_CPU_MKL
#include "MklFFT.h"
#endif

#ifdef USE_CPU_FFTW
#include "FftwFFT.h"
#endif

/**
 * @brief Create an FFT object with the specified backend.
 *
 * Factory function that instantiates the appropriate FFT implementation
 * based on the backend parameter. Supports all boundary conditions.
 *
 * @tparam T   Numeric type (double or std::complex<double>)
 * @tparam DIM Dimensionality (1, 2, or 3)
 *
 * @param nx      Grid dimensions
 * @param bc      Boundary conditions per dimension
 * @param backend FFT backend to use (MKL or FFTW)
 *
 * @return Pointer to newly created FFT object (caller owns memory)
 *
 * @throws Exception if the requested backend is not available
 */
template <typename T, int DIM>
FFT<T>* createFFT(std::array<int, DIM> nx, std::array<BoundaryCondition, DIM> bc, FFTBackend backend)
{
    switch (backend)
    {
#ifdef USE_CPU_MKL
        case FFTBackend::MKL:
            return new MklFFT<T, DIM>(nx, bc);
#endif

#ifdef USE_CPU_FFTW
        case FFTBackend::FFTW:
            return new FftwFFT<T, DIM>(nx, bc);
#endif

        default:
            throw_with_line_number("Requested FFT backend is not available. "
                "Compile with USE_CPU_MKL or USE_CPU_FFTW to enable backends.");
    }
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
 * @param backend FFT backend to use (MKL or FFTW)
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
