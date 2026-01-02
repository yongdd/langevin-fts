/**
 * @file FFT.h
 * @brief Abstract interface for Fast Fourier Transform operations.
 *
 * This header defines the abstract FFT class which provides a common interface
 * for FFT operations. Concrete implementations include MklFFT (using Intel MKL)
 * and CUDA-based FFT using cuFFT.
 *
 * **FFT Convention:**
 *
 * - Forward: Real-to-complex transform (r2c)
 * - Backward: Complex-to-real transform (c2r) with normalization
 *
 * The backward transform includes the 1/N normalization factor so that
 * backward(forward(x)) = x (up to numerical precision).
 *
 * @see MklFFT for Intel MKL implementation
 *
 * @example
 * @code
 * // Create FFT object (via factory)
 * FFT<double>* fft = ...;
 *
 * // Perform forward transform
 * double* real_data = ...;           // size: Nx * Ny * Nz
 * std::complex<double>* complex_data = ...;  // size: Nx * Ny * (Nz/2+1)
 * fft->forward(real_data, complex_data);
 *
 * // Perform backward transform
 * fft->backward(complex_data, real_data);
 * @endcode
 */

#ifndef FFT_H_
#define FFT_H_

#include <array>
#include <complex>
#include "Exception.h"

/**
 * @class FFT
 * @brief Abstract base class for Fast Fourier Transform.
 *
 * Defines the interface for real-to-complex and complex-to-real FFT
 * operations used in the pseudo-spectral method.
 *
 * @tparam T Input type for forward transform (typically double)
 *
 * **Grid Size Convention:**
 *
 * For a real array of size Nx × Ny × Nz, the complex array has size
 * Nx × Ny × (Nz/2 + 1) due to Hermitian symmetry.
 */
template <typename T>
class FFT
{
public:
    /**
     * @brief Virtual destructor.
     */
    virtual ~FFT() {};

    /**
     * @brief Perform forward (real-to-complex) FFT.
     *
     * @param rdata Input: real array of size Nx × Ny × Nz
     * @param cdata Output: complex array of size Nx × Ny × (Nz/2+1)
     *
     * @note The complex output uses Hermitian symmetry packing.
     */
    virtual void forward (T *rdata, std::complex<double> *cdata) = 0;

    /**
     * @brief Perform backward (complex-to-real) FFT.
     *
     * @param cdata Input: complex array of size Nx × Ny × (Nz/2+1)
     * @param rdata Output: real array of size Nx × Ny × Nz
     *
     * @note Includes 1/N normalization factor.
     */
    virtual void backward(std::complex<double> *cdata, T *rdata) = 0;
};

#endif
