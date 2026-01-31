/**
 * @file FFT.h
 * @brief Abstract interface for spectral transforms (FFT, DCT, DST).
 *
 * This header defines the FFT abstract base class that supports:
 *
 * - PERIODIC (FFT): Interleaved real/imaginary complex coefficients
 * - REFLECTING (DCT): Real coefficients
 * - ABSORBING (DST): Real coefficients
 *
 * **Interfaces:**
 *
 * - `forward(T*, double*)` / `backward(double*, T*)`: Universal interface for all BCs
 * - `forward(T*, complex<double>*)` / `backward(complex<double>*, T*)`: Periodic BC only
 *
 * **Design Rationale:**
 *
 * This allows storing transform objects of different dimensions (1D, 2D, 3D)
 * in a single FFT<T>* pointer without void* casts and if-else dispatch.
 * Since all simulations run in fixed dimensions, a single pointer suffices.
 *
 * @see FftwFFT for FFTW implementation
 * @see CudaFFT for CUDA implementation
 */

#ifndef FFT_H_
#define FFT_H_

#include <complex>
#include "Exception.h"

/**
 * @enum FFTBackend
 * @brief Available FFT backend implementations.
 *
 * Two backends are available for CPU:
 * - FFTW: FFTW3 library with O(N log N) for all transform types (GPL license)
 * - MKL: Intel MKL with O(N log N) for FFT, O(N^2) for DCT/DST (proprietary)
 */
enum class FFTBackend
{
    FFTW,  ///< FFTW3 library (O(N log N) for all transform types)
    MKL    ///< Intel MKL (O(N log N) for FFT, O(N^2) for DCT/DST)
};

/**
 * @class FFT
 * @brief Abstract base class for spectral transforms.
 *
 * Provides a dimension-independent interface for forward/backward transforms
 * including FFT (periodic), DCT (reflecting), and DST (absorbing).
 *
 * @tparam T Input/output type (double or std::complex<double>)
 *
 * **Grid Size Convention:**
 *
 * For a real array of size Nx × Ny × Nz, the complex array has size
 * Nx × Ny × (Nz/2 + 1) due to Hermitian symmetry.
 *
 * **FFT Convention:**
 *
 * - Forward: Real-to-complex transform (r2c)
 * - Backward: Complex-to-real transform (c2r) with normalization
 *
 * The backward transform includes the 1/N normalization factor so that
 * backward(forward(x)) = x (up to numerical precision).
 *
 * @example
 * @code
 * FFT<double>* fft = ...;
 * double* real_data = ...;
 * std::complex<double>* complex_data = ...;
 * fft->forward(real_data, complex_data);
 * fft->backward(complex_data, real_data);
 * @endcode
 */
template <typename T>
class FFT
{
public:
    /**
     * @brief Virtual destructor.
     */
    virtual ~FFT() {}

    /**
     * @brief Perform forward transform (real coefficient interface).
     *
     * Works for all boundary conditions:
     * - PERIODIC: real-to-complex FFT (output is interleaved real/imag)
     * - REFLECTING: DCT-II
     * - ABSORBING: DST-II
     *
     * @param rdata Input: real-space data
     * @param cdata Output: spectral coefficients (double* for all BCs)
     */
    virtual void forward(T* rdata, double* cdata) = 0;

    /**
     * @brief Perform backward transform (real coefficient interface).
     *
     * Works for all boundary conditions:
     * - PERIODIC: complex-to-real IFFT with 1/N normalization
     * - REFLECTING: DCT-III with normalization
     * - ABSORBING: DST-III with normalization
     *
     * @param cdata Input: spectral coefficients
     * @param rdata Output: real-space data
     */
    virtual void backward(double* cdata, T* rdata) = 0;

    /**
     * @brief Perform forward (real-to-complex) FFT.
     *
     * @param rdata Input: real array of size Nx × Ny × Nz
     * @param cdata Output: complex array of size Nx × Ny × (Nz/2+1)
     *
     * @note The complex output uses Hermitian symmetry packing.
     * @throws Exception if any dimension is non-periodic
     */
    virtual void forward(T* rdata, std::complex<double>* cdata) = 0;

    /**
     * @brief Perform backward (complex-to-real) FFT.
     *
     * @param cdata Input: complex array of size Nx × Ny × (Nz/2+1)
     * @param rdata Output: real array of size Nx × Ny × Nz
     *
     * @note Includes 1/N normalization factor.
     * @throws Exception if any dimension is non-periodic
     */
    virtual void backward(std::complex<double>* cdata, T* rdata) = 0;
};

#endif
