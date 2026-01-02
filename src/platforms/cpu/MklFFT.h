/**
 * @file MklFFT.h
 * @brief Intel MKL implementation of Fast Fourier Transform.
 *
 * This header provides MklFFT, which implements the FFT interface using
 * Intel Math Kernel Library's Discrete Fourier Transform Interface (DFTI).
 * MKL provides highly optimized FFT routines for Intel processors.
 *
 * **MKL DFTI Features Used:**
 *
 * - Real-to-complex (r2c) forward transforms
 * - Complex-to-real (c2r) backward transforms
 * - Out-of-place transforms (input and output are separate arrays)
 * - Automatic handling of Hermitian symmetry
 *
 * **Performance Notes:**
 *
 * - MKL is highly optimized for Intel CPUs
 * - Supports multi-threading via OpenMP
 * - Grid sizes that are powers of 2 are most efficient
 *
 * @see FFT for the abstract interface
 * @see CudaFFT for GPU implementation using cuFFT
 *
 * @example
 * @code
 * // Create 3D FFT for 32x32x32 grid
 * std::array<int, 3> nx = {32, 32, 32};
 * MklFFT<double, 3> fft(nx);
 *
 * // Allocate arrays
 * double* real_data = new double[32*32*32];
 * std::complex<double>* complex_data = new std::complex<double>[32*32*17];
 *
 * // Forward transform
 * fft.forward(real_data, complex_data);
 *
 * // Backward transform (includes 1/N normalization)
 * fft.backward(complex_data, real_data);
 * @endcode
 */

#ifndef MKL_FFT_H_
#define MKL_FFT_H_

#include <array>
#include <complex>
#include "FFT.h"
#include "mkl_service.h"
#include "mkl_dfti.h"

/**
 * @class MklFFT
 * @brief Intel MKL-based FFT implementation.
 *
 * Uses Intel MKL's DFTI (Discrete Fourier Transform Interface) for
 * high-performance FFT operations. Supports 1D, 2D, and 3D transforms.
 *
 * @tparam T   Input type for forward transform (typically double)
 * @tparam DIM Dimensionality of the transform (1, 2, or 3)
 *
 * **Implementation Details:**
 *
 * - Uses separate DFTI descriptors for forward and backward transforms
 * - Forward: DFTI_REAL domain, computes r2c transform
 * - Backward: DFTI_REAL domain, computes c2r transform with 1/N normalization
 * - Both transforms are configured as out-of-place (NOT_INPLACE)
 *
 * **Memory Layout:**
 *
 * For a real array of dimensions Nx × Ny × Nz:
 * - Real array size: Nx * Ny * Nz
 * - Complex array size: Nx * Ny * (Nz/2 + 1)
 *
 * @note The backward transform includes normalization by 1/N where
 *       N = Nx * Ny * Nz, so backward(forward(x)) = x.
 */
template <typename T, int DIM>
class MklFFT : public FFT<T>
{
private:
    DFTI_DESCRIPTOR_HANDLE hand_forward = NULL;   ///< MKL descriptor for forward (r2c) transform
    DFTI_DESCRIPTOR_HANDLE hand_backward = NULL;  ///< MKL descriptor for backward (c2r) transform

public:
    /**
     * @brief Construct MKL FFT object for given grid dimensions.
     *
     * Creates and commits DFTI descriptors for both forward and backward
     * transforms. The descriptors are configured for:
     * - Double precision real domain
     * - Out-of-place computation
     * - CCE (Complex Conjugate-Even) storage format
     *
     * @param nx Grid dimensions [Nx] for 1D, [Nx, Ny] for 2D, [Nx, Ny, Nz] for 3D
     *
     * @throws Exception if DFTI descriptor creation or commit fails
     *
     * @example
     * @code
     * // 1D FFT for 128 points
     * MklFFT<double, 1> fft1d({128});
     *
     * // 2D FFT for 64x64 grid
     * MklFFT<double, 2> fft2d({64, 64});
     *
     * // 3D FFT for 32x32x32 grid
     * MklFFT<double, 3> fft3d({32, 32, 32});
     * @endcode
     */
    MklFFT(std::array<int, DIM> nx);

    /**
     * @brief Destructor. Frees MKL DFTI descriptors.
     *
     * Calls DftiFreeDescriptor for both forward and backward handles.
     */
    ~MklFFT();

    /**
     * @brief Perform forward (real-to-complex) FFT.
     *
     * Computes the DFT of a real-valued array, producing a complex array
     * with Hermitian symmetry (only half of complex coefficients stored).
     *
     * @param rdata Input: real array of size Nx × Ny × Nz
     * @param cdata Output: complex array of size Nx × Ny × (Nz/2+1)
     *
     * @note The transform is computed out-of-place; rdata is not modified.
     *
     * @example
     * @code
     * double real_in[32*32*32];
     * std::complex<double> complex_out[32*32*17];
     * fft.forward(real_in, complex_out);
     * @endcode
     */
    void forward (T *rdata, std::complex<double> *cdata) override;

    /**
     * @brief Perform backward (complex-to-real) FFT.
     *
     * Computes the inverse DFT of a complex array (with Hermitian symmetry)
     * producing a real-valued output. Includes 1/N normalization.
     *
     * @param cdata Input: complex array of size Nx × Ny × (Nz/2+1)
     * @param rdata Output: real array of size Nx × Ny × Nz
     *
     * @note The input cdata may be modified during computation.
     * @note The 1/N normalization is applied so backward(forward(x)) = x.
     *
     * @example
     * @code
     * std::complex<double> complex_in[32*32*17];
     * double real_out[32*32*32];
     * fft.backward(complex_in, real_out);
     * @endcode
     */
    void backward(std::complex<double> *cdata, T *rdata) override;
};

#endif