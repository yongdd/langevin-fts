/**
 * @file MklFFT.h
 * @brief Intel MKL implementation of spectral transforms with mixed BC support.
 *
 * This header provides MklFFT, which implements spectral transforms using
 * Intel MKL's DFTI and supports all boundary conditions (FFT/DCT/DST).
 *
 * **Supported Boundary Conditions:**
 *
 * - PERIODIC: Standard FFT (real-to-complex) using MKL DFTI
 * - REFLECTING: DCT-II forward, DCT-III backward (Neumann BC)
 * - ABSORBING: DST-II forward, DST-III backward (Dirichlet BC)
 *
 * **Memory Layout:**
 *
 * - Periodic BC: Complex array has size Nx × Ny × (Nz/2+1) for Hermitian symmetry
 * - Non-periodic BC: "Complex" array has same size as real array (all real coefficients)
 *
 * @see FFT for the abstract interface
 * @see CudaFFT for GPU implementation
 */

#ifndef MKL_FFT_H_
#define MKL_FFT_H_

#include <array>
#include <vector>
#include <complex>
#include "FFT.h"
#include "ComputationBox.h"
#include "mkl_service.h"
#include "mkl_dfti.h"

/**
 * @class MklFFT
 * @brief Intel MKL-based FFT with support for all boundary conditions.
 *
 * Uses Intel MKL's DFTI for periodic FFT and custom DCT/DST for non-periodic.
 *
 * @tparam T   Input type for forward transform (typically double)
 * @tparam DIM Dimensionality of the transform (1, 2, or 3)
 *
 * **Backward Compatibility:**
 *
 * When constructed without boundary conditions (or all periodic), this class
 * behaves exactly like the original MklFFT, inheriting from FFT<T>.
 *
 * **For Non-Periodic BC:**
 *
 * Use the double* interface: forward(T*, double*) and backward(double*, T*)
 * The complex<double>* interface will throw if any dimension is non-periodic.
 */
template <typename T, int DIM>
class MklFFT : public FFT<T>
{
private:
    std::array<int, DIM> nx_;                      ///< Grid dimensions
    std::array<BoundaryCondition, DIM> bc_;        ///< Boundary conditions per dimension
    int total_grid_;                               ///< Total real grid size
    int total_complex_grid_;                       ///< Total "complex" grid size
    bool is_all_periodic_;                         ///< True if all BCs are periodic

    // MKL DFTI descriptors for periodic FFT
    DFTI_DESCRIPTOR_HANDLE hand_forward_ = nullptr;   ///< Forward (r2c) transform
    DFTI_DESCRIPTOR_HANDLE hand_backward_ = nullptr;  ///< Backward (c2r) transform

    // Working buffers for DCT/DST
    std::vector<double> work_buffer_;              ///< Workspace for transforms
    std::vector<double> temp_buffer_;              ///< Temporary buffer for dim-by-dim

    // Precomputed sine/cosine tables for DCT/DST
    std::vector<std::vector<double>> sin_tables_;  ///< Sine tables for DST
    std::vector<std::vector<double>> cos_tables_;  ///< Cosine tables for DCT

    /**
     * @brief Initialize MKL DFTI descriptors for periodic FFT.
     */
    void initPeriodicFFT();

    /**
     * @brief Initialize DCT/DST trig tables for non-periodic BC.
     */
    void initNonPeriodicFFT();

    /**
     * @brief Precompute sine/cosine tables for DCT/DST.
     */
    void precomputeTrigTables();

    /**
     * @brief Apply DCT-II (forward) along specified dimension.
     */
    void applyDCT2Forward(double* data, int dim);

    /**
     * @brief Apply DCT-III (backward/inverse) along specified dimension.
     */
    void applyDCT3Backward(double* data, int dim);

    /**
     * @brief Apply DST-II (forward) along specified dimension.
     */
    void applyDST2Forward(double* data, int dim);

    /**
     * @brief Apply DST-III (backward/inverse) along specified dimension.
     */
    void applyDST3Backward(double* data, int dim);

    /**
     * @brief Get stride information for dimension-by-dimension transform.
     */
    void getStrides(int dim, int& stride, int& num_transforms) const;

public:
    /**
     * @brief Construct MKL FFT object for given grid dimensions (periodic BC).
     *
     * For backward compatibility, creates an FFT with all periodic boundaries.
     *
     * @param nx Grid dimensions [Nx] for 1D, [Nx, Ny] for 2D, [Nx, Ny, Nz] for 3D
     */
    MklFFT(std::array<int, DIM> nx);

    /**
     * @brief Construct MKL FFT object with specified boundary conditions.
     *
     * @param nx Grid dimensions
     * @param bc Boundary conditions per dimension
     */
    MklFFT(std::array<int, DIM> nx, std::array<BoundaryCondition, DIM> bc);

    /**
     * @brief Destructor. Frees MKL DFTI descriptors and buffers.
     */
    ~MklFFT();

    /**
     * @brief Check if all boundary conditions are periodic.
     * @return true if all periodic, false otherwise
     */
    bool is_all_periodic() const { return is_all_periodic_; }

    /**
     * @brief Get total "complex" grid size.
     *
     * For periodic: Nx × Ny × (Nz/2+1)
     * For non-periodic: same as real grid size
     *
     * @return Total complex grid size
     */
    int get_total_complex_grid() const { return total_complex_grid_; }

    /**
     * @brief Get total real grid size.
     * @return Total grid size
     */
    int get_total_grid() const { return total_grid_; }

    /**
     * @brief Get boundary condition for a dimension.
     * @param dim Dimension index
     * @return Boundary condition
     */
    BoundaryCondition get_bc(int dim) const { return bc_[dim]; }

    /**
     * @brief Perform forward FFT (complex output interface).
     *
     * Implements FFT<T> interface for backward compatibility.
     *
     * @param rdata Input: real array
     * @param cdata Output: complex array
     *
     * @throws Exception if any dimension is non-periodic
     */
    void forward(T *rdata, std::complex<double> *cdata) override;

    /**
     * @brief Perform backward FFT (complex input interface).
     *
     * Implements FFT<T> interface for backward compatibility.
     *
     * @param cdata Input: complex array
     * @param rdata Output: real array
     *
     * @throws Exception if any dimension is non-periodic
     */
    void backward(std::complex<double> *cdata, T *rdata) override;

    /**
     * @brief Perform forward transform (real coefficient interface).
     *
     * Works for all boundary conditions:
     * - PERIODIC: real-to-complex FFT (output is interleaved real/imag)
     * - REFLECTING: DCT-II
     * - ABSORBING: DST-II
     *
     * @param rdata Input: real array of size total_grid
     * @param cdata Output: coefficients (size total_complex_grid)
     */
    void forward(T *rdata, double *cdata) override;

    /**
     * @brief Perform backward transform (real coefficient interface).
     *
     * Works for all boundary conditions:
     * - PERIODIC: complex-to-real IFFT with 1/N normalization
     * - REFLECTING: DCT-III with normalization
     * - ABSORBING: DST-III with normalization
     *
     * @param cdata Input: Fourier coefficients
     * @param rdata Output: real array
     */
    void backward(double *cdata, T *rdata) override;
};

#endif
