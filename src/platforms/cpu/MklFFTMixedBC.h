/**
 * @file MklFFTMixedBC.h
 * @brief Intel MKL FFT with mixed boundary conditions (DCT/DST support).
 *
 * This header provides MklFFTMixedBC, which implements transforms for
 * pseudo-spectral method with mixed boundary conditions per dimension:
 * - PERIODIC: Standard FFT (real-to-complex)
 * - REFLECTING: DCT-II forward, DCT-III backward (Neumann BC)
 * - ABSORBING: DST-II forward, DST-III backward (Dirichlet BC)
 *
 * For DCT/DST, the transform is real-to-real (no complex storage needed).
 * The wavenumber for reflecting/absorbing is k = π*n/L (half of periodic).
 *
 * @see FFT for the abstract interface
 * @see MklFFT for periodic-only implementation
 */

#ifndef MKL_FFT_MIXED_BC_H_
#define MKL_FFT_MIXED_BC_H_

#include <array>
#include <vector>
#include <complex>
#include <map>
#include <string>
#include "ComputationBox.h"
#include "mkl_service.h"
#include "mkl_dfti.h"

/**
 * @class MklFFTMixedBC
 * @brief MKL-based FFT with mixed boundary conditions per dimension.
 *
 * Supports different boundary conditions in each spatial direction:
 * - PERIODIC: Uses standard FFT
 * - REFLECTING: Uses DCT-II (forward) and DCT-III (backward)
 * - ABSORBING: Uses DST-II (forward) and DST-III (backward)
 *
 * @tparam T   Input type for forward transform (typically double)
 * @tparam DIM Dimensionality of the transform (1, 2, or 3)
 *
 * **Key Differences from Standard FFT:**
 *
 * For non-periodic BCs, the transform is purely real-to-real, so the
 * "complex" array stores real Fourier coefficients. The wavenumber
 * factor is π/L instead of 2π/L.
 *
 * **Memory Layout:**
 *
 * When all dimensions are non-periodic:
 * - "Complex" array has same size as real array (all real coefficients)
 *
 * When mixing periodic and non-periodic:
 * - Periodic dimensions use half-complex storage
 * - Non-periodic dimensions use full real storage
 */
template <typename T, int DIM>
class MklFFTMixedBC
{
private:
    std::array<int, DIM> nx_;                      ///< Grid dimensions
    std::array<BoundaryCondition, DIM> bc_;        ///< Boundary conditions per dimension
    int total_grid_;                               ///< Total real grid size
    int total_complex_grid_;                       ///< Total "complex" grid size

    // Working buffers
    std::vector<double> work_buffer_;              ///< Workspace for transforms
    std::vector<double> temp_buffer_;              ///< Temporary buffer for dim-by-dim transforms

    // Precomputed sine/cosine tables for DCT/DST
    std::vector<std::vector<double>> sin_tables_;  ///< Sine tables for DST
    std::vector<std::vector<double>> cos_tables_;  ///< Cosine tables for DCT

    // MKL DFTI descriptor for periodic dimensions (if any)
    DFTI_DESCRIPTOR_HANDLE fft_handle_ = nullptr;
    bool has_periodic_dim_;
    int periodic_dim_idx_;                         ///< Which dimension is periodic (for 1D FFT)

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
     * @brief Construct MKL FFT object with mixed boundary conditions.
     *
     * @param nx Grid dimensions
     * @param bc Boundary conditions per dimension
     */
    MklFFTMixedBC(std::array<int, DIM> nx, std::array<BoundaryCondition, DIM> bc);

    /**
     * @brief Destructor. Frees MKL descriptors.
     */
    ~MklFFTMixedBC();

    /**
     * @brief Get total "complex" grid size.
     *
     * For purely non-periodic: same as real grid size
     * For mixed: accounts for half-complex in periodic dimensions
     *
     * @return Total complex grid size
     */
    int get_total_complex_grid() const { return total_complex_grid_; }

    /**
     * @brief Check if all boundary conditions are periodic.
     * @return true if all periodic, false otherwise
     */
    bool is_all_periodic() const;

    /**
     * @brief Perform forward transform (real space -> Fourier space).
     *
     * For PERIODIC: real-to-complex FFT
     * For REFLECTING: DCT-II
     * For ABSORBING: DST-II
     *
     * @param rdata Input: real array of size total_grid
     * @param cdata Output: coefficients (complex for periodic, real for non-periodic)
     */
    void forward(T *rdata, double *cdata);

    /**
     * @brief Perform backward transform (Fourier space -> real space).
     *
     * For PERIODIC: complex-to-real IFFT with 1/N normalization
     * For REFLECTING: DCT-III with normalization
     * For ABSORBING: DST-III with normalization
     *
     * @param cdata Input: Fourier coefficients
     * @param rdata Output: real array
     */
    void backward(double *cdata, T *rdata);

    /**
     * @brief Get boundary condition for a dimension.
     * @param dim Dimension index
     * @return Boundary condition
     */
    BoundaryCondition get_bc(int dim) const { return bc_[dim]; }
};

#endif
