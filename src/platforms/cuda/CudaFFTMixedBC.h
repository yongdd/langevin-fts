/**
 * @file CudaFFTMixedBC.h
 * @brief CUDA FFT with mixed boundary conditions (DCT/DST).
 *
 * This header provides CudaFFTMixedBC, which implements DCT-II/III and
 * DST-II/III transforms on GPU for non-periodic boundary conditions.
 *
 * **Transform Types:**
 *
 * - REFLECTING (Neumann BC): DCT-II forward, DCT-III backward
 * - ABSORBING (Dirichlet BC): DST-II forward, DST-III backward
 *
 * **Implementation:**
 *
 * Since cuFFT doesn't provide native DCT/DST, we implement them using
 * custom CUDA kernels with precomputed trigonometric tables stored on GPU.
 *
 * @see MklFFTMixedBC for CPU version
 * @see CudaSolverPseudoMixedBC for solver using this class
 */

#ifndef CUDA_FFT_MIXED_BC_H_
#define CUDA_FFT_MIXED_BC_H_

#include <array>
#include <vector>
#include <complex>
#include <cufft.h>

#include "ComputationBox.h"
#include "CudaCommon.h"

/**
 * @class CudaFFTMixedBC
 * @brief GPU DCT/DST transform implementation for mixed boundary conditions.
 *
 * Implements dimension-by-dimension transforms where each dimension can
 * have either REFLECTING (DCT) or ABSORBING (DST) boundary condition.
 *
 * @tparam T   Data type (double or std::complex<double>)
 * @tparam DIM Number of dimensions (1, 2, or 3)
 *
 * **GPU Memory:**
 *
 * - Precomputed sin/cos tables on device
 * - Work buffers for intermediate results
 * - Stream-aware for concurrent execution
 */
template <typename T, int DIM>
class CudaFFTMixedBC
{
private:
    std::array<int, DIM> nx_;                    ///< Grid dimensions
    std::array<BoundaryCondition, DIM> bc_;      ///< Boundary conditions per dimension
    int total_grid_;                             ///< Total grid size
    int total_complex_grid_;                     ///< Output size (same as total_grid for DCT/DST)

    bool has_periodic_dim_;                      ///< True if any dimension is periodic
    int periodic_dim_idx_;                       ///< Index of first periodic dimension

    // Device memory for trigonometric tables
    std::vector<double*> d_sin_tables_;          ///< Device sin tables per dimension
    std::vector<double*> d_cos_tables_;          ///< Device cos tables per dimension

    // Device work buffers
    double* d_work_buffer_;                      ///< Main work buffer
    double* d_temp_buffer_;                      ///< Temporary buffer for transforms

    /**
     * @brief Precompute and upload trigonometric tables to GPU.
     */
    void precomputeTrigTables();

    /**
     * @brief Get strides for dimension-by-dimension processing.
     */
    void getStrides(int dim, int& stride, int& num_transforms) const;

    /**
     * @brief Apply DCT-II forward transform for one dimension.
     */
    void applyDCT2Forward(double* d_data, int dim, cudaStream_t stream);

    /**
     * @brief Apply DCT-III backward transform for one dimension.
     */
    void applyDCT3Backward(double* d_data, int dim, cudaStream_t stream);

    /**
     * @brief Apply DST-II forward transform for one dimension.
     */
    void applyDST2Forward(double* d_data, int dim, cudaStream_t stream);

    /**
     * @brief Apply DST-III backward transform for one dimension.
     */
    void applyDST3Backward(double* d_data, int dim, cudaStream_t stream);

public:
    /**
     * @brief Construct CUDA FFT object for mixed boundary conditions.
     *
     * @param nx Grid dimensions
     * @param bc Boundary conditions (REFLECTING or ABSORBING per dimension)
     *
     * @throws Exception if mixed periodic/non-periodic BCs are specified
     */
    CudaFFTMixedBC(std::array<int, DIM> nx, std::array<BoundaryCondition, DIM> bc);

    /**
     * @brief Destructor. Frees GPU memory.
     */
    ~CudaFFTMixedBC();

    /**
     * @brief Check if all boundaries are periodic.
     */
    bool is_all_periodic() const;

    /**
     * @brief Get total complex grid size.
     */
    int get_total_complex_grid() const { return total_complex_grid_; }

    /**
     * @brief Get total real grid size.
     */
    int get_total_grid() const { return total_grid_; }

    /**
     * @brief Forward transform (DCT-II or DST-II per dimension).
     *
     * @param d_rdata Input real data on device
     * @param d_cdata Output transformed data on device
     * @param stream  CUDA stream for async execution (default: 0)
     */
    void forward(T* d_rdata, double* d_cdata, cudaStream_t stream = 0);

    /**
     * @brief Backward transform (DCT-III or DST-III per dimension).
     *
     * @param d_cdata Input transformed data on device
     * @param d_rdata Output real data on device
     * @param stream  CUDA stream for async execution (default: 0)
     */
    void backward(double* d_cdata, T* d_rdata, cudaStream_t stream = 0);
};

#endif
