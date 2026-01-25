/**
 * @file CudaFFT.h
 * @brief CUDA spectral transforms (cuFFT, DCT, DST) for all boundary conditions.
 *
 * This header provides CudaFFT, which implements:
 * - PERIODIC: Standard cuFFT (r2c/c2r)
 * - REFLECTING (Neumann BC): DCT-II forward, DCT-III backward
 * - ABSORBING (Dirichlet BC): DST-II forward, DST-III backward
 *
 * **Interfaces:**
 *
 * - `forward(T*, double*)` / `backward(double*, T*)`: Universal interface for all BCs
 * - `forward(T*, complex<double>*)` / `backward(complex<double>*, T*)`: Periodic BC only
 *
 * **Implementation:**
 *
 * For periodic BCs, uses cuFFT for GPU-accelerated FFT.
 * For non-periodic BCs, delegates to CudaRealTransform (FCT/FST algorithm).
 *
 * @see FftwFFT for CPU version
 * @see FFT for abstract interface
 * @see CudaRealTransform for DCT/DST implementation
 */

#ifndef CUDA_FFT_H_
#define CUDA_FFT_H_

#include <array>
#include <vector>
#include <complex>
#include <cufft.h>

#include "ComputationBox.h"
#include "CudaCommon.h"
#include "FFT.h"

// Forward declarations
class CudaRealTransform1D;
class CudaRealTransform2D;
class CudaRealTransform3D;

/**
 * @class CudaFFT
 * @brief GPU spectral transform implementation for all boundary conditions.
 *
 * Implements dimension-by-dimension transforms where each dimension can
 * have PERIODIC, REFLECTING (DCT), or ABSORBING (DST) boundary condition.
 *
 * @tparam T   Data type (double or std::complex<double>)
 * @tparam DIM Number of dimensions (1, 2, or 3)
 *
 * **GPU Memory:**
 *
 * - For periodic: cuFFT plans
 * - For non-periodic: Delegates to CudaRealTransform
 * - Stream-aware for concurrent execution
 */
template <typename T, int DIM>
class CudaFFT : public FFT<T>
{
private:
    std::array<int, DIM> nx_;                    ///< Grid dimensions
    std::array<BoundaryCondition, DIM> bc_;      ///< Boundary conditions per dimension
    int total_grid_;                             ///< Total grid size
    int total_complex_grid_;                     ///< Output size

    bool is_all_periodic_;                       ///< True if all dimensions are periodic

    // cuFFT handles for periodic BC
    cufftHandle plan_forward_;                   ///< Forward FFT plan
    cufftHandle plan_backward_;                  ///< Backward FFT plan

    // Device work buffers
    double* d_work_buffer_;                      ///< Main work buffer

    // CudaRealTransform objects for non-periodic BC (forward: DCT-2/DST-2, backward: DCT-3/DST-3)
    void* rt_forward_;                           ///< Forward transform (DCT-2/DST-2)
    void* rt_backward_;                          ///< Backward transform (DCT-3/DST-3)

    /**
     * @brief Initialize cuFFT plans for periodic BC.
     */
    void initPeriodicFFT();

    /**
     * @brief Initialize CudaRealTransform for non-periodic BC.
     */
    void initNonPeriodicFFT();

public:
    /**
     * @brief Construct CUDA FFT object with all periodic BCs (backward compatible).
     *
     * @param nx Grid dimensions
     */
    CudaFFT(std::array<int, DIM> nx);

    /**
     * @brief Construct CUDA FFT object with specified boundary conditions.
     *
     * @param nx Grid dimensions
     * @param bc Boundary conditions per dimension
     */
    CudaFFT(std::array<int, DIM> nx, std::array<BoundaryCondition, DIM> bc);

    /**
     * @brief Destructor. Frees GPU memory and cuFFT plans.
     */
    ~CudaFFT();

    /**
     * @brief Check if all boundaries are periodic.
     */
    bool is_all_periodic() const { return is_all_periodic_; }

    /**
     * @brief Get total complex grid size.
     */
    int get_total_complex_grid() const { return total_complex_grid_; }

    /**
     * @brief Get total real grid size.
     */
    int get_total_grid() const { return total_grid_; }

    /**
     * @brief Get boundary condition for a dimension.
     */
    BoundaryCondition get_bc(int dim) const { return bc_[dim]; }

    /**
     * @brief Forward transform (real coefficient interface) with explicit stream.
     *
     * Works for all boundary conditions.
     *
     * @param d_rdata Input real data on device
     * @param d_cdata Output transformed data on device
     * @param stream  CUDA stream for async execution
     */
    void forward_stream(T* d_rdata, double* d_cdata, cudaStream_t stream);

    /**
     * @brief Backward transform (real coefficient interface) with explicit stream.
     *
     * Works for all boundary conditions.
     *
     * @param d_cdata Input transformed data on device
     * @param d_rdata Output real data on device
     * @param stream  CUDA stream for async execution
     */
    void backward_stream(double* d_cdata, T* d_rdata, cudaStream_t stream);

    /**
     * @brief Forward FFT (complex output interface, periodic BC only) with explicit stream.
     *
     * @param d_rdata Input real data on device
     * @param d_cdata Output complex data on device
     * @param stream  CUDA stream for async execution
     *
     * @throws Exception if any dimension is non-periodic
     */
    void forward_stream(T* d_rdata, std::complex<double>* d_cdata, cudaStream_t stream);

    /**
     * @brief Backward FFT (complex input interface, periodic BC only) with explicit stream.
     *
     * @param d_cdata Input complex data on device
     * @param d_rdata Output real data on device
     * @param stream  CUDA stream for async execution
     *
     * @throws Exception if any dimension is non-periodic
     */
    void backward_stream(std::complex<double>* d_cdata, T* d_rdata, cudaStream_t stream);

    // Override pure virtual methods from FFT<T> base class
    void forward(T* rdata, double* cdata) override { forward_stream(rdata, cdata, 0); }
    void backward(double* cdata, T* rdata) override { backward_stream(cdata, rdata, 0); }
    void forward(T* rdata, std::complex<double>* cdata) override { forward_stream(rdata, cdata, 0); }
    void backward(std::complex<double>* cdata, T* rdata) override { backward_stream(cdata, rdata, 0); }
};

#endif
