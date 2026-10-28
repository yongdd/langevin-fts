/**
 * @file CudaComputationBox.h
 * @brief GPU implementation of ComputationBox using CUDA.
 *
 * This header provides CudaComputationBox, the GPU-specific implementation
 * of ComputationBox. It extends the base class with device-memory versions
 * of integration and inner product operations using CUB reduction.
 *
 * **GPU Memory Layout:**
 *
 * - All field data resides in GPU device memory
 * - Uses CUB library for efficient parallel reductions
 * - Temporary storage allocated once and reused
 *
 * **Performance Optimizations:**
 *
 * - CUB device-wide parallel reduction for summation
 * - Single kernel launch for element-wise products + reduction
 * - Minimizes host-device synchronization
 *
 * @see ComputationBox for the interface definition
 * @see CpuComputationBox for CPU implementation
 */

#ifndef CUDA_SIMULATION_BOX_H_
#define CUDA_SIMULATION_BOX_H_

#include <vector>

#include "CudaCommon.h"
#include "ComputationBox.h"

/**
 * @class CudaComputationBox
 * @brief GPU-specific computation box with device-side operations.
 *
 * Extends ComputationBox with CUDA implementations of integration
 * and inner products that operate directly on GPU memory.
 *
 * @tparam T Numeric type (double or std::complex<double>)
 *
 * **CUB Reduction:**
 *
 * Uses NVIDIA CUB library for efficient parallel reduction:
 * - Device-wide sum reduction
 * - Temporary storage pre-allocated to avoid repeated allocations
 *
 * **Usage Pattern:**
 *
 * @code
 * CudaComputationBox<double> cb(nx, lx, bc);
 *
 * // Device array (owned; see CudaDeviceBuffer in CudaCommon.h)
 * CudaDeviceBuffer<double> d_field(n_grid);
 *
 * // Compute integral on GPU
 * double result = cb.integral_device(d_field);
 * @endcode
 */
template <typename T>
class CudaComputationBox : public ComputationBox<T>
{
private:
    CudaDeviceBuffer<CuDeviceData<T>> d_sum;       ///< Device buffer for reduction intermediate
    CudaDeviceBuffer<double> d_dv;                 ///< dV (volume element) in device memory

    /// @name CUB Reduction Storage
    /// @{
    size_t temp_storage_bytes = 0;     ///< Size of CUB temporary storage
    CudaDeviceBuffer<unsigned char> d_temp_storage;   ///< CUB temporary storage
    CudaDeviceBuffer<CuDeviceData<T>> d_sum_out;      ///< Output buffer for reduction
    /// @}

    /// @name Space Group Support
    /// @{
    CudaDeviceBuffer<double> d_basis_weights_;  ///< Device copy of the reduced-basis integration weights (orbit counts times the mask)
    /// @}

    /**
     * @brief Initialize GPU resources.
     *
     * Allocates CUB temporary storage and device buffers.
     */
    void initialize();

    /// Sum of the first n values of d_sum (CUB, legacy default stream), returned to the host.
    T reduce_sum_to_host(int n);

public:
    /**
     * @brief Construct CUDA computation box.
     *
     * @param nx   Grid dimensions [Nx, Ny, Nz]
     * @param lx   Box lengths [Lx, Ly, Lz]
     * @param bc   Boundary conditions
     * @param mask Optional mask for impenetrable regions
     */
    CudaComputationBox(std::vector<int> nx, std::vector<double> lx, std::vector<std::string> bc, const double* mask=nullptr);

    /**
     * @brief Construct CUDA computation box with lattice angles.
     *
     * @param nx     Grid dimensions [Nx, Ny, Nz]
     * @param lx     Box lengths [Lx, Ly, Lz]
     * @param bc     Boundary conditions
     * @param angles Lattice angles [alpha, beta, gamma] in degrees
     * @param mask   Optional mask for impenetrable regions
     */
    CudaComputationBox(std::vector<int> nx, std::vector<double> lx, std::vector<std::string> bc,
                       std::vector<double> angles, const double* mask=nullptr);

    /// Destructor. Frees GPU memory.
    ~CudaComputationBox() override;

    /**
     * @brief Update box lengths and recompute grid spacing.
     * @param new_lx New box lengths [Lx, Ly, Lz]
     */
    void set_lx(std::vector<double> new_lx) override;

    /**
     * @brief Update lattice parameters (box lengths and angles), recompute lattice vectors.
     * @param new_lx     New box lengths [Lx, Ly, Lz]
     * @param new_angles New lattice angles [alpha, beta, gamma] in degrees
     */
    void set_lattice_parameters(std::vector<double> new_lx, std::vector<double> new_angles) override;

    /**
     * @brief Set space group for reduced basis operations.
     *
     * Copies the reduced-basis weights (orbit counts times the mask) to device
     * memory for GPU-accelerated reductions.
     *
     * @param sg Space group pointer (nullptr to disable reduced basis)
     */
    void set_space_group(SpaceGroup* sg) override;

    /// @name Device-Side Operations
    /// @{

    /**
     * @brief Compute integral of a device array.
     *
     * Calculates: ∫ g(r) dr = Σᵢ g[i] * dV
     *
     * @param d_g Device array (size n_grid; n_basis with a space group)
     * @return Integral value
     */
    T integral_device(const CuDeviceData<T> *d_g);

    /**
     * @brief Compute mean (spatial average) of a device array.
     *
     * Calculates: (1/V) ∫ g(r) dr = integral_device(g) / volume
     *
     * @param d_g Device array (size n_grid; n_basis with a space group)
     * @return Mean value
     */
    T mean_device(const CuDeviceData<T> *d_g);

    /**
     * @brief Compute inner product of two device arrays.
     *
     * Calculates: ∫ g(r) h(r) dr = Σᵢ g[i] * h[i] * dV
     *
     * @param d_g First device array (size n_grid; n_basis with a space group)
     * @param d_h Second device array (same size)
     * @return Inner product value
     */
    T inner_product_device(const CuDeviceData<T> *d_g, const CuDeviceData<T> *d_h);

    /**
     * @brief Compute weighted inverse inner product.
     *
     * Calculates: Σᵢ g[i] * conj(h[i]) / w[i] * dV
     *
     * @param d_g First device array
     * @param d_h Second device array
     * @param d_w Weight array
     * @return Weighted inner product
     */
    T inner_product_inverse_weight_device(const CuDeviceData<T> *d_g, const CuDeviceData<T> *d_h, const CuDeviceData<T> *d_w);

    /**
     * @brief Compute multiple inner products in one kernel.
     *
     * Computes n_comp inner products for multi-species systems.
     *
     * @param n_comp Number of components
     * @param d_g    First array (size n_comp * n_grid; n_comp * n_basis with a space group)
     * @param d_h    Second array (same size)
     * @return Sum of inner products
     */
    T multi_inner_product_device(int n_comp, const CuDeviceData<T> *d_g, const CuDeviceData<T> *d_h);

    /**
     * @brief Subtract mean from device array.
     *
     * Makes the array have zero mean: g = g - integral(g) / accessible volume
     * (as ComputationBox::zero_mean)
     *
     * @param d_g Device array to modify in-place (size n_grid; n_basis with a space group)
     */
    void zero_mean_device(CuDeviceData<T> *d_g);

    /// @}
};
#endif
