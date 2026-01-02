/**
 * @file CudaArray.h
 * @brief GPU implementation of the Array class using CUDA device memory.
 *
 * This header provides CudaArray, which implements the Array interface
 * using CUDA device memory. All operations are performed on the GPU
 * using custom CUDA kernels.
 *
 * **Memory Management:**
 *
 * - Data is allocated on GPU using cudaMalloc
 * - Copy operations use cudaMemcpy for host-device transfers
 * - Element access requires device-to-host transfer (slow for single elements)
 *
 * **Performance Considerations:**
 *
 * - Bulk operations (add, multiply, etc.) are fast (GPU-parallel)
 * - Single element access via operator[] is slow (requires cudaMemcpy)
 * - Use to_vector() to get all data at once when needed
 *
 * @see Array for the interface definition
 * @see CpuArray for CPU implementation
 */

#ifndef CUDA_ARRAY_H_
#define CUDA_ARRAY_H_

#include <vector>

#include "Exception.h"
#include "Array.h"

/**
 * @class CudaArray
 * @brief GPU-specific array implementation using CUDA device memory.
 *
 * Implements all Array operations using CUDA kernels. Data resides in
 * GPU memory and is transferred to host only when explicitly requested.
 *
 * **CUDA Kernel Usage:**
 *
 * Element-wise operations use ker_* kernels from CudaCommon.h
 * with automatic block/thread configuration.
 *
 * @example
 * @code
 * // Create array on GPU
 * CudaArray arr(1024);
 *
 * // Set data from host
 * double host_data[1024];
 * arr.set_data(host_data, 1024);
 *
 * // GPU operations
 * CudaArray result(1024);
 * result.add(arr, arr);  // Runs on GPU
 *
 * // Get result back to host
 * std::vector<double> host_result = result.to_vector();
 * @endcode
 */
class CudaArray : public Array
{
private:
    double *d_data = nullptr;  ///< Pointer to data in GPU device memory

public:
    /**
     * @brief Construct array of given size on GPU (uninitialized).
     * @param new_size Number of elements
     */
    CudaArray(unsigned new_size);

    /**
     * @brief Construct array by copying from host pointer to GPU.
     * @param new_data Source data in host memory
     * @param new_size Number of elements to copy
     */
    CudaArray(double* new_data, unsigned new_size);

    /**
     * @brief Copy constructor.
     * @param array Source array to copy (can be CudaArray or CpuArray)
     */
    CudaArray(const Array& array);

    /**
     * @brief Destructor. Frees GPU memory using cudaFree.
     */
    ~CudaArray();

    /** @brief Element-wise addition on GPU: this = src_1 + src_2 */
    void add(const Array& src_1, const Array& src_2) override;

    /** @brief Element-wise subtraction on GPU: this = src_1 - src_2 */
    void subtract(const Array& src_1, const Array& src_2) override;

    /** @brief Element-wise multiplication on GPU: this = src_1 * src_2 */
    void multiply(const Array& src_1, const Array& src_2) override;

    /** @brief Element-wise division on GPU: this = src_1 / src_2 */
    void divide(const Array& src_1, const Array& src_3) override;

    /** @brief Linear scaling on GPU: this = a * src + b */
    void linear_scaling(const Array& src, const double a, const double b) override;

    /** @brief Copy assignment (device-to-device or host-to-device) */
    void operator=(const Array&) override;

    /**
     * @brief Set data from host pointer.
     * @param data Host memory pointer
     * @param size Number of elements
     */
    void set_data(double *, unsigned int) override;

    /**
     * @brief Convert to std::vector (copies from GPU to host).
     * @return Vector containing all elements
     */
    std::vector<double> to_vector() const override;

    /**
     * @brief Get raw device pointer.
     * @return Pointer to GPU memory
     * @warning Returns device pointer; cannot be dereferenced on host
     */
    double* get_ptr() const override;

    /** @brief Get array size. */
    unsigned int get_size() const override;

    /**
     * @brief Element access by index (requires device-to-host copy).
     * @param idx Element index
     * @return Element value
     * @warning Slow for repeated access; use to_vector() for bulk reads
     */
    double operator[](unsigned int) const override;
};
#endif