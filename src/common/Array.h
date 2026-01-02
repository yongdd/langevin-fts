/**
 * @file Array.h
 * @brief Abstract array class for platform-independent array operations.
 *
 * This header defines the Array abstract class which provides a unified
 * interface for array operations across different computational platforms
 * (CPU and GPU). It enables element-wise arithmetic operations without
 * exposing platform-specific memory management.
 *
 * **Supported Operations:**
 *
 * - Element-wise arithmetic: add, subtract, multiply, divide
 * - Linear scaling: result = a * src + b
 * - Data access: get pointer, convert to vector, element access
 *
 * @see CpuArray for CPU implementation
 * @see CudaArray for GPU implementation
 *
 * @example
 * @code
 * // Create arrays through factory (not shown)
 * Array* a = factory->create_array(1000);
 * Array* b = factory->create_array(1000);
 * Array* c = factory->create_array(1000);
 *
 * // Element-wise operations
 * c->add(*a, *b);       // c = a + b
 * c->subtract(*a, *b);  // c = a - b
 * c->multiply(*a, *b);  // c = a * b
 *
 * // Linear scaling
 * c->linear_scaling(*a, 2.0, 1.0);  // c = 2*a + 1
 *
 * // Access data
 * double* ptr = c->get_ptr();
 * double val = (*c)[0];
 * std::vector<double> vec = c->to_vector();
 * @endcode
 */
#ifndef ARRAY_H_
#define ARRAY_H_

#include <iostream>
#include <vector>

#include "Exception.h"

/**
 * @class Array
 * @brief Abstract base class for platform-independent array operations.
 *
 * This class defines a common interface for array operations that can be
 * implemented efficiently on different platforms (CPU with MKL, CUDA GPU).
 * All operations are element-wise.
 *
 * **Memory Model:**
 *
 * - CPU arrays: Data stored in host memory
 * - GPU arrays: Data stored in device memory
 * - Data transfer between host and device is handled transparently
 *
 * **Thread Safety:**
 *
 * Array operations are not thread-safe. For parallel operations, ensure
 * proper synchronization or use separate arrays per thread.
 */
class Array
{
protected:
    std::string device;     ///< Device type: "cpu" or "cuda"
    unsigned int size = 0;  ///< Number of elements in the array
    int device_id;          ///< Device ID (for multi-GPU systems)

public:
    /**
     * @brief Virtual destructor.
     */
    virtual ~Array() {};

    /**
     * @brief Element-wise addition: this = src_1 + src_2.
     *
     * @param src_1 First source array
     * @param src_2 Second source array
     *
     * @note All arrays must have the same size.
     */
    virtual void add(const Array& src_1, const Array& src_2)=0;

    /**
     * @brief Element-wise subtraction: this = src_1 - src_2.
     *
     * @param src_1 First source array (minuend)
     * @param src_2 Second source array (subtrahend)
     */
    virtual void subtract(const Array& src_1, const Array& src_2)=0;

    /**
     * @brief Element-wise multiplication: this = src_1 * src_2.
     *
     * @param src_1 First source array
     * @param src_2 Second source array
     */
    virtual void multiply(const Array& src_1, const Array& src_2)=0;

    /**
     * @brief Element-wise division: this = src_1 / src_2.
     *
     * @param src_1 Numerator array
     * @param src_2 Denominator array (must be non-zero)
     *
     * @warning Division by zero results in undefined behavior.
     */
    virtual void divide(const Array& src_1, const Array& src_2)=0;

    /**
     * @brief Linear scaling: this = a * src + b.
     *
     * Computes element-wise linear transformation.
     *
     * @param src Source array
     * @param a   Multiplicative coefficient
     * @param b   Additive constant
     *
     * @example
     * @code
     * // Normalize: result = (src - mean) / std
     * result->linear_scaling(*src, 1.0/std, -mean/std);
     * @endcode
     */
    virtual void linear_scaling(const Array& src, const double a, const double b)=0;

    /**
     * @brief Copy assignment operator.
     *
     * Copies data from source array to this array.
     *
     * @param src Source array to copy from
     */
    virtual void operator=(const Array& src)=0;

    /**
     * @brief Set array data from raw pointer.
     *
     * Copies data from a raw pointer into the array.
     *
     * @param data Pointer to source data
     * @param size Number of elements to copy
     */
    virtual void set_data(double *data, unsigned int size)=0;

    /**
     * @brief Convert array to std::vector.
     *
     * @return Vector containing copy of array data
     *
     * @note For GPU arrays, this triggers a device-to-host copy.
     */
    virtual std::vector<double> to_vector() const=0;

    /**
     * @brief Get raw pointer to array data.
     *
     * @return Pointer to internal data storage
     *
     * @warning For GPU arrays, this returns a device pointer that cannot
     *          be dereferenced on the host.
     */
    virtual double* get_ptr() const=0;

    /**
     * @brief Get number of elements in array.
     *
     * @return Array size
     */
    virtual unsigned int get_size() const=0;

    /**
     * @brief Element access operator.
     *
     * @param index Element index (0 to size-1)
     * @return Value at the specified index
     *
     * @note For GPU arrays, this may trigger a device-to-host copy
     *       for the single element, which is inefficient for bulk access.
     */
    virtual double operator[](unsigned int index) const=0;
};

#endif