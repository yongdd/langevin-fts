/**
 * @file CpuArray.h
 * @brief CPU implementation of the Array class.
 *
 * This header provides CpuArray, which implements the Array interface
 * for CPU-based array operations. Data is stored in standard host memory
 * and operations use simple loops (potentially OpenMP parallelized).
 *
 * @see Array for the interface definition
 * @see CudaArray for GPU implementation
 */

#ifndef CPU_ARRAY_H_
#define CPU_ARRAY_H_

#include <vector>

#include "Exception.h"
#include "Array.h"

/**
 * @class CpuArray
 * @brief CPU-specific array implementation using host memory.
 *
 * Implements all Array operations using standard C++ on host memory.
 * All element access and operations are direct without any data transfers.
 *
 * **Memory Management:**
 *
 * - Allocates memory using new[] in constructor
 * - Frees memory using delete[] in destructor
 * - Copy operations perform deep copies
 */
class CpuArray : public Array
{

private:
    double *data = nullptr;  ///< Pointer to array data in host memory

public:
    /**
     * @brief Construct array of given size (uninitialized).
     * @param new_size Number of elements
     */
    CpuArray(unsigned new_size);

    /**
     * @brief Construct array by copying from raw pointer.
     * @param new_data Source data pointer
     * @param new_size Number of elements to copy
     */
    CpuArray(double* new_data, unsigned new_size);

    /**
     * @brief Copy constructor.
     * @param array Source array to copy
     */
    CpuArray(const Array& array);

    /**
     * @brief Destructor. Frees allocated memory.
     */
    ~CpuArray();

    /** @brief Element-wise addition: this = src_1 + src_2 */
    void add(const Array& src_1, const Array& src_2) override;

    /** @brief Element-wise subtraction: this = src_1 - src_2 */
    void subtract(const Array& src_1, const Array& src_2) override;

    /** @brief Element-wise multiplication: this = src_1 * src_2 */
    void multiply(const Array& src_1, const Array& src_2) override;

    /** @brief Element-wise division: this = src_1 / src_2 */
    void divide(const Array& src_1, const Array& src_2) override;

    /** @brief Linear scaling: this = a * src + b */
    void linear_scaling(const Array& src, const double a, const double b) override;

    /** @brief Copy assignment operator */
    void operator=(const Array&) override;

    /** @brief Set data from raw pointer */
    void set_data(double *, unsigned int) override;

    /** @brief Convert to std::vector */
    std::vector<double> to_vector() const override;

    /** @brief Get raw data pointer */
    double* get_ptr() const override;

    /** @brief Get array size */
    unsigned int get_size() const override;

    /** @brief Element access by index */
    double operator[](unsigned int) const override;
};
#endif