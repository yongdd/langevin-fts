/**
 * @file CpuArray.cpp
 * @brief CPU implementation of memory array management.
 *
 * Provides CPU-side memory allocation and basic arithmetic operations
 * using standard C++ new/delete. Implements the abstract Array interface
 * for host memory operations.
 *
 * **Memory Layout:**
 *
 * Arrays are allocated contiguously in host memory for optimal cache access.
 * Standard C++ allocation ensures proper alignment for vectorization.
 *
 * **Operations:**
 *
 * - Element-wise arithmetic: add, subtract, multiply, divide
 * - Linear scaling: dst[i] = a * src[i] + b
 * - Data transfer: set_data, to_vector
 *
 * @see Array for abstract interface
 * @see CudaArray for GPU equivalent
 */

#include <iostream>
#include "CpuArray.h"

/**
 * @brief Construct CPU array with given size.
 *
 * @param new_size Number of elements to allocate
 */
CpuArray::CpuArray(unsigned new_size)
{
    this->data = new double[new_size];
    this->size = new_size;
}
CpuArray::CpuArray(double* new_data, unsigned int new_size)
{
    this->data = new double[new_size];
    this->size = new_size;
    for(unsigned int i=0; i<new_size; i++)
        this->data[i] = new_data[i];
}
CpuArray::CpuArray(const Array& array)
{
    unsigned int new_size = array.get_size();
    double *new_data = array.get_ptr();

    this->data = new double[new_size];
    this->size = new_size;
    for(unsigned int i=0; i<size; i++)
        this->data[i] = new_data[i];
}
CpuArray::~CpuArray()
{
    if (this->data != nullptr)
    {
        delete [] this->data;
        this->size = 0;
        // std::cout << "Memory has been deallocated" << std::endl;
    }
}
void CpuArray::operator=(const Array& arr)
{
    unsigned int arr_size = arr.get_size();
    double *arr_data = arr.get_ptr();

    if (this->size != arr_size)
    {
        throw_with_line_number("Sizes of arrays ("
            + std::to_string(this->size) + ", "
            + std::to_string(arr_size) + ") do not match.");
    }
    for(unsigned int i=0; i<size; i++)
        this->data[i] = arr_data[i];
}
void CpuArray::set_data(double * arr_data, unsigned int arr_size)
{
    if (this->size != arr_size)
    {
        throw_with_line_number("Sizes of arrays ("
            + std::to_string(this->size) + ", "
            + std::to_string(arr_size) + ") do not match.");
    }
    for(unsigned int i=0; i<size; i++)
        this->data[i] = arr_data[i];
}
std::vector<double> CpuArray::to_vector() const
{
    std::vector<double> vec(size);
    for(unsigned int i=0; i<size; i++)
        vec[i] = this->data[i];
    return std::move(vec);
}
double* CpuArray::get_ptr() const
{
    return this->data;
}
unsigned int CpuArray::get_size() const
{
    return this->size;
}
double CpuArray::operator[](unsigned int i) const
{
    if (size <= i)
        throw_with_line_number("Index [" + std::to_string(i) + "] is out of bound.");
    return data[i];
}

// Arithmetic operations with two arrays
void CpuArray::add(const Array& src_1, const Array& src_2)
{
    unsigned int src1_size = src_1.get_size();
    unsigned int src2_size = src_2.get_size();
    double *src1_data = src_1.get_ptr();
    double *src2_data = src_2.get_ptr();

    if (this->size != src1_size || this->size != src2_size)
    {
        throw_with_line_number("Sizes of arrays ("
            + std::to_string(this->size) + ", "
            + std::to_string(src1_size)  + ", "
            + std::to_string(src2_size)  + ") do not match.");
    }

    for (unsigned int i=0; i<size; i++)
        data[i] = src1_data[i] + src2_data[i];
}
void CpuArray::subtract(const Array& src_1, const Array& src_2)
{
    unsigned int src1_size = src_1.get_size();
    unsigned int src2_size = src_2.get_size();
    double *src1_data = src_1.get_ptr();
    double *src2_data = src_2.get_ptr();

    if (this->size != src1_size || this->size != src2_size)
    {
        throw_with_line_number("Sizes of arrays ("
            + std::to_string(this->size) + ", "
            + std::to_string(src1_size)  + ", "
            + std::to_string(src2_size)  + ") do not match.");
    }

    for (unsigned int i=0; i<size; i++)
        data[i] = src1_data[i] - src2_data[i];
}
void CpuArray::multiply(const Array& src_1, const Array& src_2)
{
    unsigned int src1_size = src_1.get_size();
    unsigned int src2_size = src_2.get_size();
    double *src1_data = src_1.get_ptr();
    double *src2_data = src_2.get_ptr();

    if (this->size != src1_size || this->size != src2_size)
    {
        throw_with_line_number("Sizes of arrays ("
            + std::to_string(this->size) + ", "
            + std::to_string(src1_size)  + ", "
            + std::to_string(src2_size)  + ") do not match.");
    }

    for (unsigned int i=0; i<size; i++)
        data[i] = src1_data[i] * src2_data[i];
}
void CpuArray::divide(const Array& src_1, const Array& src_2)
{
    unsigned int src1_size = src_1.get_size();
    unsigned int src2_size = src_2.get_size();
    double *src1_data = src_1.get_ptr();
    double *src2_data = src_2.get_ptr();

    if (this->size != src1_size || this->size != src2_size)
    {
        throw_with_line_number("Sizes of arrays ("
            + std::to_string(this->size) + ", "
            + std::to_string(src1_size)  + ", "
            + std::to_string(src2_size)  + ") do not match.");
    }

    for (unsigned int i=0; i<size; i++)
        data[i] = src1_data[i] / src2_data[i];
}
// Arithmetic operations with an array and a float number
void CpuArray::linear_scaling(const Array& src, const double a, const double b)
{
    unsigned int src_size = src.get_size();
    double *src_data = src.get_ptr();

    if (this->size != src_size)
    {
        throw_with_line_number("Sizes of arrays ("
            + std::to_string(this->size) + ", "
            + std::to_string(src_size) + ") do not match.");
    }

    for (unsigned int i=0; i<size; i++)
        data[i] = a*src_data[i] + b;
}