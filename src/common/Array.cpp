/**
 * @file Array.cpp
 * @brief Implementation of abstract Array class.
 *
 * This file contains global operator overloading implementations for
 * the Array class. Currently disabled (commented out) as these operations
 * are implemented within platform-specific derived classes.
 *
 * @note The Array class itself is abstract; see CpuArray and CudaArray
 *       for concrete implementations.
 */

#include <iostream>
#include "Array.h"

// // global operator overloading
// Array operator+(const double a, const Array& array)
// {
//     return std::move(array.operator+(a));
// }
// Array operator-(const double a, const Array& array)
// {
//     return std::move(array.constant_minus_array(a));
// }
// Array operator*(const double a, const Array& array)
// {
//     return std::move(array.operator*(a));
// }
// Array operator/(const double a, const Array& array)
// {
//     return std::move(array.constant_divide_array(a));
// }