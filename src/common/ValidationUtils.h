/**
 * @file ValidationUtils.h
 * @brief Common validation utilities to reduce duplicate error checking.
 *
 * Provides reusable validation functions for parameter checking, range
 * validation, and container lookups. These utilities consolidate repeated
 * validation patterns found throughout the codebase.
 *
 * @note All validation functions throw exceptions on failure using
 *       throw_with_line_number() for consistent error reporting.
 */

#ifndef VALIDATION_UTILS_H_
#define VALIDATION_UTILS_H_

#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <cmath>

#include "Exception.h"

namespace validation {

/**
 * @brief Validate that a value is positive.
 *
 * @tparam T Numeric type (int, double, etc.)
 * @param value Value to check
 * @param name  Variable name for error message
 * @throws Exception if value <= 0
 */
template<typename T>
inline void require_positive(T value, const std::string& name)
{
    if (value <= 0)
    {
        throw_with_line_number(name + " must be a positive number, got: " + std::to_string(value))
    }
}

/**
 * @brief Validate that a value is non-negative.
 *
 * @tparam T Numeric type
 * @param value Value to check
 * @param name  Variable name for error message
 * @throws Exception if value < 0
 */
template<typename T>
inline void require_non_negative(T value, const std::string& name)
{
    if (value < 0)
    {
        throw_with_line_number(name + " must be a non-negative number, got: " + std::to_string(value))
    }
}

/**
 * @brief Validate that a value is within an inclusive range.
 *
 * @tparam T Numeric type
 * @param value   Value to check
 * @param min_val Minimum allowed value (inclusive)
 * @param max_val Maximum allowed value (inclusive)
 * @param name    Variable name for error message
 * @throws Exception if value < min_val or value > max_val
 */
template<typename T>
inline void require_in_range(T value, T min_val, T max_val, const std::string& name)
{
    if (value < min_val || value > max_val)
    {
        throw_with_line_number(name + " (" + std::to_string(value) +
            ") must be in range [" + std::to_string(min_val) +
            ", " + std::to_string(max_val) + "]")
    }
}

/**
 * @brief Validate that all elements in a vector are positive.
 *
 * @tparam T Numeric type
 * @param values Vector of values to check
 * @param name   Variable name for error message
 * @throws Exception if any value <= 0
 */
template<typename T>
inline void require_all_positive(const std::vector<T>& values, const std::string& name)
{
    std::string str_values = "";
    for (size_t i = 0; i < values.size(); i++)
    {
        if (i > 0) str_values += ", ";
        str_values += std::to_string(values[i]);
    }

    for (const auto& v : values)
    {
        if (v <= 0)
        {
            throw_with_line_number(name + " (" + str_values + ") must contain only positive numbers")
        }
    }
}

/**
 * @brief Validate that two containers have matching sizes.
 *
 * @param size1 Size of first container
 * @param size2 Size of second container
 * @param name1 Name of first container
 * @param name2 Name of second container
 * @throws Exception if sizes don't match
 */
inline void require_same_size(size_t size1, size_t size2,
    const std::string& name1, const std::string& name2)
{
    if (size1 != size2)
    {
        throw_with_line_number("The sizes of " + name1 + " (" + std::to_string(size1) +
            ") and " + name2 + " (" + std::to_string(size2) + ") must match.")
    }
}

/**
 * @brief Check if a map contains a key.
 *
 * @tparam Map Map type
 * @tparam Key Key type
 * @param container Map to search
 * @param key       Key to find
 * @return true if key exists
 */
template<typename Map, typename Key>
inline bool contains(const Map& container, const Key& key)
{
    return container.find(key) != container.end();
}

/**
 * @brief Require that a map contains a key.
 *
 * @tparam Map Map type
 * @tparam Key Key type
 * @param container     Map to search
 * @param key           Key that must exist
 * @param container_name Name of container for error message
 * @throws Exception if key not found
 */
template<typename Map, typename Key>
inline void require_key(const Map& container, const Key& key, const std::string& container_name)
{
    if (!contains(container, key))
    {
        throw_with_line_number("Key not found in " + container_name)
    }
}

/**
 * @brief Require that a map contains a string key.
 *
 * Specialized version for string keys that includes the key in the error message.
 *
 * @tparam Map Map type with string keys
 * @param container     Map to search
 * @param key           String key that must exist
 * @param container_name Name of container for error message
 * @throws Exception if key not found
 */
template<typename Map>
inline void require_string_key(const Map& container, const std::string& key,
    const std::string& container_name)
{
    if (container.find(key) == container.end())
    {
        throw_with_line_number("Key '" + key + "' not found in " + container_name)
    }
}

/**
 * @brief Validate and normalize a mask array.
 *
 * Ensures all mask values are either 0.0 or 1.0 within tolerance.
 *
 * @param mask      Mask array to validate and normalize
 * @param size      Number of elements
 * @param tolerance Tolerance for floating point comparison
 * @throws Exception if any value is not approximately 0.0 or 1.0
 */
inline void validate_mask(double* mask, int size, double tolerance = 1e-7)
{
    for (int i = 0; i < size; i++)
    {
        if (std::abs(mask[i]) < tolerance)
        {
            mask[i] = 0.0;
        }
        else if (std::abs(mask[i] - 1.0) < tolerance)
        {
            mask[i] = 1.0;
        }
        else
        {
            throw_with_line_number("mask[" + std::to_string(i) + "] must be 0.0 or 1.0, got: " +
                std::to_string(mask[i]))
        }
    }
}

/**
 * @brief Convert string to lowercase.
 *
 * @param str Input string
 * @return Lowercase version of string
 */
inline std::string to_lower(const std::string& str)
{
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);
    return result;
}

} // namespace validation

#endif // VALIDATION_UTILS_H_
