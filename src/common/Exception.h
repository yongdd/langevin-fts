/**
 * @file Exception.h
 * @brief Custom exception class with enhanced error reporting for polymer FTS library.
 *
 * This header provides a custom exception class that extends std::runtime_error
 * with additional context information including file name, line number, and
 * function name where the exception was thrown. This facilitates debugging by
 * providing precise location information in error messages.
 *
 * @note Use the provided macros throw_with_line_number() and throw_without_line_number()
 *       instead of throwing Exception directly to automatically capture location info.
 *
 * @example
 * @code
 * // Throwing an exception with line number information
 * if (invalid_parameter) {
 *     throw_with_line_number("Invalid parameter: value must be positive");
 * }
 *
 * // Output: File: "src/common/MyClass.cpp", line: 42, function <myFunction>
 * //         Invalid parameter: value must be positive
 * @endcode
 */

#ifndef FTS_Exception_H_
#define FTS_Exception_H_

#include <string>
#include <stdexcept>

/**
 * @class Exception
 * @brief Enhanced runtime exception with source location information.
 *
 * Extends std::runtime_error to include file name, line number, and function
 * name in the error message. This provides detailed context for debugging
 * when exceptions are caught and logged.
 *
 * @note Prefer using throw_with_line_number() or throw_without_line_number()
 *       macros which automatically capture __FILE__, __LINE__, and __func__.
 */
class Exception : public std::runtime_error {
public:
    /**
     * @brief Construct exception with full location information.
     *
     * @param arg   Error message describing the exception
     * @param file  Source file name (use __FILE__ macro)
     * @param line  Line number in source file (use __LINE__ macro)
     * @param func  Function name (use __func__ macro)
     *
     * @example
     * @code
     * throw Exception("Memory allocation failed", __FILE__, __LINE__, __func__);
     * @endcode
     */
    Exception(const std::string &arg, const char *file, int line, const char *func):
        std::runtime_error("File: \"" + std::string(file) + "\", line: " + std::to_string(line) + ", function <" + std::string(func) + ">\n\t" + arg){};

    /**
     * @brief Construct exception without line number.
     *
     * @param arg   Error message describing the exception
     * @param file  Source file name (use __FILE__ macro)
     * @param func  Function name (use __func__ macro)
     *
     * @example
     * @code
     * throw Exception("Configuration error", __FILE__, __func__);
     * @endcode
     */
    Exception(const std::string &arg, const char *file, const char *func):
        std::runtime_error("File: \"" + std::string(file) + "\", function <" + std::string(func) + ">\n\t" + arg){};

    /**
     * @brief Destructor (noexcept).
     */
    ~Exception() throw() {};
};

/**
 * @def throw_with_line_number(arg)
 * @brief Convenience macro to throw Exception with automatic location capture.
 *
 * Automatically captures __FILE__, __LINE__, and __func__ at the throw site.
 *
 * @param arg Error message string describing the exception
 *
 * @example
 * @code
 * if (value < 0) {
 *     throw_with_line_number("Value must be non-negative, got: " + std::to_string(value));
 * }
 * @endcode
 */
#define throw_with_line_number(arg) throw Exception(arg, __FILE__, __LINE__, __func__);

/**
 * @def throw_without_line_number(arg)
 * @brief Convenience macro to throw Exception without line number.
 *
 * Automatically captures __FILE__ and __func__ at the throw site.
 * Use when line number is not needed or when throwing from header-only code.
 *
 * @param arg Error message string describing the exception
 *
 * @example
 * @code
 * if (!file_exists) {
 *     throw_without_line_number("Configuration file not found");
 * }
 * @endcode
 */
#define throw_without_line_number(arg) throw Exception(arg, __FILE__, __func__);

#endif