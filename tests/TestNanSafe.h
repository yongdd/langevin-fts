/**
 * @file TestNanSafe.h
 * @brief Maxima for the test checks that keep a NaN.
 *
 * std::max_element and std::max compare with operator<, which is false for every
 * comparison with NaN, so both skip a NaN that is not the first value: an error
 * array or an error accumulator holding a NaN would then pass its tolerance check.
 * These helpers return NaN as soon as one value is NaN.
 */

#ifndef TEST_NAN_SAFE_H_
#define TEST_NAN_SAFE_H_

#include <cmath>
#include <limits>

/// Larger of a and b, or NaN if either is NaN (use instead of std::max)
inline double max_or_nan(double a, double b)
{
    if (std::isnan(a) || std::isnan(b))
        return std::numeric_limits<double>::quiet_NaN();
    return a < b ? b : a;
}

/// Largest value of a range, or NaN if any value is NaN (use instead of *std::max_element)
template <typename Range>
double max_or_nan(const Range& values)
{
    double result = -std::numeric_limits<double>::infinity();
    for (const auto& v : values)
        result = max_or_nan(result, static_cast<double>(v));
    return result;
}

/// Largest absolute value of a range, or NaN if any value is NaN
template <typename Range>
double max_abs_or_nan(const Range& values)
{
    double result = 0.0;
    for (const auto& v : values)
        result = max_or_nan(result, std::abs(static_cast<double>(v)));
    return result;
}

#endif
