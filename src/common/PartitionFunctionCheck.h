#ifndef PARTITION_FUNCTION_CHECK_H_
#define PARTITION_FUNCTION_CHECK_H_

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <vector>

// Compare complex values, not their magnitudes. Scaling before subtraction
// avoids overflow and makes the check invariant under a uniform field shift.
template <typename T>
double relative_partition_error(const std::vector<T>& values, T reference)
{
    double scale = std::abs(reference);
    if (!std::isfinite(scale))
        return std::numeric_limits<double>::infinity();
    for (const auto& value : values)
    {
        const double magnitude = std::abs(value);
        if (!std::isfinite(magnitude))
            return std::numeric_limits<double>::infinity();
        scale = std::max(scale, magnitude);
    }
    if (values.empty() || scale == 0.0 || std::abs(reference) == 0.0)
        return std::numeric_limits<double>::infinity();
    double error = 0.0;
    for (const auto& value : values)
        error = std::max(error, std::abs(value / scale - reference / scale));
    return error;
}

// The error of every polymer: its contour values against the Q that normalizes
// its concentrations. A uniform field shift can make Q arbitrarily small, hence
// the relative (complex) error.
template <typename T>
std::vector<double> relative_partition_errors(const std::vector<std::vector<T>>& values, const std::vector<T>& references)
{
    std::vector<double> errors;
    errors.reserve(references.size());
    for (std::size_t p = 0; p < references.size(); p++)
        errors.push_back(relative_partition_error(values[p], references[p]));
    return errors;
}

#endif
