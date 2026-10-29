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

// A grand-canonical chain does not divide by Q, so cancellation in its spatial
// integral is regular. Compare contour cuts on the scale of their absolute
// integrands, before that cancellation, with the same quadrature normalization.
template <typename T>
double absolute_integrand_partition_error(const std::vector<T>& values, T reference,
                                         const std::vector<double>& integrals)
{
    const double invalid = std::numeric_limits<double>::infinity();
    if (values.empty() || values.size() != integrals.size() || !std::isfinite(std::abs(reference)))
        return invalid;
    double scale = 0.0;
    double largest_partition = std::abs(reference);
    for (std::size_t i = 0; i < values.size(); ++i)
    {
        const double magnitude = std::abs(values[i]);
        if (!std::isfinite(magnitude) || !std::isfinite(integrals[i]) || integrals[i] < 0.0)
            return invalid;
        largest_partition = std::max(largest_partition, magnitude);
        scale = std::max(scale, integrals[i]);
    }
    if (scale == 0.0)
        return largest_partition == 0.0 ? 0.0 : invalid;
    double error = 0.0;
    for (const auto& value : values)
    {
        const double residual = std::abs(value / scale - reference / scale);
        if (!std::isfinite(residual))
            return invalid;
        error = std::max(error, residual);
    }
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
