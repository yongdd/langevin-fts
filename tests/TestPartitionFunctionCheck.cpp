#include <complex>
#include <iostream>
#include <limits>
#include "PartitionFunctionCheck.h"

int main()
{
    using C = std::complex<double>;
    const double tiny = 1e-250;
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    // Absolute-magnitude checks incorrectly accept the second and third cases.
    const bool ok =
        relative_partition_error<C>({C(tiny, tiny), C(tiny, tiny)}, C(tiny, tiny)) == 0.0 &&
        relative_partition_error<C>({C(tiny, 0), C(0, tiny)}, C(tiny, 0)) > 1.0 &&
        relative_partition_error<double>({tiny, 1.01*tiny}, tiny) > 0.009 &&
        relative_partition_error<double>({tiny, tiny}, 2*tiny) == 0.5 &&
        !std::isfinite(relative_partition_error<double>({0., 0.}, 0.)) &&
        !std::isfinite(relative_partition_error<double>({inf}, 1.)) &&
        !std::isfinite(relative_partition_error<double>({nan}, 1.)) &&
        !std::isfinite(relative_partition_error<double>({}, 1.));
    if (!ok) {
        std::cerr << "Relative complex partition check regression failed.\n";
        return 1;
    }

    // GCE cancellation is compared against the absolute spatial integrand,
    // not the (possibly zero) result of that integral. Every invalid datum
    // still fails, and a finite mismatch is not excused merely because Q=0.
    const bool gce_ok =
        absolute_integrand_partition_error<C>({C(0.0), C(0.0)}, C(0.0), {0.0, 0.0}) == 0.0 &&
        absolute_integrand_partition_error<C>({C(0.0), C(0.0, 1e-16)}, C(0.0), {1.0, 1.0}) < 1e-14 &&
        absolute_integrand_partition_error<C>({C(0.0), C(0.0, 0.1)}, C(0.0), {1.0, 1.0}) > 0.09 &&
        !std::isfinite(absolute_integrand_partition_error<double>({0.0, 1.0}, 0.0, {0.0, 0.0})) &&
        !std::isfinite(absolute_integrand_partition_error<double>({0.0}, 1.0, {0.0})) &&
        !std::isfinite(absolute_integrand_partition_error<double>({inf}, 0.0, {1.0})) &&
        !std::isfinite(absolute_integrand_partition_error<double>({nan}, 0.0, {1.0})) &&
        !std::isfinite(absolute_integrand_partition_error<double>({0.0}, inf, {1.0})) &&
        !std::isfinite(absolute_integrand_partition_error<double>({0.0}, nan, {1.0})) &&
        !std::isfinite(absolute_integrand_partition_error<double>({0.0}, 0.0, {inf})) &&
        !std::isfinite(absolute_integrand_partition_error<double>({0.0}, 0.0, {nan})) &&
        !std::isfinite(absolute_integrand_partition_error<double>({0.0}, 0.0, {-1.0})) &&
        !std::isfinite(absolute_integrand_partition_error<double>({0.0}, 0.0, {})) &&
        !std::isfinite(absolute_integrand_partition_error<double>({}, 0.0, {}));
    if (!gce_ok)
    {
        std::cerr << "Absolute-integrand partition check regression failed.\n";
        return 1;
    }
    const double expected = absolute_integrand_partition_error<C>(
        {C(0.1, 0.2), C(0.12, 0.19)}, C(0.1, 0.2), {1.0, 2.0});
    for (const C factor : {C(tiny, tiny), C(1e200, -1e200), C(-2.0, 3.0)})
    {
        const double actual = absolute_integrand_partition_error<C>(
            {factor*C(0.1, 0.2), factor*C(0.12, 0.19)}, factor*C(0.1, 0.2),
            {std::abs(factor), 2.0*std::abs(factor)});
        if (!std::isfinite(actual) || std::abs(actual - expected) > 1e-15)
        {
            std::cerr << "GCE partition check changed under uniform field scaling.\n";
            return 1;
        }
    }
    return 0;
}
