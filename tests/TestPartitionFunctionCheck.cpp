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
    return 0;
}
