#include <iostream>
#include <sstream>
#include <complex>

#include "CpuComputationBox.h"

// Explicit template instantiation for double and std::complex<double>
template class CpuComputationBox<double>;
template class CpuComputationBox<std::complex<double>>;