#include <iostream>
#include <sstream>
#include <complex>

#include "CpuComputationBox.h"

// Explicit template instantiation
template class CpuComputationBox<double>;
template class CpuComputationBox<std::complex<double>>;