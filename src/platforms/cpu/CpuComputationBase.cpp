/**
 * @file CpuComputationBase.cpp
 * @brief Common base class implementation for CPU propagator computation.
 *
 * The accessors shared with the reduce-memory variant live in
 * CpuComputationCommon; this file only constructs the base.
 *
 * **Template Instantiations:**
 *
 * - CpuComputationBase<double>: Real fields
 * - CpuComputationBase<std::complex<double>>: Complex fields
 *
 * @see CpuComputationContinuous for continuous chain implementation
 * @see CpuComputationDiscrete for discrete chain implementation
 */

#include <cmath>

#include "CpuComputationBase.h"

template <typename T>
CpuComputationBase<T>::CpuComputationBase(
    ComputationBox<T>* cb,
    Molecules *molecules,
    PropagatorComputationOptimizer *propagator_computation_optimizer)
    : CpuComputationCommon<T>(cb, molecules, propagator_computation_optimizer, /*n_streams=*/0)
{
}

// Explicit template instantiation
#include "TemplateInstantiations.h"
INSTANTIATE_CLASS(CpuComputationBase);
