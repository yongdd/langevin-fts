/**
 * @file TestComputationFactory.h
 * @brief Create a propagator computation the way the Python layer does.
 *
 * The C++ tests build their computation boxes and molecules directly and get
 * the propagator computation from the platform factory, so they exercise the
 * same construction path as `PropagatorSolver`.
 */

#ifndef TEST_COMPUTATION_FACTORY_H_
#define TEST_COMPUTATION_FACTORY_H_

#include <complex>
#include <memory>
#include <string>
#include <type_traits>

#include "PlatformSelector.h"
#include "PropagatorComputation.h"

/**
 * @brief Propagator computation on one platform.
 *
 * @param platform         "cpu-fftw", "cpu-mkl" or "cuda"
 * @param reduce_memory    Store only checkpoints (reduce-memory mode)
 * @param numerical_method "rqm4", "rk2" or "cn-adi2" (ignored for discrete chains)
 */
template <typename T>
PropagatorComputation<T>* make_propagator_computation(
    const std::string& platform, bool reduce_memory,
    ComputationBox<T>* cb, Molecules* molecules,
    PropagatorComputationOptimizer* propagator_computation_optimizer,
    const std::string& numerical_method = "rqm4")
{
    std::unique_ptr<AbstractFactory<T>> factory;
    if constexpr (std::is_same_v<T, double>)
        factory.reset(PlatformSelector::create_factory_real(platform, reduce_memory));
    else
        factory.reset(PlatformSelector::create_factory_complex(platform, reduce_memory));
    return factory->create_propagator_computation(cb, molecules, propagator_computation_optimizer, numerical_method);
}

#endif
