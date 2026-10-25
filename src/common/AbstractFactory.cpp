/**
 * @file AbstractFactory.cpp
 * @brief Creation of the propagator computation, the same on every platform.
 *
 * The chain model and reduce_memory select the computation class; the
 * platform factory contributes only its backend (create_backend()).
 *
 * **Checkpointing Mode:**
 *
 * When reduce_memory=true, creates the ComputationReduceMemory* variants
 * that store only checkpoint propagators and recompute intermediate values.
 * This trades computation time for reduced memory footprint.
 *
 * **Template Instantiations:**
 *
 * - AbstractFactory<double>: Factory for real field simulations
 * - AbstractFactory<std::complex<double>>: Factory for complex field simulations
 */

#include <climits>
#include <complex>
#include <memory>
#include <type_traits>

#include "AbstractFactory.h"
#include "ComputationContinuous.h"
#include "ComputationDiscrete.h"
#include "ComputationReduceMemoryContinuous.h"
#include "ComputationReduceMemoryDiscrete.h"
#include "NumericalMethods.h"

template <typename T>
PropagatorComputation<T>* AbstractFactory<T>::create_propagator_computation(ComputationBox<T>* cb, Molecules *molecules, PropagatorComputationOptimizer* propagator_computation_optimizer, std::string numerical_method, SpaceGroup* space_group)
{
    try
    {
        if constexpr (std::is_same_v<T, std::complex<double>>)
            if (space_group != nullptr)
                throw_with_line_number("Space group reduced basis is only supported for real fields.");

        // The solver: discrete chains have their own; continuous chains take the numerical method
        const bool discrete = (molecules->get_model_name() == "discrete");
        const NumericalMethodInfo& method_info = resolve_numerical_method(molecules->get_model_name(), numerical_method);
        validate_numerical_method(method_info, std::is_same_v<T, std::complex<double>>, cb->is_orthogonal());
        const std::string solver_type = method_info.solver_type;

        // Reduce-memory mode computes one propagator at a time
        const int max_n_streams = this->reduce_memory ? 1 : INT_MAX;
        auto make_backend = [=, this]() {
            return this->create_backend(cb, molecules, method_info, space_group, max_n_streams);
        };

        typename ComputationBox<T>::SpaceGroupGuard group_change(cb, space_group);
        std::unique_ptr<PropagatorComputation<T>> computation;
        if (discrete && !this->reduce_memory)
            computation = std::make_unique<ComputationDiscrete<T>>(cb, molecules, propagator_computation_optimizer, space_group, make_backend);
        else if (discrete)
            computation = std::make_unique<ComputationReduceMemoryDiscrete<T>>(cb, molecules, propagator_computation_optimizer, space_group, make_backend);
        else if (!this->reduce_memory)
            computation = std::make_unique<ComputationContinuous<T>>(cb, molecules, propagator_computation_optimizer, solver_type, space_group, make_backend);
        else
            computation = std::make_unique<ComputationReduceMemoryContinuous<T>>(cb, molecules, propagator_computation_optimizer, solver_type, space_group, make_backend);
        group_change.commit();
        return computation.release();
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

// Explicit template instantiation
template class AbstractFactory<double>;
template class AbstractFactory<std::complex<double>>;
