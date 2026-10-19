/**
 * @file CudaComputationCommon.cu
 * @brief Accessors shared by the CUDA computation classes.
 */

#include <complex>

#include "CudaComputationCommon.h"

template <typename T>
CudaComputationCommon<T>::CudaComputationCommon(
    ComputationBox<T>* cb,
    Molecules *molecules,
    PropagatorComputationOptimizer *propagator_computation_optimizer,
    int n_streams)
    : PropagatorComputation<T>(cb, molecules, propagator_computation_optimizer),
      propagator_solver(nullptr),
      sc(nullptr),
      n_streams(n_streams),
      d_q_unity(nullptr),
      d_q_mask(nullptr)
{
    for(int i=0; i<MAX_STREAMS; i++)
    {
        streams[i][0] = nullptr;
        streams[i][1] = nullptr;
    }
}

template <typename T>
void CudaComputationCommon<T>::update_laplacian_operator()
{
    try
    {
        propagator_solver->update_laplacian_operator();
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

template <typename T>
T CudaComputationCommon<T>::get_total_partition(int polymer)
{
    try
    {
        return this->single_polymer_partitions[polymer];
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

template <typename T>
T CudaComputationCommon<T>::get_solvent_partition(int s)
{
    try
    {
        return this->single_solvent_partitions[s];
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

// Explicit template instantiation
template class CudaComputationCommon<double>;
template class CudaComputationCommon<std::complex<double>>;
