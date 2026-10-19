/**
 * @file CudaComputationCommon.cu
 * @brief Accessors shared by the CUDA computation classes.
 */

#include <climits>
#include <cstdlib>
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
        d_segment_stress_[i] = nullptr;
    }
    for(int i=0; i<MAX_STREAMS; i++)
        gpu_error_check(cudaMalloc((void**)&d_segment_stress_[i], sizeof(T)*6));
}

template <typename T>
CudaComputationCommon<T>::~CudaComputationCommon()
{
    for(int i=0; i<MAX_STREAMS; i++)
    {
        if (d_segment_stress_[i] != nullptr)
            cudaFree(d_segment_stress_[i]);
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
template <typename T>
int CudaComputationCommon<T>::n_streams_from_env(int default_n_streams)
{
    const char* env = std::getenv("OMP_NUM_THREADS");
    if (env == nullptr || *env == '\0')
        return default_n_streams;

    // Leading integer of the value (OMP_NUM_THREADS may be a nested list "4,2")
    char* end = nullptr;
    const long value = std::strtol(env, &end, 10);
    if (end == env || value <= 0 || value > INT_MAX)
        return default_n_streams;
    return static_cast<int>(value);
}

template class CudaComputationCommon<double>;
template class CudaComputationCommon<std::complex<double>>;
