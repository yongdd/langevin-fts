/**
 * @file CpuComputationReduceMemoryBase.cpp
 * @brief Common base class implementation for memory-efficient CPU propagator computation.
 *
 * Implements shared functionality between CpuComputationReduceMemoryContinuous
 * and CpuComputationReduceMemoryDiscrete, including concentration queries and
 * partition function accessors.
 *
 * @see CpuComputationReduceMemoryBase.h for class documentation
 */

#include <cmath>
#include "CpuComputationReduceMemoryBase.h"
#include "PropagatorCode.h"

template <typename T>
CpuComputationReduceMemoryBase<T>::CpuComputationReduceMemoryBase(
    ComputationBox<T>* cb,
    Molecules *molecules,
    PropagatorComputationOptimizer *propagator_computation_optimizer)
    : CpuComputationCommon<T>(cb, molecules, propagator_computation_optimizer, /*n_streams=*/1),
      total_max_n_segment(0),
      checkpoint_interval(1)
{
}

template <typename T>
void CpuComputationReduceMemoryBase<T>::compute_statistics(
    std::map<std::string, const T*> w_input,
    std::map<std::string, const T*> q_init)
{
    this->compute_propagators(w_input, q_init);
    this->compute_concentrations();
}

template <typename T>
void CpuComputationReduceMemoryBase<T>::advance_propagator_single_segment(
    T* q_init, T *q_out, int p, int v, int u)
{
    try
    {
        // Get block info from polymer
        const Block& block = this->molecules->get_polymer(p).get_block(v, u);
        std::string monomer_type = block.monomer_type;

        // Get ds_index from ContourLengthMapping
        const ContourLengthMapping& mapping = this->molecules->get_contour_length_mapping();
        int ds_index = mapping.get_ds_index(block.contour_length);

        const double *q_mask = this->cb->get_mask();
        this->propagator_solver->advance_propagator(q_init, q_out, monomer_type, q_mask, ds_index);
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

// Explicit template instantiation
#include "TemplateInstantiations.h"
INSTANTIATE_CLASS(CpuComputationReduceMemoryBase);
