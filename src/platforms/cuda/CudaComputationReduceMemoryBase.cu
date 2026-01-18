/**
 * @file CudaComputationReduceMemoryBase.cu
 * @brief Common base class implementation for memory-efficient GPU propagator computation.
 *
 * Implements shared functionality between CudaComputationReduceMemoryContinuous
 * and CudaComputationReduceMemoryDiscrete, including concentration queries and
 * partition function accessors.
 *
 * @see CudaComputationReduceMemoryBase.h for class documentation
 */

#include <cmath>
#include <vector>
#include "CudaComputationReduceMemoryBase.h"
#include "CudaCommon.h"
#include "PropagatorCode.h"

template <typename T>
CudaComputationReduceMemoryBase<T>::CudaComputationReduceMemoryBase(
    ComputationBox<T>* cb,
    Molecules *molecules,
    PropagatorComputationOptimizer *propagator_computation_optimizer)
    : PropagatorComputation<T>(cb, molecules, propagator_computation_optimizer),
      propagator_solver(nullptr),
      sc(nullptr),
      n_streams(1),
      d_q_unity(nullptr),
      d_q_mask(nullptr),
      total_max_n_segment(0),
      checkpoint_interval(1),
      d_phi(nullptr)
{
    // Initialize workspace pointers
    for(int i=0; i<MAX_STREAMS; i++)
    {
        for(int j=0; j<2; j++)
        {
            d_q_one[i][j] = nullptr;
            d_propagator_sub_dep[i][j] = nullptr;
            streams[i][j] = nullptr;
        }
    }
    d_workspace[0] = nullptr;
    d_workspace[1] = nullptr;
}

template <typename T>
void CudaComputationReduceMemoryBase<T>::alloc_checkpoint_memory(T** ptr, size_t count)
{
    const size_t bytes = count * sizeof(T);
    gpu_error_check(cudaMallocHost(reinterpret_cast<void**>(ptr), bytes));
}

template <typename T>
void CudaComputationReduceMemoryBase<T>::free_checkpoint_memory(T* ptr)
{
    cudaFreeHost(ptr);
}

template <typename T>
void CudaComputationReduceMemoryBase<T>::update_laplacian_operator()
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
void CudaComputationReduceMemoryBase<T>::compute_statistics(
    std::map<std::string, const T*> w_input,
    std::map<std::string, const T*> q_init)
{
    this->compute_propagators(w_input, q_init);
    this->compute_concentrations();
}

template <typename T>
void CudaComputationReduceMemoryBase<T>::advance_propagator_single_segment(
    T* q_init, T *q_out, std::string monomer_type)
{
    try
    {
        const int M = this->cb->get_total_grid();

        // Create device pointers
        CuDeviceData<T> *d_q_in;
        CuDeviceData<T> *d_q_out;

        gpu_error_check(cudaMalloc((void**)&d_q_in, sizeof(T)*M));
        gpu_error_check(cudaMalloc((void**)&d_q_out, sizeof(T)*M));

        // Copy to device
        gpu_error_check(cudaMemcpy(d_q_in, q_init, sizeof(T)*M, cudaMemcpyHostToDevice));

        // Advance
        propagator_solver->advance_propagator(0, d_q_in, d_q_out, monomer_type, d_q_mask);
        gpu_error_check(cudaDeviceSynchronize());

        // Copy back
        gpu_error_check(cudaMemcpy(q_out, d_q_out, sizeof(T)*M, cudaMemcpyDeviceToHost));

        // Free device memory
        cudaFree(d_q_in);
        cudaFree(d_q_out);
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

template <typename T>
void CudaComputationReduceMemoryBase<T>::get_total_concentration(std::string monomer_type, T *phi)
{
    try
    {
        const int M = this->cb->get_total_grid();
        // Initialize array
        for(int i=0; i<M; i++)
            phi[i] = 0.0;

        // For each block (phi_block is in pinned host memory)
        for(const auto& block: phi_block)
        {
            std::string key_left = std::get<1>(block.first);
            int n_segment_right = this->propagator_computation_optimizer->get_computation_block(block.first).n_segment_right;
            if (PropagatorCode::get_monomer_type_from_key(key_left) == monomer_type && n_segment_right != 0)
            {
                for(int i=0; i<M; i++)
                    phi[i] += block.second[i];
            }
        }

        // For each solvent (always in host memory)
        for(int s=0;s<this->molecules->get_n_solvent_types();s++)
        {
            if (std::get<1>(this->molecules->get_solvent(s)) == monomer_type)
            {
                T *phi_solvent_ = phi_solvent[s];
                for(int i=0; i<M; i++)
                    phi[i] += phi_solvent_[i];
            }
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

template <typename T>
void CudaComputationReduceMemoryBase<T>::get_total_concentration(int p, std::string monomer_type, T *phi)
{
    try
    {
        const int M = this->cb->get_total_grid();
        const int P = this->molecules->get_n_polymer_types();

        if (p < 0 || p > P-1)
            throw_with_line_number("Index (" + std::to_string(p) + ") must be in range [0, " + std::to_string(P-1) + "]");

        // Initialize array
        for(int i=0; i<M; i++)
            phi[i] = 0.0;

        // For each block (phi_block is in pinned host memory)
        for(const auto& block: phi_block)
        {
            int polymer_idx = std::get<0>(block.first);
            std::string key_left = std::get<1>(block.first);
            int n_segment_right = this->propagator_computation_optimizer->get_computation_block(block.first).n_segment_right;
            if (polymer_idx == p && PropagatorCode::get_monomer_type_from_key(key_left) == monomer_type && n_segment_right != 0)
            {
                for(int i=0; i<M; i++)
                    phi[i] += block.second[i];
            }
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

template <typename T>
void CudaComputationReduceMemoryBase<T>::get_total_concentration_gce(double fugacity, int p, std::string monomer_type, T *phi)
{
    try
    {
        const int M = this->cb->get_total_grid();
        const int P = this->molecules->get_n_polymer_types();

        if (p < 0 || p > P-1)
            throw_with_line_number("Index (" + std::to_string(p) + ") must be in range [0, " + std::to_string(P-1) + "]");

        // Initialize array
        for(int i=0; i<M; i++)
            phi[i] = 0.0;

        // For each block (phi_block is in pinned host memory)
        for(const auto& block: phi_block)
        {
            int polymer_idx = std::get<0>(block.first);
            std::string key_left = std::get<1>(block.first);
            int n_segment_right = this->propagator_computation_optimizer->get_computation_block(block.first).n_segment_right;
            if (polymer_idx == p && PropagatorCode::get_monomer_type_from_key(key_left) == monomer_type && n_segment_right != 0)
            {
                Polymer& pc = this->molecules->get_polymer(p);
                T norm = fugacity/pc.get_volume_fraction()*pc.get_alpha()*this->single_polymer_partitions[p];
                for(int i=0; i<M; i++)
                    phi[i] += block.second[i]*norm;
            }
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

template <typename T>
void CudaComputationReduceMemoryBase<T>::get_block_concentration(int p, T *phi)
{
    try
    {
        const int M = this->cb->get_total_grid();
        const int P = this->molecules->get_n_polymer_types();

        if (p < 0 || p > P-1)
            throw_with_line_number("Index (" + std::to_string(p) + ") must be in range [0, " + std::to_string(P-1) + "]");

        if (this->propagator_computation_optimizer->use_aggregation())
            throw_with_line_number("Disable 'aggregation' option to invoke 'get_block_concentration'.");

        Polymer& pc = this->molecules->get_polymer(p);
        std::vector<Block>& blocks = pc.get_blocks();

        for(size_t b=0; b<blocks.size(); b++)
        {
            std::string key_left  = pc.get_propagator_key(blocks[b].v, blocks[b].u);
            std::string key_right = pc.get_propagator_key(blocks[b].u, blocks[b].v);
            if (key_left < key_right)
                key_left.swap(key_right);

            T* _essential_phi_block = phi_block[std::make_tuple(p, key_left, key_right)];
            // phi_block is in pinned host memory
            for(int i=0; i<M; i++)
                phi[i+b*M] = _essential_phi_block[i];
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

template <typename T>
T CudaComputationReduceMemoryBase<T>::get_solvent_partition(int s)
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

template <typename T>
void CudaComputationReduceMemoryBase<T>::get_solvent_concentration(int s, T *phi)
{
    try
    {
        const int M = this->cb->get_total_grid();
        const int S = this->molecules->get_n_solvent_types();

        if (s < 0 || s > S-1)
            throw_with_line_number("Index (" + std::to_string(s) + ") must be in range [0, " + std::to_string(S-1) + "]");

        T *phi_solvent_ = phi_solvent[s];
        for(int i=0; i<M; i++)
            phi[i] = phi_solvent_[i];
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

// Explicit template instantiation
template class CudaComputationReduceMemoryBase<double>;
template class CudaComputationReduceMemoryBase<std::complex<double>>;
