/**
 * @file CudaComputationBase.cu
 * @brief Common base class implementation for CUDA propagator computation.
 *
 * Implements shared functionality between CudaComputationContinuous and
 * CudaComputationDiscrete including concentration queries, partition
 * function accessors, and laplacian operator updates.
 *
 * **Template Instantiations:**
 *
 * - CudaComputationBase<double>: Real fields
 * - CudaComputationBase<std::complex<double>>: Complex fields
 *
 * @see CudaComputationContinuous for continuous chain implementation
 * @see CudaComputationDiscrete for discrete chain implementation
 */

#include <complex>
#include <cmath>
#include <cuComplex.h>

#include "CudaComputationBase.h"
#include "CudaComputationBox.h"

template <typename T>
CudaComputationBase<T>::CudaComputationBase(
    ComputationBox<T>* cb,
    Molecules *molecules,
    PropagatorComputationOptimizer *propagator_computation_optimizer)
    : PropagatorComputation<T>(cb, molecules, propagator_computation_optimizer),
      propagator_solver(nullptr),
      n_streams(0),
      d_q_unity(nullptr),
      d_q_mask(nullptr),
      sc(nullptr),
      d_phi(nullptr)
{
}

template <typename T>
void CudaComputationBase<T>::update_laplacian_operator()
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
T CudaComputationBase<T>::get_total_partition(int polymer)
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
void CudaComputationBase<T>::get_total_concentration(std::string monomer_type, T *phi)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();
        const int M = this->cb->get_total_grid();

        // Initialize to zero
        gpu_error_check(cudaMemset(d_phi, 0, sizeof(T)*M));

        // For each block
        for(const auto& d_block: d_phi_block)
        {
            const auto& key = d_block.first;
            std::string key_left = std::get<1>(key);
            int n_segment_right = this->propagator_computation_optimizer->get_computation_block(key).n_segment_right;
            if (PropagatorCode::get_monomer_type_from_key(key_left) == monomer_type && n_segment_right != 0)
                ker_lin_comb<<<N_BLOCKS, N_THREADS>>>(d_phi, 1.0, d_phi, 1.0, d_block.second, M);
        }
        gpu_error_check(cudaPeekAtLastError());

        // For each solvent
        for(int s=0;s<this->molecules->get_n_solvent_types();s++)
        {
            if (std::get<1>(this->molecules->get_solvent(s)) == monomer_type)
                ker_lin_comb<<<N_BLOCKS, N_THREADS>>>(d_phi, 1.0, d_phi, 1.0, d_phi_solvent[s], M);
        }
        gpu_error_check(cudaPeekAtLastError());
        gpu_error_check(cudaMemcpy(phi, d_phi, sizeof(T)*M, cudaMemcpyDeviceToHost));
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

template <typename T>
void CudaComputationBase<T>::get_total_concentration(int p, std::string monomer_type, T *phi)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = this->cb->get_total_grid();
        const int P = this->molecules->get_n_polymer_types();

        if (p < 0 || p > P-1)
            throw_with_line_number("Index (" + std::to_string(p) + ") must be in range [0, " + std::to_string(P-1) + "]");

        // Initialize to zero
        gpu_error_check(cudaMemset(d_phi, 0, sizeof(T)*M));

        // For each block
        for(const auto& d_block: d_phi_block)
        {
            const auto& key = d_block.first;
            int polymer_idx = std::get<0>(key);
            std::string key_left = std::get<1>(key);
            int n_segment_right = this->propagator_computation_optimizer->get_computation_block(key).n_segment_right;
            if (polymer_idx == p && PropagatorCode::get_monomer_type_from_key(key_left) == monomer_type && n_segment_right != 0)
                ker_lin_comb<<<N_BLOCKS, N_THREADS>>>(d_phi, 1.0, d_phi, 1.0, d_block.second, M);
        }
        gpu_error_check(cudaPeekAtLastError());
        gpu_error_check(cudaMemcpy(phi, d_phi, sizeof(T)*M, cudaMemcpyDeviceToHost));
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

template <typename T>
void CudaComputationBase<T>::get_total_concentration_gce(double fugacity, int p, std::string monomer_type, T *phi)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = this->cb->get_total_grid();
        const int P = this->molecules->get_n_polymer_types();

        if (p < 0 || p > P-1)
            throw_with_line_number("Index (" + std::to_string(p) + ") must be in range [0, " + std::to_string(P-1) + "]");

        // Initialize to zero
        gpu_error_check(cudaMemset(d_phi, 0, sizeof(T)*M));

        // For each block
        for(const auto& d_block: d_phi_block)
        {
            const auto& key = d_block.first;
            int polymer_idx = std::get<0>(key);
            std::string key_left = std::get<1>(key);
            int n_segment_right = this->propagator_computation_optimizer->get_computation_block(key).n_segment_right;
            if (polymer_idx == p && PropagatorCode::get_monomer_type_from_key(key_left) == monomer_type && n_segment_right != 0)
            {
                Polymer& pc = this->molecules->get_polymer(p);

                CuDeviceData<T> norm;
                if constexpr (std::is_same<T, double>::value)
                    norm = fugacity/pc.get_volume_fraction()*pc.get_alpha()*this->single_polymer_partitions[p];
                else
                    norm = stdToCuDoubleComplex(fugacity/pc.get_volume_fraction()*pc.get_alpha()*this->single_polymer_partitions[p]);
                ker_lin_comb<<<N_BLOCKS, N_THREADS>>>(d_phi, norm, d_block.second, 1.0, d_phi, M);
            }
        }
        gpu_error_check(cudaPeekAtLastError());
        gpu_error_check(cudaMemcpy(phi, d_phi, sizeof(T)*M, cudaMemcpyDeviceToHost));
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

template <typename T>
void CudaComputationBase<T>::get_block_concentration(int p, T *phi)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = this->cb->get_total_grid();
        const int P = this->molecules->get_n_polymer_types();

        if (p < 0 || p > P-1)
            throw_with_line_number("Index (" + std::to_string(p) + ") must be in range [0, " + std::to_string(P-1) + "]");

        if (this->propagator_computation_optimizer->use_aggregation())
            throw_with_line_number("Disable 'aggregation' option to invoke 'get_block_concentration'.");

        // Initialize to zero
        gpu_error_check(cudaMemset(d_phi, 0, sizeof(T)*M));

        Polymer& pc = this->molecules->get_polymer(p);
        std::vector<Block>& blocks = pc.get_blocks();

        for(size_t b=0; b<blocks.size(); b++)
        {
            std::string key_left  = pc.get_propagator_key(blocks[b].v, blocks[b].u);
            std::string key_right = pc.get_propagator_key(blocks[b].u, blocks[b].v);
            if (key_left < key_right)
                key_left.swap(key_right);

            ker_lin_comb<<<N_BLOCKS, N_THREADS>>>(d_phi, 0.0, d_phi, 1.0, d_phi_block[std::make_tuple(p, key_left, key_right)], M);
            gpu_error_check(cudaPeekAtLastError());
            gpu_error_check(cudaMemcpy(&phi[b*M], d_phi, sizeof(T)*M, cudaMemcpyDeviceToHost));
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

template <typename T>
T CudaComputationBase<T>::get_solvent_partition(int s)
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
void CudaComputationBase<T>::get_solvent_concentration(int s, T *phi)
{
    try
    {
        const int M = this->cb->get_total_grid();
        const int S = this->molecules->get_n_solvent_types();

        if (s < 0 || s > S-1)
            throw_with_line_number("Index (" + std::to_string(s) + ") must be in range [0, " + std::to_string(S-1) + "]");

        gpu_error_check(cudaMemcpy(phi, d_phi_solvent[s], sizeof(T)*M, cudaMemcpyDeviceToHost));
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

// Explicit template instantiation
template class CudaComputationBase<double>;
template class CudaComputationBase<std::complex<double>>;
