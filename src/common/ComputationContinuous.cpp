/**
 * @file ComputationContinuous.cpp
 * @brief Propagator computation for continuous chains, full storage.
 */

#include <cmath>
#include <omp.h>

#include "ComputationContinuous.h"
#include "PropagatorCode.h"
#include "SimpsonRule.h"

template <typename T>
ComputationContinuous<T>::ComputationContinuous(
    ComputationBox<T>* cb,
    Molecules* molecules,
    PropagatorComputationOptimizer* propagator_computation_optimizer,
    std::string method,
    SpaceGroup* space_group,
    const std::function<ComputationBackend<T>*()>& create_backend)
    : ComputationCommon<T>(cb, molecules, propagator_computation_optimizer, space_group, create_backend),
      method(method)
{
    try
    {
        // q(r, s) at every contour step of every propagator
        for (const auto& [key, edge] : this->propagator_computation_optimizer->get_computation_propagators())
        {
            const int size = edge.max_n_segment + 1;
            propagator_size[key] = size;
            propagator[key] = new BackendArray<T>*[size]();
            for (int n = 0; n < size; n++)
                propagator[key][n] = this->backend->allocate();
        }

        // One segment of each polymer gives its partition function
        int current_p = 0;
        for (const auto& [key, phi] : this->phi_block)
        {
            const int p = std::get<0>(key);
            if (p != current_p)
                continue;
            const auto& block = this->propagator_computation_optimizer->get_computation_block(key);
            const int n_aggregated = block.v_u.size()/block.n_repeated;
            single_partition_segment.push_back(std::make_tuple(
                p,
                propagator.at(std::get<1>(key))[block.n_segment_left],  // q
                propagator.at(std::get<2>(key))[0],                     // q†
                n_aggregated));
            current_p++;
        }
    }
    catch (std::exception& exc)
    {
        // The destructor does not run for a partly constructed object
        free_propagators();
        throw_without_line_number(exc.what());
    }
}

template <typename T>
ComputationContinuous<T>::~ComputationContinuous()
{
    free_propagators();
}

template <typename T>
void ComputationContinuous<T>::free_propagators()
{
    for (const auto& [key, q] : propagator)
    {
        for (int n = 0; n < propagator_size[key]; n++)
            if (q[n] != nullptr)
                this->backend->deallocate(q[n]);
        delete[] q;
    }
    propagator.clear();
}

template <typename T>
std::vector<std::pair<BackendArray<T>*, int>> ComputationContinuous<T>::junction_sources(
    const std::vector<std::tuple<std::string, int, int>>& deps)
{
    std::vector<std::pair<BackendArray<T>*, int>> sources;
    for (const auto& [sub_key, sub_n_segment, sub_n_repeated] : deps)
        sources.push_back({propagator.at(sub_key)[sub_n_segment], sub_n_repeated});
    return sources;
}

template <typename T>
void ComputationContinuous<T>::compute_propagators(
    std::map<std::string, const T*> w_input,
    std::map<std::string, const T*> q_init)
{
    try
    {
        this->set_fields(w_input, q_init);

        // The scheduler groups the propagators so that the propagators of one
        // group only depend on propagators of earlier groups
        for (const auto& parallel_job : this->sc->get_schedule())
        {
            ParallelErrors errors;
            #pragma omp parallel for num_threads(this->n_streams)
            for (size_t job = 0; job < parallel_job.size(); job++)
            {
                errors.run([&]
                {
                    const int STREAM = omp_get_thread_num();
                    const auto& [key, n_segment_from, n_segment_to] = parallel_job[job];
                    const auto& edge = this->propagator_computation_optimizer->get_computation_propagator(key);
                    BackendArray<T>** q = propagator.at(key);

                    // Initial condition q(r, 0)
                    if (n_segment_from == 0)
                    {
                        if (edge.deps.size() == 0)
                        {
                            if (key[0] == '{')  // grafting point
                                this->backend->copy_from_host(STREAM, q[0], q_init.at(PropagatorCode::get_q_input_idx_from_key(key)));
                            else                // free chain end
                                this->backend->set_unity(STREAM, q[0]);
                        }
                        else if (key[0] == '[')  // aggregated propagators
                            this->backend->junction_sum(STREAM, q[0], junction_sources(edge.deps));
                        else if (key[0] == '(')  // junction of branches
                            this->backend->junction_product(STREAM, q[0], junction_sources(edge.deps));

                        this->backend->apply_mask(STREAM, q[0]);
                    }

                    // q(r, n+1) from q(r, n)
                    const int ds_index = PropagatorCode::get_ds_index_from_key(key);
                    for (int n = n_segment_from; n < n_segment_to; n++)
                        this->backend->advance_propagator(STREAM, q[n], q[n+1], edge.monomer_type, ds_index);
                });
            }
            this->backend->synchronize();
            errors.rethrow();
        }

        // Partition function of each polymer: Q = ∫ q(r,s) q†(r,N-s) dr / V
        for (const auto& [p, q_left, q_right, n_aggregated] : single_partition_segment)
            this->single_polymer_partitions[p] = this->backend->inner_product(q_left, q_right)
                                                 /(n_aggregated*this->cb->get_volume());
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

template <typename T>
void ComputationContinuous<T>::advance_propagator_single_segment(T* q_init, T* q_out, int p, int v, int u)
{
    try
    {
        const Block& block = this->molecules->get_polymer(p).get_block(v, u);
        const int ds_index = this->molecules->get_contour_length_mapping().get_ds_index(block.contour_length);

        BackendArray<T>* q_in_b  = this->backend->allocate();
        BackendArray<T>* q_out_b = this->backend->allocate();
        this->backend->copy_from_host(0, q_in_b, q_init);
        this->backend->advance_propagator(0, q_in_b, q_out_b, block.monomer_type, ds_index);
        this->backend->synchronize();
        this->backend->copy_to_host(q_out, q_out_b);
        this->backend->deallocate(q_in_b);
        this->backend->deallocate(q_out_b);
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

template <typename T>
void ComputationContinuous<T>::compute_concentrations()
{
    try
    {
        // Concentration of each block (several blocks when propagators are aggregated)
        ParallelErrors errors;
        #pragma omp parallel for num_threads(this->n_streams)
        for (size_t b = 0; b < this->phi_block.size(); b++)
        {
            errors.run([&]
            {
                const int STREAM = omp_get_thread_num();
                auto it = this->phi_block.begin();
                std::advance(it, b);
                const auto& [p, key_left, key_right] = it->first;
                const auto& block = this->propagator_computation_optimizer->get_computation_block(it->first);
                const int N_LEFT  = block.n_segment_left;
                const int N_RIGHT = block.n_segment_right;

                // A block without segments has no concentration
                this->backend->set_zero(STREAM, it->second);
                if (N_RIGHT == 0)
                    return;

                // φ = Σ_n c_n q(N_LEFT-n) q†(n), Simpson's rule over the segments of the block
                const std::vector<double> simpson = SimpsonRule::get_coeff(N_RIGHT);
                std::vector<SegmentPair<T>> pairs;
                for (int n = 0; n <= N_RIGHT; n++)
                    pairs.push_back({propagator.at(key_left)[N_LEFT-n], propagator.at(key_right)[n], simpson[n]});
                this->backend->add_segment_products(STREAM, it->second, pairs);

                // Normalization: local_ds * volume_fraction / alpha * n_repeated / Q
                const int ds_index = PropagatorCode::get_ds_index_from_key(key_right);
                const double local_ds = this->molecules->get_contour_length_mapping().get_ds_from_index(ds_index);
                Polymer& pc = this->molecules->get_polymer(p);
                const T norm = (local_ds*pc.get_volume_fraction()/pc.get_alpha()*block.n_repeated)
                               /this->single_polymer_partitions[p];
                this->backend->normalize_concentration(STREAM, it->second, norm, nullptr);
            });
        }
        this->backend->synchronize();
        errors.rethrow();

        this->compute_solvent_concentrations();
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

template <typename T>
void ComputationContinuous<T>::compute_stress()
{
    // Requires the propagators of the current fields (compute_propagators()).
    try
    {
        this->check_stress_supported(method);

        // Stress of each block: Simpson's rule over the segment stresses
        std::vector<std::array<T, 6>> stress_of_block(this->phi_block.size());
        ParallelErrors errors;
        #pragma omp parallel for num_threads(this->n_streams)
        for (size_t b = 0; b < this->phi_block.size(); b++)
        {
            errors.run([&]
            {
                const int STREAM = omp_get_thread_num();
                auto it = this->phi_block.begin();
                std::advance(it, b);
                const auto& [p, key_left, key_right] = it->first;
                const auto& block = this->propagator_computation_optimizer->get_computation_block(it->first);
                const int N_LEFT  = block.n_segment_left;
                const int N_RIGHT = block.n_segment_right;

                stress_of_block[b] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
                if (N_RIGHT == 0)
                    return;

                // Every segment contributes, the chain ends included: q(r,0) = 1 has no
                // stress only for periodic or reflecting walls, not for absorbing
                // walls, masks or grafting points.
                const std::vector<double> simpson = SimpsonRule::get_coeff(N_RIGHT);
                std::vector<SegmentPair<T>> pairs;
                for (int n = 0; n <= N_RIGHT; n++)
                    pairs.push_back({propagator.at(key_left)[N_LEFT-n], propagator.at(key_right)[n],
                                     simpson[n]*block.n_repeated});
                stress_of_block[b] = this->backend->block_stress(STREAM, pairs, block.monomer_type);

                const int ds_index = PropagatorCode::get_ds_index_from_key(key_right);
                const double local_ds = this->molecules->get_contour_length_mapping().get_ds_from_index(ds_index);
                for (int d = 0; d < 6; d++)
                    stress_of_block[b][d] *= local_ds;
            });
        }
        this->backend->synchronize();
        errors.rethrow();

        // Sum the blocks of each polymer
        for (int p = 0; p < this->molecules->get_n_polymer_types(); p++)
            for (int d = 0; d < 6; d++)
                this->dq_dl[p][d] = 0.0;
        size_t b = 0;
        for (const auto& [key, phi] : this->phi_block)
        {
            for (int d = 0; d < 6; d++)
                this->dq_dl[std::get<0>(key)][d] += stress_of_block[b][d];
            b++;
        }

        this->convert_stress_to_lattice_derivatives();
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

template <typename T>
void ComputationContinuous<T>::get_chain_propagator(T* q_out, int polymer, int v, int u, int n)
{
    // For debugging and testing: the propagator of one block and direction
    try
    {
        Polymer& pc = this->molecules->get_polymer(polymer);
        const std::string dep = pc.get_propagator_key(v, u);

        if (!this->propagator_computation_optimizer->get_computation_propagators().contains(dep))
            throw_with_line_number("Could not find the propagator code '" + dep + "'. Disable 'aggregation' option to obtain propagator_computation_optimizer.");

        const int N_RIGHT = this->propagator_computation_optimizer->get_computation_propagator(dep).max_n_segment;
        if (n < 0 || n > N_RIGHT)
            throw_with_line_number("n (" + std::to_string(n) + ") must be in range [0, " + std::to_string(N_RIGHT) + "]");

        this->backend->copy_to_host(q_out, propagator.at(dep)[n]);
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

template <typename T>
bool ComputationContinuous<T>::check_total_partition()
{
    // Q must be the same at every segment of every block
    const int n_polymer_types = this->molecules->get_n_polymer_types();
    std::vector<std::vector<T>> total_partitions(n_polymer_types);

    for (const auto& [key, phi] : this->phi_block)
    {
        const auto& [p, key_left, key_right] = key;
        const auto& block = this->propagator_computation_optimizer->get_computation_block(key);
        const int n_propagators = block.v_u.size();

        for (int n = 0; n <= block.n_segment_right; n++)
        {
            T total_partition = this->backend->inner_product(
                propagator.at(key_left)[block.n_segment_left-n],
                propagator.at(key_right)[n])*(block.n_repeated/this->cb->get_volume());
            total_partition /= n_propagators;
            total_partitions[p].push_back(total_partition);
        }
    }

    // A uniform field shift can make Q arbitrarily small. Check the
    // relative complex error against the Q actually used to normalize phi.
    std::cout << "Polymer id: relative contour partition error" << std::endl;
    for (int p = 0; p < n_polymer_types; p++)
    {
        const double error = relative_partition_error(total_partitions[p], this->single_polymer_partitions[p]);
        std::cout << "\t" << p << ": " << error << std::endl;
        if (!std::isfinite(error) || error > PARTITION_TOLERANCE)
            return false;
    }
    return true;
}

// Explicit template instantiation
#include "TemplateInstantiations.h"
INSTANTIATE_CLASS(ComputationContinuous);
