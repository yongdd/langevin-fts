/**
 * @file ComputationReduceMemoryContinuous.cpp
 * @brief Propagator computation for continuous chains, reduced memory.
 */

#include <algorithm>
#include <memory>
#include <cmath>

#include "ComputationReduceMemoryContinuous.h"
#include "PropagatorCode.h"
#include "SimpsonRule.h"

template <typename T>
ComputationReduceMemoryContinuous<T>::ComputationReduceMemoryContinuous(
    ComputationBox<T>* cb,
    Molecules* molecules,
    PropagatorComputationOptimizer* propagator_computation_optimizer,
    std::string method,
    SpaceGroup* space_group,
    const std::function<std::unique_ptr<ComputationBackend<T>>()>& create_backend)
    : ComputationReduceMemoryBase<T>(cb, molecules, propagator_computation_optimizer, space_group,
                                     /*first_segment=*/0, create_backend),
      method(method)
{
    try
    {
        // The start of every propagator and the segments other propagators start from
        for (const auto& [key, edge] : this->propagator_computation_optimizer->get_computation_propagators())
        {
            this->add_checkpoint(key, 0);
            for (const auto& [sub_key, sub_n_segment, sub_n_repeated] : edge.deps)
                this->add_checkpoint(sub_key, sub_n_segment);
        }
        this->add_regular_checkpoints();

        // One segment of each polymer gives its partition function
        int current_p = 0;
        for (const auto& [key, phi] : this->phi_block)
        {
            const int p = std::get<0>(key);
            if (p != current_p)
                continue;
            const auto& block = this->propagator_computation_optimizer->get_computation_block(key);
            const int n_aggregated = block.v_u.size()/block.n_repeated;
            this->add_checkpoint(std::get<1>(key), block.n_segment_left);
            this->add_checkpoint(std::get<2>(key), 0);
            single_partition_segment.push_back(std::make_tuple(
                p, std::get<1>(key), block.n_segment_left, std::get<2>(key), n_aggregated));
            current_p++;
        }
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

template <typename T>
int ComputationReduceMemoryContinuous<T>::propagator_ds_index(const std::string& key) const
{
    return PropagatorCode::get_ds_index_from_key(key);
}

template <typename T>
void ComputationReduceMemoryContinuous<T>::compute_propagators(
    std::map<std::string, const T*> w_input,
    std::map<std::string, const T*> q_init)
{
    try
    {
        this->set_fields(w_input, q_init);

        // The propagators one at a time, in the scheduler's order
        for (const auto& parallel_job : this->sc->get_schedule())
        {
            for (const auto& [key, n_segment_from, n_segment_to] : parallel_job)
            {
                const auto& edge = this->propagator_computation_optimizer->get_computation_propagator(key);

                // q(r, n_segment_from)
                if (n_segment_from == 0)
                {
                    if (edge.deps.size() == 0)
                    {
                        if (key[0] == '{')  // grafting point
                            this->backend->copy_from_host(0, this->q_work[0], q_init.at(PropagatorCode::get_q_input_idx_from_key(key)));
                        else                // free chain end
                            this->backend->set_unity(0, this->q_work[0]);
                    }
                    else
                    {
                        std::vector<std::pair<CheckpointArray<T>*, int>> sources;
                        for (const auto& [sub_key, sub_n_segment, sub_n_repeated] : edge.deps)
                            sources.push_back({this->checkpoint_at(sub_key, sub_n_segment), sub_n_repeated});
                        if (key[0] == '[')       // aggregated propagators
                            this->backend->junction_sum(0, this->q_work[0], sources);
                        else if (key[0] == '(')  // junction of branches
                            this->backend->junction_product(0, this->q_work[0], sources);
                    }
                    this->backend->apply_mask(0, this->q_work[0]);
                    this->backend->store_checkpoint(0, this->checkpoint_at(key, 0), this->q_work[0]);
                }
                else
                    this->backend->load_checkpoint(0, this->q_work[0], this->checkpoint_at(key, n_segment_from));

                // q(r, n+1) from q(r, n), keeping the checkpointed segments
                const int ds_index = PropagatorCode::get_ds_index_from_key(key);
                int cur = 0;
                for (int n = n_segment_from; n < n_segment_to; n++)
                {
                    this->backend->advance_propagator(0, this->q_work[cur], this->q_work[1-cur], edge.monomer_type, ds_index);
                    cur = 1-cur;
                    auto it = this->checkpoint.find(std::make_tuple(key, n+1));
                    if (it != this->checkpoint.end())
                        this->backend->store_checkpoint(0, it->second, this->q_work[cur]);
                }
            }
        }
        this->backend->synchronize();

        // Partition function of each polymer: Q = ∫ q(r,s) q†(r,N-s) dr / V
        for (const auto& [p, key_left, n_left, key_right, n_aggregated] : single_partition_segment)
        {
            this->backend->load_checkpoint(0, this->q_work[0], this->checkpoint_at(key_left, n_left));
            this->backend->load_checkpoint(0, this->q_work[1], this->checkpoint_at(key_right, 0));
            this->backend->synchronize();
            this->single_polymer_partitions[p] = this->backend->inner_product(this->q_work[0], this->q_work[1])
                                                 /(n_aggregated*this->cb->get_volume());
        }
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

template <typename T>
void ComputationReduceMemoryContinuous<T>::advance_propagator_single_segment(T* q_init, T* q_out, int p, int v, int u)
{
    this->check_box_space_group();
    try
    {
        const Block& block = this->molecules->get_polymer(p).get_block(v, u);
        const int ds_index = this->molecules->get_contour_length_mapping().get_ds_index(block.contour_length);

        this->backend->copy_from_host(0, this->q_work[0], q_init);
        this->backend->advance_propagator(0, this->q_work[0], this->q_work[1], block.monomer_type, ds_index);
        this->backend->synchronize();
        this->backend->copy_to_host(q_out, this->q_work[1]);
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

template <typename T>
void ComputationReduceMemoryContinuous<T>::compute_concentrations()
{
    this->check_box_space_group();
    try
    {
        // Concentration of each block (several blocks when propagators are aggregated)
        for (const auto& [key, phi] : this->phi_block)
        {
            const auto& [p, key_left, key_right] = key;
            const auto& block = this->propagator_computation_optimizer->get_computation_block(key);
            const int N_LEFT  = block.n_segment_left;
            const int N_RIGHT = block.n_segment_right;

            // A block without segments has no concentration
            this->backend->set_zero(0, phi);
            if (N_RIGHT == 0)
                continue;

            // φ = Σ_n c_n q(N_LEFT-n) q†(n), Simpson's rule over the segments of the block
            const std::vector<double> simpson = SimpsonRule::get_coeff(N_RIGHT);
            this->for_each_segment_pair(key_left, key_right, N_LEFT, 0, N_RIGHT,
                [&](int n, BackendArray<T>* q, BackendArray<T>* q_dagger)
                {
                    this->backend->add_segment_products(0, phi, {{q, q_dagger, simpson[n]}});
                });

            // Getters normalize the integral for the requested ensemble.
        }
        this->backend->synchronize();

        this->compute_solvent_concentrations();
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

template <typename T>
void ComputationReduceMemoryContinuous<T>::compute_stress()
{
    this->check_box_space_group();
    // Requires the propagators of the current fields (compute_propagators()).
    try
    {
        this->check_stress_supported(method);
        this->backend->prepare_stress();

        // Stress of each block: Simpson's rule over the segment stresses
        std::vector<std::array<T, 6>> stress_of_block;
        for (const auto& [key, phi] : this->phi_block)
        {
            const auto& [p, key_left, key_right] = key;
            const auto& block = this->propagator_computation_optimizer->get_computation_block(key);
            const int N_LEFT  = block.n_segment_left;
            const int N_RIGHT = block.n_segment_right;

            std::array<T, 6> stress = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            if (N_RIGHT > 0)
            {
                // Every segment contributes, the chain ends included (see ComputationContinuous)
                const std::vector<double> simpson = SimpsonRule::get_coeff(N_RIGHT);
                this->for_each_segment_pair(key_left, key_right, N_LEFT, 0, N_RIGHT,
                    [&](int n, BackendArray<T>* q, BackendArray<T>* q_dagger)
                    {
                        const std::array<T, 6> segment = this->backend->block_stress(
                            0, {{q, q_dagger, simpson[n]*block.n_repeated}}, block.monomer_type);
                        for (int d = 0; d < 6; d++)
                            stress[d] += segment[d];
                    });
                this->scale_by_contour_step(stress, key_right);
            }
            stress_of_block.push_back(stress);
        }

        // Sum the blocks of each polymer
        this->set_stress_from_blocks(stress_of_block);
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

template <typename T>
void ComputationReduceMemoryContinuous<T>::get_chain_propagator(T* q_out, int polymer, int v, int u, int n)
{
    this->check_box_space_group();
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

        this->recompute_segments(dep, n, 1);
        this->backend->synchronize();
        this->backend->copy_to_host(q_out, this->q_recal[0]);
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

template <typename T>
bool ComputationReduceMemoryContinuous<T>::check_total_partition()
{
    this->check_box_space_group();
    // Q must be the same at every segment of every block
    const int n_polymer_types = this->molecules->get_n_polymer_types();
    std::vector<std::vector<T>> total_partitions(n_polymer_types);

    for (const auto& [key, phi] : this->phi_block)
    {
        const auto& [p, key_left, key_right] = key;
        const auto& block = this->propagator_computation_optimizer->get_computation_block(key);
        const int n_propagators = block.v_u.size();

        this->for_each_segment_pair(key_left, key_right, block.n_segment_left, 0, block.n_segment_right,
            [&](int, BackendArray<T>* q, BackendArray<T>* q_dagger)
            {
                T total_partition = this->backend->inner_product(q, q_dagger)*(block.n_repeated/this->cb->get_volume());
                total_partition /= n_propagators;
                total_partitions[p].push_back(total_partition);
            });
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
INSTANTIATE_CLASS(ComputationReduceMemoryContinuous);
