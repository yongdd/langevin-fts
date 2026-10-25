/**
 * @file ComputationDiscrete.cpp
 * @brief Propagator computation for discrete chains, full storage.
 */

#include <cmath>
#include <memory>
#include <omp.h>

#include "ComputationDiscrete.h"
#include "PropagatorCode.h"

template <typename T>
ComputationDiscrete<T>::ComputationDiscrete(
    ComputationBox<T>* cb,
    Molecules* molecules,
    PropagatorComputationOptimizer* propagator_computation_optimizer,
    SpaceGroup* space_group,
    const std::function<std::unique_ptr<ComputationBackend<T>>()>& create_backend)
    : ComputationCommon<T>(cb, molecules, propagator_computation_optimizer, space_group, create_backend)
{
    try
    {
        for (const auto& [key, edge] : this->propagator_computation_optimizer->get_computation_propagators())
        {
            // Segments 1..max_n_segment, e.g. max_n_segment = 5:
            //   O--O--O--O--O
            //   1  2  3  4  5
            const int size = edge.max_n_segment + 1;

            propagator[key].assign(size, nullptr);
            propagator[key][0] = nullptr;
            for (int n = 1; n < size; n++)
                propagator[key][n] = this->backend->allocate();

            // Half-bond states: the junction value it starts from (n = 0) and
            // the states half a bond past the segments where junctions attach
            propagator_half_steps[key].assign(size, nullptr);
            propagator_half_steps[key][0] = (edge.deps.size() > 0) ? this->backend->allocate() : nullptr;
            for (int n = 1; n < size; n++)
                propagator_half_steps[key][n] = edge.junction_ends.contains(n) ? this->backend->allocate() : nullptr;
        }

        // One segment of each polymer gives its partition function
        int current_p = 0;
        for (const auto& [key, phi] : this->phi_block)
        {
            const int p = std::get<0>(key);
            if (p != current_p)
                continue;
            const auto& block = this->propagator_computation_optimizer->get_computation_block(key);
            if (block.n_segment_left == 0)
                continue;
            const int n_aggregated = block.v_u.size()/block.n_repeated;
            single_partition_segment.push_back(std::make_tuple(
                p,
                propagator.at(std::get<1>(key))[block.n_segment_left],  // q
                propagator.at(std::get<2>(key))[1],                     // q†
                block.monomer_type,
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
ComputationDiscrete<T>::~ComputationDiscrete()
{
    free_propagators();
}

template <typename T>
void ComputationDiscrete<T>::free_propagators()
{
    for (auto* storage : {&propagator, &propagator_half_steps})
    {
        // Iterate the vectors themselves: an entry left by a failed assign() is empty
        for (const auto& [key, q] : *storage)
            for (BackendArray<T>* a : q)
                if (a != nullptr)
                    this->backend->deallocate(a);
        storage->clear();
    }
}

template <typename T>
void ComputationDiscrete<T>::compute_propagators(
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
                    const std::string& monomer_type = edge.monomer_type;
                    BackendArray<T>** q      = propagator.at(key).data();
                    BackendArray<T>** q_half = propagator_half_steps.at(key).data();

                    // First segment q(r, 1)
                    if (n_segment_from == 0)
                    {
                        if (edge.deps.size() == 0)  // chain end: q(1) = exp(-w ds) q_init
                        {
                            if (key[0] == '{')      // grafting point
                                this->backend->copy_from_host(STREAM, q[1], q_init.at(PropagatorCode::get_q_input_idx_from_key(key)));
                            else                    // free chain end
                                this->backend->set_unity(STREAM, q[1]);
                            this->backend->multiply_boltzmann_factor(STREAM, q[1], monomer_type);
                        }
                        else if (key[0] == '[')     // aggregated propagators
                        {
                            // Propagators aggregated at a junction (n = 0) are half-bond
                            // states; the others are segments
                            std::vector<std::pair<BackendArray<T>*, int>> sources;
                            for (const auto& [sub_key, sub_n_segment, sub_n_repeated] : edge.deps)
                            {
                                BackendArray<T>* src = (sub_n_segment == 0) ? propagator_half_steps.at(sub_key)[0]
                                                                             : propagator.at(sub_key)[sub_n_segment];
                                sources.push_back({src, sub_n_repeated});
                            }
                            this->backend->junction_sum(STREAM, q[1], sources);

                            if (std::get<1>(edge.deps[0]) == 0)
                            {
                                // Junction value, then half a bond and the first segment
                                this->backend->copy(STREAM, q_half[0], q[1]);
                                this->backend->advance_propagator_half_bond_step(STREAM, q[1], q[1], monomer_type);
                                this->backend->multiply_boltzmann_factor(STREAM, q[1], monomer_type);
                            }
                            else
                            {
                                // A full bond and the next segment
                                this->backend->advance_propagator(STREAM, q[1], q[1], monomer_type, 0);
                            }
                        }
                        else if (key[0] == '(')     // junction of branches
                        {
                            // Junction value: product of the half-bond states of the branches
                            //     A
                            //     |
                            // O - . - B      . junction, - half bonds, O first segment
                            //     |
                            //     C
                            std::vector<std::pair<BackendArray<T>*, int>> sources;
                            for (const auto& [sub_key, sub_n_segment, sub_n_repeated] : edge.deps)
                                sources.push_back({propagator_half_steps.at(sub_key)[sub_n_segment], sub_n_repeated});
                            this->backend->junction_product(STREAM, q_half[0], sources);

                            // Half a bond and the first segment
                            if (n_segment_to > 0)
                            {
                                this->backend->advance_propagator_half_bond_step(STREAM, q_half[0], q[1], monomer_type);
                                this->backend->multiply_boltzmann_factor(STREAM, q[1], monomer_type);
                            }
                        }
                    }
                    if (n_segment_to == 0)
                        return;

                    int n_from = n_segment_from;
                    if (n_from == 0)
                    {
                        this->backend->apply_mask(STREAM, q[1]);
                        if (q_half[1] != nullptr)
                            this->backend->advance_propagator_half_bond_step(STREAM, q[1], q_half[1], monomer_type);
                        n_from = 1;
                    }

                    // q(r, n+1) from q(r, n); discrete chains use the global ds (ds_index 0)
                    for (int n = n_from; n < n_segment_to; n++)
                        this->backend->advance_propagator(STREAM, q[n], q[n+1], monomer_type, 0);

                    // q(r, n+1/2) where a junction attaches
                    for (int n = n_from; n < n_segment_to; n++)
                        if (q_half[n+1] != nullptr)
                            this->backend->advance_propagator_half_bond_step(STREAM, q[n+1], q_half[n+1], monomer_type);
                });
            }
            this->backend->synchronize();
            errors.rethrow();
        }

        // Partition function of each polymer: Q = ∫ q(r,n) q†(r,N+1-n) / exp(-w ds) dr / V
        for (const auto& [p, q_left, q_right, monomer_type, n_aggregated] : single_partition_segment)
            this->single_polymer_partitions[p] = this->backend->inner_product_inverse_weight(
                q_left, q_right, this->backend->boltzmann_factor(monomer_type))
                /(n_aggregated*this->cb->get_volume());

        this->propagators_computed_ = true;
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

template <typename T>
void ComputationDiscrete<T>::advance_propagator_single_segment(T* q_init, T* q_out, int p, int v, int u)
{
    this->check_box_space_group();
    try
    {
        const std::string monomer_type = this->molecules->get_polymer(p).get_block(v, u).monomer_type;

        // Two temporary fields, freed also when a step throws
        BackendArray<T>* q_in_b  = nullptr;
        BackendArray<T>* q_out_b = nullptr;
        try
        {
            q_in_b  = this->backend->allocate();
            q_out_b = this->backend->allocate();
            this->backend->copy_from_host(0, q_in_b, q_init);
            this->backend->advance_propagator(0, q_in_b, q_out_b, monomer_type, 0);
            this->backend->synchronize();
            this->backend->copy_to_host(q_out, q_out_b);
        }
        catch (...)
        {
            this->backend->deallocate(q_in_b);
            this->backend->deallocate(q_out_b);
            throw;
        }
        this->backend->deallocate(q_in_b);
        this->backend->deallocate(q_out_b);
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

template <typename T>
void ComputationDiscrete<T>::compute_concentrations()
{
    this->check_box_space_group();
    this->require_propagators();
    try
    {
        // Concentration of each block (several blocks when propagators are aggregated)
        ParallelErrors errors;
        #pragma omp parallel for num_threads(this->n_streams)
        for (size_t b = 0; b < this->phi_block_iterators.size(); b++)
        {
            errors.run([&]
            {
                const int STREAM = omp_get_thread_num();
                const auto it = this->phi_block_iterators[b];
                const auto& [p, key_left, key_right] = it->first;
                const auto& block = this->propagator_computation_optimizer->get_computation_block(it->first);
                const int N_LEFT  = block.n_segment_left;
                const int N_RIGHT = block.n_segment_right;

                // A block without segments has no concentration
                this->backend->set_zero(STREAM, it->second);
                if (N_RIGHT == 0)
                    return;

                // φ = Σ_n q(N_LEFT+1-n) q†(n) / exp(-w ds), n = 1..N_RIGHT
                std::vector<SegmentPair<T>> pairs;
                for (int n = 1; n <= N_RIGHT; n++)
                    pairs.push_back({propagator.at(key_left)[N_LEFT-n+1], propagator.at(key_right)[n], 1.0});
                this->backend->add_segment_products(STREAM, it->second, pairs);

                // Remove the shared bead factor; getters apply ensemble normalization.
                this->backend->normalize_concentration(STREAM, it->second, 1.0,
                                                       this->backend->boltzmann_factor(block.monomer_type));
            });
        }
        this->backend->synchronize();
        errors.rethrow();

        this->compute_solvent_concentrations();
        this->concentrations_computed_ = true;
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

template <typename T>
void ComputationDiscrete<T>::compute_stress()
{
    this->check_box_space_group();
    this->require_propagators();
    try
    {
        this->check_stress_supported("pseudospectral");
        this->backend->prepare_stress();

        // Stress of each block: sum over its bonds
        std::vector<std::array<T, 6>> stress_of_block(this->phi_block.size());
        ParallelErrors errors;
        #pragma omp parallel for num_threads(this->n_streams)
        for (size_t b = 0; b < this->phi_block_iterators.size(); b++)
        {
            errors.run([&]
            {
                const int STREAM = omp_get_thread_num();
                const auto it = this->phi_block_iterators[b];
                const auto& [p, key_left, key_right] = it->first;
                const auto& block = this->propagator_computation_optimizer->get_computation_block(it->first);
                const int N_LEFT  = block.n_segment_left;
                const int N_RIGHT = block.n_segment_right;
                const bool left_is_leaf  = this->propagator_computation_optimizer->get_computation_propagator(key_left).deps.size() == 0;
                const bool right_is_leaf = this->propagator_computation_optimizer->get_computation_propagator(key_right).deps.size() == 0;

                // Bond n joins q(N_LEFT-n) and q†(n). The bonds at the two ends are
                // half bonds to a junction; free ends and aggregation points have none.
                std::vector<SegmentPair<T>> pairs;
                for (int n = 0; n <= N_RIGHT; n++)
                {
                    if (n == N_LEFT)                          // half bond at the left end (v)
                    {
                        if (!left_is_leaf)
                            pairs.push_back({propagator_half_steps.at(key_left)[0], propagator.at(key_right)[N_RIGHT],
                                             static_cast<double>(block.n_repeated), true});
                    }
                    else if (n == 0 && key_right[0] == '(')   // half bond at the right end (u), a junction
                    {
                        if (!right_is_leaf)
                            pairs.push_back({propagator.at(key_left)[N_LEFT], propagator_half_steps.at(key_right)[0],
                                             static_cast<double>(block.n_repeated), true});
                    }
                    else if (n > 0)                           // full bond within the block
                    {
                        pairs.push_back({propagator.at(key_left)[N_LEFT-n], propagator.at(key_right)[n],
                                         static_cast<double>(block.n_repeated), false});
                    }
                }
                stress_of_block[b] = this->backend->block_stress(STREAM, pairs, block.monomer_type);
                this->scale_by_contour_step(stress_of_block[b], key_right);
            });
        }
        this->backend->synchronize();
        errors.rethrow();

        // Sum the blocks of each polymer
        this->set_stress_from_blocks(stress_of_block);
        this->stress_computed_ = true;
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

template <typename T>
void ComputationDiscrete<T>::get_chain_propagator(T* q_out, int polymer, int v, int u, int n)
{
    this->check_box_space_group();
    this->require_propagators();
    // For debugging and testing: the propagator of one block and direction
    try
    {
        Polymer& pc = this->molecules->get_polymer(polymer);
        const std::string dep = pc.get_propagator_key(v, u);

        if (!this->propagator_computation_optimizer->get_computation_propagators().contains(dep))
            throw_with_line_number("Could not find the propagator code '" + dep + "'. Disable 'aggregation' option to obtain propagator_computation_optimizer.");

        const int N_RIGHT = this->propagator_computation_optimizer->get_computation_propagator(dep).max_n_segment;
        if (n < 1 || n > N_RIGHT)
            throw_with_line_number("n (" + std::to_string(n) + ") must be in range [1, " + std::to_string(N_RIGHT) + "]");

        this->backend->copy_to_host(q_out, propagator.at(dep)[n]);
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

template <typename T>
std::vector<double> ComputationDiscrete<T>::compute_partition_errors()
{
    this->check_box_space_group();
    this->require_propagators();
    // Q must be the same at every segment of every block
    const int n_polymer_types = this->molecules->get_n_polymer_types();
    std::vector<std::vector<T>> total_partitions(n_polymer_types);

    for (const auto& [key, phi] : this->phi_block)
    {
        const auto& [p, key_left, key_right] = key;
        const auto& block = this->propagator_computation_optimizer->get_computation_block(key);
        const int n_propagators = block.v_u.size();
        BackendArray<T>* exp_dw = this->backend->boltzmann_factor(block.monomer_type);

        for (int n = 1; n <= block.n_segment_right; n++)
        {
            T total_partition = this->backend->inner_product_inverse_weight(
                propagator.at(key_left)[block.n_segment_left-n+1],
                propagator.at(key_right)[n], exp_dw)*(block.n_repeated/this->cb->get_volume());
            total_partition /= n_propagators;
            total_partitions[p].push_back(total_partition);
        }
    }

    return relative_partition_errors(total_partitions, this->single_polymer_partitions);
}

// Explicit template instantiation
#include "TemplateInstantiations.h"
INSTANTIATE_CLASS(ComputationDiscrete);
