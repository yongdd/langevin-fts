/**
 * @file PropagatorAggregator.cpp
 * @brief Implementation of propagator aggregation algorithms.
 *
 * Implements chain-model-specific algorithms for identifying and merging
 * equivalent propagator computations to reduce computational cost.
 *
 * @see PropagatorAggregator.h for class documentation
 */

#include <iostream>
#include <algorithm>
#include <set>

#include "PropagatorAggregator.h"
#include "PropagatorComputationOptimizer.h"
#include "PropagatorCode.h"

std::map<std::string, ComputationBlock> PropagatorAggregator::aggregate_continuous_chain(
    std::map<std::string, ComputationBlock> set_I)
{
    // Example) Continuous chain aggregation
    //
    // Input (key format: DK+ds_index):
    //   n_segment_right, key,       n_segment_left, n_repeated
    //   6,               (C)B+0,    6,              1
    //   4,               (D)B+0,    4,              3
    //   4,               (E)B+0,    4,              2
    //   2,               (F)B+0,    2,              1
    //
    //      ↓   Aggregation (sliced propagators: n_segment=0 → length_index=0)
    //
    // Output:
    //   6,               (C)B+0,              6,  1
    //   4 → 0,           (D)B+0,              4,  3  (sliced)
    //   4 → 0,           (E)B+0,              4,  2  (sliced)
    //   4,               [(D)B0:3,(E)B0:2]B+0, 4,  1  (aggregated)
    //   2,               (F)B+0,              2,  1

    // Collect unique n_segment values
    std::set<int> set_n_compute;
    for (const auto& item : set_I)
        set_n_compute.insert(item.second.n_segment_right);

    // Aggregate propagators with same n_segment
    for (const int n_segment_current : set_n_compute)
    {
        // Collect propagators with this n_segment
        std::map<std::string, ComputationBlock, ComparePropagatorKey> set_S;
        for (const auto& item : set_I)
        {
            if (item.second.n_segment_right == n_segment_current)
                set_S[item.first] = item.second;
        }

        // Skip if nothing to aggregate
        if (set_S.size() == 1)
            continue;

        // Set sliced propagators to n_segment=0
        for (const auto& item : set_S)
            set_I[item.first].n_segment_right = 0;

        // Build aggregated key: [dep1,dep2,...]monomer+ds_index
        std::string monomer_type = PropagatorCode::get_monomer_type_from_key(set_S.begin()->second.monomer_type);
        std::string aggregated_key = "[";
        std::vector<std::tuple<int, int>> all_v_u;
        int ds_index = -1;
        bool first = true;

        for (auto it = set_S.rbegin(); it != set_S.rend(); it++)
        {
            const std::string& key = it->first;

            // Extract ds_index (same for all in set_S)
            if (ds_index < 0)
                ds_index = PropagatorCode::get_ds_index_from_key(key);

            // Strip +ds_index to get DK part
            size_t plus_pos = key.rfind('+');
            std::string dk_part = (plus_pos != std::string::npos) ? key.substr(0, plus_pos) : key;

            if (!first) aggregated_key += ",";
            first = false;

            // Append "0" (length_index=0 for sliced propagators with n_segment=0)
            aggregated_key += dk_part + "0";

            if (set_I[key].n_repeated > 1)
                aggregated_key += ":" + std::to_string(set_I[key].n_repeated);

            // Collect v_u
            all_v_u.insert(all_v_u.end(), set_I[key].v_u.begin(), set_I[key].v_u.end());
        }
        aggregated_key += "]" + monomer_type + "+" + std::to_string(ds_index);

        // Add aggregated block
        set_I[aggregated_key].monomer_type = monomer_type;
        set_I[aggregated_key].n_segment_right = n_segment_current;
        set_I[aggregated_key].n_segment_left = n_segment_current;
        set_I[aggregated_key].v_u = all_v_u;
        set_I[aggregated_key].n_repeated = 1;
    }
    return set_I;
}

std::map<std::string, ComputationBlock> PropagatorAggregator::aggregate_discrete_chain(
    std::map<std::string, ComputationBlock> set_I)
{
    // Example) Discrete chain aggregation (hierarchical merging)
    //
    // Input (key format: DK+ds_index, uses n_segment directly):
    //   n_segment_right, key,       n_segment_left, n_repeated
    //   6,               (C)B+0,    6,              1
    //   4,               (D)B+0,    4,              3
    //   4,               (E)B+0,    4,              2
    //   2,               (F)B+0,    2,              1
    //
    //      ↓   Aggregation (minimum_n_segment=1 for discrete chains)
    //
    // Output:
    //   6,               (C)B+0,              6,  1
    //   4 → 1,           (D)B+0,              4,  3  (sliced to n_segment=1)
    //   4 → 1,           (E)B+0,              4,  2  (sliced to n_segment=1)
    //   3,               [(D)B1:3,(E)B1:2]B+0, 3,  1  (aggregated, n_segment=4-1=3)
    //   2,               (F)B+0,              2,  1

    constexpr int MIN_N_SEGMENT = 1;  // Discrete chains require at least 1 segment

    // set_F: final result, set_I: working set for hierarchical merging
    std::map<std::string, ComputationBlock> set_F = set_I;

    // Remove propagators with n_segment <= 1 from working set (can't be aggregated further)
    for (const auto& item : set_F)
    {
        if (item.second.n_segment_right <= MIN_N_SEGMENT)
            set_I.erase(item.first);
    }

    // Hierarchical merging: repeatedly merge propagators with largest n_segment values
    while (set_I.size() > 1)
    {
        // Find 2nd largest n_segment
        std::vector<int> n_segments;
        for (const auto& item : set_I)
            n_segments.push_back(item.second.n_segment_right);
        std::sort(n_segments.rbegin(), n_segments.rend());
        int n_segment_threshold = n_segments[1];

        // Collect propagators with n_segment >= threshold
        std::map<std::string, ComputationBlock, ComparePropagatorKey> set_S;
        for (const auto& item : set_I)
        {
            if (item.second.n_segment_right >= n_segment_threshold)
                set_S[item.first] = item.second;
        }

        // Slice: reduce n_segment_right by (threshold - 1)
        int slice_amount = n_segment_threshold - MIN_N_SEGMENT;
        for (const auto& item : set_S)
            set_F[item.first].n_segment_right -= slice_amount;

        // Build aggregated key: [dep1,dep2,...]monomer+ds_index
        std::string monomer_type = PropagatorCode::get_monomer_type_from_key(set_S.begin()->second.monomer_type);
        std::string aggregated_key = "[";
        std::vector<std::tuple<int, int>> all_v_u;
        int ds_index = -1;
        bool first = true;

        for (auto it = set_S.rbegin(); it != set_S.rend(); it++)
        {
            const std::string& key = it->first;

            if (ds_index < 0)
                ds_index = PropagatorCode::get_ds_index_from_key(key);

            // Strip +ds_index to get DK part
            size_t plus_pos = key.rfind('+');
            std::string dk_part = (plus_pos != std::string::npos) ? key.substr(0, plus_pos) : key;

            if (!first) aggregated_key += ",";
            first = false;

            // Append sliced n_segment (discrete chains use n_segment directly)
            aggregated_key += dk_part + std::to_string(set_F[key].n_segment_right);

            if (set_F[key].n_repeated > 1)
                aggregated_key += ":" + std::to_string(set_F[key].n_repeated);

            all_v_u.insert(all_v_u.end(), set_F[key].v_u.begin(), set_F[key].v_u.end());
        }
        aggregated_key += "]" + monomer_type + "+" + std::to_string(ds_index);

        // Add aggregated block to both sets
        int n_segment_agg = n_segment_threshold - MIN_N_SEGMENT;
        set_F[aggregated_key].monomer_type = monomer_type;
        set_F[aggregated_key].n_segment_right = n_segment_agg;
        set_F[aggregated_key].n_segment_left = n_segment_agg;
        set_F[aggregated_key].v_u = all_v_u;
        set_F[aggregated_key].n_repeated = 1;

        set_I[aggregated_key] = set_F[aggregated_key];

        // Remove merged propagators from working set
        for (const auto& item : set_S)
            set_I.erase(item.first);
    }

    return set_F;
}
