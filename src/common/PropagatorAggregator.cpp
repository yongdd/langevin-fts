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
    // Example)
    // 0, B:
    //   N_offset, N_compute, R, C_R,
    //   6, 6, 1, (C)B,
    //   4, 4, 3, (D)B,
    //   4, 4, 2, (E)B,
    //   2, 2, 1, (F)B,
    //
    //      ↓   Aggregation
    //
    //   6, 6, 1, (C)B,
    //   4, 0, 3, (D)B,
    //   4, 0, 2, (E)B,
    //   4, 4, 1, [(D)B0:3,(E)B0:2]B,
    //   2, 2, 1, (F)B,

    const int minimum_n_segment = 0;

    #ifndef NDEBUG
    std::cout << "--------- PropagatorAggregator::aggregate_continuous_chain (before) -----------" << std::endl;
    std::cout << "--------- map ------------" << std::endl;
    for(const auto& item : set_I)
    {
        std::cout << item.second.n_segment_right << ", " <<
                     item.first << ", " <<
                     item.second.n_segment_left << ", ";
        for(const auto& v_u : item.second.v_u)
        {
            std::cout << "("
            + std::to_string(std::get<0>(v_u)) + ","
            + std::to_string(std::get<1>(v_u)) + ")" + ", ";
        }
        std::cout << std::endl;
    }
    std::cout << "-----------------------" << std::endl;
    #endif

    // Make a set of n_compute
    std::set<int> set_n_compute;
    for(const auto& item: set_I)
        set_n_compute.insert(item.second.n_segment_right);

    // Aggregate right keys
    for(const int n_segment_current: set_n_compute)
    {
        // Add elements into set_S
        std::map<std::string, ComputationBlock, ComparePropagatorKey> set_S;
        for(const auto& item: set_I)
        {
            if (item.second.n_segment_right == n_segment_current)
                set_S[item.first] = item.second;
        }
        std::string monomer_type = PropagatorCode::get_monomer_type_from_key(set_S.begin()->second.monomer_type);

        // Skip if nothing to aggregate
        if (set_S.size() == 1 || n_segment_current < 2*minimum_n_segment)
            continue;

        // Update 'n_segment_right'
        for(const auto& item: set_S)
            set_I[item.first].n_segment_right = minimum_n_segment;

        // New 'n_segment_right' and 'n_segment_left'
        int n_segment_right = n_segment_current - minimum_n_segment;
        int n_segment_left  = n_segment_current - minimum_n_segment;

        // New 'v_u' and propagator key
        std::vector<std::tuple<int ,int>> v_u;
        std::string propagator_code = "[";
        bool is_first_sub_propagator = true;
        std::string dep_key;
        for(auto it = set_S.rbegin(); it != set_S.rend(); it++)
        {
            dep_key = it->first;

            // Update propagator key
            if(!is_first_sub_propagator)
                propagator_code += ",";
            else
                is_first_sub_propagator = false;
            propagator_code += dep_key + std::to_string(set_I[dep_key].n_segment_right);

            // The number of repeats
            if (set_I[dep_key].n_repeated > 1)
                propagator_code += ":" + std::to_string(set_I[dep_key].n_repeated);

            // Compute the union of v_u
            std::vector<std::tuple<int ,int>> dep_v_u = set_I[dep_key].v_u;
            v_u.insert(v_u.end(), dep_v_u.begin(), dep_v_u.end());
        }
        propagator_code += "]" + monomer_type;

        // Add new aggregated key to set_I
        set_I[propagator_code].monomer_type = monomer_type;
        set_I[propagator_code].n_segment_right = n_segment_right;
        set_I[propagator_code].n_segment_left = n_segment_left;
        set_I[propagator_code].v_u = v_u;
        set_I[propagator_code].n_repeated = 1;

        #ifndef NDEBUG
        std::cout << "---------- PropagatorAggregator::aggregate_continuous_chain (in progress) -----------" << std::endl;
        std::cout << "--------- map (" + std::to_string(set_I.size()) + ") -----------" << std::endl;
        for(const auto& item : set_I)
        {
            std::cout << item.second.n_segment_right << ", " <<
                        item.first << ", " <<
                        item.second.n_segment_left << ", ";
            for(const auto& v_u : item.second.v_u)
            {
                std::cout << "("
                + std::to_string(std::get<0>(v_u)) + ","
                + std::to_string(std::get<1>(v_u)) + ")" + ", ";
            }
            std::cout << std::endl;
        }
        std::cout << "-----------------------" << std::endl;
        #endif
    }
    return set_I;
}

std::map<std::string, ComputationBlock> PropagatorAggregator::aggregate_discrete_chain(
    std::map<std::string, ComputationBlock> set_I)
{
    // Example)
    // 0, B:
    //   N_offset, N_compute, R, C_R,
    //   6, 6, 1, (C)B,
    //   4, 4, 3, (D)B,
    //   4, 4, 2, (E)B,
    //   2, 2, 1, (F)B,
    //
    //      ↓   Aggregation
    //
    //   6, 6, 1, (C)B,
    //   4, 1, 3, (D)B,
    //   4, 1, 2, (E)B,
    //   3, 3, 1, [(D)B1:3,(E)B1:2]B,
    //   2, 2, 1, (F)B,

    #ifndef NDEBUG
    std::cout << "--------- PropagatorAggregator::aggregate_discrete_chain (before) -----------" << std::endl;
    std::cout << "--------- map ------------" << std::endl;
    for(const auto& item : set_I)
    {
        std::cout << item.second.n_segment_right << ", " <<
                     item.first << ", " <<
                     item.second.n_segment_left << ", ";
        for(const auto& v_u : item.second.v_u)
        {
            std::cout << "("
            + std::to_string(std::get<0>(v_u)) + ","
            + std::to_string(std::get<1>(v_u)) + ")" + ", ";
        }
        std::cout << std::endl;
    }
    std::cout << "-----------------------" << std::endl;
    #endif

    // Copy
    std::map<std::string, ComputationBlock> set_F = set_I;

    // Minimum 'n_segment_2nd_largest' is 2 because of discrete chain
    for(const auto& item: set_F)
    {
        if (item.second.n_segment_right <= 1)
            set_I.erase(item.first);
    }

    while(set_I.size() > 1)
    {
        int minimum_n_segment = 1;

        // Make a list of 'n_segment_right'
        std::vector<int> vector_n_segment_right;
        for(const auto& item: set_I)
            vector_n_segment_right.push_back(item.second.n_segment_right);

        // Sort the list in ascending order
        std::sort(vector_n_segment_right.rbegin(), vector_n_segment_right.rend());

        // Find the 2nd largest n_segment
        int n_segment_2nd_largest = vector_n_segment_right[1]; // The second largest element.

        // Add elements into set_S
        std::map<std::string, ComputationBlock, ComparePropagatorKey> set_S;
        for(const auto& item: set_I)
        {
            if (item.second.n_segment_right >= n_segment_2nd_largest)
                set_S[item.first] = item.second;
        }
        std::string monomer_type = PropagatorCode::get_monomer_type_from_key(set_S.begin()->second.monomer_type);

        // Update 'n_segment_right'
        for(const auto& item: set_S)
            set_F[item.first].n_segment_right -= n_segment_2nd_largest - minimum_n_segment;

        // New 'n_segment_right' and 'n_segment_left'
        int n_segment_right = n_segment_2nd_largest - minimum_n_segment;
        int n_segment_left  = n_segment_2nd_largest - minimum_n_segment;

        // New 'v_u' and propagator key
        std::vector<std::tuple<int ,int>> v_u;
        std::string propagator_code = "[";
        bool is_first_sub_propagator = true;
        std::string dep_key;
        for(auto it = set_S.rbegin(); it != set_S.rend(); it++)
        {
            dep_key = it->first;

            // Update propagator key
            if(!is_first_sub_propagator)
                propagator_code += ",";
            else
                is_first_sub_propagator = false;
            propagator_code += dep_key + std::to_string(set_F[dep_key].n_segment_right);

            // The number of repeats
            if (set_F[dep_key].n_repeated > 1)
                propagator_code += ":" + std::to_string(set_F[dep_key].n_repeated);

            // Compute the union of v_u
            std::vector<std::tuple<int ,int>> dep_v_u = set_F[dep_key].v_u;
            v_u.insert(v_u.end(), dep_v_u.begin(), dep_v_u.end());
        }
        propagator_code += "]" + monomer_type;

        // Add new aggregated key to set_F
        set_F[propagator_code].monomer_type = monomer_type;
        set_F[propagator_code].n_segment_right = n_segment_right;
        set_F[propagator_code].n_segment_left = n_segment_left;
        set_F[propagator_code].v_u = v_u;
        set_F[propagator_code].n_repeated = 1;

        // Add new aggregated key to set_I
        set_I[propagator_code] = set_F[propagator_code];

        // set_I = set_I - set_S
        for(const auto& item: set_S)
            set_I.erase(item.first);

        #ifndef NDEBUG
        std::cout << "---------- PropagatorAggregator::aggregate_discrete_chain (in progress) -----------" << std::endl;
        std::cout << "--------- map (" + std::to_string(set_I.size()) + ") -----------" << std::endl;
        for(const auto& item : set_I)
        {
            std::cout << item.second.n_segment_right << ", " <<
                        item.first << ", " <<
                        item.second.n_segment_left << ", ";
            for(const auto& v_u : item.second.v_u)
            {
                std::cout << "("
                + std::to_string(std::get<0>(v_u)) + ","
                + std::to_string(std::get<1>(v_u)) + ")" + ", ";
            }
            std::cout << std::endl;
        }
        std::cout << "-----------------------" << std::endl;
        #endif
    }

    return set_F;
}
