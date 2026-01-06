/**
 * @file PropagatorComputationOptimizer.cpp
 * @brief Implementation of propagator computation optimizer.
 *
 * Analyzes polymer architectures to identify equivalent propagator computations
 * that can be shared across chains or within branched structures. This
 * optimization significantly reduces computational cost for complex mixtures.
 *
 * **Optimization Strategy:**
 *
 * The optimizer identifies two types of redundancy:
 *
 * 1. **Intra-polymer aggregation**: Identical branches within a single chain
 *    (e.g., star polymers with identical arms) can share propagator computation
 *
 * 2. **Inter-polymer sharing**: Different polymer species with identical
 *    sub-structures can reuse computed propagators
 *
 * **Data Structures:**
 *
 * - computation_propagators: Map of propagator keys to computation metadata
 * - computation_blocks: Map of (polymer_id, left_key, right_key) to block info
 *
 * **Algorithm Reference:**
 *
 * See *J. Chem. Theory Comput.* **2025**, 21, 3676 for the optimization algorithm.
 *
 * @see PropagatorCode for key generation
 * @see Scheduler for parallel execution scheduling
 */

#include <iostream>
#include <cctype>
#include <cmath>
#include <numbers>
#include <cassert>
#include <algorithm>
#include <stack>
#include <set>

#include "PropagatorComputationOptimizer.h"
#include "PropagatorAggregator.h"
#include "Molecules.h"
#include "Polymer.h"
#include "Exception.h"
#include "ValidationUtils.h"

/**
 * @brief Compare propagator keys for ordered storage.
 *
 * Keys are compared first by height (nesting depth), then lexicographically.
 * This ordering ensures that dependencies are processed before dependents.
 *
 * @param str1 First propagator key
 * @param str2 Second propagator key
 * @return true if str1 should come before str2
 */
bool ComparePropagatorKey::operator()(const std::string& str1, const std::string& str2) const
{
    // First compare heights
    int height_str1 = PropagatorCode::get_height_from_key(str1);
    int height_str2 = PropagatorCode::get_height_from_key(str2);

    if (height_str1 < height_str2)
        return true;
    else if(height_str1 > height_str2)
        return false;

    // Second compare their strings
    return str1 > str2;
}

/**
 * @brief Construct optimizer for given molecular system.
 *
 * Analyzes all polymer species and builds optimized computation schedule.
 * If aggregation is enabled, identifies equivalent propagators.
 *
 * @param molecules                       Molecular system to optimize
 * @param aggregate_propagator_computation Enable branch aggregation
 *
 * @throws Exception if no polymers in the system
 */
PropagatorComputationOptimizer::PropagatorComputationOptimizer(Molecules* molecules, bool aggregate_propagator_computation)
{
    if(molecules->get_n_polymer_types() == 0)
        throw_with_line_number("There is no chain. Add polymers first.");

    // Finalize the contour length mapping (regenerates propagator keys using length indices)
    molecules->finalize_contour_length_mapping();

    // Store pointer to the mapping for converting length_index to n_segment
    this->contour_length_mapping = &molecules->get_contour_length_mapping();

    this->aggregate_propagator_computation = aggregate_propagator_computation;
    this->model_name = molecules->get_model_name();
    for(int p=0; p<molecules->get_n_polymer_types();p++)
    {
        add_polymer(molecules->get_polymer(p), p);
    }
}

/**
 * @brief Add a polymer to the optimization analysis.
 *
 * Processes the polymer's block structure and identifies computation blocks.
 * If aggregation is enabled, merges equivalent branches.
 *
 * **Steps:**
 *
 * 1. Extract all edges and their propagator keys
 * 2. Apply aggregation algorithm (if enabled)
 * 3. Update global computation_blocks and computation_propagators maps
 *
 * @param pc         Polymer to add
 * @param polymer_id Index of polymer in the system
 */
void PropagatorComputationOptimizer::add_polymer(Polymer& pc, int polymer_id)
{
    // Temporary map for the new polymer
    std::map<std::string, std::map<std::string, ComputationBlock>> computation_blocks_new_polymer;
    std::map<std::tuple<int, int>, std::string> v_u_to_right_key;

    // Find computation_blocks in new_polymer
    std::vector<Block> blocks = pc.get_blocks();
    for(size_t b=0; b<blocks.size(); b++)
    {
        int v = blocks[b].v;
        int u = blocks[b].u;
        std::string key_left  = pc.get_propagator_key(v, u);
        std::string key_right = pc.get_propagator_key(u, v);

        if (key_left < key_right){
            key_left.swap(key_right);
            std::swap(v,u);
        }

        computation_blocks_new_polymer[key_left][key_right].monomer_type = blocks[b].monomer_type;
        computation_blocks_new_polymer[key_left][key_right].n_segment_right = blocks[b].n_segment;
        computation_blocks_new_polymer[key_left][key_right].n_segment_left = blocks[b].n_segment;
        computation_blocks_new_polymer[key_left][key_right].v_u.push_back(std::make_tuple(v,u));
        computation_blocks_new_polymer[key_left][key_right].n_repeated = computation_blocks_new_polymer[key_left][key_right].v_u.size();

        v_u_to_right_key[std::make_tuple(v,u)] = key_right;
    }

    // Total segment number
    int total_segment_number = 0;
    if (this->model_name == "continuous")
    {
        for(size_t b=0; b<blocks.size(); b++)
            total_segment_number += blocks[b].n_segment;
    }
    else if (this->model_name == "discrete")
    {
        for(size_t b=0; b<blocks.size(); b++)
        {
            total_segment_number += blocks[b].n_segment-1;
            if(is_junction(pc, blocks[b].v))
                total_segment_number ++;
            if(is_junction(pc, blocks[b].u))
                total_segment_number ++;
        }
    }
    this->total_segment_numbers.push_back(total_segment_number);

    // Aggregation
    if (this->aggregate_propagator_computation)
    {
        // Aggregated keys (initially empty)
        std::map<std::string, std::vector<std::string>> aggregated_blocks;

        // Find aggregated branches in computation_blocks_new_polymer
        for(auto& item : computation_blocks_new_polymer)
        {
            // Left key and right keys
            std::string left_key = item.first;
            std::map<std::string, ComputationBlock> right_keys = item.second;

            // std::cout << "left_key: " << left_key << std::endl;

            // Aggregate propagators for given left key
            std::map<std::string, ComputationBlock> set_I;
            if (model_name == "continuous")
                set_I = PropagatorAggregator::aggregate_continuous_chain(right_keys);
            else if (model_name == "discrete")
                set_I = PropagatorAggregator::aggregate_discrete_chain(right_keys);
            else if (model_name == "")
            {
                throw_with_line_number("Chain model name is not set!")
            }
            else
            {
                throw_with_line_number("Invalid model name: " + model_name + "!")
            }

            // Replace the second map of computation_blocks_new_polymer with 'set_I'
            computation_blocks_new_polymer[left_key] = set_I;

            for(auto& item : set_I)
            {
                //if(item.first[0] == '[')
                if(item.first[0] == '[' && item.second.n_segment_right == item.second.n_segment_left)
                    aggregated_blocks[left_key].push_back(item.first);
            }

            // Remove right keys from other left keys related to aggregated keys, and create new right keys
            substitute_right_keys(
                pc, v_u_to_right_key,
                computation_blocks_new_polymer,
                aggregated_blocks, left_key);
        }
    }

    // Add results to computation_blocks and computation_propagators
    for(const auto& v_item : computation_blocks_new_polymer)
    {
        for(const auto& u_item : v_item.second)
        {
            std::string key_v = v_item.first;
            std::string key_u = u_item.first;

            int n_segment_right = u_item.second.n_segment_right;
            int n_segment_left = u_item.second.n_segment_left;
            int n_repeated = u_item.second.n_repeated;

            // Add blocks
            auto key = std::make_tuple(polymer_id, key_v, key_u);

            computation_blocks[key].monomer_type = PropagatorCode::get_monomer_type_from_key(key_v);
            computation_blocks[key].n_segment_right = n_segment_right;
            computation_blocks[key].n_segment_left = n_segment_left;
            computation_blocks[key].v_u = u_item.second.v_u;
            computation_blocks[key].n_repeated = n_repeated;
            // std::cout << "computation_blocks[key].n_repeated: " << key_v << ", " << key_u  << ", " << computation_blocks[key].n_repeated << std::endl;
            bool is_junction_left = false;
            bool is_junction_right = false;
            if (PropagatorCode::get_height_from_key(key_v) > 0)
                is_junction_left = true;
            if (PropagatorCode::get_height_from_key(key_u) > 0)
                is_junction_right = true;

            // Update propagators
            update_computation_propagator_map(computation_propagators, key_v, n_segment_left,  is_junction_right);
            update_computation_propagator_map(computation_propagators, key_u, n_segment_right, is_junction_left);
        }
    }
}

/** @brief Check if aggregation is enabled. */
bool PropagatorComputationOptimizer::use_aggregation() const
{
    return aggregate_propagator_computation;
}

/**
 * @brief Update right keys after aggregation.
 *
 * When branches are aggregated, the keys of dependent propagators must
 * be updated to reference the new aggregated key.
 *
 * @param pc                              Polymer graph
 * @param v_u_to_right_key               Map from (v,u) to right key
 * @param computation_blocks_new_polymer Block map to update
 * @param aggregated_blocks              Newly aggregated keys
 * @param left_key                       Left key being processed
 */
void PropagatorComputationOptimizer::substitute_right_keys(
    Polymer& pc, 
    std::map<std::tuple<int, int>, std::string>& v_u_to_right_key,
    std::map<std::string, std::map<std::string, ComputationBlock>> & computation_blocks_new_polymer,
    std::map<std::string, std::vector<std::string>>& aggregated_blocks,
    std::string left_key)
{

    for(auto& aggregated_key : aggregated_blocks[left_key])
    {
        auto& computation_block = computation_blocks_new_polymer[left_key][aggregated_key];

        // std::cout << "aggregated_key: " << aggregated_key << std::endl;
        // For each v_u
        for(auto& v_u : computation_block.v_u)
        {
            // (u) ----- (v) ----- (j) 
            //               ----- (k)
            int v = std::get<0>(v_u);
            int u = std::get<1>(v_u);

            auto neighbor_nodes_v = pc.get_adjacent_nodes()[v];
            // Remove 'u' from 'neighbor_nodes_v'
            neighbor_nodes_v.erase(std::remove(neighbor_nodes_v.begin(), neighbor_nodes_v.end(), u), neighbor_nodes_v.end());
            // std::cout << "(v_u): " << v <<  ", " << v << std::endl;

            // For each neighbor_node of v
            for(auto& j : neighbor_nodes_v)
            {
                std::string dep_j = pc.get_propagator_key(j, v);
                // std::cout << dep_j << ", " << pc.get_block(j,v).n_segment << std::endl;

                // Make new key using DKN format for dep_codes (no semicolons)
                // Convert aggregated_key (DK+M format) to code format (DKN)
                // Strip +ds_index from aggregated_key and add n_segment_right
                size_t plus_pos = aggregated_key.rfind('+');
                std::string agg_dk_part = (plus_pos != std::string::npos) ? aggregated_key.substr(0, plus_pos) : aggregated_key;
                std::string agg_code = agg_dk_part + std::to_string(computation_block.n_segment_right);

                std::string new_u_key = "(" + agg_code;
                std::vector<std::string> sub_codes;

                for(auto& k : neighbor_nodes_v)
                {
                    if (k != j)
                    {
                        // Convert propagator key (DK+M) to code format (DKN)
                        std::string prop_key = pc.get_propagator_key(k,v);
                        size_t prop_plus_pos = prop_key.rfind('+');
                        std::string prop_dk_part = (prop_plus_pos != std::string::npos) ? prop_key.substr(0, prop_plus_pos) : prop_key;
                        std::string prop_code = prop_dk_part + std::to_string(pc.get_block(k,v).n_segment);
                        sub_codes.push_back(prop_code);
                    }
                }
                std::sort(sub_codes.begin(),sub_codes.end());
                for(auto& item : sub_codes)
                    new_u_key += item;

                // Get ds_index for the new key (format: DK+M)
                int ds_index = contour_length_mapping->get_ds_index(pc.get_block(j,v).contour_length);
                new_u_key += ")" + pc.get_block(j,v).monomer_type + "+" + std::to_string(ds_index);

                // Remove 'v_u' from 'computation_blocks_new_polymer'
                // std::cout << "v_u_to_right_key[std::make_tuple(j,v)]: " << v_u_to_right_key[std::make_tuple(j,v)] << std::endl; 
                computation_blocks_new_polymer[dep_j].erase(v_u_to_right_key[std::make_tuple(j,v)]);

                // Add new key
                if (computation_blocks_new_polymer[dep_j].find(new_u_key) == computation_blocks_new_polymer[dep_j].end())
                {
                    computation_blocks_new_polymer[dep_j][new_u_key].monomer_type = pc.get_block(j,v).monomer_type;
                    computation_blocks_new_polymer[dep_j][new_u_key].n_segment_right = pc.get_block(j,v).n_segment;
                    computation_blocks_new_polymer[dep_j][new_u_key].n_segment_left = pc.get_block(j,v).n_segment;
                    computation_blocks_new_polymer[dep_j][new_u_key].v_u.push_back(std::make_tuple(j,v));

                    if (aggregated_key[0] == '[')
                        computation_blocks_new_polymer[dep_j][new_u_key].n_repeated = 1;
                    else
                        computation_blocks_new_polymer[dep_j][new_u_key].n_repeated = computation_block.n_repeated;

                    aggregated_blocks[dep_j].push_back(new_u_key);
                }
                else
                {
                    computation_blocks_new_polymer[dep_j][new_u_key].v_u.push_back(std::make_tuple(j,v));
                    int v0 = std::get<1>(computation_blocks_new_polymer[dep_j][new_u_key].v_u[0]);
                    if(v0 == v)
                        computation_blocks_new_polymer[dep_j][new_u_key].n_repeated += computation_block.n_repeated;
                }
                // std::cout << "dep_j, new_u_key, n_segment_right, n_segment_left : " << dep_j << ", " << new_u_key << ", " << n_segment_right << ", " << n_segment_left << std::endl;
            }
        }
    }
}

/**
 * @brief Update computation_propagators map with new key.
 *
 * Adds or updates entry for a propagator key, tracking maximum segment
 * count and junction end positions.
 *
 * @param computation_propagators Map to update
 * @param new_key                 Propagator key
 * @param new_n_segment          Segment count for this usage
 * @param is_junction_end        Whether this ends at a junction
 */
void PropagatorComputationOptimizer::update_computation_propagator_map(
    std::map<std::string, ComputationEdge, ComparePropagatorKey>& computation_propagators,
    std::string new_key, int new_n_segment, bool is_junction_end)
{
    if (computation_propagators.find(new_key) == computation_propagators.end())
    {
        // Parse deps from key
        auto parsed_deps = PropagatorCode::get_deps_from_key(new_key);

        // For non-aggregated keys, the segment values in deps are length_index
        // and need to be converted to n_segment.
        // For aggregated keys (containing '[' anywhere), the segment values are already
        // n_segment from the slicing process, so no conversion is needed.
        // Keys containing '[' include:
        // - Direct aggregated keys like "[A3,B2]C+1"
        // - Mixed keys from substitute_right_keys like "([A3,B2]C5)D+1"
        if (new_key.find('[') != std::string::npos)
        {
            // Aggregated key or key containing aggregated sub-keys:
            // deps already contain n_segment values, but sub_keys need ds_index appended
            // All deps in an aggregated key share the same ds_index as the outer key
            int ds_index = PropagatorCode::get_ds_index_from_key(new_key);
            std::vector<std::tuple<std::string, int, int>> deps_with_ds_index;
            deps_with_ds_index.reserve(parsed_deps.size());
            for (const auto& dep : parsed_deps)
            {
                std::string sub_key = std::get<0>(dep) + "+" + std::to_string(ds_index);
                deps_with_ds_index.push_back(std::make_tuple(sub_key, std::get<1>(dep), std::get<2>(dep)));
            }
            computation_propagators[new_key].deps = deps_with_ds_index;
        }
        else
        {
            // Non-aggregated key: convert length_index to n_segment
            computation_propagators[new_key].deps = convert_deps_to_n_segment(parsed_deps);
        }

        computation_propagators[new_key].monomer_type = PropagatorCode::get_monomer_type_from_key(new_key);
        computation_propagators[new_key].max_n_segment = new_n_segment;
        computation_propagators[new_key].height = PropagatorCode::get_height_from_key(new_key);
    }
    else
    {
        if (computation_propagators[new_key].max_n_segment < new_n_segment)
            computation_propagators[new_key].max_n_segment = new_n_segment;
    }
    if (is_junction_end)
        computation_propagators[new_key].junction_ends.insert(new_n_segment);
}

/**
 * @brief Check if a node is a junction point.
 *
 * Junctions have more than one adjacent node (branching points).
 * Chain ends have exactly one adjacent node.
 *
 * @param pc   Polymer graph
 * @param node Node index to check
 * @return true if node is a junction
 */
bool PropagatorComputationOptimizer::is_junction(Polymer& pc, int node)
{
    if (pc.get_adjacent_nodes()[node].size() == 1)
        return false;
    else
        return true;
}

/**
 * @brief Convert length_index to n_segment in deps tuple and add ds_index to sub_key.
 *
 * The propagator keys use length_index instead of n_segment.
 * This method converts the parsed length_index back to n_segment
 * using the ContourLengthMapping, and also appends ds_index to the sub_key
 * so it matches the format used in computation_propagators.
 *
 * @param deps Parsed dependencies with length_index (sub_key without ds_index)
 * @return Dependencies with n_segment and sub_key including ds_index
 */
std::vector<std::tuple<std::string, int, int>> PropagatorComputationOptimizer::convert_deps_to_n_segment(
    const std::vector<std::tuple<std::string, int, int>>& deps) const
{
    std::vector<std::tuple<std::string, int, int>> converted_deps;
    converted_deps.reserve(deps.size());

    for (const auto& dep : deps)
    {
        std::string sub_key = std::get<0>(dep);
        int length_index = std::get<1>(dep);
        int n_repeated = std::get<2>(dep);

        // Convert length_index to n_segment using the mapping
        // The length_index is 1-based, get the contour_length and then n_segment
        double contour_length = contour_length_mapping->get_length_from_index(length_index);
        int n_segment = contour_length_mapping->get_n_segment(contour_length);

        // Append ds_index to sub_key to match DK+M format
        int ds_index = contour_length_mapping->get_ds_index(contour_length);
        sub_key += "+" + std::to_string(ds_index);

        converted_deps.push_back(std::make_tuple(sub_key, n_segment, n_repeated));
    }

    return converted_deps;
}

/** @brief Get number of unique propagator computations. */
int PropagatorComputationOptimizer::get_n_computation_propagator_codes() const
{
    return computation_propagators.size();
}

/** @brief Get all computation propagators. */
std::map<std::string, ComputationEdge, ComparePropagatorKey>& PropagatorComputationOptimizer::get_computation_propagators()
{
    return computation_propagators;
}

/** @brief Get computation propagator by key. */
ComputationEdge& PropagatorComputationOptimizer::get_computation_propagator(std::string key)
{
    validation::require_string_key(computation_propagators, key, "computation_propagators");
    return computation_propagators[key];
}

/** @brief Get all computation blocks. */
std::map<std::tuple<int, std::string, std::string>, ComputationBlock>& PropagatorComputationOptimizer::get_computation_blocks()
{
    return computation_blocks;
}

/** @brief Get computation block by (polymer_id, left_key, right_key). */
ComputationBlock& PropagatorComputationOptimizer::get_computation_block(std::tuple<int, std::string, std::string> key)
{
    validation::require_key(computation_blocks, key,
        "computation_blocks for (" + std::to_string(std::get<0>(key)) + ", " +
        std::get<1>(key) + ", " + std::get<2>(key) + ")");
    return computation_blocks[key];
}

/**
 * @brief Print computation blocks for debugging.
 *
 * Displays contour length mapping followed by all blocks with their
 * aggregation status, junction flags, segment counts, and (v,u) node pairs.
 */
void PropagatorComputationOptimizer::display_blocks() const
{
    // Print contour length mapping
    if (contour_length_mapping != nullptr && contour_length_mapping->finalized())
    {
        contour_length_mapping->print_mapping();
    }

    // Print blocks
    std::cout << "--------------- Blocks ---------------" << std::endl;
    std::cout << "Polymer id, left key:\n\taggregated, (left, right) is_junction, (left, right) n_segment, right key, n_repeat, {v, u} list" << std::endl;

    const int MAX_PRINT_LENGTH = 500;
    std::tuple<int, std::string> v_tuple = std::make_tuple(-1, "");

    for(const auto& item : computation_blocks)
    {
        // Print polymer id, left key
        const std::string v_string = std::get<1>(item.first);
        if (v_tuple != std::make_tuple(std::get<0>(item.first), v_string))
        {
            std::cout << std::endl << std::to_string(std::get<0>(item.first)) + ", ";
            if (v_string.size() <= MAX_PRINT_LENGTH)
                std::cout << v_string;
            else
                std::cout << v_string.substr(0,MAX_PRINT_LENGTH-5) + " ... <omitted>, " ;
            std::cout << ":" << std::endl;
            v_tuple = std::make_tuple(std::get<0>(item.first), v_string);
        }

        // Print if aggregated
        const std::string u_string = std::get<2>(item.first);
        std::cout << "\t ";
        if (u_string.find('[') == std::string::npos)
            std::cout << "X, ";
        else
            std::cout << "O, ";

        // Print is_free_end (left, right)
        std::cout << "(";
        if (PropagatorCode::get_height_from_key(v_string) > 0)
            std::cout << "O, ";
        else
            std::cout << "X, ";

        if (PropagatorCode::get_height_from_key(u_string) > 0)
            std::cout << "O), ";
        else
            std::cout << "X), ";

        // Print n_segment (left, right)
        std::cout << "(" + std::to_string(item.second.n_segment_left) + ", " + std::to_string(item.second.n_segment_right) + "), ";

        // Print right key
        if (u_string.size() <= MAX_PRINT_LENGTH)
            std::cout << u_string;
        else
            std::cout << u_string.substr(0,MAX_PRINT_LENGTH-5) + " ... <omitted>" ;

        // Print n_repeat
        std::cout << ", " + std::to_string(item.second.n_repeated);        

        // Print v_u list
        for(const auto& v_u : item.second.v_u)
        {
            std::cout << ", {"
            + std::to_string(std::get<0>(v_u)) + ","
            + std::to_string(std::get<1>(v_u)) + "}";
        }
        std::cout << std::endl;
    }
    //std::cout << "------------------------------------" << std::endl;
}

/**
 * @brief Print only the summary statistics (no detailed propagator/block info).
 *
 * Shows total computational steps before and after optimization,
 * and the efficiency gain percentage.
 */
void PropagatorComputationOptimizer::display_statistics() const
{
    int total_mde_steps_without_reduction = 0;
    int reduced_mde_steps = 0;

    for(const auto& n_segment : total_segment_numbers)
    {
        total_mde_steps_without_reduction += 2*n_segment;
    }

    for(const auto& item : computation_propagators)
    {
        if (this->model_name == "continuous")
            reduced_mde_steps += item.second.max_n_segment;
        else if (this->model_name == "discrete")
        {
            reduced_mde_steps += item.second.max_n_segment-1;
            reduced_mde_steps += item.second.junction_ends.size();
            if (item.second.deps.size() > 0)
                reduced_mde_steps++;
        }
    }

    if (this->model_name == "continuous")
        std::cout << "Propagator solver: total MDE steps = " << total_mde_steps_without_reduction;
    else if (this->model_name == "discrete")
        std::cout << "Propagator solver: total integral equation steps = " << total_mde_steps_without_reduction;

    std::cout << ", after optimization = " << reduced_mde_steps;

    double percent = 100*(1.0 - static_cast<double>(reduced_mde_steps)/static_cast<double>(total_mde_steps_without_reduction));
    percent = std::round(percent*100)/100;
    std::cout << ", reduction = " << percent << " %" << std::endl;
}

/**
 * @brief Print propagator list with optimization statistics.
 *
 * Shows each propagator's height, aggregation status, max segment count,
 * and dependencies. Also reports total computational cost reduction.
 */
void PropagatorComputationOptimizer::display_propagators() const
{
    // Print propagators
    std::vector<std::tuple<std::string, int, int>> sub_deps;
    int total_mde_steps_without_reduction = 0;
    int reduced_mde_steps = 0;

    std::cout << "--------------- Propagators ---------------" << std::endl;
    std::cout << "Key:\n\theight, aggregated, max_n_segment, # dependencies, junction_ends" << std::endl;

    for(const auto& item : total_segment_numbers)
    {
        total_mde_steps_without_reduction += 2*item;
    }

    for(const auto& item : computation_propagators)
    {
        if (this->model_name == "continuous")
            reduced_mde_steps += item.second.max_n_segment;
        else if (this->model_name == "discrete")
        {
            reduced_mde_steps += item.second.max_n_segment-1;
            reduced_mde_steps += item.second.junction_ends.size();
            if (item.second.deps.size() > 0)
                reduced_mde_steps++;
        }
        const int MAX_PRINT_LENGTH = 500;

        if (item.first.size() <= MAX_PRINT_LENGTH)
            std::cout << item.first;
        else
            std::cout << item.first.substr(0,MAX_PRINT_LENGTH-5) + " ... <omitted> " ;

        std::cout << ":\n\t ";
        std::cout << item.second.height << ", ";
        if (item.first.find('[') == std::string::npos)
            std::cout << "X, ";
        else
            std::cout << "O, ";

        // Print max_n_segment
        std::cout << item.second.max_n_segment << ", ";

        // Print number of dependency
        std::cout << item.second.deps.size() << ", ";

        // Print indices for junction_ends
        std::cout << "{";
        for (auto it = item.second.junction_ends.begin(); it != item.second.junction_ends.end(); ++it)
        {
            std::cout << *it;
            if (std::next(it) != item.second.junction_ends.end()) {
                std::cout << ", ";
            }
        }
        std::cout << "}, "<< std::endl;
    }
    if (this->model_name == "continuous")
        std::cout << "Total number of modified diffusion equation steps (time complexity) to compute propagators: " << total_mde_steps_without_reduction << std::endl;    
    else if (this->model_name == "discrete")
        std::cout << "(Total number of integral equation steps (time complexity to compute propagators: " << total_mde_steps_without_reduction << std::endl;    
    std::cout << "Total number of steps after optimizing computation : " << reduced_mde_steps << std::endl;

    double percent = 100*(1.0 - static_cast<double>(reduced_mde_steps)/static_cast<double>(total_mde_steps_without_reduction));
    percent = std::round(percent*100)/100; //rounding
    std::cout << "Computational cost reduction (higher is better) : " << percent << " %" << std::endl;
    //std::cout << "------------------------------------" << std::endl;
}

/**
 * @brief Print sub-propagators with dependency information.
 *
 * Alternative display format showing propagator dependencies
 * in a compact format.
 */
void PropagatorComputationOptimizer::display_sub_propagators() const
{
    // Print sub propagators
    int total_segments = 0;
    std::cout << "--------- Propagators ---------" << std::endl;
    std::cout << "Key:\n\taggregated, max_n_segment, height, deps," << std::endl;
    
    for(const auto& item : computation_propagators)
    {
        total_segments += item.second.max_n_segment;

        std::cout << item.first;
        std::cout << ":\n\t ";
        if (item.first.find('[') == std::string::npos)
            std::cout << "X, ";
        else
            std::cout << "O, ";
        std::cout << item.second.max_n_segment << ", " << item.second.height;

        // Use already-converted deps (n_segment) from computation_propagators
        // rather than parsing from key (which contains length_index)
        for(size_t i=0; i<item.second.deps.size(); i++)
        {
            std::cout << ", "  << std::get<0>(item.second.deps[i]) << ":" << std::get<1>(item.second.deps[i]);
        }
        std::cout << std::endl;
    }
    std::cout << "Total number of modified diffusion equation (or integral equation for discrete chain model) steps to compute propagators: " << total_segments << std::endl;
    std::cout << "------------------------------------" << std::endl;
}