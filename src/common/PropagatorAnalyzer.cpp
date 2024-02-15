#include <iostream>
#include <cctype>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <stack>
#include <set>

#include "PropagatorAnalyzer.h"
#include "Molecules.h"
#include "Polymer.h"
#include "Exception.h"

bool ComparePropagatorKey::operator()(const std::string& str1, const std::string& str2)
{
    // First compare heights
    int height_str1 = PropagatorCode::get_height_from_key(str1);
    int height_str2 = PropagatorCode::get_height_from_key(str2);

    if (height_str1 < height_str2)
        return true;
    else if(height_str1 > height_str2)
        return false;

    // Second compare their strings
    return str1 < str2;
}

PropagatorAnalyzer::PropagatorAnalyzer(Molecules* molecules, bool aggregate_propagator_computation)
{
    if(molecules->get_n_polymer_types() == 0)
        throw_with_line_number("There is no chain. Add polymers first.");

    this->aggregate_propagator_computation = aggregate_propagator_computation;
    this->model_name = molecules->get_model_name();
    for(int p=0; p<molecules->get_n_polymer_types();p++)
    {
        add_polymer(molecules->get_polymer(p), p);
    }
}
void PropagatorAnalyzer::add_polymer(Polymer& pc, int polymer_count)
{
    // Temporary map for the new polymer
    std::map<std::tuple<int, std::string>, std::map<std::string, ComputationBlock >> computation_blocks_new_polymer;

    // Find computation_blocks in new_polymer
    std::vector<Block> blocks = pc.get_blocks();
    for(size_t b=0; b<blocks.size(); b++)
    {
        int v = blocks[b].v;
        int u = blocks[b].u;
        std::string dep_v = pc.get_propagator_key(v, u);
        std::string dep_u = pc.get_propagator_key(u, v);

        if (dep_v < dep_u){
            dep_v.swap(dep_u);
            std::swap(v,u);
        }

        auto key1 = std::make_tuple(polymer_count, dep_v);
        computation_blocks_new_polymer[key1][dep_u].monomer_type = blocks[b].monomer_type;
        computation_blocks_new_polymer[key1][dep_u].n_segment_compute = blocks[b].n_segment;
        computation_blocks_new_polymer[key1][dep_u].n_segment_offset = blocks[b].n_segment;
        computation_blocks_new_polymer[key1][dep_u].v_u.push_back(std::make_tuple(v,u));
    }

    if (this->aggregate_propagator_computation)
    {
        // Find aggregated branches in computation_blocks_new_polymer
        std::map<std::tuple<int, std::string>, std::map<std::string, ComputationBlock >> aggregated_blocks;
        for(auto& item : computation_blocks_new_polymer)
        {
            
            std::vector<std::tuple<int, int>> total_v_u_list;
            // Find all (v,u) pairs in aggregated_blocks for the given key
            for(auto& second_key : aggregated_blocks[item.first]) // map <tuple, v_u_vector>
            {
                for(auto& aggregation_v_u : second_key.second.v_u) 
                {
                    total_v_u_list.push_back(aggregation_v_u);
                    // std::cout << "(v_u): " << std::get<0>(aggregation_v_u) <<  ", " << std::get<1>(aggregation_v_u) << std::endl;
                }
            }

            // Remove keys of second map in computation_blocks_new_polymer, which exist in aggregated_blocks.
            for(auto it = item.second.cbegin(); it != item.second.cend();) // map <tuple, v_u_vector>
            {
                bool removed = false;
                for(auto& v_u : total_v_u_list)
                {
                    if ( std::find(it->second.v_u.begin(), it->second.v_u.end(), v_u) != it->second.v_u.end())
                    {
                        it = item.second.erase(it);
                        removed = true;
                        break;
                    }
                }
                if (!removed)
                    ++it;
            }

            // After the removal is done, add the aggregated branches
            for(auto& second_key : aggregated_blocks[item.first]) 
                computation_blocks_new_polymer[item.first][second_key.first] = second_key.second;

            // Aggregate propagators for given key
            // If the number of elements in the second map is only 1, it will return the map without aggregation.
            // Not all elements of aggregated_second_map are aggregated.
            std::map<std::string, ComputationBlock> aggregated_second_map;
            if (model_name == "continuous")
                aggregated_second_map = PropagatorAnalyzer::aggregate_propagator_continuous_chain(item.second);
            else if (model_name == "discrete")
                aggregated_second_map = PropagatorAnalyzer::aggregate_propagator_discrete_chain(item.second);
            else if (model_name == "")
                std::cout << "Chain model name is not set!" << std::endl;
            else
                std::cout << "Invalid model name: " << model_name << "!" << std::endl;

            // Replace the second map of computation_blocks_new_polymer with aggregated_second_map
            computation_blocks_new_polymer[item.first].clear();
            for(auto& aggregated_propagator_code : aggregated_second_map)
                computation_blocks_new_polymer[item.first][aggregated_propagator_code.first] = aggregated_propagator_code.second;

            // For each aggregated_propagator_code
            for(auto& aggregated_propagator_code : aggregated_second_map)
            {
                int n_segment_compute = aggregated_propagator_code.second.n_segment_compute;
                std::string dep_key = aggregated_propagator_code.first;
                int n_segment_offset = aggregated_propagator_code.second.n_segment_offset;

                // Skip, if it is not aggregated
                if ( dep_key[0] != '[' || n_segment_compute != n_segment_offset)
                    continue;

                // For each v_u 
                for(auto& v_u : aggregated_propagator_code.second.v_u)
                {
                    auto& v_adj_nodes = pc.get_adjacent_nodes()[std::get<0>(v_u)];
                    // For each v_adj_node
                    for(auto& v_adj_node : v_adj_nodes)
                    {
                        if (v_adj_node != std::get<1>(v_u))
                        {
                            // std::cout << "(v_u): " << v_adj_node <<  ", " << std::get<0>(v_u) << std::endl;
                            int v = v_adj_node;
                            int u = std::get<0>(v_u);

                            std::string dep_v = pc.get_propagator_key(v, u);
                            std::string dep_u = pc.get_propagator_key(u, v);
                            // std::cout << dep_v << ", " << dep_u << ", " << pc.get_block(v,u).n_segment << std::endl;

                            auto key = std::make_tuple(polymer_count, dep_v);
                            // pc.get_block(v,u).monomer_type
                            // pc.get_block(v,u).n_segment

                            // Make new key
                            std::string new_u_key = "(" + dep_key
                                + std::to_string(n_segment_compute);

                            for(auto& v_adj_node_dep : v_adj_nodes)
                            {
                                if (v_adj_node_dep != v && v_adj_node_dep != std::get<1>(v_u))
                                    new_u_key += pc.get_block(v_adj_node_dep,u).monomer_type + std::to_string(pc.get_block(v_adj_node_dep,u).n_segment);
                                
                            }
                            new_u_key += ")" + pc.get_block(v,u).monomer_type;

                            // Add the new key
                            aggregated_blocks[key][new_u_key].monomer_type = pc.get_block(v,u).monomer_type;
                            aggregated_blocks[key][new_u_key].n_segment_compute = pc.get_block(v,u).n_segment;
                            aggregated_blocks[key][new_u_key].n_segment_offset = pc.get_block(v,u).n_segment;
                            aggregated_blocks[key][new_u_key].v_u.push_back(std::make_tuple(v,u));
                        }
                    }
                }
            }
        }
    }

    // Add results to computation_blocks and computation_propagator_codes
    for(const auto& v_item : computation_blocks_new_polymer)
    {
        for(const auto& u_item : v_item.second)
        {
            int polymer_id = std::get<0>(v_item.first);

            std::string key_v = std::get<1>(v_item.first);
            std::string key_u = u_item.first;

            int n_segment_compute = u_item.second.n_segment_compute;
            int n_segment_offset = u_item.second.n_segment_offset;

            // Add blocks
            auto key = std::make_tuple(polymer_id, key_v, key_u);

            computation_blocks[key].monomer_type = PropagatorCode::get_monomer_type_from_key(key_v);
            computation_blocks[key].n_segment_compute = n_segment_compute;
            computation_blocks[key].n_segment_offset = n_segment_offset;
            computation_blocks[key].v_u = u_item.second.v_u;

            // Update propagators
            update_computation_propagator_map(computation_propagator_codes, key_v, n_segment_offset);
            update_computation_propagator_map(computation_propagator_codes, key_u, n_segment_compute);
        }
    }
}
std::map<std::string, ComputationBlock> PropagatorAnalyzer::aggregate_propagator_continuous_chain(std::map<std::string, ComputationBlock> not_aggregated_yet_second_map)
{
    // Example)
    // 0, B:
    //   6, 0, 6, (C)B, 1,
    //   4, 0, 4, (D)B, 3,
    //   4, 0, 4, (E)B, 2,
    //   2, 0, 2, (F)B, 1,
    //
    //      ↓   Aggregation
    //  
    //   6, 0, 2, (C)B, 1,  // done
    //   4, 0, 0, (D)B, 3,  // done
    //   4, 0, 0, (E)B, 2,  // done
    //   2, 0, 2, (F)B, 1,
    //   6, 2, 4, [(C)B2:1,(D)B0:3,(E)B0:2]B,
    //
    //      ↓   Aggregation
    //  
    //   6, 0, 2, (C)B, 1,  // done
    //   4, 0, 0, (D)B, 3,  // done
    //   4, 0, 0, (E)B, 2,  // done
    //   2, 0, 0, (F)B, 1,  // done
    //   6, 2, 2, [(C)B2:1,(D)B0:3,(E)B0:2]B,             // done
    //   6, 4, 2, [[(C)B2:1,(D)B0:3,(E)B0:2]B2,(F)B2:1]B  // done

    std::map<std::string, ComputationBlock> remaining_keys;
    std::map<std::string, ComputationBlock> aggregated_second_map;
    std::map<std::string, ComputationBlock> aggregated_second_map_total;

    // Because of our SimpsonRule implementation, whose weights of odd number n_segments and even number n_segments are slightly different,
    // aggregations for blocks of odd number and of even number are separately performed.

    // For even number
    for(const auto& item : not_aggregated_yet_second_map)
    {
        if (item.second.n_segment_compute % 2 == 0)
            remaining_keys[item.first] = item.second;
    }
    aggregated_second_map_total = aggregate_propagator_common(remaining_keys, 0);

    // For odd number
    remaining_keys.clear();
    for(const auto& item : not_aggregated_yet_second_map)
    {
        if (item.second.n_segment_compute % 2 == 1)
            remaining_keys[item.first] = item.second;
    }
    aggregated_second_map = aggregate_propagator_common(remaining_keys, 0);

    // Merge maps
    aggregated_second_map_total.insert(std::begin(aggregated_second_map), std::end(aggregated_second_map));

    return aggregated_second_map_total;

}
std::map<std::string, ComputationBlock> PropagatorAnalyzer::aggregate_propagator_discrete_chain(std::map<std::string, ComputationBlock> not_aggregated_yet_second_map)
{

    // Example)
    // 0, B:
    //   6, 0, 6, (C)B, 1,
    //   4, 0, 4, (D)B, 3,
    //   4, 0, 4, (E)B, 2,
    //   2, 0, 2, (F)B, 1,
    //
    //      ↓   Aggregation
    //  
    //   6, 0, 3, (C)B, 1,  // done
    //   4, 0, 1, (D)B, 3,  // done
    //   4, 0, 1, (E)B, 2,  // done
    //   2, 0, 2, (F)B, 1,
    //   6, 3, 3, [(C)B3:1,(D)B1:3,(E)B1:2]B,
    //
    //      ↓   Aggregation
    //  
    //   6, 0, 3, (C)B, 1,  // done
    //   4, 0, 1, (D)B, 3,  // done
    //   4, 0, 1, (E)B, 2,  // done
    //   2, 0, 1, (F)B, 1,  // done
    //   6, 3, 2, [(C)B3:1,(D)B1:3,(E)B1:2]B,             // done
    //   6, 5, 1, [[(C)B3:1,(D)B1:3,(E)B1:2]B2,(F)B1:1]B  // done

    // std::map<std::string, ComputationBlock> remaining_keys;
    // for(const auto& item : not_aggregated_yet_second_map)
    // {
    //     remaining_keys[item.first] = item.second;
    // }

    return aggregate_propagator_common(not_aggregated_yet_second_map, 1);
}

std::map<std::string, ComputationBlock> PropagatorAnalyzer::aggregate_propagator_common(std::map<std::string, ComputationBlock> remaining_keys, int minimum_n_segment)
{
    int current_n_segment;
    int n_segment_compute;
    int n_segment_offset;

    std::string aggregated_propagator_code;
    std::vector<std::tuple<int ,int>> v_u_total;

    std::map<std::string, ComputationBlock> aggregated_second_map;
    // Tuple <n_segment_compute, key, n_segment_offset, v_u_list>
    std::vector<std::tuple<int, std::string, int, std::vector<std::tuple<int ,int>>>> same_aggregation_level_list;

    // std::cout << "---------map------------" << std::endl;
    // for(const auto& item : remaining_keys)
    // {
    //     std::cout << item.second.n_segment_compute << ", " <<
    //                  item.first << ", " <<
    //                  item.second.n_segment_offset << ", ";
    //     for(const auto& v_u : item.second.v_u)
    //     {
    //         std::cout << "("
    //         + std::to_string(std::get<0>(v_u)) + ","
    //         + std::to_string(std::get<1>(v_u)) + ")" + ", ";
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << "-----------------------" << std::endl;

    // int count = 0;
    current_n_segment = 0;
    for(const auto& item: remaining_keys)
        current_n_segment = std::max(current_n_segment, item.second.n_segment_compute);
    while(!remaining_keys.empty())
    {
        // Count ++;
        // If (count == 10)
        //     break;
        // std::cout << "------remaining_keys------" << std::endl;
        // for(const auto& item: remaining_keys)
        // {
        //     std::cout << item.second.n_segment_compute << ", "  
        //               << item.first << ", " 
        //               << item.second.n_segment_offset << ", " << std::endl;
        // }
        // std::cout << "-------------" << std::endl;

        // Find keys that have the same aggregation level from remaining_keys
        std::set<int, std::greater<int>> n_segment_set; // for finding the largest n_segment that is not in same_aggregation_level_list.
        for (auto it = remaining_keys.cbegin(); it != remaining_keys.cend(); )
        {
            bool erased = false;
            if (it->second.n_segment_compute <= 1)
            {
                aggregated_second_map[it->first] = it->second;
                remaining_keys.erase(it++);
                erased = true;
            }

            if (!erased)
            {
                if (current_n_segment <= it->second.n_segment_compute)
                    same_aggregation_level_list.push_back(std::make_tuple(
                        it->second.n_segment_compute,
                        it->first,
                        it->second.n_segment_offset,
                        it->second.v_u));
                else
                    n_segment_set.insert(it->second.n_segment_compute);
                ++it;
            }
        }
        // If it empty, decrease current_n_segment.
        if (same_aggregation_level_list.empty())
            current_n_segment = *std::next(n_segment_set.begin(), 0);
        else
        {
            // std::cout << "------same_aggregation_level_list------" << std::endl;
            // for(size_t i=0; i<same_aggregation_level_list.size(); i++)
            //     std::cout << std::get<0>(same_aggregation_level_list[i]) << ", " << std::get<1>(same_aggregation_level_list[i]) << std::endl;
            // std::cout << "------------------------------------" << std::endl;

            v_u_total.clear();
            // If there is only one element
            if (same_aggregation_level_list.size() == 1)
            {
                //  No the second largest element
                if (n_segment_set.size() == 0)
                {
                    // Add to map
                    aggregated_second_map[std::get<1>(same_aggregation_level_list[0])].monomer_type = PropagatorCode::get_monomer_type_from_key(std::get<1>(same_aggregation_level_list[0]));
                    aggregated_second_map[std::get<1>(same_aggregation_level_list[0])].n_segment_compute = std::get<0>(same_aggregation_level_list[0]);
                    aggregated_second_map[std::get<1>(same_aggregation_level_list[0])].n_segment_offset  = std::get<2>(same_aggregation_level_list[0]);
                    aggregated_second_map[std::get<1>(same_aggregation_level_list[0])].v_u                 = std::get<3>(same_aggregation_level_list[0]);

                    // Erase element
                    remaining_keys.erase(std::get<1>(same_aggregation_level_list[0]));
                }
                // Lower 'current_n_segment' to the next level and repeat
                else
                    current_n_segment = *std::next(n_segment_set.begin(), 0);
            }
            // Superposition
            else
            {
                // Sort same_aggregation_level_list with height in descending order
                std::sort(same_aggregation_level_list.begin(), same_aggregation_level_list.end(),
                    [](auto const &t1, auto const &t2)
                        {
                            return PropagatorCode::get_height_from_key(std::get<1>(t1)) > PropagatorCode::get_height_from_key(std::get<1>(t2));
                        }
                );

                // Add one by one
                std::string dep_key;
                std::vector<std::tuple<int ,int>> dep_v_u ;
                
                for(size_t i=0; i<same_aggregation_level_list.size(); i++)
                {
                    n_segment_compute = std::get<0>(same_aggregation_level_list[i]) - current_n_segment + minimum_n_segment;

                    dep_key = std::get<1>(same_aggregation_level_list[i]);
                    n_segment_offset = std::get<2>(same_aggregation_level_list[i]);
                    dep_v_u = std::get<3>(same_aggregation_level_list[i]);

                    v_u_total.insert(v_u_total.end(), dep_v_u.begin(), dep_v_u.end());
                    if (i==0)
                        aggregated_propagator_code = "[" + dep_key + std::to_string(n_segment_compute);
                    else
                        aggregated_propagator_code += "," + dep_key + std::to_string(n_segment_compute);

                    if (dep_key.find('[') == std::string::npos)
                        aggregated_propagator_code += ":" + std::to_string(dep_v_u.size());

                    // Add to map
                    aggregated_second_map[std::get<1>(same_aggregation_level_list[i])].monomer_type = PropagatorCode::get_monomer_type_from_key(dep_key);
                    aggregated_second_map[std::get<1>(same_aggregation_level_list[i])].n_segment_compute = n_segment_compute;
                    aggregated_second_map[std::get<1>(same_aggregation_level_list[i])].n_segment_offset  = n_segment_offset;
                    aggregated_second_map[std::get<1>(same_aggregation_level_list[i])].v_u                 = dep_v_u;
                }
                aggregated_propagator_code += "]" + PropagatorCode::get_monomer_type_from_key(dep_key);
                n_segment_compute = current_n_segment - minimum_n_segment;

                // Add to remaining_keys
                remaining_keys[aggregated_propagator_code].monomer_type = PropagatorCode::get_monomer_type_from_key(aggregated_propagator_code);
                remaining_keys[aggregated_propagator_code].n_segment_compute = n_segment_compute;
                remaining_keys[aggregated_propagator_code].n_segment_offset  = n_segment_compute;
                remaining_keys[aggregated_propagator_code].v_u                 = v_u_total;

                // Erase elements
                for(size_t i=0; i<same_aggregation_level_list.size(); i++)
                    remaining_keys.erase(std::get<1>(same_aggregation_level_list[i]));
            }
            same_aggregation_level_list.clear();
        }
    }
    return aggregated_second_map;
}
bool PropagatorAnalyzer::is_aggregated() const
{
    return aggregate_propagator_computation;
}
void PropagatorAnalyzer::update_computation_propagator_map(std::map<std::string, ComputationEdge, ComparePropagatorKey>& computation_propagator_codes, std::string new_key, int new_n_segment)
{
    if (computation_propagator_codes.find(new_key) == computation_propagator_codes.end())
    {
        computation_propagator_codes[new_key].deps = PropagatorCode::get_deps_from_key(new_key);
        computation_propagator_codes[new_key].monomer_type = PropagatorCode::get_monomer_type_from_key(new_key);
        computation_propagator_codes[new_key].max_n_segment = new_n_segment;
        computation_propagator_codes[new_key].height = PropagatorCode::get_height_from_key(new_key);
    }
    else
    {
        if (computation_propagator_codes[new_key].max_n_segment < new_n_segment)
            computation_propagator_codes[new_key].max_n_segment = new_n_segment;
    }
}
int PropagatorAnalyzer::get_n_computation_propagator_codes() const
{
    return computation_propagator_codes.size();
}
std::map<std::string, ComputationEdge, ComparePropagatorKey>& PropagatorAnalyzer::get_computation_propagator_codes()
{
    return computation_propagator_codes;
}
ComputationEdge& PropagatorAnalyzer::get_computation_propagator_code(std::string key)
{
    if (computation_propagator_codes.find(key) == computation_propagator_codes.end())
        throw_with_line_number("There is no such key (" + key + ").");

    return computation_propagator_codes[key];
}
std::map<std::tuple<int, std::string, std::string>, ComputationBlock>& PropagatorAnalyzer::get_computation_blocks()
{
    return computation_blocks;
}
ComputationBlock& PropagatorAnalyzer::get_computation_block(std::tuple<int, std::string, std::string> key)
{
    if (computation_blocks.find(key) == computation_blocks.end())
        throw_with_line_number("There is no such key (" + std::to_string(std::get<0>(key)) + ", " + 
            std::get<1>(key) + ", " + std::get<2>(key) + ").");

    return computation_blocks[key];
}
void PropagatorAnalyzer::display_blocks() const
{
    // Print blocks
    std::cout << "--------- Blocks ---------" << std::endl;
    std::cout << "Polymer id, key1:\n\taggregated, n_segment (offset, compute), key2, {v, u} list" << std::endl;

    const int MAX_PRINT_LENGTH = 500;
    std::tuple<int, std::string> v_tuple = std::make_tuple(-1, "");

    for(const auto& item : computation_blocks)
    {
        // Print polymer id, key1
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
        // Print n_segment (offset, compute)
        std::cout << "(" + std::to_string(item.second.n_segment_offset) + ", " + std::to_string(item.second.n_segment_compute) + "), ";

        // Print key2
        if (u_string.size() <= MAX_PRINT_LENGTH)
            std::cout << u_string;
        else
            std::cout << u_string.substr(0,MAX_PRINT_LENGTH-5) + " ... <omitted>" ;

        // Print v_u list
        for(const auto& v_u : item.second.v_u)
        {
            std::cout << ", {"
            + std::to_string(std::get<0>(v_u)) + ","
            + std::to_string(std::get<1>(v_u)) + "}";
        }
        std::cout << std::endl;
    }
    std::cout << "------------------------------------" << std::endl;
}
void PropagatorAnalyzer::display_propagators() const
{
    // Print propagators
    std::vector<std::tuple<std::string, int, int>> sub_deps;
    int total_segments = 0;

    std::cout << "--------- Propagators ---------" << std::endl;
    std::cout << "Key:\n\taggregated, max_n_segment, height" << std::endl;
    
    for(const auto& item : computation_propagator_codes)
    {
        total_segments += item.second.max_n_segment;

        const int MAX_PRINT_LENGTH = 500;

        if (item.first.size() <= MAX_PRINT_LENGTH)
            std::cout << item.first;
        else
            std::cout << item.first.substr(0,MAX_PRINT_LENGTH-5) + " ... <omitted> " ;

        std::cout << ":\n\t ";
        if (item.first.find('[') == std::string::npos)
            std::cout << "X, ";
        else
            std::cout << "O, ";
        std::cout << item.second.max_n_segment << ", " << item.second.height << std::endl;
    }
    std::cout << "Total number of modified diffusion equation (or integral equation for discrete chain model) steps to compute propagators: " << total_segments << std::endl;
    std::cout << "------------------------------------" << std::endl;
}

void PropagatorAnalyzer::display_sub_propagators() const
{
    // Print sub propagators
    std::vector<std::tuple<std::string, int, int>> sub_deps;
    int total_segments = 0;
    std::cout << "--------- Propagators ---------" << std::endl;
    std::cout << "Key:\n\taggregated, max_n_segment, height, deps," << std::endl;
    
    for(const auto& item : computation_propagator_codes)
    {
        total_segments += item.second.max_n_segment;

        std::cout << item.first;
        std::cout << ":\n\t ";
        if (item.first.find('[') == std::string::npos)
            std::cout << "X, ";
        else
            std::cout << "O, ";
        std::cout << item.second.max_n_segment << ", " << item.second.height;

        sub_deps = PropagatorCode::get_deps_from_key(item.first);
        for(size_t i=0; i<sub_deps.size(); i++)
        {
            std::cout << ", "  << std::get<0>(sub_deps[i]) << ":" << std::get<1>(sub_deps[i]);
        }
        std::cout << std::endl;
    }
    std::cout << "Total number of modified diffusion equation (or integral equation for discrete chain model) steps to compute propagators: " << total_segments << std::endl;
    std::cout << "------------------------------------" << std::endl;
}