#include <iostream>
#include <cctype>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <stack>
#include <set>

#include "Molecules.h"
#include "Exception.h"

// //----------------- Constructor ----------------------------
// Molecules::Molecules(
//     std::string model_name, double ds, std::map<std::string, double> bond_lengths, bool reduce_propagator_computation)
// {
//     // Checking chain model
//     std::transform(model_name.begin(), model_name.end(), model_name.begin(),
//                    [](unsigned char c)
//     {
//         return std::tolower(c);
//     });

//     if (model_name != "continuous" && model_name != "discrete")
//     {
//         throw_with_line_number(model_name + " is an invalid chain model. This must be 'Continuous' or 'Discrete'.");
//     }
//     this->model_name = model_name;

//     // Save variables
//     try
//     {
//         this->ds = ds;
//         this->bond_lengths = bond_lengths;
//         this->reduce_propagator_computation = reduce_propagator_computation;
//     }
//     catch(std::exception& exc)
//     {
//         throw_without_line_number(exc.what());
//     }
// }
// void Molecules::add_polymer(
//     double volume_fraction,
//     std::vector<BlockInput> block_inputs,
//     std::map<int, std::string> chain_end_to_q_init)
// {
//     std::string propagator_code;
//     distinct_polymers.push_back(Polymer(ds, bond_lengths, 
//         volume_fraction, block_inputs, chain_end_to_q_init));

//     Polymer& pc = distinct_polymers.back();

//     // Construct starting vertices 'v', ending vertices 'u', 
//     std::vector<int> v;
//     std::vector<int> u;
//     for(size_t i=0; i<block_inputs.size(); i++)
//     {
//         v.push_back(block_inputs[i].v);
//         u.push_back(block_inputs[i].u);
//     }

//     // Generate propagator code for each block and each direction
//     std::map<std::pair<int, int>, std::pair<std::string, int>> memory;
//     for (int i=0; i<pc.get_n_blocks(); i++)
//     {
//         propagator_code = generate_propagator_code(
//             memory,
//             pc.get_blocks(),
//             pc.get_adjacent_nodes(),
//             pc.get_block_indexes(),
//             chain_end_to_q_init,
//             v[i], u[i]).first;
//         pc.set_propagator_key(propagator_code, v[i], u[i]);

//         propagator_code = generate_propagator_code(
//             memory,
//             pc.get_blocks(),
//             pc.get_adjacent_nodes(),
//             pc.get_block_indexes(),
//             chain_end_to_q_init,
//             u[i], v[i]).first;
//         pc.set_propagator_key(propagator_code, u[i], v[i]);
//     }

//     // Temporary map for the new polymer
//     std::map<std::tuple<int, std::string>, std::map<std::string, EssentialBlock >> essential_blocks_new_polymer;

//     // Find essential_blocks in new_polymer
//     std::vector<Block> blocks = pc.get_blocks();
//     for(size_t b=0; b<blocks.size(); b++)
//     {
//         int v = blocks[b].v;
//         int u = blocks[b].u;
//         std::string dep_v = pc.get_propagator_key(v, u);
//         std::string dep_u = pc.get_propagator_key(u, v);

//         if (dep_v < dep_u){
//             dep_v.swap(dep_u);
//             std::swap(v,u);
//         }

//         auto key1 = std::make_tuple(distinct_polymers.size()-1, dep_v);
//         essential_blocks_new_polymer[key1][dep_u].monomer_type = blocks[b].monomer_type;
//         essential_blocks_new_polymer[key1][dep_u].n_segment_allocated = blocks[b].n_segment;
//         essential_blocks_new_polymer[key1][dep_u].n_segment_offset    = 0;
//         essential_blocks_new_polymer[key1][dep_u].n_segment_original  = blocks[b].n_segment;
//         essential_blocks_new_polymer[key1][dep_u].v_u.push_back(std::make_tuple(v,u));
//     }

//     if (this->reduce_propagator_computation)
//     {
//         // Find superposed branches in essential_blocks_new_polymer
//         std::map<std::tuple<int, std::string>, std::map<std::string, EssentialBlock >> superposed_blocks;
//         for(auto& item : essential_blocks_new_polymer)
//         {
            
//             std::vector<std::tuple<int, int>> total_v_u_list;
//             // Find all (v,u) pairs in superposed_blocks for the given key
//             for(auto& second_key : superposed_blocks[item.first]) // map <tuple, v_u_vector>
//             {
//                 for(auto& superposition_v_u : second_key.second.v_u) 
//                 {
//                     total_v_u_list.push_back(superposition_v_u);
//                     // std::cout << "(v_u): " << std::get<0>(superposition_v_u) <<  ", " << std::get<1>(superposition_v_u) << std::endl;
//                 }
//             }

//             // Remove keys of second map in essential_blocks_new_polymer, which exist in superposed_blocks.
//             for(auto it = item.second.cbegin(); it != item.second.cend();) // map <tuple, v_u_vector>
//             {
//                 bool removed = false;
//                 for(auto& v_u : total_v_u_list)
//                 {
//                     if ( std::find(it->second.v_u.begin(), it->second.v_u.end(), v_u) != it->second.v_u.end())
//                     {
//                         it = item.second.erase(it);
//                         removed = true;
//                         break;
//                     }
//                 }
//                 if (!removed)
//                     ++it;
//             }

//             // After the removal is done, add the superposed branches
//             for(auto& second_key : superposed_blocks[item.first]) 
//                 essential_blocks_new_polymer[item.first][second_key.first] = second_key.second;

//             // Superpose propagators for given key
//             // If the number of elements in the second map is only 1, it will return the map without superposition.
//             // Not all elements of superposed_second_map are superposed.
//             std::map<std::string, EssentialBlock> superposed_second_map;
//             if (model_name == "continuous")
//                 superposed_second_map = superpose_propagator_of_continuous_chain(item.second);
//             else if (model_name == "discrete")
//                 superposed_second_map = superpose_propagator_of_discrete_chain(item.second);

//             // Replace the second map of essential_blocks_new_polymer with superposed_second_map
//             essential_blocks_new_polymer[item.first].clear();
//             for(auto& superposed_propagator_code : superposed_second_map)
//                 essential_blocks_new_polymer[item.first][superposed_propagator_code.first] = superposed_propagator_code.second;

//             // For each superposed_propagator_code
//             for(auto& superposed_propagator_code : superposed_second_map)
//             {
//                 int n_segment_allocated = superposed_propagator_code.second.n_segment_allocated;
//                 std::string dep_key = superposed_propagator_code.first;
//                 int n_segment_offset = superposed_propagator_code.second.n_segment_offset;
//                 int n_segment_original = superposed_propagator_code.second.n_segment_original;

//                 // Skip, if it is not superposed
//                 if ( dep_key[0] != '[' || n_segment_offset+n_segment_allocated != n_segment_original)
//                     continue;

//                 // For each v_u 
//                 for(auto& v_u : superposed_propagator_code.second.v_u)
//                 {
//                     auto& v_adj_nodes = pc.get_adjacent_nodes()[std::get<0>(v_u)];
//                     // For each v_adj_node
//                     for(auto& v_adj_node : v_adj_nodes)
//                     {
//                         if (v_adj_node != std::get<1>(v_u))
//                         {
//                             // std::cout << "(v_u): " << v_adj_node <<  ", " << std::get<0>(v_u) << std::endl;
//                             int v = v_adj_node;
//                             int u = std::get<0>(v_u);

//                             std::string dep_v = pc.get_propagator_key(v, u);
//                             std::string dep_u = pc.get_propagator_key(u, v);
//                             // std::cout << dep_v << ", " << dep_u << ", " << pc.get_block(v,u).n_segment << std::endl;

//                             auto key = std::make_tuple(distinct_polymers.size()-1, dep_v);
//                             // pc.get_block(v,u).monomer_type
//                             // pc.get_block(v,u).n_segment

//                             // Make new key
//                             std::string new_u_key = "(" + dep_key
//                                 + std::to_string(n_segment_allocated);

//                             for(auto& v_adj_node_dep : v_adj_nodes)
//                             {
//                                 if (v_adj_node_dep != v && v_adj_node_dep != std::get<1>(v_u))
//                                     new_u_key += pc.get_block(v_adj_node_dep,u).monomer_type + std::to_string(pc.get_block(v_adj_node_dep,u).n_segment);
                                
//                             }
//                             new_u_key += ")" + pc.get_block(v,u).monomer_type;

//                             // Add the new key
//                             superposed_blocks[key][new_u_key].monomer_type = pc.get_block(v,u).monomer_type;
//                             superposed_blocks[key][new_u_key].n_segment_allocated = pc.get_block(v,u).n_segment;
//                             superposed_blocks[key][new_u_key].n_segment_offset = 0;
//                             superposed_blocks[key][new_u_key].n_segment_original = pc.get_block(v,u).n_segment;
//                             superposed_blocks[key][new_u_key].v_u.push_back(std::make_tuple(v,u));
//                         }
//                     }
//                 }
//             }
//         }
//     }

//     // Add results to essential_blocks and essential_propagator_codes
//     for(const auto& v_item : essential_blocks_new_polymer)
//     {
//         for(const auto& u_item : v_item.second)
//         {
//             int polymer_id = std::get<0>(v_item.first);

//             std::string key_v = std::get<1>(v_item.first);
//             std::string key_u = u_item.first;

//             int n_segment_allocated = u_item.second.n_segment_allocated;
//             int n_segment_offset = u_item.second.n_segment_offset;
//             int n_segment_original = u_item.second.n_segment_original;

//             // Add blocks
//             auto key = std::make_tuple(polymer_id, key_v, key_u);

//             essential_blocks[key].monomer_type = PropagatorCode::get_monomer_type_from_key(key_v);
//             essential_blocks[key].n_segment_allocated = n_segment_allocated;
//             essential_blocks[key].n_segment_offset    = n_segment_offset;
//             essential_blocks[key].n_segment_original  = n_segment_original;
//             essential_blocks[key].v_u = u_item.second.v_u;

//             // Update propagators
//             update_essential_propagator_code(essential_propagator_codes, key_v, n_segment_original);
//             update_essential_propagator_code(essential_propagator_codes, key_u, n_segment_allocated);
//         }
//     }
// }
// std::string Molecules::get_model_name() const
// {
//     return model_name;
// }
// double Molecules::get_ds() const
// {
//     return ds;
// }
// bool Molecules::is_using_superposition() const
// {
//     return reduce_propagator_computation;
// }
// int Molecules::get_n_polymer_types() const
// {
//     return distinct_polymers.size();
// }
// Polymer& Molecules::get_polymer(const int p)
// {
//     return distinct_polymers[p];
// }
// const std::map<std::string, double>& Molecules::get_bond_lengths() const
// {
//     return bond_lengths;
// }