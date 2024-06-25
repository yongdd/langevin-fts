#include <iostream>
#include <cctype>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <stack>
#include <set>

#include "Polymer.h"
#include "PropagatorCode.h"
#include "Exception.h"

//----------------- Constructor ----------------------------
Polymer::Polymer(
    int index, 
    double ds, std::map<std::string, double> bond_lengths, 
    double volume_fraction, std::vector<BlockInput> block_inputs,
    std::map<int, std::string> chain_end_to_q_init)
{

    // Check the name of monomer_type. Only alphabets and underscore are allowed.
    for(const auto& item : bond_lengths)
    {
        if (!std::all_of(item.first.begin(), item.first.end(), [](unsigned char c){ return std::isalpha(c) || c=='_' ; }))
            throw_with_line_number("\"" + item.first + "\" is an invalid monomer_type name. Only alphabets and underscore(_) are allowed.");
        
        if (item.second <= 0)
            throw_with_line_number("bond_lengths[\"" + item.first + "\"] must be a positive number.");
    }

    // Check block lengths, segments, types
    for(size_t i=0; i<block_inputs.size(); i++)
    {
        if( block_inputs[i].contour_length <= 0)
            throw_with_line_number("block_inputs[" + std::to_string(i) + "].contour_lengths (" +std::to_string(block_inputs[i].contour_length) + ") must be a positive number.");
        if( std::abs(std::lround(block_inputs[i].contour_length/ds)-block_inputs[i].contour_length/ds) > 1.e-6)
            throw_with_line_number("block_inputs[" + std::to_string(i) + "].contour_lengths/ds (" + std::to_string(block_inputs[i].contour_length) + "/" + std::to_string(ds) + ") is not an integer.");
        if( bond_lengths.count(block_inputs[i].monomer_type) == 0 )
            throw_with_line_number("block_inputs[" + std::to_string(i) + "].monomer_type (\"" + block_inputs[i].monomer_type + "\") is not in bond_lengths.");
    }

    // chain_end_to_q_init
    this->chain_end_to_q_init = chain_end_to_q_init;

    // Save variables
    try
    {
        if(index >= 25)
            throw_with_line_number("In this version, the maximum number of polymer types is 25.");

        std::string str_polymer_idx = generateString(index);
        std::cout<< "str_polymer_idx: " << str_polymer_idx << std::endl;

        this->volume_fraction = volume_fraction;
        for(size_t i=0; i<block_inputs.size(); i++){

            std::string str_block_idx = generateString(i);
            std::cout<< "str_block_idx: " << str_block_idx << std::endl;
            std::string monomer_type = block_inputs[i].monomer_type + str_polymer_idx + str_block_idx;

            blocks.push_back({
                monomer_type,                                           // monomer_type
                (int) std::lround(block_inputs[i].contour_length/ds),   // n_segment
                block_inputs[i].contour_length,                         // contour_length
                block_inputs[i].v,                                      // starting node
                block_inputs[i].u});                                    // ending node
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }

    // Compute alpha, sum of relative contour lengths
    double alpha{0.0};
    for(size_t i=0; i<blocks.size(); i++){
        alpha += blocks[i].contour_length;
    }
    this->alpha = alpha;

    // Construct starting vertices 'v', ending vertices 'u', 
    std::vector<int> v;
    std::vector<int> u;
    for(size_t i=0; i<block_inputs.size(); i++)
    {
        v.push_back(block_inputs[i].v);
        u.push_back(block_inputs[i].u);
    }

    // Construct adjacent_nodes
    for(size_t i=0; i<v.size(); i++)
    {
        adjacent_nodes[v[i]].push_back(u[i]);
        adjacent_nodes[u[i]].push_back(v[i]);
        // V and u must be a non-negative integer for depth first search
        if( v[i] < 0 )
            throw_with_line_number("v[" + std::to_string(i) + "] (" + std::to_string(v[i]) + ") must be a non-negative integer.");
        if( u[i] < 0 )
            throw_with_line_number("u[" + std::to_string(i) + "] (" + std::to_string(u[i]) + ") must be a non-negative integer.");
        if( v[i] == u[i] )
            throw_with_line_number("v[" + std::to_string(i) + "] and u[" + std::to_string(i) + "] must be different integers.");
    }

    // // Print adjacent_nodes
    // for(const auto& node : adjacent_nodes){
    //     std::cout << node.first << ": [";
    //     for(int i=0; i<node.second.size()-1; i++)
    //         std::cout << node.second[i] << ",";
    //     std::cout << node.second[node.second.size()-1] << "]" << std::endl;
    // }

    // Construct edge nodes
    for (size_t i=0; i<v.size(); i++)
    {
        if (edge_to_block_index.count(std::make_pair(v[i], u[i])) > 0)
        {
            throw_with_line_number("There are duplicated edges. Please check the edge between ("
                + std::to_string(v[i]) + ", " + std::to_string(u[i]) + ").");
        }
        else
        {
            edge_to_block_index[std::make_pair(v[i],u[i])] = i;
            edge_to_block_index[std::make_pair(u[i],v[i])] = i;
        }
    }

    // Detect a cycle and isolated nodes in the block copolymer graph using depth first search
    std::map<int, bool> is_visited;
    for (size_t i = 0; i < v.size(); i++)
        is_visited[v[i]] = false;
    for (size_t i = 0; i < u.size(); i++)
        is_visited[u[i]] = false;

    std::stack<std::pair<int,int>> connected_nodes;
    // Starting node is v[0]
    connected_nodes.push(std::make_pair(v[0],-1));
    // Perform depth first search
    while (!connected_nodes.empty())
    {
        // Get one item and visit
        int cur = connected_nodes.top().first;
        int parent = connected_nodes.top().second;
        is_visited[cur] = true;

        // Remove the item
        connected_nodes.pop();

        // Add adjacent_nodes at stack 
        auto nodes = adjacent_nodes[cur];
        for(size_t i=0; i<nodes.size();i++)
        {
            if (is_visited[nodes[i]] && nodes[i] != parent)
            {
                throw_with_line_number("A cycle is detected, which contains nodes " 
                    + std::to_string(nodes[i]) + " and " + std::to_string(parent)
                    + ". Only acyclic branched polymers are allowed.");
            }
            else if(!is_visited[nodes[i]])
            {
                connected_nodes.push(std::make_pair(nodes[i], cur));
            }
        }
    }

    // Collect isolated nodes
    std::set<int> isolated_nodes_set;
    for (size_t i=0; i<v.size(); i++)
    {
        if (!is_visited[v[i]])
            isolated_nodes_set.insert(v[i]);
    }
    for (size_t i=0; i<u.size(); i++)
    {
        if (!is_visited[u[i]])
            isolated_nodes_set.insert(u[i]);
    }

    // Print isolated nodes
    std::vector<int> isolated_nodes(isolated_nodes_set.begin(), isolated_nodes_set.end());
    if (isolated_nodes.size() > 0)
    {
        std::string error_message = "There is no route from node " + std::to_string(v[0]) + " to nodes: "
            + std::to_string(isolated_nodes[0]);
        for (size_t i=1; i<isolated_nodes.size(); i++)
            error_message += ", " + std::to_string(isolated_nodes[i]);
        throw_with_line_number(error_message + ".");
    }

    // Generate propagator codes and assign each of them to each of propagator
    auto propagator_codes = PropagatorCode::generate_codes(*this, chain_end_to_q_init);
    for(size_t i=0; i<propagator_codes.size(); i++)
    {
        int v = std::get<0>(propagator_codes[i]);
        int u = std::get<1>(propagator_codes[i]);
        std::string propagator_key = PropagatorCode::get_key_from_code(std::get<2>(propagator_codes[i]));
        this->set_propagator_key(propagator_key, v, u);
    }

}
int Polymer::get_n_blocks() const
{
    return blocks.size();
}
int Polymer::get_n_segment(const int idx) const
{
    return blocks[idx].n_segment;
}
int Polymer::get_n_segment_total() const
{
    int total_n{0};
    for(size_t i=0; i<blocks.size(); i++)
        total_n += blocks[i].n_segment;
    return total_n;
}
double Polymer::get_alpha() const
{
    return alpha;
}
double Polymer::get_volume_fraction() const
{
    return volume_fraction;
}
int Polymer::get_block_index_from_edge(const int v, const int u)
{
    if (edge_to_block_index.find(std::make_pair(v,u)) == edge_to_block_index.end())
        throw_with_line_number("There is no such edge (" + std::to_string(v) + ", " + std::to_string(u) + ")."); 
    return edge_to_block_index[std::make_pair(v, u)];
}
struct Block& Polymer::get_block(const int v, const int u)
{
    if (edge_to_block_index.find(std::make_pair(v,u)) == edge_to_block_index.end())
        throw_with_line_number("There is no such edge (" + std::to_string(v) + ", " + std::to_string(u) + ")."); 
    return blocks[edge_to_block_index[std::make_pair(v, u)]];
}
std::vector<Block>& Polymer::get_blocks()
{
    return blocks;
}
std::map<int, std::vector<int>>& Polymer::get_adjacent_nodes()
{
    return adjacent_nodes;
}
std::map<std::pair<int, int>, int>& Polymer::get_block_indexes()
{
    return edge_to_block_index;
}
void Polymer::set_propagator_key(const std::string code, const int v, const int u)
{
    edge_to_propagator_key[std::make_pair(v, u)] = code;
}
std::string Polymer::get_propagator_key(const int v, const int u) {
    if (edge_to_propagator_key.find(std::make_pair(v,u)) == edge_to_propagator_key.end())
        throw_with_line_number("There is no such block (v, u): (" + std::to_string(v) + ", " + std::to_string(u) + ")."); 
    return edge_to_propagator_key[std::make_pair(v,u)];
}