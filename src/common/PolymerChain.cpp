#include <iostream>
#include <cctype>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <stack>
#include <set>

#include "PolymerChain.h"
#include "Exception.h"

//----------------- Constructor ----------------------------
PolymerChain::PolymerChain(
    double ds, std::map<std::string, double> bond_lengths, 
    double volume_fraction, 
    std::vector<std::string> block_monomer_types,
    std::vector<double> contour_lengths,
    std::vector<int> v, std::vector<int> u,
    std::map<int, std::string> chain_end_to_q_init)
{
    // check block size
    if( block_monomer_types.size() != contour_lengths.size())
        throw_with_line_number("The sizes of block_monomer_types (" + std::to_string(block_monomer_types.size()) + 
            ") and contour_lengths (" +std::to_string(contour_lengths.size()) + ") must be consistent.");

    if( block_monomer_types.size() != v.size())
        throw_with_line_number("The sizes of block_monomer_types (" + std::to_string(block_monomer_types.size()) + 
            ") and edges v (" +std::to_string(v.size()) + ") must be consistent.");

    if( block_monomer_types.size() != u.size())
        throw_with_line_number("The sizes of block_monomer_types (" + std::to_string(block_monomer_types.size()) + 
            ") and edges u (" +std::to_string(v.size()) + ") must be consistent.");

    // check the name of monomer_type. Only alphabets and underscore(_) are allowed.
    for(const auto& item : bond_lengths)
    {
        if (!std::all_of(item.first.begin(), item.first.end(), [](unsigned char c){ return std::isalpha(c) || c=='_' ; }))
            throw_with_line_number("\"" + item.first + "\" is an invalid monomer_type name. Only alphabets and underscore(_) are allowed.");
        
        if (item.second <= 0)
            throw_with_line_number("bond_lengths[\"" + item.first + "\"] must be a positive number.");
    }

    // check block lengths, segments, types
    for(size_t i=0; i<contour_lengths.size(); i++)
    {
        if( contour_lengths[i] <= 0)
            throw_with_line_number("contour_lengths[" + std::to_string(i) + "] (" +std::to_string(contour_lengths[i]) + ") must be a positive number.");
        if( std::abs(std::lround(contour_lengths[i]/ds)-contour_lengths[i]/ds) > 1.e-6)
            throw_with_line_number("contour_lengths[" + std::to_string(i) + "]/ds (" + std::to_string(contour_lengths[i]) + "/" + std::to_string(ds) + ") is not an integer.");
        if( bond_lengths.count(block_monomer_types[i]) == 0 )
            throw_with_line_number("block_monomer_types[" + std::to_string(i) + "] (\"" + block_monomer_types[i] + "\") is not in bond_lengths.");
    }

    // // check chain_end_to_q_init
    // if( chain_end_to_q_init.size() > 0)
    //     throw_with_line_number("Currently, \'chain_end_to_q_init\' is not supported.");
    this->chain_end_to_q_init = chain_end_to_q_init;

    // save variables
    try
    {
        this->volume_fraction = volume_fraction;
        for(size_t i=0; i<contour_lengths.size(); i++){
            blocks.push_back({
                block_monomer_types[i],                     // monomer_type
                (int) std::lround(contour_lengths[i]/ds),   // n_segment
                contour_lengths[i],                         // contour_length
                v[i], u[i]});                               // nodes
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }

    // compute alpha, sum of relative contour lengths
    double alpha{0.0};
    for(size_t i=0; i<blocks.size(); i++){
        alpha += blocks[i].contour_length;
    }
    this->alpha = alpha;

    // construct adjacent_nodes
    for(size_t i=0; i<v.size(); i++)
    {
        adjacent_nodes[v[i]].push_back(u[i]);
        adjacent_nodes[u[i]].push_back(v[i]);
        // v and u must be a non-negative integer for depth first search
        if( v[i] < 0 )
            throw_with_line_number("v[" + std::to_string(i) + "] (" + std::to_string(v[i]) + ") must be a non-negative integer.");
        if( u[i] < 0 )
            throw_with_line_number("u[" + std::to_string(i) + "] (" + std::to_string(u[i]) + ") must be a non-negative integer.");
        if( v[i] == u[i] )
            throw_with_line_number("v[" + std::to_string(i) + "] and u[" + std::to_string(i) + "] must be different integers.");
    }

    // // print adjacent_nodes
    // for(const auto& node : adjacent_nodes){
    //     std::cout << node.first << ": [";
    //     for(int i=0; i<node.second.size()-1; i++)
    //         std::cout << node.second[i] << ",";
    //     std::cout << node.second[node.second.size()-1] << "]" << std::endl;
    // }

    // construct edge nodes
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

    // detect a cycle and isolated nodes in the block copolymer graph using depth first search
    std::map<int, bool> is_visited;
    for (size_t i = 0; i < v.size(); i++)
        is_visited[v[i]] = false;
    for (size_t i = 0; i < u.size(); i++)
        is_visited[u[i]] = false;

    std::stack<std::pair<int,int>> connected_nodes;
    // starting node is v[0]
    connected_nodes.push(std::make_pair(v[0],-1));
    // perform depth first search
    while (!connected_nodes.empty())
    {
        // get one item and visit
        int cur = connected_nodes.top().first;
        int parent = connected_nodes.top().second;
        is_visited[cur] = true;

        // remove the item
        connected_nodes.pop();

        // add adjacent_nodes at stack 
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

    // collect isolated nodes
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

    // print isolated nodes
    std::vector<int> isolated_nodes(isolated_nodes_set.begin(), isolated_nodes_set.end());
    if (isolated_nodes.size() > 0)
    {
        std::string error_message = "There is no route from node " + std::to_string(v[0]) + " to nodes: "
            + std::to_string(isolated_nodes[0]);
        for (size_t i=1; i<isolated_nodes.size(); i++)
            error_message += ", " + std::to_string(isolated_nodes[i]);
        throw_with_line_number(error_message + ".");
    }
}
int PolymerChain::get_n_blocks() const
{
    return blocks.size();
}
int PolymerChain::get_n_segment(const int idx) const
{
    return blocks[idx].n_segment;
}
int PolymerChain::get_n_segment_total() const
{
    int total_n{0};
    for(size_t i=0; i<blocks.size(); i++)
        total_n += blocks[i].n_segment;
    return total_n;
}
double PolymerChain::get_alpha() const
{
    return alpha;
}
double PolymerChain::get_volume_fraction() const
{
    return volume_fraction;
}
int PolymerChain::get_block_index_from_edge(const int v, const int u)
{
    if (edge_to_block_index.find(std::make_pair(v,u)) == edge_to_block_index.end())
        throw_with_line_number("There is no such edge (" + std::to_string(v) + ", " + std::to_string(u) + ")."); 
    return edge_to_block_index[std::make_pair(v, u)];
}
struct PolymerChainBlock& PolymerChain::get_block(const int v, const int u)
{
    if (edge_to_block_index.find(std::make_pair(v,u)) == edge_to_block_index.end())
        throw_with_line_number("There is no such edge (" + std::to_string(v) + ", " + std::to_string(u) + ")."); 
    return blocks[edge_to_block_index[std::make_pair(v, u)]];
}
std::vector<PolymerChainBlock>& PolymerChain::get_blocks()
{
    return blocks;
}
std::map<int, std::vector<int>>& PolymerChain::get_adjacent_nodes()
{
    return adjacent_nodes;
}
std::map<std::pair<int, int>, int>& PolymerChain::get_block_indexes()
{
    return edge_to_block_index;
}
void PolymerChain::set_propagator_key(const std::string code, const int v, const int u)
{
    edge_to_propagator_key[std::make_pair(v, u)] = code;
}
std::string PolymerChain::get_propagator_key(const int v, const int u) {
    if (edge_to_propagator_key.find(std::make_pair(v,u)) == edge_to_propagator_key.end())
        throw_with_line_number("There is no such block (v, u): (" + std::to_string(v) + ", " + std::to_string(u) + ")."); 
    return edge_to_propagator_key[std::make_pair(v,u)];
}