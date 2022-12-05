#include <iostream>
#include <cmath>
#include <algorithm>
#include <stack>

#include "BranchedPolymerChain.h"
#include "Exception.h"

//----------------- Constructor ----------------------------
BranchedPolymerChain::BranchedPolymerChain(
    std::string model_name, double ds, std::map<std::string, double> dict_segment_lengths,
    std::vector<std::string> block_types, std::vector<double> block_lengths, std::vector<int> v, std::vector<int> u)
{
    // checking chain model
    std::transform(model_name.begin(), model_name.end(), model_name.begin(),
                   [](unsigned char c)
    {
        return std::tolower(c);
    });

    if (model_name != "continuous" && model_name != "discrete")
    {
        throw_with_line_number(model_name + " is an invalid chain model. This must be 'Continuous' or 'Discrete'.");
    }
    this->model_name = model_name;

    // check block size
    if( block_types.size() != block_lengths.size())
        throw_with_line_number("The sizes of block_types (" + std::to_string(block_types.size()) + 
            ") and block_lengths (" +std::to_string(block_lengths.size()) + ") must be consistent.");

    if( block_types.size() != v.size())
        throw_with_line_number("The sizes of block_types (" + std::to_string(block_types.size()) + 
            ") and edges v (" +std::to_string(v.size()) + ") must be consistent.");

    if( block_types.size() != u.size())
        throw_with_line_number("The sizes of block_types (" + std::to_string(block_types.size()) + 
            ") and edges u (" +std::to_string(v.size()) + ") must be consistent.");

    // check block lengths, segments, types
    for(int i=0; i<block_lengths.size(); i++)
    {
        if( block_lengths[i] <= 0)
            throw_with_line_number("block_lengths[" + std::to_string(i) + "] (" +std::to_string(block_lengths[i]) + ") must be a positive number.");
        if( std::abs(std::lround(block_lengths[i]/ds)-block_lengths[i]/ds) > 1.e-6)
            throw_with_line_number("block_lengths[" + std::to_string(i) + "]/ds (" + std::to_string(block_lengths[i]) + "/" + std::to_string(ds) + ") is not an integer.");
        if( dict_segment_lengths.count(block_types[i]) == 0 )
            throw_with_line_number("block_types[" + std::to_string(i) + "] (\"" + block_types[i] + "\") is not in dict_segment_lengths.");

        this->bond_length_sq.push_back(dict_segment_lengths[block_types[i]]*dict_segment_lengths[block_types[i]]);
        this->n_segments.push_back(std::lround(block_lengths[i]/ds));
    }

    // construct adjacent_nodes
    for(int i=0; i<block_lengths.size(); i++){
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

    // print adjacent_nodes
    for(const auto& node : adjacent_nodes){
        std::cout << node.first << ": [";
        for(int i=0; i<node.second.size()-1; i++){ 
            std::cout << node.second[i] << ",";
        }
        std::cout << node.second[node.second.size()-1] << "]" << std::endl;
    }

    // detect a cycle and isolated nodes in the block copolymer graph using depth first search
    std::map<int, bool> is_visited;
    for (int i = 0; i < block_lengths.size(); i++)
        is_visited[v[i]] = false;

    std::stack<std::pair<int,int>> connected_nodes;
    connected_nodes.push(std::make_pair(v[0],-1));
    while (!connected_nodes.empty()){
        //std::cout << "connected_nodes" << connected_nodes.top() << std::endl;

        // pop item and visit
        int cur = connected_nodes.top().first;
        int parent = connected_nodes.top().second;
        is_visited[cur] = true;
        connected_nodes.pop();

        // add adjacent_nodes at stack
        auto nodes = adjacent_nodes[cur];
        for(int i=0; i<nodes.size();i++){
            if (is_visited[nodes[i]] && nodes[i] != parent){
                throw_with_line_number("A cycle is detected, which contains nodes " 
                    + std::to_string(nodes[i]) + " and " + std::to_string(parent)
                    + ". Only acyclic block copolymer is allowed.");
            }
            else if(! is_visited[nodes[i]]){
                connected_nodes.push(std::make_pair(nodes[i], cur));
            }
        }


    }
    for (int i=0; i<block_lengths.size(); i++){
        if (!is_visited[v[i]])
            throw_with_line_number("There are disconnected nodes. Please check node number: " + std::to_string(v[i]) + ".");
    }

    // construct edge nodes
    for (int i=0; i<block_lengths.size(); i++){
        if( edges.count(std::make_pair(v[i], u[i])) > 0 ){
            throw_with_line_number("There are duplicated edges. Please check edge : ("
                + std::to_string(v[i]) + ", " + std::to_string(u[i]) + ").");
        }
        else{
            edges[std::make_pair(v[i], u[i])] = std::make_pair(block_types[i], n_segments[i]);
            edges[std::make_pair(u[i], v[i])] = std::make_pair(block_types[i], n_segments[i]);
        }
    }

    // find unique sub branches using `dynamic programming`
    for (int i=0; i<block_lengths.size(); i++){
        get_text_of_ordered_branches(v[i], u[i]);
        get_text_of_ordered_branches(u[i], v[i]);
    }

    // print unique sub branches
    for(const auto& item : dependencies){
        std::cout << item.first << ":\n\t";
        std::cout << "{max_segments: " << max_segments[item.first] << ",\n\tdependencies: [";
        int count = item.second.size(); 
        if(count > 0){
            for(int i=0; i<count-1; i++){ 
                std::cout << "[" << item.second[i].first << ", " <<  item.second[i].second << "], " ;
            }
            std::cout << "[" << item.second[count-1].first << ", " <<  item.second[count-1].second << "]" ;
        }
        std::cout << "]}" << std::endl;
    }

    // save variable
    try
    {

        //this->n_segment_total = 0;
        //this->block_start = {0};
        //for(int i=0; i<block_lengths.size(); i++)
        //{
            //this->n_segment_total += std::lround(block_lengths[i]/ds);
            //block_start.push_back(block_start.back() + n_segments[i]);
        //}
        this->block_types = block_types;
        this->ds = ds;
    }
    catch(std::exception& exc)
    {
        std::cerr << "Exception caught : " << exc.what() << std::endl;
        //throw_without_line_number(exc.what());
    }
}

std::string BranchedPolymerChain::get_model_name()
{
    return model_name;
}
double BranchedPolymerChain::get_ds()
{
    return ds;
}
std::vector<double> BranchedPolymerChain::get_bond_length_sq()
{
    return bond_length_sq;
}
double BranchedPolymerChain::get_bond_length_sq(int block)
{
    return bond_length_sq[block];
}
int BranchedPolymerChain::get_n_block()
{
    return block_types.size();
}
int BranchedPolymerChain::get_n_segment(int block)
{
    return n_segments[block];
}
// std::vector<int> BranchedPolymerChain::get_n_segment()
// {
//     return n_segments;
// }
// int BranchedPolymerChain::get_n_segment_total()
// {
//     return n_segment_total;
// }
std::vector<std::string> BranchedPolymerChain::get_block_type()
{
    return block_types;
}
std::string BranchedPolymerChain::get_block_type(int block_number)
{
    return block_types[block_number];
}

std::pair<std::string, int> BranchedPolymerChain::get_text_of_ordered_branches(int in_node, int out_node)
{
    // find children
    std::vector<std::string> edge_text;
    std::vector<std::pair<std::string,int>> edge_dict;
    std::pair<std::string,int> text_and_segments;

    //std::cout << "[" + std::to_string(in_node) + ", " +  std::to_string(out_node) + "]:";
    for(int i=0; i<adjacent_nodes[in_node].size(); i++){
        if (adjacent_nodes[in_node][i] != out_node){
            //std::cout << "(" << in_node << ", " << adjacent_nodes[in_node][i] << ")";
            text_and_segments = get_text_of_ordered_branches(adjacent_nodes[in_node][i], in_node);
            edge_text.push_back(text_and_segments.first + std::to_string(text_and_segments.second));
            edge_dict.push_back(text_and_segments);
            //std::cout << text_and_segments.first << " " << text_and_segments.second << std::endl;
        }
    }
    //std::cout << std::endl;

    std::string text;
    if(edge_text.size() == 0)
        text = "";
    else{
        std::sort (edge_text.begin(), edge_text.end());
        text += "(";
        for(int i=0; i<edge_text.size(); i++){
            text += edge_text[i];
        }
        text += ")";
    }
    text += edges[std::make_pair(in_node, out_node)].first;  // species;
    if(dependencies.count(text) > 0){
         if(max_segments[text] < edges[std::make_pair(in_node, out_node)].second){  // segments;
             max_segments[text] = edges[std::make_pair(in_node, out_node)].second;
         }
    }
    else{
        dependencies[text] = edge_dict;
        max_segments[text] = edges[std::make_pair(in_node, out_node)].second;
        //std::cout << edges[std::make_pair(in_node, out_node)].second << std::endl;
    }
    //std::cout << text << ", " << edges[std::make_pair(in_node, out_node)].second << std::endl;
    return std::make_pair(text, edges[std::make_pair(in_node, out_node)].second);
}

// std::vector<int> BranchedPolymerChain::get_block_start()
// {
//     return block_start;
// }