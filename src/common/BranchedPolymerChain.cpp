#include <iostream>
#include <cctype>
#include <cmath>
#include <algorithm>
#include <stack>

#include "BranchedPolymerChain.h"
#include "Exception.h"

//----------------- Constructor ----------------------------
BranchedPolymerChain::BranchedPolymerChain(
    std::string model_name, double ds, std::map<std::string, double> dict_bond_lengths,
    std::vector<std::string> block_species, std::vector<double> contour_lengths, std::vector<int> v, std::vector<int> u,
    std::map<int, int> v_to_grafting_index)
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
    if( block_species.size() != contour_lengths.size())
        throw_with_line_number("The sizes of block_species (" + std::to_string(block_species.size()) + 
            ") and contour_lengths (" +std::to_string(contour_lengths.size()) + ") must be consistent.");

    if( block_species.size() != v.size())
        throw_with_line_number("The sizes of block_species (" + std::to_string(block_species.size()) + 
            ") and edges v (" +std::to_string(v.size()) + ") must be consistent.");

    if( block_species.size() != u.size())
        throw_with_line_number("The sizes of block_species (" + std::to_string(block_species.size()) + 
            ") and edges u (" +std::to_string(v.size()) + ") must be consistent.");

    // check block lengths, segments, types
    for(int i=0; i<contour_lengths.size(); i++)
    {
        if( contour_lengths[i] <= 0)
            throw_with_line_number("contour_lengths[" + std::to_string(i) + "] (" +std::to_string(contour_lengths[i]) + ") must be a positive number.");
        if( std::abs(std::lround(contour_lengths[i]/ds)-contour_lengths[i]/ds) > 1.e-6)
            throw_with_line_number("contour_lengths[" + std::to_string(i) + "]/ds (" + std::to_string(contour_lengths[i]) + "/" + std::to_string(ds) + ") is not an integer.");
        if( dict_bond_lengths.count(block_species[i]) == 0 )
            throw_with_line_number("block_species[" + std::to_string(i) + "] (\"" + block_species[i] + "\") is not in dict_bond_lengths.");
    }

    // compute alpha, sum of relative contour lengths
    double alpha{0.0};
    for(int i=0; i<blocks.size(); i++){
        alpha += blocks[i].contour_length;
    }
    this->alpha = alpha;

    // construct adjacent_nodes
    for(int i=0; i<contour_lengths.size(); i++)
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

    // save variables
    try
    {
        this->ds = ds;
        this->dict_bond_lengths = dict_bond_lengths;
        for(int i=0; i<contour_lengths.size(); i++){
            blocks.push_back({
                block_species[i],                         // species
                (int) std::lround(contour_lengths[i]/ds),   // n_segment
                contour_lengths[i],                         // contour_length
                v[i], u[i]});                               // nodes
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }

    // detect a cycle and isolated nodes in the block copolymer graph using depth first search
    std::map<int, bool> is_visited;
    for (int i = 0; i < contour_lengths.size(); i++)
        is_visited[v[i]] = false;

    std::stack<std::pair<int,int>> connected_nodes;
    connected_nodes.push(std::make_pair(v[0],-1));
    while (!connected_nodes.empty())
    {
        //std::cout << "connected_nodes" << connected_nodes.top() << std::endl;

        // pop item and visit
        int cur = connected_nodes.top().first;
        int parent = connected_nodes.top().second;
        is_visited[cur] = true;
        connected_nodes.pop();

        // add adjacent_nodes at stack
        auto nodes = adjacent_nodes[cur];
        for(int i=0; i<nodes.size();i++)
        {
            if (is_visited[nodes[i]] && nodes[i] != parent)
            {
                throw_with_line_number("A cycle is detected, which contains nodes " 
                    + std::to_string(nodes[i]) + " and " + std::to_string(parent)
                    + ". Only acyclic block copolymer is allowed.");
            }
            else if(! is_visited[nodes[i]])
            {
                connected_nodes.push(std::make_pair(nodes[i], cur));
            }
        }
    }
    for (int i=0; i<contour_lengths.size(); i++)
    {
        if (!is_visited[v[i]])
            throw_with_line_number("There are disconnected nodes. Please check node number: " + std::to_string(v[i]) + ".");
    }

    // construct edge nodes
    for (int i=0; i<contour_lengths.size(); i++)
    {
        if (edge_to_array.count(std::make_pair(v[i], u[i])) > 0)
        {
            throw_with_line_number("There are duplicated edges. Please check the edge between ("
                + std::to_string(v[i]) + ", " + std::to_string(u[i]) + ").");
        }
        else
        {
            edge_to_array[std::make_pair(v[i],u[i])] = i;
            edge_to_array[std::make_pair(u[i],v[i])] = i;
        }
    }

    // find unique sub branches using `dynamic programming`
    for (int i=0; i<contour_lengths.size(); i++)
    {
        edge_to_deps[std::make_pair(v[i],u[i])] = get_text_of_ordered_branches(v[i],u[i]).first;
        edge_to_deps[std::make_pair(u[i],v[i])] = get_text_of_ordered_branches(u[i],v[i]).first;
    }

    // // print unique sub branches
    // std::vector<std::pair<std::string, int>> sub_deps;
    // for(const auto& item : reduced_branches_max_segment)
    // {
    //     std::cout << item.first << ":\n\t";
    //     std::cout << "{max_segments: " << item.second << ",\n\tsub_deps: [";
    //     sub_deps = key_to_deps(item.first);
    //     for(int i=0; i<sub_deps.size(); i++)
    //     {
    //         std::cout << sub_deps[i].first << ":" << sub_deps[i].second << ", " ;
    //     }
    //     std::cout << "]}" << std::endl;
    // }

    // // find unique junctions
    // for(const auto& item: adjacent_nodes)
    // {
    //     std::vector<std::string> deps;
    //     for(int i=0; i<item.second.size(); i++)
    //     {
    //         std::string sub_dep = edge_to_deps[std::make_pair(item.second[i], item.first)];
    //         sub_dep += std::to_string(blocks[edge_to_array[std::make_pair(item.second[i], item.first)]].n_segment);
    //         deps.push_back(sub_dep);
    //     }

    //     std::sort(deps.begin(), deps.end());
    //     std::cout << item.first << ", " << std::endl;
    //     for(int i=0; i<deps.size(); i++)
    //     {
    //         std::cout << deps[i] << std::endl;
    //     }
    // }
}

std::string BranchedPolymerChain::get_model_name()
{
    return model_name;
}
double BranchedPolymerChain::get_ds()
{
    return ds;
}
int BranchedPolymerChain::get_n_block()
{
    return blocks.size();
}
std::string BranchedPolymerChain::get_block_species(int idx)
{
    return blocks[idx].species;
}
int BranchedPolymerChain::get_n_segment(int idx)
{
    return blocks[idx].n_segment;
}
int BranchedPolymerChain::get_n_segment_total()
{
    int total_n{0};
    for(int i=0; i<blocks.size(); i++)
        total_n += blocks[i].n_segment;
    return total_n;
}
double BranchedPolymerChain::get_alpha()
{
    return alpha;
}

std::map<std::string, double>& BranchedPolymerChain::get_dict_bond_lengths()
{
    return dict_bond_lengths;
}
struct polymer_chain_block& BranchedPolymerChain::get_block(int v, int u)
{
    if (edge_to_array.count(std::make_pair(v, u)) == 0)
        throw_with_line_number("There is no such edge (" + std::to_string(v) + ", " + std::to_string(u) + ").");
    return blocks[edge_to_array[std::make_pair(v, u)]];
}

std::vector<polymer_chain_block>& BranchedPolymerChain::get_blocks()
{
    return blocks;
}
std::string BranchedPolymerChain::get_dep(int v, int u){
    if (edge_to_deps.count(std::make_pair(v, u)) == 0)
        throw_with_line_number("There is no such edge (" + std::to_string(v) + ", " + std::to_string(u) + ").");
    return edge_to_deps[std::make_pair(v,u)];
}
// std::vector<std::string> BranchedPolymerChain::get_block_species()
// {
//     return block_species;
// }
// std::vector<int> BranchedPolymerChain::get_n_segment()
// {
//     return n_segments;
// }
// std::vector<double> BranchedPolymerChain::get_bond_length_sq()
// {
//     return bond_length_sq;
// }
// int BranchedPolymerChain::get_n_segment_total()
// {
//     return n_segment_total;
// }

std::pair<std::string, int> BranchedPolymerChain::get_text_of_ordered_branches(int in_node, int out_node)
{
    std::vector<std::string> edge_text;
    std::vector<std::pair<std::string,int>> edge_dict;
    std::pair<std::string,int> text_and_segments;

    // explore child branches
    //std::cout << "[" + std::to_string(in_node) + ", " +  std::to_string(out_node) + "]:";
    for(int i=0; i<adjacent_nodes[in_node].size(); i++)
    {
        if (adjacent_nodes[in_node][i] != out_node)
        {
            //std::cout << "(" << in_node << ", " << adjacent_nodes[in_node][i] << ")";
            text_and_segments = get_text_of_ordered_branches(adjacent_nodes[in_node][i], in_node);
            edge_text.push_back(text_and_segments.first + std::to_string(text_and_segments.second));
            edge_dict.push_back(text_and_segments);
            //std::cout << text_and_segments.first << " " << text_and_segments.second << std::endl;
        }
    }

    // merge text of child branches
    std::string text;
    if(edge_text.size() == 0)
        text = "";
    else
    {
        std::sort (edge_text.begin(), edge_text.end());
        text += "(";
        for(int i=0; i<edge_text.size(); i++)
            text += edge_text[i];
        text += ")";
    }

    // update reduced_sub_branches
    text += blocks[edge_to_array[std::make_pair(in_node, out_node)]].species;
    if(reduced_branches_max_segment.count(text) > 0)
    {
         if(reduced_branches_max_segment[text] < blocks[edge_to_array[std::make_pair(in_node, out_node)]].n_segment)
             reduced_branches_max_segment[text] = blocks[edge_to_array[std::make_pair(in_node, out_node)]].n_segment;
    }
    else
    {
        //reduced_branches_max_segment[text].sub_deps = edge_dict;
        reduced_branches_max_segment[text] = blocks[edge_to_array[std::make_pair(in_node, out_node)]].n_segment;
    }
    return std::make_pair(text, blocks[edge_to_array[std::make_pair(in_node, out_node)]].n_segment);
    // return std::make_pair("A", 10);
}
int BranchedPolymerChain::get_reduced_n_branches()
{
    return reduced_branches_max_segment.size();
}
std::vector<std::pair<std::string, int>> BranchedPolymerChain::key_to_deps(std::string key)
{
    std::stack<int> s;

    std::vector<std::pair<std::string, int>> sub_deps;
    int sub_n_segment;
    std::string sub_key;

    bool is_finding_key = true;
    int key_start = 1;
    for(int i=0; i<key.size();i++)
    {
        if( isdigit(key[i]) && is_finding_key && s.size() == 1 )
        {
            sub_key = key.substr(key_start, i-key_start);
            //std::cout << sub_key << ": " << key_start << ", " << i  << std::endl;
            is_finding_key = false;
            key_start = i;
        }
        else if( !isdigit(key[i]) && !is_finding_key && s.size() == 1)
        {
            sub_n_segment = std::stoi(key.substr(key_start, i-key_start));
            //std::cout << sub_key << ": " << key_start << ", " << i  << ", " << key.substr(key_start, i-key_start) << std::endl;
            sub_deps.push_back(std::make_pair(sub_key, sub_n_segment));
            is_finding_key = true;
            key_start = i;
        }
        if(key[i] == '(')
            s.push(i);
        else if(key[i] == ')')
            s.pop();
        //std::cout << "key_start: " << key_start << std::endl;
    }
    return sub_deps;
}
std::string BranchedPolymerChain::key_to_species(std::string key){
    int key_start = 0;
    for(int i=key.size()-1; i>=0;i--)
    {
        //std::cout << key[i] << std::endl;
        if(key[i] == ')')
        {
            key_start=i+1;
            break;
        }
    }
    //std::cout << key.substr(key_start, key.size()-key_start) << std::endl;
    return key.substr(key_start, key.size()-key_start);
}
std::map<std::string, int, std::greater<std::string>>& BranchedPolymerChain::get_reduced_branches_max_segment()
{
    return reduced_branches_max_segment;
}
int BranchedPolymerChain::get_reduced_branch_max_segment(std::string key)
{
    return reduced_branches_max_segment[key];
}