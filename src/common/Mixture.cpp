#include <iostream>
#include <cctype>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <stack>

#include "Mixture.h"
#include "Exception.h"

//----------------- Constructor ----------------------------
Mixture::Mixture(
    std::string model_name, double ds, std::map<std::string, double> bond_lengths)
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

    // save variables
    try
    {
        this->ds = ds;
        this->bond_lengths = bond_lengths;
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void Mixture::add_polymer(
    double volume_fraction,
    std::vector<std::string> block_species,
    std::vector<double> contour_lengths,
    std::vector<int> v, std::vector<int> u,
    std::map<int, int> v_to_grafting_index)
{
    std::string deps;
    distinct_polymers.push_back(PolymerChain(ds, bond_lengths, 
        volume_fraction, block_species, contour_lengths,
        v, u, v_to_grafting_index));

    PolymerChain& pc = distinct_polymers.back();

    // find unique sub branches
    for (int i=0; i<pc.get_n_blocks(); i++)
    {
        deps = get_text_of_ordered_branches(
            pc.get_blocks(), pc.get_adjacent_nodes(),
            pc.get_edge_to_array(), v[i], u[i]).first;
        pc.set_edge_to_deps(v[i], u[i], deps);

        deps = get_text_of_ordered_branches(
            pc.get_blocks(), pc.get_adjacent_nodes(),
            pc.get_edge_to_array(), u[i], v[i]).first;
        pc.set_edge_to_deps(u[i], v[i], deps);
    }

    // find unique blocks
    std::vector<PolymerChainBlock> blocks = pc.get_blocks();
    for(int b=0; b<blocks.size(); b++){
        std::string dep_v = pc.get_dep(blocks[b].v, blocks[b].u);
        std::string dep_u = pc.get_dep(blocks[b].u, blocks[b].v);
        if (dep_v > dep_u)
            dep_v.swap(dep_u);
        auto key = std::make_tuple(dep_v, dep_u, blocks[b].n_segment);
        unique_blocks[key].species = blocks[b].species;
    }

}
std::string Mixture::get_model_name() const
{
    return model_name;
}
double Mixture::get_ds() const
{
    return ds;
}
int Mixture::get_n_polymers() const
{
    return distinct_polymers.size();
}
PolymerChain& Mixture::get_polymer(const int p)
{
    return distinct_polymers[p];
}
const std::map<std::string, double>& Mixture::get_bond_lengths() const
{
    return bond_lengths;
}
std::pair<std::string, int> Mixture::get_text_of_ordered_branches(
    std::vector<PolymerChainBlock> blocks,
    std::map<int, std::vector<int>> adjacent_nodes,
    std::map<std::pair<int, int>, int> edge_to_array,
    int in_node, int out_node)
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
            text_and_segments = get_text_of_ordered_branches(
                blocks, adjacent_nodes, edge_to_array,
                adjacent_nodes[in_node][i], in_node);
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

    // update unique_sub_branches
    text += blocks[edge_to_array[std::make_pair(in_node, out_node)]].species;
    if(unique_branches.count(text) > 0)
    {
         if(unique_branches[text].max_n_segment < blocks[edge_to_array[std::make_pair(in_node, out_node)]].n_segment)
             unique_branches[text].max_n_segment = blocks[edge_to_array[std::make_pair(in_node, out_node)]].n_segment;
    }
    else
    {
        unique_branches[text].max_n_segment = blocks[edge_to_array[std::make_pair(in_node, out_node)]].n_segment;
        unique_branches[text].deps = key_to_deps(text);
        unique_branches[text].species = key_to_species(text);
        for(int i=0; i<text.size();i++)
        {
            if (text[i] != '(')
            {
                unique_branches[text].height = i;
                break;
            }
        }
    }
    return std::make_pair(text, blocks[edge_to_array[std::make_pair(in_node, out_node)]].n_segment);
    // return std::make_pair("A", 10);
}
int Mixture::get_unique_n_branches() const
{
    return unique_branches.size();
}
std::vector<std::pair<std::string, int>> Mixture::key_to_deps(std::string key)
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
std::string Mixture::key_to_species(std::string key){
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
std::map<std::string, UniqueEdge, std::greater<std::string>>& Mixture::get_unique_branches()
{
    return unique_branches;
}
UniqueEdge& Mixture::get_unique_branch(std::string key)
{
    if (unique_branches.count(key) == 0)
        throw_with_line_number("There is no such key (" + key + ").");
    return unique_branches[key];
}
std::map<std::tuple<std::string, std::string, int>, UniqueBlock>& Mixture::get_unique_blocks()
{
    return unique_blocks;
}
UniqueBlock& Mixture::get_unique_block(std::tuple<std::string, std::string, int> key)
{
    assert(unique_blocks.count(key) == 0 && "There is no such key (" +
            std::get<0>(key) + ", " + std::get<1>(key) + ", " + std::to_string(std::get<2>(key)) + ").");
    return unique_blocks[key];
}
void Mixture::display_unique_branches() const
{
    // print unique sub branches
    std::vector<std::pair<std::string, int>> sub_deps;
    std::cout << "--------- Unique Branches ---------" << std::endl;
    for(const auto& item : unique_branches)
    {
        std::cout << item.first;
        std::cout << ":\n\t{max_n_segment: " << item.second.max_n_segment;
        std::cout << ", height: " << item.second.height;
        std::cout << ",\n\tsub_deps: [";
        sub_deps = key_to_deps(item.first);
        for(int i=0; i<sub_deps.size(); i++)
        {
            std::cout << sub_deps[i].first << ":" << sub_deps[i].second << ", " ;
        }
        std::cout << "]}" << std::endl;
    }
    std::cout << "------------------------------------" << std::endl;
}
void Mixture::display_unique_blocks() const
{
    // print unique sub blocks
    std::cout << "---------- Unique Blocks ----------" << std::endl;
    for(const auto& item : unique_blocks)
    {
        const auto& key = item.first;
        std::cout << std::get<0>(key) + ", " + std::get<1>(key) + ", " + std::to_string(std::get<2>(key)) << std::endl;
    }
    std::cout << "------------------------------------" << std::endl;
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