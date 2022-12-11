#include <iostream>
#include <cctype>
#include <cmath>
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
Mixture::~Mixture()
{
    for(int p=0; p<distinct_polymers.size(); p++)
    {
        delete distinct_polymers[p];
    }
}
void Mixture::add_polymer_chain(
    double volume_fraction,
    std::vector<std::string> block_species,
    std::vector<double> contour_lengths,
    std::vector<int> v, std::vector<int> u,
    std::map<int, int> v_to_grafting_index)
{
    std::string deps;
    PolymerChain* pc = new PolymerChain(ds, bond_lengths, 
        volume_fraction, block_species, contour_lengths,
        v, u, v_to_grafting_index);
    distinct_polymers.push_back(pc);

    // find unique sub branches
    for (int i=0; i<pc->get_n_block(); i++)
    {
        deps = get_text_of_ordered_branches(
            pc->get_blocks(), pc->get_adjacent_nodes(),
            pc->get_edge_to_array(), v[i], u[i]).first;
        pc->set_edge_to_deps(v[i], u[i], deps);

        deps = get_text_of_ordered_branches(
            pc->get_blocks(), pc->get_adjacent_nodes(),
            pc->get_edge_to_array(), u[i], v[i]).first;
        pc->set_edge_to_deps(u[i], v[i], deps);
    }

    // find unique blocks
    std::vector<PolymerChainBlock>& blocks = pc->get_blocks();
    for(int b=0; b<blocks.size(); b++){
        std::string dep_v = pc->get_dep(blocks[b].v, blocks[b].u);
        std::string dep_u = pc->get_dep(blocks[b].u, blocks[b].v);
        if (dep_v > dep_u)
            dep_v.swap(dep_u);
        auto key = std::make_tuple(dep_v, dep_u, blocks[b].n_segment);
        reduced_blocks[key].species = blocks[b].species;
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
std::string Mixture::get_model_name()
{
    return model_name;
}
double Mixture::get_ds()
{
    return ds;
}
int Mixture::get_n_distinct_polymers()
{
    return distinct_polymers.size();
}
PolymerChain* Mixture::get_polymer_chain(int p)
{
    return distinct_polymers[p];
}

std::map<std::string, double>& Mixture::get_bond_lengths()
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

    // update reduced_sub_branches
    text += blocks[edge_to_array[std::make_pair(in_node, out_node)]].species;
    if(reduced_branches.count(text) > 0)
    {
         if(reduced_branches[text].max_n_segment < blocks[edge_to_array[std::make_pair(in_node, out_node)]].n_segment)
             reduced_branches[text].max_n_segment = blocks[edge_to_array[std::make_pair(in_node, out_node)]].n_segment;
    }
    else
    {
        reduced_branches[text].max_n_segment = blocks[edge_to_array[std::make_pair(in_node, out_node)]].n_segment;
        reduced_branches[text].deps = key_to_deps(text);
        reduced_branches[text].species = key_to_species(text);
    }
    return std::make_pair(text, blocks[edge_to_array[std::make_pair(in_node, out_node)]].n_segment);
    // return std::make_pair("A", 10);
}
int Mixture::get_reduced_n_branches()
{
    return reduced_branches.size();
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
std::map<std::string, ReducedEdge, std::greater<std::string>>& Mixture::get_reduced_branches()
{
    return reduced_branches;
}
ReducedEdge Mixture::get_reduced_branch(std::string key)
{
    return reduced_branches[key];
}
std::map<std::tuple<std::string, std::string, int>, ReducedBlock>& Mixture::get_reduced_blocks()
{
    return reduced_blocks;
}
ReducedBlock Mixture::get_reduced_block(std::tuple<std::string, std::string, int> key)
{
    return reduced_blocks[key];
}