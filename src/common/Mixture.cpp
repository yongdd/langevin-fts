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
    std::vector<std::string> block_monomer_types,
    std::vector<double> contour_lengths,
    std::vector<int> v, std::vector<int> u,
    std::map<int, int> v_to_grafting_index)
{
    std::string deps;
    distinct_polymers.push_back(PolymerChain(ds, bond_lengths, 
        volume_fraction, block_monomer_types, contour_lengths,
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

    // temporary maps for this single polymer
    std::map<std::tuple<int, std::string, std::string, int>, UniqueBlock> unique_blocks_one_polymer; 
    std::map<std::tuple<int, std::string>, std::map<std::tuple<int, std::string>, std::vector<std::tuple<int, int>>, std::greater<void> >> unique_block_count_one_polymer;
    std::map<std::tuple<int, std::string>, std::map<std::tuple<int, std::string>, std::vector<std::tuple<int, int>>, std::greater<void> >> unique_block_superposition_one_polymer;

    // std::map<std::string, UniqueEdge, std::greater<std::string>> unique_branches_superposition_one_polymer;

    // find unique_blocks_one_polymer
    std::vector<PolymerChainBlock> blocks = pc.get_blocks();
    for(int b=0; b<blocks.size(); b++)
    {
        int v = blocks[b].v;
        int u = blocks[b].u;
        std::string dep_v = pc.get_dep(v, u);
        std::string dep_u = pc.get_dep(u, v);

        if (dep_v < dep_u){
            dep_v.swap(dep_u);
            std::swap(v,u);
        }
        
        auto key = std::make_tuple(distinct_polymers.size()-1, dep_v, dep_u, blocks[b].n_segment);
        // auto key_minus_species = std::make_tuple(distinct_polymers.size()-1, Mixture::key_minus_species(dep_v), Mixture::key_minus_species(dep_u), blocks[b].n_segment);

        if (unique_blocks_one_polymer.count(key) == 0)
        {
            unique_blocks_one_polymer[key].monomer_type = blocks[b].monomer_type;
            unique_blocks_one_polymer[key].v_u.push_back(std::make_tuple(v,u));

            unique_blocks[key].monomer_type = blocks[b].monomer_type;
            unique_blocks[key].v_u.push_back(std::make_tuple(v,u));
        }
        else
        {
            unique_blocks_one_polymer[key].v_u.push_back(std::make_tuple(v,u));
            
            unique_blocks[key].v_u.push_back(std::make_tuple(v,u));
        }
    }

    // insert to unique_block_count_one_polymer
    for(auto& item : unique_blocks_one_polymer)
    {
        std::string dep_v = std::get<1>(item.first);
        std::string dep_u = std::get<2>(item.first);
        // if (dep_v < dep_u)
        //     dep_v.swap(dep_u);
        
        // auto block_key = std::make_tuple(distinct_polymers.size()-1, dep_v, dep_u, item.second.monomer_type);
        auto key1 = std::make_tuple(distinct_polymers.size()-1, dep_v);
        auto key2 = std::make_tuple(std::get<3>(item.first), dep_u);
        // unique_block_count_one_polymer[key].push_back(std::make_tuple(std::get<3>(item.first), dep_u, item.second.v_u));

        unique_block_count_one_polymer[key1][key2] = item.second.v_u;
        unique_block_superposition_one_polymer[key1] = {};
    }

    // // sort vectors with n_segment (descending) and keys (ascending)
    // for(auto& item : unique_block_count_one_polymer)
    // {
    //     std::sort(item.second.begin(), item.second.end(),
    //         [](auto const &t1, auto const &t2)
    //         {
    //             if (std::get<0>(t1) == std::get<0>(t2))
    //                 return std::get<1>(t1) < std::get<1>(t2);
    //             else
    //             {
    //                 return std::get<0>(t1) > std::get<0>(t2);
    //             }
    //         }
    //     );
    // }

    // print Unique Blocks
    std::cout << "----------- Unique Blocks -----------" << std::endl;
    std::cout << "Polymer id, key1: n_segment, key2, v_u" << std::endl;
    for(const auto& item : unique_block_count_one_polymer)
    {
        const auto& v_key = item.first;
        std::cout << std::to_string(std::get<0>(v_key)) + ", " 
            + std::get<1>(v_key) + ":" << std::endl;
        
        for(const auto& u_key : item.second)
        {
            std::cout << "\t" + std::to_string(std::get<0>(u_key.first)) + ", "
            + std::get<1>(u_key.first) + ", ";
            for(const auto& v_u : u_key.second)
            {
                    std::cout << "("
                    + std::to_string(std::get<0>(v_u)) + ","
                    + std::to_string(std::get<1>(v_u)) + ")" + ", ";
            }
            std::cout << std::endl;
        }
    }
    std::cout << "------------------------------------" << std::endl;

    // generate unique_block_superposition_one_polymer from unique_block_count_one_polymer
    for(auto& item : unique_block_count_one_polymer)
    {
        auto dep_u_list = item.second;
        if (dep_u_list.size() == 1)
        {
            unique_block_superposition_one_polymer[item.first][dep_u_list.begin()->first] = dep_u_list.begin()->second;
        }
        else
        {
            std::vector<std::tuple<int, int>> v_u_total_list;
            std::map<std::tuple<int, std::string>, std::vector<std::tuple<int, int>>, std::greater<void>> new_dep_u_list;
            for(auto& u_key : unique_block_superposition_one_polymer[item.first]) 
            {
                for(auto& superposition_v_u : u_key.second) 
                {
                    new_dep_u_list[u_key.first].push_back(superposition_v_u);
                    v_u_total_list.push_back(superposition_v_u);
                    // std::cout << "(v_u): " << std::get<0>(superposition_v_u) <<  ", " << std::get<1>(superposition_v_u) << std::endl;
                }
            }
            for(auto& u_key : dep_u_list) 
            {
                for(auto& new_v_u : u_key.second) 
                {
                    if ( std::find(v_u_total_list.begin(), v_u_total_list.end(), new_v_u) == v_u_total_list.end())
                    {
                        new_dep_u_list[u_key.first].push_back(new_v_u);
                        // std::cout << "u_key: " << std::get<0>(u_key.first) <<  ", " << std::get<1>(u_key.first) << std::endl;
                        // std::cout << "(v_u): " << std::get<0>(new_v_u) <<  ", " << std::get<1>(new_v_u) << std::endl;
                    }
                }
            }

            // find superposition
            auto dep_u_superposition_list = superpose_branches(new_dep_u_list);
            unique_block_superposition_one_polymer[item.first].clear();
            for(auto& dep_u_superposition : dep_u_superposition_list)
            {
                // std::cout << std::to_string(std::get<0>(dep_u_superposition)) + ", "
                //     + std::get<1>(dep_u_superposition) + ", ";
                // for(const auto& v_u : std::get<2>(dep_u_superposition))
                // {
                //         std::cout << "("
                //         + std::to_string(std::get<0>(v_u)) + ","
                //         + std::to_string(std::get<1>(v_u)) + ")" + ", ";
                // }
                // std::cout << std::endl;

                unique_block_superposition_one_polymer[item.first][std::make_tuple(std::get<0>(dep_u_superposition),std::get<1>(dep_u_superposition))]
                    = std::get<2>(dep_u_superposition);
            }

            // for each node of last dep_u_superposition
            for(auto& v_u : std::get<2>(dep_u_superposition_list.back()))
            {
                auto& v_adj_nodes = pc.get_adjacent_nodes()[std::get<0>(v_u)];
                // for each v_adj_node
                for(auto& v_adj_node : v_adj_nodes)
                {
                    if (v_adj_node != std::get<1>(v_u))
                    {
                        // std::cout << "(v_u): " << v_adj_node <<  ", " << std::get<0>(v_u) << std::endl;

                        int v = v_adj_node;
                        int u = std::get<0>(v_u);

                        std::string dep_v = pc.get_dep(v, u);
                        std::string dep_u = pc.get_dep(u, v);
                        // std::cout << dep_v << ", " << dep_u << ", " << pc.get_block(v,u).n_segment << std::endl;

                        auto key = std::make_tuple(distinct_polymers.size()-1, dep_v);
                        // pc.get_block(v,u).monomer_type
                        // pc.get_block(v,u).n_segment

                        std::tuple<int, std::string> new_u_key = std::make_tuple(pc.get_block(v,u).n_segment, "[" + std::get<1>(dep_u_superposition_list.back())
                            + std::to_string(std::get<0>(dep_u_superposition_list.back())) + "]" + pc.get_block(v,u).monomer_type);
                        
                        // std::cout << "new_u_key: " << std::get<0>(new_u_key) << ", " << std::get<1>(new_u_key) << std::endl;
                        unique_block_superposition_one_polymer[key][new_u_key].push_back(std::make_tuple(v,u));
                    }
                }
            }
            // for(auto& item_another : unique_block_count_one_polymer)
            // {
            //     if( Mixture::key_to_height(std::get<1>(item.first)) > Mixture::key_to_height(std::get<1>(item_another.first)))
            //     {
            //         // for(auto& v_u : std::get<2>(dep_u_superposition_list.back()))
            //         // {

            //         // }
            //     }
            // }

        }
    }

    // print Unique Blocks
    std::cout << "----------- Unique Blocks (Superposition) -----------" << std::endl;
    std::cout << "Polymer id, key1: n_segment, key2, v_u" << std::endl;
    for(const auto& item : unique_block_superposition_one_polymer)
    {
        const auto& v_key = item.first;
        std::cout << std::to_string(std::get<0>(v_key)) + ", " 
            + std::get<1>(v_key) + ":" << std::endl;
        
        for(const auto& u_key : item.second)
        {
            std::cout << "\t" + std::to_string(std::get<0>(u_key.first)) + ", "
            + std::get<1>(u_key.first) + ", ";
            for(const auto& v_u : u_key.second)
            {
                    std::cout << "("
                    + std::to_string(std::get<0>(v_u)) + ","
                    + std::to_string(std::get<1>(v_u)) + ")" + ", ";
            }
            std::cout << std::endl;
        }
    }
    std::cout << "------------------------------------" << std::endl;

    // copy data
    // for(const auto& item : unique_blocks_one_polymer)
    // {
    //     unique_blocks[item.first] = item.second;
    // }

    // for(const auto& item : unique_block_superposition_one_polymer)
    // {
    //     unique_block_superposition[item.first] = item.second;
    // }
    // for(const auto& item : unique_branches_superposition_one_polymer)
    // {
    //     unique_branches_superposition[item.first] = item.second;
    // }

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
    text += blocks[edge_to_array[std::make_pair(in_node, out_node)]].monomer_type;
    if(unique_branches.count(text) > 0)
    {
         if(unique_branches[text].max_n_segment < blocks[edge_to_array[std::make_pair(in_node, out_node)]].n_segment)
             unique_branches[text].max_n_segment = blocks[edge_to_array[std::make_pair(in_node, out_node)]].n_segment;
    }
    else
    {
        unique_branches[text].max_n_segment = blocks[edge_to_array[std::make_pair(in_node, out_node)]].n_segment;
        unique_branches[text].deps = key_to_deps(text);
        unique_branches[text].monomer_type = key_to_species(text);
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
void Mixture::add_unique_branch(std::map<std::string, UniqueEdge, std::greater<std::string>>& unique_branches_superposition, std::string key, int new_n_segment)
{
    if (unique_branches_superposition.count(key) == 0)
    {
        unique_branches_superposition[key].deps = Mixture::key_to_deps(key);
        unique_branches_superposition[key].max_n_segment = new_n_segment;
    }
    else
    {
        if (unique_branches_superposition[key].max_n_segment < new_n_segment)
            unique_branches_superposition[key].max_n_segment = new_n_segment;
    }
}
std::vector<std::tuple<int, std::string, std::vector<std::tuple<int ,int>>>> Mixture::superpose_branches(std::map<std::tuple<int, std::string>, std::vector<std::tuple<int, int>>, std::greater<void>> map_u_list)
{
    // Example 1)
    // 0, B:
    //   6, (C)B, 1, 
    //   6, (D)B, 3, 
    //   6, (E)B, 2, 
    //   4, (F)B, 1, 
    //
    //      ↓   Superposition
    //  
    // [C:1,D:3,E:2]B, 2
    // [[C:1,D:3,E:2]B2,F:1]B, 4

    // Example 1)
    // 0, B:
    //   6, (C)B, 1, 
    //   4, (D)B, 3, 
    //
    //      ↓   Superposition
    //  
    // (C)B, 2
    // [(C)B2,D:3]B, 4

    int n_segment;
    int current_max_n_segment = std::get<0>(map_u_list.begin()->first);
    int count = 1;
    std::string dep_u_superposition;

    std::vector<std::tuple<int, std::string, std::vector<std::tuple<int ,int>>>> dep_u_superposition_list;
    std::vector<std::tuple<int, std::string, std::vector<std::tuple<int ,int>>>> sample_n_segment_superposition_list;
    std::vector<std::tuple<int ,int>> v_u_total;

    // // print Unique Blocks
    // // std::cout << "----------- Unique Blocks -----------" << std::endl;
    // for(const auto& u_key : map_u_list)
    // {
    //     std::cout << "\t" + std::to_string(std::get<0>(u_key.first)) + ", "
    //     + std::get<1>(u_key.first) + ", " 
    //     + std::to_string(u_key.second.size()) + ", ";
    //     for(const auto& v_u : u_key.second)
    //     {
    //             std::cout << "("
    //             + std::to_string(std::get<0>(v_u)) + ","
    //             + std::to_string(std::get<1>(v_u)) + ")" + ", ";
    //     }
    //     std::cout << std::endl;
    // }
    // // std::cout << "------------------------------------" << std::endl;

    for(const auto& item : map_u_list)
    {
        // superpose
        if(current_max_n_segment > std::get<0>(item.first))
        {
            // make a key
            if (sample_n_segment_superposition_list.size() == 1)
            {
                dep_u_superposition = std::get<1>(sample_n_segment_superposition_list[0]);
            }
            else
            {
                // std::cout << "std::get<1>(sample_n_segment_superposition_list[0]): " << std::get<1>(sample_n_segment_superposition_list[0]) << std::endl;
                dep_u_superposition = "[" + Mixture::key_minus_species(std::get<1>(sample_n_segment_superposition_list[0]));
                if (Mixture::key_minus_species(std::get<1>(sample_n_segment_superposition_list[0]))[0] != '[')
                    dep_u_superposition += ":" + std::to_string(std::get<2>(sample_n_segment_superposition_list[0]).size());
                for(int i=1; i<sample_n_segment_superposition_list.size(); i++)
                {
                    dep_u_superposition +=  "," + Mixture::key_minus_species(std::get<1>(sample_n_segment_superposition_list[i]));
                    if (Mixture::key_minus_species(std::get<1>(sample_n_segment_superposition_list[i]))[0] != '[')
                        dep_u_superposition += ":" + std::to_string(std::get<2>(sample_n_segment_superposition_list[i]).size());
                }
                dep_u_superposition += "]" + Mixture::key_to_species(std::get<1>(sample_n_segment_superposition_list[0]));
            }
            // n_segment
            n_segment = current_max_n_segment-std::get<0>(item.first);

            // // display 
            // std::cout << dep_u_superposition << ", " << n_segment << ", ";
            // for(int i=0; i<v_u_total.size(); i++)
            // {
            //     std::cout << "(" + std::to_string(std::get<0>(v_u_total[i])) + "," + std::to_string(std::get<1>(v_u_total[i])) + ")," ;
            // }
            // std::cout << std::endl;

            // store result
            dep_u_superposition_list.push_back(std::make_tuple(n_segment, dep_u_superposition, v_u_total));

            // make next key
            current_max_n_segment = std::get<0>(item.first);
            sample_n_segment_superposition_list.clear();
            dep_u_superposition = "[" + dep_u_superposition + std::to_string(n_segment) + "]" + Mixture::key_to_species(std::get<1>(sample_n_segment_superposition_list[0]));
            sample_n_segment_superposition_list.push_back(std::make_tuple(n_segment, dep_u_superposition, v_u_total));
        }

        // add new u
        sample_n_segment_superposition_list.push_back(std::make_tuple(std::get<0>(item.first), std::get<1>(item.first), item.second));
        for(const auto& v_u : item.second)
            v_u_total.push_back(v_u);     

        // last element
        if (count == map_u_list.size())
        {
            // std::cout << "std::get<1>(sample_n_segment_superposition_list[0]): " << std::get<1>(sample_n_segment_superposition_list[0]) << std::endl;
            // std::cout << "std::get<1>(sample_n_segment_superposition_list[1]): " << std::get<1>(sample_n_segment_superposition_list[1]) << std::endl;

            dep_u_superposition = "[" + Mixture::key_minus_species(std::get<1>(sample_n_segment_superposition_list[0]));
            if (Mixture::key_minus_species(std::get<1>(sample_n_segment_superposition_list[0]))[0] != '[')
                dep_u_superposition += ":" + std::to_string(std::get<2>(sample_n_segment_superposition_list[0]).size());
            for(int i=1; i<sample_n_segment_superposition_list.size(); i++)
            {
                dep_u_superposition +=  "," + Mixture::key_minus_species(std::get<1>(sample_n_segment_superposition_list[i]));
                if (Mixture::key_minus_species(std::get<1>(sample_n_segment_superposition_list[i]))[0] != '[')
                    dep_u_superposition += ":" + std::to_string(std::get<2>(sample_n_segment_superposition_list[i]).size());
            }
            dep_u_superposition += "]" + Mixture::key_to_species(std::get<1>(sample_n_segment_superposition_list[0]));

            // n_segment
            n_segment = current_max_n_segment;

            // // display 
            // std::cout << dep_u_superposition << ", " << n_segment << ", ";
            // for(int i=0; i<v_u_total.size(); i++)
            // {
            //     std::cout << "(" + std::to_string(std::get<0>(v_u_total[i])) + "," + std::to_string(std::get<1>(v_u_total[i])) + ")," ;
            // }
            // std::cout << std::endl;

            // store result
            dep_u_superposition_list.push_back(std::make_tuple(n_segment, dep_u_superposition, v_u_total));
        }
        count++;
    }

    return dep_u_superposition_list;
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

std::string Mixture::key_minus_species(std::string key)
{
    if (key[0] != '[' && key[0] != '(')
    {
        return "";
    }
    else
    {
        int brace_count = 0;
        int species_idx = 0;
        for(int i=0; i<key.size();i++)
        {
            if (key[i] == '[' || key[i] == '(')
            {
                brace_count++;
            }
            else if (key[i] == ']' || key[i] == ')')
            {
                brace_count--;
                if (brace_count == 0)
                {
                    species_idx=i;
                    break;
                }
            }
        }
        // std::cout << "key.substr(1, species_idx): " << key.substr(1, species_idx-1) << std::endl;
        return key.substr(1, species_idx-1);
    }
}

std::string Mixture::key_to_species(std::string key)
{
    int key_start = 0;
    for(int i=key.size()-1; i>=0;i--)
    {
        //std::cout << key[i] << std::endl;
        if(key[i] == ')' || key[i] == ']')
        {
            key_start=i+1;
            break;
        }
    }
    //std::cout << key.substr(key_start, key.size()-key_start) << std::endl;
    return key.substr(key_start, key.size()-key_start);
}
int Mixture::key_to_height(std::string key)
{
    int height_count = 0;
    for(int i=0; i<key.size();i++)
    {
        if (key[i] == '[' || key[i] == '(')
            height_count++;
        else
            break;
    }
    return height_count;
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
std::map<std::tuple<int, std::string, std::string, int>, UniqueBlock>& Mixture::get_unique_blocks()
{
    return unique_blocks;
}
UniqueBlock& Mixture::get_unique_block(std::tuple<int, std::string, std::string, int> key)
{
    assert(unique_blocks.count(key) == 0 && "There is no such key (" +
            std::get<0>(key) + ", " + std::get<1>(key) + ", " + std::to_string(std::get<2>(key)) + ").");
    return unique_blocks[key];
}
void Mixture::display_unique_branches() const
{
    // // print unique sub branches
    // std::vector<std::pair<std::string, int>> sub_deps;
    // std::cout << "--------- Unique Branches ---------" << std::endl;
    // for(const auto& item : unique_branches)
    // {
    //     std::cout << item.first;
    //     std::cout << ":\n\t{max_n_segment: " << item.second.max_n_segment;
    //     std::cout << ", height: " << item.second.height;
    //     std::cout << ",\n\tsub_deps: [";
    //     sub_deps = key_to_deps(item.first);
    //     for(int i=0; i<sub_deps.size(); i++)
    //     {
    //         std::cout << sub_deps[i].first << ":" << sub_deps[i].second << ", " ;
    //     }
    //     std::cout << "]}" << std::endl;
    // }
    // std::cout << "------------------------------------" << std::endl;
}
void Mixture::display_unique_blocks() const
{
    // // print unique sub blocks
    // std::cout << "----------- Unique Blocks -----------" << std::endl;
    // std::cout << "Polymer id, key1, key2, n_segment, n_repeated" << std::endl;
    // for(const auto& item : unique_blocks)
    // {
    //     const auto& key = item.first;
    //     std::cout << std::to_string(std::get<0>(key)) + ", " 
    //         + std::get<1>(key) + ", " 
    //         + std::get<2>(key) + ", "
    //         + std::to_string(std::get<3>(key)) + ", "
    //         + std::to_string(item.second.n_repeated) + ", " << std::endl;
    // }

    // // print Unique Blocks
    // std::cout << "----------- Unique Blocks -----------" << std::endl;
    // std::cout << "Polymer id, key1: n_segment, key2, n_repeated" << std::endl;
    // for(const auto& item : unique_block_superposition)
    // {
    //     const auto& key = item.first;
    //     std::cout << std::to_string(std::get<0>(key)) + ", " 
    //         + std::get<1>(key) + ":" << std::endl;
        
    //     for(const auto& value : item.second){
    //         std::cout << "\t" + std::to_string(std::get<0>(value)) + ", "
    //         + std::get<1>(value) + ", " 
    //         + std::to_string(std::get<2>(value)) + ", " << std::endl;
    //     }
    // }
    // std::cout << "------------------------------------" << std::endl;

    // std::vector<std::pair<std::string, int>> sub_deps;
    // std::cout << "--------- Unique Branches ---------" << std::endl;
    // for(const auto& item : unique_branches_superposition)
    // {
    //     std::cout << item.first;
    //     std::cout << ":\n\t{max_n_segment: " << item.second.max_n_segment;
    //     std::cout << ", height: " << item.second.height;
    //     std::cout << ",\n\tsub_deps: [";
    //     sub_deps = key_to_deps(item.first);
    //     for(int i=0; i<sub_deps.size(); i++)
    //     {
    //         std::cout << sub_deps[i].first << ":" << sub_deps[i].second << ", " ;
    //     }
    //     std::cout << "]}" << std::endl;
    // }
    // std::cout << "------------------------------------" << std::endl;


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